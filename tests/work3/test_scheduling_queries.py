"""开题方案的准确条件、采样重放和语义程序回归。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from envs.work3.core_types import ActionBranch
from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import ActorCriticWork3


def test_query_continuous_slot_prefix_and_read_only() -> None:
    from envs.work3.scheduling_queries import SchedulingQueries, QueryStatus

    env = AirLineEnvWork3()
    env.reset()
    task = env.get_action_candidates()[0]
    team = tuple(env.valid_team_completion_workers(task, [])[:task.demand])
    worker = env.state.workers[team[0]]
    worker.add_interval(2.0, 10.0, "占用一")
    worker.add_interval(12.0, 20.0, "占用二")
    duration = env.duration_for_team(task, team)
    task.standard_duration *= 3.0 / duration
    before = tuple(worker.intervals)
    rng = random.getstate()
    query = SchedulingQueries(env)
    result = query.fixed_team(task.task_key, team, align=0)
    assert result.status == QueryStatus.EXACT
    assert result.start == pytest.approx(20.0)
    assert result.end == pytest.approx(23.0)
    assert tuple(worker.intervals) == before and random.getstate() == rng
    table = query.completions(task.task_key)
    assert table.best(team).team == tuple(sorted(team))
    env.step({"branch": ActionBranch.STATION_EXECUTE, "task_key": task.task_key,
              "team": team, "align": 0})
    assert task.scheduled_start == pytest.approx(result.start)


def test_query_policy_replays_without_live_queries_and_updates(tmp_path: Path) -> None:
    from models.work3.ppo_trainer import PPOTrainerWork3

    torch.set_num_threads(1)
    env = AirLineEnvWork3()
    env.reset()
    task = env.get_action_candidates()[0]
    task.current_station = 1
    env.state.aircraft[0].current_station = 1
    for index, worker_id in enumerate(env.state.station_worker_bindings[1]):
        env.worker_skills[worker_id] = frozenset({task.skill})
        env.worker_efficiencies[worker_id] = float(index + 1)
    env.get_action_candidates = lambda: [task]
    actor = ActorCriticWork3(hidden_dim=32, condition_queries=True)
    snapshot = actor.make_decision_snapshot(env, torch.zeros(32), torch.zeros(2), allow_postpone=False)
    assert snapshot.estimated_cmax is None
    assert snapshot.query_tables
    assert snapshot.graph_snapshot.role_task_facts.shape[1] == 7
    action, lp, _, record = actor.select_snapshot(snapshot, deterministic=True)
    assert action is not None and "query_task_features" in record
    assert len(record["worker_prefixes"]) == 2
    assert record["worker_prefixes"][1] == (action["team"][0],)
    from envs.work3.scheduling_queries import SchedulingQueries
    env.state.current_time += 5.0
    def reject_search(*args: object, **kwargs: object) -> None:
        raise AssertionError("PPO重放不得重新执行查询")
    original = SchedulingQueries.completions
    SchedulingQueries.completions = reject_search
    try:
        values, replay, entropy = actor.evaluate_action_log_probs(
            snapshot.state_features.unsqueeze(0), snapshot.time_features.unsqueeze(0), [record])
    finally:
        SchedulingQueries.completions = original
    assert replay.item() == pytest.approx(lp, abs=1e-5)
    before = actor.task_proj.weight.detach().clone()
    trainer = PPOTrainerWork3(actor)
    trainer.optimizer.zero_grad()
    loss = values.square().mean() - replay.mean() - 0.01 * entropy.mean()
    assert torch.isfinite(loss)
    loss.backward()
    trainer.optimizer.step()
    assert not torch.equal(before, actor.task_proj.weight)
    for module in (actor.query_worker_score, actor.query_alignment_score):
        assert module.weight.grad is not None and torch.count_nonzero(module.weight.grad)
    path = tmp_path / "query.pt"
    trainer.save_checkpoint(str(path))
    trainer.load_checkpoint(str(path))
    historical = PPOTrainerWork3(ActorCriticWork3(hidden_dim=32))
    with pytest.raises(ValueError, match="语义"):
        historical.load_checkpoint(str(path))


def test_capacity_efficiency_unknown_predecessor_and_prefix() -> None:
    from envs.work3.scheduling_queries import SchedulingQueries, QueryStatus
    from envs.work3.core_types import TaskStatus

    env = AirLineEnvWork3(max_slots_per_station=1)
    env.reset()
    task = env.get_action_candidates()[0]
    task.current_station = 1
    env.state.aircraft[0].current_station = 1
    workers = tuple(env.state.station_worker_bindings[1])
    for index, worker_id in enumerate(workers):
        env.worker_skills[worker_id] = frozenset({task.skill})
        env.worker_efficiencies[worker_id] = float(index + 1)
    task.standard_duration = 12.0
    task.in_station_offset = 9.0
    occupied = next(other for other in env.state.tasks.values() if other.task_key != task.task_key)
    occupied.scheduled_start, occupied.execution_duration = 0.0, 7.0
    env._station_occupied_tasks[1].add(occupied.task_key)
    query = SchedulingQueries(env)
    team = workers[:2]
    fixed = query.fixed_team(task.task_key, team, align=0)
    assert fixed.start == 7.0 and "station" in fixed.reasons
    assert query.fixed_team(task.task_key, team, align=1).start == 9.0
    assert query.fixed_team(task.task_key, workers[2:], align=0).duration < fixed.duration
    table = query.completions(task.task_key)
    assert workers[0] in table.best((workers[0],)).team
    assert query.completions(task.task_key).best(workers).status == QueryStatus.NO_COMPLETION
    task.predecessor_keys = (occupied.task_key,)
    occupied.status = TaskStatus.UNREADY
    unknown = query.fixed_team(task.task_key, team, align=0)
    assert unknown.status == QueryStatus.UNKNOWN and unknown.start is None
    assert unknown.features(env.state.current_time, env.state.h0)[8] == 0.0


@pytest.mark.parametrize("mode", ["postpone", "advance", "forced"])
def test_query_branch_truncation_replays_only_actual_choices(mode: str) -> None:
    torch.set_num_threads(1)
    env = AirLineEnvWork3()
    env.reset()
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=16)
    if mode == "postpone":
        task = next(task for task in env.get_action_candidates() if env.can_postpone(task))
        env.get_action_candidates = lambda: [task]
    elif mode == "advance":
        task = env.get_action_candidates()[0]
        task.material_ready_time = 100.0
        team = tuple(env.valid_team_completion_workers(task, [])[:task.demand])
        env.step({"task_key": task.task_key, "team": team, "branch": ActionBranch.STATION_EXECUTE, "align": 0})
    else:
        env.get_action_candidates = lambda: []
    with torch.no_grad():
        actor.branch_head[-1].weight.zero_()
        actor.branch_head[-1].bias[:] = torch.tensor([-100.0, 100.0])
    snapshot = actor.make_decision_snapshot(env, torch.zeros(32), torch.zeros(2))
    action, lp, _, record = actor.select_snapshot(snapshot, deterministic=True)
    assert record["action_type"] == {"postpone": "schedule", "advance": "advance_to_next_event", "forced": "forced_advance"}[mode]
    assert "worker_prefixes" not in record and "query_alignment_features" not in record
    if mode == "postpone":
        assert action["branch"] == ActionBranch.POSTPONE
    _, replay, _ = actor.evaluate_action_log_probs(snapshot.state_features.unsqueeze(0),
                                                   snapshot.time_features.unsqueeze(0), [record])
    assert replay.item() == pytest.approx(lp, abs=1e-5)


def test_actual_time_cost_closes_without_predicted_shaping() -> None:
    from utils.work3.objective_evaluator import ObjectiveWeights, evaluate_trajectory_objective

    with pytest.raises(ValueError, match="有限"):
        ObjectiveWeights(w_makespan=float("inf"))

    env = AirLineEnvWork3(weights=ObjectiveWeights(w_makespan=1.0))
    env.reset()
    task = env.get_action_candidates()[0]
    team = tuple(env.valid_team_completion_workers(task, [])[:task.demand])
    _, reward, _, _, _ = env.step({"branch": ActionBranch.STATION_EXECUTE,
                                  "task_key": task.task_key, "team": team, "align": 0})
    assert env.cost_makespan == pytest.approx(env.state.current_time / env.state.h0)
    assert -reward == pytest.approx(env.cumulative_cost)
    report = evaluate_trajectory_objective(env.state, env.weights)
    assert report.j_total == pytest.approx(env.cumulative_cost)


def test_query_training_uses_existing_lightning_and_two_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.work3.train_ppo_work3 as training

    def reject_time_cache() -> None:
        raise AssertionError("查询路径不应创建时间标签缓存")

    monkeypatch.setattr(training, "PendingTimeLabelCache", reject_time_cache)
    report = training.run_training(method_variant="QUERY", num_envs=2, steps_per_iter=4,
                                   batch_size=4, max_wall_seconds=120.0,
                                   output_ckpt=tmp_path / "query.pt")
    assert report["lightning_optimization_steps"] == 1


def test_offline_program_binds_real_candidate_and_reuses_plan(tmp_path: Path) -> None:
    import json
    from envs.work3.query_program import load_condition_program, validate_condition_program, compile_condition_program

    program = load_condition_program(Path("conf") / "work3" / "condition_program.json")
    env = AirLineEnvWork3()
    env.reset()
    task = env.get_action_candidates()[0]
    first = program.execute(env, task.task_key)
    assert first.task_key == task.task_key
    worker_id = first.best().team[0]
    env.state.workers[worker_id].add_interval(0.0, 100.0, "新占用")
    second = program.execute(env, task.task_key)
    assert second.best((worker_id,)).start >= 100.0
    with pytest.raises(ValueError, match="白名单"):
        validate_condition_program({"version": program.version, "api": "eval", "args": {}})
    source = json.loads((Path("conf") / "work3" / "condition_program.json").read_text(encoding="utf-8"))
    generated = compile_condition_program(program.description, lambda text, feedback: dict(source),
                                         cache_dir=tmp_path, attempts=1)
    def no_second_request(text: str, feedback: str) -> dict:
        raise AssertionError("同一说明和接口版本不应重新请求模型")
    assert compile_condition_program(program.description, no_second_request, cache_dir=tmp_path, attempts=1) == generated
    with pytest.raises(ValueError, match="语义核验"):
        load_condition_program(generated)


@pytest.mark.parametrize("disturbed", [False, True])
def test_query_network_completes_real_two_aircraft_batch(tmp_path: Path, disturbed: bool) -> None:
    from scripts.work3.evaluate_query import evaluate_query_trajectory
    from .test_work3_m02_disturbed_pulse_trace import _two_aircraft_baseline
    from utils.work3.objective_evaluator import ObjectiveWeights

    torch.set_num_threads(1)
    torch.manual_seed(42)
    path = tmp_path / "two_aircraft.json"
    _two_aircraft_baseline(path)
    env = AirLineEnvWork3(baseline_json_path=path, weights=ObjectiveWeights(w_makespan=1.0))
    env.reset()
    target = env.get_action_candidates()[0]
    scenario = {"scenario_id": "QUERY_CHECK", "tau": 0.0, "recovery_time": 100.0,
                "affected_task_keys": [target.task_key], "valid": True} if disturbed else None
    result = evaluate_query_trajectory(env, ActorCriticWork3(condition_queries=True, hidden_dim=32),
                                       scenario=scenario, max_decisions=100)
    assert result["success"] and result["terminated"] and not result["truncated"]
    assert result["completed_tasks"] == 10
    assert result["independent_feasible"] and result["cost_ledger_matches"]
    if disturbed:
        assert env.disturbance_event_triggered
        assert target.task_key in env.disturbance_event_results["QUERY_CHECK"]["actual_hit_task_keys"]
