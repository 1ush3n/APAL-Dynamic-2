"""开题方案的准确条件、采样重放和语义程序回归。"""

from __future__ import annotations

import random
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from apal.envs.work3.core_types import ActionBranch
from apal.envs.work3.environment import AirLineEnvWork3
from apal.models.work3.actor_critic import ActorCriticWork3


@pytest.mark.parametrize("later_worker", [False, True])
def test_search_reports_later_station_block_after_waiting_for_workers(later_worker: bool) -> None:
    from apal.core.constraints import calculate_team_synergy_factor
    from apal.envs.work3.scheduling_queries import SchedulingQueries

    env = AirLineEnvWork3(baseline_json_path=Path("data") / "work3" / "real_680_k10_baseline.json",
                         raw_data_path=Path("data") / "680.csv", max_slots_per_station=2)
    env.reset()
    for key, team, release, duration in (("0_317", (0, 5), 0.0, 10.0),
                                         ("0_706", (6, 22), 10.0, 10.0),
                                         ("0_696", (66, 80), 10.0, 10.0),
                                         ("0_18", (0, 5), 0.0, 3.0)):
        task = env.state.tasks[key]
        task.material_ready_time = release
        task.standard_duration = duration * sum(env.worker_efficiencies[w] for w in team) * calculate_team_synergy_factor(task.demand) / task.demand
        if key != "0_18":
            env.step({"task_key": key, "team": team, "branch": ActionBranch.STATION_EXECUTE, "align": 0})
    expected_conflicts = {"0_317", "0_706", "0_696"}
    if later_worker:
        blocker = next(task for task in env.get_action_candidates()
                       if task.skill == 0 and task.demand == 2 and env.can_reserve(task) and task.task_key != "0_18")
        blocker.material_ready_time = 21.0
        blocker.standard_duration = 9.0 * sum(env.worker_efficiencies[w] for w in (0, 5)) * calculate_team_synergy_factor(2) / 2
        env.step({"task_key": blocker.task_key, "team": (0, 5), "branch": ActionBranch.STATION_EXECUTE, "align": 0})
        expected_conflicts.add(blocker.task_key)
    result = SchedulingQueries(env).fixed_team("0_18", (0, 5), align=0)
    assert result.start == pytest.approx(30.0 if later_worker else 20.0)
    assert set(result.reasons) == {"worker", "station"}
    assert set(result.conflict_keys) == expected_conflicts
    assert result.features(0.0, env.state.h0)[12] == 1.0
    env.max_slots_per_station = 3
    assert SchedulingQueries(env).fixed_team("0_18", (0, 5), align=0).start == pytest.approx(10.0)


def test_equivalent_description_keeps_checkpoint_and_mixed_replay_compatible(tmp_path: Path) -> None:
    from apal.envs.work3.query_program import validate_condition_program
    from apal.models.work3.ppo_trainer import PPOTrainerWork3

    torch.set_num_threads(1)
    env = AirLineEnvWork3()
    env.reset()
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=16)
    snapshot = actor.make_decision_snapshot(env, torch.zeros(32), torch.zeros(2))
    _, lp, _, record = actor.select_snapshot(snapshot, deterministic=True)
    checkpoint = tmp_path / "program.pt"
    trainer = PPOTrainerWork3(actor)
    trainer.save_checkpoint(str(checkpoint))
    plan = json.loads((Path("conf") / "work3" / "condition_program.json").read_text(encoding="utf-8"))
    plan["description"] = "读取当前工序团队可执行窗口。"
    actor.query_program = validate_condition_program(plan)
    assert actor.query_program.fingerprint != snapshot.query_program_hashes[0]
    trainer.load_checkpoint(str(checkpoint))
    legacy = torch.load(checkpoint, weights_only=False)
    legacy["policy_input_semantics"].pop("condition_normalization", None)
    legacy["policy_input_semantics"]["query_program_hash"] = snapshot.query_program_hashes[0]
    torch.save(legacy, checkpoint)
    trainer.load_checkpoint(str(checkpoint))
    legacy["policy_input_semantics"]["condition_layout"] = ("错误字段",)
    torch.save(legacy, checkpoint)
    with pytest.raises(ValueError, match="语义"):
        trainer.load_checkpoint(str(checkpoint))
    new_snapshot = actor.make_decision_snapshot(env, torch.zeros(32), torch.zeros(2))
    assert new_snapshot.query_tables == snapshot.query_tables
    _, new_lp, _, new_record = actor.select_snapshot(new_snapshot, deterministic=True)
    _, replay, _ = actor.evaluate_action_log_probs(torch.stack([snapshot.state_features, new_snapshot.state_features]),
        torch.stack([snapshot.time_features, new_snapshot.time_features]), [record, new_record])
    assert replay.tolist() == pytest.approx([lp, new_lp], abs=1e-5)
    with pytest.raises(ValueError, match="归一化"):
        actor.evaluate_action_log_probs(snapshot.state_features.unsqueeze(0), snapshot.time_features.unsqueeze(0),
                                         [{**record, "condition_normalization": "绝对时间"}])
    with pytest.raises(ValueError):
        actor.evaluate_action_log_probs(snapshot.state_features.unsqueeze(0), snapshot.time_features.unsqueeze(0),
                                         [{**record, "query_version": "错误版本"}])
    damaged = record["query_task_features"].clone()
    damaged[0, 0] = float("nan")
    with pytest.raises(ValueError, match="非有限"):
        actor.evaluate_action_log_probs(snapshot.state_features.unsqueeze(0), snapshot.time_features.unsqueeze(0),
                                         [{**record, "query_task_features": damaged}])


def test_continuation_evaluation_preserves_progress_and_initial_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from .test_work3_m02_disturbed_pulse_trace import _two_aircraft_baseline, TEAMS
    from apal.evaluation.query import evaluate_query_trajectory
    from apal.utils.work3.objective_evaluator import ObjectiveWeights

    path = tmp_path / "continuation.json"
    _two_aircraft_baseline(path)
    env = AirLineEnvWork3(baseline_json_path=path, weights=ObjectiveWeights(w_makespan=1.0))
    env.reset()
    env.step({"task_key": "0_15", "team": TEAMS[15], "branch": ActionBranch.STATION_EXECUTE, "align": 0})
    env.step({"task_key": "0_18", "team": TEAMS[18], "branch": ActionBranch.STATION_EXECUTE, "align": 1})
    before_time, before_cost = env.state.current_time, env.cumulative_cost
    before_steps = env.step_count
    scheduled = env.state.tasks["0_18"].scheduled_start
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=16)
    seen = []
    original = actor.select_snapshot
    def capture(snapshot: object, **kwargs: object) -> tuple:
        seen.append((snapshot.current_time, env.state.tasks["0_15"].status, env.state.tasks["0_18"].scheduled_start, snapshot.episode_index))
        return original(snapshot, **kwargs)
    monkeypatch.setattr(actor, "select_snapshot", capture)
    result = evaluate_query_trajectory(env, actor, reset=False, max_decisions=100,
        scenario={"scenario_id": "续调度", "tau": before_time, "recovery_time": 12.0,
                  "affected_task_keys": ["0_18"], "valid": True})
    from apal.envs.work3.core_types import TaskStatus
    assert seen[0][0] == before_time and seen[0][1] == TaskStatus.COMPLETED
    assert seen[0][2] >= scheduled
    assert seen[0][3] == before_steps
    assert tuple(env.state.tasks["0_18"].assigned_team) == tuple(sorted(TEAMS[18]))
    assert env.state.tasks["0_18"].alignment == 1
    assert result["initial_cost"] == before_cost and result["initial_time"] == before_time
    assert result["success"] and result["cost_ledger_matches"] and result["incremental_cost_ledger_matches"]


@pytest.mark.parametrize("reserved", [False, True])
def test_effective_disturbance_preserves_committed_prefix_and_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reserved: bool,
) -> None:
    from .test_work3_m02_disturbed_pulse_trace import _two_aircraft_baseline, TEAMS
    from apal.envs.work3.core_types import TaskStatus
    from apal.envs.work3.event_queue import EventType
    from apal.evaluation.query import evaluate_query_trajectory
    from apal.utils.work3.objective_evaluator import ObjectiveWeights

    torch.set_num_threads(1)
    torch.manual_seed(42)
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=32)
    path = tmp_path / "effective_disturbance.json"
    _two_aircraft_baseline(path)
    prefixes = []
    for disturbed in (False, True):
        env = AirLineEnvWork3(baseline_json_path=path, weights=ObjectiveWeights(w_makespan=1.0))
        env.reset()
        env.step({"task_key": "0_15", "team": TEAMS[15], "branch": ActionBranch.STATION_EXECUTE, "align": 0})
        target = env.state.tasks["0_18"]
        if reserved:
            env.step({"task_key": target.task_key, "team": TEAMS[18], "branch": ActionBranch.STATION_EXECUTE, "align": 1})
        expected_status = TaskStatus.RESERVED if reserved else TaskStatus.READY
        assert target.status == expected_status
        before = (env.state.current_time, env.cumulative_cost, env.step_count, target.scheduled_start,
                  target.material_ready_time, target.generation, target.current_station,
                  tuple(target.assigned_team), target.alignment)
        prefixes.append(before)
        old_starts = [event for event in env.event_queue._heap
                      if event.task_key == target.task_key and event.event_type == EventType.TASK_START]
        starts, finishes, after_event = [], [], []
        original_start, original_finish, original_load = env._on_task_started, env._on_task_completed, env.load_scenario

        def record_start(task: Any, when: float) -> None:
            starts.append((task.task_key, when))
            original_start(task, when)

        def record_finish(task: Any) -> None:
            finishes.append(task.task_key)
            original_finish(task)

        def record_scenario(scenario: dict[str, Any]) -> None:
            original_load(scenario)
            after_event.append((target.status, target.material_ready_time, target.scheduled_start, target.generation,
                                target.current_station, tuple(target.assigned_team), target.alignment))
            if reserved:
                assert len(old_starts) == 1 and not env.event_queue._is_event_valid(old_starts[0])
                for worker_id in target.assigned_team:
                    intervals = [item for item in env.state.workers[worker_id].intervals if item.task_key == target.task_key]
                    assert len(intervals) == 1 and intervals[0].start == target.scheduled_start

        monkeypatch.setattr(env, "_on_task_started", record_start)
        monkeypatch.setattr(env, "_on_task_completed", record_finish)
        monkeypatch.setattr(env, "load_scenario", record_scenario)
        scenario = {"scenario_id": "EFFECTIVE_RESERVED" if reserved else "EFFECTIVE_READY",
                    "tau": before[0], "recovery_time": 25.0, "affected_task_keys": [target.task_key], "valid": True}
        result = evaluate_query_trajectory(env, actor, reset=False, max_decisions=100,
                                           scenario=scenario if disturbed else None)
        assert result["success"] and result["terminated"] and not result["truncated"]
        assert result["completed_tasks"] == result["total_tasks"] == 10
        assert result["independent_feasible"] and not result["constraint_violations"]
        assert result["cost_ledger_matches"] and result["incremental_cost_ledger_matches"]
        assert result["initial_time"] == before[0] and result["initial_cost"] == before[1]
        assert env.state.tasks["0_15"].status == TaskStatus.COMPLETED
        assert [key for key, _ in starts].count(target.task_key) == finishes.count(target.task_key) == 1
        if disturbed:
            assert result["disturbance_results"][scenario["scenario_id"]]["actual_hit_task_keys"] == (target.task_key,)
            assert after_event[0][1] == 25.0 and after_event[0][1] - before[4] > 0.0
            assert target.actual_start >= 25.0
            if reserved:
                assert after_event[0][0] == TaskStatus.RESERVED and after_event[0][2] > before[3]
                assert after_event[0][3] > before[5] and after_event[0][4:] == before[6:]
                assert (target.current_station, tuple(target.assigned_team), target.alignment) == before[6:]
            else:
                assert after_event[0][0] == TaskStatus.UNREADY
    assert prefixes[0] == prefixes[1]


def test_scoped_programs_bind_relations_and_conditions_into_same_policy(tmp_path: Path) -> None:
    from .test_work3_m02_disturbed_pulse_trace import _two_aircraft_baseline, TEAMS
    from apal.envs.work3.query_program import load_condition_program, compile_condition_program

    torch.set_num_threads(1)
    path = tmp_path / "program_batch.json"
    _two_aircraft_baseline(path)
    env = AirLineEnvWork3(baseline_json_path=path)
    env.reset()
    env.step({"task_key": "0_15", "team": TEAMS[15], "branch": ActionBranch.STATION_EXECUTE, "align": 0})
    env.load_scenario({"scenario_id": "已揭示物料", "tau": env.state.current_time, "recovery_time": 100.0,
                       "affected_task_keys": ["1_15"], "valid": True})
    paths = [Path("conf") / "work3" / name for name in ("program_predecessor_focus.json", "program_material_focus.json")]
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=16, query_program_path=paths[0])
    snapshots, records, probabilities = [], [], []
    scoring_inputs = []
    handle = actor.task_proj.register_forward_pre_hook(lambda module, inputs: scoring_inputs.append(inputs[0].detach().clone()))
    for program_path in paths:
        actor.query_program = load_condition_program(program_path)
        snapshot = actor.make_decision_snapshot(env, torch.zeros(32), torch.zeros(2))
        snapshots.append(snapshot)
        _, lp, _, record = actor.select_snapshot(snapshot, deterministic=True)
        records.append(record)
        probabilities.append(lp)
        raw = json.loads(program_path.read_text(encoding="utf-8"))
        def mock_generate(text: str, feedback: str) -> dict:
            assert text == raw["description"] and "已核验检索示例" in feedback
            return dict(raw)
        compiled = compile_condition_program(raw["description"], mock_generate, cache_dir=tmp_path, attempts=1,
                                             verified_examples=(raw,))
        generated = json.loads(compiled.read_text(encoding="utf-8"))
        generated["review"] = "offline_verified_retrieval_only"
        from apal.envs.work3.query_program import validate_condition_program
        manual = actor.query_program.execute_candidates(env, snapshot.candidate_task_keys)
        assert validate_condition_program(generated).execute_candidates(env, snapshot.candidate_task_keys) == manual
    keys = snapshots[0].candidate_task_keys
    assert keys == snapshots[1].candidate_task_keys and "0_18" in keys and "1_15" in keys
    for key, left, right in zip(keys, snapshots[0].query_tables, snapshots[1].query_tables):
        if key == "0_18":
            assert left.best().start is not None and right.best().start is None
        if key == "1_15":
            assert left.best().start is None and right.best().start == 100.0
    assert not torch.equal(records[0]["query_task_features"], records[1]["query_task_features"])
    handle.remove()
    assert snapshots[0].branch_masks == snapshots[1].branch_masks
    assert all(torch.equal(observed, record["query_task_features"]) for observed, record in zip(scoring_inputs, records))
    _, replay, _ = actor.evaluate_action_log_probs(torch.stack([s.state_features for s in snapshots]),
        torch.stack([s.time_features for s in snapshots]), records)
    assert replay.tolist() == pytest.approx(probabilities, abs=1e-5)


@pytest.mark.parametrize("relation", ["predecessors", "successors"])
def test_program_relation_scope_negation_and_rejections(relation: str) -> None:
    import copy
    from apal.envs.work3.query_program import validate_condition_program

    env = AirLineEnvWork3()
    env.reset()
    from apal.envs.work3.core_types import TaskStatus
    child = next(task for task in env.state.tasks.values() if task.aircraft_id == 0 and task.predecessors
                 and env.state.tasks[f"0_{task.predecessors[0]}"].status != TaskStatus.COMPLETED)
    parent = env.state.tasks[f"0_{child.predecessors[0]}"]
    source, target = (child, parent) if relation == "predecessors" else (parent, child)
    plan = json.loads((Path("conf") / "work3" / "program_material_focus.json").read_text(encoding="utf-8"))
    plan["steps"] = [{"api": "bind_tasks", "aircraft_ids": [0], "task_ids": [source.task_id]},
                     {"api": "related_tasks", "relation": relation, "include_self": False},
                     {"api": "where", "predicate": "material_wait", "negate": True}, {"api": "team_completions"}]
    keys = (target.task_key, f"1_{target.task_id}")
    tables = validate_condition_program(plan).execute_candidates(env, keys)
    assert "outside_program_scope" not in tables[0].unavailable.reasons
    assert tables[1].unavailable.reasons == ("outside_program_scope",)
    target.material_ready_time = 100.0
    assert validate_condition_program(plan).execute_candidates(env, keys)[0].unavailable.reasons == ("outside_program_scope",)
    bad = copy.deepcopy(plan)
    bad["steps"][1]["api"] = "eval"
    with pytest.raises(ValueError, match="白名单"):
        validate_condition_program(bad)
    bad = copy.deepcopy(plan)
    bad["steps"][0]["task_ids"] = [True]
    with pytest.raises(ValueError, match="整数"):
        validate_condition_program(bad)
    bad = copy.deepcopy(plan)
    bad["steps"][0]["aircraft_ids"] = [999]
    with pytest.raises(ValueError, match="不存在"):
        validate_condition_program(bad).execute_candidates(env, keys)


def test_query_continuous_slot_prefix_and_read_only() -> None:
    from apal.envs.work3.scheduling_queries import SchedulingQueries, QueryStatus

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
    from apal.models.work3.ppo_trainer import PPOTrainerWork3

    torch.set_num_threads(1)
    torch.manual_seed(42)
    env = AirLineEnvWork3()
    env.reset()
    task = env.get_action_candidates()[0]
    task.current_station = 1
    # 两种对齐必须给出不同条件，才能检验对齐查询投影的非零梯度。
    task.in_station_offset = env.state.h0
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
    assert not torch.equal(record["query_alignment_features"][0], record["query_alignment_features"][1])
    from apal.envs.work3.scheduling_queries import SchedulingQueries
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
    from apal.envs.work3.scheduling_queries import SchedulingQueries, QueryStatus
    from apal.envs.work3.core_types import TaskStatus

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
    from apal.utils.work3.objective_evaluator import ObjectiveWeights, evaluate_trajectory_objective

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
    import apal.training.work3.train as training

    def reject_time_cache() -> None:
        raise AssertionError("查询路径不应创建时间标签缓存")

    monkeypatch.setattr(training, "PendingTimeLabelCache", reject_time_cache)
    report = training.run_training(method_variant="QUERY", num_envs=2, steps_per_iter=4,
                                   batch_size=4, max_wall_seconds=120.0,
                                   query_program_path=Path("conf") / "work3" / "program_predecessor_focus.json",
                                   output_ckpt=tmp_path / "query.pt")
    assert report["lightning_optimization_steps"] == 1
    from apal.envs.work3.query_program import load_condition_program
    checkpoint = torch.load(tmp_path / "query.pt", weights_only=False)
    assert checkpoint["query_program_source"] == load_condition_program(Path("conf") / "work3" / "program_predecessor_focus.json").fingerprint
    assert report["environment_worker_cuda_initialized"] == [False, False]


@pytest.mark.parametrize("bad_result", [None, [], "bad"], ids=["null", "array", "string"])
def test_non_object_program_retries_with_feedback_and_keeps_review_gate(tmp_path: Path, bad_result: Any) -> None:
    from apal.envs.work3.query_program import compile_condition_program, load_condition_program

    source = json.loads((Path("conf") / "work3" / "condition_program.json").read_text(encoding="utf-8"))
    feedbacks: list[str] = []

    def generate(text: str, feedback: str) -> Any:
        assert text == source["description"]
        feedbacks.append(feedback)
        return bad_result if len(feedbacks) == 1 else deepcopy(source)

    generated = compile_condition_program(source["description"], generate, cache_dir=tmp_path, attempts=2,
                                         verified_examples=(deepcopy(source),))
    assert len(feedbacks) == 2
    assert "JSON对象" in feedbacks[1]
    assert "已核验检索示例：" in feedbacks[1]
    assert json.dumps((source,), ensure_ascii=False) in feedbacks[1]
    assert json.loads(generated.read_text(encoding="utf-8"))["review"] == "pending_semantic_review"
    with pytest.raises(ValueError, match="语义核验"):
        load_condition_program(generated)

    def no_second_request(text: str, feedback: str) -> dict[str, Any]:
        raise AssertionError("已缓存的候选计划不应再次请求生成，也不代表通过语义审核")

    assert compile_condition_program(source["description"], no_second_request,
                                     cache_dir=tmp_path, attempts=2) == generated


@pytest.mark.parametrize("bad_result", [None, [], "bad"], ids=["null", "array", "string"])
def test_non_object_program_stops_at_attempt_limit_without_cache(tmp_path: Path, bad_result: Any) -> None:
    from apal.envs.work3.query_program import compile_condition_program

    calls = 0

    def generate(text: str, feedback: str) -> Any:
        nonlocal calls
        calls += 1
        return bad_result

    with pytest.raises(ValueError, match="JSON对象"):
        compile_condition_program("查询当前候选工序的合法团队窗口。", generate, cache_dir=tmp_path, attempts=2)
    assert calls == 2
    assert not list(tmp_path.glob("*.json"))


def test_offline_program_binds_real_candidate_and_reuses_plan(tmp_path: Path) -> None:
    import json
    from apal.envs.work3.query_program import load_condition_program, validate_condition_program, compile_condition_program

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
    from apal.evaluation.query import evaluate_query_trajectory
    from .test_work3_m02_disturbed_pulse_trace import _two_aircraft_baseline
    from apal.utils.work3.objective_evaluator import ObjectiveWeights

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
