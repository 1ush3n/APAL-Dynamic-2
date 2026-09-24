"""W3-13运行时的新配置、种子和并行训练边界测试。"""

from __future__ import annotations

import importlib
import importlib.util
import os
import random
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT_DIR / "conf" / "work3" / "train_pilot.yaml"


@pytest.fixture
def restore_global_random_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def _runtime_config_module() -> ModuleType:
    """将缺少的新运行时模块呈现为清晰的测试失败，而不是导入错误。"""
    if importlib.util.find_spec("training.work3_runtime_config") is None:
        pytest.fail("工作三运行时配置加载器尚未实现", pytrace=False)
    return importlib.import_module("training.work3_runtime_config")


def _decision_snapshot_module() -> ModuleType:
    if importlib.util.find_spec("envs.work3.decision_snapshot") is None:
        pytest.fail("工作三决策快照与纯团队掩码尚未实现", pytrace=False)
    return importlib.import_module("envs.work3.decision_snapshot")


def _work3_vector_env_module() -> ModuleType:
    if importlib.util.find_spec("training.work3_vector_env") is None:
        pytest.fail("工作三spawn环境运行时尚未实现", pytrace=False)
    return importlib.import_module("training.work3_vector_env")


def test_runtime_config_loads_smoke_defaults_and_hashes_resolved_yaml() -> None:
    module = _runtime_config_module()

    config = module.load_work3_runtime_config(DEFAULT_CONFIG)
    resolved_yaml, fingerprint = module.resolved_config_fingerprint(config)

    assert config.runtime.run_mode == "smoke"
    assert config.runtime.method_profile == "C"
    assert config.runtime.num_envs == 1
    assert config.runtime.amp_dtype == "fp32"
    assert config.runtime.start_method == "spawn"
    assert config.runtime.dataloader_num_workers == 0
    assert config.runtime.total_env_steps == 64
    assert "num_envs: 1" in resolved_yaml
    assert len(fingerprint) == 64
    assert set(fingerprint) <= set("0123456789abcdef")


def test_runtime_config_overrides_change_resolved_values_and_fingerprint() -> None:
    module = _runtime_config_module()

    default = module.load_work3_runtime_config(DEFAULT_CONFIG)
    overridden = module.load_work3_runtime_config(
        DEFAULT_CONFIG,
        overrides=("runtime.seed=31415", "runtime.num_envs=2"),
    )
    _, default_fingerprint = module.resolved_config_fingerprint(default)
    resolved_yaml, override_fingerprint = module.resolved_config_fingerprint(overridden)

    assert overridden.runtime.seed == 31415
    assert overridden.runtime.num_envs == 2
    assert "seed: 31415" in resolved_yaml
    assert override_fingerprint != default_fingerprint


def test_runtime_config_rejects_missing_required_fields(tmp_path: Path) -> None:
    module = _runtime_config_module()
    incomplete_config = tmp_path / "incomplete.yaml"
    incomplete_config.write_text("runtime:\n  run_mode: smoke\n", encoding="utf-8")

    with pytest.raises(ValueError, match="缺少必需配置"):
        module.load_work3_runtime_config(incomplete_config)


@pytest.mark.parametrize(
    "override",
    (
        "runtime.num_envs=0",
        "runtime.amp_dtype=fp8",
        "runtime.total_env_steps=0",
        "runtime.dataloader_num_workers=1",
        "runtime.start_method=fork",
        "ppo.gamma=1.2",
        "paths.baseline=",
        "paths.baseline=' '",
    ),
)
def test_runtime_config_rejects_unsupported_or_out_of_range_values(override: str) -> None:
    module = _runtime_config_module()

    with pytest.raises((ValueError, TypeError)):
        module.load_work3_runtime_config(DEFAULT_CONFIG, overrides=(override,))


def test_runtime_seed_and_worker_episode_seed_are_reproducible(
    restore_global_random_state: None,
) -> None:
    module = _runtime_config_module()

    worker_seed = module.derive_worker_seed(42, worker_id=1, episode_index=3)
    assert worker_seed == module.derive_worker_seed(42, worker_id=1, episode_index=3)
    assert worker_seed != module.derive_worker_seed(42, worker_id=0, episode_index=3)
    assert worker_seed != module.derive_worker_seed(42, worker_id=1, episode_index=4)

    module.seed_work3_runtime(1234, deterministic=True)
    first = (random.random(), float(np.random.random()), torch.rand(3))
    module.seed_work3_runtime(1234, deterministic=True)
    second = (random.random(), float(np.random.random()), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.are_deterministic_algorithms_enabled()


def test_pilot_config_requires_a_positive_successful_batch_target() -> None:
    module = _runtime_config_module()

    with pytest.raises(ValueError, match="successful_batch_target"):
        module.load_work3_runtime_config(
            DEFAULT_CONFIG,
            overrides=("runtime.run_mode=pilot",),
        )

    pilot_config = module.load_work3_runtime_config(
        DEFAULT_CONFIG,
        overrides=(
            "runtime.run_mode=pilot",
            "runtime.successful_batch_target=1",
        ),
    )
    assert pilot_config.runtime.successful_batch_target == 1


def test_snapshot_worker_mask_preserves_legal_team_completion_and_order() -> None:
    module = _decision_snapshot_module()
    worker = module.WorkerSnapshot
    context_type = module.TeamCompletionContext
    context = context_type(
        task_key="7_12",
        station_id=1,
        required_skill=3,
        demand=2,
        workers=(
            worker(18, (3,), 1.0, ((20.0, 22.0, "other"),)),
            worker(65, (3,), 0.9, ()),
            worker(91, (1,), 1.1, ()),
        ),
    )

    first_pointer = module.worker_completion_mask(context, (), max_station_workers=5)
    second_pointer = module.worker_completion_mask(context, (18,), max_station_workers=5)
    impossible_prefix = module.worker_completion_mask(
        context,
        (18, 65),
        max_station_workers=5,
    )

    assert first_pointer == (True, True, False, False, False)
    assert second_pointer == (False, True, False, False, False)
    assert impossible_prefix == (False, False, False, False, False)


def test_snapshot_worker_mask_disables_task_without_a_full_skill_team() -> None:
    module = _decision_snapshot_module()
    context = module.TeamCompletionContext(
        task_key="2_9",
        station_id=2,
        required_skill=4,
        demand=2,
        workers=(
            module.WorkerSnapshot(31, (4,), 1.0, ()),
            module.WorkerSnapshot(32, (1, 2), 1.0, ()),
        ),
    )

    assert module.worker_completion_mask(context, (), max_station_workers=3) == (
        False,
        False,
        False,
    )


def test_environment_team_context_is_an_independent_cpu_snapshot() -> None:
    from envs.work3.environment import AirLineEnvWork3

    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()
    task = env.get_action_candidates()[0]
    context = env.get_team_completion_context(task)

    station_workers = tuple(env.state.station_worker_bindings[task.current_station])
    assert tuple(worker.worker_id for worker in context.workers) == station_workers
    assert all(isinstance(worker.calendar_intervals, tuple) for worker in context.workers)
    original_skills = context.workers[0].skills
    original_intervals = context.workers[0].calendar_intervals

    env.worker_skills[station_workers[0]] = frozenset({99})
    env.state.workers[station_workers[0]].add_interval(100.0, 101.0, "later")

    assert context.workers[0].skills == original_skills
    assert context.workers[0].calendar_intervals == original_intervals


def test_live_team_eligibility_does_not_construct_full_calendar_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from envs.work3.environment import AirLineEnvWork3

    env = AirLineEnvWork3(
        baseline_json_path=ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    )
    env.reset()
    task = env.get_ready_tasks()[0]

    def reject_snapshot_construction(*args: object, **kwargs: object) -> None:
        raise AssertionError("普通团队资格查询不应构造日历快照DTO")

    monkeypatch.setattr(env, "_team_completion_context", reject_snapshot_construction)
    valid_workers = env.valid_team_completion_workers(task, [])
    assert len(valid_workers) >= task.demand


def test_decision_snapshot_clones_tensor_and_graph_payloads_to_cpu() -> None:
    from torch_geometric.data import HeteroData

    module = _decision_snapshot_module()
    context = module.TeamCompletionContext(
        task_key="0_0",
        station_id=0,
        required_skill=-1,
        demand=1,
        workers=(module.WorkerSnapshot(4, (0,), 1.0, ()),),
    )
    state_features = torch.ones(32)
    time_features = torch.tensor([0.25, 0.5])
    task_features = torch.ones((1, 8))
    graph = HeteroData()
    graph["task"].x = torch.ones((1, 3))

    snapshot = module.DecisionSnapshot(
        worker_id=0,
        episode_id=12,
        episode_index=3,
        state_features=state_features,
        time_features=time_features,
        graph_snapshot=graph,
        candidate_task_keys=("0_0",),
        candidate_task_features=task_features,
        branch_masks=((True, False),),
        team_contexts=(context,),
        candidate_task_node_indices=(0,),
        worker_node_indices=((0,),),
        reserved_flags=(False,),
        advance_available=False,
        current_time=5.0,
        cycle_id=1,
        estimated_cmax=10.0,
        h0=10.0,
        last_transfer_time=0.0,
        graph_version="work3_graph_v2",
    )

    state_features[0] = 99.0
    time_features[0] = 99.0
    task_features[0, 0] = 99.0
    graph["task"].x[0, 0] = 99.0

    assert snapshot.state_features.device.type == "cpu"
    assert snapshot.time_features.device.type == "cpu"
    assert snapshot.candidate_task_features.device.type == "cpu"
    assert snapshot.state_features[0].item() == 1.0
    assert snapshot.time_features[0].item() == 0.25
    assert snapshot.candidate_task_features[0, 0].item() == 1.0
    assert snapshot.graph_snapshot["task"].x[0, 0].item() == 1.0


def test_snapshot_masks_match_environment_and_actor_replay_probability() -> None:
    from dataclasses import replace
    from itertools import permutations

    from envs.work3.environment import AirLineEnvWork3
    from models.work3.actor_critic import ActorCriticWork3

    env = AirLineEnvWork3(
        baseline_json_path=ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    )
    env.reset()
    actor = ActorCriticWork3(hidden_dim=32)
    actor.eval()
    with torch.no_grad():
        actor.branch_head[-1].bias[0] = 100.0
        actor.branch_head[-1].bias[1] = -100.0
    state_features = torch.zeros(32)
    time_features = torch.zeros(2)

    snapshot = actor.make_decision_snapshot(
        env,
        state_features,
        time_features,
        worker_id=2,
        episode_id=7,
        episode_index=3,
    )
    live_candidates = env.get_action_candidates()
    assert snapshot.candidate_task_keys == tuple(task.task_key for task in live_candidates)
    assert snapshot.branch_masks == tuple(
        env.get_action_branch_mask(task) for task in live_candidates
    )
    with pytest.raises(ValueError, match="图版本"):
        actor.select_snapshot(replace(snapshot, graph_version="stale_graph"))

    for task, context in zip(live_candidates, snapshot.team_contexts, strict=True):
        assert tuple(worker.worker_id for worker in context.workers) == tuple(
            env.state.station_worker_bindings[task.current_station]
        )
        prefixes = [()]
        for prefix_length in range(1, task.demand):
            prefixes.extend(
                tuple(prefix)
                for prefix in permutations(
                    env.state.station_worker_bindings[task.current_station],
                    prefix_length,
                )
            )
        for prefix in prefixes:
            expected = tuple(
                env.valid_team_completion_workers(task, prefix)
            )
            actual = _decision_snapshot_module().team_completion_worker_ids(
                context, prefix
            )
            assert actual == expected

    snapshot_result = actor.select_snapshot(snapshot, deterministic=True)
    live_result = actor.select_action(
        env, state_features, time_features, deterministic=True
    )
    assert snapshot_result[0] == live_result[0]
    assert snapshot_result[1] == pytest.approx(live_result[1], abs=1e-6)
    record = snapshot_result[3]
    assert record["candidate_branch_masks"] == snapshot.branch_masks
    selected_task_index = snapshot.candidate_task_keys.index(
        snapshot_result[0]["task_key"]
    )
    selected_branch = int(snapshot_result[0]["branch"])
    assert snapshot.branch_masks[selected_task_index][selected_branch]
    _, replay_log_probs, _ = actor.evaluate_action_log_probs(
        state_features.unsqueeze(0), time_features.unsqueeze(0), [record]
    )
    assert float(replay_log_probs[0]) == pytest.approx(snapshot_result[1], abs=1e-6)


def test_single_env_worker_matches_direct_environment_for_frozen_actions() -> None:
    from copy import deepcopy

    from envs.work3.core_types import ActionBranch, TaskRuntimeState
    from envs.work3.environment import AirLineEnvWork3
    from models.work3.heuristic_agent import HeuristicAgentWork3
    from scripts.work3.train_ppo_work3 import create_work3_single_env_runtime
    from scripts.work3.evaluate_c_vs_d import _check_completed_trajectory_feasibility
    from utils.work3.trajectory_feasibility import (
        TaskConstraintRecord,
        TrajectoryExecutionRecord,
        validate_trajectory,
    )

    _work3_vector_env_module()
    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    if not baseline_path.is_file():
        pytest.skip(f"固定动作对照实例不存在: {baseline_path}")

    planning_env = AirLineEnvWork3(baseline_json_path=baseline_path)
    planning_env.reset()
    agent = HeuristicAgentWork3(name="SingleEnvParityFixture")
    frozen_actions: list[dict[str, object]] = []
    for _ in range(12000):
        if planning_env._check_terminated():
            break
        action = agent.select_action(planning_env)
        if action is None:
            action = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
        frozen_actions.append(deepcopy(action))
        _obs, _reward, terminated, truncated, info = planning_env.step(action)
        assert not truncated
        assert not (terminated and info.get("termination_reason") == "deadlock")
    assert planning_env._check_terminated()
    assert len(frozen_actions) < 12000
    action_branches = tuple(
        ActionBranch(action.get("branch", ActionBranch.STATION_EXECUTE))
        for action in frozen_actions
    )
    assert ActionBranch.ADVANCE_TO_NEXT_EVENT in action_branches
    assert ActionBranch.STATION_EXECUTE in action_branches

    direct_env = AirLineEnvWork3(baseline_json_path=baseline_path)
    direct_env.reset()
    direct_event_trace: list[tuple[float, str, int, str | None, int]] = []
    direct_start_station_by_task: dict[str, int] = {}
    original_pop = direct_env.event_queue.pop

    def traced_pop() -> object:
        event = original_pop()
        if event is not None:
            direct_event_trace.append(
                (
                    float(event.timestamp),
                    event.event_type.name,
                    int(event.event_id),
                    event.task_key,
                    int(event.generation),
                )
            )
        return event

    direct_env.event_queue.pop = traced_pop  # type: ignore[method-assign]
    original_started = direct_env._on_task_started

    def traced_started(task: TaskRuntimeState, start_time: float) -> None:
        original_started(task, start_time)
        if task.actual_start is not None:
            direct_start_station_by_task[task.task_key] = int(
                direct_env.state.aircraft[task.aircraft_id].current_station
            )

    direct_env._on_task_started = traced_started  # type: ignore[method-assign]
    direct_completed: set[str] = set()
    direct_steps = 0
    worker = create_work3_single_env_runtime(baseline_path)
    try:
        initial_observation = worker.reset(scenario=None, episode_id=21, episode_index=0)
        snapshot = worker.snapshot()
        assert snapshot.current_time == direct_env.state.current_time
        assert snapshot.state_features.device.type == "cpu"
        assert all(value.device.type == "cpu" for value in snapshot.graph_snapshot.x_dict.values())

        for action in frozen_actions:
            trace_start = len(direct_event_trace)
            _obs, expected_reward, expected_terminated, expected_truncated, expected_info = (
                direct_env.step(action)
            )
            expected_trace = tuple(direct_event_trace[trace_start:])
            expected_new_completions = []
            for task in direct_env.state.tasks.values():
                if task.actual_end is None or task.task_key in direct_completed:
                    continue
                direct_completed.add(task.task_key)
                station_id = int(task.last_published_assignment["station"])
                expected_new_completions.append(
                    (
                        task.task_key,
                        int(task.aircraft_id),
                        int(task.task_id),
                        station_id,
                        tuple(task.assigned_team),
                        float(task.actual_start),
                        float(task.actual_end),
                        direct_start_station_by_task[task.task_key],
                    )
                )

            actual = worker.step(action)
            direct_steps += 1
            assert actual.step_count == direct_steps
            assert actual.observation == _obs
            assert actual.raw_reward == pytest.approx(expected_reward, abs=1e-9)
            assert actual.terminated is expected_terminated
            assert actual.truncated is expected_truncated
            assert actual.info == expected_info
            assert actual.processed_events == expected_trace
            actual_new_completions = tuple(
                (
                    item["task_key"],
                    item["aircraft_id"],
                    item["task_id"],
                    item["station_id"],
                    tuple(item["team"]),
                    item["start"],
                    item["end"],
                    item["aircraft_station_at_start"],
                )
                for item in actual.completed_execution_records
            )
            assert actual_new_completions == tuple(expected_new_completions)
            if actual.terminated:
                break

        assert direct_steps == len(frozen_actions)
        assert direct_env._check_terminated()
        audit = worker.close()
        assert not worker.is_alive
    finally:
        if worker.is_alive:
            worker.close()

    assert initial_observation["current_time"] == 0.0
    assert audit["success"] is True
    assert audit["completed_tasks"] == audit["total_tasks"] == len(direct_env.state.tasks)
    assert audit["step_count"] == direct_env.step_count == direct_steps
    assert list(audit["transfer_history"]) == direct_env.state.transfer_history
    assert audit["cost_breakdown"] == {
        "cost_takt": direct_env.cost_takt,
        "cost_time": direct_env.cost_time,
        "cost_team": direct_env.cost_team,
        "cost_postpone": direct_env.cost_postpone,
        "cost_revision": direct_env.cost_revision,
    }

    def independently_check(serialized_audit: dict[str, object]) -> object:
        constraints = {
            int(task_id): TaskConstraintRecord(
                demand=int(value["demand"]),
                required_skill=int(value["required_skill"]),
                predecessors=tuple(value["predecessors"]),
                fixed_station=value["fixed_station"],
                max_allowed_station=value["max_allowed_station"],
            )
            for task_id, value in serialized_audit["task_constraints"].items()
        }
        records = [
            TrajectoryExecutionRecord(
                aircraft_id=int(item["aircraft_id"]),
                task_id=int(item["task_id"]),
                station_id=int(item["station_id"]),
                team=tuple(item["team"]),
                start=float(item["start"]),
                end=float(item["end"]),
                material_ready_time=float(item["material_ready_time"]),
                station_entry_time=item["station_entry_time"],
                aircraft_station_at_start=int(item["aircraft_station_at_start"]),
            )
            for item in serialized_audit["execution_records"]
        ]
        return validate_trajectory(
            records,
            task_constraints=constraints,
            worker_skills=serialized_audit["worker_skills"],
            worker_station_bindings=serialized_audit["worker_station_bindings"],
            station_capacities=serialized_audit["station_capacities"],
        )

    report = independently_check(audit)
    assert report.is_feasible
    assert not report.violations
    direct_feasible, direct_violations = _check_completed_trajectory_feasibility(direct_env)
    assert direct_feasible == report.is_feasible
    assert direct_violations == report.violations
    assert len(direct_completed) == len(direct_env.state.tasks)


def test_worker_scenario_plan_is_stable_for_worker_episode_coordinates() -> None:
    from scripts.work3.train_ppo_work3 import build_worker_scenario_plan

    scenarios = [
        {"scenario_id": f"scenario-{index}", "tau": float(index)}
        for index in range(5)
    ]
    first = build_worker_scenario_plan(
        scenarios,
        num_workers=2,
        episodes_per_worker=3,
        seed=41,
    )
    second = build_worker_scenario_plan(
        scenarios,
        num_workers=2,
        episodes_per_worker=3,
        seed=41,
    )

    assert first == second
    assert [(item["worker_id"], item["episode_index"]) for item in first] == [
        (worker_id, episode_index)
        for episode_index in range(3)
        for worker_id in range(2)
    ]
    assert all(item["scenario"]["scenario_id"] in {
        scenario["scenario_id"] for scenario in scenarios
    } for item in first)


def test_two_spawn_workers_share_one_main_actor_and_enforce_aggregate_step_budget() -> None:
    from envs.work3.core_types import ActionBranch
    from envs.work3.environment import AirLineEnvWork3
    from models.work3.actor_critic import ActorCriticWork3
    from models.work3.heuristic_agent import HeuristicAgentWork3
    from training.work3_vector_env import Work3VectorEnv

    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    planning_env = AirLineEnvWork3(baseline_json_path=baseline_path)
    planning_env.reset()
    agent = HeuristicAgentWork3(name="VectorBudgetFixture")
    frozen_actions: list[dict[str, object]] = []
    for _ in range(32):
        action = agent.select_action(planning_env)
        if action is None:
            action = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
        frozen_actions.append(action)
        planning_env.step(action)
        if action.get("branch") == ActionBranch.ADVANCE_TO_NEXT_EVENT:
            break
    assert frozen_actions[-1]["branch"] == ActionBranch.ADVANCE_TO_NEXT_EVENT

    vector = Work3VectorEnv(
        env_kwargs={"baseline_json_path": str(baseline_path)},
        num_envs=2,
        start_method="spawn",
    )
    actor = ActorCriticWork3(hidden_dim=32)
    try:
        assert len(vector.worker_pids) == 2
        assert len(set(vector.worker_pids)) == 2
        assert all(pid != os.getpid() for pid in vector.worker_pids)
        assert vector.workers_alive == (True, True)
        assert vector.worker_cuda_initialized == (False, False)

        resets = vector.reset_all(
            scenarios=(None, None),
            episode_ids=(11, 12),
            episode_indices=(0, 0),
        )
        assert len(resets) == 2
        snapshots = vector.snapshots()
        assert snapshots[0].current_time == snapshots[1].current_time == 0.0
        assert all(snapshot.graph_snapshot["task"].x.device.type == "cpu" for snapshot in snapshots)
        assert actor.select_snapshot(snapshots[0], deterministic=True)[0] is not None

        observed: list[object] = []
        last_batch = None
        for action in frozen_actions:
            last_batch = vector.step_all(
                actions=(action, action),
                max_total_steps=17,
            )
            observed.extend(result for result in last_batch.results if result is not None)
        assert last_batch is not None
        assert last_batch.results[0] is not None
        assert last_batch.results[1] is None
        assert vector.total_env_steps == 17
        assert vector.budget_reserved_steps == 17
        assert len(observed) == 17
        assert vector.worker_step_counts == (9, 8)
    finally:
        vector.close()
    assert vector.workers_alive == (False, False)


def test_vector_reset_reports_fixed_scenario_and_actual_hit_status() -> None:
    from scripts.work3.train_ppo_work3 import build_worker_scenario_plan
    from training.work3_vector_env import Work3VectorEnv

    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    scenarios = [
        {
            "scenario_id": "VECTOR_T0_HIT",
            "tau": 0.0,
            "recovery_time": 1.0,
            "affected_task_keys": ["0_15"],
        },
        {
            "scenario_id": "VECTOR_NOT_YET_TRIGGERED",
            "tau": 1000.0,
            "recovery_time": 1001.0,
            "affected_task_keys": ["missing-task"],
        },
    ]
    plan = build_worker_scenario_plan(
        scenarios,
        num_workers=2,
        episodes_per_worker=1,
        seed=41,
    )
    with Work3VectorEnv(
        env_kwargs={"baseline_json_path": str(baseline_path)},
        num_envs=2,
        start_method="spawn",
    ) as vector:
        resets = vector.reset_from_plan(plan)

    for reset, planned in zip(resets, plan, strict=True):
        status = reset.scenario_status
        assert status["scenario_id"] == planned["scenario"]["scenario_id"]
        assert status["started"] is True
        assert status["scheduled_target_count"] == 1
        assert status["in_flight"] is True
        if status["scenario_id"] == "VECTOR_T0_HIT":
            assert status["event_triggered"] is True
            assert status["actual_hit_task_keys"] == ("0_15",)
            assert status["actual_hit_count"] == 1
        else:
            assert status["event_triggered"] is False
            assert status["actual_hit_count"] == 0
            assert status["unhit_reasons"] == {"missing-task": "event_not_triggered"}


def test_vector_worker_errors_and_incomplete_step_timeouts_are_explicit_and_cleaned() -> None:
    from envs.work3.core_types import ActionBranch
    from models.work3.actor_critic import ActorCriticWork3
    from training.work3_vector_env import Work3VectorEnv

    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    failed_vector = Work3VectorEnv(
        env_kwargs={"baseline_json_path": str(baseline_path)},
        num_envs=2,
        start_method="spawn",
    )
    try:
        failed_vector.reset_all(
            scenarios=(None, None),
            episode_ids=(30, 31),
            episode_indices=(0, 0),
        )
        failed_batch = failed_vector.step_all(
            actions=(
                {"branch": ActionBranch.STATION_EXECUTE},
                {"branch": ActionBranch.STATION_EXECUTE},
            ),
            max_total_steps=2,
        )
        assert failed_batch.worker_errors
        assert all("step" in message for _, message in failed_batch.worker_errors)
    finally:
        failed_vector.close()
    assert failed_vector.workers_alive == (False, False)

    timed_vector = Work3VectorEnv(
        env_kwargs={"baseline_json_path": str(baseline_path)},
        num_envs=2,
        start_method="spawn",
    )
    try:
        timed_vector.reset_all(
            scenarios=(None, None),
            episode_ids=(40, 41),
            episode_indices=(0, 0),
        )
        snapshots = timed_vector.snapshots()
        actor = ActorCriticWork3(hidden_dim=32)
        actions = tuple(
            actor.select_snapshot(snapshot, deterministic=True)[0]
            for snapshot in snapshots
        )
        assert all(action is not None for action in actions)
        batch = timed_vector.step_all(
            actions=tuple(action for action in actions if action is not None),
            max_total_steps=2,
            settle_timeout_seconds=1e-9,
        )
        assert batch.interrupted_worker_ids
        assert all(batch.results[index] is None for index in batch.interrupted_worker_ids)
        assert batch.total_env_steps <= 2
        assert batch.budget_reserved_steps == len(batch.dispatched_worker_ids)
    finally:
        timed_vector.close()
    assert timed_vector.workers_alive == (False, False)


def test_vector_expired_wall_clock_deadline_stops_dispatch_without_fake_result() -> None:
    import time

    from envs.work3.core_types import ActionBranch
    from training.work3_vector_env import Work3VectorEnv

    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    vector = Work3VectorEnv(
        env_kwargs={"baseline_json_path": str(baseline_path)},
        num_envs=1,
        start_method="spawn",
    )
    try:
        vector.reset_all(
            scenarios=(None,),
            episode_ids=(50,),
            episode_indices=(0,),
        )
        deadline = time.monotonic() - 1.0
        stopped = vector.step_all(
            actions=({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT},),
            max_total_steps=2,
            wall_clock_deadline=deadline,
        )
        assert stopped.wall_clock_expired is True
        assert stopped.dispatched_worker_ids == ()
        assert stopped.results == (None,)
        assert stopped.total_env_steps == 0
    finally:
        vector.close()
    assert vector.workers_alive == (False,)
