"""R05 图状态的预约、发布安排、未来日历与约束掩码反例。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import pytest
import torch

from envs.work3.core_types import ActionBranch, TaskStatus, TimeInterval
from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import (
    ActorCriticWork3,
    extract_compact_state_features,
)
from models.work3.graph_builder import (
    GRAPH_FEATURE_VERSION,
    MultiAircraftGraphBuilder,
)
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from models.work3.time_head import TimeResidualHead
from models.work3.ppo_trainer import PPOTrainerWork3
from scripts.work3.evaluate_c_vs_d import build_formal_evaluation_agent
from scripts.work3.train_ppo_work3 import compute_online_time_inputs
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


@pytest.fixture
def baseline_path() -> Path:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return path


def _make_env(baseline_path: Path) -> AirLineEnvWork3:
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    env.reset()
    env.state.current_time = 10.0
    return env


def _write_relative_feature_baseline(
    source: MultiAircraftBaseline,
    num_aircraft: int,
    path: Path,
) -> Path:
    tasks = {}
    for aircraft_id in range(num_aircraft):
        for station_id, task_id in ((1, 6), (1, 15), (2, 18), (3, 12), (4, 20), (5, 24)):
            template = source.get_task(0, task_id)
            nominal_entry = (aircraft_id + station_id - 1) * source.h0
            key = f"{aircraft_id}_{task_id}"
            tasks[key] = replace(
                template,
                aircraft_id=aircraft_id,
                task_key=key,
                cycle_idx=aircraft_id + station_id,
                baseline_start=nominal_entry + template.in_station_offset,
                baseline_end=(
                    nominal_entry + template.in_station_offset + template.duration
                ),
                nominal_station_entry=nominal_entry,
                nominal_station_exit=nominal_entry + source.h0,
            )
    MultiAircraftBaseline(
        num_aircraft=num_aircraft,
        num_stations=5,
        h0=source.h0,
        tasks=tasks,
        station_workers=source.station_workers,
    ).save_to_json(path)
    return path


def _set_relative_feature_snapshot(
    env: AirLineEnvWork3,
    current_aircraft_id: int,
    next_aircraft_id: int,
    cycle_id: int,
) -> None:
    state = env.state
    for aircraft_id, aircraft in state.aircraft.items():
        station = cycle_id - aircraft_id - 1
        aircraft.current_station = (
            5 if station >= state.num_stations
            else station if station >= 0
            else -1
        )
    state.current_cycle = cycle_id
    state.current_time = 100.0
    state.last_transfer_time = 100.0
    for task in state.tasks.values():
        aircraft_station = state.aircraft[task.aircraft_id].current_station
        if aircraft_station == 5 or (
            aircraft_station >= 0 and task.current_station < aircraft_station
        ):
            task.status = TaskStatus.COMPLETED
        elif task.aircraft_id == current_aircraft_id:
            task.status = (
                TaskStatus.READY if task.task_id == 18 else TaskStatus.COMPLETED
            )
        else:
            task.status = TaskStatus.UNREADY


def _reservable_task(env: AirLineEnvWork3):
    return next(
        task
        for task in env.get_action_candidates()
        if env.can_reserve(task) and task.current_station < env.state.num_stations - 1
    )


def _set_future_reservation(
    env: AirLineEnvWork3,
    task_key: str,
    start: float,
) -> None:
    task = env.state.tasks[task_key]
    team = tuple(task.base_team)
    assert len(team) == task.demand
    task.status = TaskStatus.RESERVED
    task.assigned_team = list(team)
    task.scheduled_start = float(start)
    task.execution_duration = 2.0
    for worker_id in team:
        calendar = env.state.workers[worker_id]
        calendar.intervals = [
            interval for interval in calendar.intervals
            if interval.task_key != task_key
        ]
        calendar.intervals.append(TimeInterval(start, start + 2.0, task_key))
        calendar.intervals.sort(key=lambda interval: interval.start)


def _assert_graph_equal(left, right) -> None:
    assert set(left.node_types) == set(right.node_types)
    assert set(left.edge_types) == set(right.edge_types)
    for node_type in left.node_types:
        assert torch.equal(left[node_type].x, right[node_type].x)
    for edge_type in left.edge_types:
        assert torch.equal(left[edge_type].edge_index, right[edge_type].edge_index)


def test_future_reservation_times_and_replay_snapshot_are_distinguishable(
    baseline_path: Path,
) -> None:
    env = _make_env(baseline_path)
    task = _reservable_task(env)
    _set_future_reservation(env, task.task_key, 20.0)

    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    first_graph = actor.build_graph_snapshot(env)
    task_index = actor._get_graph_builder(env).task_key_to_idx[task.task_key]
    first_task_features = first_graph["task"].x[task_index]

    # 新图契约：18为is_reserved，19为has_schedule，20为开工剩余时间/H0，
    # 21为has_schedule_end，22为预计完工剩余时间/H0。
    assert first_task_features.shape == (26,)
    assert first_task_features[18:23].tolist() == pytest.approx(
        [1.0, 1.0, 10.0 / env.state.h0, 1.0, 12.0 / env.state.h0]
    )
    worker_index = actor._get_graph_builder(env).worker_id_to_idx[task.assigned_team[0]]
    worker_features = first_graph["worker"].x[worker_index]
    assert worker_features.shape == (21,)
    assert worker_features[18].item() == 1.0
    assert worker_features[19:21].tolist() == pytest.approx(
        [10.0 / env.state.h0, 12.0 / env.state.h0]
    )

    cmax = 100.0
    state_feat = extract_compact_state_features(env.state, cmax)
    urgency = torch.zeros(2)
    _, old_log_prob, _, record = actor.select_action(
        env,
        state_feat,
        urgency,
        deterministic=True,
    )
    _set_future_reservation(env, task.task_key, 25.0)
    second_graph = actor.build_graph_snapshot(env)
    second_task_features = second_graph["task"].x[task_index]
    assert second_task_features[20].item() == pytest.approx(15.0 / env.state.h0)
    assert not torch.equal(first_task_features, second_task_features)

    _, replay_log_prob, _ = actor.evaluate_action_log_probs(
        state_feat.unsqueeze(0),
        urgency.unsqueeze(0),
        [record],
    )
    assert replay_log_prob.item() == pytest.approx(old_log_prob, abs=1e-6)


def test_unlaunched_aircraft_load_changes_future_not_current_station_feature(
    baseline_path: Path,
) -> None:
    """隔离当前/未来站位工作量，并验证启发式只看本周期任务。"""
    env = _make_env(baseline_path)
    current_aircraft_id = env.state.get_aircraft_at_station(0)
    unlaunched_aircraft_id = next(
        aircraft_id
        for aircraft_id, aircraft in env.state.aircraft.items()
        if aircraft.current_station == -1
    )
    current_task = next(
        task
        for task in env.state.tasks.values()
        if task.aircraft_id == current_aircraft_id
        and task.current_station == 0
        and task.status != TaskStatus.COMPLETED
    )
    future_task = next(
        task
        for task in env.state.tasks.values()
        if task.aircraft_id == unlaunched_aircraft_id
        and task.current_station == 0
        and task.status != TaskStatus.COMPLETED
    )
    for task in env.state.tasks.values():
        if task.current_station == 0 and task.task_key not in {
            current_task.task_key,
            future_task.task_key,
        }:
            task.status = TaskStatus.COMPLETED
    current_task.status = TaskStatus.READY
    current_task.duration = 2.0
    current_task.execution_duration = None
    future_task.status = TaskStatus.UNREADY
    future_task.duration = 4.0
    future_task.execution_duration = None

    assert current_aircraft_id is not None
    assert env.state.get_aircraft_at_station(0) == current_aircraft_id
    assert env.state.aircraft[unlaunched_aircraft_id].current_station == -1

    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    graph_before = actor.build_graph_snapshot(env)
    station_index = future_task.current_station
    remaining_load_before = graph_before["station"].x[station_index, 0].item()
    current_load_before = graph_before["station"].x[station_index, 4].item()
    future_load_before = graph_before["station"].x[station_index, 5].item()
    compact_before = extract_compact_state_features(env.state, 100.0)
    heuristic_before = compute_cycle_heuristic_cmax(env.state)
    assert current_load_before == pytest.approx(2.0 / env.state.h0)
    assert future_load_before == pytest.approx(4.0 / env.state.h0)
    assert remaining_load_before == pytest.approx(6.0 / env.state.h0)
    assert compact_before[1 + station_index].item() == pytest.approx(
        2.0 / env.state.h0
    )

    future_task.duration = 9.0

    graph_after = actor.build_graph_snapshot(env)
    compact_after = extract_compact_state_features(env.state, 100.0)

    assert graph_after["station"].x[station_index, 4].item() == pytest.approx(
        current_load_before
    )
    assert graph_after["station"].x[station_index, 5].item() == pytest.approx(
        9.0 / env.state.h0
    )
    assert graph_after["station"].x[station_index, 0].item() == pytest.approx(
        11.0 / env.state.h0
    )
    assert compact_after[1 + station_index].item() == pytest.approx(
        compact_before[1 + station_index].item()
    )
    assert compute_cycle_heuristic_cmax(env.state) == pytest.approx(heuristic_before)
    assert torch.isfinite(compact_after).all()


def test_empty_station_has_no_current_period_work_or_delay_features(
    baseline_path: Path,
) -> None:
    env = _make_env(baseline_path)
    empty_station = next(
        station
        for station in range(env.state.num_stations)
        if env.state.get_aircraft_at_station(station) is None
    )

    features = extract_compact_state_features(env.state, 100.0)
    graph = ActorCriticWork3(
        state_dim=32, task_feat_dim=8, hidden_dim=16
    ).build_graph_snapshot(env)

    assert features[1 + empty_station].item() == 0.0
    assert features[6 + empty_station].item() == 0.0
    assert features[11 + empty_station].item() == 0.0
    assert features[16 + empty_station].item() == 0.0
    assert graph["station"].x[empty_station, 4].item() == 0.0


def test_running_task_features_use_remaining_execution_time(
    baseline_path: Path,
) -> None:
    env = _make_env(baseline_path)
    aircraft_id = env.state.get_aircraft_at_station(0)
    running_task = next(
        task
        for task in env.state.tasks.values()
        if task.aircraft_id == aircraft_id
        and task.current_station == 0
        and task.status != TaskStatus.COMPLETED
    )
    for task in env.state.tasks.values():
        if (
            task.aircraft_id == aircraft_id
            and task.current_station == 0
            and task.task_key != running_task.task_key
        ):
            task.status = TaskStatus.COMPLETED
    running_task.status = TaskStatus.RUNNING
    running_task.duration = 5.0
    running_task.execution_duration = 5.0
    running_task.actual_start = env.state.current_time - 2.0

    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    features = extract_compact_state_features(env.state, 100.0)
    graph = actor.build_graph_snapshot(env)

    assert features[1].item() == pytest.approx(3.0 / env.state.h0)
    assert graph["station"].x[0, 4].item() == pytest.approx(
        3.0 / env.state.h0
    )

    running_task.status = TaskStatus.RESERVED
    running_task.actual_start = None
    running_task.duration = 2.0
    running_task.execution_duration = 3.0
    running_task.scheduled_start = env.state.current_time + 10.0
    reserved_features = extract_compact_state_features(env.state, 100.0)
    reserved_graph = actor.build_graph_snapshot(env)
    assert reserved_features[1].item() == pytest.approx(3.0 / env.state.h0)
    assert reserved_graph["station"].x[0, 4].item() == pytest.approx(
        3.0 / env.state.h0
    )

    running_task.status = TaskStatus.RUNNING
    running_task.execution_duration = 5.0
    running_task.actual_start = env.state.current_time - 2.0
    env.state.current_time = running_task.actual_start + running_task.execution_duration + 1.0
    finished_features = extract_compact_state_features(env.state, 100.0)
    finished_graph = actor.build_graph_snapshot(env)
    assert finished_features[1].item() == 0.0
    assert finished_graph["station"].x[0, 4].item() == 0.0


def test_relative_aircraft_features_are_batch_and_cycle_invariant(
    baseline_path: Path,
    tmp_path: Path,
) -> None:
    source = MultiAircraftBaseline.load_from_json(str(baseline_path))
    snapshots = {}
    for num_aircraft, current_id, next_id, cycle_id in (
        (7, 3, 4, 5),
        (10, 3, 4, 5),
        (14, 7, 8, 9),
    ):
        path = _write_relative_feature_baseline(
            source,
            num_aircraft,
            tmp_path / f"relative_features_k{num_aircraft}.json",
        )
        env = AirLineEnvWork3(baseline_json_path=path)
        env.reset()
        _set_relative_feature_snapshot(env, current_id, next_id, cycle_id)
        baseline = MultiAircraftBaseline.load_from_json(path)
        builder = MultiAircraftGraphBuilder(baseline)
        graph = builder.build_graph(env)
        compact = extract_compact_state_features(env.state, 100.0)
        current_key = f"{current_id}_18"
        next_key = f"{next_id}_18"
        future_key = f"{next_id + 1}_18"
        task_index = builder.task_key_to_idx
        snapshots[num_aircraft] = (
            compact[21:26],
            graph["station"].x[:, 1],
            graph["task"].x[task_index[current_key]],
            graph["task"].x[task_index[next_key]],
            graph["task"].x[task_index[future_key]],
        )

    for num_aircraft in (7, 10, 14):
        compact, stations, current_task, next_task, future_task = snapshots[num_aircraft]
        assert torch.isfinite(compact).all()
        assert torch.isfinite(stations).all()
        assert torch.isfinite(current_task).all()
        assert torch.isfinite(next_task).all()
        assert torch.isfinite(future_task).all()
        assert current_task[11].item() == pytest.approx(-1.0 / 5.0)
        assert next_task[11].item() == pytest.approx(0.0)
        assert future_task[11].item() == pytest.approx(1.0 / 5.0)
        assert current_task[15].item() == pytest.approx(0.0)
        assert next_task[15].item() == pytest.approx(1.0 / 5.0)
        assert future_task[15].item() == pytest.approx(2.0 / 5.0)
        assert compact.tolist() == pytest.approx([1, 1, 1, 1, 1])
        assert stations.tolist() == pytest.approx([1, 1, 1, 1, 1])

    for index in range(5):
        assert torch.allclose(snapshots[7][index], snapshots[10][index])
        assert torch.allclose(snapshots[10][index], snapshots[14][index])

    replay_path = _write_relative_feature_baseline(
        source, 7, tmp_path / "relative_replay_k7.json"
    )
    replay_env = AirLineEnvWork3(baseline_json_path=replay_path)
    replay_env.reset()
    for task in replay_env.state.tasks.values():
        if task.aircraft_id == 0 and task.current_station == 0 and task.task_id != 6:
            task.status = TaskStatus.COMPLETED
    replay_env.state.tasks["0_6"].status = TaskStatus.READY
    assert replay_env.get_action_candidates() == [replay_env.state.tasks["0_6"]]
    assert replay_env.get_action_branch_mask(replay_env.state.tasks["0_6"]) == (
        True,
        True,
    )
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    actor.eval()
    state_features = extract_compact_state_features(replay_env.state, 100.0)
    assert state_features[21:26].tolist() == pytest.approx([1, 0, 0, 0, 0])
    time_features = torch.zeros(2)
    for branch, bias in ((0, (100.0, -100.0)), (1, (-100.0, 100.0))):
        with torch.no_grad():
            actor.branch_head[-1].bias.copy_(torch.tensor(bias))
        action, sampled_log_prob, _, record = actor.select_action(
            replay_env,
            state_features,
            time_features,
            deterministic=True,
        )
        assert action is not None
        assert int(record["branch"]) == branch
        assert record["graph_version"] == GRAPH_FEATURE_VERSION
        assert record["graph_snapshot"]["station"].x[:, 1].tolist() == pytest.approx(
            [1, 0, 0, 0, 0]
        )
        _, replay_log_prob, _ = actor.evaluate_action_log_probs(
            state_features.unsqueeze(0),
            time_features.unsqueeze(0),
            [record],
        )
        assert replay_log_prob.item() == pytest.approx(sampled_log_prob, abs=1e-6)


def test_seven_aircraft_small_full_line_trace_keeps_loads_feasible(
    baseline_path: Path,
    tmp_path: Path,
) -> None:
    """七架同模板飞机填线、经历扰动并排空，特征与独立账本保持有效。"""
    from scripts.work3.evaluate_c_vs_d import _check_completed_trajectory_feasibility
    from scripts.work3.train_ppo_work3 import count_actual_scenario_hits
    from utils.work3.objective_evaluator import evaluate_trajectory_objective

    source = MultiAircraftBaseline.load_from_json(str(baseline_path))
    station_task_ids = (
        (1, 15), (1, 6), (1, 16),
        (2, 18), (2, 8), (2, 9), (2, 26),
        (3, 12), (3, 13), (3, 29),
        (4, 20), (4, 21), (4, 10),
        (5, 24), (5, 23), (5, 30), (5, 34),
    )
    selected_ids = {task_id for _, task_id in station_task_ids}
    tasks = {}
    for aircraft_id in range(7):
        for station_id, task_id in station_task_ids:
            template = source.get_task(0, task_id)
            assert set(template.predecessors) <= selected_ids
            nominal_entry = (aircraft_id + station_id - 1) * source.h0
            key = f"{aircraft_id}_{task_id}"
            tasks[key] = replace(
                template,
                aircraft_id=aircraft_id,
                task_key=key,
                cycle_idx=aircraft_id + station_id,
                baseline_start=nominal_entry + template.in_station_offset,
                baseline_end=(
                    nominal_entry + template.in_station_offset + template.duration
                ),
                nominal_station_entry=nominal_entry,
                nominal_station_exit=nominal_entry + source.h0,
            )

    baseline = MultiAircraftBaseline(
        num_aircraft=7,
        num_stations=5,
        h0=source.h0,
        tasks=tasks,
        station_workers=source.station_workers,
    )
    small_path = tmp_path / "seven_aircraft_five_station.json"
    baseline.save_to_json(small_path)
    env = AirLineEnvWork3(baseline_json_path=small_path)
    env.reset()
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    builder = actor._get_graph_builder(env)
    assert env.can_postpone(env.state.tasks["0_6"])
    assert not env.can_postpone(env.state.tasks["0_34"])

    initial_features = extract_compact_state_features(env.state, 100.0)
    initial_graph = builder.build_graph(env)
    assert torch.isfinite(initial_features).all()
    assert torch.isfinite(initial_graph["station"].x).all()
    assert initial_features[1].item() == pytest.approx(
        initial_graph["station"].x[0, 4].item()
    )
    assert initial_graph["station"].x[0, 5].item() > 0.0

    def run_fixed_action() -> tuple[bool, dict[str, object]]:
        ready_tasks = env.get_ready_tasks()
        if ready_tasks:
            task = ready_tasks[0]
            original = baseline.get_task(task.aircraft_id, task.task_id)
            _, _, terminated, truncated, info = env.step(
                {
                    "task_key": task.task_key,
                    "branch": ActionBranch.STATION_EXECUTE,
                    "team": original.team,
                    "align": 0,
                }
            )
        else:
            _, _, terminated, truncated, info = env.step(
                {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
            )
        assert not truncated
        return terminated, info

    decisions = 0
    while len(env.state.transfer_history) < 6:
        terminated, _ = run_fixed_action()
        decisions += 1
        assert not terminated
        assert decisions < 1000

    assert [
        sum(aircraft.current_station == station for aircraft in env.state.aircraft.values())
        for station in range(5)
    ] == [1] * 5
    target = env.state.tasks["4_12"]
    assert env.state.aircraft[4].current_station == 2
    assert target.status == TaskStatus.READY
    assert target.actual_start is None
    tau = env.state.current_time
    scenario = {
        "scenario_id": "P0_SEVEN_AIRCRAFT_FIXED_HIT",
        "tau": tau,
        "recovery_time": tau + 0.15 * env.state.h0,
        "affected_task_keys": [target.task_key],
    }
    env.load_scenario(scenario)
    assert count_actual_scenario_hits(env, scenario) == 1
    disturbed_features = extract_compact_state_features(env.state, 100.0)
    disturbed_graph = builder.build_graph(env)
    assert torch.isfinite(disturbed_features).all()
    assert torch.isfinite(disturbed_graph["station"].x).all()

    last_info: dict[str, object] = {}
    while not env._check_terminated():
        terminated, last_info = run_fixed_action()
        decisions += 1
        assert decisions < 4000
        if terminated:
            break

    assert last_info.get("success") is True
    assert len(env.state.tasks) == 7 * len(station_task_ids)
    assert all(task.status == TaskStatus.COMPLETED for task in env.state.tasks.values())
    assert len(env.state.transfer_history) == 7 + 5 - 1
    assert target.actual_start is not None
    assert target.actual_start >= scenario["recovery_time"] - env.tolerance
    assert count_actual_scenario_hits(env, scenario) == 1
    feasible, violations = _check_completed_trajectory_feasibility(env)
    assert feasible, f"独立轨迹检查失败：{violations}"
    ledger = evaluate_trajectory_objective(env, weights=env.weights)
    assert sum(env.step_rewards) == pytest.approx(-ledger.j_total, abs=1e-8)


def test_previous_published_team_is_separate_from_p0_and_changes_worker_input(
    baseline_path: Path,
) -> None:
    env = _make_env(baseline_path)
    task = next(
        t
        for t in env.state.tasks.values()
        if 0 < t.demand < len(env.state.station_worker_bindings[t.current_station])
    )
    team_a = tuple(task.base_team)
    station_workers = env.state.station_worker_bindings[task.current_station]
    extra_worker = next(w for w in station_workers if w not in team_a)
    env.worker_skills[extra_worker] = frozenset((*env.worker_skills[extra_worker], task.skill))
    team_b = (extra_worker, *team_a[1:])
    assert _team_is_valid(env, task, team_b)
    published_a = deepcopy(task.last_published_assignment)
    published_b = deepcopy(task.last_published_assignment)
    published_a["team"] = list(team_a)
    published_b["team"] = list(team_b)
    task.last_published_assignment = published_a
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    graph_a = actor.build_graph_snapshot(env)

    task.last_published_assignment = published_b
    graph_b = actor.build_graph_snapshot(env)
    task_index = actor._get_graph_builder(env).task_key_to_idx[task.task_key]

    assert torch.equal(
        graph_a["task", "baseline_team", "worker"].edge_index,
        graph_b["task", "baseline_team", "worker"].edge_index,
    )
    assert not torch.equal(
        graph_a["task", "last_published_team", "worker"].edge_index,
        graph_b["task", "last_published_team", "worker"].edge_index,
    )
    assert graph_a["task"].x[task_index, 23:26].tolist() == pytest.approx(
        graph_b["task"].x[task_index, 23:26].tolist()
    )

    action_after_a = {**published_a, "team": list(team_a)}
    change_a = env._revision_cost_components(published_a, action_after_a, task)[1]
    change_b = env._revision_cost_components(published_b, action_after_a, task)[1]
    assert change_a == pytest.approx(0.0)
    assert change_b > 0.0

    _, _, worker_nodes_a = actor.graph_encoder(graph_a)
    _, _, worker_nodes_b = actor.graph_encoder(graph_b)
    assert not torch.equal(worker_nodes_a, worker_nodes_b)


def _team_is_valid(env: AirLineEnvWork3, task, team: tuple[int, ...]) -> bool:
    try:
        env._validate_team_for_task(task, team)
    except ValueError:
        return False
    return True


def test_unrevealed_future_disturbance_does_not_change_graph_or_time_inputs(
    baseline_path: Path,
) -> None:
    env_a = _make_env(baseline_path)
    env_b = _make_env(baseline_path)
    task_key = _reservable_task(env_a).task_key
    env_b.state.tasks[task_key].task_key
    env_a.load_scenario({
        "scenario_id": "UNREVEALED_A",
        "tau": 30.0,
        "recovery_time": 50.0,
        "affected_task_keys": [task_key],
    })
    env_b.load_scenario({
        "scenario_id": "UNREVEALED_B",
        "tau": 40.0,
        "recovery_time": 80.0,
        "affected_task_keys": [task_key],
    })

    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    time_head = TimeResidualHead(in_dim=16, hidden_dim=16)
    actor.eval()
    time_head.eval()
    cmax_a = 100.0
    cmax_b = 100.0
    state_a = extract_compact_state_features(env_a.state, cmax_a)
    state_b = extract_compact_state_features(env_b.state, cmax_b)
    graph_a, urgency_a, prediction_a = compute_online_time_inputs(
        actor, time_head, env_a, state_a, cmax_a
    )
    graph_b, urgency_b, prediction_b = compute_online_time_inputs(
        actor, time_head, env_b, state_b, cmax_b
    )

    _assert_graph_equal(graph_a, graph_b)
    assert torch.equal(state_a, state_b)
    assert torch.equal(urgency_a, urgency_b)
    assert torch.equal(prediction_a, prediction_b)
    candidates_a = env_a.get_action_candidates()
    candidates_b = env_b.get_action_candidates()
    assert [task.task_key for task in candidates_a] == [
        task.task_key for task in candidates_b
    ]
    assert [env_a.get_action_branch_mask(task) for task in candidates_a] == [
        env_b.get_action_branch_mask(task) for task in candidates_b
    ]
    action_a, log_prob_a, _, _ = actor.select_action(
        env_a, state_a, urgency_a, deterministic=True
    )
    action_b, log_prob_b, _, _ = actor.select_action(
        env_b, state_b, urgency_b, deterministic=True
    )
    assert action_a == action_b
    assert float(log_prob_a) == pytest.approx(float(log_prob_b), abs=1e-7)


def test_graph_action_helpers_match_environment_for_physical_masks(
    baseline_path: Path,
) -> None:
    env = _make_env(baseline_path)
    baseline = MultiAircraftBaseline.load_from_json(str(baseline_path))
    builder = MultiAircraftGraphBuilder(baseline)

    fixed_task = _reservable_task(env)
    fixed_task.fixed_station = fixed_task.current_station
    assert env.get_action_branch_mask(fixed_task) == (True, False)
    fixed_idx = builder.task_key_to_idx[fixed_task.task_key]
    assert builder.get_action_branch_mask(env, fixed_idx) == {
        "can_stay": True,
        "can_postpone": False,
    }

    task = next(
        task
        for task in env.get_action_candidates()
        if env.can_reserve(task)
        and any(
            env.state.tasks[f"{task.aircraft_id}_{successor_id}"].current_station
            == task.current_station
            and env.state.tasks[f"{task.aircraft_id}_{successor_id}"].status
            not in (TaskStatus.COMPLETED, TaskStatus.POSTPONED)
            for successor_id in env._successors_map[task.aircraft_id][task.task_id]
        )
    )
    assert env.get_action_branch_mask(task) == (True, False)
    task_idx = builder.task_key_to_idx[task.task_key]
    assert builder.get_action_branch_mask(env, task_idx) == {
        "can_stay": True,
        "can_postpone": False,
    }

    revised_task = next(
        task for task in env.get_action_candidates() if env.can_reserve(task)
    )
    _set_future_reservation(env, revised_task.task_key, 20.0)
    assert revised_task.status == TaskStatus.RESERVED
    reserve_mask = env.get_action_branch_mask(revised_task)
    revised_idx = builder.task_key_to_idx[revised_task.task_key]
    assert builder.get_action_branch_mask(env, revised_idx) == {
        "can_stay": reserve_mask[0],
        "can_postpone": reserve_mask[1],
    }
    assert builder.get_action_candidate_indices(env) == [
        builder.task_key_to_idx[item.task_key]
        for item in env.get_action_candidates()
    ]


def test_checkpoints_declare_graph_feature_schema_and_reject_old_schema(
    tmp_path: Path,
) -> None:
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    head = TimeResidualHead(in_dim=64, hidden_dim=64)
    trainer = PPOTrainerWork3(actor_critic=actor, time_head=head)
    checkpoint_path = tmp_path / "current.pt"
    trainer.save_checkpoint(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert GRAPH_FEATURE_VERSION == "work3_graph_v4"
    assert checkpoint["graph_feature_version"] == GRAPH_FEATURE_VERSION
    assert checkpoint["graph_feature_dims"] == {
        "task": 26,
        "worker": 21,
        "station": 15,
        "skill": 11,
    }
    assert "task_features" in checkpoint["graph_feature_schema"]
    assert "worker_features" in checkpoint["graph_feature_schema"]
    assert checkpoint["graph_feature_schema"]["station_workload_semantics"] == (
        "total_remaining_processing_work_norm",
        "current_aircraft_remaining_processing_work_norm",
        "known_future_remaining_processing_work_norm",
    )
    assert checkpoint["graph_feature_schema"]["task_features"][11] == (
        "launch_pulse_offset_norm"
    )
    assert checkpoint["graph_feature_schema"]["task_features"][15] == (
        "transfers_remaining_to_target_station_norm"
    )
    assert checkpoint["graph_feature_schema"]["station_occupancy_semantics"] == (
        "binary_presence_indicator_0_or_1"
    )

    old_path = tmp_path / "legacy_graph.pt"
    old_checkpoint = {
        "actor_critic_state": actor.state_dict(),
        "time_head_state": head.state_dict(),
        "time_head_model_version": "signed_residual_v1",
        "graph_feature_version": "work3_graph_v3",
        "graph_feature_dims": {"task": 26, "worker": 21, "station": 15, "skill": 11},
    }
    torch.save(old_checkpoint, old_path)
    with pytest.raises(ValueError, match="图特征"):
        build_formal_evaluation_agent("D", old_path, device="cpu")
