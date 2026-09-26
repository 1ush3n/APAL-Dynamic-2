"""M01：五站小实例固定动作的手算轨迹验收。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline
from utils.work3.objective_evaluator import (
    ObjectiveBreakdown,
    ObjectiveWeights,
    evaluate_trajectory_objective,
)
from utils.work3.trajectory_feasibility import (
    TaskConstraintRecord,
    TrajectoryExecutionRecord,
    TrajectoryFeasibilityReport,
    validate_trajectory,
)


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")
FIXED_TASKS = ((1, 15), (2, 18), (3, 12), (4, 20), (5, 24))
FIXED_TEAMS = {
    "0_15": (72, 12),
    "0_18": (18, 65),
    "0_12": (50, 47),
    "0_20": (55, 6),
    "0_24": (28, 1),
}

# 由 p_i * demand / (0.95 * sum(worker_efficiency)) 独立预先核算；
# 两人团队的协同系数为0.95。五道工序各自占据一站，顺序执行。
EXPECTED_TASK_DURATIONS = (
    0.5457256030603916,
    0.5574953738564286,
    1.5009998684001387,
    0.9876964746422382,
    1.2146186360338305,
)


def _small_five_station_baseline(
    path: Path,
    *,
    time_scale: float = 1.0,
    time_shift: float = 0.0,
) -> None:
    source = MultiAircraftBaseline.load_from_json(BASELINE_PATH)
    tasks = {}
    for station_id, task_id in FIXED_TASKS:
        task = source.get_task(0, task_id)
        assert task.station_id == station_id
        assert not task.predecessors
        tasks[task.task_key] = replace(
            task,
            duration=task.duration * time_scale,
            in_station_offset=0.0,
            baseline_start=time_shift,
            baseline_end=time_shift + task.duration * time_scale,
            nominal_station_entry=(
                task.nominal_station_entry * time_scale + time_shift
            ),
            nominal_station_exit=(
                task.nominal_station_exit * time_scale + time_shift
            ),
        )

    MultiAircraftBaseline(
        num_aircraft=1,
        num_stations=5,
        h0=source.h0 * time_scale,
        tasks=tasks,
        station_workers=source.station_workers,
    ).save_to_json(path)


def _run_scaled_disturbed_batch(
    baseline_path: Path,
    *,
    time_scale: float,
    time_shift: float = 0.0,
) -> tuple[
    AirLineEnvWork3,
    list[float],
    list[tuple[tuple[str, ...], tuple[bool, bool], tuple[int, ...]]],
    ObjectiveBreakdown,
    TrajectoryFeasibilityReport,
]:
    _small_five_station_baseline(
        baseline_path,
        time_scale=time_scale,
        time_shift=time_shift,
    )
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()
    # 控制测试实例的实际标准工时；环境初始化会从唯一权威CSV恢复原始值。
    for task in env.state.tasks.values():
        assert task.standard_duration is not None
        task.standard_duration *= time_scale
    if time_shift:
        env.state.current_time = time_shift
        env.state.last_transfer_time = time_shift
        env.event_queue.reset(start_time=time_shift)
        for task in env.state.tasks.values():
            task.material_ready_time += time_shift
        for aircraft in env.state.aircraft.values():
            aircraft.entry_times = {
                station_id: time + time_shift
                for station_id, time in aircraft.entry_times.items()
            }
            aircraft.exit_times = {
                station_id: time + time_shift
                for station_id, time in aircraft.exit_times.items()
            }
    recovery_time = time_shift + 400.0 * time_scale
    scenario = {
        "scenario_id": "FIXED_DISTURBANCE_EVENT",
        "tau": time_shift,
        "delta": 400.0 * time_scale,
        "recovery_time": recovery_time,
        "affected_task_keys": ["0_15"],
    }
    env.load_scenario(scenario)
    assert env.state.tasks["0_15"].status.name == "UNREADY"
    assert env.disturbance_event_results[scenario["scenario_id"]][
        "actual_hit_task_keys"
    ] == ("0_15",)

    rewards: list[float] = []
    signatures: list[
        tuple[tuple[str, ...], tuple[bool, bool], tuple[int, ...]]
    ] = []
    _observation, reward, terminated, truncated, info = env.step(
        {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    )
    assert info.get("advanced") is True
    assert not terminated and not truncated
    rewards.append(reward)
    assert env.state.current_time == pytest.approx(recovery_time)

    records: list[TrajectoryExecutionRecord] = []
    for action_index, (station_id, task_id) in enumerate(FIXED_TASKS):
        task_key = f"0_{task_id}"
        candidates = tuple(
            candidate.task_key for candidate in env.get_action_candidates()
        )
        task = env.state.tasks[task_key]
        mask = env.get_action_branch_mask(task)
        valid_workers = tuple(env.valid_team_completion_workers(task, []))
        signatures.append((candidates, mask, valid_workers))
        assert task_key in candidates and mask[0]

        _observation, reward, terminated, truncated, _info = env.step(
            {
                "task_key": task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": FIXED_TEAMS[task_key],
                "align": 0,
            }
        )
        rewards.append(reward)
        assert not truncated
        assert terminated is (action_index == len(FIXED_TASKS) - 1)
        assert task.actual_start is not None and task.actual_end is not None
        aircraft = env.state.aircraft[0]
        records.append(
            TrajectoryExecutionRecord(
                aircraft_id=0,
                task_id=task_id,
                station_id=station_id - 1,
                team=FIXED_TEAMS[task_key],
                start=task.actual_start,
                end=task.actual_end,
                material_ready_time=task.material_ready_time,
                station_entry_time=aircraft.entry_times[station_id - 1],
                aircraft_station_at_start=station_id - 1,
                station_exit_time=aircraft.exit_times[station_id - 1],
            )
        )

    constraints = {
        task_id: TaskConstraintRecord(
            demand=env.state.tasks[f"0_{task_id}"].demand,
            required_skill=env.state.tasks[f"0_{task_id}"].skill,
            predecessors=env.state.tasks[f"0_{task_id}"].predecessors,
            fixed_station=env.state.tasks[f"0_{task_id}"].fixed_station,
            max_allowed_station=env.state.tasks[f"0_{task_id}"].max_allowed_station,
        )
        for _, task_id in FIXED_TASKS
    }
    feasibility = validate_trajectory(
        records,
        task_constraints=constraints,
        worker_skills=env.worker_skills,
        worker_station_bindings={
            worker_id: station_id
            for station_id, worker_ids in env.state.station_worker_bindings.items()
            for worker_id in worker_ids
        },
        station_capacities={station_id: env.max_slots_per_station for station_id in range(5)},
    )
    objective = evaluate_trajectory_objective(env)
    return env, rewards, signatures, objective, feasibility


def test_five_station_fixed_actions_match_manual_time_and_costs(tmp_path: Path) -> None:
    """固定五步动作应得到手算工期、周期偏差和团队费用。"""
    if not BASELINE_PATH.is_file():
        pytest.skip(f"{BASELINE_PATH} 不存在")

    baseline_path = tmp_path / "five_station_baseline.json"
    _small_five_station_baseline(baseline_path)
    weights = ObjectiveWeights(
        w_h=0.0,
        w_t=0.20,
        w_w=0.05,
        w_p=0.0,
        normalize_by_n=False,
    )
    env = AirLineEnvWork3(baseline_json_path=baseline_path, weights=weights)
    env.reset()

    records: list[TrajectoryExecutionRecord] = []
    rewards: list[float] = []
    for (station_id, task_id), expected_duration in zip(
        FIXED_TASKS, EXPECTED_TASK_DURATIONS, strict=True
    ):
        task_key = f"0_{task_id}"
        task = env.state.tasks[task_key]
        aircraft = env.state.aircraft[0]
        assert task.current_station == station_id - 1
        assert aircraft.current_station == station_id - 1
        assert task.status.name == "READY"

        _, reward, _terminated, truncated, _info = env.step(
            {
                "task_key": task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": FIXED_TEAMS[task_key],
                "align": 0,
            }
        )
        rewards.append(reward)
        assert not truncated
        assert task.actual_start is not None
        assert task.actual_end is not None
        assert task.actual_end - task.actual_start == pytest.approx(expected_duration)
        assert task.actual_start == pytest.approx(aircraft.entry_times[station_id - 1])

        records.append(
            TrajectoryExecutionRecord(
                aircraft_id=0,
                task_id=task_id,
                station_id=station_id - 1,
                team=FIXED_TEAMS[task_key],
                start=task.actual_start,
                end=task.actual_end,
                station_entry_time=aircraft.entry_times[station_id - 1],
                aircraft_station_at_start=station_id - 1,
                station_exit_time=aircraft.exit_times[station_id - 1],
            )
        )

    assert _terminated
    expected_makespan = sum(EXPECTED_TASK_DURATIONS)
    assert expected_makespan == pytest.approx(4.806535955993027)
    assert env.state.current_time == pytest.approx(expected_makespan)
    assert env.state.transfer_history[-1] == pytest.approx(expected_makespan)

    objective = evaluate_trajectory_objective(env, weights=weights)
    assert objective.d_time == pytest.approx(0.0)
    assert objective.j_takt == pytest.approx(0.0)
    assert objective.d_team == pytest.approx(0.5)
    assert objective.revision_team == pytest.approx(0.5)
    assert objective.j_revision == pytest.approx(0.025)
    assert env.cost_team == pytest.approx(0.025)
    assert objective.j_total == pytest.approx(0.05)
    assert sum(rewards) == pytest.approx(-objective.j_total)

    task_constraints = {
        task_id: TaskConstraintRecord(
            demand=env.state.tasks[f"0_{task_id}"].demand,
            required_skill=env.state.tasks[f"0_{task_id}"].skill,
            predecessors=(),
            fixed_station=env.state.tasks[f"0_{task_id}"].fixed_station,
            max_allowed_station=env.state.tasks[f"0_{task_id}"].max_allowed_station,
        )
        for _, task_id in FIXED_TASKS
    }
    independent_report = validate_trajectory(
        records,
        task_constraints=task_constraints,
        worker_skills=env.worker_skills,
        worker_station_bindings={
            worker_id: station_id
            for station_id, worker_ids in env.state.station_worker_bindings.items()
            for worker_id in worker_ids
        },
        station_capacities={station_id: env.max_slots_per_station for station_id in range(5)},
    )
    assert independent_report.is_feasible, independent_report.violations


def test_n01_uniform_time_scaling_preserves_feasibility_and_normalized_cost(
    tmp_path: Path,
) -> None:
    """统一缩放时间、工时、H0及恢复时刻后，固定五站动作保持可行与归一化费用。"""
    if not BASELINE_PATH.is_file():
        pytest.skip(f"{BASELINE_PATH} 不存在")

    scale = 3.0
    base_env, base_rewards, base_signatures, base_objective, base_check = (
        _run_scaled_disturbed_batch(tmp_path / "n01_base.json", time_scale=1.0)
    )
    scaled_env, scaled_rewards, scaled_signatures, scaled_objective, scaled_check = (
        _run_scaled_disturbed_batch(
            tmp_path / "n01_scaled.json",
            time_scale=scale,
        )
    )

    base_target = base_env.state.tasks["0_15"]
    scaled_target = scaled_env.state.tasks["0_15"]
    assert base_target.actual_start == pytest.approx(400.0)
    assert scaled_target.actual_start == pytest.approx(1200.0)
    assert base_target.actual_start >= base_target.material_ready_time - base_env.tolerance
    assert scaled_target.actual_start >= scaled_target.material_ready_time - scaled_env.tolerance
    assert base_env.state.h0 < base_target.actual_start - base_env.tolerance
    assert scaled_env.state.h0 < scaled_target.actual_start - scaled_env.tolerance
    assert base_env.tolerance == scaled_env.tolerance

    # The fixed absolute tolerance is not scaled; event and H0 margins here are much larger.
    assert base_target.actual_start - base_env.state.h0 > 100.0
    assert scaled_target.actual_start - scaled_env.state.h0 > 100.0
    assert base_check.is_feasible, base_check.violations
    assert scaled_check.is_feasible, scaled_check.violations
    assert scaled_check.violations == base_check.violations == {}
    assert base_signatures == scaled_signatures
    assert scaled_env.state.h0 == pytest.approx(scale * base_env.state.h0)
    assert scaled_env.state.current_time == pytest.approx(
        scale * base_env.state.current_time,
        rel=0.0,
        abs=1e-7,
    )
    assert scaled_env.state.transfer_history == pytest.approx(
        [scale * value for value in base_env.state.transfer_history],
        rel=0.0,
        abs=1e-7,
    )
    assert scaled_env.disturbance_event_results == base_env.disturbance_event_results

    for task_key, base_task in base_env.state.tasks.items():
        scaled_task = scaled_env.state.tasks[task_key]
        assert scaled_task.status == base_task.status == TaskStatus.COMPLETED
        assert scaled_task.current_station == base_task.current_station
        assert scaled_task.assigned_team == base_task.assigned_team
        for field_name in (
            "duration",
            "material_ready_time",
            "scheduled_start",
            "actual_start",
            "actual_end",
            "execution_duration",
            "cycle_start_time",
        ):
            base_value = getattr(base_task, field_name)
            scaled_value = getattr(scaled_task, field_name)
            if base_value is None:
                assert scaled_value is None, f"{task_key}.{field_name}"
            else:
                assert scaled_value == pytest.approx(
                    scale * base_value,
                    rel=0.0,
                    abs=1e-7,
                ), f"{task_key}.{field_name}"

    base_aircraft = base_env.state.aircraft[0]
    scaled_aircraft = scaled_env.state.aircraft[0]
    assert scaled_aircraft.current_station == base_aircraft.current_station
    assert scaled_aircraft.entry_times.keys() == base_aircraft.entry_times.keys()
    assert scaled_aircraft.exit_times.keys() == base_aircraft.exit_times.keys()
    for station_id, base_time in base_aircraft.entry_times.items():
        assert scaled_aircraft.entry_times[station_id] == pytest.approx(
            scale * base_time,
            rel=0.0,
            abs=1e-7,
        )
    for station_id, base_time in base_aircraft.exit_times.items():
        assert scaled_aircraft.exit_times[station_id] == pytest.approx(
            scale * base_time,
            rel=0.0,
            abs=1e-7,
        )

    assert scaled_rewards == pytest.approx(base_rewards, rel=0.0, abs=1e-10)
    assert sum(base_rewards) == pytest.approx(-base_objective.j_total, abs=1e-10)
    assert sum(scaled_rewards) == pytest.approx(-scaled_objective.j_total, abs=1e-10)
    assert base_objective.j_takt > 0.0
    assert base_objective.d_time > 0.0
    for field_name in (
        "j_takt",
        "d_time",
        "d_team",
        "j_postpone",
        "j_total",
        "revision_time",
        "revision_team",
        "j_revision",
    ):
        assert getattr(scaled_objective, field_name) == pytest.approx(
            getattr(base_objective, field_name),
            rel=0.0,
            abs=1e-10,
        ), field_name
    assert scaled_objective.takt_violation_hours == pytest.approx(
        scale * base_objective.takt_violation_hours,
        rel=0.0,
        abs=1e-7,
    )


def test_n02_translating_timeline_and_baseline_anchor_preserves_cycle_offset(
    tmp_path: Path,
) -> None:
    """整体平移事件和基准绝对时刻后，固定动作的周期相对偏差不变。"""
    if not BASELINE_PATH.is_file():
        pytest.skip(f"{BASELINE_PATH} 不存在")

    shift = 250.0
    base_path = tmp_path / "n02_base.json"
    shifted_path = tmp_path / "n02_shifted.json"
    base_env, _base_rewards, _base_signatures, base_objective, base_check = (
        _run_scaled_disturbed_batch(base_path, time_scale=1.0)
    )
    shifted_env, _shifted_rewards, _shifted_signatures, shifted_objective, shifted_check = (
        _run_scaled_disturbed_batch(
            shifted_path,
            time_scale=1.0,
            time_shift=shift,
        )
    )

    base_baseline = MultiAircraftBaseline.load_from_json(base_path)
    shifted_baseline = MultiAircraftBaseline.load_from_json(shifted_path)
    for task_key, base_task in base_baseline.tasks.items():
        shifted_task = shifted_baseline.tasks[task_key]
        assert shifted_task.baseline_start == pytest.approx(
            base_task.baseline_start + shift
        )
        assert shifted_task.baseline_end == pytest.approx(base_task.baseline_end + shift)
        assert shifted_task.nominal_station_entry == pytest.approx(
            base_task.nominal_station_entry + shift
        )
        assert shifted_task.nominal_station_exit == pytest.approx(
            base_task.nominal_station_exit + shift
        )
        assert shifted_task.in_station_offset == pytest.approx(base_task.in_station_offset)

    assert shifted_env.state.h0 == pytest.approx(base_env.state.h0)
    assert shifted_env.disturbance_event_results[
        "FIXED_DISTURBANCE_EVENT"
    ]["actual_hit_task_keys"] == base_env.disturbance_event_results[
        "FIXED_DISTURBANCE_EVENT"
    ]["actual_hit_task_keys"] == ("0_15",)
    assert _shifted_signatures == _base_signatures
    assert base_env.state.tasks["0_15"].actual_start == pytest.approx(400.0)
    assert shifted_env.state.tasks["0_15"].actual_start == pytest.approx(400.0 + shift)
    assert shifted_env.state.transfer_history == pytest.approx(
        [time + shift for time in base_env.state.transfer_history]
    )
    assert shifted_env.state.current_time == pytest.approx(
        base_env.state.current_time + shift
    )
    for aircraft_id, base_aircraft in base_env.state.aircraft.items():
        shifted_aircraft = shifted_env.state.aircraft[aircraft_id]
        assert shifted_aircraft.current_station == base_aircraft.current_station
        assert shifted_aircraft.entry_times.keys() == base_aircraft.entry_times.keys()
        assert shifted_aircraft.exit_times.keys() == base_aircraft.exit_times.keys()
        for station_id, base_time in base_aircraft.entry_times.items():
            assert shifted_aircraft.entry_times[station_id] == pytest.approx(
                base_time + shift
            )
        for station_id, base_time in base_aircraft.exit_times.items():
            assert shifted_aircraft.exit_times[station_id] == pytest.approx(
                base_time + shift
            )

    for task_key, base_task in base_env.state.tasks.items():
        shifted_task = shifted_env.state.tasks[task_key]
        assert shifted_task.status == base_task.status == TaskStatus.COMPLETED
        assert shifted_task.in_station_offset == pytest.approx(base_task.in_station_offset)
        assert shifted_task.assigned_team == base_task.assigned_team
        for field_name in (
            "material_ready_time",
            "scheduled_start",
            "actual_start",
            "actual_end",
            "cycle_start_time",
        ):
            base_value = getattr(base_task, field_name)
            shifted_value = getattr(shifted_task, field_name)
            if base_value is None:
                assert shifted_value is None, f"{task_key}.{field_name}"
            else:
                assert shifted_value == pytest.approx(
                    base_value + shift
                ), f"{task_key}.{field_name}"
        assert shifted_task.execution_duration == pytest.approx(
            base_task.execution_duration
        )

    assert base_check.is_feasible, base_check.violations
    assert shifted_check.is_feasible, shifted_check.violations
    assert shifted_check.violations == base_check.violations == {}
    assert shifted_env.cost_time == pytest.approx(base_env.cost_time)
    assert shifted_objective.d_time == pytest.approx(base_objective.d_time)
