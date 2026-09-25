"""M01：五站小实例固定动作的手算轨迹验收。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from envs.work3.core_types import ActionBranch
from envs.work3.environment import AirLineEnvWork3
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline
from utils.work3.objective_evaluator import ObjectiveWeights, evaluate_trajectory_objective
from utils.work3.trajectory_feasibility import (
    TaskConstraintRecord,
    TrajectoryExecutionRecord,
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
    0.5224849650851306,
    0.5452650666162796,
    1.8777017505628602,
    0.9380233559980313,
    1.3917908975110986,
)


def _small_five_station_baseline(path: Path) -> None:
    source = MultiAircraftBaseline.load_from_json(BASELINE_PATH)
    tasks = {}
    for station_id, task_id in FIXED_TASKS:
        task = source.get_task(0, task_id)
        assert task.station_id == station_id
        assert not task.predecessors
        tasks[task.task_key] = replace(
            task,
            in_station_offset=0.0,
            baseline_start=0.0,
            baseline_end=task.duration,
        )

    MultiAircraftBaseline(
        num_aircraft=1,
        num_stations=5,
        h0=source.h0,
        tasks=tasks,
        station_workers=source.station_workers,
    ).save_to_json(path)


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
            )
        )

    assert _terminated
    expected_makespan = sum(EXPECTED_TASK_DURATIONS)
    assert expected_makespan == pytest.approx(5.2752660357734005)
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
