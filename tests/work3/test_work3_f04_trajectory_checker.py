"""F04：独立执行轨迹复核器的反例测试。"""

from __future__ import annotations

from utils.work3.trajectory_feasibility import (
    TaskConstraintRecord,
    TrajectoryExecutionRecord,
    validate_trajectory,
)


def _valid_records() -> list[TrajectoryExecutionRecord]:
    return [
        TrajectoryExecutionRecord(
            aircraft_id=0,
            task_id=0,
            station_id=0,
            team=(0,),
            start=0.0,
            end=2.0,
            station_entry_time=0.0,
            aircraft_station_at_start=0,
        ),
        TrajectoryExecutionRecord(
            aircraft_id=0,
            task_id=1,
            station_id=0,
            team=(1,),
            start=2.0,
            end=4.0,
            station_entry_time=0.0,
            aircraft_station_at_start=0,
        ),
    ]


def _constraints() -> dict[int, TaskConstraintRecord]:
    return {
        0: TaskConstraintRecord(
            demand=1,
            required_skill=0,
            predecessors=(),
            fixed_station=0,
            max_allowed_station=0,
        ),
        1: TaskConstraintRecord(
            demand=1,
            required_skill=0,
            predecessors=(0,),
            fixed_station=0,
            max_allowed_station=0,
        ),
    }


def test_independent_checker_accepts_legal_log() -> None:
    report = validate_trajectory(
        _valid_records(),
        task_constraints=_constraints(),
        worker_skills={0: {0}, 1: {0}},
        worker_station_bindings={0: 0, 1: 0},
        station_capacities={0: 1},
    )

    assert report.is_feasible
    assert not report.violations


def test_independent_checker_detects_resource_skill_release_and_position_errors() -> None:
    records = [
        TrajectoryExecutionRecord(
            aircraft_id=0,
            task_id=0,
            station_id=0,
            team=(0,),
            start=3.0,
            end=5.0,
            material_ready_time=6.0,
            station_entry_time=4.0,
            aircraft_station_at_start=1,
        ),
        TrajectoryExecutionRecord(
            aircraft_id=0,
            task_id=1,
            station_id=0,
            team=(0,),
            start=4.0,
            end=6.0,
            station_entry_time=0.0,
            aircraft_station_at_start=0,
        ),
    ]
    report = validate_trajectory(
        records,
        task_constraints={
            **_constraints(),
            0: TaskConstraintRecord(
                demand=1,
                required_skill=1,
                predecessors=(),
                fixed_station=0,
                max_allowed_station=0,
            ),
        },
        worker_skills={0: {0}},
        worker_station_bindings={0: 0},
        station_capacities={0: 1},
    )

    assert not report.is_feasible
    assert report.violations["worker_overlap"] == 1
    assert report.violations["skill"] == 1
    assert report.violations["material_release"] == 1
    assert report.violations["station_entry"] == 1
    assert report.violations["aircraft_position"] == 1


def test_independent_checker_detects_precedence_and_station_capacity() -> None:
    records = [
        TrajectoryExecutionRecord(
            aircraft_id=0,
            task_id=0,
            station_id=0,
            team=(0,),
            start=0.0,
            end=5.0,
            station_entry_time=0.0,
            aircraft_station_at_start=0,
        ),
        TrajectoryExecutionRecord(
            aircraft_id=0,
            task_id=1,
            station_id=0,
            team=(1,),
            start=1.0,
            end=2.0,
            station_entry_time=0.0,
            aircraft_station_at_start=0,
        ),
    ]
    report = validate_trajectory(
        records,
        task_constraints=_constraints(),
        worker_skills={0: {0}, 1: {0}},
        worker_station_bindings={0: 0, 1: 0},
        station_capacities={0: 1},
    )

    assert report.violations["precedence"] == 1
    assert report.violations["station_capacity"] == 1
