"""F04：独立执行轨迹复核器的反例测试。"""

from __future__ import annotations

from types import SimpleNamespace

from scripts.work3.evaluate_c_vs_d import _check_completed_trajectory_feasibility
from scripts.work3.train_ppo_work3 import (
    _independent_feasibility_from_audit,
)
from training.work3_vector_env import _execution_record
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
            station_exit_time=4.0,
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
            station_exit_time=4.0,
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


def test_independent_checker_detects_duplicate_task_execution_records() -> None:
    """同一架飞机同一工序出现两条不重叠执行记录也必须判为重复。"""
    records = [
        TrajectoryExecutionRecord(
            aircraft_id=0,
            task_id=0,
            station_id=0,
            team=(0,),
            start=0.0,
            end=1.0,
            station_entry_time=0.0,
            aircraft_station_at_start=0,
        ),
        TrajectoryExecutionRecord(
            aircraft_id=0,
            task_id=0,
            station_id=0,
            team=(0,),
            start=1.0,
            end=2.0,
            station_entry_time=0.0,
            aircraft_station_at_start=0,
        ),
    ]
    report = validate_trajectory(
        records,
        task_constraints={0: TaskConstraintRecord(demand=1, required_skill=0)},
        worker_skills={0: {0}},
        worker_station_bindings={0: 0},
        station_capacities={0: 1},
    )

    assert report.violations == {"duplicate_task": 1}


def test_independent_checker_detects_cross_station_execution() -> None:
    """实际位于站位0却记录在站位1加工的轨迹必须被拒绝。"""
    record = TrajectoryExecutionRecord(
        aircraft_id=0,
        task_id=0,
        station_id=1,
        team=(1,),
        start=0.0,
        end=1.0,
        station_entry_time=0.0,
        aircraft_station_at_start=0,
    )
    report = validate_trajectory(
        [record],
        task_constraints={
            0: TaskConstraintRecord(
                demand=1,
                required_skill=0,
                fixed_station=0,
                max_allowed_station=0,
            )
        },
        worker_skills={1: {0}},
        worker_station_bindings={1: 1},
        station_capacities={1: 1},
    )

    assert not report.is_feasible
    assert report.violations == {
        "fixed_station": 1,
        "station_upper_bound": 1,
        "aircraft_position": 1,
    }


def test_independent_checker_detects_processing_across_synchronous_transfer() -> None:
    """工序在离站脉动前尚未完工时必须被独立检查器拒绝。"""
    record = TrajectoryExecutionRecord(
        aircraft_id=0,
        task_id=0,
        station_id=0,
        team=(0,),
        start=4.0,
        end=6.0,
        station_entry_time=0.0,
        aircraft_station_at_start=0,
        station_exit_time=5.0,
    )
    report = validate_trajectory(
        [record],
        task_constraints={
            0: TaskConstraintRecord(
                demand=1,
                required_skill=0,
                fixed_station=0,
                max_allowed_station=0,
            )
        },
        worker_skills={0: {0}},
        worker_station_bindings={0: 0},
        station_capacities={0: 1},
    )

    assert report.violations == {"processing_across_transfer": 1}


def test_evaluation_feasibility_rebuilds_station_exit_time_from_aircraft_log() -> None:
    """正式评测导出器必须把飞机离站脉动时刻传入独立检查器。"""
    task = SimpleNamespace(
        task_id=0,
        aircraft_id=0,
        demand=1,
        skill=0,
        predecessors=(),
        fixed_station=0,
        max_allowed_station=0,
        actual_start=4.0,
        actual_end=6.0,
        current_station=0,
        assigned_team=[0],
        material_ready_time=0.0,
    )
    aircraft = SimpleNamespace(entry_times={0: 0.0}, exit_times={0: 5.0})
    state = SimpleNamespace(
        tasks={"0_0": task},
        aircraft={0: aircraft},
        station_worker_bindings={0: [0]},
        num_stations=1,
    )
    env = SimpleNamespace(
        _check_terminated=lambda: True,
        state=state,
        worker_skills={0: {0}},
        max_slots_per_station=1,
    )

    feasible, violations = _check_completed_trajectory_feasibility(env)

    assert not feasible
    assert violations == {"processing_across_transfer": 1}


def test_training_audit_rejects_processing_across_synchronous_transfer() -> None:
    """训练审计重建路径也必须保留真实离站时刻并拒绝跨脉动加工。"""
    task = SimpleNamespace(
        task_key="0_0",
        task_id=0,
        aircraft_id=0,
        current_station=0,
        last_published_assignment=None,
        baseline_assignment={"station": 0},
        assigned_team=[0],
        actual_start=4.0,
        actual_end=6.0,
        material_ready_time=0.0,
    )
    aircraft = SimpleNamespace(entry_times={0: 0.0}, exit_times={0: 5.0})
    execution_record = _execution_record(
        SimpleNamespace(state=SimpleNamespace(
            tasks={"0_0": task},
            aircraft={0: aircraft},
        )),
        "0_0",
        {"0_0": 0},
    )
    audit = {
        "success": True,
        "completed_tasks": 1,
        "total_tasks": 1,
        "execution_records": [execution_record],
        "task_constraints": {
            0: {
                "demand": 1,
                "required_skill": 0,
                "predecessors": (),
                "fixed_station": 0,
                "max_allowed_station": 0,
            },
        },
        "worker_skills": {0: (0,)},
        "worker_station_bindings": {0: 0},
        "station_capacities": {0: 1},
    }

    result = _independent_feasibility_from_audit(audit)[0]

    assert result["status"] == "violations"
    assert result["violations"] == {"processing_across_transfer": 1}
