"""独立复核工作三执行日志，不调用环境的排程可行性判断。"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from core.constraints import calculate_team_synergy_factor

if TYPE_CHECKING:
    from envs.work3.environment import AirLineEnvWork3


@dataclass(frozen=True)
class TaskConstraintRecord:
    """单架次工序的静态硬约束。"""

    demand: int
    required_skill: int
    predecessors: tuple[int, ...] = ()
    fixed_station: int | None = None
    max_allowed_station: int | None = None
    standard_duration: float | None = None


@dataclass(frozen=True)
class TrajectoryExecutionRecord:
    """从最终执行日志提取的单次物理执行记录。"""

    aircraft_id: int
    task_id: int
    station_id: int
    team: tuple[int, ...]
    start: float
    end: float
    material_ready_time: float = 0.0
    station_entry_time: float | None = None
    aircraft_station_at_start: int | None = None
    station_exit_time: float | None = None


@dataclass(frozen=True)
class TrajectoryFeasibilityReport:
    """独立轨迹检查结果；仅记录实际发现的违规类型。"""

    violations: dict[str, int]
    examples: dict[str, list[dict[str, object]]] = field(default_factory=dict)

    @property
    def is_feasible(self) -> bool:
        return not any(self.violations.values())


def validate_trajectory(
    records: Iterable[TrajectoryExecutionRecord],
    *,
    task_constraints: Mapping[int, TaskConstraintRecord],
    worker_skills: Mapping[int, set[int] | frozenset[int] | Sequence[int]],
    worker_station_bindings: Mapping[int, int],
    station_capacities: Mapping[int, int],
    worker_efficiencies: Mapping[int, float] | None = None,
    tolerance: float = 1e-5,
) -> TrajectoryFeasibilityReport:
    """独立扫描执行记录中的资源、工艺、恢复、位置和转站约束。"""
    rows = list(records)
    counts: dict[str, int] = {}
    examples: dict[str, list[dict[str, object]]] = {}

    def add(kind: str, record: TrajectoryExecutionRecord, **details: object) -> None:
        counts[kind] = counts.get(kind, 0) + 1
        examples.setdefault(kind, [])
        if len(examples[kind]) < 5:
            examples[kind].append(
                {
                    "aircraft_id": record.aircraft_id,
                    "task_id": record.task_id,
                    **details,
                }
            )

    keyed: dict[tuple[int, int], TrajectoryExecutionRecord] = {}
    worker_intervals: dict[int, list[TrajectoryExecutionRecord]] = {}
    station_intervals: dict[int, list[TrajectoryExecutionRecord]] = {}

    for record in rows:
        key = (int(record.aircraft_id), int(record.task_id))
        if key in keyed:
            add("duplicate_task", record)
        keyed[key] = record
        constraint = task_constraints.get(record.task_id)
        if constraint is None:
            add("unknown_task", record)
            continue

        if record.start < -tolerance or record.end <= record.start + tolerance:
            add("invalid_time", record)
        if record.start < record.material_ready_time - tolerance:
            add("material_release", record, release_time=record.material_ready_time)
        if (
            record.station_entry_time is not None
            and record.start < record.station_entry_time - tolerance
        ):
            add("station_entry", record, entry_time=record.station_entry_time)
        if (
            record.station_exit_time is not None
            and record.end > record.station_exit_time + tolerance
        ):
            add(
                "processing_across_transfer",
                record,
                exit_time=record.station_exit_time,
            )
        if record.aircraft_station_at_start is None:
            add("missing_actual_start_station", record)
        elif int(record.aircraft_station_at_start) != int(record.station_id):
            add("aircraft_position", record)

        if constraint.fixed_station is not None and record.station_id != constraint.fixed_station:
            add("fixed_station", record, expected_station=constraint.fixed_station)
        if (
            constraint.max_allowed_station is not None
            and record.station_id > constraint.max_allowed_station
        ):
            add("station_upper_bound", record, max_station=constraint.max_allowed_station)
        if len(record.team) != int(constraint.demand) or len(set(record.team)) != len(record.team):
            add("team_size", record, expected_demand=constraint.demand)

        for worker_id in record.team:
            if worker_id not in worker_skills:
                add("unknown_worker", record, worker_id=worker_id)
            elif int(constraint.required_skill) not in set(worker_skills[worker_id]):
                add("skill", record, worker_id=worker_id)
            if worker_station_bindings.get(worker_id) != record.station_id:
                add("worker_station", record, worker_id=worker_id)
            worker_intervals.setdefault(worker_id, []).append(record)
        station_intervals.setdefault(record.station_id, []).append(record)

        if worker_efficiencies is not None:
            if constraint.standard_duration is None:
                add("missing_duration_inputs", record, field="standard_duration")
            elif constraint.standard_duration < -tolerance:
                add(
                    "invalid_duration_inputs",
                    record,
                    standard_duration=constraint.standard_duration,
                )
            elif len(record.team) == int(constraint.demand) and len(set(record.team)) == len(record.team):
                if any(worker_id not in worker_efficiencies for worker_id in record.team):
                    add("missing_duration_inputs", record, field="worker_efficiencies")
                else:
                    efficiencies = [float(worker_efficiencies[worker_id]) for worker_id in record.team]
                    if any(not math.isfinite(value) or value <= 0.0 for value in efficiencies):
                        add("invalid_duration_inputs", record, field="worker_efficiencies")
                    elif constraint.standard_duration <= tolerance:
                        expected_duration = 0.0
                    else:
                        effective_capacity = sum(efficiencies) * calculate_team_synergy_factor(
                            len(record.team)
                        )
                        if effective_capacity <= tolerance:
                            add("invalid_duration_inputs", record, field="effective_capacity")
                        else:
                            expected_duration = (
                                float(constraint.standard_duration)
                                * int(constraint.demand)
                                / effective_capacity
                            )
                            actual_duration = record.end - record.start
                            if (
                                not math.isfinite(expected_duration)
                                or not math.isfinite(actual_duration)
                            ):
                                add("invalid_duration_inputs", record, field="duration")
                            elif abs(expected_duration - actual_duration) > 1e-4:
                                add(
                                    "duration_mismatch",
                                    record,
                                    expected_duration=expected_duration,
                                    actual_duration=actual_duration,
                                    tolerance_hours=1e-4,
                                )

    for (aircraft_id, task_id), record in keyed.items():
        constraint = task_constraints.get(task_id)
        if constraint is None:
            continue
        for predecessor_id in constraint.predecessors:
            predecessor = keyed.get((aircraft_id, predecessor_id))
            if predecessor is None:
                add("missing_predecessor", record, predecessor_id=predecessor_id)
            elif predecessor.end > record.start + tolerance:
                add("precedence", record, predecessor_id=predecessor_id)
            if predecessor is not None and predecessor.station_id > record.station_id:
                add("station_precedence", record, predecessor_id=predecessor_id)

    for worker_id, intervals in worker_intervals.items():
        ordered = sorted(intervals, key=lambda item: (item.start, item.end))
        for previous, current in zip(ordered, ordered[1:]):
            if previous.end > current.start + tolerance:
                add("worker_overlap", current, worker_id=worker_id)

    for station_id, intervals in station_intervals.items():
        capacity = station_capacities.get(station_id)
        if capacity is None:
            for record in intervals:
                add("unknown_station_capacity", record)
            continue
        active_ends: list[float] = []
        ordered_intervals = sorted(intervals, key=lambda item: (item.start, item.end))
        for record in ordered_intervals:
            if record.end <= record.start + tolerance:
                continue
            while active_ends and active_ends[0] <= record.start + tolerance:
                heapq.heappop(active_ends)
            if len(active_ends) >= int(capacity):
                add(
                    "station_capacity",
                    record,
                    station_id=station_id,
                    time=record.start,
                    capacity=capacity,
                )
                break
            heapq.heappush(active_ends, record.end)

    return TrajectoryFeasibilityReport(violations=counts, examples=examples)


def check_environment_trajectory_feasibility(
    env: AirLineEnvWork3,
) -> tuple[bool, dict[str, int]]:
    """从环境实际执行字段独立重建完整轨迹并复核可行性。"""
    if not env._check_terminated():
        return False, {"incomplete_trajectory": 1}

    records: list[TrajectoryExecutionRecord] = []
    constraints: dict[int, TaskConstraintRecord] = {}
    for task in env.state.tasks.values():
        constraints[task.task_id] = TaskConstraintRecord(
            demand=task.demand,
            required_skill=task.skill,
            predecessors=tuple(task.predecessors),
            fixed_station=task.fixed_station,
            max_allowed_station=task.max_allowed_station,
            standard_duration=(
                None
                if getattr(task, "standard_duration", None) is None
                else float(task.standard_duration)
            ),
        )
        if task.actual_start is None or task.actual_end is None:
            return False, {"missing_execution_interval": 1}
        actual_start_station = getattr(task, "actual_start_station", None)
        if actual_start_station is None:
            return False, {"missing_actual_start_station": 1}
        aircraft = env.state.aircraft[task.aircraft_id]
        station_exit_time = aircraft.exit_times.get(task.current_station)
        if station_exit_time is None:
            return False, {"missing_station_exit_time": 1}
        records.append(
            TrajectoryExecutionRecord(
                aircraft_id=task.aircraft_id,
                task_id=task.task_id,
                station_id=task.current_station,
                team=tuple(task.assigned_team),
                start=float(task.actual_start),
                end=float(task.actual_end),
                material_ready_time=float(task.material_ready_time),
                station_entry_time=aircraft.entry_times.get(task.current_station),
                aircraft_station_at_start=int(actual_start_station),
                station_exit_time=float(station_exit_time),
            )
        )

    report = validate_trajectory(
        records,
        task_constraints=constraints,
        worker_skills=env.worker_skills,
        worker_efficiencies=env.worker_efficiencies,
        worker_station_bindings={
            worker_id: station_id
            for station_id, worker_ids in env.state.station_worker_bindings.items()
            for worker_id in worker_ids
        },
        station_capacities={
            station_id: env.max_slots_per_station
            for station_id in range(env.state.num_stations)
        },
    )
    return report.is_feasible, dict(report.violations)
