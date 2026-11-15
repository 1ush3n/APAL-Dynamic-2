"""只读调度条件：复用环境物理规则，完整枚举固定站位的合法团队。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import TYPE_CHECKING

import torch

from envs.work3.core_types import TaskStatus

if TYPE_CHECKING:
    from envs.work3.environment import AirLineEnvWork3

QUERY_VERSION = "work3_conditions_v1"
METHOD_ID = "hierarchical_condition_query_v1"
CONDITION_DIM = 14
CONDITION_LAYOUT = (
    "exact", "lower_bound", "unknown", "no_completion", "budget_exhausted",
    "start_delta_h0", "end_delta_h0", "duration_h0", "time_valid",
    "material_wait", "predecessor_wait", "worker_wait", "station_wait", "alignment_wait",
)


class QueryStatus(IntEnum):
    EXACT = 0
    LOWER_BOUND = 1
    UNKNOWN = 2
    NO_COMPLETION = 3
    BUDGET_EXHAUSTED = 4


@dataclass(frozen=True, slots=True)
class ScheduleCondition:
    """精确指当前已知承诺下的条件结果，不承诺未来扰动后的时刻。"""

    status: QueryStatus
    team: tuple[int, ...] = ()
    start: float | None = None
    end: float | None = None
    duration: float | None = None
    reasons: tuple[str, ...] = ()
    conflict_keys: tuple[str, ...] = ()

    def features(self, now: float, h0: float) -> torch.Tensor:
        assert h0 > 0.0
        output = torch.zeros(CONDITION_DIM, dtype=torch.float32)
        output[int(self.status)] = 1.0
        if self.start is not None and self.end is not None and self.duration is not None:
            output[5:9] = torch.tensor(((self.start - now) / h0,
                                      (self.end - now) / h0, self.duration / h0, 1.0))
        output[9:] = torch.tensor([float(reason in self.reasons) for reason in
                                  ("material", "predecessor", "worker", "station", "alignment")])
        assert output.shape == (CONDITION_DIM,) and torch.isfinite(output).all()
        return output


@dataclass(frozen=True, slots=True)
class CompletionTable:
    task_key: str
    entries: tuple[tuple[ScheduleCondition, ScheduleCondition], ...]
    unavailable: ScheduleCondition
    eligible_worker_ids: tuple[int, ...] = ()
    demand: int = 0

    def best(self, prefix: tuple[int, ...] = (), align: int = 0) -> ScheduleCondition:
        if align not in (0, 1):
            raise ValueError("对齐只能为0或1")
        if len(set(prefix)) != len(prefix):
            return ScheduleCondition(QueryStatus.NO_COMPLETION)
        selected = frozenset(prefix)
        if len(prefix) > self.demand or not selected.issubset(self.eligible_worker_ids):
            return ScheduleCondition(QueryStatus.NO_COMPLETION)
        results = [pair[align] for pair in self.entries
                   if selected.issubset(pair[align].team)]
        if not results:
            return self.unavailable if not self.entries else ScheduleCondition(QueryStatus.NO_COMPLETION)
        return min(results, key=lambda result: (float(result.start), float(result.end), result.team))

    def worker_features(self, worker_ids: tuple[int, ...], prefix: tuple[int, ...],
                        now: float, h0: float, width: int) -> torch.Tensor:
        assert len(worker_ids) <= width
        # 每位候选 [14] -> 站内候选矩阵 [W, 14]，无效填充另由资格掩码屏蔽。
        result = torch.zeros((width, CONDITION_DIM), dtype=torch.float32)
        for index, worker_id in enumerate(worker_ids):
            result[index] = self.best((*prefix, worker_id)).features(now, h0)
        return result


class SchedulingQueries:
    """生命周期限于一次CPU快照；同刻新指派后重建，不缓存可变现场。"""

    def __init__(self, env: AirLineEnvWork3) -> None:
        self.env = env

    def fixed_team(self, task_key: str, team: tuple[int, ...], *, align: int) -> ScheduleCondition:
        if align not in (0, 1):
            raise ValueError("对齐只能为0或1")
        env = self.env
        task = env.state.tasks[task_key]
        team = tuple(sorted(team))
        env._validate_team_for_task(task, team)
        release = env._predecessor_release_time(task)
        if release is None:
            return ScheduleCondition(QueryStatus.UNKNOWN, team, reasons=("predecessor",))
        if env.state.aircraft[task.aircraft_id].current_station != task.current_station:
            return ScheduleCondition(QueryStatus.UNKNOWN, team, reasons=("station",))
        if task.status in (TaskStatus.RESERVED, TaskStatus.RUNNING, TaskStatus.COMPLETED):
            raise ValueError("查询不允许重新安排已提交工序")
        now = float(env.state.current_time)
        lower = max(now, float(task.material_ready_time), float(release))
        reasons: list[str] = []
        if task.material_ready_time > now + env.tolerance:
            reasons.append("material")
        if release > now + env.tolerance:
            reasons.append("predecessor")
        if align == 1:
            target = env.state.last_transfer_time + task.in_station_offset
            if target > lower + env.tolerance:
                reasons.append("alignment")
            lower = max(lower, target)
        duration = env.duration_for_team(task, team, start_time=lower)
        start = env._find_team_earliest_slot(task.current_station, team, lower, duration)
        conflicts = set()
        for worker_id in team:
            for interval in env.state.workers[worker_id].intervals:
                if interval.end > lower + env.tolerance and interval.start < start + duration - env.tolerance:
                    if "worker" not in reasons:
                        reasons.append("worker")
                    conflicts.add(interval.task_key)
        if env._next_station_conflict_end(task.current_station, lower, lower + duration) is not None:
            reasons.append("station")
        return ScheduleCondition(QueryStatus.EXACT, team, start, start + duration,
                                 duration, tuple(reasons), tuple(sorted(conflicts)))

    def completions(self, task_key: str) -> CompletionTable:
        task = self.env.state.tasks[task_key]
        eligible = tuple(self.env.valid_team_completion_workers(task, []))
        if len(eligible) < task.demand:
            return CompletionTable(task_key, (), ScheduleCondition(QueryStatus.NO_COMPLETION), eligible, task.demand)
        if self.env._predecessor_release_time(task) is None:
            return CompletionTable(task_key, (), ScheduleCondition(QueryStatus.UNKNOWN, reasons=("predecessor",)), eligible, task.demand)
        if self.env.state.aircraft[task.aircraft_id].current_station != task.current_station:
            return CompletionTable(task_key, (), ScheduleCondition(QueryStatus.UNKNOWN, reasons=("station",)), eligible, task.demand)
        # 完整枚举保证任意逐人前缀下的最早补全；没有静默截断或效率和近似。
        entries = tuple((self.fixed_team(task_key, tuple(team), align=0),
                         self.fixed_team(task_key, tuple(team), align=1))
                        for team in combinations(eligible, task.demand))
        return CompletionTable(task_key, entries, ScheduleCondition(QueryStatus.NO_COMPLETION), eligible, task.demand)
