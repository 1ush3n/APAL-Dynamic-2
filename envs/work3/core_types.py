"""工作三核心数据结构、状态机枚举与日历模型。

严格落实顶层设计规范：
1. 工序六状态机：UNREADY, READY, RESERVED, RUNNING, COMPLETED, POSTPONED；
2. 飞机位置数组：s_k(t) in {-1, 0, 1, 2, 3, 4, 5}；
3. 工人区间日历与空隙搜索；
4. 状态快照、复原与重置。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, NamedTuple, Sequence

import numpy as np


class TaskStatus(IntEnum):
    """工序生命周期六状态枚举。

    状态流转说明：
    - UNREADY (0): 紧前工序未完工，或飞机尚未进入对应站位；
    - READY (1): 前驱全部完工、可用时刻到达且飞机在场，具备调度决策资格；
    - RESERVED (2): 策略已指派团队并排定未来时刻，预占工人日历，但尚未物理开工；
    - RUNNING (3): 任务已到达开工时刻，正在由团队物理加工；
    - COMPLETED (4): 任务加工完成，释放工人日历，解除后继工序的前驱约束；
    - POSTPONED (5): 本周期决策合法后移至下一站，当前站免除完工考核，待转站后在新站重新唤醒。
    """

    UNREADY = 0
    READY = 1
    RESERVED = 2
    RUNNING = 3
    COMPLETED = 4
    POSTPONED = 5


class ActionBranch(IntEnum):
    """自回归动作分支枚举。"""

    STATION_EXECUTE = 0  # 留在当前站位执行（后续选队并对齐）
    POSTPONE = 1        # 合法后移至紧邻下一站位（动作当场截断）
    ADVANCE_TO_NEXT_EVENT = 2  # 结束当前修订轮次并推进到下一个离散事件


class TimeInterval(NamedTuple):
    """工人日历占用区间。"""

    start: float
    end: float
    task_key: str


@dataclass
class WorkerCalendar:
    """单个工人的时间占用日历。

    维护互不重叠的时间区间列表，支持快速可用性判定、区间插入、取消预约与空闲间隙搜索。
    """

    worker_id: int
    station_id: int
    intervals: list[TimeInterval] = field(default_factory=list)

    def is_available(self, start: float, end: float, tolerance: float = 1e-5) -> bool:
        """检查工人在指定时间窗口 [start, end] 内是否完全空闲。"""
        if end <= start + tolerance:
            return True
        for iv in self.intervals:
            # 重叠判定：max(start, iv.start) < min(end, iv.end)
            if max(start, iv.start) < min(end, iv.end) - tolerance:
                return False
        return True

    def add_interval(self, start: float, end: float, task_key: str, tolerance: float = 1e-5) -> None:
        """为工序登记时间占用区间。"""
        if not self.is_available(start, end, tolerance=tolerance):
            raise ValueError(
                f"工人 {self.worker_id} 在区间 [{start:.2f}, {end:.2f}] 存在时间冲突！"
                f"当前已有区间: {self.intervals}"
            )
        self.intervals.append(TimeInterval(start, end, task_key))
        self.intervals.sort(key=lambda x: x.start)

    def remove_interval(self, task_key: str) -> bool:
        """撤销指定任务的时间占用区间（用于旧预约事件失效与改派）。"""
        initial_len = len(self.intervals)
        self.intervals = [iv for iv in self.intervals if iv.task_key != task_key]
        return len(self.intervals) < initial_len

    def find_earliest_slot(
        self, search_start: float, duration: float, tolerance: float = 1e-5
    ) -> float:
        """从 search_start 开始，寻找工人的第一个能够容纳 duration 的可行区间起点。"""
        if duration <= tolerance:
            return search_start

        candidate = search_start
        for iv in self.intervals:
            if candidate + duration <= iv.start + tolerance:
                return candidate
            if candidate < iv.end - tolerance:
                candidate = iv.end

        return candidate

    def copy(self) -> "WorkerCalendar":
        """创建日历的独立深拷贝。"""
        return WorkerCalendar(
            worker_id=self.worker_id,
            station_id=self.station_id,
            intervals=list(self.intervals),
        )


@dataclass
class TaskRuntimeState:
    """单道工序的动态运行时状态。"""

    aircraft_id: int
    task_id: int
    task_key: str
    base_station: int           # 原始基准站位 (0-based: 0 ~ 4)
    current_station: int        # 当前排定执行站位 (0-based: 0 ~ 4)
    status: TaskStatus
    duration: float                 # 基准模板团队的实际加工工时
    in_station_offset: float     # 单机模板站内周期偏移量 b_i^0
    demand: int
    skill: int
    ao_code: str
    predecessors: tuple[int, ...]  # 本机内部紧前工序 ID 列表
    standard_duration: float | None = None  # 原始实例中的标准加工工时

    assigned_team: list[int] = field(default_factory=list)
    scheduled_start: float | None = None
    actual_start: float | None = None
    actual_start_station: int | None = None
    actual_end: float | None = None

    material_ready_time: float = 0.0  # 开工可用性到达时刻 R（受扰后继承）
    postpone_count: int = 0           # 累计正式改站/后移次数 n_{ki}
    generation: int = 0               # 预约版本代数，用于失效令牌校验
    base_team: tuple[int, ...] = field(default_factory=tuple)  # 基准标准团队 W_i^0
    fixed_station: int | None = None  # 原始实例固定站位（0-based）
    max_allowed_station: int | None = None  # 工艺后继约束允许的最晚站位（0-based）
    execution_duration: float | None = None  # 本次团队对应的实际工时
    cycle_start_time: float | None = None  # 实际开工所在周期的转站时刻 P_{q-1}
    start_cost_confirmed: bool = False     # 是否已确认并结算开工偏差与团队替换费用
    baseline_assignment: dict[str, Any] = field(default_factory=dict)
    last_published_assignment: dict[str, Any] = field(default_factory=dict)
    revision_history: list[dict[str, Any]] = field(default_factory=list)
    _state_ref: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """初始化基准安排与当前正式安排快照。"""
        if not self.baseline_assignment:
            self.baseline_assignment = {
                "station": int(self.base_station),
                "team": list(self.base_team),
                "position": float(self.in_station_offset),
            }
        if not self.last_published_assignment:
            self.last_published_assignment = copy.deepcopy(self.baseline_assignment)

    def can_physically_start(self, current_time: float, tolerance: float = 1e-5) -> bool:
        """检查任务是否满足物理开工硬条件（到料到达且时刻到达）。"""
        return current_time >= self.material_ready_time - tolerance

    def reserve(self, team: Sequence[int], scheduled_start: float) -> None:
        """建立未来开工预约。"""
        self.assigned_team = [int(w) for w in team]
        self.scheduled_start = float(scheduled_start)
        self.status = TaskStatus.RESERVED

    def cancel_reservation(self) -> None:
        """取消当前预约，退回就绪状态，并递增版本代数使事件队列旧开工事件失效。"""
        self.assigned_team = []
        self.scheduled_start = None
        self.execution_duration = None
        self.status = TaskStatus.READY
        self.generation += 1

    def start_work(
        self,
        current_time: float,
        cycle_start_time: float | None = None,
        actual_start_station: int | None = None,
    ) -> None:
        """实际开工。"""
        self.actual_start = float(current_time)
        if self.actual_start_station is None and actual_start_station is not None:
            self.actual_start_station = int(actual_start_station)
        if cycle_start_time is not None:
            self.cycle_start_time = float(cycle_start_time)
        self.status = TaskStatus.RUNNING

    def complete_work(self, current_time: float) -> None:
        """实际完工。"""
        self.actual_end = float(current_time)
        self.status = TaskStatus.COMPLETED

    def postpone_to_next_station(self) -> None:
        """后移至下一站，更新状态并递增改派计数。"""
        old_station = self.current_station
        self.assigned_team = []
        self.scheduled_start = None
        self.execution_duration = None
        self.status = TaskStatus.POSTPONED
        self.current_station += 1
        self.postpone_count += 1
        self.generation += 1
        state_ref = getattr(self, "_state_ref", None)
        if state_ref is not None:
            state_ref.move_task_station(self, old_station, self.current_station)

    def awaken_in_station(self, all_predecessors_completed: bool) -> None:
        """飞机到达新站后唤醒后移任务。"""
        if all_predecessors_completed:
            self.status = TaskStatus.READY
        else:
            self.status = TaskStatus.UNREADY

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "current_station":
            old = getattr(self, "current_station", None)
            super().__setattr__(name, value)
            if old is not None and old != value:
                state_ref = getattr(self, "_state_ref", None)
                if state_ref is not None:
                    state_ref.move_task_station(self, old, value)
        else:
            super().__setattr__(name, value)

    def copy(self) -> "TaskRuntimeState":
        """深拷贝任务运行时状态。"""
        copied = TaskRuntimeState(
            aircraft_id=self.aircraft_id,
            task_id=self.task_id,
            task_key=self.task_key,
            base_station=self.base_station,
            current_station=self.current_station,
            status=self.status,
            duration=self.duration,
            standard_duration=self.standard_duration,
            in_station_offset=self.in_station_offset,
            demand=self.demand,
            skill=self.skill,
            ao_code=self.ao_code,
            predecessors=self.predecessors,
            assigned_team=list(self.assigned_team),
            scheduled_start=self.scheduled_start,
            actual_start=self.actual_start,
            actual_start_station=self.actual_start_station,
            actual_end=self.actual_end,
            material_ready_time=self.material_ready_time,
            postpone_count=self.postpone_count,
            generation=self.generation,
            base_team=self.base_team,
            fixed_station=self.fixed_station,
            max_allowed_station=self.max_allowed_station,
            execution_duration=self.execution_duration,
            cycle_start_time=self.cycle_start_time,
            start_cost_confirmed=self.start_cost_confirmed,
            baseline_assignment=copy.deepcopy(self.baseline_assignment),
            last_published_assignment=copy.deepcopy(self.last_published_assignment),
            revision_history=copy.deepcopy(self.revision_history),
        )
        copied._state_ref = getattr(self, "_state_ref", None)
        return copied


@dataclass
class AircraftRuntimeState:
    """单架飞机的运行时站位与流转追踪。"""

    aircraft_id: int
    current_station: int = -1  # -1: 未进线, 0~4: 站位1~5, 5: 已离线出站
    entry_times: dict[int, float] = field(default_factory=dict)
    exit_times: dict[int, float] = field(default_factory=dict)

    @property
    def is_in_factory(self) -> bool:
        """是否在产线五个站位中作业。"""
        return 0 <= self.current_station <= 4

    @property
    def is_completed(self) -> bool:
        """是否已离开末站完全完工。"""
        return self.current_station >= 5

    def step_to_next_station(self, timestamp: float) -> None:
        """全线脉动同步流转时向前步进一步。"""
        if 0 <= self.current_station <= 4:
            self.exit_times[self.current_station] = float(timestamp)
        self.current_station += 1
        if 0 <= self.current_station <= 4:
            self.entry_times[self.current_station] = float(timestamp)

    def copy(self) -> "AircraftRuntimeState":
        return AircraftRuntimeState(
            aircraft_id=self.aircraft_id,
            current_station=self.current_station,
            entry_times=dict(self.entry_times),
            exit_times=dict(self.exit_times),
        )


@dataclass
class MultiAircraftState:
    """多架次装配协同调度的全局环境完整快照。"""

    num_aircraft: int
    num_stations: int
    h0: float

    current_time: float = 0.0
    current_cycle: int = 1               # 脉动周期 q in [1, 14]
    last_transfer_time: float = 0.0      # 上次实际脉动转站时刻 P_{q-1}
    transfer_history: list[float] = field(default_factory=list)

    aircraft: dict[int, AircraftRuntimeState] = field(default_factory=dict)
    tasks: dict[str, TaskRuntimeState] = field(default_factory=dict)
    workers: dict[int, WorkerCalendar] = field(default_factory=dict)
    station_worker_bindings: dict[int, list[int]] = field(default_factory=dict)
    normalization_task_count: int = field(init=False)
    _ac_station_tasks: dict[tuple[int, int], list[TaskRuntimeState]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self.normalization_task_count = len(self.tasks)
        self._rebuild_ac_station_tasks()

    def _rebuild_ac_station_tasks(self) -> None:
        """根据当前任务状态全量重建 (aircraft_id, current_station) 增量索引。"""
        self._ac_station_tasks = {}
        for t in self.tasks.values():
            t._state_ref = self
            self._ac_station_tasks.setdefault((t.aircraft_id, t.current_station), []).append(t)

    def move_task_station(self, task: TaskRuntimeState, old_station: int, new_station: int) -> None:
        """当工序跨站后移时，增量维护 (aircraft_id, station_id) 映射。"""
        old_bucket = self._ac_station_tasks.get((task.aircraft_id, old_station))
        if old_bucket and task in old_bucket:
            old_bucket.remove(task)
        new_bucket = self._ac_station_tasks.setdefault((task.aircraft_id, new_station), [])
        if task not in new_bucket:
            new_bucket.append(task)

    def get_aircraft_at_station(self, station_id: int) -> int | None:
        """查询当前物理停靠在指定站位的飞机编号（至多 1 架）。"""
        for k, ac in self.aircraft.items():
            if ac.current_station == station_id:
                return k
        return None

    def get_tasks_for_station(self, station_id: int) -> list[TaskRuntimeState]:
        """获取当前停靠在指定站位的飞机归属在该站的全部活动任务 (O(1) 索引)。"""
        k = self.get_aircraft_at_station(station_id)
        if k is None:
            return []
        return list(self._ac_station_tasks.get((k, station_id), []))

    def get_ready_tasks_for_station(self, station_id: int) -> list[TaskRuntimeState]:
        """获取指定站位当前可被调度的 READY 状态任务列表。"""
        tasks = self.get_tasks_for_station(station_id)
        return [t for t in tasks if t.status == TaskStatus.READY]

    def is_station_cleared_for_transfer(self, station_id: int) -> bool:
        """检查指定站位是否满足放行转站要求。

        放行要求：归属本站的所有未后移任务必须均已完工 (COMPLETED)。
        """
        k = self.get_aircraft_at_station(station_id)
        if k is None:
            # 空站位天然就绪
            return True

        for t in self._ac_station_tasks.get((k, station_id), []):
            # 若任务停留在本站且既未完工也未后移，则站位不可放行
            if t.status not in (TaskStatus.COMPLETED, TaskStatus.POSTPONED):
                return False

        return True

    def is_all_stations_cleared_for_transfer(self) -> bool:
        """检查全线 5 个站位是否全部满足同步脉动转站放行条件。"""
        for s in range(self.num_stations):
            if not self.is_station_cleared_for_transfer(s):
                return False
        return True

    def snapshot(self) -> dict[str, Any]:
        """生成全局状态的完整可复原深拷贝快照。"""
        return {
            "current_time": self.current_time,
            "current_cycle": self.current_cycle,
            "last_transfer_time": self.last_transfer_time,
            "transfer_history": list(self.transfer_history),
            "aircraft": {k: ac.copy() for k, ac in self.aircraft.items()},
            "tasks": {k: t.copy() for k, t in self.tasks.items()},
            "workers": {w: wc.copy() for w, wc in self.workers.items()},
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """从快照完整恢复状态。"""
        self.current_time = float(snap["current_time"])
        self.current_cycle = int(snap["current_cycle"])
        self.last_transfer_time = float(snap["last_transfer_time"])
        self.transfer_history = list(snap["transfer_history"])
        self.aircraft = {k: ac.copy() for k, ac in snap["aircraft"].items()}
        self.tasks = {k: t.copy() for k, t in snap["tasks"].items()}
        self.workers = {w: wc.copy() for w, wc in snap["workers"].items()}
        self._rebuild_ac_station_tasks()


def initialize_multi_aircraft_state(
    baseline_json_path: str = "data/work3/real_283_k10_baseline.json",
) -> MultiAircraftState:
    """由多架次基准计划初始化多架次全局运行状态。"""
    from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline

    baseline = MultiAircraftBaseline.load_from_json(baseline_json_path)

    aircraft_map: dict[int, AircraftRuntimeState] = {
        k: AircraftRuntimeState(aircraft_id=k, current_station=-1)
        for k in range(baseline.num_aircraft)
    }

    tasks_map: dict[str, TaskRuntimeState] = {}
    for key, task in baseline.tasks.items():
        # 0-based station ID internally (0 ~ 4)
        base_station_0 = task.station_id - 1
        tasks_map[key] = TaskRuntimeState(
            aircraft_id=task.aircraft_id,
            task_id=task.task_id,
            task_key=key,
            base_station=base_station_0,
            current_station=base_station_0,
            status=TaskStatus.UNREADY,
            duration=task.duration,
            in_station_offset=task.in_station_offset,
            demand=task.demand,
            skill=task.skill,
            ao_code=task.ao_code,
            predecessors=task.predecessors,
            base_team=tuple(task.team),
        )

    # 构造工人日历与站位绑定
    workers_map: dict[int, WorkerCalendar] = {}
    bindings_0: dict[int, list[int]] = {}
    for s_1based, w_list in baseline.station_workers.items():
        s_0based = s_1based - 1
        bindings_0[s_0based] = list(w_list)
        for w in w_list:
            workers_map[w] = WorkerCalendar(worker_id=w, station_id=s_0based)

    state = MultiAircraftState(
        num_aircraft=baseline.num_aircraft,
        num_stations=baseline.num_stations,
        h0=baseline.h0,
        current_time=0.0,
        current_cycle=1,
        last_transfer_time=0.0,
        transfer_history=[],
        aircraft=aircraft_map,
        tasks=tasks_map,
        workers=workers_map,
        station_worker_bindings=bindings_0,
    )

    return state
