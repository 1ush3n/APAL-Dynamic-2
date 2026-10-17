"""工作三核心数据结构、状态机枚举与日历模型。

严格落实顶层设计规范：
1. 工序六状态机：UNREADY, READY, RESERVED, RUNNING, COMPLETED, POSTPONED；
2. 飞机位置数组：s_k(t) in {-1, 0, 1, 2, 3, 4, 5}；
3. 工人区间日历与空隙搜索；
4. 状态快照、复原与重置。
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
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


ADVANCE_CAUSES = (
    "explicit",
    "identical_resubmission",
    "forced_no_candidate",
    "auto_after_action",
    "none",
)


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

    assigned_team: list[int] = field(default_factory=list)  # 当前有效预约/执行团队；后移后为空
    scheduled_start: float | None = None  # 当前有效预约/执行开工时刻；后移后为 None
    actual_start: float | None = None
    actual_start_station: int | None = None
    actual_end: float | None = None

    material_ready_time: float = 0.0  # 兼容字段名：工序开工可用下界R（受扰后继承）
    postpone_count: int = 0           # 累计正式改站/后移次数 n_{ki}
    generation: int = 0               # 预约版本代数，用于失效令牌校验
    base_team: tuple[int, ...] = field(default_factory=tuple)  # 基准标准团队 W_i^0
    fixed_station: int | None = None  # 原始实例固定站位（0-based）
    max_allowed_station: int | None = None  # 工艺后继约束允许的最晚站位（0-based）
    execution_duration: float | None = None  # 本次团队对应的实际工时
    cycle_start_time: float | None = None  # 实际开工所在周期的转站时刻 P_{q-1}
    start_cost_confirmed: bool = False     # 是否已确认并结算开工偏差与团队替换费用
    baseline_assignment: dict[str, Any] = field(default_factory=dict)
    last_published_assignment: dict[str, Any] = field(default_factory=dict)  # 历史比较锚点，不代表当前资源预约
    revision_history: list[dict[str, Any]] = field(default_factory=list)
    alignment: int = 0                # 留站对齐模式：0=ASAP (尽快开工), 1=周期内标准偏置对齐
    committed: bool = False           # 是否已提交不可撤回决策（一旦预约或后移确定即不可主动改派）
    is_physical: bool = True          # 是否为实际物理加工工序
    node_type: int = 2                # 节点类型：1=虚拟摘要/子装配节点, 2=物理加工Task
    _state_ref: Any = field(default=None, repr=False, compare=False)

    @property
    def effective_assignment(self) -> Mapping[str, Any]:
        """只读当前安排投影；后移任务的团队、开工和执行工时均未确定。"""
        return MappingProxyType(
            {
                "status": self.status,
                "station": int(self.current_station),
                "team": tuple(self.assigned_team),
                "scheduled_start": self.scheduled_start,
                "execution_duration": self.execution_duration,
            }
        )

    @property
    def publication_reference(self) -> Mapping[str, Any]:
        """只读上一正式发布比较参照；其中团队不意味着当前站已占用该团队。"""
        assignment = self.last_published_assignment or self.baseline_assignment
        return MappingProxyType(
            {
                "station": assignment.get("station"),
                "team": tuple(assignment.get("team") or ()),
                "position": assignment.get("position"),
            }
        )

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

    def reserve(self, team: Sequence[int], scheduled_start: float, alignment: int = 0) -> None:
        """建立不可撤回的未来开工预约决定。"""
        self.assigned_team = [int(w) for w in team]
        self.scheduled_start = float(scheduled_start)
        self.alignment = int(alignment)
        self.status = TaskStatus.RESERVED
        self.committed = True

    def defer_reservation(self, new_scheduled_start: float, new_duration: float | None = None) -> None:
        """被动顺延预约开工时间，保持已提交的团队与对齐决策不变。"""
        self.scheduled_start = float(new_scheduled_start)
        if new_duration is not None:
            self.execution_duration = float(new_duration)
        self.generation += 1

    def cancel_reservation(self) -> None:
        """取消当前预约，退回就绪状态，并递增版本代数使事件队列旧开工事件失效。"""
        self.assigned_team = []
        self.scheduled_start = None
        self.execution_duration = None
        self.status = TaskStatus.READY
        self.committed = False
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
            alignment=self.alignment,
            committed=self.committed,
            is_physical=getattr(self, "is_physical", True),
            node_type=getattr(self, "node_type", 2),
        )
        copied._state_ref = getattr(self, "_state_ref", None)
        return copied


def remaining_processing_time(task: TaskRuntimeState, current_time: float) -> float:
    """返回任务在当前快照中的剩余加工工时，不计预约开始前的等待。"""
    if task.status == TaskStatus.COMPLETED:
        return 0.0
    duration = float(
        task.execution_duration
        if task.execution_duration is not None
        else task.duration
    )
    if task.status == TaskStatus.RUNNING and task.actual_start is not None:
        return max(0.0, float(task.actual_start) + duration - float(current_time))
    return max(0.0, duration)


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


import math


@dataclass(frozen=True)
class CycleExecutionRecord:
    """单个脉动周期的实际执行与结算事实记录 (R01)。"""

    cycle_id: int
    cycle_start_time: float
    transfer_time: float
    occupied_stations: tuple[bool, ...]
    aircraft_at_stations: tuple[int | None, ...]
    station_first_cleared_times: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_id", int(self.cycle_id))
        object.__setattr__(self, "cycle_start_time", float(self.cycle_start_time))
        object.__setattr__(self, "transfer_time", float(self.transfer_time))
        object.__setattr__(self, "occupied_stations", tuple(bool(x) for x in self.occupied_stations))
        object.__setattr__(
            self,
            "aircraft_at_stations",
            tuple(int(x) if x is not None else None for x in self.aircraft_at_stations),
        )
        if isinstance(self.station_first_cleared_times, Mapping):
            cleared = {int(k): float(v) for k, v in self.station_first_cleared_times.items()}
        else:
            cleared = {int(k): float(v) for k, v in dict(self.station_first_cleared_times).items()}
        object.__setattr__(self, "station_first_cleared_times", MappingProxyType(cleared))

    def __deepcopy__(self, memo: dict[int, Any]) -> CycleExecutionRecord:
        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return (
            type(self),
            (
                self.cycle_id,
                self.cycle_start_time,
                self.transfer_time,
                self.occupied_stations,
                self.aircraft_at_stations,
                dict(self.station_first_cleared_times),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": int(self.cycle_id),
            "cycle_start_time": float(self.cycle_start_time),
            "transfer_time": float(self.transfer_time),
            "occupied_stations": tuple(self.occupied_stations),
            "aircraft_at_stations": tuple(self.aircraft_at_stations),
            "station_first_cleared_times": {int(k): float(v) for k, v in self.station_first_cleared_times.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CycleExecutionRecord:
        cleared = data.get("station_first_cleared_times", {})
        return cls(
            cycle_id=int(data["cycle_id"]),
            cycle_start_time=float(data["cycle_start_time"]),
            transfer_time=float(data["transfer_time"]),
            occupied_stations=tuple(bool(x) for x in data["occupied_stations"]),
            aircraft_at_stations=tuple(
                int(x) if x is not None else None for x in data["aircraft_at_stations"]
            ),
            station_first_cleared_times={int(k): float(v) for k, v in cleared.items()},
        )


def validate_cycle_execution_record(rec: CycleExecutionRecord, tolerance: float = 1e-6) -> None:
    """原子校验周期执行事实记录的物理合法性与数据完整性 (U03-U04/N01)。"""
    if not isinstance(rec, CycleExecutionRecord):
        raise TypeError(f"期望 CycleExecutionRecord，收到: {type(rec)}")
    if rec.cycle_id < 1:
        raise ValueError(f"周期号非法: {rec.cycle_id}")
    if not math.isfinite(rec.cycle_start_time) or rec.cycle_start_time < -tolerance:
        raise ValueError(f"周期起点时间非有限或为负: {rec.cycle_start_time}")
    if not math.isfinite(rec.transfer_time) or rec.transfer_time < rec.cycle_start_time - tolerance:
        raise ValueError(f"脉动时刻 {rec.transfer_time} 早于起点 {rec.cycle_start_time}")
    if len(rec.occupied_stations) != 5:
        raise ValueError(f"占用站位数必须为5，当前为: {len(rec.occupied_stations)}")
    if len(rec.aircraft_at_stations) != 5:
        raise ValueError(f"站位飞机数必须为5，当前为: {len(rec.aircraft_at_stations)}")

    occupied_ac_ids: list[int] = []
    for s in range(5):
        is_occ = rec.occupied_stations[s]
        ac_id = rec.aircraft_at_stations[s]
        if is_occ:
            if ac_id is None:
                raise ValueError(f"站位 {s} 标记为占用但飞机ID为None")
            if ac_id < 0:
                raise ValueError(f"站位 {s} 飞机ID非法 (为负): {ac_id}")
            occupied_ac_ids.append(ac_id)
            if s not in rec.station_first_cleared_times:
                raise ValueError(f"占用站位 {s} (飞机 {ac_id}) 缺少首次就绪时刻 F[m, q]")
            f_val = rec.station_first_cleared_times[s]
            if not math.isfinite(f_val):
                raise ValueError(f"占用站位 {s} 的就绪时刻 F 非有限数值: {f_val}")
            if f_val < rec.cycle_start_time - tolerance:
                raise ValueError(f"占用站位 {s} 的就绪时刻 {f_val} 早于周期起点 {rec.cycle_start_time}")
            if f_val > rec.transfer_time + tolerance:
                raise ValueError(f"占用站位 {s} 的就绪时刻 {f_val} 晚于脉动时刻 {rec.transfer_time}")
        else:
            if ac_id is not None:
                raise ValueError(f"站位 {s} 标记为空但存在飞机ID {ac_id}")

    if len(occupied_ac_ids) != len(set(occupied_ac_ids)):
        raise ValueError(f"占用站位出现重复飞机ID: {occupied_ac_ids}")


def parse_and_validate_step_cycle_records(
    raw_records: Any,
    *,
    before_cycle: int | None = None,
    after_cycle: int | None = None,
) -> list[CycleExecutionRecord]:
    """从环境步返回的 info['cycle_execution_records'] 解析并严格校验连续周期记录 (R1/N01)。"""
    if raw_records is None:
        raise ValueError("缺少周期执行记录 (cycle_execution_records is None)")

    parsed_records: list[CycleExecutionRecord] = []
    if isinstance(raw_records, dict):
        items = list(raw_records.items())
        items.sort(key=lambda kv: int(kv[0]))
        for cycle_id_key, rec_data in items:
            if isinstance(rec_data, CycleExecutionRecord):
                rec = rec_data
            elif isinstance(rec_data, dict):
                rec = CycleExecutionRecord.from_dict(rec_data)
            else:
                raise TypeError(f"未知的周期记录类型: {type(rec_data)}")
            if rec.cycle_id != int(cycle_id_key):
                raise ValueError(f"周期字典键 {cycle_id_key} 与记录内周期号 {rec.cycle_id} 不一致")
            validate_cycle_execution_record(rec)
            parsed_records.append(rec)
    elif isinstance(raw_records, (list, tuple)):
        for item in raw_records:
            if isinstance(item, CycleExecutionRecord):
                rec = item
            elif isinstance(item, dict):
                rec = CycleExecutionRecord.from_dict(item)
            else:
                raise TypeError(f"未知的周期记录类型: {type(item)}")
            validate_cycle_execution_record(rec)
            parsed_records.append(rec)
        parsed_records.sort(key=lambda r: r.cycle_id)
    else:
        raise TypeError(f"未知的原始周期记录集合类型: {type(raw_records)}")

    # 校验周期连续性与起止衔接
    if before_cycle is not None and after_cycle is not None:
        expected_count = max(0, int(after_cycle) - int(before_cycle))
        if len(parsed_records) != expected_count:
            raise ValueError(
                f"周期记录数量与周期跨度不符：before={before_cycle}, after={after_cycle}, "
                f"期望 {expected_count} 条，实际收到 {len(parsed_records)} 条"
            )
        for idx, rec in enumerate(parsed_records):
            expected_cid = int(before_cycle) + idx
            if rec.cycle_id != expected_cid:
                raise ValueError(
                    f"周期记录不连续：第 {idx} 条记录周期号为 {rec.cycle_id}，期望为 {expected_cid}"
                )

    for i in range(len(parsed_records) - 1):
        r_curr = parsed_records[i]
        r_next = parsed_records[i + 1]
        if r_next.cycle_id != r_curr.cycle_id + 1:
            raise ValueError(f"周期号不连续: {r_curr.cycle_id} -> {r_next.cycle_id}")
        if abs(r_curr.transfer_time - r_next.cycle_start_time) > 1e-4:
            raise ValueError(
                f"相邻周期起止时刻不衔接：周期 {r_curr.cycle_id} 脉动时刻 {r_curr.transfer_time} "
                f"!= 周期 {r_next.cycle_id} 起点 {r_next.cycle_start_time}"
            )

    return parsed_records



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

    # 首次实际就绪时刻跟踪与高频动态演化语义摘要 (R05 & R08)
    station_first_cleared_times: dict[int, dict[int, float]] = field(default_factory=dict)
    cycle_execution_records: dict[int, CycleExecutionRecord] = field(default_factory=dict)
    last_action_branch: int = 0          # 上一决策动作分支 (0: NONE, 1: STAY, 2: POSTPONE, 3: ADVANCE)
    last_action_task_id: int = -1
    last_action_aircraft_id: int = -1
    last_action_station_id: int = -1
    micro_steps_in_timestamp: int = 0    # 同刻微决策轮数计数
    recent_postponed_count: int = 0      # 当前周期内后移工序数
    recent_reserved_count: int = 0       # 当前周期内预约工序数
    recent_events_count: int = 0         # 自上一决策以来发生的离散事件数
    recent_material_delays: int = 0      # 自上一决策以来发生的物料延迟事件数
    recent_worker_disturbances: int = 0  # 自上一决策以来发生的人员扰动事件数

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

    def get_next_cycle_station_mask(self) -> list[bool]:
        """查询下一周期 q+1 各站位是否有飞机在场（空站无效，有飞机有效）。"""
        mask = [False] * self.num_stations
        # 站位 0: 下一架待进线飞机 k = current_cycle (0-indexed)
        if self.current_cycle < self.num_aircraft:
            mask[0] = True
        # 站位 1..4: 当前站位 s-1 是否有飞机停靠
        for s in range(1, self.num_stations):
            if self.get_aircraft_at_station(s - 1) is not None:
                mask[s] = True
        return mask

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
            "station_first_cleared_times": copy.deepcopy(self.station_first_cleared_times),
            "cycle_execution_records": copy.deepcopy(self.cycle_execution_records),
            "last_action_branch": self.last_action_branch,
            "last_action_task_id": self.last_action_task_id,
            "last_action_aircraft_id": self.last_action_aircraft_id,
            "last_action_station_id": self.last_action_station_id,
            "micro_steps_in_timestamp": self.micro_steps_in_timestamp,
            "recent_postponed_count": self.recent_postponed_count,
            "recent_reserved_count": self.recent_reserved_count,
            "recent_events_count": self.recent_events_count,
            "recent_material_delays": self.recent_material_delays,
            "recent_worker_disturbances": self.recent_worker_disturbances,
            "recent_change_summary": copy.deepcopy(getattr(self, "recent_change_summary", {})),
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
        self.station_first_cleared_times = copy.deepcopy(snap.get("station_first_cleared_times", {}))
        self.cycle_execution_records = copy.deepcopy(snap.get("cycle_execution_records", {}))
        self.last_action_branch = int(snap.get("last_action_branch", 0))
        self.last_action_task_id = int(snap.get("last_action_task_id", -1))
        self.last_action_aircraft_id = int(snap.get("last_action_aircraft_id", -1))
        self.last_action_station_id = int(snap.get("last_action_station_id", -1))
        self.micro_steps_in_timestamp = int(snap.get("micro_steps_in_timestamp", 0))
        self.recent_postponed_count = int(snap.get("recent_postponed_count", 0))
        self.recent_reserved_count = int(snap.get("recent_reserved_count", 0))
        self.recent_events_count = int(snap.get("recent_events_count", 0))
        self.recent_material_delays = int(snap.get("recent_material_delays", 0))
        self.recent_worker_disturbances = int(snap.get("recent_worker_disturbances", 0))
        self.recent_change_summary = copy.deepcopy(snap.get("recent_change_summary", {}))
        self.aircraft = {k: ac.copy() for k, ac in snap["aircraft"].items()}
        self.tasks = {k: t.copy() for k, t in snap["tasks"].items()}
        self.workers = {w: wc.copy() for w, wc in snap["workers"].items()}
        for t in self.tasks.values():
            t._state_ref = self
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
