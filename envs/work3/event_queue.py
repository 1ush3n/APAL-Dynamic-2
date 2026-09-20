"""工作三最小离散事件队列引擎 (Task 2.2)。

统一支持核心事件推进：
1. DISTURBANCE: 扰动在 tau 时刻揭示，优先更新状态并撤销非法预约；
2. TASK_FINISH: 任务完工、释放工人、解锁后继；
3. MATERIAL_ARRIVE: 物料/可用性到达、就绪解锁；
4. SYNCHRONOUS_TRANSFER: 全线同步转站脉动；
5. TASK_START: 预约任务到达开工时刻，转为 RUNNING。

设计关键：
- 基于 heapq 优先队列，严格保证时间单调递增不回退；
- 同刻事件优先级严格排序，杜绝瞬时死锁与重叠；
- 内置代数标记（Generation Token）无效化机制，彻底攻坚“易错点 1（旧预约事件失效原则）”。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable


class EventType(IntEnum):
    """离散事件类型定义及同刻执行优先级。

    同刻优先级原则（数值越小优先级越高）：
    1. DISTURBANCE: 扰动先揭示，以便在开工前撤销冲突预约；
    2. TASK_FINISH: 先完工并释放工人与槽位；
    3. MATERIAL_ARRIVE: 物料到达解锁就绪；
    4. SYNCHRONOUS_TRANSFER: 本周期全部完工后脉动转站；
    5. TASK_START: 预约工序开工（此时工人已完成前序释放）。
    """

    DISTURBANCE = 1
    TASK_FINISH = 2
    MATERIAL_ARRIVE = 3
    SYNCHRONOUS_TRANSFER = 4
    TASK_START = 5


@dataclass(order=True)
class SimulationEvent:
    """离散仿真事件元组。"""

    timestamp: float
    priority: int
    event_id: int

    event_type: EventType = field(compare=False)
    task_key: str | None = field(compare=False, default=None)
    generation: int = field(compare=False, default=0)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)
    is_cancelled: bool = field(compare=False, default=False)


class DiscreteEventQueue:
    """基于最小堆的高性能离散事件队列。"""

    def __init__(self) -> None:
        self._heap: list[SimulationEvent] = []
        self._event_counter: int = 0
        self._current_time: float = 0.0

        # task_key -> 期望的最新 generation。若弹出的 event.generation < expected，则视作过期失效
        self._task_generations: dict[str, int] = {}

    @property
    def current_time(self) -> float:
        """当前仿真推进到的时间点。"""
        return self._current_time

    def size(self) -> int:
        """队列中当前剩余事件总数（含待延迟惰性丢弃的事件）。"""
        return len(self._heap)

    def is_empty(self) -> bool:
        """队列是否为空。"""
        return len(self._heap) == 0

    def push(
        self,
        event_type: EventType,
        timestamp: float,
        task_key: str | None = None,
        generation: int = 0,
        payload: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        """压入新事件。

        Args:
            event_type: 事件枚举类型.
            timestamp: 触发时刻 (必须 >= current_time).
            task_key: 关联的任务全局唯一标识 (可选).
            generation: 压入时该工序的版本代数.
            payload: 事件携带的上下文负载数据 (可选).

        Returns:
            生成的 SimulationEvent 对象.
        """
        if timestamp < self._current_time - 1e-5:
            raise ValueError(
                f"时间倒退违规！当前仿真时刻为 {self._current_time:.4f}，"
                f"尝试安排过去时刻的事件: {timestamp:.4f} (type={event_type.name})"
            )

        self._event_counter += 1
        event = SimulationEvent(
            timestamp=float(timestamp),
            priority=int(event_type),
            event_id=self._event_counter,
            event_type=event_type,
            task_key=task_key,
            generation=int(generation),
            payload=payload or {},
        )

        if task_key is not None:
            # 记录当前已知的最大 generation
            curr_gen = self._task_generations.get(task_key, 0)
            if generation > curr_gen:
                self._task_generations[task_key] = generation

        heapq.heappush(self._heap, event)
        return event

    def invalidate_task_events(self, task_key: str, new_generation: int) -> None:
        """使指定任务之前安排的所有事件（如旧预约开工事件）立即失效。

        核心实现：更新 task_key 的最新代数，后续弹出该代数之前的旧事件将被自动丢弃。
        """
        self._task_generations[task_key] = int(new_generation)

    def peek(self) -> SimulationEvent | None:
        """查看堆顶下一个有效事件（不弹出），自动剔除已失效的过期事件。"""
        while self._heap:
            top = self._heap[0]
            if self._is_event_valid(top):
                return top
            # 惰性丢弃无效事件
            heapq.heappop(self._heap)
        return None

    def pop(self) -> SimulationEvent | None:
        """弹出下一个有效事件，并将仿真时钟推进至该事件发生时刻。"""
        while self._heap:
            event = heapq.heappop(self._heap)
            if not self._is_event_valid(event):
                # 惰性丢弃过期或已取消事件
                continue

            # 推进仿真时钟，严格单调不回退
            if event.timestamp > self._current_time:
                self._current_time = event.timestamp

            return event

        return None

    def _is_event_valid(self, event: SimulationEvent) -> bool:
        """判定事件是否依然有效。"""
        if event.is_cancelled:
            return False
        if event.task_key is not None:
            expected_gen = self._task_generations.get(event.task_key, 0)
            # 仅当事件代数等于当前任务最新代数时有效
            if event.generation < expected_gen:
                return False
        return True

    def reset(self, start_time: float = 0.0) -> None:
        """清空重置事件队列。"""
        self._heap.clear()
        self._event_counter = 0
        self._current_time = float(start_time)
        self._task_generations.clear()
