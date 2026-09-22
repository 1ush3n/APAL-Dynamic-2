"""最小离散事件队列引擎白盒单元测试 (Task 2.2)。

测试覆盖：
1. 优先队列时间单调递增性与时间倒退拦截;
2. 同一时刻多事件触发的优先级排序（DISTURBANCE > FINISH > ARRIVE > TRANSFER > START）;
3. 易错点 1 核心攻坚：代数令牌（Generation Token）失效机制，确保被撤销的旧预约绝对不被弹出;
4. peek() 惰性清理与幂等性;
5. 队列重置与时钟重置。
"""

from __future__ import annotations

import pytest

from envs.work3.event_queue import (
    DiscreteEventQueue,
    EventType,
    SimulationEvent,
)


def test_event_queue_time_monotonicity() -> None:
    """测试事件按发生时刻单调递增出队，杜绝时间倒退。"""
    queue = DiscreteEventQueue()

    queue.push(EventType.TASK_FINISH, timestamp=50.0)
    queue.push(EventType.TASK_START, timestamp=10.0)
    queue.push(EventType.MATERIAL_ARRIVE, timestamp=25.0)
    queue.push(EventType.TASK_FINISH, timestamp=5.0)
    queue.push(EventType.SYNCHRONOUS_TRANSFER, timestamp=30.0)

    popped_times: list[float] = []
    while not queue.is_empty():
        ev = queue.pop()
        assert ev is not None
        popped_times.append(ev.timestamp)

    assert popped_times == [5.0, 10.0, 25.0, 30.0, 50.0]
    assert queue.current_time == 50.0

    # 尝试安排过去时刻事件，必须抛出 ValueError
    with pytest.raises(ValueError, match="时间倒退违规"):
        queue.push(EventType.TASK_START, timestamp=40.0)


def test_same_timestamp_priority_ordering() -> None:
    """测试同刻事件发生时的物理优先级排序。"""
    queue = DiscreteEventQueue()
    t = 100.0

    # 故意打乱顺序压入 5 种同刻事件
    queue.push(EventType.TASK_START, timestamp=t, payload={"name": "start"})
    queue.push(EventType.SYNCHRONOUS_TRANSFER, timestamp=t, payload={"name": "transfer"})
    queue.push(EventType.DISTURBANCE, timestamp=t, payload={"name": "disturb"})
    queue.push(EventType.MATERIAL_ARRIVE, timestamp=t, payload={"name": "arrive"})
    queue.push(EventType.TASK_FINISH, timestamp=t, payload={"name": "finish"})

    popped_types: list[EventType] = []
    while not queue.is_empty():
        ev = queue.pop()
        assert ev is not None
        popped_types.append(ev.event_type)

    # 期望优先级：TASK_FINISH(1) -> DISTURBANCE(2) -> MATERIAL_ARRIVE(3) -> SYNCHRONOUS_TRANSFER(4) -> TASK_START(5)
    expected = [
        EventType.TASK_FINISH,
        EventType.DISTURBANCE,
        EventType.MATERIAL_ARRIVE,
        EventType.SYNCHRONOUS_TRANSFER,
        EventType.TASK_START,
    ]
    assert popped_types == expected


def test_old_appointed_event_invalidation() -> None:
    """测试易错点 1（旧预约事件失效原则）：撤销预约后，旧开工事件被彻底丢弃。"""
    queue = DiscreteEventQueue()

    # 1. 任务 0_1 在当前时刻 10.0 安排了未来 30.0 的开工事件 (gen=0)
    queue.push(
        EventType.TASK_START,
        timestamp=30.0,
        task_key="0_1",
        generation=0,
        payload={"action": "old_start_at_30"},
    )

    # 2. 其他工序在 20.0 完工
    queue.push(
        EventType.TASK_FINISH,
        timestamp=20.0,
        task_key="0_2",
        generation=0,
        payload={"action": "other_finish"},
    )

    # 3. 队列弹出第 1 个事件 (20.0)
    ev1 = queue.pop()
    assert ev1 is not None
    assert ev1.timestamp == 20.0
    assert ev1.task_key == "0_2"
    assert queue.current_time == 20.0

    # 4. 在时刻 20.0 发生扰动，将 0_1 的恢复时间推迟至 50.0！
    # 原定于 30.0 的开工事件必须失效！
    # 使 0_1 的 generation 升级为 1
    queue.invalidate_task_events(task_key="0_1", new_generation=1)

    # 5. 任务 0_1 重新安排在 60.0 开工 (gen=1)
    queue.push(
        EventType.TASK_START,
        timestamp=60.0,
        task_key="0_1",
        generation=1,
        payload={"action": "new_start_at_60"},
    )

    # 6. 另一个任务在 40.0 完工
    queue.push(
        EventType.TASK_FINISH,
        timestamp=40.0,
        task_key="0_3",
        generation=0,
        payload={"action": "task_0_3_finish"},
    )

    # 7. 连续弹出后续事件：
    # 期望：原定 30.0 的旧开工事件直接被惰性丢弃，跳过它，直接弹出 40.0 的任务，再弹出 60.0 的新任务！
    ev2 = queue.pop()
    assert ev2 is not None
    assert ev2.timestamp == 40.0
    assert ev2.task_key == "0_3"

    ev3 = queue.pop()
    assert ev3 is not None
    assert ev3.timestamp == 60.0
    assert ev3.task_key == "0_1"
    assert ev3.payload["action"] == "new_start_at_60"

    # 队列为空
    assert queue.is_empty()


def test_peek_and_cancellation() -> None:
    """测试 peek() 查看堆顶与显式 is_cancelled 标记。"""
    queue = DiscreteEventQueue()

    ev1 = queue.push(EventType.TASK_FINISH, timestamp=15.0)
    ev2 = queue.push(EventType.TASK_START, timestamp=25.0)

    # 查看堆顶
    top = queue.peek()
    assert top is not None
    assert top.timestamp == 15.0
    assert queue.current_time == 0.0  # peek 不推进时钟

    # 手动取消 ev1
    ev1.is_cancelled = True

    # 再次 peek，自动略过 ev1，看到 ev2
    new_top = queue.peek()
    assert new_top is not None
    assert new_top.timestamp == 25.0


def test_queue_reset() -> None:
    """测试队列清空与重置。"""
    queue = DiscreteEventQueue()
    queue.push(EventType.TASK_FINISH, timestamp=100.0)
    queue.pop()
    assert queue.current_time == 100.0

    queue.reset(start_time=0.0)
    assert queue.is_empty()
    assert queue.current_time == 0.0
    assert queue.size() == 0
