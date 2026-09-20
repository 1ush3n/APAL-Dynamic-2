"""工作三核心状态模型与工序状态机白盒单元测试 (Task 2.1)。

测试覆盖：
1. 工序六状态机完整流转与合法性判定;
2. 预约取消与版本代数 (Generation) 自增机制;
3. 合法后移状态继承与物料恢复时间 R 属性持久保留;
4. 单工人日历区间排他性、空隙搜索与撤销释放;
5. 飞机在场物理位置 s_k(t) 步进与完工出线;
6. 多机全局状态快照 (Snapshot) 与复原 (Restore) 精确一致性;
7. 站位放行与全线转站前置条件核验。
"""

from __future__ import annotations

import pytest
from pathlib import Path

from envs.work3.core_types import (
    ActionBranch,
    AircraftRuntimeState,
    MultiAircraftState,
    TaskRuntimeState,
    TaskStatus,
    TimeInterval,
    WorkerCalendar,
    initialize_multi_aircraft_state,
)


def test_task_status_lifecycle() -> None:
    """测试常规工序生命周期：UNREADY -> READY -> RESERVED -> RUNNING -> COMPLETED。"""
    task = TaskRuntimeState(
        aircraft_id=0,
        task_id=10,
        task_key="0_10",
        base_station=0,
        current_station=0,
        status=TaskStatus.UNREADY,
        duration=2.5,
        in_station_offset=1.0,
        demand=2,
        skill=1,
        ao_code="AO_010",
        predecessors=(),
        material_ready_time=15.0,
    )

    assert task.status == TaskStatus.UNREADY
    assert not task.can_physically_start(current_time=10.0)
    assert task.can_physically_start(current_time=15.0)

    # 1. 前驱完成后转为 READY
    task.status = TaskStatus.READY

    # 2. 策略指派团队并建立预约
    team = [12, 72]
    task.reserve(team=team, scheduled_start=15.0)
    assert task.status == TaskStatus.RESERVED
    assert task.assigned_team == [12, 72]
    assert task.scheduled_start == 15.0

    # 3. 到达开工时刻
    task.start_work(current_time=15.0)
    assert task.status == TaskStatus.RUNNING
    assert task.actual_start == 15.0

    # 4. 加工完毕完工
    task.complete_work(current_time=17.5)
    assert task.status == TaskStatus.COMPLETED
    assert task.actual_end == 17.5


def test_reservation_cancellation_and_generation() -> None:
    """测试预约取消（易错点 1 基石）：状态退回 READY，清空团队，代数自增。"""
    task = TaskRuntimeState(
        aircraft_id=1,
        task_id=20,
        task_key="1_20",
        base_station=1,
        current_station=1,
        status=TaskStatus.READY,
        duration=3.0,
        in_station_offset=2.0,
        demand=1,
        skill=0,
        ao_code="AO_020",
        predecessors=(),
    )
    initial_gen = task.generation

    # 建立预约
    task.reserve(team=[5], scheduled_start=25.0)
    assert task.status == TaskStatus.RESERVED
    assert task.generation == initial_gen

    # 受扰后撤销预约
    task.cancel_reservation()
    assert task.status == TaskStatus.READY
    assert task.assigned_team == []
    assert task.scheduled_start is None
    assert task.generation == initial_gen + 1, "撤销预约必须使代数自增以失效旧开工事件"


def test_task_postponement_and_retention() -> None:
    """测试合法后移与物料下界持久继承（易错点 2 基石）。"""
    task = TaskRuntimeState(
        aircraft_id=2,
        task_id=30,
        task_key="2_30",
        base_station=1,
        current_station=1,
        status=TaskStatus.READY,
        duration=4.0,
        in_station_offset=0.5,
        demand=2,
        skill=2,
        ao_code="AO_030",
        predecessors=(),
        material_ready_time=120.0,  # 受扰后的恢复时间 R
    )

    # 执行合法后移
    task.postpone_to_next_station()
    assert task.status == TaskStatus.POSTPONED
    assert task.current_station == 2, "后移后目标站位必须步进至下一站"
    assert task.postpone_count == 1
    assert task.material_ready_time == 120.0, "后移后恢复时间 R 必须严格保留，不得遗失"

    # 飞机转站到达新站，重新唤醒
    task.awaken_in_station(all_predecessors_completed=True)
    assert task.status == TaskStatus.READY
    assert task.material_ready_time == 120.0, "新站就绪后依然受限于原恢复时间 R"
    assert not task.can_physically_start(current_time=100.0)
    assert task.can_physically_start(current_time=120.0)


def test_worker_calendar_operations() -> None:
    """测试工人日历可用性检测、空隙搜索与区间取消。"""
    wc = WorkerCalendar(worker_id=10, station_id=1)

    # 初始全空闲
    assert wc.is_available(0.0, 10.0)

    # 添加 [10, 20] 占用
    wc.add_interval(10.0, 20.0, "task_A")
    assert not wc.is_available(5.0, 15.0)
    assert not wc.is_available(12.0, 18.0)
    assert wc.is_available(0.0, 10.0)
    assert wc.is_available(20.0, 30.0)

    # 添加 [30, 40] 占用
    wc.add_interval(30.0, 40.0, "task_B")

    # 搜索能够容纳 8 小时工期的最早空隙 (从 5 开始)
    # [5, 10] 仅有 5 小时不足；[20, 30] 有 10 小时，能够容纳！
    earliest_slot = wc.find_earliest_slot(search_start=5.0, duration=8.0)
    assert earliest_slot == 20.0

    # 搜索能够容纳 15 小时工期的最早空隙 (从 5 开始)
    # [20, 30] 只有 10 小时不足；最早要在 [40, 55]
    earliest_large = wc.find_earliest_slot(search_start=5.0, duration=15.0)
    assert earliest_large == 40.0

    # 撤销 task_A，[10, 20] 被释放
    assert wc.remove_interval("task_A")
    assert wc.is_available(10.0, 20.0)
    assert not wc.remove_interval("non_existent")


def test_aircraft_state_and_stepping() -> None:
    """测试飞机位置 s_k in {-1, 0, 1, 2, 3, 4, 5} 的物理流转与状态判断。"""
    ac = AircraftRuntimeState(aircraft_id=0, current_station=-1)

    assert not ac.is_in_factory
    assert not ac.is_completed

    # 进入站位 0 (站位 1)
    ac.step_to_next_station(timestamp=0.0)
    assert ac.current_station == 0
    assert ac.is_in_factory
    assert ac.entry_times[0] == 0.0

    # 依次流转过站
    for expected_s in range(1, 5):
        t = expected_s * 292.0
        ac.step_to_next_station(timestamp=t)
        assert ac.current_station == expected_s
        assert ac.is_in_factory
        assert ac.exit_times[expected_s - 1] == t
        assert ac.entry_times[expected_s] == t

    # 离开末站 (进入 5)
    ac.step_to_next_station(timestamp=1460.0)
    assert ac.current_station == 5
    assert not ac.is_in_factory
    assert ac.is_completed


def test_multi_aircraft_state_initialization_and_snapshot() -> None:
    """测试全局多机环境状态初始化、快照与复原。"""
    baseline_path = "data/work3/real_283_k10_baseline.json"
    if not Path(baseline_path).is_file():
        pytest.skip(f"{baseline_path} 不存在")

    state = initialize_multi_aircraft_state(baseline_path)

    assert state.num_aircraft == 10
    assert state.num_stations == 5
    assert len(state.aircraft) == 10
    assert len(state.tasks) == 2830
    assert len(state.workers) == 25

    # 模拟发生一些状态改变
    state.current_time = 105.5
    state.current_cycle = 2
    state.last_transfer_time = 100.0
    state.aircraft[0].current_station = 1
    sample_key = "0_2"
    state.tasks[sample_key].reserve(team=[12, 72], scheduled_start=110.0)
    state.workers[12].add_interval(110.0, 115.0, sample_key)

    # 拍摄快照
    snap = state.snapshot()

    # 进一步扰动破坏当前状态
    state.current_time = 999.0
    state.current_cycle = 99
    state.tasks[sample_key].status = TaskStatus.COMPLETED
    state.workers[12].intervals.clear()

    # 从快照精确还原
    state.restore(snap)
    assert state.current_time == 105.5
    assert state.current_cycle == 2
    assert state.last_transfer_time == 100.0
    assert state.tasks[sample_key].status == TaskStatus.RESERVED
    assert state.tasks[sample_key].scheduled_start == 110.0
    assert len(state.workers[12].intervals) == 1


def test_station_transfer_clearance_logic() -> None:
    """测试全线同步转站放行判定逻辑。"""
    baseline_path = "data/work3/real_283_k10_baseline.json"
    if not Path(baseline_path).is_file():
        pytest.skip(f"{baseline_path} 不存在")

    state = initialize_multi_aircraft_state(baseline_path)

    # 初始时，所有飞机都在 -1 (线外)，所有站位均无在场飞机 -> 空站天然放行
    assert state.is_all_stations_cleared_for_transfer()

    # 飞机 0 进驻站位 0
    state.aircraft[0].current_station = 0
    # 此时飞机 0 在站位 0 的任务全为 UNREADY，站位 0 不可放行
    assert not state.is_station_cleared_for_transfer(0)
    assert not state.is_all_stations_cleared_for_transfer()

    # 将飞机 0 在站位 0 的所有任务设为 COMPLETED (除一道后移任务外)
    st0_tasks = [
        t for t in state.tasks.values() if t.aircraft_id == 0 and t.current_station == 0
    ]
    for t in st0_tasks[:-1]:
        t.status = TaskStatus.COMPLETED

    # 剩余最后一道工序后移至下一站
    st0_tasks[-1].postpone_to_next_station()

    # 现在站位 0 的本站任务全部处于 COMPLETED 或 POSTPONED，站位 0 满足放行条件！
    assert state.is_station_cleared_for_transfer(0)
    assert state.is_all_stations_cleared_for_transfer()
