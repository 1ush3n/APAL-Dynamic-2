"""易错点 1 攻坚白盒单元测试：突发扰动导致旧预约事件失效与工人日历释放 (Task 4.2)。

核心断言：
1. 工序在未来排定预约开工时刻 t_sched，工人日历被预占；
2. 在 tau < t_sched 时刻发生突发可用性扰动，推迟恢复时刻至 R > t_sched；
3. 扰动揭示瞬间：
   - 必须立刻撤销工人日历的预占区间，工人重新恢复可用；
   - 递增代数令牌 (Generation Token)，使原 TASK_START 事件失效；
   - 任务退回 UNREADY，并在 R 时刻挂载 MATERIAL_ARRIVE 事件；
4. 仿真时钟越过原排定时刻 t_sched 时，绝不开工；
5. 时钟推进至 R 时，工序恢复 READY，且实际开工时刻必定 >= R。
"""

from __future__ import annotations

from pathlib import Path
import pytest

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from envs.work3.event_queue import EventType


@pytest.fixture
def baseline_path() -> str:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return str(path)


def test_reserved_task_invalidation_on_disturbance(baseline_path: str) -> None:
    """测试已预约工序在扰动推迟物料后，预约被取消、工人日历被释放且旧开工事件失效。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    ready = env.get_ready_tasks()
    assert len(ready) >= 2

    task = ready[0]
    team = tuple(env.state.station_worker_bindings[task.current_station][: task.demand])

    # 1. 制造一个未来预约 (align=1 且搜索起点推后，确保 t_sched > current_time)
    task.in_station_offset = 20.0  # 人为设置较大的对齐偏移，使其预约在 20.0h
    env.step({
        "task_key": task.task_key,
        "branch": ActionBranch.STATION_EXECUTE,
        "team": team,
        "align": 1,
    })

    assert task.status == TaskStatus.RESERVED
    assert task.scheduled_start is not None and task.scheduled_start >= 20.0
    t_sched = task.scheduled_start
    gen_before = task.generation

    # 检查指派工人的日历已被预占
    for w in team:
        assert not env.state.workers[w].is_available(t_sched, t_sched + task.duration)

    # 2. 在 tau = 5.0h 触发扰动，物料恢复时间推迟至 R = 35.0h (R > t_sched)
    tau = 5.0
    r_recovery = 35.0
    scenario_payload = {
        "tau": tau,
        "recovery_time": r_recovery,
        "affected_task_keys": [task.task_key],
    }

    env.load_scenario(scenario_payload)

    # 推进时钟至 tau，触发 DISTURBANCE 事件
    # 模拟外部推进或事件驱动
    while env.state.current_time < tau:
        event = env.event_queue.pop()
        if event is None:
            break
        env.state.current_time = event.timestamp
        if event.event_type == EventType.DISTURBANCE:
            env._handle_disturbance_event(event.timestamp, event.payload)
            break

    # ======================== 易错点 1 核心断言 ========================
    # 断言 1: 工序已退回 UNREADY，且版本代数递增
    assert task.status == TaskStatus.UNREADY
    assert task.generation > gen_before
    assert task.scheduled_start is None
    assert task.assigned_team == []

    # 断言 2: 工人日历预占已被完全释放
    for w in team:
        assert env.state.workers[w].is_available(t_sched, t_sched + task.duration), (
            f"工人 {w} 的预占区间未被释放！"
        )

    # 断言 3 & 4: 推进仿真时钟，验证在 t_sched 时刻不违规开工，且在 R 时刻恢复为 READY
    while task.status != TaskStatus.READY:
        if env.event_queue.is_empty():
            break
        event = env.event_queue.pop()
        if event is None:
            break
        env.state.current_time = event.timestamp

        # 校验：原预约时刻 t_sched 绝不触发旧 TASK_START
        if event.event_type == EventType.TASK_START and event.task_key == task.task_key:
            pytest.fail(f"旧 TASK_START 事件未失效！工序在 {env.state.current_time:.2f}h 违规开工！")

        # 派发到料事件
        if event.event_type == EventType.MATERIAL_ARRIVE:
            t = env.state.tasks[event.task_key]
            env._check_and_update_task_readiness(t)

    assert task.status == TaskStatus.READY
    assert env.state.current_time >= r_recovery - 1e-5, f"任务在物料到达时刻 {r_recovery} 之前过早恢复！"
    assert task.material_ready_time >= r_recovery


def test_running_task_not_interrupted_by_disturbance(baseline_path: str) -> None:
    """测试已物理开工 (RUNNING) 的作业在遭遇扰动时硬冻结，绝不中断。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    ready = env.get_ready_tasks()
    task = ready[0]
    team = tuple(env.state.station_worker_bindings[task.current_station][: task.demand])

    # 立即开工
    env.step({
        "task_key": task.task_key,
        "branch": ActionBranch.STATION_EXECUTE,
        "team": team,
        "align": 0,
    })
    assert task.status == TaskStatus.RUNNING
    actual_start = task.actual_start

    # 在开工后注入扰动
    scenario_payload = {
        "tau": actual_start + 1.0,
        "recovery_time": actual_start + 50.0,
        "affected_task_keys": [task.task_key],
    }
    env.load_scenario(scenario_payload)

    # 触发扰动
    while env.event_queue.peek() and env.event_queue.peek().timestamp <= actual_start + 1.0:
        e = env.event_queue.pop()
        if e.event_type == EventType.DISTURBANCE:
            env._handle_disturbance_event(e.timestamp, e.payload)

    # 断言作业仍然保持 RUNNING，未被中断或取消
    assert task.status == TaskStatus.RUNNING
    assert task.actual_start == actual_start
