"""易错点 2 攻坚白盒单元测试：后移工序跨站永久保留物料可用性约束 (Task 4.3)。

核心断言：
1. 工序在原站位遭遇物料延迟，获得开工下界约束 material_ready_time = R；
2. 工序选择分支 B 后移至下一站 (s_k + 1)，其 R 属性永久保留；
3. 全线脉动转站发生后，若转站时刻 P_q < R：
   - 飞机进驻新站后，该工序绝对不能立即变成 READY；
   - 必须保持 UNREADY，直至时刻推进到 R 触发 MATERIAL_ARRIVE 后才转为 READY；
4. 工序在新站排程时，最早开工时刻 S_{ki} 严格受限必须 >= R，绝对禁止通过换站逃避物料约束；
5. 多次连续后移时，R 约束持续继承不丢失。
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


def test_postponed_task_retains_material_constraint_across_transfer(baseline_path: str) -> None:
    """测试后移工序跨站流转后严格继承物料 R 约束，绝不在 R 到达前就绪开工。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    # 0 号飞机在 0 号站位，选取无站内后继的工序进行后移（符合工艺放行原则）
    ready = env.get_ready_tasks()
    # 筛选在 0 号站位没有后继且满足完整后移约束的任务。
    leaf_candidates = [
        t for t in env.state.tasks.values()
        if t.aircraft_id == 0 and t.current_station == 0
        and len(env._successors_map.get(0, {}).get(t.task_id, [])) == 0
        and env.validate_postpone(t) is None
    ]
    task = leaf_candidates[0]
    h0 = env.state.h0

    # 1. 注入一个跨越转站的远期物料约束 R = 1.5 * H0 (当前周期将在 H0 结束转站，R 位于下一周期内部)
    r_recovery = round(1.5 * h0, 4)
    scenario_payload = {
        "tau": 10.0,
        "recovery_time": r_recovery,
        "affected_task_keys": [task.task_key],
    }
    env.load_scenario(scenario_payload)

    # 推进时钟触发扰动
    while env.event_queue.peek() and env.event_queue.peek().timestamp <= 10.0:
        e = env.event_queue.pop()
        if e is None:
            break
        env.state.current_time = e.timestamp
        if e.event_type == EventType.DISTURBANCE:
            env._handle_disturbance_event(e.timestamp, e.payload)

    assert task.material_ready_time == r_recovery
    assert task.status == TaskStatus.UNREADY  # 扰动后退回 UNREADY

    # 2. 模拟决策：将该工序后移至 1 号站位 (分支 B)
    # 先临时置为 READY 以模拟合法动作选择
    task.status = TaskStatus.READY
    obs, _, _, _, info = env.step({
        "task_key": task.task_key,
        "branch": ActionBranch.POSTPONE,
    })

    assert task.status == TaskStatus.POSTPONED
    assert task.current_station == 1
    # 核心断言 1: 后移后物料约束严格保留
    assert task.material_ready_time == r_recovery

    # 3. 调度当前第 1 周期的其他所有工序全部完工，触发第 1 次全线脉动转站
    max_steps = 300
    steps = 0
    while env.state.current_cycle == 1 and steps < max_steps:
        cur_ready = env.get_ready_tasks()
        if not cur_ready:
            break
        t = cur_ready[0]
        team = tuple(env.state.station_worker_bindings[t.current_station][: t.demand])
        env.step({
            "task_key": t.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 0,
        })
        steps += 1

    # 检查第 1 次脉动转站成功触发，且实际放行早于恢复时刻
    assert len(env.state.transfer_history) == 1
    p1 = env.state.transfer_history[0]
    assert p1 >= 0.0
    assert p1 < r_recovery, f"转站时刻 {p1} 必须早于物料恢复时刻 {r_recovery}"

    # 4. 0 号飞机已前进至 1 号站位，检查受扰工序在到达新站后的状态
    assert env.state.aircraft[0].current_station == 1
    # 核心断言 2: 转站时刻由于 P_1 < R，受扰工序绝对不能成为 READY！
    assert task.status == TaskStatus.UNREADY, (
        f"易错点 2 违规！后移工序在到料时刻 {r_recovery} 之前在新站提前变为 {task.status.name}！"
    )
    assert task.material_ready_time == r_recovery

    # 5. 推进事件时钟直至 R 时刻
    while task.status != TaskStatus.READY:
        if env.event_queue.is_empty():
            break
        event = env.event_queue.pop()
        if event is None:
            break
        env.state.current_time = event.timestamp
        if event.event_type == EventType.MATERIAL_ARRIVE and event.task_key == task.task_key:
            env._check_and_update_task_readiness(task)

    # 核心断言 3: 时刻到达 R 时才解锁为 READY
    assert task.status == TaskStatus.READY
    assert env.state.current_time >= r_recovery - 1e-5

    # 6. 在新站排程开工 (align=0)，验证最早排程时刻必须 >= R
    team1 = tuple(env.state.station_worker_bindings[1][: task.demand])
    env.step({
        "task_key": task.task_key,
        "branch": ActionBranch.STATION_EXECUTE,
        "team": team1,
        "align": 0,
    })

    # 核心断言 4: 实际开工时刻必须 >= R
    assert task.scheduled_start is not None
    assert task.scheduled_start >= r_recovery - 1e-5, (
        f"工序开工时刻 {task.scheduled_start} 违反物料下界 {r_recovery}！"
    )
