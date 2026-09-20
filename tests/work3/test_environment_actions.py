"""工作三仿真环境调度动作分支 A 与 B 白盒单元测试 (Task 2.3 & Task 2.4)。

测试覆盖：
1. reset() 初始化与站位 0 首批工序就绪激活;
2. 分支 A：立即开工 (RUNNING) 与工人日历占用、完工事件安排;
3. 分支 A：二元对齐 alpha=1 严格遵守 P_{q-1} + b_i^0 下界;
4. 分支 A：工人暂忙时自动建立未来预约 (RESERVED)，不阻塞其他调度;
5. 分支 B：合法后移至下一站 (POSTPONED)、动作当场截断（不选队/不占工人）、恢复时间 R 属性持久保留;
6. 分支 B：末站（站位 5 / index 4）后移刚性拦截报错;
7. 团队资质与人数硬约束校验拦截。
"""

from __future__ import annotations

import pytest
from pathlib import Path

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3


@pytest.fixture
def env() -> AirLineEnvWork3:
    """初始化仿真环境实例。"""
    baseline_path = "data/work3/real_283_k10_baseline.json"
    if not Path(baseline_path).is_file():
        pytest.skip(f"{baseline_path} 不存在")
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()
    return env


def test_env_reset_and_initial_readiness(env: AirLineEnvWork3) -> None:
    """测试 reset() 后环境状态正确初始化，0号飞机进驻0号站位且工序就绪。"""
    assert env.state.current_time == 0.0
    assert env.state.current_cycle == 1
    assert env.state.aircraft[0].current_station == 0
    assert env.state.aircraft[1].current_station == -1

    ready_tasks = env.get_ready_tasks()
    assert len(ready_tasks) > 0
    # 所有就绪任务必须属于 0 号飞机且在 0 号站位
    for t in ready_tasks:
        assert t.aircraft_id == 0
        assert t.current_station == 0
        assert t.status == TaskStatus.READY


def test_branch_a_immediate_execution(env: AirLineEnvWork3) -> None:
    """测试分支 A：资源空闲时立即开工 (Task 2.3)。"""
    ready_tasks = env.get_ready_tasks()
    task = ready_tasks[0]
    station_workers = env.state.station_worker_bindings[task.current_station]
    team = station_workers[: task.demand]

    action = {
        "task_key": task.task_key,
        "branch": ActionBranch.STATION_EXECUTE,
        "team": team,
        "align": 0,
    }

    obs, reward, terminated, truncated, info = env.step(action)

    assert info["scheduled_status"] == "RUNNING"
    assert info["scheduled_start"] == 0.0
    assert task.status == TaskStatus.RUNNING
    assert task.actual_start == 0.0
    assert task.assigned_team == list(team)

    # 校验工人在日历中记录了区间 [0.0, task.duration]
    for w in team:
        wc = env.state.workers[w]
        assert not wc.is_available(0.0, task.duration)
        assert wc.intervals[0].task_key == task.task_key


def test_branch_a_binary_alignment(env: AirLineEnvWork3) -> None:
    """测试分支 A：二元对齐 alpha=1 严格生效 (Task 2.3)。"""
    ready_tasks = env.get_ready_tasks()
    # 挑选一个有正偏移 b_i^0 的任务（或者人为赋予）
    task = ready_tasks[0]
    task.in_station_offset = 12.5  # 模拟 b_i^0 = 12.5

    station_workers = env.state.station_worker_bindings[task.current_station]
    team = station_workers[: task.demand]

    action = {
        "task_key": task.task_key,
        "branch": ActionBranch.STATION_EXECUTE,
        "team": team,
        "align": 1,  # 对齐开启
    }

    obs, reward, terminated, truncated, info = env.step(action)

    assert task.status == TaskStatus.RESERVED
    assert task.scheduled_start == 12.5, f"二元对齐未生效: 期望 12.5, 实际 {task.scheduled_start}"
    assert info["scheduled_status"] == "RESERVED"


def test_branch_a_future_reservation_when_worker_busy(env: AirLineEnvWork3) -> None:
    """测试分支 A：工人暂忙时成功建立未来预约，不阻塞决策 (Task 2.3)。"""
    ready_tasks = env.get_ready_tasks()
    assert len(ready_tasks) >= 2, "需要至少 2 道就绪工序"

    task1 = ready_tasks[0]
    task2 = ready_tasks[1]

    station_workers = env.state.station_worker_bindings[task1.current_station]
    team1 = station_workers[: task1.demand]
    team2 = station_workers[: task2.demand]

    # 第一个任务立即开工，占用 [0, task1.duration]
    env.step({
        "task_key": task1.task_key,
        "branch": ActionBranch.STATION_EXECUTE,
        "team": team1,
        "align": 0,
    })
    assert task1.status == TaskStatus.RUNNING

    # 第二个任务也选用冲突的工人，必须被推迟至 task1 完工之后
    env.step({
        "task_key": task2.task_key,
        "branch": ActionBranch.STATION_EXECUTE,
        "team": team2,
        "align": 0,
    })

    assert task2.status == TaskStatus.RESERVED
    assert task2.scheduled_start is not None
    assert task2.scheduled_start >= task1.duration - 1e-5


def test_branch_b_postpone_and_truncation(env: AirLineEnvWork3) -> None:
    """测试分支 B：合法后移当场截断，保留物料约束 R (Task 2.4)。"""
    ready_tasks = env.get_ready_tasks()
    task = ready_tasks[0]
    task.material_ready_time = 45.0  # 注入恢复时间 R

    action = {
        "task_key": task.task_key,
        "branch": ActionBranch.POSTPONE,
    }

    obs, reward, terminated, truncated, info = env.step(action)

    assert task.status == TaskStatus.POSTPONED
    assert task.current_station == 1, "后移后目标站位步进到 1"
    assert task.postpone_count == 1
    assert task.material_ready_time == 45.0, "恢复时间 R 必须永久继承"
    assert task.assigned_team == [], "后移分支必须当场截断，不选团队"
    assert task.scheduled_start is None, "后移分支必须当场截断，不预约工人"


def test_branch_b_final_station_forbidden(env: AirLineEnvWork3) -> None:
    """测试分支 B：末站（站位 5 / index 4）绝对禁止后移 (Task 2.4)。"""
    ready_tasks = env.get_ready_tasks()
    task = ready_tasks[0]
    # 模拟该任务当前在末站 (站位 4)
    task.current_station = 4
    env.state.aircraft[task.aircraft_id].current_station = 4

    action = {
        "task_key": task.task_key,
        "branch": ActionBranch.POSTPONE,
    }

    with pytest.raises(ValueError, match="末站.*绝对禁止后移"):
        env.step(action)


def test_team_validation_errors(env: AirLineEnvWork3) -> None:
    """测试团队人数不符或跨站工人指派的严格报错拦截。"""
    ready_tasks = env.get_ready_tasks()
    task = ready_tasks[0]

    # 1. 人数不足
    with pytest.raises(ValueError, match="需求.*人"):
        env.step({
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": [],
        })

    # 2. 指派非本站工人 (如不存在或不属于站位的工人)
    with pytest.raises(ValueError, match="不属于站位"):
        env.step({
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": list(range(990, 990 + task.demand)),
        })
