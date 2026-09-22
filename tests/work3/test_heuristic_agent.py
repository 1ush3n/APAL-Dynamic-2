"""Task 5.3 基线 C 策略运行器专项单元测试。

验证点：
1. 无扰动基准计划自主跑通：基线 C 顺利完成 14 周期全线装配，2,830 道工序 100% 完工，后移次数为 0；
2. 超期后移保节拍规则触发验证：当工序物料 R > P_{q-1} + H_0 时，基线 C 果断选择分支 B (POSTPONE)；
3. 代表性解耦场景闭环验证：在 9 类代表性场景下，基线 C 自主推进生产，输出完整的 step_records 供轨迹收集使用。
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from models.work3.heuristic_agent import HeuristicAgentWork3


@pytest.fixture
def baseline_path() -> str:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return str(path)


@pytest.fixture
def scenarios_data() -> list[dict]:
    path = Path("data/work3/scenarios_9class.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_baseline_c_nominal_trajectory_run(baseline_path: str) -> None:
    """测试基线 C 在无扰动工况下自主完成 14 周期生产，0 后移。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    agent = HeuristicAgentWork3()

    result = agent.run_trajectory(env)

    # 1. 验证全部工序完成
    assert result["completed_tasks"] == 2830
    assert result["transfer_count"] == 14
    # 2. 无扰动时不应触发后移
    assert result["cost_postpone"] == 0.0
    # 3. 验证每步均有效记录了启发式时间估计与实际转站时刻
    records = result["step_records"]
    assert len(records) > 0
    assert all("estimated_cmax" in r and "actual_transfer_time" in r for r in records)


def test_baseline_c_postpone_rule_trigger(baseline_path: str) -> None:
    """测试超期后移保节拍规则：当 R > P_{q-1} + H_0 时，基线 C 果断触发分支 B。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()
    agent = HeuristicAgentWork3()

    task = next(
        task for task in env.state.tasks.values()
        if env.validate_postpone(task) is None
    )
    for ready_task in env.get_ready_tasks():
        ready_task.status = TaskStatus.UNREADY
    task.status = TaskStatus.READY
    h0 = env.state.h0

    # 注入严重超期物料延迟 R = H_0 + 50.0 (超过当前名义周期结束点)
    task.material_ready_time = h0 + 50.0

    action = agent.select_action(env)
    assert action is not None
    # 断言首选动作为后移分支 B
    assert action["task_key"] == task.task_key
    assert action["branch"] == ActionBranch.POSTPONE


def test_baseline_c_decoupled_scenario_runs(
    baseline_path: str, scenarios_data: list[dict]
) -> None:
    """测试基线 C 在代表性解耦场景下自主完成流水线生产并生成轨迹记录。"""
    scenario = next((s for s in scenarios_data if s["scenario_id"] == "MID_HIGH_S1"), None)
    assert scenario is not None

    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    agent = HeuristicAgentWork3()

    result = agent.run_trajectory(env, scenario=scenario)

    assert result["completed_tasks"] == 2830
    assert result["transfer_count"] == 14
    assert result["total_decisions"] > 0
    assert len(result["step_records"]) == result["total_decisions"]
