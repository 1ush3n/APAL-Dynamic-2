"""固定启发式参照策略运行器专项单元测试，不代表正式图策略方法C。

验证点：
1. 无扰动基准计划自主跑通：启发式参照策略完成14周期全线装配；
2. 超期后移规则触发：当工序恢复时间超过当前周期边界时，启发式参照策略选择POSTPONE；
3. 代表性场景下输出完整step_records供离线M4轨迹采集。
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


def test_heuristic_agent_nominal_trajectory_run(baseline_path: str) -> None:
    """测试启发式参照策略在无扰动工况下完成生产且不后移。"""
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


def test_heuristic_agent_postpone_rule_trigger(baseline_path: str) -> None:
    """测试恢复时间越过当前周期边界时启发式参照策略触发后移。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()
    agent = HeuristicAgentWork3()

    task = env.state.tasks["0_16"]
    for other_task in env.state.tasks.values():
        if other_task.task_key != task.task_key:
            other_task.status = TaskStatus.COMPLETED
    task.status = TaskStatus.READY
    h0 = env.state.h0

    # 注入严重超期物料延迟 R = H_0 + 50.0 (超过当前名义周期结束点)
    task.material_ready_time = h0 + 50.0

    action = agent.select_action(env)
    assert action is not None
    # 断言首选动作为后移分支 B
    assert action["task_key"] == task.task_key
    assert action["branch"] == ActionBranch.POSTPONE


def test_heuristic_agent_decoupled_scenario_runs(
    baseline_path: str, scenarios_data: list[dict]
) -> None:
    """测试启发式参照策略在代表性场景下完成生产并生成轨迹记录。"""
    scenario = next((s for s in scenarios_data if s["scenario_id"] == "MID_HIGH_S1"), None)
    assert scenario is not None

    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    agent = HeuristicAgentWork3()

    result = agent.run_trajectory(env, scenario=scenario)

    assert result["completed_tasks"] == 2830
    assert result["transfer_count"] == 14
    assert result["total_decisions"] > 0
    assert len(result["step_records"]) == result["total_decisions"]
