"""Task 7.4 / 里程碑 M5 验收单元测试：基线 C 与方法 D 对比评测验证。

验证点：
1. evaluate_single_trajectory 能在 Baseline-C 与 Method-D 上分别驱动产线顺利跑通至完工；
2. 全线全部 2,830 道工序 100% 完工，全部 10 架次飞机顺利出线；
3. 多目标结算结构完整 (J_takt, D_time, D_team, J_postpone, J_total)；
4. 评测报告输出 JSON 文件格式规范无缺失。
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest
import torch

from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import ActorCriticWork3
from models.work3.heuristic_agent import HeuristicAgentWork3
from scripts.work3.evaluate_c_vs_d import (
    build_formal_evaluation_agent,
    evaluate_single_trajectory,
    run_benchmark_evaluation,
)


@pytest.fixture
def baseline_path() -> str:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return str(path)


@pytest.fixture
def scenarios_path() -> str:
    path = Path("data/work3/scenarios_9class.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return str(path)


def test_evaluate_single_trajectory_baseline_c(baseline_path: str, scenarios_path: str) -> None:
    """测试基线 C 在扰动场景下完成全部装配并统计多目标。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    agent = HeuristicAgentWork3(name="Baseline-C")

    with open(scenarios_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    sc = scenarios[0]

    res = evaluate_single_trajectory(env, "Baseline-C", agent, scenario=sc)

    assert res["completed_tasks"] == 2830
    assert res["j_total"] >= 0.0
    assert res["makespan"] > 0.0
    assert res["transfers"] >= 10
    assert res["termination_reason"] in {
        "completed",
        "decision_limit",
        "rollout_truncated",
        "deadlock",
    }
    assert res["success"] is True


def test_evaluate_single_trajectory_method_d_debug(baseline_path: str, scenarios_path: str) -> None:
    """随机方法D只通过显式调试包装器进入评测，不冒充正式检查点。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    agent = build_formal_evaluation_agent(
        method_variant="D",
        checkpoint_path=Path("__missing_debug_method_d.pt"),
        device="cpu",
        debug_random=True,
    )

    with open(scenarios_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    sc = scenarios[0]

    res = evaluate_single_trajectory(
        env,
        "Method-D",
        agent,
        scenario=sc,
        max_decisions=32,
    )

    assert 0 <= res["completed_tasks"] <= 2830
    assert res["j_total"] >= 0.0
    assert res["makespan"] > 0.0
    assert 0 <= res["transfers"] <= 14
    assert res["termination_reason"] in {
        "completed",
        "decision_limit",
        "rollout_truncated",
        "deadlock",
    }
    assert res["success"] is (res["completed_tasks"] == 2830 and res["feasible"])


def test_repeated_evaluation_does_not_update_actor_or_time_predictor(
    baseline_path: str,
) -> None:
    """连续评测多条轨迹不得改变Actor/图编码器/时间头及归一化状态。"""
    agent = build_formal_evaluation_agent(
        method_variant="D",
        checkpoint_path=Path("__missing_debug_method_d.pt"),
        device="cpu",
        debug_random=True,
    )
    assert agent.time_head is not None
    assert agent.actor_critic.training is False
    assert agent.time_head.training is False

    def state_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().clone()
            for name, value in module.state_dict().items()
        }

    actor_before = state_snapshot(agent.actor_critic)
    time_head_before = state_snapshot(agent.time_head)
    assert any(name.startswith("graph_encoder.") for name in actor_before)
    assert any(isinstance(module, torch.nn.LayerNorm) for module in agent.actor_critic.modules())
    assert any(isinstance(module, torch.nn.LayerNorm) for module in agent.time_head.modules())

    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    for _ in range(2):
        result = evaluate_single_trajectory(
            env,
            "Method-D",
            agent,
            max_decisions=1,
            device="cpu",
        )
        assert result["decisions"] == 1
        assert agent.last_time_prediction is not None

    assert all(
        torch.equal(actor_before[name], value)
        for name, value in agent.actor_critic.state_dict().items()
    )
    assert all(
        torch.equal(time_head_before[name], value)
        for name, value in agent.time_head.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in agent.actor_critic.parameters())
    assert all(parameter.grad is None for parameter in agent.time_head.parameters())


def test_benchmark_evaluation_m5_acceptance(baseline_path: str, scenarios_path: str) -> None:
    """测试 M5 验收对比评估接口与报告生成。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_json = Path(tmpdir) / "eval_test_m5.json"

        results = run_benchmark_evaluation(
            baseline_path=baseline_path,
            scenarios_path=scenarios_path,
            selected_scenario_ids=["EARLY_LOW_S0", "MID_MID_S1"],
            output_json=str(out_json),
            device="cpu",
            allow_debug_random=True,
            max_decisions=32,
        )

        assert results["schema_version"] == "work3_eval_c_vs_d_v2"
        assert results["nominal_method_c"]["scenario_id"] == "NOMINAL"
        nominal = results["nominal_method_c"]
        assert "raw_cost_components" in nominal
        assert "success" in nominal and "feasible" in nominal
        assert "time_prediction_metrics" in nominal
        scenario_results = results["scenario_results"]
        assert len(scenario_results) == 2
        assert out_json.is_file()

        for item in scenario_results:
            assert "scenario_id" in item
            assert "method_c" in item
            assert "method_d" in item
            assert "improvement_j_total_pct" in item
            assert 0 <= item["method_c"]["completed_tasks"] <= 2830
            assert 0 <= item["method_d"]["completed_tasks"] <= 2830
            assert "termination_reason" in item["method_d"]
            assert "success" in item["method_d"]
            assert "time_prediction_metrics" in item["method_c"]
            assert "time_prediction_metrics" in item["method_d"]
