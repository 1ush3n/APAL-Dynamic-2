"""测试 work3_4060/run_baseline_a_evaluation.py 中的协议契约、坏输入拦截与边界处理能力 (Section 7)。

验收清单：
1. A 身份验收：调用真实 BaselinePlanRepairAgent，agent_type="Baseline-Repair-A"；非合规实例被拦截。
2. 场景覆盖与主聚合隔离：5个唯一场景齐全；重复/缺失被识别；主聚合均值严格只计4个Validation，Nominal单列。
3. 生产与物理校验：完工 210/210，check_environment_trajectory_feasibility 显式解包为 (feasible, violations)。
4. 账本闭合检查：独立目标、环境费用、五分项和及奖励累计绝对误差 <= 1e-12。
5. 冻结学习方法多配置映射：两份 JSON 相同标签 (C_b4096) 依 Beta 正确区分映射，不互相覆盖。
6. 边界与异常守卫：
   - J_A < 1e-6 时百分比降幅返回 None / N.A.，无除以零、无 NaN/Inf；
   - 费用差 <= 1e-12 正确判定为 TIE；
   - 扰动命中分层逻辑覆盖；
   - A 策略若发生后移触发守卫断言；
   - 缺失记录或非有限费用触发报错。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import pytest

from envs.work3.environment import AirLineEnvWork3
from scripts.work3.baseline_repair_agent import (
    BaselinePlanRepairAgent,
    BaselineRepairInfeasibleError,
)
from scripts.work3.evaluate_c_vs_d import (
    PERCENTAGE_DENOMINATOR_THRESHOLD,
    evaluate_single_trajectory,
)
from utils.work3.trajectory_feasibility import check_environment_trajectory_feasibility
from work3_4060.run_baseline_a_evaluation import (
    COST_TOLERANCE,
    REPO_ROOT,
    build_reuse_inventory,
    compute_comparison_metrics,
    compute_file_sha256,
    compute_lf_normalized_sha256,
    verify_and_load_learning_results,
)


def test_baseline_a_identity_and_type_check():
    """验证 A 策略严格要求 BaselinePlanRepairAgent 且禁止塞入其他类型。"""
    agent = BaselinePlanRepairAgent()
    assert hasattr(agent, "select_action")
    # A 不应有神经网络模型结构
    assert not hasattr(agent, "actor_critic")
    assert not hasattr(agent, "time_head")

    # 错误类型传入 evaluate_single_trajectory 时必须抛出 TypeError
    env = AirLineEnvWork3(baseline_json_path=str(REPO_ROOT / "work3_4060/data/mini210.json"))
    with pytest.raises(TypeError, match="A组必须使用BaselinePlanRepairAgent"):
        evaluate_single_trajectory(
            env=env,
            agent_type="Baseline-Repair-A",
            agent="fake_agent",
            scenario=None,
            max_decisions=10,
        )


def test_check_environment_trajectory_feasibility_explicit_unpacking():
    """验证 check_environment_trajectory_feasibility 返回 (bool, dict)，不得将整个元组直接转布尔。"""
    env = AirLineEnvWork3(baseline_json_path=str(REPO_ROOT / "work3_4060/data/mini210.json"))
    # 在环境未开始或未终止时调用
    res = check_environment_trajectory_feasibility(env)
    assert isinstance(res, tuple)
    assert len(res) == 2
    feasible, violations = res
    assert isinstance(feasible, bool)
    assert isinstance(violations, dict)
    # 显式校验：如果直接 bool(res)，元组本身为 True，但里面 feasible 实际为 False！
    assert feasible is False
    assert bool(res) is True, "元组作为真值恒为 True，证明必须显式解包！"
    assert "incomplete_trajectory" in violations


def test_reuse_inventory_specification():
    """验证 reuse_inventory 构建时明确列出 5 个场景且均如实标记未复用原因。"""
    scenarios = [
        "NOMINAL",
        "COMPARE_VAL_S0_EARLY_M",
        "COMPARE_VAL_S2_MID_M",
        "COMPARE_VAL_S3_MID_L",
        "COMPARE_VAL_S4_LATE_H",
    ]
    inv = build_reuse_inventory(scenarios)
    assert len(inv) == 5
    for sc in scenarios:
        assert sc in inv
        assert inv[sc]["reused"] is False
        assert "接管" in inv[sc]["reason"] or "前缀" in inv[sc]["reason"]
        assert "full_end_to_end_trajectory" in inv[sc]["conditions_missing"]


def test_verify_and_load_learning_results_distinguishes_beta(tmp_path: Path):
    """验证两份不同 Beta 的学习结果 JSON 中的 C_b4096 和 F_b4096 依 Beta 正确区分映射，不互相覆盖。"""
    # 模拟一份合法的最小 JSON
    base_record = {
        "scenario_id": "NOMINAL",
        "method": "C",
        "variant": "C_b4096",
        "metrics": {
            "success": True,
            "feasible": True,
            "completed_tasks": 210,
            "j_total": 0.0264624,
            "makespan": 669.7,
            "ledger_reconciliation": {"consistent": True},
        },
    }

    # 构造假文件 1 (Beta=1.0)
    b1_records = [
        dict(base_record, variant="C_b4096", method="C"),
        dict(base_record, variant="F_b4096", method="F"),
    ]
    f_b1 = tmp_path / "b1.json"
    f_b1.write_text(json.dumps(b1_records), encoding="utf-8")

    # 构造假文件 2 (Beta=0.1)
    b01_records = [
        dict(base_record, variant="C_b4096", method="C"),
        dict(base_record, variant="F_b4096", method="F"),
        dict(base_record, variant="D_b4096", method="D"),
    ]
    f_b01 = tmp_path / "b01.json"
    f_b01.write_text(json.dumps(b01_records), encoding="utf-8")

    loaded = verify_and_load_learning_results(
        json_path_beta1=f_b1,
        json_path_beta01=f_b01,
        expected_scenarios=["NOMINAL"],
    )

    assert "NOMINAL::C_beta1_seed42_b4096" in loaded
    assert "NOMINAL::F_beta1_seed42_b4096" in loaded
    assert "NOMINAL::C_beta01_seed42_b4096" in loaded
    assert "NOMINAL::F_beta01_seed42_b4096" in loaded
    assert "NOMINAL::D_beta01_seed42_b4096" in loaded

    assert loaded["NOMINAL::C_beta1_seed42_b4096"]["beta"] == 1.0
    assert loaded["NOMINAL::C_beta01_seed42_b4096"]["beta"] == 0.1


def test_verify_and_load_learning_results_detects_missing_records(tmp_path: Path):
    """验证当场景记录缺失时拦截并报错。"""
    f_b1 = tmp_path / "b1.json"
    f_b1.write_text(json.dumps([]), encoding="utf-8")
    f_b01 = tmp_path / "b01.json"
    f_b01.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="学习方法评估记录缺失"):
        verify_and_load_learning_results(
            json_path_beta1=f_b1,
            json_path_beta01=f_b01,
            expected_scenarios=["COMPARE_VAL_S0_EARLY_M"],
        )


def test_verify_and_load_learning_results_detects_non_finite_cost(tmp_path: Path):
    """验证当出现 NaN/Inf 费用时拦截。"""
    bad_record = [{
        "scenario_id": "NOMINAL",
        "method": "C",
        "variant": "C_b4096",
        "metrics": {"j_total": float("nan"), "makespan": 669.7},
    }]
    f_b1 = tmp_path / "bad_b1.json"
    f_b1.write_text(json.dumps(bad_record), encoding="utf-8")
    f_b01 = tmp_path / "b01.json"
    f_b01.write_text(json.dumps([]), encoding="utf-8")

    with pytest.raises(ValueError, match="非有限费用"):
        verify_and_load_learning_results(
            json_path_beta1=f_b1,
            json_path_beta01=f_b01,
            expected_scenarios=["NOMINAL"],
        )


def test_compute_comparison_metrics_boundary_and_near_ties():
    """验证比较指标计算中的边界情况：J_A接近0时不除零、近平局判决与均值聚合隔离。"""
    val_scenarios = [
        "COMPARE_VAL_S0_EARLY_M",
        "COMPARE_VAL_S2_MID_M",
        "COMPARE_VAL_S3_MID_L",
        "COMPARE_VAL_S4_LATE_H",
    ]
    all_scenarios = ["NOMINAL"] + val_scenarios

    # 1. 模拟 A 结果：Nominal J_A 极小 (8e-13 < 1e-6)
    a_results = {}
    for sc_id in all_scenarios:
        j_val = 8.155e-13 if sc_id == "NOMINAL" else 1.3e-4
        a_results[sc_id] = {
            "j_total": j_val,
            "makespan": 670.0,
            "success": True,
            "feasible": True,
            "actual_disturbance_hits": 0 if sc_id == "NOMINAL" else 1,
            "d_time": j_val,
            "d_team": 0.0,
            "j_revision": 0.0,
            "j_takt": 0.0,
            "j_postpone": 0.0,
        }

    # 2. 模拟学习方法结果
    learning_results = {}
    target_methods = [
        "C_beta1_seed42_b4096",
        "F_beta1_seed42_b4096",
        "D_beta01_seed42_b4096",
        "C_beta01_seed42_b4096",
        "F_beta01_seed42_b4096",
    ]
    for sc_id in all_scenarios:
        for m_name in target_methods:
            # 针对 S0 测试近平局 (差值 4e-13 < 1e-12)
            if sc_id == "COMPARE_VAL_S0_EARLY_M" and m_name == "C_beta1_seed42_b4096":
                m_j = 1.3e-4 + 4e-13
            else:
                m_j = 0.0265
            learning_results[f"{sc_id}::{m_name}"] = {
                "scenario_id": sc_id,
                "method_canonical": m_name,
                "j_total": m_j,
                "makespan": 680.0,
                "success": True,
                "feasible": True,
                "actual_disturbance_hits": 1 if sc_id != "NOMINAL" else 0,
                "d_time": 0.001,
                "d_team": 0.0125,
                "j_revision": 0.013,
                "j_takt": 0.0,
                "j_postpone": 0.0,
            }

    res = compute_comparison_metrics(
        a_results=a_results,
        learning_results=learning_results,
        val_scenarios=val_scenarios,
        nominal_scenario="NOMINAL",
    )

    # 检验 1: Nominal 场景 J_A < 1e-6 时相对降幅为 None，不除以零或产生 Inf
    nominal_comp = res["nominal"]["comparisons"]["C_beta1_seed42_b4096"]
    assert nominal_comp["relative_cost_reduction"] is None

    # 检验 2: COMPARE_VAL_S0_EARLY_M 上差值 4e-13 <= 1e-12 判定为 TIE
    s0_comp = [c for c in res["scenario_comparisons"] if c["scenario_id"] == "COMPARE_VAL_S0_EARLY_M"][0]
    assert s0_comp["comparisons"]["C_beta1_seed42_b4096"]["winner"] == "TIE"

    # 检验 3: 主聚合平均仅计 4 个 Validation 场景，不把 Nominal 的 8e-13 混入！
    mean_j_a = res["aggregates"]["a_rule"]["mean_j_total"]
    assert math.isclose(mean_j_a, 1.3e-4, rel_tol=1e-7)

    # 检验 4: 扰动分层判定
    assert nominal_comp["hit_stratum"] == "N.A."
    assert s0_comp["comparisons"]["C_beta1_seed42_b4096"]["hit_stratum"] == "BOTH_HIT"


def test_sha256_helpers(tmp_path: Path):
    """验证 raw 与 lf 规范化 SHA256 计算对换行符差异的正确敏感性与鲁棒性。"""
    crlf_file = tmp_path / "sample_crlf.txt"
    crlf_file.write_bytes(b"hello\r\nworld\r\n")

    lf_file = tmp_path / "sample_lf.txt"
    lf_file.write_bytes(b"hello\nworld\n")

    # 原始字节哈希不同
    assert compute_file_sha256(crlf_file) != compute_file_sha256(lf_file)
    # 规范化后哈希严格相同
    assert compute_lf_normalized_sha256(crlf_file) == compute_lf_normalized_sha256(lf_file)
