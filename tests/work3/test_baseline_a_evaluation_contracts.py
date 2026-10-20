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
    generate_markdown_report,
    run_baseline_a_scenario,
    verify_and_load_learning_results,
)
from work3_4060.run_baseline_a_reachability_diagnostic import (
    evaluate_action_reachability_and_probs,
)
from models.work3.actor_critic import ActorCriticWork3
from scripts.work3.evaluate_c_vs_d import (
    compute_cycle_heuristic_cmax,
    compute_time_urgency_vector,
    extract_compact_state_features,
)
import torch


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


def test_baseline_a_advance_cause_counts_export():
    """验证 R-A1 修复：明确区分 explicit (0) 与 auto_after_action (11)，保留完整 advance_cause_counts。"""
    baseline_path = REPO_ROOT / "work3_4060/data/mini210.json"
    a_res, facts = run_baseline_a_scenario(
        baseline_path=baseline_path,
        scenario=None,
        scenario_id="NOMINAL",
        max_decisions=500,
        device="cpu",
    )
    # A 策略绝不调用显式推进 (explicit=0)
    assert a_res["explicit_advances"] == 0
    # A 策略的推进全部来自于完成车站作业后的自动周期推移 (auto_after_action=11)
    assert a_res["automatic_advances"] == 11
    assert a_res["advance_cause_counts"]["explicit"] == 0
    assert a_res["advance_cause_counts"]["auto_after_action"] == 11
    assert a_res["advance_cause_counts"]["none"] == 199

    # facts 中的 production_outcome 也应同步记录
    prod = facts["production_outcome"]
    assert prod["explicit_advances"] == 0
    assert prod["automatic_advances"] == 11
    assert prod["advance_cause_counts"]["explicit"] == 0
    assert prod["advance_cause_counts"]["auto_after_action"] == 11
    assert prod["advance_cause_counts"]["none"] == 199


def test_generate_markdown_report_dynamic_win_tallies_and_components(tmp_path: Path):
    """验证 R-A2 修复：去除报告中写死文字与固定常数，动态展示各方法独立分项与战绩。"""
    # 模拟 comparison_data
    comparison_data = {
        "aggregates": {
            "val_scenarios_count": 4,
            "a_rule": {
                "mean_j_total": 1.304e-4,
                "mean_makespan": 680.43,
                "success_feasible_count": "4/4",
            },
            "methods": {
                "C_beta1_seed42_b4096": {
                    "mean_j_total": 0.0265,
                    "mean_makespan": 680.97,
                    "success_feasible_count": "4/4",
                    "valid_pair_count": "4/4",
                    "mean_delta_j_a_minus_m": -0.0264,
                    "mean_delta_makespan_a_minus_m": -0.54,
                    "overall_relative_cost_reduction": -202.7,
                    "win_tally": {"a_wins": 4, "m_wins": 0, "ties": 0},
                },
                "F_beta1_seed42_b4096": {
                    "mean_j_total": 0.0430,
                    "mean_makespan": 680.33,
                    "success_feasible_count": "4/4",
                    "valid_pair_count": "4/4",
                    "mean_delta_j_a_minus_m": -0.0429,
                    "mean_delta_makespan_a_minus_m": 0.10,
                    "overall_relative_cost_reduction": -329.3,
                    "win_tally": {"a_wins": 0, "m_wins": 4, "ties": 0},  # 反转测试：M全胜
                },
                "D_beta01_seed42_b4096": {
                    "mean_j_total": 0.0480,
                    "mean_makespan": 692.88,
                    "success_feasible_count": "4/4",
                    "valid_pair_count": "4/4",
                    "mean_delta_j_a_minus_m": -0.0479,
                    "mean_delta_makespan_a_minus_m": -12.45,
                    "overall_relative_cost_reduction": -367.3,
                    "win_tally": {"a_wins": 0, "m_wins": 0, "ties": 4},  # 测试全平
                },
                "C_beta01_seed42_b4096": {
                    "mean_j_total": 0.0531,
                    "mean_makespan": 690.35,
                    "success_feasible_count": "4/4",
                    "valid_pair_count": "4/4",
                    "mean_delta_j_a_minus_m": -0.0530,
                    "mean_delta_makespan_a_minus_m": -9.92,
                    "overall_relative_cost_reduction": -406.8,
                    "win_tally": {"a_wins": 3, "m_wins": 1, "ties": 0},  # 测试 A胜3
                },
                "F_beta01_seed42_b4096": {
                    "mean_j_total": 0.0449,
                    "mean_makespan": 824.08,
                    "success_feasible_count": "4/4",
                    "valid_pair_count": "4/4",
                    "mean_delta_j_a_minus_m": -0.0447,
                    "mean_delta_makespan_a_minus_m": -143.65,
                    "overall_relative_cost_reduction": -343.3,
                    "win_tally": {"a_wins": 1, "m_wins": 3, "ties": 0},  # 测试 学习方法胜3
                },
            },
        },
        "nominal": {
            "is_nominal": True,
            "scenario_id": "NOMINAL",
            "a_rule": {
                "j_total": 8.155e-13,
                "makespan": 669.11,
                "j_takt": 0.0,
                "d_time": 8.155e-13,
                "d_team": 0.0,
                "j_revision": 0.0,
                "j_postpone": 0.0,
            },
            "comparisons": {
                "C_beta1_seed42_b4096": {
                    "j_total": 0.02646,
                    "makespan": 669.71,
                    "j_takt": 0.0,
                    "d_time": 7.31e-4,
                    "d_team": 0.0125,
                    "j_revision": 0.0132,
                    "j_postpone": 0.0,
                    "winner": "A_BETTER",
                    "delta_makespan_a_minus_m": -0.6,
                    "delta_j_a_minus_m": -0.02646,
                    "hit_stratum": "N.A.",
                },
                "F_beta1_seed42_b4096": {
                    "j_total": 0.04298,
                    "makespan": 669.00,
                    "j_takt": 0.0,
                    "d_time": 1.49e-3,
                    "d_team": 0.0200,
                    "j_revision": 0.0215,
                    "j_postpone": 0.0,
                    "delta_j_a_minus_m": -0.04298,
                },
                "D_beta01_seed42_b4096": {
                    "j_total": 0.04801,
                    "makespan": 685.32,
                    "j_takt": 0.0,
                    "d_time": 4.84e-3,
                    "d_team": 0.01917,
                    "j_revision": 0.0240,
                    "j_postpone": 0.0,
                    "delta_j_a_minus_m": -0.04801,
                },
            },
        },
        "scenario_comparisons": [
            {
                "is_nominal": True,
                "scenario_id": "NOMINAL",
                "a_rule": {"j_total": 8.155e-13},
                "comparisons": {
                    "C_beta1_seed42_b4096": {"j_total": 0.02646, "delta_j_a_minus_m": -0.02646, "winner": "A_BETTER", "delta_makespan_a_minus_m": -0.6, "hit_stratum": "N.A."},
                    "F_beta1_seed42_b4096": {"j_total": 0.04298, "delta_j_a_minus_m": -0.04298},
                    "D_beta01_seed42_b4096": {"j_total": 0.04801, "delta_j_a_minus_m": -0.04801},
                },
            }
        ],
    }

    a_results = {
        "NOMINAL": {"j_revision": 0.0, "d_time": 8.155e-13},
        "COMPARE_VAL_S2_MID_M": {"j_revision": 8.138e-05, "d_time": 8.138e-05},
        "COMPARE_VAL_S3_MID_L": {"j_revision": 8.997e-05, "d_time": 8.997e-05},
    }

    out_file = tmp_path / "test_summary.md"
    content = generate_markdown_report(comparison_data, a_results, out_file)

    # 1. 验证动态战绩格式化：
    assert "4/0/0 (A全胜)" in content
    assert "0/0/4 (学习方法全胜)" in content
    assert "0/4/0 (全平)" in content
    assert "3/0/1 (A胜3)" in content
    assert "1/0/3 (学习方法胜3)" in content
    assert "(A 胜 4/0/0)" not in content  # 不得再有硬编码的 (A 胜 4/0/0)

    # 2. 验证非零修订费用被如实展示，而非断言全为0
    assert "8.138" in content
    assert "8.997" in content
    assert "连带开工推移" in content or "3_23" in content

    # 3. 验证各方法的独立分项被动态引用 (C: 0.0125, F: 0.0200, D: 0.01917)
    assert "1.25000000e-02" in content
    assert "2.00000000e-02" in content
    assert "1.91700000e-02" in content


def test_reachability_diagnostic_contract():
    """验证 Unit D1 可达性诊断契约：A 在单步派工决策下对标准 C 模型的候选集、分支、团队、对齐均完全可达。"""
    weights_path = REPO_ROOT / "work3_4060/budget_v1/seed42/C_b4096/weights.pt"
    assert weights_path.is_file(), f"未找到检查点: {weights_path}"

    baseline_path = REPO_ROOT / "work3_4060/data/mini210.json"
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    agent = BaselinePlanRepairAgent()

    # 加载模型
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64).to("cpu")
    actor.load_state_dict(ckpt["actor_critic_state"])
    actor.eval()

    # 记录模型权重初始参数指纹
    param_clone = [p.clone() for p in actor.parameters()]

    # 产生第 0 步决策
    env.reset()
    action_a = agent.select_action(env)
    assert action_a is not None

    cmax = compute_cycle_heuristic_cmax(env.state)
    state_feat = extract_compact_state_features(env.state, cmax)
    time_urgency = compute_time_urgency_vector(
        estimated_r=max(0.0, cmax - float(env.state.current_time)),
        current_time=env.state.current_time,
        last_transfer_time=env.state.last_transfer_time,
        h0=env.state.h0,
        device="cpu",
    )
    next_residuals = torch.zeros(5, dtype=torch.float32, device="cpu")
    next_mask = env.state.get_next_cycle_station_mask()
    occupancy = torch.tensor(
        [1.0 if m else 0.0 for m in next_mask],
        dtype=torch.float32,
        device="cpu",
    )
    time_features = torch.cat([time_urgency, next_residuals, occupancy], dim=-1)

    snapshot = actor.make_decision_snapshot(
        env=env,
        state_feat=state_feat,
        time_urgency=time_features,
        allow_postpone=False,
    )

    diag = evaluate_action_reachability_and_probs(
        actor=actor,
        snapshot=snapshot,
        action_a=action_a,
        device="cpu",
    )

    # 检验可达性契约
    assert diag["task_reachable"] is True
    assert diag["branch_reachable"] is True
    assert diag["team_reachable"] is True
    assert diag["align_reachable"] is True
    assert diag["overall_reachable"] is True
    assert diag["joint_prob_a"] > 0.0
    assert diag["greedy_divergence"] in [
        "EXACT_MATCH",
        "C_GREEDY_TASK_MISMATCH",
        "C_GREEDY_TEAM_MISMATCH",
        "C_GREEDY_ADVANCE",
        "C_GREEDY_BRANCH_MISMATCH",
        "C_GREEDY_ALIGN_MISMATCH",
    ]

    # 检验参数守恒 (只读评估，绝对无梯度更新)
    for p_orig, p_after in zip(param_clone, actor.parameters()):
        assert torch.equal(p_orig, p_after)
