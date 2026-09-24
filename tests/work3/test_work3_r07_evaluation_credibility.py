"""R07 专项测试：评测可信度契约（未完工/不可行轨迹禁止计入改善百分比、命中子集分列、JSON 严禁 NaN/Inf）。"""

from __future__ import annotations

import json

import scripts.work3.evaluate_c_vs_d as eval_mod
from scripts.work3.evaluate_c_vs_d import run_benchmark_evaluation


def test_r07_incomplete_or_infeasible_trajectory_cannot_produce_improvement_pct(
    monkeypatch,
    tmp_path,
):
    """当 C 或 D 任一轨迹未完工 (success=False) 或不可行 (feasible=False) 时，improvement_j_total_pct 必须为 None。"""
    split_file = tmp_path / "test_split.json"
    scenarios_file = tmp_path / "scenarios.json"
    out_file = tmp_path / "eval_out.json"

    fake_scenarios = [
        {
            "scenario_id": "EARLY_LOW_S0",
            "timing": "EARLY",
            "intensity": "LOW",
            "station_id": 0,
            "aircraft_id": 5,
            "tau": 1500.0,
            "delta": 40.0,
            "recovery_time": 1540.0,
            "affected_task_keys": ["5_1"],
        },
        {
            "scenario_id": "MID_HIGH_S1",
            "timing": "MID",
            "intensity": "HIGH",
            "station_id": 1,
            "aircraft_id": 4,
            "tau": 1550.0,
            "delta": 100.0,
            "recovery_time": 1650.0,
            "affected_task_keys": ["4_2"],
        },
    ]
    scenarios_file.write_text(json.dumps(fake_scenarios), encoding="utf-8")
    split_file.write_text(json.dumps(fake_scenarios), encoding="utf-8")

    monkeypatch.setattr(eval_mod, "build_formal_evaluation_agent", lambda *a, **kw: object())
    monkeypatch.setattr(eval_mod, "AirLineEnvWork3", lambda *a, **kw: object())

    def _fake_eval(env, agent_type, agent, scenario=None, max_decisions=10000, device="cpu"):
        if scenario is None:
            return {
                "agent": agent_type,
                "scenario_id": "NOMINAL",
                "success": True,
                "feasible": True,
                "j_total": 4.0,
                "time_prediction_metrics": {"sample_count": 20},
            }
        sc_id = scenario["scenario_id"]
        if sc_id == "EARLY_LOW_S0":
            # C 成功，但 D 在中途死锁 (success=False)，累计成本看似更低 (0.10 < 1.00)
            is_c = agent_type == "Method-C"
            return {
                "agent": agent_type,
                "scenario_id": sc_id,
                "j_takt": 0.5 if is_c else 0.05,
                "d_time": 0.2 if is_c else 0.02,
                "d_team": 0.1 if is_c else 0.01,
                "j_postpone": 0.1 if is_c else 0.01,
                "j_revision": 0.1 if is_c else 0.01,
                "j_total": 1.00 if is_c else 0.10,
                "success": True if is_c else False,
                "feasible": True if is_c else False,
                "termination_reason": "completed" if is_c else "deadlock",
                "actual_disturbance_hits": 1,
                "raw_cost_components": {"diagnostic": float("nan")},
            }
        # MID_HIGH_S1: C 与 D 均成功且可行，且实际命中 > 0
        is_c = agent_type == "Method-C"
        return {
            "agent": agent_type,
            "scenario_id": sc_id,
            "j_takt": 1.0 if is_c else 0.8,
            "d_time": 0.5 if is_c else 0.4,
            "d_team": 0.2 if is_c else 0.2,
            "j_postpone": 0.2 if is_c else 0.1,
            "j_revision": 0.1 if is_c else 0.1,
            "j_total": 2.00 if is_c else 1.60,
            "success": True,
            "feasible": True,
            "termination_reason": "completed",
            "actual_disturbance_hits": 2,
        }

    monkeypatch.setattr(eval_mod, "evaluate_single_trajectory", _fake_eval)

    report = run_benchmark_evaluation(
        baseline_path="data/work3/real_283_k10_baseline.json",
        scenarios_path=str(scenarios_file),
        scenario_split_path=str(split_file),
        output_json=str(out_file),
    )
    results = report["scenario_results"]

    assert report["schema_version"] == "work3_eval_c_vs_d_v2"
    assert report["nominal_method_c"]["scenario_id"] == "NOMINAL"
    assert report["nominal_method_c"]["time_prediction_metrics"]["sample_count"] == 20
    # 第一条场景 D 未完工，严禁算出 +90% 伪改善，必须为 None 且标注无效原因
    assert results[0]["improvement_j_total_pct"] is None
    assert results[0].get("comparison_valid") is False
    assert "deadlock" in str(results[0].get("comparison_status", ""))

    # 第二条场景双方均成功且可行，改善幅度为 (2.0 - 1.6) / 2.0 = 20%
    assert results[1]["comparison_valid"] is True
    assert abs(float(results[1]["improvement_j_total_pct"]) - 20.0) < 1e-6
    persisted = json.loads(out_file.read_text(encoding="utf-8"))
    assert persisted["scenario_results"][0]["method_c"]["raw_cost_components"]["diagnostic"] is None
    json.dumps(persisted, allow_nan=False)
    assert persisted["credibility_summary"]["valid_comparison_count"] == 1


def test_r07_credibility_summary_separates_hit_subset_and_forbids_nan_json(tmp_path):
    """汇总报告必须无条件统计失败率，并将双方实际命中扰动 (actual_disturbance_hits > 0) 的有效子集单列，JSON 无 NaN/Inf。"""
    assert hasattr(eval_mod, "summarize_benchmark_credibility")
    sample_items = [
        {
            "scenario_id": "S_FAIL",
            "method_c": {"success": False, "feasible": False, "termination_reason": "deadlock", "actual_disturbance_hits": 1, "j_total": 0.5},
            "method_d": {"success": True, "feasible": True, "termination_reason": "completed", "actual_disturbance_hits": 1, "j_total": 1.2},
            "improvement_j_total_pct": None,
            "comparison_valid": False,
            "comparison_status": "invalid_incomplete_or_infeasible: c_reason=deadlock, d_reason=completed",
        },
        {
            "scenario_id": "S_ZERO_HIT",
            "method_c": {"success": True, "feasible": True, "termination_reason": "completed", "actual_disturbance_hits": 0, "j_total": 1.0},
            "method_d": {"success": True, "feasible": True, "termination_reason": "completed", "actual_disturbance_hits": 0, "j_total": 0.5},
            "improvement_j_total_pct": 50.0,
            "comparison_valid": True,
            "both_hit_disturbance": False,
            "comparison_status": "valid_both_completed_zero_hit",
        },
        {
            "scenario_id": "S_BOTH_HIT",
            "method_c": {"success": True, "feasible": True, "termination_reason": "completed", "actual_disturbance_hits": 2, "j_total": 2.0},
            "method_d": {"success": True, "feasible": True, "termination_reason": "completed", "actual_disturbance_hits": 2, "j_total": 1.5},
            "improvement_j_total_pct": 25.0,
            "comparison_valid": True,
            "both_hit_disturbance": True,
            "comparison_status": "valid_both_completed_and_hit",
        },
        {
            "scenario_id": "S_ONE_SIDED_HIT",
            "method_c": {"success": True, "feasible": True, "termination_reason": "completed", "actual_disturbance_hits": 1, "j_total": 2.0},
            "method_d": {"success": True, "feasible": True, "termination_reason": "completed", "actual_disturbance_hits": 0, "j_total": 1.5},
            "improvement_j_total_pct": 25.0,
            "comparison_valid": True,
            "both_hit_disturbance": False,
            "comparison_status": "valid_both_completed_one_sided_hit",
        },
    ]
    summary = eval_mod.summarize_benchmark_credibility(sample_items)
    one_sided = eval_mod.compare_c_vs_d_pair(
        sample_items[3]["method_c"], sample_items[3]["method_d"]
    )
    assert one_sided["comparison_valid"] is True
    assert one_sided["both_hit_disturbance"] is False
    assert one_sided["comparison_status"] == "valid_both_completed_one_sided_hit"
    assert summary["total_scenarios"] == 4
    assert abs(summary["method_c_failure_rate"] - 0.25) < 1e-6
    assert summary["method_d_failure_rate"] == 0.0
    assert summary["valid_comparison_count"] == 3
    assert abs(summary["mean_improvement_valid_only_pct"] - (100.0 / 3.0)) < 1e-6
    assert summary["valid_and_both_hit_count"] == 1
    assert abs(summary["mean_improvement_valid_and_hit_only_pct"] - 25.0) < 1e-6
    assert summary["valid_zero_hit_count"] == 1
    assert summary["mean_improvement_valid_zero_hit_pct"] == 50.0
    assert summary["valid_one_sided_hit_count"] == 1
    assert summary["mean_improvement_valid_one_sided_hit_pct"] == 25.0

    # 严格 JSON 序列化无 NaN / Inf
    serialized = json.dumps(summary, allow_nan=False)
    assert "NaN" not in serialized and "Infinity" not in serialized


def test_r07_prediction_mae_uses_only_cycles_with_real_transfer_labels():
    records = [
        (1, 12.0, 14.0, 10.0),
        (1, 18.0, 19.0, 10.0),
        (2, 25.0, 28.0, 10.0),
    ]

    metrics = eval_mod.summarize_cycle_prediction_errors(records, [20.0])

    assert metrics["labeled_cycle_count"] == 1
    assert metrics["sample_count"] == 2
    assert metrics["heuristic_mae_hours"] == 5.0
    assert metrics["heuristic_mae_h0"] == 0.5
    assert metrics["corrected_mae_hours"] == 3.5
    assert metrics["corrected_mae_h0"] == 0.35

    no_completed_cycles = eval_mod.summarize_cycle_prediction_errors(records, [])
    assert no_completed_cycles["labeled_cycle_count"] == 0
    assert no_completed_cycles["sample_count"] == 0
    assert no_completed_cycles["heuristic_mae_hours"] is None
    assert no_completed_cycles["corrected_mae_h0"] is None
