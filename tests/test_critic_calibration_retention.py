"""价值头专属校准 + PPO 保持性匹配对照的针对性验证（第 9 节 1–6 项）。"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import work3_4060.run_critic_calibration_retention as cal
from work3_4060.c_reproduction_protocol import (
    actor_fingerprint,
    load_json,
    sha256_file,
    state_dict_fingerprint,
)


# 1. 只有价值专属参数被校准；策略子集不变；完整指纹按设计改变


def test_only_value_head_is_calibrated() -> None:
    if not cal.CAL_MODEL.is_file():
        pytest.skip("缺少校准副本")
    if not cal.BC_WEIGHTS.is_file():
        pytest.skip("缺少 BC 副本")
    calibrated, payload = cal.load_calibrated_actor(cal.CAL_MODEL)
    bc, _ = cal.prev.load_diagnostic_actor(
        cal.BC_WEIGHTS,
        expected_teacher_manifest_sha256=load_json(cal.BC_MANIFEST)["teacher"]["manifest_sha256"],
    )
    groups = cal.prev.parameter_groups(calibrated)
    assert groups["counts"]["value"] == 4
    assert all(name.startswith("critic.") for name in groups["groups"]["value"])

    policy = groups["groups"]["shared"] + groups["groups"]["policy"]
    assert cal.subset_fingerprint(calibrated.state_dict(), policy) == cal.subset_fingerprint(
        bc.state_dict(), policy
    ), "策略子集指纹必须严格不变"
    assert actor_fingerprint(calibrated) != actor_fingerprint(bc), "完整指纹（含 critic）应按设计改变"
    value_changed = any(
        not torch.equal(calibrated.state_dict()[name], bc.state_dict()[name])
        for name in groups["groups"]["value"]
    )
    assert value_changed, "价值头必须被更新"

    invariance = load_json(cal.CAL_INVARIANCE)
    assert invariance["policy_unchanged"] is True
    assert invariance["greedy_action_mismatch"] == 0
    assert invariance["max_abs_log_prob_difference"] <= cal.POLICY_PROB_TOLERANCE


# 2. MC 递推、gamma 索引、终局零 bootstrap、势函数平移；非有限标签拒绝


def test_mc_recursion_and_shift_identity() -> None:
    gamma, beta = 0.99, 1.0
    phi = [-0.5, -0.5, 0.0]
    raw = [-0.3, 0.2, 0.0]
    shaped = [raw[i] + beta * (gamma * phi[i + 1] - phi[i]) if i + 1 < 3 else raw[i] - beta * phi[i]
              for i in range(3)]
    samples = [
        {"raw_reward": raw[i], "shaped_reward": shaped[i], "phi": phi[i]} for i in range(3)
    ]
    mc = cal._mc_targets(samples, gamma=gamma, beta=beta)
    # 从尾部递推：G_2 = r_2, G_1 = r_1 + gamma*G_2, G_0 = r_0 + gamma*G_1
    assert mc["shaped"][2] == pytest.approx(shaped[2])
    assert mc["shaped"][1] == pytest.approx(shaped[1] + gamma * mc["shaped"][2])
    assert mc["shaped"][0] == pytest.approx(shaped[0] + gamma * mc["shaped"][1])
    # 逐 t 的势函数平移（折扣从 t 重新起算）
    for index in range(3):
        assert mc["shaped"][index] - mc["raw"][index] == pytest.approx(-beta * phi[index], abs=1e-12)
    assert mc["identity_check"]["passed"] is True


def test_non_finite_mc_labels_are_rejected() -> None:
    cal.validate_mc_targets({"shaped": [1.0, 2.0], "raw": [0.5, 0.5]})
    with pytest.raises(ValueError):
        cal.validate_mc_targets({"shaped": [1.0, float("nan")], "raw": [0.0, 0.0]})
    with pytest.raises(ValueError):
        cal.validate_mc_targets({"shaped": [float("inf")], "raw": [0.0]})
    with pytest.raises(ValueError):
        cal.validate_mc_targets({"shaped": [], "raw": []})


# 3. 留出样本不参与优化；固定预算；终点是最后一次权重而非最好检查点


def test_holdout_never_trained_and_budget_fixed() -> None:
    if not cal.CAL_HISTORY.is_file():
        pytest.skip("缺少校准历史")
    rows = [json.loads(line) for line in cal.CAL_HISTORY.read_text(encoding="utf-8").splitlines() if line.strip()]
    fit = load_json(cal.CAL_FIT)
    holdout = set(fit["samples"]["holdout"])
    for row in rows:
        assert not (set(row["batch_indices"]) & holdout), "留出样本不得进入优化"
    assert fit["calibration_steps"] <= cal.CAL_MAX_STEPS
    assert len(rows) == fit["calibration_steps"]
    assert fit["stop_reason"] in {"max_steps", "max_wall_seconds", "non_finite_loss"}
    # 终点必须是最后一次扫描，不是按留出误差挑出来的最好点
    assert fit["final_scan"] == fit["scans"][-1]
    assert int(fit["scans"][-1]["step"]) == fit["calibration_steps"]


def test_scans_cover_start_interval_and_end() -> None:
    fit = load_json(cal.CAL_FIT)
    steps = [int(scan["step"]) for scan in fit["scans"]]
    assert steps[0] == 0
    assert steps[-1] == fit["calibration_steps"]
    assert all((b - a) == cal.CAL_SCAN_INTERVAL for a, b in zip(steps, steps[1:]) if b != steps[-1])


# 4. 校准优化器不进入 PPO；PPO 全参数可训练；RNG 隔离与首段配对可核验


def test_calibration_state_does_not_leak_into_ppo() -> None:
    if not cal.CAL_MODEL.is_file():
        pytest.skip("缺少校准副本")
    payload = torch.load(cal.CAL_MODEL, map_location="cpu", weights_only=False)
    assert "optimizer" not in json.dumps(sorted(payload.keys())), "校准优化器状态不得保存进模型"
    summary = load_json(cal.PPO_SUMMARY)
    assert summary["calibration"]["model_sha256"] == sha256_file(cal.CAL_MODEL)
    assert summary["hyperparameters"]["learning_rate"] == 3e-4

    step0 = torch.load(cal.PPO_CKPT_DIR / "step0000.pt", map_location="cpu", weights_only=False)
    assert state_dict_fingerprint(step0["actor_critic_state"]) == payload["actor_fingerprint"], (
        "PPO step0000 必须等于校准副本权重"
    )
    assert step0["optimizer_steps"] == 0


def test_first_rollout_pairing_recorded_and_consistent() -> None:
    pairing = load_json(cal.PPO_PAIRING)
    assert pairing["paired"] is True
    assert pairing["same_states"] is True
    assert pairing["same_actions"] is True
    assert pairing["max_abs_log_prob_difference"] <= pairing["tolerance"]
    assert pairing["steps_compared"] == cal.prev.ROLLOUT_STEPS


# 5. 有序 PPO 概率与无序团队评分分开；C 入口无强制 align、无教师注入


def test_ordered_ppo_path_and_no_intervention() -> None:
    assert not hasattr(cal, "AlignForcedAgent")
    assert not hasattr(cal.prev, "AlignForcedAgent")
    train_source = inspect.getsource(cal.stage_train)
    assert "deterministic=False" in train_source, "训练必须随机采样"
    assert "align" not in train_source.lower() or "align=" not in train_source, "训练不得强制 align"
    evaluate_source = inspect.getsource(cal.stage_evaluate_one)
    assert "scenario=None" in evaluate_source
    assert "FormalEvaluationAgent" in evaluate_source
    config = cal.PPO_CONFIG.read_text(encoding="utf-8")
    assert "disturbance_enabled: false" in config
    assert "scenario: NOMINAL" in config
    # 有序 PPO 概率与无序团队边缘评分是两个口径：本模块只做前者，不做 logsumexp 聚合
    assert "logsumexp" not in inspect.getsource(cal.stage_train)


# 6. 勘误、差值方向、近零分母与状态优先级


def test_errata_fixed_shift_is_exact() -> None:
    payload = load_json(cal.OUTPUT_DIR / "errata_verification.json")
    assert payload["all_within_tolerance"] is True
    for item in payload["episodes"]:
        assert abs(item["residual"]) <= 1e-9
        if item["terminated"]:
            assert item["expected_shift_general"] == pytest.approx(
                -payload["beta"] * item["phi_start"], abs=1e-12
            )


def test_effect_direction_and_protocol_precedence() -> None:
    assert cal._effect_status(1.0, 0.5) == "校准臂费用更低"
    assert cal._effect_status(0.5, 1.0) == "校准臂费用更高"
    assert cal._effect_status(1.0, 1.0) == "近似持平"
    expected = [{"path": "a"}, {"path": "b"}]
    assert cal.prev.classify_protocol_status(expected, [{"path": "a", "status": "PASS"}, {"path": "b", "status": "PASS"}]) == "PASS"
    assert cal.prev.classify_protocol_status(expected, [{"path": "a", "status": "PASS"}]) == "INCOMPLETE"
    assert cal.prev.classify_protocol_status(expected, [{"path": "a", "status": "FAIL"}]) == "FAIL"


def test_comparison_has_no_percentage_denominator() -> None:
    if not cal.COMPARISON_CSV.is_file():
        pytest.skip("缺少对比表")
    header = cal.COMPARISON_CSV.read_text(encoding="utf-8").splitlines()[0]
    assert "%" not in header and "percent" not in header.lower()
    rows = [line.split(",") for line in cal.COMPARISON_CSV.read_text(encoding="utf-8").splitlines()[1:] if line]
    assert len(rows) == len(cal.CHECKPOINTS)
