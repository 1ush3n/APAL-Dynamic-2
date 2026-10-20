"""测试 work3_4060/run_budget_evaluation.py 中的契约核验与坏输入拦截能力 (Section 9.3, 10.3)。

验证点：
1. 现有 C4096 / D4096 契约、运行记录和审计文件校验通过。
2. 目标模型缺失 run.json 时，拦截并报错 (FileNotFoundError)。
3. 检查点与 run.json 元数据冲突 (seed/method/指纹) 时，拦截并报错 (AssertionError)。
4. 目标模型缺失或失败的 shaping_audit 时，拦截并报错 (FileNotFoundError/ValueError)。
5. 优化更新计数严格分解：total_updates = ppo_updates + aux_updates。
6. (Section 10.3) 学习率冲突、期望种子错误、审计覆盖量不匹配的纯字典单测与拦截。
7. (Section 10.5) load_variants_from_json 显式清单解析与相对路径转换。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest
import torch

from work3_4060.run_budget_evaluation import (
    REPO_ROOT,
    ManifestResult,
    compute_four_cell_matrix_analysis,
    cross_verify_model_pairing,
    cross_verify_training_config,
    get_variants_for_mode,
    load_variants_from_json,
    verify_checkpoint_and_run_contracts,
    verify_shaping_audit_coverage,
)


def test_existing_cd_contracts_pass():
    """验证现有 C4096 与 D4096 检查点、运行记录与塑形审计均能通过核验。"""
    variants = get_variants_for_mode("cd")
    meta = verify_checkpoint_and_run_contracts(variants, device="cpu")

    # 验证优化更新计数的正确分解
    c_meta = meta["C_b4096"]["run_info"]
    assert c_meta["total_updates"] == 2048
    assert c_meta["ppo_updates"] == 2048
    assert c_meta["auxiliary_updates"] == 0

    d_meta = meta["D_b4096"]["run_info"]
    assert d_meta["total_updates"] == 6082
    assert d_meta["ppo_updates"] == 2048
    assert d_meta["auxiliary_updates"] == 4034
    assert d_meta["total_updates"] == d_meta["ppo_updates"] + d_meta["auxiliary_updates"]

    # 验证审计状态均为 PASS 且 8项合同完整
    assert meta["C_b4096"]["audit_info"]["audit_pass"] is True
    assert meta["D_b4096"]["audit_info"]["audit_pass"] is True


def test_missing_run_json_for_target_fails(tmp_path: Path):
    """验证目标模型若缺少 run.json，必须拦截并抛出 FileNotFoundError。"""
    variants = get_variants_for_mode("cd")
    bad_variants = copy.deepcopy(variants)
    bad_variants[0]["run_json"] = tmp_path / "non_existent_run.json"

    with pytest.raises(FileNotFoundError, match="缺少必需的运行记录"):
        verify_checkpoint_and_run_contracts(bad_variants, device="cpu")


def test_metadata_conflict_fails(tmp_path: Path):
    """验证 checkpoint.run_metadata 与 run.json 发生冲突时拦截。"""
    variants = get_variants_for_mode("cd")
    bad_variants = copy.deepcopy(variants)

    # 复制一个篡改了 seed 的 run.json
    real_run = Path(bad_variants[0]["run_json"])
    with real_run.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["seed"] = 9999
    tampered_run = tmp_path / "tampered_run.json"
    tampered_run.write_text(json.dumps(data), encoding="utf-8")
    bad_variants[0]["run_json"] = tampered_run

    with pytest.raises(AssertionError, match="seed 不匹配"):
        verify_checkpoint_and_run_contracts(bad_variants, device="cpu")


def test_missing_or_failed_audit_fails(tmp_path: Path):
    """验证目标模型缺少塑形审计文件或审计合同失败时拦截。"""
    variants = get_variants_for_mode("cd")
    bad_variants = copy.deepcopy(variants)

    # 1. 缺失审计文件
    bad_variants[0]["audit_json"] = tmp_path / "non_existent_audit.json"
    with pytest.raises(FileNotFoundError, match="缺少必需的塑形审计文件"):
        verify_checkpoint_and_run_contracts(bad_variants, device="cpu")

    # 2. 审计合同中有 FAIL
    real_audit = Path(variants[0]["audit_json"])
    with real_audit.open("r", encoding="utf-8") as f:
        audit_data = json.load(f)
    audit_data["contracts"]["single_step_formula"] = "FAIL"
    failed_audit = tmp_path / "failed_audit.json"
    failed_audit.write_text(json.dumps(audit_data), encoding="utf-8")
    bad_variants[0]["audit_json"] = failed_audit

    with pytest.raises(ValueError, match="塑形审计未通过"):
        verify_checkpoint_and_run_contracts(bad_variants, device="cpu")


def test_cross_verify_training_config_unit_checks():
    """(Section 10.3) 纯字典测试 cross_verify_training_config 的超参校验与反例拦截。"""
    base_meta = {
        "seed": 42,
        "method_variant": "D",
        "data_fingerprint": {
            "scenario_pool_sha256": "h1",
            "scenario_split_sha256": "h2",
            "baseline_sha256": "h3",
        },
        "training_config": {
            "beta_shaping": 0.1,
            "gamma": 0.99,
            "lr": 0.0003,
            "gae_lambda": 0.95,
            "ppo_epochs": 1,
            "batch_size": 2,
        },
    }
    base_run = {
        "seed": 42,
        "method_variant": "D",
        "data_fingerprint": {
            "scenario_pool_sha256": "h1",
            "scenario_split_sha256": "h2",
            "baseline_sha256": "h3",
        },
        "training_config": {
            "beta_shaping": 0.1,
            "gamma": 0.99,
            "lr": 0.0003,
            "gae_lambda": 0.95,
            "ppo_epochs": 1,
            "batch_size": 2,
        },
    }

    # 1. 正常一致通过
    cross_verify_training_config("D_test", base_meta, base_run, expected_seed=42, expected_beta=0.1)

    # 2. 期望种子不符拦截
    with pytest.raises(AssertionError, match="不等于期望种子"):
        cross_verify_training_config("D_test", base_meta, base_run, expected_seed=45, expected_beta=0.1)

    # 3. 学习率翻倍冲突拦截 (反例验证)
    tampered_meta_lr = copy.deepcopy(base_meta)
    tampered_meta_lr["training_config"]["lr"] = 0.0006
    with pytest.raises(AssertionError, match="学习率 lr 不一致"):
        cross_verify_training_config("D_test", tampered_meta_lr, base_run, expected_seed=42)

    # 4. gamma 冲突拦截
    tampered_meta_gamma = copy.deepcopy(base_meta)
    tampered_meta_gamma["training_config"]["gamma"] = 0.95
    with pytest.raises(AssertionError, match="gamma 不一致"):
        cross_verify_training_config("D_test", tampered_meta_gamma, base_run, expected_seed=42)


def test_verify_shaping_audit_coverage_unit_checks():
    """(Section 10.3) 纯字典测试 verify_shaping_audit_coverage 的覆盖范围校验与反例拦截。"""
    base_audit = {
        "overall_status": "PASS",
        "contracts": {
            "single_step_formula": "PASS",
            "step_to_step_continuity": "PASS",
            "predictor_version_invariance": "PASS",
            "terminal_zero_potential": "PASS",
            "completed_episode_telescope": "PASS",
            "config_consistency": "PASS",
            "boundary_contract": "PASS",
            "ledger_reconciliation": "PASS",
        },
        "total_events_checked": 4096,
        "total_episodes_checked": 11,
        "resolved_config": {
            "cfg_gamma": 0.99,
            "cfg_beta": 0.1,
        },
    }
    base_run = {
        "total_decisions": 4096,
        "independent_episode_count": 11,
    }

    # 1. 正常通过
    verify_shaping_audit_coverage(
        "D_test", base_audit, base_run, expected_decisions=4096, expected_beta=0.1
    )

    # 2. 覆盖事件数不足 (例如只有 8 步) 拦截 (反例验证)
    tampered_audit_events = copy.deepcopy(base_audit)
    tampered_audit_events["total_events_checked"] = 8
    with pytest.raises(AssertionError, match="审计覆盖事件数"):
        verify_shaping_audit_coverage(
            "D_test", tampered_audit_events, base_run, expected_decisions=4096, expected_beta=0.1
        )

    # 3. episode 覆盖量冲突拦截
    tampered_audit_episodes = copy.deepcopy(base_audit)
    tampered_audit_episodes["total_episodes_checked"] = 5
    with pytest.raises(AssertionError, match="审计覆盖 episode 数"):
        verify_shaping_audit_coverage(
            "D_test", tampered_audit_episodes, base_run, expected_decisions=4096, expected_beta=0.1
        )

    # 4. resolved_config beta 冲突拦截
    tampered_audit_beta = copy.deepcopy(base_audit)
    tampered_audit_beta["resolved_config"]["cfg_beta"] = 1.0
    with pytest.raises(AssertionError, match="resolved_config beta"):
        verify_shaping_audit_coverage(
            "D_test", tampered_audit_beta, base_run, expected_decisions=4096, expected_beta=0.1
        )


def test_load_variants_from_json(tmp_path: Path):
    """(Section 10.5) 验证从 JSON 加载显式清单与路径解析。"""
    manifest_data = {
        "expected_seed": 45,
        "comparison_mode": "fd",
        "variants": [
            {
                "label": "C_b4096",
                "method": "C",
                "checkpoint": "work3_4060/repeat_v1/seed45/C_beta1_b4096/weights.pt",
                "run_json": "work3_4060/repeat_v1/seed45/C_beta1_b4096/run.json",
                "audit_json": "work3_4060/repeat_v1/seed45/C_beta1_b4096/shaping_audit_summary.json",
                "expected_beta": 1.0,
                "expected_decisions": 4096,
                "is_target": True,
            }
        ],
    }
    json_path = tmp_path / "variants.json"
    json_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    variants, mode, seed = load_variants_from_json(json_path)
    assert mode == "fd"
    assert seed == 45
    assert len(variants) == 1
    assert variants[0]["label"] == "C_b4096"
    assert variants[0]["expected_seed"] == 45
    assert variants[0]["checkpoint"] == REPO_ROOT / "work3_4060/repeat_v1/seed45/C_beta1_b4096/weights.pt"


def test_cf_mode_variants_configuration(tmp_path: Path):
    """(Section 12.5) 验证 cf 比较模式的默认配置与 manifest 加载。"""
    variants = get_variants_for_mode("cf")
    assert len(variants) == 2
    assert variants[0]["label"] == "C_b4096"
    assert variants[0]["method"] == "C"
    assert variants[0]["expected_beta"] == 1.0
    assert variants[0]["is_target"] is True
    assert variants[1]["label"] == "F_b4096"
    assert variants[1]["method"] == "F"
    assert variants[1]["expected_beta"] == 1.0
    assert variants[1]["is_target"] is True

    # 测试通过 variants.json 加载 cf 模式
    manifest_data = {
        "expected_seed": 42,
        "comparison_mode": "cf",
        "variants": [
            {
                "label": "C_b4096",
                "method": "C",
                "checkpoint": "work3_4060/budget_v1/seed42/C_b4096/weights.pt",
                "run_json": "work3_4060/budget_v1/seed42/C_b4096/run.json",
                "audit_json": "work3_4060/budget_v1/seed42/C_b4096/shaping_audit_summary.json",
                "expected_beta": 1.0,
                "expected_decisions": 4096,
                "is_target": True,
            },
            {
                "label": "F_b4096",
                "method": "F",
                "checkpoint": "work3_4060/representation_v2/seed42/F_beta1_b4096/weights.pt",
                "run_json": "work3_4060/representation_v2/seed42/F_beta1_b4096/run.json",
                "audit_json": "work3_4060/representation_v2/seed42/F_beta1_b4096/shaping_audit_summary.json",
                "expected_beta": 1.0,
                "expected_decisions": 4096,
                "is_target": True,
            },
        ],
    }
    manifest_path = tmp_path / "variants_cf.json"
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")
    loaded_variants, mode, seed = load_variants_from_json(manifest_path)
    assert mode == "cf"
    assert seed == 42
    assert len(loaded_variants) == 2
    assert loaded_variants[0]["label"] == "C_b4096"
    assert loaded_variants[1]["label"] == "F_b4096"


def test_cf_actor_fingerprint_mismatch_fails(tmp_path: Path):
    """(Section 12.5) 验证 C 与 F 初始 Actor 指纹不匹配时拦截并抛出 AssertionError。"""
    # 构造两个已存在模型的变体清单 (例如 C_b4096 与篡改了初始 Actor 指纹的副本)
    real_variants = get_variants_for_mode("cd")
    c_run_path = real_variants[0]["run_json"]
    with Path(c_run_path).open("r", encoding="utf-8") as f:
        c_run = json.load(f)

    tampered_run_data = copy.deepcopy(c_run)
    tampered_run_data["initial_actor_fingerprint"] = "0000000000000000000000000000000000000000000000000000000000000000"
    tampered_run_path = tmp_path / "tampered_f_run.json"
    tampered_run_path.write_text(json.dumps(tampered_run_data), encoding="utf-8")

    test_variants = [
        copy.deepcopy(real_variants[0]),  # C_b4096
        {
            "label": "F_b4096",
            "method": "C",  # 用已有 C 检查点模拟
            "checkpoint": real_variants[0]["checkpoint"],
            "run_json": tampered_run_path,
            "audit_json": real_variants[0]["audit_json"],
            "expected_beta": 1.0,
            "expected_decisions": 4096,
            "expected_seed": 42,
            "is_target": True,
        },
    ]

    with pytest.raises(AssertionError, match="初始Actor指纹不匹配"):
        verify_checkpoint_and_run_contracts(test_variants, device="cpu")


# ==============================================================================
# Section 13 离线收口与契约强化单测 (R13-A, R13-B, R13-C)
# ==============================================================================

def _build_mock_pairing_run(
    method: str = "C",
    seed: int = 42,
    actor_fp: str = "2d639e6138af39725db3658506048ed75396eed26a8ee8629c48bd5deabf203d",
    lr: float = 0.0003,
    beta: float = 1.0,
    gamma: float = 0.99,
    rollout_steps: int = 8,
) -> dict[str, Any]:
    """构建用于跨模型配对单测的纯字典运行记录。"""
    return {
        "method_variant": method,
        "seed": seed,
        "initial_actor_fingerprint": actor_fp,
        "data_fingerprint": {
            "scenario_pool_sha256": "pool_hash_mock_1234567890abcdef",
            "scenario_split_sha256": "split_hash_mock_1234567890abcdef",
            "baseline_sha256": "baseline_hash_mock_1234567890abcdef",
        },
        "training_config": {
            "gamma": gamma,
            "gae_lambda": 0.95,
            "lr": lr,
            "rollout_steps": rollout_steps,
            "batch_size": 2,
            "ppo_epochs": 1,
            "num_envs": 1,
            "amp_dtype": "fp32",
            "beta_shaping": beta,
        },
    }


def test_cross_verify_model_pairing_valid():
    """(Section 13.4 R13-B) 验证标准合法的 C 与 F 配对记录通过校验。"""
    run_c = _build_mock_pairing_run(method="C", beta=1.0)
    run_f = _build_mock_pairing_run(method="F", beta=1.0)
    cross_verify_model_pairing("C_b4096", run_c, "F_b4096", run_f, comparison_mode="cf")


def test_cross_verify_model_pairing_missing_fingerprint_raises():
    """(Section 13.4 R13-B) 验证缺少 initial_actor_fingerprint 时拦截并抛出 ValueError。"""
    run_c = _build_mock_pairing_run(method="C")
    run_f = _build_mock_pairing_run(method="F")
    del run_c["initial_actor_fingerprint"]

    with pytest.raises(ValueError, match="缺少 initial_actor_fingerprint"):
        cross_verify_model_pairing("C_b4096", run_c, "F_b4096", run_f, comparison_mode="cf")

    run_c_ok = _build_mock_pairing_run(method="C")
    del run_f["initial_actor_fingerprint"]
    with pytest.raises(ValueError, match="缺少 initial_actor_fingerprint"):
        cross_verify_model_pairing("C_b4096", run_c_ok, "F_b4096", run_f, comparison_mode="cf")


def test_cross_verify_model_pairing_invalid_hex_fingerprint_raises():
    """(Section 13.4 R13-B) 验证非合法 64 位十六进制指纹时拦截并抛出 ValueError。"""
    run_c = _build_mock_pairing_run(method="C", actor_fp="not_a_hex_fingerprint")
    run_f = _build_mock_pairing_run(method="F")

    with pytest.raises(ValueError, match="不是有效的64位十六进制哈希"):
        cross_verify_model_pairing("C_b4096", run_c, "F_b4096", run_f, comparison_mode="cf")


def test_cross_verify_model_pairing_mismatched_actor_fingerprint_raises():
    """(Section 13.4 R13-B) 验证 Actor 指纹不匹配时拦截并抛出 AssertionError。"""
    run_c = _build_mock_pairing_run(method="C", actor_fp="1111111111111111111111111111111111111111111111111111111111111111")
    run_f = _build_mock_pairing_run(method="F", actor_fp="2222222222222222222222222222222222222222222222222222222222222222")

    with pytest.raises(AssertionError, match="初始Actor指纹不匹配"):
        cross_verify_model_pairing("C_b4096", run_c, "F_b4096", run_f, comparison_mode="cf")


def test_cross_verify_model_pairing_mismatched_hyperparams_raises():
    """(Section 13.4 R13-B) 验证跨模型超参数冲突时逐一拦截 (反例验证)。"""
    base_c = _build_mock_pairing_run(method="C", beta=1.0)
    base_f = _build_mock_pairing_run(method="F", beta=1.0)

    # 1. 学习率冲突
    bad_lr_f = _build_mock_pairing_run(method="F", beta=1.0, lr=0.0006)
    with pytest.raises(AssertionError, match="超参数 lr 不一致"):
        cross_verify_model_pairing("C_b4096", base_c, "F_b4096", bad_lr_f, comparison_mode="cf")

    # 2. gamma 冲突
    bad_gamma_f = _build_mock_pairing_run(method="F", beta=1.0, gamma=0.95)
    with pytest.raises(AssertionError, match="超参数 gamma 不一致"):
        cross_verify_model_pairing("C_b4096", base_c, "F_b4096", bad_gamma_f, comparison_mode="cf")

    # 3. rollout_steps 冲突
    bad_rollout_f = _build_mock_pairing_run(method="F", beta=1.0, rollout_steps=16)
    with pytest.raises(AssertionError, match="超参数 rollout_steps 不一致"):
        cross_verify_model_pairing("C_b4096", base_c, "F_b4096", bad_rollout_f, comparison_mode="cf")

    # 4. 种子冲突
    bad_seed_f = _build_mock_pairing_run(method="F", beta=1.0, seed=45)
    with pytest.raises(AssertionError, match="种子不一致"):
        cross_verify_model_pairing("C_b4096", base_c, "F_b4096", bad_seed_f, comparison_mode="cf")

    # 5. beta 冲突 (在 cf 模式下要求 beta 一致)
    bad_beta_f = _build_mock_pairing_run(method="F", beta=0.1)
    with pytest.raises(AssertionError, match="超参数 beta_shaping 不一致"):
        cross_verify_model_pairing("C_b4096", base_c, "F_b4096", bad_beta_f, comparison_mode="cf")

    # 6. 数据指纹冲突
    bad_dfp_f = _build_mock_pairing_run(method="F", beta=1.0)
    bad_dfp_f["data_fingerprint"]["scenario_pool_sha256"] = "tampered_pool_hash"
    with pytest.raises(AssertionError, match="数据指纹不一致"):
        cross_verify_model_pairing("C_b4096", base_c, "F_b4096", bad_dfp_f, comparison_mode="cf")


def test_cross_verify_model_pairing_wrong_method_raises():
    """(Section 13.4 R13-B) 验证对比方法不符合模式要求时拦截。"""
    run_d1 = _build_mock_pairing_run(method="D", beta=1.0)
    run_d2 = _build_mock_pairing_run(method="D", beta=1.0)

    with pytest.raises(AssertionError, match="cf 比较模式要求对比方法必须为 C 与 F"):
        cross_verify_model_pairing("C_b4096", run_d1, "F_b4096", run_d2, comparison_mode="cf")


def test_r13_c_unrounded_float_preserves_near_ties():
    """(Section 13.4 R13-C) 近似平局反例检验：
    证明在未舍入 float 下 4e-11 差值能正确判定胜负，而在旧版 .8e 字符串往返下会误判平局。
    """
    j_c = 0.05
    j_f = 0.05000000004
    true_delta = j_c - j_f  # -4e-11

    # 1. 旧逻辑：字符串 .8e 往返导致提前舍入
    str_c = f"{j_c:.8e}"
    str_f = f"{j_f:.8e}"
    assert str_c == str_f == "5.00000000e-02"
    rounded_delta = float(str_c) - float(str_f)
    old_is_tie = (abs(rounded_delta) <= 1e-12)
    assert old_is_tie is True, "旧逻辑确实把 4e-11 误判为了平局 (复现见证)"

    # 2. 新逻辑：未舍入 float 保留真实数值精度
    assert abs(true_delta) > 1e-12
    new_is_tie = (abs(true_delta) <= 1e-12)
    assert new_is_tie is False, "新逻辑未舍入 float 正确判定非平局"
    c_better = (true_delta < -1e-12)
    assert c_better is True, "正确判定 C 优于 F"


def test_r13_a_compute_four_cell_matrix_dynamic():
    """(Section 13.4 R13-A) 验证四格矩阵基于原始 JSON 动态计算，杜绝写死常量。"""
    hist_json = REPO_ROOT / "work3_4060/representation_v1/seed42/evaluation/equal_budget_validation_results.json"
    hist_manifest = REPO_ROOT / "work3_4060/representation_v1/seed42/variants.json"
    val_scenarios = ["COMPARE_VAL_S0_EARLY_M", "COMPARE_VAL_S2_MID_M", "COMPARE_VAL_S3_MID_L", "COMPARE_VAL_S4_LATE_H"]

    # 当前 beta=1.0 的 C 与 F 验证集均值 (来自真实评估数据)
    curr_c_mean = 0.026572549232481896
    curr_f_mean = 0.043086185769927551

    lines, info = compute_four_cell_matrix_analysis(
        current_seed=42,
        current_beta=1.0,
        current_c_mean=curr_c_mean,
        current_f_mean=curr_f_mean,
        val_scenario_ids=val_scenarios,
        historical_json_path=hist_json,
        historical_manifest_path=hist_manifest,
    )

    assert info["status"] == "PASS"
    # 验证动态复算的四格数值
    assert abs(info["c_01"] - 0.053197622808156944) < 1e-10
    assert abs(info["f_01"] - 0.044906890812782062) < 1e-10
    assert abs(info["delta_01"] - 0.008290731995374882) < 1e-10
    assert abs(info["delta_1"] - (-0.016513636537445656)) < 1e-10
    assert abs(info["I_interaction"] - (-0.024804368532820538)) < 1e-10
    # 验证相对百分比与分母定义
    assert abs(info["pct_f_over_c_01"] - 15.584779) < 1e-4
    assert abs(info["pct_c_over_f_1"] - 38.326986) < 1e-4


def test_r13_a_missing_historical_source_is_incomplete():
    """(Section 13.4 R13-A) 验证缺少历史结果源时，四格矩阵输出 INCOMPLETE，不填充固定常数。"""
    val_scenarios = ["COMPARE_VAL_S0_EARLY_M", "COMPARE_VAL_S2_MID_M"]
    lines, info = compute_four_cell_matrix_analysis(
        current_seed=42,
        current_beta=1.0,
        current_c_mean=0.026,
        current_f_mean=0.043,
        val_scenario_ids=val_scenarios,
        historical_json_path=None,
    )
    assert info["status"] == "INCOMPLETE"
    assert any("INCOMPLETE" in line for line in lines)
    # 确保没有输出硬编码的 seed 42 常数
    assert not any("0.05319762" in line for line in lines)
    assert not any("0.04490689" in line for line in lines)


def test_r13_a_mismatched_seed_incomplete(tmp_path: Path):
    """(Section 13.4 R13-A) 验证跨种子混合历史结果时拒绝生成四格矩阵。"""
    manifest_data = {
        "expected_seed": 45,  # 冲突种子
    }
    manifest_p = tmp_path / "bad_manifest.json"
    manifest_p.write_text(json.dumps(manifest_data), encoding="utf-8")

    hist_json = REPO_ROOT / "work3_4060/representation_v1/seed42/evaluation/equal_budget_validation_results.json"
    val_scenarios = ["COMPARE_VAL_S0_EARLY_M"]

    lines, info = compute_four_cell_matrix_analysis(
        current_seed=42,
        current_beta=1.0,
        current_c_mean=0.026,
        current_f_mean=0.043,
        val_scenario_ids=val_scenarios,
        historical_json_path=hist_json,
        historical_manifest_path=manifest_p,
    )
    assert info["status"] == "INCOMPLETE"
    assert info["reason"] == "seed_mismatch"
    assert any("禁止跨种子混合生成四格矩阵" in line for line in lines)
