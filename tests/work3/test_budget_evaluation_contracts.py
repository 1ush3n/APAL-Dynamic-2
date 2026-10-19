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
