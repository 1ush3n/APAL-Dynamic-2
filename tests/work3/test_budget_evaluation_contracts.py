"""测试 work3_4060/run_budget_evaluation.py 中的契约核验与坏输入拦截能力 (Section 9.3)。

验证点：
1. 现有 C4096 / D4096 契约、运行记录和审计文件校验通过。
2. 目标模型缺失 run.json 时，拦截并报错 (FileNotFoundError)。
3. 检查点与 run.json 元数据冲突 (seed/method/指纹) 时，拦截并报错 (AssertionError)。
4. 目标模型缺失或失败的 shaping_audit 时，拦截并报错 (FileNotFoundError/ValueError)。
5. 优化更新计数严格分解：total_updates = ppo_updates + aux_updates。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import pytest
import torch

from work3_4060.run_budget_evaluation import (
    REPO_ROOT,
    get_variants_for_mode,
    verify_checkpoint_and_run_contracts,
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
    assert len(meta["C_b4096"]["audit_info"]["failed_contracts"]) == 0
    assert len(meta["D_b4096"]["audit_info"]["failed_contracts"]) == 0


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
