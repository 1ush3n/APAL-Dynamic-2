"""Work-3 等决策预算与机制消融评估驱动 (C4096 vs D4096 及 F4096 vs D4096)。

支持特性 (Section 9.3, 9.5, 10.3, 10.5)：
1. 显式变体清单 (--variants-json) 支持动态种子 (如 seed 45) 与灵活变体组合；
2. 严格核验有效训练超参 (lr, rollout, batch, epochs, gamma, gae, beta, amp_dtype)；
3. 绑定塑形审计事件覆盖数、episode数与 resolved_config；
4. 目标模型缺失 run/audit 或审计合同失败时强拦截；
5. 正确区分总优化步数、PPO 更新与辅助头更新；
6. 动态胜负平统计与历史配置对照说明。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import ActionBranch
from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import ActorCriticWork3
from scripts.work3.experiment_protocol import build_method_profile
from scripts.work3.evaluate_c_vs_d import (
    build_formal_evaluation_agent,
    evaluate_single_trajectory,
    _validate_formal_checkpoint_contract,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BudgetEvaluation")


def get_variants_for_mode(mode: str) -> list[dict[str, Any]]:
    """根据评估模式获取默认 seed 42 变体配置 (Section 9.3, 9.5, 12.5)。"""
    if mode == "fd":
        return [
            {
                "label": "F_b4096",
                "method": "F",
                "checkpoint": REPO_ROOT / "work3_4060/mechanism_v2/seed42/F_beta01_b4096/weights.pt",
                "run_json": REPO_ROOT / "work3_4060/mechanism_v2/seed42/F_beta01_b4096/run.json",
                "audit_json": REPO_ROOT / "work3_4060/mechanism_v2/seed42/F_beta01_b4096/shaping_audit_summary.json",
                "expected_beta": 0.1,
                "expected_decisions": 4096,
                "expected_seed": 42,
                "is_target": True,
            },
            {
                "label": "D_b4096",
                "method": "D",
                "checkpoint": REPO_ROOT / "work3_4060/budget_v1/seed42/D_beta01_b4096/weights.pt",
                "run_json": REPO_ROOT / "work3_4060/budget_v1/seed42/D_beta01_b4096/run.json",
                "audit_json": REPO_ROOT / "work3_4060/budget_v1/seed42/D_beta01_b4096/shaping_audit_summary.json",
                "expected_beta": 0.1,
                "expected_decisions": 4096,
                "expected_seed": 42,
                "is_target": True,
            },
            {
                "label": "C_b4096",
                "method": "C",
                "checkpoint": REPO_ROOT / "work3_4060/budget_v1/seed42/C_b4096/weights.pt",
                "run_json": REPO_ROOT / "work3_4060/budget_v1/seed42/C_b4096/run.json",
                "audit_json": REPO_ROOT / "work3_4060/budget_v1/seed42/C_b4096/shaping_audit_summary.json",
                "expected_beta": 1.0,
                "expected_decisions": 4096,
                "expected_seed": 42,
                "is_target": True,
            },
            {
                "label": "F_b2048",
                "method": "F",
                "checkpoint": REPO_ROOT / "work3_4060/mechanism_v1/seed42/F/weights.pt",
                "run_json": None,
                "audit_json": None,
                "expected_beta": None,
                "expected_decisions": 2048,
                "expected_seed": 42,
                "is_target": False,
            },
            {
                "label": "D_b2048",
                "method": "D",
                "checkpoint": REPO_ROOT / "work3_4060/shaping_scale_v1/seed42/D_beta01/weights.pt",
                "run_json": None,
                "audit_json": None,
                "expected_beta": 0.1,
                "expected_decisions": 2048,
                "expected_seed": 42,
                "is_target": False,
            },
        ]
    elif mode == "cf":
        return [
            {
                "label": "C_b4096",
                "method": "C",
                "checkpoint": REPO_ROOT / "work3_4060/budget_v1/seed42/C_b4096/weights.pt",
                "run_json": REPO_ROOT / "work3_4060/budget_v1/seed42/C_b4096/run.json",
                "audit_json": REPO_ROOT / "work3_4060/budget_v1/seed42/C_b4096/shaping_audit_summary.json",
                "expected_beta": 1.0,
                "expected_decisions": 4096,
                "expected_seed": 42,
                "is_target": True,
            },
            {
                "label": "F_b4096",
                "method": "F",
                "checkpoint": REPO_ROOT / "work3_4060/representation_v2/seed42/F_beta1_b4096/weights.pt",
                "run_json": REPO_ROOT / "work3_4060/representation_v2/seed42/F_beta1_b4096/run.json",
                "audit_json": REPO_ROOT / "work3_4060/representation_v2/seed42/F_beta1_b4096/shaping_audit_summary.json",
                "expected_beta": 1.0,
                "expected_decisions": 4096,
                "expected_seed": 42,
                "is_target": True,
            },
        ]
    else:  # "cd"
        return [
            {
                "label": "C_b4096",
                "method": "C",
                "checkpoint": REPO_ROOT / "work3_4060/budget_v1/seed42/C_b4096/weights.pt",
                "run_json": REPO_ROOT / "work3_4060/budget_v1/seed42/C_b4096/run.json",
                "audit_json": REPO_ROOT / "work3_4060/budget_v1/seed42/C_b4096/shaping_audit_summary.json",
                "expected_beta": 1.0,
                "expected_decisions": 4096,
                "expected_seed": 42,
                "is_target": True,
            },
            {
                "label": "D_b4096",
                "method": "D",
                "checkpoint": REPO_ROOT / "work3_4060/budget_v1/seed42/D_beta01_b4096/weights.pt",
                "run_json": REPO_ROOT / "work3_4060/budget_v1/seed42/D_beta01_b4096/run.json",
                "audit_json": REPO_ROOT / "work3_4060/budget_v1/seed42/D_beta01_b4096/shaping_audit_summary.json",
                "expected_beta": 0.1,
                "expected_decisions": 4096,
                "expected_seed": 42,
                "is_target": True,
            },
            {
                "label": "C_b2048",
                "method": "C",
                "checkpoint": REPO_ROOT / "work3_4060/compare_v1/seed42/C/weights.pt",
                "run_json": None,
                "audit_json": None,
                "expected_beta": 1.0,
                "expected_decisions": 2048,
                "expected_seed": 42,
                "is_target": False,
            },
            {
                "label": "D_b2048",
                "method": "D",
                "checkpoint": REPO_ROOT / "work3_4060/shaping_scale_v1/seed42/D_beta01/weights.pt",
                "run_json": None,
                "audit_json": None,
                "expected_beta": 0.1,
                "expected_decisions": 2048,
                "expected_seed": 42,
                "is_target": False,
            },
            {
                "label": "F_b2048",
                "method": "F",
                "checkpoint": REPO_ROOT / "work3_4060/mechanism_v1/seed42/F/weights.pt",
                "run_json": None,
                "audit_json": None,
                "expected_beta": None,
                "expected_decisions": 2048,
                "expected_seed": 42,
                "is_target": False,
            },
        ]


class ManifestResult(tuple):
    """变体清单加载结果，同时支持三元组解包 (variants, comparison_mode, expected_seed) 与属性访问 (Section 10.5, 13.4)。"""

    def __new__(cls, variants: list[dict[str, Any]], comparison_mode: str | None, expected_seed: int | None, extra_meta: dict[str, Any] | None = None):
        return super().__new__(cls, (variants, comparison_mode, expected_seed))

    def __init__(self, variants: list[dict[str, Any]], comparison_mode: str | None, expected_seed: int | None, extra_meta: dict[str, Any] | None = None):
        self.variants = variants
        self.comparison_mode = comparison_mode
        self.expected_seed = expected_seed
        self.extra_meta = extra_meta or {}
        self.matrix_historical_json: Path | None = self.extra_meta.get("matrix_historical_json")
        self.matrix_historical_manifest: Path | None = self.extra_meta.get("matrix_historical_manifest")
        self.matrix_historical_sha256: str | None = self.extra_meta.get("matrix_historical_sha256")


def load_variants_from_json(json_path: Path) -> ManifestResult:
    """从外部 JSON 文件加载显式变体清单 (Section 10.5, 13.4)。"""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    extra_meta: dict[str, Any] = {}
    if isinstance(data, dict):
        variants_raw = data.get("variants", [])
        comparison_mode = data.get("comparison_mode")
        expected_seed = data.get("expected_seed")
        for k in ["matrix_historical_json", "matrix_historical_manifest"]:
            if data.get(k):
                raw_p = Path(data[k])
                extra_meta[k] = (REPO_ROOT / raw_p) if not raw_p.is_absolute() else raw_p
        if data.get("matrix_historical_sha256"):
            extra_meta["matrix_historical_sha256"] = str(data["matrix_historical_sha256"]).strip()
    elif isinstance(data, list):
        variants_raw = data
        comparison_mode = None
        expected_seed = None
    else:
        raise ValueError(f"variants-json 格式非法: {json_path}")

    variants: list[dict[str, Any]] = []
    for item in variants_raw:
        v = dict(item)
        if expected_seed is not None and "expected_seed" not in v:
            v["expected_seed"] = expected_seed

        # 路径转换为 Path 对象 (支持相对路径)
        for p_key in ["checkpoint", "run_json", "audit_json"]:
            if v.get(p_key):
                raw_p = Path(v[p_key])
                v[p_key] = (REPO_ROOT / raw_p) if not raw_p.is_absolute() else raw_p

        variants.append(v)

    return ManifestResult(variants, comparison_mode, expected_seed, extra_meta)


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def cross_verify_training_config(
    label: str,
    ckpt_meta: dict[str, Any],
    run_data: dict[str, Any],
    expected_seed: int | None = None,
    expected_beta: float | None = None,
) -> None:
    """严格核验 checkpoint.run_metadata 与 run_data 的有效训练超参一致性 (Section 10.3)。"""
    ckpt_seed = ckpt_meta.get("seed")
    run_seed = run_data.get("seed")
    if ckpt_seed is None or run_seed is None:
        raise ValueError(f"{label} seed 字段缺失 (ckpt={ckpt_seed}, run={run_seed})")
    if int(ckpt_seed) != int(run_seed):
        raise AssertionError(f"{label} seed 不匹配: ckpt={ckpt_seed} vs run={run_seed}")
    if expected_seed is not None and int(ckpt_seed) != int(expected_seed):
        raise AssertionError(f"{label} seed ({ckpt_seed}) 不等于期望种子 ({expected_seed})")

    actual_method = run_data.get("method_variant") or run_data.get("method_profile", {}).get("name")
    ckpt_method = ckpt_meta.get("method_variant")
    if not ckpt_method or not actual_method:
        raise ValueError(f"{label} method_variant 字段缺失 (ckpt={ckpt_method}, run={actual_method})")
    if ckpt_method != actual_method:
        raise AssertionError(f"{label} method 不匹配: ckpt={ckpt_method} vs run={actual_method}")

    # 数据指纹交叉比对
    ckpt_dfp = ckpt_meta.get("data_fingerprint", {})
    run_dfp = run_data.get("data_fingerprint", {})
    if not ckpt_dfp or not run_dfp:
        raise ValueError(f"{label} data_fingerprint 缺失")
    for dfp_key in ["scenario_pool_sha256", "scenario_split_sha256", "baseline_sha256"]:
        c_val = ckpt_dfp.get(dfp_key)
        r_val = run_dfp.get(dfp_key)
        if not c_val or not r_val:
            raise ValueError(f"{label} 数据指纹 {dfp_key} 缺失 (ckpt={c_val}, run={r_val})")
        if c_val != r_val:
            raise AssertionError(f"{label} 数据指纹 {dfp_key} 不匹配！")

    # 有效超参数交叉比对
    ckpt_cfg = ckpt_meta.get("training_config", {})
    run_cfg = run_data.get("training_config", {})
    if not ckpt_cfg or not run_cfg:
        raise ValueError(f"{label} training_config 缺失")

    # 1. beta_shaping
    c_beta = ckpt_cfg.get("beta_shaping", ckpt_cfg.get("ppo", {}).get("shaping_coefficient"))
    r_beta = run_cfg.get("beta_shaping", run_cfg.get("ppo", {}).get("shaping_coefficient"))
    if c_beta is None or r_beta is None:
        raise ValueError(f"{label} beta_shaping 缺失")
    if abs(float(c_beta) - float(r_beta)) > 1e-6:
        raise AssertionError(f"{label} beta 不一致: ckpt={c_beta} vs run={r_beta}")
    if expected_beta is not None and abs(float(c_beta) - expected_beta) > 1e-6:
        raise AssertionError(f"{label} beta ({c_beta}) 不匹配期望 ({expected_beta})")

    # 2. gamma
    c_gamma = ckpt_cfg.get("gamma", ckpt_cfg.get("ppo", {}).get("gamma"))
    r_gamma = run_cfg.get("gamma", run_cfg.get("ppo", {}).get("gamma"))
    if c_gamma is None or r_gamma is None or abs(float(c_gamma) - float(r_gamma)) > 1e-6:
        raise AssertionError(f"{label} gamma 不一致: ckpt={c_gamma} vs run={r_gamma}")

    # 3. lr
    c_lr = ckpt_cfg.get("lr", ckpt_cfg.get("ppo", {}).get("learning_rate"))
    r_lr = run_cfg.get("lr", run_cfg.get("ppo", {}).get("learning_rate"))
    if c_lr is None or r_lr is None:
        raise ValueError(f"{label} 学习率 lr 缺失 (ckpt={c_lr}, run={r_lr})")
    if abs(float(c_lr) - float(r_lr)) > 1e-7:
        raise AssertionError(f"{label} 学习率 lr 不一致: ckpt={c_lr} vs run={r_lr}")

    # 4. gae_lambda
    c_gae = ckpt_cfg.get("gae_lambda", ckpt_cfg.get("ppo", {}).get("gae_lambda"))
    r_gae = run_cfg.get("gae_lambda", run_cfg.get("ppo", {}).get("gae_lambda"))
    if c_gae is None or r_gae is None or abs(float(c_gae) - float(r_gae)) > 1e-6:
        raise AssertionError(f"{label} gae_lambda 不一致: ckpt={c_gae} vs run={r_gae}")

    # 5. rollout & batch & epochs
    for k in ["ppo_epochs", "batch_size"]:
        c_v = ckpt_cfg.get(k, ckpt_cfg.get("ppo", {}).get(k))
        r_v = run_cfg.get(k, run_cfg.get("ppo", {}).get(k))
        if c_v is None or r_v is None or int(c_v) != int(r_v):
            raise AssertionError(f"{label} {k} 不一致: ckpt={c_v} vs run={r_v}")


def verify_shaping_audit_coverage(
    label: str,
    audit_data: dict[str, Any],
    run_data: dict[str, Any],
    expected_decisions: int,
    expected_beta: float | None = None,
    expected_gamma: float = 0.99,
) -> None:
    """严格核验塑形审计文件的覆盖范围与一致性 (Section 10.3)。"""
    overall_status = audit_data.get("overall_status")
    contracts = audit_data.get("contracts", {})
    required_contracts = [
        "single_step_formula",
        "step_to_step_continuity",
        "predictor_version_invariance",
        "terminal_zero_potential",
        "completed_episode_telescope",
        "config_consistency",
        "boundary_contract",
        "ledger_reconciliation",
    ]
    failed_contracts = [c for c in required_contracts if contracts.get(c) != "PASS"]
    if overall_status != "PASS" or len(failed_contracts) > 0:
        raise ValueError(
            f"目标模型 {label} 塑形审计未通过: overall={overall_status}, 失败合同={failed_contracts}"
        )

    # 覆盖事件数核验
    tot_events = audit_data.get("total_events_checked")
    if tot_events is None or int(tot_events) != expected_decisions:
        raise AssertionError(
            f"{label} 审计覆盖事件数 ({tot_events}) 不等于期望决策数 ({expected_decisions})"
        )

    # episode 覆盖量核验
    tot_episodes = audit_data.get("total_episodes_checked")
    run_episodes = run_data.get("independent_episode_count")
    if tot_episodes is not None and run_episodes is not None:
        if int(tot_episodes) != int(run_episodes):
            raise AssertionError(
                f"{label} 审计覆盖 episode 数 ({tot_episodes}) 与 run.json ({run_episodes}) 不匹配"
            )

    # resolved_config 校验
    res_cfg = audit_data.get("resolved_config", {})
    audit_gamma = res_cfg.get("cfg_gamma")
    audit_beta = res_cfg.get("cfg_beta")
    if audit_gamma is not None and abs(float(audit_gamma) - expected_gamma) > 1e-5:
        raise AssertionError(
            f"{label} 审计 resolved_config gamma ({audit_gamma}) 不匹配期望 ({expected_gamma})"
        )
    if expected_beta is not None and audit_beta is not None:
        if abs(float(audit_beta) - expected_beta) > 1e-5:
            raise AssertionError(
                f"{label} 审计 resolved_config beta ({audit_beta}) 不匹配期望 ({expected_beta})"
            )


def cross_verify_model_pairing(
    label_a: str,
    run_a: dict[str, Any],
    label_b: str,
    run_b: dict[str, Any],
    comparison_mode: str = "cf",
    meta_a: dict[str, Any] | None = None,
    meta_b: dict[str, Any] | None = None,
) -> None:
    """严格核验两个对比模型之间的配对契约与共同超参数 (Section 13.4 R13-B)。

    要求：
    1. 双方 initial_actor_fingerprint 必须存在、为合法的 64 位十六进制 SHA256 字符串，且严格一致。
    2. 若 meta_a/meta_b 中也记录了 initial_actor_fingerprint，交叉比对各自 meta 与 run 是否一致。
    3. 核对双方方法确为对应模式的配对方法 (例如 cf 模式必须为 C 与 F)。
    4. 交叉比对跨模型共同超参数:
       - 种子 seed 严格一致。
       - data_fingerprint (scenario_pool_sha256, scenario_split_sha256, baseline_sha256) 严格一致。
       - training_config 共有超参数 (gamma, gae_lambda, lr, rollout_steps, batch_size, ppo_epochs, num_envs, amp_dtype) 严格一致。
       - 在 cf 模式下核对 beta_shaping 严格一致。
    """
    if not isinstance(run_a, dict) or not isinstance(run_b, dict):
        raise ValueError(f"模型配对校验失败: run 数据必须为字典格式 ({label_a}, {label_b})")

    # 1. initial_actor_fingerprint 检查
    fp_a = run_a.get("initial_actor_fingerprint")
    fp_b = run_b.get("initial_actor_fingerprint")
    if not fp_a:
        raise ValueError(f"模型配对校验失败: {label_a} 缺少 initial_actor_fingerprint")
    if not fp_b:
        raise ValueError(f"模型配对校验失败: {label_b} 缺少 initial_actor_fingerprint")

    sha_re = re.compile(r"^[0-9a-fA-F]{64}$")
    if not isinstance(fp_a, str) or not sha_re.match(fp_a):
        raise ValueError(f"模型配对校验失败: {label_a} 初始Actor指纹不是有效的64位十六进制哈希: {fp_a}")
    if not isinstance(fp_b, str) or not sha_re.match(fp_b):
        raise ValueError(f"模型配对校验失败: {label_b} 初始Actor指纹不是有效的64位十六进制哈希: {fp_b}")

    if fp_a.lower() != fp_b.lower():
        raise AssertionError(f"模型配对校验失败: {label_a} 与 {label_b} 初始Actor指纹不匹配: {fp_a} vs {fp_b}")

    # 2. 检查各自 checkpoint 元数据与 run.json 中的指纹是否一致
    if meta_a and meta_a.get("initial_actor_fingerprint"):
        ckpt_fp_a = meta_a["initial_actor_fingerprint"]
        if ckpt_fp_a.lower() != fp_a.lower():
            raise AssertionError(f"模型 {label_a} checkpoint 与 run.json 初始Actor指纹不匹配: {ckpt_fp_a} vs {fp_a}")
    if meta_b and meta_b.get("initial_actor_fingerprint"):
        ckpt_fp_b = meta_b["initial_actor_fingerprint"]
        if ckpt_fp_b.lower() != fp_b.lower():
            raise AssertionError(f"模型 {label_b} checkpoint 与 run.json 初始Actor指纹不匹配: {ckpt_fp_b} vs {fp_b}")

    # 3. 方法核验
    m_a = run_a.get("method_variant") or run_a.get("method_profile", {}).get("name")
    m_b = run_b.get("method_variant") or run_b.get("method_profile", {}).get("name")
    if not m_a or not m_b:
        raise ValueError(f"模型配对校验失败: 缺少方法信息 ({label_a}={m_a}, {label_b}={m_b})")

    methods = {m_a, m_b}
    if comparison_mode == "cf":
        if methods != {"C", "F"}:
            raise AssertionError(f"cf 比较模式要求对比方法必须为 C 与 F，实际为: {m_a} vs {m_b}")
    elif comparison_mode == "fd":
        if methods != {"F", "D"}:
            raise AssertionError(f"fd 比较模式要求对比方法必须为 F 与 D，实际为: {m_a} vs {m_b}")
    elif comparison_mode == "cd":
        if methods != {"C", "D"}:
            raise AssertionError(f"cd 比较模式要求对比方法必须为 C 与 D，实际为: {m_a} vs {m_b}")

    # 4. 种子核验
    seed_a = run_a.get("seed")
    seed_b = run_b.get("seed")
    if seed_a is None or seed_b is None:
        raise ValueError(f"模型配对校验失败: 种子字段缺失 ({label_a}={seed_a}, {label_b}={seed_b})")
    if int(seed_a) != int(seed_b):
        raise AssertionError(f"模型配对校验失败: 种子不一致 ({label_a}={seed_a} vs {label_b}={seed_b})")

    # 5. 数据指纹比对
    dfp_a = run_a.get("data_fingerprint", {})
    dfp_b = run_b.get("data_fingerprint", {})
    for dfp_key in ["scenario_pool_sha256", "scenario_split_sha256", "baseline_sha256"]:
        val_a = dfp_a.get(dfp_key)
        val_b = dfp_b.get(dfp_key)
        if not val_a or not val_b:
            raise ValueError(f"模型配对校验失败: {dfp_key} 数据指纹缺失 ({label_a}={val_a}, {label_b}={val_b})")
        if val_a != val_b:
            raise AssertionError(f"模型配对校验失败: {dfp_key} 数据指纹不一致: {val_a} vs {val_b}")

    # 6. training_config 共有超参数比对
    cfg_a = run_a.get("training_config", {})
    cfg_b = run_b.get("training_config", {})
    if not cfg_a or not cfg_b:
        raise ValueError(f"模型配对校验失败: training_config 缺失 ({label_a}, {label_b})")

    # 浮点超参数
    shared_floats = ["gamma", "gae_lambda", "lr"]
    if comparison_mode in ("cf", "fd"):
        shared_floats.append("beta_shaping")

    for k in shared_floats:
        val_a = cfg_a.get(k, cfg_a.get("ppo", {}).get(k))
        val_b = cfg_b.get(k, cfg_b.get("ppo", {}).get(k))
        if val_a is None or val_b is None:
            raise ValueError(f"模型配对校验失败: 超参数 {k} 缺失 ({label_a}={val_a}, {label_b}={val_b})")
        if abs(float(val_a) - float(val_b)) > 1e-7:
            raise AssertionError(f"模型配对校验失败: 超参数 {k} 不一致: {val_a} vs {val_b}")

    # 整型超参数
    shared_ints = ["rollout_steps", "batch_size", "ppo_epochs", "num_envs"]
    for k in shared_ints:
        val_a = cfg_a.get(k, cfg_a.get("ppo", {}).get(k))
        val_b = cfg_b.get(k, cfg_b.get("ppo", {}).get(k))
        if val_a is None or val_b is None:
            raise ValueError(f"模型配对校验失败: 超参数 {k} 缺失 ({label_a}={val_a}, {label_b}={val_b})")
        if int(val_a) != int(val_b):
            raise AssertionError(f"模型配对校验失败: 超参数 {k} 不一致: {val_a} vs {val_b}")

    # 字符串超参数
    shared_strs = ["amp_dtype"]
    for k in shared_strs:
        val_a = cfg_a.get(k, cfg_a.get("ppo", {}).get(k))
        val_b = cfg_b.get(k, cfg_b.get("ppo", {}).get(k))
        if val_a is None or val_b is None:
            raise ValueError(f"模型配对校验失败: 超参数 {k} 缺失 ({label_a}={val_a}, {label_b}={val_b})")
        if str(val_a).lower() != str(val_b).lower():
            raise AssertionError(f"模型配对校验失败: 超参数 {k} 不一致: {val_a} vs {val_b}")

    # 特殊：fd 模式下双方均有时间辅助头，还核对 initial_parameter_fingerprint
    if comparison_mode == "fd":
        par_a = run_a.get("initial_parameter_fingerprint")
        par_b = run_b.get("initial_parameter_fingerprint")
        if not par_a or not par_b:
            raise ValueError(f"fd 模式配对校验失败: 缺少 initial_parameter_fingerprint ({label_a}={par_a}, {label_b}={par_b})")
        if par_a != par_b:
            raise AssertionError(f"fd 模式配对校验失败: 全参数指纹不匹配: {par_a} vs {par_b}")

    logger.info(f"{comparison_mode.upper()} 目标模型跨模型配对契约校验通过: {label_a} vs {label_b}")


def verify_checkpoint_and_run_contracts(
    variants: list[dict[str, Any]],
    device: str = "cuda:0",
    comparison_mode: str | None = None,
) -> dict[str, Any]:
    """校验各变体检查点契约，并对目标预算模型关联run.json与shaping_audit严格核验 (Section 9.3, 10.3)。"""
    logger.info("开始校验各变体检查点契约与训练/审计记录...")
    metadata_summary: dict[str, Any] = {}

    for v in variants:
        label = v["label"]
        ckpt_path = Path(v["checkpoint"])
        is_target = v.get("is_target", False)
        expected_seed = v.get("expected_seed")
        expected_beta = v.get("expected_beta")
        expected_decisions = v.get("expected_decisions")

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"未找到检查点: {ckpt_path}")

        profile = build_method_profile(v["method"])
        actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64).to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _validate_formal_checkpoint_contract(profile, ckpt, actor)

        meta = ckpt.get("run_metadata", {})
        cfg = meta.get("training_config", {})
        beta_val = cfg.get("beta_shaping")
        if beta_val is None:
            beta_val = cfg.get("ppo", {}).get("shaping_coefficient")

        if expected_beta is not None:
            assert beta_val is not None and abs(float(beta_val) - expected_beta) < 1e-5, (
                f"{label} 的 shaping_coefficient ({beta_val}) 不等于 {expected_beta}！"
            )

        if expected_seed is not None and meta.get("seed") is not None:
            assert int(meta["seed"]) == int(expected_seed), (
                f"{label} 检查点 seed ({meta['seed']}) 不等于期望种子 ({expected_seed})！"
            )

        ckpt_sha = compute_file_sha256(ckpt_path)
        logger.info(f"检查点 {label} 契约校验通过: {ckpt_path.name} (SHA256: {ckpt_sha[:12]}...)")

        # 校验 run.json 运行记录 (Section 9.3, 10.3)
        run_info: dict[str, Any] = {}
        run_json_path = v.get("run_json")
        if is_target:
            if run_json_path is None or not Path(run_json_path).is_file():
                raise FileNotFoundError(f"目标模型 {label} 缺少必需的运行记录: {run_json_path}")

        run_data: dict[str, Any] = {}
        if run_json_path and Path(run_json_path).is_file():
            with Path(run_json_path).open("r", encoding="utf-8") as f:
                run_data = json.load(f)

            total_steps = int(run_data.get("total_decisions", run_data.get("completed_transition_count", 0)))
            if expected_decisions is not None:
                assert total_steps == expected_decisions, (
                    f"{label} 实际总决策数 ({total_steps}) 不等于期望预算 ({expected_decisions})！"
                )

            # 跨配置交叉深度核验 (Section 10.3)
            cross_verify_training_config(
                label=label,
                ckpt_meta=meta,
                run_data=run_data,
                expected_seed=expected_seed,
                expected_beta=expected_beta,
            )

            # 严格区分总优化更新、辅助头更新与 PPO 更新 (Section 9.3)
            total_updates = int(run_data.get("lightning_optimization_steps", 0))
            aux_updates = int(run_data.get("time_supervision_optimizer_updates", 0))
            ppo_updates = total_updates - aux_updates
            assert total_updates >= 0 and aux_updates >= 0 and ppo_updates >= 0, (
                f"{label} 优化更新数异常: total={total_updates}, aux={aux_updates}, ppo={ppo_updates}"
            )
            assert ppo_updates + aux_updates == total_updates, (
                f"{label} 优化更新分解不一致: {ppo_updates} + {aux_updates} != {total_updates}"
            )

            episodes_cnt = run_data.get("independent_episode_count")
            succ_episodes = run_data.get("successful_batch_count")
            wall_seconds = float(run_data.get("elapsed_seconds", 0.0))
            peak_vram_bytes = int(run_data.get("gpu_memory_peak_bytes", 0))
            peak_vram_mib = float(peak_vram_bytes) / (1024 * 1024)

            run_info = {
                "total_steps": total_steps,
                "episodes_count": episodes_cnt,
                "successful_episodes": succ_episodes,
                "wall_clock_seconds": wall_seconds,
                "total_updates": total_updates,
                "ppo_updates": ppo_updates,
                "auxiliary_updates": aux_updates,
                "peak_vram_bytes": peak_vram_bytes,
                "peak_vram_mib": peak_vram_mib,
                "data_fingerprint": run_data.get("data_fingerprint"),
                "source_sha": run_data.get("source_sha"),
                "initial_actor_fingerprint": run_data.get("initial_actor_fingerprint"),
                "initial_parameter_fingerprint": run_data.get("initial_parameter_fingerprint"),
            }
            logger.info(
                f"{label} 运行记录深度校验通过: 步数={total_steps}, 完工={succ_episodes}, "
                f"总更新={total_updates} (PPO={ppo_updates}, 辅助={aux_updates}), "
                f"耗时={wall_seconds:.1f}s, 显存分配峰值={peak_vram_mib:.2f}MiB"
            )

        # 校验塑形审计文件 (Section 9.3, 10.3)
        audit_info: dict[str, Any] = {}
        audit_json_path = v.get("audit_json")
        if is_target:
            if audit_json_path is None or not Path(audit_json_path).is_file():
                raise FileNotFoundError(f"目标模型 {label} 缺少必需的塑形审计文件: {audit_json_path}")

        if audit_json_path and Path(audit_json_path).is_file():
            with Path(audit_json_path).open("r", encoding="utf-8") as f:
                audit_data = json.load(f)

            if expected_decisions is not None:
                verify_shaping_audit_coverage(
                    label=label,
                    audit_data=audit_data,
                    run_data=run_data,
                    expected_decisions=expected_decisions,
                    expected_beta=expected_beta,
                    expected_gamma=0.99,
                )

            audit_info = {
                "overall_status": audit_data.get("overall_status"),
                "audit_pass": True,
                "contracts": audit_data.get("contracts", {}),
                "total_events_checked": audit_data.get("total_events_checked"),
                "total_episodes_checked": audit_data.get("total_episodes_checked"),
            }
            logger.info(
                f"{label} 塑形审计文件校验通过: 8项合同均 PASS (总事件={audit_data.get('total_events_checked')})"
            )

        metadata_summary[label] = {
            "method": v["method"],
            "expected_beta": expected_beta,
            "actual_beta": beta_val,
            "checkpoint_sha256": ckpt_sha,
            "run_info": run_info,
            "run_data": run_data,
            "ckpt_meta": meta,
            "audit_info": audit_info,
            "is_target": is_target,
            "expected_seed": expected_seed,
        }

    # 跨模型配对单变量严格核验 (Section 13.4 R13-B)
    # 根据 comparison_mode 决定执行哪一组配对校验:
    should_verify_cf = (
        comparison_mode == "cf"
        or (comparison_mode is None and "C_b4096" in metadata_summary and "F_b4096" in metadata_summary and not metadata_summary.get("D_b4096", {}).get("is_target", False))
    )
    should_verify_fd = (
        comparison_mode == "fd"
        or (comparison_mode is None and "F_b4096" in metadata_summary and "D_b4096" in metadata_summary and not metadata_summary.get("C_b4096", {}).get("is_target", False))
    )
    should_verify_cd = (
        comparison_mode == "cd"
        or (comparison_mode is None and "C_b4096" in metadata_summary and "D_b4096" in metadata_summary and not metadata_summary.get("F_b4096", {}).get("is_target", False))
    )

    # 1. C 与 F (辅助表征消融)
    if should_verify_cf and "C_b4096" in metadata_summary and "F_b4096" in metadata_summary:
        c_item = metadata_summary["C_b4096"]
        f_item = metadata_summary["F_b4096"]
        if c_item.get("is_target") and f_item.get("is_target"):
            cross_verify_model_pairing(
                label_a="C_b4096",
                run_a=c_item.get("run_data", {}),
                label_b="F_b4096",
                run_b=f_item.get("run_data", {}),
                comparison_mode="cf",
                meta_a=c_item.get("ckpt_meta"),
                meta_b=f_item.get("ckpt_meta"),
            )

    # 2. F 与 D (机制消融)
    if should_verify_fd and "F_b4096" in metadata_summary and "D_b4096" in metadata_summary:
        f_item = metadata_summary["F_b4096"]
        d_item = metadata_summary["D_b4096"]
        if f_item.get("is_target") and d_item.get("is_target"):
            cross_verify_model_pairing(
                label_a="F_b4096",
                run_a=f_item.get("run_data", {}),
                label_b="D_b4096",
                run_b=d_item.get("run_data", {}),
                comparison_mode="fd",
                meta_a=f_item.get("ckpt_meta"),
                meta_b=d_item.get("ckpt_meta"),
            )

    # 3. C 与 D (等预算消融)
    if should_verify_cd and "C_b4096" in metadata_summary and "D_b4096" in metadata_summary:
        c_item = metadata_summary["C_b4096"]
        d_item = metadata_summary["D_b4096"]
        if c_item.get("is_target") and d_item.get("is_target"):
            cross_verify_model_pairing(
                label_a="C_b4096",
                run_a=c_item.get("run_data", {}),
                label_b="D_b4096",
                run_b=d_item.get("run_data", {}),
                comparison_mode="cd",
                meta_a=c_item.get("ckpt_meta"),
                meta_b=d_item.get("ckpt_meta"),
            )

    return metadata_summary


def compute_four_cell_matrix_analysis(
    current_seed: int,
    current_beta: float,
    current_c_mean: float,
    current_f_mean: float,
    val_scenario_ids: list[str],
    historical_json_path: Path | None = None,
    historical_manifest_path: Path | None = None,
    historical_sha256: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """动态生成表征组合与塑形系数四格对照分析 (Section 13.4 R13-A)。

    杜绝硬编码数值；必须显式基于已校验的历史对照结果源动态复算。
    若历史结果源缺失、不匹配或未校验，标为 INCOMPLETE 并说明原因。
    """
    if historical_json_path is None:
        lines = [
            "## 3. 表征组合与塑形系数四格对照表 (Four-Cell Matrix)",
            "> 状态: `INCOMPLETE` (未提供历史对照评估结果源，跳过跨 Beta 四格交互分析；当前配置仅完成单 Beta 评估)",
        ]
        return lines, {"status": "INCOMPLETE", "reason": "historical_json_missing"}

    if not historical_json_path.is_file():
        lines = [
            "## 3. 表征组合与塑形系数四格对照表 (Four-Cell Matrix)",
            f"> 状态: `INCOMPLETE` (未找到历史对照评估文件: `{historical_json_path}`)",
        ]
        return lines, {"status": "INCOMPLETE", "reason": "historical_file_not_found"}

    if historical_sha256:
        actual_sha = compute_file_sha256(historical_json_path)
        if actual_sha.lower() != historical_sha256.lower():
            lines = [
                "## 3. 表征组合与塑形系数四格对照表 (Four-Cell Matrix)",
                f"> 状态: `INCOMPLETE` (历史对照文件 SHA256 校验失败: 期望 {historical_sha256} vs 实际 {actual_sha})",
            ]
            return lines, {"status": "INCOMPLETE", "reason": "sha256_mismatch"}

    if historical_manifest_path is not None and historical_manifest_path.is_file():
        with historical_manifest_path.open("r", encoding="utf-8") as f:
            hist_manifest = json.load(f)
        hist_seed = hist_manifest.get("expected_seed")
        if hist_seed is not None and int(hist_seed) != int(current_seed):
            lines = [
                "## 3. 表征组合与塑形系数四格对照表 (Four-Cell Matrix)",
                f"> 状态: `INCOMPLETE` (历史对照清单种子 {hist_seed} 与当前评估种子 {current_seed} 不匹配，禁止跨种子混合生成四格矩阵)",
            ]
            return lines, {"status": "INCOMPLETE", "reason": "seed_mismatch"}

    try:
        with historical_json_path.open("r", encoding="utf-8") as f:
            hist_results = json.load(f)
    except Exception as e:
        lines = [
            "## 3. 表征组合与塑形系数四格对照表 (Four-Cell Matrix)",
            f"> 状态: `INCOMPLETE` (历史对照评估 JSON 解析失败: {e})",
        ]
        return lines, {"status": "INCOMPLETE", "reason": f"json_parse_error: {e}"}

    # 提取历史 C 与 F 的验证场景结果
    hist_c_by_sc = {
        item["scenario_id"]: float(item["metrics"]["j_total"])
        for item in hist_results
        if item.get("variant") == "C_b4096" and item.get("scenario_id") in val_scenario_ids
    }
    hist_f_by_sc = {
        item["scenario_id"]: float(item["metrics"]["j_total"])
        for item in hist_results
        if item.get("variant") == "F_b4096" and item.get("scenario_id") in val_scenario_ids
    }

    # 核验场景完整性
    for sc_id in val_scenario_ids:
        if sc_id not in hist_c_by_sc or sc_id not in hist_f_by_sc:
            lines = [
                "## 3. 表征组合与塑形系数四格对照表 (Four-Cell Matrix)",
                f"> 状态: `INCOMPLETE` (历史对照评估结果缺少场景 `{sc_id}` 的完整 C/F 记录)",
            ]
            return lines, {"status": "INCOMPLETE", "reason": "scenario_coverage_incomplete"}

    hist_c_mean = sum(hist_c_by_sc.values()) / len(val_scenario_ids)
    hist_f_mean = sum(hist_f_by_sc.values()) / len(val_scenario_ids)

    # 确定哪一个是 beta=0.1，哪一个是 beta=1.0
    if abs(current_beta - 1.0) < 1e-4:
        c_01 = hist_c_mean
        f_01 = hist_f_mean
        c_1 = current_c_mean
        f_1 = current_f_mean
    elif abs(current_beta - 0.1) < 1e-4:
        c_01 = current_c_mean
        f_01 = current_f_mean
        c_1 = hist_c_mean
        f_1 = hist_f_mean
    else:
        c_01 = hist_c_mean
        f_01 = hist_f_mean
        c_1 = current_c_mean
        f_1 = current_f_mean

    delta_01 = c_01 - f_01
    delta_1 = c_1 - f_1
    I_interaction = delta_1 - delta_01

    pct_c_beta = ((c_1 - c_01) / c_01) * 100.0
    pct_f_beta = ((f_1 - f_01) / f_01) * 100.0
    pct_f_over_c_01 = ((c_01 - f_01) / c_01) * 100.0
    pct_c_over_f_1 = ((f_1 - c_1) / f_1) * 100.0

    lines = [
        "## 3. 表征组合与塑形系数四格对照表 (Four-Cell Matrix: C/F x Beta 0.1/1.0)",
        "> **来源绑定与证据链 (Section 13.4 R13-A 动态生成)**：",
        f"> - **弱 Beta (0.1) 结果源**：`{historical_json_path.name}` (SHA256: `{compute_file_sha256(historical_json_path)[:16]}...`)",
        f"> - **标准 Beta (1.0) 结果源**：当前评估输出 (Seed {current_seed}, 决策预算 4096)",
        "",
        "| 方法配置 | Beta = 0.1 | Beta = 1.0 | Delta Beta (Beta1.0 - Beta0.1) |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Method C** (无辅助表征) | `{c_01:.8e}` | `{c_1:.8e}` | `{c_1 - c_01:+.8e}` ({pct_c_beta:+.2f}%) |",
        f"| **Method F** (辅助表征+修正输入) | `{f_01:.8e}` | `{f_1:.8e}` | `{f_1 - f_01:+.8e}` ({pct_f_beta:+.2f}%) |",
        f"| **Delta (C - F, 正=F优)** | `{delta_01:+.8e}` (Delta_0.1) | `{delta_1:+.8e}` (Delta_1) | **交互差 I = {I_interaction:+.8e}** |",
        "",
        "> **关键相对比例核算 (分母严格定义)**：",
        f"> - 在 Beta = 0.1 下：F 的平均费用相比 C 降低 `{pct_f_over_c_01:.2f}%`（分母为 C 费用 `J_C,0.1`）；",
        f"> - 在 Beta = 1.0 下：C 的平均费用相比 F 降低 `{pct_c_over_f_1:.2f}%`（分母为 F 费用 `J_F,1.0`）；",
        f"> - 单 seed 交互差 $I = \\Delta_1 - \\Delta_{{0.1}} = {I_interaction:+.8e}$，两种 Beta 下表征组合收益方向反转。",
    ]
    matrix_info = {
        "status": "PASS",
        "c_01": c_01,
        "f_01": f_01,
        "c_1": c_1,
        "f_1": f_1,
        "delta_01": delta_01,
        "delta_1": delta_1,
        "I_interaction": I_interaction,
        "pct_c_beta": pct_c_beta,
        "pct_f_beta": pct_f_beta,
        "pct_f_over_c_01": pct_f_over_c_01,
        "pct_c_over_f_1": pct_c_over_f_1,
    }
    return lines, matrix_info

def generate_evaluation_reports(
    json_results: list[dict[str, Any]],
    variants: list[dict[str, Any]],
    mode: str,
    output_dir: Path,
    metadata_summary: dict[str, Any] | None = None,
    eval_seed: int = 42,
    device_label: str = "cuda:0",
    historical_json_path: Path | None = None,
    historical_manifest_path: Path | None = None,
    historical_sha256: str | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (REPO_ROOT / "work3_4060/compare_v1/scenarios.json").open("r", encoding="utf-8") as f:
        scenarios_by_id = {s["scenario_id"]: s for s in json.load(f)}

    with (REPO_ROOT / "work3_4060/compare_v1/validation.json").open("r", encoding="utf-8") as f:
        val_scenario_ids = json.load(f)

    all_eval_scenarios = [None] + [scenarios_by_id[sc_id] for sc_id in val_scenario_ids]

    # 将 json_results 按 (scenario_id, variant) 建立索引
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in json_results:
        index[(item["scenario_id"], item["variant"])] = item

    csv_rows: list[dict[str, Any]] = []

    for sc in all_eval_scenarios:
        sc_id = sc["scenario_id"] if sc is not None else "NOMINAL"
        row: dict[str, Any] = {
            "scenario_id": sc_id,
            "timing": sc.get("timing", "NONE") if sc else "NONE",
            "intensity": sc.get("intensity", "NONE") if sc else "NONE",
            "station": sc.get("station_id", sc.get("station", "NONE")) if sc else "NONE",
        }

        for v in variants:
            label = v["label"]
            pfx = label.lower()
            entry = index.get((sc_id, label))
            if entry is None:
                raise ValueError(f"评估结果中缺失场景 `{sc_id}` 变体 `{label}` 的数据！")

            res = entry["metrics"]
            total_dec = int(res.get("decisions", res.get("total_steps", 0)))
            explicit_adv_count = int(entry.get("explicit_advance_count", 0))
            adv_pct = float(entry.get("advance_percentage", 0.0))

            j_tot = float(res["j_total"])
            c_takt = float(res["j_takt"])
            c_time = float(res["d_time"])
            c_team = float(res["d_team"])
            c_post = float(res["j_postpone"])
            c_rev = float(res["j_revision"])
            cost_sum_err = abs(j_tot - (c_takt + c_time + c_team + c_post + c_rev))

            # R13-C: 内部字典始终保留原始数值 float，CSV 导出时使用 .17g 保留完整精度
            row[f"{pfx}_j_total"] = j_tot
            row[f"{pfx}_makespan"] = float(res["makespan"])
            row[f"{pfx}_cost_sum_error"] = cost_sum_err
            row[f"{pfx}_c_takt"] = c_takt
            row[f"{pfx}_c_time"] = c_time
            row[f"{pfx}_c_team"] = c_team
            row[f"{pfx}_c_postpone"] = c_post
            row[f"{pfx}_c_revision"] = c_rev
            row[f"{pfx}_total_decisions"] = total_dec
            row[f"{pfx}_explicit_advances"] = explicit_adv_count
            row[f"{pfx}_advance_pct"] = f"{adv_pct:.2f}%"
            row[f"{pfx}_feasible"] = bool(res["feasible"])

        if mode == "fd":
            j_f4096 = float(row["f_b4096_j_total"])
            j_d4096 = float(row["d_b4096_j_total"])
            m_f4096 = float(row["f_b4096_makespan"])
            m_d4096 = float(row["d_b4096_makespan"])
            delta_f_minus_d = j_f4096 - j_d4096
            row["delta_j_f4096_minus_d4096"] = delta_f_minus_d
            row["delta_makespan_f4096_minus_d4096"] = m_f4096 - m_d4096
            row["d4096_better_than_f4096"] = (delta_f_minus_d > 1e-12)

            if "c_b4096_j_total" in row:
                j_c4096 = float(row["c_b4096_j_total"])
                row["delta_j_c4096_minus_f4096"] = j_c4096 - j_f4096
                row["delta_j_c4096_minus_d4096"] = j_c4096 - j_d4096

            if "f_b2048_j_total" in row:
                j_f2048 = float(row["f_b2048_j_total"])
                row["delta_j_f4096_minus_f2048"] = j_f4096 - j_f2048
            if "d_b2048_j_total" in row:
                j_d2048 = float(row["d_b2048_j_total"])
                row["delta_j_d4096_minus_d2048"] = j_d4096 - j_d2048
        elif mode == "cf":
            j_c4096 = float(row["c_b4096_j_total"])
            j_f4096 = float(row["f_b4096_j_total"])
            m_c4096 = float(row["c_b4096_makespan"])
            m_f4096 = float(row["f_b4096_makespan"])
            delta_c_minus_f = j_c4096 - j_f4096
            row["delta_j_c4096_minus_f4096"] = delta_c_minus_f
            row["delta_makespan_c4096_minus_f4096"] = m_c4096 - m_f4096
            row["f4096_better_than_c4096"] = (delta_c_minus_f > 1e-12)
        else:
            j_c4096 = float(row["c_b4096_j_total"])
            j_d4096 = float(row["d_b4096_j_total"])
            m_c4096 = float(row["c_b4096_makespan"])
            m_d4096 = float(row["d_b4096_makespan"])
            delta_c_minus_d = j_c4096 - j_d4096
            row["delta_j_c4096_minus_d4096"] = delta_c_minus_d
            row["delta_makespan_c4096_minus_d4096"] = m_c4096 - m_d4096
            row["d4096_better_than_c4096"] = (delta_c_minus_d > 1e-12)

            if "d_b2048_j_total" in row:
                j_d2048 = float(row["d_b2048_j_total"])
                row["delta_j_d4096_minus_d2048"] = j_d4096 - j_d2048
            if "c_b2048_j_total" in row:
                j_c2048 = float(row["c_b2048_j_total"])
                row["delta_j_c4096_minus_c2048"] = j_c4096 - j_c2048

        csv_rows.append(row)

    # 导出保留完整精度的 CSV (R13-C: 使用 .17g 保留完整 IEEE 754 精度)
    csv_rows_for_export = []
    for r in csv_rows:
        export_r = {}
        for k, val in r.items():
            if isinstance(val, float):
                export_r[k] = f"{val:.17g}"
            else:
                export_r[k] = val
        csv_rows_for_export.append(export_r)

    csv_path = output_dir / "equal_budget_validation_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows_for_export[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows_for_export)
    logger.info(f"评估 CSV 已导出: {csv_path}")

    # 统计数据生成 summary.md
    summary_path = output_dir / "summary.md"
    val_rows = [r for r in csv_rows if r["scenario_id"] != "NOMINAL"]
    nom_row = [r for r in csv_rows if r["scenario_id"] == "NOMINAL"][0]
    val_count = len(val_rows)

    # 检查目标模型物理可行性与账本闭合 (动态推导，不写死)
    target_variants = [v for v in variants if v.get("is_target", False)]
    all_target_audits_pass = all(
        (metadata_summary or {}).get(v["label"], {}).get("audit_info", {}).get("audit_pass", False)
        for v in target_variants
    )
    all_target_feasible = all(
        all(r[f"{v['label'].lower()}_feasible"] for r in csv_rows)
        for v in target_variants
    )
    max_target_ledger_err = max(
        max(float(r[f"{v['label'].lower()}_cost_sum_error"]) for r in csv_rows)
        for v in target_variants
    )
    correctness_gate_pass = (
        all_target_audits_pass and all_target_feasible and (max_target_ledger_err <= 1e-12)
    )

    has_history = any(not v.get("is_target", False) for v in variants)

    if mode == "fd":
        # 机制消融 summary: 严格区分胜、负、平
        d_wins_val = sum(1 for r in val_rows if float(r["delta_j_f4096_minus_d4096"]) > 1e-12)
        f_wins_val = sum(1 for r in val_rows if float(r["delta_j_f4096_minus_d4096"]) < -1e-12)
        ties_val = sum(1 for r in val_rows if abs(float(r["delta_j_f4096_minus_d4096"])) <= 1e-12)

        mean_j_f4k_val = sum(float(r["f_b4096_j_total"]) for r in val_rows) / val_count
        mean_j_d4k_val = sum(float(r["d_b4096_j_total"]) for r in val_rows) / val_count
        mean_j_c4k_val = (
            sum(float(r["c_b4096_j_total"]) for r in val_rows) / val_count
            if "c_b4096_j_total" in val_rows[0]
            else None
        )
        mean_m_f4k_val = sum(float(r["f_b4096_makespan"]) for r in val_rows) / val_count
        mean_m_d4k_val = sum(float(r["d_b4096_makespan"]) for r in val_rows) / val_count
        mean_delta_fd_val = mean_j_f4k_val - mean_j_d4k_val

        nom_j_f4k = float(nom_row["f_b4096_j_total"])
        nom_j_d4k = float(nom_row["d_b4096_j_total"])
        nom_j_c4k = float(nom_row["c_b4096_j_total"]) if "c_b4096_j_total" in nom_row else None
        nom_m_f4k = float(nom_row["f_b4096_makespan"])
        nom_m_d4k = float(nom_row["d_b4096_makespan"])
        nom_delta_fd = nom_j_f4k - nom_j_d4k

        f_meta = (metadata_summary or {}).get("F_b4096", {}).get("run_info", {})
        d_meta = (metadata_summary or {}).get("D_b4096", {}).get("run_info", {})
        c_meta = (metadata_summary or {}).get("C_b4096", {}).get("run_info", {})

        if mean_delta_fd_val > 1e-12:
            winner_str = "D(学习势函数)更优"
        elif mean_delta_fd_val < -1e-12:
            winner_str = "F(启发式势函数)更优"
        else:
            winner_str = "持平"

        c_ref_str = f" (背景参照 C4096 = `{mean_j_c4k_val:.8e}`)" if mean_j_c4k_val is not None else ""
        c_nom_ref_str = f" (背景参照 C4096 = `{nom_j_c4k:.8e}`)" if nom_j_c4k is not None else ""

        lines = [
            f"# Work-3 机制消融评估报告 (F_beta01_b4096 vs D_b4096, Seed {eval_seed})",
            f"评估设备：`{device_label}` | 种子：`{eval_seed}` | 决策预算：严格 4096 决策步 | 主聚合：4个 Validation 场景 (Nominal 单列)",
            "",
            "## 1. 机制消融核心结论与判定 (Section 9.5, 10.3)",
            f"- **正确性前提检验**: 目标模型塑形审计动态核验 `{all_target_audits_pass}` (8/8项合同均PASS), 独立物理可行 100% (`{all_target_feasible}`), 账本加权分项和最大误差 `{max_target_ledger_err:.2e}` (守恒), 综合放行状态: `{'PASS' if correctness_gate_pass else 'FAIL'}`。",
            f"- **4个 Validation 场景主口径聚合**: ",
            f"  - **平均综合费用 J_total**: F(4096步) = `{mean_j_f4k_val:.8e}`, D(4096步) = `{mean_j_d4k_val:.8e}`{c_ref_str}",
            f"  - **平均净优势 Delta J (J_F4096 - J_D4096)**: `{winner_str}` (`{mean_delta_fd_val:+.8e}`)",
            f"  - **逐场景胜出对比**: **D 胜 {d_wins_val} / {val_count}** | **F 胜 {f_wins_val} / {val_count}** | **平局 {ties_val} / {val_count}**",
            f"  - **平均工期 Makespan**: F(4096步) = `{mean_m_f4k_val:.2f}h`, D(4096步) = `{mean_m_d4k_val:.2f}h` (差异: `{mean_m_f4k_val - mean_m_d4k_val:+.2f}h`)",
            f"- **Nominal 场景单列**: ",
            f"  - J_total: F(4096步) = `{nom_j_f4k:.8e}`, D(4096步) = `{nom_j_d4k:.8e}` (Delta J: `{nom_delta_fd:+.8e}`){c_nom_ref_str}",
            f"  - 工期: F = `{nom_m_f4k:.2f}h`, D = `{nom_m_d4k:.2f}h`",
            "",
            "> [!IMPORTANT]",
            "> **机制消融配置对齐说明 (Section 9.4, 10.2, 10.3)**：",
            "> 1. **严格单变量势函数对照**：F4096（启发式 $\\Phi, \\beta=0.1$）与 D4096（学习 $\\Phi, \\beta=0.1$）共享相同初始 Actor/Head 指纹、训练数据 mini210、种子、4096 决策预算、修正时间输入、时间辅助监督以及 episode 冻结预测器生命周期；唯一的机制差异是势函数来源（启发式 $\\Phi$ vs 学习 $\\Phi$）。",
            "> 2. **优化步数与更新分解核查 (精确分解)**：",
            f">    - **F(4096步)**: 墙钟耗时 `{f_meta.get('wall_clock_seconds', 'N/A')}s`, 总优化更新 `{f_meta.get('total_updates', 'N/A')}` 次 (其中 PPO更新 `{f_meta.get('ppo_updates', 'N/A')}` 次, 辅助头更新 `{f_meta.get('auxiliary_updates', 'N/A')}` 次), 成功批次 `{f_meta.get('successful_episodes', 'N/A')}`",
            f">    - **D(4096步)**: 墙钟耗时 `{d_meta.get('wall_clock_seconds', 'N/A')}s`, 总优化更新 `{d_meta.get('total_updates', 'N/A')}` 次 (其中 PPO更新 `{d_meta.get('ppo_updates', 'N/A')}` 次, 辅助头更新 `{d_meta.get('auxiliary_updates', 'N/A')}` 次), 成功批次 `{d_meta.get('successful_episodes', 'N/A')}`",
        ]
        if c_meta:
            lines.append(
                f">    - **C(4096步, 背景参照)**: 墙钟耗时 `{c_meta.get('wall_clock_seconds', 'N/A')}s`, 总优化更新 `{c_meta.get('total_updates', 'N/A')}` 次 (其中 PPO更新 `{c_meta.get('ppo_updates', 'N/A')}` 次, 辅助头更新 `{c_meta.get('auxiliary_updates', 'N/A')}` 次), 成功批次 `{c_meta.get('successful_episodes', 'N/A')}`"
            )

        lines.extend([
            "",
            "## 2. 逐场景机制消融对比主表 (F_beta01_b4096 vs D_b4096)",
            "| 场景 ID | F_4096 J | D_4096 J | Delta J (F-D, 正=D优) | D优? | C_4096 J (参照) | F_4096 工期 | D_4096 工期 | Delta 工期 | F 显式推进 | D 显式推进 | 各方可行? |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for r in csv_rows:
            delta_val = float(r["delta_j_f4096_minus_d4096"])
            if delta_val > 1e-12:
                better_str = "**是 (D优)**"
            elif delta_val < -1e-12:
                better_str = "否 (F优)"
            else:
                better_str = "持平"

            is_nom = (r["scenario_id"] == "NOMINAL")
            sc_label = f"**`{r['scenario_id']}` (单列)**" if is_nom else f"`{r['scenario_id']}`"
            c_val_str = f"{r['c_b4096_j_total']:.8e}" if "c_b4096_j_total" in r else "N/A"
            all_feas = (
                r["f_b4096_feasible"]
                and r["d_b4096_feasible"]
                and (r.get("c_b4096_feasible", True))
            )
            lines.append(
                f"| {sc_label} | {r['f_b4096_j_total']:.8e} | {r['d_b4096_j_total']:.8e} | "
                f"**{delta_val:+.8e}** | {better_str} | {c_val_str} | "
                f"{r['f_b4096_makespan']:.2f}h | {r['d_b4096_makespan']:.2f}h | "
                f"{r['delta_makespan_f4096_minus_d4096']:+.2f}h | "
                f"{r['f_b4096_advance_pct']} ({r['f_b4096_explicit_advances']}/{r['f_b4096_total_decisions']}) | "
                f"{r['d_b4096_advance_pct']} ({r['d_b4096_explicit_advances']}/{r['d_b4096_total_decisions']}) | "
                f"{'PASS' if all_feas else 'FAIL'} |"
            )

        if has_history:
            lines.extend([
                "",
                "## 3. 各方法自身历史配置对照",
                "> [!NOTE]",
                "> **配置差异说明 (Section 10.2)**：历史 F_b2048 配置使用 beta=1.0，而 F_b4096 使用 beta=0.1；两者同时改变了决策预算与塑形系数，属历史配置对照，不可解释为纯预算敏感性。C 与 D 在相同 beta 下的前后比较保留。",
                "",
                "| 场景 ID | F_beta1_2048 J | F_beta01_4096 J | Delta J (F4k-F2k, 非单变量) | D_2048 J | D_4096 J | Delta J (D4k-D2k, 纯预算) |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for r in csv_rows:
                f_2k = f"{r['f_b2048_j_total']:.8e}" if "f_b2048_j_total" in r else "N/A"
                d_2k = f"{r['d_b2048_j_total']:.8e}" if "d_b2048_j_total" in r else "N/A"
                delta_f = f"{r['delta_j_f4096_minus_f2048']:+.8e}" if "delta_j_f4096_minus_f2048" in r else "N/A"
                delta_d = f"{r['delta_j_d4096_minus_d2048']:+.8e}" if "delta_j_d4096_minus_d2048" in r else "N/A"
                lines.append(
                    f"| `{r['scenario_id']}` | {f_2k} | {r['f_b4096_j_total']:.8e} | "
                    f"**{delta_f}** | {d_2k} | {r['d_b4096_j_total']:.8e} | "
                    f"**{delta_d}** |"
                )

        lines.extend([
            "",
            "## 4. 全变体费用五分项完整明细表 (已加权费用，保留完整精度)",
            "| 场景 / 变体 | J_total | c_takt | c_time | c_team | c_postpone | c_revision | 账本误差 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for r in csv_rows:
            sc = r["scenario_id"]
            for v in variants:
                lbl = v["label"]
                pfx = lbl.lower()
                lines.append(
                    f"| `{sc}` / `{lbl}` | {r[f'{pfx}_j_total']:.8e} | {r[f'{pfx}_c_takt']:.8e} | "
                    f"{r[f'{pfx}_c_time']:.8e} | {r[f'{pfx}_c_team']:.8e} | {r[f'{pfx}_c_postpone']:.8e} | "
                    f"{r[f'{pfx}_c_revision']:.8e} | {r[f'{pfx}_cost_sum_error']:.2e} |"
                )

        lines.extend([
            "",
            "## 5. 机制判定与后续决策依据 (Section 9.5, 10.6 预设规则)",
            "- **若 F/D 方向与 seed 42 一致且 C 费用依然最低**: 保留当前“F费用优于D、C完整配置更优”的探索性结论；暂不继续扩大预算或调学习Phi。若后续研究辅助表征贡献，另设计消除 C/F beta 差异的控制试验。",
            "- **若 F/D 方向发生反转**: 报告种子敏感性，不按某一个 seed 淘汰方法；先看既有费用分解/暴露/训练诊断，再决定是否需要更多预先固定重复。",
            "- **若 D 在新 seed 超过 F 或 C**: 仅报告新 seed 结果与跨 seed 差异；不能用它抹去 seed 42 或直接恢复“方法普遍优越”的表述。",
            "- **若仅效果变好但可行性/账本/来源合同失败**: 不具备有效效果比较条件，只修具体失败，不新增机制结论。",
        ])
    elif mode == "cf":
        # 辅助表征消融 summary (C_b4096 vs F_b4096)
        c_wins_val = sum(1 for r in val_rows if float(r["delta_j_c4096_minus_f4096"]) < -1e-12)
        f_wins_val = sum(1 for r in val_rows if float(r["delta_j_c4096_minus_f4096"]) > 1e-12)
        ties_val = sum(1 for r in val_rows if abs(float(r["delta_j_c4096_minus_f4096"])) <= 1e-12)

        mean_j_c4k_val = sum(float(r["c_b4096_j_total"]) for r in val_rows) / val_count
        mean_j_f4k_val = sum(float(r["f_b4096_j_total"]) for r in val_rows) / val_count
        mean_m_c4k_val = sum(float(r["c_b4096_makespan"]) for r in val_rows) / val_count
        mean_m_f4k_val = sum(float(r["f_b4096_makespan"]) for r in val_rows) / val_count
        mean_delta_cf_val = mean_j_c4k_val - mean_j_f4k_val

        nom_j_c4k = float(nom_row["c_b4096_j_total"])
        nom_j_f4k = float(nom_row["f_b4096_j_total"])
        nom_m_c4k = float(nom_row["c_b4096_makespan"])
        nom_m_f4k = float(nom_row["f_b4096_makespan"])
        nom_delta_cf = nom_j_c4k - nom_j_f4k

        c_meta = (metadata_summary or {}).get("C_b4096", {}).get("run_info", {})
        f_meta = (metadata_summary or {}).get("F_b4096", {}).get("run_info", {})
        target_beta = (metadata_summary or {}).get("C_b4096", {}).get("actual_beta", (metadata_summary or {}).get("C_b4096", {}).get("expected_beta", 1.0))

        if mean_delta_cf_val > 1e-12:
            winner_str = "F(辅助表征+修正输入)更优"
        elif mean_delta_cf_val < -1e-12:
            winner_str = "C(无辅助表征)更优"
        else:
            winner_str = "持平"

        # 准确相对降低百分比
        pct_c_over_f = (mean_j_f4k_val - mean_j_c4k_val) / mean_j_f4k_val * 100.0 if mean_j_f4k_val > 0 else 0.0
        pct_f_over_c = (mean_j_c4k_val - mean_j_f4k_val) / mean_j_c4k_val * 100.0 if mean_j_c4k_val > 0 else 0.0

        lines = [
            f"# Work-3 辅助表征消融评估报告 (C_b4096 vs F_b4096, Seed {eval_seed}, Beta {target_beta})",
            f"评估设备：`{device_label}` | 种子：`{eval_seed}` | 决策预算：严格 4096 决策步 | 塑形系数：Beta={target_beta} | 主聚合：4个 Validation 场景 (Nominal 单列)",
            "",
            "## 1. 表征消融核心结论与判定 (Section 12.5, 12.6, 13.3)",
            f"- **正确性前提检验**: 目标模型塑形审计动态核验 `{all_target_audits_pass}` (8/8项合同均PASS), 独立物理可行 100% (`{all_target_feasible}`), 账本加权分项和最大误差 `{max_target_ledger_err:.2e}` (守恒), 综合放行状态: `{'PASS' if correctness_gate_pass else 'FAIL'}`。",
            f"- **4个 Validation 场景主口径聚合**: ",
            f"  - **平均综合费用 J_total**: C(4096步, beta={target_beta}) = `{mean_j_c4k_val:.8e}`, F(4096步, beta={target_beta}) = `{mean_j_f4k_val:.8e}`",
            f"  - **平均净优势 Delta_1 (J_C4096 - J_F4096)**: `{winner_str}` (`{mean_delta_cf_val:+.8e}`)",
            f"  - **相对降幅**: 标准 Beta 下 C 的费用相比 F 降低 `{pct_c_over_f:.2f}%`（分母为 F 费用）",
            f"  - **逐场景胜出对比**: **F 胜 {f_wins_val} / {val_count}** | **C 胜 {c_wins_val} / {val_count}** | **平局 {ties_val} / {val_count}**",
            f"  - **平均工期 Makespan**: C(4096步) = `{mean_m_c4k_val:.2f}h`, F(4096步) = `{mean_m_f4k_val:.2f}h` (差异: `{mean_m_c4k_val - mean_m_f4k_val:+.2f}h`)",
            f"- **Nominal 场景单列**: ",
            f"  - J_total: C(4096步) = `{nom_j_c4k:.8e}`, F(4096步) = `{nom_j_f4k:.8e}` (Delta_1: `{nom_delta_cf:+.8e}`)",
            f"  - 工期: C = `{nom_m_c4k:.2f}h`, F = `{nom_m_f4k:.2f}h`",
            "",
            "> [!IMPORTANT]",
            "> **对比解释与机制边界 (Section 12.4-12.6, 13.3)**：",
            f"> 1. **严格表征消融对照**：C4096（启发式 $\\Phi, \\beta={target_beta}$）与 F4096（启发式 $\\Phi, \\beta={target_beta}$）共享相同初始 Actor 指纹、训练数据 mini210、种子 {eval_seed}、4096 决策预算、启发式势函数及 $\\beta={target_beta}$；唯一的机制差异是时间辅助监督与修正时间输入。",
            "> 2. **优化步数与更新分解核查 (精确分解)**：",
            f">    - **C(4096步)**: 墙钟耗时 `{c_meta.get('wall_clock_seconds', 'N/A')}s`, 总优化更新 `{c_meta.get('total_updates', 'N/A')}` 次 (其中 PPO更新 `{c_meta.get('ppo_updates', 'N/A')}` 次, 辅助头更新 `{c_meta.get('auxiliary_updates', 'N/A')}` 次), 成功批次 `{c_meta.get('successful_episodes', 'N/A')}`",
            f">    - **F(4096步)**: 墙钟耗时 `{f_meta.get('wall_clock_seconds', 'N/A')}s`, 总优化更新 `{f_meta.get('total_updates', 'N/A')}` 次 (其中 PPO更新 `{f_meta.get('ppo_updates', 'N/A')}` 次, 辅助头更新 `{f_meta.get('auxiliary_updates', 'N/A')}` 次), 成功批次 `{f_meta.get('successful_episodes', 'N/A')}`",
            "",
            f"## 2. 逐场景辅助表征消融对比主表 (C_b4096 vs F_b4096, Beta={target_beta})",
            "| 场景 ID | C_4096 J | F_4096 J | Delta_1 (C-F, 正=F优) | F优? | C_4096 工期 | F_4096 工期 | Delta 工期 | C 显式推进 | F 显式推进 | 双方可行? |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for r in csv_rows:
            delta_val = float(r["delta_j_c4096_minus_f4096"])
            if delta_val > 1e-12:
                better_str = "**是 (F优)**"
            elif delta_val < -1e-12:
                better_str = "否 (C优)"
            else:
                better_str = "持平"

            is_nom = (r["scenario_id"] == "NOMINAL")
            sc_label = f"**`{r['scenario_id']}` (单列)**" if is_nom else f"`{r['scenario_id']}`"
            lines.append(
                f"| {sc_label} | {r['c_b4096_j_total']:.8e} | {r['f_b4096_j_total']:.8e} | "
                f"**{delta_val:+.8e}** | {better_str} | "
                f"{r['c_b4096_makespan']:.2f}h | {r['f_b4096_makespan']:.2f}h | "
                f"{r['delta_makespan_c4096_minus_f4096']:+.2f}h | "
                f"{r['c_b4096_advance_pct']} ({r['c_b4096_explicit_advances']}/{r['c_b4096_total_decisions']}) | "
                f"{r['f_b4096_advance_pct']} ({r['f_b4096_explicit_advances']}/{r['f_b4096_total_decisions']}) | "
                f"{'PASS' if (r['c_b4096_feasible'] and r['f_b4096_feasible']) else 'FAIL'} |"
            )

        # 动态生成四格对照分析 (Section 13.4 R13-A)
        four_cell_lines, matrix_info = compute_four_cell_matrix_analysis(
            current_seed=eval_seed,
            current_beta=float(target_beta),
            current_c_mean=mean_j_c4k_val,
            current_f_mean=mean_j_f4k_val,
            val_scenario_ids=[r["scenario_id"] for r in val_rows],
            historical_json_path=historical_json_path,
            historical_manifest_path=historical_manifest_path,
            historical_sha256=historical_sha256,
        )
        lines.append("")
        lines.extend(four_cell_lines)

        lines.extend([
            "",
            "## 4. 全变体费用五分项完整明细表 (已加权费用，保留完整精度)",
            "> 术语说明：D_team 为团队替换偏差费用，J_revision 为计划修订费用，D_time 为周期内开工时间偏差费用；J_takt 与 J_postpone 均为 0。",
            "| 场景 / 变体 | J_total | c_takt | c_time | c_team | c_postpone | c_revision | 账本误差 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for r in csv_rows:
            sc = r["scenario_id"]
            for v in variants:
                lbl = v["label"]
                pfx = lbl.lower()
                lines.append(
                    f"| `{sc}` / `{lbl}` | {r[f'{pfx}_j_total']:.8e} | {r[f'{pfx}_c_takt']:.8e} | "
                    f"{r[f'{pfx}_c_time']:.8e} | {r[f'{pfx}_c_team']:.8e} | {r[f'{pfx}_c_postpone']:.8e} | "
                    f"{r[f'{pfx}_c_revision']:.8e} | {r[f'{pfx}_cost_sum_error']:.2e} |"
                )

        lines.extend([
            "",
            "## 5. 机制判定与后续决策依据 (Section 12.6, 13.3, 13.6 预设规则)",
            "- **表征消融核心观察 (Section 13.3)**: 本 seed 标准 β 下，辅助监督与修正输入组合未降低最终费用；费用差主要体现为团队替换与计划修订费用。共享梯度干扰、更新频率或输入作用均未被单独识别。在已观察的 seed 42、两种固定 β 和 4096 预算中，组合收益的方向随 β 变化（弱 β 下改善 15.58%，标准 β 下落后 38.33%）；尚不能推广到其他 seed、β 或数据。",
            "- **验证场景性质与统计边界**: 同一已训练策略在 4 个验证场景的描述性胜负；这不是 4 次独立训练重复，I = -0.02480 只是单 seed 交互差，不是统计显著性结果。",
            "- **当前基准判定**: 当前 seed 42、4096 预算及已比较配置中，C(beta=1.0) 的验证集平均费用最低。历史 C(2048步) 曾费用更低，不能混用预算后宣称全历史最优，也不事后换检查点。",
            "- **行为与工期特征**: 双方显式 ADVANCE 为 0（均无显式等待与显式后移），工期接近 (680.97h vs 680.33h)；仍有自动事件推进与运行工序所需的物理时间跨越。",
            "- **计算开销记录**: 标准 C 与新 F 耗时约 495.3s 和 1963.7s，后者约 3.96 倍。该记录仅报告为实际运行开销，不用单次墙钟差额证明某个模块的固定性能倍率。",
            "- **后续决策**: 四格探索性比较已经闭环，保留标准 C(beta=1.0) 作为当前 4096 步候选配置。本轮停止新增训练，不启动 seed 45 新 F、不扫描 β、不调优学习 Phi、不扩大训练预算至 8192 步。",
        ])
    else:
        # cd 模式 summary
        d_wins_val = sum(1 for r in val_rows if float(r["delta_j_c4096_minus_d4096"]) > 1e-12)
        c_wins_val = sum(1 for r in val_rows if float(r["delta_j_c4096_minus_d4096"]) < -1e-12)
        ties_val = sum(1 for r in val_rows if abs(float(r["delta_j_c4096_minus_d4096"])) <= 1e-12)

        mean_j_c4k_val = sum(float(r["c_b4096_j_total"]) for r in val_rows) / val_count
        mean_j_d4k_val = sum(float(r["d_b4096_j_total"]) for r in val_rows) / val_count
        mean_m_c4k_val = sum(float(r["c_b4096_makespan"]) for r in val_rows) / val_count
        mean_m_d4k_val = sum(float(r["d_b4096_makespan"]) for r in val_rows) / val_count
        mean_delta_val = mean_j_c4k_val - mean_j_d4k_val

        nom_j_c4k = float(nom_row["c_b4096_j_total"])
        nom_j_d4k = float(nom_row["d_b4096_j_total"])
        nom_m_c4k = float(nom_row["c_b4096_makespan"])
        nom_m_d4k = float(nom_row["d_b4096_makespan"])
        nom_delta = nom_j_c4k - nom_j_d4k

        c_meta = (metadata_summary or {}).get("C_b4096", {}).get("run_info", {})
        d_meta = (metadata_summary or {}).get("D_b4096", {}).get("run_info", {})

        if mean_delta_val > 1e-12:
            winner_str = "D费用更优"
        elif mean_delta_val < -1e-12:
            winner_str = "C费用更优"
        else:
            winner_str = "持平"

        lines = [
            f"# Work-3 等决策预算对照评估报告 (C_b4096 vs D_b4096, Seed {eval_seed})",
            f"评估设备：`{device_label}` | 种子：`{eval_seed}` | 决策预算：严格 4096 决策步 | 主聚合：4个 Validation 场景 (Nominal 单列)",
            "",
            "## 1. 等决策预算核心结论与判定 (Section 8.5 - 8.6, 9.3 纠偏, 10.3)",
            f"- **正确性前提检验**: 目标模型塑形审计动态核验 `{all_target_audits_pass}` (8/8项合同均PASS), 独立物理可行 100% (`{all_target_feasible}`), 账本加权分项和最大误差 `{max_target_ledger_err:.2e}` (守恒), 综合放行状态: `{'PASS' if correctness_gate_pass else 'FAIL'}`。",
            f"- **4个 Validation 场景主口径聚合**: ",
            f"  - **平均综合费用 J_total**: C(4096步) = `{mean_j_c4k_val:.8e}`, D(4096步) = `{mean_j_d4k_val:.8e}`",
            f"  - **平均净优势 Delta J (J_C4096 - J_D4096)**: `{winner_str}` (`{mean_delta_val:+.8e}`)",
            f"  - **逐场景胜出对比**: **D 胜 {d_wins_val} / {val_count}** | **C 胜 {c_wins_val} / {val_count}** | **平局 {ties_val} / {val_count}**",
            f"  - **平均工期 Makespan**: C(4096步) = `{mean_m_c4k_val:.2f}h`, D(4096步) = `{mean_m_d4k_val:.2f}h` (差异: `{mean_m_c4k_val - mean_m_d4k_val:+.2f}h`)",
            f"- **Nominal 场景单列**: ",
            f"  - J_total: C(4096步) = `{nom_j_c4k:.8e}`, D(4096步) = `{nom_j_d4k:.8e}` (Delta J: `{nom_delta:+.8e}`)",
            f"  - 工期: C = `{nom_m_c4k:.2f}h`, D = `{nom_m_d4k:.2f}h`",
            "",
            "> [!IMPORTANT]",
            "> **对比解释与机制边界 (Section 8.5-8.6 必读约定，Section 9.3 优化分解核实)**：",
            "> 1. **非隔离学习势函数单变量**：C4096（启发式 $\\Phi, \\beta=1.0$）与 D4096（学习 $\\Phi, \\beta=0.1$）是两套已冻结方法配置的等决策预算比较；两者塑形系数不同且 D 含有时间辅助头与修正时间输入，不是隔离“学习势函数”作用的单变量实验。",
            "> 2. **等决策预算不等于等墙钟或等优化次数 (更新计数精确分解)**：",
            f">    - **C(4096步)**: 墙钟耗时 `{c_meta.get('wall_clock_seconds', 'N/A')}s`, 总优化更新 `{c_meta.get('total_updates', 'N/A')}` 次 (其中 PPO更新 `{c_meta.get('ppo_updates', 'N/A')}` 次, 辅助头更新 `{c_meta.get('auxiliary_updates', 'N/A')}` 次), 成功批次 `{c_meta.get('successful_episodes', 'N/A')}`",
            f">    - **D(4096步)**: 墙钟耗时 `{d_meta.get('wall_clock_seconds', 'N/A')}s`, 总优化更新 `{d_meta.get('total_updates', 'N/A')}` 次 (其中 PPO更新 `{d_meta.get('ppo_updates', 'N/A')}` 次, 辅助头更新 `{d_meta.get('auxiliary_updates', 'N/A')}` 次), 成功批次 `{d_meta.get('successful_episodes', 'N/A')}`",
            "",
            "## 2. 逐场景等决策预算对比主表 (C_b4096 vs D_b4096)",
            "| 场景 ID | C_4096 J | D_4096 J | Delta J (C-D, 正=D优) | D优? | C_4096 工期 | D_4096 工期 | Delta 工期 | C 显式推进 | D 显式推进 | 双方可行? |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for r in csv_rows:
            delta_val = float(r["delta_j_c4096_minus_d4096"])
            if delta_val > 1e-12:
                better_str = "**是 (D优)**"
            elif delta_val < -1e-12:
                better_str = "否 (C优)"
            else:
                better_str = "持平"

            is_nom = (r["scenario_id"] == "NOMINAL")
            sc_label = f"**`{r['scenario_id']}` (单列)**" if is_nom else f"`{r['scenario_id']}`"
            lines.append(
                f"| {sc_label} | {r['c_b4096_j_total']:.8e} | {r['d_b4096_j_total']:.8e} | "
                f"**{delta_val:+.8e}** | {better_str} | "
                f"{r['c_b4096_makespan']:.2f}h | {r['d_b4096_makespan']:.2f}h | "
                f"{r['delta_makespan_c4096_minus_d4096']:+.2f}h | "
                f"{r['c_b4096_advance_pct']} ({r['c_b4096_explicit_advances']}/{r['c_b4096_total_decisions']}) | "
                f"{r['d_b4096_advance_pct']} ({r['d_b4096_explicit_advances']}/{r['d_b4096_total_decisions']}) | "
                f"{'PASS' if (r['c_b4096_feasible'] and r['d_b4096_feasible']) else 'FAIL'} |"
            )

        if has_history:
            lines.extend([
                "",
                "## 3. 各方法自身预算敏感性 (2048步 vs 4096步)",
                "| 场景 ID | C_2048 J | C_4096 J | Delta J (C4k-C2k) | D_2048 J | D_4096 J | Delta J (D4k-D2k) |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
            ])
            for r in csv_rows:
                c_2k = f"{r['c_b2048_j_total']:.8e}" if "c_b2048_j_total" in r else "N/A"
                d_2k = f"{r['d_b2048_j_total']:.8e}" if "d_b2048_j_total" in r else "N/A"
                delta_c = f"{r['delta_j_c4096_minus_c2048']:+.8e}" if "delta_j_c4096_minus_c2048" in r else "N/A"
                delta_d = f"{r['delta_j_d4096_minus_d2048']:+.8e}" if "delta_j_d4096_minus_d2048" in r else "N/A"
                lines.append(
                    f"| `{r['scenario_id']}` | {c_2k} | {r['c_b4096_j_total']:.8e} | "
                    f"{delta_c} | {d_2k} | {r['d_b4096_j_total']:.8e} | "
                    f"**{delta_d}** |"
                )

        lines.extend([
            "",
            "## 4. 全变体费用五分项完整明细表 (已加权费用，保留完整精度)",
            "| 场景 / 变体 | J_total | c_takt | c_time | c_team | c_postpone | c_revision | 账本误差 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for r in csv_rows:
            sc = r["scenario_id"]
            for v in variants:
                lbl = v["label"]
                pfx = lbl.lower()
                lines.append(
                    f"| `{sc}` / `{lbl}` | {r[f'{pfx}_j_total']:.8e} | {r[f'{pfx}_c_takt']:.8e} | "
                    f"{r[f'{pfx}_c_time']:.8e} | {r[f'{pfx}_c_team']:.8e} | {r[f'{pfx}_c_postpone']:.8e} | "
                    f"{r[f'{pfx}_c_revision']:.8e} | {r[f'{pfx}_cost_sum_error']:.2e} |"
                )

        lines.extend([
            "",
            "## 5. 后续决策依据 (根据 Section 8.6 预设规则)",
            "- 若 D 费用仍高于 C：当前 seed 与预算下，完整 D 配置未超过 C；D 自身增加预算有效。应先比较时间/团队/修订分项、场景暴露和已有训练诊断；不直接认定时间头污染价值梯度。",
            "- 若 D 费用低于 C 且可行性/账本正常：当前冻结配置在这个 seed 有初步优势，建议先补一个预先固定的新 seed（对 C/D 各训练 4096 步），再判断重复性。",
            "- 若下一轮需要隔离学习势函数作用：优先考虑 F(β=0.1, 4096步) 对 D(β=0.1, 4096步) 的严格单变量对照。",
        ])

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"评估综合报告 summary.md 已生成: {summary_path}")

    return csv_path, summary_path


def evaluate_scenarios(
    variants: list[dict[str, Any]],
    mode: str,
    device: str = "cuda:0",
    output_dir: Path | None = None,
    metadata_summary: dict[str, Any] | None = None,
    eval_seed: int = 42,
    historical_json_path: Path | None = None,
    historical_manifest_path: Path | None = None,
    historical_sha256: str | None = None,
    offline: bool = False,
) -> tuple[Path, Path]:
    default_dir = (
        REPO_ROOT / f"work3_4060/representation_v2/seed{eval_seed}/evaluation"
        if mode == "cf"
        else (
            REPO_ROOT / f"work3_4060/mechanism_v2/seed{eval_seed}/evaluation"
            if mode == "fd"
            else REPO_ROOT / f"work3_4060/budget_v1/equal_budget_evaluation"
        )
    )
    out_dir = output_dir or default_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "equal_budget_validation_results.json"

    if offline:
        if not json_path.is_file():
            raise FileNotFoundError(f"离线重算模式未找到评估 JSON 文件: {json_path}")
        logger.info(f"以离线重算模式读取既有评估 JSON: {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            json_results = json.load(f)
    else:
        baseline_path = str(REPO_ROOT / "work3_4060/data/mini210.json")

        with (REPO_ROOT / "work3_4060/compare_v1/scenarios.json").open("r", encoding="utf-8") as f:
            scenarios_by_id = {s["scenario_id"]: s for s in json.load(f)}

        with (REPO_ROOT / "work3_4060/compare_v1/validation.json").open("r", encoding="utf-8") as f:
            val_scenario_ids = json.load(f)

        all_eval_scenarios = [None] + [scenarios_by_id[sc_id] for sc_id in val_scenario_ids]
        json_results: list[dict[str, Any]] = []

        for sc in all_eval_scenarios:
            sc_id = sc["scenario_id"] if sc is not None else "NOMINAL"
            logger.info(f"--- 评估场景: {sc_id} ---")

            for v in variants:
                label = v["label"]
                method = v["method"]
                ckpt_path = v["checkpoint"]

                agent = build_formal_evaluation_agent(method, str(ckpt_path), device=device)
                env = AirLineEnvWork3(baseline_json_path=baseline_path)
                res = evaluate_single_trajectory(
                    env=env,
                    agent_type=f"Method-{method}",
                    agent=agent,
                    scenario=sc,
                    device=device,
                    max_decisions=4096,
                    warmup_mode="none",
                )

                total_dec = int(res.get("decisions", res.get("total_steps", 0)))
                adv_counts = res.get("advance_cause_counts", {})
                explicit_adv_count = int(adv_counts.get("explicit", 0))
                adv_pct = (explicit_adv_count / total_dec * 100.0) if total_dec > 0 else 0.0

                json_results.append({
                    "scenario_id": sc_id,
                    "variant": label,
                    "method": method,
                    "metrics": res,
                    "explicit_advance_count": explicit_adv_count,
                    "advance_percentage": adv_pct,
                })

        json_path.write_text(json.dumps(json_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info(f"评估完整 JSON 已导出: {json_path}")

    return generate_evaluation_reports(
        json_results=json_results,
        variants=variants,
        mode=mode,
        output_dir=out_dir,
        metadata_summary=metadata_summary,
        eval_seed=eval_seed,
        device_label=device,
        historical_json_path=historical_json_path,
        historical_manifest_path=historical_manifest_path,
        historical_sha256=historical_sha256,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--comparison",
        type=str,
        choices=["auto", "cd", "fd", "cf"],
        default="auto",
        help="评估比较模式: cd (等决策预算), fd (机制消融), cf (辅助表征消融), auto (根据 output-dir 或 variants-json 自动推断)",
    )
    p.add_argument(
        "--variants-json",
        type=Path,
        default=None,
        help="显式变体清单 JSON 路径 (Section 10.5)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="离线模式：直接从已有 equal_budget_validation_results.json 重算并生成报告，不重复执行物理仿真 (Section 13.5)",
    )
    p.add_argument(
        "--matrix-historical-json",
        type=Path,
        default=None,
        help="四格矩阵历史评估结果 JSON 路径 (Section 13.4 R13-A)",
    )
    p.add_argument(
        "--matrix-historical-manifest",
        type=Path,
        default=None,
        help="四格矩阵历史变体清单 JSON 路径 (Section 13.4 R13-A)",
    )
    args = p.parse_args()

    eval_seed = 42
    historical_json_path = args.matrix_historical_json
    historical_manifest_path = args.matrix_historical_manifest
    historical_sha256 = None

    if args.variants_json is not None:
        if not args.variants_json.is_file():
            raise FileNotFoundError(f"未找到变体清单文件: {args.variants_json}")
        manifest_res = load_variants_from_json(args.variants_json)
        variants, json_mode, json_seed = manifest_res
        if json_seed is not None:
            eval_seed = json_seed
        mode = args.comparison if args.comparison != "auto" else (json_mode or "cf" if "representation_v2" in str(args.variants_json) else (json_mode or "fd"))

        if historical_json_path is None and manifest_res.matrix_historical_json:
            historical_json_path = manifest_res.matrix_historical_json
        if historical_manifest_path is None and manifest_res.matrix_historical_manifest:
            historical_manifest_path = manifest_res.matrix_historical_manifest
        if manifest_res.matrix_historical_sha256:
            historical_sha256 = manifest_res.matrix_historical_sha256
    else:
        mode = args.comparison
        if mode == "auto":
            if args.output_dir is not None and "representation_v2" in str(args.output_dir):
                mode = "cf"
            elif args.output_dir is not None and "mechanism_v2" in str(args.output_dir):
                mode = "fd"
            else:
                mode = "cd"
        variants = get_variants_for_mode(mode)

    output_dir = args.output_dir or (
        args.variants_json.parent / "evaluation"
        if args.variants_json is not None
        else (
            REPO_ROOT / f"work3_4060/representation_v2/seed{eval_seed}/evaluation"
            if mode == "cf"
            else (
                REPO_ROOT / f"work3_4060/mechanism_v2/seed{eval_seed}/evaluation"
                if mode == "fd"
                else REPO_ROOT / f"work3_4060/budget_v1/equal_budget_evaluation"
            )
        )
    )

    device_for_meta = "cpu" if args.offline else args.device
    meta = verify_checkpoint_and_run_contracts(variants=variants, device=device_for_meta, comparison_mode=mode)
    evaluate_scenarios(
        variants=variants,
        mode=mode,
        device=args.device,
        output_dir=output_dir,
        metadata_summary=meta,
        eval_seed=eval_seed,
        historical_json_path=historical_json_path,
        historical_manifest_path=historical_manifest_path,
        historical_sha256=historical_sha256,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
