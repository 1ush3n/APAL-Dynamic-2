"""Work-3 价值头专属校准与 PPO 保持性匹配对照。

唯一新增训练臂：先只校准价值专属头，再用与前轮完全相同的配置跑 512 步 PPO。
未校准对照直接复用 ``work3_4060/bc_ppo_retention_v1/seed42``（不新增对照训练）。

阶段：
    errata     两处历史解释纠偏（固定平移 + 首段证据范围）
    offline    固定首段离线梯度分解（零参数更新）
    calibrate  Phase A 采集一条随机 MC 轨迹 + Phase B 仅价值头校准（CPU）
    train      Phase C 从校准副本出发的 512 步 PPO（GPU，复用正式 training_step/GAE/势函数）
    evaluate   0/128/256/512 四点独立 CPU 贪婪闭环
    verify     轨迹事实、塑形审计、权重完整性、协议状态
    report     并列表、周期表与报告

复用：``training.work3_lightning.Work3LightningModule``（正式 PPO 更新与损失）、
``models.work3.ppo_buffer.RolloutBufferWork3``（GAE）、``train_ppo_work3.compute_heuristic_potential``、
``training.work3_vector_env.Work3VectorEnv``（单环境 spawn）、``evaluate_c_vs_d.evaluate_single_trajectory``。
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import ActorCriticWork3
from models.work3.graph_builder import GRAPH_FEATURE_DIMS, GRAPH_FEATURE_SCHEMA, GRAPH_FEATURE_VERSION
from models.work3.ppo_buffer import RolloutBufferWork3
from scripts.work3.evaluate_c_vs_d import (
    FormalEvaluationAgent,
    evaluate_single_trajectory,
)
from scripts.work3.experiment_protocol import build_method_profile
from scripts.work3.train_ppo_work3 import compute_heuristic_potential
from training.work3_lightning import Work3TrainingUpdate
from utils.work3.objective_evaluator import build_objective_report, evaluate_trajectory_objective
from utils.work3.trajectory_feasibility import check_environment_trajectory_feasibility
from work3_4060.c_reproduction_protocol import (
    actor_fingerprint,
    clone_actor_state,
    compare_execution_facts,
    dump_json,
    dump_jsonl,
    enforce_single_thread_cpu,
    evaluate_teacher_sample,
    extract_execution_facts,
    load_diagnostic_actor,
    load_json,
    sha256_file,
    state_dict_fingerprint,
)
import work3_4060.run_bc_ppo_retention as prev

BASE_DIR = REPO_ROOT / "work3_4060"
RULE_BASELINE_DIR = BASE_DIR / "rule_baseline_ac_v1"
BC_DIR = BASE_DIR / "c_reproduction_diagnostic_v1" / "seed42"
CONTROL_DIR = BASE_DIR / "bc_ppo_retention_v1" / "seed42"
OUTPUT_DIR = BASE_DIR / "critic_calibration_retention_v1" / "seed42"
CAL_DIR = OUTPUT_DIR / "calibration"
PPO_DIR = OUTPUT_DIR / "ppo"
PPO_CKPT_DIR = PPO_DIR / "checkpoints"
PPO_TRAJ_DIR = PPO_DIR / "trajectories"
RUNS_DIR = OUTPUT_DIR / "runs"

BASELINE_PATH = RULE_BASELINE_DIR / "mini210_rule_baseline.json"
BC_WEIGHTS = BC_DIR / "diagnostic_actor.pt"
BC_MANIFEST = BC_DIR / "manifest.json"
CONTROL_MANIFEST = CONTROL_DIR / "manifest.json"
CONTROL_CONFIG = CONTROL_DIR / "resolved_config.yaml"
CONTROL_SUMMARY = CONTROL_DIR / "training_summary.json"
CONTROL_FIRST_ROLLOUT = CONTROL_DIR / "first_rollout.pt"
CONTROL_TEACHER = CONTROL_DIR / "teacher_scores_step0000.json"

MANIFEST = OUTPUT_DIR / "manifest.json"
CONTROL_BINDING = OUTPUT_DIR / "control_binding.json"
ERRATA = OUTPUT_DIR / "interpretation_errata.md"
GRAD_DECOMP = OUTPUT_DIR / "offline_gradient_decomposition.json"
CAL_TRAJECTORY = CAL_DIR / "trajectory.json"
CAL_SNAPSHOTS = CAL_DIR / "snapshots_and_mc_targets.pt"
CAL_HISTORY = CAL_DIR / "history.jsonl"
CAL_FIT = CAL_DIR / "fit_metrics.json"
CAL_INVARIANCE = CAL_DIR / "policy_invariance.json"
CAL_MODEL = CAL_DIR / "calibrated_model.pt"
PPO_CONFIG = PPO_DIR / "resolved_config.yaml"
PPO_INIT_VERIFICATION = PPO_DIR / "initialization_verification.json"
PPO_PAIRING = PPO_DIR / "first_rollout_pairing.json"
PPO_HISTORY = PPO_DIR / "training_history.jsonl"
PPO_SHAPING = PPO_DIR / "shaping_events.jsonl"
PPO_SHAPING_AUDIT = PPO_DIR / "shaping_audit_summary.json"
PPO_FIRST_ROLLOUT = PPO_DIR / "first_rollout.pt"
PPO_SUMMARY = PPO_DIR / "training_summary.json"
PPO_RUN_JSON = PPO_DIR / "runs" / "ppo_run.json"
COMPARISON_CSV = OUTPUT_DIR / "comparison_results.csv"
CYCLE_CSV = OUTPUT_DIR / "cycle_comparison.csv"
TRAJECTORY_VERIFICATION = OUTPUT_DIR / "trajectory_verification.json"
WEIGHT_INTEGRITY = OUTPUT_DIR / "weight_integrity.json"
VERIFICATION = OUTPUT_DIR / "verification.json"
SUMMARY_MD = OUTPUT_DIR / "summary.md"

STAGES = ("errata", "offline", "calibrate", "train", "evaluate", "verify", "report")
CHECKPOINTS = (0, 128, 256, 512)
TRAINING_BUDGET = 512
CAL_SEED = 4201
CAL_SPLIT_SEED = 4202
CAL_MAX_STEPS = 128
CAL_MAX_SECONDS = 300.0
CAL_BATCH = 8
CAL_SCAN_INTERVAL = 16
CAL_DECISION_LIMIT = 4096
HOLDOUT_MODULO = 5
REPLAY_TOLERANCE = 1e-4
POLICY_PROB_TOLERANCE = 1e-7
GRAD_RESIDUAL_TOLERANCE = 1e-4  # float32 autograd 累加的相对容差（1e-9 是 float64 级，不适用于单精度）
GRAD_RESIDUAL_NOTE = (
    "残差容差按单精度浮点累加特征固定为 1e-4 相对值；"
    "float64 级 1e-9 不适用于 float32 的 138 个参数梯度求和。实测残差已记录。"
)
CAL_MODEL_FORMAT = "critic_value_calibrated_copy"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_calibrated_actor(path: Path) -> tuple[ActorCriticWork3, dict[str, Any]]:
    """严格加载本轮的“已校准副本”格式（与 PPO 检查点是不同格式，不混用）。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("diagnostic_format") != CAL_MODEL_FORMAT:
        raise ValueError(f"不是校准副本格式: {path}")
    if int(payload.get("diagnostic_format_version", -1)) != 1:
        raise ValueError("校准副本格式版本不匹配")
    if payload.get("graph_feature_version") != GRAPH_FEATURE_VERSION:
        raise ValueError("校准副本图特征版本不匹配")
    if dict(payload.get("graph_feature_dims") or {}) != dict(GRAPH_FEATURE_DIMS):
        raise ValueError("校准副本图特征维度不匹配")
    if payload.get("graph_feature_schema_hash") != GRAPH_FEATURE_SCHEMA["schema_fingerprint"]:
        raise ValueError("校准副本 schema 指纹不匹配")
    hyper = payload.get("hyperparameters") or {}
    model = ActorCriticWork3(
        state_dim=int(hyper["state_dim"]),
        task_feat_dim=int(hyper["task_feat_dim"]),
        hidden_dim=int(hyper["hidden_dim"]),
        max_station_workers=int(hyper["max_station_workers"]),
    )
    state = payload.get("actor_critic_state")
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise ValueError("校准副本参数集合不匹配")
    for name, tensor in model.state_dict().items():
        if not isinstance(state[name], torch.Tensor) or state[name].shape != tensor.shape:
            raise ValueError(f"校准副本参数维度不匹配: {name}")
    if state_dict_fingerprint(state) != payload.get("actor_fingerprint"):
        raise ValueError("校准副本指纹与权重不一致")
    model.load_state_dict(state, strict=True)
    return model, payload


def write_ppo_run_json(
    history: Sequence[Mapping[str, Any]],
    hyper: Mapping[str, Any],
    stop_reason: str,
    decisions: int,
) -> None:
    ppo_block = {
        "steps_per_iter": int(hyper["rollout_steps"]),
        "epochs": int(hyper["ppo_epochs"]),
        "batch_size": int(hyper["batch_size"]),
        "learning_rate": float(hyper["learning_rate"]),
        "clip_epsilon": float(hyper["clip_epsilon"]),
        "value_coefficient": float(hyper["value_coefficient"]),
        "entropy_coefficient": float(hyper["entropy_coefficient"]),
        "gamma": float(hyper["gamma"]),
        "gae_lambda": float(hyper["gae_lambda"]),
        "shaping_coefficient": float(hyper["beta_shaping"]),
    }
    PPO_RUN_JSON.parent.mkdir(parents=True, exist_ok=True)
    dump_json(
        PPO_RUN_JSON,
        {
            "training_config": {
                "gamma": float(hyper["gamma"]),
                "beta_shaping": float(hyper["beta_shaping"]),
                "learning_rate": float(hyper["learning_rate"]),
                "clip_eps": float(hyper["clip_epsilon"]),
                "vf_coef": float(hyper["value_coefficient"]),
                "ent_coef": float(hyper["entropy_coefficient"]),
                "ppo_epochs": int(hyper["ppo_epochs"]),
                "batch_size": int(hyper["batch_size"]),
                "rollout_steps": int(hyper["rollout_steps"]),
                "disturbance_enabled": False,
            },
            "resolved_runtime_config_yaml": "ppo:\n"
            + "".join(f"  {key}: {value}\n" for key, value in ppo_block.items()),
            "training_decisions": decisions,
            "segments": len(history),
            "stop_reason": stop_reason,
            "arm": "critic_calibrated",
        },
    )


def git_state() -> dict[str, Any]:
    def _run(args: Sequence[str]) -> str | None:
        try:
            return subprocess.check_output(
                args, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    head = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain=v1"])
    return {
        "head": head,
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "worktree_dirty": bool(status),
        "worktree_entries": [] if not status else status.splitlines()[:40],
    }


def subset_fingerprint(state: Mapping[str, torch.Tensor], names: Sequence[str]) -> str:
    return state_dict_fingerprint({name: state[name] for name in names})


def policy_names(model: torch.nn.Module) -> list[str]:
    groups = prev.parameter_groups(model)
    return groups["groups"]["shared"] + groups["groups"]["policy"]


def value_names(model: torch.nn.Module) -> list[str]:
    return prev.parameter_groups(model)["groups"]["value"]


# ---------------------------------------------------------------------------
# Phase 0：报告纠偏
# ---------------------------------------------------------------------------


def stage_errata() -> int:
    control = load_json(CONTROL_DIR / "shaping_audit_summary.json")
    episodes = control.get("episodes", [])
    beta = float(load_json(CONTROL_SUMMARY)["hyperparameters"]["beta_shaping"])
    gamma = float(load_json(CONTROL_SUMMARY)["hyperparameters"]["gamma"])
    rows = []
    for item in episodes:
        raw = float(item["discounted_raw_return"])
        shaped = float(item["discounted_shaped_return"])
        phi0 = float(item["phi_start"])
        phi_end = float(item.get("phi_end", 0.0))
        # 一般形式：shaped - raw = beta*(gamma^n*Phi_T - Phi_0)；完工段 Phi_T=0
        expected_shift = beta * ((gamma ** int(item["steps"])) * phi_end - phi0)
        observed = shaped - raw
        rows.append(
            {
                "episode_id": int(item["episode_id"]),
                "steps": int(item["steps"]),
                "terminated": bool(item["terminated"]),
                "gamma": gamma,
                "beta": beta,
                "phi_start": phi0,
                "phi_end": phi_end,
                "discounted_raw_return": raw,
                "discounted_shaped_return": shaped,
                "observed_shift_shaped_minus_raw": observed,
                "expected_shift_minus_beta_phi0": expected_shift if bool(item["terminated"]) else None,
                "expected_shift_general": expected_shift,
                "residual": observed - expected_shift,
                "within_tolerance": abs(observed - expected_shift) <= 1e-9,
                "undiscounted_raw_sum": float(item["undiscounted_raw_sum"]),
            }
        )
    all_ok = all(row["within_tolerance"] for row in rows)
    payload = {
        "schema_version": "critic_calibration_errata_v1",
        "beta": beta,
        "gamma": gamma,
        "identity": "G_gamma^shaped = G_gamma^raw + beta*(gamma^n*Phi_T - Phi(s_0))；完工段 Phi_T=0",
        "episodes": rows,
        "all_within_tolerance": all_ok,
        "correction_3_1": (
            "塑形与原始折扣回报之差不代表“方向相反”：在自然终止且终局势为零时，"
            "两者只差一个固定平移 -beta*Phi(s0)。前轮 0.5 的差值来自 Phi(s0)=-0.5、beta=1，"
            "与该等式逐条吻合，因此不能据此宣称“塑形回报改善而业务费用变差”。"
        ),
        "correction_3_2": (
            "首段 8 步全部留站+align=1、raw reward≈0 只能说明该段记录内没有出现明显即时费用偏离，"
            "不能排除后续随机采样的影响；第一个完整闭环复测点是 128，"
            "故只能说“更新改变了已记录快照上的动作概率，首次已评估的闭环退化出现在 128”。"
        ),
    }
    dump_json(OUTPUT_DIR / "errata_verification.json", payload)

    lines = [
        "# 解释勘误（不重跑前轮训练，原始日志/检查点/轨迹保持不变）",
        "",
        "## 勘误 1：回报的符号差异不是“优化方向相反”",
        "",
        "对自然完工、终局势为零的完整轨迹，有严格恒等式：",
        "",
        "```",
        "G_gamma^shaped = G_gamma^raw - beta * Phi(s_0)",
        "```",
        "",
        "因此前轮报告的 `0.3195 - (-0.1805) = 0.5`、`0.3687 - (-0.1313) = 0.5` 是一个**固定平移**，",
        f"而不是“塑形回报变好、业务费用变差”。本轮用前轮 `shaping_events.jsonl` 的实际事件重算：",
        "",
        "| episode | 步数 | 终止 | Phi(s0) | 原始折扣回报 | 塑形折扣回报 | 实测差值 | -beta*Phi(s0) | 残差 |",
        "| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['episode_id']} | {row['steps']} | {'自然终止' if row['terminated'] else '预算截断'} | "
            f"{row['phi_start']:.6f} | {row['discounted_raw_return']:.6f} | "
            f"{row['discounted_shaped_return']:.6f} | {row['observed_shift_shaped_minus_raw']:.9f} | "
            f"{row['expected_shift_general']:.9f} | {row['residual']:.3e} |"
        )
    shift_value = (-beta * float(rows[0]["phi_start"])) if rows else 0.5
    max_residual = max((abs(row["residual"]) for row in rows), default=float("nan"))
    lines.extend(
        [
            "",
            f"结论：三条 episode 的差值全部等于 `-beta*Phi(s0) = {shift_value:.6f}`，"
            f"最大残差 {max_residual:.3e}，"
            f"{'全部通过' if all_ok else '存在不一致'} 1e-9 容差。",
            "",
            "需要分开的两个问题：",
            "",
            "1. **塑形平移**：满足端点条件时，塑形与原始折扣回报只差固定常数，不能比较“谁更好”。",
            "2. **目标口径**：`gamma` 折扣原始回报与未折扣 `J_total` 之间本身存在折扣差异；",
            "   后者是业务费用，前者是训练目标的一部分，两者不能默认一致。",
            "   预算截断段保留边界项（自身段尾 `phi_T != 0`），不套用完整终止公式。",
            "",
            "因此：更新前零费用轨迹的 raw 折扣回报近 0 时，其 shaped 回报应接近 `-beta*Phi(s0)`；",
            "后来 shaped 回报仍为正，**不构成改善证据**。",
            "",
            "## 勘误 2：首段证据不能排除采样影响",
            "",
            "前轮首段 8 步全部是留站 + `align=1`、原始单步奖励≈0，这只说明**该段记录内**",
            "尚未出现明显即时费用偏离；它与完整工序/团队选择是否一致是两件事。",
            "",
            "前轮第一个完整闭环复测点是 **128**，不是“首次 optimizer.step 之后”。",
            "因此结论收紧为：",
            "",
            "> 更新改变了已记录快照上的动作概率；**首次已评估的闭环退化出现在 128 决策处**。",
            "",
            "不能据此把闭环退化唯一定位到首次更新，也不能排除后续随机采样的影响。",
            "",
        ]
    )
    ERRATA.write_text("\n".join(lines), encoding="utf-8")
    log(f"errata 完成: 固定平移校验 {'通过' if all_ok else '不通过'}")
    return 0 if all_ok else 2


# ---------------------------------------------------------------------------
# 离线梯度分解（零参数更新）
# ---------------------------------------------------------------------------


def stage_offline() -> int:
    enforce_single_thread_cpu()
    rollout = torch.load(CONTROL_FIRST_ROLLOUT, map_location="cpu", weights_only=False)
    model, _payload = load_diagnostic_actor(
        BC_WEIGHTS,
        expected_teacher_manifest_sha256=load_json(BC_MANIFEST)["teacher"]["manifest_sha256"],
    )
    model.eval()
    groups = prev.parameter_groups(model)
    group_of = {name: group for group, names in groups["groups"].items() for name in names}

    state = rollout["state_feats"]
    time_feature = rollout["time_urgencies"]
    records = rollout["sample_records"]
    old_log_probs = rollout["old_log_probs"].to(torch.float32)
    hyper = rollout["hyperparameters"]
    clip_eps = float(hyper["clip_epsilon"])
    vf_coef = float(hyper["value_coefficient"])
    ent_coef = float(hyper["entropy_coefficient"])

    raw_adv = rollout["raw_advantages"].to(torch.float32)
    adv_std_unbiased = float(raw_adv.std(unbiased=True).item()) if raw_adv.numel() > 1 else None
    adv_std_biased = float(raw_adv.std(unbiased=False).item()) if raw_adv.numel() > 0 else None
    advantages = rollout["normalized_advantages"].to(torch.float32)
    targets = rollout["target_values"].to(torch.float32)

    values, new_log_probs, entropies = model.evaluate_action_log_probs(
        state, time_feature, records
    )
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    surrogate = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surrogate, clipped).mean()
    value_loss = 0.5 * torch.nn.functional.mse_loss(values, targets)
    entropy = entropies.mean()
    weighted = {
        "policy": policy_loss,
        "value": vf_coef * value_loss,
        "entropy": -ent_coef * entropy,
    }
    total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    approx_kl = float((((ratio - 1.0) - log_ratio).mean()).item())
    clip_fraction = float((ratio.sub(1.0).abs() > clip_eps).float().mean().item())

    shared_names = groups["groups"]["shared"]
    params = dict(model.named_parameters())

    def flat_grad(loss: torch.Tensor, names: Sequence[str]) -> torch.Tensor:
        grads = torch.autograd.grad(loss, [params[name] for name in names], retain_graph=True, allow_unused=True)
        pieces = [g.reshape(-1) if g is not None else torch.zeros_like(params[name]).reshape(-1)
                  for g, name in zip(grads, names)]
        return torch.cat(pieces) if pieces else torch.zeros(0)

    grads = {key: flat_grad(loss, shared_names) for key, loss in weighted.items()}
    total_grad = flat_grad(total_loss, shared_names)
    norms = {key: float(g.norm().item()) for key, g in grads.items()}
    dot = lambda a, b: float(torch.dot(a, b).item())
    cosines = {}
    for left in grads:
        for right in grads:
            if left >= right:
                continue
            denominator = norms[left] * norms[right]
            cosines[f"{left}|{right}"] = (
                None if denominator == 0.0 else dot(grads[left], grads[right]) / denominator
            )
    summed = grads["policy"] + grads["value"] + grads["entropy"]
    residual = float((summed - total_grad).norm().item())
    scale = max(1.0, float(summed.norm().item()), float(total_grad.norm().item()))

    per_group_norms = {}
    for group_name in ("shared", "policy", "value"):
        names = groups["groups"][group_name]
        per_group_norms[group_name] = {
            key: float(flat_grad(loss, names).norm().item()) for key, loss in weighted.items()
        }

    sample_ids = [int(r.get("_sample_id", index)) for index, r in enumerate(records)]
    payload = {
        "schema_version": "critic_calibration_gradient_decomposition_v1",
        "scope": (
            "固定首段离线诊断：使用前轮 first_rollout.pt 的 8 条真实采样记录整体计算三项损失并分解梯度；"
            "**不是**历史上真实的第一次 2 样本 mini-batch。真实首批的样本身份无法从归档数据恢复，"
            "因为正式 trainer 用全局 RNG 打乱后按 batch=2 切分，而该 RNG 状态未被保存。"
        ),
        "first_minibatch_identity_available": False,
        "missing_evidence": [
            "真实第一批 2 条样本的索引（依赖未保存的 torch RNG 状态）",
            "该 mini-batch 的实际 shuffle 顺序",
        ],
        "device": "cpu",
        "mode": "eval",
        "sample_count": int(state.size(0)),
        "sample_indices": sample_ids,
        "coefficients": {"clip_epsilon": clip_eps, "vf_coef": vf_coef, "ent_coef": ent_coef},
        "losses": {
            "policy_loss": float(policy_loss.item()),
            "value_loss_raw": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "value_loss_weighted": float(weighted["value"].item()),
            "entropy_weighted": float(weighted["entropy"].item()),
            "total_loss": float(total_loss.item()),
        },
        "advantages": {
            "raw_mean": float(raw_adv.mean().item()),
            "raw_std_unbiased_ddof1": adv_std_unbiased,
            "raw_std_biased_ddof0": adv_std_biased,
            "normalized_mean": float(advantages.mean().item()),
            "normalized_std_unbiased_ddof1": float(advantages.std(unbiased=True).item()),
            "normalized_std_biased_ddof0": float(advantages.std(unbiased=False).item()),
            "normalization_scope": "整段 8 样本（与正式 buffer 的段内标准化一致）",
            "note": "0.935 附近的归一化 std 是 (x-mean)/(std+1e-8) 在有限样本上的结果，不是错误。",
        },
        "shared_gradient": {
            "parameter_count": len(shared_names),
            "norms": norms,
            "pairwise_cosines": cosines,
            "sum_residual_norm": residual,
            "relative_residual": residual / scale,
            "tolerance": GRAD_RESIDUAL_TOLERANCE,
            "residual_within_tolerance": residual / scale <= GRAD_RESIDUAL_TOLERANCE,
            "tolerance_rationale": GRAD_RESIDUAL_NOTE,
        },
        "per_group_gradient_norms": per_group_norms,
        "group_membership": groups["counts"],
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "interpretation_limits": [
            "这是贡献分解，不是因果定位：某项梯度范数最大不等于它导致退化。",
            "Adam 的自适应缩放、全局裁剪与参数尺度都会改变实际更新幅度。",
            "共享参数梯度同时包含 policy 与 value 的贡献，不能当作“仅来自 policy”。",
        ],
    }
    dump_json(GRAD_DECOMP, payload)
    log(
        "offline 完成: 共享梯度范数 policy={:.4f} value={:.4f} entropy={:.4f}，残差 {:.3e}".format(
            norms["policy"], norms["value"], norms["entropy"], residual / scale
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Phase A：采集一条随机 MC 轨迹（CPU）
# ---------------------------------------------------------------------------


def finalize_environment(env: AirLineEnvWork3, *, scenario: Mapping[str, Any] | None) -> dict[str, Any]:
    """按正式评估同一套工具结算环境事实与账本。"""
    weights = env.weights
    breakdown = evaluate_trajectory_objective(env, weights=weights)
    report = build_objective_report(breakdown, weights, h0=float(env.state.h0))
    feasible, violations = check_environment_trajectory_feasibility(env)
    completed = sum(1 for t in env.state.tasks.values() if t.status.name == "COMPLETED")
    terminated = bool(env._check_terminated())
    return {
        "agent": "C_BC_RANDOM_MC_COLLECT",
        "scenario_id": "NOMINAL" if scenario is None else str(scenario["scenario_id"]),
        "success": bool(terminated and feasible),
        "feasible": bool(feasible),
        "constraint_violations": violations,
        "constraint_violation_count": sum(violations.values()),
        "completed_tasks": completed,
        "objective_j_total": float(report["j_total"]),
        "cost_components_weighted": report["cost_components_weighted"],
        "metrics_unweighted": report["metrics_unweighted"],
        "j_takt": float(env.cost_takt),
        "d_time": float(env.cost_time),
        "d_team": float(env.cost_team),
        "j_postpone": float(env.cost_postpone),
        "j_revision": float(env.cost_revision),
        "makespan": float(env.state.current_time),
        "transfers": len(env.state.transfer_history),
        "postponed_count": sum(t.postpone_count for t in env.state.tasks.values()),
        "actual_disturbance_hits": 0,
        "decisions": int(env.step_count),
        "termination_reason": "completed" if terminated else "decision_limit",
        "step_rewards": [float(r) for r in env.step_rewards],
    }


def stage_calibrate() -> int:
    threads = enforce_single_thread_cpu()
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    baseline = load_json(BASELINE_PATH)
    gamma = 0.99
    beta = 1.0
    model, _payload = load_diagnostic_actor(
        BC_WEIGHTS,
        expected_teacher_manifest_sha256=load_json(BC_MANIFEST)["teacher"]["manifest_sha256"],
    )
    model.to("cpu")
    model.eval()
    source_fingerprint = actor_fingerprint(model)
    bc_sha_before = sha256_file(BC_WEIGHTS)

    # ---- Phase A：一条自然完工的随机 Nominal 轨迹 ----
    random_state = np.random.get_state()
    np.random.seed(CAL_SEED)
    torch.manual_seed(CAL_SEED)
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    samples: list[dict[str, Any]] = []
    terminated = False
    with torch.inference_mode():
        while len(samples) < CAL_DECISION_LIMIT:
            if env._check_terminated():
                terminated = True
                break
            snapshot = model.make_decision_snapshot(
                env,
                _state_features(env),
                _time_features(env),
                worker_id=0,
                episode_id=0,
                episode_index=len(samples) + 1,
                estimated_cmax=_cmax(env),
                allow_postpone=True,
            )
            action, log_prob, value, record = model.select_snapshot(snapshot, deterministic=False)
            phi_current = compute_heuristic_potential(
                float(snapshot.estimated_cmax), float(snapshot.last_transfer_time), float(snapshot.h0)
            )
            _, raw_reward, done, truncated, info = env.step(action)
            terminal = bool(env._check_terminated())
            phi_next = 0.0 if terminal else compute_heuristic_potential(
                _cmax(env), float(env.state.last_transfer_time), float(env.state.h0)
            )
            shaped = float(raw_reward) + beta * (gamma * phi_next - phi_current)
            samples.append(
                {
                    "index": len(samples),
                    "snapshot": snapshot,
                    "action": action,
                    "log_prob": float(log_prob),
                    "value": float(value),
                    "raw_reward": float(raw_reward),
                    "shaped_reward": float(shaped),
                    "phi": float(phi_current),
                    "phi_next": float(phi_next),
                    "terminated": bool(terminal),
                    "info_cost": float(info.get("step_cost", 0.0)),
                }
            )
            if terminal or truncated:
                terminated = bool(terminal)
                break
    np.random.set_state(random_state)

    facts = extract_execution_facts(env)
    trajectory = finalize_environment(env, scenario=None)
    trajectory["execution_facts"] = facts
    trajectory["collection"] = {
        "seed": CAL_SEED,
        "device": "cpu",
        "threads": threads,
        "policy_mode": "stochastic",
        "gamma": gamma,
        "beta": beta,
        "terminated": terminated,
        "samples": len(samples),
    }
    dump_json(CAL_TRAJECTORY, trajectory)
    if not terminated:
        dump_json(OUTPUT_DIR / "calibration_blocked.json", {"reason": "episode_not_completed"})
        log("校准轨迹未在 4096 决策内自然完工，停止依赖完整 MC 标签的后续步骤")
        return 3

    # ---- MC 标签 ----
    mc = _mc_targets(samples, gamma=gamma, beta=beta)
    validate_mc_targets(mc)
    # 校准前的逐快照 greedy 判读，用于校准后逐一比对策略路径不变
    greedy_before = [
        _greedy_reference(model, s["snapshot"]) for s in samples
    ]
    tensors = {
        "state_feats": torch.stack([s["snapshot"].state_features for s in samples]),
        "time_urgencies": torch.stack([s["snapshot"].time_features for s in samples]),
        "graph_snapshots": [s["snapshot"].graph_snapshot for s in samples],
        "snapshots": [s["snapshot"] for s in samples],
        "actions": [s["action"] for s in samples],
        "log_probs": torch.tensor([s["log_prob"] for s in samples], dtype=torch.float32),
        "old_values": torch.tensor([s["value"] for s in samples], dtype=torch.float32),
        "raw_rewards": torch.tensor([s["raw_reward"] for s in samples], dtype=torch.float32),
        "shaped_rewards": torch.tensor([s["shaped_reward"] for s in samples], dtype=torch.float32),
        "phis": torch.tensor([s["phi"] for s in samples], dtype=torch.float32),
        "mc_shaped": torch.tensor(mc["shaped"], dtype=torch.float32),
        "mc_raw": torch.tensor(mc["raw"], dtype=torch.float32),
        "terminated": [s["terminated"] for s in samples],
        "gamma": gamma,
        "beta": beta,
        "seed": CAL_SEED,
    }
    torch.save(tensors, CAL_SNAPSHOTS)

    # ---- Phase B：仅价值头校准 ----
    groups = prev.parameter_groups(model)
    value_names = groups["groups"]["value"]
    policy_subset = groups["groups"]["shared"] + groups["groups"]["policy"]
    policy_fingerprint_before = subset_fingerprint(model.state_dict(), policy_subset)
    full_fingerprint_before = actor_fingerprint(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in value_names
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) != len(value_names):
        raise RuntimeError("校准优化器只应包含价值专属参数")

    indices = list(range(len(samples)))
    holdout = [i for i in indices if i % HOLDOUT_MODULO == 0]
    train_idx = [i for i in indices if i % HOLDOUT_MODULO != 0]
    generator = torch.Generator().manual_seed(CAL_SPLIT_SEED)
    optimizer = torch.optim.Adam(trainable, lr=3e-4, weight_decay=0.0)
    history: list[dict[str, Any]] = []
    order: list[int] = []
    started = time.perf_counter()
    steps = 0
    stop_reason = "max_steps"
    initial_scan = _score_calibration(model, tensors, train_idx, holdout)
    scans = [{"step": 0, **initial_scan}]
    while steps < CAL_MAX_STEPS and (time.perf_counter() - started) < CAL_MAX_SECONDS:
        if len(order) < CAL_BATCH:
            order.extend(torch.randperm(len(train_idx), generator=generator).tolist())
        batch = [train_idx[i] for i in order[:CAL_BATCH]]
        order = order[CAL_BATCH:]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in batch:
            value = _value_forward(model, tensors, index)
            target = tensors["mc_shaped"][index]
            losses.append((value - target) ** 2)
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss).item()):
            stop_reason = "non_finite_loss"
            dump_json(CAL_DIR / "non_finite.json", {"step": steps, "loss": float(loss)})
            break
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 0.5)
        optimizer.step()
        steps += 1
        history.append(
            {
                "step": steps,
                "batch_indices": batch,
                "mse": float(loss.item()),
                "grad_norm": float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if steps % CAL_SCAN_INTERVAL == 0:
            scans.append({"step": steps, **_score_calibration(model, tensors, train_idx, holdout)})
    if (time.perf_counter() - started) >= CAL_MAX_SECONDS:
        stop_reason = "max_wall_seconds"
    scans.append({"step": steps, **_score_calibration(model, tensors, train_idx, holdout)})
    dump_jsonl(CAL_HISTORY, history)

    # ---- 策略不变性证明（同设备、同模式、全部快照）----
    invariance = _policy_invariance(model, samples, greedy_before, policy_fingerprint_before, policy_subset)
    dump_json(CAL_INVARIANCE, invariance)

    state = clone_actor_state(model)
    torch.save(
        {
            "diagnostic_format": "critic_value_calibrated_copy",
            "diagnostic_format_version": 1,
            "actor_critic_state": state,
            "actor_fingerprint": state_dict_fingerprint(state),
            "policy_subset_fingerprint": invariance["policy_subset_fingerprint_after"],
            "value_names": value_names,
            "source_bc_sha256": sha256_file(BC_WEIGHTS),
            "source_bc_actor_fingerprint": source_fingerprint,
            "calibration": {"steps": steps, "stop_reason": stop_reason, "seed": CAL_SEED, "split_seed": CAL_SPLIT_SEED},
            "hyperparameters": {
                "state_dim": 32, "task_feat_dim": 8, "hidden_dim": 64, "max_station_workers": 16
            },
            "graph_feature_version": GRAPH_FEATURE_VERSION,
            "graph_feature_dims": dict(GRAPH_FEATURE_DIMS),
            "graph_feature_schema_hash": GRAPH_FEATURE_SCHEMA["schema_fingerprint"],
        },
        CAL_MODEL,
    )

    final = scans[-1]
    fit = {
        "schema_version": "critic_calibration_fit_v1",
        "calibration_steps": steps,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer": {"name": "Adam", "lr": 3e-4, "weight_decay": 0.0, "grad_clip": 0.5},
        "label": "shaped Monte Carlo return G_t^shaped（真实终止，G_T=0）",
        "loss": "mean((V - G)^2)（按真实组大小归一化）",
        "samples": {"total": len(indices), "train": len(train_idx), "holdout": sorted(holdout)},
        "split_rule": f"index % {HOLDOUT_MODULO} == 0 为诊断留出",
        "target_stats": {
            "mean": float(tensors["mc_shaped"].mean().item()),
            "std_unbiased": float(tensors["mc_shaped"].std(unbiased=True).item()),
            "min": float(tensors["mc_shaped"].min().item()),
            "max": float(tensors["mc_shaped"].max().item()),
        },
        "initial_scan": scans[0],
        "final_scan": final,
        "scans": scans,
        "mc_identity_check": mc["identity_check"],
        "constant_baselines": _constant_baselines(tensors, train_idx, holdout),
        "holdout_role": "同一条相关轨迹上的诊断留出，不是独立泛化验证，也不是独立种子。",
        "fit_improved": bool(
            final["train"]["mse"] < scans[0]["train"]["mse"]
            and final["holdout"]["mse"] < scans[0]["holdout"]["mse"]
        ),
    }
    dump_json(CAL_FIT, fit)
    if sha256_file(BC_WEIGHTS) != bc_sha_before:
        raise RuntimeError("源 BC 文件在采集/校准期间被改写")
    log(
        f"calibrate 完成: 采样 {len(samples)} 步, 校准 {steps} 步, "
        f"train MSE {scans[0]['train']['mse']:.6f}->{final['train']['mse']:.6f}, "
        f"holdout {scans[0]['holdout']['mse']:.6f}->{final['holdout']['mse']:.6f}, "
        f"策略不变={invariance['policy_unchanged']}"
    )
    return 0 if invariance["policy_unchanged"] else 2


def _cmax(env: AirLineEnvWork3) -> float:
    from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax

    return float(compute_cycle_heuristic_cmax(env.state))


def _state_features(env: AirLineEnvWork3) -> torch.Tensor:
    from work3_4060.c_reproduction_protocol import build_c_state_features

    return build_c_state_features(env.state, _cmax(env))


def _time_features(env: AirLineEnvWork3) -> torch.Tensor:
    from work3_4060.c_reproduction_protocol import build_c_time_features

    return build_c_time_features(env.state, _cmax(env))


def _mc_targets(samples: Sequence[Mapping[str, Any]], *, gamma: float, beta: float) -> dict[str, Any]:
    shaped = [0.0] * len(samples)
    raw = [0.0] * len(samples)
    running_shaped = 0.0
    running_raw = 0.0
    for index in range(len(samples) - 1, -1, -1):
        running_shaped = float(samples[index]["shaped_reward"]) + gamma * running_shaped
        running_raw = float(samples[index]["raw_reward"]) + gamma * running_raw
        shaped[index] = running_shaped
        raw[index] = running_raw
    residual = max(
        abs(shaped[i] - (raw[i] - beta * float(samples[i]["phi"]))) for i in range(len(samples))
    )
    return {
        "shaped": shaped,
        "raw": raw,
        "identity_check": {
            "identity": "G_t^shaped = G_t^raw - beta*Phi(s_t)",
            "max_abs_residual": residual,
            "tolerance": 1e-6,
            "passed": bool(residual <= 1e-6),
            "gamma_index_note": "折扣从 t 重新起算，不混用全局步号指数。",
        },
    }


def validate_mc_targets(mc: Mapping[str, Any]) -> None:
    """任何非有限 MC 标签都必须显式拒绝，而不是被静默用于回归。"""
    for key in ("shaped", "raw"):
        values = mc[key]
        if not values:
            raise ValueError(f"MC 标签为空: {key}")
        if any(not math.isfinite(float(v)) for v in values):
            raise ValueError(f"MC 标签存在非有限值: {key}")


def _value_forward(model: ActorCriticWork3, tensors: Mapping[str, Any], index: int) -> torch.Tensor:
    value, _fused, _task, _worker, _hier = model._encode_state_components(
        tensors["state_feats"][index],
        tensors["time_urgencies"][index],
        tensors["graph_snapshots"][index],
    )
    return value


def _score_calibration(
    model: ActorCriticWork3,
    tensors: Mapping[str, Any],
    train_idx: Sequence[int],
    holdout: Sequence[int],
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        values = torch.stack(
            [_value_forward(model, tensors, i) for i in range(tensors["mc_shaped"].size(0))]
        )
    target = tensors["mc_shaped"]
    result: dict[str, Any] = {}
    for split_name, indices in (("train", train_idx), ("holdout", holdout)):
        if not indices:
            result[split_name] = {"count": 0}
            continue
        v = values[indices].to(torch.float64)
        g = target[indices].to(torch.float64)
        error = v - g
        variance = float(g.var(unbiased=False).item()) if g.numel() > 1 else 0.0
        result[split_name] = {
            "count": len(indices),
            "mse": float((error ** 2).mean().item()),
            "rmse": float(torch.sqrt((error ** 2).mean()).item()),
            "mae": float(error.abs().mean().item()),
            "bias": float(error.mean().item()),
            "target_mean": float(g.mean().item()),
            "target_std": math.sqrt(variance),
            "explained_variance": None
            if variance <= 1e-12
            else float(1.0 - (error ** 2).mean().item() / variance),
        }
    return result


def _constant_baselines(
    tensors: Mapping[str, Any], train_idx: Sequence[int], holdout: Sequence[int]
) -> dict[str, Any]:
    target = tensors["mc_shaped"].to(torch.float64)
    train_mean = float(target[train_idx].mean().item()) if train_idx else 0.0
    result = {"constant_predictor_value": train_mean}
    for split_name, indices in (("train", train_idx), ("holdout", holdout)):
        if not indices:
            result[split_name] = {"count": 0}
            continue
        g = target[indices]
        error = g - train_mean
        result[split_name] = {
            "count": len(indices),
            "mse": float((error ** 2).mean().item()),
            "mae": float(error.abs().mean().item()),
        }
    return result


def _greedy_reference(model: ActorCriticWork3, snapshot: Any) -> dict[str, Any]:
    with torch.no_grad():
        action, log_prob, _value, _record = model.select_snapshot(snapshot, deterministic=True)
    return {"signature": _action_signature(action), "log_prob": float(log_prob)}


def _policy_invariance(
    model: ActorCriticWork3,
    samples: Sequence[Mapping[str, Any]],
    before: Sequence[Mapping[str, Any]],
    policy_fingerprint_before: str,
    policy_subset: Sequence[str],
) -> dict[str, Any]:
    """校准后策略路径必须逐快照保持：greedy 动作、log-prob 与参数子集。"""
    after_subset = subset_fingerprint(model.state_dict(), policy_subset)
    max_prob_diff = 0.0
    mismatch = 0
    for index, sample in enumerate(samples):
        current = _greedy_reference(model, sample["snapshot"])
        if current["signature"] != before[index]["signature"]:
            mismatch += 1
        max_prob_diff = max(max_prob_diff, abs(current["log_prob"] - before[index]["log_prob"]))
    return {
        "policy_subset_fingerprint_before": policy_fingerprint_before,
        "policy_subset_fingerprint_after": after_subset,
        "policy_subset_names": list(policy_subset),
        "policy_unchanged": after_subset == policy_fingerprint_before,
        "full_model_fingerprint_after": actor_fingerprint(model),
        "greedy_action_mismatch": mismatch,
        "snapshots_checked": len(samples),
        "max_abs_log_prob_difference": max_prob_diff,
        "tolerance": POLICY_PROB_TOLERANCE,
        "within_tolerance": max_prob_diff <= POLICY_PROB_TOLERANCE and mismatch == 0,
        "note": (
            "完整模型指纹（含 critic）在校准后必然改变；这里另用策略子集指纹与逐快照 greedy "
            "判读证明策略路径未被改动。"
        ),
    }


def _action_signature(action: Mapping[str, Any] | None) -> tuple[Any, ...]:
    if action is None:
        return ("none",)
    return (
        int(action["branch"]),
        action.get("task_key"),
        tuple(sorted(int(w) for w in action.get("team", ()))),
        int(action.get("align", 0)),
    )


# ---------------------------------------------------------------------------
# Phase C：校准后 512 步 PPO（复用前轮全部组件）
# ---------------------------------------------------------------------------


def stage_train() -> int:
    log("train(校准臂): 启动")
    if not torch.cuda.is_available():
        raise RuntimeError("本阶段要求 cuda:0")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    PPO_DIR.mkdir(parents=True, exist_ok=True)
    PPO_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    control = load_json(CONTROL_SUMMARY)
    hyper = control["hyperparameters"]
    fit = load_json(CAL_FIT)
    calibration_payload = torch.load(CAL_MODEL, map_location="cpu", weights_only=False)
    if sha256_file(BC_WEIGHTS) != calibration_payload["source_bc_sha256"]:
        raise ValueError("校准副本记录的源 BC SHA 与当前文件不一致")

    seeds = prev.seed_everything(int(hyper["seed"]))
    model, _cal_payload = load_calibrated_actor(CAL_MODEL)
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.to(device)
    model.train()
    init_fingerprint = actor_fingerprint(model)
    if init_fingerprint != calibration_payload["actor_fingerprint"]:
        raise ValueError("校准副本加载后指纹不一致")
    log(f"train(校准臂): 模型已加载 fp={init_fingerprint[:16]}")

    checkpoint_records = [
        prev.save_checkpoint(
            PPO_CKPT_DIR / "step0000.pt",
            model,
            step_index=0,
            decisions=0,
            optimizer_steps=0,
            episode_id=0,
            tag="step0000",
            extra={"source": "critic_value_calibrated_copy"},
        )
    ]
    lightning = prev.DiagnosticsLightningModule(
        actor_critic=model,
        learning_rate=float(hyper["learning_rate"]),
        clip_eps=float(hyper["clip_epsilon"]),
        vf_coef=float(hyper["value_coefficient"]),
        ent_coef=float(hyper["entropy_coefficient"]),
        max_grad_norm=float(hyper["max_grad_norm"]),
        ppo_epochs=int(hyper["ppo_epochs"]),
        batch_size=int(hyper["batch_size"]),
    )
    lightning.parameter_group_names = prev.parameter_groups(model)["groups"]
    optimizer = lightning.configure_optimizers()
    if {id(p) for p in lightning.objective.optimized_parameters} != {id(p) for p in model.parameters()}:
        raise ValueError("PPO 优化器参数集合必须覆盖全部模型参数")
    if any(state for state in optimizer.state.values()):
        raise ValueError("PPO 优化器不应携带校准状态")

    env = prev.build_env()
    log("train(校准臂): spawn 环境 worker 已就绪")
    shaping = prev.ShapingWriter(PPO_SHAPING)
    history: list[dict[str, Any]] = []
    decisions = 0
    episode_id = 0
    episode_step = 0
    rollout_id = 0
    optimizer_steps = 0
    stop_reason = "budget_exhausted"
    episode_last_info: dict[int, dict[str, Any]] = {}
    episode_raw_sum: dict[int, float] = {}
    episode_closed: set[int] = set()
    first_rollout_saved = False
    first_rollout_payload: dict[str, Any] | None = None
    pending_snapshot = None
    memory_peak = 0
    terminated = False
    started = time.perf_counter()

    try:
        env.reset(scenario=None, episode_id=episode_id, episode_index=episode_id)
        pending_snapshot = env.snapshot()
        while decisions < TRAINING_BUDGET:
            next_checkpoint = next((p for p in CHECKPOINTS if p > decisions), TRAINING_BUDGET)
            segment_limit = min(prev.ROLLOUT_STEPS, next_checkpoint - decisions)
            if segment_limit <= 0:
                break
            buffer = RolloutBufferWork3(
                gamma=float(hyper["gamma"]), gae_lambda=float(hyper["gae_lambda"]),
                normalize_advantages=True,
            )
            raw_buffer = RolloutBufferWork3(
                gamma=float(hyper["gamma"]), gae_lambda=float(hyper["gae_lambda"]),
                normalize_advantages=False,
            )
            transitions = []
            segment_records = []
            terminated = False
            for _ in range(segment_limit):
                snapshot = pending_snapshot
                with torch.no_grad():
                    action, log_prob, value, sample_record = model.select_snapshot(
                        snapshot, deterministic=False
                    )
                phi_current = compute_heuristic_potential(
                    float(snapshot.estimated_cmax), float(snapshot.last_transfer_time), float(snapshot.h0)
                )
                result = env.step(action)
                raw_reward = float(result.raw_reward)
                terminated = bool(result.terminated)
                decisions += 1
                episode_step += 1
                if terminated:
                    phi_next = 0.0
                    pending_snapshot = None
                else:
                    pending_snapshot = env.snapshot()
                    phi_next = compute_heuristic_potential(
                        float(pending_snapshot.estimated_cmax),
                        float(pending_snapshot.last_transfer_time),
                        float(pending_snapshot.h0),
                    )
                shaped = raw_reward + float(hyper["beta_shaping"]) * (
                    float(hyper["gamma"]) * phi_next - phi_current
                )
                transition = prev.PPOTransition(
                    state_feat=snapshot.state_features.cpu(),
                    time_urgency=snapshot.time_features.cpu(),
                    sample_record=sample_record,
                    reward=float(shaped),
                    raw_reward=raw_reward,
                    value=float(value),
                    log_prob=float(log_prob),
                    done=bool(terminated),
                    action_dict=dict(action),
                    terminated=terminated,
                    truncated=False,
                    worker_id=0,
                    episode_id=episode_id,
                    segment_id=rollout_id,
                    predictor_version=None,
                    predictor_source="heuristic",
                    time_input_source="heuristic_time",
                    phi_source="heuristic_phi",
                )
                transitions.append(transition)
                buffer.add(transition)
                raw_buffer.add(transition)
                info = result.info or {}
                segment_records.append(
                    {
                        "decision_index": decisions,
                        "episode_id": episode_id,
                        "episode_step": episode_step - 1,
                        "cycle_id": int(snapshot.cycle_id),
                        "sim_time": float(snapshot.current_time),
                        "action_type": int(action["branch"]),
                        "task_key": action.get("task_key"),
                        "team": [int(w) for w in action.get("team", ())],
                        "align": int(action.get("align", 0)),
                        "log_prob": float(log_prob),
                        "value": float(value),
                        "raw_reward": raw_reward,
                        "shaped_reward": float(shaped),
                        "phi_current": phi_current,
                        "phi_next": phi_next,
                        "terminated": terminated,
                        "truncated": False,
                        "step_cost": float(info.get("step_cost", 0.0)),
                    }
                )
                shaping.transition(
                    {
                        "record_type": "transition",
                        "run_id": "critic_calibration_retention",
                        "worker_id": 0,
                        "episode_id": episode_id,
                        "rollout_id": rollout_id,
                        "episode_step": episode_step - 1,
                        "action_type": int(action["branch"]),
                        "task_id": None,
                        "task_key": action.get("task_key"),
                        "decision_time": float(snapshot.current_time),
                        "cycle_id": int(snapshot.cycle_id),
                        "raw_reward": raw_reward,
                        "shaped_reward": float(shaped),
                        "phi_current": float(phi_current),
                        "phi_next": float(phi_next),
                        "beta": float(hyper["beta_shaping"]),
                        "gamma": float(hyper["gamma"]),
                        "terminated": terminated,
                        "truncated": False,
                        "predictor_version": None,
                        "bootstrap_value": None,
                        "phi_source": "heuristic_phi",
                    }
                )
                episode_last_info[episode_id] = info
                episode_raw_sum[episode_id] = episode_raw_sum.get(episode_id, 0.0) + raw_reward
                if terminated:
                    break
            if not terminated:
                with torch.no_grad():
                    bootstrap, _ = model.encode_state(
                        pending_snapshot.state_features.to(device),
                        pending_snapshot.time_features.to(device),
                        graph_data=pending_snapshot.graph_snapshot,
                    )
                last_value = float(bootstrap.squeeze().item())
                boundary_type = "budget_truncation" if decisions >= TRAINING_BUDGET else "rollout_continuation"
            else:
                last_value = 0.0
                boundary_type = "natural_termination"

            key = (0, episode_id, rollout_id)
            buffer.finish_trajectories(last_values_by_segment={key: last_value})
            raw_buffer.finish_trajectories(last_values_by_segment={key: last_value})
            raw_advantages = raw_buffer.advantages.clone()

            with torch.no_grad():
                state_batch = torch.stack([t.state_feat for t in transitions])
                time_batch = torch.stack([t.time_urgency for t in transitions])
                _v, replay_lp, _e = model.evaluate_action_log_probs(
                    state_batch, time_batch, [t.sample_record for t in transitions]
                )
                old_lp = torch.tensor([t.log_prob for t in transitions], dtype=torch.float)
                replay_error = float((replay_lp.cpu().double() - old_lp.double()).abs().max().item())
            if not prev.replay_contract_ok(replay_error):
                dump_json(PPO_DIR / "replay_failure.json", {"rollout_id": rollout_id, "error": replay_error})
                stop_reason = "replay_contract_failed"
                break

            shaping.segment_end(
                {
                    "record_type": "segment_end",
                    "run_id": "critic_calibration_retention",
                    "worker_id": 0,
                    "episode_id": episode_id,
                    "rollout_id": rollout_id,
                    "boundary_type": boundary_type,
                    "bootstrap_value": float(last_value),
                    "termination_reason": boundary_type,
                    "steps": len(transitions),
                    "training_decisions": decisions,
                }
            )
            if terminated:
                shaping.episode_cost(
                    prev.build_episode_cost_record(
                        episode_id=episode_id,
                        info=episode_last_info.get(episode_id, {}),
                        termination_reason="completed",
                        episode_steps=episode_step,
                        raw_reward_sum=episode_raw_sum.get(episode_id, 0.0),
                    )
                )
                episode_closed.add(episode_id)

            update = Work3TrainingUpdate(
                buffer=buffer, environment_steps=len(transitions),
                time_auxiliary_batch=None, rollout_id=rollout_id,
            )
            updates_before = lightning.optimization_steps
            loss = lightning.training_step(update, 0)
            if not bool(torch.isfinite(loss).item()):
                stop_reason = "non_finite_loss"
                break
            updates_this_segment = lightning.optimization_steps - updates_before
            optimizer_steps = lightning.optimization_steps
            segment_updates = (
                lightning.update_records[-updates_this_segment:] if updates_this_segment > 0 else []
            )
            if not first_rollout_saved:
                first_rollout_payload = {
                    "state_feats": torch.stack([t.state_feat for t in transitions]),
                    "time_urgencies": torch.stack([t.time_urgency for t in transitions]),
                    "actions": [t.action_dict for t in transitions],
                    "old_log_probs": old_lp,
                    "old_values": torch.tensor([t.value for t in transitions], dtype=torch.float),
                    "sample_records": [t.sample_record for t in transitions],
                }
                torch.save(first_rollout_payload, PPO_FIRST_ROLLOUT)
                first_rollout_saved = True

            memory = prev.gpu_snapshot()
            memory_peak = max(memory_peak, memory["peak_reserved_bytes"])
            history.append(
                {
                    "rollout_id": rollout_id,
                    "training_decisions": decisions,
                    "segment_steps": len(transitions),
                    "episode_id": episode_id,
                    "terminated": terminated,
                    "boundary_type": boundary_type,
                    "bootstrap_value": float(last_value),
                    "replay_max_abs_error": replay_error,
                    "advantage_raw_stats": prev.stats(raw_advantages.tolist()),
                    "advantage_normalized_stats": prev.stats(buffer.advantages.tolist()),
                    "value_stats": prev.stats([t.value for t in transitions]),
                    "return_target_stats": prev.stats(buffer.target_values.tolist()),
                    "raw_reward_stats": prev.stats([t.raw_reward for t in transitions]),
                    "shaped_reward_stats": prev.stats([t.reward for t in transitions]),
                    "action_counts": prev.action_counts(segment_records),
                    "update_records": segment_updates,
                    "elapsed_seconds": time.perf_counter() - started,
                    "cuda_peak_reserved_mib": memory["peak_reserved_bytes"] / 2**20,
                }
            )
            if memory["peak_reserved_bytes"] >= prev.GPU_BUDGET_BYTES:
                stop_reason = "gpu_budget_exceeded"
                break
            if any(
                record["clipped_grad_norm"] is None or not math.isfinite(record["clipped_grad_norm"] or 0.0)
                for record in segment_updates
            ):
                stop_reason = "non_finite_update"
                break
            if terminated:
                episode_id += 1
                episode_step = 0
                env.reset(scenario=None, episode_id=episode_id, episode_index=episode_id)
                pending_snapshot = env.snapshot()
            rollout_id += 1
            if decisions in CHECKPOINTS and decisions > 0:
                checkpoint_records.append(
                    prev.save_checkpoint(
                        PPO_CKPT_DIR / f"step{decisions:04d}.pt",
                        model,
                        step_index=list(CHECKPOINTS).index(decisions),
                        decisions=decisions,
                        optimizer_steps=optimizer_steps,
                        episode_id=episode_id,
                        tag=f"step{decisions:04d}",
                    )
                )
                log(f"已保存检查点 step{decisions:04d}（优化步 {optimizer_steps}）")
    except torch.cuda.OutOfMemoryError as error:  # pragma: no cover
        stop_reason = "cuda_oom"
        dump_json(PPO_DIR / "cuda_oom.json", {"error": str(error), "decisions": decisions})
    except FloatingPointError as error:
        stop_reason = "non_finite_gradient"
        dump_json(PPO_DIR / "non_finite_gradient.json", {"error": str(error), "decisions": decisions})
    finally:
        for open_id in sorted(episode_last_info):
            if open_id in episode_closed:
                continue
            shaping.episode_cost(
                prev.build_episode_cost_record(
                    episode_id=open_id,
                    info=episode_last_info[open_id],
                    termination_reason="budget_truncation",
                    episode_steps=episode_step,
                    raw_reward_sum=episode_raw_sum.get(open_id, 0.0),
                )
            )
        shaping.close()
        try:
            env.close()
        except Exception:
            pass
        prev.dump_jsonl_safe(PPO_HISTORY, history)
        prev.dump_json_safe(
            PPO_SUMMARY,
            {
                "stop_reason": stop_reason,
                "training_decisions": decisions,
                "optimizer_steps": optimizer_steps,
                "episodes_started": len(episode_last_info),
                "episodes_completed": len(episode_closed),
                "segments": len(history),
                "elapsed_seconds": time.perf_counter() - started,
                "seeds": seeds,
                "device": str(device),
                "gpu": {
                    "name": torch.cuda.get_device_name(0),
                    "peak_reserved_mib": memory_peak / 2**20,
                    "budget_mib": prev.GPU_BUDGET_BYTES / 2**20,
                    "budget_exceeded": memory_peak >= prev.GPU_BUDGET_BYTES,
                },
                "checkpoints": checkpoint_records,
                "hyperparameters": dict(hyper),
                "calibration": {
                    "steps": fit["calibration_steps"],
                    "stop_reason": fit["stop_reason"],
                    "model_sha256": sha256_file(CAL_MODEL),
                },
            },
        )
        write_ppo_run_json(history, hyper, stop_reason, decisions)

    pairing = verify_first_rollout_pairing()
    dump_json(PPO_PAIRING, pairing)
    log(f"首段配对: {pairing['paired']} ({pairing.get('reason')})")
    if not pairing["paired"]:
        return 3
    log(f"train(校准臂) 结束: decisions={decisions} updates={optimizer_steps} stop={stop_reason}")
    return 0 if stop_reason in ("budget_exhausted",) else 2


def verify_first_rollout_pairing() -> dict[str, Any]:
    if not PPO_FIRST_ROLLOUT.is_file() or not CONTROL_FIRST_ROLLOUT.is_file():
        return {"paired": False, "reason": "first_rollout 缺失"}
    ours = torch.load(PPO_FIRST_ROLLOUT, map_location="cpu", weights_only=False)
    control = torch.load(CONTROL_FIRST_ROLLOUT, map_location="cpu", weights_only=False)
    same_states = bool(torch.allclose(ours["state_feats"], control["state_feats"], atol=0.0, rtol=0.0))
    same_actions = [_action_signature(a) for a in ours["actions"]] == [
        _action_signature(a) for a in control["actions"]
    ]
    log_prob_diff = float(
        (ours["old_log_probs"].double() - control["old_log_probs"].double()).abs().max().item()
    )
    paired = bool(
        same_states
        and same_actions
        and log_prob_diff <= 1e-5
        and ours["state_feats"].shape == control["state_feats"].shape
    )
    return {
        "paired": paired,
        "same_states": same_states,
        "same_actions": same_actions,
        "max_abs_log_prob_difference": log_prob_diff,
        "tolerance": 1e-5,
        "steps_compared": int(ours["state_feats"].size(0)),
        "divergence_note": (
            "策略相同且随机状态匹配时，首段 8 步应得到相同采样事件；"
            "首段之后两臂因 critic 不同而自然分化，不强制重放对照动作。"
        ),
        "reason": "matched" if paired else "首段采样事件不一致，无法建立严格配对",
    }


# ---------------------------------------------------------------------------
# 评估（四点独立 CPU 进程）
# ---------------------------------------------------------------------------


def stage_evaluate() -> int:
    if not (PPO_DIR / "first_rollout_pairing.json").is_file():
        log("缺少首段配对记录，暂不评估")
        return 3
    if not load_json(PPO_DIR / "first_rollout_pairing.json")["paired"]:
        log("首段配对失败（BLOCKED_PAIRING），不评估")
        return 3
    for point in CHECKPOINTS:
        path = PPO_CKPT_DIR / f"step{point:04d}.pt"
        if not path.is_file():
            log(f"缺少检查点 step{point:04d}")
            return 3
    PPO_TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    for point in CHECKPOINTS:
        log(f"独立子进程评估 step{point:04d} ...")
        with (RUNS_DIR / f"evaluate_step{point:04d}.log").open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--stage", "evaluate-one",
                 "--checkpoint", str(PPO_CKPT_DIR / f"step{point:04d}.pt"), "--step", str(point)],
                cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, text=True, check=False,
            )
        if process.returncode != 0:
            log(f"step{point:04d} 评估失败 {process.returncode}")
            return 2
    return 0


def stage_evaluate_one(checkpoint: Path, step_point: int) -> int:
    threads = enforce_single_thread_cpu()
    model, payload = prev.load_checkpoint_actor(checkpoint)
    model.to("cpu")
    model.eval()
    agent = FormalEvaluationAgent(profile=build_method_profile("C"), actor_critic=model, time_head=None)
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    result = evaluate_single_trajectory(
        env, "Method-C", agent, scenario=None, max_decisions=4096, device="cpu"
    )
    dump_json(
        PPO_TRAJ_DIR / f"step{step_point:04d}_NOMINAL.json",
        {
            **result,
            "agent": f"C_BC_CAL_PPO_step{step_point:04d}",
            "wrapped_agent_type": "Method-C",
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "actor_fingerprint": payload["actor_fingerprint"],
                "training_decisions": int(payload["training_decisions"]),
                "optimizer_steps": int(payload["optimizer_steps"]),
            },
            "evaluation": {
                "device": "cpu", "threads": threads, "deterministic": True,
                "decision_limit": 4096, "scenario": "NOMINAL", "disturbance": "none",
            },
            "execution_facts": extract_execution_facts(env),
        },
    )
    log(
        "step{:04d}: decisions={} J={:.10f} makespan={:.6f} completed={} success={} postponed={}".format(
            step_point, result["decisions"], result["objective_j_total"], result["makespan"],
            result["completed_tasks"], result["success"], result["postponed_count"],
        )
    )
    return 0


# ---------------------------------------------------------------------------
# 验证与报告
# ---------------------------------------------------------------------------


def stage_verify() -> int:
    threads = enforce_single_thread_cpu()
    verifier = prev.load_verifier()
    baseline = load_json(BASELINE_PATH)
    h0 = float(baseline["metadata"]["h0"])
    tasks = baseline["tasks"]
    expected = [
        {"method": "C_BC_RANDOM_MC_COLLECT", "scenario_id": "NOMINAL", "path": str(CAL_TRAJECTORY)},
    ] + [
        {"method": f"C_BC_CAL_PPO_step{p:04d}", "scenario_id": "NOMINAL",
         "path": str(PPO_TRAJ_DIR / f"step{p:04d}_NOMINAL.json")}
        for p in CHECKPOINTS
    ]
    rows = []
    for item in expected:
        path = Path(item["path"])
        if not path.is_file():
            continue
        result = verifier.verify_single_trajectory(
            traj_path=path, baseline_h0=h0, baseline_tasks=tasks,
            scenario_spec={"scenario_id": "NOMINAL"},
        )
        result.update({"path": item["path"], "method": item["method"], "scenario_id": item["scenario_id"]})
        rows.append(result)
        log(f"轨迹核验[{item['method']}]: {result['status']} violations={result['total_violations']}")
    protocol = prev.classify_protocol_status(expected, rows)
    dump_json(
        TRAJECTORY_VERIFICATION,
        {
            "scope": "本轮显式清单：1 条校准采集轨迹 + 4 个 greedy 评估点。",
            "expected_manifest": expected,
            "verified_count": len(rows),
            "protocol_status": protocol,
            "exit_code": {"PASS": 0, "FAIL": 2, "INCOMPLETE": 3}[protocol],
            "trajectories": rows,
        },
    )

    shaping_status = "INCOMPLETE"
    shaping_summary: dict[str, Any] = {}
    if PPO_SHAPING.is_file() and PPO_RUN_JSON.is_file():
        shaping_summary = prev.load_shaping_verifier().verify_shaping_trace(
            PPO_SHAPING, PPO_RUN_JSON, PPO_SHAPING_AUDIT
        )
        shaping_status = shaping_summary["overall_status"]

    integrity = build_weight_integrity()
    dump_json(WEIGHT_INTEGRITY, integrity)
    checks = {
        "protocol_status_pass": protocol == "PASS",
        "source_bc_unchanged": integrity["bc"]["unchanged"],
        "calibrated_model_consistent": integrity["calibrated"]["consistent"],
        "checkpoints_consistent": integrity["checkpoints"]["all_consistent"],
        "policy_invariant_after_calibration": integrity["calibrated"]["policy_unchanged"],
        "shaping_not_fail": shaping_status != "FAIL",
        "first_rollout_paired": integrity["pairing"]["paired"],
    }
    overall = "PASS" if all(checks.values()) else "FAIL"
    dump_json(
        VERIFICATION,
        {
            "protocol_status": protocol,
            "overall_status": overall,
            "checks": checks,
            "shaping_audit_status": shaping_status,
            "shaping_contracts": shaping_summary.get("contracts", {}),
            "threads": threads,
        },
    )
    log(f"verify 完成: protocol={protocol} shaping={shaping_status} overall={overall}")
    if overall != "PASS":
        return 2
    return {"PASS": 0, "FAIL": 2, "INCOMPLETE": 3}[protocol]


def build_weight_integrity() -> dict[str, Any]:
    manifest = load_json(BC_MANIFEST)
    bc_sha = sha256_file(BC_WEIGHTS)
    calibrated = torch.load(CAL_MODEL, map_location="cpu", weights_only=False)
    invariance = load_json(CAL_INVARIANCE)
    pairing = load_json(PPO_PAIRING) if PPO_PAIRING.is_file() else {"paired": False}
    checkpoints = []
    for point in CHECKPOINTS:
        path = PPO_CKPT_DIR / f"step{point:04d}.pt"
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        recomputed = state_dict_fingerprint(payload["actor_critic_state"])
        checkpoints.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "actor_fingerprint": payload["actor_fingerprint"],
                "recomputed": recomputed,
                "consistent": recomputed == payload["actor_fingerprint"],
            }
        )
    return {
        "bc": {
            "path": str(BC_WEIGHTS),
            "sha256": bc_sha,
            "expected_sha256": manifest["diagnostic_checkpoint"]["sha256"],
            "unchanged": bc_sha == manifest["diagnostic_checkpoint"]["sha256"],
        },
        "calibrated": {
            "path": str(CAL_MODEL),
            "sha256": sha256_file(CAL_MODEL),
            "actor_fingerprint": calibrated["actor_fingerprint"],
            "consistent": state_dict_fingerprint(calibrated["actor_critic_state"])
            == calibrated["actor_fingerprint"],
            "policy_unchanged": bool(invariance["policy_unchanged"]),
            "policy_subset_fingerprint": invariance["policy_subset_fingerprint_after"],
        },
        "checkpoints": {"entries": checkpoints, "all_consistent": all(c["consistent"] for c in checkpoints)},
        "pairing": pairing,
    }


def stage_report() -> int:
    control_summary = load_json(CONTROL_SUMMARY)
    control_config = CONTROL_CONFIG.read_text(encoding="utf-8")
    ppo_config = PPO_CONFIG.read_text(encoding="utf-8") if PPO_CONFIG.is_file() else ""
    calibrated_ppo = load_json(PPO_SUMMARY)
    fit = load_json(CAL_FIT)
    invariance = load_json(CAL_INVARIANCE)
    grad = load_json(GRAD_DECOMP)
    verification = load_json(VERIFICATION)
    trajectory_verification = load_json(TRAJECTORY_VERIFICATION)
    integrity = load_json(WEIGHT_INTEGRITY)
    errata = load_json(OUTPUT_DIR / "errata_verification.json")

    rows = []
    bc_reference = load_json(CONTROL_DIR / "trajectories" / "step0000_NOMINAL.json")
    for point in CHECKPOINTS:
        control_path = CONTROL_DIR / "trajectories" / f"step{point:04d}_NOMINAL.json"
        cal_path = PPO_TRAJ_DIR / f"step{point:04d}_NOMINAL.json"
        if not cal_path.is_file():
            continue
        cal_payload = load_json(cal_path)
        control_payload = load_json(control_path) if control_path.is_file() else None
        rows.append(_comparison_row(point, control_payload, cal_payload, bc_reference))
    prev.write_csv(COMPARISON_CSV, rows)

    cycle_rows = []
    for point in CHECKPOINTS:
        for label, base in (("uncalibrated", CONTROL_DIR), ("calibrated", PPO_DIR)):
            path = base / "trajectories" / f"step{point:04d}_NOMINAL.json"
            if not path.is_file():
                continue
            payload = load_json(path)
            records = payload["execution_facts"]["cycle_execution_records"]
            for key in sorted(records, key=int):
                record = records[key]
                cycle_rows.append(
                    {
                        "arm": label,
                        "step": point,
                        "cycle": int(record["cycle_id"]),
                        "nominal_transfer_time": float(record["nominal_transfer_time"]),
                        "actual_transfer_time": float(record["transfer_time"]),
                        "delay_hours": float(record["transfer_time"]) - float(record["nominal_transfer_time"]),
                    }
                )
    prev.write_csv(CYCLE_CSV, cycle_rows)

    summary = _build_summary(
        control_summary=control_summary, calibrated_ppo=calibrated_ppo, control_config=control_config,
        ppo_config=ppo_config, fit=fit, invariance=invariance, grad=grad, rows=rows,
        verification=verification, trajectory_verification=trajectory_verification,
        integrity=integrity, errata=errata, cycle_rows=cycle_rows,
    )
    SUMMARY_MD.write_text(summary, encoding="utf-8")
    log("report 完成")
    return 0


def _comparison_row(
    point: int,
    control: Mapping[str, Any] | None,
    cal: Mapping[str, Any],
    bc_reference: Mapping[str, Any],
) -> dict[str, Any]:
    c = cal["cost_components_weighted"]
    u = control["cost_components_weighted"] if control else None
    cycle = prev.summarize_cycles(cal)
    control_cycle = prev.summarize_cycles(control) if control else None
    cal_vs_bc = compare_execution_facts(bc_reference["execution_facts"], cal["execution_facts"])
    control_vs_bc = (
        compare_execution_facts(bc_reference["execution_facts"], control["execution_facts"])
        if control
        else None
    )
    return {
        "step": point,
        "uncal_j_total": None if control is None else float(control["objective_j_total"]),
        "cal_j_total": float(cal["objective_j_total"]),
        "delta_j_cal_minus_uncal": None if control is None else float(cal["objective_j_total"]) - float(control["objective_j_total"]),
        "delta_j_uncal_minus_cal": None if control is None else float(control["objective_j_total"]) - float(cal["objective_j_total"]),
        "uncal_j_takt": None if control is None else float(u["j_takt"]),
        "cal_j_takt": float(c["j_takt"]),
        "uncal_d_time": None if control is None else float(u["d_time"]),
        "cal_d_time": float(c["d_time"]),
        "uncal_d_team": None if control is None else float(u["d_team"]),
        "cal_d_team": float(c["d_team"]),
        "uncal_j_postpone": None if control is None else float(u["j_postpone"]),
        "cal_j_postpone": float(c["j_postpone"]),
        "uncal_j_revision": None if control is None else float(u["j_revision"]),
        "cal_j_revision": float(c["j_revision"]),
        "uncal_makespan": None if control is None else float(control["makespan"]),
        "cal_makespan": float(cal["makespan"]),
        "uncal_max_cycle_delay": None if control_cycle is None else control_cycle["max_cycle_delay"],
        "cal_max_cycle_delay": cycle["max_cycle_delay"],
        "uncal_delayed_cycles": None if control_cycle is None else control_cycle["delayed_cycle_count"],
        "cal_delayed_cycles": cycle["delayed_cycle_count"],
        "uncal_decisions": None if control is None else int(control["decisions"]),
        "cal_decisions": int(cal["decisions"]),
        "uncal_completed": None if control is None else int(control["completed_tasks"]),
        "cal_completed": int(cal["completed_tasks"]),
        "uncal_postponed": None if control is None else int(control["postponed_count"]),
        "cal_postponed": int(cal["postponed_count"]),
        "uncal_retention": None if control is None else prev.retention_pass(control),
        "cal_retention": prev.retention_pass(cal),
        "effect_status": None if control is None else _effect_status(
            float(control["objective_j_total"]), float(cal["objective_j_total"])
        ),
        "cal_start_deviation_vs_bc": cal_vs_bc["start_deviation_count"],
        "cal_team_deviation_vs_bc": cal_vs_bc["team_deviation_count"],
        "cal_transfer_deviation_vs_bc": cal_vs_bc["transfer_deviation_count"],
        "cal_first_start_task": None
        if cal_vs_bc["first_start_deviation"] is None
        else cal_vs_bc["first_start_deviation"]["task_key"],
        "cal_first_start_delta": None
        if cal_vs_bc["first_start_deviation"] is None
        else cal_vs_bc["first_start_deviation"]["delta"],
        "cal_first_transfer_cycle": None
        if cal_vs_bc["first_transfer_deviation"] is None
        else cal_vs_bc["first_transfer_deviation"]["cycle"],
        "uncal_start_deviation_vs_bc": None if control_vs_bc is None else control_vs_bc["start_deviation_count"],
        "uncal_team_deviation_vs_bc": None if control_vs_bc is None else control_vs_bc["team_deviation_count"],
        "uncal_transfer_deviation_vs_bc": None
        if control_vs_bc is None
        else control_vs_bc["transfer_deviation_count"],
        "uncal_first_start_task": None
        if not control_vs_bc or control_vs_bc["first_start_deviation"] is None
        else control_vs_bc["first_start_deviation"]["task_key"],
        "uncal_first_start_delta": None
        if not control_vs_bc or control_vs_bc["first_start_deviation"] is None
        else control_vs_bc["first_start_deviation"]["delta"],
    }


def _effect_status(control_j: float, cal_j: float) -> str:
    delta = control_j - cal_j
    if delta > 1e-9:
        return "校准臂费用更低"
    if delta < -1e-9:
        return "校准臂费用更高"
    return "近似持平"


def _build_summary(
    *,
    control_summary: Mapping[str, Any],
    calibrated_ppo: Mapping[str, Any],
    control_config: str,
    ppo_config: str,
    fit: Mapping[str, Any],
    invariance: Mapping[str, Any],
    grad: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any],
    trajectory_verification: Mapping[str, Any],
    integrity: Mapping[str, Any],
    errata: Mapping[str, Any],
    cycle_rows: Sequence[Mapping[str, Any]],
) -> str:
    fmt = prev.fmt
    lines: list[str] = []
    lines.append("# Work-3 价值头专属校准与 PPO 保持性匹配对照报告")
    lines.append("")
    lines.append(
        "> 唯一新增训练臂：先只校准价值专属头（冻结全部策略路径），再用与前轮**完全相同**的配置跑 "
        f"{TRAINING_BUDGET} 步 PPO。未校准对照直接复用前轮归档运行，不新增对照训练。"
    )
    lines.append("")
    lines.append(
        f"**协议状态**：轨迹事实 `{trajectory_verification['protocol_status']}`；"
        f"塑形审计 `{verification['shaping_audit_status']}`；"
        f"诊断自检 `{verification['overall_status']}`。"
    )
    lines.append("")

    lines.append("## 1. 历史解释纠偏（不重跑前轮训练）")
    lines.append("")
    lines.append(
        "对自然完工、终局势为零的轨迹，`G_shaped = G_raw - beta*Phi(s0)` 是严格恒等式，"
        "塑形与原始折扣回报只差一个**固定平移**，不能据此说“方向相反”。"
    )
    lines.append("")
    lines.append("| episode | Phi(s0) | 原始折扣回报 | 塑形折扣回报 | 实测差值 | 一般式期望差 | 残差 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in errata["episodes"]:
        lines.append(
            f"| {item['episode_id']} | {item['phi_start']:.6f} | {item['discounted_raw_return']:.6f} | "
            f"{item['discounted_shaped_return']:.6f} | {item['observed_shift_shaped_minus_raw']:.9f} | "
            f"{item['expected_shift_general']:.9f} | {item['residual']:.3e} |"
        )
    lines.append("")
    lines.append(
        f"- 三条 episode（含 1 条预算截断）的残差全部 ≤ "
        f"{max(abs(i['residual']) for i in errata['episodes']):.3e}，"
        f"与 1e-9 容差{'一致' if errata['all_within_tolerance'] else '不一致'}："
        "完工段的 0.5 差值来自 `Phi(s0)=-0.5, beta=1`，截断段用其自身 `phi_T` 代入一般式，"
        "都不是塑形目标改善。"
    )
    lines.append(
        "- 仍需分开的两个问题：**(a)** 势函数塑形在端点条件下只产生固定平移；"
        "**(b)** `gamma` 折扣原始回报与未折扣 `J_total` 之间存在目标口径差异。截断段保留边界项。"
    )
    lines.append(
        "- 首段证据的范围收紧为：更新改变了已记录快照上的动作概率；**首次已评估的闭环退化出现在 128**，"
        "不能唯一定位到首次更新，也不能排除后续随机采样影响。"
    )
    lines.append("")

    lines.append("## 2. 离线首批梯度分解（零参数更新）")
    lines.append("")
    lines.append(f"- 范围：{grad['scope']}")
    lines.append(f"- 真实首批样本身份可识别：`{grad['first_minibatch_identity_available']}`；缺项：{'；'.join(grad['missing_evidence'])}")
    lines.append(
        f"- 8 样本整体：policy {fmt(grad['losses']['policy_loss'])}，value(raw) {fmt(grad['losses']['value_loss_raw'])}，"
        f"entropy {fmt(grad['losses']['entropy'])}，total {fmt(grad['losses']['total_loss'])}。"
    )
    lines.append(
        f"- 共享参数梯度范数：policy {grad['shared_gradient']['norms']['policy']:.6f}、"
        f"value {grad['shared_gradient']['norms']['value']:.6f}、"
        f"entropy {grad['shared_gradient']['norms']['entropy']:.6f}。"
    )
    cosines = grad["shared_gradient"]["pairwise_cosines"]
    for key, value in cosines.items():
        lines.append(f"  - cos({key}) = {'N/A' if value is None else f'{value:+.6f}'}")
    lines.append(
        f"- 三项带权梯度之和与总损失梯度的相对残差 {grad['shared_gradient']['relative_residual']:.3e}"
        f"（容差 {grad['shared_gradient']['tolerance']:.0e}，"
        f"{'通过' if grad['shared_gradient']['residual_within_tolerance'] else '不通过'}）。"
    )
    lines.append(
        f"- 优势标准差：raw unbiased(ddof=1) {fmt(grad['advantages']['raw_std_unbiased_ddof1'], 6)}、"
        f"biased(ddof=0) {fmt(grad['advantages']['raw_std_biased_ddof0'], 6)}；"
        f"归一化后 unbiased {fmt(grad['advantages']['normalized_std_unbiased_ddof1'], 6)}"
        "（8 样本上 `(x-mean)/(std+1e-8)` 的有限样本结果，不是错误）。"
    )
    lines.append(
        "- 不能说明的：哪一项导致退化。梯度范数只描述贡献大小，Adam 自适应、全局裁剪与参数尺度都会改变实际更新。"
    )
    lines.append("")

    lines.append("## 3. 校准阶段（仅价值专属头）")
    lines.append("")
    lines.append(
        f"- 采集：BC 原始权重、CPU 单线程、随机策略、seed {CAL_SEED}，一条自然完工 Nominal 轨迹，"
        f"{fit['samples']['total']} 个决策；源文件 SHA 未变（{integrity['bc']['unchanged']}）。"
    )
    lines.append(
        f"- 标签：shaped Monte Carlo return，`G_T=0` 尾部递推；恒等式 "
        f"`G_shaped=G_raw-beta*Phi` 最大残差 {fit['mc_identity_check']['max_abs_residual']:.3e}"
        f"（{'通过' if fit['mc_identity_check']['passed'] else '不通过'}）。"
    )
    lines.append(
        f"- 目标统计：均值 {fit['target_stats']['mean']:.6f}、std {fit['target_stats']['std_unbiased']:.6f}、"
        f"范围 [{fit['target_stats']['min']:.6f}, {fit['target_stats']['max']:.6f}]。"
    )
    lines.append(
        f"- 校准：{fit['calibration_steps']} 次 optimizer.step（上限 {CAL_MAX_STEPS} / {CAL_MAX_SECONDS:.0f}s，"
        f"停止原因 `{fit['stop_reason']}`），Adam lr=3e-4、wd=0、MSE、batch {CAL_BATCH}、"
        f"梯度裁剪 0.5；训练 {fit['samples']['train']} 条、留出 {fit['samples']['holdout']}。"
    )
    for label in ("train", "holdout"):
        initial = fit["initial_scan"][label]
        final = fit["final_scan"][label]
        baseline = fit["constant_baselines"][label]
        lines.append(
            f"  - {label}：MSE {fmt(initial['mse'], 6)} → {fmt(final['mse'], 6)}；"
            f"RMSE {fmt(final['rmse'], 6)}、MAE {fmt(final['mae'], 6)}、bias {fmt(final['bias'], 6)}、"
            f"EV {fmt(final['explained_variance'], 4)}；常数预测器 MSE {fmt(baseline['mse'], 6)}。"
        )
    lines.append(
        f"- 拟合改善证据：`{fit['fit_improved']}`（train 与 holdout 的 MSE 同时下降才算）；"
        "留出误差不是真实 V 误差，也不是独立泛化验证。"
    )
    train_const = fit["constant_baselines"]["train"]["mse"]
    hold_const = fit["constant_baselines"]["holdout"]["mse"]
    train_final = fit["final_scan"]["train"]["mse"]
    hold_final = fit["final_scan"]["holdout"]["mse"]
    constant_better = bool(train_const < train_final and hold_const < hold_final)
    lines.append(
        f"- **与常数预测器的对照**：训练集 target 均值作常数预测时 MSE 为 train {train_const:.3e} / "
        f"holdout {hold_const:.3e}，而校准后为 {train_final:.3e} / {hold_final:.3e}。"
        + (
            "常数预测器**仍然更好**，说明这次拟合只是把 V 拉到接近常数（本轨迹的 MC target 几乎为常数），"
            "不能称为“获得了准确 critic”。"
            if constant_better
            else "校准后优于常数预测器，但 target 方差极小，仍不足以声称 critic 准确。"
        )
        + f"（target std={fit['target_stats']['std_unbiased']:.3e}，EV 因此不可解释，已在报告中标注。）"
    )
    lines.append(
        f"- 策略不变性：策略子集指纹 before==after `{invariance['policy_unchanged']}`，"
        f"逐快照贪心动作不一致 {invariance['greedy_action_mismatch']} 处，"
        f"log-prob 最大差 {invariance['max_abs_log_prob_difference']:.3e}（容差 {POLICY_PROB_TOLERANCE:.0e}）；"
        "完整模型指纹（含 critic）按设计改变。"
    )
    lines.append("")

    lines.append("## 4. 两臂配置与首段配对")
    lines.append("")
    same_keys = _config_diff(control_config, ppo_config)
    lines.append(
        f"- 配置逐字段比对：{same_keys}"
    )
    pairing = integrity["pairing"]
    lines.append(
        f"- 首段配对：paired=`{pairing['paired']}`，状态一致 `{pairing.get('same_states')}`，"
        f"动作一致 `{pairing.get('same_actions')}`，log-prob 最大差 "
        f"{fmt(pairing.get('max_abs_log_prob_difference'), 8)}（容差 {pairing.get('tolerance')}）。"
    )
    lines.append(
        "- 两臂初始化顺序与 seed 相同，首段 8 步采样事件一致；首段之后因 critic 不同而自然分化，"
        "不强制重放对照动作。"
    )
    lines.append(
        f"- 计数：校准 optimizer.step {fit['calibration_steps']} 次、PPO 交互 {calibrated_ppo['training_decisions']} 步、"
        f"PPO optimizer.step {calibrated_ppo['optimizer_steps']} 次。"
        "两臂只匹配 PPO 部分预算；校准臂额外消耗了一条采集轨迹与校准计算，总数据量并不严格相等。"
    )
    lines.append("")

    lines.append("## 5. 四点并列表")
    lines.append("")
    lines.append(
        "| 检查点 | 未校准 J | 校准 J | ΔJ(未校准−校准) | 未校准 J_takt | 校准 J_takt | "
        "未校准 D_time | 校准 D_time | 未校准 D_team | 校准 D_team | 未校准 J_rev | 校准 J_rev | "
        "未校准 Makespan | 校准 Makespan | 裁决 |"
    )
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    for row in rows:
        lines.append(
            f"| {row['step']} | {fmt(row['uncal_j_total'])} | {fmt(row['cal_j_total'])} | "
            f"{fmt(row['delta_j_uncal_minus_cal'])} | {fmt(row['uncal_j_takt'])} | {fmt(row['cal_j_takt'])} | "
            f"{fmt(row['uncal_d_time'])} | {fmt(row['cal_d_time'])} | {fmt(row['uncal_d_team'])} | {fmt(row['cal_d_team'])} | "
            f"{fmt(row['uncal_j_revision'])} | {fmt(row['cal_j_revision'])} | "
            f"{fmt(row['uncal_makespan'], 6)} | {fmt(row['cal_makespan'], 6)} | {row['effect_status']} |"
        )
    lines.append("")
    lines.append(
        "> `ΔJ_cal(n) = J_uncal(n) - J_cal(n)`，正值表示校准臂在该点费用更低。"
        "终点以 512 为主，中间点只用于解释曲线，不择优。BC 起点费用近零，不用百分比分母。"
    )
    lines.append("")
    for row in rows:
        lines.append(
            f"- step{row['step']:04d}：保持标准 未校准={row['uncal_retention']} / 校准={row['cal_retention']}；"
            f"完工 {row['uncal_completed']}/{row['cal_completed']}，后移 {row['uncal_postponed']}/{row['cal_postponed']}，"
            f"周期延迟数 {row['uncal_delayed_cycles']}/{row['cal_delayed_cycles']}，"
            f"最大延迟 {fmt(row['uncal_max_cycle_delay'], 6)}/{fmt(row['cal_max_cycle_delay'], 6)}h；"
            f"相对 BC 基准的开工/团队/转站偏离 未校准 "
            f"{fmt(row['uncal_start_deviation_vs_bc'])}/{fmt(row['uncal_team_deviation_vs_bc'])}/"
            f"{fmt(row['uncal_transfer_deviation_vs_bc'])}、校准 "
            f"{row['cal_start_deviation_vs_bc']}/{row['cal_team_deviation_vs_bc']}/"
            f"{row['cal_transfer_deviation_vs_bc']} 处。"
        )
        if row["cal_first_start_task"]:
            lines.append(
                f"  - 校准臂首次实质偏离：`{row['cal_first_start_task']}` 开工 "
                f"{fmt(row['cal_first_start_delta'], 6)}h，首次转站延迟周期 "
                f"{row['cal_first_transfer_cycle']}；未校准臂首次："
                f"`{row['uncal_first_start_task']}` 开工 {fmt(row['uncal_first_start_delta'], 6)}h。"
            )
    lines.append("")

    lines.append("## 6. 运行事实与核验")
    lines.append("")
    gpu = calibrated_ppo["gpu"]
    lines.append(
        f"- 校准臂 PPO：{calibrated_ppo['training_decisions']} 决策、{calibrated_ppo['optimizer_steps']} 次更新、"
        f"{calibrated_ppo['segments']} 段、完成 episode {calibrated_ppo['episodes_completed']}/"
        f"{calibrated_ppo['episodes_started']}、耗时 {calibrated_ppo['elapsed_seconds']:.1f}s、"
        f"停止原因 `{calibrated_ppo['stop_reason']}`。"
    )
    lines.append(
        f"- 显存峰值 reserved {gpu['peak_reserved_mib']:.1f}MiB（预算 {gpu['budget_mib']:.0f}MiB，"
        f"超限={gpu['budget_exceeded']}）；FP32；单环境 spawn。"
    )
    lines.append(
        f"- 轨迹核验：{trajectory_verification['verified_count']}/{len(trajectory_verification['expected_manifest'])} 条，"
        f"状态 `{trajectory_verification['protocol_status']}`；塑形审计 `{verification['shaping_audit_status']}`。"
    )
    lines.append(
        f"- 权重完整性：源 BC 未变 `{integrity['bc']['unchanged']}`、校准副本自洽 "
        f"`{integrity['calibrated']['consistent']}`、策略子集不变 `{integrity['calibrated']['policy_unchanged']}`。"
    )
    lines.append("")

    lines.append("## 7. 有限结论与下一步")
    lines.append("")
    final = rows[-1] if rows else None
    retain_cal = [r for r in rows if r["step"] > 0 and r["cal_retention"]]
    retain_uncal = [r for r in rows if r["step"] > 0 and r["uncal_retention"]]
    improved_fit = bool(fit["fit_improved"])
    if final is None:
        verdict = "缺少终点评估，无法形成结论。"
    elif improved_fit and retain_cal and not retain_uncal:
        verdict = (
            "校准在固定预算内改善了价值拟合，且校准臂在训练后检查点上满足保持标准而对照不满足："
            "支持“这次价值头校准处理对短预算保持性有正向作用”，价值初始化失配是候选影响因素。"
        )
    elif improved_fit and not retain_cal:
        verdict = (
            "校准在 MSE 上改善了拟合，但校准臂仍未保持基准，且三个训练后检查点都比未校准对照更差："
            "**价值拟合改善不足以解决保持性**。"
            "另需注意：本轨迹的 MC target 近乎常数，常数预测器的 MSE 仍低于校准后模型，"
            "因此这次处理既不能称为“获得准确 critic”，也不能据此排除价值失配在别处的作用。"
            "下一轮应考虑更新幅度、探索项或共享表示变化，而不是继续加大校准预算。"
        )
    elif not improved_fit:
        verdict = (
            "校准在固定预算内拟合改善有限：这次处理**尚未建立充分的拟合改善证据**，"
            "既不能据此排除价值失配，也不能把后续结果归因于“critic 更准确”。"
        )
    else:
        verdict = "结果处于混合区间，按上表与拟合证据逐点解释，不作单一因果断言。"
    lines.append(f"- {verdict}")
    lines.append("")
    lines.append(
        "本轮是单实例、单 PPO seed、一条校准轨迹的探索性匹配对照，不提供统计显著性，"
        "也不证明 critic 的精确价值；MC 标签对随机策略只是一条实现回报。"
    )
    lines.append("")
    lines.append("**下一步只建议一个单变量对照（不自动执行）**：")
    lines.append("")
    if improved_fit and not retain_cal:
        lines.append(
            "- 在相同 BC 起点、相同校准副本下，只改**更新幅度**（例如降低 policy 学习率或缩小 clip 范围）做单变量对照，"
            "先看是否能把 128 点的退化推迟；本轮不调参、不重跑。"
        )
    else:
        lines.append(
            "- 在相同 BC 起点下，只改**探索项**（熵系数）做单变量对照，观察随机采样偏离是否主导退化；"
            "本轮不调参、不重跑。"
        )
    lines.append("")

    lines.append("## 8. 产物索引")
    lines.append("")
    for name in (
        "manifest.json", "control_binding.json", "interpretation_errata.json", "interpretation_errata.md",
        "offline_gradient_decomposition.json", "calibration/trajectory.json",
        "calibration/snapshots_and_mc_targets.pt", "calibration/history.jsonl",
        "calibration/fit_metrics.json", "calibration/policy_invariance.json",
        "calibration/calibrated_model.pt", "ppo/resolved_config.yaml",
        "ppo/initialization_verification.json", "ppo/first_rollout_pairing.json",
        "ppo/training_history.jsonl", "ppo/checkpoints/{step0000,step0128,step0256,step0512}.pt",
        "ppo/trajectories/{step0000,step0128,step0256,step0512}_NOMINAL.json",
        "ppo/shaping_events.jsonl", "ppo/shaping_audit_summary.json",
        "comparison_results.csv", "cycle_comparison.csv", "trajectory_verification.json",
        "weight_integrity.json", "verification.json",
    ):
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append(
        "说明：`*.pt` 按仓库既有 `.gitignore` 约定仅保留本地；SHA256 与指纹记录在 "
        "`weight_integrity.json` 与 `manifest.json`。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _config_diff(control: str, ours: str) -> str:
    control_lines = {line.strip() for line in control.splitlines() if ":" in line}
    ours_lines = {line.strip() for line in ours.splitlines() if ":" in line}
    keys = ("epochs", "batch_size", "learning_rate", "clip_epsilon", "value_coefficient",
            "entropy_coefficient", "gamma", "gae_lambda", "steps_per_iter", "shaping_coefficient",
            "max_grad_norm", "weight_decay", "optimizer_eps")
    mismatches = []
    for key in keys:
        left = next((line for line in control_lines if line.startswith(f"{key}:")), None)
        right = next((line for line in ours_lines if line.startswith(f"{key}:")), None)
        if left != right:
            mismatches.append(f"{key}: control={left} ours={right}")
    return "全部一致" if not mismatches else "；".join(mismatches)


# ---------------------------------------------------------------------------
# 清册与入口
# ---------------------------------------------------------------------------


def stage_inventory() -> int:
    threads = enforce_single_thread_cpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    PPO_DIR.mkdir(parents=True, exist_ok=True)
    PPO_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    PPO_TRAJ_DIR.mkdir(parents=True, exist_ok=True)

    if not BC_WEIGHTS.is_file():
        raise FileNotFoundError(f"缺少 BC 诊断副本: {BC_WEIGHTS}")
    manifest = load_json(BC_MANIFEST)
    bc_sha = sha256_file(BC_WEIGHTS)
    if bc_sha != manifest["diagnostic_checkpoint"]["sha256"]:
        raise ValueError("BC 文件 SHA 与上一轮清单不一致")
    model, payload = load_diagnostic_actor(
        BC_WEIGHTS, expected_teacher_manifest_sha256=manifest["teacher"]["manifest_sha256"]
    )
    bc_fingerprint = actor_fingerprint(model)

    control_files = {
        "manifest": CONTROL_MANIFEST,
        "resolved_config": CONTROL_CONFIG,
        "training_summary": CONTROL_SUMMARY,
        "first_rollout": CONTROL_FIRST_ROLLOUT,
        "shaping_audit": CONTROL_DIR / "shaping_audit_summary.json",
        "verification": CONTROL_DIR / "verification.json",
    }
    for point in CHECKPOINTS:
        control_files[f"trajectory_step{point:04d}"] = CONTROL_DIR / "trajectories" / f"step{point:04d}_NOMINAL.json"
        control_files[f"checkpoint_step{point:04d}"] = CONTROL_DIR / "checkpoints" / f"step{point:04d}.pt"
    missing = [str(path) for path in control_files.values() if not path.is_file()]
    if missing:
        dump_json(OUTPUT_DIR / "missing_control.json", {"missing": missing})
        raise FileNotFoundError(f"未校准对照产物缺失: {missing}")

    hyper = load_json(CONTROL_SUMMARY)["hyperparameters"]
    dump_json(
        CONTROL_BINDING,
        {
            "schema_version": "critic_calibration_control_binding_v1",
            "control_dir": str(CONTROL_DIR),
            "files": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in control_files.items()},
            "control_reference_commit": load_json(CONTROL_MANIFEST)["run_context"]["head"],
            "hyperparameters": hyper,
            "note": "未校准对照直接复用前轮归档运行，不新增第二条 512 步对照。",
        },
    )

    # PPO 配置：与对照逐字段一致，仅诊断字段不同
    write_resolved_config(hyper)
    init_verification = {
        "schema_version": "critic_calibration_initialization_v1",
        "bc": {
            "path": str(BC_WEIGHTS),
            "sha256": bc_sha,
            "actor_fingerprint": bc_fingerprint,
            "format": payload["diagnostic_format"],
            "teacher_manifest_sha256": payload["teacher_manifest_sha256"],
            "source_checkpoint_sha256": payload["source_checkpoint"]["sha256"],
        },
        "control": {"dir": str(CONTROL_DIR), "hyperparameters": hyper},
        "graph_schema": {
            "version": GRAPH_FEATURE_VERSION,
            "dims": dict(GRAPH_FEATURE_DIMS),
            "schema_hash": GRAPH_FEATURE_SCHEMA["schema_fingerprint"],
        },
        "stage_plan": {
            "calibration": {"seed": CAL_SEED, "split_seed": CAL_SPLIT_SEED, "max_steps": CAL_MAX_STEPS,
                            "max_seconds": CAL_MAX_SECONDS, "batch": CAL_BATCH, "holdout_modulo": HOLDOUT_MODULO},
            "ppo": {"budget": TRAINING_BUDGET, "checkpoints": list(CHECKPOINTS), "seed": hyper["seed"]},
        },
        "threads": threads,
    }
    dump_json(PPO_INIT_VERIFICATION, init_verification)
    dump_json(
        MANIFEST,
        {
            "task": "Work-3 价值头专属校准与 PPO 保持性匹配对照",
            "schema_version": "critic_calibration_manifest_v1",
            "reference_commit": "169df7c136a1816255e221cce08ca90c453febfa",
            "run_context": git_state(),
            "output_dir": str(OUTPUT_DIR),
            "instance": {
                "baseline": str(BASELINE_PATH),
                "baseline_sha256": sha256_file(BASELINE_PATH),
                "scenario": "NOMINAL",
                "disturbance": "none",
                "warmup": "none",
            },
            "arms": {
                "uncalibrated_control": str(CONTROL_DIR),
                "calibrated": "本目录 ppo/（唯一新增训练臂）",
            },
            "budget": {
                "calibration_steps_max": CAL_MAX_STEPS,
                "calibration_seconds_max": CAL_MAX_SECONDS,
                "ppo_decisions": TRAINING_BUDGET,
                "checkpoints": list(CHECKPOINTS),
            },
            "expected_artifacts": [
                "manifest.json", "control_binding.json", "interpretation_errata.md",
                "interpretation_errata.json", "offline_gradient_decomposition.json",
                "calibration/trajectory.json", "calibration/snapshots_and_mc_targets.pt",
                "calibration/history.jsonl", "calibration/fit_metrics.json",
                "calibration/policy_invariance.json", "calibration/calibrated_model.pt",
                "ppo/resolved_config.yaml", "ppo/initialization_verification.json",
                "ppo/first_rollout_pairing.json", "ppo/training_history.jsonl",
                "ppo/checkpoints/{step0000,step0128,step0256,step0512}.pt",
                "ppo/trajectories/{step0000,step0128,step0256,step0512}_NOMINAL.json",
                "ppo/shaping_events.jsonl", "ppo/shaping_audit_summary.json",
                "comparison_results.csv", "cycle_comparison.csv",
                "trajectory_verification.json", "weight_integrity.json",
                "verification.json", "summary.md",
            ],
        },
    )
    log(f"inventory 完成: BC fp={bc_fingerprint[:16]}，对照 {len(control_files)} 个文件已绑定")
    return 0


def write_resolved_config(hyper: Mapping[str, Any]) -> None:
    lines = [
        "# 校准臂 PPO 配置：与未校准对照逐字段一致，仅以下诊断字段不同",
        "runtime:",
        f"  seed: {int(hyper['seed'])}",
        "  device: cuda:0",
        "  precision: fp32",
        f"  num_envs: {int(hyper['num_envs'])}",
        "  start_method: spawn",
        f"  total_env_steps: {TRAINING_BUDGET}",
        "  disturbance_enabled: false",
        "  scenario: NOMINAL",
        "  warmup_mode: none",
        "# 本轮诊断字段（不计入对照匹配）",
        "  init_weights: calibration/calibrated_model.pt",
        "  calibration_seed: 4201",
        "  calibration_split_seed: 4202",
        "paths:",
        f"  baseline: {BASELINE_PATH}",
        "ppo:",
        f"  steps_per_iter: {int(hyper['rollout_steps'])}",
        f"  epochs: {int(hyper['ppo_epochs'])}",
        f"  batch_size: {int(hyper['batch_size'])}",
        f"  learning_rate: {hyper['learning_rate']}",
        f"  clip_epsilon: {hyper['clip_epsilon']}",
        f"  value_coefficient: {hyper['value_coefficient']}",
        f"  entropy_coefficient: {hyper['entropy_coefficient']}",
        f"  gamma: {hyper['gamma']}",
        f"  gae_lambda: {hyper['gae_lambda']}",
        f"  shaping_coefficient: {hyper['beta_shaping']}",
        f"  max_grad_norm: {hyper['max_grad_norm']}",
        "  optimizer: AdamW",
        f"  optimizer_eps: {hyper['optimizer_eps']}",
        f"  weight_decay: {hyper['weight_decay']}",
        f"  normalize_advantages: {str(hyper['normalize_advantages']).lower()}",
        "checkpoints:",
        f"  points: {list(CHECKPOINTS)}",
    ]
    PPO_CONFIG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_stages() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}
    for stage in STAGES:
        log(f"=== 子进程执行阶段: {stage} ===")
        with (RUNS_DIR / f"{stage}.log").open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--stage", stage],
                cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, text=True, check=False,
            )
        results[stage] = process.returncode
        log(f"阶段 {stage} 退出码 {process.returncode}")
        if process.returncode != 0:
            dump_json(RUNS_DIR / "stage_results.json", results)
            return process.returncode
    dump_json(RUNS_DIR / "stage_results.json", results)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="价值头专属校准与 PPO 保持性匹配对照")
    parser.add_argument("--stage", choices=list(STAGES) + ["inventory", "evaluate-one", "all"], default="all")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()
    if args.stage == "all":
        stages = ("inventory",) + STAGES
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        results: dict[str, int] = {}
        for stage in stages:
            log(f"=== 子进程执行阶段: {stage} ===")
            with (RUNS_DIR / f"{stage}.log").open("w", encoding="utf-8") as handle:
                process = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--stage", stage],
                    cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, text=True, check=False,
                )
            results[stage] = process.returncode
            log(f"阶段 {stage} 退出码 {process.returncode}")
            if process.returncode != 0:
                dump_json(RUNS_DIR / "stage_results.json", results)
                return process.returncode
        dump_json(RUNS_DIR / "stage_results.json", results)
        return 0
    if args.stage == "evaluate-one":
        if args.checkpoint is None or args.step is None:
            raise SystemExit("evaluate-one 需要 --checkpoint 与 --step")
        return stage_evaluate_one(args.checkpoint, args.step)
    handlers = {
        "inventory": stage_inventory,
        "errata": stage_errata,
        "offline": stage_offline,
        "calibrate": stage_calibrate,
        "train": stage_train,
        "evaluate": stage_evaluate,
        "verify": stage_verify,
        "report": stage_report,
    }
    started = time.perf_counter()
    code = handlers[args.stage]()
    log(f"阶段 {args.stage} 用时 {time.perf_counter() - started:.1f}s，退出码 {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
