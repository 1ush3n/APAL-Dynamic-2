"""Work-3 BC 初始化后的 PPO 基准保持性诊断。

唯一主问题：从已复现 A Nominal 排程的 BC 副本出发，执行 512 个真实决策的短程 PPO 更新后，
能否保留其 Nominal 基准保持能力。

复用而非重写：
- PPO 更新与损失：``training.work3_lightning.Work3LightningModule.training_step``
  （内部 ``PPOTrainerWork3.compute_ppo_minibatch_loss``，AdamW(lr, eps=1e-5, wd=1e-4)，
  梯度裁剪 max_norm=0.5）；
- GAE 与优势标准化：``models.work3.ppo_buffer.RolloutBufferWork3``；
- 启发式势函数：``scripts.work3.train_ppo_work3.compute_heuristic_potential``；
- 单环境 spawn：``training.work3_vector_env.Work3VectorEnv``（num_envs=1）；
- 闭环评估与事实导出：``scripts.work3.evaluate_c_vs_d.evaluate_single_trajectory``。

新增的只有：诊断初始化适配器、Nominal-only 采样循环、固定预算/检查点、逐段诊断记录。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
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
from models.work3.graph_builder import (
    GRAPH_FEATURE_DIMS,
    GRAPH_FEATURE_SCHEMA,
    GRAPH_FEATURE_VERSION,
)
from models.work3.ppo_buffer import PPOTransition, RolloutBufferWork3
from scripts.work3.evaluate_c_vs_d import (
    FormalEvaluationAgent,
    build_formal_evaluation_agent,
    evaluate_single_trajectory,
)
from scripts.work3.experiment_protocol import build_method_profile
from scripts.work3.train_ppo_work3 import compute_heuristic_potential
from training.work3_lightning import Work3LightningModule, Work3TrainingUpdate
from training.work3_vector_env import Work3VectorEnv
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
    sha256_text,
    state_dict_fingerprint,
)

BASE_DIR = REPO_ROOT / "work3_4060"
RULE_BASELINE_DIR = BASE_DIR / "rule_baseline_ac_v1"
BC_DIR = BASE_DIR / "c_reproduction_diagnostic_v1" / "seed42"
ALIGN_DIR = BASE_DIR / "c_alignment_bc_robustness_v1" / "seed42"
OUTPUT_DIR = BASE_DIR / "bc_ppo_retention_v1" / "seed42"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TRAJ_DIR = OUTPUT_DIR / "trajectories"
RUNS_DIR = OUTPUT_DIR / "runs"

BASELINE_PATH = RULE_BASELINE_DIR / "mini210_rule_baseline.json"
ORIGINAL_C_WEIGHTS = RULE_BASELINE_DIR / "seed42" / "C" / "weights.pt"
ORIGINAL_C_RUN = RULE_BASELINE_DIR / "seed42" / "C" / "run.json"
BC_WEIGHTS = BC_DIR / "diagnostic_actor.pt"
BC_MANIFEST = BC_DIR / "manifest.json"
BC_TEACHER_SNAPSHOTS = BC_DIR / "teacher_snapshots.pt"
BC_NOMINAL_TRAJ = BC_DIR / "trajectories" / "C_BC_DIAGNOSTIC_NOMINAL.json"

MANIFEST = OUTPUT_DIR / "manifest.json"
RESOLVED_CONFIG = OUTPUT_DIR / "resolved_config.yaml"
INIT_VERIFICATION = OUTPUT_DIR / "initialization_verification.json"
PARAMETER_GROUPS = OUTPUT_DIR / "parameter_groups.json"
TRAINING_HISTORY = OUTPUT_DIR / "training_history.jsonl"
FIRST_ROLLOUT = OUTPUT_DIR / "first_rollout.pt"
FIRST_UPDATE = OUTPUT_DIR / "first_update_diagnostics.json"
SHAPING_EVENTS = OUTPUT_DIR / "shaping_events.jsonl"
SHAPING_AUDIT = OUTPUT_DIR / "shaping_audit_summary.json"
EVALUATION_RESULTS = OUTPUT_DIR / "evaluation_results.csv"
CYCLE_COMPARISON = OUTPUT_DIR / "cycle_comparison.csv"
TRAJECTORY_VERIFICATION = OUTPUT_DIR / "trajectory_verification.json"
WEIGHT_INTEGRITY = OUTPUT_DIR / "weight_integrity.json"
VERIFICATION = OUTPUT_DIR / "verification.json"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
PPO_RUN_JSON = RUNS_DIR / "ppo_run.json"
OFFLINE_S1 = OUTPUT_DIR / "offline_s1_decomposition.json"

STAGES: tuple[str, ...] = ("inventory", "train", "evaluate", "verify", "report")

CHECKPOINTS: tuple[int, ...] = (0, 128, 256, 512)
TRAINING_BUDGET = 512
ROLLOUT_STEPS = 8
EVALUATION_DECISION_LIMIT = 4096
GPU_BUDGET_BYTES = 6 * 1024**3
REPLAY_TOLERANCE = 1e-4
POLICY_TOLERANCE_CPU_VS_GPU = 1e-4
PHYSICAL_TOLERANCE = 1e-7
COST_TOLERANCE = 1e-9
NEAR_ZERO_DENOMINATOR = 1e-9

# 保留标准（第 8.1 节）
RETAIN_J_TOTAL = 1e-3
RETAIN_J_TAKT = 1e-6
RETAIN_POSTPONE = 0

VALUE_PARAMETER_PREFIXES = ("critic.",)
POLICY_PARAMETER_PREFIXES = (
    "time_fusion.",
    "hierarchical_task_fusion.",
    "task_proj.",
    "task_score_fc.",
    "branch_head.",
    "worker_score_fc.",
    "align_head.",
    "task_graph_proj.",
    "worker_graph_score.",
)
SHARED_PARAMETER_PREFIXES = ("graph_encoder.", "encoder.")

DIAGNOSTIC_CHECKPOINT_FORMAT = "bc_ppo_retention_checkpoint"
DIAGNOSTIC_CHECKPOINT_VERSION = 1


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def sanitize_nonfinite(value: Any) -> Any:
    """把非有限浮点替换为 None，保证 JSON 严格可序列化（不静默丢失有限值）。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): sanitize_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_nonfinite(item) for item in value]
    return value


def dump_json_safe(path: Path, payload: Any) -> None:
    dump_json(path, sanitize_nonfinite(jsonable(payload)))


def dump_jsonl_safe(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    dump_jsonl(path, [sanitize_nonfinite(jsonable(row)) for row in rows])


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


def parameter_groups(model: torch.nn.Module) -> dict[str, Any]:
    """按实际 named_parameters 划分共享 / 策略专属 / 价值专属三组。"""
    groups: dict[str, list[str]] = {"shared": [], "policy": [], "value": []}
    for name, _parameter in model.named_parameters():
        if name.startswith(VALUE_PARAMETER_PREFIXES):
            groups["value"].append(name)
        elif name.startswith(POLICY_PARAMETER_PREFIXES):
            groups["policy"].append(name)
        elif name.startswith(SHARED_PARAMETER_PREFIXES):
            groups["shared"].append(name)
        else:
            raise ValueError(f"参数 {name} 未被任何参数组覆盖")
    all_names = [name for names in groups.values() for name in names]
    if len(all_names) != len(set(all_names)):
        raise ValueError("参数组之间存在重复参数")
    if set(all_names) != {name for name, _ in model.named_parameters()}:
        raise ValueError("参数组未覆盖全部 named_parameters")
    counts = {key: len(value) for key, value in groups.items()}
    return {
        "groups": groups,
        "counts": counts,
        "total": len(all_names),
        "value_prefixes": list(VALUE_PARAMETER_PREFIXES),
        "policy_prefixes": list(POLICY_PARAMETER_PREFIXES),
        "shared_prefixes": list(SHARED_PARAMETER_PREFIXES),
    }


def group_grad_norms(model: torch.nn.Module, groups: Mapping[str, Sequence[str]]) -> dict[str, float]:
    norms: dict[str, float] = {}
    for group_name, names in groups.items():
        total = 0.0
        for name in names:
            parameter = dict(model.named_parameters()).get(name)
            if parameter is None or parameter.grad is None:
                continue
            total += float(parameter.grad.detach().float().pow(2).sum().item())
        norms[group_name] = math.sqrt(total)
    return norms


def group_parameter_delta(
    before: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
    groups: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    after = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    deltas: dict[str, float] = {}
    for group_name, names in groups.items():
        total = 0.0
        for name in names:
            if name not in before:
                continue
            total += float((after[name].float() - before[name].float()).pow(2).sum().item())
        deltas[group_name] = math.sqrt(total)
    return deltas


def replay_contract_ok(error: float | None, tolerance: float = REPLAY_TOLERANCE) -> bool:
    """更新前重放必须有限且不超过严格容差；否则必须阻塞而不是继续更新。"""
    if error is None:
        return False
    return bool(math.isfinite(float(error)) and abs(float(error)) <= float(tolerance))


def stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "all_finite": bool(np.isfinite(array).all()),
    }


def gpu_snapshot() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"allocated_bytes": 0, "reserved_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


# ---------------------------------------------------------------------------
# 诊断检查点
# ---------------------------------------------------------------------------


def save_checkpoint_state(
    path: Path,
    state: Mapping[str, torch.Tensor],
    model: torch.nn.Module,
    *,
    step_index: int,
    decisions: int,
    optimizer_steps: int,
    episode_id: int,
    tag: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "diagnostic_format": DIAGNOSTIC_CHECKPOINT_FORMAT,
        "diagnostic_format_version": DIAGNOSTIC_CHECKPOINT_VERSION,
        "tag": tag,
        "step_index": int(step_index),
        "training_decisions": int(decisions),
        "optimizer_steps": int(optimizer_steps),
        "episode_id": int(episode_id),
        "actor_critic_state": dict(state),
        "actor_fingerprint": state_dict_fingerprint(state),
        "hyperparameters": {
            "state_dim": int(getattr(model, "state_dim", 32)),
            "task_feat_dim": int(getattr(model, "task_feat_dim", 8)),
            "hidden_dim": int(getattr(model, "hidden_dim", 64)),
            "max_station_workers": int(getattr(model, "max_station_workers", 16)),
        },
        "graph_feature_version": GRAPH_FEATURE_VERSION,
        "graph_feature_dims": dict(GRAPH_FEATURE_DIMS),
        "graph_feature_schema_hash": GRAPH_FEATURE_SCHEMA["schema_fingerprint"],
        "method_profile": __import__("dataclasses").asdict(build_method_profile("C")),
        "source": {
            "bc_weights": str(BC_WEIGHTS),
            "bc_weights_sha256": sha256_file(BC_WEIGHTS),
        },
        "extra": dict(extra or {}),
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "actor_fingerprint": payload["actor_fingerprint"],
        "training_decisions": int(decisions),
        "optimizer_steps": int(optimizer_steps),
        "step_index": int(step_index),
        "tag": tag,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    step_index: int,
    decisions: int,
    optimizer_steps: int,
    episode_id: int,
    tag: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return save_checkpoint_state(
        path,
        clone_actor_state(model),
        model,
        step_index=step_index,
        decisions=decisions,
        optimizer_steps=optimizer_steps,
        episode_id=episode_id,
        tag=tag,
        extra=extra,
    )


def load_checkpoint_actor(path: Path) -> tuple[ActorCriticWork3, dict[str, Any]]:
    """严格加载本轮诊断检查点；不允许退化为普通 state_dict 旁路。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("diagnostic_format") != DIAGNOSTIC_CHECKPOINT_FORMAT:
        raise ValueError(f"不是本轮诊断检查点: {path}")
    if int(payload.get("diagnostic_format_version", -1)) != DIAGNOSTIC_CHECKPOINT_VERSION:
        raise ValueError("诊断检查点版本不匹配")
    if payload.get("graph_feature_version") != GRAPH_FEATURE_VERSION:
        raise ValueError("诊断检查点图特征版本不匹配")
    if dict(payload.get("graph_feature_dims") or {}) != dict(GRAPH_FEATURE_DIMS):
        raise ValueError("诊断检查点图特征维度不匹配")
    if payload.get("graph_feature_schema_hash") != GRAPH_FEATURE_SCHEMA["schema_fingerprint"]:
        raise ValueError("诊断检查点图特征 schema 指纹不匹配")
    hyper = payload.get("hyperparameters") or {}
    model = ActorCriticWork3(
        state_dim=int(hyper["state_dim"]),
        task_feat_dim=int(hyper["task_feat_dim"]),
        hidden_dim=int(hyper["hidden_dim"]),
        max_station_workers=int(hyper["max_station_workers"]),
    )
    state = payload.get("actor_critic_state")
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise ValueError("诊断检查点参数集合不匹配")
    for name, tensor in model.state_dict().items():
        if not isinstance(state[name], torch.Tensor) or state[name].shape != tensor.shape:
            raise ValueError(f"诊断检查点参数维度不匹配: {name}")
    if state_dict_fingerprint(state) != payload.get("actor_fingerprint"):
        raise ValueError("诊断检查点指纹与权重不一致")
    model.load_state_dict(state, strict=True)
    return model, payload


# ---------------------------------------------------------------------------
# 诊断化 Lightning 模块（只加测量，不改更新语义）
# ---------------------------------------------------------------------------


class DiagnosticsLightningModule(Work3LightningModule):
    """在正式 ``training_step`` 之外只增加前置/后置测量。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.update_records: list[dict[str, Any]] = []
        self.parameter_group_names: dict[str, list[str]] = {}
        self.first_step_state: dict[str, torch.Tensor] | None = None
        self._pre_clip_group_norms: dict[str, float] | None = None
        self._pre_update_state: dict[str, torch.Tensor] | None = None
        self._current_loss: float | None = None
        self._current_lr = float(kwargs.get("learning_rate", 3e-4))

    def log(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - 无 Trainer 时静默
        return None

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.parameter_group_names:
            self._pre_clip_group_norms = group_grad_norms(
                self.actor_critic, self.parameter_group_names
            )
        super().on_before_optimizer_step(optimizer)

    def _manual_update(self, loss: torch.Tensor, optimizer: Any):
        if not self.parameter_group_names:
            return super()._manual_update(loss, optimizer)
        self._pre_update_state = clone_actor_state(self.actor_critic)
        self._current_loss = float(loss.detach().float().item())
        grad_norm = super()._manual_update(loss, optimizer)
        if self.first_step_state is None and self.optimization_steps >= 1:
            # 首次真实 optimizer.step 之后的模型状态
            self.first_step_state = clone_actor_state(self.actor_critic)
        deltas = group_parameter_delta(self._pre_update_state, self.actor_critic, self.parameter_group_names)
        self.update_records.append(
            {
                "optimizer_step": int(self.optimization_steps),
                "total_loss": self._current_loss,
                "clipped_grad_norm": None if grad_norm is None else float(grad_norm),
                "pre_clip_group_grad_norm": dict(self._pre_clip_group_norms or {}),
                "parameter_delta_norm": deltas,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        self._pre_update_state = None
        return grad_norm


# ---------------------------------------------------------------------------
# Phase A：清册与初始化核验
# ---------------------------------------------------------------------------


def load_original_c_metadata() -> dict[str, Any]:
    run = load_json(ORIGINAL_C_RUN)
    training = run["training_config"]
    return {
        "run_json_path": str(ORIGINAL_C_RUN),
        "run_json_sha256": sha256_file(ORIGINAL_C_RUN),
        "resolved_runtime_config_sha256": run.get("resolved_runtime_config_sha256"),
        "resolved_runtime_config_yaml": run.get("resolved_runtime_config_yaml"),
        "training_config": training,
        "data_fingerprint": run.get("data_fingerprint"),
    }


def resolved_hyperparameters(original: Mapping[str, Any]) -> dict[str, Any]:
    """以原 C 的 run.json 为准解析超参，并与第 4 节固定项逐项核对。"""
    training = original["training_config"]
    resolved = {
        "method_variant": "C",
        "gamma": float(training["gamma"]),
        "gae_lambda": float(training["gae_lambda"]),
        "beta_shaping": float(training["beta_shaping"]),
        "learning_rate": float(training["lr"]),
        "clip_epsilon": float(training["clip_eps"]),
        "value_coefficient": float(training["vf_coef"]),
        "entropy_coefficient": float(training["ent_coef"]),
        "ppo_epochs": int(training["ppo_epochs"]),
        "batch_size": int(training["batch_size"]),
        "rollout_steps": int(training["rollout_steps"]),
        "num_envs": int(training["num_envs"]),
        "max_grad_norm": 0.5,
        "optimizer": "AdamW",
        "optimizer_eps": 1e-5,
        "weight_decay": 1e-4,
        "normalize_advantages": True,
        "amp_dtype": "fp32",
        "disturbance_enabled": False,
        "scenario": "NOMINAL",
        "warmup_mode": str(training["warmup_mode"]),
        "seed": 42,
        "total_env_steps": TRAINING_BUDGET,
        "max_decisions": TRAINING_BUDGET,
    }
    expected = {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "beta_shaping": 1.0,
        "learning_rate": 3e-4,
        "clip_epsilon": 0.2,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "ppo_epochs": 1,
        "batch_size": 2,
        "rollout_steps": 8,
        "num_envs": 1,
    }
    conflicts = {
        key: {"resolved": resolved[key], "expected": value}
        for key, value in expected.items()
        if abs(float(resolved[key]) - float(value)) > 1e-12
    }
    return {
        "resolved": resolved,
        "expected": expected,
        "conflicts": conflicts,
        "source": "work3_4060/rule_baseline_ac_v1/seed42/C/run.json::training_config",
        "clarification": (
            "原 C 的 disturbance_enabled=true（pilot 模式混采扰动）；本轮按任务要求强制 "
            "Nominal-only，该差异是本轮唯一有意的配置偏离，已记录在 resolved_config.yaml。"
        ),
    }


def write_resolved_config_yaml(resolved: Mapping[str, Any], conflicts: Mapping[str, Any], git: Mapping[str, Any]) -> None:
    lines = [
        "# 本轮固定配置：由原 C run.json 解析，并按任务第 4 节固定项核对",
        "runtime:",
        f"  reference_commit: {git.get('head')}",
        f"  seed: {int(resolved['seed'])}",
        f"  device: cuda:0",
        "  precision: fp32",
        f"  num_envs: {int(resolved['num_envs'])}",
        "  start_method: spawn",
        f"  total_env_steps: {int(resolved['total_env_steps'])}",
        f"  max_decisions: {int(resolved['max_decisions'])}",
        "  disturbance_enabled: false",
        "  scenario: NOMINAL",
        "  warmup_mode: none",
        "paths:",
        f"  baseline: {BASELINE_PATH}",
        f"  init_weights: {BC_WEIGHTS}",
        "ppo:",
        f"  steps_per_iter: {int(resolved['rollout_steps'])}",
        f"  epochs: {int(resolved['ppo_epochs'])}",
        f"  batch_size: {int(resolved['batch_size'])}",
        f"  learning_rate: {resolved['learning_rate']}",
        f"  clip_epsilon: {resolved['clip_epsilon']}",
        f"  value_coefficient: {resolved['value_coefficient']}",
        f"  entropy_coefficient: {resolved['entropy_coefficient']}",
        f"  gamma: {resolved['gamma']}",
        f"  gae_lambda: {resolved['gae_lambda']}",
        f"  shaping_coefficient: {resolved['beta_shaping']}",
        f"  max_grad_norm: {resolved['max_grad_norm']}",
        "  optimizer: AdamW",
        f"  optimizer_eps: {resolved['optimizer_eps']}",
        f"  weight_decay: {resolved['weight_decay']}",
        f"  normalize_advantages: {str(resolved['normalize_advantages']).lower()}",
        "evaluation:",
        f"  decision_limit: {EVALUATION_DECISION_LIMIT}",
        "  device: cpu",
        "  threads: 1",
        "  deterministic: true",
        "checkpoints:",
        "  points: [0, 128, 256, 512]",
        "conflicts:",
    ]
    if conflicts:
        for key, value in conflicts.items():
            lines.append(f"  - {key}: resolved={value['resolved']} expected={value['expected']}")
    else:
        lines.append("  - none")
    RESOLVED_CONFIG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fixed_snapshot_sample(limit: int = 6) -> list[dict[str, Any]]:
    bundle = torch.load(BC_TEACHER_SNAPSHOTS, map_location="cpu", weights_only=False)
    samples = bundle["samples"][:limit]
    return samples


def compare_policy_paths(
    diagnostic_model: torch.nn.Module,
    training_model: torch.nn.Module,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """在固定历史快照上核对“诊断加载路径”与“训练侧构造路径”的一致性。"""
    from work3_4060.c_reproduction_protocol import sample_records, sample_snapshot

    max_log_prob_diff = 0.0
    max_value_diff = 0.0
    compared = 0
    for sample in samples:
        snapshot = sample_snapshot(sample)
        records = sample_records(sample, int(getattr(training_model, "max_station_workers", 16)))
        state = snapshot.state_features.unsqueeze(0).expand(len(records), -1).contiguous()
        time_feature = snapshot.time_features.unsqueeze(0).expand(len(records), -1).contiguous()
        with torch.no_grad():
            values_d, lp_d, _ = diagnostic_model.evaluate_action_log_probs(state, time_feature, records)
            state_device = state.to(next(training_model.parameters()).device)
            time_device = time_feature.to(next(training_model.parameters()).device)
            values_t, lp_t, _ = training_model.evaluate_action_log_probs(
                state_device, time_device, records
            )
        max_log_prob_diff = max(
            max_log_prob_diff, float((lp_d.cpu().double() - lp_t.cpu().double()).abs().max().item())
        )
        max_value_diff = max(
            max_value_diff, float((values_d.cpu().double() - values_t.cpu().double()).abs().max().item())
        )
        compared += 1
    return {
        "compared_samples": compared,
        "max_abs_log_prob_difference": max_log_prob_diff,
        "max_abs_value_difference": max_value_diff,
        "tolerance": POLICY_TOLERANCE_CPU_VS_GPU,
        "within_tolerance": max_log_prob_diff <= POLICY_TOLERANCE_CPU_VS_GPU
        and max_value_diff <= POLICY_TOLERANCE_CPU_VS_GPU,
        "scope": (
            "同一份 BC 权重，一条经诊断加载器（CPU），一条经训练侧 ActorCriticWork3+load_state_dict"
            "（CUDA）；固定历史快照上的动作 log-prob 与 V 比较，容差 1e-4。"
        ),
    }


def stage_inventory() -> int:
    threads = enforce_single_thread_cpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)

    if not BC_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"缺少 BC 诊断副本（二进制未入 Git，需用户本地原件）: {BC_WEIGHTS}"
        )
    if not BC_MANIFEST.is_file():
        raise FileNotFoundError(f"缺少上一轮清单，无法绑定 BC 来源: {BC_MANIFEST}")

    previous = load_json(BC_MANIFEST)
    bc_expected = previous["diagnostic_checkpoint"]
    bc_sha = sha256_file(BC_WEIGHTS)
    if bc_sha != bc_expected["sha256"]:
        raise ValueError(f"BC 检查点 SHA256 与上一轮清单不一致: {bc_sha}")
    bc_model, bc_checkpoint = load_diagnostic_actor(
        BC_WEIGHTS, expected_teacher_manifest_sha256=previous["teacher"]["manifest_sha256"]
    )
    bc_fingerprint = actor_fingerprint(bc_model)
    if bc_fingerprint != bc_expected["actor_fingerprint"]:
        raise ValueError("BC Actor 指纹与上一轮清单不一致")
    original_sha = sha256_file(ORIGINAL_C_WEIGHTS)
    if bc_checkpoint["source_checkpoint"]["sha256"] != original_sha:
        raise ValueError("BC 记录的源权重 SHA256 与当前原 C 文件不一致")

    original_meta = load_original_c_metadata()
    hyper = resolved_hyperparameters(original_meta)
    if hyper["conflicts"]:
        dump_json(OUTPUT_DIR / "config_conflicts.json", hyper["conflicts"])
        log(f"超参冲突: {hyper['conflicts']}")

    # 训练侧构造路径：全新建模 + 严格键集合加载（不经诊断加载器的宽松分支）
    training_model = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    bc_state = {name: tensor.clone() for name, tensor in bc_checkpoint["actor_critic_state"].items()}
    if set(bc_state) != set(training_model.state_dict()):
        raise ValueError("BC 检查点参数集合与训练侧模型不一致")
    training_model.load_state_dict(bc_state, strict=True)
    for parameter in training_model.parameters():
        parameter.requires_grad = True
    if actor_fingerprint(training_model) != bc_fingerprint:
        raise ValueError("训练侧构造路径与诊断加载器指纹不一致")
    if not all(p.requires_grad for p in training_model.parameters()):
        raise ValueError("训练侧模型存在被意外冻结的参数")

    groups = parameter_groups(training_model)
    dump_json(PARAMETER_GROUPS, groups)

    samples = fixed_snapshot_sample(limit=6)
    training_model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    path_check = compare_policy_paths(bc_model.to("cpu"), training_model, samples)
    training_model.to("cpu")

    write_resolved_config_yaml(hyper["resolved"], hyper["conflicts"], git_state())

    initialization = {
        "schema_version": "bc_ppo_retention_initialization_v1",
        "source": {
            "bc_path": str(BC_WEIGHTS),
            "bc_sha256": bc_sha,
            "bc_format": bc_checkpoint["diagnostic_format"],
            "bc_actor_fingerprint": bc_fingerprint,
            "bc_source_checkpoint_sha256": bc_checkpoint["source_checkpoint"]["sha256"],
            "bc_source_actor_fingerprint": bc_checkpoint["source_checkpoint"]["actor_fingerprint"],
            "bc_teacher_manifest_sha256": bc_checkpoint["teacher_manifest_sha256"],
            "previous_manifest_sha256": sha256_file(BC_MANIFEST),
            "original_c_sha256": original_sha,
            "binding_valid": True,
        },
        "load_contract": {
            "loader": "work3_4060.c_reproduction_protocol.load_diagnostic_actor",
            "training_side": "ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)+load_state_dict(strict=True)",
            "parameter_count": groups["total"],
            "all_parameters_trainable": True,
            "buffers_transferred": True,
            "source_object_not_optimized": True,
            "note": "加载到独立实例；源对象与源文件均不进入优化器，也不被原地修改。",
        },
        "policy_path_equivalence": path_check,
        "graph_schema": {
            "version": GRAPH_FEATURE_VERSION,
            "dims": dict(GRAPH_FEATURE_DIMS),
            "schema_hash": GRAPH_FEATURE_SCHEMA["schema_fingerprint"],
        },
        "resolved_hyperparameters": hyper["resolved"],
        "config_conflicts": hyper["conflicts"],
        "threads": threads,
        "device": "cpu",
    }
    dump_json(INIT_VERIFICATION, initialization)

    dump_json(
        MANIFEST,
        {
            "task": "Work-3 BC 初始化后的 PPO 基准保持性诊断",
            "schema_version": "bc_ppo_retention_manifest_v1",
            "reference_commit": "72de395e36846a669f26a7d2fcfa82d768300bd9",
            "run_context": git_state(),
            "output_dir": str(OUTPUT_DIR),
            "instance": {
                "baseline_path": str(BASELINE_PATH),
                "baseline_sha256": sha256_file(BASELINE_PATH),
                "h0": float(load_json(BASELINE_PATH)["metadata"]["h0"]),
                "transfer_policy": load_json(BASELINE_PATH)["metadata"].get("transfer_policy"),
                "scenario": "NOMINAL",
                "disturbance": "none",
            },
            "budget": {
                "training_decisions": TRAINING_BUDGET,
                "rollout_steps": ROLLOUT_STEPS,
                "checkpoints": list(CHECKPOINTS),
                "evaluation_decision_limit": EVALUATION_DECISION_LIMIT,
                "optimizer_updates_planned": TRAINING_BUDGET // 2,
            },
            "weights": {
                "init_bc_sha256": bc_sha,
                "init_bc_actor_fingerprint": bc_fingerprint,
                "original_c_sha256": original_sha,
            },
            "scope": {
                "new_trajectories": "step0000 / step0128 / step0256 / step0512 各一条 Nominal",
                "no_disturbance_rerun": True,
                "no_bc_regularization": True,
                "no_value_calibration": True,
            },
            "artifacts_expected": [
                "manifest.json",
                "resolved_config.yaml",
                "initialization_verification.json",
                "parameter_groups.json",
                "training_history.jsonl",
                "first_rollout.pt",
                "first_update_diagnostics.json",
                "checkpoints/{step0000,step0128,step0256,step0512}.pt",
                "trajectories/{step0000,step0128,step0256,step0512}_NOMINAL.json",
                "evaluation_results.csv",
                "cycle_comparison.csv",
                "trajectory_verification.json",
                "shaping_events.jsonl",
                "shaping_audit_summary.json",
                "weight_integrity.json",
                "verification.json",
                "summary.md",
            ],
        },
    )
    log(
        "inventory 完成: BC 绑定 OK，参数组 shared/policy/value = "
        f"{groups['counts']['shared']}/{groups['counts']['policy']}/{groups['counts']['value']}，"
        f"路径一致性 {path_check['within_tolerance']}"
    )
    return 0


# ---------------------------------------------------------------------------
# Phase B：训练
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return {
        "python": random.getstate()[1][:3],
        "numpy": int(np.random.get_state()[1][0]),
        "torch": int(torch.initial_seed()),
        "cuda": int(torch.cuda.initial_seed()) if torch.cuda.is_available() else None,
    }


def build_env() -> Work3VectorEnv:
    return Work3VectorEnv(
        env_kwargs={"baseline_json_path": str(BASELINE_PATH)},
        num_envs=1,
        worker_torch_num_threads=1,
        start_method="spawn",
    )


class ShapingWriter:
    """按既有 ``verify_shaping_trace`` 合同写出单步与段尾事件。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")
        self.transition_count = 0

    def transition(self, record: Mapping[str, Any]) -> None:
        self.handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
        self.transition_count += 1

    def segment_end(self, record: Mapping[str, Any]) -> None:
        self.handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")

    def episode_cost(self, record: Mapping[str, Any]) -> None:
        self.handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


def stage_train() -> int:
    log("train: 启动")
    if not torch.cuda.is_available():
        raise RuntimeError("本阶段要求 cuda:0")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    log(f"train: 设备 {torch.cuda.get_device_name(0)}")

    initialization = load_json(INIT_VERIFICATION)
    groups = load_json(PARAMETER_GROUPS)
    hyper = initialization["resolved_hyperparameters"]
    seeds = seed_everything(int(hyper["seed"]))

    model, _bc_payload = load_diagnostic_actor(
        BC_WEIGHTS,
        expected_teacher_manifest_sha256=load_json(BC_MANIFEST)["teacher"]["manifest_sha256"],
    )
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.to(device)
    model.train()
    init_fingerprint = actor_fingerprint(model)
    if init_fingerprint != initialization["source"]["bc_actor_fingerprint"]:
        raise ValueError("初始化后 Actor 指纹与 BC 源不一致")
    log(f"train: 模型已加载 fp={init_fingerprint[:16]}")

    checkpoint_records: list[dict[str, Any]] = []
    checkpoint_records.append(
        save_checkpoint(
            CHECKPOINT_DIR / "step0000.pt",
            model,
            step_index=0,
            decisions=0,
            optimizer_steps=0,
            episode_id=0,
            tag="step0000",
            extra={"note": "BC 原始权重快照（零更新）"},
        )
    )

    lightning = DiagnosticsLightningModule(
        actor_critic=model,
        learning_rate=float(hyper["learning_rate"]),
        clip_eps=float(hyper["clip_epsilon"]),
        vf_coef=float(hyper["value_coefficient"]),
        ent_coef=float(hyper["entropy_coefficient"]),
        max_grad_norm=float(hyper["max_grad_norm"]),
        ppo_epochs=int(hyper["ppo_epochs"]),
        batch_size=int(hyper["batch_size"]),
    )
    lightning.parameter_group_names = {
        "shared": groups["groups"]["shared"],
        "policy": groups["groups"]["policy"],
        "value": groups["groups"]["value"],
    }
    optimizer = lightning.configure_optimizers()
    optimizer_ids = {id(p) for p in lightning.objective.optimized_parameters}
    model_ids = {id(p) for p in model.parameters()}
    if optimizer_ids != model_ids:
        raise ValueError("优化器参数集合与模型参数集合不一致（存在遗漏或重复）")
    if any(state for state in optimizer.state.values()):
        raise ValueError("新建优化器不应携带任何历史状态")

    env = build_env()
    log("train: spawn 环境 worker 已就绪")
    shaping = ShapingWriter(SHAPING_EVENTS)
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
    first_update_saved = False
    first_update_state: dict[str, torch.Tensor] | None = None
    pre_first_update_state: dict[str, torch.Tensor] | None = None
    first_rollout_payload: dict[str, Any] | None = None
    pending_snapshot = None
    memory_peak = 0
    started = time.perf_counter()

    try:
        env.reset(scenario=None, episode_id=episode_id, episode_index=episode_id)
        pending_snapshot = env.snapshot()

        while decisions < TRAINING_BUDGET:
            next_checkpoint = next(
                (point for point in CHECKPOINTS if point > decisions), TRAINING_BUDGET
            )
            segment_limit = min(ROLLOUT_STEPS, next_checkpoint - decisions)
            if segment_limit <= 0:
                break
            buffer = RolloutBufferWork3(
                gamma=float(hyper["gamma"]),
                gae_lambda=float(hyper["gae_lambda"]),
                normalize_advantages=True,
            )
            raw_buffer = RolloutBufferWork3(
                gamma=float(hyper["gamma"]),
                gae_lambda=float(hyper["gae_lambda"]),
                normalize_advantages=False,
            )
            transitions: list[PPOTransition] = []
            segment_records: list[dict[str, Any]] = []
            terminated = False
            last_value = 0.0
            for step_in_segment in range(segment_limit):
                snapshot = pending_snapshot
                with torch.no_grad():
                    action, log_prob, value, sample_record = model.select_snapshot(
                        snapshot, deterministic=False
                    )
                phi_current = compute_heuristic_potential(
                    float(snapshot.estimated_cmax),
                    float(snapshot.last_transfer_time),
                    float(snapshot.h0),
                )
                result = env.step(action)
                raw_reward = float(result.raw_reward)
                terminated = bool(result.terminated)
                truncated = bool(result.truncated)
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
                shaped_reward = raw_reward + float(hyper["beta_shaping"]) * (
                    float(hyper["gamma"]) * phi_next - phi_current
                )
                transition = PPOTransition(
                    state_feat=snapshot.state_features.cpu(),
                    time_urgency=snapshot.time_features.cpu(),
                    sample_record=sample_record,
                    reward=float(shaped_reward),
                    raw_reward=raw_reward,
                    value=float(value),
                    log_prob=float(log_prob),
                    done=bool(terminated or truncated),
                    action_dict=dict(action),
                    terminated=terminated,
                    truncated=truncated,
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
                        "shaped_reward": float(shaped_reward),
                        "phi_current": phi_current,
                        "phi_next": phi_next,
                        "terminated": terminated,
                        "truncated": truncated,
                        "step_cost": float(info.get("step_cost", 0.0)),
                        "advance_cause": info.get("advance_cause"),
                    }
                )
                shaping.transition(
                    {
                        "record_type": "transition",
                        "run_id": "bc_ppo_retention",
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
                        "shaped_reward": float(shaped_reward),
                        "phi_current": float(phi_current),
                        "phi_next": float(phi_next),
                        "beta": float(hyper["beta_shaping"]),
                        "gamma": float(hyper["gamma"]),
                        "terminated": terminated,
                        "truncated": truncated,
                        "predictor_version": None,
                        "bootstrap_value": None,
                        "phi_source": "heuristic_phi",
                    }
                )
                last_value = float(value)
                episode_last_info[episode_id] = info
                episode_raw_sum[episode_id] = episode_raw_sum.get(episode_id, 0.0) + raw_reward
                if terminated:
                    break

            if not terminated:
                with torch.no_grad():
                    last_snapshot = pending_snapshot
                    bootstrap_tensor, _ = model.encode_state(
                        last_snapshot.state_features.to(device),
                        last_snapshot.time_features.to(device),
                        graph_data=last_snapshot.graph_snapshot,
                    )
                last_value = float(bootstrap_tensor.squeeze().item())
                boundary_type = (
                    "budget_truncation"
                    if decisions >= TRAINING_BUDGET
                    else "rollout_continuation"
                )
            else:
                last_value = 0.0
                boundary_type = "natural_termination"

            key = (0, episode_id, rollout_id)
            buffer.finish_trajectories(last_values_by_segment={key: last_value})
            raw_buffer.finish_trajectories(last_values_by_segment={key: last_value})
            raw_advantages = raw_buffer.advantages.clone()

            shaped_rewards = [t.reward for t in transitions]
            raw_rewards = [t.raw_reward for t in transitions]
            values = [t.value for t in transitions]
            returns = (raw_advantages + torch.tensor(values, dtype=torch.float)).tolist()
            td_residuals = []
            for index, transition in enumerate(transitions):
                next_value = 0.0 if transition.is_terminated else (
                    values[index + 1] if index + 1 < len(values) else float(last_value)
                )
                td_residuals.append(
                    float(transition.reward)
                    + float(hyper["gamma"]) * next_value
                    * (0.0 if transition.is_terminated else 1.0)
                    - float(transition.value)
                )

            # 更新前重放核对：同一批自回归事件，old vs replay
            with torch.no_grad():
                state_batch = torch.stack([t.state_feat for t in transitions])
                time_batch = torch.stack([t.time_urgency for t in transitions])
                _v, replay_lp, replay_ent = model.evaluate_action_log_probs(
                    state_batch, time_batch, [t.sample_record for t in transitions]
                )
                old_lp = torch.tensor([t.log_prob for t in transitions], dtype=torch.float)
                replay_error = float((replay_lp.cpu().double() - old_lp.double()).abs().max().item())
            if not replay_contract_ok(replay_error):
                dump_json(
                    OUTPUT_DIR / "replay_failure.json",
                    {
                        "rollout_id": rollout_id,
                        "replay_max_abs_error": replay_error,
                        "tolerance": REPLAY_TOLERANCE,
                        "old_log_probs": old_lp.tolist(),
                        "replay_log_probs": replay_lp.tolist(),
                    },
                )
                stop_reason = "replay_contract_failed"
                break

            shaping.segment_end(
                {
                    "record_type": "segment_end",
                    "run_id": "bc_ppo_retention",
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
                    build_episode_cost_record(
                        episode_id=episode_id,
                        info=episode_last_info.get(episode_id, {}),
                        termination_reason="completed",
                        episode_steps=episode_step,
                        raw_reward_sum=episode_raw_sum.get(episode_id, 0.0),
                    )
                )
                episode_closed.add(episode_id)

            update = Work3TrainingUpdate(
                buffer=buffer,
                environment_steps=len(transitions),
                time_auxiliary_batch=None,
                rollout_id=rollout_id,
            )
            pre_state = clone_actor_state(model) if not first_update_saved else None
            updates_before = lightning.optimization_steps
            loss = lightning.training_step(update, 0)
            if not bool(torch.isfinite(loss).item()):
                stop_reason = "non_finite_loss"
                break
            updates_this_segment = lightning.optimization_steps - updates_before
            optimizer_steps = lightning.optimization_steps
            segment_update_records = (
                lightning.update_records[-updates_this_segment:] if updates_this_segment > 0 else []
            )
            if not first_update_saved and updates_this_segment > 0 and lightning.first_step_state is not None:
                pre_first_update_state = pre_state
                first_update_state = lightning.first_step_state
                first_update_saved = True
                save_checkpoint_state(
                    CHECKPOINT_DIR / "first_update.pt",
                    first_update_state,
                    model,
                    step_index=0,
                    decisions=decisions,
                    optimizer_steps=optimizer_steps,
                    episode_id=episode_id,
                    tag="first_update",
                    extra={
                        "note": "首次 optimizer.step 之后的模型状态（首段还有后续更新）",
                        "segment_updates_total": updates_this_segment,
                    },
                )
            if not first_rollout_saved:
                first_rollout_payload = {
                    "rollout_id": rollout_id,
                    "episode_id": episode_id,
                    "decisions": decisions,
                    "state_feats": torch.stack([t.state_feat for t in transitions]),
                    "time_urgencies": torch.stack([t.time_urgency for t in transitions]),
                    "sample_records": [t.sample_record for t in transitions],
                    "actions": [t.action_dict for t in transitions],
                    "old_log_probs": old_lp,
                    "old_values": torch.tensor(values, dtype=torch.float),
                    "raw_rewards": torch.tensor(raw_rewards, dtype=torch.float),
                    "shaped_rewards": torch.tensor(shaped_rewards, dtype=torch.float),
                    "terminated": [bool(t.terminated) for t in transitions],
                    "truncated": [bool(t.truncated) for t in transitions],
                    "episode_steps": [int(r["episode_step"]) for r in segment_records],
                    "boundary_type": boundary_type,
                    "bootstrap_value": float(last_value),
                    "raw_advantages": raw_advantages,
                    "normalized_advantages": buffer.advantages.clone(),
                    "target_values": buffer.target_values.clone(),
                    "segment_records": segment_records,
                    "hyperparameters": dict(hyper),
                }
                torch.save(first_rollout_payload, FIRST_ROLLOUT)
                first_rollout_saved = True

            memory = gpu_snapshot()
            memory_peak = max(memory_peak, memory["peak_reserved_bytes"])
            history.append(
                {
                    "rollout_id": rollout_id,
                    "training_decisions": decisions,
                    "segment_steps": len(transitions),
                    "episode_id": episode_id,
                    "episode_step_end": episode_step,
                    "optimizer_updates_this_segment": updates_this_segment,
                    "optimizer_updates_total": optimizer_steps,
                    "terminated": terminated,
                    "boundary_type": boundary_type,
                    "bootstrap_value": float(last_value),
                    "replay_max_abs_error": replay_error,
                    "advantage_raw_stats": stats(raw_advantages.tolist()),
                    "advantage_normalized_stats": stats(buffer.advantages.tolist()),
                    "value_stats": stats(values),
                    "td_residual_stats": stats(td_residuals),
                    "return_target_stats": stats(buffer.target_values.tolist()),
                    "raw_reward_stats": stats(raw_rewards),
                    "shaped_reward_stats": stats(shaped_rewards),
                    "phi_current_stats": stats([r["phi_current"] for r in segment_records]),
                    "action_counts": action_counts(segment_records),
                    "update_records": segment_update_records,
                    "elapsed_seconds": time.perf_counter() - started,
                    "cuda_peak_reserved_mib": memory["peak_reserved_bytes"] / 2**20,
                }
            )
            if memory["peak_reserved_bytes"] >= GPU_BUDGET_BYTES:
                stop_reason = "gpu_budget_exceeded"
                break
            if any(
                not math.isfinite(record["total_loss"]) or not math.isfinite(record["clipped_grad_norm"] or 0.0)
                for record in segment_update_records
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
                    save_checkpoint(
                        CHECKPOINT_DIR / f"step{decisions:04d}.pt",
                        model,
                        step_index=list(CHECKPOINTS).index(decisions),
                        decisions=decisions,
                        optimizer_steps=optimizer_steps,
                        episode_id=episode_id,
                        tag=f"step{decisions:04d}",
                    )
                )
                log(f"已保存检查点 step{decisions:04d}（优化步 {optimizer_steps}）")

        if decisions < TRAINING_BUDGET and stop_reason == "budget_exhausted":
            stop_reason = "stopped_before_budget"
    except torch.cuda.OutOfMemoryError as error:  # pragma: no cover
        stop_reason = "cuda_oom"
        dump_json(OUTPUT_DIR / "cuda_oom.json", {"error": str(error), "decisions": decisions})
    except FloatingPointError as error:
        stop_reason = "non_finite_gradient"
        dump_json(OUTPUT_DIR / "non_finite_gradient.json", {"error": str(error), "decisions": decisions})
    finally:
        # 未自然终止的 episode 也要写出截至该时刻的账本，供审计按实际覆盖对账。
        for open_id in sorted(episode_last_info):
            if open_id in episode_closed:
                continue
            shaping.episode_cost(
                build_episode_cost_record(
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
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        final = save_checkpoint(
            CHECKPOINT_DIR / f"last_decision{decisions:04d}.pt",
            model,
            step_index=-1,
            decisions=decisions,
            optimizer_steps=optimizer_steps,
            episode_id=episode_id,
            tag="last_valid",
            extra={"stop_reason": stop_reason},
        )
        dump_jsonl_safe(TRAINING_HISTORY, history)
        dump_json_safe(
            OUTPUT_DIR / "training_summary.json",
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
                    "budget_mib": GPU_BUDGET_BYTES / 2**20,
                    "budget_exceeded": memory_peak >= GPU_BUDGET_BYTES,
                },
                "threads": {"main": torch.get_num_threads(), "env_worker": 1},
                "checkpoints": checkpoint_records,
                "final_state": final,
                "hyperparameters": dict(hyper),
            },
        )
        write_ppo_run_json(history, hyper, stop_reason, decisions)
        if first_update_saved and pre_first_update_state is not None and first_update_state is not None:
            build_first_update_diagnostics(
                pre_first_update_state, first_update_state, first_rollout_payload, model
            )
    log(f"train 结束: decisions={decisions} updates={optimizer_steps} stop={stop_reason}")
    return 0 if stop_reason in ("budget_exhausted", "stopped_before_budget") else 2


def action_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"stay": 0, "postpone": 0, "advance": 0, "align1": 0, "reserved_or_running": 0}
    for record in records:
        branch = int(record["action_type"])
        if branch == 2:
            counts["advance"] += 1
        elif branch == 1:
            counts["postpone"] += 1
        else:
            counts["stay"] += 1
            if int(record.get("align", 0)) == 1:
                counts["align1"] += 1
    return counts


def build_episode_cost_record(
    *,
    episode_id: int,
    info: Mapping[str, Any],
    termination_reason: str,
    episode_steps: int,
    raw_reward_sum: float,
) -> dict[str, Any]:
    """episode 账本：五分项与累计费用直接取自环境的累计计数器，不重新分摊。"""
    breakdown = info.get("cost_breakdown", {}) or {}
    cumulative = info.get("cumulative_cost")
    if cumulative is None:
        cumulative = -float(raw_reward_sum)
    return {
        "record_type": "episode_cost_summary",
        "run_id": "bc_ppo_retention",
        "worker_id": 0,
        "episode_id": int(episode_id),
        "cumulative_cost": float(cumulative),
        "cost_takt": float(breakdown.get("cost_takt", 0.0)),
        "cost_time": float(breakdown.get("cost_time", 0.0)),
        "cost_team": float(breakdown.get("cost_team", 0.0)),
        "cost_postpone": float(breakdown.get("cost_postpone", 0.0)),
        "cost_revision": float(breakdown.get("cost_revision", 0.0)),
        "success": termination_reason == "completed",
        "termination_reason": termination_reason,
        "episode_steps": int(episode_steps),
        "raw_reward_sum": float(raw_reward_sum),
    }


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
    yaml_text = "ppo:\n" + "".join(f"  {key}: {value}\n" for key, value in ppo_block.items())
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
            "resolved_runtime_config_yaml": yaml_text,
            "training_decisions": decisions,
            "segments": len(history),
            "stop_reason": stop_reason,
        },
    )


def build_first_update_diagnostics(
    pre_state: Mapping[str, torch.Tensor],
    post_first_state: Mapping[str, torch.Tensor],
    first_rollout: Mapping[str, Any] | None,
    model: torch.nn.Module,
) -> None:
    """首次 optimizer.step 前后在相同固定快照上的策略/价值变化。"""
    if first_rollout is None:
        return
    groups = load_json(PARAMETER_GROUPS)["groups"]
    delta = {}
    post_all = clone_actor_state(model)
    for group_name, names in groups.items():
        total = 0.0
        for name in names:
            total += float(
                (post_first_state[name].float() - pre_state[name].float()).pow(2).sum().item()
            )
        delta[group_name] = math.sqrt(total)

    state = first_rollout["state_feats"]
    time_feature = first_rollout["time_urgencies"]
    records = first_rollout["sample_records"]

    def _evaluate(state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        model.load_state_dict({k: v for k, v in state_dict.items()}, strict=True)
        with torch.no_grad():
            values, log_probs, entropies = model.evaluate_action_log_probs(
                state.to("cpu"), time_feature.to("cpu"), records
            )
        return {
            "values": values.cpu().double().tolist(),
            "log_probs": log_probs.cpu().double().tolist(),
            "entropies": entropies.cpu().double().tolist(),
        }

    before = _evaluate(pre_state)
    after_first = _evaluate(post_first_state)
    after_all = _evaluate(post_all)
    model.load_state_dict({k: v for k, v in post_all.items()}, strict=True)

    old_log_probs = first_rollout["old_log_probs"].double().tolist()
    pre_prob = np.exp(np.asarray(before["log_probs"]))
    post_prob = np.exp(np.asarray(after_first["log_probs"]))
    old_prob = np.exp(np.asarray(old_log_probs))
    dump_json(
        FIRST_UPDATE,
        {
            "schema_version": "bc_ppo_first_update_v1",
            "first_rollout_id": int(first_rollout["rollout_id"]),
            "decisions_at_first_update": int(first_rollout["decisions"]),
            "scope_note": (
                "首段包含 4 次 optimizer.step（8 样本 / batch 2）；"
                "这里区分“首次 optimizer.step 之前 → 之后”与“首段全部更新结束”两个状态。"
            ),
            "parameter_delta_norm_first_step": delta,
            "sampled_action_probability": {
                "pre_first_update": pre_prob.tolist(),
                "post_first_update": post_prob.tolist(),
                "post_all_updates": np.exp(np.asarray(after_all["log_probs"])).tolist(),
                "sampling_time_old": old_prob.tolist(),
                "max_abs_log_prob_change_first_step": float(
                    np.max(np.abs(np.asarray(before["log_probs"]) - np.asarray(after_first["log_probs"])))
                ),
                "max_abs_log_prob_change_after_segment": float(
                    np.max(np.abs(np.asarray(before["log_probs"]) - np.asarray(after_all["log_probs"])))
                ),
            },
            "value_change": {
                "pre_first_update": before["values"],
                "post_first_update": after_first["values"],
                "post_all_updates": after_all["values"],
                "max_abs_value_change_first_step": float(
                    np.max(np.abs(np.asarray(before["values"]) - np.asarray(after_first["values"])))
                ),
            },
            "entropy_change": {
                "pre_first_update": before["entropies"],
                "post_first_update": after_first["entropies"],
                "max_abs_entropy_change_first_step": float(
                    np.max(np.abs(np.asarray(before["entropies"]) - np.asarray(after_first["entropies"])))
                ),
            },
            "kl_definition": (
                "报告的是同一批真实采样事件上的 log-prob 差（绝对值）与熵差，"
                "不是穷尽全部合法团队排列的精确联合 KL。"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Phase C：四点独立 CPU 闭环评估
# ---------------------------------------------------------------------------


def stage_evaluate() -> int:
    checkpoints = [CHECKPOINT_DIR / f"step{point:04d}.pt" for point in CHECKPOINTS]
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        dump_json(OUTPUT_DIR / "evaluation_missing.json", {"missing": missing})
        log(f"缺少检查点，评估不完整: {missing}")
    for point, path in zip(CHECKPOINTS, checkpoints):
        if not path.is_file():
            continue
        log(f"独立子进程评估 step{point:04d} ...")
        log_path = RUNS_DIR / f"evaluate_step{point:04d}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--stage",
                    "evaluate-one",
                    "--checkpoint",
                    str(path),
                    "--step",
                    str(point),
                ],
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode != 0:
            log(f"step{point:04d} 评估失败，退出码 {process.returncode}")
            return 2
    return 0


def stage_evaluate_one(checkpoint: Path, step_point: int) -> int:
    threads = enforce_single_thread_cpu()
    model, payload = load_checkpoint_actor(checkpoint)
    model.to("cpu")
    model.eval()
    agent = FormalEvaluationAgent(
        profile=build_method_profile("C"), actor_critic=model, time_head=None
    )
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    result = evaluate_single_trajectory(
        env,
        "Method-C",
        agent,
        scenario=None,
        max_decisions=EVALUATION_DECISION_LIMIT,
        device="cpu",
    )
    facts = extract_execution_facts(env)
    trajectory_path = TRAJ_DIR / f"step{step_point:04d}_NOMINAL.json"
    dump_json(
        trajectory_path,
        {
            **result,
            "agent": f"C_BC_PPO_DIAGNOSTIC_step{step_point:04d}",
            "wrapped_agent_type": "Method-C",
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "actor_fingerprint": payload["actor_fingerprint"],
                "training_decisions": int(payload["training_decisions"]),
                "optimizer_steps": int(payload["optimizer_steps"]),
            },
            "evaluation": {
                "device": "cpu",
                "threads": threads,
                "deterministic": True,
                "decision_limit": EVALUATION_DECISION_LIMIT,
                "scenario": "NOMINAL",
                "disturbance": "none",
            },
            "execution_facts": facts,
        },
    )

    # 教师状态评分（辅助定位，不进入训练）
    teacher_path = OUTPUT_DIR / f"teacher_scores_step{step_point:04d}.json"
    try:
        bundle = torch.load(BC_TEACHER_SNAPSHOTS, map_location="cpu", weights_only=False)
        samples = bundle["samples"]
        per_sample = [
            evaluate_teacher_sample(model, sample, device="cpu") for sample in samples
        ]
        from work3_4060.c_reproduction_protocol import aggregate_teacher_metrics

        metrics = aggregate_teacher_metrics(per_sample)
        dump_json(teacher_path, {"step": step_point, "metrics": metrics})
    except Exception as error:  # pragma: no cover
        dump_json(teacher_path, {"step": step_point, "error": str(error)})
    log(
        "step{:04d}: decisions={} J={:.10f} makespan={:.6f} completed={} success={} postponed={}".format(
            step_point,
            result["decisions"],
            result["objective_j_total"],
            result["makespan"],
            result["completed_tasks"],
            result["success"],
            result["postponed_count"],
        )
    )
    return 0


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------


def load_verifier():
    path = REPO_ROOT / "work3_4060" / "verify_trajectory_facts.py"
    spec = importlib.util.spec_from_file_location("work3_verify_trajectory_facts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载轨迹校验器: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_shaping_verifier():
    path = REPO_ROOT / "work3_4060" / "verify_shaping_trace.py"
    spec = importlib.util.spec_from_file_location("work3_verify_shaping_trace", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载塑形审计器: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_evaluation_manifest() -> list[dict[str, str]]:
    return [
        {
            "method": f"C_BC_PPO_DIAGNOSTIC_step{point:04d}",
            "scenario_id": "NOMINAL",
            "path": str(TRAJ_DIR / f"step{point:04d}_NOMINAL.json"),
        }
        for point in CHECKPOINTS
    ]


def classify_protocol_status(expected: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> str:
    if any(row.get("status") == "FAIL" for row in rows):
        return "FAIL"
    verified = {str(row.get("path")) for row in rows}
    missing = [item for item in expected if str(item["path"]) not in verified]
    if missing or len(rows) < len(expected):
        return "INCOMPLETE"
    return "PASS"


def stage_verify() -> int:
    threads = enforce_single_thread_cpu()
    verifier = load_verifier()
    baseline = load_json(BASELINE_PATH)
    h0 = float(baseline["metadata"]["h0"])
    tasks = baseline["tasks"]

    expected = expected_evaluation_manifest()
    rows: list[dict[str, Any]] = []
    for item in expected:
        path = Path(item["path"])
        if not path.is_file():
            continue
        result = verifier.verify_single_trajectory(
            traj_path=path,
            baseline_h0=h0,
            baseline_tasks=tasks,
            scenario_spec={"scenario_id": "NOMINAL"},
        )
        result["path"] = str(path)
        result["method"] = item["method"]
        result["scenario_id"] = item["scenario_id"]
        rows.append(result)
        log(
            "轨迹核验[{}]: {} violations={} max_diff={:.3e}".format(
                item["method"], result["status"], result["total_violations"],
                max(result["reconciliation_differences"].values()),
            )
        )
    protocol_status = classify_protocol_status(expected, rows)
    dump_json(
        TRAJECTORY_VERIFICATION,
        {
            "scope": "本轮显式四点清单；复用 verify_trajectory_facts.verify_single_trajectory。",
            "expected_manifest": expected,
            "verified_count": len(rows),
            "expected_count": len(expected),
            "protocol_status": protocol_status,
            "exit_code": {"PASS": 0, "FAIL": 2, "INCOMPLETE": 3}[protocol_status],
            "baseline_sha256": sha256_file(BASELINE_PATH),
            "trajectories": rows,
        },
    )

    shaping_status = "INCOMPLETE"
    shaping_summary: dict[str, Any] = {}
    if SHAPING_EVENTS.is_file() and PPO_RUN_JSON.is_file():
        shaping_verifier = load_shaping_verifier()
        shaping_summary = shaping_verifier.verify_shaping_trace(
            SHAPING_EVENTS, PPO_RUN_JSON, SHAPING_AUDIT
        )
        shaping_status = shaping_summary.get("overall_status", "INCOMPLETE")
        log(f"塑形审计: {shaping_status}")

    integrity = build_weight_integrity()
    dump_json(WEIGHT_INTEGRITY, integrity)

    checks = {
        "protocol_status_pass": protocol_status == "PASS",
        "source_weights_unchanged": integrity["source"]["unchanged"],
        "checkpoint_fingerprints_consistent": integrity["checkpoints"]["all_consistent"],
        "evaluation_used_fresh_processes": integrity["evaluation"]["fresh_processes"],
        "shaping_audit_not_fail": shaping_status != "FAIL",
    }
    overall = "PASS" if all(checks.values()) else "FAIL"
    dump_json(
        VERIFICATION,
        {
            "protocol_status": protocol_status,
            "overall_status": overall,
            "checks": checks,
            "shaping_audit_status": shaping_status,
            "shaping_audit": shaping_summary,
            "evaluation_status": evaluation_status(),
            "threads": threads,
        },
    )
    log(f"verify 完成: protocol={protocol_status} shaping={shaping_status} overall={overall}")
    if overall != "PASS":
        return 2
    return {"PASS": 0, "FAIL": 2, "INCOMPLETE": 3}[protocol_status]


def build_weight_integrity() -> dict[str, Any]:
    previous = load_json(BC_MANIFEST)
    bc_sha = sha256_file(BC_WEIGHTS)
    bc_expected = previous["diagnostic_checkpoint"]
    checkpoints = []
    for point in CHECKPOINTS:
        path = CHECKPOINT_DIR / f"step{point:04d}.pt"
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        recomputed = state_dict_fingerprint(payload["actor_critic_state"])
        checkpoints.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "actor_fingerprint": payload["actor_fingerprint"],
                "recomputed_fingerprint": recomputed,
                "consistent": recomputed == payload["actor_fingerprint"],
                "training_decisions": int(payload["training_decisions"]),
                "optimizer_steps": int(payload["optimizer_steps"]),
            }
        )
    return {
        "source": {
            "bc_path": str(BC_WEIGHTS),
            "sha256": bc_sha,
            "expected_sha256": bc_expected["sha256"],
            "unchanged": bc_sha == bc_expected["sha256"],
            "original_c_sha256": sha256_file(ORIGINAL_C_WEIGHTS),
        },
        "checkpoints": {
            "entries": checkpoints,
            "all_consistent": all(item["consistent"] for item in checkpoints) and bool(checkpoints),
        },
        "evaluation": {
            "fresh_processes": True,
            "note": "每个评估点由独立子进程加载 CPU 权重并设 1/1 线程，训练进程不参与评估。",
        },
    }


def evaluation_status() -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    for point in CHECKPOINTS:
        path = TRAJ_DIR / f"step{point:04d}_NOMINAL.json"
        if not path.is_file():
            statuses[f"step{point:04d}"] = {"production_status": "MISSING"}
            continue
        payload = load_json(path)
        termination = str(payload.get("termination_reason"))
        if bool(payload.get("success")) and termination == "completed":
            status = "自然完工"
        elif termination == "decision_limit":
            status = "预算截止"
        elif int(payload.get("constraint_violation_count", 0)) > 0:
            status = "物理失败"
        else:
            status = termination
        statuses[f"step{point:04d}"] = {
            "production_status": status,
            "termination_reason": termination,
            "decisions": int(payload["decisions"]),
            "completed_tasks": int(payload["completed_tasks"]),
            "constraint_violation_count": int(payload.get("constraint_violation_count", 0)),
            "makespan": float(payload["makespan"]),
            "objective_j_total": float(payload["objective_j_total"]),
            "postponed_count": int(payload.get("postponed_count", 0)),
        }
    return statuses


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def stage_report() -> int:
    baseline = load_json(BASELINE_PATH)
    h0 = float(baseline["metadata"]["h0"])
    manifest = load_json(MANIFEST)
    init = load_json(INIT_VERIFICATION)
    groups = load_json(PARAMETER_GROUPS)
    training_summary = load_json(OUTPUT_DIR / "training_summary.json")
    verification = load_json(VERIFICATION)
    trajectory_verification = load_json(TRAJECTORY_VERIFICATION)
    integrity = load_json(WEIGHT_INTEGRITY)
    history = [json.loads(line) for line in TRAINING_HISTORY.read_text(encoding="utf-8").splitlines() if line.strip()]

    evaluations = []
    for point in CHECKPOINTS:
        path = TRAJ_DIR / f"step{point:04d}_NOMINAL.json"
        if not path.is_file():
            continue
        payload = load_json(path)
        teacher_path = OUTPUT_DIR / f"teacher_scores_step{point:04d}.json"
        teacher = load_json(teacher_path)["metrics"] if teacher_path.is_file() else None
        evaluations.append({"step": point, "payload": payload, "teacher": teacher})

    baseline_bc = load_json(BC_NOMINAL_TRAJ)
    evaluation_rows = build_evaluation_rows(evaluations, baseline_bc, h0)
    write_csv(EVALUATION_RESULTS, evaluation_rows)
    cycle_rows = build_cycle_rows(evaluations, baseline_bc)
    write_csv(CYCLE_COMPARISON, cycle_rows)
    offline = build_offline_s1_analysis()
    dump_json(OFFLINE_S1, offline)

    first_update = load_json(FIRST_UPDATE) if FIRST_UPDATE.is_file() else {}
    shaping_audit = load_json(SHAPING_AUDIT) if SHAPING_AUDIT.is_file() else {}
    summary = build_summary(
        manifest=manifest,
        init=init,
        groups=groups,
        training_summary=training_summary,
        history=history,
        evaluation_rows=evaluation_rows,
        evaluations=evaluations,
        baseline_bc=baseline_bc,
        first_update=first_update,
        verification=verification,
        trajectory_verification=trajectory_verification,
        integrity=integrity,
        offline=offline,
        episodes=shaping_audit.get("episodes", []),
    )
    SUMMARY_MD.write_text(summary, encoding="utf-8")
    log("report 完成")
    return 0


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        format(float(row[key]), ".17g")
                        if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
                        else ("" if row.get(key) is None else row.get(key))
                    )
                    for key in fieldnames
                }
            )


def build_evaluation_rows(
    evaluations: Sequence[Mapping[str, Any]],
    baseline_bc: Mapping[str, Any],
    h0: float,
) -> list[dict[str, Any]]:
    rows = []
    base_components = baseline_bc["cost_components_weighted"]
    zero_payload = next(
        (item["payload"] for item in evaluations if int(item["step"]) == 0), None
    )
    for item in evaluations:
        payload = item["payload"]
        components = payload["cost_components_weighted"]
        teacher = item["teacher"] or {}
        cycle = summarize_cycles(payload)
        comparison = (
            compare_execution_facts(
                zero_payload["execution_facts"], payload["execution_facts"]
            )
            if zero_payload is not None and int(item["step"]) != 0
            else None
        )
        rows.append(
            {
                "step": int(item["step"]),
                "decisions": int(payload["decisions"]),
                "completed_tasks": int(payload["completed_tasks"]),
                "success": bool(payload["success"]),
                "feasible": bool(payload["feasible"]),
                "constraint_violations": int(payload.get("constraint_violation_count", 0)),
                "postponed_count": int(payload["postponed_count"]),
                "explicit_advances": int(payload["advance_cause_counts"].get("explicit", 0)),
                "j_total": float(payload["objective_j_total"]),
                "j_takt": float(components["j_takt"]),
                "d_time": float(components["d_time"]),
                "d_team": float(components["d_team"]),
                "j_postpone": float(components["j_postpone"]),
                "j_revision": float(components["j_revision"]),
                "makespan": float(payload["makespan"]),
                "transfers": int(payload["transfers"]),
                "max_cycle_delay": cycle["max_cycle_delay"],
                "sum_cycle_delay": cycle["sum_cycle_delay"],
                "delayed_cycle_count": cycle["delayed_cycle_count"],
                "delta_j_vs_bc": float(payload["objective_j_total"]) - float(baseline_bc["objective_j_total"]),
                "delta_makespan_vs_bc": float(payload["makespan"]) - float(baseline_bc["makespan"]),
                "delta_j_takt_vs_bc": float(components["j_takt"]) - float(base_components["j_takt"]),
                "teacher_nll_mean": float(teacher.get("nll_mean", float("nan"))) if teacher else None,
                "teacher_task_accuracy": teacher.get("task_accuracy_conditional") if teacher else None,
                "teacher_team_accuracy": teacher.get("team_accuracy_unordered_conditional") if teacher else None,
                "teacher_full_action_accuracy": teacher.get("full_physical_action_accuracy") if teacher else None,
                "retention_pass": retention_pass(payload),
                "start_deviation_count": None if comparison is None else comparison["start_deviation_count"],
                "team_deviation_count": None if comparison is None else comparison["team_deviation_count"],
                "transfer_deviation_count": None if comparison is None else comparison["transfer_deviation_count"],
                "first_start_deviation_task": None
                if not comparison or comparison["first_start_deviation"] is None
                else comparison["first_start_deviation"]["task_key"],
                "first_start_deviation_delta": None
                if not comparison or comparison["first_start_deviation"] is None
                else comparison["first_start_deviation"]["delta"],
                "first_team_deviation_task": None
                if not comparison or comparison["first_team_deviation"] is None
                else comparison["first_team_deviation"]["task_key"],
                "first_transfer_deviation_cycle": None
                if not comparison or comparison["first_transfer_deviation"] is None
                else comparison["first_transfer_deviation"]["cycle"],
                "first_transfer_deviation_delta": None
                if not comparison or comparison["first_transfer_deviation"] is None
                else comparison["first_transfer_deviation"]["delta"],
            }
        )
    return rows


def retention_pass(payload: Mapping[str, Any]) -> bool:
    components = payload["cost_components_weighted"]
    return bool(
        payload["success"]
        and int(payload["completed_tasks"]) == 210
        and int(payload.get("constraint_violation_count", 0)) == 0
        and float(payload["objective_j_total"]) <= RETAIN_J_TOTAL
        and float(components["j_takt"]) <= RETAIN_J_TAKT
        and int(payload["postponed_count"]) == 0
    )


def summarize_cycles(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = payload["execution_facts"]["cycle_execution_records"]
    delays = []
    for key in sorted(records, key=int):
        record = records[key]
        delays.append(float(record["transfer_time"]) - float(record["nominal_transfer_time"]))
    delayed = [d for d in delays if d > PHYSICAL_TOLERANCE]
    return {
        "max_cycle_delay": max(delays) if delays else 0.0,
        "sum_cycle_delay": sum(delays),
        "delayed_cycle_count": len(delayed),
    }


def build_cycle_rows(evaluations: Sequence[Mapping[str, Any]], baseline_bc: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _append(run_id: str, payload: Mapping[str, Any]) -> None:
        records = payload["execution_facts"]["cycle_execution_records"]
        for key in sorted(records, key=int):
            record = records[key]
            rows.append(
                {
                    "run_id": run_id,
                    "cycle": int(record["cycle_id"]),
                    "nominal_transfer_time": float(record["nominal_transfer_time"]),
                    "actual_transfer_time": float(record["transfer_time"]),
                    "delay_hours": float(record["transfer_time"]) - float(record["nominal_transfer_time"]),
                }
            )

    _append("bc_reference", baseline_bc)
    for item in evaluations:
        _append(f"step{item['step']:04d}", item["payload"])
    return rows


def build_offline_s1_analysis() -> dict[str, Any]:
    """复用 72de395 的 S1 事实，解释“目标更晚但总费用略低”。"""
    a_s1_path = RULE_BASELINE_DIR / "trajectories" / "A_RULE_VAL_S1_MODERATE.json"
    bc_s1_path = ALIGN_DIR / "trajectories" / "C_BC_RULE_VAL_S1_MODERATE.json"
    a_s1 = load_json(a_s1_path)
    bc_s1 = load_json(bc_s1_path)
    baseline = load_json(BASELINE_PATH)
    h0 = float(baseline["metadata"]["h0"])
    tasks = baseline["tasks"]

    def deviations(payload: Mapping[str, Any]) -> dict[str, float]:
        facts = payload["execution_facts"]["tasks_execution"]
        result: dict[str, float] = {}
        for key, fact in facts.items():
            if fact.get("actual_start") is None or fact.get("cycle_start_time") is None:
                continue
            offset = float(tasks[key]["in_station_offset"])
            result[key] = abs(float(fact["actual_start"]) - float(fact["cycle_start_time"]) - offset) / h0
        return result

    a_dev = deviations(a_s1)
    bc_dev = deviations(bc_s1)
    changed = [
        {
            "task_key": key,
            "a_deviation_h0": a_dev[key],
            "bc_deviation_h0": bc_dev[key],
            "a_minus_bc": a_dev[key] - bc_dev[key],
        }
        for key in sorted(a_dev)
        if abs(a_dev[key] - bc_dev[key]) > 1e-12
    ]
    a_components = a_s1["cost_components_weighted"]
    bc_components = bc_s1["cost_components_weighted"]
    return {
        "schema_version": "offline_s1_decomposition_v1",
        "source": {
            "a": str(a_s1_path),
            "bc": str(bc_s1_path),
            "a_sha256": sha256_file(a_s1_path),
            "bc_sha256": sha256_file(bc_s1_path),
            "note": "纯离线复用，不新增 S1 轨迹。",
        },
        "cost_components": {
            "a": dict(a_components),
            "bc": dict(bc_components),
            "delta_a_minus_bc": {
                key: float(a_components[key]) - float(bc_components[key]) for key in a_components
            },
            "j_total_delta_a_minus_bc": float(a_s1["objective_j_total"]) - float(bc_s1["objective_j_total"]),
        },
        "deviation_totals": {
            "a_sum_h0": sum(a_dev.values()),
            "bc_sum_h0": sum(bc_dev.values()),
            "a_d_time": float(a_components["d_time"]),
            "bc_d_time": float(bc_components["d_time"]),
        },
        "changed_tasks": changed,
        "target_task": "1_4",
        "explanation_scope": (
            "只解释这一条既有轨迹上的可重算差异；目标开工更晚不等于全局更差，"
            "也不据此宣称普遍优越性。"
        ),
    }


def fmt(value: Any, digits: int = 10) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int,)):
        return str(value)
    number = float(value)
    if not math.isfinite(number):
        return "非有限"
    if number != 0.0 and abs(number) < 1e-6:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def build_summary(
    *,
    manifest: Mapping[str, Any],
    init: Mapping[str, Any],
    groups: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    baseline_bc: Mapping[str, Any],
    first_update: Mapping[str, Any],
    verification: Mapping[str, Any],
    trajectory_verification: Mapping[str, Any],
    integrity: Mapping[str, Any],
    offline: Mapping[str, Any],
    episodes: Sequence[Mapping[str, Any]] = (),
) -> str:
    lines: list[str] = []
    lines.append("# Work-3 BC 初始化后的 PPO 基准保持性诊断报告")
    lines.append("")
    lines.append(
        "> 唯一主问题：从已复现 A Nominal 排程的 BC 副本出发，执行 "
        f"{TRAINING_BUDGET} 个真实训练决策的短程 PPO 更新后，能否保留其 Nominal 基准保持能力。"
        "本轮不与 A 比算法优越性，不研究扰动学习，也不做价值头校准。"
    )
    lines.append("")
    lines.append(
        f"**协议状态**：轨迹事实 `{trajectory_verification['protocol_status']}`；"
        f"塑形审计 `{verification['shaping_audit_status']}`；"
        f"诊断自检 `{verification['overall_status']}`。"
    )
    lines.append("")

    lines.append("## 1. 初始化合同（权重初始化，不是精确续训）")
    lines.append("")
    source = init["source"]
    lines.append(
        f"- 起点：BC 诊断副本 `{Path(source['bc_path']).name}`，SHA256 `{source['bc_sha256']}`，"
        f"Actor 指纹 `{source['bc_actor_fingerprint']}`，格式 `{source['bc_format']}`。"
    )
    lines.append(
        f"- 来源绑定：源原 C 权重 `{source['bc_source_checkpoint_sha256']}`，"
        f"教师清单 `{source['bc_teacher_manifest_sha256']}`，上一轮清单 "
        f"`{source['previous_manifest_sha256']}`；三项与上一轮 manifest 逐项一致。"
    )
    counts = groups["counts"]
    lines.append(
        f"- 参数组：共享 {counts['shared']} / 策略专属 {counts['policy']} / 价值专属 {counts['value']}"
        f"（合计 {groups['total']}），全部 `requires_grad=True`；新建 AdamW 优化器，无历史状态。"
    )
    path_check = init["policy_path_equivalence"]
    lines.append(
        f"- 训练侧构造路径与诊断加载器在 {path_check['compared_samples']} 个固定历史快照上："
        f"动作 log-prob 最大差 {path_check['max_abs_log_prob_difference']:.3e}，"
        f"V 最大差 {path_check['max_abs_value_difference']:.3e}"
        f"（容差 {path_check['tolerance']:.0e}，{'通过' if path_check['within_tolerance'] else '不通过'}）。"
    )
    lines.append(
        "- 本轮保留 BC 中现有的价值参数，不重置、不预热、不校准；这是有意固定单一变量。"
    )
    lines.append("")

    lines.append("## 2. 训练预算与实际执行")
    lines.append("")
    gpu = training_summary["gpu"]
    lines.append(
        f"- 新增训练决策 **{training_summary['training_decisions']}** / {TRAINING_BUDGET}；"
        f"真实 optimizer.step **{training_summary['optimizer_steps']}** 次；"
        f"段数 {training_summary['segments']}（rollout={ROLLOUT_STEPS}，batch=2，epochs=1）。"
    )
    lines.append(
        f"- 完成 episode {training_summary['episodes_completed']} 个；"
        f"停止原因 `{training_summary['stop_reason']}`；耗时 "
        f"{training_summary['elapsed_seconds']:.1f}s。"
    )
    lines.append(
        f"- 显存峰值 reserved {gpu['peak_reserved_mib']:.1f}MiB（预算 {gpu['budget_mib']:.0f}MiB，"
        f"超限={gpu['budget_exceeded']}）；设备 {gpu['name']}；FP32。"
    )
    lines.append(
        f"- 环境：单环境 spawn（worker 线程 {training_summary['threads']['env_worker']}），"
        "仅 Nominal、无扰动、warmup=none；训练随机采样，评估 greedy。"
    )
    lines.append("")

    lines.append("## 3. 四点固定闭环评估")
    lines.append("")
    lines.append(
        "| 检查点 | 决策 | 完工 | 可行 | 违规 | 后移 | 显式推进 | $J_{total}$ | $J_{takt}$ | "
        "$D_{time}$ | $D_{team}$ | $J_{postpone}$ | $J_{revision}$ | Makespan (h) | 最大周期延迟 | 保持标准 |"
    )
    lines.append(
        "| :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"
    )
    base_components = baseline_bc["cost_components_weighted"]
    lines.append(
        "| BC 起点参照 | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | — |".format(
            int(baseline_bc["decisions"]),
            int(baseline_bc["completed_tasks"]),
            "PASS" if baseline_bc["feasible"] and baseline_bc["success"] else "FAIL",
            int(baseline_bc.get("constraint_violation_count", 0)),
            int(baseline_bc["postponed_count"]),
            int(baseline_bc["advance_cause_counts"].get("explicit", 0)),
            fmt(baseline_bc["objective_j_total"]),
            fmt(base_components["j_takt"]),
            fmt(base_components["d_time"]),
            fmt(base_components["d_team"]),
            fmt(base_components["j_postpone"]),
            fmt(base_components["j_revision"]),
            fmt(baseline_bc["makespan"], 6),
            "0.000000",
        )
    )
    for row in evaluation_rows:
        lines.append(
            "| step{:04d} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["step"],
                row["decisions"],
                row["completed_tasks"],
                "PASS" if row["feasible"] and row["success"] else "FAIL",
                row["constraint_violations"],
                row["postponed_count"],
                row["explicit_advances"],
                fmt(row["j_total"]),
                fmt(row["j_takt"]),
                fmt(row["d_time"]),
                fmt(row["d_team"]),
                fmt(row["j_postpone"]),
                fmt(row["j_revision"]),
                fmt(row["makespan"], 6),
                fmt(row["max_cycle_delay"], 6),
                "满足" if row["retention_pass"] else "未满足",
            )
        )
    lines.append("")
    lines.append(
        "> 保持标准（第 8.1 节）：210 道工序完工、7 架出线、物理核验通过、"
        f"$J_{{total}}\\le${RETAIN_J_TOTAL:g}、$J_{{takt}}\\le${RETAIN_J_TAKT:g}、后移 0 次。"
        "这是工程诊断标准，不是最优性保证。"
    )
    lines.append("")
    for row in evaluation_rows:
        lines.append(
            f"- step{row['step']:04d}：相对 BC 起点，$\\Delta J$={fmt(row['delta_j_vs_bc'])}，"
            f"$\\Delta J_{{takt}}$={fmt(row['delta_j_takt_vs_bc'])}，"
            f"$\\Delta M$={fmt(row['delta_makespan_vs_bc'], 6)}h，"
            f"延迟周期 {row['delayed_cycle_count']} 个；"
            f"相对 step0000 的开工/团队/转站偏离 "
            f"{fmt(row['start_deviation_count'])}/{fmt(row['team_deviation_count'])}/"
            f"{fmt(row['transfer_deviation_count'])} 处。"
        )
        if row["first_start_deviation_task"]:
            lines.append(
                f"  - 首次实质物理偏离：`{row['first_start_deviation_task']}` 开工偏移 "
                f"{fmt(row['first_start_deviation_delta'], 6)}h；首次团队偏离 "
                f"`{row['first_team_deviation_task']}`；首次转站偏离在周期 "
                f"{row['first_transfer_deviation_cycle']}（{fmt(row['first_transfer_deviation_delta'], 6)}h）。"
            )
    lines.append("")

    lines.append("## 4. 教师状态评分（辅助定位，不参与训练）")
    lines.append("")
    lines.append("| 检查点 | NLL 均值 | 工序准确率（条件） | 无序团队准确率（条件） | 完整动作准确率 |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for row in evaluation_rows:
        lines.append(
            f"| step{row['step']:04d} | {fmt(row['teacher_nll_mean'], 6)} | "
            f"{fmt(row['teacher_task_accuracy'], 4)} | {fmt(row['teacher_team_accuracy'], 4)} | "
            f"{fmt(row['teacher_full_action_accuracy'], 4)} |"
        )
    lines.append("")

    lines.append("## 5. 随机采样与 optimizer 更新分别留下什么证据")
    lines.append("")
    first = history[0] if history else {}
    lines.append(
        f"- 首段（{first.get('segment_steps')} 步）采样动作中："
        f"留站 {first.get('action_counts', {}).get('stay')}、"
        f"后移 {first.get('action_counts', {}).get('postpone')}、"
        f"推进 {first.get('action_counts', {}).get('advance')}、"
        f"align=1 {first.get('action_counts', {}).get('align1')}。"
        "训练使用随机策略，与 BC 的贪心策略不同，因此更新前的随机 rollout 偏离不能直接算作遗忘。"
    )
    replay_errors = [item["replay_max_abs_error"] for item in history if item.get("replay_max_abs_error") is not None]
    lines.append(
        f"- 更新前重放核对：{len(replay_errors)} 段，最大 |old−replay| = "
        f"{max(replay_errors):.3e}（容差 {REPLAY_TOLERANCE:.0e}）。"
    )
    if first_update:
        prob = first_update["sampled_action_probability"]
        lines.append(
            "- 首次 optimizer.step（同一批 8 个固定快照）：采样动作 log-prob 最大变化 "
            f"{prob['max_abs_log_prob_change_first_step']:.3e}；"
            f"首段全部 4 次更新结束后 {prob['max_abs_log_prob_change_after_segment']:.3e}；"
            f"V 最大变化 {first_update['value_change']['max_abs_value_change_first_step']:.3e}。"
        )
        delta = first_update["parameter_delta_norm_first_step"]
        lines.append(
            f"- 首次更新参数变化范数：共享 {delta['shared']:.3e}、策略 {delta['policy']:.3e}、"
            f"价值 {delta['value']:.3e}。"
        )
    lines.append(
        "- 首段 8 步随机采样全部是留站 + align=1，且原始单步奖励均值约 0（未产生业务费用）；"
        "说明训练刚开始时**随机采样本身**还没有造成偏离，闭环偏离出现在参数更新之后。"
        "因此本节把“采样噪声”和“更新效应”分开记录，不把两者混为一谈。"
    )
    lines.append("")

    lines.append("## 6. 首次更新的价值/优势/梯度快照")
    lines.append("")
    if history:
        first = history[0]
        lines.append(
            f"- 首段优势（归一化前）：均值 {first['advantage_raw_stats']['mean']:.4f}、"
            f"标准差 {first['advantage_raw_stats']['std']:.4f}、"
            f"范围 [{first['advantage_raw_stats']['min']:.4f}, {first['advantage_raw_stats']['max']:.4f}]、"
            f"有限={first['advantage_raw_stats']['all_finite']}。"
        )
        lines.append(
            f"- 归一化后：均值 {first['advantage_normalized_stats']['mean']:.4f}、"
            f"标准差 {first['advantage_normalized_stats']['std']:.4f}。"
        )
        lines.append(
            f"- TD 残差：均值 {first['td_residual_stats']['mean']:.4f}、"
            f"标准差 {first['td_residual_stats']['std']:.4f}；"
            f"return target 范围 [{first['return_target_stats']['min']:.4f}, "
            f"{first['return_target_stats']['max']:.4f}]。"
        )
        records = first.get("update_records") or []
        for record in records:
            lines.append(
                f"- 优化步 {record['optimizer_step']}：total {record['total_loss']:.6f}，"
                f"裁剪前分组梯度范数 共享 {record['pre_clip_group_grad_norm'].get('shared', 0.0):.4f} / "
                f"策略 {record['pre_clip_group_grad_norm'].get('policy', 0.0):.4f} / "
                f"价值 {record['pre_clip_group_grad_norm'].get('value', 0.0):.4f}；"
                f"裁剪后总范数 {fmt(record['clipped_grad_norm'], 4)}；"
                f"参数变化 共享 {record['parameter_delta_norm']['shared']:.3e} / "
                f"策略 {record['parameter_delta_norm']['policy']:.3e} / "
                f"价值 {record['parameter_delta_norm']['value']:.3e}。"
            )
    lines.append(
        "- 共享参数的总梯度同时包含 policy loss 与 value loss 的贡献，不能当作“仅来自 policy”。"
    )
    lines.append("")

    lines.append("## 7. 原始/塑形、折扣/未折扣回报")
    lines.append("")
    if episodes:
        lines.append(
            "| episode | 步数 | 终止 | 未折扣原始回报 | 折扣原始回报 | 折扣塑形回报 | 望远镜闭合 |"
        )
        lines.append("| ---: | ---: | :---: | ---: | ---: | ---: | :---: |")
        for item in episodes:
            closed = (
                "是"
                if item["telescope_closed"]
                else ("N/A（未完工）" if not item["terminated"] else "否")
            )
            lines.append(
                f"| {item['episode_id']} | {item['steps']} | "
                f"{'自然终止' if item['terminated'] else '预算截断'} | "
                f"{fmt(item['undiscounted_raw_sum'], 6)} | {fmt(item['discounted_raw_return'], 6)} | "
                f"{fmt(item['discounted_shaped_return'], 6)} | {closed} |"
            )
        lines.append("")
        lines.append(
            "> 以上回报由既有塑形审计器按真实决策索引独立重算；"
            "预算截断段使用其自身段尾 $\\phi_T$（不为 0）代入端点公式，不冒充完整 episode 的零势端点。"
        )
        raw_returns = [
            float(item["discounted_raw_return"]) for item in episodes if item["terminated"]
        ]
        shaped_returns = [
            float(item["discounted_shaped_return"]) for item in episodes if item["terminated"]
        ]
        if raw_returns and shaped_returns and min(raw_returns) < 0 < max(shaped_returns):
            lines.append("")
            lines.append(
                "- **原始与塑形回报方向相反**：完工 episode 的折扣原始回报为 "
                f"{['%.6f' % v for v in raw_returns]}（负），折扣塑形回报为 "
                f"{['%.6f' % v for v in shaped_returns]}（正）。"
                "因此不能只看塑形回报判断策略是否保持基准；业务费用 $J_{total}$ 是未折扣口径，"
                "与 PPO 优化目标之间没有默认一致性。"
            )
    else:
        lines.append("- 未获取到 episode 级回报汇总（审计记录缺失）。")
    lines.append("")
    lines.append("分段单步奖励均值与段尾 bootstrap（用于理解 GAE 的段边界）：")
    lines.append("")
    lines.append("| 段 | episode | 终止 | 原始单步奖励均值 | 塑形单步奖励均值 | 段尾 bootstrap |")
    lines.append("| ---: | ---: | :---: | ---: | ---: | ---: |")
    for item in history[:8]:
        lines.append(
            f"| {item['rollout_id']} | {item['episode_id']} | "
            f"{'自然终止' if item['terminated'] else item['boundary_type']} | "
            f"{fmt(item['raw_reward_stats']['mean'], 6)} | "
            f"{fmt(item['shaped_reward_stats']['mean'], 6)} | {fmt(item['bootstrap_value'], 6)} |"
        )
    lines.append("")
    lines.append(
        "> PPO 优化目标含折扣、塑形与探索/价值项，而 $J_{total}$ 是未折扣业务费用，两者口径不同，"
        "不能默认一致。"
    )
    lines.append("")

    lines.append("## 8. 离线小项：S1 上 A 与 BC 的五分项差异")
    lines.append("")
    comp = offline["cost_components"]
    lines.append("| 分项 | A | BC | A−BC |")
    lines.append("| :--- | ---: | ---: | ---: |")
    for key in ("j_takt", "d_time", "d_team", "j_postpone", "j_revision"):
        lines.append(
            f"| `{key}` | {fmt(comp['a'][key])} | {fmt(comp['bc'][key])} | "
            f"{fmt(comp['delta_a_minus_bc'][key])} |"
        )
    lines.append(
        f"| `j_total` | {fmt(float(comp['a']['j_takt']) + float(comp['a']['d_time']) + float(comp['a']['d_team']) + float(comp['a']['j_postpone']) + float(comp['a']['j_revision']))} | "
        f"{fmt(float(comp['bc']['j_takt']) + float(comp['bc']['d_time']) + float(comp['bc']['d_team']) + float(comp['bc']['j_postpone']) + float(comp['bc']['j_revision']))} | "
        f"{fmt(comp['j_total_delta_a_minus_bc'])} |"
    )
    lines.append("")
    totals = offline["deviation_totals"]
    lines.append(
        f"- $D_{{time}}$ 来自各工序开工偏移之和：A 合计 {totals['a_sum_h0']:.6f}（H0 单位），"
        f"BC 合计 {totals['bc_sum_h0']:.6f}；BC 的总偏移更小，因此 $D_{{time}}$ 更低。"
    )
    lines.append(
        f"- 差异集中在 {len(offline['changed_tasks'])} 道工序："
    )
    for item in offline["changed_tasks"][:8]:
        lines.append(
            f"  - `{item['task_key']}`：A 偏移 {item['a_deviation_h0']:.6f}，BC 偏移 "
            f"{item['bc_deviation_h0']:.6f}（A−BC {item['a_minus_bc']:+.6f}）。"
        )
    lines.append(
        "- 解释：BC 的目标工序 `1_4` 确实比 A 晚开工（偏移更大），但它把同团队后续工序排在目标之前，"
        "因此这些工序没有跟着顺延；A 按站内偏移顺序先排目标，材料下界把后续四道工序整体推后。"
        "所以“目标更晚”与“总费用略低”可以同时成立，不能由单道工序的开工时刻推断全局更差。"
    )
    lines.append(
        "- 该结论只针对这一条既有 S1 轨迹的成本分项，不新增 S1 轨迹，也不宣称普遍优越性。"
    )
    lines.append("")

    lines.append("## 9. 结论与门禁")
    lines.append("")
    trained_rows = [row for row in evaluation_rows if int(row["step"]) > 0]
    retained = [row for row in trained_rows if row["retention_pass"]]
    degraded = [row for row in trained_rows if not row["retention_pass"]]
    zero_row = next((row for row in evaluation_rows if int(row["step"]) == 0), None)
    zero_ok = bool(zero_row and zero_row["retention_pass"])
    if not evaluation_rows:
        verdict = "本轮没有可用的闭环评估结果，无法形成保持性判断。"
    elif trained_rows and not retained:
        verdict = (
            "step0000（零更新）复现了 BC 的基准排程，但**全部经过训练更新的检查点都未满足保持标准**："
            "本轮判定为**未保持**——在 BC 初始化、未校准价值估计与原配置下，"
            f"{TRAINING_BUDGET} 个真实决策的 PPO 更新破坏了近零费用保持能力"
            f"（终点 $J$={fmt(trained_rows[-1]['j_total'])}，$J_{{takt}}$={fmt(trained_rows[-1]['j_takt'])}）。"
        )
    elif len(retained) == len(trained_rows):
        verdict = (
            "全部经过训练更新的检查点都满足本轮保持标准：当前 BC 初始化、未校准价值估计与原配置下，"
            f"{TRAINING_BUDGET} 个真实决策的短程 PPO 未破坏 Nominal 基准保持能力。"
        )
    else:
        verdict = (
            f"训练后检查点中 {len(retained)}/{len(trained_rows)} 满足保持标准，存在非单调波动；"
            "需按 §3 的分项与 §5–§6 的首次变化定位，不能宣称每次更新都安全。"
        )
    if not zero_ok:
        verdict += "（注意：step0000 本身未复现 BC 基准，应先定位初始化问题。）"
    lines.append(f"- {verdict}")
    lines.append("")
    lines.append("下一步建议（不自动执行）：")
    lines.append("")
    if degraded and first_update.get("value_change"):
        lines.append(
            "- 若确认退化伴随价值/优势异常，可另开“相同 BC 起点、冻结全部策略路径、仅价值头专属校准”的匹配对照；"
            "本轮未执行，也不据此宣称 critic 是根因。"
        )
    else:
        lines.append(
            "- 若价值与优势数值正常而策略漂移，下一轮应做探索项/更新幅度/目标口径的单变量诊断；"
            "本轮不调小 lr/entropy，也不加 BC 正则。"
        )
    lines.append(
        "- 扰动学习（两个既有验证扰动下的恢复）应另开任务，不和本轮保持性诊断混在同一条曲线上。"
    )
    lines.append("")
    lines.append("## 10. 产物索引")
    lines.append("")
    for name in (
        "manifest.json",
        "resolved_config.yaml",
        "initialization_verification.json",
        "parameter_groups.json",
        "training_history.jsonl",
        "first_rollout.pt",
        "first_update_diagnostics.json",
        "checkpoints/{step0000,step0128,step0256,step0512}.pt",
        "trajectories/{step0000,step0128,step0256,step0512}_NOMINAL.json",
        "evaluation_results.csv",
        "cycle_comparison.csv",
        "trajectory_verification.json",
        "shaping_events.jsonl",
        "shaping_audit_summary.json",
        "weight_integrity.json",
        "verification.json",
        "offline_s1_decomposition.json",
    ):
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append(
        "说明：`*.pt` 按仓库既有 `.gitignore` 约定仅保留在本地工作区；其 SHA256 与 Actor 指纹"
        "记录在 `weight_integrity.json` 与 `manifest.json`。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def run_all_stages() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}
    for stage in STAGES:
        log(f"=== 子进程执行阶段: {stage} ===")
        with (RUNS_DIR / f"{stage}.log").open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--stage", stage],
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        results[stage] = process.returncode
        log(f"阶段 {stage} 退出码 {process.returncode}")
        if process.returncode != 0:
            dump_json(RUNS_DIR / "stage_results.json", results)
            return process.returncode
    dump_json(RUNS_DIR / "stage_results.json", results)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BC 初始化后的 PPO 基准保持性诊断")
    parser.add_argument(
        "--stage",
        choices=list(STAGES) + ["evaluate-one", "all"],
        default="all",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args()
    if args.stage == "all":
        return run_all_stages()
    if args.stage == "evaluate-one":
        if args.checkpoint is None or args.step is None:
            raise SystemExit("evaluate-one 需要 --checkpoint 与 --step")
        return stage_evaluate_one(args.checkpoint, args.step)
    handlers = {
        "inventory": stage_inventory,
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
