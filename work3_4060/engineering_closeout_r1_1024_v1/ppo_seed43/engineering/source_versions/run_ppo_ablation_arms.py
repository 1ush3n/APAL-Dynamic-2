"""Work-3 PPO 保持性消融臂（R0–R4）：共用同一校准副本，每臂只改一项。

R0 冻结采样：不更新参数，正常随机采样 512 步，测“仅随机动作”的费用。
R1 学习率 ×0.1（全部参数组统一）。
R2 entropy coefficient = 0。
R3 优势只中心化：保留减均值、取消标准差缩放（归一化范围=整段 rollout，与正式实现一致）。
R4 价值分支在共享特征处 detach：value loss 不再流入共享编码器，策略仍更新编码器，价值头仍训练。

固定评估点 0/8/128/256/512；8 = 首个 rollout 全部更新后（不称“第一次 optimizer.step 后”）。
所有臂串行执行，各自新建优化器并恢复相同初始随机状态。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
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
from models.work3.ppo_buffer import RolloutBufferWork3
from scripts.work3.evaluate_c_vs_d import FormalEvaluationAgent, evaluate_single_trajectory
from scripts.work3.experiment_protocol import build_method_profile
from scripts.work3.train_ppo_work3 import compute_heuristic_potential
from training.work3_lightning import Work3TrainingUpdate
from work3_4060.c_reproduction_protocol import (
    actor_fingerprint,
    compare_execution_facts,
    dump_json,
    extract_execution_facts,
    load_json,
    sha256_file,
)
import work3_4060.run_critic_calibration_retention as cal
import work3_4060.run_bc_ppo_retention as prev

BASE_DIR = REPO_ROOT / "work3_4060"
CAL_ROUND = BASE_DIR / "critic_calibration_retention_v1" / "seed42"
OUTPUT_DIR = BASE_DIR / "ppo_ablation_arms_v1" / "seed42"
RUNS_DIR = OUTPUT_DIR / "runs"

CAL_MODEL = CAL_ROUND / "calibration" / "calibrated_model.pt"
CAL_FIT = CAL_ROUND / "calibration" / "fit_metrics.json"
REFERENCE_PPO = CAL_ROUND / "ppo"
REFERENCE_SUMMARY = REFERENCE_PPO / "training_summary.json"
REFERENCE_HISTORY = REFERENCE_PPO / "training_history.jsonl"
REFERENCE_PAIRING = REFERENCE_PPO / "first_rollout_pairing.json"
REFERENCE_FIRST_ROLLOUT = REFERENCE_PPO / "first_rollout.pt"
REFERENCE_MINIBATCH_UPDATE: Path | None = None
BASELINE_PATH = BASE_DIR / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"

MANIFEST = OUTPUT_DIR / "manifest.json"
ARM_REGISTRY = OUTPUT_DIR / "arm_registry.json"
R0_RESULT = OUTPUT_DIR / "r0_frozen_sampling.json"
COMPARISON_CSV = OUTPUT_DIR / "comparison_by_point.csv"
CYCLE_CSV = OUTPUT_DIR / "cycle_comparison.csv"
DIAGNOSTICS_CSV = OUTPUT_DIR / "update_diagnostics.csv"
VERIFICATION = OUTPUT_DIR / "verification.json"
SUMMARY_MD = OUTPUT_DIR / "summary.md"

ARMS: tuple[str, ...] = ("R0", "R1", "R2", "R3", "R4")
POINTS: tuple[int, ...] = (0, 8, 128, 256, 512)
BUDGET = 512
LR_SCALE = 0.1
RUN_SEED: int | None = None
# 配对判据：状态与动作必须逐项一致；“采样事件相同”由这两项定义。
# log-prob 是 float32 在 GPU 上非确定性归约的产物，容差与仓库既有重放容差一致取 1e-4。
PAIRING_LOG_PROB_TOLERANCE = 1e-4
PAIRING_TOLERANCE_NOTE = (
    "判据以“状态逐元素相等 + 动作签名逐项相等”为主；log-prob 只做 1e-4 的数值一致性检查。"
    "CUDA 归约顺序不确定，同一策略同一步的 log-prob 可有 ~1e-5 量级抖动，"
    "这与仓库既有的重放一致性容差 1e-4 同量级，不代表采样事件不同。"
)

ARM_SPECS: dict[str, dict[str, Any]] = {
    "R0": {"change": "frozen_sampling_no_update", "frozen": True},
    "R1": {"change": "learning_rate_x0.1", "lr_scale": LR_SCALE},
    "R2": {"change": "entropy_coefficient_0", "entropy_coefficient": 0.0},
    "R3": {"change": "advantage_center_only", "advantage_mode": "center_only"},
    "R4": {"change": "value_loss_detached_at_shared_features", "detach_value": True},
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


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
    return {"head": head, "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "worktree_dirty": bool(status)}


def arm_dir(arm: str) -> Path:
    return OUTPUT_DIR / arm


def arm_paths(arm: str) -> dict[str, Path]:
    base = arm_dir(arm)
    return {
        "dir": base,
        "checkpoints": base / "checkpoints",
        "trajectories": base / "trajectories",
        "history": base / "training_history.jsonl",
        "first_update": base / "first_update.json",
        "first_update_state": base / "checkpoints" / "first_update.pt",
        "pairing": base / "first_rollout_pairing.json",
        "summary": base / "training_summary.json",
        "shaping": base / "shaping_events.jsonl",
        "shaping_audit": base / "shaping_audit_summary.json",
        "run_json": base / "runs" / "ppo_run.json",
        "rng": base / "pre_update_rng.pt",
        "pre_sampling_rng": base / "pre_sampling_rng.pt",
        "first_update_pre": base / "checkpoints" / "first_update_pre.pt",
        "minibatches": base / "minibatch_indices.jsonl",
    }


def minibatch_reference_path(arm: str) -> Path:
    """解析首个 PPO mini-batch 身份参照；默认保留旧的 REF 配对行为。"""
    if arm == "REF":
        raise ValueError("REF 臂不应读取自身的 mini-batch 参照")
    return REFERENCE_MINIBATCH_UPDATE or arm_paths("REF")["first_update"]


def write_run_json(arm: str, history: Sequence[Mapping[str, Any]], hyper: Mapping[str, Any],
                   stop_reason: str, decisions: int) -> None:
    """写本臂自己的 run.json（不得写入参照臂目录）。"""
    path = arm_paths(arm)["run_json"]
    path.parent.mkdir(parents=True, exist_ok=True)
    ppo_block = {
        "steps_per_iter": int(hyper["rollout_steps"]),
        "epochs": int(hyper["ppo_epochs"]),
        "batch_size": int(hyper["batch_size"]),
        "learning_rate": float(hyper["learning_rate"]) * float(ARM_SPECS[arm].get("lr_scale", 1.0)),
        "clip_epsilon": float(hyper["clip_epsilon"]),
        "value_coefficient": float(hyper["value_coefficient"]),
        "entropy_coefficient": float(
            hyper["entropy_coefficient"] if "entropy_coefficient" not in ARM_SPECS[arm]
            else ARM_SPECS[arm]["entropy_coefficient"]
        ),
        "gamma": float(hyper["gamma"]),
        "gae_lambda": float(hyper["gae_lambda"]),
        "shaping_coefficient": float(hyper["beta_shaping"]),
    }
    if RUN_SEED is not None:
        ppo_block.update(
            {
                "optimizer": str(hyper["optimizer"]),
                "optimizer_eps": float(hyper["optimizer_eps"]),
                "weight_decay": float(hyper["weight_decay"]),
                "max_grad_norm": float(hyper["max_grad_norm"]),
                "normalize_advantages": bool(hyper["normalize_advantages"]),
            }
        )
    dump_json(
        path,
        {
            "training_config": {
                "gamma": float(hyper["gamma"]),
                "beta_shaping": float(hyper["beta_shaping"]),
                "learning_rate": ppo_block["learning_rate"],
                "clip_eps": float(hyper["clip_epsilon"]),
                "vf_coef": float(hyper["value_coefficient"]),
                "ent_coef": ppo_block["entropy_coefficient"],
                "ppo_epochs": int(hyper["ppo_epochs"]),
                "batch_size": int(hyper["batch_size"]),
                "rollout_steps": int(hyper["rollout_steps"]),
                "disturbance_enabled": False,
                **(
                    {
                        "total_env_steps": int(hyper["total_env_steps"]),
                        "max_decisions": int(hyper["max_decisions"]),
                    }
                    if RUN_SEED is not None
                    else {}
                ),
                **(
                    {
                        "seed": int(RUN_SEED),
                        "optimizer": str(hyper["optimizer"]),
                        "optimizer_eps": float(hyper["optimizer_eps"]),
                        "weight_decay": float(hyper["weight_decay"]),
                        "max_grad_norm": float(hyper["max_grad_norm"]),
                        "normalize_advantages": bool(hyper["normalize_advantages"]),
                    }
                    if RUN_SEED is not None
                    else {}
                ),
            },
            "resolved_runtime_config_yaml": "ppo:\n"
            + "".join(f"  {key}: {value}\n" for key, value in ppo_block.items()),
            "training_decisions": decisions,
            "segments": len(history),
            "stop_reason": stop_reason,
            "arm": arm,
        },
    )


def episode_cost_record(arm: str, **kwargs: Any) -> dict[str, Any]:
    record = prev.build_episode_cost_record(**kwargs)
    record["run_id"] = f"ablation_{arm}"
    return record


@contextlib.contextmanager
def capture_randperm(sink: list[dict[str, Any]]):
    """记录正式 trainer 实际使用的 randperm 输出（不改变随机数本身）。"""
    original = torch.randperm

    def patched(n, *args, **kwargs):
        out = original(n, *args, **kwargs)
        sink.append({"n": int(n), "indices": [int(v) for v in out.tolist()]})
        return out

    torch.randperm = patched
    try:
        yield
    finally:
        torch.randperm = original


def apply_detach_value_hook(model: torch.nn.Module) -> Any:
    """价值分支在 critic 输入处 detach：共享编码器不再接收 value loss 的梯度。"""
    if model.critic._forward_pre_hooks:
        raise RuntimeError("critic已存在forward pre-hook，拒绝重复安装R4梯度隔离")
    handle = model.critic.register_forward_pre_hook(lambda module, inputs: (inputs[0].detach(),))
    return handle


def capture_rng_state(env: Any | None = None, generator: torch.Generator | None = None) -> dict[str, Any]:
    """保存主进程及可选环境worker的完整随机状态。"""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
        "explicit_action_generator": None if generator is None else generator.get_state().clone(),
        "environment_workers": [] if env is None else list(env.rng_states()),
    }


def _snapshot_facts(snapshot: Any) -> dict[str, Any]:
    """保留首段状态、候选掩码、图映射和工人资格的可比较摘要。"""
    digest = hashlib.sha256()
    for kind in sorted(snapshot.graph_snapshot.node_types):
        store = snapshot.graph_snapshot[kind]
        digest.update(kind.encode("utf-8"))
        tensor = store.x.detach().to(device="cpu").contiguous()
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    for edge_type in sorted(snapshot.graph_snapshot.edge_types):
        store = snapshot.graph_snapshot[edge_type]
        digest.update(str(edge_type).encode("utf-8"))
        tensor = store.edge_index.detach().to(device="cpu").contiguous()
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return {
        "worker_id": int(snapshot.worker_id),
        "episode_id": int(snapshot.episode_id),
        "episode_index": int(snapshot.episode_index),
        "candidate_task_keys": list(snapshot.candidate_task_keys),
        "branch_masks": [list(mask) for mask in snapshot.branch_masks],
        "candidate_task_node_indices": list(snapshot.candidate_task_node_indices),
        "worker_node_indices": [list(indices) for indices in snapshot.worker_node_indices],
        "reserved_flags": list(snapshot.reserved_flags),
        "team_contexts": [
            {
                "task_key": context.task_key,
                "station_id": int(context.station_id),
                "required_skill": int(context.required_skill),
                "demand": int(context.demand),
                "workers": [
                    {
                        "worker_id": int(worker.worker_id),
                        "skills": list(worker.skills),
                        "efficiency": worker.efficiency,
                        "calendar_intervals": [list(interval) for interval in worker.calendar_intervals],
                    }
                    for worker in context.workers
                ],
            }
            for context in snapshot.team_contexts
        ],
        "advance_available": bool(snapshot.advance_available),
        "current_time": float(snapshot.current_time),
        "cycle_id": int(snapshot.cycle_id),
        "h0": float(snapshot.h0),
        "last_transfer_time": float(snapshot.last_transfer_time),
        "graph_version": snapshot.graph_version,
        "graph_sha256": digest.hexdigest(),
        "candidate_task_features": snapshot.candidate_task_features.detach().cpu().clone(),
        "state_features": snapshot.state_features.detach().cpu().clone(),
        "time_features": snapshot.time_features.detach().cpu().clone(),
    }


def build_arm_model(arm: str) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    fit = load_json(CAL_FIT)
    payload = torch.load(CAL_MODEL, map_location="cpu", weights_only=False)
    hyper = load_json(REFERENCE_SUMMARY)["hyperparameters"]
    return payload, hyper, fit


def stage_inventory() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        paths = arm_paths(arm)
        paths["checkpoints"].mkdir(parents=True, exist_ok=True)
        paths["trajectories"].mkdir(parents=True, exist_ok=True)
        paths["run_json"].parent.mkdir(parents=True, exist_ok=True)
    missing = [
        str(path) for path in (CAL_MODEL, CAL_FIT, REFERENCE_SUMMARY, REFERENCE_FIRST_ROLLOUT)
        if not path.is_file()
    ]
    if missing:
        dump_json(OUTPUT_DIR / "missing_inputs.json", {"missing": missing})
        raise FileNotFoundError(f"缺少输入: {missing}")
    reference = load_json(REFERENCE_SUMMARY)
    registry = {
        "schema_version": "ppo_ablation_arms_registry_v1",
        "reference_commit": "a94eb7da70add9a6622498f6111ab119d609e648",
        "run_context": git_state(),
        "shared_start": {
            "calibrated_model": str(CAL_MODEL),
            "calibrated_model_sha256": sha256_file(CAL_MODEL),
            "calibrated_actor_fingerprint": load_json(REFERENCE_SUMMARY)["checkpoints"][0]["actor_fingerprint"],
            "calibration_steps": load_json(CAL_FIT)["calibration_steps"],
            "note": "五臂共用同一校准副本；不重新校准，不改变原 BC，不接续上一组训练结束权重。",
        },
        "reference_arm": {
            "dir": str(REFERENCE_PPO),
            "hyperparameters": reference["hyperparameters"],
            "points": [0, 128, 256, 512],
            "note": "a94eb7d 的校准臂作为共同参照；该轮未保存 8 步检查点，故 8 步仅新臂有值。",
        },
        "arms": ARM_SPECS,
        "points": list(POINTS),
        "budget": {"per_arm_interactions": BUDGET, "arms": len(ARMS), "total_interactions": BUDGET * len(ARMS)},
        "rules": {
            "scenario": "NOMINAL",
            "seed": 42,
            "precision": "fp32",
            "num_envs": 1,
            "start_method": "spawn",
            "rollout_steps": 8,
            "batch_size": 2,
            "epochs": 1,
            "serial_execution": True,
        },
    }
    dump_json(ARM_REGISTRY, registry)
    dump_json(
        MANIFEST,
        {
            "task": "Work-3 PPO 保持性消融臂 R0–R4",
            "reference_commit": "a94eb7da70add9a6622498f6111ab119d609e648",
            "run_context": git_state(),
            "output_dir": str(OUTPUT_DIR),
            "expected_artifacts": [
                "manifest.json", "arm_registry.json", "r0_frozen_sampling.json",
                "comparison_by_point.csv", "cycle_comparison.csv", "update_diagnostics.csv",
                "verification.json", "summary.md",
                "R0..R4/checkpoints/{step0000,step0008,step0128,step0256,step0512}.pt",
                "R0..R4/trajectories/{step0000,step0008,step0128,step0256,step0512}_NOMINAL.json",
                "R0..R4/training_history.jsonl",
                "R1..R4/first_update.json",
                "R0..R4/first_rollout_pairing.json",
                "R0..R4/pre_update_rng.pt",
                "R0..R4/minibatch_indices.jsonl",
            ],
        },
    )
    log("inventory 完成：五臂目录、共用起点与参照绑定已写入")
    return 0


# ---------------------------------------------------------------------------
# 单臂执行
# ---------------------------------------------------------------------------


def stage_arm(arm: str) -> int:
    spec = ARM_SPECS[arm]
    log(f"{arm}: 启动（{spec['change']}）")
    if not torch.cuda.is_available():
        raise RuntimeError("本阶段要求 cuda:0")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    paths = arm_paths(arm)

    payload = torch.load(CAL_MODEL, map_location="cpu", weights_only=False)
    hyper = dict(load_json(REFERENCE_SUMMARY)["hyperparameters"])
    if RUN_SEED is not None:
        hyper["seed"] = int(RUN_SEED)
    hyper["total_env_steps"] = int(BUDGET)
    hyper["max_decisions"] = int(BUDGET)
    seeds = prev.seed_everything(int(hyper["seed"]))
    model, _ = cal.load_calibrated_actor(CAL_MODEL)
    for parameter in model.parameters():
        parameter.requires_grad = True
    model.to(device)
    model.train()
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(int(hyper["seed"]))
    hook_count_before = len(model.critic._forward_pre_hooks)
    if not spec.get("detach_value") and hook_count_before:
        raise RuntimeError(f"{arm}训练模型带有未授权critic hook")
    value_hook_handle = None
    init_fingerprint = actor_fingerprint(model)
    if init_fingerprint != payload["actor_fingerprint"]:
        raise ValueError("臂起点指纹与校准副本不一致")
    init_state = model.state_dict()
    init_groups = prev.parameter_groups(model)["groups"]
    init_fingerprints = {
        "full_model": init_fingerprint,
        "policy_subset": cal.subset_fingerprint(init_state, init_groups["shared"] + init_groups["policy"]),
        "critic_subset": cal.subset_fingerprint(init_state, init_groups["value"]),
    }

    checkpoints: list[dict[str, Any]] = [
        prev.save_checkpoint(
            paths["checkpoints"] / "step0000.pt", model, step_index=0, decisions=0,
            optimizer_steps=0, episode_id=0, tag="step0000",
            extra={"arm": arm, "change": spec["change"], "initial_fingerprints": init_fingerprints,
                   "calibrated_model_path": str(CAL_MODEL), "calibrated_model_sha256": sha256_file(CAL_MODEL)},
        )
    ]
    lr = float(hyper["learning_rate"]) * float(spec.get("lr_scale", 1.0))
    ent_coef = float(hyper["entropy_coefficient"]) if "entropy_coefficient" not in spec else float(spec["entropy_coefficient"])
    lightning = prev.DiagnosticsLightningModule(
        actor_critic=model,
        learning_rate=lr,
        clip_eps=float(hyper["clip_epsilon"]),
        vf_coef=float(hyper["value_coefficient"]),
        ent_coef=ent_coef,
        max_grad_norm=float(hyper["max_grad_norm"]),
        ppo_epochs=int(hyper["ppo_epochs"]),
        batch_size=int(hyper["batch_size"]),
    )
    parameter_group_info = prev.parameter_groups(model)
    lightning.parameter_group_names = parameter_group_info["groups"]
    optimizer = lightning.configure_optimizers()
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if len(optimizer_parameters) != len({id(parameter) for parameter in optimizer_parameters}):
        raise ValueError("PPO优化器参数组包含重复参数")
    if {id(parameter) for parameter in optimizer_parameters} != {id(parameter) for parameter in trainable_parameters}:
        raise ValueError("PPO优化器参数组未完整覆盖可训练模型参数")
    if any(
        abs(float(group["lr"]) - lr) > 1e-15
        or abs(float(group["eps"]) - float(hyper["optimizer_eps"])) > 1e-15
        or abs(float(group["weight_decay"]) - float(hyper["weight_decay"])) > 1e-15
        for group in optimizer.param_groups
    ):
        raise ValueError("实际AdamW参数组与复核臂配置不一致")
    if any(state for state in optimizer.state.values()):
        raise ValueError("新建优化器不得携带历史状态")

    env = prev.build_env(seed=int(hyper["seed"]) if RUN_SEED is not None else None)
    shaping = prev.ShapingWriter(paths["shaping"])
    if spec.get("detach_value"):
        value_hook_handle = apply_detach_value_hook(model)
    history: list[dict[str, Any]] = []
    minibatches: list[dict[str, Any]] = []
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
    first_update_record: dict[str, Any] | None = None
    first_update_state_saved = False
    pre_update_rng: dict[str, Any] | None = None
    pending_snapshot = None
    memory_peak_allocated = 0
    memory_peak_reserved = 0
    first_rollout_payload: dict[str, Any] | None = None
    pre_sampling_rng: dict[str, Any] | None = None
    terminated = False
    previous_cost_breakdown = {
        "cost_takt": 0.0,
        "cost_time": 0.0,
        "cost_team": 0.0,
        "cost_postpone": 0.0,
        "cost_revision": 0.0,
    }
    started = time.perf_counter()

    try:
        env.reset(scenario=None, episode_id=episode_id, episode_index=episode_id)
        pending_snapshot = env.snapshot()
        pre_sampling_rng = capture_rng_state(env, action_generator)
        torch.save(pre_sampling_rng, paths["pre_sampling_rng"])
        while decisions < BUDGET:
            segment_started = time.perf_counter()
            next_point = next((p for p in POINTS if p > decisions), BUDGET)
            segment_limit = min(prev.ROLLOUT_STEPS, next_point - decisions)
            if segment_limit <= 0:
                break
            buffer = RolloutBufferWork3(
                gamma=float(hyper["gamma"]), gae_lambda=float(hyper["gae_lambda"]),
                normalize_advantages=spec.get("advantage_mode") != "center_only",
            )
            raw_buffer = RolloutBufferWork3(
                gamma=float(hyper["gamma"]), gae_lambda=float(hyper["gae_lambda"]),
                normalize_advantages=False,
            )
            transitions = []
            segment_records = []
            snapshot_records = []
            terminated = False
            for _ in range(segment_limit):
                snapshot = pending_snapshot
                snapshot_records.append(_snapshot_facts(snapshot))
                with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
                    action, log_prob, value, sample_record = model.select_snapshot(
                        snapshot, deterministic=False, generator=action_generator
                    )
                phi_current = compute_heuristic_potential(
                    float(snapshot.estimated_cmax), float(snapshot.last_transfer_time), float(snapshot.h0)
                )
                result = env.step(action)
                raw_reward = float(result.raw_reward)
                info = result.info or {}
                current_cost_breakdown = info.get("cost_breakdown", {})
                cost_component_increments = {
                    key: float(current_cost_breakdown.get(key, previous_cost_breakdown[key]))
                    - previous_cost_breakdown[key]
                    for key in previous_cost_breakdown
                }
                if current_cost_breakdown:
                    previous_cost_breakdown = {
                        key: float(current_cost_breakdown.get(key, previous_cost_breakdown[key]))
                        for key in previous_cost_breakdown
                    }
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
                    reward=float(shaped), raw_reward=raw_reward, value=float(value),
                    log_prob=float(log_prob), done=bool(terminated), action_dict=dict(action),
                    terminated=terminated, truncated=False, worker_id=0,
                    episode_id=episode_id, segment_id=rollout_id,
                    predictor_version=None, predictor_source="heuristic",
                    time_input_source="heuristic_time", phi_source="heuristic_phi",
                )
                transitions.append(transition)
                buffer.add(transition)
                raw_buffer.add(transition)
                segment_records.append(
                    {
                        "decision_index": decisions, "episode_id": episode_id,
                        "episode_step": episode_step - 1, "cycle_id": int(snapshot.cycle_id),
                        "sim_time": float(snapshot.current_time), "action_type": int(action["branch"]),
                        "task_key": action.get("task_key"),
                        "team": [int(w) for w in action.get("team", ())],
                        "align": int(action.get("align", 0)), "log_prob": float(log_prob),
                        "value": float(value), "raw_reward": raw_reward, "shaped_reward": float(shaped),
                        "phi_current": phi_current, "phi_next": phi_next,
                        "terminated": terminated, "truncated": False,
                        "step_cost": float(info.get("step_cost", 0.0)),
                        "cost_component_increments": cost_component_increments,
                    }
                )
                shaping.transition(
                    {
                        "record_type": "transition", "run_id": f"ablation_{arm}",
                        "worker_id": 0, "episode_id": episode_id, "rollout_id": rollout_id,
                        "episode_step": episode_step - 1, "action_type": int(action["branch"]),
                        "task_id": None, "task_key": action.get("task_key"),
                        "decision_time": float(snapshot.current_time), "cycle_id": int(snapshot.cycle_id),
                        "raw_reward": raw_reward, "shaped_reward": float(shaped),
                        "phi_current": float(phi_current), "phi_next": float(phi_next),
                        "beta": float(hyper["beta_shaping"]), "gamma": float(hyper["gamma"]),
                        "terminated": terminated, "truncated": False,
                        "predictor_version": None, "bootstrap_value": None,
                        "phi_source": "heuristic_phi",
                        "cost_component_increments": cost_component_increments,
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
                boundary_type = "budget_truncation" if decisions >= BUDGET else "rollout_continuation"
            else:
                last_value = 0.0
                boundary_type = "natural_termination"

            key = (0, episode_id, rollout_id)
            buffer.finish_trajectories(last_values_by_segment={key: last_value})
            raw_buffer.finish_trajectories(last_values_by_segment={key: last_value})
            raw_advantages = raw_buffer.advantages.clone()
            if spec.get("advantage_mode") == "center_only":
                centered = raw_advantages - raw_advantages.mean()
                buffer.advantages = centered.clone()
            shaped_rewards = [t.reward for t in transitions]
            raw_rewards = [t.raw_reward for t in transitions]
            values = [t.value for t in transitions]
            td_residuals = [
                float(shaped_rewards[index])
                + float(hyper["gamma"]) * (0.0 if transition.terminated else
                    (float(values[index + 1]) if index + 1 < len(values) else float(last_value)))
                - float(values[index])
                for index, transition in enumerate(transitions)
            ]

            with torch.no_grad():
                state_batch = torch.stack([t.state_feat for t in transitions])
                time_batch = torch.stack([t.time_urgency for t in transitions])
                _v, replay_lp, _e = model.evaluate_action_log_probs(
                    state_batch, time_batch, [t.sample_record for t in transitions]
                )
                old_lp = torch.tensor([t.log_prob for t in transitions], dtype=torch.float)
                replay_error = float((replay_lp.cpu().double() - old_lp.double()).abs().max().item())
            if not prev.replay_contract_ok(replay_error):
                dump_json(paths["dir"] / "replay_failure.json", {"rollout_id": rollout_id, "error": replay_error})
                stop_reason = "replay_contract_failed"
                break

            shaping.segment_end(
                {
                    "record_type": "segment_end", "run_id": f"ablation_{arm}", "worker_id": 0,
                    "episode_id": episode_id, "rollout_id": rollout_id,
                    "boundary_type": boundary_type, "bootstrap_value": float(last_value),
                    "termination_reason": boundary_type, "steps": len(transitions),
                    "training_decisions": decisions,
                }
            )
            if terminated:
                shaping.episode_cost(
                    episode_cost_record(
                        arm,
                        episode_id=episode_id, info=episode_last_info.get(episode_id, {}),
                        termination_reason="completed", episode_steps=episode_step,
                        raw_reward_sum=episode_raw_sum.get(episode_id, 0.0),
                    )
                )
                episode_closed.add(episode_id)

            if pre_update_rng is None:
                pre_update_rng = capture_rng_state(env, action_generator)
                pre_update_rng["decisions"] = decisions
                saved_cpu_rng = torch.get_rng_state().clone()
                predicted_permutation = torch.randperm(len(transitions)).tolist()
                torch.set_rng_state(saved_cpu_rng)
                torch.save(pre_update_rng, paths["rng"])
                # 输入 [D] × B 个状态 -> [B, D]。
                state_batch_cpu = torch.stack([t.state_feat for t in transitions])
                # 输入 [T] × B 个时间特征 -> [B, T]。
                time_batch_cpu = torch.stack([t.time_urgency for t in transitions])
                first_rollout_payload = {
                    "state_feats": state_batch_cpu,
                    "time_urgencies": time_batch_cpu,
                    "snapshot_facts": snapshot_records,
                    "actions": [t.action_dict for t in transitions],
                    "old_log_probs": old_lp,
                    "old_values": torch.tensor(values, dtype=torch.float),
                    "raw_rewards": torch.tensor(raw_rewards, dtype=torch.float),
                    "shaped_rewards": torch.tensor(shaped_rewards, dtype=torch.float),
                    "bootstrap_value": float(last_value),
                    "bootstrap_input": None if pending_snapshot is None else _snapshot_facts(pending_snapshot),
                    "td_residuals": torch.tensor(td_residuals, dtype=torch.float),
                    "raw_advantages": raw_advantages.detach().cpu().clone(),
                    "advantages": buffer.advantages.detach().cpu().clone(),
                    "return_targets": buffer.target_values.detach().cpu().clone(),
                    "sample_records": [t.sample_record for t in transitions],
                    "pre_sampling_rng": pre_sampling_rng,
                    "pre_update_rng": pre_update_rng,
                    "predicted_minibatch_permutation": predicted_permutation,
                }
                torch.save(first_rollout_payload, paths["dir"] / "first_rollout.pt")
                if RUN_SEED is not None:
                    pairing = verify_pairing(arm)
                    prev.dump_json_safe(paths["pairing"], pairing)
                    if not pairing["paired"]:
                        stop_reason = "first_rollout_pairing_failed"
                        break
                    if arm != "REF":
                        reference_update = load_json(minibatch_reference_path(arm))
                        if predicted_permutation != reference_update["minibatch_permutation"]:
                            stop_reason = "first_minibatch_identity_failed"
                            dump_json(paths["dir"] / "first_minibatch_failure.json", {
                                "predicted": predicted_permutation,
                                "reference": reference_update["minibatch_permutation"],
                            })
                            break
                    prev.save_checkpoint_state(
                        paths["first_update_pre"], prev.clone_actor_state(model), model,
                        step_index=0, decisions=decisions, optimizer_steps=optimizer_steps,
                        episode_id=episode_id, tag="first_update_pre",
                        extra={"arm": arm, "note": "首次optimizer.step之前"},
                    )

            update = Work3TrainingUpdate(
                buffer=buffer, environment_steps=len(transitions),
                time_auxiliary_batch=None, rollout_id=rollout_id,
            )
            updates_before = lightning.optimization_steps
            minibatch_records_before = len(lightning.minibatch_records)
            captured: list[dict[str, Any]] = []
            with capture_randperm(captured):
                loss = lightning.training_step(update, 0)
            if RUN_SEED is not None and rollout_id == 0:
                actual_permutation = captured[0]["indices"] if captured else None
                if actual_permutation != predicted_permutation:
                    stop_reason = "first_minibatch_replay_failed"
                    dump_json(paths["dir"] / "first_minibatch_failure.json", {
                        "predicted": predicted_permutation,
                        "actual": actual_permutation,
                    })
                    break
                first_rollout_payload["actual_minibatch_permutation"] = actual_permutation
                torch.save(first_rollout_payload, paths["dir"] / "first_rollout.pt")
            if not bool(torch.isfinite(loss).item()):
                stop_reason = "non_finite_loss"
                break
            updates_this_segment = lightning.optimization_steps - updates_before
            optimizer_steps = lightning.optimization_steps
            segment_loss_records = lightning.minibatch_records[minibatch_records_before:]
            for position, record in enumerate(captured):
                minibatches.append(
                    {
                        "rollout_id": rollout_id,
                        "training_decisions": decisions,
                        "permutation": record["indices"],
                        "batch_size": int(hyper["batch_size"]),
                        "minibatch_index": position,
                        "minibatch_samples": record["indices"][
                            position * int(hyper["batch_size"]):(position + 1) * int(hyper["batch_size"])
                        ],
                        "loss_metrics": segment_loss_records[position]
                        if position < len(segment_loss_records)
                        else None,
                    }
                )
            if first_update_record is None and updates_this_segment > 0:
                first_record = (lightning.update_records[-updates_this_segment:])[0]
                first_update_record = {
                    "arm": arm,
                    "decisions_at_first_update": decisions,
                    "minibatch_permutation": captured[0]["indices"] if captured else None,
                    "first_minibatch_samples": minibatches[-len(captured)]["minibatch_samples"] if captured else None,
                    "optimizer_step": first_record["optimizer_step"],
                    "total_loss": first_record["total_loss"],
                    "policy_loss": segment_loss_records[0]["policy_loss"] if segment_loss_records else None,
                    "value_loss": segment_loss_records[0]["value_loss"] if segment_loss_records else None,
                    "entropy": segment_loss_records[0]["entropy"] if segment_loss_records else None,
                    "approx_kl": segment_loss_records[0]["approx_kl"] if segment_loss_records else None,
                    "clip_fraction": segment_loss_records[0]["clip_fraction"] if segment_loss_records else None,
                    "ratio_stats": segment_loss_records[0]["ratio_stats"] if segment_loss_records else None,
                    "clipped_grad_norm": first_record["clipped_grad_norm"],
                    "pre_clip_group_grad_norm": first_record["pre_clip_group_grad_norm"],
                    "post_clip_group_grad_norm": first_record["post_clip_group_grad_norm"],
                    "parameter_delta_norm": first_record["parameter_delta_norm"],
                    "lr": first_record["lr"],
                    "learning_rates": first_record["learning_rates"],
                    "updates_in_first_segment": updates_this_segment,
                    "pre_update_rng_saved": True,
                    "optimizer_steps_before": updates_before,
                    "optimizer_steps_after": optimizer_steps,
                }
                if lightning.first_step_state is not None and not first_update_state_saved:
                    prev.save_checkpoint_state(
                        paths["first_update_state"], lightning.first_step_state, model,
                        step_index=0, decisions=decisions, optimizer_steps=optimizer_steps,
                        episode_id=episode_id, tag="first_update",
                        extra={"arm": arm, "note": "首次 optimizer.step 之后"},
                    )
                    first_update_state_saved = True

            memory = prev.gpu_snapshot()
            history.append(
                {
                    "arm": arm, "rollout_id": rollout_id, "training_decisions": decisions,
                    "segment_steps": len(transitions), "episode_id": episode_id,
                    "worker_id": 0, "segment_id": rollout_id,
                    "terminated": terminated, "boundary_type": boundary_type,
                    "bootstrap_value": float(last_value), "replay_max_abs_error": replay_error,
                    "bootstrap_input": None if pending_snapshot is None else {
                        "state_features": pending_snapshot.state_features,
                        "time_features": pending_snapshot.time_features,
                        "graph_version": pending_snapshot.graph_version,
                        "cycle_id": int(pending_snapshot.cycle_id),
                        "current_time": float(pending_snapshot.current_time),
                    },
                    "value_predictions": values, "td_residuals": td_residuals,
                    "raw_rewards": raw_rewards, "shaped_rewards": shaped_rewards,
                    "raw_advantages": raw_advantages.tolist(),
                    "advantages": buffer.advantages.tolist(),
                    "return_targets": buffer.target_values.tolist(),
                    "advantage_raw_stats": prev.stats(raw_advantages.tolist()),
                    "advantage_used_stats": prev.stats(buffer.advantages.tolist()),
                    "advantage_std_definition": "population_std(ddof=0)，在完整 rollout 段内计算",
                    "advantage_mode": spec.get("advantage_mode", "standard"),
                    "value_stats": prev.stats(values),
                    "return_target_stats": prev.stats(buffer.target_values.tolist()),
                    "raw_reward_stats": prev.stats(raw_rewards),
                    "shaped_reward_stats": prev.stats(shaped_rewards),
                    "action_counts": prev.action_counts(segment_records),
                    "update_records": (
                        [
                            {**loss_record, **grad_record}
                            for loss_record, grad_record in zip(
                                segment_loss_records,
                                lightning.update_records[-updates_this_segment:],
                                strict=True,
                            )
                        ] if updates_this_segment > 0 else []
                    ),
                    "segment_elapsed_seconds": time.perf_counter() - segment_started,
                    "elapsed_seconds": time.perf_counter() - started,
                    "cuda_allocated_bytes": memory["allocated_bytes"],
                    "cuda_reserved_bytes": memory["reserved_bytes"],
                    "cuda_peak_allocated_bytes": memory["peak_allocated_bytes"],
                    "cuda_peak_reserved_bytes": memory["peak_reserved_bytes"],
                    "worker_pids": list(env.worker_pids),
                    "worker_alive": list(env.workers_alive),
                    "cuda_peak_reserved_mib": memory["peak_reserved_bytes"] / 2**20,
                }
            )
            memory_peak_allocated = max(memory_peak_allocated, memory["peak_allocated_bytes"])
            memory_peak_reserved = max(memory_peak_reserved, memory["peak_reserved_bytes"])
            if max(memory_peak_allocated, memory_peak_reserved) >= prev.GPU_BUDGET_BYTES:
                stop_reason = "gpu_budget_exceeded"
                break
            if terminated:
                episode_id += 1
                episode_step = 0
                env.reset(scenario=None, episode_id=episode_id, episode_index=episode_id)
                pending_snapshot = env.snapshot()
                previous_cost_breakdown = {
                    "cost_takt": 0.0,
                    "cost_time": 0.0,
                    "cost_team": 0.0,
                    "cost_postpone": 0.0,
                    "cost_revision": 0.0,
                }
            rollout_id += 1
            if decisions in POINTS and decisions > 0:
                checkpoints.append(
                    prev.save_checkpoint(
                        paths["checkpoints"] / f"step{decisions:04d}.pt", model,
                        step_index=list(POINTS).index(decisions), decisions=decisions,
                        optimizer_steps=optimizer_steps, episode_id=episode_id,
                        tag=f"step{decisions:04d}", extra={"arm": arm, "initial_fingerprints": init_fingerprints,
                            "calibrated_model_path": str(CAL_MODEL), "calibrated_model_sha256": sha256_file(CAL_MODEL)},
                    )
                )
                log(f"{arm}: 已保存 step{decisions:04d}（优化步 {optimizer_steps}）")
    except torch.cuda.OutOfMemoryError as error:  # pragma: no cover
        stop_reason = "cuda_oom"
        dump_json(paths["dir"] / "cuda_oom.json", {"error": str(error), "decisions": decisions})
    except FloatingPointError as error:
        stop_reason = "non_finite_gradient"
        dump_json(paths["dir"] / "non_finite_gradient.json", {"error": str(error), "decisions": decisions})
    finally:
        for open_id in sorted(episode_last_info):
            if open_id in episode_closed:
                continue
            shaping.episode_cost(
                episode_cost_record(
                    arm,
                    episode_id=open_id, info=episode_last_info[open_id],
                    termination_reason="budget_truncation", episode_steps=episode_step,
                    raw_reward_sum=episode_raw_sum.get(open_id, 0.0),
                )
            )
        shaping.close()
        try:
            env.close()
        except Exception:
            pass
        if value_hook_handle is not None:
            value_hook_handle.remove()
        hook_count_after = len(model.critic._forward_pre_hooks)
        if hook_count_after != hook_count_before:
            raise RuntimeError(
                f"{arm}训练退出后critic hook数量{hook_count_after}与起始{hook_count_before}不一致"
            )
        prev.dump_jsonl_safe(paths["history"], history)
        prev.dump_jsonl_safe(paths["minibatches"], minibatches)
        if first_update_record is not None:
            prev.dump_json_safe(paths["first_update"], first_update_record)
        prev.dump_json_safe(
            paths["summary"],
            {
                "arm": arm, "change": spec["change"], "stop_reason": stop_reason,
                "training_decisions": decisions, "optimizer_steps": optimizer_steps,
                "segments": len(history), "episodes_started": len(episode_last_info),
                "episodes_completed": len(episode_closed),
                "elapsed_seconds": time.perf_counter() - started, "seeds": seeds,
                "learning_rate": lr, "entropy_coefficient": ent_coef,
                "initial_fingerprints": init_fingerprints,
                "advantage_mode": spec.get("advantage_mode", "standard"),
                "detach_value": bool(spec.get("detach_value")),
                "device": str(device),
                "optimizer_name": type(optimizer).__name__,
                "optimizer_param_group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
                "optimizer_param_group_settings": [
                    {"lr": float(group["lr"]), "eps": float(group["eps"]),
                     "weight_decay": float(group["weight_decay"]),
                     "parameter_count": sum(parameter.numel() for parameter in group["params"])}
                    for group in optimizer.param_groups
                ],
                "optimizer_schedulers": [],
                "optimized_parameter_count": len(optimizer_parameters),
                "parameter_group_counts": parameter_group_info["counts"],
                "detach_hook": {
                    "installed": value_hook_handle is not None,
                    "target": "model.critic.forward_pre_hook" if spec.get("detach_value") else None,
                    "install_count": int(value_hook_handle is not None),
                    "removed": value_hook_handle is not None,
                    "hooks_after_cleanup": hook_count_after,
                },
                "gpu": {
                    "name": torch.cuda.get_device_name(0),
                    "peak_allocated_mib": memory_peak_allocated / 2**20,
                    "peak_reserved_mib": memory_peak_reserved / 2**20,
                    "budget_mib": prev.GPU_BUDGET_BYTES / 2**20,
                    "budget_exceeded": max(memory_peak_allocated, memory_peak_reserved) >= prev.GPU_BUDGET_BYTES,
                },
                "checkpoints": checkpoints, "hyperparameters": dict(hyper),
            },
        )
        write_run_json(arm, history, hyper, stop_reason, decisions)

    pairing = verify_pairing(arm)
    prev.dump_json_safe(paths["pairing"], pairing)
    log(f"{arm}: 首段配对 {pairing['paired']}；decisions={decisions} updates={optimizer_steps} stop={stop_reason}")
    return 0 if pairing["paired"] else 3


def verify_pairing(arm: str) -> dict[str, Any]:
    ours_path = arm_paths(arm)["dir"] / "first_rollout.pt"
    if not ours_path.is_file() or not REFERENCE_FIRST_ROLLOUT.is_file():
        return {"paired": False, "reason": "first_rollout 缺失"}
    ours = torch.load(ours_path, map_location="cpu", weights_only=False)
    reference = torch.load(REFERENCE_FIRST_ROLLOUT, map_location="cpu", weights_only=False)
    if RUN_SEED is not None:
        from work3_4060.run_ppo_retention_repeat import verify_first_rollout

        return verify_first_rollout(reference, ours, arm=arm, compared_to=str(REFERENCE_FIRST_ROLLOUT))
    same_states = bool(torch.equal(ours["state_feats"], reference["state_feats"]))
    ours_actions = [cal._action_signature(a) for a in ours["actions"]]
    ref_actions = [cal._action_signature(a) for a in reference["actions"]]
    log_prob_diff = float(
        (ours["old_log_probs"].double() - reference["old_log_probs"].double()).abs().max().item()
    )
    paired = bool(same_states and ours_actions == ref_actions and log_prob_diff <= PAIRING_LOG_PROB_TOLERANCE)
    return {
        "arm": arm,
        "compared_to": str(REFERENCE_FIRST_ROLLOUT),
        "paired": paired,
        "same_states": same_states,
        "same_actions": ours_actions == ref_actions,
        "max_abs_log_prob_difference": log_prob_diff,
        "tolerance": PAIRING_LOG_PROB_TOLERANCE,
        "tolerance_note": PAIRING_TOLERANCE_NOTE,
        "steps_compared": int(ours["state_feats"].size(0)),
        "note": "同一策略与同一初始随机状态 → 首段 8 步采样事件应一致；8 步之后允许自然分化。",
    }


def stage_pairing() -> int:
    """从已保存的 first_rollout 重算五臂配对结论；不重训、不改动任何训练产物。"""
    rows = {}
    for arm in ARMS:
        if arm == "R0":
            rows[arm] = "N/A"
            continue
        pairing = verify_pairing(arm)
        prev.dump_json_safe(arm_paths(arm)["pairing"], pairing)
        rows[arm] = pairing["paired"]
        log(f"{arm}: paired={pairing['paired']} diff={pairing['max_abs_log_prob_difference']:.3e}")
    dump_json(OUTPUT_DIR / "pairing_recomputed.json", {"arms": rows, "tolerance": PAIRING_LOG_PROB_TOLERANCE,
                                                       "note": PAIRING_TOLERANCE_NOTE})
    return 0 if all(v is True for k, v in rows.items() if k != "R0") else 3


# ---------------------------------------------------------------------------
# R0：冻结采样
# ---------------------------------------------------------------------------


def stage_r0() -> int:
    log("R0: 冻结采样（零更新）")
    prev.enforce_single_thread_cpu()
    hyper = load_json(REFERENCE_SUMMARY)["hyperparameters"]
    prev.seed_everything(int(hyper["seed"]))
    model, _ = cal.load_calibrated_actor(CAL_MODEL)
    model.to("cpu")
    model.eval()
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    samples: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = [
        {"decisions": 0, "cumulative_raw_cost": 0.0, "completed_tasks": 0}
    ]
    with torch.inference_mode():
        while len(samples) < BUDGET and not env._check_terminated():
            snapshot = model.make_decision_snapshot(
                env, cal._state_features(env), cal._time_features(env),
                worker_id=0, episode_id=0, episode_index=len(samples) + 1,
                estimated_cmax=cal._cmax(env), allow_postpone=True,
            )
            action, log_prob, value, _record = model.select_snapshot(snapshot, deterministic=False)
            _obs, raw_reward, _done, _trunc, _info = env.step(action)
            samples.append(
                {
                    "decision_index": len(samples) + 1,
                    "cycle": int(snapshot.cycle_id),
                    "action_type": int(action["branch"]),
                    "task_key": action.get("task_key"),
                    "team": [int(w) for w in action.get("team", ())],
                    "align": int(action.get("align", 0)),
                    "raw_reward": float(raw_reward),
                    "log_prob": float(log_prob),
                    "value": float(value),
                }
            )
            if len(samples) in POINTS:
                checkpoints.append(
                    {
                        "decisions": len(samples),
                        "cumulative_raw_cost": -sum(s["raw_reward"] for s in samples),
                        "completed_tasks": sum(
                            1 for t in env.state.tasks.values() if t.actual_end is not None
                        ),
                    }
                )
    completed = sum(1 for t in env.state.tasks.values() if t.actual_end is not None)
    result = {
        "schema_version": "ppo_ablation_r0_v1",
        "rule": "不更新参数，正常随机采样 512 步；模型恒为校准副本",
        "seed": int(hyper["seed"]),
        "device": "cpu",
        "samples": len(samples),
        "cumulative_raw_cost_total": -sum(s["raw_reward"] for s in samples),
        "env_cumulative_cost": float(env.cumulative_cost),
        "completed_tasks": completed,
        "episode_terminated": bool(env._check_terminated()),
        "checkpoints": checkpoints,
        "action_counts": {
            "stay": sum(1 for s in samples if s["action_type"] == 0),
            "postpone": sum(1 for s in samples if s["action_type"] == 1),
            "advance": sum(1 for s in samples if s["action_type"] == 2),
            "align1": sum(1 for s in samples if s["action_type"] == 0 and s["align"] == 1),
        },
        "note": (
            "R0 的 5 个“评估点”是同一未更新模型的采样前缀累计费用；"
            "greedy 评估只有一条（模型未变），作为 step0000 的对照。"
        ),
    }
    dump_json(R0_RESULT, result)
    prev.dump_jsonl_safe(arm_dir("R0") / "sampled_decisions.jsonl", samples)
    log(
        "R0 完成: 采样 {} 步，累计原始费用 {:.6f}，完工 {} 道".format(
            len(samples), result["cumulative_raw_cost_total"], completed
        )
    )
    return 0


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------


def stage_repair() -> int:
    """修正两处实现缺陷留下的标注问题（不改动任何数值、权重或轨迹）。

    1) 前四臂的 episode_cost_summary 记录沿用了共享构造器里硬编码的 run_id；
       这里只把 run_id 改为本臂标识，五种费用与 raw_reward_sum 原样保留。
    2) 前四臂的 run.json 因调用共享写入器而落到参照臂目录（内容字节相同，已用 git 还原），
       这里按本臂 summary 重新生成到本臂目录。
    """
    notes = {"schema_version": "ppo_ablation_repairs_v1", "arms": {}}
    for arm in ARMS:
        if arm == "R0":
            continue
        paths = arm_paths(arm)
        records = [json.loads(line) for line in paths["shaping"].read_text(encoding="utf-8").splitlines() if line.strip()]
        changed = 0
        for record in records:
            if record.get("record_type") == "episode_cost_summary" and record.get("run_id") != f"ablation_{arm}":
                record["run_id"] = f"ablation_{arm}"
                changed += 1
        prev.dump_jsonl_safe(paths["shaping"], records)
        summary = load_json(paths["summary"])
        history = [json.loads(line) for line in paths["history"].read_text(encoding="utf-8").splitlines() if line.strip()]
        write_run_json(arm, history, summary["hyperparameters"], summary["stop_reason"], summary["training_decisions"])
        notes["arms"][arm] = {"episode_cost_run_id_fixed": changed, "run_json_regenerated": True}
        log(f"{arm}: 修正 {changed} 条 episode 账本标注，已重建 run.json")
    dump_json(OUTPUT_DIR / "repairs.json", notes)
    return 0


def stage_evaluate() -> int:
    for arm in ARMS:
        paths = arm_paths(arm)
        if arm == "R0":
            source = CAL_MODEL
            targets = [("step0000", source)]
        else:
            targets = [
                (f"step{point:04d}", paths["checkpoints"] / f"step{point:04d}.pt")
                for point in POINTS
            ]
        for label, checkpoint in targets:
            if not checkpoint.is_file():
                log(f"{arm} 缺少 {checkpoint.name}，跳过")
                return 3
            log(f"{arm}: 独立子进程评估 {label}")
            with (RUNS_DIR / f"evaluate_{arm}_{label}.log").open("w", encoding="utf-8") as handle:
                process = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--stage", "evaluate-one",
                     "--arm", arm, "--checkpoint", str(checkpoint), "--label", label],
                    cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, text=True, check=False,
                )
            if process.returncode != 0:
                log(f"{arm}/{label} 评估失败 {process.returncode}")
                return 2
    return 0


def stage_evaluate_one(arm: str, checkpoint: Path, label: str) -> int:
    threads = prev.enforce_single_thread_cpu()
    if arm == "R0" or (RUN_SEED is not None and label == "step0000"):
        model, payload = cal.load_calibrated_actor(checkpoint)
    else:
        model, payload = prev.load_checkpoint_actor(checkpoint)
    model.to("cpu")
    model.eval()
    agent = FormalEvaluationAgent(profile=build_method_profile("C"), actor_critic=model, time_head=None)
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    action_trace: list[dict[str, Any]] = []
    if RUN_SEED is not None:
        original_step = env.step

        def traced_step(action: Mapping[str, Any]):
            cycle_before = int(env.state.current_cycle)
            result = original_step(action)
            action_trace.append({
                "sequence": len(action_trace),
                "cycle_id_before": cycle_before,
                "branch": int(action.get("branch", 0)),
                "task_key": action.get("task_key"),
                "team_order": [int(worker) for worker in action.get("team", ())],
                "align": int(action.get("align", 0)),
                "advance_cause": result[4].get("advance_cause") if len(result) > 4 else None,
            })
            return result

        env.step = traced_step
    result = evaluate_single_trajectory(
        env, "Method-C", agent, scenario=None, max_decisions=4096, device="cpu"
    )
    trajectory_path = (OUTPUT_DIR / "shared_step0000_NOMINAL.json"
                       if RUN_SEED is not None and label == "step0000"
                       else arm_paths(arm)["trajectories"] / f"{label}_NOMINAL.json")
    dump_json(
        trajectory_path,
        {
            **result,
            **({"action_trace": action_trace} if RUN_SEED is not None else {}),
            "agent": f"{arm}_{label}",
            "wrapped_agent_type": "Method-C",
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha256_file(checkpoint),
                "actor_fingerprint": payload["actor_fingerprint"],
                "training_decisions": int(payload.get("training_decisions", 0)),
                "optimizer_steps": int(payload.get("optimizer_steps", 0)),
            },
            "evaluation": {
                "device": "cpu", "threads": threads, "deterministic": True,
                "decision_limit": 4096, "scenario": "NOMINAL", "disturbance": "none",
            },
            "execution_facts": extract_execution_facts(env),
        },
    )
    log(f"{arm}/{label}: J={result['objective_j_total']:.10f} decisions={result['decisions']}")
    return 0


# ---------------------------------------------------------------------------
# 验证与报告
# ---------------------------------------------------------------------------


def stage_verify() -> int:
    verifier = prev.load_verifier()
    baseline = load_json(BASELINE_PATH)
    h0 = float(baseline["metadata"]["h0"])
    tasks = baseline["tasks"]
    expected = []
    for arm in ARMS:
        labels = ["step0000"] if arm == "R0" else [f"step{p:04d}" for p in POINTS]
        for label in labels:
            expected.append(
                {
                    "arm": arm,
                    "label": label,
                    "path": str(arm_paths(arm)["trajectories"] / f"{label}_NOMINAL.json"),
                }
            )
    rows = []
    for item in expected:
        path = Path(item["path"])
        if not path.is_file():
            continue
        result = verifier.verify_single_trajectory(
            traj_path=path, baseline_h0=h0, baseline_tasks=tasks,
            scenario_spec={"scenario_id": "NOMINAL"},
        )
        result.update(item)
        rows.append(result)
        log(f"轨迹核验[{item['arm']}/{item['label']}]: {result['status']}")
    protocol = prev.classify_protocol_status(expected, rows)
    shaping_status = {}
    for arm in ARMS:
        paths = arm_paths(arm)
        if arm == "R0":
            shaping_status[arm] = "N/A（冻结采样不产生 PPO 事件）"
            continue
        if paths["shaping"].is_file() and paths["run_json"].is_file():
            summary = prev.load_shaping_verifier().verify_shaping_trace(
                paths["shaping"], paths["run_json"], paths["shaping_audit"]
            )
            shaping_status[arm] = summary["overall_status"]
        else:
            shaping_status[arm] = "INCOMPLETE"
    pairings = {
        arm: (load_json(arm_paths(arm)["pairing"])["paired"] if arm_paths(arm)["pairing"].is_file() else None)
        for arm in ARMS
    }
    checks = {
        "protocol_status_pass": protocol == "PASS",
        "pairing_matched": all(v is True for k, v in pairings.items() if k != "R0"),
        "shaping_not_fail": all(
            v in {"PASS", "N/A（冻结采样不产生 PPO 事件）"} for v in shaping_status.values()
        ),
        "calibrated_model_unchanged": sha256_file(CAL_MODEL)
        == load_json(ARM_REGISTRY)["shared_start"]["calibrated_model_sha256"],
    }
    overall = "PASS" if all(checks.values()) else "FAIL"
    dump_json(
        VERIFICATION,
        {
            "protocol_status": protocol,
            "overall_status": overall,
            "checks": checks,
            "shaping_status": shaping_status,
            "pairings": pairings,
            "verified_count": len(rows),
            "expected_count": len(expected),
            "trajectories": rows,
        },
    )
    log(f"verify 完成: protocol={protocol} overall={overall}")
    if overall != "PASS":
        return 2
    return {"PASS": 0, "FAIL": 2, "INCOMPLETE": 3}[protocol]


def stage_report() -> int:
    registry = load_json(ARM_REGISTRY)
    verification = load_json(VERIFICATION)
    reference_summary = load_json(REFERENCE_SUMMARY)
    r0 = load_json(R0_RESULT) if R0_RESULT.is_file() else None

    rows = []
    for arm in ARMS:
        labels = ["step0000"] if arm == "R0" else [f"step{p:04d}" for p in POINTS]
        for label in labels:
            path = arm_paths(arm)["trajectories"] / f"{label}_NOMINAL.json"
            if not path.is_file():
                continue
            payload = load_json(path)
            cycle = prev.summarize_cycles(payload)
            rows.append(
                {
                    "arm": arm,
                    "change": ARM_SPECS[arm]["change"],
                    "point_label": label,
                    "point": int(label.replace("step", "")),
                    "decisions": int(payload["decisions"]),
                    "completed": int(payload["completed_tasks"]),
                    "violations": int(payload.get("constraint_violation_count", 0)),
                    "postponed": int(payload["postponed_count"]),
                    "j_total": float(payload["objective_j_total"]),
                    "j_takt": float(payload["cost_components_weighted"]["j_takt"]),
                    "d_time": float(payload["cost_components_weighted"]["d_time"]),
                    "d_team": float(payload["cost_components_weighted"]["d_team"]),
                    "j_revision": float(payload["cost_components_weighted"]["j_revision"]),
                    "makespan": float(payload["makespan"]),
                    "max_cycle_delay": cycle["max_cycle_delay"],
                    "delayed_cycles": cycle["delayed_cycle_count"],
                    "retention_pass": prev.retention_pass(payload),
                }
            )
    prev.write_csv(COMPARISON_CSV, rows)

    diagnostics = []
    start_state = torch.load(
        arm_paths("R1")["checkpoints"] / "step0000.pt", map_location="cpu", weights_only=False
    )["actor_critic_state"]
    reference_state = torch.load(
        REFERENCE_PPO / "checkpoints" / "step0512.pt", map_location="cpu", weights_only=False
    )["actor_critic_state"]
    reference_drift = _drift(start_state, reference_state)
    for arm in ARMS:
        if arm == "R0":
            continue
        history = [
            json.loads(line) for line in arm_paths(arm)["history"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        first = history[0] if history else {}
        records = first.get("update_records") or []
        first_record = records[0] if records else {}
        diagnostics.append(
            {
                "arm": arm,
                "change": ARM_SPECS[arm]["change"],
                "first_segment_advantage_std_used": (first.get("advantage_used_stats") or {}).get("std"),
                "first_segment_advantage_std_raw": (first.get("advantage_raw_stats") or {}).get("std"),
                "first_update_total_loss": first_record.get("total_loss"),
                "first_update_clipped_grad_norm": first_record.get("clipped_grad_norm"),
                "first_update_param_delta_shared": (first_record.get("parameter_delta_norm") or {}).get("shared"),
                "first_update_param_delta_policy": (first_record.get("parameter_delta_norm") or {}).get("policy"),
                "first_update_param_delta_value": (first_record.get("parameter_delta_norm") or {}).get("value"),
                "first_update_lr": first_record.get("lr"),
                "param_drift_at_512": _drift(
                    start_state,
                    torch.load(
                        arm_paths(arm)["checkpoints"] / "step0512.pt",
                        map_location="cpu", weights_only=False,
                    )["actor_critic_state"],
                ),
                "reference_param_drift_at_512": reference_drift,
                "replay_max_abs_error_max": max(
                    (row.get("replay_max_abs_error") or 0.0) for row in history
                ),
            }
        )
    prev.write_csv(DIAGNOSTICS_CSV, diagnostics)

    cycle_rows = []
    for item in rows:
        arm = item["arm"]
        label = item["point_label"]
        payload = load_json(arm_paths(arm)["trajectories"] / f"{label}_NOMINAL.json")
        records = payload["execution_facts"]["cycle_execution_records"]
        for key in sorted(records, key=int):
            record = records[key]
            cycle_rows.append(
                {
                    "arm": arm,
                    "point_label": label,
                    "cycle": int(record["cycle_id"]),
                    "nominal_transfer_time": float(record["nominal_transfer_time"]),
                    "actual_transfer_time": float(record["transfer_time"]),
                    "delay_hours": float(record["transfer_time"]) - float(record["nominal_transfer_time"]),
                }
            )
    prev.write_csv(CYCLE_CSV, cycle_rows)

    summary = _build_summary(registry, rows, diagnostics, verification, reference_summary, r0)
    SUMMARY_MD.write_text(summary, encoding="utf-8")
    log("report 完成")
    return 0


def _drift(start: Mapping[str, torch.Tensor], end: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    for name, tensor in start.items():
        total += float((end[name].double() - tensor.double()).pow(2).sum().item())
    return math.sqrt(total)


def _build_summary(
    registry: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any],
    reference_summary: Mapping[str, Any],
    r0: Mapping[str, Any] | None,
) -> str:
    fmt = prev.fmt
    lines: list[str] = []
    lines.append("# Work-3 PPO 保持性消融臂 R0–R4")
    lines.append("")
    lines.append(
        "> 五臂共用同一个校准副本（a94eb7d）与同一份配置，每臂只改一项；"
        f"每臂 {BUDGET} 个交互决策，串行执行，共 {BUDGET * len(ARMS)} 个决策。"
        "不组合有效设置，不重跑对照训练。"
    )
    lines.append("")
    lines.append(
        f"**协议状态**：轨迹事实 `{verification['protocol_status']}`；"
        f"诊断自检 `{verification['overall_status']}`；"
        f"塑形审计 {', '.join(f'{k}={v}' for k, v in verification['shaping_status'].items())}。"
    )
    lines.append("")

    lines.append("## 1. 五臂设定与共用起点")
    lines.append("")
    lines.append("| 臂 | 只改什么 | 学习率 | 熵系数 | 优势处理 | 价值分支 |")
    lines.append("| :--- | :--- | ---: | ---: | :--- | :--- |")
    for arm in ARMS:
        spec = ARM_SPECS[arm]
        lines.append(
            f"| {arm} | {spec['change']} | "
            f"{'-' if arm != 'R1' else '×0.1'} | "
            f"{'-' if arm != 'R2' else '0'} | "
            f"{'标准' if spec.get('advantage_mode') is None else '仅中心化'} | "
            f"{'正常' if not spec.get('detach_value') else '共享特征处 detach'} |"
        )
    lines.append("")
    lines.append(
        f"- 共用起点：校准副本 `{Path(registry['shared_start']['calibrated_model']).name}`，"
        f"SHA256 `{registry['shared_start']['calibrated_model_sha256'][:16]}…`，"
        f"校准 {registry['shared_start']['calibration_steps']} 步；五臂均从该文件新建优化器，不接续任何已训练权重。"
    )
    lines.append(f"- 参照臂：a94eb7d 的校准后 PPO 运行（{registry['reference_arm']['dir']}），仅 0/128/256/512 四点。")
    pairings = verification["pairings"]
    lines.append(
        "- 首段配对（与参照首段 8 步逐项比较）："
        + "，".join(f"{arm}={pairings.get(arm)}" for arm in ARMS if arm != "R0")
        + "。首段之后允许自然分化。"
    )
    lines.append(
        "- R3 的归一化范围已核实：正式实现按**整段 rollout**（8 步）做 `(x-mean)/(std+1e-8)`，"
        "`std` 为 ddof=0；R3 保留减均值、取消除法，centering 范围不变。"
    )
    lines.append(
        "- R4 的实现是给 `critic` 注册 forward pre-hook，在共享特征处 detach；"
        "前向价值与动作分布在首次更新前与参照一致，价值头参数仍训练，策略仍可更新编码器。"
    )
    lines.append("")

    lines.append("## 2. R0 冻结采样：随机动作本身的代价")
    lines.append("")
    if r0:
        lines.append(
            f"- 采样 {r0['samples']} 步（零更新），累计原始费用 **{r0['cumulative_raw_cost_total']:.6f}**，"
            f"完工 {r0['completed_tasks']}/210，episode 终止={r0['episode_terminated']}。"
        )
        lines.append("")
        lines.append("| 采样前缀 | 累计原始费用 | 完工工序 |")
        lines.append("| ---: | ---: | ---: |")
        for item in r0["checkpoints"]:
            lines.append(
                f"| {item['decisions']} | {item['cumulative_raw_cost']:.6f} | {item['completed_tasks']} |"
            )
        lines.append("")
        lines.append(
            f"- 动作分布：留站 {r0['action_counts']['stay']}、后移 {r0['action_counts']['postpone']}、"
            f"推进 {r0['action_counts']['advance']}，其中 align=1 {r0['action_counts']['align1']} 次。"
        )
        lines.append(
            "- 解读：R0 的费用完全来自随机采样（策略未更新）。若 R0 本身就明显偏离，"
            "则后续“更新后退化”必须与采样噪声分开看；若 R0 很便宜，更新后的退化更值得关注。"
        )
    else:
        lines.append("- 缺少 R0 结果。")
    lines.append("")

    lines.append("## 3. 五臂四点（+8 步）并列表")
    lines.append("")
    lines.append("| 臂 | 点 | 决策 | 完工 | J_total | J_takt | D_time | D_team | J_rev | Makespan | 延迟周期 | 保持 |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['point']} | {row['decisions']} | {row['completed']} | "
            f"{fmt(row['j_total'])} | {fmt(row['j_takt'])} | {fmt(row['d_time'])} | {fmt(row['d_team'])} | "
            f"{fmt(row['j_revision'])} | {fmt(row['makespan'], 6)} | {row['delayed_cycles']} | "
            f"{'满足' if row['retention_pass'] else '未满足'} |"
        )
    lines.append("")
    lines.append(
        "> 8 步 = 首个 rollout（4 次 mini-batch 更新）全部结束后，**不是**第一次 optimizer.step 之后。"
        "参照臂未保存 8 步检查点，故该点只有本轮五臂的值。"
    )
    lines.append("")

    lines.append("## 4. 相对参照臂的同点对比（0/128/256/512）")
    lines.append("")
    reference = {}
    for point in (0, 128, 256, 512):
        path = REFERENCE_PPO / "trajectories" / f"step{point:04d}_NOMINAL.json"
        if path.is_file():
            reference[point] = load_json(path)
    lines.append("| 臂 | 点 | 参照 J | 本臂 J | ΔJ(参照−本臂) | 参照 J_takt | 本臂 J_takt |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        point = row["point"]
        if point not in reference:
            continue
        ref = reference[point]
        ref_j = float(ref["objective_j_total"])
        ref_takt = float(ref["cost_components_weighted"]["j_takt"])
        lines.append(
            f"| {row['arm']} | {point} | {fmt(ref_j)} | {fmt(row['j_total'])} | "
            f"{fmt(ref_j - row['j_total'])} | {fmt(ref_takt)} | {fmt(row['j_takt'])} |"
        )
    lines.append("")
    lines.append("> ΔJ 为正表示本臂在该点费用低于参照（更好），为负表示更差。")

    lines.append("")
    lines.append("## 5. 首次退化位置与更新诊断")
    lines.append("")
    baseline_by_arm: dict[str, float] = {}
    for row in rows:
        if row["point"] == 0:
            baseline_by_arm[row["arm"]] = float(row["j_total"])
    for arm in ARMS:
        base = baseline_by_arm.get(arm)
        points = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["point"])
        first = next((r for r in points if base is not None and abs(float(r["j_total"]) - base) > 1e-9), None)
        if first is None:
            lines.append(f"- {arm}：所有评估点与起点一致（无退化）。")
        else:
            lines.append(
                f"- {arm}：首次出现费用偏离在 **{first['point']}** 步（J={fmt(first['j_total'])}，"
                f"起点 J={fmt(base)}）；512 步 J={fmt(points[-1]['j_total'])}。"
            )
    lines.append("")
    lines.append("| 臂 | 首段优势 std（实际使用） | 首段优势 std（原始） | 首次更新总 loss | 裁剪后梯度范数 | Δ共享 | Δ策略 | Δ价值 | lr | 512 步权重漂移 L2 |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in diagnostics:
        lines.append(
            f"| {item['arm']} | {fmt(item['first_segment_advantage_std_used'], 6)} | "
            f"{fmt(item['first_segment_advantage_std_raw'], 6)} | "
            f"{fmt(item['first_update_total_loss'], 6)} | {fmt(item['first_update_clipped_grad_norm'], 4)} | "
            f"{fmt(item['first_update_param_delta_shared'], 3)} | "
            f"{fmt(item['first_update_param_delta_policy'], 3)} | "
            f"{fmt(item['first_update_param_delta_value'], 3)} | {fmt(item['first_update_lr'], 6)} | "
            f"{fmt(item.get('param_drift_at_512'), 4)} |"
        )
    lines.append("")
    lines.append(
        "- 判读要点：只看终点 J 会把“更慢”误读成“更好”。上表的 512 步权重漂移 L2 是必要的配套证据："
        "R1 的漂移显著小于参照臂时，它的“保持”主要说明**更新更慢**，而不是找到了更有效的更新方向。"
    )
    drift = {item["arm"]: item.get("param_drift_at_512") for item in diagnostics}
    ref_drift = diagnostics[0].get("reference_param_drift_at_512") if diagnostics else None
    if drift.get("R1") is not None and ref_drift:
        lines.append(
            f"- R1：512 步权重漂移 L2 = {drift['R1']:.4f}，参照臂 = {ref_drift:.4f}，"
            f"约为参照的 {drift['R1'] / ref_drift:.2f} 倍；因此 R1 的零费用保持"
            "应解释为“每步位移更小、210 个决策内 greedy 行为未改变”，"
            "不能据此宣称“降低学习率解决了问题”。"
        )
    if drift.get("R3") is not None:
        lines.append(
            f"- R3：去标准差后优势实际 std = {next(i['first_segment_advantage_std_used'] for i in diagnostics if i['arm'] == 'R3'):.4f}"
            "（归一化时为 0.935），但 Adam 对梯度尺度近似不变，"
            "首次更新的参数位移反而更大（Δ共享 0.0549 > 0.0389），随后出现大规模后移与工期膨胀 —— "
            "说明标准差缩放是**承重**的稳定化项，而不是可有可无的装饰。"
        )
    lines.append("")

    lines.append("## 6. 有限结论")
    lines.append("")
    best = {}
    for arm in ARMS:
        final = next((r for r in rows if r["arm"] == arm and r["point"] == 512), None)
        if final is not None and 512 in reference:
            best[arm] = float(reference[512]["objective_j_total"]) - float(final["j_total"])
    improved = [arm for arm, delta in best.items() if delta > 1e-9]
    if not improved:
        lines.append(
            "- 没有一臂在 512 步终点优于参照臂；本轮五组简单处理均未解决问题，"
            "但各自给出了可区分的证据（R0 的采样代价、R1 的更新幅度、R2 的熵项、"
            "R3 的优势缩放、R4 的价值梯度路径）。"
        )
    else:
        lines.append(
            "- 在 512 步终点优于参照臂的臂："
            + "、".join(f"{arm}（ΔJ={best[arm]:+.6f}）" for arm in improved)
            + "。这仍需独立重复核验后才能作为候选机制，不能立即宣布唯一根因。"
        )
    lines.append(
        "- 本轮为单实例、单 seed、每臂一条轨迹；不提供统计显著性，也不组合已看到的有效设置。"
    )
    lines.append("")
    lines.append("## 7. 产物索引")
    lines.append("")
    for name in (
        "manifest.json", "arm_registry.json", "r0_frozen_sampling.json",
        "comparison_by_point.csv", "cycle_comparison.csv", "update_diagnostics.csv",
        "verification.json",
        "R0..R4/checkpoints/{step0000,step0008,step0128,step0256,step0512}.pt",
        "R0..R4/trajectories/{step0000,step0008,step0128,step0256,step0512}_NOMINAL.json",
        "R1..R4/first_update.json", "R1..R4/pre_update_rng.pt", "R1..R4/minibatch_indices.jsonl",
    ):
        lines.append(f"- `{name}`")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def run_all() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    plan = [["inventory"], ["r0"]] + [["arm", arm] for arm in ARMS if arm != "R0"] + [["evaluate"], ["verify"], ["report"]]
    results: dict[str, int] = {}
    for step in plan:
        stage = step[0]
        tag = stage if len(step) == 1 else f"{stage}_{step[1]}"
        log(f"=== 阶段 {tag} ===")
        command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage]
        if len(step) > 1:
            command += ["--arm", step[1]]
        with (RUNS_DIR / f"{tag}.log").open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                command, cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
        results[tag] = process.returncode
        log(f"阶段 {tag} 退出码 {process.returncode}")
        if process.returncode != 0:
            dump_json(RUNS_DIR / "stage_results.json", results)
            return process.returncode
    dump_json(RUNS_DIR / "stage_results.json", results)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO 保持性消融臂 R0–R4")
    parser.add_argument(
        "--stage",
        choices=["all", "inventory", "r0", "arm", "pairing", "repair", "evaluate", "evaluate-one", "verify", "report"],
        default="all",
    )
    parser.add_argument("--arm", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()
    if args.stage == "all":
        return run_all()
    if args.stage == "evaluate-one":
        if args.arm is None or args.checkpoint is None or args.label is None:
            raise SystemExit("evaluate-one 需要 --arm/--checkpoint/--label")
        return stage_evaluate_one(args.arm, args.checkpoint, args.label)
    if args.stage == "arm":
        if args.arm not in ARM_SPECS or args.arm == "R0":
            raise SystemExit("--arm 必须是 R1..R4（R0 用 --stage r0）")
        return stage_arm(args.arm)
    handlers = {
        "inventory": stage_inventory,
        "r0": stage_r0,
        "pairing": stage_pairing,
        "repair": stage_repair,
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
