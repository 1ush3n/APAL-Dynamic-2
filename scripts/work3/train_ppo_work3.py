"""工作三条件分支PPO烟测与训练试点入口；本脚本不产出正式研究结论。

功能范围：
1. 烟测使用无扰动小预算；试点使用固定训练事件清单和可配置退出预算；
2. 检查图策略、条件分支动作、GAE、势函数及时间监督的训练通路；
3. 记录损失、重放误差、真实命中、转站、批次结果和资源使用；不将试点指标解释为方法效果。
4. 保存带运行元数据的模型权重；检查点不支持精确断点续训。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
import platform
import random
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from envs.work3.environment import AirLineEnvWork3
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from models.work3.potential_shaping import PotentialRewardShaper
from models.work3.ppo_buffer import PendingTimeLabelCache, PPOTransition, RolloutBufferWork3
from models.work3.ppo_trainer import PPO_CHECKPOINT_VERSION, PPOTrainerWork3
from models.work3.time_head import TimeResidualHead
from scripts.work3.collect_validation_trajectories import load_scenarios_for_split
from scripts.work3.experiment_protocol import Work3MethodProfile, build_method_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    """锁定本次训练用到的Python、NumPy、PyTorch及CUDA随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_revision() -> tuple[str | None, bool | None]:
    """返回当前Git提交及工作区是否有未提交变更。"""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if revision.returncode != 0 or status.returncode != 0:
        return None, None
    return revision.stdout.strip(), bool(status.stdout.strip())


def _peak_memory_bytes(device: torch.device) -> int | None:
    """读取当前训练进程的峰值显存或峰值常驻内存；无平台接口时返回None。"""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device))
    try:
        import psutil

        memory = psutil.Process().memory_info()
        peak_working_set = getattr(memory, "peak_wset", None)
        if peak_working_set is not None:
            return int(peak_working_set)
    except (ImportError, OSError):
        pass
    if os.name != "nt":
        try:
            import resource

            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return peak if platform.system() == "Darwin" else peak * 1024
        except (ImportError, OSError, ValueError):
            pass
    return None


def _module_fingerprint(modules: dict[str, torch.nn.Module | None]) -> str:
    digest = hashlib.sha256()
    for module_name, module in sorted(modules.items()):
        if module is None:
            continue
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(f"{module_name}.{name}".encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _compare_training_reports(
    current: dict[str, Any],
    paired: dict[str, Any],
) -> dict[str, Any]:
    """分别比较C/D外生事件计划、预算和策略实际命中模式。"""
    budget_keys = (
        "max_decisions",
        "max_wall_seconds",
        "successful_batch_target",
        "rollout_steps",
        "max_rollout_iterations",
        "ppo_epochs",
        "batch_size",
    )
    current_config = current.get("training_config", {})
    paired_config = paired.get("training_config", {})
    current_hits = {
        (item.get("episode_id"), item.get("scenario_id")): int(item.get("actual_hit_count") or 0)
        for item in current.get("scenario_log", [])
    }
    paired_hits = {
        (item.get("episode_id"), item.get("scenario_id")): int(item.get("actual_hit_count") or 0)
        for item in paired.get("scenario_log", [])
    }
    same_event_plan = current.get("event_plan_fingerprint") == paired.get("event_plan_fingerprint")
    return {
        "paired_method_variant": paired.get("method_variant"),
        "same_method_variant": current.get("method_variant") == paired.get("method_variant"),
        "same_seed": current.get("seed") == paired.get("seed"),
        "same_scenario_pool": (
            current.get("data_fingerprint", {}).get("scenario_pool_sha256")
            == paired.get("data_fingerprint", {}).get("scenario_pool_sha256")
        ),
        "same_event_plan": same_event_plan,
        "same_interaction_budget": all(
            current_config.get(key) == paired_config.get(key)
            for key in budget_keys
        ),
        "same_actual_hit_pattern": (
            same_event_plan and current_hits == paired_hits
        ),
    }


def load_training_scenarios(
    scenarios_path: str | Path,
    scenario_split_path: str | Path,
) -> list[dict[str, Any]]:
    """按固定训练清单加载有效扰动，不从现场状态重新选目标。"""
    scenarios = load_scenarios_for_split(scenarios_path, scenario_split_path)
    if not scenarios:
        raise ValueError("正式扰动训练清单为空")
    invalid = [item["scenario_id"] for item in scenarios if item.get("valid", True) is False]
    if invalid:
        raise ValueError(f"训练清单包含无效场景: {invalid}")
    return scenarios


def build_episode_scenario_plan(
    scenarios: list[dict[str, Any]],
    num_episodes: int,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """按固定种子循环打乱场景，生成可记录的episode事件计划。"""
    if not scenarios:
        raise ValueError("不能从空场景池生成episode计划")
    if num_episodes < 1:
        return []
    rng = random.Random(int(seed))
    pool = [dict(item) for item in scenarios]
    plan: list[dict[str, Any]] = []
    while len(plan) < num_episodes:
        rng.shuffle(pool)
        plan.extend(dict(item) for item in pool)
    return plan[:num_episodes]


def compute_heuristic_potential(
    estimated_cmax: float,
    last_transfer_time: float,
    h0: float,
    *,
    a: float = 0.5,
    b: float = 1.0,
) -> float:
    """计算正式C使用的启发式时间势函数。"""
    h_est = max(0.0, float(estimated_cmax) - float(last_transfer_time))
    h0_value = float(h0)
    return -float(a) * (h_est / h0_value) - float(b) * max(0.0, h_est - h0_value) / h0_value


def actual_scenario_hit_task_keys(
    env: AirLineEnvWork3,
    scenario: dict[str, Any],
) -> list[str]:
    """返回实际受到恢复时刻约束的目标工序。"""
    return [
        str(task_key)
        for task_key in scenario.get("affected_task_keys", [])
        if task_key in env.state.tasks
        and float(env.state.tasks[task_key].material_ready_time) > env.tolerance
    ]


def count_actual_scenario_hits(env: AirLineEnvWork3, scenario: dict[str, Any]) -> int:
    """按实际物料恢复时间统计已揭示的目标工序数量。"""
    return len(actual_scenario_hit_task_keys(env, scenario))


def compute_online_time_inputs(
    actor_critic: ActorCriticWork3,
    time_head: TimeResidualHead,
    env: AirLineEnvWork3,
    state_feat: torch.Tensor,
    estimated_cmax: float,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """用当前图表征和时间头生成Actor实际接收的时间输入。"""
    device = next(actor_critic.parameters()).device
    graph_snapshot = actor_critic.build_graph_snapshot(env)
    with torch.no_grad():
        shared_feature = actor_critic.encode_shared_representation(
            state_feat.to(device),
            graph_snapshot,
        ).unsqueeze(0)
        predicted_transfer, remaining, _ = time_head.predict_corrected_time(
            state_feat=shared_feature,
            estimated_cmax=estimated_cmax,
            current_time=float(env.state.current_time),
            h0=float(env.state.h0),
            last_transfer_time=float(env.state.last_transfer_time),
        )
        urgency = compute_time_urgency_vector(
            estimated_r=remaining.squeeze(0),
            current_time=env.state.current_time,
            last_transfer_time=env.state.last_transfer_time,
            h0=env.state.h0,
            device=device,
        )
    return graph_snapshot, urgency, predicted_transfer.squeeze(0)


def run_training(
    num_iterations: int | None = None,
    steps_per_iter: int = 32,
    ppo_epochs: int = 1,
    batch_size: int = 64,
    lr: float = 3e-4,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    beta_shaping: float = 1.0,
    time_loss_coef: float = 1.0,
    time_auxiliary_epochs: int = 1,
    time_auxiliary_batch_size: int = 64,
    seed: int = 42,
    run_mode: str = "smoke",
    successful_batch_target: int | None = None,
    max_decisions: int | None = None,
    max_wall_seconds: float | None = None,
    method_variant: str = "D",
    baseline_path: str | Path = "data/work3/real_283_k10_baseline.json",
    scenarios_path: str | Path = "data/work3/scenarios_9class.json",
    scenario_split_path: str | Path = "data/work3/experiment_splits/train.json",
    time_head_ckpt: str | Path = "models/work3/checkpoints/time_head_best.pt",
    output_ckpt: str | Path = "models/work3/checkpoints/method_d_model.pt",
    report_path: str | Path | None = None,
    paired_report_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """执行不具研究结论资格的烟测或有明确预算的训练试点。"""
    run_started = time.monotonic()
    if run_mode not in {"smoke", "pilot"}:
        raise ValueError("run_mode必须是'smoke'或'pilot'；正式训练预算尚未定义")
    if (num_iterations is not None and num_iterations < 1) or steps_per_iter < 1 or ppo_epochs < 0 or batch_size < 1:
        raise ValueError("迭代、采样步数、PPO轮数和批量参数不合法")
    if run_mode == "smoke":
        if num_iterations not in (None, 1) or ppo_epochs != 1 or steps_per_iter > 64:
            raise ValueError("smoke仅允许1轮、1次PPO优化及最多64个决策步")
        num_iterations = 1
        if successful_batch_target is not None:
            raise ValueError("smoke不接受完整批次成功目标")
        if max_decisions is None:
            max_decisions = steps_per_iter
        if max_decisions < 1 or max_decisions > steps_per_iter:
            raise ValueError("smoke决策上限必须在1到本轮采样步数之间")
        if batch_size < max_decisions:
            raise ValueError("smoke的batch_size必须覆盖全部采样样本，确保仅执行一次优化")
    else:
        if successful_batch_target is None or successful_batch_target < 1:
            raise ValueError("pilot必须设置正数successful_batch_target")
        if max_decisions is None and max_wall_seconds is None:
            raise ValueError("pilot必须设置max_decisions或max_wall_seconds预算上限")
    if max_decisions is not None and max_decisions < 1:
        raise ValueError("max_decisions必须为正数")
    if max_wall_seconds is not None and (
        not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0.0
    ):
        raise ValueError("max_wall_seconds必须为有限正数")

    profile = build_method_profile(method_variant)
    _seed_everything(seed)
    torch_device = torch.device(device)

    training_scenarios = load_training_scenarios(scenarios_path, scenario_split_path)
    if run_mode == "smoke":
        episode_plan = [{
            "scenario_id": "SMOKE_NO_DISTURBANCE",
            "timing": None,
            "intensity": None,
            "station_id": None,
            "aircraft_id": None,
            "affected_task_keys": [],
            "valid": True,
        }]
    else:
        episode_plan = build_episode_scenario_plan(
            training_scenarios,
            len(training_scenarios),
            seed=seed,
        )
    if Path(output_ckpt).as_posix() == "models/work3/checkpoints/method_d_model.pt":
        mode_name = "smoke" if run_mode == "smoke" else "pilot"
        output_ckpt = Path("models/work3/checkpoints") / f"{mode_name}_method_{profile.name.lower()}_weights.pt"
    output_ckpt = Path(output_ckpt)
    report_path = Path(report_path) if report_path is not None else output_ckpt.with_suffix(".run.json")
    paired_report_path = Path(paired_report_path) if paired_report_path is not None else None
    if paired_report_path is not None and not paired_report_path.is_file():
        raise FileNotFoundError(f"配对训练报告不存在：{paired_report_path}")

    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 初始化仿真环境
    logger.info(f"初始化环境: {baseline_path}")
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    # 2. 初始化Actor与共享图表征上的时间预测头
    actor_critic = ActorCriticWork3(
        state_dim=32,
        task_feat_dim=8,
        hidden_dim=64,
        max_station_workers=16,
    ).to(torch_device)
    time_head = (
        TimeResidualHead(in_dim=actor_critic.hidden_dim, hidden_dim=64).to(torch_device)
        if profile.use_time_auxiliary
        else None
    )
    time_head_initialization = "not_applicable"
    if profile.use_time_auxiliary and time_head is not None and Path(time_head_ckpt).is_file():
        ckpt_data = torch.load(time_head_ckpt, map_location="cpu")
        if (
            ckpt_data.get("model_version") == "signed_residual_v1"
            and int(ckpt_data.get("in_dim", -1)) == actor_critic.hidden_dim
        ):
            state_dict = ckpt_data.get("model_state_dict", ckpt_data)
            time_head.load_state_dict(state_dict)
            time_head_initialization = "signed_pretrained_checkpoint"
            logger.info(f"已成功载入有符号离线时间修正头: {time_head_ckpt}")
        else:
            time_head_initialization = "random_no_compatible_checkpoint"
            logger.warning(f"检查点 {time_head_ckpt} 不是当前共享图有符号版本，本次不加载")
    elif profile.use_time_auxiliary:
        time_head_initialization = "random_checkpoint_missing"
        logger.warning(f"未找到预训练时间修正头 {time_head_ckpt}，使用随机初始化头")

    shaper = None
    if profile.use_learned_time_shaping and time_head is not None:
        shaper = PotentialRewardShaper(
            time_head=time_head,
            actor_critic=actor_critic,
            a=0.5,
            b=1.0,
            beta=beta_shaping,
            gamma=gamma,
        )

    # 3. 初始化条件分支自回归 PPO 训练器
    trainer = PPOTrainerWork3(
        actor_critic=actor_critic,
        lr=lr,
        clip_eps=clip_eps,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        time_head=time_head if profile.use_time_auxiliary else None,
        time_loss_coef=time_loss_coef if profile.use_time_auxiliary else 0.0,
        device=torch_device,
    )

    buffer = RolloutBufferWork3(
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_advantages=True,
    )

    source_sha, source_tree_dirty = _source_revision()
    event_plan = [dict(item) for item in episode_plan]
    data_fingerprint = {
        "scenario_pool_sha256": _json_fingerprint(training_scenarios),
        "scenario_split_sha256": _file_fingerprint(scenario_split_path),
        "baseline_sha256": _file_fingerprint(baseline_path),
        "time_head_checkpoint_sha256": (
            _file_fingerprint(time_head_ckpt)
            if Path(time_head_ckpt).is_file()
            else None
        ),
        "event_plan_sha256": _json_fingerprint(event_plan),
    }
    config = {
        "run_mode": run_mode,
        "disturbance_enabled": run_mode == "pilot",
        "method_variant": profile.name,
        "method_profile": {
            "graph_policy": profile.graph_policy,
            "allow_postpone": profile.allow_postpone,
            "use_time_auxiliary": profile.use_time_auxiliary,
            "use_corrected_time_input": profile.use_corrected_time_input,
            "use_learned_time_shaping": profile.use_learned_time_shaping,
        },
        "time_head_initialization": time_head_initialization,
        "seed": int(seed),
        "seed_role": "training_initialization_and_episode_schedule",
        "max_rollout_iterations": num_iterations,
        "rollout_steps": int(steps_per_iter),
        "ppo_epochs": int(ppo_epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "clip_eps": float(clip_eps),
        "vf_coef": float(vf_coef),
        "ent_coef": float(ent_coef),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "beta_shaping": float(beta_shaping),
        "time_loss_coef": float(time_loss_coef),
        "time_auxiliary_epochs": int(time_auxiliary_epochs),
        "time_auxiliary_batch_size": int(time_auxiliary_batch_size),
        "successful_batch_target": successful_batch_target,
        "max_decisions": max_decisions,
        "max_wall_seconds": max_wall_seconds,
        "baseline_path": str(Path(baseline_path)),
        "scenarios_path": str(Path(scenarios_path)),
        "scenario_split_path": str(Path(scenario_split_path)),
        "time_head_checkpoint_path": str(Path(time_head_ckpt)),
        "output_checkpoint_path": str(output_ckpt),
        "report_path": str(report_path),
        "paired_report_path": None if paired_report_path is None else str(paired_report_path),
        "device": str(torch_device),
    }
    initial_parameter_fingerprint = _module_fingerprint({
        "actor_critic": actor_critic,
        "time_head": time_head,
    })
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)

    history: list[dict[str, Any]] = []
    total_env_steps = 0
    start_time = run_started
    episode_id = 0
    pending_time_labels = PendingTimeLabelCache()
    episode_plan_index = 0
    scenario_log: list[dict[str, Any]] = []
    cycle_time_labels: list[dict[str, Any]] = []
    stop_reason: str | None = None
    current_scenario: dict[str, Any] | None = None
    current_scenario_log: dict[str, Any] | None = None
    episode_done = False
    episode_termination_reason: str | None = None

    def update_current_scenario_effect_log() -> None:
        if current_scenario_log is None or current_scenario is None:
            return
        tau = current_scenario.get("tau") if run_mode == "pilot" else None
        current_scenario_log["disturbance_triggered"] = bool(
            tau is not None
            and env.state.current_time >= float(tau) - env.tolerance
        )
        hit_keys = actual_scenario_hit_task_keys(env, current_scenario)
        current_scenario_log["actual_hit_task_keys"] = hit_keys
        current_scenario_log["actual_hit_count"] = len(hit_keys)

    def start_episode() -> None:
        nonlocal episode_plan_index, current_scenario, current_scenario_log
        env.reset()
        current_scenario = dict(episode_plan[episode_plan_index % len(episode_plan)])
        episode_plan_index += 1
        if run_mode == "pilot":
            env.load_scenario(current_scenario)
        current_scenario_log = {
            "episode_id": episode_id,
            "scenario_id": current_scenario["scenario_id"],
            "timing": current_scenario["timing"],
            "intensity": current_scenario["intensity"],
            "station_id": current_scenario["station_id"],
            "aircraft_id": current_scenario["aircraft_id"],
            "disturbance_scheduled": run_mode == "pilot",
            "scheduled_tau": current_scenario.get("tau") if run_mode == "pilot" else None,
            "scheduled_recovery_time": current_scenario.get("recovery_time") if run_mode == "pilot" else None,
            "scheduled_affected_task_keys": list(current_scenario.get("affected_task_keys", [])) if run_mode == "pilot" else [],
            "scheduled_target_count": (
                len(current_scenario.get("affected_task_keys", []))
                if run_mode == "pilot"
                else 0
            ),
            "disturbance_triggered": False,
            "actual_hit_task_keys": [],
            "actual_hit_count": 0,
            "actual_transfer_times": [],
            "completed": False,
            "success": None,
            "truncated": False,
            "termination_reason": None,
        }
        scenario_log.append(current_scenario_log)
        update_current_scenario_effect_log()
        logger.info(
            "加载训练episode=%s scenario=%s disturbance=%s timing=%s intensity=%s station=%s",
            episode_id,
            current_scenario["scenario_id"],
            run_mode == "pilot",
            current_scenario["timing"],
            current_scenario["intensity"],
            current_scenario["station_id"],
        )

    def record_unlabelled_cycles(target_episode_id: int) -> None:
        if not profile.use_time_auxiliary:
            return
        for pending_cycle_id, count in sorted(
            pending_time_labels.pending_cycle_counts(target_episode_id).items()
        ):
            cycle_time_labels.append({
                "episode_id": int(target_episode_id),
                "cycle_id": int(pending_cycle_id),
                "actual_transfer_time": None,
                "label_count": 0,
                "pending_decision_count": int(count),
                "label_available": False,
            })

    def mark_budget_truncated(reason: str) -> None:
        nonlocal stop_reason
        stop_reason = reason
        update_current_scenario_effect_log()
        if current_scenario_log is not None and current_scenario_log["success"] is None:
            current_scenario_log.update({
                "completed": False,
                "success": False,
                "truncated": True,
                "termination_reason": reason,
                "actual_hit_count": count_actual_scenario_hits(env, current_scenario or {}),
            })
            record_unlabelled_cycles(episode_id)
            pending_time_labels.discard_episode(episode_id)
        if buffer.transitions and not buffer.transitions[-1].terminated:
            buffer.transitions[-1].done = True
            buffer.transitions[-1].truncated = True

    def current_budget_reason() -> str | None:
        if max_decisions is not None and total_env_steps >= max_decisions:
            return "decision_limit"
        if max_wall_seconds is not None and time.monotonic() - start_time >= max_wall_seconds:
            return "wall_time_limit"
        return None

    start_episode()

    logger.info("=" * 80)
    logger.info(
        "开始工作三%s (seed=%s, 每段最多%s步, 决策上限=%s, 墙钟上限=%s秒)",
        "无扰动烟测" if run_mode == "smoke" else "扰动训练试点",
        seed,
        steps_per_iter,
        max_decisions,
        max_wall_seconds,
    )
    logger.info("=" * 80)

    iter_idx = 0
    while stop_reason is None:
        if num_iterations is not None and iter_idx >= num_iterations:
            stop_reason = "iteration_limit"
            if run_mode == "pilot":
                mark_budget_truncated(stop_reason)
            break
        budget_reason = current_budget_reason()
        if budget_reason is not None:
            mark_budget_truncated(budget_reason)
            break
        iter_idx += 1
        iter_start = time.monotonic()
        buffer.clear()

        step_raw_rewards: list[float] = []
        step_shaped_rewards: list[float] = []
        last_terminated = False

        # -------------------------
        # Rollout 数据采集循环
        # -------------------------
        for step in range(steps_per_iter):
            if env._check_terminated() or episode_done:
                episode_success = env._check_terminated()
                final_reason = "completed" if episode_success else episode_termination_reason
                update_current_scenario_effect_log()
                if not episode_success:
                    record_unlabelled_cycles(episode_id)
                pending_time_labels.discard_episode(episode_id)
                if current_scenario_log is not None:
                    current_scenario_log["actual_hit_count"] = count_actual_scenario_hits(
                        env,
                        current_scenario or {},
                    )
                    current_scenario_log["completed"] = episode_success
                    current_scenario_log["success"] = episode_success
                    current_scenario_log["termination_reason"] = final_reason
                if shaper is not None and episode_success:
                    shaper.update_snapshot(time_head, actor_critic)
                if (
                    run_mode == "pilot"
                    and sum(item.get("success") is True for item in scenario_log)
                    >= int(successful_batch_target or 0)
                ):
                    stop_reason = "successful_batch_target_reached"
                    break
                budget_reason = current_budget_reason()
                if budget_reason is not None:
                    stop_reason = budget_reason
                    break
                episode_id += 1
                episode_done = False
                episode_termination_reason = None
                start_episode()

            budget_reason = current_budget_reason()
            if budget_reason is not None:
                mark_budget_truncated(budget_reason)
                break

            cmax_est = compute_cycle_heuristic_cmax(env.state)
            s_feat = extract_compact_state_features(env.state, cmax_est)
            graph_snapshot = actor_critic.build_graph_snapshot(env)
            if profile.use_corrected_time_input and time_head is not None:
                graph_snapshot, u_time, _ = compute_online_time_inputs(
                    actor_critic=actor_critic,
                    time_head=time_head,
                    env=env,
                    state_feat=s_feat,
                    estimated_cmax=cmax_est,
                )
            else:
                estimated_r = max(0.0, cmax_est - float(env.state.current_time))
                u_time = compute_time_urgency_vector(
                    estimated_r=estimated_r,
                    current_time=env.state.current_time,
                    last_transfer_time=env.state.last_transfer_time,
                    h0=env.state.h0,
                    device=torch_device,
                )

            # 计算当前状态势 Φ(s_t)
            if shaper is not None:
                phi_current = shaper.compute_potential(
                    state_feat=s_feat,
                    estimated_cmax=cmax_est,
                    current_time=float(env.state.current_time),
                    h0=float(env.state.h0),
                    last_transfer_time=float(env.state.last_transfer_time),
                    is_terminal=False,
                    graph_data=graph_snapshot,
                )
            else:
                phi_current = compute_heuristic_potential(
                    cmax_est,
                    env.state.last_transfer_time,
                    env.state.h0,
                )

            # 采样条件动作
            act, lp, v, rec = actor_critic.select_action(
                env=env,
                state_feat=s_feat,
                time_urgency=u_time,
                deterministic=False,
            )

            if act is None:
                raise RuntimeError("Actor未返回动作；无候选状态应通过强制推进动作进入env.step()")

            cycle_id = int(env.state.current_cycle)
            if profile.use_time_auxiliary and shaper is not None:
                pending_time_labels.add(
                    episode_id=episode_id,
                    cycle_id=cycle_id,
                    decision_id=total_env_steps,
                    state_feat=s_feat,
                    graph_snapshot=rec.get("graph_snapshot", graph_snapshot),
                    estimated_cmax=cmax_est,
                    current_time=float(env.state.current_time),
                    h0=float(env.state.h0),
                    predictor_version=shaper.snapshot_version,
                    time_urgency=u_time,
                )

            # 环境执行一步调度动作
            transfer_count_before = len(env.state.transfer_history)
            obs, raw_reward, terminated, truncated, info = env.step(act)
            update_current_scenario_effect_log()
            done = terminated or truncated
            last_terminated = bool(terminated)
            discard_unlabelled_time_samples = False
            if terminated:
                episode_done = True
                episode_termination_reason = str(
                    info.get(
                        "termination_reason",
                        "completed" if env._check_terminated() else "deadlock",
                    )
                )
                if current_scenario_log is not None:
                    episode_success = env._check_terminated()
                    current_scenario_log["completed"] = episode_success
                    current_scenario_log["success"] = episode_success
                    current_scenario_log["truncated"] = False
                    current_scenario_log["termination_reason"] = episode_termination_reason
                if not env._check_terminated():
                    discard_unlabelled_time_samples = True

            elif truncated:
                episode_done = True
                episode_termination_reason = "truncated"
                discard_unlabelled_time_samples = True
                if current_scenario_log is not None:
                    current_scenario_log["completed"] = False
                    current_scenario_log["success"] = False
                    current_scenario_log["truncated"] = True
                    current_scenario_log["termination_reason"] = "truncated"

            actual_transfers = env.state.transfer_history[transfer_count_before:]
            for offset, actual_transfer_time in enumerate(actual_transfers):
                transfer_cycle_id = cycle_id + offset
                label_count = pending_time_labels.pending_cycle_counts(episode_id).get(
                    transfer_cycle_id,
                    0,
                )
                label_available = pending_time_labels.attach_transfer(
                    episode_id=episode_id,
                    cycle_id=transfer_cycle_id,
                    actual_transfer_time=float(actual_transfer_time),
                )
                if profile.use_time_auxiliary:
                    cycle_time_labels.append({
                        "episode_id": episode_id,
                        "cycle_id": transfer_cycle_id,
                        "actual_transfer_time": float(actual_transfer_time),
                        "label_count": int(label_count) if label_available else 0,
                        "pending_decision_count": 0,
                        "label_available": bool(label_available),
                    })
                if current_scenario_log is not None:
                    current_scenario_log["actual_transfer_times"].append(
                        float(actual_transfer_time)
                    )
            if discard_unlabelled_time_samples:
                record_unlabelled_cycles(episode_id)
                pending_time_labels.discard_episode(episode_id)

            # 计算下一状态势 Φ(s_{t+1}) 与塑形奖励
            if terminated:
                phi_next = 0.0
            else:
                next_cmax_est = compute_cycle_heuristic_cmax(env.state)
                next_s_feat = extract_compact_state_features(env.state, next_cmax_est)
                next_graph_snapshot = actor_critic.build_graph_snapshot(env)
                if shaper is not None and profile.use_corrected_time_input and time_head is not None:
                    next_graph_snapshot, _, _ = compute_online_time_inputs(
                        actor_critic=actor_critic,
                        time_head=time_head,
                        env=env,
                        state_feat=next_s_feat,
                        estimated_cmax=next_cmax_est,
                    )
                    phi_next = shaper.compute_potential(
                        state_feat=next_s_feat,
                        estimated_cmax=next_cmax_est,
                        current_time=float(env.state.current_time),
                        h0=float(env.state.h0),
                        last_transfer_time=float(env.state.last_transfer_time),
                        is_terminal=False,
                        graph_data=next_graph_snapshot,
                    )
                else:
                    phi_next = compute_heuristic_potential(
                        next_cmax_est,
                        env.state.last_transfer_time,
                        env.state.h0,
                    )

            if shaper is not None:
                shaped_reward = shaper.shape_reward(
                    actual_reward=raw_reward,
                    phi_current=phi_current,
                    phi_next=phi_next,
                )
            else:
                shaped_reward = float(raw_reward) + beta_shaping * (
                    gamma * float(phi_next) - float(phi_current)
                )

            buffer.add(PPOTransition(
                state_feat=s_feat.cpu(),
                time_urgency=u_time.cpu(),
                sample_record=rec,
                reward=shaped_reward,
                raw_reward=raw_reward,
                value=v,
                log_prob=lp,
                done=done,
                action_dict=act,
                terminated=terminated,
                truncated=truncated,
            ))

            step_raw_rewards.append(raw_reward)
            step_shaped_rewards.append(shaped_reward)
            total_env_steps += 1

            if (
                run_mode == "pilot"
                and sum(item.get("success") is True for item in scenario_log)
                >= int(successful_batch_target or 0)
            ):
                stop_reason = "successful_batch_target_reached"
                break
            budget_reason = current_budget_reason()
            if budget_reason is not None:
                if not terminated and not truncated:
                    mark_budget_truncated(budget_reason)
                else:
                    stop_reason = budget_reason
                break

        if current_scenario_log is not None:
            update_current_scenario_effect_log()

        # -------------------------
        # GAE 与价值目标结算
        # -------------------------
        # 末尾 Bootstrap 状态值
        if len(buffer) > 0:
            last_cmax = compute_cycle_heuristic_cmax(env.state)
            last_s_feat = extract_compact_state_features(env.state, last_cmax)
            last_graph = actor_critic.build_graph_snapshot(env)
            if profile.use_corrected_time_input and time_head is not None:
                last_graph, last_u_time, _ = compute_online_time_inputs(
                    actor_critic=actor_critic,
                    time_head=time_head,
                    env=env,
                    state_feat=last_s_feat,
                    estimated_cmax=last_cmax,
                )
            else:
                last_u_time = compute_time_urgency_vector(
                    estimated_r=max(0.0, last_cmax - float(env.state.current_time)),
                    current_time=env.state.current_time,
                    last_transfer_time=env.state.last_transfer_time,
                    h0=env.state.h0,
                    device=torch_device,
                )
            with torch.no_grad():
                last_v, _ = actor_critic.encode_state(
                    last_s_feat.to(torch_device),
                    last_u_time.to(torch_device),
                    graph_data=last_graph,
                )
                last_val = float(last_v.squeeze().item()) if not last_terminated else 0.0

            buffer.finish_trajectory(last_value=last_val)
            time_auxiliary_batch = (
                pending_time_labels.drain_ready()
                if profile.use_time_auxiliary
                else None
            )

            # -------------------------
            # PPO 训练更新步
            # -------------------------
            metrics = trainer.train_step(
                buffer=buffer,
                ppo_epochs=ppo_epochs,
                batch_size=batch_size,
                time_auxiliary_batch=time_auxiliary_batch,
                time_auxiliary_epochs=time_auxiliary_epochs,
                time_auxiliary_batch_size=time_auxiliary_batch_size,
            )
        else:
            metrics = {}

        iter_elapsed = time.monotonic() - iter_start
        mean_raw_r = float(np.mean(step_raw_rewards)) if step_raw_rewards else 0.0
        mean_shaped_r = float(np.mean(step_shaped_rewards)) if step_shaped_rewards else 0.0

        iter_log = {
            "iteration": iter_idx,
            "total_steps": total_env_steps,
            "policy_loss": metrics.get("policy_loss", 0.0),
            "value_loss": metrics.get("value_loss", 0.0),
            "entropy": metrics.get("entropy", 0.0),
            "total_loss": metrics.get("total_loss", 0.0),
            "approx_kl": metrics.get("approx_kl", 0.0),
            "clip_fraction": metrics.get("clip_fraction", 0.0),
            "grad_norm": metrics.get("grad_norm", 0.0),
            "time_loss": metrics.get("time_loss", 0.0),
            "time_label_count": metrics.get("time_label_count", 0),
            "time_supervision_steps": metrics.get("time_supervision_steps", 0),
            "time_supervision_epochs": metrics.get("time_supervision_epochs", 0),
            "time_supervision_status": (
                "not_applicable_method_c" if not profile.use_time_auxiliary
                else "updated" if metrics.get("time_supervision_steps", 0) > 0
                else "skipped_no_real_transfer_labels"
            ),
            "sampling_replay_max_abs_error": metrics.get("sampling_replay_max_abs_error"),
            "sampling_replay_sample_count": metrics.get("sampling_replay_sample_count", 0),
            "sampling_replay_scope": "first_pre_update_minibatch_per_rollout",
            "ppo_updates": metrics.get("num_updates", 0),
            "mean_raw_reward": mean_raw_r,
            "mean_shaped_reward": mean_shaped_r,
            "elapsed_seconds": iter_elapsed,
            "method_variant": profile.name,
            "scenario_ids": sorted({item["scenario_id"] for item in scenario_log}),
            "scenario_log_count": len(scenario_log),
        }
        history.append(iter_log)
        logger.info(
            "采样/重放一致性 max_abs_error=%s, 样本数=%s; 时间监督状态=%s, 标签数=%s",
            iter_log["sampling_replay_max_abs_error"],
            iter_log["sampling_replay_sample_count"],
            iter_log["time_supervision_status"],
            iter_log["time_label_count"],
        )

        logger.info(
            f"Iter [{iter_idx:02d}] "
            f"Steps={total_env_steps:4d} | "
            f"Loss(Tot={iter_log['total_loss']:+.4f}, Pol={iter_log['policy_loss']:+.4f}, "
            f"Val={iter_log['value_loss']:.4f}) | "
            f"Ent={iter_log['entropy']:.3f} | "
            f"KL={iter_log['approx_kl']:.5f} | "
            f"Clip={iter_log['clip_fraction']:.3f} | "
            f"R_raw={mean_raw_r:.3f} | R_shaped={mean_shaped_r:.3f} | "
            f"Time={iter_elapsed:.1f}s"
        )

        if stop_reason is not None:
            break

    if stop_reason is None:
        stop_reason = "iteration_limit"
    if current_scenario_log is not None and current_scenario_log["success"] is None:
        mark_budget_truncated(stop_reason)

    total_elapsed = time.monotonic() - run_started
    successful_batch_count = sum(item.get("success") is True for item in scenario_log)
    failure_reasons: dict[str, int] = {}
    for item in scenario_log:
        if item.get("success") is not True:
            reason = str(item.get("termination_reason") or "unknown")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    first_scheduled = next(
        (item for item in scenario_log if item.get("disturbance_scheduled")),
        None,
    )
    first_hit = next(
        (item for item in scenario_log if int(item.get("actual_hit_count") or 0) > 0),
        None,
    )
    first_transfer = next(
        (
            {"episode_id": item["episode_id"], "transfer_time": transfer_time}
            for item in scenario_log
            for transfer_time in item.get("actual_transfer_times", [])
        ),
        None,
    )
    replay_errors = [
        float(item["sampling_replay_max_abs_error"])
        for item in history
        if item.get("sampling_replay_max_abs_error") is not None
    ]
    memory_peak = _peak_memory_bytes(torch_device)
    device_name = (
        torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda"
        else platform.processor() or platform.machine()
    )
    report: dict[str, Any] = {
        "report_version": "work3_training_run_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_mode": run_mode,
        "disturbance_enabled": run_mode == "pilot",
        "run_status": "successful_batch_target_reached"
        if stop_reason == "successful_batch_target_reached"
        else "budget_or_iteration_ended",
        "terminated": stop_reason == "successful_batch_target_reached",
        "truncated": stop_reason != "successful_batch_target_reached",
        "termination_reason": stop_reason,
        "research_result_eligible": False,
        "method_variant": profile.name,
        "seed": int(seed),
        "seed_role": "training_initialization_and_episode_schedule",
        "source_sha": source_sha,
        "source_tree_dirty": source_tree_dirty,
        "training_config": config,
        "data_fingerprint": data_fingerprint,
        "event_plan": event_plan,
        "event_plan_fingerprint": data_fingerprint["event_plan_sha256"],
        "planned_scenario_ids": [item["scenario_id"] for item in event_plan],
        "initial_parameter_fingerprint": initial_parameter_fingerprint,
        "device": str(torch_device),
        "device_name": device_name,
        "runtime_versions": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
        },
        "elapsed_seconds": total_elapsed,
        "memory_peak_bytes": memory_peak,
        "memory_peak_kind": (
            "cuda_max_memory_allocated" if torch_device.type == "cuda"
            else "process_peak_working_set_or_rss"
        ),
        "total_decisions": total_env_steps,
        "independent_episode_count": len({item["episode_id"] for item in scenario_log}),
        "successful_batch_count": int(successful_batch_count),
        "failure_reasons": failure_reasons,
        "actual_disturbance_hit_count": sum(
            int(item.get("actual_hit_count") or 0) for item in scenario_log
        ),
        "actual_disturbance_hit_episode_count": sum(
            int(item.get("actual_hit_count") or 0) > 0 for item in scenario_log
        ),
        "first_scheduled_disturbance": None if first_scheduled is None else {
            key: first_scheduled.get(key)
            for key in (
                "episode_id", "scenario_id", "scheduled_tau",
                "scheduled_recovery_time", "scheduled_affected_task_keys",
            )
        },
        "first_actual_hit": None if first_hit is None else {
            "episode_id": first_hit["episode_id"],
            "scenario_id": first_hit["scenario_id"],
            "actual_hit_count": first_hit["actual_hit_count"],
        },
        "first_actual_transfer": first_transfer,
        "cycle_time_labels": cycle_time_labels,
        "sampling_replay_max_abs_error": max(replay_errors) if replay_errors else None,
        "sampling_replay_sample_count": sum(
            int(item.get("sampling_replay_sample_count", 0)) for item in history
        ),
        "sampling_replay_scope": "first_pre_update_minibatch_per_rollout",
        "checkpoint_path": str(output_ckpt),
        "checkpoint_version": PPO_CHECKPOINT_VERSION,
        "checkpoint_role": "model_weights",
        "resume_capability": "non_exact",
        "history": history,
        "scenario_log": scenario_log,
    }
    if paired_report_path is not None:
        paired_report = json.loads(paired_report_path.read_text(encoding="utf-8"))
        report["paired_run_check"] = _compare_training_reports(report, paired_report)
        report["paired_report_path"] = str(paired_report_path)

    checkpoint_metadata = {
        "run_mode": run_mode,
        "seed": int(seed),
        "source_sha": source_sha,
        "source_tree_dirty": source_tree_dirty,
        "training_config": config,
        "data_fingerprint": data_fingerprint,
        "event_plan_fingerprint": data_fingerprint["event_plan_sha256"],
        "termination_reason": stop_reason,
        "successful_batch_count": int(successful_batch_count),
        "independent_episode_count": report["independent_episode_count"],
        "resume_capability": "non_exact",
    }
    trainer.save_checkpoint(str(output_ckpt), metadata=checkpoint_metadata)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    logger.info("=" * 80)
    logger.info(
        "工作三%s结束：reason=%s, steps=%s, episodes=%s, successes=%s, hits=%s, report=%s",
        run_mode,
        stop_reason,
        total_env_steps,
        report["independent_episode_count"],
        successful_batch_count,
        report["actual_disturbance_hit_count"],
        report_path,
    )
    logger.info("=" * 80)

    return {
        **report,
        "history": history,
        "total_steps": total_env_steps,
        "total_elapsed_seconds": total_elapsed,
        "checkpoint_path": output_ckpt,
        "report_path": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="工作三烟测/训练试点入口（不生成正式实验结论）")
    parser.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--seed", type=int, default=42, help="训练初始化及episode计划种子")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="可选rollout段数上限；smoke固定为1，pilot默认由成功目标/预算终止",
    )
    parser.add_argument("--steps", type=int, default=32, help="每个rollout段的最大决策步数")
    parser.add_argument("--epochs", type=int, default=1, help="PPO重放轮数；smoke固定为1")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch 大小 (默认 64)")
    parser.add_argument("--successful-batch-target", type=int, default=None)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--max-wall-seconds", type=float, default=None)
    parser.add_argument(
        "--time-auxiliary-epochs",
        type=int,
        default=1,
        help="每次PPO采样段中已补真实标签的独立监督轮数 (默认 1)",
    )
    parser.add_argument(
        "--time-auxiliary-batch-size",
        type=int,
        default=64,
        help="时间辅助监督 mini-batch 大小 (默认 64)",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率 (默认 3e-4)")
    parser.add_argument("--method", choices=("C", "D"), default="D", help="待验证的方法配置，不代表正式结果")
    parser.add_argument("--baseline", type=Path, default=Path("data/work3/real_283_k10_baseline.json"))
    parser.add_argument("--scenarios", type=Path, default=Path("data/work3/scenarios_9class.json"))
    parser.add_argument("--scenario-split", type=Path, default=Path("data/work3/experiment_splits/train.json"))
    parser.add_argument("--time-head-checkpoint", type=Path, default=Path("models/work3/checkpoints/time_head_best.pt"))
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu/cuda)")
    parser.add_argument("--output", type=Path, default=Path("models/work3/checkpoints/method_d_model.pt"), help="模型权重检查点路径")
    parser.add_argument("--report", type=Path, default=None, help="运行报告路径；默认与检查点同名.run.json")
    parser.add_argument("--paired-report", type=Path, default=None, help="另一组C/D训练报告路径，用于核对事件、预算及实际命中")
    args = parser.parse_args()

    run_training(
        num_iterations=args.iterations,
        steps_per_iter=args.steps,
        ppo_epochs=args.epochs,
        batch_size=args.batch_size,
        time_auxiliary_epochs=args.time_auxiliary_epochs,
        time_auxiliary_batch_size=args.time_auxiliary_batch_size,
        lr=args.lr,
        seed=args.seed,
        run_mode=args.mode,
        successful_batch_target=args.successful_batch_target,
        max_decisions=args.max_decisions,
        max_wall_seconds=args.max_wall_seconds,
        method_variant=args.method,
        baseline_path=args.baseline,
        scenarios_path=args.scenarios,
        scenario_split_path=args.scenario_split,
        time_head_ckpt=args.time_head_checkpoint,
        device=args.device,
        output_ckpt=args.output,
        report_path=args.report,
        paired_report_path=args.paired_report,
    )


if __name__ == "__main__":
    main()
