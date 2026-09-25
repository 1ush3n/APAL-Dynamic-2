"""工作三条件分支PPO烟测与训练试点入口；本脚本不产出正式研究结论。

功能范围：
1. 烟测使用无扰动小预算；试点使用固定训练事件清单和可配置退出预算；
2. 检查图策略、条件分支动作、GAE、势函数及时间监督的训练通路；
3. 记录损失、重放误差、真实命中、转站、批次结果和资源使用；不将试点指标解释为方法效果。
4. 保存带运行元数据的模型权重；检查点不支持精确断点续训。
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import wraps
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
from typing import Any, Callable

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import lightning.pytorch as pl

from envs.work3.environment import AirLineEnvWork3
from envs.work3.decision_snapshot import DecisionSnapshot
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.actor_critic import ActorCriticWork3
from models.work3.graph_builder import GRAPH_FEATURE_VERSION
from models.work3.potential_shaping import PotentialRewardShaper
from models.work3.ppo_buffer import PendingTimeLabelCache, PPOTransition, RolloutBufferWork3
from models.work3.ppo_trainer import PPO_CHECKPOINT_VERSION
from models.work3.time_head import TimeResidualHead
from scripts.work3.collect_validation_trajectories import load_scenarios_for_split
from scripts.work3.experiment_protocol import Work3MethodProfile, build_method_profile
from training.work3_vector_env import Work3VectorEnv
from training.work3_lightning import Work3PPODataModule, Work3LightningModule, Work3TrainingUpdate
from training.work3_runtime_config import (
    apply_work3_runtime_overrides,
    load_work3_runtime_config,
    resolved_config_fingerprint,
    resolve_work3_precision,
    seed_work3_runtime,
)
from utils.work3.trajectory_feasibility import (
    TaskConstraintRecord,
    TrajectoryExecutionRecord,
    validate_trajectory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

_ACTIVE_TRAINING_ENVS: ContextVar[list[Work3VectorEnv] | None] = ContextVar(
    "work3_active_training_envs",
    default=None,
)


def _close_training_envs_on_exit(
    function: Callable[..., Any],
) -> Callable[..., Any]:
    """在训练正常或异常退出时回收本次调用启动的向量环境。"""
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        environments: list[Work3VectorEnv] = []
        token = _ACTIVE_TRAINING_ENVS.set(environments)
        try:
            result = function(*args, **kwargs)
        except BaseException:
            for environment in reversed(environments):
                try:
                    environment.close()
                except Exception:
                    logger.exception("训练异常期间关闭工作三环境失败")
            raise
        else:
            for environment in reversed(environments):
                environment.close()
            return result
        finally:
            _ACTIVE_TRAINING_ENVS.reset(token)

    return wrapped


def _autocast_context(
    device: torch.device,
    dtype: torch.dtype | None,
) -> AbstractContextManager[None]:
    if dtype is None:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


@dataclass(slots=True)
class _TrainingWorkerEpisode:
    worker_id: int
    episode_id: int
    episode_index: int
    scenario: dict[str, Any]
    scenario_status: dict[str, Any]
    snapshot: DecisionSnapshot
    scenario_log: dict[str, Any]
    done: bool = False
    success: bool | None = None
    termination_reason: str | None = None
    predictor_version: int | None = None


def create_work3_single_env_runtime(
    baseline_path: str | Path,
    *,
    num_envs: int = 1,
    worker_torch_num_threads: int = 1,
) -> Work3VectorEnv:
    """构造FP32 spawn环境worker池；策略与PPO优化器仍由调用方持有。"""
    environment = Work3VectorEnv(
        env_kwargs={"baseline_json_path": str(Path(baseline_path))},
        num_envs=num_envs,
        worker_torch_num_threads=worker_torch_num_threads,
        start_method="spawn",
    )
    active_environments = _ACTIVE_TRAINING_ENVS.get()
    if active_environments is not None:
        active_environments.append(environment)
    return environment


def _seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """锁定本次训练用到的Python、NumPy、PyTorch及CUDA随机源。"""
    seed_work3_runtime(seed, deterministic=deterministic)


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


def _peak_host_memory_bytes() -> int | None:
    """读取训练主进程峰值常驻内存；平台不支持时返回None。"""
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


def _peak_gpu_memory_bytes(device: torch.device) -> int | None:
    """读取本次训练使用设备的峰值分配显存；未使用CUDA时返回None。"""
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def _module_fingerprint(modules: dict[str, torch.nn.Module | None]) -> str:
    digest = hashlib.sha256()
    for module_name, module in sorted(modules.items()):
        if module is None:
            continue
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(f"{module_name}.{name}".encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _independent_feasibility_from_audit(audit: Any) -> list[dict[str, Any]]:
    """只从worker导出的执行审计重建完成轨迹并独立核验。"""
    audits = (audit,) if isinstance(audit, dict) else audit
    if not isinstance(audits, (tuple, list)):
        raise TypeError("轨迹审计必须是worker审计字典或其序列")

    results: list[dict[str, Any]] = []
    for worker_id, worker_audit in enumerate(audits):
        if worker_audit is None:
            results.append({
                "worker_id": worker_id,
                "status": "audit_unavailable",
                "batch_complete": False,
                "feasible": None,
                "execution_record_count": 0,
                "violations": {},
            })
            continue
        if not isinstance(worker_audit, dict):
            raise TypeError("worker轨迹审计必须是字典或None")

        records_data = worker_audit.get("execution_records", ())
        completed = (
            worker_audit.get("success") is True
            and int(worker_audit.get("completed_tasks", -1))
            == int(worker_audit.get("total_tasks", -2))
        )
        if not completed:
            results.append({
                "worker_id": worker_id,
                "status": "incomplete_not_assessed",
                "batch_complete": False,
                "feasible": None,
                "execution_record_count": len(records_data),
                "violations": {},
            })
            continue
        if len(records_data) != int(worker_audit["total_tasks"]):
            results.append({
                "worker_id": worker_id,
                "status": "audit_inconsistent",
                "batch_complete": True,
                "feasible": False,
                "execution_record_count": len(records_data),
                "violations": {"execution_record_count_mismatch": 1},
            })
            continue

        constraints = {
            int(task_id): TaskConstraintRecord(
                demand=int(item["demand"]),
                required_skill=int(item["required_skill"]),
                predecessors=tuple(int(value) for value in item["predecessors"]),
                fixed_station=(
                    None if item["fixed_station"] is None
                    else int(item["fixed_station"])
                ),
                max_allowed_station=(
                    None if item["max_allowed_station"] is None
                    else int(item["max_allowed_station"])
                ),
            )
            for task_id, item in worker_audit["task_constraints"].items()
        }
        execution_records = [
            TrajectoryExecutionRecord(
                aircraft_id=int(item["aircraft_id"]),
                task_id=int(item["task_id"]),
                station_id=int(item["station_id"]),
                team=tuple(int(value) for value in item["team"]),
                start=float(item["start"]),
                end=float(item["end"]),
                material_ready_time=float(item["material_ready_time"]),
                station_entry_time=(
                    None if item["station_entry_time"] is None
                    else float(item["station_entry_time"])
                ),
                aircraft_station_at_start=int(item["aircraft_station_at_start"]),
            )
            for item in records_data
        ]
        feasibility = validate_trajectory(
            execution_records,
            task_constraints=constraints,
            worker_skills={
                int(worker): tuple(int(skill) for skill in skills)
                for worker, skills in worker_audit["worker_skills"].items()
            },
            worker_station_bindings={
                int(worker): int(station)
                for worker, station in worker_audit["worker_station_bindings"].items()
            },
            station_capacities={
                int(station): int(capacity)
                for station, capacity in worker_audit["station_capacities"].items()
            },
        )
        results.append({
            "worker_id": worker_id,
            "status": "feasible" if feasibility.is_feasible else "violations",
            "batch_complete": True,
            "feasible": feasibility.is_feasible,
            "execution_record_count": len(execution_records),
            "violations": dict(feasibility.violations),
        })
    return results


def _should_stop_after_success_target(
    *,
    run_mode: str,
    successful_batch_target: int | None,
    successful_batch_count: int,
    active_episode_count: int,
) -> bool:
    """成功目标达成后，仅在所有已启动episode结束时停止。"""
    return (
        run_mode == "pilot"
        and successful_batch_target is not None
        and successful_batch_count >= successful_batch_target
        and active_episode_count == 0
    )


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


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
        "num_envs",
    )
    current_config = current.get("training_config", {})
    paired_config = paired.get("training_config", {})
    current_hits = {
        (
            item.get("worker_id"),
            item.get("episode_id"),
            item.get("scenario_id"),
        ): int(item.get("actual_hit_count") or 0)
        for item in current.get("scenario_log", [])
    }
    paired_hits = {
        (
            item.get("worker_id"),
            item.get("episode_id"),
            item.get("scenario_id"),
        ): int(item.get("actual_hit_count") or 0)
        for item in paired.get("scenario_log", [])
    }
    current_event_fingerprint = current.get(
        "worker_event_plan_fingerprint",
        current.get("event_plan_fingerprint"),
    )
    paired_event_fingerprint = paired.get(
        "worker_event_plan_fingerprint",
        paired.get("event_plan_fingerprint"),
    )
    same_event_plan = current_event_fingerprint == paired_event_fingerprint
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


def build_worker_scenario_plan(
    scenarios: list[dict[str, Any]],
    *,
    num_workers: int,
    episodes_per_worker: int,
    seed: int,
) -> list[dict[str, Any]]:
    """固定生成按``(worker_id, episode_index)``索引的外生事件计划。"""
    if type(num_workers) is not int or num_workers < 1:
        raise ValueError("num_workers必须为正整数")
    if type(episodes_per_worker) is not int or episodes_per_worker < 1:
        raise ValueError("episodes_per_worker必须为正整数")
    flat_plan = build_episode_scenario_plan(
        scenarios,
        num_workers * episodes_per_worker,
        seed=seed,
    )
    return [
        {
            "worker_id": worker_id,
            "episode_index": episode_index,
            "episode_id": episode_index * num_workers + worker_id,
            "scenario": dict(flat_plan[episode_index * num_workers + worker_id]),
        }
        for episode_index in range(episodes_per_worker)
        for worker_id in range(num_workers)
    ]


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


def compute_online_snapshot_time_inputs(
    actor_critic: ActorCriticWork3,
    time_head: TimeResidualHead,
    snapshot: DecisionSnapshot,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """只依赖CPU决策快照生成修正预测，不访问worker内环境对象。"""
    device = next(actor_critic.parameters()).device
    state_features = snapshot.state_features.to(device)
    graph_snapshot = snapshot.graph_snapshot
    estimated_cmax = snapshot.estimated_cmax
    if estimated_cmax is None:
        raise ValueError("时间修正要求决策快照包含启发式完工预测")
    with torch.no_grad():
        shared_feature = actor_critic.encode_shared_representation(
            state_features,
            graph_snapshot,
        ).unsqueeze(0)
        predicted_transfer, remaining, _ = time_head.predict_corrected_time(
            state_feat=shared_feature,
            estimated_cmax=estimated_cmax,
            current_time=snapshot.current_time,
            h0=snapshot.h0,
            last_transfer_time=snapshot.last_transfer_time,
        )
        urgency = compute_time_urgency_vector(
            estimated_r=remaining.squeeze(0),
            current_time=snapshot.current_time,
            last_transfer_time=snapshot.last_transfer_time,
            h0=snapshot.h0,
            device=device,
        )
    return graph_snapshot, urgency, predicted_transfer.squeeze(0)


@_close_training_envs_on_exit
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
    num_envs: int = 1,
    env_num_threads: int = 1,
    deterministic: bool = True,
    main_num_threads: int = 1,
    settle_timeout_seconds: float | None = None,
    resolved_config_yaml: str | None = None,
    resolved_config_sha256: str | None = None,
    amp_dtype: str = "fp32",
) -> dict[str, Any]:
    """执行不具研究结论资格的烟测或有明确预算的训练试点。"""
    run_started = time.monotonic()
    if run_mode not in {"smoke", "pilot"}:
        raise ValueError("run_mode必须是'smoke'或'pilot'；正式训练预算尚未定义")
    if (
        (num_iterations is not None and num_iterations < 1)
        or steps_per_iter < 1
        or type(num_envs) is not int
        or num_envs < 1
        or type(env_num_threads) is not int
        or env_num_threads < 1
        or type(deterministic) is not bool
        or type(main_num_threads) is not int
        or main_num_threads < 1
        or ppo_epochs < 0
        or batch_size < 1
    ):
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
    if settle_timeout_seconds is not None and (
        not math.isfinite(settle_timeout_seconds) or settle_timeout_seconds <= 0.0
    ):
        raise ValueError("settle_timeout_seconds必须为有限正数")
    if (resolved_config_yaml is None) != (resolved_config_sha256 is None):
        raise ValueError("resolved_config_yaml与resolved_config_sha256必须同时提供")
    if resolved_config_yaml is not None:
        observed_hash = hashlib.sha256(resolved_config_yaml.encode("utf-8")).hexdigest()
        if observed_hash != resolved_config_sha256:
            raise ValueError("解析后YAML与其SHA256不匹配")

    torch_device = torch.device(device)
    lightning_precision, autocast_dtype = resolve_work3_precision(
        amp_dtype,
        torch_device,
    )
    profile = build_method_profile(method_variant)
    torch.set_num_threads(main_num_threads)
    _seed_everything(seed, deterministic=deterministic)

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
    worker_scenario_source = training_scenarios if run_mode == "pilot" else episode_plan
    worker_event_plan = build_worker_scenario_plan(
        worker_scenario_source,
        num_workers=num_envs,
        episodes_per_worker=len(episode_plan),
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
    env = create_work3_single_env_runtime(
        baseline_path,
        num_envs=num_envs,
        worker_torch_num_threads=env_num_threads,
    )

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

    # 3. Lightning持有唯一优化器；PPO对象只提供纯损失和检查点接口。
    lightning_module = Work3LightningModule(
        actor_critic=actor_critic,
        time_head=time_head if profile.use_time_auxiliary else None,
        learning_rate=lr,
        clip_eps=clip_eps,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        time_loss_coef=time_loss_coef if profile.use_time_auxiliary else 0.0,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size,
        time_auxiliary_epochs=time_auxiliary_epochs,
        time_auxiliary_batch_size=time_auxiliary_batch_size,
    )
    trainer = lightning_module.objective

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
        "worker_event_plan_sha256": _json_fingerprint(worker_event_plan),
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
        "num_envs": int(num_envs),
        "main_num_threads": int(main_num_threads),
        "env_num_threads": int(env_num_threads),
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
        "deterministic": deterministic,
        "settle_timeout_seconds": settle_timeout_seconds,
        "resolved_config_sha256": resolved_config_sha256,
        "baseline_path": str(Path(baseline_path)),
        "scenarios_path": str(Path(scenarios_path)),
        "scenario_split_path": str(Path(scenario_split_path)),
        "time_head_checkpoint_path": str(Path(time_head_ckpt)),
        "output_checkpoint_path": str(output_ckpt),
        "report_path": str(report_path),
        "paired_report_path": None if paired_report_path is None else str(paired_report_path),
        "device": str(torch_device),
        "amp_dtype": amp_dtype,
        "lightning_precision": lightning_precision,
    }
    initial_actor_fingerprint = _module_fingerprint({"actor_critic": actor_critic})
    initial_parameter_fingerprint = _module_fingerprint({
        "actor_critic": actor_critic,
        "time_head": time_head,
    })
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)

    history: list[dict[str, Any]] = []
    total_env_steps = 0
    start_time = run_started
    pending_time_labels = PendingTimeLabelCache()
    scenario_log: list[dict[str, Any]] = []
    cycle_time_labels: list[dict[str, Any]] = []
    lightning_fit_calls = 0
    lightning_grad_scaler_enabled: bool | None = None
    lightning_precision_state: dict[str, Any] = {
        "precision": lightning_precision,
        "amp_dtype": amp_dtype,
        "grad_scaler_state": None,
        "grad_scaler_scale": None,
    }
    stop_reason: str | None = None
    worker_states: list[_TrainingWorkerEpisode] = []
    episode_wave_index = 0

    def update_scenario_effect_log(state: _TrainingWorkerEpisode) -> None:
        state.scenario_log["disturbance_triggered"] = bool(
            state.scenario_status.get("event_triggered", False)
        )
        hit_keys = list(state.scenario_status.get("actual_hit_task_keys", ()))
        state.scenario_log["actual_hit_task_keys"] = hit_keys
        state.scenario_log["actual_hit_count"] = len(hit_keys)

    def start_episode_wave() -> None:
        nonlocal worker_states
        wave = episode_wave_index
        plan_offset = (wave % len(episode_plan)) * num_envs
        plan_entries = worker_event_plan[plan_offset : plan_offset + num_envs]
        if len(plan_entries) != num_envs:
            raise RuntimeError("固定worker/episode事件计划没有覆盖当前episode wave")
        scenarios = [dict(item["scenario"]) for item in plan_entries]
        episode_ids = [wave * num_envs + worker_id for worker_id in range(num_envs)]
        reset_results = env.reset_all(
            scenarios=[scenario if run_mode == "pilot" else None for scenario in scenarios],
            episode_ids=episode_ids,
            episode_indices=[wave] * num_envs,
        )
        snapshots = env.snapshots()
        worker_states = []
        for worker_id, (scenario, reset_result, snapshot) in enumerate(
            zip(scenarios, reset_results, snapshots, strict=True)
        ):
            episode_id = episode_ids[worker_id]
            scenario_log_entry = {
                "worker_id": worker_id,
                "episode_index": wave,
                "episode_id": episode_id,
                "scenario_id": scenario["scenario_id"],
                "timing": scenario["timing"],
                "intensity": scenario["intensity"],
                "station_id": scenario["station_id"],
                "aircraft_id": scenario["aircraft_id"],
                "disturbance_scheduled": run_mode == "pilot",
                "scheduled_tau": scenario.get("tau") if run_mode == "pilot" else None,
                "scheduled_recovery_time": scenario.get("recovery_time") if run_mode == "pilot" else None,
                "scheduled_affected_task_keys": list(scenario.get("affected_task_keys", [])) if run_mode == "pilot" else [],
                "scheduled_target_count": (
                    len(scenario.get("affected_task_keys", []))
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
            state = _TrainingWorkerEpisode(
                worker_id=worker_id,
                episode_id=episode_id,
                episode_index=wave,
                scenario=scenario,
                scenario_status=dict(reset_result.scenario_status),
                snapshot=snapshot,
                scenario_log=scenario_log_entry,
            )
            if shaper is not None:
                state.predictor_version = shaper.begin_episode(
                    worker_id=worker_id,
                    episode_id=episode_id,
                )
            scenario_log_entry["potential_snapshot_version"] = state.predictor_version
            worker_states.append(state)
            scenario_log.append(scenario_log_entry)
            update_scenario_effect_log(state)
            logger.info(
                "加载训练worker=%s episode=%s scenario=%s disturbance=%s timing=%s intensity=%s station=%s",
                worker_id,
                episode_id,
                scenario["scenario_id"],
                run_mode == "pilot",
                scenario["timing"],
                scenario["intensity"],
                scenario["station_id"],
            )

    def record_unlabelled_cycles(worker_id: int, target_episode_id: int) -> None:
        if not profile.use_time_auxiliary:
            return
        for pending_cycle_id, count in sorted(
            pending_time_labels.pending_cycle_counts(
                target_episode_id,
                worker_id=worker_id,
            ).items()
        ):
            cycle_time_labels.append({
                "worker_id": int(worker_id),
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
        for state in worker_states:
            if state.done:
                continue
            update_scenario_effect_log(state)
            state.scenario_log.update({
                "completed": False,
                "success": False,
                "truncated": True,
                "termination_reason": reason,
            })
            state.done = True
            state.success = False
            state.termination_reason = reason
            record_unlabelled_cycles(state.worker_id, state.episode_id)
            pending_time_labels.discard_episode(
                state.episode_id,
                worker_id=state.worker_id,
            )
            if shaper is not None and state.predictor_version is not None:
                shaper.end_episode(
                    worker_id=state.worker_id,
                    episode_id=state.episode_id,
                )
                state.predictor_version = None
            for transition in reversed(buffer.transitions):
                if (
                    transition.worker_id == state.worker_id
                    and transition.episode_id == state.episode_id
                    and transition.segment_id == iter_idx
                ):
                    if not transition.terminated:
                        transition.done = True
                        transition.truncated = True
                    break

    def current_budget_reason() -> str | None:
        if max_decisions is not None and total_env_steps >= max_decisions:
            return "decision_limit"
        if max_wall_seconds is not None and time.monotonic() - start_time >= max_wall_seconds:
            return "wall_time_limit"
        return None

    start_episode_wave()

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
        rollout_bootstraps: dict[tuple[int, int, int], float] = {}

        step_raw_rewards: list[float] = []
        step_shaped_rewards: list[float] = []

        # -------------------------
        # Rollout 数据采集循环
        # -------------------------
        steps_this_rollout = 0
        while steps_this_rollout < steps_per_iter and stop_reason is None:
            # ponytail: 同步episode wave简化事件配对与势函数版本引用；轨迹时长差异明显时再改为异步补位。
            if all(state.done for state in worker_states):
                if _should_stop_after_success_target(
                    run_mode=run_mode,
                    successful_batch_target=successful_batch_target,
                    successful_batch_count=sum(
                        item.get("success") is True for item in scenario_log
                    ),
                    active_episode_count=0,
                ):
                    stop_reason = "successful_batch_target_reached"
                    break
                if shaper is not None and any(state.success for state in worker_states):
                    shaper.update_snapshot(time_head, actor_critic)
                episode_wave_index += 1
                start_episode_wave()

            budget_reason = current_budget_reason()
            if budget_reason is not None:
                mark_budget_truncated(budget_reason)
                break

            remaining_global_steps = (
                max_decisions - total_env_steps
                if max_decisions is not None
                else num_envs
            )
            dispatch_limit = min(
                num_envs,
                steps_per_iter - steps_this_rollout,
                remaining_global_steps,
            )
            selected_states = [state for state in worker_states if not state.done][
                :dispatch_limit
            ]
            actions: list[dict[str, Any] | None] = [None] * num_envs
            action_contexts: dict[int, dict[str, Any]] = {}
            for action_offset, state in enumerate(selected_states):
                snapshot = state.snapshot
                cmax_est = snapshot.estimated_cmax
                if cmax_est is None:
                    raise RuntimeError("环境worker快照缺少启发式完工预测")
                state_features = snapshot.state_features
                graph_snapshot = snapshot.graph_snapshot
                action_snapshot = snapshot
                if profile.use_corrected_time_input and time_head is not None:
                    with _autocast_context(torch_device, autocast_dtype):
                        graph_snapshot, time_urgency, _ = compute_online_snapshot_time_inputs(
                            actor_critic=actor_critic,
                            time_head=time_head,
                            snapshot=snapshot,
                        )
                    action_snapshot = replace(snapshot, time_features=time_urgency)
                else:
                    time_urgency = snapshot.time_features

                if shaper is not None:
                    if state.predictor_version is None:
                        raise RuntimeError("势函数episode没有绑定预测器版本")
                    with _autocast_context(torch_device, autocast_dtype):
                        phi_current = shaper.compute_potential(
                            state_feat=state_features,
                            estimated_cmax=cmax_est,
                            current_time=snapshot.current_time,
                            h0=snapshot.h0,
                            last_transfer_time=snapshot.last_transfer_time,
                            is_terminal=False,
                            graph_data=graph_snapshot,
                            worker_id=state.worker_id,
                            episode_id=state.episode_id,
                        )
                else:
                    phi_current = compute_heuristic_potential(
                        cmax_est,
                        snapshot.last_transfer_time,
                        snapshot.h0,
                    )

                with _autocast_context(torch_device, autocast_dtype):
                    action, log_prob, value, sample_record = actor_critic.select_snapshot(
                        action_snapshot,
                        deterministic=False,
                    )
                if action is None:
                    raise RuntimeError(
                        "Actor未返回动作；无候选状态应通过强制推进动作进入env.step()"
                    )
                if profile.use_time_auxiliary and shaper is not None:
                    pending_time_labels.add(
                        episode_id=state.episode_id,
                        cycle_id=int(snapshot.cycle_id),
                        decision_id=total_env_steps + action_offset,
                        state_feat=state_features,
                        graph_snapshot=sample_record.get("graph_snapshot", graph_snapshot),
                        estimated_cmax=cmax_est,
                        current_time=snapshot.current_time,
                        h0=snapshot.h0,
                        predictor_version=state.predictor_version,
                        worker_id=state.worker_id,
                        time_urgency=time_urgency,
                    )
                actions[state.worker_id] = action
                action_contexts[state.worker_id] = {
                    "state_features": state_features,
                    "time_urgency": time_urgency,
                    "sample_record": sample_record,
                    "value": value,
                    "log_prob": log_prob,
                    "phi_current": phi_current,
                    "cycle_id": int(snapshot.cycle_id),
                }

            deadline = (
                None if max_wall_seconds is None else start_time + max_wall_seconds
            )
            step_batch = env.step_all(
                actions=actions,
                max_total_steps=max_decisions,
                settle_timeout_seconds=settle_timeout_seconds,
                wall_clock_deadline=deadline,
            )
            if step_batch.worker_errors or step_batch.interrupted_worker_ids:
                env.close()
                raise RuntimeError(
                    "向量环境step未完整结算："
                    f"errors={step_batch.worker_errors}, "
                    f"interrupted={step_batch.interrupted_worker_ids}"
                )
            if not step_batch.dispatched_worker_ids:
                reason = current_budget_reason() or "no_worker_dispatched"
                mark_budget_truncated(reason)
                break
            if any(
                step_batch.results[worker_id] is None
                for worker_id in step_batch.dispatched_worker_ids
            ):
                raise RuntimeError("向量环境step缺少已派发worker的结果")
            next_snapshots = env.snapshots()

            for worker_id in step_batch.dispatched_worker_ids:
                state = worker_states[worker_id]
                context = action_contexts[worker_id]
                snapshot = next_snapshots[worker_id]
                result = step_batch.results[worker_id]
                if result is None:
                    raise RuntimeError("向量环境worker未返回有效执行结果")
                state.snapshot = snapshot
                state.scenario_status = dict(result.scenario_status)
                update_scenario_effect_log(state)
                raw_reward = float(result.raw_reward)
                terminated = bool(result.terminated)
                truncated = bool(result.truncated)
                done = terminated or truncated
                if terminated:
                    state.done = True
                    state.success = bool(result.info.get("success", False))
                    state.termination_reason = str(
                        result.info.get(
                            "termination_reason",
                            "completed" if state.success else "deadlock",
                        )
                    )
                    state.scenario_log.update({
                        "completed": state.success,
                        "success": state.success,
                        "truncated": False,
                        "termination_reason": state.termination_reason,
                    })
                elif truncated:
                    state.done = True
                    state.success = False
                    state.termination_reason = "truncated"
                    state.scenario_log.update({
                        "completed": False,
                        "success": False,
                        "truncated": True,
                        "termination_reason": "truncated",
                    })

                actual_transfers = [
                    float(event[0])
                    for event in result.processed_events
                    if event[1] == "SYNCHRONOUS_TRANSFER"
                ]
                for offset, actual_transfer_time in enumerate(actual_transfers):
                    transfer_cycle_id = context["cycle_id"] + offset
                    label_count = pending_time_labels.pending_cycle_counts(
                        state.episode_id,
                        worker_id=worker_id,
                    ).get(transfer_cycle_id, 0)
                    label_available = pending_time_labels.attach_transfer(
                        episode_id=state.episode_id,
                        worker_id=worker_id,
                        cycle_id=transfer_cycle_id,
                        actual_transfer_time=actual_transfer_time,
                    )
                    if profile.use_time_auxiliary:
                        cycle_time_labels.append({
                            "worker_id": worker_id,
                            "episode_id": state.episode_id,
                            "cycle_id": transfer_cycle_id,
                            "actual_transfer_time": actual_transfer_time,
                            "label_count": int(label_count) if label_available else 0,
                            "pending_decision_count": 0,
                            "label_available": bool(label_available),
                        })
                    state.scenario_log["actual_transfer_times"].append(
                        actual_transfer_time
                    )

                if terminated:
                    phi_next = 0.0
                else:
                    next_cmax_est = snapshot.estimated_cmax
                    if next_cmax_est is None:
                        raise RuntimeError("环境worker下一状态快照缺少时间预测")
                    if profile.use_corrected_time_input and time_head is not None:
                        with _autocast_context(torch_device, autocast_dtype):
                            next_graph, next_time_urgency, _ = compute_online_snapshot_time_inputs(
                                actor_critic=actor_critic,
                                time_head=time_head,
                                snapshot=snapshot,
                            )
                    else:
                        next_graph = snapshot.graph_snapshot
                        next_time_urgency = snapshot.time_features
                    if shaper is not None:
                        with _autocast_context(torch_device, autocast_dtype):
                            phi_next = shaper.compute_potential(
                                state_feat=snapshot.state_features,
                                estimated_cmax=next_cmax_est,
                                current_time=snapshot.current_time,
                                h0=snapshot.h0,
                                last_transfer_time=snapshot.last_transfer_time,
                                is_terminal=False,
                                graph_data=next_graph,
                                worker_id=worker_id,
                                episode_id=state.episode_id,
                            )
                    else:
                        phi_next = compute_heuristic_potential(
                            next_cmax_est,
                            snapshot.last_transfer_time,
                            snapshot.h0,
                        )

                if truncated:
                    with torch.no_grad(), _autocast_context(torch_device, autocast_dtype):
                        truncated_value, _ = actor_critic.encode_state(
                            snapshot.state_features.to(torch_device),
                            next_time_urgency.to(torch_device),
                            graph_data=next_graph,
                        )
                    rollout_bootstraps[(worker_id, state.episode_id, iter_idx)] = float(
                        truncated_value.squeeze().item()
                    )

                shaped_reward = (
                    shaper.shape_reward(
                        actual_reward=raw_reward,
                        phi_current=context["phi_current"],
                        phi_next=phi_next,
                    )
                    if shaper is not None
                    else raw_reward + beta_shaping * (
                        gamma * float(phi_next) - float(context["phi_current"])
                    )
                )
                if done:
                    record_unlabelled_cycles(worker_id, state.episode_id)
                    pending_time_labels.discard_episode(
                        state.episode_id,
                        worker_id=worker_id,
                    )
                    if shaper is not None and state.predictor_version is not None:
                        shaper.end_episode(
                            worker_id=worker_id,
                            episode_id=state.episode_id,
                        )
                        state.predictor_version = None

                buffer.add(PPOTransition(
                    state_feat=context["state_features"].cpu(),
                    time_urgency=context["time_urgency"].cpu(),
                    sample_record=context["sample_record"],
                    reward=shaped_reward,
                    raw_reward=raw_reward,
                    value=context["value"],
                    log_prob=context["log_prob"],
                    done=done,
                    action_dict=actions[worker_id] or {},
                    terminated=terminated,
                    truncated=truncated,
                    worker_id=worker_id,
                    episode_id=state.episode_id,
                    segment_id=iter_idx,
                ))
                step_raw_rewards.append(raw_reward)
                step_shaped_rewards.append(shaped_reward)
                total_env_steps += 1
                steps_this_rollout += 1

            if _should_stop_after_success_target(
                run_mode=run_mode,
                successful_batch_target=successful_batch_target,
                successful_batch_count=sum(
                    item.get("success") is True for item in scenario_log
                ),
                active_episode_count=sum(not state.done for state in worker_states),
            ):
                stop_reason = "successful_batch_target_reached"
                break

        if stop_reason is None:
            budget_reason = current_budget_reason()
            if budget_reason is not None:
                mark_budget_truncated(budget_reason)

        for state in worker_states:
            if not state.done:
                update_scenario_effect_log(state)

        # -------------------------
        # GAE 与价值目标结算
        # -------------------------
        # 对每条worker/episode/segment独立补齐段尾bootstrap。
        if len(buffer) > 0:
            last_transitions: dict[tuple[int, int, int], PPOTransition] = {}
            for transition in buffer.transitions:
                key = (
                    transition.worker_id,
                    transition.episode_id,
                    transition.segment_id,
                )
                last_transitions[key] = transition
            for key, last_transition in last_transitions.items():
                if last_transition.terminated or key in rollout_bootstraps:
                    continue
                state = worker_states[key[0]]
                if state.episode_id != key[1]:
                    raise RuntimeError(f"rollout段尾状态与episode不一致：{key}")
                snapshot = state.snapshot
                if snapshot.estimated_cmax is None:
                    raise RuntimeError("rollout结束时环境worker快照缺少时间预测")
                if profile.use_corrected_time_input and time_head is not None:
                    with _autocast_context(torch_device, autocast_dtype):
                        graph_snapshot, last_time_urgency, _ = compute_online_snapshot_time_inputs(
                            actor_critic=actor_critic,
                            time_head=time_head,
                            snapshot=snapshot,
                        )
                else:
                    graph_snapshot = snapshot.graph_snapshot
                    last_time_urgency = snapshot.time_features
                with torch.no_grad(), _autocast_context(torch_device, autocast_dtype):
                    last_value, _ = actor_critic.encode_state(
                        snapshot.state_features.to(torch_device),
                        last_time_urgency.to(torch_device),
                        graph_data=graph_snapshot,
                    )
                rollout_bootstraps[key] = float(last_value.squeeze().item())
            buffer.finish_trajectories(
                last_values_by_segment=rollout_bootstraps,
            )
            time_auxiliary_batch = (
                pending_time_labels.drain_ready()
                if profile.use_time_auxiliary
                else None
            )

            update = Work3TrainingUpdate(
                buffer=buffer,
                environment_steps=len(buffer),
                time_auxiliary_batch=time_auxiliary_batch,
            )
            data_module = Work3PPODataModule(
                update_factory=lambda update=update: iter((update,))
            )
            lightning_trainer = pl.Trainer(
                accelerator="gpu" if torch_device.type == "cuda" else "cpu",
                devices=1,
                precision=lightning_precision,
                max_epochs=1,
                limit_train_batches=1,
                num_sanity_val_steps=0,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
                default_root_dir=report_path.parent / ".work3_lightning",
            )
            lightning_trainer.fit(lightning_module, datamodule=data_module)
            scaler = getattr(lightning_trainer.precision_plugin, "scaler", None)
            current_scaler_enabled = bool(
                scaler is not None and scaler.is_enabled()
            )
            if (
                lightning_grad_scaler_enabled is not None
                and lightning_grad_scaler_enabled != current_scaler_enabled
            ):
                raise RuntimeError("不同rollout的Lightning GradScaler状态不一致")
            lightning_grad_scaler_enabled = current_scaler_enabled
            lightning_precision_state = {
                "precision": lightning_precision,
                "amp_dtype": amp_dtype,
                "grad_scaler_state": (
                    scaler.state_dict() if current_scaler_enabled else None
                ),
                "grad_scaler_scale": (
                    float(scaler.get_scale()) if current_scaler_enabled else None
                ),
            }
            # Lightning teardown把模型和优化器状态移回CPU；后续rollout/bootstrap需要主策略留在目标设备。
            lightning_module.to(torch_device)
            lightning_fit_calls += 1
            metrics = lightning_module.last_metrics
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
            "time_supervision_optimizer_updates": metrics.get(
                "time_supervision_optimizer_updates", 0
            ),
            "time_supervision_epochs": metrics.get("time_supervision_epochs", 0),
            "environment_steps": len(buffer),
            "lightning_optimization_steps": int(lightning_module.optimization_steps),
            "lightning_optimizer_step_attempts": int(
                metrics.get("optimizer_step_attempts", 0)
            ),
            "lightning_amp_skipped_steps": int(metrics.get("amp_skipped_steps", 0)),
            "lightning_fit_calls": lightning_fit_calls,
            "time_supervision_status": (
                "not_applicable_method_c" if not profile.use_time_auxiliary
                else "updated" if metrics.get("time_supervision_steps", 0) > 0
                else "skipped_no_real_transfer_labels"
            ),
            "sampling_replay_max_abs_error": metrics.get("sampling_replay_max_abs_error"),
            "sampling_replay_sample_count": metrics.get("sampling_replay_sample_count", 0),
            "sampling_replay_scope": "first_pre_update_minibatch_per_rollout",
            "ppo_updates": metrics.get("ppo_updates", 0),
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
    if any(not state.done for state in worker_states):
        mark_budget_truncated(stop_reason)
    environment_worker_pids = list(env.worker_pids)
    environment_worker_cuda_initialized = list(env.worker_cuda_initialized)
    environment_worker_torch_num_threads = list(env.worker_torch_num_threads)
    worker_step_counts = list(env.worker_step_counts)
    trajectory_audit = env.close()
    independent_feasibility = _independent_feasibility_from_audit(trajectory_audit)

    total_elapsed = time.monotonic() - run_started
    successful_batch_count = sum(item.get("success") is True for item in scenario_log)
    time_label_count = sum(int(item.get("time_label_count", 0)) for item in history)
    time_supervision_steps = sum(
        int(item.get("time_supervision_steps", 0)) for item in history
    )
    time_supervision_optimizer_updates = sum(
        int(item.get("time_supervision_optimizer_updates", 0)) for item in history
    )
    time_head_training_status = (
        "not_applicable_method_c"
        if not profile.use_time_auxiliary
        else "trained_online"
        if time_label_count > 0 and time_supervision_optimizer_updates > 0
        else "untrained_no_successful_online_update"
    )
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
    host_memory_peak = _peak_host_memory_bytes()
    gpu_memory_peak = _peak_gpu_memory_bytes(torch_device)
    memory_peak = gpu_memory_peak if torch_device.type == "cuda" else host_memory_peak
    device_name = (
        torch.cuda.get_device_name(torch_device)
        if torch_device.type == "cuda"
        else platform.processor() or platform.machine()
    )
    required_data_hashes = (
        data_fingerprint.get("scenario_pool_sha256"),
        data_fingerprint.get("scenario_split_sha256"),
        data_fingerprint.get("baseline_sha256"),
        data_fingerprint.get("event_plan_sha256"),
        data_fingerprint.get("worker_event_plan_sha256"),
    )
    resolved_config_valid = (
        resolved_config_yaml is not None
        and resolved_config_sha256 is not None
        and hashlib.sha256(resolved_config_yaml.encode("utf-8")).hexdigest()
        == resolved_config_sha256
    )
    checkpoint_evaluation_eligible = bool(
        run_mode == "pilot"
        and successful_batch_count > 0
        and lightning_module.optimization_steps > 0
        and resolved_config_valid
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in required_data_hashes
        )
        and (
            not profile.use_time_auxiliary
            or time_head_training_status == "trained_online"
        )
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
        "checkpoint_evaluation_eligible": checkpoint_evaluation_eligible,
        "method_variant": profile.name,
        "method_profile": config["method_profile"],
        "seed": int(seed),
        "seed_role": "training_initialization_and_episode_schedule",
        "source_sha": source_sha,
        "source_tree_dirty": source_tree_dirty,
        "training_config": config,
        "resolved_runtime_config_yaml": resolved_config_yaml,
        "resolved_runtime_config_sha256": resolved_config_sha256,
        "data_fingerprint": data_fingerprint,
        "event_plan": event_plan,
        "event_plan_fingerprint": data_fingerprint["event_plan_sha256"],
        "worker_event_plan": worker_event_plan,
        "worker_event_plan_fingerprint": data_fingerprint["worker_event_plan_sha256"],
        "planned_scenario_ids": [item["scenario_id"] for item in event_plan],
        "initial_actor_fingerprint": initial_actor_fingerprint,
        "initial_parameter_fingerprint": initial_parameter_fingerprint,
        "device": str(torch_device),
        "environment_worker_pids": environment_worker_pids,
        "environment_worker_cuda_initialized": environment_worker_cuda_initialized,
        "environment_worker_torch_num_threads": environment_worker_torch_num_threads,
        "environment_worker_start_method": "spawn",
        "worker_step_counts": worker_step_counts,
        "worker_step_settlement_requests": int(env.step_settlement_requests),
        "trajectory_audit": trajectory_audit,
        "independent_feasibility": independent_feasibility,
        "device_name": device_name,
        "amp_dtype": amp_dtype,
        "lightning_precision": lightning_precision,
        "runtime_versions": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "lightning": pl.__version__,
        },
        "elapsed_seconds": total_elapsed,
        "memory_peak_bytes": memory_peak,
        "host_memory_peak_bytes": host_memory_peak,
        "gpu_memory_peak_bytes": gpu_memory_peak,
        "memory_peak_kind": (
            "cuda_max_memory_allocated" if torch_device.type == "cuda"
            else "process_peak_working_set_or_rss"
        ),
        "total_decisions": total_env_steps,
        "lightning_optimization_steps": int(lightning_module.optimization_steps),
        "lightning_optimizer_step_attempts": int(
            lightning_module.optimizer_step_attempts
        ),
        "lightning_amp_skipped_steps": int(lightning_module.amp_skipped_steps),
        "lightning_fit_calls": lightning_fit_calls,
        "lightning_grad_scaler_enabled": lightning_grad_scaler_enabled,
        "time_head_training_status": time_head_training_status,
        "time_label_count": time_label_count,
        "time_supervision_steps": time_supervision_steps,
        "time_supervision_optimizer_updates": time_supervision_optimizer_updates,
        "independent_episode_count": len({
            (item["worker_id"], item["episode_id"])
            for item in scenario_log
        }),
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
        "method_variant": profile.name,
        "method_profile": config["method_profile"],
        "initial_actor_fingerprint": initial_actor_fingerprint,
        "initial_parameter_fingerprint": initial_parameter_fingerprint,
        "source_sha": source_sha,
        "source_tree_dirty": source_tree_dirty,
        "training_config": config,
        "resolved_runtime_config_yaml": resolved_config_yaml,
        "resolved_runtime_config_sha256": resolved_config_sha256,
        "data_fingerprint": data_fingerprint,
        "event_plan_fingerprint": data_fingerprint["event_plan_sha256"],
        "termination_reason": stop_reason,
        "successful_batch_count": int(successful_batch_count),
        "lightning_optimization_steps": int(lightning_module.optimization_steps),
        "checkpoint_evaluation_eligible": checkpoint_evaluation_eligible,
        "time_head_training_status": time_head_training_status,
        "time_label_count": time_label_count,
        "time_supervision_steps": time_supervision_steps,
        "time_supervision_optimizer_updates": time_supervision_optimizer_updates,
        "lightning_precision_state": dict(lightning_precision_state),
        "potential_predictor_snapshot": (
            {
                "version": int(shaper.snapshot_version),
                "graph_feature_version": GRAPH_FEATURE_VERSION,
                "actor_state": _cpu_state_dict(shaper.frozen_actor),
                "time_head_state": _cpu_state_dict(shaper.frozen_head),
                "time_head_model_version": "signed_residual_v1",
                "time_head_in_dim": int(shaper.frozen_head.in_dim),
            }
            if shaper is not None and shaper.frozen_actor is not None
            else None
        ),
        "independent_episode_count": report["independent_episode_count"],
        "resume_capability": "non_exact",
    }
    numpy_rng_state = np.random.get_state()
    training_state = {
        "optimizer_state": lightning_module.optimizer_state_dict,
        "precision_state": dict(lightning_precision_state),
        "rng_state": {
            "python": random.getstate(),
            "numpy": (
                str(numpy_rng_state[0]),
                numpy_rng_state[1].tolist(),
                int(numpy_rng_state[2]),
                int(numpy_rng_state[3]),
                float(numpy_rng_state[4]),
            ),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch_device.type == "cuda"
                else None
            ),
        },
        "optimizer_step_counts": {
            "successful": int(lightning_module.optimization_steps),
            "attempted": int(lightning_module.optimizer_step_attempts),
            "amp_skipped": int(lightning_module.amp_skipped_steps),
            "time_supervision_successful": time_supervision_optimizer_updates,
        },
        "trainer_global_step": int(lightning_module.optimization_steps),
        "environment_steps": int(total_env_steps),
        "event_plan_position": {
            str(worker_id): sum(
                int(item.get("worker_id", -1)) == worker_id
                for item in scenario_log
            )
            for worker_id in range(num_envs)
        },
    }
    trainer.save_checkpoint(
        str(output_ckpt),
        metadata=checkpoint_metadata,
        training_state=training_state,
    )
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
        "worker_step_counts": worker_step_counts,
        "checkpoint_path": output_ckpt,
        "report_path": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="工作三烟测/训练试点入口（不生成正式实验结论）")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT_DIR / "conf" / "work3" / "train_pilot.yaml",
        help="工作三OmegaConf YAML配置",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf点式覆盖，可重复指定",
    )
    parser.add_argument("--mode", choices=("smoke", "pilot"), default=None)
    parser.add_argument("--seed", type=int, default=None, help="训练初始化及episode计划种子")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="可选rollout段数上限；smoke固定为1，pilot默认由成功目标/预算终止",
    )
    parser.add_argument("--steps", type=int, default=None, help="每个rollout段的最大决策步数")
    parser.add_argument("--num-envs", type=int, default=None, help="spawn CPU环境worker数")
    parser.add_argument("--epochs", type=int, default=None, help="PPO重放轮数")
    parser.add_argument("--batch-size", type=int, default=None, help="PPO mini-batch大小")
    parser.add_argument("--successful-batch-target", type=int, default=None)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--max-wall-seconds", type=float, default=None)
    parser.add_argument("--time-auxiliary-epochs", type=int, default=None)
    parser.add_argument("--time-auxiliary-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--method", choices=("C", "D"), default=None, help="C/D方法配置")
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--scenarios", type=Path, default=None)
    parser.add_argument("--scenario-split", type=Path, default=None)
    parser.add_argument("--time-head-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None, help="设备 (cpu/cuda)")
    parser.add_argument("--main-num-threads", type=int, default=None)
    parser.add_argument("--env-num-threads", type=int, default=None)
    parser.add_argument("--settle-timeout-seconds", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None, help="模型权重检查点路径")
    parser.add_argument("--report", type=Path, default=None, help="运行报告路径；默认与检查点同名.run.json")
    parser.add_argument("--paired-report", type=Path, default=None, help="另一组C/D训练报告路径，用于核对事件、预算及实际命中")
    args = parser.parse_args()

    runtime_config = load_work3_runtime_config(
        args.config,
        overrides=args.config_overrides,
    )
    cli_overrides = {
        "runtime.run_mode": args.mode,
        "runtime.seed": args.seed,
        "runtime.num_envs": args.num_envs,
        "runtime.successful_batch_target": args.successful_batch_target,
        "runtime.total_env_steps": args.max_decisions,
        "runtime.max_wall_seconds": args.max_wall_seconds,
        "runtime.method_profile": args.method,
        "runtime.device": args.device,
        "runtime.main_num_threads": args.main_num_threads,
        "runtime.env_num_threads": args.env_num_threads,
        "runtime.settle_timeout_seconds": args.settle_timeout_seconds,
        "paths.baseline": None if args.baseline is None else str(args.baseline),
        "runtime.scenario_pool_path": None if args.scenarios is None else str(args.scenarios),
        "runtime.scenario_split_path": None if args.scenario_split is None else str(args.scenario_split),
        "paths.time_head_checkpoint": (
            None if args.time_head_checkpoint is None else str(args.time_head_checkpoint)
        ),
        "ppo.steps_per_iter": args.steps,
        "ppo.epochs": args.epochs,
        "ppo.batch_size": args.batch_size,
        "ppo.learning_rate": args.lr,
        "ppo.time_auxiliary_epochs": args.time_auxiliary_epochs,
        "ppo.time_auxiliary_batch_size": args.time_auxiliary_batch_size,
    }
    runtime_config = apply_work3_runtime_overrides(
        runtime_config,
        {key: value for key, value in cli_overrides.items() if value is not None},
    )
    resolved_yaml, config_sha256 = resolved_config_fingerprint(runtime_config)

    run_mode = str(runtime_config.runtime.run_mode)
    method_profile = str(runtime_config.runtime.method_profile)
    steps_per_iter = int(runtime_config.ppo.steps_per_iter)
    max_decisions = int(runtime_config.runtime.total_env_steps)
    if run_mode == "smoke":
        max_decisions = min(max_decisions, steps_per_iter)
    output_path = args.output or (
        Path(runtime_config.paths.checkpoint_dir)
        / f"{run_mode}_method_{method_profile.lower()}_weights.pt"
    )
    report_path = args.report or (
        Path(runtime_config.paths.report_dir)
        / f"{run_mode}_method_{method_profile.lower()}.run.json"
    )
    run_training(
        num_iterations=args.iterations,
        steps_per_iter=steps_per_iter,
        num_envs=int(runtime_config.runtime.num_envs),
        env_num_threads=int(runtime_config.runtime.env_num_threads),
        ppo_epochs=int(runtime_config.ppo.epochs),
        batch_size=int(runtime_config.ppo.batch_size),
        time_auxiliary_epochs=int(runtime_config.ppo.time_auxiliary_epochs),
        time_auxiliary_batch_size=int(runtime_config.ppo.time_auxiliary_batch_size),
        lr=float(runtime_config.ppo.learning_rate),
        clip_eps=float(runtime_config.ppo.clip_epsilon),
        vf_coef=float(runtime_config.ppo.value_coefficient),
        ent_coef=float(runtime_config.ppo.entropy_coefficient),
        gamma=float(runtime_config.ppo.gamma),
        gae_lambda=float(runtime_config.ppo.gae_lambda),
        beta_shaping=float(runtime_config.ppo.shaping_coefficient),
        time_loss_coef=float(runtime_config.ppo.time_loss_coefficient),
        seed=int(runtime_config.runtime.seed),
        deterministic=bool(runtime_config.runtime.deterministic),
        main_num_threads=int(runtime_config.runtime.main_num_threads),
        settle_timeout_seconds=float(runtime_config.runtime.settle_timeout_seconds),
        run_mode=run_mode,
        successful_batch_target=runtime_config.runtime.successful_batch_target,
        max_decisions=max_decisions,
        max_wall_seconds=float(runtime_config.runtime.max_wall_seconds),
        method_variant=method_profile,
        baseline_path=Path(runtime_config.paths.baseline),
        scenarios_path=Path(runtime_config.runtime.scenario_pool_path),
        scenario_split_path=Path(runtime_config.runtime.scenario_split_path),
        time_head_ckpt=Path(runtime_config.paths.time_head_checkpoint),
        device=str(runtime_config.runtime.device),
        amp_dtype=str(runtime_config.runtime.amp_dtype),
        output_ckpt=output_path,
        report_path=report_path,
        paired_report_path=args.paired_report,
        resolved_config_yaml=resolved_yaml,
        resolved_config_sha256=config_sha256,
    )


if __name__ == "__main__":
    main()
