"""工作三 基线 C 与方法 D 对比评测脚本 (Task 7.4 / 里程碑 M5 验收)。

核心评测目标与技术规范：
1. 在固定独立测试事件清单下：
   - 运行正式方法 C 图策略 (ActorCriticWork3, 启发式时间输入)；
   - 运行正式方法 D 图策略 (ActorCriticWork3, deterministic=True, 学习修正时间输入)；
2. 收集生产全流程多目标核心指标：
   - J_takt: 脉动节拍超期惩罚；
   - D_time: 基准工序排程时间偏差；
   - D_team: 基准团队人员改派偏差；
   - J_postpone: 跨站后移改派惩罚；
   - J_total: 全线综合惩罚费用 (J_total = w_h J_takt + w_tau D_time + w_m D_team + w_p J_postpone)；
   - Makespan: 全线 10 架次飞机总完工时长；
3. 以同一 C 检查点报告无扰动基线；仅对双方成功且独立可行的轨迹计算真实费用改善率。
4. 对具有真实转站标签的周期报告原始启发式及修正预测 MAE（小时与 H0 归一化）。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features
from models.work3.heuristic_agent import HeuristicAgentWork3
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from models.work3.time_head import TimeResidualHead
from models.work3.ppo_trainer import PPO_CHECKPOINT_VERSION
from scripts.work3.collect_validation_trajectories import load_scenarios_for_split
from scripts.work3.experiment_protocol import Work3MethodProfile, build_method_profile
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline
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


@dataclass
class FormalEvaluationAgent:
    """正式C/D评测包装器；启发式智能体不属于该类型。"""

    profile: Work3MethodProfile
    actor_critic: ActorCriticWork3
    time_head: TimeResidualHead | None
    debug_random: bool = False
    last_time_prediction: tuple[int, float, float, float] | None = field(
        default=None, init=False, repr=False
    )

    def select_action(self, env: AirLineEnvWork3) -> dict[str, Any] | None:
        """使用启发式或学习修正时间输入调用同一图策略。"""
        device = next(self.actor_critic.parameters()).device
        cmax = compute_cycle_heuristic_cmax(env.state)
        corrected_cmax = cmax
        state_feat = extract_compact_state_features(env.state, cmax)
        if self.profile.use_corrected_time_input:
            if self.time_head is None:
                raise RuntimeError("正式方法 D 缺少共享时间头")
            graph_snapshot = self.actor_critic.build_graph_snapshot(env)
            with torch.no_grad():
                shared_feature = self.actor_critic.encode_shared_representation(
                    state_feat.to(device),
                    graph_snapshot,
                ).unsqueeze(0)
                p_corrected, remaining, _ = self.time_head.predict_corrected_time(
                    state_feat=shared_feature,
                    estimated_cmax=cmax,
                    current_time=float(env.state.current_time),
                    h0=float(env.state.h0),
                    last_transfer_time=float(env.state.last_transfer_time),
                )
                corrected_cmax = float(p_corrected.detach().cpu().item())
                time_urgency = compute_time_urgency_vector(
                    estimated_r=remaining.squeeze(0),
                    current_time=env.state.current_time,
                    last_transfer_time=env.state.last_transfer_time,
                    h0=env.state.h0,
                    device=device,
                )
        else:
            time_urgency = compute_time_urgency_vector(
                estimated_r=max(0.0, cmax - float(env.state.current_time)),
                current_time=env.state.current_time,
                last_transfer_time=env.state.last_transfer_time,
                h0=env.state.h0,
                device=device,
            )
        self.last_time_prediction = (
            int(env.state.current_cycle),
            float(cmax),
            float(corrected_cmax),
            float(env.state.h0),
        )
        action, _, _, _ = self.actor_critic.select_action(
            env=env,
            state_feat=state_feat,
            time_urgency=time_urgency,
            deterministic=True,
        )
        return action


def _validate_state_dict_shapes(
    expected: dict[str, torch.Tensor],
    observed: Any,
    component: str,
) -> None:
    if not isinstance(observed, dict) or set(observed) != set(expected):
        raise ValueError(f"正式检查点{component}结构不匹配")
    for name, tensor in expected.items():
        value = observed[name]
        if not isinstance(value, torch.Tensor) or value.shape != tensor.shape:
            raise ValueError(f"正式检查点{component}参数维度不匹配: {name}")


def _validate_formal_checkpoint_contract(
    profile: Work3MethodProfile,
    checkpoint: dict[str, Any],
    actor: ActorCriticWork3,
) -> None:
    """拒绝缺少训练来源、配置指纹或与C/D方法不一致的推理权重。"""
    from models.work3.graph_builder import GRAPH_FEATURE_DIMS, GRAPH_FEATURE_VERSION

    metadata = checkpoint.get("run_metadata")
    expected_profile = asdict(profile)
    config = metadata.get("training_config") if isinstance(metadata, dict) else None
    if (
        not isinstance(metadata, dict)
        or metadata.get("method_variant") != profile.name
        or metadata.get("method_profile") != expected_profile
        or not isinstance(config, dict)
        or config.get("method_variant") != profile.name
        or config.get("method_profile") != expected_profile
    ):
        raise ValueError(f"正式检查点方法profile不匹配: 期望{profile.name}")
    if (
        checkpoint.get("checkpoint_version") != PPO_CHECKPOINT_VERSION
        or checkpoint.get("checkpoint_role") != "model_weights"
        or checkpoint.get("resume_capability") != "non_exact"
    ):
        raise ValueError("正式检查点格式或续训能力标记不匹配")

    config_yaml = metadata.get("resolved_runtime_config_yaml")
    config_hash = metadata.get("resolved_runtime_config_sha256")
    if (
        not isinstance(config_yaml, str)
        or not config_yaml
        or not isinstance(config_hash, str)
        or hashlib.sha256(config_yaml.encode("utf-8")).hexdigest() != config_hash
        or config.get("resolved_config_sha256") != config_hash
    ):
        raise ValueError("正式检查点解析配置与SHA256不匹配")

    data_fingerprint = metadata.get("data_fingerprint")
    fingerprint_keys = (
        "scenario_pool_sha256",
        "scenario_split_sha256",
        "baseline_sha256",
        "event_plan_sha256",
        "worker_event_plan_sha256",
    )
    source_sha = metadata.get("source_sha")
    initial_fingerprint = metadata.get("initial_actor_fingerprint")
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 40
        or any(character not in "0123456789abcdef" for character in source_sha)
        or not isinstance(initial_fingerprint, str)
        or len(initial_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in initial_fingerprint)
        or not isinstance(data_fingerprint, dict)
        or any(
            not isinstance(data_fingerprint.get(key), str)
            or len(data_fingerprint[key]) != 64
            or any(c not in "0123456789abcdef" for c in data_fingerprint[key])
            for key in fingerprint_keys
        )
    ):
        raise ValueError("正式检查点缺少源码、初始化或数据指纹")

    training_state = checkpoint.get("lightning_training_state")
    expected_rng_fields = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if (
        not isinstance(training_state, dict)
        or not isinstance(training_state.get("optimizer_state"), dict)
        or not isinstance(training_state.get("precision_state"), dict)
        or not isinstance(training_state.get("rng_state"), dict)
        or not expected_rng_fields.issubset(training_state["rng_state"])
        or not isinstance(training_state.get("event_plan_position"), dict)
        or checkpoint.get("resume_capability") != "non_exact"
    ):
        raise ValueError("正式检查点缺少Lightning训练/精度状态或随机状态")
    precision_state = training_state["precision_state"]
    expected_precision = {
        "fp32": "32-true",
        "fp16": "16-mixed",
        "bf16": "bf16-mixed",
    }.get(config.get("amp_dtype"))
    if (
        expected_precision is None
        or expected_precision != config.get("lightning_precision")
        or precision_state.get("precision") != expected_precision
    ):
        raise ValueError("检查点Lightning精度与训练配置不匹配")
    scaler_state = precision_state.get("grad_scaler_state")
    if (config.get("amp_dtype") == "fp16") != isinstance(scaler_state, dict):
        raise ValueError("检查点GradScaler状态与AMP精度配置不匹配")

    if (
        metadata.get("run_mode") != "pilot"
        or int(metadata.get("successful_batch_count", 0)) < 1
        or int(metadata.get("lightning_optimization_steps", 0)) < 1
        or metadata.get("checkpoint_evaluation_eligible") is not True
    ):
        raise ValueError("检查点尚未通过完整批次训练评测门槛")

    if not profile.use_time_auxiliary:
        if "time_head_state" in checkpoint or metadata.get("potential_predictor_snapshot") is not None:
            raise ValueError("C profile检查点不得包含D时间学习预测器")
        return

    labels = int(metadata.get("time_label_count", 0))
    successful_updates = int(metadata.get("time_supervision_optimizer_updates", 0))
    if (
        metadata.get("time_head_training_status") != "trained_online"
        or labels < 1
        or successful_updates < 1
    ):
        raise ValueError("正式方法D时间头必须由真实转站标签成功训练")
    if (
        checkpoint.get("time_head_model_version") != "signed_residual_v1"
        or int(checkpoint.get("time_head_in_dim", -1)) != actor.hidden_dim
        or "time_head_state" not in checkpoint
    ):
        raise ValueError("正式方法D检查点缺少兼容的有符号时间头")
    time_head = TimeResidualHead(in_dim=actor.hidden_dim, hidden_dim=64)
    _validate_state_dict_shapes(
        time_head.state_dict(),
        checkpoint["time_head_state"],
        "在线时间头",
    )

    snapshot = metadata.get("potential_predictor_snapshot")
    if (
        not isinstance(snapshot, dict)
        or int(snapshot.get("version", -1)) < 0
        or snapshot.get("graph_feature_version") != GRAPH_FEATURE_VERSION
        or snapshot.get("time_head_model_version") != "signed_residual_v1"
        or int(snapshot.get("time_head_in_dim", -1)) != actor.hidden_dim
    ):
        raise ValueError("正式方法D缺少兼容的势函数预测器快照")
    if dict(checkpoint.get("graph_feature_dims") or {}) != dict(GRAPH_FEATURE_DIMS):
        raise ValueError("势函数预测器图特征维度不匹配")
    _validate_state_dict_shapes(
        actor.state_dict(),
        snapshot.get("actor_state"),
        "势函数图编码器",
    )
    _validate_state_dict_shapes(
        time_head.state_dict(),
        snapshot.get("time_head_state"),
        "势函数时间头",
    )


def build_formal_evaluation_agent(
    method_variant: str,
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    debug_random: bool = False,
) -> FormalEvaluationAgent:
    """加载正式C/D模型；缺检查点仅在显式调试模式允许随机权重。"""
    profile = build_method_profile(method_variant)
    torch_device = torch.device(device)
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64).to(torch_device)
    time_head: TimeResidualHead | None = None
    path = Path(checkpoint_path)
    if not path.is_file():
        if not debug_random:
            raise FileNotFoundError(f"正式方法 {profile.name} 缺少检查点: {path}")
        if profile.use_time_auxiliary:
            time_head = TimeResidualHead(in_dim=actor.hidden_dim, hidden_dim=64).to(torch_device)
        actor.eval()
        if time_head is not None:
            time_head.eval()
        return FormalEvaluationAgent(profile, actor, time_head, debug_random=True)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    from models.work3.graph_builder import GRAPH_FEATURE_DIMS, GRAPH_FEATURE_VERSION

    if checkpoint.get("graph_feature_version") != GRAPH_FEATURE_VERSION or dict(
        checkpoint.get("graph_feature_dims") or {}
    ) != dict(GRAPH_FEATURE_DIMS):
        if debug_random:
            if profile.use_time_auxiliary:
                time_head = TimeResidualHead(in_dim=actor.hidden_dim, hidden_dim=64).to(torch_device)
            actor.eval()
            if time_head is not None:
                time_head.eval()
            return FormalEvaluationAgent(profile, actor, time_head, debug_random=True)
        raise ValueError(
            f"正式方法 {profile.name} 检查点图特征版本或维度不匹配："
            f"期望 {GRAPH_FEATURE_VERSION} {GRAPH_FEATURE_DIMS}，"
            f"实际 {checkpoint.get('graph_feature_version')} {checkpoint.get('graph_feature_dims')}"
        )
    actor_state = checkpoint.get("actor_critic_state")
    if actor_state is None:
        if debug_random:
            time_head = (
                TimeResidualHead(in_dim=actor.hidden_dim, hidden_dim=64).to(torch_device)
                if profile.use_time_auxiliary
                else None
            )
            actor.eval()
            if time_head is not None:
                time_head.eval()
            return FormalEvaluationAgent(profile, actor, time_head, debug_random=True)
        raise ValueError(f"方法 {profile.name} 检查点缺少 actor_critic_state: {path}")
    try:
        _validate_formal_checkpoint_contract(profile, checkpoint, actor)
    except ValueError:
        if not debug_random:
            raise
        time_head = (
            TimeResidualHead(in_dim=actor.hidden_dim, hidden_dim=64).to(torch_device)
            if profile.use_time_auxiliary
            else None
        )
        actor.eval()
        if time_head is not None:
            time_head.eval()
        return FormalEvaluationAgent(profile, actor, time_head, debug_random=True)
    actor.load_state_dict(actor_state)

    if profile.use_time_auxiliary:
        if checkpoint.get("time_head_model_version") != "signed_residual_v1":
            if debug_random:
                time_head = TimeResidualHead(in_dim=actor.hidden_dim, hidden_dim=64).to(torch_device)
                actor.eval()
                time_head.eval()
                return FormalEvaluationAgent(profile, actor, time_head, debug_random=True)
            raise ValueError("正式方法 D 检查点缺少有符号时间头版本")
        if "time_head_state" not in checkpoint:
            if debug_random:
                time_head = TimeResidualHead(in_dim=actor.hidden_dim, hidden_dim=64).to(torch_device)
                actor.eval()
                time_head.eval()
                return FormalEvaluationAgent(profile, actor, time_head, debug_random=True)
            raise ValueError("正式方法 D 检查点缺少共享时间头")
        time_head = TimeResidualHead(in_dim=actor.hidden_dim, hidden_dim=64).to(torch_device)
        time_head.load_state_dict(checkpoint["time_head_state"])

    actor.eval()
    if time_head is not None:
        time_head.eval()
    return FormalEvaluationAgent(profile, actor, time_head, debug_random=False)


def summarize_cycle_prediction_errors(
    prediction_records: Sequence[tuple[int, float, float, float]],
    transfer_history: Sequence[float],
) -> dict[str, int | float | None]:
    """仅用有实际转站标签的周期计算启发式与修正预测 MAE。"""
    actual_by_cycle = dict(enumerate(transfer_history, start=1))
    labeled_errors: list[tuple[int, float, float, float]] = []
    for cycle_idx, heuristic_cmax, corrected_cmax, h0 in prediction_records:
        if cycle_idx not in actual_by_cycle:
            continue
        actual_time = float(actual_by_cycle[cycle_idx])
        scale = float(h0)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"周期{cycle_idx}的H0必须是有限正数")
        labeled_errors.append(
            (
                cycle_idx,
                abs(actual_time - float(heuristic_cmax)),
                abs(actual_time - float(corrected_cmax)),
                scale,
            )
        )

    sample_count = len(labeled_errors)
    if sample_count == 0:
        heuristic_mae_hours = None
        heuristic_mae_h0 = None
        corrected_mae_hours = None
        corrected_mae_h0 = None
    else:
        heuristic_mae_hours = sum(item[1] for item in labeled_errors) / sample_count
        heuristic_mae_h0 = (
            sum(item[1] / item[3] for item in labeled_errors) / sample_count
        )
        corrected_mae_hours = sum(item[2] for item in labeled_errors) / sample_count
        corrected_mae_h0 = (
            sum(item[2] / item[3] for item in labeled_errors) / sample_count
        )

    return {
        "labeled_cycle_count": len({item[0] for item in labeled_errors}),
        "sample_count": sample_count,
        "heuristic_mae_hours": heuristic_mae_hours,
        "heuristic_mae_h0": heuristic_mae_h0,
        "corrected_mae_hours": corrected_mae_hours,
        "corrected_mae_h0": corrected_mae_h0,
    }


def _check_completed_trajectory_feasibility(
    env: AirLineEnvWork3,
) -> tuple[bool, dict[str, int]]:
    """从实际执行字段重建记录，再调用独立检查器复核可行性。"""
    if not env._check_terminated():
        return False, {"incomplete_trajectory": 1}

    records: list[TrajectoryExecutionRecord] = []
    constraints: dict[int, TaskConstraintRecord] = {}
    for task in env.state.tasks.values():
        constraints[task.task_id] = TaskConstraintRecord(
            demand=task.demand,
            required_skill=task.skill,
            predecessors=tuple(task.predecessors),
            fixed_station=task.fixed_station,
            max_allowed_station=task.max_allowed_station,
        )
        if task.actual_start is None or task.actual_end is None:
            return False, {"missing_execution_interval": 1}
        aircraft = env.state.aircraft[task.aircraft_id]
        records.append(
            TrajectoryExecutionRecord(
                aircraft_id=task.aircraft_id,
                task_id=task.task_id,
                station_id=task.current_station,
                team=tuple(task.assigned_team),
                start=float(task.actual_start),
                end=float(task.actual_end),
                material_ready_time=float(task.material_ready_time),
                station_entry_time=aircraft.entry_times.get(task.current_station),
                aircraft_station_at_start=task.current_station,
            )
        )

    report = validate_trajectory(
        records,
        task_constraints=constraints,
        worker_skills=env.worker_skills,
        worker_station_bindings={
            worker_id: station_id
            for station_id, worker_ids in env.state.station_worker_bindings.items()
            for worker_id in worker_ids
        },
        station_capacities={
            station_id: env.max_slots_per_station
            for station_id in range(env.state.num_stations)
        },
    )
    return report.is_feasible, dict(report.violations)


def _count_actual_disturbance_hits(
    env: AirLineEnvWork3,
    scenario: dict[str, Any] | None,
) -> int:
    """按环境实际继承的物料恢复时刻统计已命中的受扰工序。"""
    if scenario is None:
        return 0
    return sum(
        1
        for task_key in scenario.get("affected_task_keys", [])
        if task_key in env.state.tasks
        and env.state.tasks[task_key].material_ready_time > env.tolerance
    )


def summarize_disturbance_effects(
    task_records: dict[str, Any],
    scenario: dict[str, Any] | None,
    *,
    baseline_start_by_key: dict[str, float],
    baseline_material_ready_by_key: dict[str, float] | None = None,
    event_triggered: bool = True,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """区分预设目标、实际命中、观察等待和其他飞机传播。"""
    if scenario is None:
        return {
            "target_count": 0,
            "actual_hit_count": 0,
            "actual_hit_rate": 0.0,
            "actual_hit_task_keys": [],
            "unhit_reasons": {},
            "observed_added_wait_hours": 0.0,
            "cross_aircraft_affected_task_count": 0,
            "cross_aircraft_affected_aircraft_ids": [],
            "event_triggered": False,
        }

    baseline_ready = baseline_material_ready_by_key or {}
    target_keys = [str(key) for key in scenario.get("affected_task_keys", [])]
    target_aircraft_ids = {int(scenario["aircraft_id"])}
    actual_hit_keys = [
        key
        for key in target_keys
        if key in task_records
        and float(getattr(task_records[key], "material_ready_time", 0.0))
        > float(baseline_ready.get(key, 0.0)) + tolerance
    ]
    actual_hit_key_set = set(actual_hit_keys)
    tau_value = scenario.get("tau")
    tau = None if tau_value is None else float(tau_value)
    unhit_reasons: dict[str, str] = {}
    for key in target_keys:
        if key in actual_hit_key_set:
            continue
        if not event_triggered:
            unhit_reasons[key] = "event_not_triggered"
        elif key not in task_records:
            unhit_reasons[key] = "target_not_in_instance"
        else:
            actual_start = getattr(task_records[key], "actual_start", None)
            if (
                tau is not None
                and actual_start is not None
                and float(actual_start) < tau - tolerance
            ):
                unhit_reasons[key] = "already_started_or_completed_at_event"
            else:
                unhit_reasons[key] = "no_material_delay_recorded"

    added_wait = 0.0
    for key in actual_hit_keys:
        task = task_records[key]
        actual_start = getattr(task, "actual_start", None)
        baseline_start = baseline_start_by_key.get(key)
        if actual_start is not None and baseline_start is not None:
            added_wait += max(0.0, float(actual_start) - float(baseline_start))

    cross_aircraft_keys: list[str] = []
    for key, task in task_records.items():
        if key in target_keys or int(task.aircraft_id) in target_aircraft_ids:
            continue
        actual_start = getattr(task, "actual_start", None)
        baseline_start = baseline_start_by_key.get(key)
        if (
            actual_start is not None
            and baseline_start is not None
            and float(actual_start) > float(baseline_start) + tolerance
        ):
            cross_aircraft_keys.append(str(key))

    cross_aircraft_ids = sorted({int(task_records[key].aircraft_id) for key in cross_aircraft_keys})
    target_count = len(target_keys)
    return {
        "target_count": target_count,
        "actual_hit_count": len(actual_hit_keys),
        "actual_hit_rate": len(actual_hit_keys) / target_count if target_count else 0.0,
        "actual_hit_task_keys": actual_hit_keys,
        "unhit_reasons": unhit_reasons,
        "observed_added_wait_hours": added_wait,
        "cross_aircraft_affected_task_count": len(cross_aircraft_keys),
        "cross_aircraft_affected_aircraft_ids": cross_aircraft_ids,
        "event_triggered": bool(event_triggered),
    }


def evaluate_single_trajectory(
    env: AirLineEnvWork3,
    agent_type: str,
    agent: Any,
    scenario: dict[str, Any] | None = None,
    max_decisions: int = 10000,
    device: str = "cpu",
) -> dict[str, Any]:
    """运行单条生产轨迹并结算综合目标。"""
    torch_device = torch.device(device)
    if isinstance(agent, FormalEvaluationAgent):
        agent.last_time_prediction = None
    env.reset()
    if scenario is not None:
        env.load_scenario(scenario)

    decisions = 0
    termination_reason = "decision_limit"
    terminated = False
    truncated = False
    prediction_records: list[tuple[int, float, float, float]] = []
    with torch.inference_mode():
        while decisions < max_decisions:
            candidates = env.get_action_candidates()
            if not candidates:
                if env._check_terminated():
                    termination_reason = "completed"
                    break
                _obs, _reward, terminated, truncated, info = env.step(
                    {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
                )
                if terminated:
                    termination_reason = str(info.get("termination_reason", "deadlock"))
                    break
                if truncated:
                    termination_reason = "rollout_truncated"
                    break
                continue

            if agent_type in {"Baseline-C", "Heuristic-Debug"}:
                action = agent.select_action(env)
            elif agent_type in {"Method-C", "Method-D"}:
                if isinstance(agent, FormalEvaluationAgent):
                    action = agent.select_action(env)
                else:
                    raise TypeError(f"正式{agent_type}必须使用FormalEvaluationAgent")
            else:
                raise ValueError(f"未知智能体类型: {agent_type}")

            if isinstance(agent, FormalEvaluationAgent):
                prediction = agent.last_time_prediction
                if prediction is not None:
                    prediction_records.append(prediction)

            if action is None:
                _obs, _reward, terminated, truncated, info = env.step(
                    {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
                )
                if terminated:
                    termination_reason = str(info.get("termination_reason", "deadlock"))
                    break
                if truncated:
                    termination_reason = "rollout_truncated"
                    break
                continue

            obs, reward, terminated, truncated, info = env.step(action)
            decisions += 1
            if terminated:
                termination_reason = str(
                    info.get(
                        "termination_reason",
                        "completed" if env._check_terminated() else "deadlock",
                    )
                )
                break
            if truncated:
                termination_reason = "rollout_truncated"
                break

    if decisions >= max_decisions and not terminated and not truncated:
        termination_reason = "decision_limit"

    # 统计生产指标
    completed = sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.COMPLETED)
    postponed_total = sum(t.postpone_count for t in env.state.tasks.values())
    total_cost = env.cumulative_cost
    feasible, constraint_violations = _check_completed_trajectory_feasibility(env)
    success = bool(
        feasible
        and termination_reason == "completed"
        and env._check_terminated()
    )
    baseline = MultiAircraftBaseline.load_from_json(env.baseline_json_path)
    baseline_start_by_key = {
        task.task_key: float(task.baseline_start)
        for task in baseline.tasks.values()
    }
    effect_report = summarize_disturbance_effects(
        env.state.tasks,
        scenario,
        baseline_start_by_key=baseline_start_by_key,
        event_triggered=env.disturbance_event_triggered,
        tolerance=env.tolerance,
    )

    return {
        "agent": agent_type,
        "scenario_id": scenario["scenario_id"] if scenario else "NOMINAL",
        "j_takt": env.cost_takt,
        "d_time": env.cost_time,
        "d_team": env.cost_team,
        "j_postpone": env.cost_postpone,
        "j_revision": env.cost_revision,
        "j_total": total_cost,
        "raw_cost_components": {
            "j_takt": env.cost_takt,
            "d_time": env.cost_time,
            "d_team": env.cost_team,
            "j_postpone": env.cost_postpone,
            "j_revision": env.cost_revision,
        },
        "makespan": float(env.state.current_time),
        "completed_tasks": completed,
        "success": success,
        "feasible": feasible,
        "termination_reason": termination_reason,
        "constraint_violations": constraint_violations,
        "constraint_violation_count": sum(constraint_violations.values()),
        "actual_disturbance_hits": effect_report["actual_hit_count"],
        "disturbance_effects": effect_report,
        "time_prediction_metrics": summarize_cycle_prediction_errors(
            prediction_records,
            env.state.transfer_history,
        ),
        "postponed_count": postponed_total,
        "decisions": decisions,
        "transfers": len(env.state.transfer_history),
    }


def _sanitize_for_json(value: Any) -> Any:
    """递归清洗非有限浮点数，确保 JSON 严格序列化不含 NaN 或 Inf。"""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    return value


def compare_c_vs_d_pair(
    res_c: dict[str, Any],
    res_d: dict[str, Any],
) -> dict[str, Any]:
    """仅当 C 与 D 均完工 (success=True) 且独立可行 (feasible=True) 时才计算性能改善百分比。"""
    c_ok = bool(res_c.get("success")) and bool(res_c.get("feasible"))
    d_ok = bool(res_d.get("success")) and bool(res_d.get("feasible"))
    c_hits = int(res_c.get("actual_disturbance_hits", 0))
    d_hits = int(res_d.get("actual_disturbance_hits", 0))
    c_hit = c_hits > 0
    d_hit = d_hits > 0
    both_hit = c_hit and d_hit

    if not (c_ok and d_ok):
        c_reason = str(res_c.get("termination_reason", "unknown"))
        d_reason = str(res_d.get("termination_reason", "unknown"))
        return {
            "improvement_j_total_pct": None,
            "comparison_valid": False,
            "both_hit_disturbance": False,
            "comparison_status": (
                f"invalid_incomplete_or_infeasible: c_success={c_ok}({c_reason}), "
                f"d_success={d_ok}({d_reason})"
            ),
        }

    cost_c = float(res_c.get("j_total", 0.0))
    cost_d = float(res_d.get("j_total", 0.0))
    if not (math.isfinite(cost_c) and math.isfinite(cost_d)):
        return {
            "improvement_j_total_pct": None,
            "comparison_valid": False,
            "both_hit_disturbance": False,
            "comparison_status": "invalid_non_finite_cost",
        }

    improv_pct = ((cost_c - cost_d) / cost_c) * 100.0 if cost_c > 1e-6 else 0.0
    if both_hit:
        comparison_status = "valid_both_completed_and_hit"
    elif c_hit or d_hit:
        comparison_status = "valid_both_completed_one_sided_hit"
    else:
        comparison_status = "valid_both_completed_zero_hit"

    return {
        "improvement_j_total_pct": float(improv_pct),
        "comparison_valid": True,
        "both_hit_disturbance": both_hit,
        "comparison_status": comparison_status,
    }


def summarize_benchmark_credibility(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """无条件汇总 C/D 失败率，并将有效成对完工集合与双方实际命中扰动子集分列报告。"""
    total = len(results)
    c_success = 0
    d_success = 0
    c_reasons: dict[str, int] = {}
    d_reasons: dict[str, int] = {}
    valid_improvements: list[float] = []
    valid_and_hit_improvements: list[float] = []
    valid_one_sided_hit_improvements: list[float] = []
    valid_zero_hit_improvements: list[float] = []

    for item in results:
        res_c = item.get("method_c", {})
        res_d = item.get("method_d", {})
        c_ok = bool(res_c.get("success")) and bool(res_c.get("feasible"))
        d_ok = bool(res_d.get("success")) and bool(res_d.get("feasible"))
        if c_ok:
            c_success += 1
        if d_ok:
            d_success += 1
        c_r = str(res_c.get("termination_reason", "unknown"))
        d_r = str(res_d.get("termination_reason", "unknown"))
        c_reasons[c_r] = c_reasons.get(c_r, 0) + 1
        d_reasons[d_r] = d_reasons.get(d_r, 0) + 1

        pair_eval = compare_c_vs_d_pair(res_c, res_d)
        if pair_eval["comparison_valid"] and pair_eval["improvement_j_total_pct"] is not None:
            imp = float(pair_eval["improvement_j_total_pct"])
            valid_improvements.append(imp)
            if pair_eval["both_hit_disturbance"]:
                valid_and_hit_improvements.append(imp)
            elif int(res_c.get("actual_disturbance_hits", 0)) > 0 or int(
                res_d.get("actual_disturbance_hits", 0)
            ) > 0:
                valid_one_sided_hit_improvements.append(imp)
            else:
                valid_zero_hit_improvements.append(imp)

    c_fail = total - c_success
    d_fail = total - d_success
    summary = {
        "total_scenarios": total,
        "method_c_success_count": c_success,
        "method_c_failure_count": c_fail,
        "method_c_failure_rate": (c_fail / total) if total > 0 else 0.0,
        "method_c_termination_reasons": c_reasons,
        "method_d_success_count": d_success,
        "method_d_failure_count": d_fail,
        "method_d_failure_rate": (d_fail / total) if total > 0 else 0.0,
        "method_d_termination_reasons": d_reasons,
        "valid_comparison_count": len(valid_improvements),
        "invalid_comparison_count": total - len(valid_improvements),
        "mean_improvement_valid_only_pct": (
            sum(valid_improvements) / len(valid_improvements)
            if valid_improvements
            else None
        ),
        "valid_and_both_hit_count": len(valid_and_hit_improvements),
        "mean_improvement_valid_and_hit_only_pct": (
            sum(valid_and_hit_improvements) / len(valid_and_hit_improvements)
            if valid_and_hit_improvements
            else None
        ),
        "valid_one_sided_hit_count": len(valid_one_sided_hit_improvements),
        "mean_improvement_valid_one_sided_hit_pct": (
            sum(valid_one_sided_hit_improvements)
            / len(valid_one_sided_hit_improvements)
            if valid_one_sided_hit_improvements
            else None
        ),
        "valid_zero_hit_count": len(valid_zero_hit_improvements),
        "mean_improvement_valid_zero_hit_pct": (
            sum(valid_zero_hit_improvements) / len(valid_zero_hit_improvements)
            if valid_zero_hit_improvements
            else None
        ),
    }
    return _sanitize_for_json(summary)


def run_benchmark_evaluation(
    baseline_path: str = "data/work3/real_283_k10_baseline.json",
    scenarios_path: str = "data/work3/scenarios_9class.json",
    scenario_split_path: str | Path | None = None,
    method_c_ckpt: str = "models/work3/checkpoints/method_c_model.pt",
    method_d_ckpt: str = "models/work3/checkpoints/method_d_model.pt",
    selected_scenario_ids: list[str] | None = None,
    output_json: str = "data/work3/eval_c_vs_d_m5.json",
    device: str = "cpu",
    allow_debug_random: bool = False,
    max_decisions: int = 10000,
) -> dict[str, Any]:
    """在固定外生事件上比较同架构正式方法 C 与 D。"""
    torch_device = torch.device(device)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    # 1. 载入场景库
    if scenario_split_path is not None:
        all_scenarios = load_scenarios_for_split(scenarios_path, scenario_split_path)
    else:
        with open(scenarios_path, "r", encoding="utf-8") as f:
            all_scenarios = json.load(f)

    if selected_scenario_ids:
        scenarios = [s for s in all_scenarios if s["scenario_id"] in selected_scenario_ids]
        if len(scenarios) != len(selected_scenario_ids):
            raise ValueError("选定的正式评测事件不在固定场景清单中")
    else:
        if scenario_split_path is None:
            raise ValueError("正式评测必须显式提供独立场景清单")
        scenarios = list(all_scenarios)
    if not scenarios:
        raise ValueError("正式评测场景清单为空")

    # 2. 初始化智能体
    agent_c = build_formal_evaluation_agent(
        "C",
        method_c_ckpt,
        device=device,
        debug_random=allow_debug_random,
    )
    agent_d = build_formal_evaluation_agent(
        "D",
        method_d_ckpt,
        device=device,
        debug_random=allow_debug_random,
    )

    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    nominal_started = time.time()
    nominal_method_c = evaluate_single_trajectory(
        env,
        "Method-C",
        agent_c,
        scenario=None,
        max_decisions=max_decisions,
        device=device,
    )
    nominal_method_c["evaluation_time_s"] = time.time() - nominal_started

    results: list[dict[str, Any]] = []

    logger.info("=" * 105)
    logger.info(f"{'Scenario ID':<16} | {'Metric':<10} | {'Method C':<12} | {'Method D':<12} | {'Delta (%)':<10}")
    logger.info("=" * 105)

    for sc in scenarios:
        sc_id = sc["scenario_id"]

        # 评测基线 C
        t0 = time.time()
        res_c = evaluate_single_trajectory(
            env,
            "Method-C",
            agent_c,
            scenario=sc,
            max_decisions=max_decisions,
            device=device,
        )
        t_c = time.time() - t0

        # 评测方法 D
        t0 = time.time()
        res_d = evaluate_single_trajectory(
            env,
            "Method-D",
            agent_d,
            scenario=sc,
            max_decisions=max_decisions,
            device=device,
        )
        t_d = time.time() - t0

        pair_eval = compare_c_vs_d_pair(res_c, res_d)
        improv_pct = pair_eval["improvement_j_total_pct"]
        cost_c = float(res_c["j_total"])
        cost_d = float(res_d["j_total"])

        item = _sanitize_for_json(
            {
                "scenario_id": sc_id,
                "timing": sc.get("timing"),
                "intensity": sc.get("intensity"),
                "station": sc.get("station_id"),
                "method_c": res_c,
                "method_d": res_d,
                "improvement_j_total_pct": improv_pct,
                "comparison_valid": pair_eval["comparison_valid"],
                "both_hit_disturbance": pair_eval["both_hit_disturbance"],
                "comparison_status": pair_eval["comparison_status"],
                "time_c_s": t_c,
                "time_d_s": t_d,
            }
        )
        results.append(item)

        delta_str = f"{improv_pct:>+8.2f}%" if improv_pct is not None else "INVALID"
        logger.info(
            f"{sc_id:<16} | {'J_total':<10} | {cost_c:<12.4f} | {cost_d:<12.4f} | {delta_str:<10}"
        )
        logger.info(
            f"{'':<16} | {'J_takt':<10} | {res_c['j_takt']:<12.4f} | {res_d['j_takt']:<12.4f} |"
        )
        logger.info(
            f"{'':<16} | {'D_time':<10} | {res_c['d_time']:<12.4f} | {res_d['d_time']:<12.4f} |"
        )
        logger.info(
            f"{'':<16} | {'D_team':<10} | {res_c['d_team']:<12.4f} | {res_d['d_team']:<12.4f} |"
        )
        logger.info(
            f"{'':<16} | {'J_post':<10} | {res_c['j_postpone']:<12.4f} | {res_d['j_postpone']:<12.4f} |"
        )
        logger.info("-" * 105)

    credibility_summary = summarize_benchmark_credibility(results)
    report = {
        "schema_version": "work3_eval_c_vs_d_v2",
        "nominal_method_c": nominal_method_c,
        "scenario_results": results,
        "credibility_summary": credibility_summary,
    }

    # 保存评测结果（严格禁止 NaN/Inf）
    sanitized_report = _sanitize_for_json(report)
    with Path(output_json).open("w", encoding="utf-8") as f:
        json.dump(sanitized_report, f, indent=2, ensure_ascii=False, allow_nan=False)
    logger.info(f"评测结果已持久化保存至: {output_json}")

    return sanitized_report


def main() -> None:
    parser = argparse.ArgumentParser(description="工作三 正式方法 C vs D 对比评测")
    parser.add_argument("--baseline", type=str, default="data/work3/real_283_k10_baseline.json")
    parser.add_argument("--scenarios", type=str, default="data/work3/scenarios_9class.json")
    parser.add_argument("--scenario-split", type=str, default="data/work3/experiment_splits/test.json")
    parser.add_argument("--c-ckpt", type=str, default="models/work3/checkpoints/method_c_model.pt")
    parser.add_argument("--d-ckpt", type=str, default="models/work3/checkpoints/method_d_model.pt")
    parser.add_argument("--output", type=str, default="data/work3/eval_c_vs_d_m5.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--debug-random", action="store_true", help="仅烟测时允许缺检查点随机权重")
    args = parser.parse_args()

    run_benchmark_evaluation(
        baseline_path=args.baseline,
        scenarios_path=args.scenarios,
        scenario_split_path=args.scenario_split,
        method_c_ckpt=args.c_ckpt,
        method_d_ckpt=args.d_ckpt,
        output_json=args.output,
        device=args.device,
        allow_debug_random=args.debug_random,
    )


if __name__ == "__main__":
    main()
