"""工作三 条件分支 PPO 强化学习验证训练脚本 (Task 7.3)。

核心功能与执行目标：
1. 在固定训练事件清单和 10 架次基准产线上执行正式 C/D PPO 训练验证；
2. 验证计算图反向传播、显式时间紧迫度融合、条件分支动作截断、GAE 优势计算与势函数奖励塑形全链路闭环；
3. 记录多轮 Iteration 损失曲线、KL 散度、梯度范数、策略熵以及累计回报，验证数值稳定性与初步收敛性；
4. 保存训练完成的模型检查点至 models/work3/checkpoints/method_d_model.pt。
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from pathlib import Path
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
from models.work3.ppo_trainer import PPOTrainerWork3
from models.work3.time_head import TimeResidualHead
from scripts.work3.collect_validation_trajectories import load_scenarios_for_split
from scripts.work3.experiment_protocol import Work3MethodProfile, build_method_profile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


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


def count_actual_scenario_hits(env: AirLineEnvWork3, scenario: dict[str, Any]) -> int:
    """按实际物料恢复时间统计已揭示的目标工序数量。"""
    return sum(
        1
        for task_key in scenario.get("affected_task_keys", [])
        if task_key in env.state.tasks
        and float(env.state.tasks[task_key].material_ready_time) > env.tolerance
    )


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
    num_iterations: int = 15,
    steps_per_iter: int = 256,
    ppo_epochs: int = 4,
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
    method_variant: str = "D",
    baseline_path: str = "data/work3/real_283_k10_baseline.json",
    scenarios_path: str = "data/work3/scenarios_9class.json",
    scenario_split_path: str = "data/work3/experiment_splits/train.json",
    time_head_ckpt: str = "models/work3/checkpoints/time_head_best.pt",
    output_ckpt: str = "models/work3/checkpoints/method_d_model.pt",
    device: str = "cpu",
) -> dict[str, Any]:
    """执行正式C/D分组下的扰动PPO训练流程。"""
    profile = build_method_profile(method_variant)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch_device = torch.device(device)

    training_scenarios = load_training_scenarios(scenarios_path, scenario_split_path)
    episode_plan = build_episode_scenario_plan(
        training_scenarios,
        max(1, num_iterations * 2),
        seed=seed,
    )
    if profile.name == "C" and output_ckpt == "models/work3/checkpoints/method_d_model.pt":
        output_ckpt = "models/work3/checkpoints/method_c_model.pt"

    Path(output_ckpt).parent.mkdir(parents=True, exist_ok=True)

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
    if profile.use_time_auxiliary and time_head is not None and Path(time_head_ckpt).is_file():
        ckpt_data = torch.load(time_head_ckpt, map_location="cpu")
        if (
            ckpt_data.get("model_version") == "signed_residual_v1"
            and int(ckpt_data.get("in_dim", -1)) == actor_critic.hidden_dim
        ):
            state_dict = ckpt_data.get("model_state_dict", ckpt_data)
            time_head.load_state_dict(state_dict)
            logger.info(f"已成功载入有符号离线时间修正头: {time_head_ckpt}")
        else:
            logger.warning(f"检查点 {time_head_ckpt} 不是当前共享图有符号版本，本次不加载")
    elif profile.use_time_auxiliary:
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

    history: list[dict[str, Any]] = []
    total_env_steps = 0
    start_time = time.time()
    episode_id = 0
    pending_time_labels = PendingTimeLabelCache()
    episode_plan_index = 0
    scenario_log: list[dict[str, Any]] = []
    current_scenario: dict[str, Any] | None = None
    current_scenario_log: dict[str, Any] | None = None
    episode_done = False
    episode_termination_reason: str | None = None

    def start_episode() -> None:
        nonlocal episode_plan_index, current_scenario, current_scenario_log
        env.reset()
        current_scenario = dict(episode_plan[episode_plan_index % len(episode_plan)])
        episode_plan_index += 1
        env.load_scenario(current_scenario)
        current_scenario_log = {
            "episode_id": episode_id,
            "scenario_id": current_scenario["scenario_id"],
            "timing": current_scenario["timing"],
            "intensity": current_scenario["intensity"],
            "station_id": current_scenario["station_id"],
            "aircraft_id": current_scenario["aircraft_id"],
            "actual_hit_count": None,
            "completed": False,
            "success": None,
            "termination_reason": None,
        }
        scenario_log.append(current_scenario_log)
        logger.info(
            "加载训练扰动 episode=%s scenario=%s timing=%s intensity=%s station=%s",
            episode_id,
            current_scenario["scenario_id"],
            current_scenario["timing"],
            current_scenario["intensity"],
            current_scenario["station_id"],
        )

    start_episode()

    logger.info("=" * 80)
    logger.info(f"开始条件分支 PPO 训练验证 (共 {num_iterations} 轮, 每轮 {steps_per_iter} 步, 总计 ~{num_iterations * steps_per_iter} 步)")
    logger.info("=" * 80)

    for iter_idx in range(1, num_iterations + 1):
        iter_start = time.time()
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
                episode_id += 1
                episode_done = False
                episode_termination_reason = None
                start_episode()

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
                    current_scenario_log["termination_reason"] = "truncated"

            for offset, actual_transfer_time in enumerate(
                env.state.transfer_history[transfer_count_before:]
            ):
                pending_time_labels.attach_transfer(
                    episode_id=episode_id,
                    cycle_id=cycle_id + offset,
                    actual_transfer_time=float(actual_transfer_time),
                )
            if discard_unlabelled_time_samples:
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

        if current_scenario_log is not None:
            current_scenario_log["actual_hit_count"] = count_actual_scenario_hits(
                env,
                current_scenario or {},
            )

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

        iter_elapsed = time.time() - iter_start
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
            f"Iter [{iter_idx:02d}/{num_iterations:02d}] "
            f"Steps={total_env_steps:4d} | "
            f"Loss(Tot={iter_log['total_loss']:+.4f}, Pol={iter_log['policy_loss']:+.4f}, "
            f"Val={iter_log['value_loss']:.4f}) | "
            f"Ent={iter_log['entropy']:.3f} | "
            f"KL={iter_log['approx_kl']:.5f} | "
            f"Clip={iter_log['clip_fraction']:.3f} | "
            f"R_raw={mean_raw_r:.3f} | R_shaped={mean_shaped_r:.3f} | "
            f"Time={iter_elapsed:.1f}s"
        )

        # 阶段性与最终保存检查点
        if iter_idx % 5 == 0 or iter_idx == num_iterations:
            trainer.save_checkpoint(output_ckpt)
            logger.info(f"已保存模型检查点 -> {output_ckpt}")

    total_elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"PPO 验证训练完成！总耗时: {total_elapsed:.1f}s, 最终模型: {output_ckpt}")
    logger.info("=" * 80)

    return {
        "history": history,
        "total_steps": total_env_steps,
        "total_elapsed_seconds": total_elapsed,
        "checkpoint_path": output_ckpt,
        "method_variant": profile.name,
        "scenario_log": scenario_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="工作三 条件分支 PPO 验证训练")
    parser.add_argument("--iterations", type=int, default=15, help="训练轮数 (默认 15)")
    parser.add_argument("--steps", type=int, default=256, help="每轮采集步数 (默认 256)")
    parser.add_argument("--epochs", type=int, default=4, help="PPO 重放轮数 (默认 4)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch 大小 (默认 64)")
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
    parser.add_argument("--method", choices=("C", "D"), default="D", help="正式方法分组")
    parser.add_argument("--scenarios", type=str, default="data/work3/scenarios_9class.json")
    parser.add_argument("--scenario-split", type=str, default="data/work3/experiment_splits/train.json")
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu/cuda)")
    parser.add_argument("--output", type=str, default="models/work3/checkpoints/method_d_model.pt", help="检查点输出路径")
    args = parser.parse_args()

    run_training(
        num_iterations=args.iterations,
        steps_per_iter=args.steps,
        ppo_epochs=args.epochs,
        batch_size=args.batch_size,
        time_auxiliary_epochs=args.time_auxiliary_epochs,
        time_auxiliary_batch_size=args.time_auxiliary_batch_size,
        lr=args.lr,
        method_variant=args.method,
        scenarios_path=args.scenarios,
        scenario_split_path=args.scenario_split,
        device=args.device,
        output_ckpt=args.output,
    )


if __name__ == "__main__":
    main()
