"""工作三 条件分支 PPO 强化学习验证训练脚本 (Task 7.3)。

核心功能与执行目标：
1. 在最小数据集 (data/work3/real_283_k10_baseline.json, 10 架次基准产线) 上执行 PPO 训练验证；
2. 验证计算图反向传播、显式时间紧迫度融合、条件分支动作截断、GAE 优势计算与势函数奖励塑形全链路闭环；
3. 记录多轮 Iteration 损失曲线、KL 散度、梯度范数、策略熵以及累计回报，验证数值稳定性与初步收敛性；
4. 保存训练完成的模型检查点至 models/work3/checkpoints/method_d_model.pt。
"""

from __future__ import annotations

import argparse
import logging
import math
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
from models.work3.ppo_buffer import PPOTransition, RolloutBufferWork3
from models.work3.ppo_trainer import PPOTrainerWork3
from models.work3.time_head import TimeResidualHead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


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
    seed: int = 42,
    baseline_path: str = "data/work3/real_283_k10_baseline.json",
    time_head_ckpt: str = "models/work3/checkpoints/time_head_best.pt",
    output_ckpt: str = "models/work3/checkpoints/method_d_model.pt",
    device: str = "cpu",
) -> dict[str, Any]:
    """执行小规模 PPO 验证训练流程。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_device = torch.device(device)

    Path(output_ckpt).parent.mkdir(parents=True, exist_ok=True)

    # 1. 初始化仿真环境
    logger.info(f"初始化环境: {baseline_path}")
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    # 2. 初始化时间预测头与势函数奖励塑形器
    time_head = TimeResidualHead(in_dim=32, hidden_dim=64)
    if Path(time_head_ckpt).is_file():
        ckpt_data = torch.load(time_head_ckpt, map_location="cpu")
        state_dict = ckpt_data.get("model_state_dict", ckpt_data)
        time_head.load_state_dict(state_dict)
        logger.info(f"已成功载入离线预训练时间修正头: {time_head_ckpt}")
    else:
        logger.warning(f"未找到预训练时间修正头 {time_head_ckpt}，使用随机初始化头")

    shaper = PotentialRewardShaper(
        time_head=time_head,
        a=0.5,
        b=1.0,
        beta=beta_shaping,
        gamma=gamma,
    )

    # 3. 初始化条件分支自回归 Actor-Critic 网络与 PPO 训练器
    actor_critic = ActorCriticWork3(
        state_dim=32,
        task_feat_dim=8,
        hidden_dim=64,
        max_station_workers=16,
    ).to(torch_device)

    trainer = PPOTrainerWork3(
        actor_critic=actor_critic,
        lr=lr,
        clip_eps=clip_eps,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
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
            if env._check_terminated():
                env.reset()

            candidates = env.get_action_candidates()
            if not candidates:
                env._advance_events_until_next_decision()
                candidates = env.get_action_candidates()
                if not candidates and env._check_terminated():
                    env.reset()
                    candidates = env.get_action_candidates()

            cmax_est = compute_cycle_heuristic_cmax(env.state)
            s_feat = extract_compact_state_features(env.state, cmax_est)
            r_est = max(0.0, cmax_est - float(env.state.current_time))
            u_time = compute_time_urgency_vector(
                estimated_r=r_est,
                current_time=env.state.current_time,
                last_transfer_time=env.state.last_transfer_time,
                h0=env.state.h0,
                device=torch_device,
            )

            # 计算当前状态势 Φ(s_t)
            phi_current = shaper.compute_potential(
                state_feat=s_feat,
                estimated_cmax=cmax_est,
                current_time=float(env.state.current_time),
                h0=float(env.state.h0),
                last_transfer_time=float(env.state.last_transfer_time),
                is_terminal=False,
            )

            # 采样条件动作
            act, lp, v, rec = actor_critic.select_action(
                env=env,
                state_feat=s_feat,
                time_urgency=u_time,
                deterministic=False,
            )

            if act is None:
                # 极端异常推进
                env._advance_events_until_next_decision()
                continue

            # 环境执行一步调度动作
            obs, raw_reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            last_terminated = bool(terminated)

            # 计算下一状态势 Φ(s_{t+1}) 与塑形奖励
            if terminated:
                phi_next = 0.0
            else:
                next_cmax_est = compute_cycle_heuristic_cmax(env.state)
                next_s_feat = extract_compact_state_features(env.state, next_cmax_est)
                phi_next = shaper.compute_potential(
                    state_feat=next_s_feat,
                    estimated_cmax=next_cmax_est,
                    current_time=float(env.state.current_time),
                    h0=float(env.state.h0),
                    last_transfer_time=float(env.state.last_transfer_time),
                    is_terminal=False,
                )

            shaped_reward = shaper.shape_reward(
                actual_reward=raw_reward,
                phi_current=phi_current,
                phi_next=phi_next,
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

        # -------------------------
        # GAE 与价值目标结算
        # -------------------------
        # 末尾 Bootstrap 状态值
        if len(buffer) > 0:
            last_cmax = compute_cycle_heuristic_cmax(env.state)
            last_s_feat = extract_compact_state_features(env.state, last_cmax)
            last_r_est = max(0.0, last_cmax - float(env.state.current_time))
            last_u_time = compute_time_urgency_vector(
                estimated_r=last_r_est,
                current_time=env.state.current_time,
                last_transfer_time=env.state.last_transfer_time,
                h0=env.state.h0,
                device=torch_device,
            )
            with torch.no_grad():
                last_v, _ = actor_critic.encode_state(
                    last_s_feat.unsqueeze(0).to(torch_device),
                    last_u_time.unsqueeze(0).to(torch_device),
                )
                last_val = float(last_v.squeeze().item()) if not last_terminated else 0.0

            buffer.finish_trajectory(last_value=last_val)

            # -------------------------
            # PPO 训练更新步
            # -------------------------
            metrics = trainer.train_step(
                buffer=buffer,
                ppo_epochs=ppo_epochs,
                batch_size=batch_size,
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
            "mean_raw_reward": mean_raw_r,
            "mean_shaped_reward": mean_shaped_r,
            "elapsed_seconds": iter_elapsed,
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="工作三 条件分支 PPO 验证训练")
    parser.add_argument("--iterations", type=int, default=15, help="训练轮数 (默认 15)")
    parser.add_argument("--steps", type=int, default=256, help="每轮采集步数 (默认 256)")
    parser.add_argument("--epochs", type=int, default=4, help="PPO 重放轮数 (默认 4)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch 大小 (默认 64)")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率 (默认 3e-4)")
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu/cuda)")
    parser.add_argument("--output", type=str, default="models/work3/checkpoints/method_d_model.pt", help="检查点输出路径")
    args = parser.parse_args()

    run_training(
        num_iterations=args.iterations,
        steps_per_iter=args.steps,
        ppo_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_ckpt=args.output,
    )


if __name__ == "__main__":
    main()
