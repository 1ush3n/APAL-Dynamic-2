"""工作三 条件分支 PPO 训练器 (Task 7.2)。

核心数学模型与技术规范：
依据《方法设计确认稿》第 4 节（条件分支自回归强化学习网络训练）：
1. 目标函数与损失分解：
   L_total(θ) = L_clip(θ) + c_1 · L_vf(θ) - c_2 · H(π_θ)
   - 截断代理策略损失:
     r_t(θ) = exp(log π_θ(a_t|s_t) - log π_old(a_t|s_t))
     L_clip(θ) = - E [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]
   - 价值函数损失:
     L_vf(θ) = 0.5 · E [ (V_θ(s_t) - V_target(s_t))^2 ]
   - 条件动作熵奖励 (鼓励多分支与多工人指针探索):
     H(π_θ) = E [ H(π_task) + H(π_branch) + I(branch==STAY) · (Σ H(π_w) + H(π_align)) ]
2. 梯度稳定与裁剪：
   - 最大梯度范数裁剪 (max_grad_norm = 0.5)；
   - 学习率预热与自适应衰减支持；
   - 数值稳定性断言与 NaN/Inf 熔断防护。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.work3.actor_critic import ActorCriticWork3
from models.work3.ppo_buffer import RolloutBufferWork3

logger = logging.getLogger(__name__)


class PPOTrainerWork3:
    """工作三 条件分支自回归 PPO 策略优化训练器。"""

    def __init__(
        self,
        actor_critic: ActorCriticWork3,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str | torch.device = "cpu",
    ) -> None:
        self.actor_critic = actor_critic
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.device = torch.device(device)

        self.actor_critic.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.actor_critic.parameters(),
            lr=lr,
            eps=1e-5,
            weight_decay=1e-4,
        )

    def train_step(
        self,
        buffer: RolloutBufferWork3,
        ppo_epochs: int = 4,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """使用缓冲区的 Rollout 经验执行一轮多 Epoch PPO 更新。

        Args:
            buffer: 已调用 finish_trajectory 的经验回放缓存
            ppo_epochs: 重放迭代轮数
            batch_size: mini-batch 样本大小

        Returns:
            训练过程诊断统计指标字典
        """
        self.actor_critic.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss_val = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        total_grad_norm = 0.0
        num_updates = 0

        for epoch in range(ppo_epochs):
            for batch in buffer.get_batches(batch_size=batch_size, shuffle=True):
                state_feats = batch["state_feats"].to(self.device)
                time_urgencies = batch["time_urgencies"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                target_values = batch["target_values"].to(self.device)
                sample_records = batch["sample_records"]

                # 1. 批量重放计算当前网络的新策略概率、价值与熵
                values, new_log_probs, entropies = self.actor_critic.evaluate_action_log_probs(
                    state_feats=state_feats,
                    time_urgencies=time_urgencies,
                    sample_records=sample_records,
                )

                # 2. 重要性采样比率 r_t(θ)
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                # 3. PPO 裁剪代理策略损失
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 4. 价值函数平方误差损失
                value_loss = 0.5 * F.mse_loss(values, target_values)

                # 5. 策略熵奖励损失 (-entropy 促使熵最大化)
                entropy = entropies.mean()
                entropy_loss = -entropy

                # 6. 综合损失
                loss = policy_loss + (self.vf_coef * value_loss) + (self.ent_coef * entropy_loss)

                # 7. 反向传播与梯度裁剪
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度断言检查
                grad_norm = nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.max_grad_norm,
                )

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    logger.warning(f"梯度发生异常 (grad_norm={grad_norm})，跳过本 mini-batch 更新！")
                    continue

                self.optimizer.step()

                # 8. 统计诊断数据
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy_loss += float(entropy.item())
                total_loss_val += float(loss.item())
                total_kl += approx_kl
                total_clip_frac += clip_frac
                total_grad_norm += float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                num_updates += 1

        k = max(1, num_updates)
        return {
            "policy_loss": total_policy_loss / k,
            "value_loss": total_value_loss / k,
            "entropy": total_entropy_loss / k,
            "total_loss": total_loss_val / k,
            "approx_kl": total_kl / k,
            "clip_fraction": total_clip_frac / k,
            "grad_norm": total_grad_norm / k,
            "num_updates": num_updates,
        }

    def save_checkpoint(self, path: str) -> None:
        """保存模型与优化器检查点。"""
        torch.save(
            {
                "actor_critic_state": self.actor_critic.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """恢复模型检查点。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
