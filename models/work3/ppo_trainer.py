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
from models.work3.graph_builder import (
    GRAPH_FEATURE_DIMS,
    GRAPH_FEATURE_SCHEMA,
    GRAPH_FEATURE_VERSION,
)
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
        time_head: nn.Module | None = None,
        time_loss_coef: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.actor_critic = actor_critic
        self.clip_eps = float(clip_eps)
        self.vf_coef = float(vf_coef)
        self.ent_coef = float(ent_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.time_head = time_head
        self.time_loss_coef = float(time_loss_coef)
        self.device = torch.device(device)

        self.actor_critic.to(self.device)
        if self.time_head is not None:
            self.time_head.to(self.device)
        self.optimized_parameters: list[nn.Parameter] = list(self.actor_critic.parameters())
        if self.time_head is not None:
            actor_parameter_ids = {id(parameter) for parameter in self.optimized_parameters}
            self.optimized_parameters.extend(
                parameter
                for parameter in self.time_head.parameters()
                if id(parameter) not in actor_parameter_ids
            )
        self.optimizer = torch.optim.AdamW(
            self.optimized_parameters,
            lr=lr,
            eps=1e-5,
            weight_decay=1e-4,
        )

    def compute_time_auxiliary_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """在采样快照上计算共享图表征的有符号时间监督损失。"""
        if self.time_head is None:
            raise RuntimeError("未配置时间残差头，不能计算时间辅助损失")
        state_feats = batch["state_feats"].to(self.device)
        graph_snapshots = batch["graph_snapshots"]
        targets = batch["target_residuals"].to(self.device)
        episode_ids = batch["episode_ids"]
        cycle_ids = batch["cycle_ids"]
        decision_ids = batch["decision_ids"]
        assert state_feats.ndim == 2
        assert len(graph_snapshots) == state_feats.size(0)
        assert targets.shape == (state_feats.size(0),)
        assert len(episode_ids) == state_feats.size(0)
        assert len(cycle_ids) == state_feats.size(0)
        assert len(decision_ids) == state_feats.size(0)
        assert len(set(decision_ids)) == len(decision_ids)

        shared_features = torch.stack([
            self.actor_critic.encode_shared_representation(
                state_feats[index],
                graph_snapshots[index],
            )
            for index in range(state_feats.size(0))
        ])
        predictions = self.time_head(shared_features)
        assert predictions.shape == targets.shape
        per_sample_loss = F.smooth_l1_loss(
            predictions,
            targets,
            beta=0.01,
            reduction="none",
        )
        return per_sample_loss.mean()

    def train_step(
        self,
        buffer: RolloutBufferWork3,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        time_auxiliary_batch: dict[str, Any] | None = None,
        time_auxiliary_epochs: int = 1,
        time_auxiliary_batch_size: int = 64,
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
        if self.time_head is not None:
            self.time_head.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss_val = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        total_grad_norm = 0.0
        num_updates = 0
        if time_auxiliary_epochs < 0:
            raise ValueError("time_auxiliary_epochs不能为负数")
        if time_auxiliary_batch_size <= 0:
            raise ValueError("time_auxiliary_batch_size必须为正数")

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
                loss = (
                    policy_loss
                    + (self.vf_coef * value_loss)
                    + (self.ent_coef * entropy_loss)
                )

                # 7. 反向传播与梯度裁剪
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度断言检查
                grad_norm = nn.utils.clip_grad_norm_(
                    self.optimized_parameters,
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

        time_label_count = 0
        time_supervision_steps = 0
        total_time_loss = 0.0
        if time_auxiliary_batch is not None:
            if self.time_head is None:
                raise RuntimeError("收到时间标签批次，但训练器未配置时间头")
            time_label_count = int(time_auxiliary_batch["target_residuals"].numel())
            if time_label_count and self.time_loss_coef > 0.0:
                graph_snapshots = time_auxiliary_batch["graph_snapshots"]
                episode_ids = time_auxiliary_batch["episode_ids"]
                cycle_ids = time_auxiliary_batch["cycle_ids"]
                decision_ids = time_auxiliary_batch["decision_ids"]
                if not (
                    len(graph_snapshots)
                    == len(episode_ids)
                    == len(cycle_ids)
                    == len(decision_ids)
                    == time_label_count
                ):
                    raise ValueError("时间监督批次的图与标签元数据长度不一致")
                if len(set(decision_ids)) != len(decision_ids):
                    raise ValueError("时间监督批次含重复decision_id")

                for _ in range(time_auxiliary_epochs):
                    indices = torch.randperm(time_label_count)
                    for start in range(0, time_label_count, time_auxiliary_batch_size):
                        batch_indices = indices[
                            start : start + time_auxiliary_batch_size
                        ].tolist()
                        auxiliary_minibatch = {
                            "state_feats": time_auxiliary_batch["state_feats"][batch_indices],
                            "graph_snapshots": [graph_snapshots[index] for index in batch_indices],
                            "target_residuals": time_auxiliary_batch["target_residuals"][batch_indices],
                            "episode_ids": [episode_ids[index] for index in batch_indices],
                            "cycle_ids": [cycle_ids[index] for index in batch_indices],
                            "decision_ids": [decision_ids[index] for index in batch_indices],
                        }
                        self.optimizer.zero_grad()
                        time_loss = self.compute_time_auxiliary_loss(auxiliary_minibatch)
                        auxiliary_loss = self.time_loss_coef * time_loss
                        auxiliary_loss.backward()
                        grad_norm = nn.utils.clip_grad_norm_(
                            self.optimized_parameters,
                            self.max_grad_norm,
                        )
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            logger.warning(
                                "时间辅助梯度异常，跳过本辅助 mini-batch 更新！"
                            )
                            continue
                        self.optimizer.step()
                        total_time_loss += float(time_loss.item())
                        time_supervision_steps += 1

        k = max(1, num_updates)
        return {
            "policy_loss": total_policy_loss / k,
            "value_loss": total_value_loss / k,
            "entropy": total_entropy_loss / k,
            "total_loss": total_loss_val / k,
            "approx_kl": total_kl / k,
            "clip_fraction": total_clip_frac / k,
            "grad_norm": total_grad_norm / k,
            "time_loss": total_time_loss / max(1, time_supervision_steps),
            "time_label_count": time_label_count,
            "time_supervision_steps": time_supervision_steps,
            "time_supervision_epochs": (
                time_auxiliary_epochs if time_supervision_steps else 0
            ),
            "num_updates": num_updates,
        }

    def save_checkpoint(self, path: str) -> None:
        """保存模型与优化器检查点。"""
        checkpoint: dict[str, Any] = {
            "checkpoint_version": "work3_actor_time_v2",
            "graph_feature_version": GRAPH_FEATURE_VERSION,
            "graph_feature_dims": dict(GRAPH_FEATURE_DIMS),
            "graph_feature_schema": dict(GRAPH_FEATURE_SCHEMA),
            "actor_critic_state": self.actor_critic.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        if self.time_head is not None:
            checkpoint.update({
                "time_head_state": self.time_head.state_dict(),
                "time_head_model_version": "signed_residual_v1",
                "time_head_in_dim": int(getattr(self.time_head, "in_dim", -1)),
            })
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """恢复模型检查点。"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("graph_feature_version") != GRAPH_FEATURE_VERSION or dict(
            checkpoint.get("graph_feature_dims") or {}
        ) != dict(GRAPH_FEATURE_DIMS):
            raise ValueError(
                f"检查点图特征版本或维度不匹配：期望 {GRAPH_FEATURE_VERSION} {GRAPH_FEATURE_DIMS}，"
                f"实际 {checkpoint.get('graph_feature_version')} {checkpoint.get('graph_feature_dims')}"
            )
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state"])
        if self.time_head is not None:
            if "time_head_state" not in checkpoint:
                raise ValueError("检查点缺少共享图时间头，不能作为完整工作三模型恢复")
            if checkpoint.get("time_head_model_version") != "signed_residual_v1":
                raise ValueError("检查点时间头不是有符号残差版本")
            self.time_head.load_state_dict(checkpoint["time_head_state"])
        elif "time_head_state" in checkpoint:
            raise ValueError("完整工作三检查点包含时间头，但当前训练器未配置时间头")
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
