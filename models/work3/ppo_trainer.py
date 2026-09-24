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
PPO_CHECKPOINT_VERSION = "work3_actor_time_v2"


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
        create_optimizer: bool = True,
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
        self.optimizer: torch.optim.Optimizer | None = None
        if create_optimizer:
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
        worker_ids = batch.get("worker_ids", [0] * state_feats.size(0))
        episode_ids = batch["episode_ids"]
        cycle_ids = batch["cycle_ids"]
        decision_ids = batch["decision_ids"]
        assert state_feats.ndim == 2
        assert len(graph_snapshots) == state_feats.size(0)
        assert targets.shape == (state_feats.size(0),)
        assert len(worker_ids) == state_feats.size(0)
        assert len(episode_ids) == state_feats.size(0)
        assert len(cycle_ids) == state_feats.size(0)
        assert len(decision_ids) == state_feats.size(0)
        decision_keys = tuple(zip(worker_ids, episode_ids, decision_ids, strict=True))
        if len(set(decision_keys)) != len(decision_keys):
            raise ValueError("时间监督批次含重复(worker_id, episode_id, decision_id)")

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

    def compute_ppo_minibatch_loss(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """纯计算一个PPO minibatch的损失，不反传、不创建或更新优化器。"""
        state_feats = batch["state_feats"].to(self.device)
        time_urgencies = batch["time_urgencies"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device, dtype=torch.float32)
        advantages = batch["advantages"].to(self.device, dtype=torch.float32)
        target_values = batch["target_values"].to(self.device, dtype=torch.float32)
        assert state_feats.ndim == 2 and state_feats.size(-1) == 32
        assert time_urgencies.shape == (state_feats.size(0), 2)
        assert old_log_probs.shape == advantages.shape == target_values.shape == (
            state_feats.size(0),
        )

        values, new_log_probs, entropies = self.actor_critic.evaluate_action_log_probs(
            state_feats=state_feats,
            time_urgencies=time_urgencies,
            sample_records=batch["sample_records"],
        )
        values = values.float()
        new_log_probs = new_log_probs.float()
        entropies = entropies.float()
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        surrogate = ratio * advantages
        clipped_surrogate = torch.clamp(
            ratio,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps,
        ) * advantages
        policy_loss = -torch.min(surrogate, clipped_surrogate).mean()
        value_loss = 0.5 * F.mse_loss(values, target_values)
        entropy = entropies.mean()
        total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = (ratio.sub(1.0).abs() > self.clip_eps).float().mean()
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "values": values,
            "new_log_probs": new_log_probs,
            "entropies": entropies,
            "ratio": ratio,
            "log_ratio": log_ratio,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def train_step(
        self,
        buffer: RolloutBufferWork3,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        time_auxiliary_batch: dict[str, Any] | None = None,
        time_auxiliary_epochs: int = 1,
        time_auxiliary_batch_size: int = 64,
    ) -> dict[str, Any]:
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
        sampling_replay_checked = False
        sampling_replay_max_abs_error: float | None = None
        sampling_replay_sample_count = 0
        if time_auxiliary_epochs < 0:
            raise ValueError("time_auxiliary_epochs不能为负数")
        if time_auxiliary_batch_size <= 0:
            raise ValueError("time_auxiliary_batch_size必须为正数")
        if self.optimizer is None:
            raise RuntimeError("当前PPOTrainer未创建优化器；请使用Lightning训练模块更新")

        for epoch in range(ppo_epochs):
            for batch in buffer.get_batches(batch_size=batch_size, shuffle=True):
                old_log_probs = batch["old_log_probs"].to(self.device)
                batch_losses = self.compute_ppo_minibatch_loss(batch)
                values = batch_losses["values"]
                new_log_probs = batch_losses["new_log_probs"]
                entropies = batch_losses["entropies"]

                if not sampling_replay_checked:
                    with torch.no_grad():
                        sampling_replay_max_abs_error = float(
                            (new_log_probs.detach() - old_log_probs).abs().max().item()
                        )
                    sampling_replay_sample_count = int(old_log_probs.numel())
                    sampling_replay_checked = True

                # 纯损失由共享计算路径给出；这里仅为旧入口执行优化步骤。
                log_ratio = batch_losses["log_ratio"]
                ratio = batch_losses["ratio"]
                policy_loss = batch_losses["policy_loss"]
                value_loss = batch_losses["value_loss"]
                entropy = batch_losses["entropy"]
                loss = batch_losses["total_loss"]

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
                worker_ids = time_auxiliary_batch.get(
                    "worker_ids",
                    [0] * time_label_count,
                )
                episode_ids = time_auxiliary_batch["episode_ids"]
                cycle_ids = time_auxiliary_batch["cycle_ids"]
                decision_ids = time_auxiliary_batch["decision_ids"]
                if not (
                    len(graph_snapshots)
                    == len(worker_ids)
                    == len(episode_ids)
                    == len(cycle_ids)
                    == len(decision_ids)
                    == time_label_count
                ):
                    raise ValueError("时间监督批次的图与标签元数据长度不一致")
                decision_keys = tuple(zip(worker_ids, episode_ids, decision_ids, strict=True))
                if len(set(decision_keys)) != len(decision_keys):
                    raise ValueError(
                        "时间监督批次含重复(worker_id, episode_id, decision_id)"
                    )

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
                            "worker_ids": [worker_ids[index] for index in batch_indices],
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
            "sampling_replay_max_abs_error": sampling_replay_max_abs_error,
            "sampling_replay_sample_count": sampling_replay_sample_count,
        }

    def save_checkpoint(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        """保存推理/热启动权重；不承诺恢复完整训练状态。"""
        checkpoint: dict[str, Any] = {
            "checkpoint_version": PPO_CHECKPOINT_VERSION,
            "checkpoint_role": "model_weights",
            "resume_capability": "non_exact",
            "graph_feature_version": GRAPH_FEATURE_VERSION,
            "graph_feature_dims": dict(GRAPH_FEATURE_DIMS),
            "graph_feature_schema": dict(GRAPH_FEATURE_SCHEMA),
            "actor_critic_state": self.actor_critic.state_dict(),
            "run_metadata": dict(metadata or {}),
        }
        if self.optimizer is not None:
            checkpoint["optimizer_state"] = self.optimizer.state_dict()
        if self.time_head is not None:
            checkpoint.update({
                "time_head_state": self.time_head.state_dict(),
                "time_head_model_version": "signed_residual_v1",
                "time_head_in_dim": int(getattr(self.time_head, "in_dim", -1)),
            })
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """加载模型/优化器权重；此操作不恢复环境和采样状态，不能精确续训。"""
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
        if "optimizer_state" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
