"""工作三PPO的Lightning优化生命周期与rollout数据入口。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch
from lightning.pytorch import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, IterableDataset

from models.work3.actor_critic import ActorCriticWork3
from models.work3.ppo_buffer import RolloutBufferWork3
from models.work3.ppo_trainer import PPOTrainerWork3
from models.work3.time_head import TimeResidualHead


@dataclass(frozen=True, slots=True)
class Work3TrainingUpdate:
    """一次已经采样并完成GAE结算的PPO更新输入。"""

    buffer: RolloutBufferWork3
    environment_steps: int
    time_auxiliary_batch: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.buffer.is_finalized:
            raise ValueError("Work3TrainingUpdate要求GAE已结算的rollout buffer")
        if type(self.environment_steps) is not int or self.environment_steps < 0:
            raise ValueError("environment_steps必须是非负整数")


class _Work3UpdateDataset(IterableDataset[Work3TrainingUpdate]):
    def __init__(
        self,
        update_factory: Callable[[], Iterable[Work3TrainingUpdate]],
    ) -> None:
        super().__init__()
        self._update_factory = update_factory

    def __iter__(self) -> Iterator[Work3TrainingUpdate]:
        return iter(self._update_factory())


class Work3PPODataModule(LightningDataModule):
    """在主进程按需产出rollout更新；环境并行仍由独立worker池负责。"""

    def __init__(
        self,
        *,
        update_factory: Callable[[], Iterable[Work3TrainingUpdate]],
    ) -> None:
        super().__init__()
        self._update_factory = update_factory

    def train_dataloader(self) -> DataLoader[Work3TrainingUpdate]:
        return DataLoader(
            _Work3UpdateDataset(self._update_factory),
            batch_size=None,
            num_workers=0,
            pin_memory=False,
        )


class Work3LightningModule(LightningModule):
    """唯一持有工作三优化器的LightningModule；PPOTrainer仅提供损失计算。"""

    automatic_optimization = False

    def __init__(
        self,
        *,
        actor_critic: ActorCriticWork3,
        time_head: TimeResidualHead | None = None,
        learning_rate: float = 3e-4,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        time_loss_coef: float = 0.0,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        time_auxiliary_epochs: int = 1,
        time_auxiliary_batch_size: int = 64,
    ) -> None:
        super().__init__()
        if learning_rate <= 0 or ppo_epochs < 0 or batch_size < 1:
            raise ValueError("Lightning PPO学习率与batch_size必须为正数，epoch不能为负数")
        if max_grad_norm <= 0 or time_auxiliary_epochs < 0 or time_auxiliary_batch_size < 1:
            raise ValueError("梯度裁剪和时间监督配置无效")

        self.actor_critic = actor_critic
        self.time_head = time_head
        self.learning_rate = float(learning_rate)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.batch_size = int(batch_size)
        self.time_auxiliary_epochs = int(time_auxiliary_epochs)
        self.time_auxiliary_batch_size = int(time_auxiliary_batch_size)
        self.objective = PPOTrainerWork3(
            actor_critic=actor_critic,
            lr=learning_rate,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            time_head=time_head,
            time_loss_coef=time_loss_coef,
            device="cpu",
            create_optimizer=False,
        )
        self.optimization_steps = 0
        self.environment_steps = 0
        self.last_metrics: dict[str, float | None] = {}
        self._optimizer_parameter_ids: tuple[int, ...] = ()
        self._optimizer: torch.optim.Optimizer | None = None

    @property
    def optimizers_configured_parameter_ids(self) -> tuple[int, ...]:
        return self._optimizer_parameter_ids

    def configure_optimizers(self) -> torch.optim.Optimizer:
        parameters: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for parameter in self.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
        if not parameters:
            raise RuntimeError("工作三Lightning模块没有可训练参数")
        parameter_ids = tuple(id(parameter) for parameter in parameters)
        if self._optimizer is not None:
            if parameter_ids != self._optimizer_parameter_ids:
                raise RuntimeError("Lightning重复fit期间可训练参数集合发生变化")
            return self._optimizer
        self._optimizer_parameter_ids = parameter_ids
        self._optimizer = torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            eps=1e-5,
            weight_decay=1e-4,
        )
        return self._optimizer

    def transfer_batch_to_device(
        self,
        batch: Work3TrainingUpdate,
        device: torch.device,
        dataloader_idx: int,
    ) -> Work3TrainingUpdate:
        """保留CPU回放快照；前向重放只将当前张量移到模型设备。"""
        del device, dataloader_idx
        if not isinstance(batch, Work3TrainingUpdate):
            raise TypeError("DataModule必须产出Work3TrainingUpdate")
        return batch

    def on_fit_start(self) -> None:
        self.objective.device = self.device

    def _manual_update(self, loss: torch.Tensor, optimizer: Any) -> float:
        if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("Lightning收到非有限或非标量训练损失")
        optimizer.zero_grad()
        self.manual_backward(loss)
        parameters = [parameter for parameter in self.parameters() if parameter.grad is not None]
        if any(not bool(torch.isfinite(parameter.grad).all().item()) for parameter in parameters):
            optimizer.zero_grad()
            raise FloatingPointError("Lightning PPO梯度包含NaN或Inf")
        grad_norm = self.clip_gradients(
            optimizer,
            gradient_clip_val=self.max_grad_norm,
            gradient_clip_algorithm="norm",
        )
        optimizer.step()
        self.optimization_steps += 1
        if isinstance(grad_norm, torch.Tensor):
            return float(grad_norm.detach().float().item())
        return float(grad_norm) if grad_norm is not None else 0.0

    def training_step(self, batch: Work3TrainingUpdate, batch_idx: int) -> torch.Tensor:
        del batch_idx
        if not isinstance(batch, Work3TrainingUpdate):
            raise TypeError("DataModule必须产出Work3TrainingUpdate")
        if not batch.buffer.is_finalized:
            raise ValueError("训练step收到未完成GAE的rollout buffer")

        self.actor_critic.train()
        if self.time_head is not None:
            self.time_head.train()
        optimizer = self.optimizers()
        metrics: dict[str, list[float]] = {
            "total_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "grad_norm": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        sampling_replay_max_abs_error: float | None = None
        sampling_replay_sample_count = 0
        ppo_update_count = 0
        time_supervision_steps = 0
        time_label_count = (
            int(batch.time_auxiliary_batch["target_residuals"].numel())
            if batch.time_auxiliary_batch is not None
            else 0
        )
        for _ in range(self.ppo_epochs):
            for minibatch in batch.buffer.get_batches(batch_size=self.batch_size, shuffle=True):
                losses = self.objective.compute_ppo_minibatch_loss(minibatch)
                if sampling_replay_max_abs_error is None:
                    with torch.no_grad():
                        sampling_replay_max_abs_error = float(
                            (losses["new_log_probs"] - minibatch["old_log_probs"])
                            .abs()
                            .max()
                            .item()
                        )
                    sampling_replay_sample_count = int(
                        minibatch["old_log_probs"].numel()
                    )
                grad_norm = self._manual_update(losses["total_loss"], optimizer)
                for key in (
                    "total_loss",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "approx_kl",
                    "clip_fraction",
                ):
                    metrics[key].append(float(losses[key].detach().float().item()))
                metrics["grad_norm"].append(grad_norm)
                ppo_update_count += 1

        auxiliary = batch.time_auxiliary_batch
        if (
            auxiliary is not None
            and self.time_head is not None
            and self.objective.time_loss_coef > 0.0
        ):
            count = int(auxiliary["target_residuals"].numel())
            for _ in range(self.time_auxiliary_epochs):
                indices = torch.randperm(count).tolist()
                for start in range(0, count, self.time_auxiliary_batch_size):
                    chosen = indices[start : start + self.time_auxiliary_batch_size]
                    minibatch = {
                        "state_feats": auxiliary["state_feats"][chosen],
                        "graph_snapshots": [auxiliary["graph_snapshots"][i] for i in chosen],
                        "target_residuals": auxiliary["target_residuals"][chosen],
                        "worker_ids": [auxiliary["worker_ids"][i] for i in chosen],
                        "episode_ids": [auxiliary["episode_ids"][i] for i in chosen],
                        "cycle_ids": [auxiliary["cycle_ids"][i] for i in chosen],
                        "decision_ids": [auxiliary["decision_ids"][i] for i in chosen],
                    }
                    time_loss = self.objective.compute_time_auxiliary_loss(minibatch)
                    self._manual_update(self.objective.time_loss_coef * time_loss, optimizer)
                    time_supervision_steps += 1
                    metrics.setdefault("time_loss", []).append(
                        float(time_loss.detach().float().item())
                    )

        self.environment_steps += batch.environment_steps
        self.last_metrics = {
            key: sum(values) / len(values)
            for key, values in metrics.items()
            if values
        }
        if not self.last_metrics:
            self.last_metrics = {"total_loss": 0.0}
        self.last_metrics["ppo_updates"] = float(ppo_update_count)
        self.last_metrics["time_label_count"] = float(time_label_count)
        self.last_metrics["time_supervision_steps"] = float(time_supervision_steps)
        self.last_metrics["time_supervision_epochs"] = float(self.time_auxiliary_epochs)
        self.last_metrics["optimization_steps"] = float(self.optimization_steps)
        self.last_metrics["environment_steps"] = float(self.environment_steps)
        self.last_metrics["sampling_replay_max_abs_error"] = (
            sampling_replay_max_abs_error
            if sampling_replay_max_abs_error is not None
            else None
        )
        self.last_metrics["sampling_replay_sample_count"] = float(
            sampling_replay_sample_count
        )
        for name, value in self.last_metrics.items():
            if value is None:
                continue
            self.log(f"work3/{name}", value, on_step=True, on_epoch=False, prog_bar=False)
        return torch.tensor(self.last_metrics["total_loss"], device=self.device)
