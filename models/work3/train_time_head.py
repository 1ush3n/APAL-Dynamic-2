"""工作三 时间残差预测头训练与离线验证模块 (Task 6.2)。

核心数学模型与技术规范：
依据《方法设计确认稿》第 7.2 节（修正量预测）与第 10 节代码继承规范：
1. 监督目标：
   真实脉动转站时刻 P_q 相对启发式估计 P_q^h 的归一化残差：
   y = (P_q^actual - P_q^h) / H_0
2. 轨迹级无泄漏切分 (Trajectory-level Split)：
   按完整生产流水线轨迹划分离线训练集与验证集，严防相邻 decision step 出现时序穿越；
3. 训练目标与鲁棒损失：
   采用 Huber 损失 (Smooth L1 Loss) 优化轻量级时间修正头 δ_ψ(s)；
4. 【里程碑 M4 验收标准】：
   在独立留出的验证轨迹上，对比修正完工时间 P^_q 与原始启发式 P_q^h 的 MAE 误差：
   MAE_raw = mean(|P_q^actual - P_q^h|)
   MAE_corrected = mean(|P_q^actual - P^_q|)
   要求 ΔMAE = (MAE_raw - MAE_corrected) / MAE_raw >= 15.0%。
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from models.work3.time_head import TimeResidualHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class StepResidualDataset(Dataset):
    """从生产轨迹中提取的决策步残差监督数据集。"""

    def __init__(self, trajectories: list[dict[str, Any]]) -> None:
        self.samples: list[dict[str, Any]] = []
        for traj in trajectories:
            h0 = float(traj["h0"])
            for step in traj["steps"]:
                if (
                    step.get("label_available", step.get("label_y") is not None) is False
                    or step.get("label_y") is None
                    or step.get("actual_transfer_time") is None
                ):
                    continue
                feat = step["state_feat"]
                if not isinstance(feat, torch.Tensor):
                    feat = torch.tensor(feat, dtype=torch.float)
                self.samples.append({
                    "state_feat": feat,
                    "label_y": float(step["label_y"]),
                    "estimated_cmax": float(step["estimated_cmax"]),
                    "actual_transfer_time": float(step["actual_transfer_time"]),
                    "current_time": float(step["current_time"]),
                    "h0": h0,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def collate_step_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """批处理整合函数。"""
    return {
        "state_feat": torch.stack([s["state_feat"] for s in batch]),
        "label_y": torch.tensor([s["label_y"] for s in batch], dtype=torch.float),
        "estimated_cmax": torch.tensor([s["estimated_cmax"] for s in batch], dtype=torch.float),
        "actual_transfer_time": torch.tensor([s["actual_transfer_time"] for s in batch], dtype=torch.float),
        "current_time": torch.tensor([s["current_time"] for s in batch], dtype=torch.float),
        "h0": torch.tensor([s["h0"] for s in batch], dtype=torch.float),
    }


def split_trajectories_by_scenario(
    trajectories: list[dict[str, Any]],
    val_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """去重后按scenario整体划分，拒绝没有独立验证场景的数据集。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, bytes]] = set()
    for trajectory in trajectories:
        scenario_id = str(trajectory.get("scenario_id") or "").strip()
        if not scenario_id:
            raise ValueError("每条轨迹必须包含非空scenario_id，才能进行无泄漏划分")
        fingerprint = _trajectory_supervision_fingerprint(trajectory)
        identity = (scenario_id, fingerprint)
        if identity in seen:
            continue
        seen.add(identity)
        grouped.setdefault(scenario_id, []).append(trajectory)

    scenario_ids = sorted(grouped)
    if len(scenario_ids) < 2:
        raise ValueError("时间头训练/验证切分至少需要两个独立scenario")

    rng = np.random.RandomState(seed)
    rng.shuffle(scenario_ids)
    n_val = min(
        len(scenario_ids) - 1,
        max(1, int(round(len(scenario_ids) * val_ratio))),
    )
    val_scenario_ids = set(scenario_ids[:n_val])
    train_trajs = [
        trajectory
        for scenario_id, items in grouped.items()
        if scenario_id not in val_scenario_ids
        for trajectory in items
    ]
    val_trajs = [
        trajectory
        for scenario_id, items in grouped.items()
        if scenario_id in val_scenario_ids
        for trajectory in items
    ]

    logger.info(
        f"轨迹切分完成: 训练场景={len(scenario_ids) - n_val}个、轨迹={len(train_trajs)}条 "
        f"验证场景={n_val}个、轨迹={len(val_trajs)}条 "
        f"训练步数={sum(len(t['steps']) for t in train_trajs)}, "
        f"验证步数={sum(len(t['steps']) for t in val_trajs)}"
    )
    return train_trajs, val_trajs


def _trajectory_supervision_fingerprint(trajectory: dict[str, Any]) -> bytes:
    """对时间头实际消费的特征与监督值做定长摘要，忽略trajectory_id。"""
    digest = hashlib.sha256()
    scalar_keys = (
        "current_time",
        "estimated_cmax",
        "actual_transfer_time",
        "label_y",
    )
    digest.update(np.asarray(float(trajectory["h0"]), dtype="<f4").tobytes())
    steps = trajectory["steps"]
    digest.update(len(steps).to_bytes(8, byteorder="little", signed=False))
    for step in steps:
        feature = step["state_feat"]
        if isinstance(feature, torch.Tensor):
            feature = feature.detach().cpu().numpy()
        # 输入形状[D]保持不变；统一float32字节表示，不保存整条副本的展开列表。
        feature_array = np.asarray(feature, dtype="<f4")
        assert feature_array.ndim == 1, "时间头状态特征必须为[D]"
        digest.update(np.asarray(feature_array.shape, dtype="<i8").tobytes())
        digest.update(feature_array.tobytes(order="C"))
        for key in scalar_keys:
            value = step.get(key)
            if value is None:
                digest.update(b"N")
            else:
                digest.update(b"F")
                digest.update(np.asarray(float(value), dtype="<f4").tobytes())
    return digest.digest()


def evaluate_time_head(
    model: TimeResidualHead,
    val_trajectories: list[dict[str, Any]],
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """在留出验证轨迹上评估修正时间与原始启发式的 MAE 误差。"""
    model.eval()
    val_dataset = StepResidualDataset(val_trajectories)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=collate_step_batch)

    total_raw_abs_err = 0.0
    total_corrected_abs_err = 0.0
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["state_feat"].to(device)
            y = batch["label_y"].to(device)
            est_cmax = batch["estimated_cmax"].to(device)
            actual_time = batch["actual_transfer_time"].to(device)
            current_time = batch["current_time"].to(device)
            h0 = batch["h0"].to(device)

            delta = model(x)
            loss = F.smooth_l1_loss(delta, y, beta=0.1)

            # 物理修正时间 P^_q = max{t, P_q^h + H_0 * delta}
            p_corrected = torch.maximum(current_time, est_cmax + h0 * delta)

            raw_err = torch.abs(actual_time - est_cmax).sum().item()
            corr_err = torch.abs(actual_time - p_corrected).sum().item()

            total_raw_abs_err += raw_err
            total_corrected_abs_err += corr_err
            total_loss += loss.item() * len(y)
            total_samples += len(y)

    mae_raw = total_raw_abs_err / max(1, total_samples)
    mae_corrected = total_corrected_abs_err / max(1, total_samples)

    if mae_raw > 1e-6:
        mae_improvement_pct = ((mae_raw - mae_corrected) / mae_raw) * 100.0
    else:
        mae_improvement_pct = 0.0

    return {
        "val_loss": total_loss / max(1, total_samples),
        "mae_raw_hours": mae_raw,
        "mae_corrected_hours": mae_corrected,
        "mae_improvement_pct": mae_improvement_pct,
        "total_val_samples": total_samples,
    }


def train_time_head(
    train_trajectories: list[dict[str, Any]],
    val_trajectories: list[dict[str, Any]],
    in_dim: int = 32,
    hidden_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_path: str | None = "models/work3/checkpoints/time_head_best.pt",
    device: str = "cpu",
    data_provenance: dict[str, Any] | None = None,
) -> tuple[TimeResidualHead, dict[str, Any]]:
    """训练时间修正头并执行离线验证，保存最佳检查点。"""
    torch_device = torch.device(device)
    model = TimeResidualHead(in_dim=in_dim, hidden_dim=hidden_dim).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_dataset = StepResidualDataset(train_trajectories)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_step_batch,
    )

    best_mae_improvement = -float("inf")
    best_eval_metrics: dict[str, Any] = {}

    logger.info(f"开始时间修正头辅助训练: 总样本={len(train_dataset)}, 训练轮数={epochs}...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            x = batch["state_feat"].to(torch_device)
            y = batch["label_y"].to(torch_device)
            optimizer.zero_grad()
            delta = model(x)
            # 正负残差都参加同一个有符号回归损失。
            loss = F.smooth_l1_loss(delta, y, beta=0.01)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_samples += len(y)

        scheduler.step()
        train_loss_avg = train_loss / max(1, train_samples)

        # 评估验证集
        metrics = evaluate_time_head(model, val_trajectories, device=torch_device)
        logger.info(
            f"[Epoch {epoch:02d}/{epochs:02d}] 训练Loss={train_loss_avg:.5f} | "
            f"验证Loss={metrics['val_loss']:.5f} | "
            f"原始MAE={metrics['mae_raw_hours']:.3f}h | "
            f"修正MAE={metrics['mae_corrected_hours']:.3f}h | "
            f"误差改善={metrics['mae_improvement_pct']:+.2f}%"
        )

        if metrics["mae_improvement_pct"] > best_mae_improvement:
            best_mae_improvement = metrics["mae_improvement_pct"]
            best_eval_metrics = dict(metrics)
            best_eval_metrics["best_epoch"] = epoch

            if checkpoint_path is not None:
                ckpt_p = Path(checkpoint_path)
                ckpt_p.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "in_dim": in_dim,
                    "hidden_dim": hidden_dim,
                    "model_version": "signed_residual_v1",
                    "metrics": best_eval_metrics,
                    "data_provenance": data_provenance,
                }, ckpt_p)

    logger.info(
        f"训练完成！最佳验证集误差改善={best_eval_metrics.get('mae_improvement_pct', 0.0):.2f}% "
        f"(原始 {best_eval_metrics.get('mae_raw_hours', 0.0):.3f}h -> "
        f"修正 {best_eval_metrics.get('mae_corrected_hours', 0.0):.3f}h)"
    )

    return model, best_eval_metrics
