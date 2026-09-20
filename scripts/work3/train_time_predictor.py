"""工作三 时间残差预测头训练与 M4 里程碑验收评估脚本 (Task 6.2)。

用法：
    python scripts/work3/train_time_predictor.py \
        --trajectories data/work3/val_trajectories.pt \
        --epochs 35 \
        --checkpoint models/work3/checkpoints/time_head_best.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

# 确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from models.work3.train_time_head import (
    evaluate_time_head,
    split_trajectories_by_scenario,
    train_time_head,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练时间残差预测头并进行 M4 里程碑评估")
    parser.add_argument("--trajectories", type=str, default="data/work3/val_trajectories.pt")
    parser.add_argument("--checkpoint", type=str, default="models/work3/checkpoints/time_head_best.pt")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    traj_path = Path(args.trajectories)
    if not traj_path.is_file():
        raise FileNotFoundError(f"轨迹数据集 {traj_path} 不存在，请先运行数据收集脚本！")

    logger.info(f"正在加载轨迹数据集: {traj_path}...")
    trajectories = torch.load(traj_path, weights_only=False)
    logger.info(f"成功加载 {len(trajectories)} 条生产流水线轨迹")

    # 轨迹级切分
    train_trajs, val_trajs = split_trajectories_by_scenario(
        trajectories,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 执行训练
    model, best_metrics = train_time_head(
        train_trajectories=train_trajs,
        val_trajectories=val_trajs,
        in_dim=32,
        hidden_dim=64,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=args.checkpoint,
    )

    # 打印最终里程碑 M4 验收报告
    mae_raw = best_metrics["mae_raw_hours"]
    mae_corr = best_metrics["mae_corrected_hours"]
    impr = best_metrics["mae_improvement_pct"]

    print("\n" + "=" * 70)
    print("【工作三 里程碑 M4 验收核验报告：时间残差预测头离线回归】")
    print("=" * 70)
    print(f"验证集轨迹数量: {len(val_trajs)} 条 (总计 {best_metrics['total_val_samples']} 决策步)")
    print(f"原始无学习启发式 MAE : {mae_raw:.4f} 小时")
    print(f"时间残差修正后   MAE : {mae_corr:.4f} 小时")
    print(f"预测误差降低幅度     : {impr:+.2f}%")
    print("-" * 70)

    if impr >= 15.0:
        print(f"[PASS] 误差改善幅度 {impr:.2f}% >= 15.0%，顺利达到里程碑 M4 设定标准！")
    else:
        print(f"[NOTICE] 误差改善幅度 {impr:.2f}% < 15.0%。")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
