"""工作三 时间残差预测头训练与 M4 里程碑验收评估脚本 (Task 6.2)。

用法：
    python scripts/work3/train_time_predictor.py \
        --train-trajectories data/work3/m4_train_trajectories.pt \
        --validation-trajectories data/work3/m4_validation_trajectories.pt \
        --epochs 35 \
        --checkpoint models/work3/checkpoints/time_head_m4_official.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any

# 确保项目根目录在 sys.path 中
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from models.work3.train_time_head import (
    train_time_head,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
OFFICIAL_SPLIT_PATHS = {
    "train": _ROOT / "data" / "work3" / "experiment_splits" / "train.json",
    "validation": _ROOT / "data" / "work3" / "experiment_splits" / "validation.json",
    "test": _ROOT / "data" / "work3" / "experiment_splits" / "test.json",
}


def _scenario_ids_from_manifest(path: Path, split_name: str) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("scenarios", manifest) if isinstance(manifest, dict) else manifest
    if not isinstance(entries, list):
        raise ValueError(f"{split_name}场景清单必须是列表：{path}")
    scenario_ids = []
    for entry in entries:
        value = entry if isinstance(entry, str) else (
            entry.get("scenario_id", "") if isinstance(entry, dict) else ""
        )
        scenario_ids.append(str(value).strip())
    if any(not scenario_id for scenario_id in scenario_ids):
        raise ValueError(f"{split_name}场景清单包含空scenario_id：{path}")
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError(f"{split_name}场景清单包含重复scenario_id：{path}")
    if not scenario_ids:
        raise ValueError(f"{split_name}场景清单不能为空：{path}")
    return scenario_ids


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_trajectory_splits(
    train_trajectories_path: str | Path,
    validation_trajectories_path: str | Path,
    train_split_path: str | Path,
    validation_split_path: str | Path,
    test_split_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """读取预先按事件划分的数据，并拒绝跨集合、缺失或重复场景。"""
    manifest_paths = {
        "train": Path(train_split_path),
        "validation": Path(validation_split_path),
        "test": Path(test_split_path),
    }
    manifest_ids = {
        split_name: _scenario_ids_from_manifest(path, split_name)
        for split_name, path in manifest_paths.items()
    }
    for split_name, scenario_ids in manifest_ids.items():
        for other_name, other_ids in manifest_ids.items():
            if split_name < other_name and set(scenario_ids) & set(other_ids):
                raise ValueError(
                    f"官方{split_name}/{other_name}场景清单存在重叠事件"
                )

    trajectory_paths = {
        "train": Path(train_trajectories_path),
        "validation": Path(validation_trajectories_path),
    }
    loaded: dict[str, list[dict[str, Any]]] = {}
    for split_name, path in trajectory_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{split_name}轨迹文件不存在：{path}")
        trajectories = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(trajectories, list) or not trajectories:
            raise ValueError(f"{split_name}轨迹文件必须包含非空轨迹列表：{path}")
        actual_ids = [
            str(item.get("scenario_id", "")).strip()
            for item in trajectories
            if isinstance(item, dict)
        ]
        if len(actual_ids) != len(trajectories) or any(not item for item in actual_ids):
            raise ValueError(f"{split_name}轨迹存在缺失scenario_id的记录：{path}")
        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError(f"{split_name}轨迹文件包含重复scenario_id：{path}")
        expected_ids = set(manifest_ids[split_name])
        unexpected_ids = sorted(set(actual_ids) - expected_ids)
        missing_ids = sorted(expected_ids - set(actual_ids))
        if unexpected_ids or missing_ids:
            raise ValueError(
                f"{split_name}轨迹与{split_name}场景清单不一致；"
                f"不属于{split_name}场景清单={unexpected_ids}，缺失={missing_ids}"
            )
        loaded[split_name] = trajectories

    provenance = {
        "protocol": "official_event_scenario_split_v1",
        "scenario_ids": manifest_ids,
        "manifests": {
            split_name: {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for split_name, path in manifest_paths.items()
        },
        "trajectory_artifacts": {
            split_name: {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for split_name, path in trajectory_paths.items()
        },
    }
    return loaded["train"], loaded["validation"], provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="训练时间残差预测头并进行 M4 里程碑评估")
    parser.add_argument("--train-trajectories", type=Path, required=True)
    parser.add_argument("--validation-trajectories", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_ROOT / "models" / "work3" / "checkpoints" / "time_head_m4_official.pt",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_trajs, val_trajs, data_provenance = load_official_trajectory_splits(
        train_trajectories_path=args.train_trajectories,
        validation_trajectories_path=args.validation_trajectories,
        train_split_path=OFFICIAL_SPLIT_PATHS["train"],
        validation_split_path=OFFICIAL_SPLIT_PATHS["validation"],
        test_split_path=OFFICIAL_SPLIT_PATHS["test"],
    )
    logger.info(
        "加载预先划分轨迹：训练场景=%s、验证场景=%s；测试场景仅登记不参与拟合/选模",
        len(data_provenance["scenario_ids"]["train"]),
        len(data_provenance["scenario_ids"]["validation"]),
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
        data_provenance=data_provenance,
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
