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
import math
import numbers
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


def _validate_official_trajectory(trajectory: dict[str, Any], split_name: str) -> None:
    """在拟合前核验一条正式轨迹的完成状态、数值与真实周期标签。"""
    scenario_id = str(trajectory.get("scenario_id", ""))

    def reject(field: str, reason: str, step_idx: object | None = None) -> None:
        location = f"{split_name}轨迹scenario_id={scenario_id}"
        if step_idx is not None:
            location += f" step_idx={step_idx}"
        raise ValueError(f"{location} field={field}: {reason}")

    if trajectory.get("success") is not True:
        reject("success", "轨迹未成功完成")
    if trajectory.get("termination_reason") != "completed":
        reject("termination_reason", "轨迹退出原因不是completed")
    if trajectory.get("feasible") is not True:
        reject("feasible", "独立可行性检查未通过")
    if trajectory.get("constraint_violations") != {}:
        reject("constraint_violations", "存在约束违规")

    def positive_integer(field: str) -> int:
        value = trajectory.get(field)
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
            reject(field, "必须为正整数")
        return int(value)

    total_tasks = positive_integer("total_tasks")
    completed_tasks = positive_integer("completed_tasks")
    if completed_tasks != total_tasks:
        reject("completed_tasks", f"完成数{completed_tasks}不等于总数{total_tasks}")

    def finite_number(
        value: object,
        field: str,
        step_idx: object | None = None,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            reject(field, "必须为数值且不能是布尔值或字符串", step_idx)
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            reject(field, "必须为有限数值", step_idx)
        return numeric_value

    h0 = finite_number(trajectory.get("h0"), "h0")
    if h0 <= 0.0:
        reject("h0", "必须大于0")

    raw_history = trajectory.get("transfer_history")
    if not isinstance(raw_history, (list, tuple)) or not raw_history:
        reject("transfer_history", "必须包含完整、非空的真实转站时刻序列")
    transfer_history = [
        finite_number(value, f"transfer_history[{index}]")
        for index, value in enumerate(raw_history)
    ]
    transfer_count = positive_integer("transfer_count")
    if transfer_count != len(transfer_history):
        reject("transfer_count", "与真实转站时刻序列长度不一致")

    steps = trajectory.get("steps")
    if not isinstance(steps, list) or not steps:
        reject("steps", "必须包含非空的决策步骤列表")

    # u=2^-24覆盖可能的float32标量序列化舍入；时间容差单位为小时，
    # 标签容差单位为H0归一化残差，另加1e-12的双精度计算保护量。
    unit_roundoff = 2.0**-24
    for position, step in enumerate(steps):
        if not isinstance(step, dict):
            reject("step", "步骤必须为字段映射", position)
        step_idx = step.get("step_idx", position)
        if step.get("label_available") is not True:
            reject("label_available", "缺少真实转站监督标签", step_idx)

        raw_features = step.get("state_feat")
        if isinstance(raw_features, (list, tuple)):
            if any(
                isinstance(value, bool) or not isinstance(value, numbers.Real)
                for value in raw_features
            ):
                reject("state_feat", "必须只包含数值且不能含布尔值", step_idx)
        try:
            features = torch.as_tensor(raw_features)
        except (TypeError, ValueError, RuntimeError) as exc:
            reject("state_feat", f"无法转换为数值张量（{type(exc).__name__}）", step_idx)
        if (
            features.ndim != 1
            or features.shape[0] != 32
            or features.dtype == torch.bool
            or features.is_complex()
        ):
            reject("state_feat", "必须为恰好32维的实数向量", step_idx)
        if not bool(torch.isfinite(features.to(dtype=torch.float64)).all()):
            reject("state_feat", "不得包含NaN或Inf", step_idx)

        current_time = finite_number(step.get("current_time"), "current_time", step_idx)
        estimated_cmax = finite_number(
            step.get("estimated_cmax"), "estimated_cmax", step_idx
        )
        actual_time = finite_number(
            step.get("actual_transfer_time"), "actual_transfer_time", step_idx
        )
        label_y = finite_number(step.get("label_y"), "label_y", step_idx)

        cycle_idx_value = step.get("cycle_idx")
        if (
            isinstance(cycle_idx_value, bool)
            or not isinstance(cycle_idx_value, numbers.Integral)
            or cycle_idx_value <= 0
        ):
            reject("cycle_idx", "必须为正整数", step_idx)
        cycle_idx = int(cycle_idx_value)
        if cycle_idx > len(transfer_history):
            reject("cycle_idx", "超出真实转站历史范围", step_idx)

        cycle_time = transfer_history[cycle_idx - 1]
        time_tolerance = (
            unit_roundoff
            / (1.0 - unit_roundoff)
            * (abs(actual_time) + abs(current_time) + abs(cycle_time))
            + 1e-12
        )
        if not math.isfinite(time_tolerance):
            reject("actual_transfer_time", "时间比较容差发生数值溢出", step_idx)
        if actual_time < current_time - time_tolerance:
            reject("actual_transfer_time", "早于该决策步骤的current_time", step_idx)
        if abs(actual_time - cycle_time) > time_tolerance:
            reject(
                "actual_transfer_time",
                f"与cycle_idx={cycle_idx}对应的transfer_history时刻不一致",
                step_idx,
            )

        expected_label = (actual_time - estimated_cmax) / h0
        label_tolerance = (
            unit_roundoff * abs(label_y)
            + unit_roundoff
            / (1.0 - unit_roundoff)
            * ((abs(actual_time) + abs(estimated_cmax)) / h0 + abs(label_y))
            + 1e-12
        )
        if not math.isfinite(expected_label) or not math.isfinite(label_tolerance):
            reject("label_y", "标签公式或比较容差发生数值溢出", step_idx)
        if abs(label_y - expected_label) > label_tolerance:
            reject(
                "label_y",
                "与(actual_transfer_time-estimated_cmax)/h0不一致；"
                f"允许误差为{label_tolerance:.3g}（归一化残差单位）",
                step_idx,
            )


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
        for item in trajectories:
            _validate_official_trajectory(item, split_name)
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
