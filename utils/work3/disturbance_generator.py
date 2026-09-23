"""工作三 9 类解耦扰动场景库生成器 (Task 4.1)。

核心规范：
- 锚定稳态第 6 周期 (P_5 = 5*H_0 到 P_6 = 6*H_0)；
- 3 时机使用同一目标站位内基准开工跨度的相对位置；
- 3 强度：低 (Delta=0.15 H_0, rho=5%)、中 (Delta=0.35 H_0, rho=10%)、高 (Delta=0.60 H_0, rho=18%)；
- 空间覆盖：站位 0~4 分别对应在场飞机 5~1，合计 3 x 3 x 5 = 45 个离线确定性基准测试场景；
- 生成并保存 data/work3/scenarios_9class.json。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


TIMING_OFFSETS = {
    "EARLY": 0.225,
    "MID": 0.400,
    "LATE": 0.575,
}

INTENSITY_SPECS = {
    "LOW": {"delta_ratio": 0.15, "rho": 0.05},
    "MID": {"delta_ratio": 0.35, "rho": 0.10},
    "HIGH": {"delta_ratio": 0.60, "rho": 0.18},
}

# 稳态第 6 周期中，5 个站位对应的飞机编号 (0-based 站位 0~4 对应 飞机 5~1)
CYCLE_6_STATION_AIRCRAFT = {
    0: 5,
    1: 4,
    2: 3,
    3: 2,
    4: 1,
}


@dataclass(frozen=True)
class DisturbanceScenario:
    """单个扰动测试场景元数据。"""

    scenario_id: str
    timing: str
    intensity: str
    station_id: int
    aircraft_id: int
    tau: float
    delta: float
    recovery_time: float
    candidate_count: int
    affected_count: int
    affected_task_keys: list[str]
    timing_reference: str = "station_task_span"
    valid: bool = True
    invalid_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def select_unstarted_candidates(
    tasks: Sequence[Any],
    tau: float,
) -> tuple[list[Any], str | None]:
    """只选择扰动揭示时尚未按基准开工的工序，不用全站回退。"""
    candidates = [task for task in tasks if float(task.baseline_start) >= float(tau)]
    candidates.sort(key=lambda task: (float(task.in_station_offset), int(task.task_id)))
    if not candidates:
        return [], "no_unstarted_candidate"
    return candidates, None


def generate_9class_scenarios(
    baseline_json_path: str = "data/work3/real_283_k10_baseline.json",
    output_json_path: str = "data/work3/scenarios_9class.json",
) -> list[DisturbanceScenario]:
    """生成 9 类正交解耦场景库 (45 个确定性测试用例) 并保存至 JSON。"""
    baseline = MultiAircraftBaseline.load_from_json(baseline_json_path)
    h0 = baseline.h0
    p5 = 5.0 * h0

    scenarios: list[DisturbanceScenario] = []

    for station_id, aircraft_id in CYCLE_6_STATION_AIRCRAFT.items():
        # 收集该飞机在该站位的所有工序 (baseline station_id 为 1-based: station_id + 1)
        st_tasks = [
            t for t in baseline.tasks.values()
            if t.aircraft_id == aircraft_id and t.station_id == station_id + 1
        ]
        # 任务表已明确禁止把尚未确定的绝对时刻擅自写成 H0 下界；
        # 当前首版只固定站内相对位置，实际小时量随场景元数据记录。
        t_span = max((float(t.in_station_offset) for t in st_tasks), default=0.0)
        for timing, t_ratio in TIMING_OFFSETS.items():
            tau = round(p5 + t_ratio * t_span, 4)

            for intensity, spec in INTENSITY_SPECS.items():
                delta = round(spec["delta_ratio"] * h0, 4)
                recovery_time = round(tau + delta, 4)
                rho = spec["rho"]

                scenario_id = f"{timing}_{intensity}_S{station_id}"

                candidates, invalid_reason = select_unstarted_candidates(st_tasks, tau)

                # 确定命中数量: ceil(rho * |candidates|)，至少 1 项
                k_affected = max(1, math.ceil(rho * len(candidates))) if candidates else 0
                affected_tasks = candidates[:k_affected]
                affected_keys = [t.task_key for t in affected_tasks]

                scenario = DisturbanceScenario(
                    scenario_id=scenario_id,
                    timing=timing,
                    intensity=intensity,
                    station_id=station_id,
                    aircraft_id=aircraft_id,
                    tau=tau,
                    delta=delta,
                    recovery_time=recovery_time,
                    candidate_count=len(candidates),
                    affected_count=len(affected_keys),
                    affected_task_keys=affected_keys,
                    timing_reference="station_task_span",
                    valid=invalid_reason is None,
                    invalid_reason=invalid_reason,
                )
                scenarios.append(scenario)

    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in scenarios], f, indent=2)

    return scenarios


def split_scenarios(
    scenarios: Sequence[dict[str, Any]],
    *,
    seed: int = 2026,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
) -> dict[str, list[dict[str, Any]]]:
    """按场景种子分层划分训练、验证和测试事件，确保五站测试覆盖。"""
    if not 0.0 < train_ratio < 1.0 or not 0.0 < validation_ratio < 1.0:
        raise ValueError("训练和验证比例必须在(0,1)内")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("训练与验证比例之和必须小于1")

    grouped: dict[int, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        if scenario.get("valid", True) is False:
            raise ValueError(f"不能把无效扰动场景放入正式划分: {scenario.get('scenario_id')}")
        grouped.setdefault(int(scenario["station_id"]), []).append(dict(scenario))

    rng = random.Random(int(seed))
    result = {"train": [], "validation": [], "test": []}
    for station_id in sorted(grouped):
        station_scenarios = grouped[station_id]
        station_scenarios.sort(key=lambda item: str(item["scenario_id"]))
        rng.shuffle(station_scenarios)
        n_total = len(station_scenarios)
        n_train = max(1, int(round(n_total * train_ratio)))
        n_validation = max(1, int(round(n_total * validation_ratio)))
        if n_train + n_validation >= n_total:
            n_validation = max(1, n_total - n_train - 1)
        result["train"].extend(station_scenarios[:n_train])
        result["validation"].extend(station_scenarios[n_train : n_train + n_validation])
        result["test"].extend(station_scenarios[n_train + n_validation :])

    for key in result:
        result[key].sort(key=lambda item: str(item["scenario_id"]))
    return result


def write_experiment_splits(
    splits: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
    *,
    seed: int = 2026,
) -> dict[str, Path]:
    """将无重叠事件清单写成可复现实验输入文件。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split_name in ("train", "validation", "test"):
        path = out_dir / f"{split_name}.json"
        path.write_text(
            json.dumps(splits[split_name], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        paths[split_name] = path
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "seed": int(seed),
                "source": "scenarios_9class.json",
                "split_names": ["train", "validation", "test"],
                "scenario_counts": {name: len(items) for name, items in splits.items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    paths["metadata"] = metadata_path
    return paths


if __name__ == "__main__":
    generated = generate_9class_scenarios()
    print(f"成功生成 {len(generated)} 个 9 类解耦扰动基准场景至 data/work3/scenarios_9class.json")
