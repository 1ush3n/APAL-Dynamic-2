"""工作三 9 类解耦扰动场景库生成器 (Task 4.1)。

核心规范：
- 默认锚定稳态第 6 周期 (P_5 = 5*H_0 到 P_6 = 6*H_0)；cycle_offset只改变基准周期和对应飞机编号；
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
    intended_rho: float = 0.0
    intended_count: int = 0
    actual_hit_count: int = 0
    actual_fraction: float = 0.0
    delta_h0_ratio: float = 0.0
    seed: int | None = None
    degeneracy_reason: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["event_fingerprint"] = list(compute_event_fingerprint(data))
        data["target_group_key"] = list(compute_target_group_key(data))
        return data


def compute_event_fingerprint(
    scenario: dict[str, Any] | DisturbanceScenario,
) -> tuple[int, tuple[str, ...], float, float]:
    """计算完整扰动事件指纹 (aircraft_id, sorted(affected_task_keys), tau, recovery_time)。"""
    if isinstance(scenario, DisturbanceScenario):
        aircraft_id = int(scenario.aircraft_id)
        keys = tuple(sorted(str(k) for k in scenario.affected_task_keys))
        tau = round(float(scenario.tau), 4)
        recovery = round(float(scenario.recovery_time), 4)
    else:
        aircraft_id = int(scenario["aircraft_id"])
        keys = tuple(sorted(str(k) for k in scenario["affected_task_keys"]))
        tau = round(float(scenario["tau"]), 4)
        recovery = round(float(scenario["recovery_time"]), 4)
    return (aircraft_id, keys, tau, recovery)


def compute_target_group_key(
    scenario: dict[str, Any] | DisturbanceScenario,
) -> tuple[int, int, tuple[str, ...]]:
    """计算受扰目标工序组键 (station_id, aircraft_id, sorted(affected_task_keys))。"""
    if isinstance(scenario, DisturbanceScenario):
        station_id = int(scenario.station_id)
        aircraft_id = int(scenario.aircraft_id)
        keys = tuple(sorted(str(k) for k in scenario.affected_task_keys))
    else:
        station_id = int(scenario["station_id"])
        aircraft_id = int(scenario["aircraft_id"])
        keys = tuple(sorted(str(k) for k in scenario["affected_task_keys"]))
    return (station_id, aircraft_id, keys)


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


def _detect_intensity_degeneracy(candidate_count: int, rho: float) -> str | None:
    if candidate_count <= 0:
        return "empty_candidates"
    low_c = max(1, math.ceil(INTENSITY_SPECS["LOW"]["rho"] * candidate_count))
    mid_c = max(1, math.ceil(INTENSITY_SPECS["MID"]["rho"] * candidate_count))
    high_c = max(1, math.ceil(INTENSITY_SPECS["HIGH"]["rho"] * candidate_count))
    if len({low_c, mid_c, high_c}) < 3:
        return "discrete_candidate_saturation"
    return None


def summarize_scenario_diagnostics(
    scenarios: Sequence[dict[str, Any] | DisturbanceScenario],
) -> dict[str, Any]:
    """诚实统计每个场景及 (station, intensity) 的候选数、目标命中数、实际命中数、实际比例与退化原因。"""
    rows: list[dict[str, Any]] = []
    by_station_intensity: dict[str, dict[str, Any]] = {}

    for item in scenarios:
        d = item.to_dict() if isinstance(item, DisturbanceScenario) else dict(item)
        cand_count = int(d.get("candidate_count", 0))
        affected_keys = [str(k) for k in d.get("affected_task_keys", [])]
        actual_hit = len(affected_keys)
        intensity = str(d.get("intensity", "LOW"))
        rho = float(d.get("intended_rho") or INTENSITY_SPECS.get(intensity, {}).get("rho", 0.0))
        intended_count = (
            int(d["intended_count"])
            if d.get("intended_count") is not None and int(d.get("intended_count", 0)) > 0
            else (max(1, math.ceil(rho * cand_count)) if cand_count > 0 else 0)
        )
        actual_fraction = round(actual_hit / cand_count, 6) if cand_count > 0 else 0.0
        delta_h0_ratio = float(
            d.get("delta_h0_ratio")
            or INTENSITY_SPECS.get(intensity, {}).get("delta_ratio", 0.0)
        )
        degeneracy = d.get("degeneracy_reason") or _detect_intensity_degeneracy(cand_count, rho)
        row = {
            "scenario_id": str(d["scenario_id"]),
            "station_id": int(d["station_id"]),
            "aircraft_id": int(d["aircraft_id"]),
            "timing": str(d["timing"]),
            "intensity": intensity,
            "candidate_count": cand_count,
            "intended_rho": rho,
            "intended_count": intended_count,
            "actual_hit_count": actual_hit,
            "actual_fraction": actual_fraction,
            "delta_h0_ratio": round(delta_h0_ratio, 4),
            "degeneracy_reason": degeneracy,
            "event_fingerprint": compute_event_fingerprint(d),
            "target_group_key": compute_target_group_key(d),
        }
        rows.append(row)
        group_label = f"S{row['station_id']}_{intensity}"
        by_station_intensity[group_label] = {
            "station_id": row["station_id"],
            "intensity": intensity,
            "candidate_count": cand_count,
            "intended_count": intended_count,
            "actual_hit_count": actual_hit,
            "actual_fraction": actual_fraction,
            "delta_h0_ratio": row["delta_h0_ratio"],
            "degeneracy_reason": degeneracy,
        }

    return {
        "total_scenarios": len(rows),
        "unique_event_fingerprints": len({r["event_fingerprint"] for r in rows}),
        "unique_target_groups": len({r["target_group_key"] for r in rows}),
        "rows": rows,
        "by_station_intensity": by_station_intensity,
    }


def generate_9class_scenarios(
    baseline_json_path: str = "data/work3/real_283_k10_baseline.json",
    output_json_path: str | Path | None = None,
    *,
    cycle_offset: int = 0,
) -> list[DisturbanceScenario]:
    """生成九类×五站确定性场景；cycle_offset可选后续脉动周期作独立清单。"""
    if isinstance(cycle_offset, bool) or not isinstance(cycle_offset, int):
        raise TypeError("cycle_offset必须为整数")
    if cycle_offset < 0:
        raise ValueError("cycle_offset必须为非负整数")

    baseline = MultiAircraftBaseline.load_from_json(baseline_json_path)
    h0 = baseline.h0
    cycle_start = (5.0 + cycle_offset) * h0

    scenarios: list[DisturbanceScenario] = []

    for station_id, cycle_6_aircraft_id in CYCLE_6_STATION_AIRCRAFT.items():
        aircraft_id = cycle_6_aircraft_id + cycle_offset
        # 收集该飞机在该站位的所有工序 (baseline station_id 为 1-based: station_id + 1)
        st_tasks = [
            t for t in baseline.tasks.values()
            if t.aircraft_id == aircraft_id and t.station_id == station_id + 1
        ]
        if not st_tasks:
            raise ValueError(
                f"cycle_offset={cycle_offset}时站位S{station_id}没有飞机{aircraft_id}的基准工序"
            )
        # 任务表已明确禁止把尚未确定的绝对时刻擅自写成 H0 下界；
        # 当前首版只固定站内相对位置，实际小时量随场景元数据记录。
        t_span = max((float(t.in_station_offset) for t in st_tasks), default=0.0)
        for timing, t_ratio in TIMING_OFFSETS.items():
            tau = round(cycle_start + t_ratio * t_span, 4)

            for intensity, spec in INTENSITY_SPECS.items():
                delta_ratio = float(spec["delta_ratio"])
                delta = round(delta_ratio * h0, 4)
                recovery_time = round(tau + delta, 4)
                rho = float(spec["rho"])

                scenario_id = f"{timing}_{intensity}_S{station_id}"
                if cycle_offset:
                    scenario_id += f"_CYCLE{6 + cycle_offset}"

                candidates, invalid_reason = select_unstarted_candidates(st_tasks, tau)

                # 确定命中数量: ceil(rho * |candidates|)，至少 1 项
                k_affected = max(1, math.ceil(rho * len(candidates))) if candidates else 0
                affected_tasks = candidates[:k_affected]
                affected_keys = [t.task_key for t in affected_tasks]
                cand_count = len(candidates)
                actual_hit = len(affected_keys)
                actual_frac = round(actual_hit / cand_count, 6) if cand_count > 0 else 0.0
                degeneracy = _detect_intensity_degeneracy(cand_count, rho)

                scenario = DisturbanceScenario(
                    scenario_id=scenario_id,
                    timing=timing,
                    intensity=intensity,
                    station_id=station_id,
                    aircraft_id=aircraft_id,
                    tau=tau,
                    delta=delta,
                    recovery_time=recovery_time,
                    candidate_count=cand_count,
                    affected_count=actual_hit,
                    affected_task_keys=affected_keys,
                    timing_reference="station_task_span",
                    valid=invalid_reason is None,
                    invalid_reason=invalid_reason,
                    intended_rho=rho,
                    intended_count=k_affected,
                    actual_hit_count=actual_hit,
                    actual_fraction=actual_frac,
                    delta_h0_ratio=round(delta_ratio, 4),
                    seed=None,
                    degeneracy_reason=degeneracy,
                )
                scenarios.append(scenario)

    default_name = (
        "scenarios_9class.json"
        if cycle_offset == 0
        else f"scenarios_9class_cycle{6 + cycle_offset}.json"
    )
    out_path = Path(
        output_json_path
        if output_json_path is not None
        else Path("data/work3") / default_name
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in scenarios], f, indent=2)

    return scenarios


def generate_candidate_scenario_pool(
    baseline_json_path: str = "data/work3/real_283_k10_baseline.json",
    *,
    seeds: Sequence[int] = (2026, 2027, 2028, 2029, 2030, 2031),
) -> list[dict[str, Any]]:
    """构建带显式种子与完整事件指纹的候选扰动池，支持按无重叠目标工序组拆分 train/val/test。"""
    baseline = MultiAircraftBaseline.load_from_json(baseline_json_path)
    h0 = float(baseline.h0)
    p5 = 5.0 * h0

    seen_fingerprints: set[tuple[int, tuple[str, ...], float, float]] = set()
    pool: list[dict[str, Any]] = []
    seed_list = [int(s) for s in seeds]
    num_slots = max(3, len(seed_list))

    for station_id, base_aircraft_id in CYCLE_6_STATION_AIRCRAFT.items():
        for slot_idx, seed_val in enumerate(seed_list):
            rng = random.Random(seed_val * 31 + station_id * 17 + slot_idx)
            # 先取第 6 周期该站位工序；若该站单架飞机工序数少于 3，则按稳态周期 (6, 7, 8) 轮换同站位不同在制飞机以保证目标工序组完全不重叠
            base_tasks = [
                t for t in baseline.tasks.values()
                if t.aircraft_id == base_aircraft_id and t.station_id == station_id + 1
            ]
            if len(base_tasks) < 3:
                cycle_shift = slot_idx % 3
                aircraft_id = base_aircraft_id + cycle_shift
                p_cycle = (5.0 + float(cycle_shift)) * h0
            else:
                aircraft_id = base_aircraft_id
                p_cycle = p5

            st_tasks = [
                t for t in baseline.tasks.values()
                if t.aircraft_id == aircraft_id and t.station_id == station_id + 1
            ]
            st_tasks.sort(key=lambda task: (float(task.in_station_offset), int(task.task_id)))
            t_span = max((float(t.in_station_offset) for t in st_tasks), default=0.0)

            for timing, t_ratio in TIMING_OFFSETS.items():
                tau = round(p_cycle + t_ratio * t_span, 4)
                candidates, invalid_reason = select_unstarted_candidates(st_tasks, tau)
                # 若站尾长尾任务导致固定跨度比例下未开工候选不足，按站内未开工分位点校准 tau，确保仍满足 baseline_start >= tau
                if len(candidates) < min(num_slots, len(st_tasks)) and len(st_tasks) > 0:
                    target_avail = min(num_slots, len(st_tasks))
                    quantile_idx = min(
                        max(0, int(round(t_ratio * max(1, len(st_tasks) - target_avail)))),
                        len(st_tasks) - target_avail,
                    )
                    tau = round(float(st_tasks[quantile_idx].baseline_start), 4)
                    candidates, invalid_reason = select_unstarted_candidates(st_tasks, tau)

                if invalid_reason is not None or not candidates:
                    continue

                # 将候选集按 slot 分桶以构造互不重叠的目标工序子集
                strided_subset = candidates[slot_idx::num_slots]
                if not strided_subset:
                    strided_subset = [candidates[slot_idx % len(candidates)]]

                for intensity, spec in INTENSITY_SPECS.items():
                    delta_ratio = float(spec["delta_ratio"])
                    delta = round(delta_ratio * h0, 4)
                    recovery_time = round(tau + delta, 4)
                    rho = float(spec["rho"])

                    cand_count = len(candidates)
                    intended_k = max(1, math.ceil(rho * cand_count))
                    k_slot = max(1, min(len(strided_subset), math.ceil(rho * len(strided_subset))))
                    chosen = list(strided_subset[:k_slot])
                    if len(strided_subset) > k_slot:
                        chosen = rng.sample(strided_subset, k_slot)
                    affected_keys = sorted(str(t.task_key) for t in chosen)
                    actual_hit = len(affected_keys)
                    actual_frac = round(actual_hit / cand_count, 6) if cand_count > 0 else 0.0
                    degeneracy = _detect_intensity_degeneracy(cand_count, rho)

                    scenario_obj = DisturbanceScenario(
                        scenario_id=f"{timing}_{intensity}_S{station_id}_SEED{seed_val}",
                        timing=timing,
                        intensity=intensity,
                        station_id=station_id,
                        aircraft_id=aircraft_id,
                        tau=tau,
                        delta=delta,
                        recovery_time=recovery_time,
                        candidate_count=cand_count,
                        affected_count=actual_hit,
                        affected_task_keys=affected_keys,
                        timing_reference="station_task_span",
                        valid=True,
                        invalid_reason=None,
                        intended_rho=rho,
                        intended_count=intended_k,
                        actual_hit_count=actual_hit,
                        actual_fraction=actual_frac,
                        delta_h0_ratio=round(delta_ratio, 4),
                        seed=seed_val,
                        degeneracy_reason=degeneracy,
                    )
                    scenario_dict = scenario_obj.to_dict()
                    fp = compute_event_fingerprint(scenario_dict)
                    if fp not in seen_fingerprints:
                        seen_fingerprints.add(fp)
                        pool.append(scenario_dict)

    return pool


def split_scenarios(
    scenarios: Sequence[dict[str, Any]],
    *,
    seed: int = 2026,
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    strict_group_isolation: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """按事件指纹（及可选的目标工序组）严格零重叠分层划分训练、验证和测试事件，确保五站测试覆盖。"""
    if not 0.0 < train_ratio < 1.0 or not 0.0 < validation_ratio < 1.0:
        raise ValueError("训练和验证比例必须在(0,1)内")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("训练与验证比例之和必须小于1")

    # 第一步：按完整事件指纹去重，杜绝相同事件指纹进入多个子集
    deduped: list[dict[str, Any]] = []
    seen_fp: set[tuple[int, tuple[str, ...], float, float]] = set()
    for scenario in scenarios:
        if scenario.get("valid", True) is False:
            raise ValueError(f"不能把无效扰动场景放入正式划分: {scenario.get('scenario_id')}")
        fp = compute_event_fingerprint(scenario)
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        deduped.append(dict(scenario))

    # 第二步：在每个站位内部按隔离单元（target_group_key 或 event_fingerprint）整组划分
    grouped_units: dict[int, dict[Any, list[dict[str, Any]]]] = {}
    for item in deduped:
        station_id = int(item["station_id"])
        unit_key = (
            compute_target_group_key(item)
            if strict_group_isolation
            else compute_event_fingerprint(item)
        )
        grouped_units.setdefault(station_id, {}).setdefault(unit_key, []).append(item)

    rng = random.Random(int(seed))
    result = {"train": [], "validation": [], "test": []}
    for station_id in sorted(grouped_units):
        unit_entries = list(grouped_units[station_id].items())
        unit_entries.sort(key=lambda pair: str(pair[0]))
        rng.shuffle(unit_entries)
        n_units = len(unit_entries)
        if n_units < 3:
            raise ValueError(
                f"站位 S{station_id} 的独立扰动单元数量 ({n_units}) 少于 3，无法进行零重叠 train/validation/test 划分"
            )
        n_train = max(1, int(round(n_units * train_ratio)))
        n_validation = max(1, int(round(n_units * validation_ratio)))
        if n_train + n_validation >= n_units:
            n_train = max(1, n_units - 2)
            n_validation = 1

        for _, unit_items in unit_entries[:n_train]:
            result["train"].extend(unit_items)
        for _, unit_items in unit_entries[n_train : n_train + n_validation]:
            result["validation"].extend(unit_items)
        for _, unit_items in unit_entries[n_train + n_validation :]:
            result["test"].extend(unit_items)

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
                "unique_fingerprints": {
                    name: len({compute_event_fingerprint(s) for s in items})
                    for name, items in splits.items()
                },
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
