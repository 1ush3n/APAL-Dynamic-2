"""工作三 9 类解耦扰动场景库生成器 (Task 4.1)。

核心规范：
- 锚定稳态第 6 周期 (P_5 = 5*H_0 到 P_6 = 6*H_0)；
- 3 时机：早 (0.225 H_0)、中 (0.400 H_0)、晚 (0.575 H_0)；
- 3 强度：低 (Delta=0.15 H_0, rho=5%)、中 (Delta=0.35 H_0, rho=10%)、高 (Delta=0.60 H_0, rho=18%)；
- 空间覆盖：站位 0~4 分别对应在场飞机 5~1，合计 3 x 3 x 5 = 45 个离线确定性基准测试场景；
- 生成并保存 data/work3/scenarios_9class.json。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

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

    def to_dict(self) -> dict:
        return asdict(self)


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
        # 计算该站位工序开工时间跨度 T_span
        t_span = max(t.in_station_offset for t in st_tasks)

        for timing, t_ratio in TIMING_OFFSETS.items():
            tau = round(p5 + t_ratio * t_span, 4)

            for intensity, spec in INTENSITY_SPECS.items():
                delta = round(spec["delta_ratio"] * h0, 4)
                recovery_time = round(tau + delta, 4)
                rho = spec["rho"]

                scenario_id = f"{timing}_{intensity}_S{station_id}"

                # 候选池：在 tau 时刻基准计划中尚未开工的任务 (基准开工时刻 baseline_start >= tau)
                candidates = [t for t in st_tasks if t.baseline_start >= tau]
                if not candidates:
                    candidates = list(st_tasks)

                # 按站内偏移与工序 ID 确定性排序
                candidates.sort(key=lambda t: (t.in_station_offset, t.task_id))

                # 确定命中数量: ceil(rho * |candidates|)，至少 1 项
                k_affected = max(1, math.ceil(rho * len(candidates)))
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
                )
                scenarios.append(scenario)

    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in scenarios], f, indent=2)

    return scenarios


if __name__ == "__main__":
    generated = generate_9class_scenarios()
    print(f"成功生成 {len(generated)} 个 9 类解耦扰动基准场景至 data/work3/scenarios_9class.json")
