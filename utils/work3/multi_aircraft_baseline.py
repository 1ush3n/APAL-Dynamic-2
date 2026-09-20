"""多架次飞机脉动装配流水基准排程生成与展开器。

本模块依据工作三顶层设计规范，读取单机无扰动基准排程模板（如 real_283_schedule.csv / real_283_baseline.csv），
将其按照 K 架次（默认 K=10）周期性脉动投产规则展开为全线多机协同基准排程。

物理与数学定义：
1. 飞机编号 k in [0, K-1]，名义投产时刻 T_k^launch = k * H_0；
2. 基础站位 m_{ki}^0 = m_i^0 in {1, 2, 3, 4, 5}；
3. 任务唯一身份定义为二元组 (k, i)，对应工序全局唯一标识；
4. 基准周期编号 q = k + m_i^0 in [1, K + M - 1]（对于 K=10, M=5，共 14 个周期）；
5. 飞机 k 进入站位 m_i^0 的名义周期起点为 (k + m_i^0 - 1) * H_0；
6. 任务基准开工时间 S_{ki}^0 = (k + m_i^0 - 1) * H_0 + b_i^0，其中 b_i^0 为单机模板站内偏移；
7. 任务基准完工时间 C_{ki}^0 = S_{ki}^0 + p_i；
8. 基准指派团队 W_{ki}^0 = W_i^0（继承模板固定工人）。
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingleAircraftTask:
    """单机模板中的工序元数据。

    Attributes:
        task_id: 内部工序编号 (0-based 索引, 0 ~ 289).
        ao_code: 工艺规程 AO 号.
        station_id: 归属物理站位 (1-based: 1 ~ 5).
        team: 指派的固定工人 ID 列表.
        duration: 标准工时 p_i (小时).
        in_station_offset: 周期内开工偏移量 b_i^0 (小时).
        demand: 需求人数.
        skill: 所需工种技能类型 (0 ~ 4, dummy为 -1).
        predecessors: 工艺直接紧前工序列表.
        is_physical: 是否为物理加工工序 (工期 > 0).
    """

    task_id: int
    ao_code: str
    station_id: int
    team: tuple[int, ...]
    duration: float
    in_station_offset: float
    demand: int
    skill: int
    predecessors: tuple[int, ...]
    is_physical: bool


@dataclass(frozen=True)
class MultiAircraftTask:
    """多架次展开后的工序执行实例。

    Attributes:
        aircraft_id: 飞机编号 k in [0, K-1].
        task_id: 内部工序编号 i.
        task_key: 全局唯一标识字符串 f"{aircraft_id}_{task_id}".
        station_id: 基础执行站位 m_{ki}^0 in {1, ..., 5}.
        team: 基准指派团队 W_{ki}^0.
        duration: 标准加工时间 p_i.
        in_station_offset: 周期内开工相对偏移 b_i^0.
        baseline_start: 基准全局绝对开工时刻 S_{ki}^0.
        baseline_end: 基准全局绝对完工时刻 C_{ki}^0.
        cycle_idx: 归属的名义脉动周期 q in [1, 14].
        nominal_station_entry: 本架飞机进入该站的名义时刻.
        nominal_station_exit: 本架飞机离开该站的名义时刻.
        demand: 需求人数.
        skill: 所需技能.
        ao_code: AO 编码.
        predecessors: 本机内部的前驱工序列表.
    """

    aircraft_id: int
    task_id: int
    task_key: str
    station_id: int
    team: tuple[int, ...]
    duration: float
    in_station_offset: float
    baseline_start: float
    baseline_end: float
    cycle_idx: int
    nominal_station_entry: float
    nominal_station_exit: float
    demand: int
    skill: int
    ao_code: str
    predecessors: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return asdict(self)


class MultiAircraftBaseline:
    """多架次基准计划容器与查询接口。"""

    def __init__(
        self,
        *,
        num_aircraft: int,
        num_stations: int,
        h0: float,
        tasks: dict[str, MultiAircraftTask],
        station_workers: dict[int, list[int]],
    ) -> None:
        self.num_aircraft = int(num_aircraft)
        self.num_stations = int(num_stations)
        self.h0 = float(h0)
        self.total_cycles = self.num_aircraft + self.num_stations - 1
        self.tasks = tasks
        self.station_workers = {int(k): list(v) for k, v in station_workers.items()}

        # 快速索引缓存
        self._tasks_by_aircraft: dict[int, list[MultiAircraftTask]] = {}
        self._tasks_by_cycle: dict[int, list[MultiAircraftTask]] = {}
        self._tasks_by_station_and_cycle: dict[tuple[int, int], list[MultiAircraftTask]] = {}

        for task in self.tasks.values():
            self._tasks_by_aircraft.setdefault(task.aircraft_id, []).append(task)
            self._tasks_by_cycle.setdefault(task.cycle_idx, []).append(task)
            self._tasks_by_station_and_cycle.setdefault(
                (task.station_id, task.cycle_idx), []
            ).append(task)

    @property
    def total_tasks_count(self) -> int:
        """展开后的物理工序总数。"""
        return len(self.tasks)

    @property
    def physical_tasks_per_aircraft(self) -> int:
        """单架飞机包含的物理工序数。"""
        return len(self._tasks_by_aircraft.get(0, []))

    def get_task(self, aircraft_id: int, task_id: int) -> MultiAircraftTask:
        """根据 (k, i) 二元组获取任务。"""
        key = f"{aircraft_id}_{task_id}"
        if key not in self.tasks:
            raise KeyError(f"未找到工序: aircraft_id={aircraft_id}, task_id={task_id}")
        return self.tasks[key]

    def get_tasks_for_aircraft(self, aircraft_id: int) -> list[MultiAircraftTask]:
        """获取指定飞机的全部工序。"""
        return self._tasks_by_aircraft.get(aircraft_id, [])

    def get_tasks_for_cycle(self, cycle_idx: int) -> list[MultiAircraftTask]:
        """获取指定脉动周期的全部工序。"""
        return self._tasks_by_cycle.get(cycle_idx, [])

    def get_tasks_for_station_and_cycle(
        self, station_id: int, cycle_idx: int
    ) -> list[MultiAircraftTask]:
        """获取指定站位在指定脉动周期的所有工序。"""
        return self._tasks_by_station_and_cycle.get((station_id, cycle_idx), [])

    def get_workers_for_station(self, station_id: int) -> list[int]:
        """获取指定站位绑定的工人列表。"""
        return self.station_workers.get(station_id, [])

    def to_json_dict(self) -> dict[str, Any]:
        """转换为可导出的 JSON 字典。"""
        return {
            "metadata": {
                "version": "work3_k10_baseline_v1",
                "num_aircraft": self.num_aircraft,
                "num_stations": self.num_stations,
                "h0": self.h0,
                "total_cycles": self.total_cycles,
                "physical_tasks_per_aircraft": self.physical_tasks_per_aircraft,
                "total_physical_tasks": self.total_tasks_count,
            },
            "station_workers": self.station_workers,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
        }

    def save_to_json(self, output_path: str | Path) -> None:
        """将多架次基准计划落盘为 JSON 文件。"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"多架次基线计划已保存至: {path.resolve()}")

    @classmethod
    def load_from_json(cls, json_path: str | Path) -> "MultiAircraftBaseline":
        """从 JSON 文件还原多架次基准计划。"""
        path = Path(json_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data["metadata"]
        station_workers = {int(k): v for k, v in data["station_workers"].items()}
        tasks: dict[str, MultiAircraftTask] = {}
        for key, d in data["tasks"].items():
            tasks[key] = MultiAircraftTask(
                aircraft_id=int(d["aircraft_id"]),
                task_id=int(d["task_id"]),
                task_key=str(d["task_key"]),
                station_id=int(d["station_id"]),
                team=tuple(int(w) for w in d["team"]),
                duration=float(d["duration"]),
                in_station_offset=float(d["in_station_offset"]),
                baseline_start=float(d["baseline_start"]),
                baseline_end=float(d["baseline_end"]),
                cycle_idx=int(d["cycle_idx"]),
                nominal_station_entry=float(d["nominal_station_entry"]),
                nominal_station_exit=float(d["nominal_station_exit"]),
                demand=int(d["demand"]),
                skill=int(d["skill"]),
                ao_code=str(d["ao_code"]),
                predecessors=tuple(int(p) for p in d["predecessors"]),
            )

        return cls(
            num_aircraft=int(meta["num_aircraft"]),
            num_stations=int(meta["num_stations"]),
            h0=float(meta["h0"]),
            tasks=tasks,
            station_workers=station_workers,
        )


def _parse_team(raw_team: Any) -> tuple[int, ...]:
    """解析团队字符串或列表为整数元组。"""
    if isinstance(raw_team, (list, tuple)):
        return tuple(int(x) for x in raw_team)
    if isinstance(raw_team, str):
        cleaned = raw_team.strip()
        if not cleaned or cleaned == "[]":
            return ()
        try:
            val = ast.literal_eval(cleaned)
            if isinstance(val, (list, tuple)):
                return tuple(int(x) for x in val)
        except Exception:
            # 兼容 "[50, 51]" 或 "50 51"
            cleaned = cleaned.strip("[]")
            parts = [p.strip() for p in cleaned.replace(",", " ").split() if p.strip()]
            return tuple(int(p) for p in parts)
    return ()


def load_single_aircraft_template(
    raw_data_path: str | Path,
    schedule_csv_path: str | Path,
) -> tuple[list[SingleAircraftTask], float, dict[int, list[int]]]:
    """读取原始工序定义与基准排程，解析出单机标准模板。

    Args:
        raw_data_path: 原始工艺定义 CSV (如 data/283.csv).
        schedule_csv_path: 单机可行基准排程 CSV (如 real_283_schedule.csv).

    Returns:
        tasks: 单机模板物理工序列表 (过滤掉工期为 0 的虚拟层级节点).
        h0: 基准单机节拍 / 完工时间.
        station_workers: 各站位绑定的固定工人列表.
    """
    raw_path = Path(raw_data_path)
    sched_path = Path(schedule_csv_path)

    if not raw_path.is_file():
        raise FileNotFoundError(f"原始工艺数据文件不存在: {raw_path}")
    if not sched_path.is_file():
        raise FileNotFoundError(f"基准排程文件不存在: {sched_path}")

    # 1. 加载原始工艺数据以提取 AO号、工种技能、需求人数、前驱关系
    df_raw = pd.read_csv(raw_path)
    # 建立内部索引 (0-based) 与行号的对应
    num_tasks = len(df_raw)

    # 解析前驱映射 (内部 0-based 索引)
    ao_to_tid = {str(df_raw.iloc[i]["AO号"]).strip(): i for i in range(num_tasks)}
    predecessors_map: dict[int, list[int]] = {i: [] for i in range(num_tasks)}

    col_preds = [c for c in df_raw.columns if "紧前工序" in str(c) or "predecessor" in str(c).lower()]
    pred_col_name = col_preds[0] if col_preds else None

    if pred_col_name is not None:
        for tid in range(num_tasks):
            val = df_raw.iloc[tid][pred_col_name]
            if pd.isna(val):
                continue
            for token in str(val).replace("，", ",").replace(";", ",").split(","):
                clean_tok = token.strip()
                if clean_tok and clean_tok in ao_to_tid:
                    predecessors_map[tid].append(ao_to_tid[clean_tok])

    # 2. 加载基准排程 CSV
    df_sched = pd.read_csv(sched_path)

    # 兼容 TaskID 是否包含 0-based 索引
    task_id_col = "TaskID"
    station_id_col = "StationID"

    # 计算基准 makespan H_0
    h0 = float(df_sched["End"].max())

    template_tasks: list[SingleAircraftTask] = []
    station_workers_map: dict[int, set[int]] = {s: set() for s in range(1, 6)}

    for _, row in df_sched.iterrows():
        tid = int(row[task_id_col])
        if tid < 0 or tid >= num_tasks:
            continue

        duration = float(row["Duration"])
        start = float(row["Start"])
        station_id = int(row[station_id_col])
        team = _parse_team(row["Team"])

        # 提取工艺属性
        ao_code = str(df_raw.iloc[tid].get("AO号", f"TASK_{tid}"))
        demand = int(df_raw.iloc[tid].get("需求人数", len(team)))
        skill = int(df_raw.iloc[tid].get("工种", -1))
        preds = tuple(sorted(predecessors_map[tid]))

        is_physical = duration > 1e-5

        # 仅对物理加工工序记录站位工人绑定
        if is_physical and station_id in station_workers_map:
            for w in team:
                station_workers_map[station_id].add(w)

        # 记录单机模板工序
        if is_physical:
            task = SingleAircraftTask(
                task_id=tid,
                ao_code=ao_code,
                station_id=station_id,
                team=team,
                duration=duration,
                in_station_offset=start,  # 站内偏移量 b_i^0
                demand=demand,
                skill=skill,
                predecessors=preds,
                is_physical=True,
            )
            template_tasks.append(task)

    sorted_workers = {s: sorted(station_workers_map[s]) for s in range(1, 6)}
    logger.info(
        f"成功加载单机模板: 共 {len(template_tasks)} 道物理工序, H0={h0:.4f}, "
        f"各站工人数={[len(sorted_workers[s]) for s in range(1, 6)]}"
    )

    return template_tasks, h0, sorted_workers


def expand_to_multi_aircraft_baseline(
    raw_data_path: str | Path = "data/283.csv",
    schedule_csv_path: str | Path = "data/real/real_283_baseline.csv",
    num_aircraft: int = 10,
    num_stations: int = 5,
) -> MultiAircraftBaseline:
    """将单机模板按脉动节拍展开为 K 架次基线排程。

    Args:
        raw_data_path: 原始工艺 CSV 路径.
        schedule_csv_path: 基准单机排程 CSV 路径 (支持 real_283_baseline.csv 或 real_283_schedule.csv).
        num_aircraft: 飞机架次总数 K (默认 10).
        num_stations: 物理站位总数 M (默认 5).

    Returns:
        展开后的多架次基准计划 MultiAircraftBaseline 实例.
    """
    # 路径回退机制：优先使用传入路径，若不存在则尝试默认备选
    sched_path = Path(schedule_csv_path)
    if not sched_path.is_file():
        alt_path = Path("data/r5_task_delay_v1/baselines/real/real_283_schedule.csv")
        if alt_path.is_file():
            sched_path = alt_path
        else:
            raise FileNotFoundError(f"找不到可用的基准排程文件: {schedule_csv_path}")

    template_tasks, h0, station_workers = load_single_aircraft_template(
        raw_data_path=raw_data_path,
        schedule_csv_path=sched_path,
    )

    multi_tasks: dict[str, MultiAircraftTask] = {}

    for k in range(num_aircraft):
        launch_time = k * h0
        for task in template_tasks:
            station_id = task.station_id
            # 站位名义进站与出站时刻
            nominal_entry = (k + station_id - 1) * h0
            nominal_exit = (k + station_id) * h0

            # 周期序号 q in [1, 14]
            cycle_idx = k + station_id

            # 基准开工与完工时刻
            baseline_start = nominal_entry + task.in_station_offset
            baseline_end = baseline_start + task.duration

            task_key = f"{k}_{task.task_id}"

            multi_task = MultiAircraftTask(
                aircraft_id=k,
                task_id=task.task_id,
                task_key=task_key,
                station_id=station_id,
                team=task.team,
                duration=task.duration,
                in_station_offset=task.in_station_offset,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                cycle_idx=cycle_idx,
                nominal_station_entry=nominal_entry,
                nominal_station_exit=nominal_exit,
                demand=task.demand,
                skill=task.skill,
                ao_code=task.ao_code,
                predecessors=task.predecessors,
            )
            multi_tasks[task_key] = multi_task

    baseline = MultiAircraftBaseline(
        num_aircraft=num_aircraft,
        num_stations=num_stations,
        h0=h0,
        tasks=multi_tasks,
        station_workers=station_workers,
    )

    logger.info(
        f"多架次基准计划展开完毕: K={num_aircraft}, 总工序数={baseline.total_tasks_count}, "
        f"总脉动周期数={baseline.total_cycles}, 总跨度={baseline.total_cycles * h0:.2f}h"
    )

    return baseline


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output_file = Path("data/work3/real_283_k10_baseline.json")
    baseline = expand_to_multi_aircraft_baseline(
        raw_data_path="data/283.csv",
        schedule_csv_path="data/real/real_283_baseline.csv",
        num_aircraft=10,
        num_stations=5,
    )
    baseline.save_to_json(output_file)
    print(f"[SUCCESS] Multi-aircraft baseline generated: {output_file}, total tasks: {baseline.total_tasks_count}")
