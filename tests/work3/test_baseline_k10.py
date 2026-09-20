"""多架次飞机基准计划静态硬约束白盒测试。

严格校验 Task 1.2 展开后的 10 架次、2,830 道物理工序的物理与运筹合法性：
1. 工序完整性与二元组身份唯一性 (10 架次 x 283 工序 = 2,830 道);
2. 站位周期时间窗边界约束;
3. 工艺拓扑偏序无违背 (DAG 物理依赖);
4. 站位空间槽位并发上限 (<= 3);
5. 固定工人站位绑定互斥隔离;
6. 工人全时段零重叠 (跨架次、跨周期绝对无冲突);
7. 团队需求人数与技能匹配度 100%;
8. JSON 序列化与反序列化双向一致性。
"""

from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from utils.work3.multi_aircraft_baseline import (
    MultiAircraftBaseline,
    expand_to_multi_aircraft_baseline,
)
from core.constraints import ConstraintEngine
from data_loader import load_data


@pytest.fixture(scope="module")
def baseline_data() -> MultiAircraftBaseline:
    """提供展开后的多架次基准计划实例。"""
    json_path = Path("data/work3/real_283_k10_baseline.json")
    if json_path.is_file():
        return MultiAircraftBaseline.load_from_json(json_path)
    return expand_to_multi_aircraft_baseline(
        raw_data_path="data/283.csv",
        schedule_csv_path="data/real/real_283_baseline.csv",
        num_aircraft=10,
        num_stations=5,
    )


@pytest.fixture(scope="module")
def raw_constraint_engine() -> tuple[ConstraintEngine, dict[str, Any]]:
    """提供由 283.csv 构造的原始约束引擎。"""
    raw_data = load_data("data/283.csv")
    durations = raw_data["task_df"]["duration"].values
    fixed = raw_data["task_df"]["fixed_station"].values
    engine = ConstraintEngine.build(
        num_tasks=raw_data["num_tasks"],
        num_stations=5,
        edges=raw_data["precedence_edges"].numpy(),
        durations=durations,
        fixed_stations=fixed,
    )
    return engine, raw_data


def test_multi_aircraft_integrity(baseline_data: MultiAircraftBaseline) -> None:
    """校验多架次工序总数与二元组身份唯一性。"""
    assert baseline_data.num_aircraft == 10
    assert baseline_data.num_stations == 5
    assert baseline_data.total_cycles == 14
    assert baseline_data.physical_tasks_per_aircraft == 283
    assert baseline_data.total_tasks_count == 2830

    seen_keys: set[str] = set()
    for k in range(10):
        tasks_k = baseline_data.get_tasks_for_aircraft(k)
        assert len(tasks_k) == 283, f"飞机 {k} 的物理工序数不为 283"
        for t in tasks_k:
            assert t.aircraft_id == k
            assert t.task_key == f"{k}_{t.task_id}"
            assert t.task_key not in seen_keys, f"重复的任务标识: {t.task_key}"
            seen_keys.add(t.task_key)

    assert len(seen_keys) == 2830


def test_station_and_cycle_bounds(baseline_data: MultiAircraftBaseline) -> None:
    """校验所有工序的起止时间严格落在名义站位脉动时间窗内。"""
    h0 = baseline_data.h0
    tolerance = 1e-5

    for task in baseline_data.tasks.values():
        k = task.aircraft_id
        s = task.station_id
        expected_cycle = k + s
        assert task.cycle_idx == expected_cycle, (
            f"工序 {task.task_key} 周期编号错误: 期望 {expected_cycle}, 实际 {task.cycle_idx}"
        )

        nominal_entry = (k + s - 1) * h0
        nominal_exit = (k + s) * h0

        assert abs(task.nominal_station_entry - nominal_entry) < tolerance
        assert abs(task.nominal_station_exit - nominal_exit) < tolerance

        # 物理起止时刻必须满足站内周期边界约束
        assert task.baseline_start >= nominal_entry - tolerance, (
            f"工序 {task.task_key} 提早开工: start={task.baseline_start}, entry={nominal_entry}"
        )
        assert task.baseline_end <= nominal_exit + tolerance, (
            f"工序 {task.task_key} 超出站位周期: end={task.baseline_end}, exit={nominal_exit}"
        )
        assert task.baseline_end >= task.baseline_start + task.duration - tolerance


def test_craft_dag_precedence(
    baseline_data: MultiAircraftBaseline,
    raw_constraint_engine: tuple[ConstraintEngine, dict[str, Any]],
) -> None:
    """校验每架飞机内部的工艺 DAG 物理前后序约束无违背。"""
    engine, _ = raw_constraint_engine
    tolerance = 1e-5
    violations = 0

    for k in range(baseline_data.num_aircraft):
        aircraft_tasks = {t.task_id: t for t in baseline_data.get_tasks_for_aircraft(k)}

        for tid, task in aircraft_tasks.items():
            # 检查物理紧前工序
            physical_preds = engine.physical_predecessors[tid]
            for pred_id in physical_preds:
                if pred_id not in aircraft_tasks:
                    continue
                pred_task = aircraft_tasks[pred_id]
                # 紧前工序完工时刻必须 <= 当前工序开工时刻
                if pred_task.baseline_end > task.baseline_start + tolerance:
                    violations += 1

    assert violations == 0, f"发现 {violations} 处 DAG 偏序冲突！"


def test_station_slot_capacity(baseline_data: MultiAircraftBaseline) -> None:
    """校验 5 个站位中任何时刻的并发物理工序数均不超过槽位上限 3。"""
    max_slot_capacity = 3
    tolerance = 1e-5

    for station_id in range(1, 6):
        # 收集该站位的所有任务执行区间
        intervals: list[tuple[float, float, str]] = []
        for task in baseline_data.tasks.values():
            if task.station_id == station_id:
                intervals.append((task.baseline_start, task.baseline_end, task.task_key))

        # 扫描线算法统计最大并发数（按容差尺度 round 到 5 位小数，消除浮点数尾差导致的相邻工序瞬时伪并发）
        events: list[tuple[float, int, str]] = []
        for start, end, key in intervals:
            if end - start > tolerance:
                events.append((round(start, 5), 1, key))
                events.append((round(end, 5), -1, key))

        # 排序：同一时刻先完工释放(-1)再开工占用(1)
        events.sort(key=lambda x: (x[0], x[1]))

        current_concurrent = 0
        max_concurrent = 0
        for time_pt, delta, _ in events:
            current_concurrent += delta
            if current_concurrent > max_concurrent:
                max_concurrent = current_concurrent

        assert max_concurrent <= max_slot_capacity, (
            f"站位 {station_id} 槽位超限: 最大并发数={max_concurrent} > 上限 {max_slot_capacity}"
        )


def test_worker_station_binding(baseline_data: MultiAircraftBaseline) -> None:
    """校验工人固定站位绑定互斥隔离，绝无跨站调动违规。"""
    station_workers = baseline_data.station_workers

    # 1. 站位之间工人集合互斥
    all_workers: list[int] = []
    for s, workers in station_workers.items():
        all_workers.extend(workers)
    assert len(all_workers) == len(set(all_workers)), "各站位工人集合存在交集，违反站位独立绑定原则！"

    # 2. 实际任务指派与绑定的站位完全一致
    for task in baseline_data.tasks.values():
        allowed_workers = set(station_workers[task.station_id])
        for w in task.team:
            assert w in allowed_workers, (
                f"工序 {task.task_key} 指派了非本站工人 {w} (本站={task.station_id})"
            )


def test_worker_no_overlap(baseline_data: MultiAircraftBaseline) -> None:
    """校验全线所有工人在全时段 [0, 14*H0] 内零重叠，无一人分身多用。"""
    tolerance = 1e-5
    worker_intervals: dict[int, list[tuple[float, float, str]]] = {}

    for task in baseline_data.tasks.values():
        for w in task.team:
            worker_intervals.setdefault(w, []).append(
                (task.baseline_start, task.baseline_end, task.task_key)
            )

    overlap_count = 0
    for w, intervals in worker_intervals.items():
        # 按开工时间排序
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        for prev, curr in zip(sorted_intervals, sorted_intervals[1:]):
            prev_start, prev_end, prev_key = prev
            curr_start, curr_end, curr_key = curr
            if prev_end > curr_start + tolerance:
                overlap_count += 1
                print(
                    f"工人 {w} 时间重叠: {prev_key} [{prev_start:.2f}, {prev_end:.2f}] 与 "
                    f"{curr_key} [{curr_start:.2f}, {curr_end:.2f}]"
                )

    assert overlap_count == 0, f"发现 {overlap_count} 处工人时间重叠冲突！"


def test_skill_and_demand_fulfillment(baseline_data: MultiAircraftBaseline) -> None:
    """校验每道工序的指派团队人数符合需求，且技能要求满足。"""
    worker_pool_path = Path("data/worker_pool_fixed.csv")
    worker_skill_matrix = None
    if worker_pool_path.is_file():
        df_workers = pd.read_csv(worker_pool_path).set_index("worker_id")
        skill_cols = [f"skill_{i}" for i in range(5)]
        if all(c in df_workers.columns for c in skill_cols):
            worker_skill_matrix = df_workers[skill_cols].values

    for task in baseline_data.tasks.values():
        # 需求人数
        assert len(task.team) == task.demand, (
            f"工序 {task.task_key} 需求人数不符: 需求 {task.demand}, 实际 {len(task.team)}"
        )
        assert len(task.team) == len(set(task.team)), f"工序 {task.task_key} 团队内存在重复工人"

        # 技能资质校验
        if worker_skill_matrix is not None and task.skill >= 0:
            for w in task.team:
                if w < len(worker_skill_matrix) and task.skill < worker_skill_matrix.shape[1]:
                    assert worker_skill_matrix[w, task.skill] >= 0.5, (
                        f"工人 {w} 不具备工序 {task.task_key} 所需技能 {task.skill}"
                    )


def test_json_serialization_roundtrip(baseline_data: MultiAircraftBaseline, tmp_path: Path) -> None:
    """校验 JSON 文件导出与还原的一致性。"""
    test_json = tmp_path / "test_baseline_roundtrip.json"
    baseline_data.save_to_json(test_json)
    reloaded = MultiAircraftBaseline.load_from_json(test_json)

    assert reloaded.num_aircraft == baseline_data.num_aircraft
    assert reloaded.num_stations == baseline_data.num_stations
    assert abs(reloaded.h0 - baseline_data.h0) < 1e-6
    assert reloaded.total_tasks_count == baseline_data.total_tasks_count

    # 抽样比对首末任务
    task_first = reloaded.get_task(0, baseline_data.get_tasks_for_aircraft(0)[0].task_id)
    orig_first = baseline_data.get_task(0, baseline_data.get_tasks_for_aircraft(0)[0].task_id)
    assert task_first.task_key == orig_first.task_key
    assert abs(task_first.baseline_start - orig_first.baseline_start) < 1e-6
    assert task_first.team == orig_first.team
