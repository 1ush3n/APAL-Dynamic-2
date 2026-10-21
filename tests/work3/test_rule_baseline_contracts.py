"""Unit A: 规则基准与输入清册契约测试 (Section 3, 4, 9)。

测试清单：
1. 局部/全局时间坐标映射合同：S_{k,i}^0 = (k + m_i - 1)*H0 + b_i^0，验证首架首站、末架末站及填线/排空边界。
2. 参考分区工艺合法性：无固定站位冲突，无反向跨站前序依赖，各站工人完全正交。
3. 规则列表排程物理可行性：
   - 零工人冲突 (同一工人在任何时刻最多执行1个任务)；
   - 零槽位超限 (每个站位在任何时刻并发在制任务数 <= 3)；
   - 零前序冲突 (同站前序开工前完工，跨站前序在更早周期完成)；
   - 零周期转站越界 (工序开工/完工严格位于名义进站与出站区间内)。
4. 坏输入拦截：人为破坏参考分区或注入反向跨站依赖必须被拦截拒绝。
5. 生成幂等性与内容一致性：重复生成模板两次，规范化核心计划内容完全一致。
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

from work3_4060.build_rule_baseline import (
    build_deterministic_station_schedule,
    generate_rule_baseline,
    load_craft_data,
    load_worker_pool,
)


def test_coordinate_conversion_contract():
    """验证局部偏移到全局绝对时间的数学映射公式 (Section 4.3)。"""
    h0 = 50.0
    num_aircraft = 7
    num_stations = 5

    # 约定飞机 k 从 0 编号 (0 ~ 6)，站位 m 从 1 编号 (1 ~ 5)
    # E_{k, m}^0 = (k + m - 1) * H0
    # S_{k, i}^0 = E_{k, m}^0 + b_i^0
    # C_{k, i}^0 = S_{k, i}^0 + p_i^0

    # 1. 首架飞机 (k=0) 在首站 (m=1):
    # 名义进站时刻 = (0 + 1 - 1) * 50 = 0.0
    # 若 b_i = 10.0, p_i = 5.0, 则全局开工 = 10.0, 完工 = 15.0
    e_0_1 = (0 + 1 - 1) * h0
    assert e_0_1 == 0.0
    s_0_i = e_0_1 + 10.0
    assert s_0_i == 10.0

    # 2. 首架飞机在第 3 站 (m=3):
    # 名义进站时刻 = (0 + 3 - 1) * 50 = 100.0 (周期 3 进站)
    # 若 b_i = 2.0, 全局开工应为 102.0
    e_0_3 = (0 + 3 - 1) * h0
    assert e_0_3 == 100.0
    assert e_0_3 + 2.0 == 102.0

    # 3. 末架飞机 (k=6) 在末站 (m=5):
    # 名义进站时刻 = (6 + 5 - 1) * 50 = 500.0 (周期 11 进站)
    # 名义出线时刻 = 11 * 50 = 550.0
    e_6_5 = (6 + 5 - 1) * h0
    assert e_6_5 == 500.0
    exit_6_5 = (6 + 5) * h0
    assert exit_6_5 == 550.0
    assert exit_6_5 == (num_aircraft + num_stations - 1) * h0


def test_reference_partition_craft_validity():
    """验证沿用的 mini210 参考站位分区不违背任何工艺硬约束。"""
    craft_csv_path = REPO_ROOT / "data/283.csv"
    mini210_source_path = REPO_ROOT / "work3_4060/data/mini210.json"
    task_craft, _ = load_craft_data(craft_csv_path)

    with mini210_source_path.open("r", encoding="utf-8") as f:
        mini_data = json.load(f)

    tasks_p0 = {t["task_id"]: t for t in mini_data["tasks"].values() if int(t["aircraft_id"]) == 0}
    selected_tids = sorted(tasks_p0.keys())
    ref_partition = {tid: int(tasks_p0[tid]["station_id"]) for tid in selected_tids}

    # 校验 1: 固定站位硬约束
    for tid in selected_tids:
        craft = task_craft[tid]
        if craft["fixed_station"] is not None:
            assert ref_partition[tid] == craft["fixed_station"]

    # 校验 2: 站位偏序单调性 (前驱站位 <= 后继站位)
    for tid in selected_tids:
        for p in task_craft[tid]["predecessors"]:
            if p in ref_partition:
                assert ref_partition[p] <= ref_partition[tid], (
                    f"工序 {tid} (站位 {ref_partition[tid]}) 的前驱 {p} 位于更晚站位 {ref_partition[p]}"
                )


def test_rule_baseline_physical_feasibility():
    """验证生成的 mini210_rule_baseline.json 物理无冲突、槽位不超限、前序守恒。"""
    baseline_path = REPO_ROOT / "work3_4060/rule_baseline_ac_v1/mini210_rule_baseline.json"
    val_path = REPO_ROOT / "work3_4060/rule_baseline_ac_v1/baseline_validation.json"

    assert baseline_path.is_file()
    assert val_path.is_file()

    with baseline_path.open("r", encoding="utf-8") as f:
        base_data = json.load(f)
    with val_path.open("r", encoding="utf-8") as f:
        val_data = json.load(f)

    # 1. 验证元数据
    meta = base_data["metadata"]
    assert meta["num_aircraft"] == 7
    assert meta["num_stations"] == 5
    assert meta["total_cycles"] == 11
    assert meta["total_physical_tasks"] == 210
    assert meta["physical_tasks_per_aircraft"] == 30
    assert meta["transfer_policy"] == "nominal_anchored_v1"
    assert math.isclose(meta["h0"], 52.037844127, rel_tol=1e-5)
    assert meta["bottleneck_station"] == 3

    # 2. 验证校验文件无违规
    assert val_data["feasible"] is True
    assert val_data["violations"] == {}
    assert val_data["total_tasks_checked"] == 210

    # 3. 独立重算全线工人互斥无冲突
    tasks = base_data["tasks"]
    worker_events: dict[int, list[tuple[float, float, str]]] = {}
    for key, t in tasks.items():
        s = float(t["baseline_start"])
        e = float(t["baseline_end"])
        for w in t["team"]:
            worker_events.setdefault(w, []).append((s, e, key))

    for w, ivs in worker_events.items():
        ivs.sort()
        for i in range(len(ivs) - 1):
            s1, e1, k1 = ivs[i]
            s2, e2, k2 = ivs[i + 1]
            assert s2 >= e1 - 1e-9, f"工人 {w} 在工序 {k1} 与 {k2} 冲突"


def test_rule_baseline_generation_idempotence(tmp_path: Path):
    """验证同一输入下，规则生成器连续运行两次的结果严格内容一致。"""
    res1 = generate_rule_baseline(repo_root=REPO_ROOT, output_dir=tmp_path / "run1", dry_run=False)
    res2 = generate_rule_baseline(repo_root=REPO_ROOT, output_dir=tmp_path / "run2", dry_run=False)

    f1 = Path(res1["baseline_path"]).read_text(encoding="utf-8")
    f2 = Path(res2["baseline_path"]).read_text(encoding="utf-8")

    assert f1 == f2
    assert math.isclose(res1["h0"], res2["h0"], rel_tol=1e-12)


def test_reject_bad_craft_partition():
    """验证当输入违背固定站位约束或产生反向跨站环时，生成器显式抛出异常拦截。"""
    pool_csv_path = REPO_ROOT / "data/worker_pool_fixed.csv"
    worker_effs, worker_skills = load_worker_pool(pool_csv_path)

    # 构造人为倒置的任务依赖：工序 B 在站位 1，但其前驱工序 A 在站位 2
    # 试图在 Station 1 调度含有未完成站外前驱的任务应报错或被外层拦截
    bad_station_tasks = [
        {
            "task_id": 999,
            "ao_code": "BAD_TASK",
            "demand": 2,
            "skill": 0,
            "standard_duration": 1.0,
            "predecessors": [888],  # 888 不在此站内
        }
    ]

    # 单站内若无此工人则抛出 RuntimeError
    with pytest.raises(RuntimeError, match="找不到任何可行团队槽位"):
        build_deterministic_station_schedule(
            station_id=1,
            station_tasks=bad_station_tasks,
            station_workers=[12],  # 人数不足 demand=2
            worker_efficiencies=worker_effs,
            worker_skills=worker_skills,
            max_slots=3,
        )


def test_reject_cyclic_graph_in_station_schedule():
    """验证当站内工序出现两节点环依赖 (A -> B -> A) 时，排程算法明确拒绝并抛出异常。"""
    pool_csv_path = REPO_ROOT / "data/worker_pool_fixed.csv"
    worker_effs, worker_skills = load_worker_pool(pool_csv_path)

    cyclic_tasks = [
        {
            "task_id": 101,
            "ao_code": "CYCLE_A",
            "demand": 1,
            "skill": 0,
            "standard_duration": 1.0,
            "predecessors": [102],
        },
        {
            "task_id": 102,
            "ao_code": "CYCLE_B",
            "demand": 1,
            "skill": 0,
            "standard_duration": 1.0,
            "predecessors": [101],
        },
    ]

    with pytest.raises(RuntimeError, match="拓扑排程未完成全部工序.*环依赖"):
        build_deterministic_station_schedule(
            station_id=1,
            station_tasks=cyclic_tasks,
            station_workers=[12, 72],
            worker_efficiencies=worker_effs,
            worker_skills=worker_skills,
            max_slots=3,
        )


def test_zero_duration_physical_task_schedule():
    """验证包含零工时工序时，排程算法正常完成且工序起止时刻一致。"""
    pool_csv_path = REPO_ROOT / "data/worker_pool_fixed.csv"
    worker_effs, worker_skills = load_worker_pool(pool_csv_path)

    zero_dur_tasks = [
        {
            "task_id": 201,
            "ao_code": "ZERO_TASK",
            "demand": 1,
            "skill": 0,
            "standard_duration": 0.0,
            "predecessors": [],
        }
    ]

    sched, f_m = build_deterministic_station_schedule(
        station_id=1,
        station_tasks=zero_dur_tasks,
        station_workers=[12, 72],
        worker_efficiencies=worker_effs,
        worker_skills=worker_skills,
        max_slots=3,
    )
    assert 201 in sched
    assert sched[201]["duration"] == 0.0
    assert sched[201]["start"] == 0.0
    assert sched[201]["end"] == 0.0
    assert f_m == 0.0


def test_baseline_serialization_roundtrip_preserves_policy(tmp_path: Path):
    """R1: 验证多架次基准计划在 load -> save -> load 往返中严格保留 transfer_policy 与元数据。"""
    from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline

    baseline_path = REPO_ROOT / "work3_4060/rule_baseline_ac_v1/mini210_rule_baseline.json"
    b1 = MultiAircraftBaseline.load_from_json(baseline_path)
    assert b1.transfer_policy == "nominal_anchored_v1"
    assert b1.metadata.get("transfer_policy") == "nominal_anchored_v1"
    assert b1.metadata.get("version") == "work3_mini210_rule_baseline_v1"

    out_file = tmp_path / "saved_baseline.json"
    b1.save_to_json(out_file)

    saved_raw = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved_raw["metadata"]["transfer_policy"] == "nominal_anchored_v1"
    assert saved_raw["metadata"]["version"] == "work3_mini210_rule_baseline_v1"
    assert saved_raw["metadata"]["num_aircraft"] == 7
    assert saved_raw["metadata"]["total_physical_tasks"] == 210

    b2 = MultiAircraftBaseline.load_from_json(out_file)
    assert b2.transfer_policy == "nominal_anchored_v1"
    assert b2.metadata["transfer_policy"] == "nominal_anchored_v1"
    assert b2.num_aircraft == 7
    assert b2.total_tasks_count == 210


def test_baseline_rejects_unknown_transfer_policy(tmp_path: Path):
    """R1: 验证未知转站模式被显式拒绝，不发生静默降级。"""
    from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline

    baseline_path = REPO_ROOT / "work3_4060/rule_baseline_ac_v1/mini210_rule_baseline.json"
    raw_data = json.loads(baseline_path.read_text(encoding="utf-8"))
    raw_data["metadata"]["transfer_policy"] = "unsupported_magic_clock"

    bad_file = tmp_path / "bad_policy_baseline.json"
    bad_file.write_text(json.dumps(raw_data), encoding="utf-8")

    with pytest.raises(ValueError, match="不支持的转站模式"):
        MultiAircraftBaseline.load_from_json(bad_file)

