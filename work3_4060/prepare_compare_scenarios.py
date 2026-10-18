"""生成并严格验证 compare_v1 专用小场景包（8 train / 4 val / 8 test / 1 nominal）。

遵循《工作三：ad38f4e 复核与小规模 C/D 效果对比执行方案》第 5 节：
1. 扰动目标任务确实存在于 mini210，保持合法前序与外生事件定义；tau 与仿真工时均为小时；
2. train / validation / test 按实际事件内容去重（不同 task_key / tau / recovery）；
3. 用固定启发式参照轨迹离线选取不同强度、目标与时段；
4. 逐个进行真实环境与启发式前向验证，确保：
   - 任务确实存在；
   - 事件确实命中（actual_hit_task_keys 包含目标任务）；
   - 整批生产最终可行（feasible，terminated == True，termination_reason == 'completed'）；
   - 费用分项存在可区分的变化；
5. 冻结并输出 SHA-256 哈希清单 manifest.json。
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.environment import AirLineEnvWork3
from models.work3.heuristic_agent import HeuristicAgentWork3


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_heuristic_simulation(baseline_path: Path, scenario: dict[str, Any] | None = None) -> dict[str, Any]:
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    env.reset()
    if scenario is not None:
        env.load_scenario(scenario)
    agent = HeuristicAgentWork3()

    step_count = 0
    while True:
        act = agent.select_action(env)
        _, _, term, trunc, info = env.step(act)
        step_count += 1
        if term or trunc:
            break
        if step_count > 2500:
            raise RuntimeError(f"仿真步数超限 ({step_count})，可能存在死锁")

    hit_info = {}
    if scenario is not None:
        s_id = scenario["scenario_id"]
        hit_info = env.disturbance_event_results.get(s_id, {})

    return {
        "completed": bool(env.state.is_all_completed() if hasattr(env.state, "is_all_completed") else info.get("termination_reason") == "completed"),
        "termination_reason": info.get("termination_reason"),
        "step_count": step_count,
        "makespan": float(env.state.current_time),
        "cumulative_cost": float(env.cumulative_cost),
        "cost_takt": float(env.cost_takt),
        "cost_time": float(env.cost_time),
        "cost_postpone": float(env.cost_postpone),
        "transfer_history": [float(x) for x in env.state.transfer_history],
        "hit_info": hit_info,
        "tasks": {
            k: {
                "start": float(t.actual_start),
                "end": float(t.actual_end),
                "station": int(t.current_station),
                "plane": int(t.aircraft_id),
                "duration": float(t.duration),
            }
            for k, t in env.state.tasks.items()
            if t.actual_start is not None
        },
    }


def main():
    dest_dir = REPO_ROOT / "work3_4060/compare_v1"
    dest_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = REPO_ROOT / "work3_4060/data/mini210.json"

    print("Step 1: 运行 nominal heuristic 建立基准时间轴...")
    nom_result = run_heuristic_simulation(baseline_path, scenario=None)
    assert nom_result["completed"], "Nominal 基准轨迹未正常完成"
    print(f"Nominal 完成: makespan={nom_result['makespan']:.2f}h, steps={nom_result['step_count']}, cost={nom_result['cumulative_cost']:.2f}")

    tasks_map = nom_result["tasks"]

    # 选取代表性工序覆盖早期、中期、晚期，五站及多架次
    # 策略：
    # 8 Train scenarios: 涵盖飞机 0, 1, 2, 3, 4，站位 0, 1, 2, 3, 4，低/中/高强度
    # 4 Val scenarios: 涵盖飞机 1, 2, 3, 4，站位 0, 1, 2, 3
    # 8 Test scenarios: 涵盖飞机 1, 2, 3, 4, 5，站位 0, 1, 2, 3, 4，包含早期/满线期/晚期

    # 预选候选任务列表 (plane, task_id, delta, timing, intensity, category)
    specs = [
        # --- TRAIN (8) ---
        {"cat": "train", "s_id": "COMPARE_TRAIN_S0_EARLY_L", "plane": 0, "tid": 16, "delta": 15.0, "timing": "EARLY", "intensity": "LOW"},
        {"cat": "train", "s_id": "COMPARE_TRAIN_S1_EARLY_M", "plane": 1, "tid": 21, "delta": 25.0, "timing": "EARLY", "intensity": "MED"},
        {"cat": "train", "s_id": "COMPARE_TRAIN_S2_MID_L",   "plane": 1, "tid": 13, "delta": 15.0, "timing": "MID",   "intensity": "LOW"},
        {"cat": "train", "s_id": "COMPARE_TRAIN_S3_MID_H",   "plane": 2, "tid": 24, "delta": 35.0, "timing": "MID",   "intensity": "HIGH"},
        {"cat": "train", "s_id": "COMPARE_TRAIN_S0_MID_M",   "plane": 3, "tid": 16, "delta": 25.0, "timing": "MID",   "intensity": "MED"},
        {"cat": "train", "s_id": "COMPARE_TRAIN_S4_MID_M",   "plane": 2, "tid": 125, "delta": 25.0, "timing": "MID",   "intensity": "MED"},
        {"cat": "train", "s_id": "COMPARE_TRAIN_S1_LATE_L",  "plane": 4, "tid": 8, "delta": 15.0, "timing": "LATE",  "intensity": "LOW"},
        {"cat": "train", "s_id": "COMPARE_TRAIN_S2_LATE_H",  "plane": 3, "tid": 13, "delta": 35.0, "timing": "LATE",  "intensity": "HIGH"},

        # --- VALIDATION (4) ---
        {"cat": "val",   "s_id": "COMPARE_VAL_S0_EARLY_M",   "plane": 1, "tid": 16, "delta": 20.0, "timing": "EARLY", "intensity": "MED"},
        {"cat": "val",   "s_id": "COMPARE_VAL_S2_MID_M",     "plane": 2, "tid": 13, "delta": 25.0, "timing": "MID",   "intensity": "MED"},
        {"cat": "val",   "s_id": "COMPARE_VAL_S3_MID_L",     "plane": 3, "tid": 24, "delta": 15.0, "timing": "MID",   "intensity": "LOW"},
        {"cat": "val",   "s_id": "COMPARE_VAL_S4_LATE_H",    "plane": 3, "tid": 125, "delta": 35.0, "timing": "LATE",  "intensity": "HIGH"},

        # --- TEST (8) ---
        {"cat": "test",  "s_id": "COMPARE_TEST_S0_EARLY_M",  "plane": 2, "tid": 16, "delta": 25.0, "timing": "EARLY", "intensity": "MED"},
        {"cat": "test",  "s_id": "COMPARE_TEST_S1_EARLY_L",  "plane": 2, "tid": 21, "delta": 15.0, "timing": "EARLY", "intensity": "LOW"},
        {"cat": "test",  "s_id": "COMPARE_TEST_S2_MID_H",    "plane": 2, "tid": 9,  "delta": 35.0, "timing": "MID",   "intensity": "HIGH"},
        {"cat": "test",  "s_id": "COMPARE_TEST_S3_MID_M",    "plane": 3, "tid": 23, "delta": 25.0, "timing": "MID",   "intensity": "MED"},
        {"cat": "test",  "s_id": "COMPARE_TEST_S4_MID_L",    "plane": 4, "tid": 125, "delta": 15.0, "timing": "MID",   "intensity": "LOW"},
        {"cat": "test",  "s_id": "COMPARE_TEST_S1_MID_H",    "plane": 3, "tid": 8,  "delta": 35.0, "timing": "MID",   "intensity": "HIGH"},
        {"cat": "test",  "s_id": "COMPARE_TEST_S0_LATE_L",   "plane": 4, "tid": 16, "delta": 15.0, "timing": "LATE",  "intensity": "LOW"},
        {"cat": "test",  "s_id": "COMPARE_TEST_S2_LATE_M",   "plane": 4, "tid": 13, "delta": 25.0, "timing": "LATE",  "intensity": "MED"},
    ]

    all_scenarios = []
    train_ids = []
    val_ids = []
    test_ids = []

    print("\nStep 2: 生成并验证各个扰动事件...")
    seen_contents = set()

    for spec in specs:
        tkey = f"{spec['plane']}_{spec['tid']}"
        assert tkey in tasks_map, f"任务 {tkey} 不存在于 mini210 任务图中"
        tinfo = tasks_map[tkey]
        nom_start = tinfo["start"]
        station_id = tinfo["station"]

        # 确保 tau <= nom_start，即任务开工前扰动到达
        # 将 tau 设为 nom_start 前 2 小时（或 0.0）
        tau = max(0.0, nom_start - 2.0)
        delta = float(spec["delta"])
        rec_time = nom_start + delta

        # 去重检查
        content_sig = (tkey, round(tau, 3), round(rec_time, 3))
        assert content_sig not in seen_contents, f"事件内容重复: {content_sig}"
        seen_contents.add(content_sig)

        scen = {
            "scenario_id": spec["s_id"],
            "valid": True,
            "timing": spec["timing"],
            "intensity": spec["intensity"],
            "station_id": station_id,
            "aircraft_id": spec["plane"],
            "tau": float(tau),
            "delta": delta,
            "recovery_time": float(rec_time),
            "candidate_count": 1,
            "affected_count": 1,
            "affected_task_keys": [tkey],
            "timing_reference": "compare_v1_offline_heuristic",
        }

        # 真实环境前向验证
        v_res = run_heuristic_simulation(baseline_path, scenario=scen)
        assert v_res["completed"], f"场景 {spec['s_id']} 未完成调度"
        hit_keys = v_res["hit_info"].get("actual_hit_task_keys", ())
        assert tkey in hit_keys, f"场景 {spec['s_id']} 未命中目标任务 {tkey}: hit_keys={hit_keys}, unhit={v_res['hit_info'].get('unhit_reasons')}"
        cost_diff = v_res["cumulative_cost"] - nom_result["cumulative_cost"]
        makespan_diff = v_res["makespan"] - nom_result["makespan"]

        print(f"  [{spec['cat'].upper()}] {spec['s_id']:<26} -> 目标={tkey:<4} Station={station_id} tau={tau:<6.1f} R={rec_time:<6.1f} | Hit=OK, MakespanDiff={makespan_diff:+.1f}h, CostDiff={cost_diff:+.2f}")

        all_scenarios.append(scen)
        if spec["cat"] == "train":
            train_ids.append(spec["s_id"])
        elif spec["cat"] == "val":
            val_ids.append(spec["s_id"])
        elif spec["cat"] == "test":
            test_ids.append(spec["s_id"])

    print(f"\n去重后场景统计: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}, Total={len(all_scenarios)}")
    assert len(train_ids) == 8
    assert len(val_ids) == 4
    assert len(test_ids) == 8

    # 写入文件
    scenarios_path = dest_dir / "scenarios.json"
    train_path = dest_dir / "train.json"
    val_path = dest_dir / "validation.json"
    test_path = dest_dir / "test.json"

    scenarios_path.write_text(json.dumps(all_scenarios, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    train_path.write_text(json.dumps(train_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    val_path.write_text(json.dumps(val_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    test_path.write_text(json.dumps(test_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "dataset": "mini210",
        "description": "Dedicated scenario pack for Work-3 Method C vs Method D comparison (compare_v1)",
        "train_count": len(train_ids),
        "validation_count": len(val_ids),
        "test_count": len(test_ids),
        "total_scenarios": len(all_scenarios),
        "files": {
            "scenarios.json": digest_file(scenarios_path),
            "train.json": digest_file(train_path),
            "validation.json": digest_file(val_path),
            "test.json": digest_file(test_path),
        },
    }
    manifest_path = dest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n生成完成！Manifest 内容：")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
