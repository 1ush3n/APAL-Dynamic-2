"""Prepare and calibrate disturbance scenarios for Rule Baseline A/C comparison.

Outputs:
- work3_4060/rule_baseline_ac_v1/scenarios.json
- work3_4060/rule_baseline_ac_v1/train.json
- work3_4060/rule_baseline_ac_v1/validation.json
- work3_4060/rule_baseline_ac_v1/scenario_calibration.json

Contracts:
1. R = P_q^0 + rho * H0, where P_q^0 is the nominal entry of the task's station cycle, rho in (0, 1).
2. tau < S_{k, i}^0, R > tau.
3. 1 Nominal + 4 Train + 2 Validation scenarios.
4. Validation on BaselinePlanRepairAgent (A):
   - actual_disturbance_hits >= 1 for all disturbances.
   - Target actual start >= R.
   - At least one validation scenario produces strictly positive J_takt > 0.
   - All trajectories physically feasible and complete all 210 tasks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.environment import AirLineEnvWork3
from scripts.work3.baseline_repair_agent import BaselinePlanRepairAgent
from scripts.work3.evaluate_c_vs_d import evaluate_single_trajectory
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_and_calibrate_scenarios(version: str = "v1") -> None:
    if version == "v1":
        dest_dir = REPO_ROOT / "work3_4060/rule_baseline_ac_v1"
        timing_ref = "nominal_entry_referenced_preentry_v1"
    elif version == "v2":
        dest_dir = REPO_ROOT / "work3_4060/rule_baseline_ac_v2"
        timing_ref = "nominal_exit_referenced_postexit_v2"
    else:
        raise ValueError(f"Unknown version: {version}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = dest_dir / "mini210_rule_baseline.json"
    if not baseline_path.is_file():
        v1_baseline = REPO_ROOT / "work3_4060/rule_baseline_ac_v1/mini210_rule_baseline.json"
        if v1_baseline.is_file():
            shutil.copy2(v1_baseline, baseline_path)
        else:
            raise FileNotFoundError(f"Source baseline not found: {v1_baseline}")
    
    baseline = MultiAircraftBaseline.load_from_json(baseline_path)
    h0 = float(baseline.h0)
    print(f"\n[PrepareScenarios] Loaded {version} baseline with H0 = {h0:.6f}h, tasks = {len(baseline.tasks)}")

    # 4 Train scenarios + 2 Validation scenarios
    # Each scenario defines:
    # - scenario_id
    # - split: "train" | "val"
    # - aircraft_id, station_id, task_key
    # - rho: fraction in (0, 1)
    scenario_specs = [
        {
            "scenario_id": "RULE_TRAIN_S1_RHO10",
            "split": "train",
            "aircraft_id": 1,
            "station_id": 1,
            "task_key": "1_3",
            "rho": 0.10,
            "timing": "EARLY",
            "intensity": "LOW",
            "description": "Station 1 early task material delay rho=0.10",
        },
        {
            "scenario_id": "RULE_TRAIN_S2_RHO25",
            "split": "train",
            "aircraft_id": 2,
            "station_id": 2,
            "task_key": "2_8",
            "rho": 0.25,
            "timing": "MID",
            "intensity": "MED",
            "description": "Station 2 task material delay rho=0.25",
        },
        {
            "scenario_id": "RULE_TRAIN_S3_RHO50",
            "split": "train",
            "aircraft_id": 2,
            "station_id": 3,
            "task_key": "2_27",
            "rho": 0.50,
            "timing": "MID",
            "intensity": "HIGH",
            "description": "Station 3 bottleneck task material delay rho=0.50",
        },
        {
            "scenario_id": "RULE_TRAIN_S4_RHO75",
            "split": "train",
            "aircraft_id": 3,
            "station_id": 4,
            "task_key": "3_10",
            "rho": 0.75,
            "timing": "LATE",
            "intensity": "HIGH",
            "description": "Station 4 task material delay rho=0.75",
        },
        {
            "scenario_id": "RULE_VAL_S1_MODERATE",
            "split": "val",
            "aircraft_id": 1,
            "station_id": 1,
            "task_key": "1_4",
            "rho": 0.20,
            "timing": "EARLY",
            "intensity": "MED",
            "description": "Validation 1: Station 1 moderate critical path delay rho=0.20",
        },
        {
            "scenario_id": "RULE_VAL_S2_STRONG",
            "split": "val",
            "aircraft_id": 1,
            "station_id": 3,
            "task_key": "1_27",
            "rho": 0.60,
            "timing": "MID",
            "intensity": "HIGH",
            "description": "Validation 2: Station 3 strong bottleneck delay rho=0.60",
        },
    ]

    scenarios_list: list[dict[str, Any]] = []
    train_ids: list[str] = []
    val_ids: list[str] = []

    for spec in scenario_specs:
        s_id = spec["scenario_id"]
        plane = spec["aircraft_id"]
        station = spec["station_id"]
        task_key = spec["task_key"]
        rho = spec["rho"]

        # Ensure task exists in baseline
        task = baseline.tasks.get(task_key)
        assert task is not None, f"Task {task_key} not in baseline"
        assert task.aircraft_id == plane, f"Task aircraft mismatch: {task.aircraft_id} != {plane}"
        assert task.station_id == station, f"Task station mismatch: {task.station_id} != {station}"

        # P_q^0: nominal entry for plane at station
        p_entry = (plane + station - 1) * h0
        tau = max(0.0, p_entry - 5.0)
        if version == "v1":
            recovery_time = p_entry + rho * h0
        else:  # v2: exit-referenced R = P_q^0 + rho * H0
            p_exit = (plane + station) * h0
            recovery_time = p_exit + rho * h0
        delta = recovery_time - tau

        # Contract assertions
        assert task.baseline_start > tau, f"Nominal start {task.baseline_start} not > tau {tau}"
        assert recovery_time > tau, f"Recovery {recovery_time} not > tau {tau}"

        scen_dict = {
            "scenario_id": s_id,
            "valid": True,
            "split": spec["split"],
            "timing": spec["timing"],
            "intensity": spec["intensity"],
            "aircraft_id": plane,
            "station_id": station,
            "target_task_key": task_key,
            "rho": float(rho),
            "tau": float(tau),
            "delta": float(delta),
            "recovery_time": float(recovery_time),
            "candidate_count": 1,
            "affected_count": 1,
            "affected_task_keys": [task_key],
            "description": spec["description"],
            "timing_reference": timing_ref,
        }
        scenarios_list.append(scen_dict)
        if spec["split"] == "train":
            train_ids.append(s_id)
        else:
            val_ids.append(s_id)

    # Save scenarios.json, train.json, validation.json
    scenarios_path = dest_dir / "scenarios.json"
    train_path = dest_dir / "train.json"
    val_path = dest_dir / "validation.json"

    with scenarios_path.open("w", encoding="utf-8") as f:
        json.dump(scenarios_list, f, indent=2)
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train_ids, f, indent=2)
    with val_path.open("w", encoding="utf-8") as f:
        json.dump(val_ids, f, indent=2)

    print(f"[PrepareScenarios] Wrote scenarios ({len(scenarios_list)}), train ({len(train_ids)}), val ({len(val_ids)})")

    # Now run verification and calibration using BaselinePlanRepairAgent (A)
    print("\n[PrepareScenarios] Running verification on BaselinePlanRepairAgent (A)...")
    calibration_records: dict[str, Any] = {}

    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    agent = BaselinePlanRepairAgent()

    # 1. Nominal
    nom_res = evaluate_single_trajectory(env, "Baseline-Repair-A", agent, scenario=None)
    assert nom_res["success"] is True and nom_res["feasible"] is True
    assert nom_res["completed_tasks"] == 210
    assert nom_res["objective_j_total"] <= 1e-9
    calibration_records["NOMINAL"] = {
        "scenario_id": "NOMINAL",
        "split": "nominal",
        "success": nom_res["success"],
        "feasible": nom_res["feasible"],
        "completed_tasks": nom_res["completed_tasks"],
        "actual_disturbance_hits": nom_res["actual_disturbance_hits"],
        "j_total": nom_res["objective_j_total"],
        "j_takt": nom_res["j_takt"],
        "d_time": nom_res["d_time"],
        "d_team": nom_res["d_team"],
        "j_postpone": nom_res["j_postpone"],
        "j_revision": nom_res["j_revision"],
        "makespan": nom_res["makespan"],
    }
    print(f"  [NOMINAL] J_total={nom_res['objective_j_total']:.6e}, Makespan={nom_res['makespan']:.4f}h, Completed={nom_res['completed_tasks']}/210")

    # 2. Scenarios
    has_positive_j_takt = False
    for scen in scenarios_list:
        s_id = scen["scenario_id"]
        res = evaluate_single_trajectory(env, "Baseline-Repair-A", agent, scenario=scen)

        # Verification checks
        assert res["success"] is True, f"{s_id} failed success"
        assert res["feasible"] is True, f"{s_id} failed feasibility"
        assert res["completed_tasks"] == 210, f"{s_id} completed {res['completed_tasks']}/210"
        assert res["actual_disturbance_hits"] >= 1, f"{s_id} actual_disturbance_hits == 0"

        # Check target task actual start >= recovery_time
        target_task_key = scen["affected_task_keys"][0]
        actual_start = env.state.tasks[target_task_key].actual_start
        assert actual_start is not None and actual_start >= scen["recovery_time"] - 1e-5, (
            f"{s_id} target task {target_task_key} actual start {actual_start} < recovery_time {scen['recovery_time']}"
        )

        if scen["split"] == "val" and res["j_takt"] > 1e-6:
            has_positive_j_takt = True

        calibration_records[s_id] = {
            "scenario_id": s_id,
            "split": scen["split"],
            "target_task_key": target_task_key,
            "target_actual_start": float(actual_start),
            "target_nominal_start": float(baseline.tasks[target_task_key].baseline_start),
            "recovery_time": scen["recovery_time"],
            "rho": scen["rho"],
            "success": res["success"],
            "feasible": res["feasible"],
            "completed_tasks": res["completed_tasks"],
            "actual_disturbance_hits": res["actual_disturbance_hits"],
            "j_total": float(res["objective_j_total"]),
            "j_takt": float(res["j_takt"]),
            "d_time": float(res["d_time"]),
            "d_team": float(res["d_team"]),
            "j_postpone": float(res["j_postpone"]),
            "j_revision": float(res["j_revision"]),
            "makespan": float(res["makespan"]),
            "makespan_delta_vs_nominal": float(res["makespan"] - nom_res["makespan"]),
        }
        print(
            f"  [{scen['split'].upper()}] {s_id:<22} -> target={target_task_key:<4} start={actual_start:.2f}>=R={scen['recovery_time']:.2f} "
            f"| hits={res['actual_disturbance_hits']} J_total={res['objective_j_total']:.6f} J_takt={res['j_takt']:.6f} "
            f"MakespanDiff={res['makespan'] - nom_res['makespan']:+.2f}h"
        )

    assert has_positive_j_takt, "At least one validation scenario must produce strictly positive J_takt > 0"
    print("\n[PrepareScenarios] All calibration assertions PASSED!")

    # Calibration report
    calibration_report = {
        "baseline_metadata": baseline.metadata,
        "baseline_sha256": sha256_file(baseline_path),
        "scenarios_sha256": sha256_file(scenarios_path),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(val_path),
        "results": calibration_records,
        "acceptance_criteria": {
            "all_feasible": all(r["feasible"] for r in calibration_records.values()),
            "all_completed_210": all(r["completed_tasks"] == 210 for r in calibration_records.values()),
            "validation_measurable_impact": any(
                (r["j_takt"] > 1e-6 or r["d_time"] > 1e-6 or abs(r.get("makespan_delta_vs_nominal", 0.0)) > 1e-6)
                for sid, r in calibration_records.items() if r.get("split") == "val"
            ),
            "validation_has_positive_j_takt": has_positive_j_takt,
            "nominal_j_total_near_zero": calibration_records["NOMINAL"]["j_total"] <= 1e-9,
        },
    }

    cal_path = dest_dir / "scenario_calibration.json"
    with cal_path.open("w", encoding="utf-8") as f:
        json.dump(calibration_report, f, indent=2)
    print(f"[PrepareScenarios] Wrote calibration report to {cal_path}")


if __name__ == "__main__":
    print("=== Generating & Calibrating Rule Scenarios v1 ===")
    generate_and_calibrate_scenarios("v1")
    print("\n=== Generating & Calibrating Rule Scenarios v2 (Offline A Only) ===")
    generate_and_calibrate_scenarios("v2")
