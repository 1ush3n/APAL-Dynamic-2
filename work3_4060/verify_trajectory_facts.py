"""Independent Trajectory Physical Facts and Cost Reconciler.

Strictly verifies that archived trajectory files satisfy:
1. Worker exclusivity (zero overlap): any worker is assigned to at most 1 task at any point in time.
2. Station capacity (zero overflow): any station has at most 3 concurrent running tasks at any point in time.
3. Craft precedence constraints (zero inverted edges): all 210 tasks start only after their predecessor tasks complete.
4. Station dwell interval: tasks start and end within aircraft's actual entry and exit times at that station.
5. Independent cost recalculation:
   - J_takt: from 11 cycle execution records transfer times (w_h = 1.0)
   - D_time: from tasks actual start vs cycle start time vs in-station offset (w_t = 0.20, scale = 1/210)
   - D_team: from tasks actual team vs base team overlap (w_w = 0.05, scale = 1/210)
   - J_postpone: from tasks postpone counts (w_p = 1.0)
   - J_revision: from tasks revision history
   - J_total: weighted sum of components
   All reconciled against reported trajectory values within 1e-9 tolerance.
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VerifyTrajectoryFacts")


def load_baseline_tasks(baseline_path: Path) -> tuple[float, dict[str, Any], dict[str, Any]]:
    with baseline_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    h0 = float(data["metadata"]["h0"])
    tasks = data["tasks"]
    metadata = data["metadata"]
    return h0, tasks, metadata


def verify_single_trajectory(
    traj_path: Path,
    baseline_h0: float,
    baseline_tasks: dict[str, Any],
    tolerance: float = 1e-5,
    cost_tolerance: float = 1e-9,
) -> dict[str, Any]:
    with traj_path.open("r", encoding="utf-8") as f:
        traj_data = json.load(f)

    exec_facts = traj_data.get("execution_facts", {})
    if not exec_facts:
        tasks_map = traj_data.get("tasks_execution", {})
        aircraft_map = traj_data.get("aircraft_execution", {})
        cycle_map = traj_data.get("cycle_execution_records", {})
        step_rewards = traj_data.get("step_rewards", [])
    else:
        tasks_map = exec_facts.get("tasks_execution", {})
        aircraft_map = exec_facts.get("aircraft_execution", {})
        cycle_map = exec_facts.get("cycle_execution_records", {})
        step_rewards = exec_facts.get("step_rewards", [])

    if not tasks_map:
        raise ValueError(f"Trajectory {traj_path} does not contain tasks_execution facts!")

    violations: dict[str, list[dict[str, Any]]] = {
        "worker_overlap": [],
        "station_overflow": [],
        "precedence_inversion": [],
        "station_interval_violation": [],
        "cost_reconciliation_mismatch": [],
    }

    # 1. Worker Exclusivity (Zero Overlap, Positive Intervals with Heap/Tolerance)
    worker_tasks: dict[int, list[dict[str, Any]]] = {}
    for task_key, t_fact in tasks_map.items():
        s = t_fact.get("actual_start")
        e = t_fact.get("actual_end")
        if s is None or e is None:
            continue
        for w in t_fact.get("team", []):
            worker_tasks.setdefault(int(w), []).append(t_fact)

    for w, t_list in worker_tasks.items():
        positive_intervals = [
            t for t in t_list if float(t["actual_end"]) > float(t["actual_start"]) + tolerance
        ]
        ordered = sorted(positive_intervals, key=lambda x: (float(x["actual_start"]), float(x["actual_end"])))
        for prev, curr in zip(ordered, ordered[1:]):
            if float(prev["actual_end"]) > float(curr["actual_start"]) + tolerance:
                violations["worker_overlap"].append({
                    "worker": w,
                    "task1": prev["task_key"],
                    "interval1": (float(prev["actual_start"]), float(prev["actual_end"])),
                    "task2": curr["task_key"],
                    "interval2": (float(curr["actual_start"]), float(curr["actual_end"])),
                    "overlap": float(prev["actual_end"]) - float(curr["actual_start"]),
                })

    # 2. Station Capacity (Max 3 Concurrent Tasks, Active Ends Heap with Tolerance)
    station_tasks: dict[int, list[dict[str, Any]]] = {}
    for task_key, t_fact in tasks_map.items():
        s = t_fact.get("actual_start")
        e = t_fact.get("actual_end")
        st = t_fact.get("station_id")
        if s is None or e is None or st is None:
            continue
        station_tasks.setdefault(int(st), []).append(t_fact)

    for st, t_list in station_tasks.items():
        active_ends: list[float] = []
        positive_intervals = [
            t for t in t_list if float(t["actual_end"]) > float(t["actual_start"]) + tolerance
        ]
        ordered = sorted(positive_intervals, key=lambda x: (float(x["actual_start"]), float(x["actual_end"])))
        for t in ordered:
            s_val = float(t["actual_start"])
            e_val = float(t["actual_end"])
            while active_ends and active_ends[0] <= s_val + tolerance:
                heapq.heappop(active_ends)
            if len(active_ends) >= 3:
                violations["station_overflow"].append({
                    "station": st,
                    "time": s_val,
                    "concurrency": len(active_ends) + 1,
                    "task": t["task_key"],
                })
                break
            heapq.heappush(active_ends, e_val)

    # 3. Craft Precedence Integrity
    for task_key, t_fact in tasks_map.items():
        s = t_fact.get("actual_start")
        if s is None:
            continue
        base_t = baseline_tasks.get(task_key, {})
        preds = base_t.get("predecessors", [])
        ac_id = t_fact.get("aircraft_id")
        for p in preds:
            pred_key = f"{ac_id}_{p}"
            pred_fact = tasks_map.get(pred_key)
            if pred_fact is None:
                continue
            pred_end = pred_fact.get("actual_end")
            if pred_end is not None and float(s) < float(pred_end) - tolerance:
                violations["precedence_inversion"].append({
                    "task": task_key,
                    "actual_start": float(s),
                    "predecessor": pred_key,
                    "predecessor_actual_end": float(pred_end),
                    "inversion": float(pred_end) - float(s),
                })

    # 4. Station Dwell Interval Legality
    for task_key, t_fact in tasks_map.items():
        s = t_fact.get("actual_start")
        e = t_fact.get("actual_end")
        st = t_fact.get("station_id")
        ac_id = str(t_fact.get("aircraft_id"))
        if s is None or e is None or st is None:
            continue
        ac_info = aircraft_map.get(ac_id, {})
        entry_times = ac_info.get("entry_times", {})
        exit_times = ac_info.get("exit_times", {})
        st_str = str(st)
        if st_str in entry_times and st_str in exit_times:
            entry_t = float(entry_times[st_str])
            exit_t = float(exit_times[st_str])
            if float(s) < entry_t - tolerance:
                violations["station_interval_violation"].append({
                    "task": task_key,
                    "aircraft": ac_id,
                    "station": st,
                    "actual_start": float(s),
                    "station_entry": entry_t,
                })
            if float(e) > exit_t + tolerance:
                violations["station_interval_violation"].append({
                    "task": task_key,
                    "aircraft": ac_id,
                    "station": st,
                    "actual_end": float(e),
                    "station_exit": exit_t,
                })

    # 5. Independent Cost Recalculation and Reconciliation
    # Weights defined in ObjectiveWeights:
    # w_h = 1.0, w_t = 0.20, w_w = 0.05, w_p = 1.0
    w_h = 1.0
    w_t = 0.20
    w_w = 0.05
    w_p = 1.0

    # 5a. J_takt
    recalc_j_takt = 0.0
    sorted_cycle_ids = sorted([int(q) for q in cycle_map.keys()])
    transfer_history: list[float] = [0.0]
    for q in sorted_cycle_ids:
        rec = cycle_map[str(q)]
        p_q = float(rec["transfer_time"])
        p_prev = float(rec["cycle_start_time"])
        duration = p_q - p_prev
        overdue = max(0.0, duration - baseline_h0)
        takt_penalty = overdue / baseline_h0
        recalc_j_takt += w_h * takt_penalty
        transfer_history.append(p_q)

    # 5b. D_time and D_team
    d_time_sum = 0.0
    d_team_sum = 0.0
    total_tasks_count = len(tasks_map)
    scale = 1.0 / total_tasks_count if total_tasks_count > 0 else 0.0

    for task_key, t_fact in tasks_map.items():
        s = t_fact.get("actual_start")
        if s is None:
            continue
        s_val = float(s)

        cycle_start_time = t_fact.get("cycle_start_time")
        if cycle_start_time is not None:
            p_cycle_start = float(cycle_start_time)
        else:
            p_cycle_start = 0.0
            for p in transfer_history:
                if p <= s_val + 1e-5:
                    p_cycle_start = p
                else:
                    break

        b_ki = s_val - p_cycle_start
        base_t = baseline_tasks.get(task_key, {})
        b_i0 = float(base_t.get("in_station_offset", 0.0))
        d_time_sum += abs(b_ki - b_i0) / baseline_h0

        # Team change
        w_actual = set(int(w) for w in t_fact.get("team", []))
        w_base = set(int(w) for w in base_t.get("team", []))
        demand = len(w_base)
        if demand > 0:
            overlap = len(w_actual & w_base)
            d_team_sum += 1.0 - (overlap / demand)

    recalc_d_time = w_t * (d_time_sum * scale)
    recalc_d_team = w_w * (d_team_sum * scale)

    # 5c. J_postpone
    postpone_total = sum(int(t_fact.get("postpone_count", 0)) for t_fact in tasks_map.values())
    recalc_j_postpone = w_p * postpone_total

    # 5d. J_revision
    rev_cost_sum = 0.0
    for task_key, t_fact in tasks_map.items():
        for r in t_fact.get("revision_history", []):
            rev_cost_sum += float(r.get("revision_cost", 0.0))
    recalc_j_revision = rev_cost_sum

    # 5e. J_total
    recalc_j_total = recalc_j_takt + recalc_d_time + recalc_d_team + recalc_j_postpone + recalc_j_revision

    # Reconciliation with reported values
    reported_j_takt = float(traj_data.get("j_takt", 0.0))
    reported_d_time = float(traj_data.get("d_time", 0.0))
    reported_d_team = float(traj_data.get("d_team", 0.0))
    reported_j_postpone = float(traj_data.get("j_postpone", 0.0))
    reported_j_revision = float(traj_data.get("j_revision", 0.0))
    reported_j_total = float(traj_data.get("objective_j_total", traj_data.get("j_total", 0.0)))

    reconciliation_diffs = {
        "j_takt_diff": abs(recalc_j_takt - reported_j_takt),
        "d_time_diff": abs(recalc_d_time - reported_d_time),
        "d_team_diff": abs(recalc_d_team - reported_d_team),
        "j_postpone_diff": abs(recalc_j_postpone - reported_j_postpone),
        "j_revision_diff": abs(recalc_j_revision - reported_j_revision),
        "j_total_diff": abs(recalc_j_total - reported_j_total),
    }

    for comp_name, diff_val in reconciliation_diffs.items():
        if diff_val > cost_tolerance:
            violations["cost_reconciliation_mismatch"].append({
                "component": comp_name,
                "difference": diff_val,
                "tolerance": cost_tolerance,
            })

    total_violations = sum(len(v) for v in violations.values())
    status = "PASS" if total_violations == 0 else "FAIL"

    return {
        "trajectory_file": traj_path.name,
        "status": status,
        "total_violations": total_violations,
        "violations": violations,
        "recalculated_costs": {
            "j_takt": recalc_j_takt,
            "d_time": recalc_d_time,
            "d_team": recalc_d_team,
            "j_postpone": recalc_j_postpone,
            "j_revision": recalc_j_revision,
            "j_total": recalc_j_total,
        },
        "reported_costs": {
            "j_takt": reported_j_takt,
            "d_time": reported_d_time,
            "d_team": reported_d_team,
            "j_postpone": reported_j_postpone,
            "j_revision": reported_j_revision,
            "j_total": reported_j_total,
        },
        "reconciliation_differences": reconciliation_diffs,
    }


def verify_all_trajectories(
    traj_dir: Path,
    baseline_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    baseline_h0, baseline_tasks, _ = load_baseline_tasks(baseline_path)
    scenarios = ["NOMINAL", "RULE_VAL_S1_MODERATE", "RULE_VAL_S2_STRONG"]

    # Select unique 6 trajectories: 3 for A, 3 for C
    found_files: list[Path] = []
    for sc in scenarios:
        for prefix in ["Baseline-Repair-A", "A"]:
            p = traj_dir / f"{prefix}_{sc}.json"
            if p.is_file():
                found_files.append(p)
                break
        for prefix in ["Method-C", "C"]:
            p = traj_dir / f"{prefix}_{sc}.json"
            if p.is_file():
                found_files.append(p)
                break

    if not found_files:
        raise FileNotFoundError(f"No valid trajectory files found in {traj_dir}")

    logger.info(f"=== Starting Independent Trajectory Facts & Cost Verification on {len(found_files)} files ===")
    results: list[dict[str, Any]] = []
    all_pass = True

    for f_path in found_files:
        res = verify_single_trajectory(
            traj_path=f_path,
            baseline_h0=baseline_h0,
            baseline_tasks=baseline_tasks,
        )
        results.append(res)
        logger.info(
            f"  [{res['status']}] {f_path.name:<32} -> Violations={res['total_violations']}, "
            f"MaxDiff={max(res['reconciliation_differences'].values()):.3e}"
        )
        if res["status"] != "PASS":
            all_pass = False

    summary = {
        "overall_status": "PASS" if all_pass else "FAIL",
        "total_trajectories_verified": len(results),
        "all_pass": all_pass,
        "trajectories": results,
    }

    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Verification report saved to {output_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent Trajectory Facts & Cost Verifier")
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=REPO_ROOT / "work3_4060/rule_baseline_ac_v1/trajectories",
        help="Directory with trajectory files",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO_ROOT / "work3_4060/rule_baseline_ac_v1/mini210_rule_baseline.json",
        help="Path to rule baseline JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "work3_4060/rule_baseline_ac_v1/trajectory_verification.json",
        help="Output report JSON",
    )
    args = parser.parse_args()

    summary = verify_all_trajectories(
        traj_dir=args.traj_dir,
        baseline_path=args.baseline,
        output_path=args.output,
    )

    if not summary["all_pass"]:
        logger.error("Trajectory verification FAILED!")
        sys.exit(1)
    else:
        logger.info("All trajectory physical and financial hard contracts STRICTLY PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
