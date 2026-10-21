"""Independent Trajectory Physical Facts and Cost Reconciler.

Strictly verifies that archived trajectory files satisfy:
1. Manifest and schema integrity:
   - All 6 canonical method x scenario combinations must be present (A/C x Nominal/Val S1/Val S2).
   - Any alias files must strictly match their canonical file SHA256.
   - All 210 tasks must be present (completed, finite timestamps, valid team/station).
   - All 7 aircraft and 11 cycles must be complete and valid.
   - Finite value contracts: any NaN or Inf immediately fails.
2. Hard physical constraints:
   - Worker exclusivity: zero overlapping intervals per worker (tolerance 1e-7h).
   - Station capacity: at most 3 concurrent running tasks (tolerance 1e-7h).
   - Craft precedence: all DAG predecessor tasks complete before successor starts.
   - Station dwell interval: tasks start and end within aircraft's dwell interval at that station.
3. Scenario binding:
   - Nominal scenarios must have 0 disturbance hits.
   - Disturbance scenarios must have actual disturbance hits >= 1, and target task actual start >= recovery time R.
4. Independent financial and cost reconciliation (within 1e-9):
   - J_takt: w_h * sum(overdue / H0) from cycle records (w_h = 1.0)
   - D_time: (w_t / N) * sum(|b_ki - b_i0| / H0) (w_t = 0.20, N = 210)
   - D_team: (w_w / N) * sum(1 - overlap / demand) (w_w = 0.05, N = 210)
   - J_postpone: w_p * sum(lambda_1 * n + lambda_2 * max(0, n - 1)) (w_p = 1.0, lambda_1 = 0.10, lambda_2 = 0.20)
   - J_revision: independently recalculated from before/after diffs (w_rev_t = 0.05, w_rev_w = 0.05, N = 210)
   - J_total: sum of components, matches reported within 1e-9.
   - Step rewards: sum(rewards) + J_total == 0 (tolerance 1e-6).

Exit Codes:
  0 = PASS
  2 = FAIL (prioritized over INCOMPLETE)
  3 = INCOMPLETE
"""

from __future__ import annotations

import argparse
import hashlib
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

from utils.work3.objective_evaluator import ObjectiveWeights, calculate_postpone_penalty

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VerifyTrajectoryFacts")

PHYSICAL_TIME_TOLERANCE = 1e-7
COST_TOLERANCE = 1e-9
REWARD_SUM_TOLERANCE = 1e-6
FIXED_TASK_COUNT = 210

CANONICAL_MANIFEST = [
    {"agent": "A", "scenario_id": "NOMINAL", "filename": "A_NOMINAL.json", "alias": "Baseline-Repair-A_NOMINAL.json"},
    {"agent": "A", "scenario_id": "RULE_VAL_S1_MODERATE", "filename": "A_RULE_VAL_S1_MODERATE.json", "alias": "Baseline-Repair-A_RULE_VAL_S1_MODERATE.json"},
    {"agent": "A", "scenario_id": "RULE_VAL_S2_STRONG", "filename": "A_RULE_VAL_S2_STRONG.json", "alias": "Baseline-Repair-A_RULE_VAL_S2_STRONG.json"},
    {"agent": "C", "scenario_id": "NOMINAL", "filename": "C_NOMINAL.json", "alias": "Method-C_NOMINAL.json"},
    {"agent": "C", "scenario_id": "RULE_VAL_S1_MODERATE", "filename": "C_RULE_VAL_S1_MODERATE.json", "alias": "Method-C_RULE_VAL_S1_MODERATE.json"},
    {"agent": "C", "scenario_id": "RULE_VAL_S2_STRONG", "filename": "C_RULE_VAL_S2_STRONG.json", "alias": "Method-C_RULE_VAL_S2_STRONG.json"},
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_baseline_tasks(baseline_path: Path) -> tuple[float, dict[str, Any], dict[str, Any]]:
    with baseline_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    h0 = float(data["metadata"]["h0"])
    tasks = data["tasks"]
    metadata = data["metadata"]
    return h0, tasks, metadata


def load_scenarios_pool(scenarios_path: Path | None) -> dict[str, dict[str, Any]]:
    if scenarios_path is None or not scenarios_path.is_file():
        # Fallback default path
        scenarios_path = REPO_ROOT / "work3_4060/rule_baseline_ac_v1/scenarios.json"
    if not scenarios_path.is_file():
        return {}
    with scenarios_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {s["scenario_id"]: s for s in data}


def verify_single_trajectory(
    traj_path: Path,
    baseline_h0: float,
    baseline_tasks: dict[str, Any],
    scenario_spec: dict[str, Any] | None = None,
    weights: ObjectiveWeights | None = None,
    physical_tolerance: float = PHYSICAL_TIME_TOLERANCE,
    cost_tolerance: float = COST_TOLERANCE,
) -> dict[str, Any]:
    if weights is None:
        weights = ObjectiveWeights()

    with traj_path.open("r", encoding="utf-8") as f:
        traj_data = json.load(f)

    exec_facts = traj_data.get("execution_facts", {})
    if not exec_facts:
        tasks_map = traj_data.get("tasks_execution")
        aircraft_map = traj_data.get("aircraft_execution")
        cycle_map = traj_data.get("cycle_execution_records")
        step_rewards = traj_data.get("step_rewards")
    else:
        tasks_map = exec_facts.get("tasks_execution")
        aircraft_map = exec_facts.get("aircraft_execution")
        cycle_map = exec_facts.get("cycle_execution_records")
        step_rewards = exec_facts.get("step_rewards")

    violations: dict[str, list[dict[str, Any]]] = {
        "schema_violation": [],
        "non_finite_value": [],
        "task_set_mismatch": [],
        "aircraft_set_mismatch": [],
        "cycle_set_mismatch": [],
        "step_rewards_violation": [],
        "worker_overlap": [],
        "station_overflow": [],
        "precedence_inversion": [],
        "station_interval_violation": [],
        "scenario_binding_violation": [],
        "revision_history_invalid": [],
        "cost_reconciliation_mismatch": [],
    }

    # --- 1. Schema, Finiteness, and Set Integrity Checks ---
    if not isinstance(tasks_map, dict):
        violations["schema_violation"].append({"error": "tasks_execution missing or not a dict"})
        tasks_map = {}
    if not isinstance(aircraft_map, dict):
        violations["schema_violation"].append({"error": "aircraft_execution missing or not a dict"})
        aircraft_map = {}
    if not isinstance(cycle_map, dict):
        violations["schema_violation"].append({"error": "cycle_execution_records missing or not a dict"})
        cycle_map = {}
    if not isinstance(step_rewards, list):
        violations["schema_violation"].append({"error": "step_rewards missing or not a list"})
        step_rewards = []

    # 1a. Task Set Equality
    expected_task_keys = set(baseline_tasks.keys())
    actual_task_keys = set(tasks_map.keys())
    missing_tasks = expected_task_keys - actual_task_keys
    extra_tasks = actual_task_keys - expected_task_keys
    if missing_tasks:
        violations["task_set_mismatch"].append({
            "missing_count": len(missing_tasks),
            "missing_sample": sorted(list(missing_tasks))[:5],
        })
    if extra_tasks:
        violations["task_set_mismatch"].append({
            "extra_count": len(extra_tasks),
            "extra_sample": sorted(list(extra_tasks))[:5],
        })

    # 1b. Task Attribute and Finiteness Check
    for task_key, t_fact in tasks_map.items():
        if not isinstance(t_fact, dict):
            violations["schema_violation"].append({"task": task_key, "error": "not a dict"})
            continue
        s = t_fact.get("actual_start")
        e = t_fact.get("actual_end")
        dur = t_fact.get("duration")
        status = t_fact.get("status")

        if status != "COMPLETED":
            violations["schema_violation"].append({"task": task_key, "error": f"status is {status}, expected COMPLETED"})

        for name, val in [("actual_start", s), ("actual_end", e), ("duration", dur)]:
            if val is None or not math.isfinite(float(val)):
                violations["non_finite_value"].append({"task": task_key, "field": name, "value": val})

        if s is not None and e is not None and math.isfinite(float(s)) and math.isfinite(float(e)):
            s_val = float(s)
            e_val = float(e)
            if s_val < -physical_tolerance or e_val < s_val - physical_tolerance:
                violations["schema_violation"].append({"task": task_key, "error": f"invalid interval [{s_val}, {e_val}]"})

    # 1c. Aircraft Set Equality (0 to 6)
    expected_aircraft_ids = {str(k) for k in range(7)}
    actual_aircraft_ids = set(aircraft_map.keys())
    if actual_aircraft_ids != expected_aircraft_ids:
        violations["aircraft_set_mismatch"].append({
            "expected": sorted(list(expected_aircraft_ids)),
            "actual": sorted(list(actual_aircraft_ids)),
        })
    else:
        for k_str, ac_info in aircraft_map.items():
            if not isinstance(ac_info, dict):
                violations["schema_violation"].append({"aircraft": k_str, "error": "not a dict"})
                continue
            entry_times = ac_info.get("entry_times", {})
            exit_times = ac_info.get("exit_times", {})
            expected_stations = {str(st) for st in range(5)}
            if set(entry_times.keys()) != expected_stations or set(exit_times.keys()) != expected_stations:
                violations["aircraft_set_mismatch"].append({
                    "aircraft": k_str,
                    "error": "missing stations in entry/exit times",
                })
            for st_str in expected_stations:
                en = entry_times.get(st_str)
                ex = exit_times.get(st_str)
                if en is None or not math.isfinite(float(en)) or ex is None or not math.isfinite(float(ex)):
                    violations["non_finite_value"].append({"aircraft": k_str, "station": st_str, "entry": en, "exit": ex})
                elif float(en) > float(ex) + physical_tolerance:
                    violations["schema_violation"].append({
                        "aircraft": k_str, "station": st_str,
                        "error": f"entry {en} > exit {ex}",
                    })

    # 1d. Cycle Records Completeness (1 to 11)
    expected_cycle_ids = {str(q) for q in range(1, 12)}
    actual_cycle_ids = set(cycle_map.keys())
    if actual_cycle_ids != expected_cycle_ids:
        violations["cycle_set_mismatch"].append({
            "expected": sorted(list(expected_cycle_ids), key=int),
            "actual": sorted(list(actual_cycle_ids), key=lambda x: int(x) if x.isdigit() else 0),
        })
    else:
        # Check cycle contiguity and finiteness
        sorted_qs = sorted([int(q) for q in actual_cycle_ids])
        prev_transfer: float | None = None
        for q in sorted_qs:
            rec = cycle_map[str(q)]
            c_start = rec.get("cycle_start_time")
            t_time = rec.get("transfer_time")
            if c_start is None or not math.isfinite(float(c_start)) or t_time is None or not math.isfinite(float(t_time)):
                violations["non_finite_value"].append({"cycle": q, "cycle_start_time": c_start, "transfer_time": t_time})
                continue
            c_start_val = float(c_start)
            t_time_val = float(t_time)
            if t_time_val < c_start_val - physical_tolerance:
                violations["schema_violation"].append({"cycle": q, "error": f"transfer_time {t_time_val} < cycle_start_time {c_start_val}"})
            if prev_transfer is not None and abs(c_start_val - prev_transfer) > 1e-4:
                violations["schema_violation"].append({
                    "cycle": q,
                    "error": f"cycle_start_time {c_start_val} does not match previous transfer_time {prev_transfer}",
                })
            prev_transfer = t_time_val

    # 1e. Step Rewards Integrity & Ledger
    reported_j_total = float(traj_data.get("objective_j_total", traj_data.get("j_total", 0.0)))
    if not math.isfinite(reported_j_total):
        violations["non_finite_value"].append({"field": "j_total", "value": reported_j_total})

    if not step_rewards:
        violations["step_rewards_violation"].append({"error": "step_rewards is empty"})
    else:
        non_finite_rewards = [r for r in step_rewards if not math.isfinite(float(r))]
        if non_finite_rewards:
            violations["non_finite_value"].append({"field": "step_rewards", "error": f"{len(non_finite_rewards)} non-finite rewards"})
        else:
            rewards_sum = sum(float(r) for r in step_rewards)
            diff_rewards = abs(rewards_sum - (-reported_j_total))
            if diff_rewards > REWARD_SUM_TOLERANCE:
                violations["step_rewards_violation"].append({
                    "error": "sum(step_rewards) != -J_total",
                    "rewards_sum": rewards_sum,
                    "expected_sum": -reported_j_total,
                    "difference": diff_rewards,
                    "tolerance": REWARD_SUM_TOLERANCE,
                })

    # --- 2. Hard Physical Constraints Checks ---
    # 2a. Worker Exclusivity (Zero Overlap)
    worker_tasks: dict[int, list[dict[str, Any]]] = {}
    for task_key, t_fact in tasks_map.items():
        s = t_fact.get("actual_start")
        e = t_fact.get("actual_end")
        if s is None or e is None or not math.isfinite(float(s)) or not math.isfinite(float(e)):
            continue
        for w in t_fact.get("team", []):
            worker_tasks.setdefault(int(w), []).append(t_fact)

    for w, t_list in worker_tasks.items():
        positive_intervals = [
            t for t in t_list if float(t["actual_end"]) > float(t["actual_start"]) + physical_tolerance
        ]
        ordered = sorted(positive_intervals, key=lambda x: (float(x["actual_start"]), float(x["actual_end"])))
        for prev, curr in zip(ordered, ordered[1:]):
            if float(prev["actual_end"]) > float(curr["actual_start"]) + physical_tolerance:
                violations["worker_overlap"].append({
                    "worker": w,
                    "task1": prev["task_key"],
                    "interval1": (float(prev["actual_start"]), float(prev["actual_end"])),
                    "task2": curr["task_key"],
                    "interval2": (float(curr["actual_start"]), float(curr["actual_end"])),
                    "overlap": float(prev["actual_end"]) - float(curr["actual_start"]),
                })

    # 2b. Station Capacity (Max 3 Concurrent Tasks)
    station_tasks: dict[int, list[dict[str, Any]]] = {}
    for task_key, t_fact in tasks_map.items():
        s = t_fact.get("actual_start")
        e = t_fact.get("actual_end")
        st = t_fact.get("station_id")
        if s is None or e is None or st is None:
            continue
        if not math.isfinite(float(s)) or not math.isfinite(float(e)):
            continue
        station_tasks.setdefault(int(st), []).append(t_fact)

    for st, t_list in station_tasks.items():
        active_ends: list[float] = []
        positive_intervals = [
            t for t in t_list if float(t["actual_end"]) > float(t["actual_start"]) + physical_tolerance
        ]
        ordered = sorted(positive_intervals, key=lambda x: (float(x["actual_start"]), float(x["actual_end"])))
        for t in ordered:
            s_val = float(t["actual_start"])
            e_val = float(t["actual_end"])
            while active_ends and active_ends[0] <= s_val + physical_tolerance:
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

    # 2c. Craft Precedence Integrity
    for task_key, t_fact in tasks_map.items():
        s = t_fact.get("actual_start")
        if s is None or not math.isfinite(float(s)):
            continue
        base_t = baseline_tasks.get(task_key, {})
        preds = base_t.get("predecessors", [])
        ac_id = t_fact.get("aircraft_id")
        for p in preds:
            pred_key = f"{ac_id}_{p}"
            pred_fact = tasks_map.get(pred_key)
            if pred_fact is None:
                violations["precedence_inversion"].append({
                    "task": task_key,
                    "predecessor": pred_key,
                    "error": "predecessor task missing from execution facts",
                })
                continue
            pred_end = pred_fact.get("actual_end")
            if pred_end is None or not math.isfinite(float(pred_end)):
                violations["non_finite_value"].append({"task": pred_key, "field": "actual_end", "value": pred_end})
            elif float(s) < float(pred_end) - physical_tolerance:
                violations["precedence_inversion"].append({
                    "task": task_key,
                    "actual_start": float(s),
                    "predecessor": pred_key,
                    "predecessor_actual_end": float(pred_end),
                    "inversion": float(pred_end) - float(s),
                })

    # 2d. Station Dwell Interval Legality
    for task_key, t_fact in tasks_map.items():
        s = t_fact.get("actual_start")
        e = t_fact.get("actual_end")
        st = t_fact.get("station_id")
        ac_id = str(t_fact.get("aircraft_id"))
        if s is None or e is None or st is None:
            continue
        if not math.isfinite(float(s)) or not math.isfinite(float(e)):
            continue
        ac_info = aircraft_map.get(ac_id, {})
        entry_times = ac_info.get("entry_times", {})
        exit_times = ac_info.get("exit_times", {})
        st_str = str(st)
        if st_str not in entry_times or st_str not in exit_times:
            violations["station_interval_violation"].append({
                "task": task_key, "aircraft": ac_id, "station": st,
                "error": "dwell times missing for station",
            })
            continue
        en_val = float(entry_times[st_str])
        ex_val = float(exit_times[st_str])
        if float(s) < en_val - physical_tolerance:
            violations["station_interval_violation"].append({
                "task": task_key, "aircraft": ac_id, "station": st,
                "actual_start": float(s), "station_entry": en_val,
            })
        if float(e) > ex_val + physical_tolerance:
            violations["station_interval_violation"].append({
                "task": task_key, "aircraft": ac_id, "station": st,
                "actual_end": float(e), "station_exit": ex_val,
            })

    # --- 3. Scenario Binding & Disturbance Verification ---
    reported_scen_id = traj_data.get("scenario_id")
    actual_hits = traj_data.get("actual_disturbance_hits", 0)

    if scenario_spec is not None:
        expected_scen_id = scenario_spec.get("scenario_id")
        if reported_scen_id != expected_scen_id:
            violations["scenario_binding_violation"].append({
                "error": f"scenario_id mismatch: reported {reported_scen_id} != expected {expected_scen_id}",
            })
        if expected_scen_id == "NOMINAL":
            if actual_hits != 0:
                violations["scenario_binding_violation"].append({
                    "error": f"NOMINAL scenario reported actual_disturbance_hits={actual_hits}, expected 0",
                })
        else:
            # Disturbance scenario
            if actual_hits < 1:
                violations["scenario_binding_violation"].append({
                    "error": f"disturbance scenario {expected_scen_id} reported actual_disturbance_hits={actual_hits}, expected >= 1",
                })
            recovery_time = scenario_spec.get("recovery_time")
            target_key = scenario_spec.get("target_task_key") or (scenario_spec.get("affected_task_keys") or [None])[0]
            if target_key and target_key in tasks_map and recovery_time is not None:
                tgt_start = tasks_map[target_key].get("actual_start")
                if tgt_start is not None and math.isfinite(float(tgt_start)):
                    if float(tgt_start) < float(recovery_time) - physical_tolerance:
                        violations["scenario_binding_violation"].append({
                            "error": f"target task {target_key} actual_start {tgt_start} < recovery_time {recovery_time}",
                            "target_task": target_key,
                            "actual_start": float(tgt_start),
                            "recovery_time": float(recovery_time),
                        })

    # --- 4. Independent Financial & Cost Recalculation ---
    # 4a. J_takt
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
        recalc_j_takt += weights.w_h * takt_penalty
        transfer_history.append(p_q)

    # 4b. D_time and D_team (Scale fixed to 1 / FIXED_TASK_COUNT)
    d_time_sum = 0.0
    d_team_sum = 0.0
    scale_n = 1.0 / FIXED_TASK_COUNT

    for task_key, t_fact in tasks_map.items():
        s = t_fact.get("actual_start")
        if s is None or not math.isfinite(float(s)):
            continue
        s_val = float(s)

        cycle_start_time = t_fact.get("cycle_start_time")
        if cycle_start_time is not None and math.isfinite(float(cycle_start_time)):
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

        w_actual = set(int(w) for w in t_fact.get("team", []))
        w_base = set(int(w) for w in base_t.get("team", []))
        demand = len(w_base)
        if demand > 0:
            overlap = len(w_actual & w_base)
            d_team_sum += 1.0 - (overlap / demand)

    recalc_d_time = weights.w_t * (d_time_sum * scale_n)
    recalc_d_team = weights.w_w * (d_team_sum * scale_n)

    # 4c. J_postpone with progressive lambda_1 and lambda_2 penalty
    recalc_j_postpone = 0.0
    for task_key, t_fact in tasks_map.items():
        n = int(t_fact.get("postpone_count", 0))
        if n > 0:
            recalc_j_postpone += calculate_postpone_penalty(
                n, lambda_1=weights.lambda_1, lambda_2=weights.lambda_2
            )
    recalc_j_postpone = weights.w_p * recalc_j_postpone

    # 4d. J_revision: Independently recalculated from physical before/after diffs
    w_rev_t = weights.w_t if weights.w_revision_time is None else weights.w_revision_time
    w_rev_w = weights.w_w if weights.w_revision_team is None else weights.w_revision_team

    recalc_j_revision = 0.0
    for task_key, t_fact in tasks_map.items():
        base_t = baseline_tasks.get(task_key, {})
        base_pos = float(base_t.get("in_station_offset", 0.0))
        base_team = list(base_t.get("team", []))
        demand = int(base_t.get("demand", len(base_team)))

        rev_history = t_fact.get("revision_history", [])
        if not rev_history:
            continue

        prev_after: dict[str, Any] | None = None
        for idx, rev in enumerate(rev_history):
            before = rev.get("before", {})
            after = rev.get("after", {})

            # Anchor check on first revision
            if idx == 0:
                pos_before = before.get("position")
                team_before = before.get("team", [])
                if pos_before is not None and abs(float(pos_before) - base_pos) > 1e-4:
                    violations["revision_history_invalid"].append({
                        "task": task_key, "error": f"first revision before position {pos_before} != base {base_pos}",
                    })
                if set(team_before) != set(base_team):
                    violations["revision_history_invalid"].append({
                        "task": task_key, "error": f"first revision before team {team_before} != base {base_team}",
                    })
            else:
                # Chain continuity check
                if prev_after is not None:
                    if abs(float(before.get("position", 0.0)) - float(prev_after.get("position", 0.0))) > 1e-4:
                        violations["revision_history_invalid"].append({
                            "task": task_key, "revision_idx": idx, "error": "revision chain position discontinuity",
                        })
                    if set(before.get("team", [])) != set(prev_after.get("team", [])):
                        violations["revision_history_invalid"].append({
                            "task": task_key, "revision_idx": idx, "error": "revision chain team discontinuity",
                        })

            prev_after = after

            # Calculate physical deltas
            pos_b = float(before.get("position", 0.0))
            pos_a = float(after.get("position", 0.0))
            t_diff = abs(pos_a - pos_b) / baseline_h0

            team_b = set(before.get("team", []))
            team_a = set(after.get("team", []))
            if demand > 0:
                overlap = len(team_b & team_a)
                team_diff = 1.0 - (overlap / demand)
            else:
                team_diff = 0.0

            item_rev_cost = scale_n * (w_rev_t * t_diff + w_rev_w * team_diff)
            recalc_j_revision += item_rev_cost

    # 4e. J_total
    recalc_j_total = recalc_j_takt + recalc_d_time + recalc_d_team + recalc_j_postpone + recalc_j_revision

    # Reconciliation with reported values
    reported_j_takt = float(traj_data.get("j_takt", 0.0))
    reported_d_time = float(traj_data.get("d_time", 0.0))
    reported_d_team = float(traj_data.get("d_team", 0.0))
    reported_j_postpone = float(traj_data.get("j_postpone", 0.0))
    reported_j_revision = float(traj_data.get("j_revision", 0.0))

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
    scenarios_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    baseline_h0, baseline_tasks, _ = load_baseline_tasks(baseline_path)
    scenarios_pool = load_scenarios_pool(scenarios_path)

    results: list[dict[str, Any]] = []
    missing_manifest_files: list[str] = []
    alias_mismatches: list[dict[str, str]] = []

    has_any_fail = False

    logger.info(f"=== Starting Independent Trajectory Facts & Cost Verification on manifest ===")

    for item in CANONICAL_MANIFEST:
        c_name = item["filename"]
        alias_name = item["alias"]
        c_path = traj_dir / c_name
        alias_path = traj_dir / alias_name

        if not c_path.is_file():
            missing_manifest_files.append(c_name)
            logger.error(f"Required canonical file missing: {c_name}")
            continue

        # If alias file exists, check content equality with canonical file
        if alias_path.is_file():
            h_c = sha256_file(c_path)
            h_a = sha256_file(alias_path)
            if h_c != h_a:
                alias_mismatches.append({
                    "canonical": c_name, "alias": alias_name,
                    "canonical_sha256": h_c, "alias_sha256": h_a,
                })
                logger.error(f"Alias hash mismatch: {alias_name} != {c_name}")

        # Bind scenario
        sc_id = item["scenario_id"]
        sc_spec = {"scenario_id": "NOMINAL"} if sc_id == "NOMINAL" else scenarios_pool.get(sc_id)

        res = verify_single_trajectory(
            traj_path=c_path,
            baseline_h0=baseline_h0,
            baseline_tasks=baseline_tasks,
            scenario_spec=sc_spec,
        )
        results.append(res)
        logger.info(
            f"  [{res['status']}] {c_name:<30} -> Violations={res['total_violations']}, "
            f"MaxDiff={max(res['reconciliation_differences'].values()):.3e}"
        )
        if res["status"] != "PASS":
            has_any_fail = True

    if alias_mismatches:
        has_any_fail = True

    # Priority: FAIL (exit 2) > INCOMPLETE (exit 3) > PASS (exit 0)
    if has_any_fail:
        overall_status = "FAIL"
        exit_code = 2
    elif missing_manifest_files or len(results) < len(CANONICAL_MANIFEST):
        overall_status = "INCOMPLETE"
        exit_code = 3
    else:
        overall_status = "PASS"
        exit_code = 0

    summary = {
        "overall_status": overall_status,
        "exit_code": exit_code,
        "total_manifest_items": len(CANONICAL_MANIFEST),
        "verified_canonical_count": len(results),
        "missing_manifest_files": missing_manifest_files,
        "alias_mismatches": alias_mismatches,
        "all_pass": overall_status == "PASS",
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
        "--scenarios",
        type=Path,
        default=REPO_ROOT / "work3_4060/rule_baseline_ac_v1/scenarios.json",
        help="Path to scenarios JSON",
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
        scenarios_path=args.scenarios,
        output_path=args.output,
    )

    if summary["exit_code"] == 0:
        logger.info("All trajectory physical and financial hard contracts STRICTLY PASSED! (exit 0)")
        sys.exit(0)
    elif summary["exit_code"] == 3:
        logger.warning("Trajectory verification INCOMPLETE due to missing files/data! (exit 3)")
        sys.exit(3)
    else:
        logger.error(f"Trajectory verification FAILED with violations! (exit 2)")
        sys.exit(2)


if __name__ == "__main__":
    main()
