"""Diagnostic script for Task 2 on rule_baseline_ac_v1:
1. Run Nominal + all 6 scenarios in rule_baseline_ac_v1 with BaselinePlanRepairAgent (A).
2. Record complete cycle-by-cycle metrics, station readiness, aircraft exits, and 5-component costs.
3. Verify the strong disturbance counterexample RULE_VAL_S2_STRONG:
   - Cycle 4-9 delay of ~28.63h;
   - Cycle 10-11 recovery during line draining;
   - Final makespan unchanged vs Nominal;
   - J_takt strictly positive due to cycle delays.
4. Categorize each scenario:
   - local_absorbed
   - persistent_delay
   - amplification
   - recovered (draining recovery)
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from envs.work3.environment import AirLineEnvWork3
from scripts.work3.baseline_repair_agent import BaselinePlanRepairAgent
from work3_4060.diagnostics.postpone_compatible_agent import PostponeCompatibleRepairAgent

BASELINE_PATH = repo_root / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"
SCENARIOS_PATH = repo_root / "work3_4060" / "rule_baseline_ac_v1" / "scenarios.json"
OUTPUT_DIR = repo_root / "work3_4060" / "diagnostics" / "task2_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_rule_simulation(
    scenario: dict[str, Any] | None,
) -> dict[str, Any]:
    env = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH))
    env.reset()
    if scenario is not None:
        env.load_scenario(scenario)

    agent = BaselinePlanRepairAgent()
    step_count = 0
    while not env._check_terminated() and step_count < 3000:
        act = agent.select_action(env)
        if act is None:
            break
        env.step(act)
        step_count += 1

    s_id = scenario["scenario_id"] if scenario else "NOMINAL"
    hit_info = env.disturbance_event_results.get(s_id, {})

    ac_exits = {}
    for ac_id, ac in env.state.aircraft.items():
        ac_exits[ac_id] = {str(st): float(t) for st, t in ac.exit_times.items()}

    tasks_timing = {}
    for t_key, t in env.state.tasks.items():
        tasks_timing[t_key] = {
            "start": float(t.actual_start) if t.actual_start is not None else None,
            "end": float(t.actual_end) if t.actual_end is not None else None,
            "station": int(t.current_station),
            "material_ready_time": float(t.material_ready_time),
        }

    # Station first cleared times
    cleared_times = {}
    for c_idx, s_map in getattr(env.state, "station_first_cleared_times", {}).items():
        cleared_times[str(c_idx)] = {str(s): float(t) for s, t in s_map.items()}

    return {
        "scenario_id": s_id,
        "completed": bool(env.state.is_all_completed() if hasattr(env.state, "is_all_completed") else env._check_terminated()),
        "step_count": step_count,
        "makespan": float(env.state.current_time),
        "cumulative_cost": float(env.cumulative_cost),
        "cost_takt": float(env.cost_takt),
        "cost_time": float(env.cost_time),
        "cost_team": float(env.cost_team),
        "cost_postpone": float(env.cost_postpone),
        "cost_revision": float(env.cost_revision),
        "transfer_history": [float(x) for x in env.state.transfer_history],
        "station_first_cleared": cleared_times,
        "aircraft_exits": ac_exits,
        "hit_info": hit_info,
        "tasks_timing": tasks_timing,
    }


def main():
    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    print(f"Loaded {len(scenarios)} rule_baseline_ac_v1 scenarios.")

    # Run Nominal
    nom = run_rule_simulation(None)
    print(f"Rule Baseline Nominal: makespan={nom['makespan']:.6f}h, J_total={nom['cumulative_cost']:.6e}, steps={nom['step_count']}")

    results = {}
    summary_rows = []
    cat_counts = {
        "local_absorbed": 0,
        "persistent_delay": 0,
        "amplification": 0,
        "recovered": 0,
        "unhit": 0,
    }

    for sc in scenarios:
        s_id = sc["scenario_id"]
        res = run_rule_simulation(sc)
        results[s_id] = res

        target_keys = sc.get("affected_task_keys", [])
        target_count = len(target_keys)
        actual_hit_keys = res["hit_info"].get("actual_hit_task_keys", [])
        actual_hit_count = len(actual_hit_keys)

        tau = sc["tau"]
        r = sc["recovery_time"]
        delta_field = sc.get("delta")
        r_minus_tau = r - tau

        delta_m = res["makespan"] - nom["makespan"]
        delta_j = res["cumulative_cost"] - nom["cumulative_cost"]

        # Lower bound change & actual start change for target
        start_shifts = {}
        lower_bound_shifts = {}
        for tk in target_keys:
            nom_start = nom["tasks_timing"][tk]["start"]
            sc_start = res["tasks_timing"][tk]["start"]
            start_shifts[tk] = sc_start - nom_start if nom_start is not None and sc_start is not None else None
            lower_bound_shifts[tk] = res["tasks_timing"][tk]["material_ready_time"] - nom["tasks_timing"][tk]["material_ready_time"]

        # Cycle transfer delays
        transfer_delays = []
        for c_idx, (t_sc, t_nom) in enumerate(zip(res["transfer_history"], nom["transfer_history"])):
            transfer_delays.append(t_sc - t_nom)
        max_transfer_delay = max(transfer_delays) if transfer_delays else 0.0

        # Classification
        if actual_hit_count == 0:
            category = "unhit"
        elif max_transfer_delay <= 1e-4 and abs(delta_m) <= 1e-4:
            category = "local_absorbed"
        elif abs(delta_m) <= 1e-4 and max_transfer_delay > 1e-4:
            category = "recovered"
        elif delta_m > r_minus_tau + 1e-4:
            category = "amplification"
        else:
            category = "persistent_delay"

        cat_counts[category] += 1

        summary_rows.append({
            "scenario_id": s_id,
            "split": sc.get("split"),
            "category": category,
            "target_keys": target_keys,
            "target_count": target_count,
            "actual_hit_count": actual_hit_count,
            "tau": tau,
            "recovery_time": r,
            "delta_field": delta_field,
            "R_minus_tau": r_minus_tau,
            "delta_vs_R_minus_tau_diff": r_minus_tau - delta_field if delta_field is not None else None,
            "lower_bound_shifts": lower_bound_shifts,
            "start_shifts": start_shifts,
            "transfer_delays": transfer_delays,
            "max_transfer_delay": max_transfer_delay,
            "delta_makespan": delta_m,
            "delta_j_total": delta_j,
            "makespan": res["makespan"],
            "cumulative_cost": res["cumulative_cost"],
            "cost_takt": res["cost_takt"],
            "cost_time": res["cost_time"],
            "cost_team": res["cost_team"],
            "cost_postpone": res["cost_postpone"],
            "cost_revision": res["cost_revision"],
        })

        print(f"[{s_id}] hit={actual_hit_count}/{target_count}, R-tau={r_minus_tau:.2f}h, max_trans_delay={max_transfer_delay:.2f}h, delta_M={delta_m:.2f}h, J_takt={res['cost_takt']:.6f}, cat={category}")

    print(f"\nRule Baseline A Category Counts (6 scenarios): {cat_counts}")

    # Focus on RULE_VAL_S2_STRONG details
    s2_strong = [r for r in summary_rows if r["scenario_id"] == "RULE_VAL_S2_STRONG"][0]
    print("\n--- Detailed Cycle Breakdown for RULE_VAL_S2_STRONG ---")
    for c_idx, delay in enumerate(s2_strong["transfer_delays"]):
        p_nom = nom["transfer_history"][c_idx]
        p_sc = results["RULE_VAL_S2_STRONG"]["transfer_history"][c_idx]
        print(f"Cycle {c_idx+1:2d}: Nominal P = {p_nom:8.4f}h, Scen P = {p_sc:8.4f}h, Delay = {delay:+8.4f}h")

    report = {
        "nominal": nom,
        "scenarios_summary": summary_rows,
        "category_counts": cat_counts,
        "detailed_results": results,
    }

    out_file = OUTPUT_DIR / "task2_rule_baseline_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {out_file}")


if __name__ == "__main__":
    main()
