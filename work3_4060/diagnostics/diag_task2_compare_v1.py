"""Diagnostic script for Task 2:
1. Re-run and review all 20 scenarios of compare_v1 + nominal.
2. Evaluate with Baseline A (BaselinePlanRepairAgent) and HeuristicAgentWork3.
3. Compare against recorded compare_v1 results for Method C and D.
4. Record for each scenario:
   - target_count
   - actual_hit_count
   - material_ready_time change (lower bound change)
   - actual start change vs nominal
   - station readiness times (station_first_cleared_times)
   - cycle transfer times (transfer_history)
   - aircraft exit times
   - batch makespan
   - 5 component costs (j_takt, d_time, d_team, j_postpone, j_revision, j_total)
   - R - tau vs delta explanation
   - Propagation category: local absorption, persistent calendar delay, downstream amplification, downstream recovery.
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
from models.work3.heuristic_agent import HeuristicAgentWork3

BASELINE_PATH_COMPARE = repo_root / "work3_4060" / "data" / "mini210.json"
SCENARIOS_PATH_COMPARE = repo_root / "work3_4060" / "compare_v1" / "scenarios.json"
OUTPUT_DIR = repo_root / "work3_4060" / "diagnostics" / "task2_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_agent_on_scenario(
    agent_name: str,
    agent_obj: Any,
    scenario: dict[str, Any] | None,
    baseline_path: Path,
) -> dict[str, Any]:
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    env.reset()
    if scenario is not None:
        env.load_scenario(scenario)

    step_count = 0
    while not env._check_terminated() and step_count < 3000:
        act = agent_obj.select_action(env)
        if act is None:
            break
        env.step(act)
        step_count += 1

    s_id = scenario["scenario_id"] if scenario else "NOMINAL"
    hit_info = env.disturbance_event_results.get(s_id, {})
    
    # Aircraft exit times
    ac_exits = {}
    for ac_id, ac in env.state.aircraft.items():
        ac_exits[ac_id] = {str(st): float(t) for st, t in ac.exit_times.items()}

    # Task actual starts and ends
    tasks_timing = {}
    for t_key, t in env.state.tasks.items():
        tasks_timing[t_key] = {
            "start": float(t.actual_start) if t.actual_start is not None else None,
            "end": float(t.actual_end) if t.actual_end is not None else None,
            "station": int(t.current_station),
            "material_ready_time": float(t.material_ready_time),
        }

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
        "station_first_cleared": {
            str(c): {str(s): float(t) for s, t in s_map.items()}
            for c, s_map in getattr(env.state, "station_first_cleared_times", {}).items()
        },
        "aircraft_exits": ac_exits,
        "hit_info": hit_info,
        "tasks_timing": tasks_timing,
    }


def main():
    with open(SCENARIOS_PATH_COMPARE, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    print(f"Loaded {len(scenarios)} compare_v1 scenarios.")

    # 1. Run BaselinePlanRepairAgent on Nominal + all 20 scenarios
    print("\n--- Running BaselinePlanRepairAgent on compare_v1 (mini210.json) ---")
    agent_a = BaselinePlanRepairAgent()
    nom_a = run_agent_on_scenario("Baseline-Repair-A", agent_a, None, BASELINE_PATH_COMPARE)
    print(f"Baseline A Nominal: makespan={nom_a['makespan']:.4f}h, J_total={nom_a['cumulative_cost']:.6f}, steps={nom_a['step_count']}")

    results_a = {}
    for sc in scenarios:
        s_id = sc["scenario_id"]
        res = run_agent_on_scenario("Baseline-Repair-A", agent_a, sc, BASELINE_PATH_COMPARE)
        results_a[s_id] = res

    # 2. Run HeuristicAgentWork3 on Nominal + all 20 scenarios
    print("\n--- Running HeuristicAgentWork3 on compare_v1 (mini210.json) ---")
    agent_h = HeuristicAgentWork3()
    nom_h = run_agent_on_scenario("Heuristic", agent_h, None, BASELINE_PATH_COMPARE)
    print(f"Heuristic Nominal: makespan={nom_h['makespan']:.4f}h, J_total={nom_h['cumulative_cost']:.6f}, steps={nom_h['step_count']}")

    results_h = {}
    for sc in scenarios:
        s_id = sc["scenario_id"]
        res = run_agent_on_scenario("Heuristic", agent_h, sc, BASELINE_PATH_COMPARE)
        results_h[s_id] = res

    # 3. Analyze propagation and classify
    # Let's see the comparison for Baseline A
    print("\n=== Analysis for Baseline A on compare_v1 (20 Scenarios) ===")
    summary_rows_a = []
    
    cat_counts_a = {
        "local_absorbed": 0,
        "persistent_delay": 0,
        "amplification": 0,
        "recovered": 0,
        "unhit": 0,
    }

    for sc in scenarios:
        s_id = sc["scenario_id"]
        res = results_a[s_id]
        
        target_keys = sc.get("affected_task_keys", [])
        target_count = len(target_keys)
        actual_hit_keys = res["hit_info"].get("actual_hit_task_keys", [])
        actual_hit_count = len(actual_hit_keys)
        
        tau = sc["tau"]
        r = sc["recovery_time"]
        delta_param = sc.get("delta")
        r_minus_tau = r - tau
        
        delta_m = res["makespan"] - nom_a["makespan"]
        delta_j = res["cumulative_cost"] - nom_a["cumulative_cost"]

        # Check target task actual start shift vs nominal
        start_shifts = {}
        for tk in target_keys:
            nom_start = nom_a["tasks_timing"][tk]["start"]
            sc_start = res["tasks_timing"][tk]["start"]
            start_shifts[tk] = sc_start - nom_start if nom_start is not None and sc_start is not None else None

        # Check transfer history delay relative to nominal
        transfer_delays = []
        for c_idx, (t_sc, t_nom) in enumerate(zip(res["transfer_history"], nom_a["transfer_history"])):
            transfer_delays.append(t_sc - t_nom)
        max_transfer_delay = max(transfer_delays) if transfer_delays else 0.0

        # Classification logic:
        # 1. unhit: actual_hit_count == 0
        # 2. local_absorbed: hit, but max_transfer_delay == 0 and delta_m == 0
        # 3. persistent_delay: transfer delays exist, and final delta_m > 0 (or constant delay persisted to end)
        # 4. recovered: transfer delays exist in intermediate cycles, but delta_m == 0 at batch end
        # 5. amplification: delta_m > r_minus_tau (or delay grew)
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

        cat_counts_a[category] += 1

        summary_rows_a.append({
            "scenario_id": s_id,
            "category": category,
            "target_keys": target_keys,
            "target_count": target_count,
            "actual_hit_count": actual_hit_count,
            "tau": tau,
            "recovery_time": r,
            "delta_field": delta_param,
            "R_minus_tau": r_minus_tau,
            "start_shifts": start_shifts,
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
            "transfer_delays": transfer_delays,
        })
        print(f"[{s_id}] hit={actual_hit_count}/{target_count}, R-tau={r_minus_tau:.2f}h, max_trans_delay={max_transfer_delay:.2f}h, delta_M={delta_m:.2f}h, cat={category}")

    print(f"\nBaseline A Category Counts on compare_v1 (20 scenarios): {cat_counts_a}")

    # Output detailed report
    report = {
        "nominal_a": nom_a,
        "nominal_h": nom_h,
        "scenarios_summary_a": summary_rows_a,
        "category_counts_a": cat_counts_a,
        "detailed_results_a": results_a,
        "detailed_results_h": results_h,
    }

    out_file = OUTPUT_DIR / "task2_compare_v1_evaluation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
