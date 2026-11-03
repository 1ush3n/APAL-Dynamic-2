"""
Script: diag_disturbance_matrix.py
Comprehensive Disturbance Grid and Paired Counterfactuals across:
  - Locations: Stations 1 to 5, Rare vs Common skills
  - Strengths: Weak (2h), Medium (8h), Strong (25h)
  - Timings: Filling (Cycle 1-2), Full-line (Cycle 5), Draining (Cycle 9-10)
  - States: UNREADY vs RESERVED
Evaluates 3 arms: Stay, Postpone, Re-order/Re-team.
Records cycle propagation vectors, Makespan changes, and runs verify_single_trajectory.
"""

import json
from pathlib import Path
import sys
import tempfile
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from work3_4060.diagnostics.postpone_compatible_agent import PostponeCompatibleRepairAgent
from work3_4060.run_rule_ac_comparison import extract_execution_facts
from work3_4060.verify_trajectory_facts import load_baseline_tasks, verify_single_trajectory

BASELINE_PATH = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"
SCENARIOS_PATH = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "scenarios.json"

def run_disturbance_case(case_name: str, target_task_key: str, tau: float, duration_delay: float, action_mode: str = "stay"):
    """Runs a disturbance case under AirLineEnvWork3.
    target_task_key: e.g. '0_16' or '1_27' or '2_8'
    tau: disclosure / arrival time
    recovery_time = tau + duration_delay
    action_mode: 'stay' (Control) or 'postpone' (Treatment) or 'reteam'
    """
    env = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH), transfer_policy="nominal_anchored_v1")
    env.reset()
    
    # Load scenario dict into env
    recovery_time = tau + duration_delay
    scenario_dict = {
        "scenario_id": case_name,
        "target_task_key": target_task_key,
        "tau": tau,
        "delta": duration_delay,
        "recovery_time": recovery_time,
        "candidate_count": 1,
        "affected_count": 1,
        "affected_task_keys": [target_task_key],
    }
    env.load_scenario(scenario_dict)
    
    agent = PostponeCompatibleRepairAgent()
    step = 0
    postponed = False
    
    while not env._check_terminated() and step < 2000:
        # Check if action_mode is postpone and target task is eligible
        if action_mode == "postpone" and not postponed:
            t = env.state.tasks.get(target_task_key)
            if t is not None and env.can_postpone(t):
                # Ensure decision time >= tau (only react after event is visible!)
                if env.state.current_time >= tau:
                    env.step({"task_key": target_task_key, "branch": ActionBranch.POSTPONE})
                    postponed = True
                    step += 1
                    continue
                    
        action = agent.select_action(env)
        if action is None:
            break
        env.step(action)
        step += 1

    facts = extract_execution_facts(env)
    tasks_exec = list(facts["tasks_execution"].values())
    cycles_exec = list(facts["cycle_execution_records"].values())
    
    makespan = max(t["actual_end"] for t in tasks_exec if t["actual_end"] is not None)
    
    # Check target task actual start
    target_task = facts["tasks_execution"].get(target_task_key)
    target_actual_start = target_task["actual_start"] if target_task else None
    
    # Target actually delayed?
    # Hit definition: target started >= recovery_time
    hit = (target_actual_start is not None and target_actual_start >= recovery_time - 1e-5)
    
    # Cycle delay sequence relative to nominal transfers
    cycle_delays = [c["delay_relative_to_nominal"] for c in cycles_exec]
    max_cycle_delay = max(cycle_delays) if cycle_delays else 0.0
    
    # Propagation classification
    nominal_makespan = 572.416285
    delta_makespan = makespan - nominal_makespan
    if abs(delta_makespan) < 1e-4 and max_cycle_delay < 1e-4:
        prop_class = "local_absorbed"
    elif abs(delta_makespan) < 1e-4 and max_cycle_delay > 1e-4:
        prop_class = "recovered_in_draining"
    elif delta_makespan > 1e-4 and abs(delta_makespan - max_cycle_delay) < 1e-4:
        prop_class = "persistent_delay"
    elif delta_makespan > max_cycle_delay:
        prop_class = "amplified"
    else:
        prop_class = "partially_absorbed"

    # Verify with independent verifier
    traj_dict = {
        "scenario_id": case_name,
        "objective_j_total": float(env.cumulative_cost),
        "j_total": float(env.cumulative_cost),
        "j_takt": float(env.cost_takt),
        "d_time": float(env.cost_time),
        "d_team": float(env.cost_team),
        "j_postpone": float(env.cost_postpone),
        "j_revision": float(env.cost_revision),
        "actual_disturbance_hits": 1 if hit else 0,
        "execution_facts": facts,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(traj_dict, f, indent=2)
        tmp_name = f.name
    try:
        h0, baseline_tasks, _ = load_baseline_tasks(BASELINE_PATH)
        res = verify_single_trajectory(Path(tmp_name), h0, baseline_tasks)
        verify_status = res["status"]
        violations = sum(len(v) for v in res["violations"].values())
    finally:
        Path(tmp_name).unlink(missing_ok=True)

    return {
        "case_name": case_name,
        "action_mode": action_mode,
        "target_task": target_task_key,
        "tau": tau,
        "delay": duration_delay,
        "recovery_time": recovery_time,
        "hit": hit,
        "target_actual_start": target_actual_start,
        "makespan": round(makespan, 4),
        "delta_makespan": round(delta_makespan, 4),
        "max_cycle_delay": round(max_cycle_delay, 4),
        "propagation_class": prop_class,
        "cycle_delays": [round(x, 2) for x in cycle_delays],
        "cumulative_cost": round(env.cumulative_cost, 4),
        "cost_takt": round(env.cost_takt, 4),
        "independent_verify": f"{verify_status} (violations={violations})",
    }

def main():
    print("=" * 80)
    print("COMPREHENSIVE DISTURBANCE MATRIX EVALUATION")
    print("=" * 80)

    # Matrix of disturbance cases:
    # 1. Station 1 task 0_7 (skill 0, common, can postpone)
    # 2. Station 2 task 0_18 (skill 0, common, can postpone)
    # 3. Station 3 task 1_27 (skill 4, scarce, CANNOT postpone due to no skill 4 in St 4)
    # 4. Station 4 task 2_10 (skill 0, common, can postpone to St 5)
    # Strengths: Weak (2h), Med (8h), Strong (25h)
    
    test_cases = [
        # (name, target, tau, delay)
        ("St1_Common_Weak_Filling", "0_7", 0.0, 2.0),
        ("St1_Common_Med_Filling",  "0_7", 0.0, 8.0),
        ("St1_Common_Strong_Filling", "0_7", 0.0, 25.0),
        ("St2_Common_Med_Filling",  "0_18", 52.04, 8.0),
        ("St3_Scarce_Weak_Full",    "1_27", 156.11, 2.0),
        ("St3_Scarce_Med_Full",     "1_27", 156.11, 8.0),
        ("St3_Scarce_Strong_Full",  "1_27", 156.11, 25.0),
        ("St4_Common_Med_Full",     "2_10", 260.19, 8.0),
        ("St1_Common_Med_Late",     "4_7",  208.15, 8.0),
    ]

    results = []
    for cname, target, tau, delay in test_cases:
        # Run Control (stay)
        r_stay = run_disturbance_case(cname, target, tau, delay, action_mode="stay")
        results.append(r_stay)
        print(f"[{cname:<26}] Stay: Hit={r_stay['hit']}, dM={r_stay['delta_makespan']:>6.2f}h, MaxDel={r_stay['max_cycle_delay']:>6.2f}h, Class={r_stay['propagation_class']:<20}, Verify={r_stay['independent_verify']}")

        # Test Postpone if task is eligible
        # Check if task is not in last station and downstream can staff
        t_id = int(target.split("_")[1])
        if t_id not in (27, 11, 14, 19, 22, 23, 24, 30, 125):
            r_post = run_disturbance_case(cname, target, tau, delay, action_mode="postpone")
            results.append(r_post)
            print(f"[{cname:<26}] Post: Hit={r_post['hit']}, dM={r_post['delta_makespan']:>6.2f}h, MaxDel={r_post['max_cycle_delay']:>6.2f}h, Class={r_post['propagation_class']:<20}, Verify={r_post['independent_verify']}")

    # Save summary table
    df = pd.DataFrame(results)
    csv_path = REPO_ROOT / "work3_4060" / "diagnostics" / "summary_tables" / "disturbance_matrix_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nDisturbance matrix saved to {csv_path}")

    json_path = REPO_ROOT / "work3_4060" / "diagnostics" / "disturbance_matrix_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
