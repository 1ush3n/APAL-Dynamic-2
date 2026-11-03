"""
Script: reproduce_station3_clue.py
Diagnoses and re-verifies the physical Station 3 bottleneck clue:
Tasks: 12, 13, 27, 28, 29 on Workers: 0, 47, 50, 51, 66, 76.
Skill 4 is rare (only workers 0, 66 have it, needed by task 27).
Compares baseline dispatching vs candidate improved dispatching.
Evaluates both under nominal_anchored_v1 (fixed H0=52.0378h) and actual_cleared_v0 / calibrated H0=49.4424h.
Runs independent trajectory physical verifier verify_single_trajectory.
"""

import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from work3_4060.diagnostics.postpone_compatible_agent import PostponeCompatibleRepairAgent
from work3_4060.run_rule_ac_comparison import extract_execution_facts
from work3_4060.verify_trajectory_facts import (
    load_baseline_tasks,
    verify_single_trajectory,
)

BASELINE_PATH = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"

class ImprovedStation3Agent:
    """Agent that specifically fixes the rare-skill bottleneck on Station 2 (physical Station 3).
    
    Candidate dispatching on Station 2:
      - 27 on [0, 66] at entry
      - 12 on [47, 76] at entry
      - 29 on [50, 51] at entry
      - 13 on [47, 76] after 12
      - 28 on [47, 76] after 13
    On all other stations: standard baseline dispatching.
    """
    def __init__(self):
        self.default_agent = PostponeCompatibleRepairAgent()

    def select_action(self, env: AirLineEnvWork3):
        if env._check_terminated():
            return None

        candidates = [
            task for task in env.get_action_candidates()
            if task.status != TaskStatus.RESERVED and env.can_reserve(task)
        ]
        if not candidates:
            return {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}

        # Check if there are candidate tasks in Station 2 (physical Station 3)
        st2_candidates = [t for t in candidates if t.current_station == 2]
        if st2_candidates:
            # Desired priority on station 2: 27, then 12, then 29, then 13, then 28
            # Teams:
            # 27 -> (0, 66)
            # 12 -> (47, 76)
            # 29 -> (50, 51)
            # 13 -> (47, 76)
            # 28 -> (47, 76)
            custom_plan = {
                27: (0, 66),
                12: (47, 76),
                29: (50, 51),
                13: (47, 76),
                28: (47, 76),
            }
            # Try to dispatch in order: 27, 12, 29, 13, 28
            priority_order = [27, 12, 29, 13, 28]
            by_tid = {t.task_id: t for t in st2_candidates}
            for target_tid in priority_order:
                if target_tid in by_tid:
                    task = by_tid[target_tid]
                    team = custom_plan[target_tid]
                    # Verify team validity at current moment
                    try:
                        env._validate_team_for_task(task, team)
                        start = self.default_agent._earliest_start(env, task, team)
                        if start is not None:
                            return {
                                "task_key": task.task_key,
                                "branch": ActionBranch.STATION_EXECUTE,
                                "team": team,
                                "align": 0,
                            }
                    except Exception:
                        pass

        # Otherwise fallback to default agent
        return self.default_agent.select_action(env)


def run_simulation(transfer_policy: str = "nominal_anchored_v1", use_improved: bool = False, baseline_h0_override: float = None):
    env = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH), transfer_policy=transfer_policy)
    env.reset()
    if baseline_h0_override is not None:
        env.state.h0 = baseline_h0_override

    agent = ImprovedStation3Agent() if use_improved else PostponeCompatibleRepairAgent()
    step = 0
    max_steps = 2000
    while not env._check_terminated() and step < max_steps:
        action = agent.select_action(env)
        if action is None:
            break
        env.step(action)
        step += 1

    facts = extract_execution_facts(env)
    
    # Calculate physical metrics directly from execution facts
    tasks_exec = list(facts["tasks_execution"].values())
    cycles_exec = list(facts["cycle_execution_records"].values())
    aircraft_exec = facts["aircraft_execution"]

    makespan = max(t["actual_end"] for t in tasks_exec if t["actual_end"] is not None)
    aircraft_exit_times = [
        aircraft_exec[str(k)]["exit_times"].get("4", aircraft_exec[str(k)]["exit_times"].get(4, None))
        for k in range(len(aircraft_exec))
    ]
    
    # Station clearance times per cycle
    # In Station 2 (physical Station 3):
    st2_tasks = [t for t in tasks_exec if t["station_id"] == 2]
    # Group by aircraft_id
    st2_by_ac = {}
    for t in st2_tasks:
        ac_id = t["aircraft_id"]
        if ac_id not in st2_by_ac:
            st2_by_ac[ac_id] = []
        st2_by_ac[ac_id].append(t)

    st2_clearance_durations = {}
    for ac_id, t_list in st2_by_ac.items():
        ac_entry = aircraft_exec[str(ac_id)]["entry_times"].get("2", aircraft_exec[str(ac_id)]["entry_times"].get(2, 0.0))
        st_end = max(t["actual_end"] for t in t_list)
        st2_clearance_durations[ac_id] = st_end - ac_entry

    # Cycle transfer times
    cycle_transfers = [c["transfer_time"] for c in cycles_exec]
    cycle_overdues = [c["delay_relative_to_nominal"] for c in cycles_exec]

    # Calculate worker busy hours and utilization
    worker_busy = {w: 0.0 for w in range(25)}
    for t in tasks_exec:
        dur = t["actual_end"] - t["actual_start"]
        for w in t["team"]:
            worker_busy[w] = worker_busy.get(w, 0.0) + dur

    # Station 2 workers: [0, 47, 50, 51, 66, 76]
    st2_workers = [0, 47, 50, 51, 66, 76]
    st2_worker_busy = {w: worker_busy.get(w, 0.0) for w in st2_workers}

    # Verify with independent verifier
    traj_dict = {
        "scenario_id": "NOMINAL",
        "objective_j_total": float(env.cumulative_cost),
        "j_total": float(env.cumulative_cost),
        "j_takt": float(env.cost_takt),
        "d_time": float(env.cost_time),
        "d_team": float(env.cost_team),
        "j_postpone": float(env.cost_postpone),
        "j_revision": float(env.cost_revision),
        "actual_disturbance_hits": 0,
        "execution_facts": facts,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(traj_dict, f, indent=2)
        tmp_name = f.name
    try:
        h0, baseline_tasks, _ = load_baseline_tasks(BASELINE_PATH)
        eval_h0 = baseline_h0_override if baseline_h0_override is not None else h0
        res = verify_single_trajectory(Path(tmp_name), eval_h0, baseline_tasks)
        verify_status = res["status"]
        violations = sum(len(v) for v in res["violations"].values())
    finally:
        Path(tmp_name).unlink(missing_ok=True)

    return {
        "use_improved": use_improved,
        "transfer_policy": transfer_policy,
        "h0": env.state.h0,
        "makespan": makespan,
        "total_steps": step,
        "completed_tasks": len(tasks_exec),
        "aircraft_exit_times": aircraft_exit_times,
        "st2_clearance_durations": st2_clearance_durations,
        "cycle_transfers": cycle_transfers,
        "cycle_overdues": cycle_overdues,
        "total_takt_overdue": sum(cycle_overdues),
        "st2_worker_busy": st2_worker_busy,
        "cumulative_cost": env.cumulative_cost,
        "cost_takt": env.cost_takt,
        "cost_team": env.cost_team,
        "cost_time": env.cost_time,
        "verify_status": verify_status,
        "verify_violations": violations,
    }

def main():
    print("=" * 80)
    print("RE-VERIFYING STATION 3 BOTTLENECK CLUE ACROSS REGIMES")
    print("=" * 80)

    # 1. Baseline Replay under nominal_anchored_v1 (H0 = 52.037844h)
    print("\n1. Running BASELINE under nominal_anchored_v1 (H0=52.0378h)...")
    r_base_nom = run_simulation(transfer_policy="nominal_anchored_v1", use_improved=False)
    print(f"   Makespan: {r_base_nom['makespan']:.4f}h, Verify: {r_base_nom['verify_status']} (violations={r_base_nom['verify_violations']})")
    print(f"   St3 clearance duration ac0: {r_base_nom['st2_clearance_durations'][0]:.6f}h")
    print(f"   Exits: {[round(x, 2) for x in r_base_nom['aircraft_exit_times']]}")

    # 2. Improved Station 3 under nominal_anchored_v1 (fixed H0 = 52.037844h)
    print("\n2. Running IMPROVED Station 3 under nominal_anchored_v1 (fixed H0=52.0378h)...")
    r_imp_nom = run_simulation(transfer_policy="nominal_anchored_v1", use_improved=True)
    print(f"   Makespan: {r_imp_nom['makespan']:.4f}h, Verify: {r_imp_nom['verify_status']} (violations={r_imp_nom['verify_violations']})")
    print(f"   St3 clearance duration ac0: {r_imp_nom['st2_clearance_durations'][0]:.6f}h")
    print(f"   St3 clearing delta: {r_imp_nom['st2_clearance_durations'][0] - r_base_nom['st2_clearance_durations'][0]:.6f}h")
    print(f"   Exits: {[round(x, 2) for x in r_imp_nom['aircraft_exit_times']]}")

    # 3. Improved Station 3 with calibrated H0 = 49.442367h
    print("\n3. Running IMPROVED Station 3 with calibrated H0 = 49.442367h...")
    calibrated_h0 = 49.44236659895083
    r_imp_calib = run_simulation(transfer_policy="nominal_anchored_v1", use_improved=True, baseline_h0_override=calibrated_h0)
    print(f"   Makespan: {r_imp_calib['makespan']:.4f}h, Verify: {r_imp_calib['verify_status']} (violations={r_imp_calib['verify_violations']})")
    print(f"   Exits: {[round(x, 2) for x in r_imp_calib['aircraft_exit_times']]}")
    print(f"   Makespan saving: {r_base_nom['makespan'] - r_imp_calib['makespan']:.4f}h ({(r_base_nom['makespan'] - r_imp_calib['makespan'])/r_base_nom['makespan']*100:.2f}%)")

    # 4. Asynchronous / Actual-cleared regime comparison
    print("\n4. Running Baseline vs Improved under actual_cleared_v0 (pull line)...")
    r_base_act = run_simulation(transfer_policy="actual_cleared_v0", use_improved=False)
    r_imp_act = run_simulation(transfer_policy="actual_cleared_v0", use_improved=True)
    print(f"   Baseline actual_cleared: Makespan = {r_base_act['makespan']:.4f}h, St3 clear = {r_base_act['st2_clearance_durations'][0]:.6f}h")
    print(f"   Improved actual_cleared: Makespan = {r_imp_act['makespan']:.4f}h, St3 clear = {r_imp_act['st2_clearance_durations'][0]:.6f}h")
    print(f"   Makespan saving in actual_cleared: {r_base_act['makespan'] - r_imp_act['makespan']:.4f}h")

    # Save detailed JSON output
    out_dir = REPO_ROOT / "work3_4060" / "diagnostics" / "summary_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "baseline_nominal_anchored": r_base_nom,
        "improved_nominal_anchored_fixed_h0": r_imp_nom,
        "improved_nominal_anchored_calibrated_h0": r_imp_calib,
        "baseline_actual_cleared": r_base_act,
        "improved_actual_cleared": r_imp_act,
    }
    with open(REPO_ROOT / "work3_4060" / "diagnostics" / "station3_clue_reproduction.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("\nResults saved to work3_4060/diagnostics/station3_clue_reproduction.json")

if __name__ == "__main__":
    main()
