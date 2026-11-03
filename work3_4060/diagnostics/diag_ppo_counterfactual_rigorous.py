"""
Rigorous Counterfactual Audit of PPO ADVANCE Actions.
Checks whether at each ADVANCE decision, there were actually FREE qualified workers
in the candidate task's station that could physically start the task immediately.
"""

import json
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from scripts.work3.evaluate_c_vs_d import build_formal_evaluation_agent
from work3_4060.verify_trajectory_facts import load_baseline_tasks, verify_single_trajectory
from work3_4060.run_rule_ac_comparison import extract_execution_facts

BASELINE_PATH = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"
CHECKPOINT_PATH = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "seed42" / "C" / "weights.pt"

def audit_ppo_advances():
    agent = build_formal_evaluation_agent("C", CHECKPOINT_PATH, device="cpu")
    env = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH), transfer_policy="nominal_anchored_v1")
    env.reset()

    step = 0
    advance_records = []
    
    total_advances = 0
    advances_with_immediate_work_possible = 0
    advances_forced_by_worker_busy = 0
    
    while not env._check_terminated() and step < 1000:
        action, diag = agent.select_action(env, return_diagnostics=True)
        cands = diag.get("candidate_task_keys", ())
        
        is_advance = False
        if action is None or action.get("branch") == ActionBranch.ADVANCE_TO_NEXT_EVENT:
            is_advance = True
            
        if is_advance:
            total_advances += 1
            curr_time = float(env.state.current_time)
            
            # Check station by station
            feasible_immediate_tasks = []
            
            # Find free workers in each station at curr_time
            # A worker is busy if they are assigned to a task that is currently IN_PROGRESS
            # or if the station concurrency limit is reached.
            station_busy_workers = {s: set() for s in range(5)}
            station_active_task_count = {s: 0 for s in range(5)}
            
            for t in env.state.tasks.values():
                if t.status == TaskStatus.RUNNING:
                    st = t.current_station
                    station_active_task_count[st] += 1
                    if hasattr(t, "assigned_team") and t.assigned_team:
                        station_busy_workers[st].update(t.assigned_team)
                        
            # Now evaluate each candidate task
            for t_key in cands:
                task = env.state.tasks[t_key]
                st = task.current_station
                
                # Check concurrency
                if station_active_task_count[st] >= env.max_slots_per_station:
                    continue
                    
                # Check free workers in this station
                st_workers = env.state.station_worker_bindings.get(st, [])
                free_wids = [w for w in st_workers if w not in station_busy_workers[st]]
                
                # Check qualified workers
                req_skill = task.skill
                req_dem = task.demand
                qual_free = [w for w in free_wids if req_skill in env.worker_skills[w]]
                
                if len(qual_free) >= req_dem:
                    feasible_immediate_tasks.append({
                        "task_key": t_key,
                        "station": st,
                        "skill": req_skill,
                        "demand": req_dem,
                        "qualified_free_workers": qual_free
                    })
                    
            if len(feasible_immediate_tasks) > 0:
                advances_with_immediate_work_possible += 1
                verdict = "AVOIDABLE_ADVANCE"
            else:
                advances_forced_by_worker_busy += 1
                verdict = "FORCED_ADVANCE_WORKERS_BUSY"
                
            advance_records.append({
                "step": step,
                "current_time": curr_time,
                "candidate_count": len(cands),
                "verdict": verdict,
                "feasible_immediate_tasks": feasible_immediate_tasks
            })
            step_action = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
        else:
            step_action = action
            
        env.step(step_action)
        step += 1

    facts = extract_execution_facts(env)
    
    # Run trajectory verifier on PPO trajectory
    import tempfile
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
    h0, baseline_tasks, _ = load_baseline_tasks(BASELINE_PATH)
    res = verify_single_trajectory(Path(tmp_name), h0, baseline_tasks)
    
    summary = {
        "total_steps": step,
        "total_advances": total_advances,
        "advances_forced_by_worker_busy": advances_forced_by_worker_busy,
        "advances_with_immediate_work_possible": advances_with_immediate_work_possible,
        "verifier_result": res,
        "sample_advances": advance_records[:30]
    }
    
    out_path = REPO_ROOT / "work3_4060" / "diagnostics" / "ppo_counterfactual_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Total steps: {step}")
    print(f"Total ADVANCE decisions: {total_advances}")
    print(f"  - FORCED (workers busy / slots full): {advances_forced_by_worker_busy} ({advances_forced_by_worker_busy/total_advances*100:.1f}%)")
    print(f"  - AVOIDABLE (qualified workers FREE right now): {advances_with_immediate_work_possible} ({advances_with_immediate_work_possible/total_advances*100:.1f}%)")
    print(f"Trajectory verification: status={res['status']}, violations={len(res.get('violations', []))}")
    print(f"Results saved to {out_path}")
    return summary

if __name__ == "__main__":
    audit_ppo_advances()
