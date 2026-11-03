"""
Paired Two-Arm Counterfactual Continuation Experiment (ADVANCE vs Immediate Dispatch).
Evaluates >= 20 sites across early, middle, and late cycles where immediate dispatch is verified feasible.
Arm A: Execute ADVANCE, then continue with PPO policy to termination.
Arm B: Execute Immediate Dispatch, then continue with PPO policy to termination.
Whole-trajectory Intervention: Greedily dispatch whenever immediate dispatch is verified feasible.
All resulting trajectories are verified by verify_single_trajectory.
"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from scripts.work3.evaluate_c_vs_d import build_formal_evaluation_agent
from work3_4060.verify_trajectory_facts import load_baseline_tasks, verify_single_trajectory
from utils.work3.objective_evaluator import evaluate_trajectory_objective
from work3_4060.run_rule_ac_comparison import extract_execution_facts
from work3_4060.diagnostics.immediate_dispatch_oracle import enumerate_immediate_dispatches

BASELINE_PATH = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"
CHECKPOINT_PATH = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "seed42" / "C" / "weights.pt"
REAUDIT_PATH = REPO_ROOT / "work3_4060" / "diagnostics" / "immediate_dispatch_reaudit.json"

def rollout_to_termination(env: AirLineEnvWork3, agent, max_steps: int = 1000):
    """Rolls out the environment using agent until termination."""
    steps = 0
    while not env._check_terminated() and steps < max_steps:
        action = agent.select_action(env)
        if action is None:
            action = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
        env.step(action)
        steps += 1
    return steps

def extract_trajectory_metrics(env: AirLineEnvWork3, h0: float, baseline_tasks: dict):
    """Extracts makespan, costs, and validates trajectory."""
    breakdown = evaluate_trajectory_objective(env)
    
    makespan_last_task = max((float(t.actual_end) for t in env.state.tasks.values() if t.actual_end is not None), default=0.0)
    final_aircraft_exit_time = max((float(ac.exit_times[4]) for ac in env.state.aircraft.values() if 4 in ac.exit_times), default=float(env.state.current_time))
    
    st_clearances = {}
    for s in range(5):
        st_clearances[f"station_{s}"] = max((float(t.actual_end) for t in env.state.tasks.values() if t.actual_end is not None and t.current_station == s), default=0.0)
        
    # Run independent verifier
    facts = extract_execution_facts(env)
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
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(traj_dict, f, indent=2)
        tmp_name = f.name
        
    verifier_res = verify_single_trajectory(Path(tmp_name), h0, baseline_tasks)
    try:
        Path(tmp_name).unlink()
    except Exception:
        pass
        
    return {
        "makespan_last_task": makespan_last_task,
        "final_aircraft_exit_time": final_aircraft_exit_time,
        "station_clearances": st_clearances,
        "j_total": breakdown.j_total,
        "j_takt": breakdown.j_takt,
        "d_time": breakdown.d_time,
        "d_team": breakdown.d_team,
        "j_postpone": breakdown.j_postpone,
        "takt_violation_hours": breakdown.takt_violation_hours,
        "num_transfers": breakdown.num_transfers,
        "verifier_status": verifier_res.get("status", "UNKNOWN"),
        "violations_count": len(verifier_res.get("violations", []))
    }

def run_paired_experiments():
    print("=" * 80)
    print("RUNNING PAIRED TWO-ARM COUNTERFACTUAL CONTINUATION EXPERIMENTS")
    print("=" * 80)
    
    agent = build_formal_evaluation_agent("C", CHECKPOINT_PATH, device="cpu")
    h0, baseline_tasks, _ = load_baseline_tasks(BASELINE_PATH)
    
    with open(REAUDIT_PATH, "r", encoding="utf-8") as f:
        reaudit_data = json.load(f)
        
    verified_records = [
        r for r in reaudit_data["advance_records"] 
        if r["category"] == "VERIFIED_IMMEDIATE_DISPATCH_AVAILABLE"
    ]
    
    # Sample >= 20 sites across early, mid, late
    num_records = len(verified_records)
    print(f"Total available verified immediate dispatch sites: {num_records}")
    
    # Stratified sampling of 24 sites across the trajectory
    indices = np.linspace(0, num_records - 1, 24, dtype=int)
    sampled_records = [verified_records[i] for i in indices]
    
    site_results = []
    
    env_template = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH), transfer_policy="nominal_anchored_v1")
    env_template.reset()

    for idx, rec in enumerate(sampled_records):
        step_idx = rec["step"]
        cur_t = rec["current_time"]
        num_imm = rec["num_immediate_dispatches"]
        sample_imm = rec["immediate_dispatches_sample"]
        
        print(f"\n[{idx+1}/24] Evaluating Site Step {step_idx} (t={cur_t:.2f}h, {num_imm} immediate candidates)...", flush=True)
        
        # Reconstruct base env up to step_idx from template clone
        env_base = copy.deepcopy(env_template)
        
        # Step through original action history
        for act in rec["env_action_history"]:
            env_base.step(act)
            
        # Verify state
        assert abs(float(env_base.state.current_time) - cur_t) < 1e-5
        
        # Arm A: Take ADVANCE
        env_arm_a = copy.deepcopy(env_base)
        env_arm_a.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
        rollout_to_termination(env_arm_a, agent)
        metrics_a = extract_trajectory_metrics(env_arm_a, h0, baseline_tasks)
        
        # Arm B: Take Immediate Dispatch (pick candidate with longest duration for maximal work clearing)
        all_imm = enumerate_immediate_dispatches(env_base, align=0)
        # Sort by duration desc, then task_key
        best_candidate = sorted(all_imm, key=lambda x: (x["duration"], x["task_key"]), reverse=True)[0]
        
        dispatch_action = {
            "branch": ActionBranch.STATION_EXECUTE,
            "task_key": best_candidate["task_key"],
            "team": list(best_candidate["team"]),
            "align": 0
        }
        
        env_arm_b = copy.deepcopy(env_base)
        env_arm_b.step(dispatch_action)
        rollout_to_termination(env_arm_b, agent)
        metrics_b = extract_trajectory_metrics(env_arm_b, h0, baseline_tasks)
        
        delta_makespan = metrics_b["makespan_last_task"] - metrics_a["makespan_last_task"]
        delta_penalty = metrics_b["j_total"] - metrics_a["j_total"]
        
        res_entry = {
            "site_index": idx + 1,
            "step": step_idx,
            "current_time": cur_t,
            "target_task": best_candidate["task_key"],
            "target_task_duration": best_candidate["duration"],
            "target_station": best_candidate["station_id"],
            "arm_a_advance": metrics_a,
            "arm_b_immediate": metrics_b,
            "delta_makespan": delta_makespan,
            "delta_penalty": delta_penalty,
        }
        site_results.append(res_entry)
        
        print(f"  Arm A (ADVANCE)   -> Last Task Makespan: {metrics_a['makespan_last_task']:.2f}h, J_total: {metrics_a['j_total']:.4f}, Verifier: {metrics_a['verifier_status']}", flush=True)
        print(f"  Arm B (IMMEDIATE) -> Last Task Makespan: {metrics_b['makespan_last_task']:.2f}h, J_total: {metrics_b['j_total']:.4f}, Verifier: {metrics_b['verifier_status']}", flush=True)
        print(f"  Delta (B - A): Makespan = {delta_makespan:+.2f}h, J_total = {delta_penalty:+.4f}", flush=True)

    # =========================================================================
    # Whole-Trajectory Heuristic Intervention
    # =========================================================================
    print("\n" + "=" * 80)
    print("RUNNING WHOLE-TRAJECTORY HEURISTIC INTERVENTION")
    print("=" * 80)
    
    # 1. Pure PPO baseline
    env_ppo = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH), transfer_policy="nominal_anchored_v1")
    env_ppo.reset()
    rollout_to_termination(env_ppo, agent)
    metrics_pure_ppo = extract_trajectory_metrics(env_ppo, h0, baseline_tasks)
    
    # 2. Greedy Immediate Dispatch Intervention
    env_intervene = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH), transfer_policy="nominal_anchored_v1")
    env_intervene.reset()
    
    intervene_steps = 0
    interventions_applied = 0
    
    while not env_intervene._check_terminated() and intervene_steps < 1000:
        action, diag = agent.select_action(env_intervene, return_diagnostics=True)
        is_advance = (action is None or action.get("branch") == ActionBranch.ADVANCE_TO_NEXT_EVENT)
        
        if is_advance:
            imm_cands = enumerate_immediate_dispatches(env_intervene, align=0)
            if len(imm_cands) > 0:
                best_cand = sorted(imm_cands, key=lambda x: (x["duration"], x["task_key"]), reverse=True)[0]
                action = {
                    "branch": ActionBranch.STATION_EXECUTE,
                    "task_key": best_cand["task_key"],
                    "team": list(best_cand["team"]),
                    "align": 0
                }
                interventions_applied += 1
            else:
                action = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
        elif action is None:
            action = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
            
        env_intervene.step(action)
        intervene_steps += 1
        
    metrics_intervene = extract_trajectory_metrics(env_intervene, h0, baseline_tasks)
    
    print(f"Whole-trajectory Intervention Applied: {interventions_applied} immediate dispatches substituted for ADVANCE.")
    print(f"Pure PPO Baseline:           Makespan: {metrics_pure_ppo['makespan_last_task']:.2f}h, J_total: {metrics_pure_ppo['j_total']:.4f}, Verifier: {metrics_pure_ppo['verifier_status']}")
    print(f"Intervention (Greedy ASAP):  Makespan: {metrics_intervene['makespan_last_task']:.2f}h, J_total: {metrics_intervene['j_total']:.4f}, Verifier: {metrics_intervene['verifier_status']}")
    print(f"Whole-trajectory Delta:      Makespan = {metrics_intervene['makespan_last_task'] - metrics_pure_ppo['makespan_last_task']:+.2f}h, J_total = {metrics_intervene['j_total'] - metrics_pure_ppo['j_total']:+.4f}")

    # Compile comprehensive results
    out_data = {
        "num_sites_evaluated": len(site_results),
        "sampled_sites": site_results,
        "whole_trajectory_comparison": {
            "pure_ppo": metrics_pure_ppo,
            "intervention": metrics_intervene,
            "interventions_applied": interventions_applied,
            "delta_makespan": metrics_intervene["makespan_last_task"] - metrics_pure_ppo["makespan_last_task"],
            "delta_penalty": metrics_intervene["j_total"] - metrics_pure_ppo["j_total"],
        }
    }
    
    out_path = REPO_ROOT / "work3_4060" / "diagnostics" / "paired_counterfactual_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nPaired experiment results saved to {out_path}")

if __name__ == "__main__":
    run_paired_experiments()
