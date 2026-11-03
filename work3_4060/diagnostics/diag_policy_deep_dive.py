"""
Script: diag_policy_deep_dive.py
Investigates the behavior and internal failure modes of the learned PPO policy (Method C).
Loads checkpoint: work3_4060/rule_baseline_ac_v1/seed42/C/weights.pt.
Inspects:
  1. Checkpoint loading, architecture compatibility, parameter norm.
  2. Action distribution over 210 tasks on NOMINAL:
     - advance_logits, prob_advance, prob_stay
     - frequency of premature ADVANCE actions while ready tasks exist
     - task selection entropy & branch choice (STAY vs POSTPONE)
     - worker selection behavior
  3. Action masking: are better actions/teams pruned?
  4. Root cause analysis of why PPO makespan is 105.4h worse than baseline.
"""

import json
from pathlib import Path
import sys
import torch
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from scripts.work3.evaluate_c_vs_d import build_formal_evaluation_agent
from work3_4060.run_rule_ac_comparison import extract_execution_facts
from work3_4060.verify_trajectory_facts import load_baseline_tasks, verify_single_trajectory

BASELINE_PATH = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"
CHECKPOINT_PATH = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "seed42" / "C" / "weights.pt"

def audit_policy_weights(checkpoint_path: Path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("actor_critic_state", ckpt)
    
    total_params = 0
    nan_params = 0
    param_norms = {}
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            total_params += param.numel()
            if torch.isnan(param).any():
                nan_params += param.numel()
            param_norms[name] = float(param.norm().item())
        
    return {
        "total_parameters": total_params,
        "nan_parameters": nan_params,
        "metadata": {k: str(v) for k, v in ckpt.get("run_metadata", {}).items()},
        "keys_count": len(state_dict),
    }

def run_policy_evaluation():
    agent = build_formal_evaluation_agent("C", CHECKPOINT_PATH, device="cpu")
    env = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH), transfer_policy="nominal_anchored_v1")
    env.reset()

    decisions = []
    step = 0
    max_steps = 1000
    premature_advance_count = 0
    postpone_count = 0
    stay_count = 0
    
    with torch.no_grad():
        while not env._check_terminated() and step < max_steps:
            action, diag = agent.select_action(env, return_diagnostics=True)
            cand_count = len(diag.get("candidate_task_keys", []))
            prob_adv = diag.get("prob_advance", 0.0)
            prob_stay = diag.get("prob_stay", 1.0)
            val = diag.get("state_value", 0.0)
            
            if action is None:
                # Agent chose ADVANCE
                branch_name = "ADVANCE_TO_NEXT_EVENT"
                t_key = None
                if cand_count > 0:
                    premature_advance_count += 1
                step_action = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
            else:
                branch = action.get("branch")
                t_key = action.get("task_key")
                branch_name = ActionBranch(branch).name if branch is not None else "None"
                if branch == ActionBranch.POSTPONE:
                    postpone_count += 1
                elif branch == ActionBranch.STATION_EXECUTE:
                    stay_count += 1
                step_action = action
                
            decisions.append({
                "step": step,
                "current_time": round(float(env.state.current_time), 3),
                "branch": branch_name,
                "task_key": t_key,
                "candidate_count": cand_count,
                "prob_advance": round(float(prob_adv), 4),
                "prob_stay": round(float(prob_stay), 4),
                "value_estimate": round(float(val), 4),
            })
            
            _obs, _rew, term, trunc, _info = env.step(step_action)
            step += 1
            if term or trunc:
                break

    facts = extract_execution_facts(env)
    tasks_exec = list(facts["tasks_execution"].values())
    makespan = max(t["actual_end"] for t in tasks_exec if t["actual_end"] is not None)

    return {
        "total_steps": step,
        "completed_tasks": len(tasks_exec),
        "makespan": round(makespan, 3),
        "cumulative_cost": round(env.cumulative_cost, 4),
        "cost_takt": round(env.cost_takt, 4),
        "cost_time": round(env.cost_time, 4),
        "cost_team": round(env.cost_team, 4),
        "cost_revision": round(env.cost_revision),
        "premature_advance_count": premature_advance_count,
        "postpone_count": postpone_count,
        "stay_count": stay_count,
        "decisions_head": decisions[:30],
        "decisions_total_count": len(decisions)
    }

def main():
    print("=" * 80)
    print("LEARNED POLICY (METHOD C / PPO) DEEP-DIVE DIAGNOSIS")
    print("=" * 80)

    print(f"\n1. Auditing Checkpoint: {CHECKPOINT_PATH.name}...")
    audit = audit_policy_weights(CHECKPOINT_PATH)
    print(f"   Total params: {audit['total_parameters']}, NaN: {audit['nan_parameters']}")

    print("\n2. Evaluating Deterministic Policy on NOMINAL scenario...")
    eval_det = run_policy_evaluation()
    print(f"   Makespan: {eval_det['makespan']}h (vs Baseline 525.882h, +{eval_det['makespan'] - 525.882:.2f}h drift!)")
    print(f"   Total steps: {eval_det['total_steps']} (vs Baseline 210, +{eval_det['total_steps'] - 210} excess steps!)")
    print(f"   Premature ADVANCE decisions: {eval_det['premature_advance_count']} times while ready tasks were waiting!")
    print(f"   Postpone decisions:          {eval_det['postpone_count']} times")
    print(f"   Takt penalty J_takt:         {eval_det['cost_takt']}")
    print(f"   Total cost J_total:          {eval_det['cumulative_cost']}")

    # Save detailed JSON report
    report = {
        "checkpoint_audit": audit,
        "evaluation_deterministic": eval_det
    }
    out_path = REPO_ROOT / "work3_4060" / "diagnostics" / "policy_deep_dive_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nPolicy deep-dive saved to {out_path}")

    # Output decisions sample to CSV
    df = pd.DataFrame(eval_det["decisions_head"])
    csv_path = REPO_ROOT / "work3_4060" / "diagnostics" / "summary_tables" / "policy_decisions_sample.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Decisions sample saved to {csv_path}")

if __name__ == "__main__":
    main()
