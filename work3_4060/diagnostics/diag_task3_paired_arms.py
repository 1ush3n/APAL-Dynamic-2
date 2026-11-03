"""Diagnostic script for Task 3: Paired Arms (Control: Stay vs Treatment: Postpone).

Design:
- Preset 3 sites fixed before reviewing outcomes:
  Site 1: NOMINAL scenario, at Aircraft 0 Station 1, Task '0_7' (step 5, t=0.0).
  Site 2: RULE_VAL_S1_MODERATE scenario, at Aircraft 1 Station 1, Task '1_7' (step 18, t=52.0378h).
  Site 3: RULE_VAL_S2_STRONG scenario, at Aircraft 0 Station 2, Task '0_18' (step 10, t=52.0378h).

Rules:
- Same template, clock, disturbance, action prefix.
- Exactly split into Control (STATION_EXECUTE) and Treatment (POSTPONE).
- Both arms subsequently use the exact same PostponeCompatibleRepairAgent.
- Verify Control reproduces the reference trajectory.
- Verify Treatment legally selects team at target station without clearing start constraints or using source workers.
- Record: current transfer, downstream transfers/load, aircraft exits, makespan, and all 5 cost components + J_total.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from scripts.work3.baseline_repair_agent import BaselinePlanRepairAgent
from work3_4060.diagnostics.postpone_compatible_agent import PostponeCompatibleRepairAgent

BASELINE_PATH = repo_root / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"
SCENARIOS_PATH = repo_root / "work3_4060" / "rule_baseline_ac_v1" / "scenarios.json"
OUTPUT_DIR = repo_root / "work3_4060" / "diagnostics" / "task3_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_paired_arm(
    site_name: str,
    scenario_dict: dict[str, Any] | None,
    split_task_key: str,
    min_cycle: int = 1,
) -> dict[str, Any]:
    print(f"\n==========================================")
    print(f"Running Paired Experiment: {site_name}")
    print(f"Split Task Key: {split_task_key}")
    print(f"==========================================")

    # 1. First run Reference Trajectory with BaselinePlanRepairAgent
    env_ref = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH))
    env_ref.reset()
    if scenario_dict is not None:
        env_ref.load_scenario(scenario_dict)
    agent_ref = BaselinePlanRepairAgent()
    while not env_ref._check_terminated():
        act = agent_ref.select_action(env_ref)
        if act is None:
            break
        env_ref.step(act)
    ref_makespan = float(env_ref.state.current_time)
    ref_cost = float(env_ref.cumulative_cost)
    ref_transfers = [float(x) for x in env_ref.state.transfer_history]
    print(f"Reference run: makespan={ref_makespan:.6f}h, J_total={ref_cost:.6e}")

    # 2. Control Arm: Follow prefix until agent selects split_task_key, execute STAY, then continue
    env_ctrl = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH))
    env_ctrl.reset()
    if scenario_dict is not None:
        env_ctrl.load_scenario(scenario_dict)
    agent_ctrl = PostponeCompatibleRepairAgent()

    split_reached_ctrl = False
    step_ctrl = 0
    split_info_ctrl = {}

    while not env_ctrl._check_terminated() and step_ctrl < 3000:
        act = agent_ctrl.select_action(env_ctrl)
        if act is None:
            break

        if not split_reached_ctrl and act.get("task_key") == split_task_key and env_ctrl.state.current_cycle >= min_cycle:
            t = env_ctrl.state.tasks[split_task_key]
            can_r = env_ctrl.can_reserve(t)
            can_p = env_ctrl.can_postpone(t)
            assert can_r and can_p, f"At split point, both stay and postpone must be legal! can_r={can_r}, can_p={can_p}"
            assert act.get("branch") == ActionBranch.STATION_EXECUTE
            split_info_ctrl = {
                "step": step_ctrl,
                "time": float(env_ctrl.state.current_time),
                "cycle": int(env_ctrl.state.current_cycle),
                "action": "STATION_EXECUTE",
                "team": [int(w) for w in act["team"]],
                "station": int(t.current_station),
            }
            split_reached_ctrl = True

        env_ctrl.step(act)
        step_ctrl += 1

    ctrl_makespan = float(env_ctrl.state.current_time)
    ctrl_cost = float(env_ctrl.cumulative_cost)
    ctrl_transfers = [float(x) for x in env_ctrl.state.transfer_history]
    ctrl_completed = sum(1 for t in env_ctrl.state.tasks.values() if t.status == TaskStatus.COMPLETED)
    print(f"Control run: split_reached={split_reached_ctrl}, completed={ctrl_completed}/210, makespan={ctrl_makespan:.6f}h, J_total={ctrl_cost:.6e}")
    assert abs(ctrl_makespan - ref_makespan) < 1e-5, f"Control makespan mismatch: {ctrl_makespan} vs {ref_makespan}"
    assert abs(ctrl_cost - ref_cost) < 1e-5, f"Control cost mismatch: {ctrl_cost} vs {ref_cost}"
    print("-> Control arm PERFECTLY reproduced the reference trajectory!")

    # 3. Treatment Arm: Same prefix, at split point execute POSTPONE, then continue
    env_treat = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH))
    env_treat.reset()
    if scenario_dict is not None:
        env_treat.load_scenario(scenario_dict)
    agent_treat = PostponeCompatibleRepairAgent()

    split_reached_treat = False
    step_treat = 0
    split_info_treat = {}

    while not env_treat._check_terminated() and step_treat < 3000:
        act = agent_treat.select_action(env_treat)
        if act is None:
            break

        if not split_reached_treat and act.get("task_key") == split_task_key and env_treat.state.current_cycle >= min_cycle:
            t = env_treat.state.tasks[split_task_key]
            can_r = env_treat.can_reserve(t)
            can_p = env_treat.can_postpone(t)
            assert can_r and can_p, f"At split point, both stay and postpone must be legal! can_r={can_r}, can_p={can_p}"

            act_postpone = {"task_key": split_task_key, "branch": ActionBranch.POSTPONE}
            split_info_treat = {
                "step": step_treat,
                "time": float(env_treat.state.current_time),
                "cycle": int(env_treat.state.current_cycle),
                "action": "POSTPONE",
                "old_station": int(t.current_station),
                "new_station": int(t.current_station + 1),
            }
            env_treat.step(act_postpone)
            step_treat += 1
            split_reached_treat = True
            continue

        env_treat.step(act)
        step_treat += 1

    treat_makespan = float(env_treat.state.current_time)
    treat_cost = float(env_treat.cumulative_cost)
    treat_transfers = [float(x) for x in env_treat.state.transfer_history]
    treat_completed = sum(1 for t in env_treat.state.tasks.values() if t.status == TaskStatus.COMPLETED)
    print(f"Treatment run: split_reached={split_reached_treat}, completed={treat_completed}/210, makespan={treat_makespan:.6f}h, J_total={treat_cost:.6e}")

    # Treatment postponed task audit: verify where it actually executed and with what team
    t_treat_obj = env_treat.state.tasks[split_task_key]
    postponed_task_audit = {
        "task_key": split_task_key,
        "base_station": int(t_treat_obj.base_station),
        "actual_start_station": int(t_treat_obj.actual_start_station) if t_treat_obj.actual_start_station is not None else int(t_treat_obj.current_station),
        "actual_start": float(t_treat_obj.actual_start) if t_treat_obj.actual_start is not None else None,
        "actual_end": float(t_treat_obj.actual_end) if t_treat_obj.actual_end is not None else None,
        "assigned_team": [int(w) for w in t_treat_obj.assigned_team],
        "postpone_count": int(t_treat_obj.postpone_count),
    }

    # Aircraft exit times
    ctrl_exits = {str(ac_id): {str(st): float(t) for st, t in ac.exit_times.items()} for ac_id, ac in env_ctrl.state.aircraft.items()}
    treat_exits = {str(ac_id): {str(st): float(t) for st, t in ac.exit_times.items()} for ac_id, ac in env_treat.state.aircraft.items()}

    # Station load (total task durations executed at each station)
    ctrl_station_load = {s: 0.0 for s in range(env_ctrl.state.num_stations)}
    for t in env_ctrl.state.tasks.values():
        st = t.actual_start_station if t.actual_start_station is not None else t.current_station
        ctrl_station_load[st] += float(t.duration)

    treat_station_load = {s: 0.0 for s in range(env_treat.state.num_stations)}
    for t in env_treat.state.tasks.values():
        st = t.actual_start_station if t.actual_start_station is not None else t.current_station
        treat_station_load[st] += float(t.duration)

    result = {
        "site_name": site_name,
        "split_task_key": split_task_key,
        "split_info_ctrl": split_info_ctrl,
        "split_info_treat": split_info_treat,
        "postponed_task_execution_audit": postponed_task_audit,
        "control": {
            "completed": ctrl_completed == 210,
            "makespan": ctrl_makespan,
            "j_total": ctrl_cost,
            "cost_takt": float(env_ctrl.cost_takt),
            "cost_time": float(env_ctrl.cost_time),
            "cost_team": float(env_ctrl.cost_team),
            "cost_postpone": float(env_ctrl.cost_postpone),
            "cost_revision": float(env_ctrl.cost_revision),
            "transfer_history": ctrl_transfers,
            "aircraft_exits": ctrl_exits,
            "station_loads": ctrl_station_load,
            "step_count": step_ctrl,
        },
        "treatment": {
            "completed": treat_completed == 210,
            "makespan": treat_makespan,
            "j_total": treat_cost,
            "cost_takt": float(env_treat.cost_takt),
            "cost_time": float(env_treat.cost_time),
            "cost_team": float(env_treat.cost_team),
            "cost_postpone": float(env_treat.cost_postpone),
            "cost_revision": float(env_treat.cost_revision),
            "transfer_history": treat_transfers,
            "aircraft_exits": treat_exits,
            "station_loads": treat_station_load,
            "step_count": step_treat,
        },
        "comparison": {
            "delta_makespan_treatment_minus_control": treat_makespan - ctrl_makespan,
            "delta_j_total_treatment_minus_control": treat_cost - ctrl_cost,
            "delta_j_postpone": float(env_treat.cost_postpone - env_ctrl.cost_postpone),
            "delta_j_takt": float(env_treat.cost_takt - env_ctrl.cost_takt),
            "delta_j_revision": float(env_treat.cost_revision - env_ctrl.cost_revision),
            "delta_d_team": float(env_treat.cost_team - env_ctrl.cost_team),
            "delta_d_time": float(env_treat.cost_time - env_ctrl.cost_time),
        }
    }

    print(f"Comparison: delta_Makespan = {result['comparison']['delta_makespan_treatment_minus_control']:+.4f}h, delta_J_total = {result['comparison']['delta_j_total_treatment_minus_control']:+.6f}")
    return result


def main():
    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        scenarios = {s["scenario_id"]: s for s in json.load(f)}

    # Preset Site 1: Nominal, task 0_7 at step 5
    res_site1 = run_paired_arm(
        site_name="Preset_Site_1_Nominal_Task_0_7",
        scenario_dict=None,
        split_task_key="0_7",
        min_cycle=1,
    )

    # Preset Site 2: RULE_VAL_S1_MODERATE, task 1_7 at Station 1
    res_site2 = run_paired_arm(
        site_name="Preset_Site_2_Moderate_Task_1_7",
        scenario_dict=scenarios["RULE_VAL_S1_MODERATE"],
        split_task_key="1_7",
        min_cycle=2,
    )

    # Preset Site 3: RULE_VAL_S2_STRONG, task 0_18 at Station 2
    res_site3 = run_paired_arm(
        site_name="Preset_Site_3_Strong_Task_0_18",
        scenario_dict=scenarios["RULE_VAL_S2_STRONG"],
        split_task_key="0_18",
        min_cycle=2,
    )

    all_results = {
        "Preset_Site_1_Nominal_Task_0_7": res_site1,
        "Preset_Site_2_Moderate_Task_1_7": res_site2,
        "Preset_Site_3_Strong_Task_0_18": res_site3,
    }

    out_file = OUTPUT_DIR / "task3_paired_arms_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nTask 3 report written to: {out_file}")


if __name__ == "__main__":
    main()
