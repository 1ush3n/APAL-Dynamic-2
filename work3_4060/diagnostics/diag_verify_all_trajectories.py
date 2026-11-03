"""
Independent Fact Verification for all 13 Work-3 diagnostic trajectories.

Uses the authoritative verify_trajectory_facts.py engine to perform strict,
physical, and financial checks on all 13 diagnostic trajectories:
  - 5 witness trajectories (6->7, 15->16, 8->9, 12->13, 20->21)
  - 1 joint 5-pairs trajectory (10 tasks postponed)
  - 1 joint 17-tasks capacity trajectory
  - 6 paired arm trajectories (Site 1, 2, 3 Control & Treatment)

Constraints checked:
  1. Worker exclusivity (overlap count == 0)
  2. Station concurrency (active concurrent tasks <= 3)
  3. Precedence DAG topology (predecessor actual_end <= successor actual_start)
  4. Station dwell intervals (all tasks within [entry_time, exit_time])
  5. Ledger balance (recalculated costs == environment cumulative costs, diff < 1e-15)
"""

import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import ActionBranch
from envs.work3.environment import AirLineEnvWork3
from work3_4060.diagnostics.postpone_compatible_agent import PostponeCompatibleRepairAgent
from work3_4060.run_rule_ac_comparison import extract_execution_facts
from work3_4060.verify_trajectory_facts import (
    load_baseline_tasks,
    load_scenarios_pool,
    verify_single_trajectory,
)

BASELINE_PATH = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"
SCENARIOS_PATH = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "scenarios.json"


def verify_env_trajectory(env, name: str, scenario_spec: dict = None) -> dict:
    facts = extract_execution_facts(env)
    traj_dict = {
        "scenario_id": scenario_spec["scenario_id"] if scenario_spec else "NOMINAL",
        "objective_j_total": float(env.cumulative_cost),
        "j_total": float(env.cumulative_cost),
        "j_takt": float(env.cost_takt),
        "d_time": float(env.cost_time),
        "d_team": float(env.cost_team),
        "j_postpone": float(env.cost_postpone),
        "j_revision": float(env.cost_revision),
        "actual_disturbance_hits": 1 if getattr(env, "disturbance_event_triggered", False) else 0,
        "execution_facts": facts,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(traj_dict, f, indent=2)
        tmp_name = f.name
    try:
        h0, baseline_tasks, _ = load_baseline_tasks(BASELINE_PATH)
        res = verify_single_trajectory(Path(tmp_name), h0, baseline_tasks, scenario_spec=scenario_spec)
        violations = res["violations"]
        total_violations = sum(len(v) for v in violations.values())
        recalc_j = res["recalculated_costs"]["j_total"]
        diff = abs(recalc_j - env.cumulative_cost)
        print(f"[{name:<28}] status={res['status']} violations={total_violations} recalc_j={recalc_j:.8f} env_j={env.cumulative_cost:.8f} diff={diff:.2e}")
        if total_violations > 0:
            for k, v in violations.items():
                if v:
                    print(f"   Violation {k}: {v[:2]}")
        return res
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def run_all_verifications():
    print("=" * 80)
    print("STARTING INDEPENDENT FACT VERIFICATION ON 13 DIAGNOSTIC TRAJECTORIES")
    print("=" * 80)
    scenarios = load_scenarios_pool(SCENARIOS_PATH)
    results = {}

    # 1-5: Five Witness Trajectories
    pairs = [(6, 7), (15, 16), (8, 9), (12, 13), (20, 21)]
    for u, v in pairs:
        env = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH))
        env.reset()
        agent = PostponeCompatibleRepairAgent()
        step = 0
        u_key = f"0_{u}"
        v_key = f"0_{v}"
        u_postponed = False
        v_postponed = False
        while not env._check_terminated() and step < 2000:
            ac0 = env.state.aircraft[0]
            t_u = env.state.tasks[u_key]
            t_v = env.state.tasks[v_key]
            if ac0.current_station == t_v.current_station and not v_postponed:
                if env.can_postpone(t_v):
                    env.step({"task_key": v_key, "branch": ActionBranch.POSTPONE})
                    step += 1
                    v_postponed = True
                    if env.can_postpone(t_u):
                        env.step({"task_key": u_key, "branch": ActionBranch.POSTPONE})
                        step += 1
                        u_postponed = True
                    continue
            act = agent.select_action(env)
            if act is None:
                break
            env.step(act)
            step += 1
        name = f"Witness {u}->{v}"
        results[name] = verify_env_trajectory(env, name)

    # 6: Joint 5-Pairs (10 tasks)
    env_5p = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH))
    env_5p.reset()
    ag_5p = PostponeCompatibleRepairAgent()
    postponed_set = set()
    step = 0
    while not env_5p._check_terminated() and step < 2000:
        ac0 = env_5p.state.aircraft[0]
        action_taken = False
        for u, v in pairs:
            u_key = f"0_{u}"
            v_key = f"0_{v}"
            t_u = env_5p.state.tasks[u_key]
            t_v = env_5p.state.tasks[v_key]
            if ac0.current_station == t_v.current_station:
                if v_key not in postponed_set and env_5p.can_postpone(t_v):
                    env_5p.step({"task_key": v_key, "branch": ActionBranch.POSTPONE})
                    postponed_set.add(v_key)
                    step += 1
                    env_5p.step({"task_key": u_key, "branch": ActionBranch.POSTPONE})
                    postponed_set.add(u_key)
                    step += 1
                    action_taken = True
                    break
        if action_taken:
            continue
        act = ag_5p.select_action(env_5p)
        if act is None:
            break
        env_5p.step(act)
        step += 1
    results["Joint 5-Pairs (10 tasks)"] = verify_env_trajectory(env_5p, "Joint 5-Pairs (10 tasks)")

    # 7: Joint 17-Tasks Capacity
    env_17 = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH))
    env_17.reset()
    ag_17 = PostponeCompatibleRepairAgent()
    all_17_eligible = [6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 20, 21, 25, 26, 28, 29]
    postponed_all = set()
    step = 0
    while not env_17._check_terminated() and step < 2000:
        ac = env_17.state.aircraft[0]
        can_p_tasks = [
            t for t in env_17.state.get_tasks_for_station(ac.current_station)
            if t.aircraft_id == 0 and t.task_id in all_17_eligible and t.task_key not in postponed_all and env_17.can_postpone(t)
        ]
        if can_p_tasks:
            t_to_postpone = sorted(can_p_tasks, key=lambda x: -len(x.predecessors))[0]
            env_17.step({"task_key": t_to_postpone.task_key, "branch": ActionBranch.POSTPONE})
            postponed_all.add(t_to_postpone.task_key)
            step += 1
            continue
        act = ag_17.select_action(env_17)
        if act is None:
            break
        env_17.step(act)
        step += 1
    results["Joint 17-Tasks Capacity"] = verify_env_trajectory(env_17, "Joint 17-Tasks Capacity")

    # 8-13: Paired Arms (Control vs Treatment)
    sites = [
        ("Site 1: Nominal Control", None, "0_7", 1, False),
        ("Site 1: Nominal Treatment", None, "0_7", 1, True),
        ("Site 2: Moderate Control", scenarios.get("RULE_VAL_S1_MODERATE"), "1_7", 2, False),
        ("Site 2: Moderate Treatment", scenarios.get("RULE_VAL_S1_MODERATE"), "1_7", 2, True),
        ("Site 3: Strong Control", scenarios.get("RULE_VAL_S2_STRONG"), "0_18", 2, False),
        ("Site 3: Strong Treatment", scenarios.get("RULE_VAL_S2_STRONG"), "0_18", 2, True),
    ]

    for name, scen, split_key, min_c, is_treatment in sites:
        env_site = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH))
        env_site.reset()
        if scen:
            env_site.load_scenario(scen)
        ag_site = PostponeCompatibleRepairAgent()
        step = 0
        split_done = False
        while not env_site._check_terminated() and step < 3000:
            act = ag_site.select_action(env_site)
            if act is None:
                break
            if is_treatment and not split_done and act.get("task_key") == split_key and env_site.state.current_cycle >= min_c:
                env_site.step({"task_key": split_key, "branch": ActionBranch.POSTPONE})
                step += 1
                split_done = True
                continue
            env_site.step(act)
            step += 1
        results[name] = verify_env_trajectory(env_site, name, scenario_spec=scen)

    print("=" * 80)
    all_pass = all(r["status"] == "PASS" for r in results.values())
    print(f"VERIFICATION COMPLETE: {len(results)}/13 TRAJECTORIES VERIFIED, ALL PASS = {all_pass}")
    print("=" * 80)
    return all_pass


if __name__ == "__main__":
    success = run_all_verifications()
    sys.exit(0 if success else 1)
