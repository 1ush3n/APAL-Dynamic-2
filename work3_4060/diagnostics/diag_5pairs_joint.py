"""Explicit joint test for all 5 pairs (10 tasks) postponed together in one trajectory."""
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from work3_4060.diagnostics.postpone_compatible_agent import PostponeCompatibleRepairAgent

BASELINE_PATH = repo_root / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"

def main():
    pairs = [(6, 7), (15, 16), (8, 9), (12, 13), (20, 21)]
    env = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH))
    env.reset()
    agent = PostponeCompatibleRepairAgent()

    postponed = set()
    step = 0
    witness_log = []

    while not env._check_terminated() and step < 2000:
        # Check pairs
        ac0 = env.state.aircraft[0]
        action_taken = False
        for u, v in pairs:
            u_key = f"0_{u}"
            v_key = f"0_{v}"
            t_u = env.state.tasks[u_key]
            t_v = env.state.tasks[v_key]

            # If both not postponed and aircraft is at their station
            if ac0.current_station == t_v.current_station:
                if v_key not in postponed and env.can_postpone(t_v):
                    # Check u is blocked
                    assert not env.can_postpone(t_u), f"Predecessor {u_key} should be blocked before {v_key} postpones!"
                    env.step({"task_key": v_key, "branch": ActionBranch.POSTPONE})
                    postponed.add(v_key)
                    step += 1
                    # Now check u is unlocked!
                    assert env.can_postpone(t_u), f"Predecessor {u_key} should unlock after {v_key} postpones!"
                    env.step({"task_key": u_key, "branch": ActionBranch.POSTPONE})
                    postponed.add(u_key)
                    step += 1
                    action_taken = True
                    witness_log.append((u, v, float(env.state.current_time), int(t_u.current_station)))
                    break
        if action_taken:
            continue

        act = agent.select_action(env)
        if act is None:
            break
        env.step(act)
        step += 1

    completed = sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.COMPLETED)
    print(f"5 Pairs Joint Postponement Result:")
    print(f"Postponed {len(postponed)} tasks: {sorted(list(postponed))}")
    print(f"Witness log: {witness_log}")
    print(f"Completed {completed}/{len(env.state.tasks)} tasks. Terminated={env._check_terminated()}")
    print(f"Makespan: {env.state.current_time:.6f}h, J_total: {env.cumulative_cost:.6f}, cost_postpone: {env.cost_postpone:.4f}, cost_takt: {env.cost_takt:.6f}")

if __name__ == "__main__":
    main()
