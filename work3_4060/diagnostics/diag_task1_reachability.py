import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from envs.work3.environment import AirLineEnvWork3
from envs.work3.core_types import ActionBranch, TaskStatus

def test_static_and_initial_masks():
    baseline_path = repo_root / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    obs = env.reset()

    print(f"Env initialized. Current time: {env.state.current_time}, cycle: {env.state.current_cycle}")
    
    # Check all tasks for aircraft 0
    ac0_tasks = {t.task_id: t for t in env.state.tasks.values() if t.aircraft_id == 0}
    
    print("\n--- Static Eligibility & Initial Postpone Masks for Aircraft 0 ---")
    static_eligible = []
    initial_postpone_masks = {}
    blocked_by_successor = {}
    
    for task_id in sorted(ac0_tasks.keys()):
        t = ac0_tasks[task_id]
        err = env.validate_postpone(t)
        can_p = env.can_postpone(t)
        can_r = env.can_reserve(t)
        
        # Check if statically eligible (ignoring runtime successor status)
        # Static conditions: current_station < num_stations - 1, target station has qualified workers, no fixed station violation
        target_station = t.current_station + 1
        workers_ok = len(env.valid_team_completion_workers(t, [], station_id=target_station)) >= t.demand if target_station < env.state.num_stations else False
        st_not_terminal = t.current_station < env.state.num_stations - 1
        
        station_map = {
            p_id: env.state.tasks[f"{t.aircraft_id}_{p_id}"].current_station
            for p_id in env.constraint_engine.physical_predecessors[t.task_id]
            if f"{t.aircraft_id}_{p_id}" in env.state.tasks
        }
        process_ok = env.constraint_engine.station_violation(t.task_id, target_station, station_map) is None if st_not_terminal else False

        is_static = st_not_terminal and workers_ok and process_ok
        if is_static:
            static_eligible.append(task_id)
            
        initial_postpone_masks[task_id] = can_p
        if err:
            blocked_by_successor[task_id] = err
            
        print(f"Task 0_{task_id:2d} (st={t.current_station+1}): can_reserve={can_r}, can_postpone={can_p}, static={is_static}, err={err}")

    print(f"\nTotal tasks for ac 0: {len(ac0_tasks)}")
    print(f"Statically eligible tasks for ac 0: {len(static_eligible)}, IDs: {static_eligible}")
    initial_true = [tid for tid, val in initial_postpone_masks.items() if val]
    print(f"Initial can_postpone=True for ac 0: {len(initial_true)}, IDs: {initial_true}")
    initial_false_among_static = [tid for tid in static_eligible if not initial_postpone_masks[tid]]
    print(f"Statically eligible but initial can_postpone=False: {len(initial_false_among_static)}, IDs: {initial_false_among_static}")
    for tid in initial_false_among_static:
        print(f"  Task 0_{tid}: blocked reason: {blocked_by_successor[tid]}")

if __name__ == "__main__":
    test_static_and_initial_masks()
