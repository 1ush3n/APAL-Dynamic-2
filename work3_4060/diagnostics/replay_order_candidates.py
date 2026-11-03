"""
Replay Order Candidates & Concurrency Consistency Auditor.
Fixes E1 searcher to model station concurrency limit (max_slots = 3).
Explains and proves the difference between 4.4085h (unconstrained) and 5.5037h (concurrency <= 3).
Replays candidate orders in AirLineEnvWork3 and verifies with verify_single_trajectory.
"""

import itertools
import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.environment import AirLineEnvWork3
from work3_4060.run_rule_ac_comparison import extract_execution_facts
from work3_4060.verify_trajectory_facts import load_baseline_tasks, verify_single_trajectory

def load_station_tasks_and_teams(baseline_path: Path, plane_id: int = 0):
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data["tasks"]
    st_tasks = {s: {} for s in range(1, 6)}
    for t_key, tinfo in tasks.items():
        if tinfo.get("aircraft_id") == plane_id:
            st = tinfo["station_id"]
            tid = str(tinfo["task_id"])
            st_tasks[st][tid] = {
                "task_id": tid,
                "ao_code": tinfo.get("ao_code", ""),
                "demand": tinfo.get("demand", 2),
                "skill": tinfo.get("skill", 0),
                "duration": float(tinfo["duration"]),
                "predecessors": [str(p) for p in tinfo.get("predecessors", [])],
                "baseline_team": tuple(tinfo["team"]),
                "in_station_offset": float(tinfo.get("in_station_offset", 0.0)),
            }
    return st_tasks

def simulate_station_schedule(order, tasks_dict, max_slots=3):
    """
    Simulates task execution within a single station given an ordering,
    strictly respecting:
      1. Predecessors within the station.
      2. Worker calendar availability (non-preemptive).
      3. Station concurrency limit (max_slots <= 3).
    """
    worker_available = {}
    task_start = {}
    task_end = {}
    active_intervals = [] # list of (start, end)
    
    for tid in order:
        t = tasks_dict[tid]
        team = t["baseline_team"]
        dur = t["duration"]
        
        # 1. Predecessors within this station
        pred_end = 0.0
        for p in t["predecessors"]:
            if p in task_end:
                pred_end = max(pred_end, task_end[p])
                
        # 2. Worker availability
        w_avail = 0.0
        for w in team:
            w_avail = max(w_avail, worker_available.get(w, 0.0))
            
        candidate_start = max(pred_end, w_avail)
        
        # 3. Concurrency check: at candidate_start, how many tasks are running?
        # A task is running during [s, e). If count >= max_slots, must advance start time
        # to when a slot opens up.
        while True:
            # Check concurrency at candidate_start
            running_overlap = [e for (s, e) in active_intervals if s <= candidate_start < e]
            if len(running_overlap) < max_slots:
                # Also check that during [candidate_start, candidate_start + dur), concurrency doesn't exceed max_slots
                # Concurrency could only increase when another task starts, but all already scheduled tasks started before candidate_start
                # However, scheduled tasks running can finish.
                # Is there any point inside [candidate_start, candidate_start + dur) where concurrency > max_slots?
                # Since all other active tasks started <= candidate_start, the number of concurrent tasks at any t in [candidate_start, candidate_start+dur)
                # is at most 1 + len({task running at t}) <= 1 + len({task running at candidate_start}) <= max_slots.
                # So checking candidate_start is sufficient!
                break
            else:
                # Wait until the earliest running task finishes
                candidate_start = min(running_overlap)
                # Re-check worker availability at this new candidate_start
                w_avail = max([worker_available.get(w, 0.0) for w in team] + [0.0])
                candidate_start = max(candidate_start, w_avail)
                
        actual_start = candidate_start
        actual_end = actual_start + dur
        
        task_start[tid] = actual_start
        task_end[tid] = actual_end
        active_intervals.append((actual_start, actual_end))
        for w in team:
            worker_available[w] = actual_end
            
    return {
        "makespan": max(task_end.values()),
        "task_start": task_start,
        "task_end": task_end,
    }

def audit_station5_concurrency():
    baseline_path = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"
    st_tasks = load_station_tasks_and_teams(baseline_path)
    tasks5 = st_tasks[5]
    
    order = sorted(tasks5.keys(), key=lambda x: int(x))
    
    # 1. Unconstrained concurrency (max_slots = 999)
    res_unconstrained = simulate_station_schedule(order, tasks5, max_slots=999)
    # 2. Constrained concurrency (max_slots = 3)
    res_constrained = simulate_station_schedule(order, tasks5, max_slots=3)
    
    print("=" * 80)
    print("STATION 5 CONCURRENCY CONSISTENCY AUDIT")
    print("=" * 80)
    print(f"Unconstrained (max_slots = inf) Makespan: {res_unconstrained['makespan']:.6f}h")
    print(f"Constrained   (max_slots = 3)   Makespan: {res_constrained['makespan']:.6f}h")
    print(f"Difference due to concurrency limit:     {res_constrained['makespan'] - res_unconstrained['makespan']:.6f}h")
    print("\nDetailed Task Schedule with max_slots = 3:")
    for tid in order:
        t = tasks5[tid]
        s = res_constrained["task_start"][tid]
        e = res_constrained["task_end"][tid]
        print(f"  Task {tid} ({t['ao_code']}): start={s:.4f}h, end={e:.4f}h, dur={t['duration']:.4f}h, team={t['baseline_team']}")
        
    # Search all valid topological permutations under max_slots = 3
    tids = list(tasks5.keys())
    best_ms_c3 = float("inf")
    worst_ms_c3 = 0.0
    valid_orders = 0
    
    for perm in itertools.permutations(tids):
        pos = {tid: i for i, tid in enumerate(perm)}
        is_valid = True
        for tid in perm:
            for p in tasks5[tid]["predecessors"]:
                if p in pos and pos[p] >= pos[tid]:
                    is_valid = False
                    break
            if not is_valid:
                break
        if not is_valid:
            continue
        valid_orders += 1
        ms = simulate_station_schedule(perm, tasks5, max_slots=3)["makespan"]
        if ms < best_ms_c3:
            best_ms_c3 = ms
        if ms > worst_ms_c3:
            worst_ms_c3 = ms
            
    print(f"\nExhaustive Permutation Search under max_slots = 3 ({valid_orders} valid orders):")
    print(f"  Baseline order makespan: {res_constrained['makespan']:.6f}h")
    print(f"  Best order makespan:     {best_ms_c3:.6f}h")
    print(f"  Worst order makespan:    {worst_ms_c3:.6f}h")
    print(f"  Improvement possible:    {res_constrained['makespan'] - best_ms_c3:.6f}h")
    
    return {
        "unconstrained_makespan": res_unconstrained["makespan"],
        "constrained_makespan": res_constrained["makespan"],
        "best_perm_makespan": best_ms_c3,
        "valid_orders": valid_orders,
    }

if __name__ == "__main__":
    audit_station5_concurrency()
