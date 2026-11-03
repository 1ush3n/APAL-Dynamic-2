"""
Rigorous E1 Order Search under Fixed Baseline Teams.
Exhaustively searches all legal topological orderings within Station 3 and Station 2
keeping the baseline's team assignments fixed.
Proves whether reordering alone (without changing teams) can yield any improvement.
"""

import itertools
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def load_station_tasks_and_teams(baseline_path: Path):
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = data["tasks"]
    
    # Extract tasks by station for aircraft 0
    st_tasks = {s: {} for s in range(1, 6)}
    for t_key, tinfo in tasks.items():
        if tinfo.get("aircraft_id") == 0:
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

def simulate_order(order, tasks_dict):
    # Simulates execution of tasks in given order on their fixed baseline teams
    # Returns makespan
    worker_available = {}
    task_end = {}
    
    for tid in order:
        t = tasks_dict[tid]
        team = t["baseline_team"]
        # Predecessor constraint within this station
        pred_end = 0.0
        for p in t["predecessors"]:
            if p in task_end:
                pred_end = max(pred_end, task_end[p])
        # Worker availability
        w_avail = 0.0
        for w in team:
            w_avail = max(w_avail, worker_available.get(w, 0.0))
            
        start = max(pred_end, w_avail)
        dur = t["duration"]
        end = start + dur
        
        task_end[tid] = end
        for w in team:
            worker_available[w] = end
            
    return max(task_end.values())

def exhaustive_search_e1():
    baseline_path = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"
    st_tasks = load_station_tasks_and_teams(baseline_path)
    
    results = {}
    for st_num in range(1, 6):
        tasks = st_tasks[st_num]
        tids = list(tasks.keys())
        if not tids:
            continue
            
        # Check all permutations
        valid_orders = 0
        best_makespan = float("inf")
        worst_makespan = 0.0
        order_makespans = []
        
        for perm in itertools.permutations(tids):
            # Check DAG validity
            pos = {tid: i for i, tid in enumerate(perm)}
            is_valid = True
            for tid in perm:
                for p in tasks[tid]["predecessors"]:
                    if p in pos and pos[p] >= pos[tid]:
                        is_valid = False
                        break
                if not is_valid:
                    break
            if not is_valid:
                continue
                
            valid_orders += 1
            ms = simulate_order(perm, tasks)
            order_makespans.append(ms)
            if ms < best_makespan:
                best_makespan = ms
            if ms > worst_makespan:
                worst_makespan = ms
                
        baseline_order = sorted(tids, key=lambda x: int(x))
        baseline_ms = simulate_order(baseline_order, tasks)
        
        results[f"Station_{st_num}"] = {
            "num_tasks": len(tids),
            "total_permutations": len(list(itertools.permutations(tids))),
            "valid_topological_orders": valid_orders,
            "baseline_order": baseline_order,
            "baseline_makespan": round(baseline_ms, 6),
            "best_order_makespan": round(best_makespan, 6),
            "worst_order_makespan": round(worst_makespan, 6),
            "improvement_possible": round(baseline_ms - best_makespan, 6)
        }
        
    out_path = REPO_ROOT / "work3_4060" / "diagnostics" / "e1_exhaustive_order_search.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("=" * 70)
    print("E1 EXHAUSTIVE ORDER SEARCH RESULTS ACROSS ALL 5 STATIONS (FIXED BASELINE TEAMS)")
    print("=" * 70)
    for st_name, res in results.items():
        print(f"\n{st_name}:")
        print(f"  Valid topological orders: {res['valid_topological_orders']} / {res['total_permutations']}")
        print(f"  Baseline makespan:        {res['baseline_makespan']:.6f}h")
        print(f"  Best order makespan:      {res['best_order_makespan']:.6f}h")
        print(f"  Worst order makespan:     {res['worst_order_makespan']:.6f}h")
        print(f"  Improvement possible:     {res['improvement_possible']:.6f}h")
        if res['improvement_possible'] == 0.0:
            print(f"  >> PROVEN: In {st_name}, no topological ordering can reduce makespan below baseline with fixed teams!")
        else:
            print(f"  >> Order can reduce makespan by {res['improvement_possible']:.6f}h!")
            
    return results

if __name__ == "__main__":
    exhaustive_search_e1()
