"""
Audit Skill Scarcity and Exact Station Lower Bounds across Mini210, 283, 680, 2338, 3182.
Searches for rare-skill lockup and computes exact station lower bounds.
"""

import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.constraints import calculate_team_synergy_factor

def load_workers(pool_path: Path):
    df = pd.read_csv(pool_path)
    workers = {}
    for _, r in df.iterrows():
        wid = int(r["worker_id"])
        eff = float(r["efficiency"])
        skills = set()
        for s in range(5):
            if float(r.get(f"skill_{s}", 0.0)) >= 0.5:
                skills.add(s)
        workers[wid] = {"id": wid, "eff": eff, "skills": skills}
    return workers

def analyze_station_skill_scarcity(station_name: str, station_tasks: list, station_workers: list, all_workers: dict):
    # Worker skill counts
    skill_workers = {s: [] for s in range(5)}
    for wid in station_workers:
        w = all_workers[wid]
        for s in w["skills"]:
            skill_workers[s].append(wid)
            
    # Task skill demand
    task_skills = {s: [] for s in range(5)}
    total_work_hours = 0.0
    skill_work_hours = {s: 0.0 for s in range(5)}
    max_single_task_duration = 0.0
    longest_task = None
    
    for t in station_tasks:
        s = t["skill"]
        dur = t.get("standard_duration", t.get("duration", 0.0))
        dem = t["demand"]
        if s >= 0:
            task_skills[s].append(t)
            # Estimate duration on best available team
            # Find best team for this task
            eligible = [wid for wid in station_workers if s in all_workers[wid]["skills"]]
            if len(eligible) >= dem:
                # Top dem by eff
                top_w = sorted(eligible, key=lambda w: all_workers[w]["eff"], reverse=True)[:dem]
                eff_sum = sum(all_workers[w]["eff"] for w in top_w)
                syn = calculate_team_synergy_factor(dem)
                act_dur = dur * dem / (eff_sum * syn)
            else:
                act_dur = dur
            if act_dur > max_single_task_duration:
                max_single_task_duration = act_dur
                longest_task = t
            total_work_hours += dur * dem
            skill_work_hours[s] += dur * dem

    # Worker capacity by skill
    skill_worker_capacity = {}
    for s in range(5):
        w_list = skill_workers[s]
        total_eff = sum(all_workers[w]["eff"] for w in w_list)
        skill_worker_capacity[s] = total_eff

    # Lower bound 1: longest task duration
    lb_task = max_single_task_duration
    
    # Lower bound 2: total work load / total worker capacity
    total_st_eff = sum(all_workers[w]["eff"] for w in station_workers)
    lb_total_load = total_work_hours / total_st_eff if total_st_eff > 0 else 0.0
    
    # Lower bound 3: skill specific work load / skill worker capacity
    lb_skill = 0.0
    bottleneck_skill = None
    for s in range(5):
        if skill_work_hours[s] > 0 and skill_worker_capacity[s] > 0:
            lb_s = skill_work_hours[s] / skill_worker_capacity[s]
            if lb_s > lb_skill:
                lb_skill = lb_s
                bottleneck_skill = s
                
    lb_station = max(lb_task, lb_total_load, lb_skill)
    
    # Check for potential skill lockup
    # A skill lockup risk happens when a skill is scarce (e.g. <= 2 workers) but those workers are also used for common skills
    lockup_risks = []
    for s, w_list in skill_workers.items():
        if 0 < len(w_list) <= 2 and len(task_skills[s]) > 0:
            # Check if these workers also have other skills that have many tasks
            other_tasks = sum(len(task_skills[os]) for os in range(5) if os != s and any(os in all_workers[w]["skills"] for w in w_list))
            if other_tasks > 0:
                lockup_risks.append({
                    "scarce_skill": s,
                    "qualified_workers": w_list,
                    "num_tasks_needing_scarce_skill": len(task_skills[s]),
                    "other_tasks_competing_for_same_workers": other_tasks
                })
                
    return {
        "station_name": station_name,
        "num_tasks": len(station_tasks),
        "num_workers": len(station_workers),
        "workers": station_workers,
        "worker_skills_count": {s: len(w_list) for s, w_list in skill_workers.items()},
        "task_skills_count": {s: len(t_list) for s, t_list in task_skills.items()},
        "total_standard_work_hours": total_work_hours,
        "longest_task": longest_task["ao_code"] if longest_task else None,
        "longest_task_est_dur": max_single_task_duration,
        "lower_bound_longest_task": lb_task,
        "lower_bound_total_load": lb_total_load,
        "lower_bound_skill_load": lb_skill,
        "bottleneck_skill": bottleneck_skill,
        "station_lower_bound": lb_station,
        "lockup_risks": lockup_risks
    }

def main():
    pool_path = REPO_ROOT / "data" / "worker_pool_fixed.csv"
    all_workers = load_workers(pool_path)
    
    # Analyze mini210 stations
    resources_path = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "resources.json"
    template_path = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "rule_template.json"
    
    with open(resources_path, "r", encoding="utf-8") as f:
        res_data = json.load(f)
    with open(template_path, "r", encoding="utf-8") as f:
        tmpl_data = json.load(f)
        
    st_workers = res_data["station_workers"]
    tasks_by_st = {int(s): [] for s in range(1, 6)}
    for tid, tinfo in tmpl_data["tasks"].items():
        st = tinfo["station_id"]
        tasks_by_st[st].append(tinfo)
        
    mini210_station_analysis = []
    print("=" * 80)
    print("MINI210 ALL STATIONS SKILL SCARCITY & LOWER BOUND AUDIT")
    print("=" * 80)
    for st in range(1, 6):
        w_list = st_workers[str(st)]
        t_list = tasks_by_st[st]
        analysis = analyze_station_skill_scarcity(f"Station_{st}", t_list, w_list, all_workers)
        mini210_station_analysis.append(analysis)
        print(f"\nStation {st}: {analysis['num_tasks']} tasks, {analysis['num_workers']} workers")
        print(f"  Worker counts per skill: {analysis['worker_skills_count']}")
        print(f"  Task counts per skill:   {analysis['task_skills_count']}")
        print(f"  Longest task: {analysis['longest_task']} ({analysis['longest_task_est_dur']:.2f}h)")
        print(f"  Station Lower Bound: {analysis['station_lower_bound']:.2f}h (Task={analysis['lower_bound_longest_task']:.2f}h, Load={analysis['lower_bound_total_load']:.2f}h, Skill={analysis['lower_bound_skill_load']:.2f}h)")
        if analysis['lockup_risks']:
            print(f"  * LOCKUP RISKS DETECTED: {analysis['lockup_risks']}")
        else:
            print("  - No obvious scarce-skill lockup risk.")
            
    # Save output
    out_path = REPO_ROOT / "work3_4060" / "diagnostics" / "skill_scarcity_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mini210_station_analysis, f, indent=2)
    print(f"\nDetailed skill scarcity audit saved to {out_path}")

if __name__ == "__main__":
    main()
