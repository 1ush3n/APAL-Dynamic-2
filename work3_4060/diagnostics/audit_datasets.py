"""
Comprehensive Data & Structural Audit Script for APAL-Dynamic-2.
Audits: 283, 680, 2338, 3182 raw datasets, reference schedules, and mini projections.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
R3_REF_DIR = DATA_DIR / "r3" / "b" / "r"

def audit_raw_csv(name: str, path: Path):
    df = pd.read_csv(path)
    # Check column names
    # Usually: 序号,AO号,类型,专业编码,工种,紧前工序AO号,需求人数,加工时间/h,限定站位,部位容量
    total_nodes = len(df)
    
    # Identify column names
    col_ao = "AO号" if "AO号" in df.columns else "task_id"
    col_type = "类型" if "类型" in df.columns else "node_type"
    col_skill = "工种" if "工种" in df.columns else "skill_type"
    col_preds = "紧前工序AO号" if "紧前工序AO号" in df.columns else "predecessors"
    col_demand = "需求人数" if "需求人数" in df.columns else "demand_workers"
    col_dur = "加工时间/h" if "加工时间/h" in df.columns else "duration"
    col_fixed = "限定站位" if "限定站位" in df.columns else "fixed_station"
    
    node_types = df[col_type].value_counts().to_dict()
    milestone_nodes = int((df[col_type] == 1).sum())
    physical_nodes = int((df[col_type] == 2).sum())
    
    # Skill counts for physical nodes
    phys_df = df[df[col_type] == 2]
    skill_counts = phys_df[col_skill].value_counts().to_dict()
    demand_counts = phys_df[col_demand].value_counts().to_dict()
    
    # Duration statistics for physical nodes
    durations = phys_df[col_dur].astype(float)
    dur_stats = {
        "min": float(durations.min()),
        "max": float(durations.max()),
        "mean": float(durations.mean()),
        "median": float(durations.median()),
        "sum": float(durations.sum()),
        "zero_count": int((durations == 0).sum()),
        "missing_count": int(durations.isna().sum())
    }
    
    # Top 5 longest tasks
    top5_longest = []
    top5_idx = durations.nlargest(5).index
    for idx in top5_idx:
        row = df.loc[idx]
        top5_longest.append({
            "ao": str(row[col_ao]),
            "duration": float(row[col_dur]),
            "skill": int(row[col_skill]),
            "demand": int(row[col_demand]),
            "fixed_station": str(row[col_fixed]) if pd.notna(row[col_fixed]) else None
        })
        
    # Fixed station tasks
    fixed_tasks = []
    for idx, row in df[df[col_fixed].notna()].iterrows():
        fixed_tasks.append({
            "ao": str(row[col_ao]),
            "type": int(row[col_type]),
            "fixed_station": int(float(row[col_fixed])),
            "duration": float(row[col_dur]),
            "skill": int(row[col_skill]) if pd.notna(row[col_skill]) else None,
            "demand": int(row[col_demand]) if pd.notna(row[col_demand]) else None
        })
        
    # Precedence edges
    edge_count = 0
    ao_set = set(df[col_ao].astype(str))
    missing_preds = []
    for idx, row in df.iterrows():
        preds_str = str(row[col_preds]) if pd.notna(row[col_preds]) else ""
        if preds_str.strip() and preds_str.strip().lower() != "nan":
            parts = [p.strip() for p in preds_str.split(",") if p.strip()]
            edge_count += len(parts)
            for p in parts:
                if p not in ao_set:
                    missing_preds.append((str(row[col_ao]), p))
                    
    return {
        "name": name,
        "total_nodes": total_nodes,
        "milestone_virtual_nodes": milestone_nodes,
        "physical_nodes": physical_nodes,
        "precedence_edges_count": edge_count,
        "missing_predecessor_refs": len(missing_preds),
        "skill_distribution_physical": skill_counts,
        "demand_distribution_physical": demand_counts,
        "duration_stats_physical": dur_stats,
        "top5_longest_tasks": top5_longest,
        "fixed_station_tasks": fixed_tasks,
        "ao_list": df[col_ao].astype(str).tolist()
    }

def audit_ref_schedule(name: str, path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Check columns
    # Real schedule columns often include: task_id, station_id, start_time, finish_time, etc.
    col_st = None
    for c in ["StationID", "station_id", "station", "Station", "工位", "站位"]:
        if c in df.columns:
            col_st = c
            break
            
    col_dur = None
    for c in ["Duration", "duration", "加工时间", "工时"]:
        if c in df.columns:
            col_dur = c
            break
            
    col_end = None
    for c in ["End", "finish_time", "end_time", "end", "完工时间"]:
        if c in df.columns:
            col_end = c
            break
            
    tasks_per_station = df[col_st].value_counts().sort_index().to_dict() if col_st else {}
    hours_per_station = df.groupby(col_st)[col_dur].sum().to_dict() if (col_st and col_dur) else {}
    makespan = float(df[col_end].max()) if col_end else None
    
    return {
        "name": name,
        "total_tasks_scheduled": len(df),
        "tasks_per_station": {str(k): int(v) for k, v in tasks_per_station.items()},
        "work_hours_per_station": {str(k): float(v) for k, v in hours_per_station.items()},
        "schedule_makespan": makespan
    }

def main():
    datasets = ["283", "680", "2338", "3182"]
    results = {}
    for d in datasets:
        path = DATA_DIR / f"{d}.csv"
        results[d] = audit_raw_csv(d, path)
        
    # Check nesting relations
    s283 = set(results["283"]["ao_list"])
    s680 = set(results["680"]["ao_list"])
    s2338 = set(results["2338"]["ao_list"])
    s3182 = set(results["3182"]["ao_list"])
    
    nesting = {
        "283_in_680": s283.issubset(s680),
        "283_not_in_680_count": len(s283 - s680),
        "680_in_2338": s680.issubset(s2338),
        "680_not_in_2338_count": len(s680 - s2338),
        "2338_in_3182": s2338.issubset(s3182),
        "2338_not_in_3182_count": len(s2338 - s3182),
        "283_in_3182": s283.issubset(s3182),
    }
    
    # Audit reference schedules
    ref_schedules = {}
    for d in datasets:
        ref_path = R3_REF_DIR / f"real_{d}_schedule.csv"
        ref_schedules[d] = audit_ref_schedule(d, ref_path)
        
    # Mini210 audit
    mini210_path = REPO_ROOT / "work3_4060" / "data" / "mini210.json"
    mini210_audit = None
    if mini210_path.exists():
        with open(mini210_path, "r", encoding="utf-8") as f:
            mini_data = json.load(f)
        # Check task keys and tasks
        tasks = mini_data.get("tasks", {})
        mini210_audit = {
            "num_tasks_in_json": len(tasks),
            "h0": mini_data.get("metadata", {}).get("h0"),
            "num_aircraft": mini_data.get("metadata", {}).get("num_aircraft", 7),
        }
        
    # Dump report
    output_path = REPO_ROOT / "work3_4060" / "diagnostics" / "data_structure_audit.json"
    clean_results = {}
    for k, v in results.items():
        v_copy = dict(v)
        del v_copy["ao_list"]
        clean_results[k] = v_copy
        
    report = {
        "raw_datasets": clean_results,
        "nesting_analysis": nesting,
        "reference_schedules": ref_schedules,
        "mini210_audit": mini210_audit
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"Data audit saved to {output_path}")
    print("\n=== RAW DATA AUDIT SUMMARY ===")
    for d in datasets:
        r = clean_results[d]
        print(f"Dataset {d:>4}: Total={r['total_nodes']:>4}, Phys={r['physical_nodes']:>4}, Virt={r['milestone_virtual_nodes']:>4}, Edges={r['precedence_edges_count']:>5}, MaxDur={r['duration_stats_physical']['max']:.1f}h, FixedTasks={len(r['fixed_station_tasks'])}")
    print("\n=== NESTING ANALYSIS ===")
    print(f"283 subset of 680?  {nesting['283_in_680']} (missing: {nesting['283_not_in_680_count']})")
    print(f"680 subset of 2338? {nesting['680_in_2338']} (missing: {nesting['680_not_in_2338_count']})")
    print(f"2338 subset of 3182? {nesting['2338_in_3182']} (missing: {nesting['2338_not_in_3182_count']})")
    print(f"283 subset of 3182? {nesting['283_in_3182']}")

    print("\n=== REFERENCE SCHEDULES DISTRIBUTION ===")
    for d, s in ref_schedules.items():
        if s:
            print(f"Dataset {d:>4}: Makespan={s['schedule_makespan']}h, TasksPerSt={s['tasks_per_station']}")

if __name__ == "__main__":
    main()
