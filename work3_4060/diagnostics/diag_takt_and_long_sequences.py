"""
Script: diag_takt_and_long_sequences.py
Verifies Nominal Takt Time vs Real Line Capacity across 1, 3, 7, 20, 50 Aircraft.
Compares:
  - Nominal Baseline (H0 = 52.0378h, nominal_anchored_v1)
  - Calibrated Takt  (H0 = 49.4424h, nominal_anchored_v1)
  - Pull Line / Asynchronous (actual_cleared_v0)
  - Old Compare Baseline (H0 = 292.5726h, nominal_anchored_v1)
Analyzes pipeline phases: Filling, Full-line Steady State, Draining.
Measures steady-state exit interval, throughput rate, and clearing margins.
"""

import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from work3_4060.diagnostics.postpone_compatible_agent import PostponeCompatibleRepairAgent
from work3_4060.diagnostics.reproduce_station3_clue import ImprovedStation3Agent

BASELINE_PATH = REPO_ROOT / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"

def simulate_long_sequence(num_aircraft: int, h0: float, transfer_policy: str, use_improved: bool):
    """Simulates a sequence of num_aircraft under specified clock regime and agent."""
    # We load baseline json template and expand to num_aircraft if necessary,
    # or simulate the station recurrence dynamically based on station clearing times.
    # In mini210:
    # Station 1: 14.227h (fixed team)
    # Station 2: 10.393h (baseline) or 8.178h (improved)
    # Station 3: 52.038h (baseline) or 49.442h (improved)
    # Station 4: 6.553h
    # Station 5: 5.504h
    
    # Station durations per plane
    if use_improved:
        st_durations = [14.227162, 8.178273, 49.442367, 6.552986, 5.503731]
    else:
        st_durations = [14.227162, 10.393067, 52.037844, 6.552986, 5.503731]
        
    num_stations = 5
    total_cycles = num_aircraft + num_stations - 1
    
    # Track cycle times and transfers
    cycle_records = []
    current_time = 0.0
    last_transfer_time = 0.0
    plane_exits = []
    
    # Aircraft positions: plane_id -> station_id (0 to 4, or -1 before entry, 5 after exit)
    # At cycle q (1-indexed, q = 1..total_cycles):
    for q in range(1, total_cycles + 1):
        cycle_start = last_transfer_time
        # Determine which plane is at which station
        # Plane k enters Station 0 at cycle k+1 (1-indexed: plane 0 enters at cycle 1)
        active_stations = {}
        for k in range(num_aircraft):
            # Plane k station at cycle q:
            # At cycle q, plane k is at station s = (q - 1) - k
            s = (q - 1) - k
            if 0 <= s < num_stations:
                active_stations[s] = k
                
        # Phase identification
        if q < num_stations:
            phase = "filling"
        elif q <= num_aircraft:
            phase = "full_line"
        else:
            phase = "draining"
            
        # Clearing time of each active station in this cycle
        latest_finish = cycle_start
        cleared_times = {}
        for s, k in active_stations.items():
            st_dur = st_durations[s]
            fin = cycle_start + st_dur
            cleared_times[s] = fin
            if fin > latest_finish:
                latest_finish = fin
                
        # Transfer time calculation based on transfer_policy
        if transfer_policy == "nominal_anchored_v1":
            nominal_boundary = float(q) * float(h0)
            transfer_time = max(nominal_boundary, last_transfer_time, latest_finish)
        elif transfer_policy == "actual_cleared_v0":
            transfer_time = max(latest_finish, cycle_start)
        else:
            raise ValueError(f"Unknown transfer policy {transfer_policy}")
            
        cycle_duration = transfer_time - cycle_start
        delay_rel_nominal = max(0.0, transfer_time - q * h0)
        
        # Check if any plane exited at this transfer
        # Plane k exits when it finishes station 4 (i.e. s == 4)
        if 4 in active_stations:
            exiting_plane = active_stations[4]
            plane_exits.append((exiting_plane, transfer_time))
            
        cycle_records.append({
            "cycle": q,
            "phase": phase,
            "cycle_start": cycle_start,
            "transfer_time": transfer_time,
            "cycle_duration": cycle_duration,
            "delay_rel_nominal": delay_rel_nominal,
            "active_stations": active_stations
        })
        last_transfer_time = transfer_time
        
    makespan = last_transfer_time
    
    # Calculate steady-state metrics (full-line phase)
    full_line_cycles = [c for c in cycle_records if c["phase"] == "full_line"]
    if full_line_cycles:
        steady_state_cycle_duration = np.mean([c["cycle_duration"] for c in full_line_cycles])
        steady_state_throughput = 1000.0 / steady_state_cycle_duration # planes per 1000 hours
    else:
        steady_state_cycle_duration = None
        steady_state_throughput = None
        
    # Exit intervals in steady state
    if len(plane_exits) >= 2:
        intervals = [plane_exits[i][1] - plane_exits[i-1][1] for i in range(1, len(plane_exits))]
        steady_state_exit_interval = np.mean(intervals)
    else:
        steady_state_exit_interval = None

    return {
        "num_aircraft": num_aircraft,
        "h0": h0,
        "transfer_policy": transfer_policy,
        "use_improved": use_improved,
        "total_cycles": total_cycles,
        "makespan": makespan,
        "steady_state_cycle_duration": steady_state_cycle_duration,
        "steady_state_exit_interval": steady_state_exit_interval,
        "steady_state_throughput_per_1000h": steady_state_throughput,
        "first_plane_exit": plane_exits[0][1] if plane_exits else None,
        "last_plane_exit": plane_exits[-1][1] if plane_exits else None,
    }

def main():
    print("=" * 80)
    print("NOMINAL TAKT VS REAL CAPACITY ACROSS BATCH SIZES (1, 3, 7, 20, 50)")
    print("=" * 80)

    configs = [
        # (name, h0, transfer_policy, use_improved)
        ("Baseline_Nominal_H52", 52.037844, "nominal_anchored_v1", False),
        ("Improved_Nominal_FixedH52", 52.037844, "nominal_anchored_v1", True),
        ("Improved_Calibrated_H49", 49.442367, "nominal_anchored_v1", True),
        ("Baseline_ActualCleared_Pull", 52.037844, "actual_cleared_v0", False),
        ("Improved_ActualCleared_Pull", 49.442367, "actual_cleared_v0", True),
        ("Old_Compare_Baseline_H292", 292.572600, "nominal_anchored_v1", False),
    ]

    batch_sizes = [1, 3, 7, 20, 50]
    all_results = []

    for name, h0, policy, imp in configs:
        for n_ac in batch_sizes:
            res = simulate_long_sequence(n_ac, h0, policy, imp)
            res["regime"] = name
            all_results.append(res)
            
    df = pd.DataFrame(all_results)
    
    # Print formatted summary table for N=7, N=20, N=50
    print("\n--- COMPARISON AT N=7 (ORIGINAL BATCH) ---")
    df7 = df[df["num_aircraft"] == 7][["regime", "makespan", "first_plane_exit", "last_plane_exit", "steady_state_exit_interval", "steady_state_throughput_per_1000h"]]
    print(df7.to_string(index=False))

    print("\n--- COMPARISON AT N=20 (EXTENDED BATCH) ---")
    df20 = df[df["num_aircraft"] == 20][["regime", "makespan", "first_plane_exit", "last_plane_exit", "steady_state_exit_interval", "steady_state_throughput_per_1000h"]]
    print(df20.to_string(index=False))

    print("\n--- COMPARISON AT N=50 (LONG PRODUCTION SEQUENCE) ---")
    df50 = df[df["num_aircraft"] == 50][["regime", "makespan", "first_plane_exit", "last_plane_exit", "steady_state_exit_interval", "steady_state_throughput_per_1000h"]]
    print(df50.to_string(index=False))

    # Save CSV
    csv_path = REPO_ROOT / "work3_4060" / "diagnostics" / "summary_tables" / "takt_capacity_long_sequences.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nFull table saved to {csv_path}")

    # Save JSON
    json_path = REPO_ROOT / "work3_4060" / "diagnostics" / "takt_capacity_long_sequences.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
