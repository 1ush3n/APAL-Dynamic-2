import sys
from pathlib import Path
import threading
import time
import subprocess
import numpy as np
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.profile_diagnostic import run_benchmark

def monitor_gpu(stop_event, samples, interval=0.1):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            ).strip()
            parts = [x.strip() for x in out.split(",")]
            gpu_util = float(parts[0])
            mem_used = float(parts[1])
            samples.append((time.time(), gpu_util, mem_used))
        except Exception:
            pass
        time.sleep(interval)

def main():
    samples = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpu, args=(stop_event, samples, 0.1))
    monitor_thread.daemon = True
    monitor_thread.start()

    t_start = time.time()
    res, _ = run_benchmark(
        data_path="data/283.csv",
        num_envs=4,
        worker_threads="auto",
        ipc_fusion=True,
        max_steps=100,
        episodes=2,
        tag="gpu_util_measure",
    )
    t_end = time.time()
    stop_event.set()
    monitor_thread.join(timeout=2.0)

    overall_utils = [s[1] for s in samples if t_start <= s[0] <= t_end]
    overall_mems = [s[2] for s in samples if t_start <= s[0] <= t_end]

    result = {
        "dataset": "data/283.csv",
        "num_envs": 4,
        "total_duration_sec": t_end - t_start,
        "sample_count": len(overall_utils),
        "overall_avg_gpu_util_pct": float(np.mean(overall_utils)) if overall_utils else 0.0,
        "overall_median_gpu_util_pct": float(np.median(overall_utils)) if overall_utils else 0.0,
        "overall_std_gpu_util_pct": float(np.std(overall_utils)) if overall_utils else 0.0,
        "max_gpu_util_pct": float(np.max(overall_utils)) if overall_utils else 0.0,
        "avg_vram_mb": float(np.mean(overall_mems)) if overall_mems else 0.0,
        "max_vram_mb": float(np.max(overall_mems)) if overall_mems else 0.0,
        "ratio_under_15_pct": float(np.mean([1 if u < 15 else 0 for u in overall_utils]) * 100) if overall_utils else 0.0,
        "ratio_15_to_50_pct": float(np.mean([1 if 15 <= u < 50 else 0 for u in overall_utils]) * 100) if overall_utils else 0.0,
        "ratio_above_50_pct": float(np.mean([1 if u >= 50 else 0 for u in overall_utils]) * 100) if overall_utils else 0.0,
    }

    print("\n" + "=" * 65)
    print("EMPIRICAL GPU UTILIZATION REPORT (data/283.csv, 4 envs)")
    print("=" * 65)
    print(f"Total Measured Time:        {result['total_duration_sec']:.2f} s ({result['sample_count']} samples)")
    print(f"Overall Average GPU Util:   {result['overall_avg_gpu_util_pct']:.1f}%")
    print(f"Overall Median GPU Util:    {result['overall_median_gpu_util_pct']:.1f}%")
    print(f"Standard Deviation:         {result['overall_std_gpu_util_pct']:.1f}%")
    print(f"Peak GPU Utilization:       {result['max_gpu_util_pct']:.1f}%")
    print(f"Peak VRAM Usage:            {result['max_vram_mb']:.1f} MB")
    print("-" * 65)
    print("Time Distribution by Load Bracket:")
    print(f"  Low (< 15%, CPU gaps/syncs):     {result['ratio_under_15_pct']:.1f}% of execution time")
    print(f"  Medium (15% - 50%, GNN Rollout): {result['ratio_15_to_50_pct']:.1f}% of execution time")
    print(f"  High (>= 50%, PPO Update/Dense): {result['ratio_above_50_pct']:.1f}% of execution time")
    print("=" * 65)

    with open("gpu_utilization_283.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
