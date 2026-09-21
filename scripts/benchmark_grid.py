# -*- coding: utf-8 -*-
"""系统对比不同 worker_threads, ipc_fusion 和 num_envs 组合下的真实训练吞吐。"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from scripts.profile_diagnostic import run_benchmark


def main():
    experiments = [
        {"tag": "T_auto_IPC_false", "worker_threads": "auto", "ipc_fusion": False, "num_envs": 4},
        {"tag": "T_1_IPC_false",    "worker_threads": 1,      "ipc_fusion": False, "num_envs": 4},
        {"tag": "T_1_IPC_true",     "worker_threads": 1,      "ipc_fusion": True,  "num_envs": 4},
    ]

    all_results = []
    for exp in experiments:
        print(f"\n>>> Running Experiment: {exp['tag']} (envs={exp['num_envs']}, threads={exp['worker_threads']}, fusion={exp['ipc_fusion']})")
        res, _ = run_benchmark(
            data_path="data/283.csv",
            num_envs=exp["num_envs"],
            worker_threads=exp["worker_threads"],
            ipc_fusion=exp["ipc_fusion"],
            max_steps=100,
            episodes=2,
            seed=42,
            profile_cuda=False,
            tag=exp["tag"],
        )
        all_results.append(res)
        print(f"  Result [{exp['tag']}]: SPS={res['overall_sps']:.2f}, WallTime={res['total_wall_time']:.2f}s (Rollout={res['total_rollout_sec']:.2f}s, PPO={res['total_update_sec']:.2f}s)")
        print(f"  Per step ms: IPC+Mask={res['mean_ipc_mask_ms']:.2f}ms, EnvStep={res['mean_env_step_ms']:.2f}ms, Rebuild={res['mean_rebuild_ms']:.2f}ms, Fwd={res['mean_forward_ms']:.2f}ms")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_file = PROJECT_ROOT / "benchmark_grid_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll benchmark results written to {out_file}")

if __name__ == "__main__":
    main()
