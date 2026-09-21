# -*- coding: utf-8 -*-
"""
双缓冲流水线与同步流水线性能对比基准测试脚本。
对比 2 环境与 4 环境下同步 vs 双缓冲，严格控制内存占用防止 OOM。
输出 SPS、阶段耗时分解（Forward、Env Step、IPC）及加速比。
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = sys.executable


def run_benchmark(num_envs: int, double_buffer: bool, max_steps: int = 48, repeats: int = 2, seed: int = 42) -> dict:
    cmd = [
        PYTHON_EXE,
        str(PROJECT_ROOT / "scripts" / "benchmark_rollout_fastpath.py"),
        "--data", str(PROJECT_ROOT / "data" / "283.csv"),
        "--num-envs", str(num_envs),
        "--max-steps", str(max_steps),
        "--repeats", str(repeats),
        "--seed", str(seed),
        "--ipc-fusion",
    ]
    if double_buffer:
        cmd.append("--double-buffer")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    mode_name = f"{num_envs} envs ({'Double-Buffer' if double_buffer else 'Sync Baseline'})"
    print(f"\n{'='*25} Running: {mode_name} {'='*25}", flush=True)
    
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    summary = None
    records = []

    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        try:
            print(line, end="", flush=True)
        except Exception:
            pass
        clean = line.strip()
        if clean.startswith("{") and clean.endswith("}"):
            try:
                data = json.loads(clean)
                if "mean_sps" in data:
                    summary = data
                elif "sps" in data:
                    records.append(data)
            except Exception:
                pass

    proc.stdout.close()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"Benchmark failed with return code {return_code}")

    if summary is None:
        raise ValueError("Could not find summary JSON in benchmark output")

    # 运行间内存清理与冷却
    gc.collect()
    time.sleep(2.0)

    return {
        "num_envs": num_envs,
        "double_buffer": double_buffer,
        "summary": summary,
        "records": records,
    }


def main():
    benchmarks = [
        {"num_envs": 2, "double_buffer": False},
        {"num_envs": 2, "double_buffer": True},
        {"num_envs": 4, "double_buffer": False},
        {"num_envs": 4, "double_buffer": True},
    ]

    results = []
    for b in benchmarks:
        res = run_benchmark(
            num_envs=b["num_envs"],
            double_buffer=b["double_buffer"],
            max_steps=48,
            repeats=2,
            seed=42,
        )
        results.append(res)

    print("\n" + "="*80)
    print("APAL DOUBLE-BUFFERING BENCHMARK RESULTS SUMMARY")
    print("="*80)
    
    table = []
    for r in results:
        s = r["summary"]
        recs = r["records"]
        avg_fwd = sum(rec["forward_ms"] for rec in recs) / len(recs) if recs else 0.0
        avg_env = sum(rec["environment_step_ms"] for rec in recs) / len(recs) if recs else 0.0
        avg_ipc = sum(rec["ipc_mask_ms"] for rec in recs) / len(recs) if recs else 0.0
        mode = "Double-Buffer" if r["double_buffer"] else "Sync Baseline"
        table.append({
            "mode": f"{r['num_envs']} Envs {mode}",
            "mean_sps": s["mean_sps"],
            "fwd_ms": avg_fwd,
            "env_ms": avg_env,
            "ipc_ms": avg_ipc,
            "vram_mb": s["peak_cuda_allocated_mb"],
        })

    print(f"{'Configuration':<28} | {'Mean SPS':<10} | {'Fwd (ms)':<10} | {'Env (ms)':<10} | {'VRAM (MB)':<10}")
    print("-" * 80)
    for row in table:
        print(f"{row['mode']:<28} | {row['mean_sps']:<10.2f} | {row['fwd_ms']:<10.2f} | {row['env_ms']:<10.2f} | {row['vram_mb']:<10.1f}")

    if len(table) >= 2:
        speedup_2 = table[1]["mean_sps"] / max(table[0]["mean_sps"], 1e-9)
        print(f"\n>>> 2-Environment Double-Buffer Speedup: {speedup_2:.2f}x ({((speedup_2 - 1.0)*100):+.1f}%)")
    if len(table) >= 4:
        speedup_4 = table[3]["mean_sps"] / max(table[2]["mean_sps"], 1e-9)
        print(f">>> 4-Environment Double-Buffer Speedup: {speedup_4:.2f}x ({((speedup_4 - 1.0)*100):+.1f}%)")

    output_path = PROJECT_ROOT / "logs" / "double_buffer_benchmark_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"table": table, "raw": results}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
