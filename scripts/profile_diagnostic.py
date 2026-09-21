# -*- coding: utf-8 -*-
"""APAL 训练链路端到端性能瓶颈深度诊断脚本。

诊断维度：
1. Rollout 各阶段精确耗时细分 (Environment step, Action mask, IPC, Rebuild, Policy forward, Action decode, CUDA sync)；
2. PPO Update 阶段耗时细分 (GAE, Graph preparation, GPU forward, Backward, Optimizer step)；
3. IPC 命令数、字节数与序列化耗时；
4. PyTorch CPU 线程数配置对比 (worker_threads = 1, 2, auto)；
5. GPU 活动剖析 (通过 torch.profiler 采集 CUDA kernel, memcpy, aten::item 等)；
6. 吞吐量 (Steps/s) 与 Amdahl 定律分析。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from configs import configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.seed import set_seed
from training.rollout_service import APALRolloutService
from utils.vector_env import EnvCreator, VectorEnv


def run_benchmark(
    *,
    data_path: str = "data/283.csv",
    num_envs: int = 4,
    worker_threads: Any = "auto",
    ipc_fusion: bool = False,
    max_steps: int = 100,
    episodes: int = 2,
    seed: int = 42,
    profile_cuda: bool = False,
    tag: str = "benchmark",
) -> dict[str, Any]:
    resolved_data = str((PROJECT_ROOT / data_path).resolve())
    set_seed(seed)

    overrides = {
        "data_file_path": resolved_data,
        "train_data_path_or_dir": resolved_data,
        "num_envs": int(num_envs),
        "rollout_max_steps": int(max_steps),
        "seed": int(seed),
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
        "enable_shadow_mask_verification": False,
        "rollout_heartbeat_interval_sec": 0.0,
        "enable_rollout_ipc_fusion": bool(ipc_fusion),
        "vector_env_worker_threads": worker_threads,
        "k_epochs": 2,
        "batch_size": 32,
    }
    configs.update_from_dict(overrides)

    start_method = "forkserver" if platform.system() == "Linux" else "spawn"
    vector_env = VectorEnv(
        EnvCreator(resolved_data, seed_offset=seed),
        num_envs=num_envs,
        start_method=start_method,
        worker_threads=worker_threads,
        init_timeout_sec=120.0,
        command_timeout_sec=120.0,
    )
    eval_env = AirLineEnv_Graph(resolved_data, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HBGATPN(configs).to(device)
    agent = PPOAgent(
        model=model,
        lr=float(configs.lr),
        gamma=float(configs.gamma),
        k_epochs=int(configs.k_epochs),
        eps_clip=float(configs.eps_clip),
        device=device,
        batch_size=int(configs.batch_size),
        total_timesteps=episodes,
        config=configs,
    )
    service = APALRolloutService(
        agent=agent,
        vector_env=vector_env,
        eval_env=eval_env,
        config=configs,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    actual_worker_threads = vector_env.worker_threads
    resolved_threads_info = [audit.get("torch_num_threads") for audit in vector_env.worker_audits]

    rollout_records = []
    update_records = []

    # CPU/GPU 监控初始采样
    cpu_percentages = []
    process = psutil.Process(os.getpid())

    total_wall_start = time.perf_counter()

    cuda_prof = None
    if profile_cuda and device.type == "cuda":
        cuda_prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        cuda_prof.__enter__()

    try:
        for ep in range(1, episodes + 1):
            cpu_percentages.append(psutil.cpu_percent(interval=None))
            # 1. 采集 Rollout
            memories, metrics = service._collect_episode(ep)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            rollout_info = {
                "episode": ep,
                "steps": metrics.environment_steps,
                "total_seconds": metrics.total_seconds,
                "sps": metrics.steps_per_second,
                "ipc_mask_ms": metrics.ipc_mask_ms,
                "forward_ms": metrics.forward_ms,
                "rebuild_ms": metrics.rebuild_ms,
                "env_step_ms": metrics.environment_step_ms,
            }
            rollout_records.append(rollout_info)

            # 2. 模拟 PPO Update
            merged = memories[0]
            for m in memories[1:]:
                merged.states.extend(m.states)
                merged.actions.extend(m.actions)
                merged.logprobs.extend(m.logprobs)
                merged.rewards.extend(m.rewards)
                merged.is_terminals.extend(m.is_terminals)
                merged.is_truncated.extend(m.is_truncated)
                merged.masks.extend(m.masks)
                merged.values.extend(m.values)
                merged.old_task_logprob.extend(m.old_task_logprob)
                merged.old_station_logprob.extend(m.old_station_logprob)
                merged.old_team_logprob.extend(m.old_team_logprob)
                merged.old_V_task.extend(m.old_V_task)
                merged.old_V_station.extend(m.old_V_station)
                merged.old_V_worker.extend(m.old_V_worker)
                merged.gated_team_traces.extend(m.gated_team_traces)
                merged.anchor_proposal_traces.extend(m.anchor_proposal_traces)
                merged.worker_pointer_v2_behavior_traces.extend(m.worker_pointer_v2_behavior_traces)

            t_update_start = time.perf_counter()
            agent.validate_snapshot_homogeneity(merged.states)
            update_metrics = agent.update(merged, eval_env, current_ep=ep)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_update = time.perf_counter() - t_update_start
            update_records.append(t_update)

            cpu_percentages.append(psutil.cpu_percent(interval=None))
    finally:
        if cuda_prof is not None:
            cuda_prof.__exit__(None, None, None)
        vector_env.close()

    total_wall_time = time.perf_counter() - total_wall_start
    total_steps = sum(r["steps"] for r in rollout_records)
    total_rollout_sec = sum(r["total_seconds"] for r in rollout_records)
    total_update_sec = sum(update_records)

    cuda_summary = ""
    if cuda_prof is not None:
        cuda_summary = cuda_prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)

    result = {
        "tag": tag,
        "data": data_path,
        "num_envs": num_envs,
        "worker_threads": actual_worker_threads,
        "resolved_worker_threads": resolved_threads_info,
        "ipc_fusion": ipc_fusion,
        "episodes": episodes,
        "total_steps": total_steps,
        "total_wall_time": total_wall_time,
        "overall_sps": total_steps / max(1e-6, total_wall_time),
        "rollout_sps": total_steps / max(1e-6, total_rollout_sec),
        "total_rollout_sec": total_rollout_sec,
        "total_update_sec": total_update_sec,
        "rollout_fraction": total_rollout_sec / max(1e-6, total_wall_time),
        "update_fraction": total_update_sec / max(1e-6, total_wall_time),
        "mean_ipc_mask_ms": np.mean([r["ipc_mask_ms"] for r in rollout_records]),
        "mean_forward_ms": np.mean([r["forward_ms"] for r in rollout_records]),
        "mean_rebuild_ms": np.mean([r["rebuild_ms"] for r in rollout_records]),
        "mean_env_step_ms": np.mean([r["env_step_ms"] for r in rollout_records]),
        "peak_cuda_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0**2)) if device.type == "cuda" else 0.0,
        "cpu_percent_samples": [float(x) for x in cpu_percentages if x > 0],
    }
    return result, cuda_summary


def main():
    parser = argparse.ArgumentParser(description="APAL 性能瓶颈诊断工具")
    parser.add_argument("--data", type=str, default="data/283.csv")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--worker-threads", type=str, default="auto")
    parser.add_argument("--ipc-fusion", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--profile-cuda", action="store_true")
    parser.add_argument("--tag", type=str, default="run")
    args = parser.parse_args()

    threads_val = args.worker_threads
    if threads_val.isdigit():
        threads_val = int(threads_val)

    res, cuda_summary = run_benchmark(
        data_path=args.data,
        num_envs=args.num_envs,
        worker_threads=threads_val,
        ipc_fusion=args.ipc_fusion,
        max_steps=args.max_steps,
        episodes=args.episodes,
        profile_cuda=args.profile_cuda,
        tag=args.tag,
    )

    print("\n" + "=" * 80)
    print(f"DIAGNOSTIC REPORT: {res['tag']}")
    print("=" * 80)
    print(f"Dataset:            {res['data']}")
    print(f"Num Envs:           {res['num_envs']} (worker_threads={res['worker_threads']}, actual={res['resolved_worker_threads']})")
    print(f"IPC Fusion:         {res['ipc_fusion']}")
    print(f"Total Steps:        {res['total_steps']}")
    print(f"Total Wall Time:    {res['total_wall_time']:.2f} s")
    print(f"Overall Throughput: {res['overall_sps']:.1f} Steps/s")
    print(f"Rollout Time:       {res['total_rollout_sec']:.2f} s ({res['rollout_fraction']*100:.1f}%)")
    print(f"PPO Update Time:    {res['total_update_sec']:.2f} s ({res['update_fraction']*100:.1f}%)")
    print("-" * 80)
    print("Rollout Step Breakdown (avg ms per step):")
    print(f"  IPC + Mask:       {res['mean_ipc_mask_ms']:.2f} ms")
    print(f"  Env Step:         {res['mean_env_step_ms']:.2f} ms")
    print(f"  Graph Rebuild:    {res['mean_rebuild_ms']:.2f} ms")
    print(f"  GPU Forward:      {res['mean_forward_ms']:.2f} ms")
    print(f"Peak VRAM:          {res['peak_cuda_mb']:.1f} MB")
    if cuda_summary:
        print("-" * 80)
        print("CUDA Profiler Top Kernels:")
        print(cuda_summary)
    print("=" * 80)

    # 保存结果 JSON
    out_file = PROJECT_ROOT / f"benchmark_{args.tag}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
