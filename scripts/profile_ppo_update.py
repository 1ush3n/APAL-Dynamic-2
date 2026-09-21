# -*- coding: utf-8 -*-
"""PPO update 内部各阶段耗时微基准剖析。"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

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
from utils.vector_env import EnvCreator, VectorEnv

def main():
    data_path = str((PROJECT_ROOT / "data/283.csv").resolve())
    set_seed(42)
    configs.update_from_dict({
        "data_file_path": data_path,
        "train_data_path_or_dir": data_path,
        "num_envs": 4,
        "rollout_max_steps": 20,
        "seed": 42,
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
        "enable_shadow_mask_verification": False,
        "rollout_heartbeat_interval_sec": 0.0,
        "enable_rollout_ipc_fusion": True,
        "vector_env_worker_threads": 1,
        "k_epochs": 2,
        "batch_size": 32,
    })

    from training.rollout_service import APALRolloutService
    start_method = "spawn"
    vector_env = VectorEnv(
        EnvCreator(data_path, seed_offset=42),
        num_envs=4,
        start_method=start_method,
        worker_threads=1,
    )
    eval_env = AirLineEnv_Graph(data_path, seed=42)
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
        total_timesteps=1,
        config=configs,
    )
    service = APALRolloutService(
        agent=agent,
        vector_env=vector_env,
        eval_env=eval_env,
        config=configs,
        device=device,
    )

    try:
        # 收集 20 步的轨迹数据
        memories, metrics = service._collect_episode(1)
        merged = memories[0]
        for m in memories[1:]:
            service._merge_memories(merged, [m])

        print(f"Collected memory with {len(merged.states)} samples.")

        # 详细分析 update 内部耗时
        t0 = time.perf_counter()
        # 1. GAE
        advantages, rewards = agent.compute_gae_returns(
            rewards=merged.rewards,
            terminals=merged.is_terminals,
            values=merged.values,
            gamma=agent.gamma,
            gae_lambda=agent.gae_lambda,
            truncated=merged.is_truncated,
        )
        if advantages.std() > 1e-7:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        else:
            advantages = advantages - advantages.mean()
        t_gae = (time.perf_counter() - t0) * 1000

        # 2. Update once
        t0 = time.perf_counter()
        update_metrics = agent.update(merged, eval_env, current_ep=1)
        torch.cuda.synchronize(device)
        t_update_total = (time.perf_counter() - t0) * 1000

        print(f"PPO Update Breakdown:")
        print(f"  GAE computation:     {t_gae:6.2f} ms")
        print(f"  Total Update (2 ep): {t_update_total:6.2f} ms ({t_update_total/1000:.2f} s)")
        print(f"  Update metrics: {list(update_metrics.keys())[:8]}")

    finally:
        vector_env.close()

if __name__ == "__main__":
    main()
