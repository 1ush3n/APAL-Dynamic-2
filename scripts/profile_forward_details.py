# -*- coding: utf-8 -*-
"""深度剖析 select_actions_batch 内部各项耗时。"""

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
        "enable_rollout_ipc_fusion": False,
        "vector_env_worker_threads": 1,
    })

    vector_env = VectorEnv(
        EnvCreator(data_path, seed_offset=42),
        num_envs=4,
        start_method="spawn",
        worker_threads=1,
    )
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

    try:
        vector_env.reset_all()
        masks_list, snapshots = vector_env.get_masks_and_snapshots_all()
        states = [
            vector_env.envs[i].rebuild_state_from_snapshot(snapshots[i])
            for i in range(4)
        ]

        # 预热 GPU
        with torch.inference_mode():
            _ = agent.select_actions_batch(
                obs_list=states,
                mask_task_list=[m[0] for m in masks_list],
                mask_station_matrix_list=[m[1] for m in masks_list],
                mask_worker_list=[m[2] for m in masks_list],
                deterministic=False,
                baseline_snapshots=snapshots,
            )
        torch.cuda.synchronize(device)

        # 详细微基准测试：对 select_actions_batch 的每个阶段分别计时
        from torch_geometric.data import Batch

        timings = {
            "pyg_batch_from_data_list": [],
            "h2d_memcpy": [],
            "gnn_encoder": [],
            "critic_encoder": [],
            "slice_features": [],
            "task_head_fwd": [],
            "task_sync_item": [],
            "station_head_fwd": [],
            "station_sync_item": [],
            "worker_head_fwd_and_loop": [],
        }

        with torch.inference_mode():
            for _ in range(10):
                # 1. PyG Batch
                t0 = time.perf_counter()
                batch_obs = Batch.from_data_list(states)
                task_ptr = batch_obs['task'].ptr.tolist()
                station_ptr = batch_obs['station'].ptr.tolist()
                worker_ptr = batch_obs['worker'].ptr.tolist()
                timings["pyg_batch_from_data_list"].append((time.perf_counter() - t0) * 1000)

                # 2. H2D
                t0 = time.perf_counter()
                batch_obs = batch_obs.to(device)
                torch.cuda.synchronize(device)
                timings["h2d_memcpy"].append((time.perf_counter() - t0) * 1000)

                # 3. GNN encoder
                t0 = time.perf_counter()
                x_dict_batch, global_context_batch = agent.policy(batch_obs)
                torch.cuda.synchronize(device)
                timings["gnn_encoder"].append((time.perf_counter() - t0) * 1000)

                # 4. Critic
                t0 = time.perf_counter()
                state_values_batch = agent.policy.get_value(batch_obs, actor_x_dict_encoded=x_dict_batch)
                torch.cuda.synchronize(device)
                timings["critic_encoder"].append((time.perf_counter() - t0) * 1000)

                # 5. Slicing and heads for 4 envs
                t_slice = 0.0
                t_task_fwd = 0.0
                t_task_sync = 0.0
                t_station_fwd = 0.0
                t_station_sync = 0.0
                t_worker = 0.0

                for i in range(4):
                    t0 = time.perf_counter()
                    t_start, t_end = task_ptr[i], task_ptr[i + 1]
                    s_start, s_end = station_ptr[i], station_ptr[i + 1]
                    w_start, w_end = worker_ptr[i], worker_ptr[i + 1]
                    task_embs = x_dict_batch['task'][t_start:t_end]
                    station_embs = x_dict_batch['station'][s_start:s_end]
                    worker_embs = x_dict_batch['worker'][w_start:w_end]
                    global_context_i = global_context_batch[i].unsqueeze(0)
                    m_task = masks_list[i][0].to(device)
                    t_slice += (time.perf_counter() - t0) * 1000

                    # Task head
                    t0 = time.perf_counter()
                    task_logits = agent.policy.task_head(task_embs, global_context_i, mask=None)
                    torch.cuda.synchronize(device)
                    t_task_fwd += (time.perf_counter() - t0) * 1000

                    # Task sync & sampling
                    t0 = time.perf_counter()
                    from torch.distributions import Categorical
                    from ppo_agent import _finalize_action_logits
                    task_logits, _ = _finalize_action_logits(task_logits, m_task.reshape(task_logits.shape), decision="task")
                    task_dist = Categorical(logits=task_logits)
                    task_act = task_dist.sample()
                    t_idx = task_act.item()
                    t_task_sync += (time.perf_counter() - t0) * 1000

                    # Station head
                    t0 = time.perf_counter()
                    selected_task_emb = task_embs[t_idx].unsqueeze(0)
                    station_embs_i = station_embs.unsqueeze(0)
                    station_logits = agent.policy.station_head(selected_task_emb, station_embs_i, mask=None)
                    torch.cuda.synchronize(device)
                    t_station_fwd += (time.perf_counter() - t0) * 1000

                    # Station sync & sampling
                    t0 = time.perf_counter()
                    m_station = masks_list[i][1][t_idx].unsqueeze(0).to(device)
                    station_logits, _ = _finalize_action_logits(station_logits, m_station.reshape(station_logits.shape), decision="station")
                    station_dist = Categorical(logits=station_logits)
                    station_act = station_dist.sample()
                    s_idx = station_act.item()
                    t_station_sync += (time.perf_counter() - t0) * 1000

                    # Worker selection
                    t0 = time.perf_counter()
                    demand = agent.get_task_demand(states[i]['task'].x, t_idx)
                    worker_embs_i = worker_embs.unsqueeze(0)
                    for _ in range(demand):
                        w_logits = agent.policy.worker_head.forward_choice(selected_task_emb, worker_embs_i, mask=None)
                        torch.cuda.synchronize(device)
                        w_dist = Categorical(logits=w_logits)
                        w_act = w_dist.sample()
                        _ = w_act.item()
                    t_worker += (time.perf_counter() - t0) * 1000

                timings["slice_features"].append(t_slice)
                timings["task_head_fwd"].append(t_task_fwd)
                timings["task_sync_item"].append(t_task_sync)
                timings["station_head_fwd"].append(t_station_fwd)
                timings["station_sync_item"].append(t_station_sync)
                timings["worker_head_fwd_and_loop"].append(t_worker)

        print("\n--- select_actions_batch Micro-benchmark (mean over 10 runs, ms) ---")
        total_ms = 0.0
        for k, v in timings.items():
            mean_v = sum(v) / len(v)
            total_ms += mean_v
            print(f"  {k:30s}: {mean_v:6.2f} ms")
        print(f"  {'TOTAL':30s}: {total_ms:6.2f} ms")

    finally:
        vector_env.close()

if __name__ == "__main__":
    main()
