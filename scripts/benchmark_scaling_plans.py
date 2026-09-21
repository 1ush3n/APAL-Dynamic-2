# -*- coding: utf-8 -*-
"""
方案 1 与 方案 2 针对性性能与 GPU 利用率测试基准脚本：
方案 1：增加并行环境数 num_envs (4 -> 6 -> 8)
方案 2：增大 PPO Update 的 mini_batch_size (32 -> 64 -> 128)

配备实时 GPU 利用率 (nvidia-smi) 与 主机物理内存 (Host RAM) 采样监视器，
全过程杜绝 OOM，准确统计平均 GPU 利用率、峰值显存与吞吐变化。
"""

from __future__ import annotations

import gc
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import Config, configs
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.seed import set_seed
from training.rollout_service import APALRolloutService
from utils.vector_env import EnvCreator, VectorEnv


class SystemResourceMonitor:
    """高频采样 GPU 利用率、VRAM 与 主机物理内存。"""

    def __init__(self, interval_sec: float = 0.15):
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread = None
        self.gpu_utils: list[float] = []
        self.vram_used_mb: list[float] = []
        self.host_ram_used_gb: list[float] = []

    def _sample_loop(self):
        while not self._stop_event.is_set():
            try:
                # 采样 GPU
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                    text=True,
                    timeout=1.0,
                ).strip()
                parts = [p.strip() for p in out.split(",")]
                if len(parts) >= 2:
                    self.gpu_utils.append(float(parts[0]))
                    self.vram_used_mb.append(float(parts[1]))
            except Exception:
                pass

            try:
                # 采样 Host RAM
                ram = psutil.virtual_memory()
                self.host_ram_used_gb.append((ram.total - ram.available) / (1024.0**3))
            except Exception:
                pass

            time.sleep(self.interval_sec)

    def start(self):
        self.gpu_utils.clear()
        self.vram_used_mb.clear()
        self.host_ram_used_gb.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        avg_gpu = sum(self.gpu_utils) / max(1, len(self.gpu_utils))
        peak_gpu = max(self.gpu_utils) if self.gpu_utils else 0.0
        avg_vram = sum(self.vram_used_mb) / max(1, len(self.vram_used_mb))
        peak_vram = max(self.vram_used_mb) if self.vram_used_mb else 0.0
        peak_host_ram = max(self.host_ram_used_gb) if self.host_ram_used_gb else 0.0

        return {
            "avg_gpu_util_pct": avg_gpu,
            "peak_gpu_util_pct": peak_gpu,
            "avg_vram_mb": avg_vram,
            "peak_vram_mb": peak_vram,
            "peak_host_ram_gb": peak_host_ram,
            "samples_count": len(self.gpu_utils),
        }


def test_scheme_1_rollout_scaling() -> list[dict]:
    """方案 1 测试：增加并行环境数 num_envs (4 -> 6 -> 8)。"""
    print("\n" + "="*80)
    print("【方案 1 测试】：Rollout 阶段增加并行环境数 num_envs (4 -> 6 -> 8)")
    print("="*80, flush=True)

    data_path = PROJECT_ROOT / "data" / "283.csv"
    env_counts = [4, 6, 8]
    max_steps = 48
    repeats = 2
    results = []

    for n_env in env_counts:
        # 预先检查主机内存
        vmem = psutil.virtual_memory()
        free_gb = vmem.available / (1024**3)
        print(f"\n>>> 准备测试 num_envs={n_env} | 当前可用物理内存: {free_gb:.2f} GB ...", flush=True)
        if free_gb < 2.0:
            print(f"WARNING: 可用物理内存过低 ({free_gb:.2f} GB)，跳过更大环境以防 OOM")
            break

        overrides = {
            "data_file_path": str(data_path),
            "train_data_path_or_dir": str(data_path),
            "num_envs": n_env,
            "rollout_max_steps": max_steps,
            "seed": 42,
            "randomize_durations": False,
            "enable_dynamic_events": False,
            "enable_station_breakdown": False,
            "enable_material_delay": False,
            "rollout_heartbeat_interval_sec": 0.0,
            "enable_rollout_ipc_fusion": True,
            "rollout_double_buffer": False,
        }
        configs.update_from_dict(overrides)
        set_seed(42)

        start_method = "forkserver" if platform.system() == "Linux" else "spawn"
        vec_env = VectorEnv(
            EnvCreator(str(data_path), seed_offset=42),
            num_envs=n_env,
            start_method=start_method,
            worker_threads=1,
            init_timeout_sec=float(configs.vector_env_init_timeout_sec),
            command_timeout_sec=float(configs.vector_env_command_timeout_sec),
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
        service = APALRolloutService(
            agent=agent,
            vector_env=vec_env,
            eval_env=vec_env.envs[0],
            config=configs,
            device=device,
        )

        monitor = SystemResourceMonitor(interval_sec=0.15)
        monitor.start()

        sps_list = []
        fwd_list = []
        env_list = []

        try:
            for rep in range(repeats):
                _, metrics = service._collect_episode(rep + 1)
                sps_list.append(metrics.steps_per_second)
                fwd_list.append(metrics.forward_ms)
                env_list.append(metrics.environment_step_ms)
        finally:
            res_stats = monitor.stop()
            service.close()
            del service
            del vec_env
            del agent
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            time.sleep(2.0)

        record = {
            "num_envs": n_env,
            "mean_sps": sum(sps_list) / len(sps_list),
            "fwd_ms": sum(fwd_list) / len(fwd_list),
            "env_ms": sum(env_list) / len(env_list),
            "avg_gpu_util_pct": res_stats["avg_gpu_util_pct"],
            "peak_gpu_util_pct": res_stats["peak_gpu_util_pct"],
            "peak_vram_mb": res_stats["peak_vram_mb"],
            "peak_host_ram_gb": res_stats["peak_host_ram_gb"],
        }
        results.append(record)
        print(f"num_envs={n_env:2d} 完成: Mean SPS={record['mean_sps']:.2f} | "
              f"GPU Util={record['avg_gpu_util_pct']:.1f}% (Peak {record['peak_gpu_util_pct']:.1f}%) | "
              f"Peak VRAM={record['peak_vram_mb']:.0f}MB | Peak Host RAM={record['peak_host_ram_gb']:.1f}GB", flush=True)

    return results


def test_scheme_2_ppo_update_batch_size() -> list[dict]:
    """方案 2 测试：增大 PPO Update mini_batch_size (32 -> 64 -> 128)。"""
    print("\n" + "="*80)
    print("【方案 2 测试】：PPO Update 阶段增大 batch_size (32 -> 64 -> 128)")
    print("="*80, flush=True)

    data_path = PROJECT_ROOT / "data" / "283.csv"
    batch_sizes = [32, 64, 128]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 首先以 4 环境采样一批固定的代表性经验（256 个 transitions）用于公平对齐
    print(">>> 正在预热并采样基准经验池 (256 步 transition) ...", flush=True)
    overrides = {
        "data_file_path": str(data_path),
        "train_data_path_or_dir": str(data_path),
        "num_envs": 4,
        "rollout_max_steps": 64,
        "seed": 42,
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
        "enable_rollout_ipc_fusion": True,
        "rollout_double_buffer": False,
    }
    configs.update_from_dict(overrides)
    set_seed(42)

    vec_env = VectorEnv(
        EnvCreator(str(data_path), seed_offset=42),
        num_envs=4,
        start_method="spawn",
        worker_threads=1,
    )
    model = HBGATPN(configs).to(device)
    agent = PPOAgent(
        model=model,
        lr=float(configs.lr),
        gamma=float(configs.gamma),
        k_epochs=4,
        eps_clip=float(configs.eps_clip),
        device=device,
        batch_size=32,
        total_timesteps=1,
        config=configs,
    )
    service = APALRolloutService(
        agent=agent,
        vector_env=vec_env,
        eval_env=vec_env.envs[0],
        config=configs,
        device=device,
    )

    memories, _ = service._collect_episode(1)
    # 合并为一个统一经验池
    from training.memory import Memory
    merged_memory = Memory()
    for m in memories:
        merged_memory.states.extend(m.states)
        merged_memory.actions.extend(m.actions)
        merged_memory.rewards.extend(m.rewards)
        merged_memory.is_terminals.extend(m.is_terminals)
        merged_memory.is_truncated.extend(m.is_truncated)
        merged_memory.logprobs.extend(m.logprobs)
        merged_memory.values.extend(m.values)
        merged_memory.masks.extend(m.masks)
        merged_memory.old_task_logprob.extend(m.old_task_logprob)
        merged_memory.old_station_logprob.extend(m.old_station_logprob)
        merged_memory.old_team_logprob.extend(m.old_team_logprob)
        merged_memory.old_V_task.extend(m.old_V_task)
        merged_memory.old_V_station.extend(m.old_V_station)
        merged_memory.old_V_worker.extend(m.old_V_worker)
        merged_memory.gated_team_traces.extend(m.gated_team_traces)
        merged_memory.anchor_proposal_traces.extend(m.anchor_proposal_traces)
        merged_memory.worker_pointer_v2_behavior_traces.extend(m.worker_pointer_v2_behavior_traces)

    total_transitions = len(merged_memory.actions)
    print(f">>> 经验池准备就绪：包含 {total_transitions} 个动作决策步。关闭环境释放内存 ...", flush=True)
    service.close()
    del service
    del vec_env
    gc.collect()
    time.sleep(1.0)

    results = []

    for bs in batch_sizes:
        print(f"\n>>> 正在测试 PPO Update batch_size={bs} (k_epochs=4, 共 {total_transitions} 步) ...", flush=True)
        agent.batch_size = bs
        configs.batch_size = bs

        # 预热并同步 CUDA
        torch.cuda.synchronize(device)
        monitor = SystemResourceMonitor(interval_sec=0.10)
        monitor.start()

        t0 = time.perf_counter()
        metrics = agent.update(merged_memory, None, current_ep=1)
        torch.cuda.synchronize(device)
        update_time = time.perf_counter() - t0

        res_stats = monitor.stop()
        gc.collect()
        torch.cuda.empty_cache()

        total_passes = total_transitions * 4 # 4 epochs
        update_sps = total_passes / max(update_time, 1e-6)

        record = {
            "batch_size": bs,
            "update_time_sec": update_time,
            "update_sps": update_sps,
            "avg_gpu_util_pct": res_stats["avg_gpu_util_pct"],
            "peak_gpu_util_pct": res_stats["peak_gpu_util_pct"],
            "peak_vram_mb": res_stats["peak_vram_mb"],
            "loss_total": float(metrics.get("Loss/Total", 0.0)),
        }
        results.append(record)
        print(f"batch_size={bs:3d} 完成: Update耗时={update_time:.2f}s | "
              f"Update吞吐={update_sps:.1f} transitions/s | "
              f"GPU Util={record['avg_gpu_util_pct']:.1f}% (Peak {record['peak_gpu_util_pct']:.1f}%) | "
              f"Peak VRAM={record['peak_vram_mb']:.0f}MB", flush=True)

    return results


def main():
    res1 = test_scheme_1_rollout_scaling()
    res2 = test_scheme_2_ppo_update_batch_size()

    print("\n" + "="*90)
    print("【综合测试报告汇总】")
    print("="*90)

    print("\n[方案 1：Rollout 阶段增加并行环境数 num_envs]")
    print(f"{'num_envs':<10} | {'Mean SPS':<10} | {'Fwd (ms)':<10} | {'Avg GPU %':<11} | {'Peak GPU %':<11} | {'Peak VRAM':<10} | {'Host RAM':<10}")
    print("-" * 90)
    for r in res1:
        print(f"{r['num_envs']:<10d} | {r['mean_sps']:<10.2f} | {r['fwd_ms']:<10.2f} | {r['avg_gpu_util_pct']:<10.1f}% | {r['peak_gpu_util_pct']:<10.1f}% | {r['peak_vram_mb']:<8.0f}MB | {r['peak_host_ram_gb']:<8.1f}GB")

    print("\n[方案 2：PPO Update 阶段增大 batch_size]")
    print(f"{'batch_size':<10} | {'Update耗时(s)':<13} | {'Update吞吐(t/s)':<15} | {'Avg GPU %':<11} | {'Peak GPU %':<11} | {'Peak VRAM':<10}")
    print("-" * 90)
    for r in res2:
        print(f"{r['batch_size']:<10d} | {r['update_time_sec']:<13.2f} | {r['update_sps']:<15.1f} | {r['avg_gpu_util_pct']:<10.1f}% | {r['peak_gpu_util_pct']:<10.1f}% | {r['peak_vram_mb']:<8.0f}MB")

    # 保存 JSON
    out_file = PROJECT_ROOT / "logs" / "scaling_plans_benchmark_results.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"scheme_1_rollout": res1, "scheme_2_update": res2}, f, indent=2, ensure_ascii=False)
    print(f"\n完整评测结果已写入：{out_file}")


if __name__ == "__main__":
    main()
