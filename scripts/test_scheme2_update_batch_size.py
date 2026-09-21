# -*- coding: utf-8 -*-
"""
方案 2 针对性测试：增大 PPO Update 的 mini_batch_size (32 -> 64 -> 128)
评测内容：
1. PPO 更新耗时 (s)
2. PPO 更新吞吐 (transitions / s)
3. GPU 真实利用率 (平均 % 与 峰值 %)
4. 峰值显存 (VRAM MB)
5. 梯度更新收敛与数值稳定性
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

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


class GpuMonitor:
    def __init__(self, interval: float = 0.08):
        self.interval = interval
        self._stop = threading.Event()
        self.utils = []
        self.vrams = []

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                    text=True,
                    timeout=0.5,
                ).strip()
                parts = [p.strip() for p in out.split(",")]
                if len(parts) >= 2:
                    self.utils.append(float(parts[0]))
                    self.vrams.append(float(parts[1]))
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.utils.clear()
        self.vrams.clear()
        self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self) -> dict:
        self._stop.set()
        self._t.join(timeout=1.0)
        return {
            "avg_gpu_util": sum(self.utils) / max(1, len(self.utils)),
            "peak_gpu_util": max(self.utils) if self.utils else 0.0,
            "peak_vram_mb": max(self.vrams) if self.vrams else 0.0,
            "samples": len(self.utils),
        }


def main():
    from configs import configs
    from models.hb_gat_pn import HBGATPN
    from ppo_agent import PPOAgent
    from training.rollout_service import APALRolloutService
    from utils.vector_env import EnvCreator, VectorEnv

    data_path = PROJECT_ROOT / "data" / "283.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")

    # 1. 采样代表性轨迹 (4 环境, 64 步, ~256 transitions)
    print("\n[Step 1] 正在采样基准代表性轨迹 (256 步 transitions) ...", flush=True)
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
    service.close()
    del service
    del vec_env
    gc.collect()
    torch.cuda.empty_cache()

    # 合并经验池
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
    print(f"经验池包含 {total_transitions} 个动作步。所有多进程子进程已彻底销毁，内存已完全回收。\n")

    # 创建用于单进程 state 重建的轻量环境对象
    rebuild_env = EnvCreator(str(data_path), seed_offset=42)(0)

    # 2. 对比 batch_size: 32 vs 64 vs 128
    batch_sizes = [32, 64, 128]
    results = []

    for bs in batch_sizes:
        print(f"{'='*30} 测试 batch_size = {bs} {'='*30}", flush=True)
        agent.batch_size = bs
        configs.batch_size = bs

        # 制作独立深拷贝内存防止 update 就地修改影响后续对比
        import copy
        mem_copy = copy.deepcopy(merged_memory)

        # 显存与状态重置
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        monitor = GpuMonitor(interval=0.06)
        monitor.start()

        t0 = time.perf_counter()
        metrics = agent.update(mem_copy, rebuild_env, current_ep=1)
        torch.cuda.synchronize(device)
        update_duration = time.perf_counter() - t0

        stats = monitor.stop()
        total_eval_steps = total_transitions * 4  # k_epochs=4
        throughput = total_eval_steps / max(update_duration, 1e-6)

        rec = {
            "batch_size": bs,
            "duration_sec": update_duration,
            "throughput_tps": throughput,
            "avg_gpu_util": stats["avg_gpu_util"],
            "peak_gpu_util": stats["peak_gpu_util"],
            "peak_vram_mb": stats["peak_vram_mb"],
            "loss": float(metrics.get("Loss/Total", 0.0)),
        }
        results.append(rec)
        print(f"batch_size={bs:3d} 结果: 耗时={update_duration:.2f}s | "
              f"吞吐={throughput:.1f} transitions/s | "
              f"平均GPU利用率={stats['avg_gpu_util']:.1f}% | "
              f"峰值GPU利用率={stats['peak_gpu_util']:.1f}% | "
              f"峰值显存={stats['peak_vram_mb']:.0f} MB", flush=True)

        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1.0)

    # 3. 输出汇总表格
    print("\n" + "="*80)
    print("【方案 2 测试结果汇总】：PPO Update 增大 mini_batch_size (32 -> 64 -> 128)")
    print("="*80)
    print(f"{'Batch Size':<12} | {'耗时 (秒)':<10} | {'更新吞吐 (t/s)':<16} | {'平均 GPU 利用率':<15} | {'峰值 GPU 利用率':<15} | {'峰值显存 (MB)':<12}")
    print("-" * 85)
    for r in results:
        print(f"{r['batch_size']:<12d} | {r['duration_sec']:<10.2f} | {r['throughput_tps']:<16.1f} | {r['avg_gpu_util']:<13.1f}% | {r['peak_gpu_util']:<13.1f}% | {r['peak_vram_mb']:<12.0f}")

    if len(results) >= 2:
        speedup_64 = results[1]["throughput_tps"] / results[0]["throughput_tps"]
        print(f"\n>>> batch_size 32 -> 64 吞吐提升: {speedup_64:.2f}x ({((speedup_64 - 1.0)*100):+.1f}%)")
    if len(results) >= 3:
        speedup_128 = results[2]["throughput_tps"] / results[0]["throughput_tps"]
        print(f">>> batch_size 32 -> 128 吞吐提升: {speedup_128:.2f}x ({((speedup_128 - 1.0)*100):+.1f}%)")

    out_file = PROJECT_ROOT / "logs" / "scheme2_batch_size_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至 {out_file}")


if __name__ == "__main__":
    main()
