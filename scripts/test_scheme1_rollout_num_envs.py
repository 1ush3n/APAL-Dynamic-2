# -*- coding: utf-8 -*-
"""
方案 1 针对性测试：Rollout 阶段增加并行环境数 num_envs (4 -> 6 -> 8)
评测内容：
1. Rollout SPS (Steps Per Second)
2. GPU 前向推断耗时 (Forward ms) 与 CPU 环境步进耗时 (Env ms)
3. GPU 真实平均利用率与峰值利用率 (%)
4. 峰值显存 (VRAM MB) 与主机物理内存 (Host RAM GB)
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


class GpuMonitor:
    def __init__(self, interval: float = 0.10):
        self.interval = interval
        self._stop = threading.Event()
        self.utils = []
        self.vrams = []
        self.rams = []

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
            try:
                ram = psutil.virtual_memory()
                self.rams.append((ram.total - ram.available) / (1024.0**3))
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.utils.clear()
        self.vrams.clear()
        self.rams.clear()
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
            "peak_host_ram_gb": max(self.rams) if self.rams else 0.0,
        }


def run_single_num_envs(n_envs: int, max_steps: int = 48, repeats: int = 2) -> dict:
    from configs import configs
    from models.hb_gat_pn import HBGATPN
    from ppo_agent import PPOAgent
    from runtime.seed import set_seed
    from training.rollout_service import APALRolloutService
    from utils.vector_env import EnvCreator, VectorEnv

    data_path = PROJECT_ROOT / "data" / "283.csv"
    overrides = {
        "data_file_path": str(data_path),
        "train_data_path_or_dir": str(data_path),
        "num_envs": n_envs,
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
        num_envs=n_envs,
        start_method=start_method,
        worker_threads=1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HBGATPN(configs).to(device)
    agent = PPOAgent(
        model=model,
        lr=float(configs.lr),
        gamma=float(configs.gamma),
        k_epochs=1,
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

    monitor = GpuMonitor(interval=0.10)
    monitor.start()

    sps_list, fwd_list, env_list = [], [], []
    try:
        for rep in range(repeats):
            _, metrics = service._collect_episode(rep + 1)
            sps_list.append(metrics.steps_per_second)
            fwd_list.append(metrics.forward_ms)
            env_list.append(metrics.environment_step_ms)
    finally:
        stats = monitor.stop()
        service.close()

    return {
        "num_envs": n_envs,
        "mean_sps": sum(sps_list) / len(sps_list),
        "fwd_ms": sum(fwd_list) / len(fwd_list),
        "env_ms": sum(env_list) / len(env_list),
        "avg_gpu_util": stats["avg_gpu_util"],
        "peak_gpu_util": stats["peak_gpu_util"],
        "peak_vram_mb": stats["peak_vram_mb"],
        "peak_host_ram_gb": stats["peak_host_ram_gb"],
    }


def main():
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        # 子进程单组测试模式
        n = int(sys.argv[1])
        res = run_single_num_envs(n)
        print("RESULT_JSON:" + json.dumps(res, ensure_ascii=False), flush=True)
        return

    # 主控流程：分别在独立子进程中启动 4, 6, 8 环境，彻底杜绝多组间内存堆积
    env_list = [4, 6, 8]
    all_results = []

    print("\n" + "="*85)
    print("【方案 1 测试】：Rollout 阶段增加并行环境数 num_envs (4 -> 6 -> 8)")
    print("="*85, flush=True)

    for n in env_list:
        vmem = psutil.virtual_memory()
        free_gb = vmem.available / (1024**3)
        print(f"\n>>> 正在启动子进程测试 num_envs={n} (当前可用物理内存: {free_gb:.2f} GB) ...", flush=True)

        cmd = [sys.executable, str(Path(__file__).resolve()), str(n)]
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        res_data = None
        for line in iter(proc.stdout.readline, ""):
            line_str = line.strip()
            if line_str.startswith("RESULT_JSON:"):
                res_data = json.loads(line_str.removeprefix("RESULT_JSON:"))
            elif line_str:
                print("  " + line_str, flush=True)

        proc.stdout.close()
        proc.wait()

        if res_data is not None:
            all_results.append(res_data)
            print(f"num_envs={n:2d} 完成: Mean SPS={res_data['mean_sps']:.2f} | "
                  f"Fwd耗时={res_data['fwd_ms']:.1f}ms | "
                  f"平均GPU利用率={res_data['avg_gpu_util']:.1f}% | "
                  f"峰值GPU利用率={res_data['peak_gpu_util']:.1f}% | "
                  f"峰值显存={res_data['peak_vram_mb']:.0f}MB | "
                  f"峰值内存={res_data['peak_host_ram_gb']:.1f}GB", flush=True)

        time.sleep(2.0)

    # 输出汇总表格
    print("\n" + "="*90)
    print("【方案 1 测试结果汇总】：Rollout 阶段增加并行环境数 num_envs (4 -> 6 -> 8)")
    print("="*90)
    print(f"{'num_envs':<10} | {'Mean SPS':<10} | {'Fwd (ms)':<10} | {'Env (ms)':<10} | {'平均 GPU %':<11} | {'峰值 GPU %':<11} | {'峰值显存 (MB)':<12} | {'峰值内存 (GB)':<12}")
    print("-" * 95)
    for r in all_results:
        print(f"{r['num_envs']:<10d} | {r['mean_sps']:<10.2f} | {r['fwd_ms']:<10.1f} | {r['env_ms']:<10.1f} | {r['avg_gpu_util']:<10.1f}% | {r['peak_gpu_util']:<10.1f}% | {r['peak_vram_mb']:<12.0f} | {r['peak_host_ram_gb']:<12.1f}")

    if len(all_results) >= 2:
        speedup_6 = all_results[1]["mean_sps"] / all_results[0]["mean_sps"]
        print(f"\n>>> num_envs 4 -> 6 SPS 提升: {speedup_6:.2f}x ({((speedup_6 - 1.0)*100):+.1f}%)")
    if len(all_results) >= 3:
        speedup_8 = all_results[2]["mean_sps"] / all_results[0]["mean_sps"]
        print(f">>> num_envs 4 -> 8 SPS 提升: {speedup_8:.2f}x ({((speedup_8 - 1.0)*100):+.1f}%)")

    out_file = PROJECT_ROOT / "logs" / "scheme1_num_envs_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至 {out_file}")


if __name__ == "__main__":
    main()
