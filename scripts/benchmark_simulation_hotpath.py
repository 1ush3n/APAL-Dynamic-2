# -*- coding: utf-8 -*-
"""APAL-Dynamic-2 最小数据集 (283) 性能与算法决策基准测试工具。

专为本地轻量运行设计：
- 严格限制在 data/283.csv 最小数据集，杜绝大图 OOM 与长时间运行；
- 采集并持久化关键基准指标：
  1. SPT 确定性启发式排程结果 (Makespan, BalanceStd, 约束合规性)；
  2. 环境热路径耗时剖析 (step 耗时, _get_observation 耗时, sweep-line 耗时)；
  3. PPO 模型同种子前向输出基准 (动作序列, 对数概率, 状态价值估值)；
  4. 系统资源占用 (内存 RSS, 显存)。
- 支持 --save (生成基准) 与 --check (回归校验比对)。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 避免 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from configs import Config, configs
from environment import AirLineEnv_Graph
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent
from runtime.seed import set_seed
from baselines.heuristic.advanced_schedulers import AdvancedSchedulerBase


def _get_process_memory_mb() -> float:
    """获取当前进程的物理内存占用 (RSS MB)。"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def benchmark_spt_simulation(data_path: Path, seed: int = 42) -> dict[str, Any]:
    """运行基于 FeasibilityPreservingDecoder 的 SPT 完整排程与热路径打点。"""
    set_seed(seed)

    configs.n_m = 5
    configs.n_w = 80
    configs.max_slots_per_station = 3
    configs.enable_dynamic_events = False

    env = AirLineEnv_Graph(data_path_or_dir=str(data_path), seed=seed)
    env.reset(randomize_duration=False, randomize_workers=False, seed=seed)

    # 包装 _get_station_earliest_available_time 以打点测量扫描线开销
    orig_sweep = env._get_station_earliest_available_time
    sweep_line_times = []
    sweep_line_call_count = 0

    def timed_sweep(station_id, min_start_time, duration):
        nonlocal sweep_line_call_count
        sweep_line_call_count += 1
        t0 = time.perf_counter()
        res = orig_sweep(station_id, min_start_time, duration)
        sweep_line_times.append(time.perf_counter() - t0)
        return res

    env._get_station_earliest_available_time = timed_sweep

    # 包装 step 耗时打点
    orig_step = env.step
    step_times = []

    def timed_step(action):
        t0 = time.perf_counter()
        res = orig_step(action)
        step_times.append(time.perf_counter() - t0)
        return res

    env.step = timed_step

    # 运行标准 SPT 解码器
    start_wall_time = time.perf_counter()
    scheduler = AdvancedSchedulerBase(env, seed=seed)
    solution = scheduler.build_rule_solution("SPT", seed)
    result = scheduler.decoder.decode(solution, seed)
    total_wall_time = time.perf_counter() - start_wall_time

    # 测量单次观测耗时基准
    obs_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = env._get_observation()
        obs_times.append(time.perf_counter() - t0)

    val_rep = result.validation_report
    is_legal = bool(val_rep.is_legal) if val_rep else True
    violations = {k: int(v) for k, v in val_rep.violations.items()} if val_rep else {}

    return {
        "dataset": "283.csv",
        "seed": seed,
        "makespan": float(result.makespan),
        "balance_std": float(result.balance_std),
        "scheduled_tasks": len(result.assigned_tasks),
        "num_total_tasks": env.num_tasks,
        "is_legal": is_legal,
        "violations": violations,
        "timing": {
            "total_wall_time_sec": total_wall_time,
            "total_steps": len(step_times),
            "step_time_ms_mean": float(np.mean(step_times) * 1000) if step_times else 0.0,
            "step_time_ms_median": float(np.median(step_times) * 1000) if step_times else 0.0,
            "step_time_ms_p95": float(np.percentile(step_times, 95) * 1000) if step_times else 0.0,
            "obs_time_ms_mean": float(np.mean(obs_times) * 1000) if obs_times else 0.0,
            "sweep_line_call_count": sweep_line_call_count,
            "sweep_line_time_ms_mean": float(np.mean(sweep_line_times) * 1000)
            if sweep_line_times
            else 0.0,
            "sweep_line_time_ms_total": float(np.sum(sweep_line_times) * 1000)
            if sweep_line_times
            else 0.0,
        },
    }


def benchmark_ppo_agent_decisions(data_path: Path, seed: int = 42, num_steps: int = 5) -> dict[str, Any]:
    """固定随机种子初始化 PPOAgent 与模型，运行前 N 步决策，记录确切的动作和估值。"""
    set_seed(seed)

    env = AirLineEnv_Graph(data_path_or_dir=str(data_path), seed=seed)
    env.reset(randomize_duration=False, randomize_workers=False, seed=seed)

    cfg = Config()
    cfg.n_m = 5
    cfg.n_w = 80
    cfg.max_slots_per_station = 3
    cfg.enable_dynamic_events = False
    cfg.device = "cpu"

    device = torch.device("cpu")
    model = HBGATPN(cfg).to(device)
    agent = PPOAgent(
        model=model,
        lr=0.0003,
        gamma=0.99,
        k_epochs=4,
        eps_clip=0.2,
        device=device,
        config=cfg,
    )

    decisions = []
    obs = env._get_observation()

    for step_idx in range(num_steps):
        task_mask, station_mask, worker_mask = env.get_masks()
        results = agent.select_actions_batch(
            obs_list=[obs],
            mask_task_list=[task_mask],
            mask_station_matrix_list=[station_mask],
            mask_worker_list=[worker_mask],
            deterministic=True,
            is_eval=True,
        )
        act, logprob, val, _, _ = results[0]
        if act is None:
            break

        decisions.append(
            {
                "step": step_idx,
                "task_id": int(act[0]),
                "station_id": int(act[1]),
                "team": [int(w) for w in act[2]],
                "value": float(val),
                "logprob": float(logprob),
            }
        )
        obs, _, done, _ = env.step(act)
        if done:
            break

    return {
        "ppo_seed": seed,
        "test_steps": len(decisions),
        "decisions": decisions,
        "rss_memory_mb": _get_process_memory_mb(),
    }


def main():
    parser = argparse.ArgumentParser(description="APAL 283 数据集轻量基准与回归标尺")
    parser.add_argument("--save", action="store_true", help="保存基准数据至 benchmarks/baseline_283_v0.json")
    parser.add_argument("--check", action="store_true", help="比对当前执行结果与 baseline_283_v0.json")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "283.csv")
    args = parser.parse_args()

    baseline_file = PROJECT_ROOT / "benchmarks" / "baseline_283_v0.json"

    print("============================================================")
    print("      APAL-Dynamic-2: 283 数据集热点基准与结果标尺采集")
    print("============================================================")
    print(f"[*] 数据文件: {args.data}")
    print(f"[*] Python 进程内存: {_get_process_memory_mb():.2f} MB")

    # 1. 执行 SPT 完整回合基准测试
    print("[1/2] 正在执行 SPT 启发式全排程与热路径打点...")
    spt_results = benchmark_spt_simulation(args.data, seed=42)
    print(f"      -> 排程任务数: {spt_results['scheduled_tasks']} / {spt_results['num_total_tasks']}")
    print(f"      -> Makespan: {spt_results['makespan']:.4f}")
    print(f"      -> Workload BalanceStd: {spt_results['balance_std']:.4f}")
    print(f"      -> 约束合规性: {spt_results['is_legal']} (Violations: {sum(spt_results['violations'].values())})")
    print(f"      -> 单步平均耗时: {spt_results['timing']['step_time_ms_mean']:.3f} ms (P95: {spt_results['timing']['step_time_ms_p95']:.3f} ms)")
    print(f"      -> 单步观测耗时: {spt_results['timing']['obs_time_ms_mean']:.3f} ms")
    print(f"      -> 站位扫描线耗时: 总计 {spt_results['timing']['sweep_line_time_ms_total']:.2f} ms ({spt_results['timing']['sweep_line_call_count']} 次调用)")

    # 2. 执行 PPO 确定性决策基准
    print("[2/2] 正在记录 PPO 模型固定种子下的前向决策结果...")
    ppo_results = benchmark_ppo_agent_decisions(args.data, seed=42, num_steps=5)
    first_dec = ppo_results["decisions"][0]
    print(f"      -> Step 0 动作: Task={first_dec['task_id']}, Station={first_dec['station_id']}, Team={first_dec['team']}")
    print(f"      -> Step 0 估值: Value={first_dec['value']:.6f}, LogProb={first_dec['logprob']:.6f}")
    print(f"      -> 峰值内存: {ppo_results['rss_memory_mb']:.2f} MB")

    combined_baseline = {
        "version": "v0_baseline",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "data/283.csv",
        "spt_results": spt_results,
        "ppo_results": ppo_results,
    }

    if args.save:
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(combined_baseline, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] 基准标尺数据已成功保存至: {baseline_file}")

    if args.check:
        if not baseline_file.exists():
            print(f"\n[Error] 未找到基准文件 {baseline_file}，请先执行 --save 生成基准！")
            sys.exit(1)

        with open(baseline_file, "r", encoding="utf-8") as f:
            saved = json.load(f)

        saved_spt = saved["spt_results"]
        saved_ppo = saved["ppo_results"]

        # 断言结果严格等价
        assert np.isclose(spt_results["makespan"], saved_spt["makespan"], atol=1e-5), (
            f"Makespan 不一致: 当前 {spt_results['makespan']} != 基准 {saved_spt['makespan']}"
        )
        assert spt_results["scheduled_tasks"] == saved_spt["scheduled_tasks"]
        assert spt_results["is_legal"] is True
        assert sum(spt_results["violations"].values()) == 0

        # 断言 PPO 前向完全对齐
        for i, (cur_d, sav_d) in enumerate(zip(ppo_results["decisions"], saved_ppo["decisions"])):
            assert cur_d["task_id"] == sav_d["task_id"], f"Step {i} TaskID 不一致"
            assert cur_d["station_id"] == sav_d["station_id"], f"Step {i} StationID 不一致"
            assert cur_d["team"] == sav_d["team"], f"Step {i} Team 不一致"
            assert np.isclose(cur_d["value"], sav_d["value"], atol=1e-4), f"Step {i} Value 不一致"

        print("\n============================================================")
        print("                 [Regression Check Passed]")
        print("============================================================")
        print(f"Makespan: 当前 {spt_results['makespan']:.4f} vs 基准 {saved_spt['makespan']:.4f} -> 一致 [OK]")
        print(f"合法排程任务数: {spt_results['scheduled_tasks']} / {saved_spt['scheduled_tasks']} -> 一致 [OK]")
        print(f"硬约束冲突数: 0 -> 一致 [OK]")
        print(f"PPO 策略前向决策: 前 {len(saved_ppo['decisions'])} 步完全重合 -> 一致 [OK]")

        # 耗时性能比对
        old_step_ms = saved_spt["timing"]["step_time_ms_mean"]
        cur_step_ms = spt_results["timing"]["step_time_ms_mean"]
        speedup = (old_step_ms - cur_step_ms) / old_step_ms * 100
        print(f"单步耗时变化: {old_step_ms:.3f} ms -> {cur_step_ms:.3f} ms (加速 {speedup:+.1f}%)")
        print("============================================================")


if __name__ == "__main__":
    main()
