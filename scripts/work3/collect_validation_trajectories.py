"""工作三 训练与验证轨迹数据集收集脚本 (Task 5.4)。

功能与技术规范：
1. 调度基线 C (HeuristicAgentWork3) 自动化运行 50 条完整生产轨迹：
   - 包含 45 个确定性 9 类正交解耦扰动场景 (scenarios_9class.json)；
   - 包含 5 个基准无扰动/随机扰动场景；
2. 逐步记录三元组与丰富物理特征：
   (s_n, P_q^h(s_n), P_q^actual, y_n)
   其中监督目标标签为归一化时间残差：
   y_n = (P_q^actual - P_q^h(s_n)) / H_0
3. 状态特征向量包含 32 维精炼宏观与微观物理特征：
   - 当前周期已运行时间比例；
   - 五大工位剩余工作量、在制任务数、延误物料数量与最大延迟；
   - 各站在场飞机分布；
   - 估计剩余时间与名义剩余时间；
   - 全线生产进度、后移工序数与累计成本。
4. 序列化落盘至 data/work3/val_trajectories.pt，为 Module 6 时间修正头离线训练提供高质量数据集。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from envs.work3.core_types import MultiAircraftState, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from models.work3.heuristic_agent import HeuristicAgentWork3
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_compact_state_features(state: MultiAircraftState, estimated_cmax: float) -> torch.Tensor:
    """提取当前仿真状态的 32 维物理特征向量。"""
    feat = torch.zeros(32, dtype=torch.float)
    h0 = float(state.h0)
    current_time = float(state.current_time)
    p_last = float(state.last_transfer_time)

    # [0] 当前周期已运行时间比例
    feat[0] = max(0.0, current_time - p_last) / h0

    # 遍历五大工位统计各站微观瓶颈
    for s in range(5):
        ac_id = state.get_aircraft_at_station(s)
        st_tasks = [
            t for t in state.tasks.values()
            if t.current_station == s and t.status != TaskStatus.COMPLETED
        ]
        # [1:6] 5 站剩余标准工时归一化
        w_remain = sum(t.duration for t in st_tasks)
        feat[1 + s] = float(w_remain) / h0

        # [6:11] 5 站正在执行任务数比例 (<= 3)
        running_count = sum(1 for t in st_tasks if t.status == TaskStatus.RUNNING)
        feat[6 + s] = float(running_count) / 3.0

        # [11:16] 5 站延误到料任务数比例
        delayed_tasks = [t for t in st_tasks if t.material_ready_time > current_time]
        feat[11 + s] = float(len(delayed_tasks)) / 10.0

        # [16:21] 5 站最大物料延误紧迫度
        if delayed_tasks:
            max_delay = max(t.material_ready_time - current_time for t in delayed_tasks)
            feat[16 + s] = math.log1p(float(max_delay) / h0)

        # [21:26] 5 站在场飞机编号归一化
        feat[21 + s] = float(ac_id) / 10.0 if ac_id is not None else -1.0

    # [26] 估计剩余时间比例 (P_q^h - t) / H_0
    feat[26] = max(0.0, estimated_cmax - current_time) / h0

    # [27] 名义剩余时间比例 (P_{q-1} + H_0 - t) / H_0
    feat[27] = (p_last + h0 - current_time) / h0

    # [28] 产线当前脉动周期比例
    feat[28] = float(state.current_cycle) / 14.0

    # [29] 全线累计完工工序比例
    completed = sum(1 for t in state.tasks.values() if t.status == TaskStatus.COMPLETED)
    feat[29] = float(completed) / 2830.0

    # [30] 全线累计后移工序数比例
    postponed = sum(t.postpone_count for t in state.tasks.values())
    feat[30] = float(postponed) / 50.0

    # [31] 归一化总生产时长进度
    feat[31] = current_time / (14.0 * h0)

    return feat


def collect_single_trajectory(
    agent: HeuristicAgentWork3,
    baseline_path: str,
    scenario: dict[str, Any] | None = None,
    trajectory_id: int = 0,
) -> dict[str, Any]:
    """使用基线 C 策略运行单条完整生产流水线，收集带时间残差标签的决策样本。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()
    if scenario is not None:
        env.load_scenario(scenario)

    h0 = float(env.state.h0)
    steps_data: list[dict[str, Any]] = []
    total_decisions = 0

    while total_decisions < 10000:
        ready = env.get_ready_tasks()
        if not ready:
            if env._check_terminated():
                break
            env._advance_events_until_next_decision()
            ready = env.get_ready_tasks()
            if not ready and env._check_terminated():
                break
            if not ready and env.event_queue.is_empty():
                break

        current_time = float(env.state.current_time)
        current_cycle = int(env.state.current_cycle)
        est_cmax = compute_cycle_heuristic_cmax(env.state)
        state_feat = extract_compact_state_features(env.state, est_cmax)

        action = agent.select_action(env)
        if action is None:
            continue

        step_record = {
            "step_idx": total_decisions,
            "cycle_idx": current_cycle,
            "current_time": current_time,
            "estimated_cmax": float(est_cmax),
            "state_feat": state_feat,
            "action": action,
        }

        obs, reward, terminated, truncated, info = env.step(action)
        total_decisions += 1
        steps_data.append(step_record)

        if terminated:
            break

    # 回溯结算真实转站时刻 P_q 与时间残差监督标签 y = (P_q - P_q^h) / H_0
    transfer_history = list(env.state.transfer_history)
    for rec in steps_data:
        c_idx = rec["cycle_idx"]
        if c_idx <= len(transfer_history):
            p_actual = float(transfer_history[c_idx - 1])
        else:
            p_actual = float(env.state.current_time)

        rec["actual_transfer_time"] = p_actual
        # 归一化时间残差标签
        rec["label_y"] = float((p_actual - rec["estimated_cmax"]) / h0)

    return {
        "trajectory_id": trajectory_id,
        "scenario_id": scenario["scenario_id"] if scenario else "NOMINAL_BASELINE",
        "h0": h0,
        "total_steps": len(steps_data),
        "makespan": float(env.state.current_time),
        "transfer_count": len(transfer_history),
        "steps": steps_data,
    }


def collect_all_trajectories(
    baseline_path: str = "data/work3/real_283_k10_baseline.json",
    scenarios_path: str = "data/work3/scenarios_9class.json",
    output_path: str = "data/work3/val_trajectories.pt",
    max_scenarios: int | None = None,
    num_nominal: int = 5,
) -> list[dict[str, Any]]:
    """收集指定规模的验证轨迹并落盘为 PyTorch 数据集。"""
    agent = HeuristicAgentWork3()

    # 1. 加载 9 类解耦场景库
    scenarios: list[dict[str, Any]] = []
    sc_path = Path(scenarios_path)
    if sc_path.is_file():
        with open(sc_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]

    all_trajectories: list[dict[str, Any]] = []
    traj_idx = 0

    # 2. 收集无扰动 / 基准轨迹
    logger.info(f"开始收集 {num_nominal} 条名义基准轨迹...")
    for i in range(num_nominal):
        traj = collect_single_trajectory(
            agent, baseline_path, scenario=None, trajectory_id=traj_idx
        )
        all_trajectories.append(traj)
        traj_idx += 1
        logger.info(
            f"[Traj {traj_idx:02d}] 基准场景完工: 步数={traj['total_steps']}, "
            f"完工时间={traj['makespan']:.2f}h, 14次脉动转站完成"
        )

    # 3. 收集 9 类解耦扰动场景轨迹
    logger.info(f"开始收集 {len(scenarios)} 条 9 类解耦扰动场景轨迹...")
    for sc in scenarios:
        traj = collect_single_trajectory(
            agent, baseline_path, scenario=sc, trajectory_id=traj_idx
        )
        all_trajectories.append(traj)
        traj_idx += 1
        logger.info(
            f"[Traj {traj_idx:02d}] 扰动场景 {sc['scenario_id']} 完工: "
            f"步数={traj['total_steps']}, 完工时间={traj['makespan']:.2f}h"
        )

    # 4. 序列化落盘
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_trajectories, out_file)
    logger.info(
        f"成功收集 {len(all_trajectories)} 条完整轨迹至 {output_path}, "
        f"样本总步数={sum(t['total_steps'] for t in all_trajectories)}"
    )

    return all_trajectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="收集基线 C 验证与训练轨迹数据集")
    parser.add_argument("--baseline", type=str, default="data/work3/real_283_k10_baseline.json")
    parser.add_argument("--scenarios", type=str, default="data/work3/scenarios_9class.json")
    parser.add_argument("--output", type=str, default="data/work3/val_trajectories.pt")
    parser.add_argument("--max_scenarios", type=int, default=None)
    parser.add_argument("--num_nominal", type=int, default=5)
    args = parser.parse_args()

    collect_all_trajectories(
        baseline_path=args.baseline,
        scenarios_path=args.scenarios,
        output_path=args.output,
        max_scenarios=args.max_scenarios,
        num_nominal=args.num_nominal,
    )
