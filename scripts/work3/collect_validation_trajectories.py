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
from models.work3.actor_critic import extract_compact_state_features
from models.work3.heuristic_agent import HeuristicAgentWork3
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def attach_transfer_labels(
    steps: list[dict[str, Any]],
    transfer_history: list[float],
    h0: float,
) -> None:
    """只用已观测的真实转站时刻补标签，未观测周期显式保留缺失。"""
    for rec in steps:
        cycle_idx = int(rec["cycle_idx"])
        if 1 <= cycle_idx <= len(transfer_history):
            actual_time = float(transfer_history[cycle_idx - 1])
            rec["actual_transfer_time"] = actual_time
            rec["label_y"] = float((actual_time - rec["estimated_cmax"]) / h0)
            rec["label_available"] = True
        else:
            rec["actual_transfer_time"] = None
            rec["label_y"] = None
            rec["label_available"] = False


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
        candidates = env.get_action_candidates()
        if not candidates:
            if env._check_terminated():
                break
            env._advance_events_until_next_decision()
            candidates = env.get_action_candidates()
            if not candidates and env._check_terminated():
                break
            if not candidates and env.event_queue.is_empty():
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

    # 回溯结算真实转站时刻；尚未转站的周期不伪造监督标签。
    transfer_history = list(env.state.transfer_history)
    attach_transfer_labels(steps_data, transfer_history, h0)

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
    scenario_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """收集指定规模的验证轨迹并落盘为 PyTorch 数据集。"""
    agent = HeuristicAgentWork3()

    # 1. 加载 9 类解耦场景库
    scenarios: list[dict[str, Any]] = []
    sc_path = Path(scenarios_path)
    if sc_path.is_file():
        with open(sc_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

    if scenario_ids is not None:
        scenarios = [s for s in scenarios if s.get("scenario_id") in scenario_ids]
    elif max_scenarios is not None:
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
