"""工作三 基线 C 与方法 D 对比评测脚本 (Task 7.4 / 里程碑 M5 验收)。

核心评测目标与技术规范：
1. 在 9 类代表性解耦扰动场景 (data/work3/scenarios_9class.json) 下：
   - 运行基线 C 启发式调度智能体 (HeuristicAgentWork3)；
   - 运行方法 D 条件分支自回归 PPO 智能体 (ActorCriticWork3, deterministic=True)；
2. 收集生产全流程多目标核心指标：
   - J_takt: 脉动节拍超期惩罚；
   - D_time: 基准工序排程时间偏差；
   - D_team: 基准团队人员改派偏差；
   - J_postpone: 跨站后移改派惩罚；
   - J_total: 全线综合惩罚费用 (J_total = w_h J_takt + w_tau D_time + w_m D_team + w_p J_postpone)；
   - Makespan: 全线 10 架次飞机总完工时长；
3. 输出基线 C 与方法 D 的详细对照评估表与改进百分比，达成里程碑 M5 验收目标。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from envs.work3.core_types import TaskStatus
from envs.work3.environment import AirLineEnvWork3
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features
from models.work3.heuristic_agent import HeuristicAgentWork3
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_single_trajectory(
    env: AirLineEnvWork3,
    agent_type: str,
    agent: Any,
    scenario: dict[str, Any] | None = None,
    max_decisions: int = 10000,
    device: str = "cpu",
) -> dict[str, Any]:
    """运行单条生产轨迹并结算综合目标。"""
    torch_device = torch.device(device)
    env.reset()
    if scenario is not None:
        env.load_scenario(scenario)

    decisions = 0
    with torch.inference_mode():
        while decisions < max_decisions:
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

            if agent_type == "Baseline-C":
                action = agent.select_action(env)
            elif agent_type == "Method-D":
                cmax = compute_cycle_heuristic_cmax(env.state)
                s_feat = extract_compact_state_features(env.state, cmax)
                r_est = max(0.0, cmax - float(env.state.current_time))
                u_time = compute_time_urgency_vector(
                    estimated_r=r_est,
                    current_time=env.state.current_time,
                    last_transfer_time=env.state.last_transfer_time,
                    h0=env.state.h0,
                    device=torch_device,
                )
                action, _, _, _ = agent.select_action(
                    env=env,
                    state_feat=s_feat,
                    time_urgency=u_time,
                    deterministic=True,
                )
            else:
                raise ValueError(f"未知智能体类型: {agent_type}")

            if action is None:
                env._advance_events_until_next_decision()
                continue

            obs, reward, terminated, truncated, info = env.step(action)
            decisions += 1
            if terminated:
                break

    # 统计生产指标
    completed = sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.COMPLETED)
    postponed_total = sum(t.postpone_count for t in env.state.tasks.values())
    total_cost = env.cumulative_cost

    return {
        "agent": agent_type,
        "scenario_id": scenario["scenario_id"] if scenario else "NOMINAL",
        "j_takt": env.cost_takt,
        "d_time": env.cost_time,
        "d_team": env.cost_team,
        "j_postpone": env.cost_postpone,
        "j_revision": env.cost_revision,
        "j_total": total_cost,
        "makespan": float(env.state.current_time),
        "completed_tasks": completed,
        "postponed_count": postponed_total,
        "decisions": decisions,
        "transfers": len(env.state.transfer_history),
    }


def run_benchmark_evaluation(
    baseline_path: str = "data/work3/real_283_k10_baseline.json",
    scenarios_path: str = "data/work3/scenarios_9class.json",
    method_d_ckpt: str = "models/work3/checkpoints/method_d_model.pt",
    selected_scenario_ids: list[str] | None = None,
    output_json: str = "data/work3/eval_c_vs_d_m5.json",
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """在代表性扰动场景上全面对比基线 C 与方法 D。"""
    torch_device = torch.device(device)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    # 1. 载入场景库
    with open(scenarios_path, "r", encoding="utf-8") as f:
        all_scenarios: list[dict[str, Any]] = json.load(f)

    if selected_scenario_ids:
        scenarios = [s for s in all_scenarios if s["scenario_id"] in selected_scenario_ids]
    else:
        # 默认选取覆盖 9 类代表性扰动模式的典型场景
        target_ids = [
            "EARLY_LOW_S0", "EARLY_MID_S0", "EARLY_HIGH_S0",
            "MID_LOW_S1",   "MID_MID_S1",   "MID_HIGH_S1",
            "LATE_LOW_S2",  "LATE_MID_S2",  "LATE_HIGH_S4",
        ]
        scenarios = [s for s in all_scenarios if s["scenario_id"] in target_ids]
        if not scenarios:
            scenarios = all_scenarios[:9]

    # 2. 初始化智能体
    agent_c = HeuristicAgentWork3(name="Baseline-C")

    agent_d = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64).to(torch_device)
    if Path(method_d_ckpt).is_file():
        ckpt_data = torch.load(method_d_ckpt, map_location=torch_device)
        state_dict = ckpt_data.get("actor_critic_state", ckpt_data)
        agent_d.load_state_dict(state_dict)
        logger.info(f"成功载入方法 D 检查点: {method_d_ckpt}")
    else:
        logger.warning(f"未找到检查点 {method_d_ckpt}，使用初始权重评测")
    agent_d.eval()

    env = AirLineEnvWork3(baseline_json_path=baseline_path)

    results: list[dict[str, Any]] = []

    logger.info("=" * 105)
    logger.info(f"{'Scenario ID':<16} | {'Metric':<10} | {'Baseline C':<12} | {'Method D':<12} | {'Delta (%)':<10}")
    logger.info("=" * 105)

    for sc in scenarios:
        sc_id = sc["scenario_id"]

        # 评测基线 C
        t0 = time.time()
        res_c = evaluate_single_trajectory(env, "Baseline-C", agent_c, scenario=sc, device=device)
        t_c = time.time() - t0

        # 评测方法 D
        t0 = time.time()
        res_d = evaluate_single_trajectory(env, "Method-D", agent_d, scenario=sc, device=device)
        t_d = time.time() - t0

        # 计算综合目标优化幅度
        cost_c = res_c["j_total"]
        cost_d = res_d["j_total"]
        if cost_c > 1e-6:
            improv_pct = ((cost_c - cost_d) / cost_c) * 100.0
        else:
            improv_pct = 0.0

        item = {
            "scenario_id": sc_id,
            "timing": sc.get("timing"),
            "intensity": sc.get("intensity"),
            "station": sc.get("station_id"),
            "baseline_c": res_c,
            "method_d": res_d,
            "improvement_j_total_pct": improv_pct,
            "time_c_s": t_c,
            "time_d_s": t_d,
        }
        results.append(item)

        logger.info(
            f"{sc_id:<16} | {'J_total':<10} | {cost_c:<12.4f} | {cost_d:<12.4f} | {improv_pct:>+8.2f}%"
        )
        logger.info(
            f"{'':<16} | {'J_takt':<10} | {res_c['j_takt']:<12.4f} | {res_d['j_takt']:<12.4f} |"
        )
        logger.info(
            f"{'':<16} | {'D_time':<10} | {res_c['d_time']:<12.4f} | {res_d['d_time']:<12.4f} |"
        )
        logger.info(
            f"{'':<16} | {'D_team':<10} | {res_c['d_team']:<12.4f} | {res_d['d_team']:<12.4f} |"
        )
        logger.info(
            f"{'':<16} | {'J_post':<10} | {res_c['j_postpone']:<12.4f} | {res_d['j_postpone']:<12.4f} |"
        )
        logger.info("-" * 105)

    # 保存评测结果
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"评测结果已持久化保存至: {output_json}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="工作三 基线 C vs 方法 D 对比评测")
    parser.add_argument("--baseline", type=str, default="data/work3/real_283_k10_baseline.json")
    parser.add_argument("--scenarios", type=str, default="data/work3/scenarios_9class.json")
    parser.add_argument("--ckpt", type=str, default="models/work3/checkpoints/method_d_model.pt")
    parser.add_argument("--output", type=str, default="data/work3/eval_c_vs_d_m5.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_benchmark_evaluation(
        baseline_path=args.baseline,
        scenarios_path=args.scenarios,
        method_d_ckpt=args.ckpt,
        output_json=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
