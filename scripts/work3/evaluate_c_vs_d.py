"""工作三 基线 C 与方法 D 对比评测脚本 (Task 7.4 / 里程碑 M5 验收)。

核心评测目标与技术规范：
1. 在固定独立测试事件清单下：
   - 运行正式方法 C 图策略 (ActorCriticWork3, 启发式时间输入)；
   - 运行正式方法 D 图策略 (ActorCriticWork3, deterministic=True, 学习修正时间输入)；
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
from dataclasses import dataclass
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
from models.work3.time_head import TimeResidualHead
from scripts.work3.collect_validation_trajectories import load_scenarios_for_split
from scripts.work3.experiment_protocol import Work3MethodProfile, build_method_profile
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline
from utils.work3.trajectory_feasibility import (
    TaskConstraintRecord,
    TrajectoryExecutionRecord,
    validate_trajectory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FormalEvaluationAgent:
    """正式C/D评测包装器；启发式智能体不属于该类型。"""

    profile: Work3MethodProfile
    actor_critic: ActorCriticWork3
    time_head: TimeResidualHead | None
    debug_random: bool = False

    def select_action(self, env: AirLineEnvWork3) -> dict[str, Any] | None:
        """使用启发式或学习修正时间输入调用同一图策略。"""
        device = next(self.actor_critic.parameters()).device
        cmax = compute_cycle_heuristic_cmax(env.state)
        state_feat = extract_compact_state_features(env.state, cmax)
        if self.profile.use_corrected_time_input:
            if self.time_head is None:
                raise RuntimeError("正式方法 D 缺少共享时间头")
            graph_snapshot = self.actor_critic.build_graph_snapshot(env)
            with torch.no_grad():
                shared_feature = self.actor_critic.encode_shared_representation(
                    state_feat.to(device),
                    graph_snapshot,
                ).unsqueeze(0)
                _, remaining, _ = self.time_head.predict_corrected_time(
                    state_feat=shared_feature,
                    estimated_cmax=cmax,
                    current_time=float(env.state.current_time),
                    h0=float(env.state.h0),
                    last_transfer_time=float(env.state.last_transfer_time),
                )
                time_urgency = compute_time_urgency_vector(
                    estimated_r=remaining.squeeze(0),
                    current_time=env.state.current_time,
                    last_transfer_time=env.state.last_transfer_time,
                    h0=env.state.h0,
                    device=device,
                )
        else:
            time_urgency = compute_time_urgency_vector(
                estimated_r=max(0.0, cmax - float(env.state.current_time)),
                current_time=env.state.current_time,
                last_transfer_time=env.state.last_transfer_time,
                h0=env.state.h0,
                device=device,
            )
        action, _, _, _ = self.actor_critic.select_action(
            env=env,
            state_feat=state_feat,
            time_urgency=time_urgency,
            deterministic=True,
        )
        return action


def build_formal_evaluation_agent(
    method_variant: str,
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    debug_random: bool = False,
) -> FormalEvaluationAgent:
    """加载正式C/D模型；缺检查点仅在显式调试模式允许随机权重。"""
    profile = build_method_profile(method_variant)
    torch_device = torch.device(device)
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64).to(torch_device)
    time_head: TimeResidualHead | None = None
    path = Path(checkpoint_path)
    if not path.is_file():
        if not debug_random:
            raise FileNotFoundError(f"正式方法 {profile.name} 缺少检查点: {path}")
        if profile.use_time_auxiliary:
            time_head = TimeResidualHead(in_dim=actor.hidden_dim, hidden_dim=64).to(torch_device)
        actor.eval()
        if time_head is not None:
            time_head.eval()
        return FormalEvaluationAgent(profile, actor, time_head, debug_random=True)

    checkpoint = torch.load(path, map_location=torch_device)
    actor_state = checkpoint.get("actor_critic_state")
    if actor_state is None:
        if debug_random:
            time_head = (
                TimeResidualHead(in_dim=actor.hidden_dim, hidden_dim=64).to(torch_device)
                if profile.use_time_auxiliary
                else None
            )
            actor.eval()
            if time_head is not None:
                time_head.eval()
            return FormalEvaluationAgent(profile, actor, time_head, debug_random=True)
        raise ValueError(f"方法 {profile.name} 检查点缺少 actor_critic_state: {path}")
    actor.load_state_dict(actor_state)

    if profile.use_time_auxiliary:
        if checkpoint.get("time_head_model_version") != "signed_residual_v1":
            if debug_random:
                time_head = TimeResidualHead(in_dim=actor.hidden_dim, hidden_dim=64).to(torch_device)
                actor.eval()
                time_head.eval()
                return FormalEvaluationAgent(profile, actor, time_head, debug_random=True)
            raise ValueError("正式方法 D 检查点缺少有符号时间头版本")
        if "time_head_state" not in checkpoint:
            if debug_random:
                time_head = TimeResidualHead(in_dim=actor.hidden_dim, hidden_dim=64).to(torch_device)
                actor.eval()
                time_head.eval()
                return FormalEvaluationAgent(profile, actor, time_head, debug_random=True)
            raise ValueError("正式方法 D 检查点缺少共享时间头")
        time_head = TimeResidualHead(in_dim=actor.hidden_dim, hidden_dim=64).to(torch_device)
        time_head.load_state_dict(checkpoint["time_head_state"])

    actor.eval()
    if time_head is not None:
        time_head.eval()
    return FormalEvaluationAgent(profile, actor, time_head, debug_random=False)


def _check_completed_trajectory_feasibility(
    env: AirLineEnvWork3,
) -> tuple[bool, dict[str, int]]:
    """从实际执行字段重建记录，再调用独立检查器复核可行性。"""
    if not env._check_terminated():
        return False, {"incomplete_trajectory": 1}

    records: list[TrajectoryExecutionRecord] = []
    constraints: dict[int, TaskConstraintRecord] = {}
    for task in env.state.tasks.values():
        constraints[task.task_id] = TaskConstraintRecord(
            demand=task.demand,
            required_skill=task.skill,
            predecessors=tuple(task.predecessors),
            fixed_station=task.fixed_station,
            max_allowed_station=task.max_allowed_station,
        )
        if task.actual_start is None or task.actual_end is None:
            return False, {"missing_execution_interval": 1}
        aircraft = env.state.aircraft[task.aircraft_id]
        records.append(
            TrajectoryExecutionRecord(
                aircraft_id=task.aircraft_id,
                task_id=task.task_id,
                station_id=task.current_station,
                team=tuple(task.assigned_team),
                start=float(task.actual_start),
                end=float(task.actual_end),
                material_ready_time=float(task.material_ready_time),
                station_entry_time=aircraft.entry_times.get(task.current_station),
                aircraft_station_at_start=task.current_station,
            )
        )

    report = validate_trajectory(
        records,
        task_constraints=constraints,
        worker_skills=env.worker_skills,
        worker_station_bindings={
            worker_id: station_id
            for station_id, worker_ids in env.state.station_worker_bindings.items()
            for worker_id in worker_ids
        },
        station_capacities={
            station_id: env.max_slots_per_station
            for station_id in range(env.state.num_stations)
        },
    )
    return report.is_feasible, dict(report.violations)


def _count_actual_disturbance_hits(
    env: AirLineEnvWork3,
    scenario: dict[str, Any] | None,
) -> int:
    """按环境实际继承的物料恢复时刻统计已命中的受扰工序。"""
    if scenario is None:
        return 0
    return sum(
        1
        for task_key in scenario.get("affected_task_keys", [])
        if task_key in env.state.tasks
        and env.state.tasks[task_key].material_ready_time > env.tolerance
    )


def summarize_disturbance_effects(
    task_records: dict[str, Any],
    scenario: dict[str, Any] | None,
    *,
    baseline_start_by_key: dict[str, float],
    baseline_material_ready_by_key: dict[str, float] | None = None,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """区分预设目标、实际命中、观察等待和其他飞机传播。"""
    if scenario is None:
        return {
            "target_count": 0,
            "actual_hit_count": 0,
            "actual_hit_rate": 0.0,
            "actual_hit_task_keys": [],
            "observed_added_wait_hours": 0.0,
            "cross_aircraft_affected_task_count": 0,
            "cross_aircraft_affected_aircraft_ids": [],
        }

    baseline_ready = baseline_material_ready_by_key or {}
    target_keys = [str(key) for key in scenario.get("affected_task_keys", [])]
    target_aircraft_ids = {int(scenario["aircraft_id"])}
    actual_hit_keys = [
        key
        for key in target_keys
        if key in task_records
        and float(getattr(task_records[key], "material_ready_time", 0.0))
        > float(baseline_ready.get(key, 0.0)) + tolerance
    ]

    added_wait = 0.0
    for key in actual_hit_keys:
        task = task_records[key]
        actual_start = getattr(task, "actual_start", None)
        baseline_start = baseline_start_by_key.get(key)
        if actual_start is not None and baseline_start is not None:
            added_wait += max(0.0, float(actual_start) - float(baseline_start))

    cross_aircraft_keys: list[str] = []
    for key, task in task_records.items():
        if key in target_keys or int(task.aircraft_id) in target_aircraft_ids:
            continue
        actual_start = getattr(task, "actual_start", None)
        baseline_start = baseline_start_by_key.get(key)
        if (
            actual_start is not None
            and baseline_start is not None
            and float(actual_start) > float(baseline_start) + tolerance
        ):
            cross_aircraft_keys.append(str(key))

    cross_aircraft_ids = sorted({int(task_records[key].aircraft_id) for key in cross_aircraft_keys})
    target_count = len(target_keys)
    return {
        "target_count": target_count,
        "actual_hit_count": len(actual_hit_keys),
        "actual_hit_rate": len(actual_hit_keys) / target_count if target_count else 0.0,
        "actual_hit_task_keys": actual_hit_keys,
        "observed_added_wait_hours": added_wait,
        "cross_aircraft_affected_task_count": len(cross_aircraft_keys),
        "cross_aircraft_affected_aircraft_ids": cross_aircraft_ids,
    }


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
    termination_reason = "decision_limit"
    terminated = False
    truncated = False
    with torch.inference_mode():
        while decisions < max_decisions:
            candidates = env.get_action_candidates()
            if not candidates:
                if env._check_terminated():
                    termination_reason = "completed"
                    break
                env._advance_events_until_next_decision()
                candidates = env.get_action_candidates()
                if not candidates and env._check_terminated():
                    termination_reason = "completed"
                    break
                if not candidates and env.event_queue.is_empty():
                    termination_reason = "deadlock"
                    break

            if agent_type in {"Baseline-C", "Heuristic-Debug"}:
                action = agent.select_action(env)
            elif agent_type in {"Method-C", "Method-D"}:
                if isinstance(agent, FormalEvaluationAgent):
                    action = agent.select_action(env)
                else:
                    raise TypeError(f"正式{agent_type}必须使用FormalEvaluationAgent")
            else:
                raise ValueError(f"未知智能体类型: {agent_type}")

            if action is None:
                if env.event_queue.is_empty():
                    termination_reason = "deadlock"
                    break
                env._advance_events_until_next_decision()
                continue

            obs, reward, terminated, truncated, info = env.step(action)
            decisions += 1
            if terminated:
                termination_reason = str(
                    info.get(
                        "termination_reason",
                        "completed" if env._check_terminated() else "deadlock",
                    )
                )
                break
            if truncated:
                termination_reason = "rollout_truncated"
                break

    if decisions >= max_decisions and not terminated and not truncated:
        termination_reason = "decision_limit"

    # 统计生产指标
    completed = sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.COMPLETED)
    postponed_total = sum(t.postpone_count for t in env.state.tasks.values())
    total_cost = env.cumulative_cost
    feasible, constraint_violations = _check_completed_trajectory_feasibility(env)
    success = bool(
        feasible
        and termination_reason == "completed"
        and env._check_terminated()
    )
    baseline = MultiAircraftBaseline.load_from_json(env.baseline_json_path)
    baseline_start_by_key = {
        task.task_key: float(task.baseline_start)
        for task in baseline.tasks.values()
    }
    effect_report = summarize_disturbance_effects(
        env.state.tasks,
        scenario,
        baseline_start_by_key=baseline_start_by_key,
        tolerance=env.tolerance,
    )

    return {
        "agent": agent_type,
        "scenario_id": scenario["scenario_id"] if scenario else "NOMINAL",
        "j_takt": env.cost_takt,
        "d_time": env.cost_time,
        "d_team": env.cost_team,
        "j_postpone": env.cost_postpone,
        "j_revision": env.cost_revision,
        "j_total": total_cost,
        "raw_cost_components": {
            "j_takt": env.cost_takt,
            "d_time": env.cost_time,
            "d_team": env.cost_team,
            "j_postpone": env.cost_postpone,
            "j_revision": env.cost_revision,
        },
        "makespan": float(env.state.current_time),
        "completed_tasks": completed,
        "success": success,
        "feasible": feasible,
        "termination_reason": termination_reason,
        "constraint_violations": constraint_violations,
        "constraint_violation_count": sum(constraint_violations.values()),
        "actual_disturbance_hits": effect_report["actual_hit_count"],
        "disturbance_effects": effect_report,
        "postponed_count": postponed_total,
        "decisions": decisions,
        "transfers": len(env.state.transfer_history),
    }


def run_benchmark_evaluation(
    baseline_path: str = "data/work3/real_283_k10_baseline.json",
    scenarios_path: str = "data/work3/scenarios_9class.json",
    scenario_split_path: str | Path | None = None,
    method_c_ckpt: str = "models/work3/checkpoints/method_c_model.pt",
    method_d_ckpt: str = "models/work3/checkpoints/method_d_model.pt",
    selected_scenario_ids: list[str] | None = None,
    output_json: str = "data/work3/eval_c_vs_d_m5.json",
    device: str = "cpu",
    allow_debug_random: bool = False,
    max_decisions: int = 10000,
) -> list[dict[str, Any]]:
    """在固定外生事件上比较同架构正式方法 C 与 D。"""
    torch_device = torch.device(device)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    # 1. 载入场景库
    if scenario_split_path is not None:
        all_scenarios = load_scenarios_for_split(scenarios_path, scenario_split_path)
    else:
        with open(scenarios_path, "r", encoding="utf-8") as f:
            all_scenarios = json.load(f)

    if selected_scenario_ids:
        scenarios = [s for s in all_scenarios if s["scenario_id"] in selected_scenario_ids]
        if len(scenarios) != len(selected_scenario_ids):
            raise ValueError("选定的正式评测事件不在固定场景清单中")
    else:
        if scenario_split_path is None:
            raise ValueError("正式评测必须显式提供独立场景清单")
        scenarios = list(all_scenarios)
    if not scenarios:
        raise ValueError("正式评测场景清单为空")

    # 2. 初始化智能体
    agent_c = build_formal_evaluation_agent(
        "C",
        method_c_ckpt,
        device=device,
        debug_random=allow_debug_random,
    )
    agent_d = build_formal_evaluation_agent(
        "D",
        method_d_ckpt,
        device=device,
        debug_random=allow_debug_random,
    )

    env = AirLineEnvWork3(baseline_json_path=baseline_path)

    results: list[dict[str, Any]] = []

    logger.info("=" * 105)
    logger.info(f"{'Scenario ID':<16} | {'Metric':<10} | {'Method C':<12} | {'Method D':<12} | {'Delta (%)':<10}")
    logger.info("=" * 105)

    for sc in scenarios:
        sc_id = sc["scenario_id"]

        # 评测基线 C
        t0 = time.time()
        res_c = evaluate_single_trajectory(
            env,
            "Method-C",
            agent_c,
            scenario=sc,
            max_decisions=max_decisions,
            device=device,
        )
        t_c = time.time() - t0

        # 评测方法 D
        t0 = time.time()
        res_d = evaluate_single_trajectory(
            env,
            "Method-D",
            agent_d,
            scenario=sc,
            max_decisions=max_decisions,
            device=device,
        )
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
            "method_c": res_c,
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
    parser = argparse.ArgumentParser(description="工作三 正式方法 C vs D 对比评测")
    parser.add_argument("--baseline", type=str, default="data/work3/real_283_k10_baseline.json")
    parser.add_argument("--scenarios", type=str, default="data/work3/scenarios_9class.json")
    parser.add_argument("--scenario-split", type=str, default="data/work3/experiment_splits/test.json")
    parser.add_argument("--c-ckpt", type=str, default="models/work3/checkpoints/method_c_model.pt")
    parser.add_argument("--d-ckpt", type=str, default="models/work3/checkpoints/method_d_model.pt")
    parser.add_argument("--output", type=str, default="data/work3/eval_c_vs_d_m5.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--debug-random", action="store_true", help="仅烟测时允许缺检查点随机权重")
    args = parser.parse_args()

    run_benchmark_evaluation(
        baseline_path=args.baseline,
        scenarios_path=args.scenarios,
        scenario_split_path=args.scenario_split,
        method_c_ckpt=args.c_ckpt,
        method_d_ckpt=args.d_ckpt,
        output_json=args.output,
        device=args.device,
        allow_debug_random=args.debug_random,
    )


if __name__ == "__main__":
    main()
