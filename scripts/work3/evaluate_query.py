"""查询方法的独立评估入口；不复用历史C/D的时间预测和方法字母。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from omegaconf import OmegaConf

from envs.work3.core_types import TaskStatus
from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import ActorCriticWork3
from models.work3.ppo_trainer import PPOTrainerWork3
from training.work3_runtime_config import load_work3_runtime_config, resolve_work3_precision, seed_work3_runtime
from utils.work3.objective_evaluator import ObjectiveWeights, build_objective_report, evaluate_trajectory_objective
from utils.work3.trajectory_feasibility import check_environment_trajectory_feasibility


def evaluate_query_trajectory(
    env: AirLineEnvWork3, actor: ActorCriticWork3, *, scenario: dict[str, Any] | None = None,
    max_decisions: int, max_wall_seconds: float | None = None, amp_dtype: str = "fp32",
) -> dict[str, Any]:
    """同一Actor生成无扰动安排，或在已有承诺及已揭示扰动后继续执行。"""
    if not actor.condition_queries or type(max_decisions) is not int or max_decisions < 1:
        raise ValueError("评估要求QUERY Actor及正整数决策预算")
    if max_wall_seconds is not None and (not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
        raise ValueError("时间预算必须为正有限数")
    device = next(actor.parameters()).device
    _, dtype = resolve_work3_precision(amp_dtype, device)
    env.reset()
    if scenario is not None:
        env.load_scenario(scenario)
    actor.eval()
    started = time.monotonic()
    reason = "decision_limit"
    truncated = False
    steps = 0
    with torch.inference_mode():
        for index in range(max_decisions):
            if env._check_terminated():
                reason = "completed"
                break
            if max_wall_seconds is not None and time.monotonic() - started >= max_wall_seconds:
                reason = "time_budget"
                break
            snapshot = actor.make_decision_snapshot(env, torch.zeros(32), torch.zeros(2), episode_index=index)
            with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                action, _, _, _ = actor.select_snapshot(snapshot, deterministic=True)
            _, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                reason = str(info.get("termination_reason", "rollout_truncated"))
                break
    complete = env._check_terminated()
    if complete:
        reason = "completed"
    feasible, violations = check_environment_trajectory_feasibility(env)
    objective = evaluate_trajectory_objective(env, weights=env.weights)
    ledger_matches = (math.isclose(env.cumulative_cost, objective.j_total, rel_tol=1e-8, abs_tol=1e-8)
                      and math.isclose(-sum(env.step_rewards), objective.j_total, rel_tol=1e-8, abs_tol=1e-8))
    return {
        "method": actor.input_semantics(), "success": complete and feasible and ledger_matches,
        "terminated": complete, "truncated": truncated or reason in {"time_budget", "decision_limit"},
        "termination_reason": reason, "decisions": steps, "elapsed_seconds": time.monotonic() - started,
        "completed_tasks": sum(task.status == TaskStatus.COMPLETED for task in env.state.tasks.values()),
        "total_tasks": len(env.state.tasks), "makespan": env.state.current_time if complete else None,
        "partial_current_time": env.state.current_time, "transfer_policy": env.transfer_policy,
        "independent_feasible": feasible, "constraint_violations": violations,
        "cost_ledger_matches": ledger_matches,
        "objective": build_objective_report(objective, env.weights, h0=env.state.h0),
        "objective_weights": asdict(env.weights), "scenario_id": (scenario or {}).get("scenario_id"),
        "disturbance_results": env.disturbance_event_results,
        "performance_improvement_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="QUERY方法无扰动/受扰评估")
    parser.add_argument("--config", type=Path, default=Path("conf") / "work3" / "train_query.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scenario-id", default=None, help="省略即无扰动；指定时从配置的场景库读取")
    parser.add_argument("--max-decisions", type=int, default=10000)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"评估报告已存在：{args.report}")
    config = load_work3_runtime_config(args.config, args.set)
    if config.runtime.method_profile != "QUERY":
        raise ValueError("此入口只接受QUERY配置")
    seed_work3_runtime(config.runtime.seed, deterministic=config.runtime.deterministic)
    torch.set_num_threads(config.runtime.main_num_threads)
    actor = ActorCriticWork3(condition_queries=True).to(config.runtime.device)
    PPOTrainerWork3(actor, device=config.runtime.device, create_optimizer=False).load_checkpoint(str(args.checkpoint))
    env = AirLineEnvWork3(baseline_json_path=Path(config.paths.baseline),
                         raw_data_path=Path(config.paths.raw_data),
                         weights=ObjectiveWeights(**OmegaConf.to_container(config.objective, resolve=True)))
    scenario = None
    if args.scenario_id is not None:
        scenarios = json.loads(Path(config.runtime.scenario_pool_path).read_text(encoding="utf-8"))
        matches = [item for item in scenarios if item["scenario_id"] == args.scenario_id and item.get("valid", True)]
        if len(matches) != 1:
            raise ValueError("场景ID不存在、不唯一或标记为无效")
        scenario = matches[0]
    report = evaluate_query_trajectory(env, actor, scenario=scenario, max_decisions=args.max_decisions,
                                       max_wall_seconds=config.runtime.max_wall_seconds,
                                       amp_dtype=config.runtime.amp_dtype)
    report["checkpoint"] = str(args.checkpoint.resolve())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    report["checkpoint_training_mode"] = checkpoint.get("run_metadata", {}).get("run_mode")
    report["checkpoint_evaluation_eligible"] = checkpoint.get("run_metadata", {}).get("checkpoint_evaluation_eligible", False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("success", "termination_reason", "completed_tasks", "total_tasks")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
