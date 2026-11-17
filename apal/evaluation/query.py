"""查询方法的独立评估入口；不复用历史C/D的时间预测和方法字母。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from apal.runtime import legacy_pickle
import torch
from omegaconf import OmegaConf

from apal.envs.work3.core_types import TaskStatus
from apal.envs.work3.environment import AirLineEnvWork3
from apal.models.work3.actor_critic import ActorCriticWork3
from apal.models.work3.ppo_trainer import PPOTrainerWork3
from apal.training.work3.runtime_config import load_work3_runtime_config, load_work3_episode_manifest, resolve_work3_precision, seed_work3_runtime
from apal.utils.work3.objective_evaluator import ObjectiveWeights, build_objective_report, evaluate_trajectory_objective
from apal.utils.work3.trajectory_feasibility import check_environment_trajectory_feasibility


def evaluate_query_trajectory(
    env: AirLineEnvWork3, actor: ActorCriticWork3, *, scenario: dict[str, Any] | None = None,
    max_decisions: int, max_wall_seconds: float | None = None, amp_dtype: str = "fp32",
    reset: bool = True,
) -> dict[str, Any]:
    """默认从零评估；reset=False继承现场，以初始费用为基准核对后续增量。"""
    if not actor.condition_queries or type(max_decisions) is not int or max_decisions < 1:
        raise ValueError("评估要求QUERY Actor及正整数决策预算")
    if max_wall_seconds is not None and (not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
        raise ValueError("时间预算必须为正有限数")
    device = next(actor.parameters()).device
    _, dtype = resolve_work3_precision(amp_dtype, device)
    if type(reset) is not bool:
        raise TypeError("reset必须为布尔值")
    if reset:
        env.reset()
    initial_time, initial_cost = float(env.state.current_time), float(env.cumulative_cost)
    initial_environment_steps = int(env.step_count)
    initial_objective = evaluate_trajectory_objective(env, weights=env.weights).j_total
    reward_offset = len(env.step_rewards)
    initial_commitments = {task.task_key: {"status": task.status.name, "station": task.current_station,
        "team": list(task.assigned_team), "start": task.scheduled_start, "align": task.alignment}
        for task in env.state.tasks.values() if task.status in (TaskStatus.RESERVED, TaskStatus.RUNNING, TaskStatus.COMPLETED)}
    if scenario is not None:
        env.load_scenario(scenario)
    scenario_cost = float(env.cumulative_cost) - initial_cost
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
            snapshot = actor.make_decision_snapshot(env, torch.zeros(32), torch.zeros(2), episode_index=int(env.step_count))
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
    incremental_cost = float(env.cumulative_cost) - initial_cost
    incremental_matches = math.isclose(-sum(env.step_rewards[reward_offset:]) + scenario_cost,
                                      incremental_cost, rel_tol=1e-8, abs_tol=1e-8)
    initial_matches = math.isclose(initial_cost, initial_objective, rel_tol=1e-8, abs_tol=1e-8)
    ledger_matches = (initial_matches and incremental_matches
                      and math.isclose(env.cumulative_cost, objective.j_total, rel_tol=1e-8, abs_tol=1e-8))
    return {
        "method": actor.input_semantics(), "success": complete and feasible and ledger_matches,
        "terminated": complete, "truncated": truncated or reason in {"time_budget", "decision_limit"},
        "termination_reason": reason, "decisions": steps, "elapsed_seconds": time.monotonic() - started,
        "completed_tasks": sum(task.status == TaskStatus.COMPLETED for task in env.state.tasks.values()),
        "total_tasks": len(env.state.tasks), "makespan": env.state.current_time if complete else None,
        "C_task": max(task.actual_end for task in env.state.tasks.values()) if complete else None,
        "partial_current_time": env.state.current_time, "transfer_policy": env.transfer_policy,
        "independent_feasible": feasible, "constraint_violations": violations,
        "cost_ledger_matches": ledger_matches,
        "reset": reset, "initial_time": initial_time, "initial_cost": initial_cost,
        "initial_environment_steps": initial_environment_steps,
        "initial_objective": initial_objective, "initial_cost_ledger_matches": initial_matches,
        "initial_commitments": initial_commitments, "scenario_cost_delta": scenario_cost,
        "incremental_cost": incremental_cost, "incremental_cost_ledger_matches": incremental_matches,
        "remaining_elapsed_time": float(env.state.current_time) - initial_time,
        "objective": build_objective_report(objective, env.weights, h0=env.state.h0),
        "objective_weights": asdict(env.weights), "scenario_id": (scenario or {}).get("scenario_id"),
        "disturbance_results": env.disturbance_event_results,
        "performance_improvement_verified": False,
    }


def evaluate_query_instances(
    manifest: Path, actor: ActorCriticWork3, *, split: str, weights: ObjectiveWeights,
    max_decisions: int, max_wall_seconds: float | None = None, amp_dtype: str = "fp32",
) -> dict[str, Any]:
    """公开清单逐例评估；证书与最优值不进入环境或策略，失败保留原报告。"""
    instances = load_work3_episode_manifest(manifest, split=split)
    device = next(actor.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    results = []
    for instance in instances:
        env = AirLineEnvWork3(baseline_json_path=Path(instance["baseline"]),
            raw_data_path=Path(instance["raw_data"]), worker_pool_path=Path(instance["worker_pool"]),
            weights=weights, **instance["env_config"])
        report = evaluate_query_trajectory(env, actor, scenario=instance["scenario"],
            max_decisions=max_decisions, max_wall_seconds=max_wall_seconds, amp_dtype=amp_dtype)
        report.update({key: instance[key] for key in ("instance_id", "parent_group", "family", "split", "public_sha256")})
        results.append(report)
    return {"manifest": str(Path(manifest).resolve()), "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            "split": split, "instances": results, "completed": sum(row["success"] for row in results),
            "total": len(results), "elapsed_seconds": sum(row["elapsed_seconds"] for row in results),
            "device": str(device), "gpu_memory_peak_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)}


def main() -> None:
    parser = argparse.ArgumentParser(description="QUERY方法无扰动/受扰评估")
    parser.add_argument("--config", type=Path, default=Path("conf") / "work3" / "train_query.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scenario-id", default=None, help="省略即无扰动；指定时从配置的场景库读取")
    parser.add_argument("--max-decisions", type=int, default=10000)
    parser.add_argument("--instance-manifest", type=Path, default=None, help="可选公开实例清单；与单场景选择互斥")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    if args.instance_manifest is not None and args.scenario_id is not None:
        raise ValueError("实例清单已有场景，不能同时指定单场景")
    if args.report.exists():
        raise FileExistsError(f"评估报告已存在：{args.report}")
    config = load_work3_runtime_config(args.config, args.set)
    if config.runtime.method_profile != "QUERY":
        raise ValueError("此入口只接受QUERY配置")
    seed_work3_runtime(config.runtime.seed, deterministic=config.runtime.deterministic)
    torch.set_num_threads(config.runtime.main_num_threads)
    actor = ActorCriticWork3(condition_queries=True, query_program_path=config.paths.get("query_program")).to(config.runtime.device)
    PPOTrainerWork3(actor, device=config.runtime.device, create_optimizer=False).load_checkpoint(str(args.checkpoint))
    weights = ObjectiveWeights(**OmegaConf.to_container(config.objective, resolve=True))
    scenario = None
    if args.scenario_id is not None:
        scenarios = json.loads(Path(config.runtime.scenario_pool_path).read_text(encoding="utf-8"))
        matches = [item for item in scenarios if item["scenario_id"] == args.scenario_id and item.get("valid", True)]
        if len(matches) != 1:
            raise ValueError("场景ID不存在、不唯一或标记为无效")
        scenario = matches[0]
    if args.instance_manifest is not None:
        report = evaluate_query_instances(args.instance_manifest, actor, split=args.split, weights=weights,
            max_decisions=args.max_decisions, max_wall_seconds=config.runtime.max_wall_seconds, amp_dtype=config.runtime.amp_dtype)
    else:
        env = AirLineEnvWork3(baseline_json_path=Path(config.paths.baseline),
                             raw_data_path=Path(config.paths.raw_data), weights=weights)
        report = evaluate_query_trajectory(env, actor, scenario=scenario, max_decisions=args.max_decisions,
                                           max_wall_seconds=config.runtime.max_wall_seconds,
                                           amp_dtype=config.runtime.amp_dtype)
    report["checkpoint"] = str(args.checkpoint.resolve())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, pickle_module=legacy_pickle)
    report["checkpoint_training_mode"] = checkpoint.get("run_metadata", {}).get("run_mode")
    report["checkpoint_stage"] = checkpoint.get("run_metadata", {}).get("stage", "ppo")
    report["checkpoint_evaluation_eligible"] = checkpoint.get("run_metadata", {}).get("checkpoint_evaluation_eligible", False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = ("completed", "total") if args.instance_manifest is not None else ("success", "termination_reason", "completed_tasks", "total_tasks")
    print(json.dumps({key: report[key] for key in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
