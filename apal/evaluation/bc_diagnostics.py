"""固定BC轨迹的自主排程、教师指标及两种选模规则诊断入口。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch

from apal.data.work3.certified_examples import verify_certificate
from apal.envs.work3.environment import AirLineEnvWork3
from apal.evaluation.query import evaluate_query_trajectory
from apal.models.work3.actor_critic import ActorCriticWork3
from apal.models.work3.ppo_trainer import PPOTrainerWork3
from apal.project import PROJECT_ROOT
from apal.runtime.artifacts import create_run_context, resolve_path
from apal.training.work3.runtime_config import load_work3_episode_manifest, seed_work3_runtime
from apal.utils.work3.objective_evaluator import ObjectiveWeights


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_digest(path: Path) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    for name, value in sorted(payload["actor_critic_state"].items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(str((name, str(tensor.dtype), tuple(tensor.shape))).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def json_facts(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_facts(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_facts(item) for item in value]
    return value


def select_checkpoint(points: list[dict[str, Any]], rule: str) -> dict[str, Any]:
    """同轨迹比较；先保留完成分母，再比较合法终局gap，允许选择W0。"""
    if not points or len({point["total"] for point in points}) != 1 or points[0]["total"] <= 0:
        raise ValueError("检查点必须使用相同非空评估面板")
    if rule == "validation_nll":
        if any(not math.isfinite(point["validation_nll"]) for point in points):
            raise ValueError("验证NLL必须有限")
        return min(points, key=lambda point: (point["validation_nll"], point["update"]))
    if rule != "closure":
        raise ValueError("未知检查点选择规则")
    completed = max(point["legal_complete"] for point in points)
    eligible = [point for point in points if point["legal_complete"] == completed]
    gaps = [point["mean_gap"] for point in eligible if point["mean_gap"] is not None]
    if gaps:
        minimum = min(gaps)
        eligible = [point for point in eligible if point["mean_gap"] is not None
                    and point["mean_gap"] <= minimum + 1e-9]
    return min(eligible, key=lambda point: point["update"])


def result_row(model: dict[str, Any], case: dict[str, Any], report: dict[str, Any], *,
               optimum: float, phase: str) -> dict[str, Any]:
    terminated, truncated = bool(report["terminated"]), bool(report["truncated"])
    legal = bool(report["success"] and terminated and not truncated and report["independent_feasible"]
                 and report["cost_ledger_matches"])
    c_exit = report["makespan"] if terminated else None
    gap = None
    if legal and optimum > 0:
        if c_exit is None or not math.isfinite(c_exit) or c_exit < optimum - 1e-9:
            raise ValueError("合法终局低于证书或缺少有限完工时间")
        gap = (c_exit - optimum) / optimum
    return {"type": "evaluation", "phase": phase, "seed": model["seed"],
        "model_id": model["model_id"], "update": model["update"], "selection_rule": model["rule"],
        "checkpoint": model["path"], "split": case["split"], "instance_id": case["instance_id"],
        "parent_group": case["parent_group"], "family": case["family"], "success": legal,
        "terminated": terminated, "feasible": bool(report["independent_feasible"]), "truncated": truncated,
        "termination_reason": report["termination_reason"], "cost_ledger_matches": report["cost_ledger_matches"],
        "C_exit": c_exit, "C_task": report.get("C_task") if terminated else None,
        "certified_C_exit": optimum, "gap": gap, "decisions": report["decisions"],
        "elapsed_seconds": report["elapsed_seconds"], "reused_from": ""}


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = [row["gap"] for row in rows if row["gap"] is not None]
    return {"total": len(rows), "legal_complete": sum(row["success"] for row in rows),
        "mean_gap": statistics.mean(gaps) if gaps else None,
        "median_gap": statistics.median(gaps) if gaps else None, "gap_count": len(gaps),
        "truncated": sum(row["truncated"] for row in rows)}


class BCDiagnosis:
    """仅管理本轮固定模型与标量证据，采样/评分/物理执行仍委托原实现。"""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.source = resolve_path(config.source_run, PROJECT_ROOT)
        self.context = create_run_context(PROJECT_ROOT, experiment_name=config.artifacts.experiment_name,
            runs_root=config.artifacts.runs_root, run_id=config.artifacts.run_id)
        self.root = self.context.run_dir
        self.plan_path = self.context.configs_dir / "diagnostic_plan.json"
        self.plan = read_json(self.plan_path) if self.plan_path.exists() else None
        self.rows_path = self.context.reports_dir / "results.json"
        self.rows = read_json(self.rows_path) if self.rows_path.exists() else []
        self.cache_path = self.context.reports_dir / "evaluation_cache.json"
        self.cache = read_json(self.cache_path) if self.cache_path.exists() else {}
        torch.set_num_threads(int(config.runtime.main_num_threads))

    def prepare(self) -> None:
        if self.plan is not None:
            raise FileExistsError(self.plan_path)
        self.root.mkdir(parents=True, exist_ok=False)
        create_run_context(PROJECT_ROOT, experiment_name=self.context.experiment_name,
            runs_root=self.config.artifacts.runs_root, run_id=self.context.run_id, create_dirs=True)
        cases = {split: load_work3_episode_manifest(self.source / "data" / "pilot" / f"{split}_instances.json", split=split)
                 for split in ("train", "val")}
        panel = []
        for family, count in self.config.evaluation.train_panel.items():
            selected = [row["instance_id"] for row in cases["train"] if row["family"] == family][:int(count)]
            if len(selected) != count:
                raise ValueError("训练曲线面板缺少指定构造族")
            panel.extend(selected)
        models = []
        for seed in self.config.runtime.seeds:
            report = read_json(self.source / "artifacts" / "reports" / f"bc_seed{seed}.json")
            for role, filename, update in (("initial", f"initial_seed{seed}.pt", 0),
                    ("nll_best", f"bc_seed{seed}.pt", report["best_update"]),
                    ("final", f"bc_seed{seed}_final.pt", report["optimization_steps"])):
                path = self.source / "checkpoints" / filename
                models.append({"model_id": f"old_{role}_seed{seed}", "seed": int(seed), "update": update,
                    "rule": role, "path": str(path), "sha256": sha256(path), "parameters_sha256": parameter_digest(path)})
        identities = {}
        program = PROJECT_ROOT / "conf" / "work3" / "condition_program.json"
        identities[str(program)] = sha256(program)
        for name in ("train_instances.json", "val_instances.json"):
            path = self.source / "data" / "pilot" / name
            identities[str(path)] = sha256(path)
        for name in ("train_set_records.pt", "validation_set_records.pt"):
            path = self.source / "data" / name
            identities[str(path)] = sha256(path)
        for rows in cases.values():
            for case in rows:
                certificate = verify_certificate(Path(case["baseline"]).parent.parent)
                case["certified_C_exit"] = certificate["certified_C_exit"]
        self.plan = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "created_epoch": time.time(), "config": OmegaConf.to_container(self.config, resolve=True),
            "source_identities": identities, "models": models, "cases": cases,
            "train_panel": panel, "selection_uses_old_test": False,
            "curve_training_subset": "清单顺序选结构4、顺序2、后移2；不按模型表现选取"}
        write_json(self.plan_path, self.plan)
        (self.context.configs_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(self.config), encoding="utf-8")

    def validate_plan(self) -> None:
        if self.plan is None:
            raise FileNotFoundError("请先执行prepare冻结本轮身份与面板")
        if OmegaConf.to_container(self.config, resolve=True) != self.plan["config"]:
            raise ValueError("本轮配置已冻结，不能更改后复用已有评估或缓存")
        for name, expected in self.plan["source_identities"].items():
            if sha256(Path(name)) != expected:
                raise ValueError(f"冻结输入已改变: {name}")

    def actor(self, model: dict[str, Any]) -> ActorCriticWork3:
        if sha256(Path(model["path"])) != model["sha256"]:
            raise ValueError("源权重文件身份改变")
        seed_work3_runtime(model["seed"], deterministic=bool(self.config.runtime.deterministic))
        actor = ActorCriticWork3(condition_queries=True).to(self.config.runtime.device)
        PPOTrainerWork3(actor, device=self.config.runtime.device, create_optimizer=False).load_checkpoint(model["path"])
        return actor.eval()

    def environment(self, case: dict[str, Any]) -> AirLineEnvWork3:
        return AirLineEnvWork3(baseline_json_path=Path(case["baseline"]),
            raw_data_path=Path(case["raw_data"]), worker_pool_path=Path(case["worker_pool"]),
            weights=ObjectiveWeights(**OmegaConf.to_container(self.config.objective)), **case["env_config"])

    def save_rows(self) -> None:
        write_json(self.rows_path, self.rows)
        write_json(self.cache_path, self.cache)
        if self.rows:
            path = self.context.reports_dir / "results.csv"
            pending = path.with_suffix(".pending.csv")
            with pending.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
            pending.replace(path)

    def evaluate(self, model: dict[str, Any], split: str, *, phase: str, panel: bool = False) -> list[dict[str, Any]]:
        cases = self.plan["cases"][split]
        if panel and split == "train":
            cases = [row for row in cases if row["instance_id"] in self.plan["train_panel"]]
        actor, output = None, []
        prior = {(row["phase"], row["model_id"], row["split"], row["instance_id"]): row for row in self.rows}
        for case in cases:
            if time.time() - self.plan["created_epoch"] > self.config.runtime.total_wall_seconds:
                raise TimeoutError("本轮总时间预算已到，保留已有结果")
            identity = (phase, model["model_id"], split, case["instance_id"])
            if identity in prior:
                output.append(prior[identity]); continue
            key = f"{model['parameters_sha256']}:{split}:{case['instance_id']}"
            reused = ""
            if key in self.cache:
                report = self.cache[key]
                reused = report["cache_origin"]
            else:
                old_role = "initial" if model["rule"] == "initial" else "bc" if model["rule"] == "nll_best" else None
                old = self.source / "eval" / f"eval_{old_role}_seed{model['seed']}_val.json"
                if phase == "A" and split == "val" and old_role is not None:
                    old_report = read_json(old)
                    if Path(old_report["checkpoint"]).resolve() != Path(model["path"]).resolve():
                        raise ValueError("旧验证结果的权重来源不匹配")
                    if old_report["manifest_sha256"] != sha256(self.source / "data/pilot/val_instances.json"):
                        raise ValueError("旧验证结果的公开输入身份不匹配")
                    report = next(row for row in old_report["instances"] if row["instance_id"] == case["instance_id"])
                    if report["objective_weights"] != OmegaConf.to_container(self.config.objective):
                        raise ValueError("旧评估奖励与当前纯C_exit配置不一致")
                    reused = str(old)
                else:
                    actor = actor or self.actor(model)
                    env = self.environment(case)
                    report = evaluate_query_trajectory(env, actor, scenario=case["scenario"],
                        max_decisions=int(self.config.evaluation.max_decisions),
                        max_wall_seconds=float(self.config.evaluation.max_wall_seconds), amp_dtype="fp32")
                self.cache[key] = {**report, "cache_origin": reused or model["model_id"]}
            row = result_row(model, case, report, optimum=case["certified_C_exit"], phase=phase)
            row["reused_from"] = reused
            self.rows.append(row); output.append(row)
            self.save_rows()
        print(json.dumps({"模型": model["model_id"], "阶段": phase, "集合": split, **summarize_rows(output)}, ensure_ascii=False), flush=True)
        return output

    def baseline(self) -> None:
        self.validate_plan()
        for model in self.plan["models"]:
            self.evaluate(model, "train", phase="A")
            self.evaluate(model, "val", phase="A")

    def rerun(self) -> None:
        from apal.training.work3.behavior_cloning import run_behavior_cloning
        self.validate_plan()
        for seed in self.config.runtime.seeds:
            output = self.context.checkpoint_dir / f"rerun_seed{seed}.pt"
            report_path = self.context.reports_dir / f"rerun_seed{seed}.json"
            if report_path.exists():
                continue
            old = read_json(self.source / "artifacts/reports" / f"bc_seed{seed}.json")
            config = OmegaConf.create(old["resolved_config"])
            config.bc.diagnostic_checkpoint_dir = str(self.context.checkpoint_dir / "diagnostic_tmp" / f"seed{seed}")
            (self.context.configs_dir / f"rerun_seed{seed}.yaml").write_text(OmegaConf.to_yaml(config), encoding="utf-8")
            run_behavior_cloning(config, output=output, report=report_path)
            current = read_json(report_path)
            equality = {"seed": int(seed), "old_updates": old["optimization_steps"], "new_updates": current["optimization_steps"],
                "validation_checks_exact": old["validation_checks"] == current["validation_checks"],
                "initial_nll_exact": old["initial_validation_nll"] == current["initial_validation_nll"],
                "old_best_update": old["best_update"], "new_best_update": current["best_update"]}
            for role, old_path, new_path in (("best", self.source / "checkpoints" / f"bc_seed{seed}.pt", output),
                    ("final", self.source / "checkpoints" / f"bc_seed{seed}_final.pt", output.with_name(output.stem + "_final.pt"))):
                equality[f"{role}_parameters_exact"] = parameter_digest(old_path) == parameter_digest(new_path)
            write_json(self.context.reports_dir / f"reproduction_seed{seed}.json", equality)
            print(json.dumps(equality, ensure_ascii=False), flush=True)

    def curve(self) -> None:
        from apal.evaluation.bc_prefix_diagnostics import teacher_metrics
        from apal.training.work3.behavior_cloning import load_demonstrations
        self.validate_plan()
        points = []
        path = self.context.reports_dir / "curves.json"
        existing = {row["model_id"]: row for row in read_json(path)} if path.exists() else {}
        for seed in self.config.runtime.seeds:
            report = read_json(self.context.reports_dir / f"rerun_seed{seed}.json")
            for saved in report["diagnostic_checkpoints"]:
                step = saved["update"]
                model = {"model_id": f"rerun_seed{seed}_step{step}", "seed": int(seed), "update": step,
                    "path": saved["path"], "sha256": sha256(Path(saved["path"])),
                    "parameters_sha256": parameter_digest(Path(saved["path"])), "rule": "diagnostic_interval"}
                if model["model_id"] in existing:
                    points.append(existing[model["model_id"]]); continue
                train_rows = self.evaluate(model, "train", phase="B", panel=True)
                val_rows = self.evaluate(model, "val", phase="B")
                actor = self.actor(model)
                metrics = {}
                for split, filename in (("train", "train_set_records.pt"), ("val", "validation_set_records.pt")):
                    samples = load_demonstrations(self.source / "data" / filename, actor,
                        split="train" if split == "train" else "validation", team_objective="set")
                    samples = [row for row in samples if row["label_scope"] and (split == "val" or row["instance_id"] in self.plan["train_panel"])]
                    metrics[split] = teacher_metrics(actor, samples, batch_size=4)
                previous = max(0, step - int(self.config.training.diagnostic_interval))
                history = [row["loss"] for row in report["history"] if previous < row["update"] <= step]
                point = {**model, "validation_nll": saved["validation_nll"],
                    "training_sampled_nll_recent": statistics.mean(history) if history else None,
                    "train": summarize_rows(train_rows), "val": summarize_rows(val_rows), "teacher": metrics,
                    "trajectory_source": "本轮同设置重跑的新保存点，不伪称旧历史检查点"}
                points.append(point); write_json(path, points)
        choices = {}
        for seed in self.config.runtime.seeds:
            selected = [{**row, **row["val"]} for row in points if row["seed"] == seed]
            choices[str(seed)] = {rule: select_checkpoint(selected, rule)["model_id"] for rule in ("validation_nll", "closure")}
        write_json(self.context.reports_dir / "selection_comparison.json", choices)

    def autonomous_trace(self, model: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
        """旁路记录原评估器；不重写解码，不额外消耗随机数。"""
        actor, env = self.actor(model), self.environment(case)
        trace, events = [], []
        original_select, original_pop, original_workers = actor.select_snapshot, env.event_queue.pop, actor._worker_logits
        worker_logits = []
        retained = ("action_type", "candidate_task_keys", "candidate_branch_masks", "query_task_features",
            "query_branch_features", "query_worker_features", "query_alignment_features", "worker_prefixes",
            "worker_valid_masks", "station_worker_ids", "chosen_worker_indices", "chosen_team", "align",
            "advance_available", "prob_advance", "prob_stay", "branch", "query_program_hashes")

        def observed_select(snapshot: Any, *args: Any, **kwargs: Any) -> Any:
            worker_logits.clear()
            result = original_select(snapshot, *args, **kwargs)
            action, log_prob, value, record = result
            row = {"decision": len(trace), "time": snapshot.current_time, "cycle": snapshot.cycle_id,
                "action": json_facts(action), "log_prob": log_prob, "value": value,
                "record": {key: json_facts(record[key]) for key in retained if key in record}}
            row["worker_logits_before_mask"] = list(worker_logits)
            device = next(actor.parameters()).device
            if record.get("query_worker_features"):
                row["worker_query_logit_terms"] = [actor.query_worker_score(features.to(device)).squeeze(-1).cpu().tolist()
                    for features in record["query_worker_features"]]
            if "task_idx" in record and int(action["branch"]) == 0:
                table = snapshot.query_tables[int(record["task_idx"])]
                row["complete_team_windows"] = [[asdict(condition) for condition in pair] for pair in table.entries]
                row["commitments"] = {key: {"status": task.status.name, "team": list(task.assigned_team),
                    "station": task.current_station, "start": task.scheduled_start,
                    "duration": task.execution_duration, "align": task.alignment}
                    for key, task in env.state.tasks.items() if task.assigned_team}
                row["worker_calendars"] = {str(key): [list(interval) for interval in worker.intervals]
                    for key, worker in env.state.workers.items() if worker.intervals}
            trace.append(row)
            return result

        def observed_workers(*args: Any, **kwargs: Any) -> Any:
            result = original_workers(*args, **kwargs)
            worker_logits.append(result.detach().cpu().tolist())
            return result

        def observed_pop() -> Any:
            event = original_pop()
            if event is not None:
                events.append({"time": event.timestamp, "priority": event.priority, "event_id": event.event_id,
                    "type": event.event_type.name, "task_key": event.task_key, "generation": event.generation})
            return event

        try:
            actor.select_snapshot, env.event_queue.pop = observed_select, observed_pop
            actor._worker_logits = observed_workers
            report = evaluate_query_trajectory(env, actor, scenario=case["scenario"],
                max_decisions=int(self.config.evaluation.max_decisions),
                max_wall_seconds=float(self.config.evaluation.max_wall_seconds), amp_dtype="fp32")
        finally:
            actor.select_snapshot, env.event_queue.pop = original_select, original_pop
            actor._worker_logits = original_workers
        expected = self.cache[f"{model['parameters_sha256']}:{case['split']}:{case['instance_id']}"]
        keys = ("success", "terminated", "truncated", "termination_reason", "decisions", "makespan", "C_task",
            "completed_tasks", "total_tasks", "independent_feasible", "constraint_violations", "cost_ledger_matches", "objective")
        differences = [key for key in keys if report[key] != expected[key]]
        if differences:
            raise AssertionError(f"旁路记录改变原自主排程结果: {differences}")
        final_tasks = {key: {"status": task.status.name, "station": task.current_station,
            "team": list(task.assigned_team), "align": task.alignment, "scheduled_start": task.scheduled_start,
            "duration": task.execution_duration, "actual_start": task.actual_start, "actual_end": task.actual_end}
            for key, task in env.state.tasks.items()}
        return {"model": model, "report": report, "trace": trace, "events": events, "final_tasks": final_tasks,
            "baseline_scientific_fields_exact": True, "comparison_keys": keys}

    def prefixes(self) -> None:
        from apal.data.work3.certified_examples import load_witness_actions
        from apal.evaluation.bc_prefix_diagnostics import score_prefix_decision
        self.validate_plan()
        selection_report = read_json(self.context.reports_dir / "representatives_selection.json")
        selected = [row for family in selection_report["families"] for row in family["selected"]]
        for selection in selected:
            split, instance_id, seed = selection["split"], selection["instance_id"], selection["seed"]
            case = next(row for row in self.plan["cases"][split] if row["instance_id"] == instance_id)
            path = self.context.traces_dir / f"prefix_{seed}_{instance_id}.json"
            if path.exists():
                continue
            models = [model for model in self.plan["models"] if model["seed"] == seed]
            actors = {model["model_id"]: self.actor(model) for model in models}
            env = self.environment(case)
            env.reset()
            if case["scenario"]:
                env.load_scenario(case["scenario"])
            common = []
            for index, teacher in enumerate(load_witness_actions(Path(case["baseline"]).parent.parent)):
                snapshot = next(iter(actors.values())).make_decision_snapshot(env, torch.zeros(32), torch.zeros(2))
                common.append({"decision": index, "time": snapshot.current_time,
                    "teacher_action": json_facts(teacher["action"]),
                    "models": {name: score_prefix_decision(actor, snapshot, teacher["action"],
                        teacher["label_scope"]) for name, actor in actors.items()}})
                env.step(teacher["action"])
            if not env._check_terminated() or abs(env.state.current_time - case["certified_C_exit"]) > 1e-9:
                raise AssertionError("共同教师前缀未到达已核证下界")
            autonomous = {model["model_id"]: self.autonomous_trace(model, case)
                for model in models if model["rule"] != "final"}
            write_json(path, json_facts({"selection": selection, "common_teacher_prefix": common,
                "autonomous": autonomous, "teacher_C_exit": env.state.current_time,
                "interpretation": "最早不同不等于错误；本文件不声称穷尽多最优动作或条件最优后缀"}))
            print(json.dumps({"代表": instance_id, "seed": seed, "教师步数": len(common), "文件": str(path)}, ensure_ascii=False), flush=True)

    def branches(self) -> None:
        """最多两个真实前缀，仅替换完整团队后由同一原策略续排。"""
        self.validate_plan()
        pairs = []
        for path in sorted(self.context.traces_dir.glob("prefix_*.json")):
            if len(pairs) >= 2:
                break
            trace = read_json(path)
            selection = trace["selection"]
            if selection["family"] != "structured":
                continue
            trajectory = next(row for name, row in trace["autonomous"].items() if "nll_best" in name)
            model = trajectory["model"]
            case = next(row for row in self.plan["cases"][selection["split"]]
                        if row["instance_id"] == selection["instance_id"])
            for row in trajectory["trace"]:
                if not row.get("complete_team_windows"):
                    continue
                action = row["action"]
                conditions = [pair[action["align"]] for pair in row["complete_team_windows"]]
                chosen = next(condition for condition in conditions if set(condition["team"]) == set(action["team"]))
                alternative = min(conditions, key=lambda condition: (condition["end"], condition["team"]))
                if alternative["end"] >= chosen["end"] - 1e-9 or alternative["duration"] >= chosen["duration"] - 1e-9:
                    continue
                outcomes = {}
                for name, selected_action in (("original", action), ("alternative_team", {**action, "team": alternative["team"]})):
                    env, actor = self.environment(case), self.actor(model)
                    env.reset()
                    if case["scenario"]:
                        env.load_scenario(case["scenario"])
                    for previous in trajectory["trace"][:row["decision"]]:
                        env.step(previous["action"])
                    before_time, before_cost = env.state.current_time, env.cumulative_cost
                    env.step(selected_action)
                    report = evaluate_query_trajectory(env, actor, reset=False,
                        max_decisions=int(self.config.evaluation.max_decisions),
                        max_wall_seconds=float(self.config.evaluation.max_wall_seconds), amp_dtype="fp32")
                    outcomes[name] = {"action": selected_action, "prefix_time": before_time,
                        "prefix_cost": before_cost, "report": report}
                if outcomes["original"]["report"]["makespan"] != trajectory["report"]["makespan"]:
                    raise AssertionError("同前缀原动作续排未复现原工期")
                pairs.append({"instance_id": case["instance_id"], "seed": model["seed"], "decision": row["decision"],
                    "task_key": action["task_key"], "chosen_query": chosen, "alternative_query": alternative,
                    "outcomes": outcomes, "interpretation": "仅在同一后续策略下的条件对照，不证明全局坏动作或新最优标签"})
                break
        write_json(self.context.reports_dir / "counterfactuals.json", pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="BC训练实例、学习轨迹与闭环选模诊断")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "conf/work3/bc_diagnosis.yaml")
    parser.add_argument("--phase", choices=("prepare", "baseline", "rerun", "curve", "prefixes", "branches"), required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set))
    diagnosis = BCDiagnosis(config)
    started = time.monotonic()
    succeeded = False
    try:
        {"prepare": diagnosis.prepare, "baseline": diagnosis.baseline, "rerun": diagnosis.rerun,
         "curve": diagnosis.curve, "prefixes": diagnosis.prefixes, "branches": diagnosis.branches}[args.phase]()
        succeeded = True
    finally:
        receipt = {"phase": args.phase, "argv": sys.argv, "python": sys.executable,
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "finished_epoch": time.time(), "elapsed_seconds": time.monotonic() - started, "success": succeeded}
        with (diagnosis.context.reports_dir / "executions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
