"""人数采样的单因素BC对照；原训练与离线评估入口保持独立。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import time
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import WeightedRandomSampler

from apal.evaluation.bc_diagnostics import (
    BCDiagnosis, json_facts, parameter_digest, read_json, sha256, summarize_rows, write_json,
)
from apal.models.work3.actor_critic import ActorCriticWork3
from apal.project import PROJECT_ROOT
from apal.runtime.artifacts import resolve_path
from apal.training.work3.behavior_cloning import (
    _batch_coverage, _demand_sampling_weights, _sampling_audit,
    load_demonstrations, run_behavior_cloning,
)


def verify_sampling_sequence(report: dict[str, Any], samples: list[dict[str, Any]],
                             config: DictConfig) -> dict[str, Any]:
    """独立重建每epoch含尾批的索引，与训练实际batch逐项精确核对。"""
    assert samples and report["sample_index_audit_complete"]
    weights = _demand_sampling_weights(samples, config.bc.demand_sampling_weights)
    assert weights is not None
    sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True,
        generator=torch.Generator().manual_seed(int(config.runtime.seed)))
    batch_size = int(config.bc.batch_size)
    remaining: list[int] = []
    hashes, demands, choices, scopes = [], Counter(), Counter(), Counter()
    presented, target_count, epochs, tails = 0, 0, 0, 0
    for update in report["history"]:
        if not remaining:
            remaining = list(sampler)
            epochs += 1
        indices, remaining = remaining[:batch_size], remaining[batch_size:]
        assert update["sample_index_audit_complete"] and all(type(index) is int for index in update["sample_indices"])
        assert update["sample_indices"] == indices, f"实际采样索引不一致，更新{update['update']}"
        batch = [samples[index] for index in indices]
        audit = _sampling_audit(batch, indices)
        assert all(update[key] == value for key, value in audit.items()), "实际batch身份或监督次数不一致"
        actual_demands, actual_choices = _batch_coverage(batch)
        assert update["demand_counts"] == actual_demands and update["real_choice_counts"] == actual_choices
        assert update["sample_count"] == len(batch)
        hashes.append(audit["sample_sequence_sha256"])
        demands.update(actual_demands); choices.update(actual_choices); scopes.update(audit["scope_counts"])
        presented += len(batch)
        target_count += audit["train_024_0_2_count"]
        tails += int(len(batch) < batch_size)
    digest = hashlib.sha256("\n".join(hashes).encode("ascii")).hexdigest()
    assert digest == report["sampled_batch_sequence_sha256"]
    assert presented == report["sampled_decisions"] and target_count == report["train_024_0_2_count"]
    assert dict(demands) == report["sampled_team_demand_counts"]
    assert dict(choices) == report["sampled_real_choice_counts"] and dict(scopes) == report["sampled_scope_counts"]
    return {"exact_batch_indices": True, "exact_batch_identity_hashes": True, "updates": len(hashes),
        "presented_records": presented, "epochs_started": epochs, "tail_batches": tails,
        "sequence_sha256": digest, "team_demand_counts": dict(demands), "scope_counts": dict(scopes),
        "real_choice_counts": dict(choices), "train_024_0_2_count": target_count}


def _initial_model(diagnosis: BCDiagnosis, seed: int) -> dict[str, Any]:
    matches = [model for model in diagnosis.plan["models"] if model["seed"] == seed and model["rule"] == "initial"]
    if len(matches) != 1:
        raise ValueError("冻结计划必须为每个种子提供唯一W0")
    model = matches[0]
    if sha256(Path(model["path"])) != model["sha256"]:
        raise ValueError("冻结W0权重身份改变")
    return model


def _records(diagnosis: BCDiagnosis, phase: str) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    source = diagnosis.source / "data" / "train_set_records.pt"
    training = source
    if phase == "single":
        previous = resolve_path(diagnosis.config.previous_diagnosis, PROJECT_ROOT)
        training = previous / "artifacts" / "single_instance_records.pt"
        payload = torch.load(training, map_location="cpu", weights_only=False)
        if (payload.get("subset_source_sha256") != sha256(source)
                or payload.get("subset_instance_id") != diagnosis.config.experiment.single_instance
                or {row["instance_id"] for row in payload["records"]} != {diagnosis.config.experiment.single_instance}):
            raise ValueError("既有单例子集的来源或实例身份不一致")
    actor = ActorCriticWork3(condition_queries=True)
    train = load_demonstrations(training, actor, split="train", team_objective="set")
    validation = load_demonstrations(diagnosis.source / "data" / "validation_set_records.pt", actor,
                                   split="validation", team_objective="set")
    train = [row for row in train if row["label_scope"]]
    validation = [row for row in validation if row["label_scope"]]
    if not train or len({id(row) for row in train}) != len(train):
        raise ValueError("真实训练记录必须非空且各数据行对象唯一，才能精确审计")
    return training, train, validation


def _training_config(diagnosis: BCDiagnosis, phase: str, seed: int, condition: str,
                     training: Path) -> DictConfig:
    config = OmegaConf.create(read_json(diagnosis.source / "artifacts" / "reports" / f"bc_seed{seed}.json")["resolved_config"])
    frozen = diagnosis.config
    if config.runtime.seed != seed or Path(config.paths.initial_checkpoint).resolve() != Path(_initial_model(diagnosis, seed)["path"]).resolve():
        raise ValueError("原训练配置与该种子的W0不一致")
    for key in ("team_objective", "hidden_dim", "max_station_workers", "batch_size", "learning_rate", "max_grad_norm", "validate_every"):
        if config.bc[key] != frozen.bc[key]:
            raise ValueError(f"本轮不能改变BC训练字段：{key}")
    for key in ("device", "amp_dtype", "deterministic", "main_num_threads"):
        if config.runtime[key] != frozen.runtime[key]:
            raise ValueError(f"本轮不能改变运行字段：{key}")
    if json_facts(OmegaConf.to_container(config.objective)) != json_facts(OmegaConf.to_container(frozen.objective)):
        raise ValueError("本轮不能改变训练目标")
    config.paths.train_records = str(training)
    config.bc.max_updates = int(frozen.bc.max_updates)
    config.bc.patience = int(frozen.bc.patience)
    config.runtime.max_wall_seconds = float(frozen.bc.max_wall_seconds)
    config.bc.demand_sampling_weights = frozen.experiment.conditions[condition]
    config.bc.diagnostic_checkpoint_dir = str(diagnosis.context.checkpoint_dir / "diagnostic_tmp" / f"{phase}_seed{seed}_{condition}")
    return config


def _validate_training(report: dict[str, Any], config: DictConfig, initial: dict[str, Any],
                       samples: list[dict[str, Any]]) -> dict[str, Any]:
    expected = int(config.bc.max_updates)
    assert report["optimization_steps"] == report["optimizer_step_attempts"] == expected == 200
    assert report["amp_skipped_steps"] == 0 and report["termination_reason"] == "max_updates"
    assert [row["update"] for row in report["validation_checks"]] == list(range(25, 201, 25))
    assert len(report["history"]) == expected and report["parameters_changed"] > 0
    assert all(math.isfinite(row["loss"]) and row["grad_norm"] is not None and math.isfinite(row["grad_norm"])
               for row in report["history"])
    assert all(math.isfinite(row["validation_nll"]) for row in report["validation_checks"])
    assert parameter_digest(Path(report["initial_checkpoint"])) == initial["parameters_sha256"]
    assert json_facts(report["resolved_config"]) == json_facts(OmegaConf.to_container(config, resolve=True))
    final = torch.load(report["final_checkpoint"], map_location="cpu", weights_only=False)
    state = final["bc_training_state"]
    assert state["optimization_steps"] == state["optimizer_step_attempts"] == expected and state["amp_skipped_steps"] == 0
    assert state["optimizer_state"] and all(torch.isfinite(value).all() for value in final["actor_critic_state"].values())
    return verify_sampling_sequence(report, samples, config)


def _checkpoint_model(diagnosis: BCDiagnosis, report: dict[str, Any], phase: str,
                      seed: int, condition: str, update: int) -> tuple[dict[str, Any], float]:
    if update == 0:
        path = Path(_initial_model(diagnosis, seed)["path"])
        nll = report["initial_validation_nll"]
    elif update == int(diagnosis.config.experiment.primary_update):
        path = Path(report["final_checkpoint"])
        nll = report["final_validation_nll"]
    else:
        saved = next(row for row in report["diagnostic_checkpoints"] if row["update"] == update)
        path, nll = Path(saved["path"]), saved["validation_nll"]
    return {"model_id": f"{phase}_seed{seed}_{condition}_step{update}", "seed": seed, "update": update,
            "rule": condition, "path": str(path), "sha256": sha256(path),
            "parameters_sha256": parameter_digest(path)}, float(nll)


def _curve_point(diagnosis: BCDiagnosis, model: dict[str, Any], phase: str,
                 train: list[dict[str, Any]], validation: list[dict[str, Any]], nll: float) -> dict[str, Any]:
    from apal.evaluation.bc_prefix_diagnostics import critical_worker_distribution, teacher_metrics

    single = phase == "single"
    instance_id = str(diagnosis.config.experiment.single_instance)
    train_rows = diagnosis.evaluate(model, "train", phase=phase,
        panel=not single, instance_ids=(instance_id,) if single else None)
    val_rows = diagnosis.evaluate(model, "val", phase=phase) if not single else []
    full_rows = None
    if not single and model["update"] == int(diagnosis.config.experiment.primary_update):
        full_rows = diagnosis.evaluate(model, "train", phase=phase)
    actor = diagnosis.actor(model)
    panel = {instance_id} if single else set(diagnosis.plan["train_panel"])
    metrics = {"train": teacher_metrics(actor, [row for row in train if row["instance_id"] in panel], batch_size=4)}
    if not single:
        metrics["val"] = teacher_metrics(actor, validation, batch_size=4)
    critical = [row for row in train if row["instance_id"] == instance_id and row["record"].get("task_key") == "0_2"]
    if len(critical) != 1:
        raise ValueError("关键教师现场必须按实例和工序唯一定位")
    critical_sample = dict(critical[0])
    context_path = diagnosis.context.reports_dir / "critical_context.json"
    if context_path.exists():
        context = read_json(context_path)
        if context["task_key"] != critical_sample["record"]["task_key"]:
            raise ValueError("关键现场的时间单位上下文不属于该工序")
        critical_sample.update(h0=context["h0"], current_time=context["current_time"])
    point = {**model, "phase": phase, "condition": model["rule"], "validation_nll": nll,
             "train": summarize_rows(train_rows), "train_instance_ids": [row["instance_id"] for row in train_rows],
             "teacher": metrics, "critical_worker_distribution": critical_worker_distribution(actor, critical_sample)}
    if val_rows:
        point["val"] = summarize_rows(val_rows)
    if full_rows is not None:
        point["train_full"] = summarize_rows(full_rows)
    return json_facts(point)


def _expansion_decision(diagnosis: BCDiagnosis) -> dict[str, Any]:
    instance = str(diagnosis.config.experiment.single_instance)
    seed, step = int(diagnosis.config.experiment.single_seed), int(diagnosis.config.experiment.primary_update)
    finals = {}
    for condition in diagnosis.config.experiment.conditions:
        model_id = f"single_seed{seed}_{condition}_step{step}"
        rows = [row for row in diagnosis.rows if row["phase"] == "single" and row["model_id"] == model_id
                and row["split"] == "train" and row["instance_id"] == instance]
        if len(rows) != 1:
            raise ValueError("两臂200步的单例合法终局尚不完整，不能做扩展决定")
        finals[condition] = rows[0]
    tolerance = float(diagnosis.config.experiment.gap_absolute_tolerance)
    legal = all(row["success"] and row["C_exit"] is not None and math.isfinite(row["C_exit"]) for row in finals.values())
    difference = finals["original"]["C_exit"] - finals["all1"]["C_exit"] if legal else None
    return {"expand": bool(legal and difference > tolerance), "phase": "single", "seed": seed,
        "instance_id": instance, "update": step, "rule": diagnosis.config.experiment.expansion_rule,
        "absolute_tolerance": tolerance, "original_minus_all1_C_exit": difference, "outcomes": finals,
        "reason": "双方合法完整且all1的C_exit严格低于original减预定容差" if legal and difference > tolerance
                  else "未满足预定合法终局与严格改善门槛，停止扩大"}


def run_phase(config: DictConfig, phase: str) -> None:
    # 冻结计划以JSON保存；需求人数键须与其中的字符串键作同语义比较。
    config = OmegaConf.create(json_facts(OmegaConf.to_container(config, resolve=True)))
    diagnosis = BCDiagnosis(config)
    diagnosis.validate_plan()
    decision_path = diagnosis.context.reports_dir / "expansion_decision.json"
    if phase == "full":
        decision = read_json(decision_path)
        if decision.get("expand") is not True or decision.get("rule") != config.experiment.expansion_rule:
            raise ValueError("单例未通过冻结扩展门槛，不运行全训练集两种子对照")
    training_path, samples, validation = _records(diagnosis, phase)
    curves_path = diagnosis.context.reports_dir / "sampling_curves.json"
    curves = read_json(curves_path) if curves_path.exists() else []
    seeds = [int(config.experiment.single_seed)] if phase == "single" else [int(seed) for seed in config.runtime.seeds]
    for seed in seeds:
        initial = _initial_model(diagnosis, seed)
        for condition in config.experiment.conditions:
            if time.time() - diagnosis.plan["created_epoch"] >= config.runtime.total_wall_seconds:
                raise TimeoutError("本轮总预算已到，保留已有采样与曲线结果")
            trial = f"{phase}_seed{seed}_{condition}"
            effective = _training_config(diagnosis, phase, seed, condition, training_path)
            output, report_path = diagnosis.context.checkpoint_dir / f"{trial}.pt", diagnosis.context.reports_dir / f"{trial}.json"
            resolved_path = diagnosis.context.configs_dir / f"{trial}.yaml"
            if resolved_path.exists() and json_facts(OmegaConf.to_container(OmegaConf.load(resolved_path))) != json_facts(OmegaConf.to_container(effective)):
                raise ValueError("既有试验配置与本次不一致")
            if not report_path.exists():
                resolved_path.write_text(OmegaConf.to_yaml(effective), encoding="utf-8")
                run_behavior_cloning(effective, output=output, report=report_path)
            report = read_json(report_path)
            audit = _validate_training(report, effective, initial, samples)
            write_json(diagnosis.context.reports_dir / f"{trial}_sampling_audit.json", audit)
            for update in config.experiment.checkpoints:
                model, nll = _checkpoint_model(diagnosis, report, phase, seed, condition, int(update))
                old = [point for point in curves if point["model_id"] == model["model_id"]]
                if old:
                    if len(old) != 1 or old[0]["sha256"] != model["sha256"]:
                        raise ValueError("已保存曲线的权重身份不匹配")
                    continue
                point = _curve_point(diagnosis, model, phase, samples, validation, nll)
                point["training_report"] = str(report_path)
                point["sampling_audit"] = str(diagnosis.context.reports_dir / f"{trial}_sampling_audit.json")
                curves.append(point)
                write_json(curves_path, curves)
                print(f"已保存 {model['model_id']} 离线曲线点", flush=True)
    if phase == "single":
        decision = _expansion_decision(diagnosis)
        write_json(decision_path, decision)
        print(f"单例扩展决定：{decision['expand']}；原工期减all1工期={decision['original_minus_all1_C_exit']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="固定原BC轨迹设置的人数采样单因素对照")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "conf" / "work3" / "bc_sampling.yaml")
    parser.add_argument("--phase", choices=("single", "full"), required=True)
    args = parser.parse_args()
    run_phase(OmegaConf.load(args.config), args.phase)


if __name__ == "__main__":
    main()
