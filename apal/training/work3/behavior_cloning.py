"""在原QUERY网络和Lightning优化生命周期上进行限定监督层的行为克隆。"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from lightning.pytorch import Callback, Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler

from apal.models.work3.actor_critic import ActorCriticWork3
from apal.models.work3.ppo_trainer import PPOTrainerWork3
from apal.project import PROJECT_ROOT
from apal.runtime.artifacts import create_run_context, resolve_path
from apal.runtime.initial_checkpoint_selection import sha256_file
from apal.training.work3.lightning import Work3LightningModule
from apal.training.work3.runtime_config import seed_work3_runtime


BC_RECORD_VERSION = "work3_bc_records_v1"
BC_RECORD_SEMANTICS_VERSION = "work3_autoregressive_layers_v1"
_LAYERS = frozenset({"advance", "task", "branch", "team", "align"})


def load_demonstrations(path: Path, actor: ActorCriticWork3, *, split: str,
                        team_objective: str = "sequence") -> list[dict[str, Any]]:
    """仅读取当前契约的CPU原始条件记录，不在训练时访问现场或重做查询。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("version") != BC_RECORD_VERSION:
        raise ValueError("行为克隆记录文件版本无效")
    if payload.get("record_semantics_version") != BC_RECORD_SEMANTICS_VERSION or payload.get("split") != split:
        raise ValueError("行为克隆记录语义版本或数据划分不匹配")
    if payload.get("policy_input_semantics") != actor.input_semantics():
        raise ValueError("行为克隆记录输入语义不匹配")
    if team_objective not in ("set", "sequence"):
        raise ValueError("团队监督只支持集合或教师序列")
    if team_objective == "set" and payload.get("team_objective") != "set":
        raise ValueError("团队集合监督要求显式集合记录及子集缓存")
    samples = payload.get("records")
    if not isinstance(samples, list) or not samples:
        raise ValueError("行为克隆记录必须为非空列表")
    for sample in samples:
        if not isinstance(sample, dict) or not all(isinstance(sample.get(key), str) and sample[key]
                                                 for key in ("instance_id", "parent_group")):
            raise ValueError("行为克隆记录缺少实例或母实例身份")
        for name, dimension in (("state_features", actor.state_dim), ("time_features", 2)):
            value = sample.get(name)
            if (not isinstance(value, torch.Tensor) or value.shape != (dimension,)
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError(f"行为克隆{name}形状或数值无效")
        for name in ("label_scope", "choice_layers"):
            scope = sample.get(name)
            if not isinstance(scope, (tuple, list)) or any(key not in _LAYERS for key in scope) or len(set(scope)) != len(scope):
                raise ValueError(f"行为克隆{name}必须列出不重复的受支持动作层")
        record = sample.get("record")
        if not isinstance(record, dict) or record.get("graph_snapshot") is None:
            raise ValueError("行为克隆记录缺少原始图快照")
        actor._validate_condition_record(record)
        if (team_objective == "set" and record.get("action_type") == "schedule"
                and int(record["branch"]) == 0 and "team_set_cache" not in record):
            raise ValueError("团队集合监督缺少子集缓存")
        if record.get("action_type") == "forced_advance" and sample["label_scope"]:
            raise ValueError("强制推进没有可监督的策略选择")
    return samples


def behavior_cloning_loss(
    components: list[dict[str, torch.Tensor]], samples: list[dict[str, Any]],
    instance_counts: dict[str, int], *, team_objective: str = "sequence",
) -> tuple[torch.Tensor, dict[str, float]]:
    """每决定平均受监督层；均匀采样时实例等权，需求重采样时保留实例长度控制。"""
    if (team_objective not in ("set", "sequence") or len(components) != len(samples)
            or not samples or not instance_counts or min(instance_counts.values()) < 1):
        raise ValueError("行为克隆组件、样本和实例计数无效")
    total = sum(instance_counts.values())
    terms, layer_values = [], {}
    for parts, sample in zip(components, samples):
        scope = sample["label_scope"]
        keys = tuple("team_set" if layer == "team" and team_objective == "set" else layer for layer in scope)
        if not scope or len(set(scope)) != len(scope) or any(layer not in _LAYERS for layer in scope) or any(key not in parts for key in keys):
            raise ValueError("行为克隆监督层在该实际动作中不存在")
        # 各已证层标量对数概率 -> 单决定标量损失；PPO序列概率保持独立。
        loss = -torch.stack([parts[key] for key in keys]).mean()
        count = instance_counts.get(sample["instance_id"], 0)
        if count < 1:
            raise ValueError("行为克隆样本的实例计数缺失")
        terms.append(loss * (total / (len(instance_counts) * count)))
        for key in keys:
            layer_values.setdefault(key, []).append(float(-parts[key].detach().item()))
    return torch.stack(terms).mean(), {key: sum(value) / len(value) for key, value in layer_values.items()}


def _replay(actor: ActorCriticWork3, samples: list[dict[str, Any]], *,
            team_objective: str = "sequence") -> list[dict[str, torch.Tensor]]:
    # 原始事实列表 -> [B,32]、[B,2]；神经表示始终由当前权重重算。
    _, _, _, components = actor.evaluate_action_log_probs(
        torch.stack([sample["state_features"] for sample in samples]),
        torch.stack([sample["time_features"] for sample in samples]),
        [sample["record"] for sample in samples], return_components=True,
        team_set_supervision=team_objective == "set",
    )
    return components


def _counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(sample["instance_id"] for sample in samples))


def _demand_sampling_weights(samples: list[dict[str, Any]], configured: Mapping | None) -> torch.Tensor | None:
    """只校准受团队监督的记录；其他动作仍保留权重1。"""
    if configured is None:
        return None
    if not isinstance(configured, Mapping) or not configured:
        raise ValueError("需求采样权重必须为非空映射")
    by_demand = {}
    for raw_demand, raw_weight in configured.items():
        try:
            demand, weight = int(raw_demand), float(raw_weight)
        except (TypeError, ValueError) as error:
            raise ValueError("需求编号必须为正整数，采样权重必须为有限正数") from error
        if isinstance(raw_demand, bool) or demand < 1 or str(demand) != str(raw_demand):
            raise ValueError("需求编号必须为正整数")
        if isinstance(raw_weight, bool) or not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("需求采样权重必须为有限正数")
        by_demand[demand] = weight
    values = []
    for row in samples:
        demand = len(row["record"].get("chosen_team", ())) if "team" in row["label_scope"] else None
        if demand == 0:
            raise ValueError("团队监督记录缺少实际团队")
        values.append(by_demand.get(demand, 1.0))
    return torch.tensor(values, dtype=torch.double)


def _batch_coverage(samples: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    demands = Counter(str(len(row["record"]["chosen_team"])) for row in samples if "team" in row["label_scope"])
    choices = Counter(layer for row in samples for layer in sorted(set(row["label_scope"]) & set(row["choice_layers"])))
    return dict(demands), dict(choices)


def _sampling_audit(samples: list[dict[str, Any]], sample_indices: list[int | None]) -> dict[str, Any]:
    """按实际batch顺序审计动作记录身份，不读取对象地址或神经表示。"""
    assert len(samples) == len(sample_indices)
    identities = [{"sample_index": index, "instance_id": row["instance_id"], **{key: row["record"].get(key)
                   for key in ("task_key", "branch", "chosen_team", "align")}}
                  for index, row in zip(sample_indices, samples)]
    encoded = json.dumps(identities, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    scopes = Counter(layer for row in samples for layer in row["label_scope"])
    return {"sample_sequence_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "sample_indices": list(sample_indices),
            "sample_index_audit_complete": all(type(index) is int for index in sample_indices),
            "scope_counts": dict(scopes),
            # 该固定样例只做呈现审计，不参与损失或抽样。
            "train_024_0_2_count": sum(row["instance_id"] == "train_024" and row["task_key"] == "0_2"
                                        for row in identities)}


def _evaluate(actor: ActorCriticWork3, samples: list[dict[str, Any]], batch_size: int,
              team_objective: str) -> float:
    training = actor.training
    actor.eval()
    total_loss = 0.0
    try:
        with torch.no_grad(), torch.amp.autocast(device_type=next(actor.parameters()).device.type, enabled=False):
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                loss, _ = behavior_cloning_loss(_replay(actor, batch, team_objective=team_objective),
                    batch, _counts(samples), team_objective=team_objective)
                total_loss += float(loss.item()) * len(batch)
    finally:
        actor.train(training)
    result = total_loss / len(samples)
    if not math.isfinite(result):
        raise FloatingPointError("行为克隆验证损失非有限")
    return result


class Work3BCModule(Work3LightningModule):
    """复用现有唯一AdamW、梯度守卫和更新计数；不增加价值或时间监督。"""

    def __init__(self, actor: ActorCriticWork3, samples: list[dict[str, Any]], config: DictConfig) -> None:
        super().__init__(actor_critic=actor, learning_rate=float(config.learning_rate),
                         max_grad_norm=float(config.max_grad_norm), ppo_epochs=0,
                         batch_size=int(config.batch_size))
        self.instance_counts = _counts(samples)
        identity_counts = Counter(id(row) for row in samples)
        self.sample_index_audit_complete = len(identity_counts) == len(samples)
        self._sample_indices = {id(row): index if identity_counts[id(row)] == 1 else None
                                for index, row in enumerate(samples)}
        self.team_objective = str(config.get("team_objective", "sequence"))
        self.history: list[dict[str, Any]] = []

    def transfer_batch_to_device(self, batch: list[dict[str, Any]], device: torch.device,
                                 dataloader_idx: int) -> list[dict[str, Any]]:
        del device, dataloader_idx
        return batch

    def training_step(self, batch: list[dict[str, Any]], batch_idx: int) -> torch.Tensor:
        del batch_idx
        started = time.perf_counter()
        self.actor_critic.train()
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            loss, layers = behavior_cloning_loss(_replay(self.actor_critic, batch, team_objective=self.team_objective),
                batch, self.instance_counts, team_objective=self.team_objective)
        grad_norm = self._manual_update(loss, self.optimizers())
        demands, choices = _batch_coverage(batch)
        self.history.append({"update": self.optimization_steps, "loss": float(loss.detach().item()),
                             "layer_nll": layers, "grad_norm": grad_norm,
                             "sample_count": len(batch), "demand_counts": demands, "real_choice_counts": choices,
                             **_sampling_audit(batch, [self._sample_indices.get(id(row)) for row in batch]),
                             "elapsed_seconds": time.perf_counter() - started})
        return loss.detach()


def _save(module: Work3BCModule, path: Path, metadata: dict[str, Any], *, stage: str = "bc") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PPOTrainerWork3(module.actor_critic, device=next(module.actor_critic.parameters()).device,
                    create_optimizer=False).save_checkpoint(str(path), metadata={
        **metadata, "stage": stage, "method_profile": "QUERY", "checkpoint_evaluation_eligible": False,
        "research_result_eligible": False,
    })
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["bc_training_state"] = {"optimizer_state": module.optimizer_state_dict,
        "optimization_steps": module.optimization_steps, "optimizer_step_attempts": module.optimizer_step_attempts,
        "amp_skipped_steps": module.amp_skipped_steps}
    torch.save(payload, path)


class _Selection(Callback):
    def __init__(self, validation: list[dict[str, Any]], config: DictConfig, output: Path,
                 metadata: dict[str, Any], initial_loss: float) -> None:
        self.validation, self.config, self.output, self.metadata = validation, config, output, metadata
        self.best_loss, self.bad_checks = initial_loss, 0
        self.best_step = 0
        self.stop_reason = "max_updates"
        self.checks: list[dict[str, Any]] = []
        self.started = time.monotonic()
        diagnostic_dir = config.bc.get("diagnostic_checkpoint_dir")
        if diagnostic_dir is not None and (not isinstance(diagnostic_dir, str) or not diagnostic_dir.strip()):
            raise ValueError("诊断权重目录必须为非空路径或null")
        self.diagnostic_dir = None if diagnostic_dir is None else resolve_path(diagnostic_dir, PROJECT_ROOT)
        self.diagnostic_checkpoints: list[dict[str, Any]] = []
        self.diagnostic_save_seconds = 0.0
        if self.diagnostic_dir is not None:
            steps = sorted({0, *range(int(config.bc.validate_every), int(config.bc.max_updates),
                                     int(config.bc.validate_every)), int(config.bc.max_updates)})
            if len(steps) > 9:
                raise ValueError("诊断最多保存初始权重和八个验证间隔点")
            for step in steps:
                path = self.diagnostic_dir / f"bc_update_{step:06d}.pt"
                if path.exists():
                    raise FileExistsError(f"诊断权重已存在：{path}")

    def _save_diagnostic(self, module: Work3BCModule, step: int, validation_nll: float) -> None:
        if self.diagnostic_dir is None:
            return
        path = self.diagnostic_dir / f"bc_update_{step:06d}.pt"
        if path.exists():
            raise FileExistsError(f"诊断权重已存在：{path}")
        started = time.monotonic()
        _save(module, path, {**self.metadata, "diagnostic_update": step,
                             "diagnostic_validation_nll": validation_nll})
        elapsed = time.monotonic() - started
        self.diagnostic_save_seconds += elapsed
        self.diagnostic_checkpoints.append({"update": step, "validation_nll": validation_nll,
                                            "path": str(path.resolve())})

    def on_train_batch_end(self, trainer: Trainer, pl_module: Work3BCModule, outputs: Any,
                           batch: Any, batch_idx: int) -> None:
        del outputs, batch, batch_idx
        step = pl_module.optimization_steps
        if step % int(self.config.bc.validate_every) == 0 or step == int(self.config.bc.max_updates):
            loss = _evaluate(pl_module.actor_critic, self.validation, int(self.config.bc.batch_size), pl_module.team_objective)
            self.checks.append({"update": step, "validation_nll": loss})
            self._save_diagnostic(pl_module, step, loss)
            if loss < self.best_loss:
                self.best_loss, self.bad_checks, self.best_step = loss, 0, step
                _save(pl_module, self.output, {**self.metadata, "selection_validation_nll": loss})
            else:
                self.bad_checks += 1
            if self.bad_checks >= int(self.config.bc.patience):
                self.stop_reason = "validation_patience"
                trainer.should_stop = True
        if time.monotonic() - self.started >= float(self.config.runtime.max_wall_seconds):
            self.stop_reason = "time_budget"
            trainer.should_stop = True


def run_behavior_cloning(config: DictConfig, *, output: Path | None = None,
                         report: Path | None = None) -> dict[str, Any]:
    """训练入口；只从同QUERY初始权重加载，BC优化器从零建立。"""
    started = time.monotonic()
    if config.runtime.amp_dtype != "fp32":
        raise ValueError("本轮行为克隆仅支持FP32")
    for key in ("max_updates", "batch_size", "validate_every", "patience"):
        if type(config.bc[key]) is not int or config.bc[key] < 1:
            raise ValueError(f"行为克隆{key}必须为正整数")
    if not math.isfinite(float(config.runtime.max_wall_seconds)) or config.runtime.max_wall_seconds <= 0:
        raise ValueError("行为克隆时间预算必须为正有限数")
    if not all(isinstance(config.paths.get(key), str) and config.paths[key]
               for key in ("initial_checkpoint", "train_records", "validation_records")):
        raise ValueError("行为克隆要求明确的初始权重、训练和验证记录路径")
    seed_work3_runtime(int(config.runtime.seed), deterministic=bool(config.runtime.deterministic))
    torch.set_num_threads(int(config.runtime.main_num_threads))
    device = torch.device(config.runtime.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=int(config.bc.hidden_dim),
                             max_station_workers=int(config.bc.max_station_workers)).to(device)
    initial = resolve_path(config.paths.initial_checkpoint, PROJECT_ROOT)
    PPOTrainerWork3(actor, device=str(device), create_optimizer=False).load_checkpoint(str(initial))
    paths = {key: resolve_path(config.paths[f"{key}_records"], PROJECT_ROOT) for key in ("train", "validation")}
    team_objective = str(config.bc.get("team_objective", "sequence"))
    loaded = {key: load_demonstrations(path, actor, split=key, team_objective=team_objective) for key, path in paths.items()}
    if {row["parent_group"] for row in loaded["train"]} & {row["parent_group"] for row in loaded["validation"]}:
        raise ValueError("训练与验证存在相同母实例来源")
    samples = {key: [row for row in rows if row["label_scope"]] for key, rows in loaded.items()}
    if not all(samples.values()):
        raise ValueError("训练和验证都需要有效监督决定")
    configured_sampling = config.bc.get("demand_sampling_weights")
    sampling_weights = _demand_sampling_weights(samples["train"], configured_sampling)
    sampling_config = None if configured_sampling is None else {str(key): float(value) for key, value in configured_sampling.items()}
    if output is None or report is None:
        context = create_run_context(PROJECT_ROOT, experiment_name=config.artifacts.experiment_name,
            runs_root=config.artifacts.runs_root, run_id=config.artifacts.run_id, create_dirs=False)
        context.run_dir.mkdir(parents=True, exist_ok=False)
        create_run_context(PROJECT_ROOT, experiment_name=context.experiment_name, runs_root=config.artifacts.runs_root,
                           run_id=context.run_id, create_dirs=True)
        output = output or context.checkpoint_dir / "bc_best.pt"
        report = report or context.reports_dir / "behavior_cloning.json"
    output, report = Path(output), Path(report)
    initial_output = output.with_name(f"{output.stem}_initial{output.suffix}")
    final_output = output.with_name(f"{output.stem}_final{output.suffix}")
    for path in (output, report, initial_output, final_output):
        if path.exists():
            raise FileExistsError(f"行为克隆输出已存在：{path}")
    module = Work3BCModule(actor, samples["train"], config.bc)
    before = {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()}
    metadata = {"initial_checkpoint": str(initial), "records": {key: str(path) for key, path in paths.items()},
                "initial_checkpoint_sha256": sha256_file(initial),
                "records_sha256": {key: sha256_file(path) for key, path in paths.items()},
                "record_semantics_version": BC_RECORD_SEMANTICS_VERSION, "team_objective": team_objective,
                "demand_sampling_weights": sampling_config,
                "seed": int(config.runtime.seed)}
    initial_loss = _evaluate(actor, samples["validation"], int(config.bc.batch_size), team_objective)
    _save(module, initial_output, metadata, stage="initial")
    _save(module, output, {**metadata, "selection_validation_nll": initial_loss})
    selection = _Selection(samples["validation"], config, output, metadata, initial_loss)
    selection._save_diagnostic(module, 0, initial_loss)
    trainer = Trainer(accelerator="gpu" if device.type == "cuda" else "cpu", devices=1, precision="32-true",
                      max_steps=int(config.bc.max_updates), max_epochs=-1, logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
                      deterministic=bool(config.runtime.deterministic), callbacks=[selection])
    sampler = None if sampling_weights is None else WeightedRandomSampler(sampling_weights,
        num_samples=len(samples["train"]), replacement=True,
        generator=torch.Generator().manual_seed(int(config.runtime.seed)))
    loader = DataLoader(samples["train"], batch_size=int(config.bc.batch_size), shuffle=sampler is None, sampler=sampler,
                        num_workers=0, collate_fn=list)
    trainer.fit(module, train_dataloaders=loader)
    final_loss = _evaluate(actor, samples["validation"], int(config.bc.batch_size), team_objective)
    _save(module, final_output, {**metadata, "final_validation_nll": final_loss})
    sampled_demands, sampled_choices, sampled_scopes = Counter(), Counter(), Counter()
    for update in module.history:
        sampled_demands.update(update["demand_counts"])
        sampled_choices.update(update["real_choice_counts"])
        sampled_scopes.update(update["scope_counts"])
    result = {"stage": "bc", "team_objective": team_objective, "precision": "fp32", "device": str(device),
              "optimization_steps": module.optimization_steps, "optimizer_step_attempts": module.optimizer_step_attempts,
              "amp_skipped_steps": module.amp_skipped_steps, "initial_validation_nll": initial_loss,
              "termination_reason": selection.stop_reason,
              "final_validation_nll": final_loss, "best_validation_nll": selection.best_loss,
              "best_update": selection.best_step, "validation_checks": selection.checks, "history": module.history,
              "parameters_changed": sum(not torch.equal(before[key], value.detach().cpu()) for key, value in actor.state_dict().items()),
              "critic_has_gradient": any(parameter.grad is not None for parameter in actor.critic.parameters()),
              "shared_encoder_has_gradient": any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                                                  for parameter in actor.graph_encoder.parameters()),
              "records": {key: len(rows) for key, rows in loaded.items()},
              "supervised_decisions": {key: len(rows) for key, rows in samples.items()},
              "choice_supervised_decisions": {key: sum(bool(set(row["label_scope"]) & set(row["choice_layers"])) for row in rows)
                                               for key, rows in samples.items()},
              "instances": {key: _counts(rows) for key, rows in samples.items()},
              "sample_weight": ("inverse_instance_record_count" if sampler is None
                                  else "demand_sampling_with_inverse_instance_length_control"),
              "demand_sampling_weights": sampling_config,
              "raw_training_team_demand_counts": _batch_coverage(samples["train"])[0],
              "sampled_team_demand_counts": dict(sampled_demands),
              "sampled_real_choice_counts": dict(sampled_choices),
              "sampled_scope_counts": dict(sampled_scopes),
              "sample_index_audit_complete": module.sample_index_audit_complete and all(
                  row["sample_index_audit_complete"] for row in module.history),
              "train_024_0_2_count": sum(row["train_024_0_2_count"] for row in module.history),
              "sampled_batch_sequence_sha256": hashlib.sha256("\n".join(
                  row["sample_sequence_sha256"] for row in module.history).encode("ascii")).hexdigest(),
              "sampled_decisions": sum(row["sample_count"] for row in module.history),
              "elapsed_seconds": time.monotonic() - started,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "checkpoint": str(output.resolve()), "initial_checkpoint": str(initial_output.resolve()),
              "source_identity": metadata,
              "final_checkpoint": str(final_output.resolve()), "resolved_config": OmegaConf.to_container(config, resolve=True),
              "checkpoint_evaluation_eligible": False, "research_result_eligible": False}
    if selection.diagnostic_dir is not None:
        result["diagnostic_checkpoints"] = selection.diagnostic_checkpoints
        result["diagnostic_save_seconds"] = selection.diagnostic_save_seconds
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="同QUERY网络的示例行为克隆")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "conf" / "work3" / "pretrain_pilot.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set))
    result = run_behavior_cloning(config, output=args.output, report=args.report)
    print(json.dumps({key: result[key] for key in ("optimization_steps", "best_validation_nll", "checkpoint")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
