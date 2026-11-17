"""教师现场的分层BC诊断；条件团队指标不冒充策略自由执行准确率。"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import math
from typing import Any, Iterator

import torch

from apal.models.work3.actor_critic import ActorCriticWork3
from apal.training.work3.behavior_cloning import _replay, behavior_cloning_loss


@contextmanager
def _teacher_mode(actor: ActorCriticWork3) -> Iterator[None]:
    modes = [(module, module.training) for module in actor.modules()]
    try:
        actor.eval()
        with torch.inference_mode(), torch.amp.autocast(
            device_type=next(actor.parameters()).device.type, enabled=False
        ):
            yield
    finally:
        for module, training in modes:
            module.training = training


def conditional_greedy_target_team(actor: ActorCriticWork3, sample: dict[str, Any]) -> bool:
    """固定教师工序与留站分支，原逐人贪心是否恰落在已证团队集合内。"""
    record = sample["record"]
    if not actor.condition_queries or int(record.get("branch", -1)) != 0 or not record.get("chosen_team"):
        raise ValueError("条件团队诊断要求完整QUERY留站团队记录")
    with _teacher_mode(actor):
        actor._validate_condition_record(record)
        device = next(actor.parameters()).device
        _, context, task_nodes, worker_nodes, hierarchy = actor._encode_state_components(
            sample["state_features"].to(device), sample["time_features"].to(device), record["graph_snapshot"])
        features = record.get("cand_feats", record.get("candidate_task_features"))
        task = actor._candidate_embeddings(features.to(device), task_nodes, hierarchy, record,
                                            record["query_task_features"])[int(record["task_idx"])]
        indices = record.get("worker_node_indices", ())
        bias = None
        if worker_nodes is not None and indices:
            # 人员节点[W,D]经原评分投影[W,1]去末维为[W]。
            node_indices = torch.as_tensor(indices, device=device)
            bias = actor.worker_graph_score(worker_nodes[node_indices]).squeeze(-1)
        actor._team_set_log_prob(context, task, bias, record)
        cache = record["team_set_cache"]
        target, mask = tuple(cache["target_worker_indices"]), 0
        for step in range(len(target)):
            entry = cache["subsets"][mask]
            tracker = torch.zeros(actor.max_station_workers, dtype=context.dtype, device=device)
            tracker[list(entry["selected_worker_indices"])] = 1.0
            logits = actor._worker_logits(context, task, tracker, entry["query_worker_features"],
                                           bias, int(record["num_st_workers"]))
            logits = actor._apply_worker_selection_mask(logits,
                torch.as_tensor(entry["worker_valid_mask"], dtype=torch.bool, device=device),
                task_key=record["task_key"], step_index=step)
            chosen = int(logits.argmax().item())
            if chosen not in target:
                return False
            mask |= 1 << target.index(chosen)
        return True


def teacher_metrics(actor: ActorCriticWork3, samples: list[dict[str, Any]],
                    batch_size: int = 4) -> dict[str, Any]:
    """重算原BC损失及其分摊，仅读记录中的事实，不访问现场或采样动作。"""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("诊断批量必须为正整数")
    selected = [sample for sample in samples if sample["label_scope"]]
    if not selected:
        raise ValueError("教师诊断需要非空监督记录")
    counts = Counter(sample["instance_id"] for sample in selected)
    total = len(selected)
    all_values: dict[str, list[float]] = {}
    choice_values: dict[str, list[float]] = {}
    contributions: dict[str, float] = {}
    by_instance = {key: {"count": count, "weighted_layer_contribution": {}} for key, count in counts.items()}
    teams: dict[str, dict[str, Any]] = {}
    original_nll = 0.0
    with _teacher_mode(actor):
        for start in range(0, total, batch_size):
            batch = selected[start:start + batch_size]
            components = _replay(actor, batch, team_objective="set")
            loss, _ = behavior_cloning_loss(components, batch, dict(counts), team_objective="set")
            original_nll += float(loss.item()) * len(batch) / total
            for sample, parts in zip(batch, components, strict=True):
                scope = sample["label_scope"]
                choice = set(sample.get("choice_layers", ()))
                if not choice.issubset(scope):
                    raise ValueError("有选择的监督层不属于实际监督范围")
                instance = sample["instance_id"]
                weight = total / (len(counts) * counts[instance])
                for layer in scope:
                    key = "team_set" if layer == "team" else layer
                    nll = float(-parts[key].item())
                    if not math.isfinite(nll):
                        raise FloatingPointError("教师分层负对数概率非有限")
                    all_values.setdefault(key, []).append(nll)
                    if layer in choice:
                        choice_values.setdefault(key, []).append(nll)
                    contribution = nll * weight / (total * len(scope))
                    contributions[key] = contributions.get(key, 0.0) + contribution
                    instance_parts = by_instance[instance]["weighted_layer_contribution"]
                    instance_parts[key] = instance_parts.get(key, 0.0) + contribution
                if "team" in scope:
                    demand = str(len(sample["record"]["chosen_team"]))
                    values = teams.setdefault(demand, {"count": 0, "probability_sum": 0.0,
                        "conditional_greedy_target_hits": 0, "choice_count": 0,
                        "choice_probability_sum": 0.0, "choice_conditional_greedy_target_hits": 0})
                    probability = float(parts["team_set"].exp().item())
                    hit = int(conditional_greedy_target_team(actor, sample))
                    values["count"] += 1
                    values["probability_sum"] += probability
                    values["conditional_greedy_target_hits"] += hit
                    if "team" in choice:
                        values["choice_count"] += 1
                        values["choice_probability_sum"] += probability
                        values["choice_conditional_greedy_target_hits"] += hit
    for values in teams.values():
        values["team_set_probability_mean"] = values.pop("probability_sum") / values["count"]
        values["conditional_greedy_target_hit_rate"] = values["conditional_greedy_target_hits"] / values["count"]
        probability_sum = values.pop("choice_probability_sum")
        count = values["choice_count"]
        values["choice_team_set_probability_mean"] = probability_sum / count if count else None
        values["choice_conditional_greedy_target_hit_rate"] = values["choice_conditional_greedy_target_hits"] / count if count else None
    contribution_sum = sum(contributions.values())
    return {"input_records": len(samples), "supervised_records": total, "original_nll": original_nll,
        "all_scope": {key: {"nll": sum(values) / len(values), "count": len(values)} for key, values in all_values.items()},
        "choice_only": {key: {"nll": sum(values) / len(values), "count": len(values)} for key, values in choice_values.items()},
        "weighted_contribution": contributions, "weighted_contribution_sum": contribution_sum,
        "decomposition_abs_error": abs(contribution_sum - original_nll), "by_instance": by_instance,
        "team_by_demand": teams,
        "weighting_scope": "原BC按实例长度加权，每记录均分受监督层；未按重采样分布重加权验证损失",
        "team_scope": "条件于教师工序与留站分支的已证团队事件；集合概率汇总全部人员顺序，命中不是自由轨迹准确率"}
