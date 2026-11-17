"""教师现场的分层BC诊断；条件团队指标不冒充策略自由执行准确率。"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import math
from typing import Any, Iterator

import torch

from apal.data.work3.demonstrations import build_team_set_cache
from apal.envs.work3.decision_snapshot import DecisionSnapshot
from apal.envs.work3.scheduling_queries import CONDITION_LAYOUT
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


def _worker_context(actor: ActorCriticWork3, sample: dict[str, Any]) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor | None
]:
    record, device = sample["record"], next(actor.parameters()).device
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
    return context, task, bias


def conditional_greedy_target_team(actor: ActorCriticWork3, sample: dict[str, Any]) -> bool:
    """固定教师工序与留站分支，原逐人贪心是否恰落在已证团队集合内。"""
    record = sample["record"]
    if not actor.condition_queries or int(record.get("branch", -1)) != 0 or not record.get("chosen_team"):
        raise ValueError("条件团队诊断要求完整QUERY留站团队记录")
    with _teacher_mode(actor):
        actor._validate_condition_record(record)
        device = next(actor.parameters()).device
        context, task, bias = _worker_context(actor, sample)
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


def critical_worker_distribution(actor: ActorCriticWork3, sample: dict[str, Any]) -> dict[str, Any]:
    """固定单人教师工序与留站分支，读取原空前缀人员分布；不执行新动作。"""
    record = sample["record"]
    if (not actor.condition_queries or int(record.get("branch", -1)) != 0
            or len(record.get("chosen_team", ())) != 1):
        raise ValueError("单人工人分布要求完整QUERY留站团队记录")
    with _teacher_mode(actor):
        actor._validate_condition_record(record)
        parts = _replay(actor, [sample], team_objective="set")[0]
        expected = float(parts["team_set"].exp().item())
        entry = record["team_set_cache"]["subsets"][0]
        if (tuple(entry["selected_worker_indices"]) != () or tuple(record["worker_prefixes"][0]) != ()
                or tuple(entry["worker_valid_mask"]) != tuple(record["worker_valid_masks"][0])
                or not torch.equal(entry["query_worker_features"], record["query_worker_features"][0])):
            raise ValueError("单人空前缀缓存与原条件或掩码不一致")
        context, task, bias = _worker_context(actor, sample)
        tracker = torch.zeros(actor.max_station_workers, dtype=context.dtype, device=context.device)
        raw = entry["query_worker_features"]
        logits = actor._worker_logits(context, task, tracker, raw, bias, int(record["num_st_workers"]))
        mask = torch.as_tensor(entry["worker_valid_mask"], dtype=torch.bool, device=context.device)
        masked = actor._apply_worker_selection_mask(logits, mask, task_key=record["task_key"], step_index=0)
        probabilities = torch.distributions.Categorical(logits=masked).probs
        worker_ids = tuple(int(worker) for worker in record["station_worker_ids"])
        target = int(record["chosen_team"][0])
        target_index = worker_ids.index(target)
        if tuple(record["chosen_worker_indices"]) != (target_index,):
            raise ValueError("教师工人ID与站内候选索引不一致")
        probability = float(probabilities[target_index].item())
        error = abs(probability - expected)
        if error > 1e-5:
            raise ValueError("单人工人分布与原集合重放概率不一致")
        h0, now = sample.get("h0"), sample.get("current_time")
        if h0 is not None and (not math.isfinite(h0) or h0 <= 0):
            raise ValueError("显式H0必须为正有限数")
        if now is not None and not math.isfinite(now):
            raise ValueError("显式当前时刻必须为有限数")
        legal = []
        for index, worker in enumerate(worker_ids):
            if not bool(mask[index]):
                continue
            query = raw[index].tolist()
            normalized = {"start_delta_h0": query[5], "end_delta_h0": query[6],
                          "duration_h0": query[7], "time_valid": bool(query[8])}
            window = None
            if normalized["time_valid"] and h0 is not None and now is not None:
                window = {"start": now + h0 * query[5], "end": now + h0 * query[6],
                          "duration": h0 * query[7]}
            legal.append({"worker_id": worker, "index": index, "mask": True,
                          "logit": float(logits[index].item()), "probability": float(probabilities[index].item()),
                          "query14": query, "normalized_window": normalized, "decoded_window": window})
        greedy_index = int(masked.argmax().item())
        return {"task_key": record["task_key"], "target_worker_id": target, "target_worker_index": target_index,
                "station_worker_ids": list(worker_ids), "worker_id_to_index": {str(worker): index for index, worker in enumerate(worker_ids)},
                "mask": mask.cpu().tolist(), "condition_layout": list(CONDITION_LAYOUT),
                "query14": raw.tolist(), "legal_workers": legal, "greedy_worker_id": worker_ids[greedy_index],
                "target_probability": probability, "original_team_set_probability": expected,
                "probability_abs_error": error, "legal_probability_sum": float(probabilities[mask].sum().item()),
                "h0": h0, "current_time": now,
                "scope": "仅条件于教师工序+留站+空选人前缀；窗口从原FP32条件解码，未给H0/当前时刻或无有效时间时不报告绝对窗口"}


def score_prefix_decision(actor: ActorCriticWork3, snapshot: DecisionSnapshot,
                          teacher_action: dict[str, Any], label_scope: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """同一教师现场读取整动作与各实际分层；不同选择不自动构成错误。"""
    if len(set(label_scope)) != len(label_scope) or any(
        layer not in ("advance", "task", "branch", "team", "align") for layer in label_scope
    ):
        raise ValueError("教师监督范围必须是有效且不重复的动作层")
    with _teacher_mode(actor):
        _, teacher_lp, _, record = actor.select_snapshot(snapshot, teacher_action=teacher_action)
        greedy, greedy_lp, _, _ = actor.select_snapshot(snapshot, deterministic=True)
        stay = record.get("action_type") == "schedule" and int(record["branch"]) == 0
        if stay:
            record["team_set_cache"] = build_team_set_cache(snapshot, record, actor.max_station_workers)
        # 单现场事实[32]、[2]加批次维为[1,32]、[1,2]；表示由当前参数重算。
        _, replay_lp, _, components = actor.evaluate_action_log_probs(
            snapshot.state_features.unsqueeze(0), snapshot.time_features.unsqueeze(0), [record],
            return_components=True, team_set_supervision=True)
        parts = components[0]
        scope = tuple(layer for layer in label_scope if layer in parts)
        available: set[str] = set()
        if record["action_type"] != "forced_advance":
            if snapshot.advance_available:
                available.add("advance")
            if record["action_type"] == "schedule":
                if len(snapshot.candidate_task_keys) > 1:
                    available.add("task")
                if sum(snapshot.branch_masks[record["task_idx"]]) > 1:
                    available.add("branch")
                if stay:
                    available.add("align")
                    if len(snapshot.query_tables[record["task_idx"]].entries) > 1:
                        available.add("team")
        choices = tuple(layer for layer in scope if layer in available)
        teacher_advance, greedy_advance = int(teacher_action["branch"]) == 2, int(greedy["branch"]) == 2
        matches: dict[str, bool | None] = {"advance": teacher_advance == greedy_advance,
                                         "task": None, "branch": None, "team": None, "align": None}
        if not teacher_advance and not greedy_advance:
            matches["task"] = teacher_action["task_key"] == greedy["task_key"]
            if matches["task"]:
                matches["branch"] = int(teacher_action["branch"]) == int(greedy["branch"])
                if matches["branch"] and int(teacher_action["branch"]) == 0:
                    matches["team"] = set(teacher_action["team"]) == set(greedy["team"])
                    if matches["team"]:
                        matches["align"] = int(teacher_action["align"]) == int(greedy["align"])
        team = None
        if stay:
            sample = {"state_features": snapshot.state_features, "time_features": snapshot.time_features,
                      "record": record}
            team = {"demand": len(record["chosen_team"]), "supervised": "team" in scope,
                    "set_probability": float(parts["team_set"].exp().item()),
                    "conditional_greedy_target_team": conditional_greedy_target_team(actor, sample)}
        actions = []
        for action in (teacher_action, greedy):
            converted = {"branch": int(action["branch"])}
            if "task_key" in action:
                converted["task_key"] = str(action["task_key"])
            if "team" in action:
                converted["team"] = [int(worker) for worker in action["team"]]
                converted["align"] = int(action["align"])
            actions.append(converted)
        return {"teacher_action": actions[0], "greedy_action": actions[1], "label_scope": list(scope),
            "choice_layers": list(choices), "choice_layer_counts": {layer: 1 for layer in choices},
            "choice_layer_comparable_counts": {layer: int(matches[layer] is not None) for layer in choices},
            "choice_layer_hits": {layer: matches[layer] for layer in choices},
            "teacher_component_log_probs": {key: float(value.item()) for key, value in parts.items()},
            "teacher_component_probabilities": {key: float(value.exp().item()) for key, value in parts.items()},
            "teacher_sequence_log_prob": float(teacher_lp), "greedy_sequence_log_prob": float(greedy_lp),
            "teacher_sampling_replay_abs_error": abs(float(teacher_lp) - float(replay_lp[0].item())),
            "teacher_team": team,
            "first_different_layer": next((layer for layer, match in matches.items() if match is False), None),
            "comparison_scope": "实际贪心整动作逐层与一条已证教师动作比较；下层只有上游相同时可比，不同不等于坏动作"}


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
                        "nll_sum": 0.0, "choice_nll_sum": 0.0,
                        "conditional_greedy_target_hits": 0, "choice_count": 0,
                        "choice_probability_sum": 0.0, "choice_conditional_greedy_target_hits": 0})
                    probability = float(parts["team_set"].exp().item())
                    hit = int(conditional_greedy_target_team(actor, sample))
                    values["count"] += 1
                    values["probability_sum"] += probability
                    values["nll_sum"] += float(-parts["team_set"].item())
                    values["conditional_greedy_target_hits"] += hit
                    if "team" in choice:
                        values["choice_count"] += 1
                        values["choice_probability_sum"] += probability
                        values["choice_nll_sum"] += float(-parts["team_set"].item())
                        values["choice_conditional_greedy_target_hits"] += hit
    for values in teams.values():
        values["team_set_nll_mean"] = values.pop("nll_sum") / values["count"]
        values["team_set_probability_mean"] = values.pop("probability_sum") / values["count"]
        values["conditional_greedy_target_hit_rate"] = values["conditional_greedy_target_hits"] / values["count"]
        probability_sum = values.pop("choice_probability_sum")
        count = values["choice_count"]
        values["choice_team_set_nll_mean"] = values.pop("choice_nll_sum") / count if count else None
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
