"""教师条件诊断沿用真实人员解码和原BC权重，不改现场或随机状态。"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from apal.data.work3.certified_examples import generate_instance, make_instance_env
from apal.data.work3.demonstrations import build_team_set_cache
from apal.models.work3.actor_critic import ActorCriticWork3


def _sample_at_greedy_team(tmp_path: Path, actor: ActorCriticWork3, demand: int) -> tuple:
    case = tmp_path / f"demand_{demand}"
    generate_instance(case, family="five_person" if demand == 5 else "structured", seed=1901)
    env = make_instance_env(case)
    actor.eval()
    for _ in range(len(env.state.tasks) + 20):
        snapshot = actor.make_decision_snapshot(env, torch.zeros(32), torch.zeros(2))
        action, _, _, record = actor.select_snapshot(snapshot, deterministic=True)
        if action["branch"] == 0 and len(action["team"]) == demand:
            record["team_set_cache"] = build_team_set_cache(snapshot, record, actor.max_station_workers)
            sample = {"instance_id": f"demand_{demand}", "state_features": snapshot.state_features,
                "time_features": snapshot.time_features, "record": record,
                "label_scope": ("task", "branch", "team", "align"),
                "choice_layers": ("team", "align")}
            return snapshot, action, sample
        env.step(action)
    raise AssertionError("真实小例未到达指定人数的组队选择")


@pytest.mark.parametrize("demand", [2, 5])
def test_conditional_greedy_event_matches_original_full_actor(tmp_path: Path, demand: int) -> None:
    from apal.evaluation.bc_prefix_diagnostics import conditional_greedy_target_team

    torch.set_num_threads(1)
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=16)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
    snapshot, action, sample = _sample_at_greedy_team(tmp_path, actor, demand)
    assert conditional_greedy_target_team(actor, sample)
    table = snapshot.query_tables[sample["record"]["task_idx"]]
    alternative = next(pair[0].team for pair in table.entries if set(pair[0].team) != set(action["team"]))
    _, _, _, alternate_record = actor.select_snapshot(snapshot, teacher_action={**action, "team": alternative})
    alternate_record["team_set_cache"] = build_team_set_cache(snapshot, alternate_record, actor.max_station_workers)
    alternate_sample = {**sample, "record": alternate_record}
    assert not conditional_greedy_target_team(actor, alternate_sample)


def test_weighted_layer_decomposition_matches_original_and_preserves_runtime(tmp_path: Path) -> None:
    from apal.evaluation.bc_prefix_diagnostics import teacher_metrics
    from apal.training.work3.behavior_cloning import _replay, behavior_cloning_loss

    torch.set_num_threads(1)
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=16)
    _, _, first = _sample_at_greedy_team(tmp_path, actor, 2)
    _, _, second = _sample_at_greedy_team(tmp_path, actor, 5)
    samples = [first, first, second]
    counts = {first["instance_id"]: 2, second["instance_id"]: 1}
    actor.eval()
    with torch.no_grad():
        expected, _ = behavior_cloning_loss(_replay(actor, samples, team_objective="set"),
                                           samples, counts, team_objective="set")
    actor.train()
    actor.graph_encoder.eval()
    modes = {name: module.training for name, module in actor.named_modules()}
    state = {key: value.clone() for key, value in actor.state_dict().items()}
    torch_rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    result = teacher_metrics(actor, samples, batch_size=2)
    assert result["original_nll"] == pytest.approx(float(expected), abs=1e-6)
    assert result["weighted_contribution_sum"] == pytest.approx(result["original_nll"], abs=1e-6)
    assert sum(result["weighted_contribution"].values()) == pytest.approx(result["original_nll"], abs=1e-6)
    assert result["all_scope"]["branch"]["count"] == 3
    assert "branch" not in result["choice_only"]
    assert result["team_by_demand"]["2"]["count"] == 2
    assert result["team_by_demand"]["2"]["team_set_nll_mean"] >= 0.0
    assert result["team_by_demand"]["2"]["choice_team_set_nll_mean"] == pytest.approx(
        result["team_by_demand"]["2"]["team_set_nll_mean"])
    assert result["team_by_demand"]["5"]["count"] == 1
    assert {name: module.training for name, module in actor.named_modules()} == modes
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    current_numpy = np.random.get_state()
    assert numpy_rng[0] == current_numpy[0] and np.array_equal(numpy_rng[1], current_numpy[1])
    assert numpy_rng[2:] == current_numpy[2:]
    assert all(torch.equal(value, actor.state_dict()[key]) for key, value in state.items())
    broken = deepcopy(first)
    broken["record"]["team_set_cache"]["version"] = "错误版本"
    with pytest.raises(ValueError):
        teacher_metrics(actor, [broken])
    assert {name: module.training for name, module in actor.named_modules()} == modes


def test_common_prefix_scores_full_actions_without_changing_actor_runtime(tmp_path: Path) -> None:
    from apal.evaluation.bc_prefix_diagnostics import score_prefix_decision

    torch.set_num_threads(1)
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=16)
    snapshot, action, sample = _sample_at_greedy_team(tmp_path, actor, 2)
    actor.train()
    actor.graph_encoder.eval()
    modes = {name: module.training for name, module in actor.named_modules()}
    rng = torch.get_rng_state().clone()
    result = score_prefix_decision(actor, snapshot, action, sample["label_scope"])
    assert result["greedy_action"]["task_key"] == action["task_key"]
    assert result["greedy_action"]["team"] == list(action["team"])
    assert result["first_different_layer"] is None
    assert result["teacher_team"]["conditional_greedy_target_team"] is True
    assert result["choice_layer_hits"]["team"] is True
    assert "team_set" in result["teacher_component_probabilities"]
    assert result["teacher_sampling_replay_abs_error"] <= 1e-6
    table = snapshot.query_tables[sample["record"]["task_idx"]]
    alternative = next(pair[0].team for pair in table.entries if set(pair[0].team) != set(action["team"]))
    changed = score_prefix_decision(actor, snapshot, {**action, "team": alternative}, sample["label_scope"])
    assert changed["first_different_layer"] == "team"
    assert changed["choice_layer_hits"]["team"] is False
    assert changed["choice_layer_hits"]["align"] is None
    assert changed["teacher_team"]["conditional_greedy_target_team"] is False
    assert torch.equal(rng, torch.get_rng_state())
    assert {name: module.training for name, module in actor.named_modules()} == modes


def test_critical_worker_distribution_uses_sparse_ids_and_original_team_probability(tmp_path: Path) -> None:
    from apal.data.work3.certified_examples import load_witness_actions
    from apal.evaluation.bc_prefix_diagnostics import critical_worker_distribution
    from apal.training.work3.behavior_cloning import _replay

    torch.set_num_threads(1)
    case = tmp_path / "single_worker"
    generate_instance(case, family="structured", seed=1901)
    env = make_instance_env(case)
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=16).eval()
    for witness in load_witness_actions(case):
        action = witness["action"]
        if action["branch"] == 0 and len(action["team"]) == 1:
            task = env.state.tasks[action["task_key"]]
            if len(env.valid_team_completion_workers(task, [])) > 1:
                break
        env.step(action)
    else:
        raise AssertionError("真实构造未覆盖多合法候选的单人动作")
    snapshot = actor.make_decision_snapshot(env, torch.zeros(32), torch.zeros(2))
    _, _, _, record = actor.select_snapshot(snapshot, teacher_action=action)
    record["team_set_cache"] = build_team_set_cache(snapshot, record, actor.max_station_workers)
    sample = {"state_features": snapshot.state_features, "time_features": snapshot.time_features,
              "record": record, "h0": env.state.h0, "current_time": env.state.current_time}
    with torch.no_grad():
        expected = float(_replay(actor, [sample], team_objective="set")[0]["team_set"].exp().item())
    assert any(index != worker for index, worker in enumerate(record["station_worker_ids"]))
    actor.train()
    actor.graph_encoder.eval()
    modes = {name: module.training for name, module in actor.named_modules()}
    rng = torch.get_rng_state().clone()
    result = critical_worker_distribution(actor, sample)
    assert result["target_probability"] == pytest.approx(expected, abs=1e-6)
    target = int(action["team"][0])
    assert result["worker_id_to_index"][str(target)] == record["station_worker_ids"].index(target)
    assert result["legal_probability_sum"] == pytest.approx(1, abs=1e-6)
    assert all(row["mask"] for row in result["legal_workers"])
    chosen = max(result["legal_workers"], key=lambda row: row["logit"])
    assert chosen["worker_id"] == result["greedy_worker_id"]
    assert all(len(row["query14"]) == 14 for row in result["legal_workers"])
    assert torch.equal(rng, torch.get_rng_state())
    assert {name: module.training for name, module in actor.named_modules()} == modes
