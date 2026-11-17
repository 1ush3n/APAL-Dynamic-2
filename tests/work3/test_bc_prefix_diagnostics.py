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
