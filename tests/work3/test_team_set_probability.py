"""团队集合概率按原合法分母计算，并与完整人员排列及梯度独立对照。"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import pytest
import torch

from apal.data.work3.demonstrations import build_team_set_cache
from apal.envs.work3.core_types import ActionBranch
from apal.models.work3.actor_critic import ActorCriticWork3


@pytest.mark.parametrize("demand", [2, 5])
def test_team_subset_probability_and_gradient_match_full_permutations(tmp_path: Path, demand: int) -> None:
    from apal.data.work3.certified_examples import generate_instance, make_instance_env, load_witness_actions

    torch.set_num_threads(1)
    torch.manual_seed(42)
    if demand == 5:
        case = tmp_path / "five"
        generate_instance(case, family="five_person", seed=401, aircraft=2)
        env = make_instance_env(case)
    else:
        case = tmp_path / "two"
        generate_instance(case, family="structured", seed=101, aircraft=2)
        env = make_instance_env(case)
        env.step(load_witness_actions(case)[0]["action"])
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=8)
    snapshot = actor.make_decision_snapshot(env, torch.zeros(32), torch.zeros(2))
    index = next(i for i, context in enumerate(snapshot.team_contexts)
                 if context.demand == demand and len(snapshot.query_tables[i].eligible_worker_ids) > demand)
    team = snapshot.query_tables[index].eligible_worker_ids[:demand]
    teacher = {"task_key": snapshot.candidate_task_keys[index], "branch": ActionBranch.STATION_EXECUTE,
               "team": team, "align": 0}
    _, _, _, record = actor.select_snapshot(snapshot, teacher_action=teacher)
    record["team_set_cache"] = build_team_set_cache(snapshot, record, actor.max_station_workers)
    assert len(record["team_set_cache"]["subsets"]) == 2 ** demand - 1
    state, times = snapshot.state_features.unsqueeze(0), snapshot.time_features.unsqueeze(0)
    old_values, old_lp, old_entropy = actor.evaluate_action_log_probs(state, times, [record])
    values, lp, entropy, parts = actor.evaluate_action_log_probs(state, times, [record],
        return_components=True, team_set_supervision=True)
    assert torch.equal(values, old_values) and torch.equal(lp, old_lp) and torch.equal(entropy, old_entropy)
    set_lp = parts[0]["team_set"]
    assert set_lp.exp().item() < 0.999
    sequence_records = [actor.select_snapshot(snapshot, teacher_action={**teacher, "team": order})[3]
                        for order in permutations(team)]
    _, _, _, sequence_parts = actor.evaluate_action_log_probs(
        state.expand(len(sequence_records), -1), times.expand(len(sequence_records), -1),
        sequence_records, return_components=True)
    exact = torch.logsumexp(torch.stack([parts["team"] for parts in sequence_parts]), dim=0)
    torch.testing.assert_close(set_lp, exact, atol=1e-6, rtol=1e-5)
    parameters = tuple(actor.parameters())
    subset_grad = torch.autograd.grad(set_lp, parameters, allow_unused=True)
    explicit_grad = torch.autograd.grad(exact, parameters, allow_unused=True)
    for left, right in zip(subset_grad, explicit_grad):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    _, _, _, reversed_record = actor.select_snapshot(snapshot, teacher_action={**teacher, "team": tuple(reversed(team))})
    reversed_record["team_set_cache"] = build_team_set_cache(snapshot, reversed_record, actor.max_station_workers)
    _, _, _, reversed_parts = actor.evaluate_action_log_probs(state, times, [reversed_record],
        return_components=True, team_set_supervision=True)
    assert torch.equal(reversed_parts[0]["team_set"], set_lp)
    from apal.training.work3.behavior_cloning import behavior_cloning_loss
    loss, layers = behavior_cloning_loss(reversed_parts, [{"instance_id": "case", "label_scope": ("team",)}],
                                        {"case": 1}, team_objective="set")
    loss.backward()
    assert "team_set" in layers
    assert actor.query_worker_score.weight.grad is not None
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
               for parameter in actor.graph_encoder.parameters())
    assert all(parameter.grad is None for parameter in actor.critic.parameters())
    with pytest.raises(ValueError, match="子集缓存"):
        actor.evaluate_action_log_probs(state, times, [{key: value for key, value in record.items()
                                                       if key != "team_set_cache"}],
                                        return_components=True, team_set_supervision=True)
