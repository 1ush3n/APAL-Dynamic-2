"""R01：逐步工人合法掩码必须随动作快照进入PPO重放。"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.distributions import Categorical

from envs.work3.core_types import ActionBranch
from envs.work3.environment import AirLineEnvWork3
from envs.work3.event_queue import EventType
from models.work3.actor_critic import ActorCriticWork3


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")


def _new_env() -> AirLineEnvWork3:
    if not BASELINE_PATH.is_file():
        pytest.skip(f"{BASELINE_PATH} 不存在")
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    return env


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(32), torch.zeros(2)


def _assert_probability_round_trip(
    sampled_log_prob: float,
    replay_log_prob: torch.Tensor,
) -> None:
    assert replay_log_prob.dtype == torch.float32
    assert float(replay_log_prob[0].detach()) == pytest.approx(
        sampled_log_prob,
        abs=1e-5,
    )
    ratio = torch.exp(replay_log_prob - sampled_log_prob)
    assert torch.isfinite(ratio).all()
    assert float(ratio[0].detach()) == pytest.approx(1.0, abs=1e-5)


def test_stay_replay_uses_sampled_skill_masks_after_live_state_changes() -> None:
    """三名合格工人下逐人条件概率及完整联合log-prob应一致。"""
    env = _new_env()
    task = env.get_action_candidates()[0]

    # 使用实际站位1的四人工池：恰有三人具备技能3，需求两人。
    # 将当前测试任务和飞机置于该站，保留环境的真实团队补全校验。
    task.current_station = 1
    task.skill = 3
    task.demand = 2
    env.state.aircraft[task.aircraft_id].current_station = 1
    station_workers = env.state.station_worker_bindings[1]
    eligible_workers = set(station_workers[:3])
    for worker_id in station_workers:
        skills = set(env.worker_skills[worker_id])
        if worker_id in eligible_workers:
            skills.add(task.skill)
        else:
            skills.discard(task.skill)
        env.worker_skills[worker_id] = frozenset(skills)
    eligible = set(env.valid_team_completion_workers(task, []))
    assert len(station_workers) == 4
    assert eligible == eligible_workers
    assert len(eligible) == 3
    assert eligible < set(station_workers)
    env.get_action_candidates = lambda: [task]  # type: ignore[method-assign]

    actor = ActorCriticWork3(hidden_dim=32)
    actor.eval()
    with torch.no_grad():
        for module in (
            actor.branch_head,
            actor.worker_score_fc,
            actor.worker_graph_score,
            actor.align_head,
        ):
            for parameter in module.parameters():
                parameter.zero_()
        actor.branch_head[-1].bias[0] = 0.0
        actor.branch_head[-1].bias[1] = -100.0
        actor.align_head[-1].bias[1] = math.log(3.0)
    state_feat, urgency = _inputs()
    action, sampled_log_prob, _, record = actor.select_action(
        env, state_feat, urgency, deterministic=True
    )

    assert action is not None
    assert action["branch"] == ActionBranch.STATION_EXECUTE
    assert len(record["chosen_worker_indices"]) == 2
    assert len(action["team"]) == task.demand
    assert set(action["team"]) <= eligible
    assert record["align"] == 1
    assert record["advance_available"] is False

    first_worker_log_prob = -math.log(3.0)
    second_worker_log_prob = -math.log(2.0)
    alignment_log_prob = math.log(0.75)
    expected_joint_log_prob = (
        math.log(1.0)  # 单一候选工序
        + math.log(1.0)  # 唯一有效留站分支
        + first_worker_log_prob
        + second_worker_log_prob
        + alignment_log_prob
    )
    assert sampled_log_prob == pytest.approx(expected_joint_log_prob, abs=1e-6)

    # 更改现场技能后，旧动作仍必须按采样时图和掩码重放。
    for worker_id in station_workers:
        env.worker_skills[worker_id] = {0, 1, 2, 3, 4}
    _, replay_log_prob, entropies = actor.evaluate_action_log_probs(
        state_feat.unsqueeze(0), urgency.unsqueeze(0), [record]
    )
    _assert_probability_round_trip(sampled_log_prob, replay_log_prob)
    assert torch.isfinite(entropies).all()

    masks = record["worker_valid_masks"]
    assert len(masks) == 2
    mask_values = [torch.as_tensor(mask, dtype=torch.bool).tolist() for mask in masks]
    expected_first = [worker_id in eligible for worker_id in station_workers]
    expected_first.extend([False] * (actor.max_station_workers - len(station_workers)))
    selected_first = record["chosen_worker_indices"][0]
    expected_second = [
        worker_id in eligible and index != selected_first
        for index, worker_id in enumerate(station_workers)
    ]
    expected_second.extend([False] * (actor.max_station_workers - len(station_workers)))
    assert mask_values == [expected_first, expected_second]
    for index, mask_values_i in enumerate(mask_values):
        mask = torch.tensor(mask_values_i, dtype=torch.bool)
        distribution = Categorical(
            logits=actor._apply_worker_selection_mask(
                torch.zeros(actor.max_station_workers),
                mask,
                task_key=task.task_key,
                step_index=index,
            )
        )
        assert torch.equal(distribution.probs[~mask], torch.zeros_like(distribution.probs[~mask]))

    legacy_record = dict(record)
    legacy_record.pop("worker_valid_masks")
    with pytest.raises(ValueError, match="缺少逐步工人合法掩码快照"):
        actor.evaluate_action_log_probs(
            state_feat.unsqueeze(0), urgency.unsqueeze(0), [legacy_record]
        )

    actor.zero_grad(set_to_none=True)
    (-replay_log_prob.sum()).backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in actor.worker_score_fc.parameters()
    )


def test_sampled_worker_order_replays_after_environment_sorts_execution_team() -> None:
    """执行团队可排序，PPO仍按采样时逐人选择顺序重放。"""
    env = _new_env()
    actor = ActorCriticWork3(hidden_dim=32)
    task = next(
        task
        for task in env.get_action_candidates()
        if env.can_reserve(task)
        and 2 <= len(env.state.station_worker_bindings[task.current_station])
        <= actor.max_station_workers
    )
    station_workers = env.state.station_worker_bindings[task.current_station]
    task.skill = 3
    task.demand = 2
    eligible_workers = set(station_workers[:2])
    for worker_id in station_workers:
        skills = set(env.worker_skills[worker_id])
        if worker_id in eligible_workers:
            skills.add(task.skill)
        else:
            skills.discard(task.skill)
        env.worker_skills[worker_id] = frozenset(skills)
    assert set(env.valid_team_completion_workers(task, [])) == eligible_workers
    env.get_action_candidates = lambda: [task]  # type: ignore[method-assign]

    actor.eval()
    first_local_index, second_local_index = sorted(
        range(2), key=lambda index: station_workers[index], reverse=True
    )[:2]
    with torch.no_grad():
        for module in (
            actor.branch_head,
            actor.worker_score_fc,
            actor.worker_graph_score,
            actor.align_head,
        ):
            for parameter in module.parameters():
                parameter.zero_()
        actor.branch_head[-1].bias[1] = -100.0
        actor.worker_score_fc[-1].bias[first_local_index] = 2.0
        actor.worker_score_fc[-1].bias[second_local_index] = 1.0

    state_feat, urgency = _inputs()
    action, sampled_log_prob, _, record = actor.select_action(
        env, state_feat, urgency, deterministic=True
    )
    assert action is not None
    assert tuple(record["chosen_worker_indices"]) == (
        first_local_index,
        second_local_index,
    )
    assert tuple(action["team"]) == tuple(record["chosen_team"])
    assert tuple(action["team"]) == (
        station_workers[first_local_index],
        station_workers[second_local_index],
    )
    assert tuple(action["team"]) != tuple(sorted(action["team"]))

    replay_inputs = (state_feat.unsqueeze(0), urgency.unsqueeze(0), [record])
    _, replay_before, _ = actor.evaluate_action_log_probs(*replay_inputs)
    _assert_probability_round_trip(sampled_log_prob, replay_before)

    execution_action = dict(action)
    execution_action["team"] = tuple(sorted(action["team"]))
    env.step(execution_action)
    assert task.assigned_team == list(execution_action["team"])
    assert set(task.assigned_team) == eligible_workers

    _, replay_after, _ = actor.evaluate_action_log_probs(*replay_inputs)
    _assert_probability_round_trip(sampled_log_prob, replay_after)


def test_postpone_replay_truncates_before_worker_head() -> None:
    env = _new_env()
    for task in env.get_ready_tasks():
        env._successors_map[task.aircraft_id][task.task_id] = []
    task = next(task for task in env.get_action_candidates() if env.can_postpone(task))
    env.get_action_candidates = lambda: [task]  # type: ignore[method-assign]

    actor = ActorCriticWork3(hidden_dim=32)
    actor.eval()
    with torch.no_grad():
        actor.branch_head[-1].bias[0] = -100.0
        actor.branch_head[-1].bias[1] = 100.0
    state_feat, urgency = _inputs()
    action, sampled_log_prob, _, record = actor.select_action(
        env, state_feat, urgency, deterministic=True
    )
    assert action is not None and action["branch"] == ActionBranch.POSTPONE
    assert "worker_valid_masks" not in record

    _, replay_log_prob, _ = actor.evaluate_action_log_probs(
        state_feat.unsqueeze(0), urgency.unsqueeze(0), [record]
    )
    _assert_probability_round_trip(sampled_log_prob, replay_log_prob)


def test_postpone_probability_and_entropy_ignore_dummy_team_and_alignment() -> None:
    """后移重放不得读取占位团队/对齐字段或未激活动作头。"""
    env = _new_env()
    for task in env.get_ready_tasks():
        env._successors_map[task.aircraft_id][task.task_id] = []
    task = next(task for task in env.get_action_candidates() if env.can_postpone(task))
    env.get_action_candidates = lambda: [task]  # type: ignore[method-assign]

    actor = ActorCriticWork3(hidden_dim=32)
    actor.eval()
    with torch.no_grad():
        for parameter in actor.branch_head.parameters():
            parameter.zero_()
        actor.branch_head[-1].bias[1] = 0.5
    state_feat, urgency = _inputs()
    action, sampled_log_prob, _, record = actor.select_action(
        env,
        state_feat,
        urgency,
        deterministic=True,
    )
    assert action is not None and action["branch"] == ActionBranch.POSTPONE

    _, base_log_prob, base_entropy = actor.evaluate_action_log_probs(
        state_feat.unsqueeze(0),
        urgency.unsqueeze(0),
        [record],
    )
    _assert_probability_round_trip(sampled_log_prob, base_log_prob)
    assert float(base_entropy[0]) > 0.0

    dummy_record = dict(record)
    dummy_record.update({
        "chosen_team": tuple(env.state.station_worker_bindings[task.current_station][:task.demand]),
        "chosen_worker_indices": tuple(range(task.demand)),
        "worker_valid_masks": tuple(
            tuple(True for _ in range(actor.max_station_workers))
            for _ in range(task.demand)
        ),
        "worker_node_indices": tuple(range(task.demand)),
        "station_worker_ids": tuple(env.state.station_worker_bindings[task.current_station]),
        "align": 1,
    })
    with torch.no_grad():
        for module in (actor.worker_score_fc, actor.worker_graph_score, actor.align_head):
            for parameter in module.parameters():
                parameter.add_(0.25)

    _, dummy_log_prob, dummy_entropy = actor.evaluate_action_log_probs(
        state_feat.unsqueeze(0),
        urgency.unsqueeze(0),
        [dummy_record],
    )
    assert torch.equal(base_log_prob, dummy_log_prob)
    assert torch.equal(base_entropy, dummy_entropy)


def test_explicit_advance_replay_has_no_worker_masks() -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    team = tuple(env.valid_team_completion_workers(task, [])[: task.demand])
    duration = env.duration_for_team(task, team)
    task.execution_duration = duration
    task.reserve(team=team, scheduled_start=20.0)
    for worker_id in team:
        env.state.workers[worker_id].add_interval(
            20.0, 20.0 + duration, task.task_key
        )
    env._station_occupied_tasks[task.current_station].add(task.task_key)
    env.event_queue.push(
        event_type=EventType.TASK_START,
        timestamp=20.0,
        task_key=task.task_key,
        generation=task.generation,
    )

    actor = ActorCriticWork3(hidden_dim=32)
    actor.eval()
    with torch.no_grad():
        actor.branch_head[-1].bias[0] = -100.0
        actor.branch_head[-1].bias[1] = 100.0
    state_feat, urgency = _inputs()
    action, sampled_log_prob, _, record = actor.select_action(
        env, state_feat, urgency, deterministic=True
    )
    assert action == {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    assert "worker_valid_masks" not in record

    _, replay_log_prob, _ = actor.evaluate_action_log_probs(
        state_feat.unsqueeze(0), urgency.unsqueeze(0), [record]
    )
    _assert_probability_round_trip(sampled_log_prob, replay_log_prob)
