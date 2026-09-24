"""R05 图状态的预约、发布安排、未来日历与约束掩码反例。"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from pathlib import Path

import pytest
import torch

from envs.work3.core_types import TaskStatus, TimeInterval
from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import (
    ActorCriticWork3,
    extract_compact_state_features,
)
from models.work3.graph_builder import MultiAircraftGraphBuilder
from models.work3.time_head import TimeResidualHead
from models.work3.ppo_trainer import PPOTrainerWork3
from scripts.work3.evaluate_c_vs_d import build_formal_evaluation_agent
from scripts.work3.train_ppo_work3 import compute_online_time_inputs
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


@pytest.fixture
def baseline_path() -> Path:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return path


def _make_env(baseline_path: Path) -> AirLineEnvWork3:
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    env.reset()
    env.state.current_time = 10.0
    return env


def _reservable_task(env: AirLineEnvWork3):
    return next(
        task
        for task in env.get_action_candidates()
        if env.can_reserve(task) and task.current_station < env.state.num_stations - 1
    )


def _set_future_reservation(
    env: AirLineEnvWork3,
    task_key: str,
    start: float,
) -> None:
    task = env.state.tasks[task_key]
    team = tuple(task.base_team)
    assert len(team) == task.demand
    task.status = TaskStatus.RESERVED
    task.assigned_team = list(team)
    task.scheduled_start = float(start)
    task.execution_duration = 2.0
    for worker_id in team:
        calendar = env.state.workers[worker_id]
        calendar.intervals = [
            interval for interval in calendar.intervals
            if interval.task_key != task_key
        ]
        calendar.intervals.append(TimeInterval(start, start + 2.0, task_key))
        calendar.intervals.sort(key=lambda interval: interval.start)


def _assert_graph_equal(left, right) -> None:
    assert set(left.node_types) == set(right.node_types)
    assert set(left.edge_types) == set(right.edge_types)
    for node_type in left.node_types:
        assert torch.equal(left[node_type].x, right[node_type].x)
    for edge_type in left.edge_types:
        assert torch.equal(left[edge_type].edge_index, right[edge_type].edge_index)


def test_future_reservation_times_and_replay_snapshot_are_distinguishable(
    baseline_path: Path,
) -> None:
    env = _make_env(baseline_path)
    task = _reservable_task(env)
    _set_future_reservation(env, task.task_key, 20.0)

    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    first_graph = actor.build_graph_snapshot(env)
    task_index = actor._get_graph_builder(env).task_key_to_idx[task.task_key]
    first_task_features = first_graph["task"].x[task_index]

    # 新图契约：18为is_reserved，19为has_schedule，20为开工剩余时间/H0，
    # 21为has_schedule_end，22为预计完工剩余时间/H0。
    assert first_task_features.shape == (26,)
    assert first_task_features[18:23].tolist() == pytest.approx(
        [1.0, 1.0, 10.0 / env.state.h0, 1.0, 12.0 / env.state.h0]
    )
    worker_index = actor._get_graph_builder(env).worker_id_to_idx[task.assigned_team[0]]
    worker_features = first_graph["worker"].x[worker_index]
    assert worker_features.shape == (21,)
    assert worker_features[18].item() == 1.0
    assert worker_features[19:21].tolist() == pytest.approx(
        [10.0 / env.state.h0, 12.0 / env.state.h0]
    )

    cmax = 100.0
    state_feat = extract_compact_state_features(env.state, cmax)
    urgency = torch.zeros(2)
    _, old_log_prob, _, record = actor.select_action(
        env,
        state_feat,
        urgency,
        deterministic=True,
    )
    _set_future_reservation(env, task.task_key, 25.0)
    second_graph = actor.build_graph_snapshot(env)
    second_task_features = second_graph["task"].x[task_index]
    assert second_task_features[20].item() == pytest.approx(15.0 / env.state.h0)
    assert not torch.equal(first_task_features, second_task_features)

    _, replay_log_prob, _ = actor.evaluate_action_log_probs(
        state_feat.unsqueeze(0),
        urgency.unsqueeze(0),
        [record],
    )
    assert replay_log_prob.item() == pytest.approx(old_log_prob, abs=1e-6)


def test_previous_published_team_is_separate_from_p0_and_changes_worker_input(
    baseline_path: Path,
) -> None:
    env = _make_env(baseline_path)
    task = next(
        t
        for t in env.state.tasks.values()
        if 0 < t.demand < len(env.state.station_worker_bindings[t.current_station])
    )
    team_a = tuple(task.base_team)
    station_workers = env.state.station_worker_bindings[task.current_station]
    extra_worker = next(w for w in station_workers if w not in team_a)
    env.worker_skills[extra_worker] = frozenset((*env.worker_skills[extra_worker], task.skill))
    team_b = (extra_worker, *team_a[1:])
    assert _team_is_valid(env, task, team_b)
    published_a = deepcopy(task.last_published_assignment)
    published_b = deepcopy(task.last_published_assignment)
    published_a["team"] = list(team_a)
    published_b["team"] = list(team_b)
    task.last_published_assignment = published_a
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    graph_a = actor.build_graph_snapshot(env)

    task.last_published_assignment = published_b
    graph_b = actor.build_graph_snapshot(env)
    task_index = actor._get_graph_builder(env).task_key_to_idx[task.task_key]

    assert torch.equal(
        graph_a["task", "baseline_team", "worker"].edge_index,
        graph_b["task", "baseline_team", "worker"].edge_index,
    )
    assert not torch.equal(
        graph_a["task", "last_published_team", "worker"].edge_index,
        graph_b["task", "last_published_team", "worker"].edge_index,
    )
    assert graph_a["task"].x[task_index, 23:26].tolist() == pytest.approx(
        graph_b["task"].x[task_index, 23:26].tolist()
    )

    action_after_a = {**published_a, "team": list(team_a)}
    change_a = env._revision_cost_components(published_a, action_after_a, task)[1]
    change_b = env._revision_cost_components(published_b, action_after_a, task)[1]
    assert change_a == pytest.approx(0.0)
    assert change_b > 0.0

    _, _, worker_nodes_a = actor.graph_encoder(graph_a)
    _, _, worker_nodes_b = actor.graph_encoder(graph_b)
    assert not torch.equal(worker_nodes_a, worker_nodes_b)


def _team_is_valid(env: AirLineEnvWork3, task, team: tuple[int, ...]) -> bool:
    try:
        env._validate_team_for_task(task, team)
    except ValueError:
        return False
    return True


def test_unrevealed_future_disturbance_does_not_change_graph_or_time_inputs(
    baseline_path: Path,
) -> None:
    env_a = _make_env(baseline_path)
    env_b = _make_env(baseline_path)
    task_key = _reservable_task(env_a).task_key
    env_b.state.tasks[task_key].task_key
    env_a.load_scenario({
        "scenario_id": "UNREVEALED_A",
        "tau": 30.0,
        "recovery_time": 50.0,
        "affected_task_keys": [task_key],
    })
    env_b.load_scenario({
        "scenario_id": "UNREVEALED_B",
        "tau": 40.0,
        "recovery_time": 80.0,
        "affected_task_keys": [task_key],
    })

    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    time_head = TimeResidualHead(in_dim=16, hidden_dim=16)
    actor.eval()
    time_head.eval()
    cmax_a = 100.0
    cmax_b = 100.0
    state_a = extract_compact_state_features(env_a.state, cmax_a)
    state_b = extract_compact_state_features(env_b.state, cmax_b)
    graph_a, urgency_a, prediction_a = compute_online_time_inputs(
        actor, time_head, env_a, state_a, cmax_a
    )
    graph_b, urgency_b, prediction_b = compute_online_time_inputs(
        actor, time_head, env_b, state_b, cmax_b
    )

    _assert_graph_equal(graph_a, graph_b)
    assert torch.equal(state_a, state_b)
    assert torch.equal(urgency_a, urgency_b)
    assert torch.equal(prediction_a, prediction_b)


def test_graph_action_helpers_match_environment_for_physical_masks(
    baseline_path: Path,
) -> None:
    env = _make_env(baseline_path)
    baseline = MultiAircraftBaseline.load_from_json(str(baseline_path))
    builder = MultiAircraftGraphBuilder(baseline)

    fixed_task = _reservable_task(env)
    fixed_task.fixed_station = fixed_task.current_station
    assert env.get_action_branch_mask(fixed_task) == (True, False)
    fixed_idx = builder.task_key_to_idx[fixed_task.task_key]
    assert builder.get_action_branch_mask(env, fixed_idx) == {
        "can_stay": True,
        "can_postpone": False,
    }

    task = next(
        task
        for task in env.get_action_candidates()
        if env.can_reserve(task)
        and any(
            env.state.tasks[f"{task.aircraft_id}_{successor_id}"].current_station
            == task.current_station
            and env.state.tasks[f"{task.aircraft_id}_{successor_id}"].status
            not in (TaskStatus.COMPLETED, TaskStatus.POSTPONED)
            for successor_id in env._successors_map[task.aircraft_id][task.task_id]
        )
    )
    assert env.get_action_branch_mask(task) == (True, False)
    task_idx = builder.task_key_to_idx[task.task_key]
    assert builder.get_action_branch_mask(env, task_idx) == {
        "can_stay": True,
        "can_postpone": False,
    }

    revised_task = next(
        task for task in env.get_action_candidates() if env.can_reserve(task)
    )
    _set_future_reservation(env, revised_task.task_key, 20.0)
    assert revised_task.status == TaskStatus.RESERVED
    reserve_mask = env.get_action_branch_mask(revised_task)
    revised_idx = builder.task_key_to_idx[revised_task.task_key]
    assert builder.get_action_branch_mask(env, revised_idx) == {
        "can_stay": reserve_mask[0],
        "can_postpone": reserve_mask[1],
    }
    assert builder.get_action_candidate_indices(env) == [
        builder.task_key_to_idx[item.task_key]
        for item in env.get_action_candidates()
    ]


def test_checkpoints_declare_graph_feature_schema_and_reject_old_schema(
    tmp_path: Path,
) -> None:
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    head = TimeResidualHead(in_dim=64, hidden_dim=64)
    trainer = PPOTrainerWork3(actor_critic=actor, time_head=head)
    checkpoint_path = tmp_path / "current.pt"
    trainer.save_checkpoint(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["graph_feature_version"] == "work3_graph_v2"
    assert checkpoint["graph_feature_dims"] == {
        "task": 26,
        "worker": 21,
        "station": 15,
        "skill": 11,
    }
    assert "task_features" in checkpoint["graph_feature_schema"]
    assert "worker_features" in checkpoint["graph_feature_schema"]

    old_path = tmp_path / "legacy_graph.pt"
    old_checkpoint = {
        "actor_critic_state": actor.state_dict(),
        "time_head_state": head.state_dict(),
        "time_head_model_version": "signed_residual_v1",
        "graph_feature_version": "work3_graph_v1",
        "graph_feature_dims": {"task": 18, "worker": 17, "station": 15, "skill": 11},
    }
    torch.save(old_checkpoint, old_path)
    with pytest.raises(ValueError, match="图特征"):
        build_formal_evaluation_agent("D", old_path, device="cpu")
