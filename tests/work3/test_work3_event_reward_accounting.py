"""R02：事件推进必须经由有奖励和终止语义的环境转移。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

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


def test_step_reward_includes_costs_until_next_legal_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """合法候选恢复前的预约开工成本应归入触发自动推进的step。"""
    env = _new_env()
    ready = env.get_ready_tasks()
    action_task, future_task = ready[:2]
    station_workers = env.state.station_worker_bindings[0]
    action_worker, future_worker = station_workers

    for task in (action_task, future_task):
        task.demand = 1
        task.skill = 0
        task.in_station_offset = 0.0

    duration = env.duration_for_team(future_task, (future_worker,))
    future_task.execution_duration = duration
    future_task.reserve((future_worker,), scheduled_start=5.0)
    env.state.workers[future_worker].add_interval(
        5.0, 5.0 + duration, future_task.task_key
    )
    env._station_occupied_tasks[0].add(future_task.task_key)
    env.event_queue.push(
        event_type=EventType.TASK_START,
        timestamp=5.0,
        task_key=future_task.task_key,
        generation=future_task.generation,
    )

    actual_candidates = env.get_action_candidates

    def candidates_after_recovery():
        if env.state.current_time < 5.0 - env.tolerance:
            return []
        return actual_candidates()

    monkeypatch.setattr(env, "get_action_candidates", candidates_after_recovery)
    initial_cost = env.cumulative_cost
    _obs, reward, terminated, truncated, info = env.step(
        {
            "task_key": action_task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": (action_worker,),
            "align": 0,
        }
    )

    assert future_task.status.name == "RUNNING"
    assert future_task.actual_start == pytest.approx(5.0)
    assert env.state.current_time == pytest.approx(5.0)
    assert env.cumulative_cost - initial_cost > 0.0
    assert info["step_cost"] == pytest.approx(env.cumulative_cost - initial_cost)
    assert reward == pytest.approx(-info["step_cost"])
    assert terminated is False
    assert truncated is False


def test_actor_emits_probability_one_forced_advance_when_no_action_exists() -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    team = tuple(env.valid_team_completion_workers(task, [])[: task.demand])
    task.in_station_offset = 0.0
    duration = env.duration_for_team(task, team)
    task.execution_duration = duration
    task.reserve(team, scheduled_start=1.0)
    for worker_id in team:
        env.state.workers[worker_id].add_interval(
            1.0, 1.0 + duration, task.task_key
        )
    env._station_occupied_tasks[task.current_station].add(task.task_key)
    env.event_queue.push(
        event_type=EventType.TASK_START,
        timestamp=1.0,
        task_key=task.task_key,
        generation=task.generation,
    )
    env.get_action_candidates = lambda: []  # type: ignore[method-assign]
    actor = ActorCriticWork3(hidden_dim=32)
    actor.eval()
    state_feat = torch.zeros(32)
    time_urgency = torch.zeros(2)

    action, log_prob, value, record = actor.select_action(
        env, state_feat, time_urgency, deterministic=True
    )

    assert action == {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    assert log_prob == pytest.approx(0.0)
    assert record["action_type"] == "forced_advance"
    assert "worker_valid_masks" not in record
    assert torch.isfinite(torch.tensor(value))

    initial_cost = env.cumulative_cost
    _obs, reward, terminated, truncated, info = env.step(action)
    assert task.status.name == "RUNNING"
    assert task.actual_start == pytest.approx(1.0)
    assert env.cumulative_cost - initial_cost > 0.0
    assert info["step_cost"] == pytest.approx(env.cumulative_cost - initial_cost)
    assert reward == pytest.approx(-info["step_cost"])
    assert terminated is False
    assert truncated is False

    _values, replay_log_prob, entropy = actor.evaluate_action_log_probs(
        state_feat.unsqueeze(0), time_urgency.unsqueeze(0), [record]
    )
    assert replay_log_prob.item() == pytest.approx(0.0)
    assert entropy.item() == pytest.approx(0.0)


def test_unchanged_reservation_without_valid_event_terminates_as_deadlock() -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    task.in_station_offset = 20.0
    team = tuple(env.valid_team_completion_workers(task, [])[: task.demand])
    action = {
        "task_key": task.task_key,
        "branch": ActionBranch.STATION_EXECUTE,
        "team": team,
        "align": 1,
    }
    env.step(action)
    assert task.scheduled_start == pytest.approx(20.0)

    # 模拟事件已失效/被撤销，但保留物理预约，重提同一安排不得空转。
    env.event_queue.reset(start_time=env.state.current_time)
    _obs, _reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert truncated is False
    assert info["success"] is False
    assert info["termination_reason"] == "deadlock"
