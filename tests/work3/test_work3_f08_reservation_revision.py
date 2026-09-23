"""F08：未开工预约修订、相邻正式修订账本与推进动作。"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pytest
import torch

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from envs.work3.event_queue import EventType
from models.work3.actor_critic import ActorCriticWork3
from utils.work3.objective_evaluator import evaluate_trajectory_objective


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")


def _new_env() -> AirLineEnvWork3:
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    return env


def _legal_teams(env: AirLineEnvWork3, task) -> list[tuple[int, ...]]:
    workers = env.state.station_worker_bindings[task.current_station]
    teams: list[tuple[int, ...]] = []
    for team in combinations(workers, task.demand):
        try:
            env._validate_team_for_task(task, team)
        except ValueError:
            continue
        teams.append(tuple(team))
    assert teams, f"工序 {task.task_key} 没有可用测试团队"
    return teams


def _task_with_two_teams(env: AirLineEnvWork3):
    for task in env.get_ready_tasks():
        teams = _legal_teams(env, task)
        if len(teams) >= 2:
            return task, teams
        # 该真实实例首批任务的技能集合可能只有一个合法团队；此处只为
        # 验证A→B→A账本，构造一个仍受站位/人数约束的无技能测试工序。
        task.skill = -1
        task.demand = 1
        teams = _legal_teams(env, task)
        if len(teams) >= 2:
            return task, teams
    pytest.skip("当前实例没有两个不同的合法测试团队")


def _reserve_at_future_time(env: AirLineEnvWork3, task, team: tuple[int, ...]) -> None:
    task.in_station_offset = 20.0
    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 1,
        }
    )
    assert task.status == TaskStatus.RESERVED
    assert task.scheduled_start == pytest.approx(20.0)


def test_reserved_task_can_be_revised_and_old_start_event_is_invalidated() -> None:
    env = _new_env()
    task, teams = _task_with_two_teams(env)
    first_team = teams[0]
    second_team = teams[1]

    _reserve_at_future_time(env, task, first_team)
    old_generation = task.generation

    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": second_team,
            "align": 1,
        }
    )

    assert task.status == TaskStatus.RESERVED
    assert task.generation > old_generation
    assert task.assigned_team == list(second_team)
    assert len(task.revision_history) == (0 if second_team == first_team else 1)

    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
    assert task.status == TaskStatus.RUNNING
    assert task.actual_start == pytest.approx(20.0)
    assert task.assigned_team == list(second_team)


def test_team_revision_recomputes_reserved_duration_calendar_and_finish_event() -> None:
    """改队后预约占用与开工后完工事件使用新团队对应工时，基准工时不变。"""
    env = _new_env()
    task = next(
        task
        for task in env.state.tasks.values()
        if task.current_station == 4 and task.skill >= 0 and task.demand == 1
    )
    eligible_workers = [
        worker_id
        for worker_id in env.state.station_worker_bindings[task.current_station]
        if task.skill in env.worker_skills[worker_id]
    ]
    slow_worker = min(eligible_workers, key=env.worker_efficiencies.__getitem__)
    fast_worker = max(eligible_workers, key=env.worker_efficiencies.__getitem__)
    assert env.worker_efficiencies[slow_worker] < env.worker_efficiencies[fast_worker]

    env.state.aircraft[task.aircraft_id].current_station = task.current_station
    task.status = TaskStatus.READY
    for predecessor_id in task.predecessors:
        predecessor = env.state.tasks[f"{task.aircraft_id}_{predecessor_id}"]
        predecessor.status = TaskStatus.COMPLETED
        predecessor.actual_end = 0.0

    baseline_duration = task.duration
    slow_duration = env.duration_for_team(task, [slow_worker])
    fast_duration = env.duration_for_team(task, [fast_worker])
    assert slow_duration > fast_duration

    _reserve_at_future_time(env, task, (slow_worker,))
    assert task.execution_duration == pytest.approx(slow_duration)

    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": (fast_worker,),
            "align": 1,
        }
    )

    assert task.duration == baseline_duration
    assert task.execution_duration == pytest.approx(fast_duration)
    assert not any(
        interval.task_key == task.task_key
        for interval in env.state.workers[slow_worker].intervals
    )
    assert any(
        interval.task_key == task.task_key
        and interval.start == pytest.approx(20.0)
        and interval.end == pytest.approx(20.0 + fast_duration)
        for interval in env.state.workers[fast_worker].intervals
    )

    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
    finish_event = env.event_queue.peek()
    assert task.status == TaskStatus.RUNNING
    assert finish_event is not None
    assert finish_event.event_type == EventType.TASK_FINISH
    assert finish_event.timestamp == pytest.approx(20.0 + fast_duration)


def test_a_to_b_to_a_records_two_revisions_and_final_team_matches_baseline() -> None:
    env = _new_env()
    task, teams = _task_with_two_teams(env)
    first_team = tuple(task.base_team)
    if first_team not in teams:
        first_team = teams[0]
    second_team = next((team for team in teams if team != first_team), None)
    if second_team is None:
        pytest.skip("当前实例该任务没有第二个合法团队")

    _reserve_at_future_time(env, task, first_team)
    for team in (second_team, first_team):
        env.step(
            {
                "task_key": task.task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": team,
                "align": 1,
            }
        )

    assert [entry["after"]["team"] for entry in task.revision_history] == [
        list(second_team),
        list(first_team),
    ]
    assert len(task.revision_history) == 2
    assert all(entry["revision_cost"] > 0.0 for entry in task.revision_history)

    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
    assert task.status == TaskStatus.RUNNING
    assert set(task.assigned_team) == set(first_team)

    breakdown = evaluate_trajectory_objective(env, weights=env.weights)
    assert breakdown.num_revisions == 2
    assert breakdown.j_revision == pytest.approx(env.cost_revision)
    assert sum(env.step_rewards) == pytest.approx(-breakdown.j_total)


def test_advance_action_moves_to_next_event_without_arbitrary_wait() -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    team = _legal_teams(env, task)[0]
    duration = env.duration_for_team(task, team)
    task.execution_duration = duration
    task.reserve(team=team, scheduled_start=20.0)
    for worker_id in team:
        env.state.workers[worker_id].add_interval(
            start=20.0,
            end=20.0 + duration,
            task_key=task.task_key,
        )
    env._station_occupied_tasks[task.current_station].add(task.task_key)
    env.event_queue.push(
        event_type=EventType.TASK_START,
        timestamp=20.0,
        task_key=task.task_key,
        generation=task.generation,
    )

    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})

    assert env.state.current_time == pytest.approx(20.0)
    assert task.status == TaskStatus.RUNNING


def test_actor_can_select_advance_action_and_replay_its_probability() -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    team = _legal_teams(env, task)[0]
    duration = env.duration_for_team(task, team)
    task.execution_duration = duration
    task.reserve(team=team, scheduled_start=20.0)
    for worker_id in team:
        env.state.workers[worker_id].add_interval(
            start=20.0,
            end=20.0 + duration,
            task_key=task.task_key,
        )
    env._station_occupied_tasks[task.current_station].add(task.task_key)
    env.event_queue.push(
        event_type=EventType.TASK_START,
        timestamp=20.0,
        task_key=task.task_key,
        generation=task.generation,
    )

    actor = ActorCriticWork3()
    with torch.no_grad():
        actor.branch_head[-1].bias.data[0] = -100.0
        actor.branch_head[-1].bias.data[1] = 100.0

    state_feat = torch.randn(32)
    time_urgency = torch.tensor([0.0, 0.0])
    action, log_prob, _, record = actor.select_action(
        env, state_feat, time_urgency, deterministic=True
    )
    assert action == {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    _, replay_log_prob, _ = actor.evaluate_action_log_probs(
        state_feat.unsqueeze(0),
        time_urgency.unsqueeze(0),
        [record],
    )
    assert record["action_type"] == "advance_to_next_event"
    assert record["advance_choice"] == 1
    assert abs(float(replay_log_prob[0]) - log_prob) < 1e-6


def test_identical_resubmission_cannot_block_time_progress() -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    _reserve_at_future_time(env, task, _legal_teams(env, task)[0])
    team = tuple(task.assigned_team)

    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 1,
        }
    )

    assert env.state.current_time == pytest.approx(20.0)
    assert task.status == TaskStatus.RUNNING
    assert task.revision_history == []


@pytest.mark.parametrize("status", [TaskStatus.RUNNING, TaskStatus.COMPLETED])
def test_running_or_completed_task_cannot_be_revised(status: TaskStatus) -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    task.status = status
    if status == TaskStatus.RUNNING:
        task.actual_start = 0.0
    else:
        task.actual_end = 1.0

    assert env.can_reserve(task) is False
    assert env.can_postpone(task) is False
    with pytest.raises(ValueError):
        env.step(
            {
                "task_key": task.task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": _legal_teams(env, task)[0],
                "align": 0,
            }
        )
