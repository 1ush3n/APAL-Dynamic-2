"""F08：未开工预约修订、相邻正式修订账本与推进动作。"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any

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


def _revision_attempt_snapshot(env: AirLineEnvWork3) -> tuple[Any, ...]:
    return (
        env.state.snapshot(),
        {
            station_id: frozenset(task_keys)
            for station_id, task_keys in env._station_occupied_tasks.items()
        },
        tuple(
            (
                event.timestamp,
                event.priority,
                event.event_id,
                event.event_type,
                event.task_key,
                event.generation,
                deepcopy(event.payload),
                event.is_cancelled,
            )
            for event in env.event_queue._heap
        ),
        env.event_queue._event_counter,
        env.event_queue._current_time,
        dict(env.event_queue._task_generations),
        (
            env.cumulative_cost,
            env.cost_takt,
            env.cost_time,
            env.cost_team,
            env.cost_postpone,
            env.cost_revision,
        ),
        env.step_count,
        tuple(env.step_rewards),
        env._transfer_scheduled_for_cycle,
    )


def test_reserved_task_can_be_revised_and_old_start_event_is_invalidated() -> None:
    env = _new_env()
    task, teams = _task_with_two_teams(env)
    first_team = teams[0]
    second_team = teams[1]

    _reserve_at_future_time(env, task, first_team)
    history_size = len(task.revision_history)
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
    assert len(task.revision_history) == history_size + int(
        set(second_team) != set(first_team)
    )

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
    history_size = len(task.revision_history)
    for team in (second_team, first_team):
        env.step(
            {
                "task_key": task.task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": team,
                "align": 1,
            }
        )

    assert [entry["after"]["team"] for entry in task.revision_history[-2:]] == [
        list(second_team),
        list(first_team),
    ]
    assert len(task.revision_history) == history_size + 2
    assert all(entry["revision_cost"] > 0.0 for entry in task.revision_history[-2:])

    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
    assert task.status == TaskStatus.RUNNING
    assert set(task.assigned_team) == set(first_team)

    breakdown = evaluate_trajectory_objective(env, weights=env.weights)
    assert breakdown.num_revisions == history_size + 2
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
    team = _legal_teams(env, task)[0]
    task.base_team = tuple(team)
    task.baseline_assignment = {
        "station": task.current_station,
        "team": list(team),
        "position": 20.0,
    }
    task.last_published_assignment = task.baseline_assignment.copy()
    _reserve_at_future_time(env, task, team)

    def effective_event_ledger() -> list[tuple[float, str, str | None, int]]:
        return sorted(
            (
                event.timestamp,
                event.event_type.name,
                event.task_key,
                event.generation,
            )
            for event in env.event_queue._heap
            if env.event_queue._is_event_valid(event)
        )

    assert len(env.event_queue._heap) == 1
    event_ledger_before = effective_event_ledger()
    assert event_ledger_before == [
        (20.0, EventType.TASK_START.name, task.task_key, task.generation)
    ]
    task_snapshot_before = (
        task.generation,
        task.scheduled_start,
        task.execution_duration,
        tuple(task.assigned_team),
        deepcopy(task.last_published_assignment),
        deepcopy(task.revision_history),
    )
    worker_calendars_before = {
        worker_id: tuple(
            (interval.start, interval.end, interval.task_key)
            for interval in worker.intervals
        )
        for worker_id, worker in env.state.workers.items()
    }
    station_occupancy_before = {
        station_id: frozenset(task_keys)
        for station_id, task_keys in env._station_occupied_tasks.items()
    }
    fee_ledger_before = (
        env.cumulative_cost,
        env.cost_takt,
        env.cost_time,
        env.cost_team,
        env.cost_postpone,
        env.cost_revision,
    )
    rewards_before = len(env.step_rewards)

    _, reward, terminated, truncated, _ = env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 1,
        }
    )

    assert env.state.current_time == pytest.approx(20.0)
    assert task.status == TaskStatus.RUNNING
    assert not terminated
    assert not truncated
    assert reward == pytest.approx(0.0)
    assert task.actual_start == pytest.approx(20.0)
    assert (
        task.generation,
        task.scheduled_start,
        task.execution_duration,
        tuple(task.assigned_team),
        task.last_published_assignment,
        task.revision_history,
    ) == task_snapshot_before
    assert {
        worker_id: tuple(
            (interval.start, interval.end, interval.task_key)
            for interval in worker.intervals
        )
        for worker_id, worker in env.state.workers.items()
    } == worker_calendars_before
    assert {
        station_id: frozenset(task_keys)
        for station_id, task_keys in env._station_occupied_tasks.items()
    } == station_occupancy_before
    assert effective_event_ledger() == [
        (
            20.0 + float(task.execution_duration),
            EventType.TASK_FINISH.name,
            task.task_key,
            task.generation,
        )
    ]
    assert fee_ledger_before == (
        env.cumulative_cost,
        env.cost_takt,
        env.cost_time,
        env.cost_team,
        env.cost_postpone,
        env.cost_revision,
    )
    assert len(env.step_rewards) == rewards_before + 1


def test_cancelled_reservation_keeps_revision_anchor_for_next_publication() -> None:
    env = _new_env()
    task, teams = _task_with_two_teams(env)
    first_team = tuple(task.base_team)
    if first_team not in teams:
        first_team = teams[0]
    second_team = next(team for team in teams if team != first_team)

    _reserve_at_future_time(env, task, first_team)
    previous_assignment = task.last_published_assignment.copy()
    history_size = len(task.revision_history)
    old_start = task.scheduled_start

    env._handle_disturbance_event(
        1.0,
        {"recovery_time": 30.0, "affected_task_keys": [task.task_key]},
    )
    assert task.status == TaskStatus.UNREADY
    assert task.last_published_assignment == previous_assignment

    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": second_team,
            "align": 0,
        }
    )

    assert task.scheduled_start is not None
    assert task.scheduled_start >= 30.0
    assert len(task.revision_history) == history_size + 1
    revision = task.revision_history[-1]
    assert revision["before"]["team"] == list(first_team)
    assert revision["after"]["team"] == list(second_team)
    assert revision["team_change"] == pytest.approx(1.0 / task.demand)

    env.state.current_time = float(old_start)
    env.process_due_events()
    assert task.status == TaskStatus.RESERVED
    assert task.actual_start is None


def test_postpone_preserves_last_published_team_and_position_as_pending_reference() -> None:
    env = _new_env()
    task = next(task for task in env.state.tasks.values() if env.can_postpone(task))
    for predecessor_id in task.predecessors:
        predecessor = env.state.tasks[f"{task.aircraft_id}_{predecessor_id}"]
        predecessor.status = TaskStatus.COMPLETED
        predecessor.actual_end = 0.0
    task.status = TaskStatus.READY
    team = _legal_teams(env, task)[0]
    _reserve_at_future_time(env, task, team)
    original_station = task.current_station
    original_generation = task.generation
    original_postpone_count = task.postpone_count
    previous_assignment = task.last_published_assignment.copy()
    revision_cost_before = env.cost_revision

    env.step({"task_key": task.task_key, "branch": ActionBranch.POSTPONE})

    assert task.current_station == original_station + 1
    assert task.current_station == previous_assignment["station"] + 1
    assert task.status == TaskStatus.POSTPONED
    assert task.assigned_team == []
    assert task.scheduled_start is None
    assert task.execution_duration is None
    assert task.postpone_count == original_postpone_count + 1
    assert task.generation == original_generation + 1
    assert task.last_published_assignment["team"] == previous_assignment["team"]
    assert task.last_published_assignment["position"] == pytest.approx(
        previous_assignment["position"]
    )
    assert all(
        interval.task_key != task.task_key
        for worker in env.state.workers.values()
        for interval in worker.intervals
    )
    assert task.task_key not in env._station_occupied_tasks[original_station]
    assert task.task_key not in env._station_occupied_tasks[task.current_station]
    assert not any(
        event.task_key == task.task_key and env.event_queue._is_event_valid(event)
        for event in env.event_queue._heap
    )
    assert env.cost_revision == pytest.approx(revision_cost_before)
    assert env.cost_postpone > 0.0


@pytest.mark.parametrize(
    ("replacement_count", "expected_replacement"),
    [(1, 1.0 / 3.0), (3, 1.0)],
)
def test_three_person_team_replacement_ratio_updates_revision_and_final_cost(
    replacement_count: int, expected_replacement: float
) -> None:
    env = _new_env()
    task = env.state.tasks["0_34"]
    env.state.aircraft[task.aircraft_id].current_station = task.current_station
    for predecessor_id in task.predecessors:
        predecessor = env.state.tasks[f"{task.aircraft_id}_{predecessor_id}"]
        predecessor.status = TaskStatus.COMPLETED
        predecessor.actual_end = 0.0
    task.status = TaskStatus.READY
    task.skill = -1  # 隔离团队替换比例账本；技能合法性由D组独立验收。

    workers = env.state.station_worker_bindings[task.current_station]
    team_a = tuple(workers[: task.demand])
    team_b = next(
        team
        for team in _legal_teams(env, task)
        if len(set(team_a) & set(team)) == task.demand - replacement_count
    )
    assert task.demand == 3
    assert len(env.state.tasks) == 2830
    task.base_team = team_a
    task.in_station_offset = 20.0
    task.baseline_assignment = {
        "station": task.current_station,
        "team": list(team_a),
        "position": 20.0,
    }
    task.last_published_assignment = task.baseline_assignment.copy()
    scale = 1.0 / 2830.0
    team_cost_before = env.cost_team
    revision_cost_before = env.cost_revision

    _reserve_at_future_time(env, task, team_a)
    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team_b,
            "align": 1,
        }
    )

    assert len(task.revision_history) == 1
    assert task.revision_history[0]["team_change"] == pytest.approx(
        expected_replacement
    )
    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})

    assert task.status == TaskStatus.RUNNING
    assert env.cost_team - team_cost_before == pytest.approx(
        env.weights.w_w * expected_replacement * scale
    )
    assert env.cost_revision - revision_cost_before == pytest.approx(
        env.weights.w_w * expected_replacement * scale
    )
    breakdown = evaluate_trajectory_objective(env, weights=env.weights)
    assert breakdown.d_team == pytest.approx(expected_replacement * scale)
    assert breakdown.j_revision == pytest.approx(env.cost_revision)
    assert sum(env.step_rewards) == pytest.approx(-breakdown.j_total)


def test_team_member_order_is_not_a_formal_revision_or_team_deviation() -> None:
    env = _new_env()
    task = next(
        task
        for task in env.get_ready_tasks()
        if task.demand >= 2 and _legal_teams(env, task)
    )
    team = _legal_teams(env, task)[0]
    task.base_team = tuple(team)
    task.baseline_assignment = {
        "station": task.current_station,
        "team": list(team),
        "position": 20.0,
    }
    task.last_published_assignment = task.baseline_assignment.copy()
    _reserve_at_future_time(env, task, team)
    history_size = len(task.revision_history)
    revision_cost_before = env.cost_revision
    team_cost_before = env.cost_team

    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": tuple(reversed(team)),
            "align": 1,
        }
    )

    assert len(task.revision_history) == history_size
    assert env.cost_revision == pytest.approx(revision_cost_before)
    assert task.status == TaskStatus.RUNNING
    assert set(task.assigned_team) == set(team)
    assert env.cost_team == pytest.approx(team_cost_before)
    breakdown = evaluate_trajectory_objective(env, weights=env.weights)
    assert breakdown.d_team == pytest.approx(0.0)


def test_exact_baseline_publication_has_no_phantom_time_revision() -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    team = _legal_teams(env, task)[0]
    task.base_team = tuple(team)
    task.in_station_offset = 0.0
    task.baseline_assignment = {
        "station": task.current_station,
        "team": list(team),
        "position": 0.0,
    }
    task.last_published_assignment = task.baseline_assignment.copy()

    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 0,
        }
    )

    assert task.scheduled_start == pytest.approx(0.0)
    assert task.revision_history == []
    assert env.cost_revision == pytest.approx(0.0)


def test_first_deviating_publication_is_compared_with_p0() -> None:
    env = _new_env()
    task, teams = _task_with_two_teams(env)
    first_team = tuple(task.base_team)
    if first_team not in teams:
        first_team = teams[0]
    second_team = next(team for team in teams if team != first_team)
    task.baseline_assignment = {
        "station": task.current_station,
        "team": list(first_team),
        "position": float(task.in_station_offset),
    }
    task.last_published_assignment = task.baseline_assignment.copy()

    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": second_team,
            "align": 1,
        }
    )

    assert len(task.revision_history) == 1
    revision = task.revision_history[0]
    assert revision["before"]["team"] == list(first_team)
    assert revision["after"]["team"] == list(second_team)
    assert revision["team_change"] == pytest.approx(1.0 / task.demand)


def test_three_person_a_to_b_to_a_charges_each_revision_and_zero_final_team_deviation() -> None:
    env = _new_env()
    task = env.state.tasks["0_34"]
    env.state.aircraft[task.aircraft_id].current_station = task.current_station
    for predecessor_id in task.predecessors:
        predecessor = env.state.tasks[f"{task.aircraft_id}_{predecessor_id}"]
        predecessor.status = TaskStatus.COMPLETED
        predecessor.actual_end = 0.0
    task.status = TaskStatus.READY

    assert task.demand == 3
    team_a = _legal_teams(env, task)[0]
    task.base_team = team_a
    team_b = next(
        team
        for team in _legal_teams(env, task)
        if len(set(team_a) & set(team)) == 2
    )
    task.in_station_offset = 20.0
    task.baseline_assignment = {
        "station": task.current_station,
        "team": list(team_a),
        "position": 20.0,
    }
    task.last_published_assignment = task.baseline_assignment.copy()
    team_cost_before = env.cost_team

    _reserve_at_future_time(env, task, team_a)
    assert task.revision_history == []
    for team in (team_b, team_a):
        env.step(
            {
                "task_key": task.task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": team,
                "align": 1,
            }
        )

    assert len(task.revision_history) == 2
    assert [revision["team_change"] for revision in task.revision_history] == pytest.approx(
        [1.0 / 3.0, 1.0 / 3.0]
    )
    assert set(task.last_published_assignment["team"]) == set(team_a)

    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})

    assert task.status == TaskStatus.RUNNING
    assert set(task.assigned_team) == set(team_a)
    assert env.cost_team == pytest.approx(team_cost_before)
    breakdown = evaluate_trajectory_objective(env, weights=env.weights)
    assert breakdown.d_team == pytest.approx(0.0)
    assert breakdown.j_revision == pytest.approx(env.cost_revision)
    assert sum(env.step_rewards) == pytest.approx(-breakdown.j_total)


def test_postponement_keeps_team_anchor_until_target_station_publication() -> None:
    env = _new_env()
    task = env.state.tasks["0_16"]
    env.state.aircraft[task.aircraft_id].current_station = task.current_station
    for predecessor_id in task.predecessors:
        predecessor = env.state.tasks[f"{task.aircraft_id}_{predecessor_id}"]
        predecessor.status = TaskStatus.COMPLETED
        predecessor.actual_end = 0.0
    task.status = TaskStatus.READY
    team_a = _legal_teams(env, task)[0]
    _reserve_at_future_time(env, task, team_a)
    history_size = len(task.revision_history)
    previous_cost_revision = env.cost_revision

    env.step({"task_key": task.task_key, "branch": ActionBranch.POSTPONE})
    assert env.cost_revision == pytest.approx(previous_cost_revision)
    assert task.last_published_assignment["team"] == list(team_a)
    assert task.last_published_assignment["position"] == pytest.approx(20.0)

    for station_id in range(env.state.num_stations):
        for other_task in env.state.get_tasks_for_station(station_id):
            if other_task.task_key != task.task_key:
                other_task.status = TaskStatus.COMPLETED
                other_task.actual_end = 0.0
    env._check_and_schedule_transfer()
    env.process_due_events()
    assert env.state.aircraft[task.aircraft_id].current_station == task.current_station
    assert task.status == TaskStatus.READY

    team_b = _legal_teams(env, task)[0]
    replacement = 1.0 - len(set(team_a) & set(team_b)) / task.demand
    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team_b,
            "align": 1,
        }
    )
    assert len(task.revision_history) == history_size + 2
    revision = task.revision_history[-1]
    assert revision["before"]["team"] == list(team_a)
    assert revision["before"]["position"] == pytest.approx(20.0)
    assert revision["after"]["team"] == list(team_b)
    assert revision["team_change"] == pytest.approx(replacement)
    breakdown = evaluate_trajectory_objective(env, weights=env.weights)
    assert breakdown.j_revision == pytest.approx(env.cost_revision)
    assert sum(env.step_rewards) == pytest.approx(-breakdown.j_total)


@pytest.mark.parametrize("status", [TaskStatus.RUNNING, TaskStatus.COMPLETED])
def test_running_or_completed_task_revisions_are_rejected_without_side_effects(
    status: TaskStatus,
) -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    team = _legal_teams(env, task)[0]
    duration = env.duration_for_team(task, team)
    task.assigned_team = list(team)
    task.scheduled_start = 0.0
    task.execution_duration = duration
    task.start_cost_confirmed = True
    task.actual_start = 0.0
    for worker_id in team:
        env.state.workers[worker_id].add_interval(
            start=0.0,
            end=duration,
            task_key=task.task_key,
        )
    task.status = status
    if status == TaskStatus.RUNNING:
        env._station_occupied_tasks[task.current_station].add(task.task_key)
        env.event_queue.push(
            event_type=EventType.TASK_FINISH,
            timestamp=duration,
            task_key=task.task_key,
            generation=task.generation,
        )
    else:
        task.actual_end = duration

    assert env.can_reserve(task) is False
    assert env.can_postpone(task) is False
    for branch in (ActionBranch.STATION_EXECUTE, ActionBranch.POSTPONE):
        before = _revision_attempt_snapshot(env)
        action = {"task_key": task.task_key, "branch": branch}
        if branch == ActionBranch.STATION_EXECUTE:
            action.update({"team": team, "align": 0})
        with pytest.raises(ValueError):
            env.step(action)
        assert _revision_attempt_snapshot(env) == before
