"""C组：预约前驱传播、到期开工复核与分支概率边界。"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import pytest
import torch

from envs.work3.core_types import ActionBranch, TaskRuntimeState, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from envs.work3.event_queue import EventType
from models.work3.actor_critic import ActorCriticWork3


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")


def _new_env() -> AirLineEnvWork3:
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    env.state.aircraft[0].current_station = 4
    env.event_queue.reset(start_time=0.0)
    return env


def _team_is_valid(
    env: AirLineEnvWork3,
    task: TaskRuntimeState,
    team: tuple[int, ...],
) -> bool:
    try:
        env._validate_team_for_task(task, team)
    except ValueError:
        return False
    return True


def _reserve_with_event(
    env: AirLineEnvWork3,
    task: TaskRuntimeState,
    *,
    start: float,
    duration: float,
    excluded_workers: set[int],
    selected_team: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    candidates = env.valid_team_completion_workers(
        task,
        [],
        station_id=task.current_station,
    )
    workers = [worker_id for worker_id in candidates if worker_id not in excluded_workers]
    assert len(workers) >= task.demand
    team = tuple(workers[: task.demand]) if selected_team is None else selected_team
    assert len(team) == task.demand
    assert set(team).issubset(workers)
    task.reserve(team=team, scheduled_start=start)
    task.execution_duration = duration
    for worker_id in team:
        env.state.workers[worker_id].add_interval(
            start=start,
            end=start + duration,
            task_key=task.task_key,
        )
    env._station_occupied_tasks[task.current_station].add(task.task_key)
    env.event_queue.push(
        event_type=EventType.TASK_START,
        timestamp=start,
        task_key=task.task_key,
        generation=task.generation,
    )
    return team


def _complete_non_chain_predecessors(
    env: AirLineEnvWork3,
    task: TaskRuntimeState,
    chain_keys: set[str],
) -> None:
    for predecessor_id in task.predecessors:
        predecessor = env.state.tasks[f"{task.aircraft_id}_{predecessor_id}"]
        if predecessor.task_key in chain_keys:
            continue
        predecessor.status = TaskStatus.COMPLETED
        predecessor.actual_end = 0.0


def _install_three_task_chain(
    env: AirLineEnvWork3,
    *,
    include_child_of_child: bool,
) -> tuple[TaskRuntimeState, TaskRuntimeState, TaskRuntimeState | None]:
    parent = env.state.tasks["0_165"]
    child = env.state.tasks["0_166"]
    grandchild = env.state.tasks["0_172"] if include_child_of_child else None
    chain_keys = {parent.task_key, child.task_key}
    if grandchild is not None:
        chain_keys.add(grandchild.task_key)

    _complete_non_chain_predecessors(env, parent, chain_keys)
    _complete_non_chain_predecessors(env, child, chain_keys)
    if grandchild is not None:
        _complete_non_chain_predecessors(env, grandchild, chain_keys)

    used_workers: set[int] = set()
    parent_team = _reserve_with_event(
        env, parent, start=8.0, duration=2.0, excluded_workers=used_workers
    )
    used_workers.update(parent_team)
    child_team = _reserve_with_event(
        env, child, start=10.0, duration=2.0, excluded_workers=used_workers
    )
    used_workers.update(child_team)
    if grandchild is not None:
        grandchild_team = _reserve_with_event(
            env,
            grandchild,
            start=12.0,
            duration=2.0,
            excluded_workers=used_workers,
        )
        used_workers.update(grandchild_team)
    return parent, child, grandchild


def _assert_task_resources_released(env: AirLineEnvWork3, task: TaskRuntimeState) -> None:
    assert task.scheduled_start is None
    assert task.assigned_team == []
    assert task.task_key not in env._station_occupied_tasks[task.current_station]
    assert not any(
        interval.task_key == task.task_key
        for worker in env.state.workers.values()
        for interval in worker.intervals
    )


def test_parent_reservation_invalidation_cancels_child_immediately() -> None:
    """父预约因恢复推迟而失效时，子预约同步失效并释放资源。"""
    env = _new_env()
    parent, child, _ = _install_three_task_chain(env, include_child_of_child=False)
    child_generation = child.generation
    parent.material_ready_time = 20.0
    env.step(
        {
            "task_key": parent.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": tuple(parent.assigned_team),
            "align": 0,
        }
    )

    assert parent.status == TaskStatus.RESERVED
    assert parent.scheduled_start == pytest.approx(20.0)
    assert child.status == TaskStatus.UNREADY
    assert child.generation > child_generation
    _assert_task_resources_released(env, child)


def test_parent_delay_propagates_through_chain_but_preserves_unrelated_booking() -> None:
    """父预约失效递归取消子链旧预约，但不撤销独立任务D。"""
    env = _new_env()
    parent, child, grandchild = _install_three_task_chain(
        env, include_child_of_child=True
    )
    assert grandchild is not None
    unrelated = env.state.tasks["0_24"]
    _complete_non_chain_predecessors(env, unrelated, set())
    used_workers = {
        worker_id
        for task in (parent, child, grandchild)
        for worker_id in task.assigned_team
    }
    unrelated_team = _reserve_with_event(
        env,
        unrelated,
        start=8.0,
        duration=4.0,
        excluded_workers=used_workers,
    )
    unrelated_start = unrelated.scheduled_start
    unrelated_intervals = {
        worker_id: tuple(env.state.workers[worker_id].intervals)
        for worker_id in unrelated_team
    }

    env.load_scenario(
        {
            "tau": 5.0,
            "recovery_time": 20.0,
            "affected_task_keys": [parent.task_key],
        }
    )
    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})

    assert parent.status == TaskStatus.UNREADY
    for dependent in (child, grandchild):
        assert dependent.status == TaskStatus.UNREADY
        _assert_task_resources_released(env, dependent)
    assert unrelated.status == TaskStatus.RESERVED
    assert unrelated.scheduled_start == unrelated_start == pytest.approx(8.0)
    assert {
        worker_id: tuple(env.state.workers[worker_id].intervals)
        for worker_id in unrelated_team
    } == unrelated_intervals


def test_parent_team_duration_change_revalidates_child_but_preserves_unrelated_booking() -> None:
    """父任务换队延长工时后撤销过早子预约，保留无关预约与资源。"""
    env = _new_env()
    parent = env.state.tasks["0_165"]
    child = env.state.tasks["0_166"]
    unrelated = env.state.tasks["0_24"]
    chain_keys = {parent.task_key, child.task_key}
    _complete_non_chain_predecessors(env, parent, chain_keys)
    _complete_non_chain_predecessors(env, child, chain_keys)
    _complete_non_chain_predecessors(env, unrelated, set())

    parent_workers = env.valid_team_completion_workers(
        parent, [], station_id=parent.current_station
    )
    parent_teams = [
        team
        for team in combinations(parent_workers, parent.demand)
        if _team_is_valid(env, parent, team)
    ]
    team_fast = min(
        parent_teams, key=lambda team: env.duration_for_team(parent, team)
    )
    team_slow = max(
        parent_teams, key=lambda team: env.duration_for_team(parent, team)
    )
    fast_duration = env.duration_for_team(parent, team_fast)
    slow_duration = env.duration_for_team(parent, team_slow)
    assert slow_duration > fast_duration

    all_parent_workers = set(team_fast) | set(team_slow)
    child_workers = env.valid_team_completion_workers(
        child, [], station_id=child.current_station
    )
    child_team = next(
        tuple(team)
        for team in combinations(child_workers, child.demand)
        if not set(team) & all_parent_workers and _team_is_valid(env, child, team)
    )
    unrelated_workers = env.valid_team_completion_workers(
        unrelated, [], station_id=unrelated.current_station
    )
    unrelated_team = next(
        tuple(team)
        for team in combinations(unrelated_workers, unrelated.demand)
        if not set(team) & (all_parent_workers | set(child_team))
        and _team_is_valid(env, unrelated, team)
    )

    parent_start = 8.0
    parent.in_station_offset = parent_start
    parent.base_team = tuple(team_fast)
    parent.baseline_assignment = {
        "station": parent.current_station,
        "team": list(team_fast),
        "position": parent_start,
    }
    parent.last_published_assignment = parent.baseline_assignment.copy()
    _reserve_with_event(
        env,
        parent,
        start=parent_start,
        duration=fast_duration,
        excluded_workers=set(),
        selected_team=tuple(team_fast),
    )
    child_start = parent_start + fast_duration
    _reserve_with_event(
        env,
        child,
        start=child_start,
        duration=env.duration_for_team(child, child_team),
        excluded_workers=all_parent_workers,
        selected_team=child_team,
    )
    _reserve_with_event(
        env,
        unrelated,
        start=20.0,
        duration=env.duration_for_team(unrelated, unrelated_team),
        excluded_workers=all_parent_workers | set(child_team),
        selected_team=unrelated_team,
    )

    unrelated_snapshot = (
        unrelated.status,
        unrelated.generation,
        tuple(unrelated.assigned_team),
        unrelated.scheduled_start,
        unrelated.execution_duration,
        tuple(unrelated.revision_history),
    )
    unrelated_intervals_before = {
        worker_id: tuple(env.state.workers[worker_id].intervals)
        for worker_id in unrelated_team
    }
    unrelated_event_before = [
        (event.timestamp, event.event_type, event.task_key, event.generation)
        for event in env.event_queue._heap
        if event.task_key == unrelated.task_key
        and env.event_queue._is_event_valid(event)
    ]
    child_generation_before = child.generation
    unaffected_costs_before = (
        env.cost_takt,
        env.cost_time,
        env.cost_team,
        env.cost_postpone,
    )

    env.step(
        {
            "task_key": parent.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": tuple(team_slow),
            "align": 1,
        }
    )

    assert parent.scheduled_start == pytest.approx(parent_start)
    assert parent.execution_duration == pytest.approx(slow_duration)
    assert child_start < parent.scheduled_start + parent.execution_duration
    assert child.status == TaskStatus.UNREADY
    assert child.generation > child_generation_before
    _assert_task_resources_released(env, child)
    assert not any(
        event.task_key == child.task_key and env.event_queue._is_event_valid(event)
        for event in env.event_queue._heap
    )
    assert (
        unrelated.status,
        unrelated.generation,
        tuple(unrelated.assigned_team),
        unrelated.scheduled_start,
        unrelated.execution_duration,
        tuple(unrelated.revision_history),
    ) == unrelated_snapshot
    assert {
        worker_id: tuple(env.state.workers[worker_id].intervals)
        for worker_id in unrelated_team
    } == unrelated_intervals_before
    assert [
        (event.timestamp, event.event_type, event.task_key, event.generation)
        for event in env.event_queue._heap
        if event.task_key == unrelated.task_key
        and env.event_queue._is_event_valid(event)
    ] == unrelated_event_before
    assert (
        env.cost_takt,
        env.cost_time,
        env.cost_team,
        env.cost_postpone,
    ) == unaffected_costs_before


def test_reserved_parent_completes_before_child_same_time_start() -> None:
    """父子均可预约；同刻父完工事件先于子开工事件，子不早于实完工。"""
    env = _new_env()
    parent, child, _ = _install_three_task_chain(env, include_child_of_child=False)

    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
    assert env.state.current_time == pytest.approx(8.0)
    assert parent.status == TaskStatus.RUNNING
    assert child.status == TaskStatus.RESERVED
    assert env.can_reserve(child)
    assert child.actual_start is None

    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
    assert parent.status == TaskStatus.COMPLETED
    assert child.status == TaskStatus.RUNNING
    assert parent.actual_end is not None
    assert child.actual_start is not None
    assert child.actual_start >= parent.actual_end - env.tolerance


def test_single_legal_branch_has_unit_probability_and_exact_ppo_replay() -> None:
    """仅后移分支合法时，采样及重放概率均为1，条件熵为0。"""
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    task = env.state.tasks["0_16"]
    for candidate in env.state.tasks.values():
        candidate.status = TaskStatus.COMPLETED
        candidate.actual_end = 0.0
    task.status = TaskStatus.UNREADY
    task.actual_end = None
    predecessor = env.state.tasks["0_15"]
    predecessor.status = TaskStatus.RUNNING
    predecessor.actual_end = None
    env.state.current_time = 5.0

    assert env.get_action_candidates() == [task]
    assert env.get_action_branch_mask(task) == (False, True)

    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    actor.eval()
    state = torch.zeros(32)
    urgency = torch.zeros(2)
    action, sampled_log_prob, _value, record = actor.select_action(
        env, state, urgency, deterministic=False
    )
    assert action is not None
    assert action["branch"] == ActionBranch.POSTPONE
    assert record["can_reserve"] is False
    assert record["can_postpone"] is True
    assert math.exp(sampled_log_prob) == pytest.approx(1.0)

    _values, replayed_log_probs, entropies = actor.evaluate_action_log_probs(
        state.unsqueeze(0), urgency.unsqueeze(0), [record]
    )
    assert replayed_log_probs.item() == pytest.approx(0.0)
    assert entropies.item() == pytest.approx(0.0)


def test_disturbance_of_same_operation_id_is_aircraft_local() -> None:
    """两架飞机具有相同task_id时，扰动只修改明确指定的task_key。"""
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    affected = env.state.tasks["0_15"]
    unaffected = env.state.tasks["1_15"]
    assert affected.task_id == unaffected.task_id
    affected.status = TaskStatus.UNREADY
    unaffected.status = TaskStatus.READY
    before_unaffected = (
        unaffected.status,
        unaffected.material_ready_time,
        unaffected.generation,
        tuple(unaffected.assigned_team),
        unaffected.scheduled_start,
    )

    env.load_scenario(
        {
            "tau": 0.0,
            "recovery_time": 20.0,
            "affected_task_keys": [affected.task_key],
        }
    )

    assert affected.material_ready_time == pytest.approx(20.0)
    assert (
        unaffected.status,
        unaffected.material_ready_time,
        unaffected.generation,
        tuple(unaffected.assigned_team),
        unaffected.scheduled_start,
    ) == before_unaffected
