"""W3-05：同刻事件在策略看到状态前处理到稳定状态。"""

from __future__ import annotations

import random
from itertools import permutations
from pathlib import Path

import torch

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from envs.work3.event_queue import EventType
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from models.work3.time_head import TimeResidualHead
from scripts.work3.evaluate_c_vs_d import summarize_disturbance_effects


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")


def _new_env() -> AirLineEnvWork3:
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    return env


def _valid_team(env: AirLineEnvWork3, task) -> tuple[int, ...]:
    candidates = env.valid_team_completion_workers(task, [])
    assert len(candidates) >= task.demand
    return tuple(candidates[: task.demand])


def test_current_timestamp_disturbance_is_processed_before_observation() -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]

    env.load_scenario(
        {
            "tau": 0.0,
            "recovery_time": 20.0,
            "affected_task_keys": [task.task_key],
        }
    )

    assert task.status == TaskStatus.UNREADY
    assert task.material_ready_time == 20.0
    assert task.task_key not in env._get_observation()["ready_task_keys"]
    assert task.actual_start is None

    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": _valid_team(env, task),
            "align": 0,
        }
    )

    assert task.status == TaskStatus.RESERVED
    assert task.scheduled_start is not None and task.scheduled_start >= 20.0
    assert task.actual_start is None


def test_due_event_processing_drains_generated_same_timestamp_events() -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    env.load_scenario(
        {
            "tau": 0.0,
            "recovery_time": 0.0,
            "affected_task_keys": [task.task_key],
        }
    )

    env.event_queue.push(
        event_type=EventType.MATERIAL_ARRIVE,
        timestamp=0.0,
        task_key=task.task_key,
        generation=task.generation,
    )
    env.process_due_events()

    assert task.status == TaskStatus.READY
    assert env.event_queue.peek() is None


def test_same_timestamp_finish_releases_before_reserved_start() -> None:
    env = _new_env()
    first, second = env.get_ready_tasks()[:2]
    team = _valid_team(env, first)

    env.step(
        {
            "task_key": first.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 0,
        }
    )
    env.step(
        {
            "task_key": second.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 0,
        }
    )
    assert first.execution_duration is not None
    finish_time = first.execution_duration
    assert second.status == TaskStatus.RESERVED
    assert second.scheduled_start == finish_time

    env.state.current_time = finish_time
    env.process_due_events()

    assert first.status == TaskStatus.COMPLETED
    assert second.status == TaskStatus.RUNNING
    assert second.actual_start == finish_time


def test_same_timestamp_disturbance_invalidates_start_before_execution() -> None:
    env = _new_env()
    first, second = env.get_ready_tasks()[:2]
    team = _valid_team(env, first)

    env.step(
        {
            "task_key": first.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 0,
        }
    )
    env.step(
        {
            "task_key": second.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 0,
        }
    )
    assert first.execution_duration is not None
    finish_time = first.execution_duration
    assert second.status == TaskStatus.RESERVED
    env.load_scenario(
        {
            "tau": finish_time,
            "recovery_time": finish_time + 5.0,
            "affected_task_keys": [second.task_key],
        }
    )

    env.state.current_time = finish_time
    env.process_due_events()

    assert first.status == TaskStatus.COMPLETED
    assert second.status == TaskStatus.UNREADY
    assert second.actual_start is None
    assert not any(interval.task_key == second.task_key for interval in env.state.workers[team[0]].intervals)


def test_stale_start_and_finish_generations_have_no_environment_side_effects() -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    team = _valid_team(env, task)

    for start_time, generation in ((20.0, 0), (25.0, 1), (30.0, 2)):
        task.in_station_offset = start_time
        env.step(
            {
                "task_key": task.task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": team,
                "align": 1,
            }
        )
        assert task.generation == generation
        assert task.scheduled_start == start_time

    cost_before_stale_events = env.cumulative_cost
    env.event_queue.push(
        EventType.TASK_START, timestamp=20.0, task_key=task.task_key, generation=0
    )
    env.event_queue.push(
        EventType.TASK_START, timestamp=25.0, task_key=task.task_key, generation=1
    )
    env.state.current_time = 25.0
    env.process_due_events()

    assert task.status == TaskStatus.RESERVED
    assert task.scheduled_start == 30.0
    assert task.actual_start is None
    assert env.cumulative_cost == cost_before_stale_events

    env.state.current_time = 30.0
    env.process_due_events()
    assert task.status == TaskStatus.RUNNING
    assert task.actual_start == 30.0
    worker_intervals_before = {
        worker_id: sum(iv.task_key == task.task_key for iv in env.state.workers[worker_id].intervals)
        for worker_id in team
    }
    cost_before_stale_finishes = env.cumulative_cost

    for generation in (0, 1):
        env.event_queue.push(
            EventType.TASK_FINISH,
            timestamp=30.0,
            task_key=task.task_key,
            generation=generation,
        )
    env.process_due_events()

    assert task.status == TaskStatus.RUNNING
    assert task.actual_end is None
    assert env.cumulative_cost == cost_before_stale_finishes
    assert {
        worker_id: sum(iv.task_key == task.task_key for iv in env.state.workers[worker_id].intervals)
        for worker_id in team
    } == worker_intervals_before


def test_disturbance_does_not_interrupt_a_task_that_already_started() -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    team = _valid_team(env, task)
    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 0,
        }
    )
    assert task.status == TaskStatus.RUNNING
    original = (task.actual_start, task.execution_duration, tuple(task.assigned_team))

    env.load_scenario(
        {
            "tau": env.state.current_time,
            "recovery_time": env.state.current_time + 20.0,
            "affected_task_keys": [task.task_key],
        }
    )

    assert task.status == TaskStatus.RUNNING
    assert (task.actual_start, task.execution_duration, tuple(task.assigned_team)) == original
    assert task.material_ready_time == 0.0
    assert env.event_queue.peek() is not None
    assert env.event_queue.peek().event_type == EventType.TASK_FINISH
    hit_report = summarize_disturbance_effects(
        env.state.tasks,
        {
            "affected_task_keys": [task.task_key],
            "aircraft_id": task.aircraft_id,
        },
        baseline_start_by_key={task.task_key: task.in_station_offset},
    )
    assert hit_report["actual_hit_count"] == 0
    assert hit_report["actual_hit_rate"] == 0.0
    assert hit_report["actual_hit_task_keys"] == []


def test_same_timestamp_mixed_events_are_independent_of_heap_insertion_order() -> None:
    def run_order(event_order: tuple[str, ...]) -> tuple[object, ...]:
        env = _new_env()
        first, second = env.get_ready_tasks()[:2]
        team = _valid_team(env, first)
        env.step(
            {
                "task_key": first.task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": team,
                "align": 0,
            }
        )
        env.step(
            {
                "task_key": second.task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": team,
                "align": 0,
            }
        )
        assert first.execution_duration is not None
        timestamp = first.execution_duration
        generation = second.generation
        events = {
            "finish": (
                EventType.TASK_FINISH,
                first.task_key,
                first.generation,
                {},
            ),
            "disturbance": (
                EventType.DISTURBANCE,
                None,
                0,
                {
                    "recovery_time": timestamp + 10.0,
                    "affected_task_keys": [second.task_key],
                },
            ),
            "material": (
                EventType.MATERIAL_ARRIVE,
                second.task_key,
                generation,
                {},
            ),
        }
        for name in event_order:
            event_type, task_key, event_generation, payload = events[name]
            env.event_queue.push(
                event_type,
                timestamp=timestamp,
                task_key=task_key,
                generation=event_generation,
                payload=payload,
            )

        env.state.current_time = timestamp
        env.process_due_events()
        return (
            env.state.current_time,
            first.status,
            first.actual_end,
            second.status,
            second.material_ready_time,
            second.scheduled_start,
            tuple(
                sorted(
                    (worker_id, iv.task_key, iv.start, iv.end)
                    for worker_id, calendar in env.state.workers.items()
                    for iv in calendar.intervals
                )
            ),
            tuple(env.state.transfer_history),
            env.cost_takt,
            env.cost_time,
            env.cost_team,
            env.cost_revision,
        )

    orders = list(permutations(("finish", "disturbance", "material")))
    expected = run_order(orders[0])
    for order in orders[1:]:
        assert run_order(order) == expected
    assert expected[1] == TaskStatus.COMPLETED
    assert expected[3] == TaskStatus.UNREADY


def test_independent_same_timestamp_disturbances_are_order_invariant() -> None:
    def run_seed(seed: int) -> tuple[object, ...]:
        env = _new_env()
        tasks = env.get_ready_tasks()[:2]
        assert len(tasks) == 2
        events = [
            (task.task_key, 10.0 + index * 10.0)
            for index, task in enumerate(tasks)
        ]
        random.Random(seed).shuffle(events)
        for task_key, recovery_time in events:
            env.event_queue.push(
                EventType.DISTURBANCE,
                timestamp=5.0,
                payload={
                    "recovery_time": recovery_time,
                    "affected_task_keys": [task_key],
                },
            )
        env.state.current_time = 5.0
        env.process_due_events()
        return (
            tuple(
                sorted(
                    (task.task_key, task.status, task.material_ready_time)
                    for task in tasks
                )
            ),
            env.cumulative_cost,
        )

    expected = run_seed(0)
    for seed in range(1, 8):
        assert run_seed(seed) == expected
    assert all(item[1] == TaskStatus.UNREADY for item in expected[0])


def test_future_scenarios_do_not_leak_into_current_actor_critic_or_time_inputs() -> None:
    env_a = _new_env()
    env_b = _new_env()
    task_a, task_b = env_a.get_ready_tasks()[:2]
    env_a.load_scenario(
        {
            "tau": 50.0,
            "recovery_time": 80.0,
            "affected_task_keys": [task_a.task_key],
        }
    )
    env_b.load_scenario(
        {
            "tau": 50.0,
            "recovery_time": 120.0,
            "affected_task_keys": [task_b.task_key],
        }
    )

    cmax_a = compute_cycle_heuristic_cmax(env_a.state)
    cmax_b = compute_cycle_heuristic_cmax(env_b.state)
    state_a = extract_compact_state_features(env_a.state, cmax_a)
    state_b = extract_compact_state_features(env_b.state, cmax_b)
    assert torch.equal(state_a, state_b)
    assert cmax_a == cmax_b
    assert {
        task.task_key: env_a.get_action_branch_mask(task)
        for task in env_a.get_action_candidates()
    } == {
        task.task_key: env_b.get_action_branch_mask(task)
        for task in env_b.get_action_candidates()
    }

    actor = ActorCriticWork3().eval()
    time_head = TimeResidualHead(in_dim=actor.hidden_dim).eval()
    graph_a = actor.build_graph_snapshot(env_a)
    graph_b = actor.build_graph_snapshot(env_b)
    assert graph_a.node_types == graph_b.node_types
    assert graph_a.edge_types == graph_b.edge_types
    for node_type in graph_a.node_types:
        assert torch.equal(graph_a[node_type].x, graph_b[node_type].x)
    for edge_type in graph_a.edge_types:
        assert torch.equal(
            graph_a[edge_type].edge_index,
            graph_b[edge_type].edge_index,
        )

    urgency_a = compute_time_urgency_vector(
        cmax_a - env_a.state.current_time,
        env_a.state.current_time,
        env_a.state.last_transfer_time,
        env_a.state.h0,
    )
    urgency_b = compute_time_urgency_vector(
        cmax_b - env_b.state.current_time,
        env_b.state.current_time,
        env_b.state.last_transfer_time,
        env_b.state.h0,
    )
    assert torch.equal(urgency_a, urgency_b)
    with torch.no_grad():
        value_a, actor_input_a = actor.encode_state(state_a, urgency_a, graph_a)
        value_b, actor_input_b = actor.encode_state(state_b, urgency_b, graph_b)
        time_input_a = actor.encode_shared_representation(state_a, graph_a)
        time_input_b = actor.encode_shared_representation(state_b, graph_b)
        residual_a = time_head(time_input_a)
        residual_b = time_head(time_input_b)

    assert torch.equal(value_a, value_b)
    assert torch.equal(actor_input_a, actor_input_b)
    assert torch.equal(time_input_a, time_input_b)
    assert torch.equal(residual_a, residual_b)


def test_recovery_time_is_hidden_until_tau_then_bounds_reservation() -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    cmax_before = compute_cycle_heuristic_cmax(env.state)
    features_before = extract_compact_state_features(env.state, cmax_before)

    env.load_scenario(
        {
            "tau": 5.0,
            "recovery_time": 20.0,
            "affected_task_keys": [task.task_key],
        }
    )
    assert task.status == TaskStatus.READY
    assert task.material_ready_time == 0.0
    assert torch.equal(
        features_before,
        extract_compact_state_features(
            env.state,
            compute_cycle_heuristic_cmax(env.state),
        ),
    )

    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})

    assert env.state.current_time == 5.0
    assert task.status == TaskStatus.UNREADY
    assert task.material_ready_time == 20.0
    assert task.actual_start is None
    cmax_after = compute_cycle_heuristic_cmax(env.state)
    features_after = extract_compact_state_features(env.state, cmax_after)
    station = task.current_station
    assert features_after[11 + station] > features_before[11 + station]

    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": _valid_team(env, task),
            "align": 0,
        }
    )
    assert task.status == TaskStatus.RESERVED
    assert task.scheduled_start is not None and task.scheduled_start >= 20.0
    assert task.actual_start is None


def test_pulse_boundary_disturbance_keeps_aircraft_task_identity() -> None:
    env = _new_env()
    target = next(
        task
        for task in env.state.tasks.values()
        if task.aircraft_id == 0 and task.current_station == 1
    )
    same_operation_next_aircraft = env.state.tasks[f"1_{target.task_id}"]
    target.status = TaskStatus.POSTPONED
    target.material_ready_time = 0.0
    for predecessor_id in target.predecessors:
        env.state.tasks[f"0_{predecessor_id}"].status = TaskStatus.COMPLETED
    other_before = (
        same_operation_next_aircraft.status,
        same_operation_next_aircraft.material_ready_time,
    )

    env.event_queue.push(
        EventType.SYNCHRONOUS_TRANSFER,
        timestamp=5.0,
        payload={"cycle": env.state.current_cycle},
    )
    env.event_queue.push(
        EventType.DISTURBANCE,
        timestamp=5.0,
        payload={
            "recovery_time": 20.0,
            "affected_task_keys": [target.task_key],
        },
    )
    env.state.current_time = 5.0
    env.process_due_events()

    assert env.state.aircraft[0].current_station == 1
    assert env.state.aircraft[1].current_station == 0
    assert target.current_station == 1
    assert target.material_ready_time == 20.0
    assert target.status == TaskStatus.UNREADY
    assert (
        same_operation_next_aircraft.status,
        same_operation_next_aircraft.material_ready_time,
    ) == other_before


def test_real_action_accounts_for_costs_during_multiple_no_decision_events() -> None:
    env = _new_env()
    target = env.get_ready_tasks()[0]
    team = _valid_team(env, target)
    current_station_tasks = env.state.get_tasks_for_station(target.current_station)
    for task in current_station_tasks:
        if task.task_key != target.task_key:
            task.status = TaskStatus.COMPLETED
            task.actual_end = 0.0

    next_station_tasks = [
        task
        for task in env.state.tasks.values()
        if task.aircraft_id == target.aircraft_id
        and task.current_station == target.current_station + 1
    ]
    for task in next_station_tasks:
        task.status = TaskStatus.UNREADY
        task.material_ready_time = 10.0
        for predecessor_id in task.predecessors:
            env.state.tasks[f"{task.aircraft_id}_{predecessor_id}"].status = TaskStatus.COMPLETED

    next_aircraft_first_station_tasks = [
        task
        for task in env.state.tasks.values()
        if task.aircraft_id == 1 and task.current_station == 0
    ]
    for task in next_aircraft_first_station_tasks:
        task.status = TaskStatus.UNREADY
        task.material_ready_time = 10.0
        for predecessor_id in task.predecessors:
            env.state.tasks[f"{task.aircraft_id}_{predecessor_id}"].status = TaskStatus.COMPLETED

    env.state.h0 = 2.0
    env.state.last_transfer_time = 0.0
    target.in_station_offset = 0.0
    original_duration = env.duration_for_team(target, team)
    target.duration *= 5.0 / original_duration
    cost_before = env.cumulative_cost

    _, reward, terminated, truncated, info = env.step(
        {
            "task_key": target.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 0,
        }
    )

    assert not terminated and not truncated
    assert env.step_count == 1
    assert len(env.step_rewards) == 1
    assert target.actual_end == 5.0
    assert env.state.transfer_history == [5.0]
    assert env.state.current_time == 10.0
    assert env.state.aircraft[target.aircraft_id].current_station == 1
    assert env.cost_takt == 1.5
    assert info["step_cost"] == env.cumulative_cost - cost_before
    assert reward == -info["step_cost"]
    assert env.step_rewards[-1] == reward
