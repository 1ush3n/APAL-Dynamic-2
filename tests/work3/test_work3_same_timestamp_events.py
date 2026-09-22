"""W3-05：同刻事件在策略看到状态前处理到稳定状态。"""

from __future__ import annotations

from pathlib import Path

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from envs.work3.event_queue import EventType


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
            "recovery_time": 0.0,
            "affected_task_keys": [task.task_key],
        }
    )

    assert task.status == TaskStatus.READY
    assert env.event_queue.peek() is None


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
