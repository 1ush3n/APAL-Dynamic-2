"""F03：决策、预约和实际开工资格分离。"""

from __future__ import annotations

from pathlib import Path

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")


def _new_env() -> AirLineEnvWork3:
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    return env


def _valid_team(env: AirLineEnvWork3, task) -> tuple[int, ...]:
    workers = env.valid_team_completion_workers(task, [])
    assert len(workers) >= task.demand
    return tuple(workers[: task.demand])


def test_future_recovery_task_can_be_reserved_but_not_started_early() -> None:
    env = _new_env()
    task = env.get_ready_tasks()[0]
    task.status = TaskStatus.UNREADY
    task.material_ready_time = 20.0
    env.state.current_time = 5.0

    assert env.can_reserve(task)
    assert env.can_postpone(task) is False
    assert task in env.get_action_candidates()
    assert env.get_action_branch_mask(task) == (True, False)

    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": _valid_team(env, task),
            "align": 0,
        }
    )

    assert task.status == TaskStatus.RESERVED
    assert task.scheduled_start is not None
    assert task.scheduled_start >= 20.0 - 1e-5
    assert all(
        env.state.workers[worker_id].is_available(5.0, 20.0, tolerance=env.tolerance)
        for worker_id in task.assigned_team
    )

    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
    assert task.actual_start is not None
    assert task.actual_start >= 20.0 - env.tolerance


def test_unplanned_predecessor_blocks_reservation_but_not_legal_postpone() -> None:
    env = _new_env()
    task = env.state.tasks["0_16"]
    task.status = TaskStatus.UNREADY
    env.state.current_time = 5.0

    assert env.can_reserve(task) is False
    assert env.can_postpone(task)
    assert env.get_action_branch_mask(task) == (False, True)

    env.step({"task_key": task.task_key, "branch": ActionBranch.POSTPONE})

    assert task.status == TaskStatus.POSTPONED
    assert task.current_station == 1


def test_reserved_predecessor_provides_future_reservation_lower_bound() -> None:
    env = _new_env()
    predecessor = env.state.tasks["0_15"]
    successor = env.state.tasks["0_16"]
    predecessor_team = _valid_team(env, predecessor)
    predecessor_duration = env.duration_for_team(predecessor, predecessor_team)

    predecessor.execution_duration = predecessor_duration
    predecessor.reserve(team=predecessor_team, scheduled_start=10.0)
    for worker_id in predecessor_team:
        env.state.workers[worker_id].add_interval(
            start=10.0,
            end=10.0 + predecessor_duration,
            task_key=predecessor.task_key,
        )
    env._station_occupied_tasks[predecessor.current_station].add(predecessor.task_key)
    successor.status = TaskStatus.UNREADY
    env.state.current_time = 5.0

    assert env.can_reserve(successor)
    env.step(
        {
            "task_key": successor.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": _valid_team(env, successor),
            "align": 0,
        }
    )

    assert successor.status == TaskStatus.RESERVED
    assert successor.scheduled_start is not None
    assert successor.scheduled_start >= 10.0 + predecessor_duration - 1e-5


def test_candidate_pool_excludes_tasks_with_no_legal_branch() -> None:
    env = _new_env()
    task = env.state.tasks["0_2"]
    task.status = TaskStatus.UNREADY
    env.state.current_time = 5.0
    env.state.aircraft[task.aircraft_id].current_station = 1

    assert env.can_reserve(task) is False
    assert env.can_postpone(task) is False
    assert task not in env.get_action_candidates()
