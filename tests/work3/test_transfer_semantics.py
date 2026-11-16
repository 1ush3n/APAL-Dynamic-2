"""F02：实际转站时刻不以名义 H0 作为硬等待下界。"""

from __future__ import annotations

from pathlib import Path

from envs.work3.core_types import TaskStatus
from envs.work3.environment import AirLineEnvWork3
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")


def _cleared_env(current_time: float) -> AirLineEnvWork3:
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    env.state.h0 = 10.0
    env.state.current_time = current_time
    for task in env.state.tasks.values():
        task.status = TaskStatus.COMPLETED
        task.actual_end = current_time
    for aircraft in env.state.aircraft.values():
        aircraft.current_station = -1
    return env


def test_transfer_can_happen_before_nominal_h0() -> None:
    env = _cleared_env(current_time=2.0)

    env._check_and_schedule_transfer()
    event = env.event_queue.peek()

    assert event is not None
    assert event.timestamp == 2.0


def test_transfer_after_nominal_h0_is_not_pulled_earlier() -> None:
    env = _cleared_env(current_time=12.0)

    env._check_and_schedule_transfer()
    event = env.event_queue.peek()

    assert event is not None
    assert event.timestamp == 12.0


def test_heuristic_estimate_does_not_add_h0_when_no_work_remains() -> None:
    env = _cleared_env(current_time=2.0)

    assert compute_cycle_heuristic_cmax(env.state) == 2.0
