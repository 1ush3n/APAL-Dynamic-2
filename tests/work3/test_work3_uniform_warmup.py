from __future__ import annotations

import pytest

from envs.work3.environment import AirLineEnvWork3
from utils.work3.uniform_baseline_warmup import (
    run_uniform_baseline_warmup,
    validate_scenario_after_warmup,
)


def _station_aircraft_counts(env: AirLineEnvWork3) -> dict[int, int]:
    return {
        station_id: sum(
            aircraft.is_in_factory and aircraft.current_station == station_id
            for aircraft in env.state.aircraft.values()
        )
        for station_id in range(env.state.num_stations)
    }


def test_uniform_baseline_warmup_fills_all_stations_then_runs_one_transfer() -> None:
    environments = [AirLineEnvWork3(), AirLineEnvWork3()]
    summaries = []
    for env in environments:
        env.reset()
        summaries.append(run_uniform_baseline_warmup(env))

    first, second = summaries
    assert first.completed is True
    assert first.termination_reason == "completed"
    assert first.first_full_station_transfer_count is not None
    assert first.transfer_count == first.first_full_station_transfer_count + 1
    assert len(first.transfer_times) == first.transfer_count
    assert first.transfer_times[-1] == pytest.approx(first.completion_time)
    assert _station_aircraft_counts(environments[0]) == {
        station_id: 1 for station_id in range(5)
    }
    assert first.prefix_sha256 == second.prefix_sha256
    assert first.transfer_times == second.transfer_times
    assert first.step_count == second.step_count
    assert first.cost == pytest.approx(second.cost, abs=1e-10)


def test_main_disturbance_cannot_be_retimed_before_warmup_finishes() -> None:
    env = AirLineEnvWork3()
    env.reset()
    warmup = run_uniform_baseline_warmup(env)

    with pytest.raises(ValueError, match="扰动时刻早于统一基准暖机完成时刻"):
        validate_scenario_after_warmup(
            {"scenario_id": "TOO_EARLY", "tau": warmup.completion_time - 1.0},
            warmup,
        )

    validate_scenario_after_warmup(
        {"scenario_id": "AT_BOUNDARY", "tau": warmup.completion_time},
        warmup,
    )


def test_warmup_step_budget_exhaustion_is_not_reported_as_completed() -> None:
    env = AirLineEnvWork3()
    env.reset()

    warmup = run_uniform_baseline_warmup(env, max_steps=0)

    assert warmup.completed is False
    assert warmup.termination_reason == "step_limit"
    assert warmup.step_count == 0
    assert warmup.transfer_count == 0
    assert env.state.current_time == pytest.approx(0.0)
    with pytest.raises(ValueError, match="统一基准暖机未完成"):
        validate_scenario_after_warmup(
            {"scenario_id": "NO_WARMUP", "tau": 1000.0},
            warmup,
        )
