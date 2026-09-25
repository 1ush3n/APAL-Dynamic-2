"""工作三实验共用的基准暖机前缀。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any

from envs.work3.core_types import ActionBranch
from envs.work3.environment import AirLineEnvWork3
from models.work3.heuristic_agent import HeuristicAgentWork3


@dataclass(frozen=True, slots=True)
class UniformBaselineWarmupResult:
    completed: bool
    termination_reason: str
    step_count: int
    first_full_station_transfer_count: int | None
    transfer_count: int
    transfer_times: tuple[float, ...]
    completion_time: float
    cost: float
    prefix_sha256: str


def _station_aircraft_counts(env: AirLineEnvWork3) -> tuple[int, ...]:
    counts = [0] * env.state.num_stations
    for aircraft in env.state.aircraft.values():
        if aircraft.is_in_factory:
            counts[aircraft.current_station] += 1
    return tuple(counts)


def _action_for_hash(action: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in action.items():
        if key == "branch" and hasattr(value, "name"):
            normalized[key] = value.name
        elif isinstance(value, tuple):
            normalized[key] = list(value)
        else:
            normalized[key] = value
    return normalized


def run_uniform_baseline_warmup(
    env: AirLineEnvWork3,
    *,
    max_steps: int | None = None,
    wall_clock_deadline: float | None = None,
) -> UniformBaselineWarmupResult:
    """运行确定性基准前缀，直至填满各站并再完成一次真实同步脉动。"""
    if max_steps is not None and (type(max_steps) is not int or max_steps < 0):
        raise ValueError("max_steps必须是非负整数或None")
    if wall_clock_deadline is not None and not math.isfinite(wall_clock_deadline):
        raise ValueError("wall_clock_deadline必须是有限monotonic时间戳")
    if env.state.num_stations != 5:
        raise ValueError("统一基准暖机要求五站实例")

    initial_transfer_count = len(env.state.transfer_history)
    initial_time = float(env.state.current_time)
    initial_cost = float(env.cumulative_cost)
    digest = hashlib.sha256()
    agent = HeuristicAgentWork3(name="UniformBaselineWarmup")
    first_full_station_transfer_count: int | None = None
    step_count = 0
    termination_reason = "step_limit"

    while True:
        station_counts = _station_aircraft_counts(env)
        full_line = all(count == 1 for count in station_counts)
        transfers_since_start = len(env.state.transfer_history) - initial_transfer_count

        if full_line and first_full_station_transfer_count is None:
            first_full_station_transfer_count = transfers_since_start
        if (
            full_line
            and first_full_station_transfer_count is not None
            and transfers_since_start > first_full_station_transfer_count
        ):
            termination_reason = "completed"
            break
        if env._check_terminated():
            termination_reason = "batch_completed_before_warmup"
            break
        if max_steps is not None and step_count >= max_steps:
            termination_reason = "step_limit"
            break
        if wall_clock_deadline is not None and time.monotonic() >= wall_clock_deadline:
            termination_reason = "wall_time_limit"
            break

        candidates = env.get_action_candidates()
        action = agent.select_action(env) if candidates else None
        if action is None:
            action = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}

        time_before = float(env.state.current_time)
        transfer_count_before = len(env.state.transfer_history)
        _observation, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        record = {
            "action": _action_for_hash(action),
            "time_before": time_before,
            "time_after": float(env.state.current_time),
            "reward": float(reward),
            "transfers": [
                float(value)
                for value in env.state.transfer_history[transfer_count_before:]
            ],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": info.get("termination_reason"),
        }
        digest.update(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        if terminated or truncated:
            if terminated and env._check_terminated():
                termination_reason = "batch_completed_before_warmup"
            else:
                termination_reason = str(
                    info.get("termination_reason", "truncated")
                )
            break

    transfer_times = tuple(
        float(value)
        for value in env.state.transfer_history[initial_transfer_count:]
    )
    return UniformBaselineWarmupResult(
        completed=termination_reason == "completed",
        termination_reason=termination_reason,
        step_count=step_count,
        first_full_station_transfer_count=first_full_station_transfer_count,
        transfer_count=len(transfer_times),
        transfer_times=transfer_times,
        completion_time=float(env.state.current_time),
        cost=float(env.cumulative_cost - initial_cost),
        prefix_sha256=digest.hexdigest(),
    )


def validate_scenario_after_warmup(
    scenario: Mapping[str, Any] | None,
    warmup: UniformBaselineWarmupResult,
    *,
    tolerance: float = 1e-8,
) -> None:
    """拒绝在统一前缀完成前已到期的主扰动，不改写固定tau。"""
    if not warmup.completed:
        raise ValueError("统一基准暖机未完成，不能进入主扰动调度")
    if scenario is None:
        return
    tau_value = scenario.get("tau")
    if isinstance(tau_value, bool) or not isinstance(tau_value, (int, float)):
        raise ValueError("固定扰动场景缺少有限数值tau")
    tau = float(tau_value)
    if not math.isfinite(tau):
        raise ValueError("固定扰动场景tau必须为有限数值")
    if tau < warmup.completion_time - tolerance:
        raise ValueError("扰动时刻早于统一基准暖机完成时刻；不得重定时场景")
