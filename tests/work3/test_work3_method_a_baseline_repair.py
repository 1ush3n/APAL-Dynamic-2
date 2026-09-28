"""P2-2 A组：确定性基准计划修复器的功能验收。"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from scripts.work3.evaluate_c_vs_d import evaluate_single_trajectory
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")
TASKS_BY_STATION = ((1, 15), (1, 6), (2, 18), (3, 12), (4, 20), (5, 24))


def _write_two_aircraft_baseline(path: Path) -> None:
    source = MultiAircraftBaseline.load_from_json(BASELINE_PATH)
    tasks = {}
    for aircraft_id in range(2):
        for station_id, task_id in TASKS_BY_STATION:
            source_task = source.get_task(0, task_id)
            assert not source_task.predecessors
            entry = (aircraft_id + station_id - 1) * source.h0
            task_key = f"{aircraft_id}_{task_id}"
            tasks[task_key] = replace(
                source_task,
                aircraft_id=aircraft_id,
                task_key=task_key,
                in_station_offset=0.0,
                baseline_start=entry,
                baseline_end=entry + source_task.duration,
                cycle_idx=aircraft_id + station_id,
                nominal_station_entry=entry,
                nominal_station_exit=entry + source.h0,
            )

    MultiAircraftBaseline(
        num_aircraft=2,
        num_stations=5,
        h0=source.h0,
        tasks=tasks,
        station_workers=source.station_workers,
    ).save_to_json(path)


def _agent() -> Any:
    from scripts.work3.baseline_repair_agent import BaselinePlanRepairAgent

    return BaselinePlanRepairAgent()


def _env(path: Path) -> AirLineEnvWork3:
    env = AirLineEnvWork3(baseline_json_path=path)
    env.reset()
    return env


def test_a_repair_keeps_base_station_uses_earliest_baseline_order_and_never_postpones(
    tmp_path: Path,
) -> None:
    """站内基准位置并列时按飞机顺序及工序号稳定排序，动作只留站。"""
    baseline_path = tmp_path / "two_aircraft_baseline.json"
    _write_two_aircraft_baseline(baseline_path)
    env = _env(baseline_path)
    env.state.tasks["0_15"].in_station_offset = 0.0
    env.state.tasks["0_6"].in_station_offset = 0.0

    action = _agent().select_action(env)

    assert action is not None
    assert action["task_key"] == "0_6"
    assert action["branch"] == ActionBranch.STATION_EXECUTE
    assert tuple(action["team"]) == env.state.tasks["0_6"].base_team
    assert action["align"] == 1
    assert all(task.current_station == task.base_station for task in env.state.tasks.values())


@pytest.mark.parametrize(
    ("busy_worker", "expected_worker"),
    [(12, 72), (None, 12)],
)
def test_a_repair_fallback_ranks_legal_teams_by_slot_then_worker_id(
    tmp_path: Path,
    busy_worker: int | None,
    expected_worker: int,
) -> None:
    """基准团队硬约束失效后，先选可行开工更早者，再按工人ID破同分。"""
    baseline_path = tmp_path / "two_aircraft_baseline.json"
    _write_two_aircraft_baseline(baseline_path)
    env = _env(baseline_path)
    task = env.state.tasks["0_15"]
    task.demand = 1
    task.skill = -1
    task.base_team = (999,)
    env.get_action_candidates = lambda: [task]  # type: ignore[method-assign]
    if busy_worker is not None:
        env.state.workers[busy_worker].add_interval(
            env.state.current_time,
            env.state.current_time + 5.0,
            "test_busy_worker",
        )

    action = _agent().select_action(env)

    assert action is not None
    assert tuple(action["team"]) == (expected_worker,)
    assert action["branch"] == ActionBranch.STATION_EXECUTE


def test_a_repair_keeps_a_hard_valid_base_team_even_when_busy(
    tmp_path: Path,
) -> None:
    """基准队暂忙但仍合法时，不用空闲替代队，统一日历延后其预约。"""
    baseline_path = tmp_path / "two_aircraft_baseline.json"
    _write_two_aircraft_baseline(baseline_path)
    env = _env(baseline_path)
    target = env.state.tasks["0_18"]
    agent = _agent()
    for _ in range(20):
        if target.status == TaskStatus.READY:
            break
        action = agent.select_action(env)
        assert action is not None
        env.step(action)
    assert target.status == TaskStatus.READY
    alternatives = next(
        team
        for team in combinations(
            env.valid_team_completion_workers(target, []),
            target.demand,
        )
        if set(team) != set(target.base_team)
    )
    env._validate_team_for_task(target, alternatives)
    now = float(env.state.current_time)
    env.state.workers[target.base_team[0]].add_interval(
        now,
        now + 5.0,
        "test_busy_base_member",
    )
    get_candidates = env.get_action_candidates
    env.get_action_candidates = lambda: [target]  # type: ignore[method-assign]

    action = agent.select_action(env)

    env.get_action_candidates = get_candidates  # type: ignore[method-assign]
    assert action is not None
    assert tuple(action["team"]) == target.base_team
    env.step(action)
    assert target.status == TaskStatus.RESERVED
    assert target.scheduled_start is not None
    assert target.scheduled_start >= now + 5.0 - env.tolerance


def test_a_repair_advances_instead_of_republishing_an_existing_reservation(
    tmp_path: Path,
) -> None:
    """仅有待执行预约时走推进动作，不重复发布或产生新修订。"""
    baseline_path = tmp_path / "two_aircraft_baseline.json"
    _write_two_aircraft_baseline(baseline_path)
    env = _env(baseline_path)
    target = env.state.tasks["0_15"]
    target.status = TaskStatus.RESERVED
    target.scheduled_start = env.state.current_time + 20.0
    target.assigned_team = tuple(target.base_team)
    env.get_action_candidates = lambda: [target]  # type: ignore[method-assign]
    cumulative_cost = env.cumulative_cost
    revision_count = len(target.revision_history)

    action = _agent().select_action(env)

    assert action == {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    assert target.scheduled_start == env.state.current_time + 20.0
    assert env.cumulative_cost == cumulative_cost
    assert len(target.revision_history) == revision_count


def test_a_repair_completes_two_aircraft_five_station_batch_deterministically(
    tmp_path: Path,
) -> None:
    """A从空线完成双机五站批次，独立可行且复跑排程一致。"""
    baseline_path = tmp_path / "two_aircraft_baseline.json"
    _write_two_aircraft_baseline(baseline_path)
    traces = []
    for _ in range(2):
        env = AirLineEnvWork3(baseline_json_path=baseline_path)
        result = evaluate_single_trajectory(
            env,
            "Baseline-Repair-A",
            _agent(),
            max_decisions=100,
        )
        assert result["success"] is True
        assert result["feasible"] is True
        assert result["completed_tasks"] == 12
        assert result["postponed_count"] == 0
        assert all(
            task.actual_start_station == task.base_station
            and task.current_station == task.base_station
            for task in env.state.tasks.values()
        )
        traces.append(
            {
                key: (
                    task.actual_start,
                    task.actual_end,
                    tuple(task.assigned_team),
                    task.actual_start_station,
                )
                for key, task in env.state.tasks.items()
            }
        )
    assert traces[0] == traces[1]


def test_a_policy_cannot_observe_future_disturbance_payload(
    tmp_path: Path,
) -> None:
    """只改变尚未揭示的事件目标与恢复时刻，不改变A当前动作。"""
    baseline_path = tmp_path / "two_aircraft_baseline.json"
    _write_two_aircraft_baseline(baseline_path)
    actions = []
    for scenario_id, target_key, recovery_time in (
        ("HIDDEN_A", "0_18", 120.0),
        ("HIDDEN_B", "0_24", 240.0),
    ):
        env = _env(baseline_path)
        env.load_scenario(
            {
                "scenario_id": scenario_id,
                "aircraft_id": 0,
                "tau": 100.0,
                "recovery_time": recovery_time,
                "affected_task_keys": [target_key],
            }
        )
        action = _agent().select_action(env)
        assert env.disturbance_event_triggered is False
        actions.append(action)

    assert actions[0] == actions[1]
    assert actions[0] is not None
    assert actions[0]["branch"] == ActionBranch.STATION_EXECUTE


def test_a_repair_handles_revealed_start_delay_without_postponing(
    tmp_path: Path,
) -> None:
    """事件揭示后A保留原站位，并确保目标实际开工不早于R。"""
    baseline_path = tmp_path / "two_aircraft_baseline.json"
    _write_two_aircraft_baseline(baseline_path)
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    result = evaluate_single_trajectory(
        env,
        "Baseline-Repair-A",
        _agent(),
        scenario={
            "scenario_id": "A_REVEALED_START_DELAY",
            "aircraft_id": 0,
            "tau": 0.0,
            "recovery_time": 5.0,
            "affected_task_keys": ["0_18"],
        },
        max_decisions=100,
    )

    target = env.state.tasks["0_18"]
    assert result["success"] is True
    assert result["feasible"] is True
    assert result["disturbance_effects"]["actual_hit_count"] == 1
    assert target.actual_start is not None and target.actual_start >= 5.0
    assert result["postponed_count"] == 0


def test_a_marks_unstaffable_fixed_station_task_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """固定站位无满足人数的团队时返回明确失败，不借后移规避。"""
    baseline_path = tmp_path / "two_aircraft_baseline.json"
    _write_two_aircraft_baseline(baseline_path)
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    original_reset = env.reset

    def reset_with_unstaffable_runtime_task() -> dict[str, Any]:
        observation = original_reset()
        env.state.tasks["0_15"].demand = 3
        return observation

    monkeypatch.setattr(env, "reset", reset_with_unstaffable_runtime_task)

    result = evaluate_single_trajectory(
        env,
        "Baseline-Repair-A",
        _agent(),
        max_decisions=20,
    )

    assert result["success"] is False
    assert result["termination_reason"] == "baseline_repair_infeasible"
    assert "0_15" in result["failure_reason"]
    assert result["postponed_count"] == 0
