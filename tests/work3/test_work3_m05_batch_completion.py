"""M05：从真实基准缩放架次数并验证无扰动完整出线。"""

from __future__ import annotations

from pathlib import Path

import pytest

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from scripts.work3.evaluate_c_vs_d import _check_completed_trajectory_feasibility
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")


@pytest.mark.parametrize("num_aircraft", (1, 2))
def test_real_baseline_subset_completes_without_disturbance(
    num_aircraft: int,
    tmp_path: Path,
) -> None:
    """按1/2架次配置缩小真实实例，仍须全任务完成并独立可行。"""
    if not BASELINE_PATH.is_file():
        pytest.skip(f"{BASELINE_PATH} 不存在")

    source = MultiAircraftBaseline.load_from_json(BASELINE_PATH)
    tasks = {
        task_key: task
        for task_key, task in source.tasks.items()
        if task.aircraft_id < num_aircraft
    }
    baseline = MultiAircraftBaseline(
        num_aircraft=num_aircraft,
        num_stations=source.num_stations,
        h0=source.h0,
        tasks=tasks,
        station_workers=source.station_workers,
    )
    assert baseline.total_tasks_count == (
        num_aircraft * source.physical_tasks_per_aircraft
    )

    baseline_path = tmp_path / f"real_baseline_{num_aircraft}_aircraft.json"
    baseline.save_to_json(baseline_path)
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    decisions = 0
    max_decisions = baseline.total_tasks_count * 4 + 100
    last_info: dict[str, object] = {}
    while not env._check_terminated():
        if decisions >= max_decisions:
            pytest.fail(f"小批次超过保护步数：{decisions}")

        ready_tasks = env.get_ready_tasks()
        if ready_tasks:
            task = ready_tasks[0]
            original = baseline.get_task(task.aircraft_id, task.task_id)
            action = {
                "task_key": task.task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": original.team,
                "align": 1,
            }
        else:
            if env.event_queue.is_empty():
                pytest.fail("无扰动小批次在未终止且无未来事件时死锁")
            action = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}

        _, _, terminated, truncated, last_info = env.step(action)
        decisions += 1
        assert not truncated
        if terminated:
            break

    assert last_info.get("termination_reason") == "completed"
    assert all(task.status == TaskStatus.COMPLETED for task in env.state.tasks.values())
    assert all(aircraft.is_completed for aircraft in env.state.aircraft.values())
    assert len(env.state.transfer_history) == num_aircraft + env.state.num_stations - 1
    assert not env.disturbance_event_triggered

    feasible, violations = _check_completed_trajectory_feasibility(env)
    assert feasible, f"独立轨迹可行性检查失败：{violations}"
