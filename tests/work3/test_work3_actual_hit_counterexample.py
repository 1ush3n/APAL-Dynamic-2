"""确定性真实命中反例：五站填满并完成一个周期后固定延迟一项工序。"""

from __future__ import annotations

from pathlib import Path

import pytest

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from scripts.work3.evaluate_c_vs_d import _check_completed_trajectory_feasibility
from scripts.work3.train_ppo_work3 import count_actual_scenario_hits
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline
from utils.work3.objective_evaluator import evaluate_trajectory_objective


def test_fixed_disturbance_hits_and_completes_full_batch_with_consistent_ledger() -> None:
    baseline_path = Path("data/work3/real_283_k10_baseline.json")
    if not baseline_path.is_file():
        pytest.skip(f"{baseline_path} 不存在")

    baseline = MultiAircraftBaseline.load_from_json(baseline_path)
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    env.reset()

    # 固定基准指派策略推进至五站填满后的下一次正常同步转站。
    while len(env.state.transfer_history) < 5:
        ready = env.get_ready_tasks()
        if not ready:
            assert not env._check_terminated(), "尚未达到反例注入点就提前完成批次"
            assert not env.event_queue.is_empty(), "尚未达到反例注入点就发生死锁"
            env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
            continue

        task = ready[0]
        team = baseline.get_task(task.aircraft_id, task.task_id).team
        _, _, terminated, truncated, _ = env.step(
            {
                "task_key": task.task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": team,
                "align": 1,
            }
        )
        assert not terminated and not truncated, "反例注入前生产轨迹意外结束"

    tau = env.state.current_time
    delta = 0.15 * env.state.h0  # LOW标定：按基准节拍比例生成
    recovery_time = tau + delta
    target_key = "5_2"

    assert env.state.current_time == pytest.approx(tau, abs=1e-8)
    station_counts = [
        sum(
            aircraft.current_station == station_id and not aircraft.is_completed
            for aircraft in env.state.aircraft.values()
        )
        for station_id in range(env.state.num_stations)
    ]
    assert len(env.state.transfer_history) == 5
    assert station_counts == [1] * env.state.num_stations
    target = env.state.tasks[target_key]
    assert target.current_station == 0
    assert target.status == TaskStatus.READY
    assert target.actual_start is None

    scenario = {
        "scenario_id": "W3_FIXED_ACTUAL_HIT_S0",
        "timing": "EARLY",
        "intensity": "LOW",
        "station_id": 0,
        "aircraft_id": 5,
        "tau": tau,
        "delta": delta,
        "recovery_time": recovery_time,
        "affected_task_keys": [target_key],
    }
    env.load_scenario(scenario)
    assert target.status == TaskStatus.UNREADY
    assert target.material_ready_time == pytest.approx(recovery_time)
    assert count_actual_scenario_hits(env, scenario) == 1

    last_info: dict[str, object] = {}
    decisions = 0
    while decisions < 10000:
        ready = env.get_ready_tasks()
        if ready:
            task = ready[0]
            team = baseline.get_task(task.aircraft_id, task.task_id).team
            _, _, terminated, truncated, last_info = env.step(
                {
                    "task_key": task.task_key,
                    "branch": ActionBranch.STATION_EXECUTE,
                    "team": team,
                    "align": 1,
                }
            )
        else:
            _, _, terminated, truncated, last_info = env.step(
                {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
            )
        decisions += 1
        assert not truncated, "完整批次反例不应被采样上限截断"
        if terminated:
            break

    assert decisions < 10000, "完整批次反例超过保护步数"
    assert last_info.get("success") is True
    assert last_info.get("termination_reason") == "completed"
    assert all(task.status == TaskStatus.COMPLETED for task in env.state.tasks.values())
    assert len(env.state.tasks) == 2830
    assert len(env.state.transfer_history) == 14
    assert target.actual_start is not None
    assert target.actual_start >= recovery_time - env.tolerance
    assert count_actual_scenario_hits(env, scenario) == 1

    feasible, violations = _check_completed_trajectory_feasibility(env)
    assert feasible, f"独立轨迹可行性检查失败：{violations}"

    independent_ledger = evaluate_trajectory_objective(env, weights=env.weights)
    assert sum(env.step_rewards) == pytest.approx(-independent_ledger.j_total, abs=1e-8)
    assert env.cumulative_cost == pytest.approx(independent_ledger.j_total, abs=1e-8)
