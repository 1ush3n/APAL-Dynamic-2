"""Unit tests for versioned nominal-anchored transfer clock (nominal_anchored_v1) and Rule A acceptance."""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from envs.work3.core_types import ActionBranch, MultiAircraftState, TaskStatus
from envs.work3.event_queue import EventType
from envs.work3.environment import AirLineEnvWork3
from scripts.work3.baseline_repair_agent import BaselinePlanRepairAgent
from scripts.work3.evaluate_c_vs_d import evaluate_single_trajectory
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


BASELINE_PATH = Path("work3_4060/rule_baseline_ac_v1/mini210_rule_baseline.json")


def test_baseline_metadata_and_policy() -> None:
    """Verify baseline file specifies nominal_anchored_v1 and valid H0."""
    baseline = MultiAircraftBaseline.load_from_json(BASELINE_PATH)
    assert baseline.transfer_policy == "nominal_anchored_v1"
    assert baseline.metadata.get("transfer_policy") == "nominal_anchored_v1"
    assert baseline.h0 > 0.0
    assert math.isclose(baseline.h0, 52.037844126999545, rel_tol=1e-5)


def test_early_finish_waits_for_nominal_boundary() -> None:
    """When stations clear earlier than nominal boundary P_q^0, transfer waits until P_q^0."""
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    assert env.transfer_policy == "nominal_anchored_v1"
    
    h0 = env.state.h0
    nom_boundary_1 = 1 * h0
    
    # Mark tasks in station 0 for plane 0 as completed early at 10.0h
    for t in env.state.get_tasks_for_station(0):
        t.status = TaskStatus.COMPLETED
        t.actual_end = 10.0
    env.state.current_time = 10.0
    env._record_station_first_cleared_if_ready()
    
    # Check if transfer gets scheduled at nom_boundary_1
    env._check_and_schedule_transfer()
    
    # Find scheduled SYNCHRONOUS_TRANSFER event
    transfer_events = [ev for ev in env.event_queue._heap if ev.event_type.name == "SYNCHRONOUS_TRANSFER"]
    assert len(transfer_events) == 1
    assert math.isclose(transfer_events[0].timestamp, nom_boundary_1, abs_tol=1e-5)


def test_delayed_finish_extends_transfer_and_subsequent_catchup() -> None:
    """When clear time exceeds nominal boundary, transfer is delayed to F_q; next cycle catches up if ready in real env."""
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    h0 = env.state.h0

    # 1. 模拟周期 1 发生严重超期：所有站位清空时刻推迟至 h0 + 5.0h (> h0)
    late_f1 = h0 + 5.0
    for t in env.state.get_tasks_for_station(0):
        t.status = TaskStatus.COMPLETED
        t.actual_end = late_f1
    env.state.current_time = late_f1
    env._record_station_first_cleared_if_ready()
    env._check_and_schedule_transfer()

    # 推进事件直到周期 1 转站执行完毕
    while env.state.current_cycle == 1 and not env.event_queue.is_empty():
        ev = env.event_queue.pop()
        env._dispatch_event(ev)

    assert env.state.current_cycle == 2
    assert math.isclose(env.state.last_transfer_time, late_f1, abs_tol=1e-5)
    cycle1_rec = env.state.cycle_execution_records[1]
    assert math.isclose(cycle1_rec.transfer_time, late_f1, abs_tol=1e-5)
    assert math.isclose(cycle1_rec.delay_relative_to_nominal, 5.0, abs_tol=1e-5)

    # 2. 周期 2 任务顺利提前清空：清空时刻为 late_f1 + 5.0h (约为 h0 + 10.0h < 2 * h0)
    early_f2 = late_f1 + 5.0
    nom_boundary_2 = 2 * h0
    assert early_f2 < nom_boundary_2

    for s in range(env.state.num_stations):
        for t in env.state.get_tasks_for_station(s):
            t.status = TaskStatus.COMPLETED
            t.actual_end = early_f2
    env.state.current_time = early_f2
    env._record_station_first_cleared_if_ready()
    env._check_and_schedule_transfer()

    # 推进事件直到周期 2 转站执行完毕
    while env.state.current_cycle == 2 and not env.event_queue.is_empty():
        ev = env.event_queue.pop()
        env._dispatch_event(ev)

    assert env.state.current_cycle == 3
    # 验证回补成功：转站时间恢复到名义节拍锚定边界 2 * h0！
    assert math.isclose(env.state.last_transfer_time, nom_boundary_2, abs_tol=1e-5)
    cycle2_rec = env.state.cycle_execution_records[2]
    assert math.isclose(cycle2_rec.transfer_time, nom_boundary_2, abs_tol=1e-5)
    assert math.isclose(cycle2_rec.delay_relative_to_nominal, 0.0, abs_tol=1e-5)


def test_safety_guard_rejects_transfer_when_station_not_cleared() -> None:
    """验证安全守卫：当站位有正在执行的任务或未清空时，拒绝执行同步转站。"""
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()

    # 人为向事件队列推入一个当前时刻的转站事件，但站位 0 仍有未完工任务
    env.event_queue.push(
        event_type=EventType.SYNCHRONOUS_TRANSFER,
        timestamp=env.state.current_time,
        payload={"cycle": 1},
    )
    assert not env.state.is_all_stations_cleared_for_transfer()

    # 处理此转站事件
    ev = env.event_queue.pop()
    env._dispatch_event(ev)

    # 转站必须被拒绝：周期仍为 1，last_transfer_time 仍为 0.0
    assert env.state.current_cycle == 1
    assert env.state.last_transfer_time == 0.0
    assert 1 not in env.state.cycle_execution_records


def test_zero_time_loop_guard() -> None:
    """Ensure that calling ADVANCE_TO_NEXT_EVENT advances simulation time or makes concrete progress."""
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    start_time = env.state.current_time

    # Reserve the first candidate
    agent = BaselinePlanRepairAgent()
    action = agent.select_action(env)
    assert action is not None
    assert action["branch"] == ActionBranch.STATION_EXECUTE
    env.step(action)

    # Next, advance to next event
    advance_action = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    obs, r, term, trunc, info = env.step(advance_action)
    assert env.state.current_time > start_time or term or trunc


def test_baseline_repair_agent_a_acceptance_on_nominal() -> None:
    """Full acceptance of Rule A on Nominal under rule baseline: 210 tasks, 0 violations, J_total <= 1e-9."""
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    agent = BaselinePlanRepairAgent()
    res = evaluate_single_trajectory(env, "Baseline-Repair-A", agent)

    assert res["success"] is True
    assert res["feasible"] is True
    assert res["completed_tasks"] == 210
    assert len(env.state.tasks) == 210
    assert res["constraint_violation_count"] == 0
    assert res["postponed_count"] == 0
    assert res["transfers"] == 11
    assert res["objective_j_total"] <= 1e-9
    assert math.isclose(res["makespan"], 11 * env.state.h0, rel_tol=1e-6)
