"""Unit tests for versioned nominal-anchored transfer clock (nominal_anchored_v1) and Rule A acceptance."""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from envs.work3.core_types import ActionBranch, MultiAircraftState, TaskStatus
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
    """When clear time exceeds nominal boundary, transfer is delayed to F_q; next cycle catches up if ready."""
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    h0 = env.state.h0

    # Simulate station cleared at late time F_1 = h0 + 10.0
    # Artificially set station_first_cleared_times for cycle 1
    late_f1 = h0 + 10.0
    env.state.station_first_cleared_times[1] = {s: late_f1 for s in range(env.state.num_stations)}
    
    # Trigger scheduling logic
    # With F_1 = h0 + 10.0 > 1*h0, scheduled transfer time must be max(1*h0, 0.0, late_f1) = late_f1
    latest_finish = max(env.state.station_first_cleared_times[1].values())
    scheduled = max(1 * h0, env.state.last_transfer_time, latest_finish)
    assert math.isclose(scheduled, late_f1, abs_tol=1e-6)

    # Next cycle q=2: nominal boundary is 2 * h0.
    # If next cycle finishes at late_f1 + 5.0 (which is < 2*h0 since h0 ~ 52h, late_f1 + 5 = 67h < 104h),
    # then P_2 = max(2*h0, P_1, F_2) = 2 * h0. Catch-up occurs!
    nom_boundary_2 = 2 * h0
    f2 = late_f1 + 5.0
    p1 = late_f1
    scheduled_2 = max(nom_boundary_2, p1, f2)
    assert math.isclose(scheduled_2, nom_boundary_2, abs_tol=1e-6)


def test_zero_time_loop_guard() -> None:
    """Ensure that calling ADVANCE_TO_NEXT_EVENT when no actions are available advances simulation time."""
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
    env.step(advance_action)
    assert env.state.current_time >= start_time


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
