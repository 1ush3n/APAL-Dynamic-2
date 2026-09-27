"""M02：单机命中并后移后，追踪同步脉动对另一架飞机的传播。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")
FIVE_STATION_TASKS = ((1, 15), (2, 18), (3, 12), (4, 20), (5, 24))
TEAMS = {
    15: (72, 12),
    18: (18, 65),
    12: (50, 51),
    20: (55, 6),
    24: (28, 1),
}
FIRST_TASK_END = 0.5457256030603916


def _two_aircraft_baseline(path: Path) -> None:
    source = MultiAircraftBaseline.load_from_json(BASELINE_PATH)
    tasks = {}
    for aircraft_id in range(2):
        for station_id, task_id in FIVE_STATION_TASKS:
            source_task = source.get_task(0, task_id)
            assert not source_task.predecessors
            entry = (aircraft_id + station_id - 1) * source.h0
            task_key = f"{aircraft_id}_{task_id}"
            tasks[task_key] = replace(
                source_task,
                aircraft_id=aircraft_id,
                task_key=task_key,
                in_station_offset=(10.0 if aircraft_id == 0 and task_id == 18 else 0.0),
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


def _two_aircraft_env_after_first_pulse(tmp_path: Path) -> AirLineEnvWork3:
    if not BASELINE_PATH.is_file():
        pytest.skip(f"{BASELINE_PATH} 不存在")

    baseline_path = tmp_path / "two_aircraft_progress_baseline.json"
    _two_aircraft_baseline(baseline_path)
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()
    assert env.state.num_stations == 5
    assert len(env.state.aircraft) == 2
    env.step(
        {
            "task_key": "0_15",
            "branch": ActionBranch.STATION_EXECUTE,
            "team": TEAMS[15],
            "align": 0,
        }
    )
    assert env.state.current_time == pytest.approx(FIRST_TASK_END)
    assert env.state.tasks["0_18"].status == TaskStatus.READY
    assert env.state.tasks["1_15"].status == TaskStatus.READY
    return env


def _reserve_small_task_at(env: AirLineEnvWork3, task_key: str, start_time: float) -> None:
    task = env.state.tasks[task_key]
    task.in_station_offset = start_time - env.state.last_transfer_time
    task_id = int(task_key.split("_")[1])
    env.step(
        {
            "task_key": task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": TEAMS[task_id],
            "align": 1,
        }
    )
    assert task.status == TaskStatus.RESERVED
    assert task.scheduled_start == pytest.approx(start_time)


def test_progress_is_available_with_only_a_modifiable_reservation(
    tmp_path: Path,
) -> None:
    """五站双机小例中只剩可修订预约时，策略可选推进至其开工事件。"""
    env = _two_aircraft_env_after_first_pulse(tmp_path)
    reserved = env.state.tasks["0_18"]
    _reserve_small_task_at(env, reserved.task_key, 20.0)
    other_task = env.state.tasks["1_15"]
    other_task.status = TaskStatus.COMPLETED
    other_task.actual_end = env.state.current_time

    assert env.get_ready_tasks() == []
    assert env.get_action_candidates() == [reserved]
    assert env.event_queue.peek() is not None
    assert env.event_queue.peek().timestamp == pytest.approx(20.0)

    _, _, terminated, truncated, info = env.step(
        {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    )

    assert info["advanced"] is True
    assert env.state.current_time == pytest.approx(20.0)
    assert reserved.status == TaskStatus.RUNNING
    assert not terminated
    assert not truncated


def test_identical_resubmission_advances_and_records_remaining_ready_task(
    tmp_path: Path,
) -> None:
    """重复预约会自动推进，但另一架飞机的READY任务仍在返回状态中可见。"""
    env = _two_aircraft_env_after_first_pulse(tmp_path)
    reserved = env.state.tasks["0_18"]
    remaining_ready = env.state.tasks["1_15"]
    _reserve_small_task_at(env, reserved.task_key, 20.0)
    assert remaining_ready in env.get_action_candidates()
    assert reserved in env.get_action_candidates()

    cost_before = env.cumulative_cost
    rewards_before = len(env.step_rewards)
    _, reward, terminated, truncated, info = env.step(
        {
            "task_key": reserved.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": TEAMS[18],
            "align": 1,
        }
    )

    assert info["revision_changed"] is False
    assert info["advanced"] is True
    assert info["scheduled_status"] == "RESERVED"
    assert env.state.current_time == pytest.approx(20.0)
    assert reserved.status == TaskStatus.RUNNING
    assert remaining_ready.status == TaskStatus.READY
    assert remaining_ready in env.get_action_candidates()
    assert reward == pytest.approx(-(env.cumulative_cost - cost_before))
    assert info["step_cost"] == pytest.approx(env.cumulative_cost - cost_before)
    assert len(env.step_rewards) == rewards_before + 1
    assert not terminated
    assert not truncated


def test_same_timestamp_disturbance_precedes_reserved_start_in_small_line(
    tmp_path: Path,
) -> None:
    """五站双机小例中τ与预约开工同刻时，扰动先撤销尚未开工的预约。"""
    env = _two_aircraft_env_after_first_pulse(tmp_path)
    target = env.state.tasks["0_18"]
    _reserve_small_task_at(env, target.task_key, 20.0)
    env.load_scenario(
        {
            "scenario_id": "P12_SAME_TIME_START",
            "tau": 20.0,
            "recovery_time": 25.0,
            "affected_task_keys": [target.task_key],
        }
    )

    _, _, terminated, truncated, info = env.step(
        {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    )

    assert info["advanced"] is True
    assert env.state.current_time == pytest.approx(20.0)
    assert target.status == TaskStatus.UNREADY
    assert target.material_ready_time == pytest.approx(25.0)
    assert target.actual_start is None
    assert env.disturbance_event_results["P12_SAME_TIME_START"][
        "actual_hit_task_keys"
    ] == (target.task_key,)
    assert not terminated
    assert not truncated


def test_no_future_event_is_a_deadlock_not_a_successful_advance(
    tmp_path: Path,
) -> None:
    """五站双机小例无合法任务与未来事件时显式推进按失败终止记录。"""
    if not BASELINE_PATH.is_file():
        pytest.skip(f"{BASELINE_PATH} 不存在")
    baseline_path = tmp_path / "two_aircraft_deadlock_baseline.json"
    _two_aircraft_baseline(baseline_path)
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()
    env.event_queue.reset(start_time=env.state.current_time)
    for task in env.state.tasks.values():
        task.status = TaskStatus.UNREADY

    _, _, terminated, truncated, info = env.step(
        {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    )

    assert terminated is True
    assert truncated is False
    assert info["advanced"] is False
    assert info["success"] is False
    assert info["termination_reason"] == "deadlock"


def test_one_aircraft_disturbance_and_postpone_delays_other_aircraft_pulse(
    tmp_path: Path,
) -> None:
    """未受扰飞机完成本站作业后，仍须等待受扰飞机放行同步转站。"""
    if not BASELINE_PATH.is_file():
        pytest.skip(f"{BASELINE_PATH} 不存在")

    baseline_path = tmp_path / "two_aircraft_baseline.json"
    _two_aircraft_baseline(baseline_path)
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    scenario = {
        "scenario_id": "M02_FIXED_SINGLE_AIRCRAFT_HIT",
        "tau": 1.0,
        "recovery_time": 20.0,
        "affected_task_keys": ["0_18"],
    }
    env.load_scenario(scenario)

    # 固定首个动作完成第一脉动，飞机0进入站2，飞机1同步进入站1。
    _, _, terminated, truncated, _ = env.step(
        {
            "task_key": "0_15",
            "branch": ActionBranch.STATION_EXECUTE,
            "team": TEAMS[15],
            "align": 0,
        }
    )
    target = env.state.tasks["0_18"]
    other_first = env.state.tasks["1_15"]
    other_second = env.state.tasks["1_18"]
    assert not terminated and not truncated
    assert env.state.current_time == pytest.approx(FIRST_TASK_END)
    assert target.status == TaskStatus.READY
    assert env.state.aircraft[1].current_station == 0

    # 飞机进入站2后先登记10小时开工预约；tau=1时扰动命中该未开工预约。
    env.step(
        {
            "task_key": "0_18",
            "branch": ActionBranch.STATION_EXECUTE,
            "team": TEAMS[18],
            "align": 1,
        }
    )
    assert target.status == TaskStatus.RESERVED
    assert target.scheduled_start == pytest.approx(FIRST_TASK_END + 10.0)
    env.step(
        {
            "task_key": "1_15",
            "branch": ActionBranch.STATION_EXECUTE,
            "team": TEAMS[15],
            "align": 0,
        }
    )
    _, _, terminated, truncated, info = env.step(
        {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    )
    assert info.get("advanced") is True
    assert not terminated and not truncated
    assert env.state.current_time == pytest.approx(scenario["tau"])
    assert target.status == TaskStatus.UNREADY
    assert target.material_ready_time == pytest.approx(20.0)
    assert target.scheduled_start is None
    assert target.assigned_team == []
    assert not any(
        interval.task_key == target.task_key
        for calendar in env.state.workers.values()
        for interval in calendar.intervals
    )
    assert env.disturbance_event_results[scenario["scenario_id"]][
        "actual_hit_task_keys"
    ] == ("0_18",)

    # 被命中的未开工工序虽未到料，仍可合法后移；这次动作释放本站放行。
    assert env.get_action_branch_mask(target) == (True, True)
    env.step({"task_key": "0_18", "branch": ActionBranch.POSTPONE})
    # step已推进到下一次脉动；入站刷新把POSTPONED转为等待恢复的UNREADY。
    assert target.status == TaskStatus.UNREADY
    assert target.current_station == 2
    assert target.postpone_count == 1
    assert target.material_ready_time == pytest.approx(20.0)
    assert target.assigned_team == []

    # 未受扰飞机完成站1任务；下一个同步脉动把它推进至站2。
    assert other_first.actual_end is not None
    assert other_first.actual_start == pytest.approx(FIRST_TASK_END)
    assert env.state.aircraft[1].current_station == 1
    assert env.state.transfer_history[-1] == pytest.approx(other_first.actual_end)

    # 后移工序到站3后改选该站合法团队，预约在物料恢复时开工。
    assert target.current_station == env.state.aircraft[0].current_station
    env.step(
        {
            "task_key": "0_18",
            "branch": ActionBranch.STATION_EXECUTE,
            "team": TEAMS[12],
            "align": 0,
        }
    )
    assert target.status == TaskStatus.RESERVED
    assert target.scheduled_start is not None and target.scheduled_start >= 20.0

    # 两架飞机分处站3和站2，可并行处理；未受扰飞机先完成站2任务。
    env.step(
        {
            "task_key": "0_12",
            "branch": ActionBranch.STATION_EXECUTE,
            "team": TEAMS[12],
            "align": 0,
        }
    )
    env.step(
        {
            "task_key": "1_18",
            "branch": ActionBranch.STATION_EXECUTE,
            "team": TEAMS[18],
            "align": 0,
        }
    )
    while target.status != TaskStatus.RUNNING:
        _, _, terminated, truncated, info = env.step(
            {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
        )
        assert info.get("advanced") is True
        assert not terminated and not truncated

    assert other_second.status == TaskStatus.COMPLETED
    assert env.state.aircraft[1].current_station == 1
    assert 1 not in env.state.aircraft[1].exit_times

    while target.status != TaskStatus.COMPLETED:
        _, _, terminated, truncated, info = env.step(
            {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
        )
        assert info.get("advanced") is True
        assert not terminated and not truncated

    pulse_time = target.actual_end
    assert pulse_time is not None and pulse_time >= 20.0
    assert other_second.actual_end is not None and other_second.actual_end < pulse_time
    assert env.state.transfer_history[-1] == pytest.approx(pulse_time)
    assert env.state.aircraft[1].exit_times[1] == pytest.approx(pulse_time)
    assert env.state.aircraft[1].entry_times[2] == pytest.approx(pulse_time)
    assert env.state.tasks["1_12"].status == TaskStatus.READY

    # 被延迟的全线脉动一到，未受扰飞机下一站任务即可开工并完成。
    env.step(
        {
            "task_key": "1_12",
            "branch": ActionBranch.STATION_EXECUTE,
            "team": TEAMS[12],
            "align": 0,
        }
    )
    other_third = env.state.tasks["1_12"]
    while other_third.status != TaskStatus.COMPLETED:
        _, _, terminated, truncated, info = env.step(
            {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
        )
        assert info.get("advanced") is True
        assert not terminated and not truncated
    assert other_third.actual_start == pytest.approx(pulse_time)
    assert all(
        task.actual_start is not None and task.actual_end is not None
        for task in (other_first, other_second, other_third, target)
    )
