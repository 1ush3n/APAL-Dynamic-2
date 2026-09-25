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
FIRST_TASK_END = 0.5224849650851306


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
