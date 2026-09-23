"""B组：同步脉动、跨架次物理位置与批次终止边界。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from envs.work3.event_queue import EventType
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")


def _new_env() -> AirLineEnvWork3:
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    return env


def _configure_station_finish_events(
    env: AirLineEnvWork3,
    *,
    last_transfer_time: float,
    finish_times: tuple[float, float, float, float, float],
) -> dict[int, int]:
    """为五架当前在制飞机构造真实TASK_FINISH事件，隔离测试同步放行。"""
    env.reset()
    env.state.h0 = 10.0
    env.state.last_transfer_time = last_transfer_time
    env.state.current_time = last_transfer_time
    env.state.current_cycle = 1 if last_transfer_time == 0.0 else 2
    env.state.transfer_history = [] if last_transfer_time == 0.0 else [last_transfer_time]
    env.event_queue.reset(start_time=last_transfer_time)
    env._transfer_scheduled_for_cycle = 0
    env._station_occupied_tasks = {station: set() for station in range(5)}

    for aircraft in env.state.aircraft.values():
        aircraft.current_station = -1

    initial_positions: dict[int, int] = {}
    for station_id, finish_time in enumerate(finish_times):
        aircraft_id = station_id
        env.state.aircraft[aircraft_id].current_station = station_id
        initial_positions[aircraft_id] = station_id
        station_tasks = env.state._ac_station_tasks[(aircraft_id, station_id)]
        for task in station_tasks:
            task.status = TaskStatus.COMPLETED
            task.actual_end = last_transfer_time

        running_task = next(task for task in station_tasks if task.duration > env.tolerance)
        running_task.status = TaskStatus.RUNNING
        running_task.actual_start = last_transfer_time
        running_task.actual_end = None
        running_task.execution_duration = finish_time - last_transfer_time
        running_task.assigned_team = list(running_task.base_team)
        env._station_occupied_tasks[station_id].add(running_task.task_key)
        env.event_queue.push(
            event_type=EventType.TASK_FINISH,
            timestamp=finish_time,
            task_key=running_task.task_key,
            generation=running_task.generation,
        )

    return initial_positions


def _assert_unique_in_line_stations(env: AirLineEnvWork3) -> int:
    stations = [
        aircraft.current_station
        for aircraft in env.state.aircraft.values()
        if aircraft.is_in_factory
    ]
    assert len(stations) <= env.state.num_stations
    assert len(stations) == len(set(stations))
    return len(stations)


@pytest.mark.parametrize(
    ("last_transfer_time", "finish_times", "expected_transfer", "expected_cycle_h"),
    [
        (0.0, (2.0, 4.0, 12.0, 6.0, 8.0), 12.0, 12.0),
        (7.0, (19.0, 19.0, 19.0, 19.0, 19.0), 19.0, 12.0),
    ],
)
def test_transfer_waits_for_latest_real_station_finish(
    last_transfer_time: float,
    finish_times: tuple[float, float, float, float, float],
    expected_transfer: float,
    expected_cycle_h: float,
) -> None:
    """不同站位完工事件未全部发生前，任何飞机都不能先行转站。"""
    env = _new_env()
    initial_positions = _configure_station_finish_events(
        env,
        last_transfer_time=last_transfer_time,
        finish_times=finish_times,
    )

    while not env.state.transfer_history or env.state.transfer_history[-1] != expected_transfer:
        event = env.event_queue.peek()
        assert event is not None
        assert event.event_type in (EventType.TASK_FINISH, EventType.SYNCHRONOUS_TRANSFER)
        env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
        if not env.state.transfer_history or env.state.transfer_history[-1] != expected_transfer:
            assert {
                aircraft_id: env.state.aircraft[aircraft_id].current_station
                for aircraft_id in initial_positions
            } == initial_positions

    assert env.state.transfer_history[-1] == pytest.approx(expected_transfer)
    assert expected_transfer - last_transfer_time == pytest.approx(expected_cycle_h)
    assert env.state.aircraft[0].current_station == 1
    assert env.state.aircraft[1].current_station == 2
    assert env.state.aircraft[2].current_station == 3
    assert env.state.aircraft[3].current_station == 4
    assert env.state.aircraft[4].current_station == 5
    assert env.cost_takt == pytest.approx(env.weights.w_h * 0.2)


def test_active_or_cancelled_reservation_never_counts_as_cycle_completion() -> None:
    """未开工预约阻止放行；扰动取消预约后任务仍未完成，继续阻止放行。"""
    env = _new_env()
    env.state.current_time = 3.0
    env.event_queue.reset(start_time=3.0)
    for aircraft in env.state.aircraft.values():
        aircraft.current_station = -1
    env.state.aircraft[0].current_station = 0
    for task in env.state.tasks.values():
        task.status = TaskStatus.COMPLETED
        task.actual_end = 3.0

    task = env.state.tasks["0_2"]
    team = tuple(env.valid_team_completion_workers(task, [])[: task.demand])
    assert len(team) == task.demand
    task.status = TaskStatus.RESERVED
    task.actual_start = None
    task.actual_end = None
    task.assigned_team = list(team)
    task.scheduled_start = 8.0
    task.execution_duration = 2.0
    for worker_id in team:
        env.state.workers[worker_id].add_interval(
            start=8.0,
            end=10.0,
            task_key=task.task_key,
        )
    env._station_occupied_tasks[0].add(task.task_key)
    env.event_queue.push(
        event_type=EventType.TASK_START,
        timestamp=8.0,
        task_key=task.task_key,
        generation=task.generation,
    )

    env._check_and_schedule_transfer()
    assert all(
        event.event_type != EventType.SYNCHRONOUS_TRANSFER
        for event in env.event_queue._heap
    )

    env.load_scenario(
        {"tau": 3.0, "recovery_time": 20.0, "affected_task_keys": [task.task_key]}
    )

    assert task.status == TaskStatus.UNREADY
    assert task.actual_end is None
    assert not any(
        interval.task_key == task.task_key
        for worker in env.state.workers.values()
        for interval in worker.intervals
    )
    assert all(
        event.event_type != EventType.SYNCHRONOUS_TRANSFER
        for event in env.event_queue._heap
    )


def test_line_filling_and_draining_preserve_one_aircraft_per_station() -> None:
    """真实脉动事件下，空线填充、满线运行和排空保持实体守恒与站位互斥。"""
    env = _new_env()
    observed_counts = [_assert_unique_in_line_stations(env)]

    while not env._check_terminated():
        for station_id in range(env.state.num_stations):
            aircraft_id = env.state.get_aircraft_at_station(station_id)
            if aircraft_id is None:
                continue
            for task in env.state._ac_station_tasks[(aircraft_id, station_id)]:
                task.status = TaskStatus.COMPLETED
                task.actual_end = env.state.current_time

        env._check_and_schedule_transfer()
        event = env.event_queue.peek()
        assert event is not None
        assert event.event_type == EventType.SYNCHRONOUS_TRANSFER
        env.process_due_events()
        observed_counts.append(_assert_unique_in_line_stations(env))

        assert len(env.state.transfer_history) <= env.state.num_aircraft + env.state.num_stations - 1

    assert observed_counts == [1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0]
    assert len(env.state.transfer_history) == 14
    assert all(task.status == TaskStatus.COMPLETED for task in env.state.tasks.values())


def test_terminated_batch_ignores_future_stale_transfer_event() -> None:
    """最后一架飞机出线后，队列残留的未来脉动事件不能再推进时间或计次。"""
    env = _new_env()
    env.state.current_time = 2.0
    env.state.last_transfer_time = 0.0
    env.state.current_cycle = 14
    env.state.transfer_history = [0.0] * 13
    env.event_queue.reset(start_time=2.0)
    env._transfer_scheduled_for_cycle = 0
    env._station_occupied_tasks = {station: set() for station in range(5)}

    for aircraft in env.state.aircraft.values():
        aircraft.current_station = 5
    env.state.aircraft[0].current_station = 4
    for task in env.state.tasks.values():
        task.status = TaskStatus.COMPLETED
        task.actual_end = 2.0

    env._check_and_schedule_transfer()
    real_transfer = env.event_queue.peek()
    assert real_transfer is not None
    assert real_transfer.event_type == EventType.SYNCHRONOUS_TRANSFER
    assert real_transfer.timestamp == pytest.approx(2.0)
    env.event_queue.push(
        event_type=EventType.SYNCHRONOUS_TRANSFER,
        timestamp=5.0,
        payload={"cycle": 14},
    )

    _observation, _reward, terminated, truncated, _info = env.step(
        {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    )

    assert terminated
    assert not truncated
    assert env.state.current_time == pytest.approx(2.0)
    assert env.state.transfer_history[-1] == pytest.approx(2.0)
    assert len(env.state.transfer_history) == 14
    assert env.state.current_cycle == 15
    assert all(aircraft.is_completed for aircraft in env.state.aircraft.values())


def test_same_time_transfers_are_finite_and_each_changes_line_state() -> None:
    """同刻连续脉动只在真实改变飞机位置时发生，批次完成后立即停止。"""
    env = _new_env()
    env.state.current_time = 2.0
    env.event_queue.reset(start_time=2.0)
    env._transfer_scheduled_for_cycle = 0
    env._station_occupied_tasks = {station: set() for station in range(5)}

    for aircraft in env.state.aircraft.values():
        aircraft.current_station = -1
    env.state.aircraft[0].current_station = 0
    env.state.aircraft[0].entry_times[0] = 2.0
    for task in env.state.tasks.values():
        task.status = TaskStatus.COMPLETED
        task.actual_end = 2.0

    env._check_and_schedule_transfer()
    _observation, _reward, terminated, truncated, _info = env.step(
        {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    )

    assert terminated
    assert not truncated
    assert env.state.current_time == pytest.approx(2.0)
    assert len(env.state.transfer_history) == 14
    assert all(timestamp == pytest.approx(2.0) for timestamp in env.state.transfer_history)
    assert all(aircraft.is_completed for aircraft in env.state.aircraft.values())
    assert env.event_queue.peek() is None


def test_postponed_task_waits_for_target_aircraft_and_station_to_clear() -> None:
    """跨站任务在飞机到站前不可见；原占站飞机脉动离开后才可预约加工。"""
    env = _new_env()
    task = env.state.tasks["0_18"]
    env.state.aircraft[0].current_station = 1
    env.state.aircraft[1].current_station = 2
    assert task.current_station == 1
    assert env.can_postpone(task)
    original_target_task_keys = {
        candidate.task_key
        for candidate in env.state._ac_station_tasks[(0, 2)]
    }

    env.step({"task_key": task.task_key, "branch": ActionBranch.POSTPONE})

    assert task.current_station == 2
    assert env.state.aircraft[0].current_station == 1
    assert env.state.aircraft[1].current_station == 2
    assert task not in env.get_action_candidates()
    assert task not in env.state.get_tasks_for_station(2)
    assert task.assigned_team == []
    assert task.task_key not in env._station_occupied_tasks[2]

    for candidate in env.state._ac_station_tasks[(0, 1)]:
        if candidate.task_key != task.task_key:
            candidate.status = TaskStatus.COMPLETED
            candidate.actual_end = env.state.current_time
    for candidate in env.state._ac_station_tasks[(1, 2)]:
        candidate.status = TaskStatus.COMPLETED
        candidate.actual_end = env.state.current_time

    env._check_and_schedule_transfer()
    transfer = env.event_queue.peek()
    assert transfer is not None
    assert transfer.event_type == EventType.SYNCHRONOUS_TRANSFER
    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})

    assert env.state.aircraft[0].current_station == 2
    assert env.state.aircraft[1].current_station == 3
    assert _assert_unique_in_line_stations(env) == 2
    station_two_keys = {
        candidate.task_key for candidate in env.state.get_tasks_for_station(2)
    }
    assert task in env.state.get_tasks_for_station(2)
    assert original_target_task_keys.issubset(station_two_keys)
    assert task.status == TaskStatus.READY

    team = tuple(
        env.valid_team_completion_workers(task, [], station_id=2)[: task.demand]
    )
    assert len(team) == task.demand
    assert set(team).issubset(env.state.station_worker_bindings[2])
    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 0,
        }
    )
    assert task.actual_start is not None
    assert task.actual_start >= env.state.aircraft[0].entry_times[2] - env.tolerance


@pytest.mark.parametrize("actual_pulse", [2.0, 12.0])
def test_next_aircraft_enters_at_actual_pulse_without_rewriting_baseline(
    tmp_path: Path, actual_pulse: float
) -> None:
    """下一架飞机按实际首脉动进线，基准名义投产记录保持不变。"""
    baseline = MultiAircraftBaseline.load_from_json(BASELINE_PATH)
    next_aircraft_tasks = [
        replace(task, nominal_station_entry=10.0)
        for task in baseline.get_tasks_for_aircraft(1)
        if task.station_id == 1
    ]
    assert next_aircraft_tasks
    baseline.tasks.update({task.task_key: task for task in next_aircraft_tasks})
    baseline_path = tmp_path / "baseline.json"
    baseline.save_to_json(baseline_path)
    baseline_bytes = baseline_path.read_bytes()

    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()
    env.state.current_time = actual_pulse
    env.event_queue.reset(start_time=actual_pulse)
    for task in env.state._ac_station_tasks[(0, 0)]:
        task.status = TaskStatus.COMPLETED
        task.actual_end = actual_pulse

    env._check_and_schedule_transfer()
    _observation, _reward, _terminated, _truncated, _info = env.step(
        {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    )

    next_aircraft = env.state.aircraft[1]
    assert next_aircraft.current_station == 0
    assert next_aircraft.entry_times[0] == pytest.approx(actual_pulse)
    stored_baseline = MultiAircraftBaseline.load_from_json(baseline_path)
    assert all(
        stored_baseline.tasks[task.task_key].nominal_station_entry == 10.0
        for task in next_aircraft_tasks
    )
    assert baseline_path.read_bytes() == baseline_bytes


def test_repeated_postpone_requires_real_transfer_and_adds_repeat_cost() -> None:
    """同周期不能连跳；真实到达下一站后再次合法后移并增加累进费用。"""
    env = _new_env()
    task = env.state.tasks["0_29"]
    env.state.current_time = 5.0
    env.state.last_transfer_time = 2.0
    env.state.current_cycle = 3
    env.state.transfer_history = [1.0, 2.0]
    env.event_queue.reset(start_time=5.0)
    env._transfer_scheduled_for_cycle = 2
    for aircraft in env.state.aircraft.values():
        aircraft.current_station = -1
    env.state.aircraft[0].current_station = 2
    env.state.aircraft[1].current_station = 1
    env.state.aircraft[2].current_station = 0
    env._station_occupied_tasks = {station: set() for station in range(5)}

    station_two_tasks = env.state._ac_station_tasks[(0, 2)]
    for candidate in station_two_tasks:
        candidate.status = TaskStatus.COMPLETED
        candidate.actual_end = 5.0
    blocker = next(
        candidate
        for candidate in station_two_tasks
        if candidate.task_key != task.task_key
        and len(env.valid_team_completion_workers(candidate, [])) >= candidate.demand
    )
    task.status = TaskStatus.READY
    blocker.status = TaskStatus.READY
    for predecessor_id in blocker.predecessors:
        predecessor = env.state.tasks[f"{blocker.aircraft_id}_{predecessor_id}"]
        predecessor.status = TaskStatus.COMPLETED
        predecessor.actual_end = 5.0

    assert env.can_postpone(task)
    assert env.can_reserve(blocker)
    env.step({"task_key": task.task_key, "branch": ActionBranch.POSTPONE})
    first_postpone_cost = env.cost_postpone
    assert task.current_station == 3
    assert env.state.aircraft[0].current_station == 2
    assert not env.can_postpone(task)
    with pytest.raises(ValueError):
        env.step({"task_key": task.task_key, "branch": ActionBranch.POSTPONE})
    assert env.cost_postpone == pytest.approx(first_postpone_cost)

    blocker.status = TaskStatus.COMPLETED
    blocker.actual_end = 5.0
    for aircraft_id, station_id in ((1, 1), (2, 0)):
        for candidate in env.state._ac_station_tasks[(aircraft_id, station_id)]:
            candidate.status = TaskStatus.COMPLETED
            candidate.actual_end = 5.0
    env._check_and_schedule_transfer()
    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})

    assert env.state.aircraft[0].current_station == 3
    assert task.status == TaskStatus.READY
    assert env.can_postpone(task)
    cost_before_repeat = env.cost_postpone
    env.step({"task_key": task.task_key, "branch": ActionBranch.POSTPONE})

    assert task.postpone_count == 2
    assert task.current_station == 4
    assert env.cost_postpone - cost_before_repeat == pytest.approx(env.weights.w_p * 0.3)
