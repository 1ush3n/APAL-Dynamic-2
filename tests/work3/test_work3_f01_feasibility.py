"""F01：站位容量全区间检查与联合资源搜索反例。"""

import copy
from itertools import permutations
from pathlib import Path

import pytest

from envs.work3.core_types import ActionBranch
from envs.work3.environment import AirLineEnvWork3, NoFeasibleSlotError
from envs.work3.event_queue import EventType


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")


@pytest.fixture
def env() -> AirLineEnvWork3:
    """创建容量为1的独立工作三环境。"""
    if not BASELINE_PATH.is_file():
        pytest.skip(f"{BASELINE_PATH} 不存在")
    instance = AirLineEnvWork3(
        baseline_json_path=BASELINE_PATH,
        max_slots_per_station=1,
    )
    instance.reset()
    return instance


def mark_station_interval(
    env: AirLineEnvWork3,
    station_id: int,
    start: float,
    duration: float,
) -> None:
    """在环境的站位占用索引中登记一个测试区间。"""
    task = next(
        task for task in env.state.tasks.values() if task.current_station == station_id
    )
    task.scheduled_start = start
    task.duration = duration
    env._station_occupied_tasks[station_id].add(task.task_key)


def mark_station_intervals(
    env: AirLineEnvWork3,
    station_id: int,
    intervals: list[tuple[float, float]],
) -> list[str]:
    """按给定端点把多个独立任务登记到站位占用日历。"""
    tasks = [
        task for task in env.state.tasks.values()
        if task.current_station == station_id
    ][: len(intervals)]
    assert len(tasks) == len(intervals)

    for task, (start, end) in zip(tasks, intervals):
        task.scheduled_start = start
        task.execution_duration = end - start
        env._station_occupied_tasks[station_id].add(task.task_key)
    return [task.task_key for task in tasks]


def test_station_capacity_scans_entire_candidate_interval(env: AirLineEnvWork3) -> None:
    """候选区间内部发生容量冲突时必须判定不可用。"""
    mark_station_interval(env, station_id=0, start=2.0, duration=1.0)

    assert not env._is_station_slot_available(0, 0.0, 10.0)


def test_station_capacity_uses_half_open_endpoint_semantics(env: AirLineEnvWork3) -> None:
    """已有区间的完工时刻可作为候选区间的开工时刻。"""
    mark_station_interval(env, station_id=0, start=2.0, duration=1.0)

    assert env._is_station_slot_available(0, 3.0, 4.0)


def test_team_search_skips_long_station_occupation_without_fake_success(
    env: AirLineEnvWork3,
) -> None:
    """长站位占用不能触发固定步长上限后返回非法时间。"""
    mark_station_interval(env, station_id=0, start=0.0, duration=2000.0)
    worker_id = env.state.station_worker_bindings[0][0]

    scheduled_start = env._find_team_earliest_slot(
        station_id=0,
        team=[worker_id],
        search_start=0.0,
        duration=1.0,
    )

    assert scheduled_start >= 2000.0
    assert env._is_station_slot_available(0, scheduled_start, scheduled_start + 1.0)


def test_team_search_can_fill_station_gap_before_future_reservation(
    env: AirLineEnvWork3,
) -> None:
    """未来预约不应阻止更早的合法回填空隙。"""
    mark_station_interval(env, station_id=0, start=20.0, duration=2.0)
    worker_id = env.state.station_worker_bindings[0][0]

    scheduled_start = env._find_team_earliest_slot(
        station_id=0,
        team=[worker_id],
        search_start=16.0,
        duration=2.0,
    )

    assert scheduled_start == pytest.approx(16.0)


def test_team_busy_only_after_candidate_gap_does_not_push_earlier_work(
    env: AirLineEnvWork3,
) -> None:
    """A03：工人[20,22)忙碌时，较早的[16,18)仍可预约。"""
    worker_id = env.state.station_worker_bindings[0][0]
    env.state.workers[worker_id].add_interval(20.0, 22.0, "future_block")

    earliest = env._find_team_earliest_slot(
        station_id=0,
        team=[worker_id],
        search_start=16.0,
        duration=2.0,
    )

    assert earliest == pytest.approx(16.0)


def test_capacity_two_counts_overlapping_intervals_as_three_with_candidate() -> None:
    """A05：容量2时，既有区间在[2,3)重叠，候选作业会形成三重并发。"""
    env = AirLineEnvWork3(
        baseline_json_path=BASELINE_PATH,
        max_slots_per_station=2,
    )
    env.reset()
    mark_station_intervals(env, 0, [(1.0, 4.0), (2.0, 3.0)])

    assert not env._is_station_slot_available(0, 0.0, 5.0)


def test_station_conflict_advancement_rechecks_worker_calendar(
    env: AirLineEnvWork3,
) -> None:
    """A06：站位先把候选推至4，随后必须越过工人[3,5)占用至5。"""
    mark_station_interval(env, 0, 0.0, 4.0)
    worker_id = env.state.station_worker_bindings[0][0]
    env.state.workers[worker_id].add_interval(3.0, 5.0, "worker_block")

    earliest = env._find_team_earliest_slot(
        station_id=0,
        team=[worker_id],
        search_start=0.0,
        duration=2.0,
    )

    assert earliest == pytest.approx(5.0)


def test_team_search_finds_common_gap_not_individual_gaps(
    env: AirLineEnvWork3,
) -> None:
    """A07：两名工人的空闲区间不重合时，团队最早共同开工时刻为5。"""
    worker_a, worker_b = env.state.station_worker_bindings[0][:2]
    env.state.workers[worker_a].add_interval(0.0, 2.0, "a_block")
    env.state.workers[worker_b].add_interval(3.0, 5.0, "b_block")

    earliest = env._find_team_earliest_slot(
        station_id=0,
        team=[worker_a, worker_b],
        search_start=0.0,
        duration=2.0,
    )

    assert earliest == pytest.approx(5.0)


def test_capacity_scan_is_independent_of_simultaneous_interval_input_order(
    env: AirLineEnvWork3,
) -> None:
    """A08：同刻释放/占用时改变输入区间顺序，不改变容量结论。"""
    env.max_slots_per_station = 2
    tasks = [
        task for task in env.state.tasks.values()
        if task.current_station == 0
    ][:2]
    outcomes: list[bool] = []

    for intervals in permutations([(0.0, 2.0), (2.0, 4.0)]):
        env._station_occupied_tasks[0].clear()
        for task, (start, end) in zip(tasks, intervals):
            task.scheduled_start = start
            task.execution_duration = end - start
            env._station_occupied_tasks[0].add(task.task_key)
        outcomes.append(env._is_station_slot_available(0, 1.0, 3.0))

    assert outcomes == [True, True]


def test_capacity_tolerance_does_not_hide_material_overlap(
    env: AirLineEnvWork3,
) -> None:
    """A09：容差外的重叠必须拒绝，容差内边界按统一容差视为相接。"""
    mark_station_interval(env, 0, 0.0, 2.0)
    tolerance = env.tolerance

    assert not env._is_station_slot_available(0, 2.0 - 2.0 * tolerance, 3.0)
    assert env._is_station_slot_available(0, 2.0 - tolerance / 2.0, 3.0)
    assert env._is_station_slot_available(0, 2.0 + 2.0 * tolerance, 3.0)


def test_failed_reservation_search_restores_old_reservation_transaction(
    env: AirLineEnvWork3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A10：搜索异常时旧预约、资源、事件代数和费用账本完整不变。"""
    task = env.get_ready_tasks()[0]
    team = tuple(env.valid_team_completion_workers(task, [])[: task.demand])
    duration = env.duration_for_team(task, team)
    task.execution_duration = duration
    task.reserve(team=team, scheduled_start=20.0)
    for worker_id in team:
        env.state.workers[worker_id].add_interval(
            20.0, 20.0 + duration, task.task_key
        )
    env._station_occupied_tasks[task.current_station].add(task.task_key)
    env.event_queue.push(
        EventType.TASK_START,
        timestamp=20.0,
        task_key=task.task_key,
        generation=task.generation,
    )

    task_before = (
        task.status,
        tuple(task.assigned_team),
        task.scheduled_start,
        task.execution_duration,
        task.generation,
        copy.deepcopy(task.revision_history),
        copy.deepcopy(task.current_assignment),
    )
    calendars_before = {
        worker_id: tuple(calendar.intervals)
        for worker_id, calendar in env.state.workers.items()
    }
    occupied_before = {
        station_id: frozenset(keys)
        for station_id, keys in env._station_occupied_tasks.items()
    }
    events_before = tuple(
        (event.timestamp, event.priority, event.event_id, event.task_key, event.generation)
        for event in env.event_queue._heap
    )
    generations_before = dict(env.event_queue._task_generations)
    costs_before = (
        env.cumulative_cost,
        env.cost_takt,
        env.cost_time,
        env.cost_team,
        env.cost_postpone,
        env.cost_revision,
        tuple(env.step_rewards),
    )

    def fail_search(**_: object) -> float:
        raise NoFeasibleSlotError("injected search failure")

    monkeypatch.setattr(env, "_find_team_earliest_slot", fail_search)
    with pytest.raises(NoFeasibleSlotError, match="injected search failure"):
        env.step(
            {
                "task_key": task.task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": team,
                "align": 0,
            }
        )

    assert task_before == (
        task.status,
        tuple(task.assigned_team),
        task.scheduled_start,
        task.execution_duration,
        task.generation,
        copy.deepcopy(task.revision_history),
        copy.deepcopy(task.current_assignment),
    )
    assert calendars_before == {
        worker_id: tuple(calendar.intervals)
        for worker_id, calendar in env.state.workers.items()
    }
    assert occupied_before == {
        station_id: frozenset(keys)
        for station_id, keys in env._station_occupied_tasks.items()
    }
    assert events_before == tuple(
        (event.timestamp, event.priority, event.event_id, event.task_key, event.generation)
        for event in env.event_queue._heap
    )
    assert generations_before == env.event_queue._task_generations
    assert costs_before == (
        env.cumulative_cost,
        env.cost_takt,
        env.cost_time,
        env.cost_team,
        env.cost_postpone,
        env.cost_revision,
        tuple(env.step_rewards),
    )


@pytest.mark.parametrize(
    "duration",
    [-1e-6, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_team_search_rejects_invalid_duration(
    env: AirLineEnvWork3,
    duration: float,
) -> None:
    """A11：负数、NaN或无穷工时必须拒绝，不能返回看似可行的时间。"""
    worker_id = env.state.station_worker_bindings[0][0]

    with pytest.raises(ValueError, match="工时"):
        env._find_team_earliest_slot(
            station_id=0,
            team=[worker_id],
            search_start=0.0,
            duration=duration,
        )


def test_station_capacity_blocks_distinct_workers_from_overlapping_future_slots(
    env: AirLineEnvWork3,
) -> None:
    """A12：资源层候选工人空闲，也仍受既有站位预约容量限制。"""
    worker_a, worker_b = env.state.station_worker_bindings[0][:2]
    mark_station_interval(env, 0, 20.0, 2.0)
    existing_task = next(iter(env._station_occupied_tasks[0]))
    env.state.workers[worker_a].add_interval(20.0, 22.0, existing_task)

    assert worker_a != worker_b
    assert env.state.workers[worker_b].is_available(20.0, 22.0)
    earliest = env._find_team_earliest_slot(
        station_id=0,
        team=[worker_b],
        search_start=20.0,
        duration=2.0,
    )

    assert earliest == pytest.approx(22.0)
