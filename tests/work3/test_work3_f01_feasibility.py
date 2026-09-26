"""F01：站位容量全区间检查与联合资源搜索反例。"""

import copy
from itertools import permutations
from pathlib import Path
import random

import pytest

from envs.work3.core_types import ActionBranch, TaskStatus
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


def endpoint_oracle_capacity_available(
    intervals: list[tuple[float, float]],
    capacity: int,
    start: float,
    end: float,
) -> bool:
    """用独立端点分段枚举判断半开候选区间是否满足容量。"""
    if end <= start:
        return True
    points = {start, end}
    for interval_start, interval_end in intervals:
        if interval_start < end and start < interval_end:
            points.add(max(start, interval_start))
            points.add(min(end, interval_end))

    ordered_points = sorted(points)
    for left, right in zip(ordered_points, ordered_points[1:]):
        if left >= right:
            continue
        active = sum(
            interval_start <= left < interval_end
            for interval_start, interval_end in intervals
        )
        if active >= capacity:
            return False
    return True


def endpoint_oracle_earliest_start(
    station_intervals: list[tuple[float, float]],
    worker_intervals: list[list[tuple[float, float]]],
    capacity: int,
    search_start: float,
    duration: float,
) -> float:
    """枚举搜索起点与资源释放端点，独立求最早联合可行时刻。"""
    all_intervals = [*station_intervals, *(iv for rows in worker_intervals for iv in rows)]
    candidate_starts = {search_start}
    candidate_starts.update(
        interval_end
        for _, interval_end in all_intervals
        if interval_end >= search_start
    )
    for candidate_start in sorted(candidate_starts):
        candidate_end = candidate_start + duration
        workers_available = all(
            not any(
                candidate_start < interval_end and interval_start < candidate_end
                for interval_start, interval_end in intervals
            )
            for intervals in worker_intervals
        )
        if workers_available and endpoint_oracle_capacity_available(
            station_intervals,
            capacity,
            candidate_start,
            candidate_end,
        ):
            return candidate_start
    raise AssertionError("有限资源端点后应存在联合可行时刻")


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


def test_n03_random_resource_calendars_match_independent_endpoint_oracle(
    env: AirLineEnvWork3,
) -> None:
    """固定种子的随机小日历，其容量与最早联合预约均匹配独立端点oracle。"""
    rng = random.Random(20260926)
    assert not endpoint_oracle_capacity_available([(2.0, 3.0)], 1, 0.0, 10.0)
    assert endpoint_oracle_capacity_available([(2.0, 3.0)], 1, 3.0, 4.0)
    saw_available = False
    saw_conflict = False
    saw_station_intervals = False
    saw_worker_intervals = False
    saw_delayed_start = False
    saw_worker_caused_delay = False
    saw_station_caused_delay = False

    for case_index in range(40):
        for task in env.state.tasks.values():
            if task.current_station == 0:
                task.scheduled_start = None
                task.execution_duration = None
        env._station_occupied_tasks[0].clear()
        for calendar in env.state.workers.values():
            calendar.intervals.clear()

        capacity = rng.randint(1, 3)
        env.max_slots_per_station = capacity
        station_intervals = [
            (float(start), float(start + duration))
            for start, duration in (
                (rng.randint(0, 14), rng.randint(1, 5))
                for _ in range(rng.randint(0, 5))
            )
        ]
        saw_station_intervals |= bool(station_intervals)
        mark_station_intervals(env, 0, station_intervals)

        station_worker_ids = env.state.station_worker_bindings[0]
        team = rng.sample(station_worker_ids, k=rng.randint(1, 2))
        worker_intervals: list[list[tuple[float, float]]] = []
        for team_index, worker_id in enumerate(team):
            intervals: list[tuple[float, float]] = []
            for interval_index in range(rng.randint(0, 4)):
                start = float(rng.randint(0, 14))
                end = start + float(rng.randint(1, 4))
                if all(
                    not (start < prior_end and prior_start < end)
                    for prior_start, prior_end in intervals
                ):
                    intervals.append((start, end))
                    env.state.workers[worker_id].add_interval(
                        start,
                        end,
                        f"n03_{case_index}_{team_index}_{interval_index}",
                    )
            worker_intervals.append(intervals)
        saw_worker_intervals |= any(worker_intervals)

        probe_start = float(rng.randint(0, 16))
        probe_duration = float(rng.randint(1, 5))
        expected_available = endpoint_oracle_capacity_available(
            station_intervals,
            capacity,
            probe_start,
            probe_start + probe_duration,
        )
        actual_available = env._is_station_slot_available(
            0,
            probe_start,
            probe_start + probe_duration,
        )
        assert actual_available is expected_available, f"容量用例 {case_index}"
        saw_available |= expected_available
        saw_conflict |= not expected_available

        search_start = float(rng.randint(0, 12))
        duration = float(rng.randint(1, 5))
        expected_start = endpoint_oracle_earliest_start(
            station_intervals,
            worker_intervals,
            capacity,
            search_start,
            duration,
        )
        station_only_start = endpoint_oracle_earliest_start(
            station_intervals,
            [],
            capacity,
            search_start,
            duration,
        )
        worker_only_start = endpoint_oracle_earliest_start(
            [],
            worker_intervals,
            capacity,
            search_start,
            duration,
        )
        actual_start = env._find_team_earliest_slot(
            station_id=0,
            team=team,
            search_start=search_start,
            duration=duration,
        )
        assert actual_start == pytest.approx(expected_start), f"预约用例 {case_index}"
        saw_delayed_start |= expected_start > search_start
        saw_worker_caused_delay |= expected_start > station_only_start
        saw_station_caused_delay |= expected_start > worker_only_start

    assert saw_available and saw_conflict
    assert saw_station_intervals and saw_worker_intervals and saw_delayed_start
    assert saw_worker_caused_delay and saw_station_caused_delay


def test_n04_more_capacity_or_removed_occupancy_never_delays_static_search(
    env: AirLineEnvWork3,
) -> None:
    """固定站位日历下增加容量或删除占用，只能保持或提前最早预约。"""
    occupied_keys = mark_station_intervals(
        env,
        0,
        [(0.0, 3.0), (4.0, 6.0)],
    )
    worker_id = env.state.station_worker_bindings[0][0]
    search_args = {
        "station_id": 0,
        "team": [worker_id],
        "search_start": 0.0,
        "duration": 2.0,
    }
    original_occupancy = frozenset(env._station_occupied_tasks[0])
    original_worker_intervals = tuple(env.state.workers[worker_id].intervals)

    earliest_capacity_one = env._find_team_earliest_slot(**search_args)
    env.max_slots_per_station = 2
    earliest_capacity_two = env._find_team_earliest_slot(**search_args)

    assert earliest_capacity_one == pytest.approx(6.0)
    assert earliest_capacity_two == pytest.approx(0.0)
    assert frozenset(env._station_occupied_tasks[0]) == original_occupancy
    assert tuple(env.state.workers[worker_id].intervals) == original_worker_intervals

    env.max_slots_per_station = 1
    removed_key = occupied_keys[1]
    env._station_occupied_tasks[0].remove(removed_key)
    env.state.tasks[removed_key].scheduled_start = None
    env.state.tasks[removed_key].execution_duration = None
    earliest_after_removal = env._find_team_earliest_slot(**search_args)

    assert earliest_after_removal == pytest.approx(3.0)
    assert earliest_after_removal <= earliest_capacity_one


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
        copy.deepcopy(task.last_published_assignment),
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
        copy.deepcopy(task.last_published_assignment),
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


@pytest.mark.parametrize("recovery_time", [0.0, 1.0])
def test_zero_duration_task_finishes_before_next_policy_observation(
    env: AirLineEnvWork3,
    recovery_time: float,
) -> None:
    """零工时任务无论立即或未来开工，都应在开工同刻完成。"""
    task = env.get_ready_tasks()[0]
    team = tuple(env.valid_team_completion_workers(task, [])[: task.demand])
    assert len(team) == task.demand
    assert len(env.get_action_candidates()) > 1
    task.standard_duration = 0.0
    task.material_ready_time = recovery_time
    if recovery_time > 0.0:
        task.status = TaskStatus.UNREADY

    _, _, terminated, _, _ = env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 0,
        }
    )
    if recovery_time > 0.0:
        assert task.status == TaskStatus.RESERVED
        env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})

    assert task.status == TaskStatus.COMPLETED
    assert task.actual_start == pytest.approx(recovery_time)
    assert task.actual_end == pytest.approx(recovery_time)
    assert task.task_key not in env._station_occupied_tasks[task.current_station]
    assert all(
        env.state.workers[worker_id].is_available(recovery_time, recovery_time + 1.0)
        for worker_id in team
    )
    assert not terminated


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
