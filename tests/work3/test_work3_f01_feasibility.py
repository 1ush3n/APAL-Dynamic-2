"""F01：站位容量全区间检查与联合资源搜索反例。"""

from pathlib import Path

import pytest

from envs.work3.environment import AirLineEnvWork3


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
