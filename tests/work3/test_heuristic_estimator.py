"""Task 5.2 周期启发式完工时间估计器专项单元测试。

验证点：
1. 无扰动名义基准估计：初始状态下估计不早于当前时刻；
2. 延误工序最早完工下界增强：突发缺料 r_ki 触发下界 r_ki + d_ki 生效；
3. 后移工序解耦效应：严重滞后工序后移至下一站后，原站位完工下界迅速回落；
4. 时间单调性与物理合理性：全过程 P_q^h >= t；
5. 排空期 (Drain-out) 鲁棒性：空工位不会产生异常崩溃。
"""

from __future__ import annotations

from pathlib import Path
import pytest

from envs.work3.core_types import (
    ActionBranch,
    AircraftRuntimeState,
    MultiAircraftState,
    TaskRuntimeState,
    TaskStatus,
)
from envs.work3.environment import AirLineEnvWork3
from models.work3.heuristic_estimator import (
    compute_cycle_heuristic_cmax,
    compute_station_estimated_finish,
)


@pytest.fixture
def baseline_path() -> str:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return str(path)


def test_nominal_baseline_estimate(baseline_path: str) -> None:
    """测试初始无扰动状态下，周期估计值不早于当前时刻。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    h0 = env.state.h0
    est_p1 = compute_cycle_heuristic_cmax(env.state)

    assert est_p1 >= env.state.current_time - 1e-4


def test_delayed_task_finish_bound_enhancement(baseline_path: str) -> None:
    """测试物理增强下界：突发延误工序最早完工时刻 r_ki + d_ki 能敏感提升全线估计。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    h0 = env.state.h0
    ready = env.get_ready_tasks()
    assert len(ready) > 0
    task = ready[0]

    # 注入远超当前节拍的物料到达时间 R = H_0 + 50.0
    r_delayed = h0 + 50.0
    task.material_ready_time = r_delayed

    est_delayed = compute_cycle_heuristic_cmax(env.state)
    expected_min = r_delayed + task.duration

    # 断言估计值直接突破名义节拍，灵敏反映物料延误下界
    assert est_delayed >= expected_min - 1e-4, (
        f"估计值未反映延误下界: 期望至少 {expected_min:.2f}, 实际={est_delayed:.2f}"
    )


def test_postpone_decoupling_effect(baseline_path: str) -> None:
    """测试后移解耦效应：当严重延误工序被后移至下一站，原站位的工位下界回落。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    h0 = env.state.h0
    task = next(
        task for task in env.state.tasks.values()
        if task.current_station == 0 and env.validate_postpone(task) is None
    )
    task.status = TaskStatus.READY
    st0 = task.current_station

    # 注入大延误
    r_delayed = h0 + 50.0
    task.material_ready_time = r_delayed

    f0_before = compute_station_estimated_finish(env.state, st0, h0)
    assert f0_before >= r_delayed + task.duration - 1e-4

    # 执行分支 B 后移至下一站
    env.step({
        "task_key": task.task_key,
        "branch": ActionBranch.POSTPONE,
    })

    # 断言该任务已被移出站位 0 (进入站位 1)
    assert task.current_station == 1
    assert task.status == TaskStatus.POSTPONED

    # 检查站位 0 的完工估计已迅速回落 (不再受该工序 r_delayed 拖累)
    f0_after = compute_station_estimated_finish(env.state, st0, h0)
    assert f0_after < r_delayed, f"后移后站位 0 估计值应回落: 实际={f0_after:.2f}"


def test_estimator_monotonicity_and_drainout(baseline_path: str) -> None:
    """测试估计器在多决策步推进中的时间下界保真度与排空期鲁棒性。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    # 推进 20 步正常生产
    for _ in range(20):
        ready = env.get_ready_tasks()
        if not ready:
            break
        task = ready[0]
        valid_workers = env.valid_team_completion_workers(task, [])
        assert len(valid_workers) >= task.demand
        team = tuple(valid_workers[: task.demand])
        env.step({
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 1,
        })
        est = compute_cycle_heuristic_cmax(env.state)
        # 断言绝对不发生时间倒退；H0不构成估计硬下界
        assert est >= env.state.current_time - 1e-4


def test_running_task_workload_uses_only_remaining_processing_time() -> None:
    """团队实际工时10小时且已完成8小时的任务只剩2小时。"""
    running = TaskRuntimeState(
        aircraft_id=0,
        task_id=0,
        task_key="0_0",
        base_station=0,
        current_station=0,
        status=TaskStatus.RUNNING,
        duration=12.0,
        in_station_offset=0.0,
        demand=1,
        skill=0,
        ao_code="",
        predecessors=(),
        actual_start=0.0,
        execution_duration=10.0,
    )
    state = MultiAircraftState(
        num_aircraft=1,
        num_stations=1,
        h0=10.0,
        current_time=8.0,
        aircraft={0: AircraftRuntimeState(aircraft_id=0, current_station=0)},
        tasks={running.task_key: running},
        station_worker_bindings={0: [0, 1]},
    )

    # 手算：团队工时10、已加工8，完工下界为10；剩余2除以2人得9；最终取max为10。
    assert compute_station_estimated_finish(state, 0, state.h0) == 10.0
