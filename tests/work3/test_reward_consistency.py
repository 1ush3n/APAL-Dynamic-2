"""奖励计账与真实综合目标数学等价性白盒测试 (Task 3.3 / 里程碑 M2 核心验收)。

核心验收标准：
1. 真实综合目标独立计算器 (Task 3.1) 与单步奖励增量计价 (Task 3.2) 代数完全一致；
2. 无扰动基准轨迹综合代价 J_total == 0.0，单步奖励均为 0.0；
3. 后移改派增量扣费与终局账本绝对一致；
4. 人员替换增量扣费与终局账本绝对一致；
5. 节拍超期增量扣费与终局账本绝对一致；
6. 随机动作长轨迹下，断言 |sum(r_n) - (-J_total)| < 10^-6；
7. 达成【里程碑 M2: 奖励计账与目标函数闭环】验收标准。
"""

from __future__ import annotations

import itertools
from pathlib import Path
import random
import pytest

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline
from utils.work3.objective_evaluator import (
    ObjectiveBreakdown,
    ObjectiveWeights,
    calculate_postpone_penalty,
    evaluate_trajectory_objective,
)


@pytest.fixture
def baseline_path() -> str:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return str(path)


def test_baseline_zero_reward_consistency(baseline_path: str) -> None:
    """测试 1: 无扰动基准排程回放，验证总成本为 0 且每步奖励严格为 0。"""
    baseline = MultiAircraftBaseline.load_from_json(baseline_path)
    weights = ObjectiveWeights(w_h=1.0, w_t=0.20, w_w=0.05, w_p=1.0)
    env = AirLineEnvWork3(baseline_json_path=baseline_path, weights=weights)
    env.reset()

    # 运行前两周期
    while env.state.current_cycle <= 2:
        ready = env.get_ready_tasks()
        if not ready:
            break
        task = ready[0]
        orig_task = baseline.get_task(task.aircraft_id, task.task_id)
        obs, reward, terminated, truncated, info = env.step({
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": orig_task.team,
            "align": 1,
        })
        # 基准回放无位置偏差、无换人、无后移，单步奖励必须严格为 0
        assert abs(reward) < 1e-10, f"基准回放出现非零单步奖励: {reward}"

    breakdown = evaluate_trajectory_objective(env, weights=weights)
    assert abs(breakdown.j_total) < 1e-10
    assert abs(breakdown.j_takt) < 1e-10
    assert abs(breakdown.d_time) < 1e-10
    assert abs(breakdown.d_team) < 1e-10
    assert abs(breakdown.j_postpone) < 1e-10
    assert abs(sum(env.step_rewards) - (-breakdown.j_total)) < 1e-12


def test_start_time_cost_uses_cycle_relative_offset(baseline_path: str) -> None:
    """周期起点7、基准偏移3、实际开工10时，相对偏差和费用均为0。"""
    weights = ObjectiveWeights(
        w_h=0.0,
        w_t=1.0,
        w_w=0.0,
        w_p=0.0,
        normalize_by_n=False,
    )
    env = AirLineEnvWork3(baseline_json_path=baseline_path, weights=weights)
    env.reset()
    task = next(iter(env.state.tasks.values()))
    task.in_station_offset = 3.0
    task.assigned_team = list(task.base_team)
    env.state.last_transfer_time = 7.0

    env._on_task_started(task, start_time=10.0)

    assert task.cycle_start_time == pytest.approx(7.0)
    assert task.actual_start == pytest.approx(10.0)
    assert env.cost_time == pytest.approx(0.0)
    assert evaluate_trajectory_objective(env, weights=weights).d_time == pytest.approx(0.0)


@pytest.mark.parametrize("actual_start", [8.0, 12.0])
def test_start_time_cost_is_symmetric_around_baseline_offset(
    baseline_path: str,
    actual_start: float,
) -> None:
    """周期相对开工偏移1/5相对基准偏移3，费用都为2/H0。"""
    weights = ObjectiveWeights(
        w_h=0.0,
        w_t=1.0,
        w_w=0.0,
        w_p=0.0,
        normalize_by_n=False,
    )
    env = AirLineEnvWork3(baseline_json_path=baseline_path, weights=weights)
    env.reset()
    task = next(iter(env.state.tasks.values()))
    task.in_station_offset = 3.0
    task.assigned_team = list(task.base_team)
    env.state.last_transfer_time = 7.0

    env._on_task_started(task, start_time=actual_start)

    expected_cost = 2.0 / env.state.h0
    assert env.cost_time == pytest.approx(expected_cost)
    assert evaluate_trajectory_objective(env, weights=weights).d_time == pytest.approx(
        expected_cost
    )


def test_alignment_does_not_wait_past_earliest_resource_slot(
    baseline_path: str,
) -> None:
    """对齐目标10但团队忙至12时，预约应从12开始而非继续等待。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()
    task = env.get_ready_tasks()[0]
    task.in_station_offset = 3.0
    env.state.last_transfer_time = 7.0
    team = tuple(
        env.valid_team_completion_workers(task, [])[: task.demand]
    )
    assert len(team) == task.demand
    for worker_id in team:
        env.state.workers[worker_id].add_interval(
            start=0.0,
            end=12.0,
            task_key=f"j03-resource-block-{worker_id}",
        )

    _obs, _reward, _terminated, _truncated, info = env.step({
        "task_key": task.task_key,
        "branch": ActionBranch.STATION_EXECUTE,
        "team": team,
        "align": 1,
    })

    assert task.status == TaskStatus.RESERVED
    assert info["scheduled_start"] == pytest.approx(12.0)
    assert task.scheduled_start == pytest.approx(12.0)


def test_postpone_incremental_consistency(baseline_path: str) -> None:
    """测试 2: 工序后移改派的增量扣费与终局账本绝对一致。"""
    weights = ObjectiveWeights(w_h=1.0, w_t=0.20, w_w=0.05, w_p=1.0, lambda_1=0.15, lambda_2=0.30)
    env = AirLineEnvWork3(baseline_json_path=baseline_path, weights=weights)
    env.reset()

    ready = env.get_ready_tasks()
    assert len(ready) > 0

    # 选取第 1 道工序执行后移 (n=1)
    task1 = next(task for task in env.state.tasks.values() if env.validate_postpone(task) is None)
    task1.status = TaskStatus.READY
    obs, reward1, _, _, info1 = env.step({
        "task_key": task1.task_key,
        "branch": ActionBranch.POSTPONE,
    })

    expected_cost1 = weights.w_p * weights.lambda_1
    assert abs(reward1 - (-expected_cost1)) < 1e-10
    assert info1["cost_postpone_inc"] == expected_cost1

    # 验证独立账本与环境累计成本
    breakdown1 = evaluate_trajectory_objective(env, weights=weights)
    assert abs(breakdown1.j_postpone - expected_cost1) < 1e-10
    assert abs(sum(env.step_rewards) - (-breakdown1.j_total)) < 1e-12


def test_team_replacement_consistency(baseline_path: str) -> None:
    """测试 3: 选派非基准团队时，团队替换扣费与终局账本绝对一致。"""
    weights = ObjectiveWeights(w_h=1.0, w_t=0.20, w_w=0.50, w_p=1.0, normalize_by_n=True)
    env = AirLineEnvWork3(baseline_json_path=baseline_path, weights=weights)
    env.reset()

    ready = env.get_ready_tasks()
    task = ready[0]
    allowed_workers = env.valid_team_completion_workers(task, [])
    base_team = set(task.base_team)

    # 构造一个与基准完全不同的可用团队
    alt_workers = [w for w in allowed_workers if w not in base_team]
    if len(alt_workers) >= task.demand:
        alt_team = tuple(alt_workers[: task.demand])
    else:
        # 部分重叠
        alt_team = tuple(allowed_workers[: task.demand])

    obs, reward, _, _, info = env.step({
        "task_key": task.task_key,
        "branch": ActionBranch.STATION_EXECUTE,
        "team": alt_team,
        "align": 1,
    })

    # 验证独立账本计算
    breakdown = evaluate_trajectory_objective(env, weights=weights)
    assert abs(breakdown.d_team - env.cost_team) < 1e-10
    assert abs(sum(env.step_rewards) - (-breakdown.j_total)) < 1e-12


def test_delayed_takt_violation_consistency(baseline_path: str) -> None:
    """测试 4: 节拍超期时，转站时刻增量扣费与终局账本绝对一致。"""
    weights = ObjectiveWeights(w_h=2.0, w_t=0.20, w_w=0.05, w_p=1.0)
    env = AirLineEnvWork3(baseline_json_path=baseline_path, weights=weights)
    env.reset()

    # 人为将当前时间推进到超过 H0 的时刻，模拟缺料导致的全线等待
    delay_hours = 50.0
    h0 = env.state.h0

    # 正常调度第 1 周期的所有任务，但设置开工延后
    while env.state.current_cycle == 1:
        ready = env.get_ready_tasks()
        if not ready:
                if env.get_action_candidates():
                    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
                    continue
                if env.event_queue.is_empty():
                    break
                env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
                continue
        t = ready[0]
        # 人为推迟物料到达时间
        t.material_ready_time = max(t.material_ready_time, h0 + delay_hours)
        valid_workers = env.valid_team_completion_workers(t, [])
        assert len(valid_workers) >= t.demand
        env.step({
            "task_key": t.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": tuple(valid_workers[: t.demand]),
            "align": 0,
        })

    # 转站应已触发，实际周期持续时间应大于 H0
    assert len(env.state.transfer_history) == 1
    p1 = env.state.transfer_history[0]
    assert p1 > h0

    breakdown = evaluate_trajectory_objective(env, weights=weights)
    assert breakdown.j_takt > 0.0
    assert abs(env.cost_takt - (weights.w_h * breakdown.j_takt)) < 1e-10
    assert abs(sum(env.step_rewards) - (-breakdown.j_total)) < 1e-10


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_randomized_trajectory_mathematical_equivalence(baseline_path: str, seed: int) -> None:
    """测试 5 (里程碑 M2 核心验收): 随机/混合调度长轨迹下的代数恒等性断言。

    验收标准：
    |sum(r_n) - (-J_total)| < 10^-6
    所有分项（节拍超期、开工位置、团队替换、改派惩罚）与独立账本误差小于 10^-6。
    """
    random.seed(seed)
    weights = ObjectiveWeights(
        w_h=1.5,
        w_t=0.25,
        w_w=0.10,
        w_p=1.2,
        lambda_1=0.10,
        lambda_2=0.20,
        normalize_by_n=True,
    )
    env = AirLineEnvWork3(baseline_json_path=baseline_path, weights=weights)
    env.reset()

    # 运行 300 步混合决策（涵盖就地执行、换队、不同对齐与合法后移）
    max_steps = 300
    for step_i in range(max_steps):
        ready = env.get_ready_tasks()
        if not ready:
            if env._check_terminated():
                break
            if env.event_queue.is_empty():
                break
            env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
            ready = env.get_ready_tasks()
            if not ready:
                break

        # 随机挑选一道就绪工序
        task = random.choice(ready)

        # 若非末站工序，以 15% 概率执行后移
        if env.validate_postpone(task) is None and random.random() < 0.15:
            action = {
                "task_key": task.task_key,
                "branch": ActionBranch.POSTPONE,
            }
        else:
            # 留在当前站执行，随机选队
            station_workers = env.state.station_worker_bindings[task.current_station]
            # 从该站工人中随机挑选 demand 个
            valid_workers = env.valid_team_completion_workers(task, [])
            assert len(valid_workers) >= task.demand
            team = tuple(random.sample(valid_workers, task.demand))
            align = random.choice([0, 1])
            action = {
                "task_key": task.task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": team,
                "align": align,
            }

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

    # 结算终局综合目标
    breakdown = evaluate_trajectory_objective(env, weights=weights)
    sum_rewards = sum(env.step_rewards)
    expected_sum_rewards = -breakdown.j_total

    # ======================== 里程碑 M2 核心验收断言 ========================
    # 1. 验证累计奖励与终局总目标代数严格恒等 (误差 < 10^-6)
    reward_diff = abs(sum_rewards - expected_sum_rewards)
    assert reward_diff < 1e-6, (
        f"[Seed {seed}] 奖励账本不一致！"
        f"sum(r_n)={sum_rewards:.8f}, -J_total={expected_sum_rewards:.8f}, 差值={reward_diff:.2e}"
    )

    # 2. 验证各分项与独立账本精确一致
    assert abs(env.cost_takt - (weights.w_h * breakdown.j_takt)) < 1e-6
    assert abs(env.cost_time - (weights.w_t * breakdown.d_time)) < 1e-6
    assert abs(env.cost_team - (weights.w_w * breakdown.d_team)) < 1e-6
    assert abs(env.cost_postpone - (weights.w_p * breakdown.j_postpone)) < 1e-6

    print(
        f"\n[MILESTONE M2 SEED {seed} PASSED] 步数={len(env.step_rewards)}, "
        f"sum(r_n)={sum_rewards:.6f}, -J_total={expected_sum_rewards:.6f}, "
        f"差值={reward_diff:.2e} (< 1e-6 严格通过!)"
    )
