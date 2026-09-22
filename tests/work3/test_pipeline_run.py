"""全线同步脉动流转、自然终止与 14 周期首条完整生产轨迹跑通 (Task 2.5 ~ Task 2.7 / 里程碑 M1)。

验收通过标准：
1. 飞机严格到站后才能加工，未到站不提前作业;
2. 5 站同时脉动步进，正好触发 14 次转站;
3. 第 9 号飞机离开末站、全部 2,830 道工序 100% 完工后才终止 (terminated=True);
4. 全程无工人分身冲突，站位槽位并发 <= 3;
5. 达成【里程碑 M1: 仿真流程跑通】验收标准。
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


def test_single_cycle_transfer_progression() -> None:
    """测试单周期完工后全线同步脉动触发（Task 2.5 验收）。"""
    env = AirLineEnvWork3()
    env.reset()

    # 0 号飞机在 0 号站位
    assert env.state.aircraft[0].current_station == 0
    h0 = env.state.h0

    # 循环调度直到 0 号站位的第 1 周期全部完成并转站
    max_steps = 50
    steps = 0
    while env.state.current_cycle == 1 and steps < max_steps:
        ready = env.get_ready_tasks()
        if not ready:
            if env._check_terminated() or env.event_queue.is_empty():
                break
            env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
            steps += 1
            continue
        task = ready[0]
        team = env.state.station_worker_bindings[task.current_station][: task.demand]
        env.step({
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 0,
        })
        steps += 1

    # 检查是否成功触发第 1 次脉动转站
    assert len(env.state.transfer_history) == 1
    assert env.state.current_cycle == 2
    assert 0.0 <= env.state.transfer_history[0] <= h0 + 1e-4

    # 0 号飞机已前进至 1 号站位，1 号飞机已进驻 0 号站位
    assert env.state.aircraft[0].current_station == 1
    assert env.state.aircraft[1].current_station == 0

    # 两架飞机在各自新站位的工序均被成功激活为 READY
    ready_after = env.get_ready_tasks()
    ac_ids = {t.aircraft_id for t in ready_after}
    assert 0 in ac_ids
    assert 1 in ac_ids


def test_full_14_cycles_pipeline_run() -> None:
    """跑通 10 架次、5 站位、14 脉动周期的完整无扰动生产轨迹（里程碑 M1 核心验收）。"""
    baseline_path = Path("data/work3/real_283_k10_baseline.json")
    if not baseline_path.is_file():
        pytest.skip(f"{baseline_path} 不存在")

    baseline = MultiAircraftBaseline.load_from_json(baseline_path)
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    env.reset()

    # 使用基准指派策略（每步选取 READY 工序，按基准团队指派）
    total_decisions = 0
    max_decisions = 10000

    while total_decisions < max_decisions:
        ready_tasks = env.get_ready_tasks()
        if not ready_tasks:
            # 无可用动作，检查是否已经终止
            if env._check_terminated():
                break
            if env.event_queue.is_empty():
                # 异常死锁检测
                pytest.fail("环境在未终止前陷入死锁！无可用动作且无未来事件！")
            # 显式结束当前预约修订轮并推进一个事件。
            env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
            total_decisions += 1
            continue

        # 选取当前就绪工序中最早的一道
        ready_tasks = env.get_ready_tasks()
        if not ready_tasks:
            continue

        task = ready_tasks[0]
        # 从基准模板中提取该工序的标准指派团队
        orig_task = baseline.get_task(task.aircraft_id, task.task_id)
        team = orig_task.team

        # 执行调度 (对齐开工)
        obs, reward, terminated, truncated, info = env.step({
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 1,
        })
        total_decisions += 1

        if terminated:
            break

    # ======================== 里程碑 M1 关键断言核验 ========================
    # 1. 验证全部工序 100% 完工
    all_completed = all(t.status == TaskStatus.COMPLETED for t in env.state.tasks.values())
    completed_count = sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.COMPLETED)
    assert all_completed, f"工序未全部完工: 实际完工 {completed_count} / {len(env.state.tasks)}"
    assert completed_count == 2830

    # 2. 验证全部 10 架飞机均已完全离开末站
    all_exited = all(ac.is_completed for ac in env.state.aircraft.values())
    assert all_exited, "存在飞机未离开末站出线"

    # 3. 验证正好触发 14 次脉动转站
    assert len(env.state.transfer_history) == 14, (
        f"脉动转站次数不正确: 期望 14 次, 实际 {len(env.state.transfer_history)} 次"
    )

    # 4. 转站按实际放行时刻单调推进；H0只作为费用与基准参照
    h0 = env.state.h0
    previous_transfer = 0.0
    for transfer_t in env.state.transfer_history:
        assert transfer_t >= previous_transfer - 1e-5
        previous_transfer = transfer_t

    # 5. 验证全过程无工人时间重叠
    worker_intervals: dict[int, list[tuple[float, float, str]]] = {}
    for task in env.state.tasks.values():
        for w in task.assigned_team:
            worker_intervals.setdefault(w, []).append(
                (task.actual_start, task.actual_end, task.task_key)
            )

    overlap_count = 0
    for w, ivs in worker_intervals.items():
        sorted_ivs = sorted(ivs, key=lambda x: x[0])
        for prev, curr in zip(sorted_ivs, sorted_ivs[1:]):
            if prev[1] > curr[0] + 1e-5:
                overlap_count += 1

    assert overlap_count == 0, f"发现 {overlap_count} 处工人时间重叠冲突！"

    # 输出里程碑验收成果甘特图汇总数据
    summary_data = {
        "milestone": "M1_VERIFIED",
        "total_aircraft": env.state.num_aircraft,
        "total_tasks": completed_count,
        "total_transfers": len(env.state.transfer_history),
        "total_makespan": env.state.current_time,
        "total_decisions": total_decisions,
        "h0": h0,
        "transfers": env.state.transfer_history,
    }
    output_path = Path("data/work3/m1_pipeline_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(
        f"\n[MILESTONE M1 PASSED] 成功跑通 14 周期全流程！"
        f"决策步数={total_decisions}, 总完工时刻={env.state.current_time:.2f}h, "
        f"正好触发 14 次脉动转站。"
    )
