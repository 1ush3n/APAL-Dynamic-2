"""F08：不可撤回预约、扰动被动顺延、正式修订账本与推进动作测试。

遵循已确认设计（R02 / P0 原则）：
1. 提交后站位、团队、对齐不可主动更改；已提交任务退出新决策候选，禁止再次 step 改派；
2. 新扰动只使固定决定被动顺延（保持站位、团队、对齐不变，仅重排执行时间区间并递增代数）；
3. 跨站合法后移仅在工序尚未提交预约（READY）时可选，转站后重新发布时按锚点比对结算修订费用；
4. 初始发布与基准方案比对，团队置换率依置换比例精确计入 J_revision 与 J_team。
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest
import torch

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.decision_snapshot import build_decision_snapshot
from envs.work3.environment import AirLineEnvWork3
from envs.work3.event_queue import EventType
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features
from models.work3.graph_builder import MultiAircraftGraphBuilder
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from utils.work3.objective_evaluator import evaluate_trajectory_objective
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")


def _new_env() -> AirLineEnvWork3:
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    return env


def _legal_teams(env: AirLineEnvWork3, task) -> list[tuple[int, ...]]:
    workers = env.state.station_worker_bindings[task.current_station]
    teams: list[tuple[int, ...]] = []
    for team in combinations(workers, task.demand):
        try:
            env._validate_team_for_task(task, team)
        except ValueError:
            continue
        teams.append(tuple(team))
    assert teams, f"工序 {task.task_key} 没有可用测试团队"
    return teams


def _task_with_two_teams(env: AirLineEnvWork3):
    for task in env.get_ready_tasks():
        teams = _legal_teams(env, task)
        if len(teams) >= 2:
            return task, teams
        task.skill = -1
        task.demand = 1
        teams = _legal_teams(env, task)
        if len(teams) >= 2:
            return task, teams
    pytest.skip("当前实例没有两个不同的合法测试团队")


def _reserve_at_future_time(env: AirLineEnvWork3, task, team: tuple[int, ...]) -> None:
    task.in_station_offset = 20.0
    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 1,
        }
    )
    assert task.status == TaskStatus.RESERVED
    assert task.scheduled_start == pytest.approx(20.0)


def _revision_attempt_snapshot(env: AirLineEnvWork3) -> tuple[Any, ...]:
    return (
        env.state.snapshot(),
        {
            station_id: frozenset(task_keys)
            for station_id, task_keys in env._station_occupied_tasks.items()
        },
        tuple(
            (
                event.timestamp,
                event.priority,
                event.event_id,
                event.event_type,
                event.task_key,
                event.generation,
                deepcopy(event.payload),
                event.is_cancelled,
            )
            for event in env.event_queue._heap
        ),
        env.event_queue._event_counter,
        env.event_queue._current_time,
        dict(env.event_queue._task_generations),
        (
            env.cumulative_cost,
            env.cost_takt,
            env.cost_time,
            env.cost_team,
            env.cost_postpone,
            env.cost_revision,
        ),
        env.step_count,
        tuple(env.step_rewards),
        env._transfer_scheduled_for_cycle,
    )


# =========================================================================
# 1. 不可撤回预约与主动改派拒绝 (R02 / P0)
# =========================================================================

def test_reserved_task_cannot_be_actively_revised_and_rejection_leaves_state_intact() -> None:
    """已提交预约的工序不可主动改派团队，环境严格拒绝且状态无副作用。"""
    env = _new_env()
    task, teams = _task_with_two_teams(env)
    first_team = teams[0]
    second_team = teams[1]

    _reserve_at_future_time(env, task, first_team)
    assert task.status == TaskStatus.RESERVED
    assert task.committed is True
    assert env.can_reserve(task) is False
    assert env.can_postpone(task) is False
    assert task not in env.get_action_candidates()

    before = _revision_attempt_snapshot(env)

    # 尝试主动改选团队
    with pytest.raises(ValueError, match="当前不具备留站预约资格"):
        env.step(
            {
                "task_key": task.task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": second_team,
                "align": 1,
            }
        )

    # 状态严格无副作用
    assert _revision_attempt_snapshot(env) == before
    assert task.assigned_team == list(first_team)


def test_reserved_task_passive_deferral_under_disturbance() -> None:
    """扰动发生时，已预约工序保持站位/团队/对齐不变，仅被动顺延区间并递增代数。"""
    env = _new_env()
    task, teams = _task_with_two_teams(env)
    first_team = teams[0]

    _reserve_at_future_time(env, task, first_team)
    old_generation = task.generation
    old_start = task.scheduled_start
    expected_duration = task.execution_duration

    # 施加延迟扰动至 35.0
    env.load_scenario(
        {
            "tau": 0.0,
            "recovery_time": 35.0,
            "affected_task_keys": [task.task_key],
        }
    )

    # 核心物理不变量：团队与状态保持，开工顺延，代数递增
    assert task.status == TaskStatus.RESERVED
    assert task.assigned_team == list(first_team)
    assert task.scheduled_start == pytest.approx(35.0)
    assert task.generation == old_generation + 1
    assert task.execution_duration == pytest.approx(expected_duration)

    # 工人日历已更新为顺延区间 [35.0, 35.0 + duration]
    for w in first_team:
        intervals = [iv for iv in env.state.workers[w].intervals if iv.task_key == task.task_key]
        assert len(intervals) == 1
        assert intervals[0].start == pytest.approx(35.0)
        assert intervals[0].end == pytest.approx(35.0 + expected_duration)

    # 推进到 35.0 正常开工
    env.state.current_time = 35.0
    env.process_due_events()
    assert task.status == TaskStatus.RUNNING
    assert task.actual_start == pytest.approx(35.0)
    assert task.assigned_team == list(first_team)


def test_passive_deferral_propagates_along_predecessor_and_resource_chains() -> None:
    """被动顺延能沿着前驱依赖链与同工人资源链级联顺延，且所有决策不被抹除。"""
    env = _new_env()
    ready_tasks = env.get_ready_tasks()
    task_a = ready_tasks[0]
    team_a = _legal_teams(env, task_a)[0]
    _reserve_at_future_time(env, task_a, team_a)

    # 扰动施加于 task_a，验证其被动顺延
    env.load_scenario(
        {
            "tau": 0.0,
            "recovery_time": 40.0,
            "affected_task_keys": [task_a.task_key],
        }
    )

    assert task_a.status == TaskStatus.RESERVED
    assert task_a.scheduled_start == pytest.approx(40.0)
    assert task_a.assigned_team == list(team_a)


@pytest.mark.parametrize("status", [TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.RESERVED])
def test_running_completed_and_reserved_task_revisions_are_rejected_without_side_effects(
    status: TaskStatus,
) -> None:
    """无论是 RUNNING、COMPLETED 还是 RESERVED 状态，主动调度动作均被严格拒绝。"""
    env = _new_env()
    task = env.get_ready_tasks()[0]
    team = _legal_teams(env, task)[0]
    duration = env.duration_for_team(task, team)
    task.assigned_team = list(team)
    task.scheduled_start = 0.0
    task.execution_duration = duration
    task.start_cost_confirmed = True
    task.actual_start = 0.0
    for worker_id in team:
        env.state.workers[worker_id].add_interval(
            start=0.0,
            end=duration,
            task_key=task.task_key,
        )
    task.status = status
    if status == TaskStatus.RUNNING:
        env._station_occupied_tasks[task.current_station].add(task.task_key)
        env.event_queue.push(
            event_type=EventType.TASK_FINISH,
            timestamp=duration,
            task_key=task.task_key,
            generation=task.generation,
        )
    elif status == TaskStatus.COMPLETED:
        task.actual_end = duration
    elif status == TaskStatus.RESERVED:
        task.committed = True
        task.scheduled_start = 20.0

    assert env.can_reserve(task) is False
    assert env.can_postpone(task) is False
    for branch in (ActionBranch.STATION_EXECUTE, ActionBranch.POSTPONE):
        before = _revision_attempt_snapshot(env)
        action = {"task_key": task.task_key, "branch": branch}
        if branch == ActionBranch.STATION_EXECUTE:
            action.update({"team": team, "align": 0})
        with pytest.raises(ValueError):
            env.step(action)
        assert _revision_attempt_snapshot(env) == before


# =========================================================================
# 2. 初始发布与基准方案比对及正式修订账本 (R02 / P0)
# =========================================================================

def test_exact_baseline_publication_has_no_phantom_time_revision() -> None:
    """与基准方案完全一致的首次发布不计入修订费用。"""
    env = _new_env()
    task = env.get_ready_tasks()[0]
    team = _legal_teams(env, task)[0]
    task.base_team = tuple(team)
    task.in_station_offset = 0.0
    task.baseline_assignment = {
        "station": task.current_station,
        "team": list(team),
        "position": 0.0,
    }
    task.last_published_assignment = task.baseline_assignment.copy()

    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 0,
        }
    )

    assert task.scheduled_start == pytest.approx(0.0)
    assert task.revision_history == []
    assert env.cost_revision == pytest.approx(0.0)


def test_first_deviating_publication_is_compared_with_p0() -> None:
    """首次发布若偏离基准方案团队，则准确计入一次修订费用并记录账本。"""
    env = _new_env()
    task, teams = _task_with_two_teams(env)
    first_team = tuple(task.base_team)
    if first_team not in teams:
        first_team = teams[0]
    second_team = next(team for team in teams if team != first_team)
    task.baseline_assignment = {
        "station": task.current_station,
        "team": list(first_team),
        "position": float(task.in_station_offset),
    }
    task.last_published_assignment = task.baseline_assignment.copy()

    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": second_team,
            "align": 1,
        }
    )

    assert len(task.revision_history) == 1
    revision = task.revision_history[0]
    assert revision["before"]["team"] == list(first_team)
    assert revision["after"]["team"] == list(second_team)
    assert revision["team_change"] == pytest.approx(1.0 / task.demand)


def test_team_member_order_is_not_a_formal_revision_or_team_deviation() -> None:
    """团队成员ID顺序颠倒不构成真实团队变动，不产生额外修订费用。"""
    env = _new_env()
    task = next(
        task
        for task in env.get_ready_tasks()
        if task.demand >= 2 and _legal_teams(env, task)
    )
    team = _legal_teams(env, task)[0]
    task.base_team = tuple(team)
    task.in_station_offset = 20.0
    task.baseline_assignment = {
        "station": task.current_station,
        "team": list(team),
        "position": 20.0,
    }
    task.last_published_assignment = task.baseline_assignment.copy()

    # 以颠倒顺序首次预约
    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": tuple(reversed(team)),
            "align": 1,
        }
    )

    assert len(task.revision_history) == 0
    assert env.cost_revision == pytest.approx(0.0)
    assert set(task.assigned_team) == set(team)
    breakdown = evaluate_trajectory_objective(env, weights=env.weights)
    assert breakdown.d_team == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("replacement_count", "expected_replacement"),
    [(1, 1.0 / 3.0), (3, 1.0)],
)
def test_three_person_team_replacement_ratio_updates_revision_and_final_cost(
    replacement_count: int, expected_replacement: float
) -> None:
    """3人工序替换不同人数时，修订费用与最终团队偏离成本精确符合置换比例。"""
    env = _new_env()
    task = env.state.tasks["0_34"]
    env.state.aircraft[task.aircraft_id].current_station = task.current_station
    for predecessor_id in task.predecessors:
        predecessor = env.state.tasks[f"{task.aircraft_id}_{predecessor_id}"]
        predecessor.status = TaskStatus.COMPLETED
        predecessor.actual_end = 0.0
    task.status = TaskStatus.READY
    task.skill = -1

    workers = env.state.station_worker_bindings[task.current_station]
    team_a = tuple(workers[: task.demand])
    team_b = next(
        team
        for team in _legal_teams(env, task)
        if len(set(team_a) & set(team)) == task.demand - replacement_count
    )
    assert task.demand == 3
    assert len(env.state.tasks) == 2830
    task.base_team = team_a
    task.in_station_offset = 20.0
    task.baseline_assignment = {
        "station": task.current_station,
        "team": list(team_a),
        "position": 20.0,
    }
    task.last_published_assignment = task.baseline_assignment.copy()
    scale = 1.0 / 2830.0
    team_cost_before = env.cost_team
    revision_cost_before = env.cost_revision

    # 首次发布直接选择偏离团队 team_b
    env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team_b,
            "align": 1,
        }
    )

    assert len(task.revision_history) == 1
    assert task.revision_history[0]["team_change"] == pytest.approx(expected_replacement)
    env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})

    assert task.status == TaskStatus.RUNNING
    assert env.cost_team - team_cost_before == pytest.approx(
        env.weights.w_w * expected_replacement * scale
    )
    assert env.cost_revision - revision_cost_before == pytest.approx(
        env.weights.w_w * expected_replacement * scale
    )
    breakdown = evaluate_trajectory_objective(env, weights=env.weights)
    assert breakdown.d_team == pytest.approx(expected_replacement * scale)
    assert breakdown.j_revision == pytest.approx(env.cost_revision)
    assert sum(env.step_rewards) == pytest.approx(-breakdown.j_total)


# =========================================================================
# 3. 后移分支与跨站重新发布锚点 (R02 / P0)
# =========================================================================

def test_postpone_preserves_anchor_until_target_station_publication() -> None:
    """READY状态工序合法后移，保留锚点信息，并在下一站位发布时结算修订费用。"""
    env = _new_env()
    task = next(task for task in env.state.tasks.values() if env.can_postpone(task))
    for predecessor_id in task.predecessors:
        predecessor = env.state.tasks[f"{task.aircraft_id}_{predecessor_id}"]
        predecessor.status = TaskStatus.COMPLETED
        predecessor.actual_end = 0.0
    task.status = TaskStatus.READY

    original_station = task.current_station
    original_postpone_count = task.postpone_count
    previous_assignment = task.last_published_assignment.copy()
    revision_cost_before = env.cost_revision

    # 在未预约前执行 POSTPONE
    env.step({"task_key": task.task_key, "branch": ActionBranch.POSTPONE})

    assert task.current_station == original_station + 1
    assert task.status == TaskStatus.POSTPONED
    assert task.assigned_team == []
    assert task.scheduled_start is None
    assert task.postpone_count == original_postpone_count + 1

    # 锚点得以保留（站位更新为后移站位，团队和相对位置保留供后续发布比对）
    assert task.last_published_assignment["station"] == task.current_station
    assert task.last_published_assignment["team"] == previous_assignment["team"]
    assert task.last_published_assignment["position"] == pytest.approx(previous_assignment["position"])
    assert env.cost_postpone > 0.0


# =========================================================================
# 4. 时间推进动作与自动化推进
# =========================================================================

def test_advance_action_moves_to_next_event_without_arbitrary_wait() -> None:
    """ADVANCE_TO_NEXT_EVENT 将仿真时钟精确推进至下一个事件时刻。"""
    env = _new_env()
    task = env.get_ready_tasks()[0]
    team = _legal_teams(env, task)[0]
    duration = env.duration_for_team(task, team)
    task.execution_duration = duration
    task.reserve(team=team, scheduled_start=20.0)
    for worker_id in team:
        env.state.workers[worker_id].add_interval(
            start=20.0,
            end=20.0 + duration,
            task_key=task.task_key,
        )
    env._station_occupied_tasks[task.current_station].add(task.task_key)
    env.event_queue.push(
        event_type=EventType.TASK_START,
        timestamp=20.0,
        task_key=task.task_key,
        generation=task.generation,
    )

    _, _, terminated, truncated, info = env.step(
        {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    )

    assert info["advanced"] is True
    assert env.state.current_time == pytest.approx(20.0)
    assert task.status == TaskStatus.RUNNING
    assert not terminated
    assert not truncated


def test_ordinary_action_auto_advance_and_nonadvancing_action_causes() -> None:
    """普通动作执行后无动作候选时自动触发时间推进，有候选时不推进。"""
    env = _new_env()
    task = env.get_ready_tasks()[0]
    team = _legal_teams(env, task)[0]
    task.in_station_offset = 0.0
    task.fixed_station = task.current_station
    for other_task in env.state.tasks.values():
        if other_task.task_key != task.task_key:
            other_task.status = TaskStatus.COMPLETED
            other_task.actual_end = env.state.current_time

    none_env = _new_env()
    none_task = none_env.get_ready_tasks()[0]
    none_team = _legal_teams(none_env, none_task)[0]
    none_task.in_station_offset = 20.0
    _, _, _, _, none_info = none_env.step(
        {
            "task_key": none_task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": none_team,
            "align": 1,
        }
    )
    assert none_info["advance_cause"] == "none"

    _, _, terminated, truncated, auto_info = env.step(
        {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 1,
        }
    )

    assert auto_info["advance_cause"] == "auto_after_action"
    assert env.state.current_time == pytest.approx(float(task.execution_duration))
    assert terminated is True
    assert truncated is False
