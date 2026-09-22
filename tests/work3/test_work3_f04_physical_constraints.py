"""F04：技能、团队工时、站位约束和后移合法性反例。"""

import csv
from pathlib import Path

import pytest

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3


BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")
WORKER_POOL_PATH = Path("data/worker_pool_fixed.csv")


def load_worker_profiles() -> dict[int, dict[str, float]]:
    with WORKER_POOL_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            int(row["worker_id"]): {key: float(value) for key, value in row.items() if key != "worker_id"}
            for row in csv.DictReader(handle)
        }


@pytest.fixture
def env() -> AirLineEnvWork3:
    if not BASELINE_PATH.is_file() or not WORKER_POOL_PATH.is_file():
        pytest.skip("工作三F04所需实例或工人池不存在")
    instance = AirLineEnvWork3(
        baseline_json_path=BASELINE_PATH,
        worker_pool_path=WORKER_POOL_PATH,
    )
    instance.reset()
    return instance


def test_incompatible_skill_team_is_rejected(env: AirLineEnvWork3) -> None:
    """站位绑定不能替代工序技能校验。"""
    profiles = load_worker_profiles()
    task = next(
        task
        for task in env.state.tasks.values()
        if task.current_station == 1 and task.skill == 4
    )
    bad_worker = next(
        worker_id
        for worker_id in env.state.station_worker_bindings[task.current_station]
        if profiles[worker_id][f"skill_{task.skill}"] < 0.5
    )
    team = [bad_worker] + [
        worker_id
        for worker_id in env.state.station_worker_bindings[task.current_station]
        if worker_id != bad_worker
    ][: task.demand - 1]

    with pytest.raises(ValueError, match="技能"):
        env._validate_team_for_task(task, tuple(team))


def test_duration_for_team_uses_efficiency_and_team_synergy(env: AirLineEnvWork3) -> None:
    """同一工序的不同合法团队应按原规则得到不同工时。"""
    profiles = load_worker_profiles()
    task = next(
        task
        for task in env.state.tasks.values()
        if task.current_station == 4 and task.skill >= 0 and task.demand == 1
    )
    eligible = [
        worker_id
        for worker_id in env.state.station_worker_bindings[task.current_station]
        if profiles[worker_id][f"skill_{task.skill}"] >= 0.5
    ]
    team_a, team_b = eligible[:1], eligible[-1:]
    assert profiles[team_a[0]]["efficiency"] != profiles[team_b[0]]["efficiency"]

    duration_a = env.duration_for_team(task, team_a)
    duration_b = env.duration_for_team(task, team_b)

    assert duration_a != pytest.approx(duration_b)
    assert duration_a == pytest.approx(
        task.duration * task.demand / profiles[team_a[0]]["efficiency"]
    )


def test_fixed_station_task_cannot_be_postponed(env: AirLineEnvWork3) -> None:
    """原始固定站位任务不能通过后移动作离开固定站位。"""
    task = env.state.tasks["0_2"]
    task.status = TaskStatus.READY
    env.state.aircraft[task.aircraft_id].current_station = task.current_station

    with pytest.raises(ValueError, match="固定站位"):
        env.step({"task_key": task.task_key, "branch": ActionBranch.POSTPONE})


def test_postponing_parent_with_same_station_successor_is_rejected(
    env: AirLineEnvWork3,
) -> None:
    """不能后移父工序并把仍依赖它的本站后继留在原站。"""
    parent = next(
        task
        for task in env.state.tasks.values()
        if any(
            env.state.tasks.get(f"{task.aircraft_id}_{successor_id}") is not None
            and env.state.tasks[f"{task.aircraft_id}_{successor_id}"].current_station
            == task.current_station
            for successor_id in env._successors_map[task.aircraft_id][task.task_id]
        )
    )
    parent.status = TaskStatus.READY
    env.state.aircraft[parent.aircraft_id].current_station = parent.current_station

    with pytest.raises(ValueError, match="后继"):
        env.step({"task_key": parent.task_key, "branch": ActionBranch.POSTPONE})


def test_worker_selection_exposes_only_skill_complete_candidates(env: AirLineEnvWork3) -> None:
    """逐步选人时，候选集合必须仍能补全合法团队。"""
    task = next(
        task
        for task in env.state.tasks.values()
        if task.current_station == 1 and task.skill == 4 and task.demand >= 2
    )
    candidates = env.valid_team_completion_workers(task, [])
    assert len(candidates) >= task.demand
    assert all(
        task.skill in env.worker_skills[worker_id]
        for worker_id in candidates
    )

    selected = [candidates[0]]
    remaining = env.valid_team_completion_workers(task, selected)
    assert len(remaining) >= task.demand - len(selected)


def test_missing_worker_pool_fails_explicitly() -> None:
    """缺失技能/效率源文件时不能静默退化为固定工时。"""
    if not BASELINE_PATH.is_file():
        pytest.skip(f"{BASELINE_PATH} 不存在")

    with pytest.raises(FileNotFoundError, match="工人池"):
        AirLineEnvWork3(
            baseline_json_path=BASELINE_PATH,
            worker_pool_path=Path("data/work3/not_found_worker_pool.csv"),
        )
