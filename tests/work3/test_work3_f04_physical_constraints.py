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


@pytest.mark.parametrize("invalid_team_kind", ["out_of_station", "duplicate", "understaffed"])
def test_environment_rejects_invalid_team_membership_and_size(
    env: AirLineEnvWork3, invalid_team_kind: str
) -> None:
    """绕过策略直接提交的团队仍须满足站位、去重和固定需求人数。"""
    task = next(task for task in env.get_ready_tasks() if task.demand >= 2)
    valid_team = env.valid_team_completion_workers(task, [])[: task.demand]
    assert len(valid_team) == task.demand

    if invalid_team_kind == "out_of_station":
        bound = set(env.state.station_worker_bindings[task.current_station])
        outsider = next(worker_id for worker_id in env.worker_efficiencies if worker_id not in bound)
        submitted_team = [*valid_team[:-1], outsider]
    elif invalid_team_kind == "duplicate":
        submitted_team = [valid_team[0]] * task.demand
    else:
        submitted_team = valid_team[:-1]

    with pytest.raises(ValueError):
        env.step(
            {
                "task_key": task.task_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": submitted_team,
                "align": 0,
            }
        )


def test_postpone_remains_available_when_only_target_station_has_a_legal_team(
    env: AirLineEnvWork3,
) -> None:
    """本站无合格团队时应屏蔽留站，但目标站有团队仍允许后移。"""
    task = env.state.tasks["0_18"]
    env.state.aircraft[task.aircraft_id].current_station = task.current_station
    task.status = TaskStatus.READY

    current_workers = env.state.station_worker_bindings[task.current_station]
    target_workers = env.state.station_worker_bindings[task.current_station + 1]
    assert sum(task.skill in env.worker_skills[worker_id] for worker_id in target_workers) >= task.demand
    for worker_id in current_workers:
        env.worker_skills[worker_id] = frozenset(
            skill for skill in env.worker_skills[worker_id] if skill != task.skill
        )

    assert env.valid_team_completion_workers(task, []) == []
    assert not env.can_reserve(task)
    assert env.get_action_branch_mask(task) == (False, True)


def test_postpone_is_blocked_when_next_station_cannot_staff_the_task(
    env: AirLineEnvWork3,
) -> None:
    """目标站人数充足但缺少所需技能时，后移不能制造不可执行任务。"""
    task = env.state.tasks["0_27"]
    env.state.aircraft[task.aircraft_id].current_station = task.current_station
    task.status = TaskStatus.READY
    for predecessor_id in task.predecessors:
        predecessor = env.state.tasks[f"{task.aircraft_id}_{predecessor_id}"]
        predecessor.status = TaskStatus.COMPLETED
        predecessor.actual_end = 0.0

    current_workers = env.state.station_worker_bindings[task.current_station]
    target_workers = env.state.station_worker_bindings[task.current_station + 1]
    assert len(current_workers) >= task.demand
    assert sum(task.skill in env.worker_skills[worker_id] for worker_id in target_workers) < task.demand
    assert env.can_reserve(task)

    assert env.get_action_branch_mask(task) == (True, False)
    with pytest.raises(ValueError):
        env.step({"task_key": task.task_key, "branch": ActionBranch.POSTPONE})


def test_postpone_is_allowed_when_successor_is_already_at_target_station(
    env: AirLineEnvWork3,
) -> None:
    """后继已在下一站时，不能把所有存在后继的工序一概屏蔽。"""
    parent = env.state.tasks["0_10"]
    successor = env.state.tasks["0_11"]
    assert parent.task_id in env.constraint_engine.physical_predecessors[successor.task_id]
    assert successor.current_station == parent.current_station + 1
    env.state.aircraft[parent.aircraft_id].current_station = parent.current_station
    parent.status = TaskStatus.READY

    assert env.can_postpone(parent)
    env.step({"task_key": parent.task_key, "branch": ActionBranch.POSTPONE})
    assert parent.status == TaskStatus.POSTPONED
    assert parent.current_station == successor.current_station


def test_postpone_is_blocked_at_last_station(env: AirLineEnvWork3) -> None:
    """末站工序不能通过后移绕过批次完成要求。"""
    task = next(task for task in env.state.tasks.values() if task.current_station == 4)
    env.state.aircraft[task.aircraft_id].current_station = 4
    task.status = TaskStatus.READY

    assert not env.can_postpone(task)
    with pytest.raises(ValueError):
        env.step({"task_key": task.task_key, "branch": ActionBranch.POSTPONE})


def test_postpone_respects_task_station_upper_bound(env: AirLineEnvWork3) -> None:
    """实例最晚站位约束也必须出现在后移动作资格中。"""
    task = env.state.tasks["0_18"]
    env.state.aircraft[task.aircraft_id].current_station = task.current_station
    task.status = TaskStatus.READY
    task.max_allowed_station = task.current_station

    assert not env.can_postpone(task)
    with pytest.raises(ValueError):
        env.step({"task_key": task.task_key, "branch": ActionBranch.POSTPONE})


def test_worker_identity_does_not_change_duration_or_team_eligibility(
    env: AirLineEnvWork3,
) -> None:
    """仅替换工人编号、保留属性时，技能可行性与工时必须不变。"""
    task = next(
        task
        for task in env.state.tasks.values()
        if task.current_station == 4 and task.skill >= 0 and task.demand == 1
    )
    worker_id = next(
        worker_id
        for worker_id in env.state.station_worker_bindings[task.current_station]
        if task.skill in env.worker_skills[worker_id]
    )
    replacement_id = max(env.worker_efficiencies) + 1000
    env.worker_efficiencies[replacement_id] = env.worker_efficiencies[worker_id]
    env.worker_skills[replacement_id] = env.worker_skills[worker_id]
    replacement_calendar = env.state.workers[worker_id].copy()
    replacement_calendar.worker_id = replacement_id
    env.state.workers[replacement_id] = replacement_calendar
    env.state.station_worker_bindings[task.current_station].append(replacement_id)

    original_duration = env.duration_for_team(task, [worker_id])
    replacement_duration = env.duration_for_team(task, [replacement_id])

    assert replacement_id in env.valid_team_completion_workers(task, [])
    assert replacement_duration == pytest.approx(original_duration)


@pytest.mark.parametrize("corruption", ["missing_skill_column", "invalid_efficiency", "invalid_skill"])
def test_malformed_worker_pool_fails_with_data_validation_error(
    tmp_path: Path, corruption: str
) -> None:
    """必需技能/效率数据缺失或格式损坏时，不能静默继续构造环境。"""
    if not BASELINE_PATH.is_file() or not WORKER_POOL_PATH.is_file():
        pytest.skip("工作三F04所需实例或工人池不存在")

    with WORKER_POOL_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if corruption == "missing_skill_column":
        fieldnames.remove("skill_0")
        for row in rows:
            row.pop("skill_0", None)
    elif corruption == "invalid_efficiency":
        rows[0]["efficiency"] = "0"
    else:
        rows[0]["skill_0"] = "not-a-number"

    malformed_path = tmp_path / "worker_pool_malformed.csv"
    with malformed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError):
        AirLineEnvWork3(
            baseline_json_path=BASELINE_PATH,
            worker_pool_path=malformed_path,
        )
