"""PostponeCompatibleRepairAgent: 后移兼容的确定性修复控制器。
当任务处于原基准站位时，行为与 BaselinePlanRepairAgent 完全严格一致；
当任务合法后移到下游站位时，在目标站位合法工人池中寻找满足技能要求且开工最早的团队，继续执行调度。
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

from envs.work3.core_types import ActionBranch, TaskRuntimeState, TaskStatus
from envs.work3.environment import AirLineEnvWork3, NoFeasibleSlotError


class PostponeCompatibleInfeasibleError(RuntimeError):
    """后移兼容控制器无法为当前任务找到合法站内执行方案。"""


class PostponeCompatibleRepairAgent:
    """支持工序后移的确定性计划修复策略。"""

    def select_action(self, env: AirLineEnvWork3) -> dict[str, Any] | None:
        if env._check_terminated():
            return None

        impossible = self._first_unstaffable_present_task(env)
        if impossible is not None:
            raise PostponeCompatibleInfeasibleError(
                f"当前站位无满足技能/人数的团队: {impossible.task_key} (站位 {impossible.current_station})"
            )

        candidates = [
            task
            for task in env.get_action_candidates()
            if task.status != TaskStatus.RESERVED and env.can_reserve(task)
        ]
        if not candidates:
            return {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}

        task = min(
            candidates,
            key=lambda item: (
                float(item.in_station_offset),
                int(item.current_station),
                int(item.aircraft_id),
                int(item.task_id),
            ),
        )
        team = self._select_team(env, task)
        return {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": team,
            "align": 1,
        }

    @staticmethod
    def _first_unstaffable_present_task(
        env: AirLineEnvWork3,
    ) -> TaskRuntimeState | None:
        for task in env.state.tasks.values():
            if task.status in (TaskStatus.RUNNING, TaskStatus.RESERVED, TaskStatus.COMPLETED):
                continue
            aircraft = env.state.aircraft[task.aircraft_id]
            if aircraft.current_station != task.current_station:
                continue
            # 在当前站位检查是否有足够合格工人
            if len(env.valid_team_completion_workers(task, [])) < task.demand:
                return task
        return None

    @classmethod
    def _select_team(
        cls,
        env: AirLineEnvWork3,
        task: TaskRuntimeState,
    ) -> tuple[int, ...]:
        baseline_team = tuple(int(worker_id) for worker_id in task.base_team)
        
        # 仅在任务未发生后移 (current_station == base_station) 时尝试原基准团队
        if task.current_station == task.base_station:
            try:
                env._validate_team_for_task(task, baseline_team)
                start = cls._earliest_start(env, task, baseline_team)
                if start is not None:
                    return baseline_team
            except ValueError:
                pass

        # 否则（后移到下游站位，或基准团队不可用）选择目标站位的最优合法替代团队
        return cls._select_replacement_team(env, task, baseline_team)

    @classmethod
    def _select_replacement_team(
        cls,
        env: AirLineEnvWork3,
        task: TaskRuntimeState,
        baseline_team: tuple[int, ...],
    ) -> tuple[int, ...]:
        valid_workers = tuple(
            sorted(env.valid_team_completion_workers(task, []))
        )
        ranked: list[tuple[int, float, tuple[int, ...]]] = []
        for team in combinations(valid_workers, task.demand):
            try:
                env._validate_team_for_task(task, team)
            except ValueError:
                continue
            start = cls._earliest_start(env, task, team)
            if start is None:
                continue
            replacements = len(set(baseline_team) - set(team))
            ranked.append((replacements, start, tuple(team)))

        if not ranked:
            raise PostponeCompatibleInfeasibleError(
                f"无合法替代团队/预约区间: {task.task_key} 在站位 {task.current_station}"
            )
        # 排序策略：优先最早开工时间，其次最少替换人数，最后团队编号字典序
        ranked.sort(key=lambda x: (x[1], x[0], x[2]))
        return ranked[0][2]

    @staticmethod
    def _earliest_start(
        env: AirLineEnvWork3,
        task: TaskRuntimeState,
        team: Sequence[int],
    ) -> float | None:
        predecessor_release = env._predecessor_release_time(task)
        if predecessor_release is None:
            return None
        search_start = max(
            float(env.state.current_time),
            float(task.material_ready_time),
            float(predecessor_release),
            float(env.state.last_transfer_time + task.in_station_offset),
        )
        duration = env.duration_for_team(
            task,
            team,
            start_time=search_start,
        )
        try:
            return env._find_team_earliest_slot(
                station_id=task.current_station,
                team=team,
                search_start=search_start,
                duration=duration,
            )
        except NoFeasibleSlotError:
            return None
