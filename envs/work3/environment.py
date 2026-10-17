"""工作三多架次飞机脉动装配协同调度核心仿真环境 (AirLineEnvWork3)。

严格遵循顶层设计与 Ponytail 准则（标准脉动控速转站，易错点全闭环）：
- 维护多架次全局状态与离散事件队列；
- 支持自回归动作分支 A（留在当前站：指派团队、二元对齐、立即开工或未来预约）；
- 支持自回归动作分支 B（合法后移至下一站：动作当场截断，保留恢复时间 R）；
- 支持全线同步脉动判定与推进 (Task 2.5)：各站放行后严格按脉动节拍 H0 步进；
- 支持生产自然终止 (Task 2.6)：全部 10 架飞机离线出站且 2,830 道工序完成才终止。
"""

from __future__ import annotations

import csv
import copy
import logging
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from core.constraints import ConstraintEngine, calculate_team_synergy_factor
from data_loader import load_data
from envs.work3.core_types import (
    ADVANCE_CAUSES,
    ActionBranch,
    AircraftRuntimeState,
    CycleExecutionRecord,
    MultiAircraftState,
    TaskRuntimeState,
    TaskStatus,
    initialize_multi_aircraft_state,
)
from envs.work3.decision_snapshot import (
    TeamCompletionContext,
    WorkerSnapshot,
    team_completion_worker_ids_from_profiles,
)
from envs.work3.event_queue import DiscreteEventQueue, EventType, SimulationEvent
from utils.work3.objective_evaluator import ObjectiveWeights, calculate_postpone_penalty

logger = logging.getLogger(__name__)


class NoFeasibleSlotError(RuntimeError):
    """在保护范围内找不到同时满足资源约束的预约时刻。"""


class AirLineEnvWork3:
    """多架次飞机脉动装配协同调度环境。"""

    def __init__(
        self,
        baseline_json_path: str | Path = "data/work3/real_283_k10_baseline.json",
        raw_data_path: str | Path = "data/283.csv",
        worker_pool_path: str | Path = "data/worker_pool_fixed.csv",
        max_slots_per_station: int = 3,
        tolerance: float = 1e-5,
        weights: ObjectiveWeights | None = None,
        max_steps_per_rollout: int | None = None,
    ) -> None:
        self.baseline_json_path = baseline_json_path
        self.raw_data_path = Path(raw_data_path)
        self.worker_pool_path = Path(worker_pool_path)
        self.max_slots_per_station = int(max_slots_per_station)
        self.tolerance = float(tolerance)
        self.weights: ObjectiveWeights = weights if weights is not None else ObjectiveWeights()
        self.max_steps_per_rollout: int | None = max_steps_per_rollout
        self.step_count: int = 0

        self.state: MultiAircraftState = initialize_multi_aircraft_state(self.baseline_json_path)
        self._raw_task_durations: np.ndarray = np.empty(0, dtype=float)
        self.worker_efficiencies: dict[int, float] = {}
        self.worker_skills: dict[int, frozenset[int]] = {}
        self.constraint_engine = self._load_domain_metadata()
        self._attach_task_domain_metadata()
        self.event_queue: DiscreteEventQueue = DiscreteEventQueue()
        self.disturbance_event_triggered: bool = False
        # 按场景记录事件处理时的目标命中与未命中原因，避免用最终时刻反推同刻顺序。
        self.disturbance_event_results: dict[str, dict[str, Any]] = {}
        self._transfer_scheduled_for_cycle: int = 0
        self.total_tasks: int = len(self.state.tasks)
        self._step_cycle_records: list[CycleExecutionRecord] = []
        self._station_occupied_tasks: dict[int, set[str]] = {
            s: set() for s in range(self.state.num_stations)
        }

        # 累计成本与单步奖励记账账本 (Task 3.2 / 里程碑 M2)
        self.cumulative_cost: float = 0.0
        self.cost_takt: float = 0.0
        self.cost_time: float = 0.0
        self.cost_team: float = 0.0
        self.cost_postpone: float = 0.0
        self.cost_revision: float = 0.0
        self.step_rewards: list[float] = []

        # 构建工序直接后继索引：aircraft_id -> task_id -> list of successor task_ids
        self._successors_map: dict[int, dict[int, list[int]]] = {
            k: {t.task_id: [] for t in self.state.tasks.values() if t.aircraft_id == k}
            for k in range(self.state.num_aircraft)
        }
        for task in self.state.tasks.values():
            for pred_id in task.predecessors:
                if pred_id in self._successors_map[task.aircraft_id]:
                    self._successors_map[task.aircraft_id][pred_id].append(task.task_id)

    def reset(self) -> dict[str, Any]:
        """重置仿真环境到初始生产状态（0号飞机进驻0号站位）。"""
        self.state = initialize_multi_aircraft_state(self.baseline_json_path)
        self._attach_task_domain_metadata()
        self.event_queue.reset(start_time=0.0)
        self.disturbance_event_triggered = False
        self.disturbance_event_results.clear()
        self._transfer_scheduled_for_cycle = 0
        self._step_cycle_records.clear()
        self.total_tasks = len(self.state.tasks)
        self._station_occupied_tasks = {s: set() for s in range(self.state.num_stations)}

        self.cumulative_cost = 0.0
        self.cost_takt = 0.0
        self.cost_time = 0.0
        self.cost_team = 0.0
        self.cost_postpone = 0.0
        self.cost_revision = 0.0
        self.step_rewards.clear()
        self.step_count = 0

        # 0 号飞机在时刻 0.0 进入 0 号站位
        ac0 = self.state.aircraft[0]
        ac0.current_station = 0
        ac0.entry_times[0] = 0.0

        # 解锁0号飞机在0号站位首批无前驱且开工可用条件满足的工序
        self._refresh_aircraft_station_readiness(aircraft_id=0, station_id=0)

        # 初始化首次就绪时刻事实记录与演化语义 (R05 & R08)
        self.state.station_first_cleared_times = {1: {}}
        self.state.last_action_branch = 0
        self.state.micro_steps_in_timestamp = 0
        self.state.recent_postponed_count = 0
        self.state.recent_reserved_count = 0
        self._last_step_time = 0.0
        self._record_station_first_cleared_if_ready()

        return self._get_observation()

    def _record_station_first_cleared_if_ready(self) -> None:
        """检查并记录当前周期各占位站位的首次就绪时刻 F[m, q] (R05)。"""
        cycle = self.state.current_cycle
        cleared_map = self.state.station_first_cleared_times.setdefault(cycle, {})
        for s in range(self.state.num_stations):
            if s in cleared_map:
                continue
            ac_id = self.state.get_aircraft_at_station(s)
            if ac_id is not None:
                if self.state.is_station_cleared_for_transfer(s):
                    # 空站无效；有飞机无未完工序仍有效且F等于当前时刻（周期初则为脉动时刻）
                    cleared_map[s] = float(self.state.current_time)

    @staticmethod
    def _find_domain_column(
        fieldnames: Sequence[str], aliases: Sequence[str], label: str
    ) -> str:
        for alias in aliases:
            if alias in fieldnames:
                return alias
        raise ValueError(f"原始工艺数据缺少{label}字段，已检查: {tuple(aliases)}")

    @staticmethod
    def _parse_optional_station(value: str, *, task_id: int) -> int:
        text = str(value).strip()
        if not text:
            return -1
        try:
            station_id = int(float(text)) - 1
        except ValueError as exc:
            raise ValueError(f"工序 {task_id} 的固定站位无法解析: {value!r}") from exc
        if station_id < 0:
            raise ValueError(f"工序 {task_id} 的固定站位必须为正数: {value!r}")
        return station_id

    def _load_domain_metadata(self) -> ConstraintEngine:
        """加载工作一、二沿用的技能、效率和工艺站位约束。"""
        if not self.raw_data_path.is_file():
            raise FileNotFoundError(f"原始工艺数据不存在: {self.raw_data_path}")
        if not self.worker_pool_path.is_file():
            raise FileNotFoundError(f"工人池技能/效率数据不存在: {self.worker_pool_path}")

        with self.worker_pool_path.open("r", encoding="utf-8-sig", newline="") as handle:
            worker_rows = list(csv.DictReader(handle))
        if not worker_rows:
            raise ValueError(f"工人池为空: {self.worker_pool_path}")

        required_worker_columns = {"worker_id", "efficiency", *(f"skill_{i}" for i in range(5))}
        missing_worker_columns = required_worker_columns - set(worker_rows[0])
        if missing_worker_columns:
            raise ValueError(f"工人池缺少字段: {sorted(missing_worker_columns)}")

        for row in worker_rows:
            worker_id = int(row["worker_id"])
            if worker_id in self.worker_efficiencies:
                raise ValueError(f"工人池存在重复工人ID: {worker_id}")
            efficiency = float(row["efficiency"])
            if not math.isfinite(efficiency) or efficiency <= 0.0:
                raise ValueError(f"工人 {worker_id} 的效率必须为正有限数: {efficiency}")
            self.worker_efficiencies[worker_id] = efficiency
            self.worker_skills[worker_id] = frozenset(
                skill_id
                for skill_id in range(5)
                if float(row[f"skill_{skill_id}"]) >= 0.5
            )

        bound_workers = {
            worker_id
            for worker_ids in self.state.station_worker_bindings.values()
            for worker_id in worker_ids
        }
        missing_bound_workers = sorted(bound_workers - self.worker_efficiencies.keys())
        if missing_bound_workers:
            raise ValueError(f"基准站位绑定的工人未出现在工人池: {missing_bound_workers}")

        data = load_data(self.raw_data_path)
        df_raw = data["task_df"]
        num_raw_tasks = data["num_tasks"]
        edge_array = data["precedence_edges"].numpy()

        durations = df_raw["duration"].values.astype(float)
        fixed_stations = np.full(num_raw_tasks, -1, dtype=np.int64)
        if "fixed_station" in df_raw.columns:
            for i, v in enumerate(df_raw["fixed_station"].values):
                if not pd.isna(v) and str(v).strip() != "":
                    fixed_stations[i] = int(float(v)) - 1

        required_skills: dict[int, int] = {}
        demands: dict[int, int] = {}
        for i in range(num_raw_tasks):
            required_skills[i] = int(df_raw.iloc[i]["skill_type"])
            demands[i] = int(df_raw.iloc[i]["demand_workers"])

        node_types = (
            df_raw["node_type"].values.astype(int)
            if "node_type" in df_raw.columns
            else (df_raw["类型"].values.astype(int) if "类型" in df_raw.columns else None)
        )

        engine = ConstraintEngine.build(
            num_tasks=num_raw_tasks,
            num_stations=self.state.num_stations,
            edges=edge_array,
            durations=durations,
            fixed_stations=fixed_stations,
            node_types=node_types,
        )

        successors: list[list[int]] = [[] for _ in range(num_raw_tasks)]
        indegree = [len(preds) for preds in engine.predecessors]
        for task_id, predecessors in enumerate(engine.predecessors):
            for predecessor in predecessors:
                successors[predecessor].append(task_id)
        queue = [task_id for task_id, degree in enumerate(indegree) if degree == 0]
        topological_order: list[int] = []
        while queue:
            task_id = queue.pop()
            topological_order.append(task_id)
            for successor in successors[task_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    queue.append(successor)
        if len(topological_order) != num_raw_tasks:
            raise ValueError("原始工艺依赖图存在环，无法建立后移合法性边界")

        max_allowed = np.full(num_raw_tasks, self.state.num_stations - 1, dtype=np.int64)
        max_allowed[engine.fixed_stations >= 0] = engine.fixed_stations[engine.fixed_stations >= 0]
        for task_id in reversed(topological_order):
            for predecessor in engine.predecessors[task_id]:
                max_allowed[predecessor] = min(max_allowed[predecessor], max_allowed[task_id])

        self._raw_required_skills = required_skills
        self._raw_demands = demands
        self._raw_task_durations = durations.copy()
        return engine.with_max_allowed_stations(max_allowed)

    def _attach_task_domain_metadata(self) -> None:
        existing_task_ids_by_ac = {
            k: {t.task_id for t in self.state.tasks.values() if t.aircraft_id == k}
            for k in range(self.state.num_aircraft)
        }
        for task in self.state.tasks.values():
            task_id = int(task.task_id)
            if not 0 <= task_id < self.constraint_engine.num_tasks:
                raise ValueError(f"基准任务 {task.task_key} 不在原始工艺数据中")
            if not bool(self.constraint_engine.physical_mask[task_id]):
                raise ValueError(f"基准任务 {task.task_key} 对应原始虚拟工序")
            if int(task.skill) != int(self._raw_required_skills[task_id]):
                raise ValueError(f"任务 {task.task_key} 的技能与原始工艺数据不一致")
            if int(task.demand) != int(self._raw_demands[task_id]):
                raise ValueError(f"任务 {task.task_key} 的需求人数与原始工艺数据不一致")
            task.standard_duration = float(self._raw_task_durations[task_id])
            fixed_station = int(self.constraint_engine.fixed_stations[task_id])
            task.fixed_station = fixed_station if fixed_station >= 0 else None
            task.max_allowed_station = int(self.constraint_engine.max_allowed_stations[task_id])
            ac_task_ids = existing_task_ids_by_ac.get(task.aircraft_id, set())
            task.predecessors = tuple(
                p for p in self.constraint_engine.physical_predecessors[task_id]
                if p in ac_task_ids
            )

        self._successors_map = {
            k: {t.task_id: [] for t in self.state.tasks.values() if t.aircraft_id == k}
            for k in range(self.state.num_aircraft)
        }
        for task in self.state.tasks.values():
            for pred_id in task.predecessors:
                if pred_id in self._successors_map[task.aircraft_id]:
                    self._successors_map[task.aircraft_id][pred_id].append(task.task_id)

    def valid_team_completion_workers(
        self,
        task: TaskRuntimeState,
        selected_team: Sequence[int],
        *,
        station_id: int | None = None,
    ) -> list[int]:
        """返回指定站位上加入部分团队后仍可补全合法团队的工人。"""
        candidate_station = task.current_station if station_id is None else int(station_id)
        return list(
            team_completion_worker_ids_from_profiles(
                self.state.station_worker_bindings.get(candidate_station, ()),
                self.worker_skills,
                self.worker_efficiencies,
                required_skill=int(task.skill),
                demand=int(task.demand),
                selected_worker_ids=tuple(int(worker_id) for worker_id in selected_team),
            )
        )

    def get_team_completion_context(
        self,
        task: TaskRuntimeState,
        *,
        station_id: int | None = None,
    ) -> TeamCompletionContext:
        """导出独立于现场对象的只读团队技能、效率及日历快照。"""
        return self._team_completion_context(
            task,
            station_id=station_id,
        )

    def _team_completion_context(
        self,
        task: TaskRuntimeState,
        *,
        station_id: int | None,
    ) -> TeamCompletionContext:
        """构造策略快照；高频资格检查直接使用纯技能/效率字典规则。"""
        candidate_station = task.current_station if station_id is None else int(station_id)
        worker_ids = tuple(self.state.station_worker_bindings.get(candidate_station, ()))
        snapshots: list[WorkerSnapshot] = []
        for worker_id in worker_ids:
            calendar = self.state.workers.get(worker_id)
            intervals = () if calendar is None else tuple(
                (float(interval.start), float(interval.end), str(interval.task_key))
                for interval in calendar.intervals
            )
            snapshots.append(
                WorkerSnapshot(
                    worker_id=int(worker_id),
                    skills=tuple(sorted(self.worker_skills.get(worker_id, ()))),
                    efficiency=self.worker_efficiencies.get(worker_id),
                    calendar_intervals=intervals,
                )
            )
        return TeamCompletionContext(
            task_key=task.task_key,
            station_id=candidate_station,
            required_skill=int(task.skill),
            demand=int(task.demand),
            workers=tuple(snapshots),
        )

    def _predecessor_release_time(self, task: TaskRuntimeState) -> float | None:
        """返回前驱已完成、预约或运行时可证明的最早释放时刻。"""
        release_time = self.state.current_time
        for predecessor_id in task.predecessors:
            predecessor = self.state.tasks.get(f"{task.aircraft_id}_{predecessor_id}")
            if predecessor is None:
                return None
            if predecessor.status == TaskStatus.COMPLETED and predecessor.actual_end is not None:
                release_time = max(release_time, float(predecessor.actual_end))
                continue
            if predecessor.status == TaskStatus.RESERVED:
                planned_start = predecessor.scheduled_start
            elif predecessor.status == TaskStatus.RUNNING:
                planned_start = predecessor.actual_start
            else:
                return None
            if planned_start is not None and predecessor.execution_duration is not None:
                release_time = max(
                    release_time,
                    float(planned_start + predecessor.execution_duration),
                )
                continue
            return None
        return release_time

    def _passive_defer_reserved_task(
        self,
        task: TaskRuntimeState,
        cause: str = "disturbance",
    ) -> None:
        """为已预约工序被动顺延可行执行时间区间，严格保持已提交的站位、团队与对齐决策不变。"""
        if task.status != TaskStatus.RESERVED or not task.assigned_team:
            return

        # 1. 释放该任务当前在工人日历中占用的区间
        for w in task.assigned_team:
            self.state.workers[w].remove_interval(task.task_key)

        # 2. 计算满足当前时刻、物料恢复时刻、已揭示前驱及旧预约时刻的最早搜索起点（仅顺延不提前）
        predecessor_release = self._predecessor_release_time(task)
        old_scheduled_start = float(task.scheduled_start) if task.scheduled_start is not None else 0.0
        search_start = max(
            self.state.current_time,
            float(task.material_ready_time),
            0.0 if predecessor_release is None else float(predecessor_release),
            old_scheduled_start,
        )
        if getattr(task, "alignment", 0) == 1:
            align_target = self.state.last_transfer_time + task.in_station_offset
            search_start = max(search_start, float(align_target))

        # 3. 计算在 search_start 开工时相同团队的实际执行工时
        execution_duration = self.duration_for_team(
            task,
            task.assigned_team,
            start_time=search_start,
        )

        # 4. 在同一站位和同一团队的时间日历中寻找最早可行无冲突槽位
        new_start = self._find_team_earliest_slot(
            station_id=task.current_station,
            team=task.assigned_team,
            search_start=search_start,
            duration=execution_duration,
        )

        # 5. 更新任务的可行预约时刻与工时，递增代数使旧事件失效
        task.scheduled_start = float(new_start)
        task.execution_duration = float(execution_duration)
        task.generation += 1

        # 6. 在工人日历中登记新区间
        for w in task.assigned_team:
            self.state.workers[w].add_interval(
                start=new_start,
                end=new_start + execution_duration,
                task_key=task.task_key,
            )

        # 7. 更新事件队列：旧开工事件失效，挂载新时刻的 TASK_START 事件
        self.event_queue.invalidate_task_events(task.task_key, task.generation)
        self.event_queue.push(
            event_type=EventType.TASK_START,
            timestamp=new_start,
            task_key=task.task_key,
            generation=task.generation,
        )

        # 8. 顺延资源链冲突任务：检查同团队工人中与新区间发生实际半开区间重叠的其他预约
        new_end = new_start + execution_duration
        conflicted_keys = set()
        for w in task.assigned_team:
            for iv in list(self.state.workers[w].intervals):
                other_key = iv.task_key
                if other_key == task.task_key or other_key in conflicted_keys:
                    continue
                if max(new_start, iv.start) < min(new_end, iv.end) - self.tolerance:
                    other_task = self.state.tasks.get(other_key)
                    if other_task is not None and other_task.status == TaskStatus.RESERVED:
                        conflicted_keys.add(other_key)

        for other_key in sorted(conflicted_keys):
            other_task = self.state.tasks.get(other_key)
            if other_task is not None:
                self._passive_defer_reserved_task(other_task, cause="resource_conflict")

        # 9. 顺延前驱链后继任务：沿工艺后继检查已预约的后继工序
        self._propagate_passive_deferral(task)

    def _propagate_passive_deferral(self, task: TaskRuntimeState) -> None:
        """沿工艺后继传播被动顺延，保持后继工序已决定的团队与站位不变。"""
        if task.scheduled_start is None or task.execution_duration is None:
            return
        task_end = task.scheduled_start + task.execution_duration
        successors = self._successors_map.get(task.aircraft_id, {}).get(task.task_id, [])
        for succ_id in successors:
            succ_key = f"{task.aircraft_id}_{succ_id}"
            succ_task = self.state.tasks.get(succ_key)
            if succ_task is not None and succ_task.status == TaskStatus.RESERVED:
                if succ_task.scheduled_start is not None and succ_task.scheduled_start < task_end - self.tolerance:
                    self._passive_defer_reserved_task(succ_task, cause="predecessor_delay")

    def _invalidate_dependent_reservations(self, task: TaskRuntimeState) -> None:
        """保持不可撤回原则，后继预约不撤销团队，仅被动顺延。"""
        self._propagate_passive_deferral(task)

    def can_reserve(self, task: TaskRuntimeState) -> bool:
        """判断任务是否可以登记留站预约，不代表当前时刻可以实际开工。"""
        if task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.RESERVED):
            return False
        aircraft = self.state.aircraft[task.aircraft_id]
        if aircraft.current_station != task.current_station:
            return False
        if len(self.valid_team_completion_workers(task, [])) < task.demand:
            return False
        return self._predecessor_release_time(task) is not None

    def can_postpone(self, task: TaskRuntimeState) -> bool:
        """判断任务是否可以独立后移，不要求开工可用下界已在当前时刻满足。"""
        if task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.RESERVED):
            return False
        return self.validate_postpone(task) is None

    def get_action_branch_mask(self, task: TaskRuntimeState) -> tuple[bool, bool]:
        """返回任务的``(stay_mask, postpone_mask)``。"""
        return self.can_reserve(task), self.can_postpone(task)

    def get_action_candidates(self) -> list[TaskRuntimeState]:
        """返回至少有一个合法动作分支的任务。"""
        candidates: list[TaskRuntimeState] = []
        for station_id in range(self.state.num_stations):
            for task in self.state.get_tasks_for_station(station_id):
                if any(self.get_action_branch_mask(task)):
                    candidates.append(task)
        return candidates

    def duration_for_team(
        self,
        task: TaskRuntimeState,
        team: Sequence[int],
        start_time: float | None = None,
    ) -> float:
        """按工作一、二的效率求和与团队协同折减计算本次团队工时。"""
        del start_time  # 当前工作三域数据没有疲劳状态，保留参数以保持工时接口可扩展。
        team_tuple = tuple(int(worker_id) for worker_id in team)
        self._validate_team_for_task(task, team_tuple)
        if task.standard_duration is None:
            raise ValueError(f"任务 {task.task_key} 缺少原始实例标准工时")
        if task.standard_duration <= self.tolerance:
            return 0.0
        efficiency_sum = sum(self.worker_efficiencies[worker_id] for worker_id in team_tuple)
        effective_capacity = efficiency_sum * calculate_team_synergy_factor(len(team_tuple))
        if effective_capacity <= self.tolerance:
            raise ValueError(f"团队 {team_tuple} 的有效工时能力不足")
        return float(task.standard_duration * task.demand / effective_capacity)

    def _assignment_snapshot(
        self,
        task: TaskRuntimeState,
        *,
        station: int,
        team: Sequence[int] | None,
        scheduled_start: float | None,
    ) -> dict[str, Any]:
        """生成可写入正式安排账本的不可变值快照。"""
        position = None
        if scheduled_start is not None:
            position = float(scheduled_start - self.state.last_transfer_time)
        return {
            "station": int(station),
            "team": None if team is None else [int(worker_id) for worker_id in team],
            "scheduled_start": None if scheduled_start is None else float(scheduled_start),
            "position": position,
        }

    def _revision_cost_components(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        task: TaskRuntimeState,
    ) -> tuple[float, float, bool]:
        """计算相邻正式安排的时间位置、团队与站位变化。"""
        before_position = before.get("position")
        after_position = after.get("position")
        time_change = 0.0
        if before_position is not None and after_position is not None:
            time_change = abs(float(after_position) - float(before_position)) / self.state.h0

        before_team = before.get("team")
        after_team = after.get("team")
        team_change = 0.0
        if before_team is not None and after_team is not None and task.demand > 0:
            overlap = len(set(before_team) & set(after_team))
            team_change = 1.0 - overlap / task.demand

        station_changed = int(before.get("station")) != int(after.get("station"))
        return time_change, team_change, station_changed

    def _record_formal_revision(
        self,
        task: TaskRuntimeState,
        after: dict[str, Any],
        *,
        reason: str,
        additional_cost: float = 0.0,
    ) -> float:
        """提交一次正式修订并计入相邻安排的增量账本。"""
        # 费用比较锚点是上一次正式发布，不读取当前预约字段。
        before = dict(task.last_published_assignment or task.baseline_assignment)
        time_change, team_change, station_changed = self._revision_cost_components(
            before, after, task
        )
        changed = (
            station_changed
            or time_change > self.tolerance
            or team_change > self.tolerance
        )
        task.last_published_assignment = dict(after)
        if not changed:
            return 0.0

        scale = (1.0 / self.total_tasks) if (self.weights.normalize_by_n and self.total_tasks > 0) else 1.0
        revision_time_weight = (
            self.weights.w_t
            if self.weights.w_revision_time is None
            else self.weights.w_revision_time
        )
        revision_team_weight = (
            self.weights.w_w
            if self.weights.w_revision_team is None
            else self.weights.w_revision_team
        )
        revision_cost = scale * (
            revision_time_weight * time_change + revision_team_weight * team_change
        )
        total_increment = revision_cost + float(additional_cost)
        task.revision_history.append(
            {
                "reason": reason,
                "before": before,
                "after": dict(after),
                "time_change": time_change,
                "team_change": team_change,
                "station_changed": station_changed,
                "revision_cost": total_increment,
            }
        )
        self.cost_revision += revision_cost
        self.cumulative_cost += revision_cost
        return total_increment

    def _release_reserved_resources(self, task: TaskRuntimeState) -> None:
        """临时释放未开工预约，供事务式改派使用。"""
        for worker_id in task.assigned_team:
            self.state.workers[worker_id].remove_interval(task.task_key)
        self._station_occupied_tasks[task.current_station].discard(task.task_key)

    def _restore_reserved_resources(self, task: TaskRuntimeState) -> None:
        """恢复事务失败前的未开工预约资源。"""
        if task.scheduled_start is None or task.execution_duration is None:
            return
        for worker_id in task.assigned_team:
            self.state.workers[worker_id].add_interval(
                start=task.scheduled_start,
                end=task.scheduled_start + task.execution_duration,
                task_key=task.task_key,
            )
        self._station_occupied_tasks[task.current_station].add(task.task_key)

    def _advance_to_next_event(self) -> bool:
        """推进到单个下一个离散事件，不允许指定任意等待时长。"""
        self.process_due_events()
        if self._check_terminated():
            return False
        next_event = self.event_queue.peek()
        if next_event is None:
            return False
        self.state.current_time = max(self.state.current_time, float(next_event.timestamp))
        self.process_due_events()
        return True

    def validate_postpone(self, task: TaskRuntimeState) -> str | None:
        """统一检查当前任务后移到紧邻下一站的工艺合法性。"""
        if task.current_station >= self.state.num_stations - 1:
            return f"工序 {task.task_key} 位于末站，禁止后移"
        aircraft = self.state.aircraft[task.aircraft_id]
        if aircraft.current_station != task.current_station:
            return f"工序 {task.task_key} 的飞机实际不在当前站位，禁止后移"

        target_station = task.current_station + 1
        if len(
            self.valid_team_completion_workers(task, [], station_id=target_station)
        ) < task.demand:
            return f"工序 {task.task_key} 在目标站 {target_station} 无满足技能与人数要求的团队，禁止后移"

        if task.fixed_station is not None and target_station != task.fixed_station:
            return f"工序 {task.task_key} 受固定站位 {task.fixed_station + 1} 约束，禁止后移"
        if task.max_allowed_station is not None and target_station > task.max_allowed_station:
            return f"工序 {task.task_key} 超过允许最晚站位，禁止后移"

        station_map = {
            predecessor_id: self.state.tasks[f"{task.aircraft_id}_{predecessor_id}"].current_station
            for predecessor_id in self.constraint_engine.physical_predecessors[task.task_id]
            if f"{task.aircraft_id}_{predecessor_id}" in self.state.tasks
        }
        violation = self.constraint_engine.station_violation(
            task.task_id,
            target_station,
            station_map,
        )
        if violation is not None:
            if violation["reason"] == "fixed_station_violation":
                return f"工序 {task.task_key} 受固定站位约束，禁止后移"
            return f"工序 {task.task_key} 的目标站位不满足工艺约束，禁止后移"

        same_station_successors = [
            successor_id
            for successor_id in self._successors_map[task.aircraft_id][task.task_id]
            if self.state.tasks[f"{task.aircraft_id}_{successor_id}"].current_station
            == task.current_station
            and self.state.tasks[f"{task.aircraft_id}_{successor_id}"].status
            not in (TaskStatus.COMPLETED, TaskStatus.POSTPONED)
        ]
        if same_station_successors:
            return f"工序 {task.task_key} 仍有本站未完成后继 {same_station_successors}，禁止后移"
        return None

    def load_scenario(self, scenario: Any) -> None:
        """注入扰动场景，向事件优先队列压入最高优先级的 DISTURBANCE 事件。"""
        if hasattr(scenario, "to_dict"):
            payload = scenario.to_dict()
        elif isinstance(scenario, dict):
            payload = dict(scenario)
        else:
            raise TypeError(f"不支持的场景数据格式: {type(scenario)}")

        tau = float(payload["tau"])
        self.event_queue.push(
            event_type=EventType.DISTURBANCE,
            timestamp=tau,
            payload=payload,
        )
        if tau <= self.state.current_time + self.tolerance:
            self.process_due_events()

    def get_ready_tasks(self) -> list[TaskRuntimeState]:
        """获取全线所有在场站位中处于 READY 状态的全部工序。"""
        ready: list[TaskRuntimeState] = []
        for s in range(self.state.num_stations):
            ready.extend(self.state.get_ready_tasks_for_station(s))
        return ready

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """执行单步调度动作。"""
        cost_before = self.cumulative_cost
        task_key_value = action.get("task_key")
        task_key = None if task_key_value is None else str(task_key_value)
        branch = ActionBranch(action.get("branch", ActionBranch.STATION_EXECUTE))

        self.process_due_events()

        info: dict[str, Any] = {
            "action_branch": branch.name,
            "task_key": task_key,
            "advance_cause": "none",
        }

        explicit_advance = branch == ActionBranch.ADVANCE_TO_NEXT_EVENT
        no_change_revision = False
        if explicit_advance:
            info["advance_cause"] = (
                "explicit"
                if self.get_action_candidates()
                else "forced_no_candidate"
            )
            info["advanced"] = self._advance_to_next_event()
            if not info["advanced"] and not self._check_terminated():
                info["success"] = False
                info["termination_reason"] = "deadlock"
        else:
            if task_key is None or task_key not in self.state.tasks:
                raise KeyError(f"未找到工序: {task_key}")
            task = self.state.tasks[task_key]

            ac = self.state.aircraft[task.aircraft_id]
            if ac.current_station != task.current_station:
                raise ValueError(
                    f"工序 {task_key} 归属站位 {task.current_station}，但飞机 {task.aircraft_id} "
                    f"当前物理停靠在站位 {ac.current_station}，不可作业！"
                )

        if branch == ActionBranch.STATION_EXECUTE:
            # 分支 A：留在当前站位执行
            if not self.can_reserve(task):
                raise ValueError(f"工序 {task_key} 当前不具备留站预约资格")
            team = tuple(int(w) for w in action.get("team", ()))
            align = int(action.get("align", 0))

            self._validate_team_for_task(task, team)

            predecessor_release = self._predecessor_release_time(task)
            if predecessor_release is None:
                raise ValueError(f"工序 {task_key} 的前驱尚未完成或没有有效预约")
            search_start = max(self.state.current_time, task.material_ready_time, predecessor_release)
            if align == 1:
                align_target = self.state.last_transfer_time + task.in_station_offset
                search_start = max(search_start, align_target)

            execution_duration = self.duration_for_team(
                task,
                team,
                start_time=search_start,
            )

            t_sched = self._find_team_earliest_slot(
                station_id=task.current_station,
                team=team,
                search_start=search_start,
                duration=execution_duration,
            )

            # 首次提交不可撤回决策（固定站位、团队、对齐）
            task.reserve(team=team, scheduled_start=t_sched, alignment=align)
            task.execution_duration = execution_duration
            after_assignment = self._assignment_snapshot(
                task,
                station=task.current_station,
                team=team,
                scheduled_start=t_sched,
            )
            revision_history_size = len(task.revision_history)
            revision_cost_inc = self._record_formal_revision(
                task,
                after_assignment,
                reason="station_reservation",
            )

            for w in team:
                self.state.workers[w].add_interval(
                    start=t_sched,
                    end=t_sched + execution_duration,
                    task_key=task.task_key,
                )

            if abs(t_sched - self.state.current_time) <= self.tolerance:
                self._on_task_started(task, t_sched)
                self.event_queue.push(
                    event_type=EventType.TASK_FINISH,
                    timestamp=t_sched + execution_duration,
                    task_key=task.task_key,
                    generation=task.generation,
                )
                info["scheduled_status"] = "RUNNING"
            else:
                task.status = TaskStatus.RESERVED
                self.event_queue.push(
                    event_type=EventType.TASK_START,
                    timestamp=t_sched,
                    task_key=task.task_key,
                    generation=task.generation,
                )
                info["scheduled_status"] = "RESERVED"

            self._station_occupied_tasks[task.current_station].add(task.task_key)
            info["revision_changed"] = len(task.revision_history) > revision_history_size
            info["cost_revision_inc"] = revision_cost_inc
            info["scheduled_start"] = t_sched
            self.state.recent_reserved_count += 1

        elif branch == ActionBranch.POSTPONE:
            # 分支 B：合法后移至下一站位
            if not self.can_postpone(task):
                postpone_error = self.validate_postpone(task)
                if postpone_error is None:
                    postpone_error = f"工序 {task_key} 当前状态不允许后移"
                raise ValueError(postpone_error)

            n_old = task.postpone_count
            # 保留上一次发布的团队/位置作比较参照；后移不会在目标站预约该团队。
            previous_assignment = copy.deepcopy(
                task.last_published_assignment or task.baseline_assignment
            )
            task.postpone_to_next_station()
            self.event_queue.invalidate_task_events(task.task_key, task.generation)
            n_new = task.postpone_count
            penalty_delta = calculate_postpone_penalty(
                n_new, self.weights.lambda_1, self.weights.lambda_2
            ) - calculate_postpone_penalty(
                n_old, self.weights.lambda_1, self.weights.lambda_2
            )
            cost_postpone_inc = self.weights.w_p * penalty_delta
            self.cost_postpone += cost_postpone_inc
            self.cumulative_cost += cost_postpone_inc

            after_assignment = dict(previous_assignment)
            after_assignment["station"] = int(task.current_station)
            revision_total_inc = self._record_formal_revision(
                task,
                after_assignment,
                reason="postpone_revision",
                additional_cost=cost_postpone_inc,
            )

            info["postponed_to_station"] = task.current_station
            info["postpone_count"] = task.postpone_count
            info["cost_postpone_inc"] = cost_postpone_inc
            info["cost_revision_inc"] = revision_total_inc - cost_postpone_inc
            self.state.recent_postponed_count += 1

            # 后移可能解除原站位未完任务从而形成首次就绪事实 (R05)
            self._record_station_first_cleared_if_ready()

            # 后移可能使得当前周期站位放行条件满足，检查是否可安排转站
            self._check_and_schedule_transfer()

        if not explicit_advance:
            # 动作可能产生同刻完工或转站事件；向策略返回状态前结算到稳定状态。
            self.process_due_events()

        # 完全相同的预约重交不计修订费，但确定性结束本轮并推进一个事件，即使仍有READY任务。
        if not explicit_advance and no_change_revision:
            info["advance_cause"] = "identical_resubmission"
            info["advanced"] = self._advance_to_next_event()
            if not info["advanced"] and not self._check_terminated():
                info["success"] = False
                info["termination_reason"] = "deadlock"
        # 普通调度动作后若没有合法候选，在同一step内推进至下一决策状态，不另造动作。
        elif not explicit_advance and len(self.get_action_candidates()) == 0:
            info["advance_cause"] = "auto_after_action"
            self._advance_events_until_next_decision()
            if not self.get_action_candidates() and not self._check_terminated():
                info["success"] = False
        # 记录高频动态演化语义摘要 (R08) - 仅在动作生效后更新
        self.state.last_action_branch = int(branch.value)
        if abs(self.state.current_time - getattr(self, "_last_step_time", self.state.current_time)) < self.tolerance:
            self.state.micro_steps_in_timestamp += 1
        else:
            self.state.micro_steps_in_timestamp = 1
        self._last_step_time = float(self.state.current_time)

        self.step_count += 1
        if info["advance_cause"] not in ADVANCE_CAUSES:
            raise RuntimeError(f"未知事件推进原因: {info['advance_cause']}")
        natural_termination = self._check_terminated()
        terminated = natural_termination or info.get("termination_reason") == "deadlock"
        if natural_termination:
            info["success"] = True
            info["termination_reason"] = "completed"
        truncated = bool(
            self.max_steps_per_rollout is not None
            and self.step_count >= self.max_steps_per_rollout
            and not terminated
        )
        obs = self._get_observation()

        cost_after = self.cumulative_cost
        step_cost = cost_after - cost_before
        reward = -step_cost
        self.step_rewards.append(reward)

        info["step_cost"] = step_cost
        info["cumulative_cost"] = cost_after
        info["cost_breakdown"] = {
            "cost_takt": self.cost_takt,
            "cost_time": self.cost_time,
            "cost_team": self.cost_team,
            "cost_postpone": self.cost_postpone,
            "cost_revision": self.cost_revision,
        }
        info["station_first_cleared_times"] = {
            c: dict(m) for c, m in self.state.station_first_cleared_times.items()
        }
        info["next_cycle_station_mask"] = tuple(self.state.get_next_cycle_station_mask())
        info["cycle_execution_records"] = {
            r.cycle_id: r.to_dict() for r in self._step_cycle_records
        }
        self._step_cycle_records.clear()

        return obs, reward, terminated, truncated, info

    def _validate_team_for_task(self, task: TaskRuntimeState, team: tuple[int, ...]) -> None:
        """严格校验指派团队的人数与站位绑定。"""
        if len(team) != task.demand:
            raise ValueError(f"工序 {task.task_key} 需求 {task.demand} 人，但提供了 {len(team)} 人！")
        if len(team) != len(set(team)):
            raise ValueError(f"团队人员存在重复: {team}")

        allowed_workers = set(self.state.station_worker_bindings.get(task.current_station, []))
        for w in team:
            if w not in self.worker_efficiencies:
                raise ValueError(f"工人 {w} 不存在于工人池！")
            if w not in allowed_workers:
                raise ValueError(
                    f"工人 {w} 不属于站位 {task.current_station}！"
                    f"该站允许工人为: {allowed_workers}"
                )
            if task.skill >= 0 and task.skill not in self.worker_skills[w]:
                raise ValueError(f"工人 {w} 不具备工序 {task.task_key} 所需技能 {task.skill}！")

    def _find_team_earliest_slot(
        self, station_id: int, team: Sequence[int], search_start: float, duration: float
    ) -> float:
        """寻找团队和站位资源同时可行的最早半开区间起点。"""
        if not math.isfinite(float(duration)):
            raise ValueError(f"工时必须是有限数值: {duration}")
        if duration < 0.0:
            raise ValueError(f"工时不能为负数: {duration}")
        if duration <= self.tolerance:
            return float(search_start)

        candidate = float(search_start)
        interval_count = len(self._station_occupied_tasks.get(station_id, ()))
        interval_count += sum(len(self.state.workers[w].intervals) for w in team)
        max_iterations = max(1, interval_count + 1)

        for _ in range(max_iterations):
            team_candidate = candidate
            for worker_id in team:
                worker_slot = self.state.workers[worker_id].find_earliest_slot(
                    candidate,
                    duration,
                    tolerance=self.tolerance,
                )
                team_candidate = max(team_candidate, worker_slot)

            if team_candidate > candidate + self.tolerance:
                candidate = team_candidate
                continue

            conflict_end = self._next_station_conflict_end(
                station_id,
                candidate,
                candidate + duration,
            )
            if conflict_end is None:
                return candidate
            if conflict_end <= candidate + self.tolerance:
                break
            candidate = conflict_end

        raise NoFeasibleSlotError(
            f"站位 {station_id}、团队 {tuple(team)} 在保护搜索范围内找不到 "
            f"[start={search_start}, duration={duration}] 的可行预约区间"
        )

    def _is_station_slot_available(self, station_id: int, start: float, end: float) -> bool:
        """按区间端点扫描检查半开区间 ``[start, end)`` 的站位容量。"""
        return self._next_station_conflict_end(station_id, start, end) is None

    def _next_station_conflict_end(
        self,
        station_id: int,
        start: float,
        end: float,
    ) -> float | None:
        """返回候选区间内第一次容量超限的相关完工时刻。"""
        if end <= start + self.tolerance:
            return None

        intervals: list[tuple[float, float]] = []
        for task_key in self._station_occupied_tasks.get(station_id, ()):
            t = self.state.tasks[task_key]
            if t.scheduled_start is None:
                continue
            interval_start = float(t.scheduled_start)
            interval_duration = t.execution_duration if t.execution_duration is not None else t.duration
            interval_end = interval_start + float(interval_duration)
            if interval_end <= interval_start + self.tolerance:
                continue
            if interval_end <= start + self.tolerance or interval_start >= end - self.tolerance:
                continue
            intervals.append((interval_start, interval_end))

        time_points = {float(start), float(end)}
        for interval_start, interval_end in intervals:
            if start + self.tolerance < interval_start < end - self.tolerance:
                time_points.add(interval_start)
            if start + self.tolerance < interval_end < end - self.tolerance:
                time_points.add(interval_end)

        ordered_points = sorted(time_points)
        for left, right in zip(ordered_points, ordered_points[1:]):
            if right <= left + self.tolerance:
                continue

            active_intervals = [
                (interval_start, interval_end)
                for interval_start, interval_end in intervals
                if interval_start <= left + self.tolerance
                and interval_end > left + self.tolerance
            ]
            if len(active_intervals) >= self.max_slots_per_station:
                return min(interval_end for _, interval_end in active_intervals)

        return None

    def process_due_events(self) -> None:
        """处理当前时刻全部事件，包含处理过程中生成的同刻事件。"""
        if self._check_terminated():
            return
        while True:
            event = self.event_queue.peek()
            if event is None or event.timestamp > self.state.current_time + self.tolerance:
                self._check_and_schedule_transfer()
                next_event = self.event_queue.peek()
                if next_event is None or next_event.timestamp > self.state.current_time + self.tolerance:
                    return
                continue

            event = self.event_queue.pop()
            if event is None:
                return
            self.state.current_time = event.timestamp
            self._dispatch_event(event)
            if event.event_type in (EventType.TASK_FINISH, EventType.SYNCHRONOUS_TRANSFER):
                if self._check_terminated():
                    return

    def _dispatch_event(self, event: SimulationEvent) -> None:
        """执行一个已按时间和优先级取出的事件。"""
        if event.event_type == EventType.TASK_START:
            self._handle_task_start_event(event)
        elif event.event_type == EventType.TASK_FINISH:
            task = self.state.tasks[event.task_key]
            if task.status == TaskStatus.RUNNING:
                task.complete_work(current_time=event.timestamp)
                self._station_occupied_tasks[task.current_station].discard(task.task_key)
                self._on_task_completed(task)
        elif event.event_type == EventType.MATERIAL_ARRIVE:
            task = self.state.tasks[event.task_key]
            self._check_and_update_task_readiness(task)
        elif event.event_type == EventType.SYNCHRONOUS_TRANSFER:
            self._execute_synchronous_transfer(event.timestamp)
        elif event.event_type == EventType.DISTURBANCE:
            self.disturbance_event_triggered = True
            self._handle_disturbance_event(event.timestamp, event.payload)

    def _handle_task_start_event(self, event: SimulationEvent) -> None:
        """在预约开工时重新核验物理条件，失败则撤销预约。"""
        task = self.state.tasks[event.task_key]
        if task.status != TaskStatus.RESERVED:
            return

        aircraft = self.state.aircraft[task.aircraft_id]
        predecessors_completed = all(
            self.state.tasks[f"{task.aircraft_id}_{pred_id}"].status == TaskStatus.COMPLETED
            for pred_id in task.predecessors
        )
        valid_start = (
            aircraft.current_station == task.current_station
            and task.can_physically_start(event.timestamp, self.tolerance)
            and predecessors_completed
        )
        try:
            self._validate_team_for_task(task, tuple(task.assigned_team))
        except ValueError:
            valid_start = False

        if not valid_start:
            # 物理开工条件未满足，保持决定不可撤回，被动顺延至可行时刻
            self._passive_defer_reserved_task(task, cause="preconditions_unmet")
            return

        self._on_task_started(task, event.timestamp)
        self.event_queue.push(
            event_type=EventType.TASK_FINISH,
            timestamp=event.timestamp
            + (
                task.execution_duration
                if task.execution_duration is not None
                else task.duration
            ),
            task_key=task.task_key,
            generation=task.generation,
        )

    def _advance_events_until_next_decision(self) -> None:
        """在当前step内推进，直至出现合法动作、批次完成或事件耗尽。"""
        while True:
            self.process_due_events()
            if self._check_terminated():
                return
            if self.get_action_candidates():
                return
            next_event = self.event_queue.peek()
            if next_event is None:
                return
            self.state.current_time = max(self.state.current_time, next_event.timestamp)

    def _handle_disturbance_event(self, timestamp: float, payload: dict[str, Any]) -> None:
        """处理突发工序开工可用性延迟扰动 (Task 4.2 / 不可撤回与被动顺延原则)。

        物理规则：
        1. 已开工 (RUNNING) 与已完工 (COMPLETED) 工序硬冻结，绝不强制中断；
        2. 已预约 (RESERVED) 且原排定开工时刻 t_sched < R 的工序：
           - 严格保持已提交的站位、团队与对齐决策不变；
           - 被动顺延执行区间至最早可行无冲突时刻；
           - 递增代数标记使旧开工事件失效，压入新时刻开工事件；
        3. 处于 READY 状态的工序：
           - 由于 R > current_time，失去当前就绪资格，退回 UNREADY；
           - 压入时间戳为 R 的MATERIAL_ARRIVE兼容事件；
        4. 处于 UNREADY 或 POSTPONED 状态的工序：
           - 更新开工可用下界 max(material_ready_time, R)；字段名为历史兼容名。
        """
        recovery_time = float(payload["recovery_time"])
        affected_keys = payload.get("affected_task_keys", [])
        scenario_id = str(payload.get("scenario_id", ""))
        hit_task_keys: list[str] = []
        unhit_reasons: dict[str, str] = {}

        for raw_task_key in affected_keys:
            task_key = str(raw_task_key)
            task = self.state.tasks.get(task_key)
            if task is None:
                unhit_reasons[task_key] = "target_not_in_instance"
                continue

            if task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED):
                # 物理规则 3：实际已开工与已完工作业硬冻结，绝不强制打断
                unhit_reasons[task_key] = "already_started_or_completed_at_event"
                continue

            hit_task_keys.append(task_key)
            # 统一建模尚未开工工序的开工可用下界：
            # r_{k*,i}^{new} = max(r_{k*,i}^{old}, R)。
            task.material_ready_time = max(task.material_ready_time, recovery_time)

            if task.status == TaskStatus.RESERVED:
                # 扰动只使固定决定被动顺延，不能清空决定或重新选队
                if task.scheduled_start is not None and task.scheduled_start < recovery_time - self.tolerance:
                    self._passive_defer_reserved_task(task, cause="disturbance")

            elif task.status == TaskStatus.READY:
                # 开工可用时刻推迟至未来，退回 UNREADY
                task.status = TaskStatus.UNREADY
                self.event_queue.push(
                    event_type=EventType.MATERIAL_ARRIVE,
                    timestamp=recovery_time,
                    task_key=task.task_key,
                    generation=task.generation,
                )

            elif task.status in (TaskStatus.UNREADY, TaskStatus.POSTPONED):
                pass

        self.disturbance_event_results[scenario_id] = {
            "actual_hit_task_keys": tuple(hit_task_keys),
            "unhit_reasons": unhit_reasons,
        }

    def _on_task_started(self, task: TaskRuntimeState, start_time: float) -> None:
        """工序正式进入 RUNNING 状态，结算基准开工位置偏差 D_time 与团队替换 D_team 增量。"""
        task.start_work(
            current_time=start_time,
            cycle_start_time=self.state.last_transfer_time,
            actual_start_station=self.state.aircraft[task.aircraft_id].current_station,
        )
        if task.start_cost_confirmed:
            return

        task.start_cost_confirmed = True
        scale = (1.0 / self.total_tasks) if (self.weights.normalize_by_n and self.total_tasks > 0) else 1.0

        # 1. 周期内开工位置偏差 b_ki - b_i0
        b_ki = start_time - self.state.last_transfer_time
        b_i0 = task.in_station_offset
        d_time_item = abs(b_ki - b_i0) / self.state.h0
        cost_time_inc = self.weights.w_t * d_time_item * scale

        # 2. 团队替换率 d(W, W^0) = 1 - |W cap W^0| / m_i
        w_actual = set(task.assigned_team)
        w_base = set(task.base_team)
        if task.demand > 0:
            overlap = len(w_actual & w_base)
            d_team_item = 1.0 - (overlap / task.demand)
        else:
            d_team_item = 0.0
        cost_team_inc = self.weights.w_w * d_team_item * scale

        self.cost_time += cost_time_inc
        self.cost_team += cost_team_inc
        self.cumulative_cost += (cost_time_inc + cost_team_inc)

    def _on_task_completed(self, completed_task: TaskRuntimeState) -> None:
        """工序完工后的事件链：记录站位就绪时刻并解锁同机直接物理后继。"""
        self._record_station_first_cleared_if_ready()
        k = completed_task.aircraft_id
        tid = completed_task.task_id

        successor_ids = self._successors_map.get(k, {}).get(tid, [])
        for succ_id in successor_ids:
            succ_key = f"{k}_{succ_id}"
            succ_task = self.state.tasks.get(succ_key)
            if succ_task is not None and succ_task.status == TaskStatus.UNREADY:
                self._check_and_update_task_readiness(succ_task)

    def _check_and_update_task_readiness(self, task: TaskRuntimeState) -> None:
        """检查工序的前驱完工、物理到料与进站情况，若满足则置为 READY。"""
        if task.status != TaskStatus.UNREADY:
            return

        ac = self.state.aircraft[task.aircraft_id]
        if ac.current_station != task.current_station:
            return

        for pred_id in task.predecessors:
            pred_key = f"{task.aircraft_id}_{pred_id}"
            pred_task = self.state.tasks.get(pred_key)
            if pred_task is not None and pred_task.status != TaskStatus.COMPLETED:
                return

        if self.state.current_time >= task.material_ready_time - self.tolerance:
            task.status = TaskStatus.READY
        else:
            self.event_queue.push(
                event_type=EventType.MATERIAL_ARRIVE,
                timestamp=task.material_ready_time,
                task_key=task.task_key,
                generation=task.generation,
            )

    def _refresh_aircraft_station_readiness(self, aircraft_id: int, station_id: int) -> None:
        """飞机进驻新站位时，扫描并激活该站位归属工序。"""
        for task in self.state.get_tasks_for_station(station_id):
            if task.status in (TaskStatus.UNREADY, TaskStatus.POSTPONED):
                task.status = TaskStatus.UNREADY
                self._check_and_update_task_readiness(task)

    def _check_and_schedule_transfer(self) -> None:
        """检查全线放行条件，按实际完成时刻安排同步转站。"""
        if self._check_terminated():
            return
        if not self.state.is_all_stations_cleared_for_transfer():
            return

        # 避免为同一个周期重复排定转站事件
        if self._transfer_scheduled_for_cycle == self.state.current_cycle:
            return

        # 检查当前现场是否仍有未完工但在制/预约的任务
        has_active_tasks = any(
            bool(keys) for keys in self._station_occupied_tasks.values()
        )
        if has_active_tasks:
            # 尚有任务在加工，等待其完成
            return

        latest_finish = self.state.current_time

        for s in range(self.state.num_stations):
            for t in self.state.get_tasks_for_station(s):
                if t.status == TaskStatus.COMPLETED:
                    if t.actual_end is not None and t.actual_end > latest_finish:
                        latest_finish = t.actual_end

        # H0只用于超期费用和基准比较，不构成实际转站的等待下界。
        transfer_time = max(latest_finish, self.state.current_time)

        self._transfer_scheduled_for_cycle = self.state.current_cycle
        self.event_queue.push(
            event_type=EventType.SYNCHRONOUS_TRANSFER,
            timestamp=transfer_time,
            payload={"cycle": self.state.current_cycle},
        )

    def _execute_synchronous_transfer(self, timestamp: float) -> None:
        """执行全线同步脉动流转（Task 2.5）：各机站位+1，唤醒后移任务。"""
        # 计账：结算本周期节拍超期 J_takt 增量
        duration_h = float(timestamp) - self.state.last_transfer_time
        overdue_h = max(0.0, duration_h - self.state.h0)
        takt_violation = overdue_h / self.state.h0
        cost_takt_inc = self.weights.w_h * takt_violation
        self.cost_takt += cost_takt_inc
        self.cumulative_cost += cost_takt_inc

        # 确保当前周期所有占位站位记录首次就绪时刻 (R05)
        current_q = self.state.current_cycle
        cleared_map = self.state.station_first_cleared_times.setdefault(current_q, {})
        for s in range(self.state.num_stations):
            if s not in cleared_map and self.state.get_aircraft_at_station(s) is not None:
                cleared_map[s] = float(timestamp)

        occupied = tuple(self.state.get_aircraft_at_station(s) is not None for s in range(self.state.num_stations))
        aircraft_at_stations = tuple(self.state.get_aircraft_at_station(s) for s in range(self.state.num_stations))
        cycle_rec = CycleExecutionRecord(
            cycle_id=current_q,
            cycle_start_time=float(self.state.last_transfer_time),
            transfer_time=float(timestamp),
            occupied_stations=occupied,
            aircraft_at_stations=aircraft_at_stations,
            station_first_cleared_times=dict(cleared_map),
        )
        self.state.cycle_execution_records[current_q] = cycle_rec
        self._step_cycle_records.append(cycle_rec)

        self.state.last_transfer_time = float(timestamp)
        self.state.transfer_history.append(float(timestamp))
        self.state.current_cycle += 1

        # 各在场飞机向前推进一步
        for k in range(self.state.num_aircraft):
            ac = self.state.aircraft[k]
            if ac.is_in_factory:
                ac.step_to_next_station(timestamp)
                if ac.is_in_factory:
                    self._refresh_aircraft_station_readiness(k, ac.current_station)

        # 检查下一架待进线飞机 k = current_q
        next_k = current_q
        if next_k < self.state.num_aircraft:
            ac_next = self.state.aircraft[next_k]
            if ac_next.current_station == -1:
                ac_next.current_station = 0
                ac_next.entry_times[0] = float(timestamp)
                self._refresh_aircraft_station_readiness(next_k, 0)

        # 脉动流转后刷新下一周期的状态演化计数并探测新就绪站位 (R05 & R08)
        self.state.recent_postponed_count = 0
        self.state.recent_reserved_count = 0
        self._record_station_first_cleared_if_ready()

    def _check_terminated(self) -> bool:
        """检查生产是否自然终止（Task 2.6：全部飞机离开末站，所有工序完工）。"""
        all_completed = all(t.status == TaskStatus.COMPLETED for t in self.state.tasks.values())
        all_exited = all(ac.is_completed for ac in self.state.aircraft.values())
        return all_completed and all_exited

    def _get_observation(self) -> dict[str, Any]:
        """构造当前现场观测字典。"""
        ready_tasks = self.get_ready_tasks()
        return {
            "current_time": self.state.current_time,
            "current_cycle": self.state.current_cycle,
            "last_transfer_time": self.state.last_transfer_time,
            "ready_task_keys": [t.task_key for t in ready_tasks],
            "aircraft_stations": {k: ac.current_station for k, ac in self.state.aircraft.items()},
        }
