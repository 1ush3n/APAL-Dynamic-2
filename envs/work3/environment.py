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

from core.constraints import ConstraintEngine, calculate_team_synergy_factor
from envs.work3.core_types import (
    ActionBranch,
    AircraftRuntimeState,
    MultiAircraftState,
    TaskRuntimeState,
    TaskStatus,
    initialize_multi_aircraft_state,
)
from envs.work3.decision_snapshot import (
    TeamCompletionContext,
    WorkerSnapshot,
    team_completion_worker_ids,
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
        self.worker_efficiencies: dict[int, float] = {}
        self.worker_skills: dict[int, frozenset[int]] = {}
        self.constraint_engine = self._load_domain_metadata()
        self._attach_task_domain_metadata()
        self.event_queue: DiscreteEventQueue = DiscreteEventQueue()
        self._transfer_scheduled_for_cycle: int = 0
        self.total_tasks: int = len(self.state.tasks)
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
        self._transfer_scheduled_for_cycle = 0
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

        # 解锁 0 号飞机在 0 号站位首批无前驱且物料就绪的工序
        self._refresh_aircraft_station_readiness(aircraft_id=0, station_id=0)

        return self._get_observation()

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

        with self.raw_data_path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_reader = csv.DictReader(handle)
            raw_rows = list(raw_reader)
            fieldnames = raw_reader.fieldnames or []
        if not raw_rows:
            raise ValueError(f"原始工艺数据为空: {self.raw_data_path}")

        task_id_column = self._find_domain_column(
            fieldnames, ("序号", "task_id", "TaskID"), "工序编号"
        )
        ao_column = self._find_domain_column(fieldnames, ("AO号", "ao_code", "AO"), "AO号")
        predecessor_column = self._find_domain_column(
            fieldnames, ("紧前工序AO号", "predecessors", "predecessor"), "紧前工序"
        )
        duration_column = self._find_domain_column(
            fieldnames, ("加工时间/h", "duration", "Duration"), "加工时间"
        )
        skill_column = self._find_domain_column(fieldnames, ("工种", "skill", "skill_type"), "工种")
        demand_column = self._find_domain_column(
            fieldnames, ("需求人数", "demand", "demand_workers"), "需求人数"
        )
        fixed_station_column = self._find_domain_column(
            fieldnames,
            ("限定站位", "fixed_station", "Fixed_Station", "station_constraint"),
            "固定站位",
        )

        rows_by_task_id: dict[int, dict[str, str]] = {}
        ao_to_task_id: dict[str, int] = {}
        for row in raw_rows:
            task_id = int(float(row[task_id_column])) - 1
            if task_id in rows_by_task_id:
                raise ValueError(f"原始工艺数据存在重复工序编号: {task_id}")
            rows_by_task_id[task_id] = row
            ao_to_task_id[str(row[ao_column]).strip()] = task_id

        num_raw_tasks = max(rows_by_task_id) + 1
        if set(rows_by_task_id) != set(range(num_raw_tasks)):
            raise ValueError("原始工艺工序编号不是连续的0-based集合")

        durations = np.zeros(num_raw_tasks, dtype=float)
        fixed_stations = np.full(num_raw_tasks, -1, dtype=np.int64)
        required_skills: dict[int, int] = {}
        demands: dict[int, int] = {}
        edges: list[tuple[int, int]] = []
        for task_id, row in rows_by_task_id.items():
            durations[task_id] = float(row[duration_column] or 0.0)
            fixed_stations[task_id] = self._parse_optional_station(
                row[fixed_station_column], task_id=task_id
            )
            required_skills[task_id] = int(float(row[skill_column]))
            demands[task_id] = int(float(row[demand_column]))
            raw_predecessors = str(row[predecessor_column] or "").strip()
            for token in re.split(r"[,，;；\s]+", raw_predecessors):
                if token and token in ao_to_task_id:
                    edges.append((ao_to_task_id[token], task_id))

        edge_array = np.asarray(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
        engine = ConstraintEngine.build(
            num_tasks=num_raw_tasks,
            num_stations=self.state.num_stations,
            edges=edge_array,
            durations=durations,
            fixed_stations=fixed_stations,
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
        return engine.with_max_allowed_stations(max_allowed)

    def _attach_task_domain_metadata(self) -> None:
        """把原始工艺约束写入当前多架次任务状态。"""
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
            fixed_station = int(self.constraint_engine.fixed_stations[task_id])
            task.fixed_station = fixed_station if fixed_station >= 0 else None
            task.max_allowed_station = int(self.constraint_engine.max_allowed_stations[task_id])

    def valid_team_completion_workers(
        self,
        task: TaskRuntimeState,
        selected_team: Sequence[int],
        *,
        station_id: int | None = None,
    ) -> list[int]:
        """返回指定站位上加入部分团队后仍可补全合法团队的工人。"""
        return list(
            team_completion_worker_ids(
                self._team_completion_context(
                    task,
                    station_id=station_id,
                    include_calendars=False,
                ),
                tuple(int(worker_id) for worker_id in selected_team),
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
            include_calendars=True,
        )

    def _team_completion_context(
        self,
        task: TaskRuntimeState,
        *,
        station_id: int | None,
        include_calendars: bool,
    ) -> TeamCompletionContext:
        """共享团队资格规则；常规候选检查不复制可能很长的资源日历。"""
        candidate_station = task.current_station if station_id is None else int(station_id)
        worker_ids = tuple(self.state.station_worker_bindings.get(candidate_station, ()))
        snapshots: list[WorkerSnapshot] = []
        for worker_id in worker_ids:
            calendar = self.state.workers.get(worker_id)
            intervals = () if calendar is None or not include_calendars else tuple(
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

    def _invalidate_dependent_reservations(self, task: TaskRuntimeState) -> None:
        """沿工艺后继检查预约下界，仅撤销失去有效前驱时序的后继预约。"""
        successors = self._successors_map[task.aircraft_id][task.task_id]
        if not successors:
            return
        pending = list(successors)
        visited: set[int] = set()
        while pending:
            successor_id = pending.pop()
            if successor_id in visited:
                continue
            visited.add(successor_id)
            pending.extend(self._successors_map[task.aircraft_id][successor_id])

            dependent = self.state.tasks[f"{task.aircraft_id}_{successor_id}"]
            if dependent.status != TaskStatus.RESERVED or dependent.actual_start is not None:
                continue

            release_time = self._predecessor_release_time(dependent)
            earliest_start = (
                None
                if release_time is None
                else max(
                    release_time,
                    dependent.material_ready_time,
                    self.state.current_time,
                )
            )
            if (
                earliest_start is not None
                and dependent.scheduled_start is not None
                and dependent.execution_duration is not None
                and dependent.scheduled_start >= earliest_start - self.tolerance
            ):
                continue

            self._release_reserved_resources(dependent)
            dependent.cancel_reservation()
            self.event_queue.invalidate_task_events(
                dependent.task_key,
                dependent.generation,
            )
            dependent.status = TaskStatus.UNREADY
            self._check_and_update_task_readiness(dependent)

    def can_reserve(self, task: TaskRuntimeState) -> bool:
        """判断任务是否可以登记留站预约，不代表当前时刻可以实际开工。"""
        if task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED):
            return False
        if task.status == TaskStatus.RESERVED and task.actual_start is not None:
            return False
        aircraft = self.state.aircraft[task.aircraft_id]
        if aircraft.current_station != task.current_station:
            return False
        if len(self.valid_team_completion_workers(task, [])) < task.demand:
            return False
        return self._predecessor_release_time(task) is not None

    def can_postpone(self, task: TaskRuntimeState) -> bool:
        """判断任务是否可以独立后移，不要求物料已在当前时刻恢复。"""
        if task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED):
            return False
        if task.status == TaskStatus.RESERVED and task.actual_start is not None:
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
        if task.duration <= self.tolerance:
            return 0.0
        efficiency_sum = sum(self.worker_efficiencies[worker_id] for worker_id in team_tuple)
        effective_capacity = efficiency_sum * calculate_team_synergy_factor(len(team_tuple))
        if effective_capacity <= self.tolerance:
            raise ValueError(f"团队 {team_tuple} 的有效工时能力不足")
        return float(task.duration * task.demand / effective_capacity)

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
        info: dict[str, Any] = {"action_branch": branch.name, "task_key": task_key}

        explicit_advance = branch == ActionBranch.ADVANCE_TO_NEXT_EVENT
        no_change_revision = False
        if explicit_advance:
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
            was_reserved = task.status == TaskStatus.RESERVED
            old_team = tuple(task.assigned_team)
            old_start = task.scheduled_start
            old_duration = task.execution_duration

            if was_reserved:
                self._release_reserved_resources(task)

            try:
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
            except Exception:
                if was_reserved:
                    self._restore_reserved_resources(task)
                raise

            unchanged = was_reserved and (
                set(old_team) == set(team)
                and old_start is not None
                and abs(float(old_start) - t_sched) <= self.tolerance
                and old_duration is not None
                and abs(float(old_duration) - execution_duration) <= self.tolerance
            )
            if unchanged:
                self._restore_reserved_resources(task)
                no_change_revision = True
                info["revision_changed"] = False
                info["cost_revision_inc"] = 0.0
                info["scheduled_status"] = "RESERVED"
                info["scheduled_start"] = old_start
            else:
                if was_reserved:
                    task.generation += 1
                    self.event_queue.invalidate_task_events(task.task_key, task.generation)

                task.execution_duration = execution_duration
                task.assigned_team = list(team)
                task.scheduled_start = t_sched
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
                    reason="reservation_revision",
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
                if was_reserved:
                    self._invalidate_dependent_reservations(task)
                info["revision_changed"] = len(task.revision_history) > revision_history_size
                info["cost_revision_inc"] = revision_cost_inc
                info["scheduled_start"] = t_sched

        elif branch == ActionBranch.POSTPONE:
            # 分支 B：合法后移至下一站位
            if not self.can_postpone(task):
                postpone_error = self.validate_postpone(task)
                if postpone_error is None:
                    postpone_error = f"工序 {task_key} 当前状态不允许后移"
                raise ValueError(postpone_error)

            was_reserved = task.status == TaskStatus.RESERVED
            if was_reserved:
                self._release_reserved_resources(task)
            n_old = task.postpone_count
            previous_assignment = copy.deepcopy(
                task.last_published_assignment or task.baseline_assignment
            )
            task.postpone_to_next_station()
            self.event_queue.invalidate_task_events(task.task_key, task.generation)
            if was_reserved:
                self._invalidate_dependent_reservations(task)
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

            # 后移可能使得当前周期站位放行条件满足，检查是否可安排转站
            self._check_and_schedule_transfer()

        # 没有任何合法调度动作时自动推进；存在 RESERVED 修订候选时交给显式推进动作决定。
        if not explicit_advance and no_change_revision:
            info["advanced"] = self._advance_to_next_event()
            if not info["advanced"] and not self._check_terminated():
                info["success"] = False
                info["termination_reason"] = "deadlock"
        elif not explicit_advance and len(self.get_action_candidates()) == 0:
            self._advance_events_until_next_decision()
            if not self.get_action_candidates() and not self._check_terminated():
                info["success"] = False
                info["termination_reason"] = "deadlock"

        self.step_count += 1
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
            for worker_id in task.assigned_team:
                self.state.workers[worker_id].remove_interval(task.task_key)
            self._station_occupied_tasks[task.current_station].discard(task.task_key)
            task.cancel_reservation()
            task.status = TaskStatus.UNREADY
            self.event_queue.invalidate_task_events(task.task_key, task.generation)
            self._check_and_update_task_readiness(task)
            self._invalidate_dependent_reservations(task)
            return

        self._on_task_started(task, event.timestamp)
        self.event_queue.push(
            event_type=EventType.TASK_FINISH,
            timestamp=event.timestamp + (task.execution_duration or task.duration),
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
        """处理突发工序开工可用性延迟扰动 (Task 4.2 / 易错点 1 攻坚)。

        物理规则：
        1. 已开工 (RUNNING) 与已完工 (COMPLETED) 工序硬冻结，绝不强制中断；
        2. 已预约 (RESERVED) 且原排定开工时刻 t_sched < R 的工序：
           - 立刻释放所有指派工人的时间日历预占区间；
           - 调用 task.cancel_reservation() 递增版本代数，使事件队列旧开工事件失效；
           - 将工序退回 UNREADY 状态；
           - 压入时间戳为 R 的 MATERIAL_ARRIVE 事件；
        3. 处于 READY 状态的工序：
           - 由于 R > current_time，失去当前就绪资格，退回 UNREADY；
           - 压入时间戳为 R 的 MATERIAL_ARRIVE 事件；
        4. 处于 UNREADY 或 POSTPONED 状态的工序：
           - 更新 material_ready_time = max(material_ready_time, R)。
        """
        recovery_time = float(payload["recovery_time"])
        affected_keys = payload.get("affected_task_keys", [])

        for task_key in affected_keys:
            task = self.state.tasks.get(task_key)
            if task is None:
                continue

            if task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED):
                # 物理规则 3：实际已开工与已完工作业硬冻结，绝不强制打断
                continue

            # 统一建模尚未开工工序的物料开工下界：
            # r_{k*,i}^{new} = max(r_{k*,i}^{old}, R)。
            task.material_ready_time = max(task.material_ready_time, recovery_time)

            if task.status == TaskStatus.RESERVED:
                # 易错点 1 攻坚：若原排定开工时刻早于物理恢复时刻 R，必须立刻废除旧预约！
                if task.scheduled_start is not None and task.scheduled_start < recovery_time - self.tolerance:
                    # 1. 立即释放指派工人的时间日历预占区间
                    for w in task.assigned_team:
                        self.state.workers[w].remove_interval(task.task_key)
                    # 2. 取消预约（递增代数标记使事件队列旧开工事件失效）
                    task.cancel_reservation()
                    self._station_occupied_tasks[task.current_station].discard(task.task_key)
                    self.event_queue.invalidate_task_events(task.task_key, task.generation)
                    task.status = TaskStatus.UNREADY
                    # 3. 挂载时间戳为 R 的 MATERIAL_ARRIVE 事件
                    self.event_queue.push(
                        event_type=EventType.MATERIAL_ARRIVE,
                        timestamp=recovery_time,
                        task_key=task.task_key,
                        generation=task.generation,
                    )
                    self._invalidate_dependent_reservations(task)

            elif task.status == TaskStatus.READY:
                # 物料推迟到未来时刻到达，退回 UNREADY
                task.status = TaskStatus.UNREADY
                self.event_queue.push(
                    event_type=EventType.MATERIAL_ARRIVE,
                    timestamp=recovery_time,
                    task_key=task.task_key,
                    generation=task.generation,
                )

            elif task.status in (TaskStatus.UNREADY, TaskStatus.POSTPONED):
                pass

    def _on_task_started(self, task: TaskRuntimeState, start_time: float) -> None:
        """工序正式进入 RUNNING 状态，结算基准开工位置偏差 D_time 与团队替换 D_team 增量。"""
        task.start_work(current_time=start_time, cycle_start_time=self.state.last_transfer_time)
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
        """工序完工后的事件链：解锁同机直接物理后继。"""
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

        self.state.last_transfer_time = float(timestamp)
        self.state.transfer_history.append(float(timestamp))
        current_q = self.state.current_cycle
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
