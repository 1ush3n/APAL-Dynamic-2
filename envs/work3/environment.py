"""工作三多架次飞机脉动装配协同调度核心仿真环境 (AirLineEnvWork3)。

严格遵循顶层设计与 Ponytail 准则：
- 最小可用代码，无冗余抽象；
- 维护多架次全局状态与离散事件队列；
- 支持自回归动作分支 A（留在当前站：指派团队、二元对齐、立即开工或未来预约）；
- 支持自回归动作分支 B（合法后移至下一站：动作当场截断，保留恢复时间 R）；
- 支持无可用动作时自动推进离散事件。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from envs.work3.core_types import (
    ActionBranch,
    AircraftRuntimeState,
    MultiAircraftState,
    TaskRuntimeState,
    TaskStatus,
    initialize_multi_aircraft_state,
)
from envs.work3.event_queue import DiscreteEventQueue, EventType, SimulationEvent

logger = logging.getLogger(__name__)


class AirLineEnvWork3:
    """多架次飞机脉动装配协同调度环境。"""

    def __init__(
        self,
        baseline_json_path: str = "data/work3/real_283_k10_baseline.json",
        max_slots_per_station: int = 3,
        tolerance: float = 1e-5,
    ) -> None:
        self.baseline_json_path = baseline_json_path
        self.max_slots_per_station = int(max_slots_per_station)
        self.tolerance = float(tolerance)

        self.state: MultiAircraftState = initialize_multi_aircraft_state(self.baseline_json_path)
        self.event_queue: DiscreteEventQueue = DiscreteEventQueue()

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
        self.event_queue.reset(start_time=0.0)

        # 0 号飞机在时刻 0.0 进入 0 号站位
        ac0 = self.state.aircraft[0]
        ac0.current_station = 0
        ac0.entry_times[0] = 0.0

        # 解锁 0 号飞机在 0 号站位首批无前驱且物料就绪的工序
        self._refresh_aircraft_station_readiness(aircraft_id=0, station_id=0)

        return self._get_observation()

    def get_ready_tasks(self) -> list[TaskRuntimeState]:
        """获取全线所有在场站位中处于 READY 状态的全部工序。"""
        ready: list[TaskRuntimeState] = []
        for s in range(self.state.num_stations):
            ready.extend(self.state.get_ready_tasks_for_station(s))
        return ready

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """执行单步调度动作。

        Action 格式：
        {
            "task_key": str,           # 选定工序标识 (如 "0_2")
            "branch": int | ActionBranch,  # 0: STAY (留站执行), 1: POSTPONE (后移)
            "team": Sequence[int],      # 指派工人团队 (分支 0 必填)
            "align": int,              # 二元对齐 0: 最早可行, 1: 对齐周期内位置 (分支 0)
        }
        """
        task_key = str(action["task_key"])
        branch = ActionBranch(action.get("branch", ActionBranch.STATION_EXECUTE))

        if task_key not in self.state.tasks:
            raise KeyError(f"未找到工序: {task_key}")

        task = self.state.tasks[task_key]
        if task.status != TaskStatus.READY:
            raise ValueError(f"工序 {task_key} 当前状态为 {task.status.name}，不可调度！必须为 READY。")

        # 校验工序当前所在站位是否有该飞机在场
        ac = self.state.aircraft[task.aircraft_id]
        if ac.current_station != task.current_station:
            raise ValueError(
                f"工序 {task_key} 归属站位 {task.current_station}，但飞机 {task.aircraft_id} "
                f"当前物理停靠在站位 {ac.current_station}，不可作业！"
            )

        reward = 0.0
        info: dict[str, Any] = {"action_branch": branch.name, "task_key": task_key}

        if branch == ActionBranch.STATION_EXECUTE:
            # -------------------------------------------------------------
            # 分支 A：留在当前站位执行 (Task 2.3)
            # -------------------------------------------------------------
            team = tuple(int(w) for w in action.get("team", ()))
            align = int(action.get("align", 0))

            # 校验团队资质与站位独占绑定
            self._validate_team_for_task(task, team)

            # 计算开工时间搜索起点下界
            search_start = max(self.state.current_time, task.material_ready_time)
            if align == 1:
                # 对齐下界：上次转站时刻 + 模板偏移 b_i^0
                align_target = self.state.last_transfer_time + task.in_station_offset
                search_start = max(search_start, align_target)

            # 在团队所有工人的日历与站位槽位中搜索最早可行执行区间
            t_sched = self._find_team_earliest_slot(
                station_id=task.current_station,
                team=team,
                search_start=search_start,
                duration=task.duration,
            )

            # 登记工人日历占用
            for w in team:
                self.state.workers[w].add_interval(
                    start=t_sched,
                    end=t_sched + task.duration,
                    task_key=task.task_key,
                )

            if abs(t_sched - self.state.current_time) <= self.tolerance:
                # 立即开工
                task.start_work(current_time=self.state.current_time)
                task.assigned_team = list(team)
                task.scheduled_start = t_sched
                self.event_queue.push(
                    event_type=EventType.TASK_FINISH,
                    timestamp=t_sched + task.duration,
                    task_key=task.task_key,
                    generation=task.generation,
                )
                info["scheduled_status"] = "RUNNING"
                info["scheduled_start"] = t_sched
            else:
                # 未来预约
                task.reserve(team=team, scheduled_start=t_sched)
                self.event_queue.push(
                    event_type=EventType.TASK_START,
                    timestamp=t_sched,
                    task_key=task.task_key,
                    generation=task.generation,
                )
                info["scheduled_status"] = "RESERVED"
                info["scheduled_start"] = t_sched

        elif branch == ActionBranch.POSTPONE:
            # -------------------------------------------------------------
            # 分支 B：合法后移至下一站位 (Task 2.4)
            # -------------------------------------------------------------
            if task.current_station >= self.state.num_stations - 1:
                raise ValueError(f"末站（站位 {task.current_station + 1}）工序绝对禁止后移！")

            # 动作截断：不指派团队，不占工人日历，不选对齐
            task.postpone_to_next_station()
            info["postponed_to_station"] = task.current_station
            info["postpone_count"] = task.postpone_count

        # 动作执行后，若当前现场无 READY 任务，自动推进离散事件
        if len(self.get_ready_tasks()) == 0:
            self._advance_events_until_next_decision()

        terminated = self._check_terminated()
        truncated = False
        obs = self._get_observation()

        return obs, reward, terminated, truncated, info

    def _validate_team_for_task(self, task: TaskRuntimeState, team: tuple[int, ...]) -> None:
        """严格校验指派团队的人数与站位绑定。"""
        if len(team) != task.demand:
            raise ValueError(f"工序 {task.task_key} 需求 {task.demand} 人，但提供了 {len(team)} 人！")
        if len(team) != len(set(team)):
            raise ValueError(f"团队人员存在重复: {team}")

        allowed_workers = set(self.state.station_worker_bindings.get(task.current_station, []))
        for w in team:
            if w not in allowed_workers:
                raise ValueError(
                    f"工人 {w} 不属于站位 {task.current_station}！"
                    f"该站允许工人为: {allowed_workers}"
                )

    def _find_team_earliest_slot(
        self, station_id: int, team: Sequence[int], search_start: float, duration: float
    ) -> float:
        """寻找满足团队全员空闲且站位槽位并发 <= 3 的最早时间起点。"""
        candidate = search_start
        max_search_steps = 1000

        for _ in range(max_search_steps):
            # 1. 检查团队成员全员可用性
            next_free = candidate
            for w in team:
                wc = self.state.workers[w]
                slot = wc.find_earliest_slot(candidate, duration, tolerance=self.tolerance)
                if slot > next_free:
                    next_free = slot

            if next_free > candidate + self.tolerance:
                candidate = next_free
                continue

            # 2. 检查站位槽位并发上限
            if not self._is_station_slot_available(station_id, candidate, candidate + duration):
                # 槽位超限，向后微步推移
                candidate += 0.5
                continue

            return candidate

        return candidate

    def _is_station_slot_available(self, station_id: int, start: float, end: float) -> bool:
        """检查指定站位在 [start, end] 区间内槽位并发数是否 <= max_slots_per_station - 1。"""
        # 收集该站位所有已排定区间的任务
        intervals: list[tuple[float, float]] = []
        for t in self.state.tasks.values():
            if t.current_station == station_id and t.status in (
                TaskStatus.RUNNING,
                TaskStatus.RESERVED,
            ):
                if t.scheduled_start is not None:
                    intervals.append((t.scheduled_start, t.scheduled_start + t.duration))

        # 扫描 [start, end] 期间的已有并发数
        for s_ex, e_ex in intervals:
            # 若已有重叠，统计重叠密度
            pass

        # 简易而严格的离散采样或端点重叠检查
        time_points = [start, (start + end) / 2.0, end - self.tolerance]
        for pt in time_points:
            active_count = sum(1 for s_ex, e_ex in intervals if s_ex <= pt < e_ex - self.tolerance)
            if active_count >= self.max_slots_per_station:
                return False

        return True

    def _advance_events_until_next_decision(self) -> None:
        """推进事件队列，直至产生新的就绪工序或全线已无事件。"""
        while not self.event_queue.is_empty() and len(self.get_ready_tasks()) == 0:
            event = self.event_queue.pop()
            if event is None:
                break

            self.state.current_time = event.timestamp

            if event.event_type == EventType.TASK_START:
                # 预约工序到达开工时刻，转为 RUNNING
                task = self.state.tasks[event.task_key]
                if task.status == TaskStatus.RESERVED:
                    task.start_work(current_time=event.timestamp)
                    self.event_queue.push(
                        event_type=EventType.TASK_FINISH,
                        timestamp=event.timestamp + task.duration,
                        task_key=task.task_key,
                        generation=task.generation,
                    )

            elif event.event_type == EventType.TASK_FINISH:
                # 工序完工，释放工人与后继
                task = self.state.tasks[event.task_key]
                if task.status == TaskStatus.RUNNING:
                    task.complete_work(current_time=event.timestamp)
                    self._on_task_completed(task)

            elif event.event_type == EventType.MATERIAL_ARRIVE:
                # 物料/可用性到达
                task = self.state.tasks[event.task_key]
                self._check_and_update_task_readiness(task)

            elif event.event_type == EventType.SYNCHRONOUS_TRANSFER:
                # 全线脉动转站 (Task 2.5 完整对接)
                self._execute_synchronous_transfer(event.timestamp)

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
        # 必须在当前站位停靠
        if ac.current_station != task.current_station:
            return

        # 检查所有前驱是否均已 COMPLETED
        for pred_id in task.predecessors:
            pred_key = f"{task.aircraft_id}_{pred_id}"
            pred_task = self.state.tasks.get(pred_key)
            if pred_task is not None and pred_task.status != TaskStatus.COMPLETED:
                return

        # 检查到料可用性时刻
        if self.state.current_time >= task.material_ready_time - self.tolerance:
            task.status = TaskStatus.READY
        else:
            # 安排物料到达事件
            self.event_queue.push(
                event_type=EventType.MATERIAL_ARRIVE,
                timestamp=task.material_ready_time,
                task_key=task.task_key,
                generation=task.generation,
            )

    def _refresh_aircraft_station_readiness(self, aircraft_id: int, station_id: int) -> None:
        """飞机进驻新站位时，扫描并激活该站位归属工序。"""
        for task in self.state.tasks.values():
            if task.aircraft_id == aircraft_id and task.current_station == station_id:
                if task.status in (TaskStatus.UNREADY, TaskStatus.POSTPONED):
                    task.status = TaskStatus.UNREADY
                    self._check_and_update_task_readiness(task)

    def _execute_synchronous_transfer(self, timestamp: float) -> None:
        """执行全线同步脉动流转（各机站位+1，唤醒后移任务）。"""
        self.state.last_transfer_time = float(timestamp)
        self.state.transfer_history.append(float(timestamp))
        self.state.current_cycle += 1

        for k, ac in self.state.aircraft.items():
            if ac.current_station >= 0 and not ac.is_completed:
                ac.step_to_next_station(timestamp)
                if ac.is_in_factory:
                    self._refresh_aircraft_station_readiness(k, ac.current_station)

        # 检查下一架待进线飞机
        next_k = self.state.current_cycle - 1
        if 0 <= next_k < self.state.num_aircraft:
            ac_next = self.state.aircraft[next_k]
            if ac_next.current_station == -1:
                ac_next.current_station = 0
                ac_next.entry_times[0] = float(timestamp)
                self._refresh_aircraft_station_readiness(next_k, 0)

    def _check_terminated(self) -> bool:
        """检查生产是否自然终止（全部飞机离开末站，所有工序完工）。"""
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
