"""工作三多架次飞机脉动装配协同调度核心仿真环境 (AirLineEnvWork3)。

严格遵循顶层设计与 Ponytail 准则（标准脉动控速转站，易错点全闭环）：
- 维护多架次全局状态与离散事件队列；
- 支持自回归动作分支 A（留在当前站：指派团队、二元对齐、立即开工或未来预约）；
- 支持自回归动作分支 B（合法后移至下一站：动作当场截断，保留恢复时间 R）；
- 支持全线同步脉动判定与推进 (Task 2.5)：各站放行后严格按脉动节拍 H0 步进；
- 支持生产自然终止 (Task 2.6)：全部 10 架飞机离线出站且 2,830 道工序完成才终止。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

from envs.work3.core_types import (
    ActionBranch,
    AircraftRuntimeState,
    MultiAircraftState,
    TaskRuntimeState,
    TaskStatus,
    initialize_multi_aircraft_state,
)
from envs.work3.event_queue import DiscreteEventQueue, EventType, SimulationEvent
from utils.work3.objective_evaluator import ObjectiveWeights, calculate_postpone_penalty

logger = logging.getLogger(__name__)


class AirLineEnvWork3:
    """多架次飞机脉动装配协同调度环境。"""

    def __init__(
        self,
        baseline_json_path: str = "data/work3/real_283_k10_baseline.json",
        max_slots_per_station: int = 3,
        tolerance: float = 1e-5,
        weights: ObjectiveWeights | None = None,
        max_steps_per_rollout: int | None = None,
    ) -> None:
        self.baseline_json_path = baseline_json_path
        self.max_slots_per_station = int(max_slots_per_station)
        self.tolerance = float(tolerance)
        self.weights: ObjectiveWeights = weights if weights is not None else ObjectiveWeights()
        self.max_steps_per_rollout: int | None = max_steps_per_rollout
        self.step_count: int = 0

        self.state: MultiAircraftState = initialize_multi_aircraft_state(self.baseline_json_path)
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
        self.event_queue.reset(start_time=0.0)
        self._transfer_scheduled_for_cycle = 0
        self.total_tasks = len(self.state.tasks)
        self._station_occupied_tasks = {s: set() for s in range(self.state.num_stations)}

        self.cumulative_cost = 0.0
        self.cost_takt = 0.0
        self.cost_time = 0.0
        self.cost_team = 0.0
        self.cost_postpone = 0.0
        self.step_rewards.clear()
        self.step_count = 0

        # 0 号飞机在时刻 0.0 进入 0 号站位
        ac0 = self.state.aircraft[0]
        ac0.current_station = 0
        ac0.entry_times[0] = 0.0

        # 解锁 0 号飞机在 0 号站位首批无前驱且物料就绪的工序
        self._refresh_aircraft_station_readiness(aircraft_id=0, station_id=0)

        return self._get_observation()

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

    def get_ready_tasks(self) -> list[TaskRuntimeState]:
        """获取全线所有在场站位中处于 READY 状态的全部工序。"""
        ready: list[TaskRuntimeState] = []
        for s in range(self.state.num_stations):
            ready.extend(self.state.get_ready_tasks_for_station(s))
        return ready

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """执行单步调度动作。"""
        cost_before = self.cumulative_cost
        task_key = str(action["task_key"])
        branch = ActionBranch(action.get("branch", ActionBranch.STATION_EXECUTE))

        if task_key not in self.state.tasks:
            raise KeyError(f"未找到工序: {task_key}")

        task = self.state.tasks[task_key]
        if task.status != TaskStatus.READY:
            raise ValueError(f"工序 {task_key} 当前状态为 {task.status.name}，不可调度！必须为 READY。")

        ac = self.state.aircraft[task.aircraft_id]
        if ac.current_station != task.current_station:
            raise ValueError(
                f"工序 {task_key} 归属站位 {task.current_station}，但飞机 {task.aircraft_id} "
                f"当前物理停靠在站位 {ac.current_station}，不可作业！"
            )

        info: dict[str, Any] = {"action_branch": branch.name, "task_key": task_key}

        if branch == ActionBranch.STATION_EXECUTE:
            # 分支 A：留在当前站位执行
            team = tuple(int(w) for w in action.get("team", ()))
            align = int(action.get("align", 0))

            self._validate_team_for_task(task, team)

            search_start = max(self.state.current_time, task.material_ready_time)
            if align == 1:
                align_target = self.state.last_transfer_time + task.in_station_offset
                search_start = max(search_start, align_target)

            t_sched = self._find_team_earliest_slot(
                station_id=task.current_station,
                team=team,
                search_start=search_start,
                duration=task.duration,
            )

            for w in team:
                self.state.workers[w].add_interval(
                    start=t_sched,
                    end=t_sched + task.duration,
                    task_key=task.task_key,
                )

            if abs(t_sched - self.state.current_time) <= self.tolerance:
                task.assigned_team = list(team)
                task.scheduled_start = t_sched
                self._on_task_started(task, t_sched)
                self.event_queue.push(
                    event_type=EventType.TASK_FINISH,
                    timestamp=t_sched + task.duration,
                    task_key=task.task_key,
                    generation=task.generation,
                )
                self._station_occupied_tasks[task.current_station].add(task.task_key)
                info["scheduled_status"] = "RUNNING"
                info["scheduled_start"] = t_sched
            else:
                task.reserve(team=team, scheduled_start=t_sched)
                self.event_queue.push(
                    event_type=EventType.TASK_START,
                    timestamp=t_sched,
                    task_key=task.task_key,
                    generation=task.generation,
                )
                self._station_occupied_tasks[task.current_station].add(task.task_key)
                info["scheduled_status"] = "RESERVED"
                info["scheduled_start"] = t_sched

        elif branch == ActionBranch.POSTPONE:
            # 分支 B：合法后移至下一站位
            if task.current_station >= self.state.num_stations - 1:
                raise ValueError(f"末站（站位 {task.current_station + 1}）工序绝对禁止后移！")

            n_old = task.postpone_count
            task.postpone_to_next_station()
            n_new = task.postpone_count
            penalty_delta = calculate_postpone_penalty(
                n_new, self.weights.lambda_1, self.weights.lambda_2
            ) - calculate_postpone_penalty(
                n_old, self.weights.lambda_1, self.weights.lambda_2
            )
            cost_postpone_inc = self.weights.w_p * penalty_delta
            self.cost_postpone += cost_postpone_inc
            self.cumulative_cost += cost_postpone_inc

            info["postponed_to_station"] = task.current_station
            info["postpone_count"] = task.postpone_count
            info["cost_postpone_inc"] = cost_postpone_inc

            # 后移可能使得当前周期站位放行条件满足，检查是否可安排转站
            self._check_and_schedule_transfer()

        # 若当前现场无可用调度动作，自动推进离散事件
        if len(self.get_ready_tasks()) == 0:
            self._advance_events_until_next_decision()

        self.step_count += 1
        terminated = self._check_terminated()
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
            next_free = candidate
            for w in team:
                wc = self.state.workers[w]
                slot = wc.find_earliest_slot(candidate, duration, tolerance=self.tolerance)
                if slot > next_free:
                    next_free = slot

            if next_free > candidate + self.tolerance:
                candidate = next_free
                continue

            if not self._is_station_slot_available(station_id, candidate, candidate + duration):
                candidate += 0.5
                continue

            return candidate

        return candidate

    def _is_station_slot_available(self, station_id: int, start: float, end: float) -> bool:
        """检查指定站位在 [start, end] 区间内槽位并发数是否 < max_slots_per_station (O(K) 局部查找)。"""
        intervals: list[tuple[float, float]] = []
        for task_key in self._station_occupied_tasks.get(station_id, ()):
            t = self.state.tasks[task_key]
            if t.scheduled_start is not None:
                intervals.append((t.scheduled_start, t.scheduled_start + t.duration))

        time_points = [start, (start + end) / 2.0, end - self.tolerance]
        for pt in time_points:
            active_count = sum(1 for s_ex, e_ex in intervals if s_ex <= pt < e_ex - self.tolerance)
            if active_count >= self.max_slots_per_station:
                return False

        return True

    def _advance_events_until_next_decision(self) -> None:
        """推进事件队列，直至产生新的就绪工序或全线完工退出。"""
        while len(self.get_ready_tasks()) == 0:
            # 首先检查是否满足全线同步脉动放行条件
            self._check_and_schedule_transfer()

            if self.event_queue.is_empty():
                # 既无就绪动作也无未来事件，退出循环
                break

            event = self.event_queue.pop()
            if event is None:
                break

            self.state.current_time = event.timestamp

            if event.event_type == EventType.TASK_START:
                task = self.state.tasks[event.task_key]
                if task.status == TaskStatus.RESERVED:
                    self._on_task_started(task, event.timestamp)
                    self.event_queue.push(
                        event_type=EventType.TASK_FINISH,
                        timestamp=event.timestamp + task.duration,
                        task_key=task.task_key,
                        generation=task.generation,
                    )

            elif event.event_type == EventType.TASK_FINISH:
                task = self.state.tasks[event.task_key]
                if task.status == TaskStatus.RUNNING:
                    task.complete_work(current_time=event.timestamp)
                    self._station_occupied_tasks[task.current_station].discard(task.task_key)
                    self._on_task_completed(task)
                    # 完工后检查是否解锁全线脉动转站
                    self._check_and_schedule_transfer()

            elif event.event_type == EventType.MATERIAL_ARRIVE:
                task = self.state.tasks[event.task_key]
                self._check_and_update_task_readiness(task)

            elif event.event_type == EventType.SYNCHRONOUS_TRANSFER:
                self._execute_synchronous_transfer(event.timestamp)

            elif event.event_type == EventType.DISTURBANCE:
                self._handle_disturbance_event(event.timestamp, event.payload)

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

            # 统一直接建模工序开工可用性延迟：r_{k*,i}^{new} = max(r_{k*,i}^{old}, R)
            task.material_ready_time = max(task.material_ready_time, recovery_time)

            if task.status in (TaskStatus.RUNNING, TaskStatus.COMPLETED):
                # 物理规则 3：实际已开工与已完工作业硬冻结，绝不强制打断
                continue

            elif task.status == TaskStatus.RESERVED:
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
        """检查全线站位是否满足放行转站要求，若满足且未安排过本周期转站，则安排 SYNCHRONOUS_TRANSFER。

        严格落实用户确认的规范：
        采用选项 A，严格等待各站工序最晚完工时间与名义节拍 H0，按节拍脉动转站。
        """
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

        # 选项 A：转站时刻 P_q = max(P_{q-1} + H_0, max_s F_{s,q}, current_time)
        nominal_takt_end = self.state.last_transfer_time + self.state.h0
        latest_finish = self.state.current_time

        for s in range(self.state.num_stations):
            for t in self.state.get_tasks_for_station(s):
                if t.status == TaskStatus.COMPLETED:
                    if t.actual_end is not None and t.actual_end > latest_finish:
                        latest_finish = t.actual_end

        transfer_time = max(nominal_takt_end, latest_finish, self.state.current_time)

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
