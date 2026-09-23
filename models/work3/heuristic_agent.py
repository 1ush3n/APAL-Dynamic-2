"""工作三 基线 C 启发式调度智能体 (Task 5.3)。

核心调度机制：
1. 优先在当前站位按工艺工序流推进就绪工序 (Branch A: STATION_EXECUTE)；
2. 团队选派：优先选派基准指派团队 W_i^0，或从本站绑定的空闲资质工人中优选团队，并执行基准二元对齐 (align=1)；
3. 超期后移保节拍规则 (用户已确认的物理准则)：
   若工序遭遇突发严重缺料且恢复时刻 R > P_{q-1} + H_0 (留在本站必将直接导致全线转站超期)，
   且该工序处于非末站 (站位 0~3)，则果断触发分支 B (POSTPONE) 合法后移至下一站，保住全线准时脉动转站；
   若为末站 (不可后移) 或物料在当前节拍内就绪，则留在本站排产。
4. 提供 run_trajectory() 全流程自主运行器，记录每步状态与时间估计。
"""

from __future__ import annotations

from typing import Any, Callable

from envs.work3.core_types import ActionBranch, TaskRuntimeState, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax


class HeuristicAgentWork3:
    """工作三基线 C 启发式调度智能体。"""

    def __init__(self, name: str = "Baseline-C") -> None:
        self.name = name

    def select_action(self, env: AirLineEnvWork3) -> dict[str, Any] | None:
        """根据当前产线状态，依据基准启发式规则选出下一步调度动作。"""
        candidate_tasks = env.get_action_candidates()
        if not candidate_tasks:
            return None

        non_reserved_tasks = [
            task for task in candidate_tasks if task.status != TaskStatus.RESERVED
        ]
        if not non_reserved_tasks:
            return {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
        candidate_tasks = non_reserved_tasks

        state = env.state
        h0 = float(state.h0)
        p_last = float(state.last_transfer_time)
        nominal_cycle_end = p_last + h0

        # 按站位编号、站内基准偏移与工序编号确定性排序
        reservable_tasks = [task for task in candidate_tasks if env.can_reserve(task)]
        candidate_tasks = reservable_tasks or candidate_tasks
        candidate_tasks.sort(key=lambda t: (t.current_station, t.in_station_offset, t.task_id))
        task: TaskRuntimeState = candidate_tasks[0]

        # ------------------
        # 1. 检查物理后移保节拍准则 (Branch B: POSTPONE)
        # ------------------
        # 条件：物料延迟导致在当前名义节拍内无法就绪 (R > P_{q-1} + H_0)，且非末站工序 (站位 0~3)
        is_delayed_beyond_cycle = task.material_ready_time > nominal_cycle_end + 1e-4
        can_postpone = env.validate_postpone(task) is None

        if is_delayed_beyond_cycle and can_postpone:
            return {
                "task_key": task.task_key,
                "branch": ActionBranch.POSTPONE,
            }

        # ------------------
        # 2. 留在当前站排产 (Branch A: STATION_EXECUTE)
        # ------------------
        if not env.can_reserve(task):
            # 前驱尚未完成或已知物料尚未到达时，先推进到下一个事件；
            # 只有上面的超期规则才主动后移，避免无扰动轨迹凭空增加改站。
            return {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
        st_workers = state.station_worker_bindings.get(task.current_station, [])
        demand = task.demand
        valid_workers = env.valid_team_completion_workers(task, [])

        # 优先使用基准固定指派团队
        chosen_team = list(task.base_team) if task.base_team else []
        if len(chosen_team) != demand or any(worker_id not in valid_workers for worker_id in chosen_team):
            # 兜底：按站位绑定工人顺延指派
            chosen_team = valid_workers[:demand]

        # 尝试检查是否有本站完全空闲的合格工人，若有则优先使用空闲工人
        free_workers = [
            w for w in st_workers
            if w in valid_workers
            and state.workers[w].is_available(state.current_time, state.current_time + 1e-5)
        ]
        if len(free_workers) >= demand and any(
            not state.workers[w].is_available(state.current_time, state.current_time + 1e-5)
            for w in chosen_team
        ):
            # 若基准团队中有忙碌工人，但站内有足够空闲工人，优先替换为空闲工人以避免等待
            chosen_team = free_workers[:demand]

        return {
            "task_key": task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": tuple(chosen_team),
            "align": 1,
        }

    def run_trajectory(
        self,
        env: AirLineEnvWork3,
        scenario: dict[str, Any] | None = None,
        max_decisions: int = 10000,
        step_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """运行一条完整生产轨迹直至自然终止或达到上限。

        Args:
            env: 工作三仿真环境实例；
            scenario: 可选的扰动场景字典；
            max_decisions: 单轨迹最大决策步数上限；
            step_callback: 每步回调函数，传入当前 step 元数据。

        Returns:
            轨迹结算汇总字典。
        """
        env.reset()
        if scenario is not None:
            env.load_scenario(scenario)

        total_decisions = 0
        step_records: list[dict[str, Any]] = []
        termination_reason = "decision_limit"

        while total_decisions < max_decisions:
            candidates = env.get_action_candidates()
            if not candidates:
                if env._check_terminated():
                    termination_reason = "completed"
                    break
                env._advance_events_until_next_decision()
                candidates = env.get_action_candidates()
                if not candidates and env._check_terminated():
                    termination_reason = "completed"
                    break
                if not candidates and env.event_queue.is_empty():
                    termination_reason = "deadlock"
                    break

            # 记录当前状态与启发式估计值
            current_cycle = env.state.current_cycle
            est_cmax = compute_cycle_heuristic_cmax(env.state)

            action = self.select_action(env)
            if action is None:
                continue

            # 抓取当前决策步数据快照
            step_record = {
                "step_idx": total_decisions,
                "cycle_idx": current_cycle,
                "current_time": float(env.state.current_time),
                "estimated_cmax": float(est_cmax),
                "action": dict(action),
            }

            obs, reward, terminated, truncated, info = env.step(action)
            total_decisions += 1

            if step_callback is not None:
                step_callback(step_record)
            step_records.append(step_record)

            if terminated:
                termination_reason = str(
                    info.get(
                        "termination_reason",
                        "completed" if env._check_terminated() else "deadlock",
                    )
                )
                break

        # 补充记录各周期最终实际达成的同步脉动转站时刻 P_q
        # transfer_history 中记录了转站时刻：索引 q-1 对应第 q 周期转站时刻
        transfer_history = list(env.state.transfer_history)
        for rec in step_records:
            c_idx = int(rec["cycle_idx"])
            if 1 <= c_idx <= len(transfer_history):
                actual_time = float(transfer_history[c_idx - 1])
                rec["actual_transfer_time"] = actual_time
                rec["label_y"] = float(
                    (actual_time - rec["estimated_cmax"]) / env.state.h0
                )
                rec["label_available"] = True
            else:
                rec["actual_transfer_time"] = None
                rec["label_y"] = None
                rec["label_available"] = False

        success = termination_reason == "completed" and env._check_terminated()

        return {
            "success": success,
            "termination_reason": termination_reason,
            "total_decisions": total_decisions,
            "makespan": float(env.state.current_time),
            "completed_tasks": sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.COMPLETED),
            "transfer_count": len(transfer_history),
            "cost_takt": float(env.cost_takt),
            "cost_time": float(env.cost_time),
            "cost_team": float(env.cost_team),
            "cost_postpone": float(env.cost_postpone),
            "cost_revision": float(env.cost_revision),
            "cumulative_cost": float(env.cumulative_cost),
            "step_records": step_records,
        }
