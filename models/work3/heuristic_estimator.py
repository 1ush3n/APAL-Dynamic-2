"""工作三 周期启发式完工时间估计器 (Task 5.2)。

核心数学模型与技术规范：
依据《方法设计确认稿》第 7.1 节与物理完工下界增强设计：
对当前周期 q (转站起点 P_{q-1} = last_transfer_time)，基于当前时刻 t、各站实际在制作业完工、
已知延误工序最早完工下界以及剩余总工时，输出无学习的静态基准估计完成时刻 P_q^h(s)。

站位完工下界：
  F_s = max(
      max_{u in Running(s)} (u.start + u.duration),
      max_{i in Delayed(s)} (r_i + d_i),
      t + W_remain(s) / M_s
  )
全线同步脉动预计完成时刻：
  P_q^h = max(t, max_{s=0}^4 F_s)

物理法则：
1. W_remain(s) 严格仅统计本周期归属本站未后移工序的标准工时总和 (已后移工序进入后续周期，不混入本站)；
2. M_s 为本站初始内生绑定的固定工人数；
3. 保证估计值不早于当前时刻；H_0只作为归一化和基准参照。
"""

from __future__ import annotations

from envs.work3.core_types import MultiAircraftState, TaskRuntimeState, TaskStatus


def compute_station_estimated_finish(
    state: MultiAircraftState,
    station_id: int,
    h0: float,
) -> float:
    """计算单个装配工位 s 在当前周期的启发式最早完工时间下界 F_s。"""
    current_time = float(state.current_time)
    ac_id = state.get_aircraft_at_station(station_id)

    # 若该工位无在场飞机，完工下界退化为当前时刻 t
    if ac_id is None:
        return current_time

    # 收集归属本站且未完工的活动工序
    st_tasks = [
        t for t in state.tasks.values()
        if t.aircraft_id == ac_id and t.current_station == station_id and t.status != TaskStatus.COMPLETED
    ]

    if not st_tasks:
        return current_time

    def estimated_duration(task: TaskRuntimeState) -> float:
        return float(
            task.execution_duration
            if task.execution_duration is not None
            else task.duration
        )

    # 1. 正在执行任务 (RUNNING) 的确定性完工时刻下界
    running_finishes = [
        float(t.actual_start + estimated_duration(t))
        for t in st_tasks
        if t.status == TaskStatus.RUNNING and t.actual_start is not None
    ]
    f_running = max(running_finishes) if running_finishes else current_time

    # 2. 已知延误到料工序 (DelayedTasks) 的最早完工时刻下界：r_{ki} + d_{ki}
    delayed_finishes = [
        float(t.material_ready_time + t.duration)
        for t in st_tasks
        if t.material_ready_time > current_time
    ]
    f_delayed = max(delayed_finishes) if delayed_finishes else current_time

    # 3. 站内未排工序工作量与工人供给下界：t + W_remain / M_s
    # 已正式后移 (POSTPONED) 的工序当前站位已被置为 s+1，自然不出现在 st_tasks 中
    w_remain = sum(
        max(0.0, t.actual_start + estimated_duration(t) - current_time)
        if t.status == TaskStatus.RUNNING and t.actual_start is not None
        else estimated_duration(t)
        for t in st_tasks
    )
    num_workers = len(state.station_worker_bindings.get(station_id, []))
    m_s = max(1, num_workers)
    f_workload = current_time + (w_remain / m_s)

    # 综合取最大值作为工位完工下界
    return max(current_time, f_running, f_delayed, f_workload)


def compute_cycle_heuristic_cmax(state: MultiAircraftState) -> float:
    """计算当前周期全线同步脉动完成时刻的启发式估计值 P_q^h(s)。"""
    current_time = float(state.current_time)
    # 统计五大工位的各自完工下界
    station_finishes = [
        compute_station_estimated_finish(state, s, state.h0)
        for s in range(state.num_stations)
    ]
    max_station_finish = max(station_finishes) if station_finishes else current_time

    # 全线同步脉动转站时刻取全站瓶颈与当前时刻的最大值。
    estimated_pq = max(current_time, max_station_finish)

    # 确保绝不发生时间倒退
    return max(current_time, estimated_pq)
