"""工作三真实综合目标独立计算器与终局账本评估器 (Task 3.1 / 里程碑 M2)。

严格落实顶层设计规范与《工作三_工程落地与逐步验证任务看板.md》：
1. 节拍超期：J_takt = sum_{q=1}^Q [H_q - H_0]_+ / H_0；
2. 周期内开工位置偏差：D_time = sum_{k,i} |(S_{ki} - P_{q-1}) - b_i^0| / H_0 （支持 / N 归一化）；
3. 团队替换率：D_team = (1/N) * sum_{k,i} (1 - |W_{ki} cap W_i^0| / m_i)；
4. 改派惩罚：J_postpone = sum_{k,i} g(n_{ki}) = sum_{k,i} (lambda_1 n + lambda_2 [n - 1]_+)；
5. 综合总成本：J_total = w_h * J_takt + w_t * D_time + w_w * D_team + w_p * J_postpone。

本模块完全独立于单步强化学习奖励计算，作为权威终局账本，用于验证奖励差分闭环代数恒等性。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from envs.work3.core_types import MultiAircraftState, TaskRuntimeState, TaskStatus


@dataclass(frozen=True)
class ObjectiveWeights:
    """综合调度目标与奖励权重配置参数。

    Attributes:
        w_h: 节拍超期主项惩罚权重 (默认 1.0).
        w_t: 周期内开工位置偏差权重 (默认 0.20).
        w_w: 团队替换稳定性权重 (默认 0.05).
        w_p: 改派后移综合惩罚乘数 (默认 1.0).
        lambda_1: 首次改派基准惩罚 (默认 0.10).
        lambda_2: 二次及以上累计改派累进加罚 (默认 0.20).
        normalize_by_n: 是否将 D_time 和 D_team 除以固定全线工序总数 N=2830 (默认 True).
    """

    w_h: float = 1.0
    w_t: float = 0.20
    w_w: float = 0.05
    w_p: float = 1.0
    lambda_1: float = 0.10
    lambda_2: float = 0.20
    normalize_by_n: bool = True


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """终局综合目标各分项明细。"""

    j_takt: float              # 节拍超期综合项 J_takt
    d_time: float              # 周期内开工位置偏差项 D_time (已应用归一化比例)
    d_team: float              # 团队替换率项 D_team (已应用归一化比例)
    j_postpone: float          # 累计后移改派惩罚 J_postpone
    j_total: float             # 加权综合总成本 J_total

    # 未加权的原始物理统计量（便于科研对比与报表输出）
    takt_violation_hours: float  # 累计节拍超期总工时 (小时)
    num_transfers: int           # 实际转站次数 Q
    total_postpone_count: int    # 全线工序累计后移次数总和
    num_postponed_tasks: int     # 发生过至少一次后移的独立工序数
    num_started_tasks: int       # 实际已确认开工工序数
    total_tasks: int             # 总工序基准规模 N (默认 2830)

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def calculate_postpone_penalty(n: int, lambda_1: float = 0.10, lambda_2: float = 0.20) -> float:
    """计算单个工序改派 n 次的递增惩罚 g(n) = lambda_1 * n + lambda_2 * max(0, n - 1)。"""
    if n <= 0:
        return 0.0
    return float(lambda_1 * n + lambda_2 * max(0, n - 1))


def evaluate_trajectory_objective(
    state_or_env: Any,
    weights: ObjectiveWeights | None = None,
) -> ObjectiveBreakdown:
    """从多架次仿真状态独立计算轨迹终局综合目标与各分项明细。

    Args:
        state_or_env: MultiAircraftState 实例或带有 .state 属性的 AirLineEnvWork3 实例。
        weights: 目标函数权重配置，默认使用标准基准参数。

    Returns:
        ObjectiveBreakdown 结构体，包含各项指标数值。
    """
    if weights is None:
        weights = ObjectiveWeights()

    state: MultiAircraftState = getattr(state_or_env, "state", state_or_env)
    h0 = float(state.h0)
    total_tasks = len(state.tasks)
    scale = (1.0 / total_tasks) if (weights.normalize_by_n and total_tasks > 0) else 1.0

    # 1. 节拍超期 J_takt 计算
    # P_0 = 0.0, P_1, ..., P_Q 记录在 state.transfer_history 中
    transfer_history = state.transfer_history
    takt_violation_hours = 0.0
    j_takt = 0.0
    last_p = 0.0

    for p in transfer_history:
        duration_h = p - last_p
        overdue_h = max(0.0, duration_h - h0)
        takt_violation_hours += overdue_h
        j_takt += overdue_h / h0
        last_p = p

    # 2. 周期内开工位置偏差 D_time 与团队替换率 D_team 计算
    d_time_sum = 0.0
    d_team_sum = 0.0
    num_started = 0

    for task in state.tasks.values():
        if task.actual_start is None:
            continue

        num_started += 1
        # 实际开工时刻 S_{ki}
        s_ki = task.actual_start
        # 确定开工所在周期的转站基准点 P_{q-1}
        if task.cycle_start_time is not None:
            p_cycle_start = task.cycle_start_time
        else:
            # 回退保护：从 transfer_history 中查找最后一个 <= S_{ki} 的转站时刻
            p_cycle_start = 0.0
            for p in transfer_history:
                if p <= s_ki + 1e-5:
                    p_cycle_start = p
                else:
                    break

        b_ki = s_ki - p_cycle_start
        b_i0 = task.in_station_offset
        d_time_sum += abs(b_ki - b_i0) / h0

        # 团队替换率 d(W, W^0) = 1 - |W cap W^0| / m_i
        w_actual = set(task.assigned_team)
        w_base = set(task.base_team)
        if task.demand > 0:
            overlap = len(w_actual & w_base)
            d_team_sum += 1.0 - (overlap / task.demand)

    d_time = d_time_sum * scale
    d_team = d_team_sum * scale

    # 3. 改派惩罚 J_postpone 计算
    j_postpone = 0.0
    total_postpone_count = 0
    num_postponed_tasks = 0

    for task in state.tasks.values():
        n = task.postpone_count
        if n > 0:
            total_postpone_count += n
            num_postponed_tasks += 1
            j_postpone += calculate_postpone_penalty(
                n, lambda_1=weights.lambda_1, lambda_2=weights.lambda_2
            )

    # 4. 加权综合总成本 J_total
    j_total = (
        weights.w_h * j_takt
        + weights.w_t * d_time
        + weights.w_w * d_team
        + weights.w_p * j_postpone
    )

    return ObjectiveBreakdown(
        j_takt=j_takt,
        d_time=d_time,
        d_team=d_team,
        j_postpone=j_postpone,
        j_total=j_total,
        takt_violation_hours=takt_violation_hours,
        num_transfers=len(transfer_history),
        total_postpone_count=total_postpone_count,
        num_postponed_tasks=num_postponed_tasks,
        num_started_tasks=num_started,
        total_tasks=total_tasks,
    )
