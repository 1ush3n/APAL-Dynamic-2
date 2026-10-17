"""工作三真实综合目标独立计算器与终局账本评估器。

按默认 ``normalize_by_n=True``：

* ``J_H = sum(max(H_q - H_0, 0) / H_0)``，每次同步转站结算一次；
* ``D_T = sum(abs(b_i - b_i^0) / H_0) / N``；
* ``D_W = sum(1 - |W_i ∩ W_i^0| / m_i) / N``；
* ``G = sum_i(lambda_1*n_i + lambda_2*max(n_i - 1, 0))``，不除以 N；
* ``R_T``、``R_W``分别为相邻正式版本时间位置差和团队替换率之和，按配置选择是否除以 N；
* ``J_total = w_h*J_H + w_t*D_T + w_w*D_W + w_p*G
  + w_RT*R_T + w_RW*R_W``。

未配置 ``w_RT``、``w_RW`` 时分别沿用 ``w_t``、``w_w``。最终站位偏差只作报告指标，
不在当前奖励中添加独立站位权重。本模块独立于单步奖励实现，用于核对奖励差分账本。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
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
        normalize_by_n: 是否将 D_T、D_W、R_T、R_W 除以批次开始时固定的工序数 N (默认 True).
    """

    w_h: float = 1.0
    w_t: float = 0.20
    w_w: float = 0.05
    w_p: float = 1.0
    lambda_1: float = 0.10
    lambda_2: float = 0.20
    normalize_by_n: bool = True
    # 未锁定时沿用已有时间/团队目标权重；正式实验可显式覆盖。
    w_revision_time: float | None = None
    w_revision_team: float | None = None


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """终局指标和费用明细。

    ``j_takt``、``d_time``、``d_team``、``j_postpone``、``revision_time``、
    ``revision_team``均为尚未乘目标权重的指标；``j_revision``是已加权费用。
    ``j_total``是六项目标的加权和。
    """

    j_takt: float              # 节拍超期综合项 J_takt
    d_time: float              # 周期内开工位置偏差项 D_T (已应用配置归一化，未乘权重)
    d_team: float              # 团队替换率项 D_W (已应用配置归一化，未乘权重)
    j_postpone: float          # 累计后移改派惩罚 J_postpone
    j_total: float             # 加权综合总成本 J_total

    # 未加权的原始物理统计量（便于科研对比与报表输出）
    takt_violation_hours: float  # 累计节拍超期总工时 (小时)
    num_transfers: int           # 实际转站次数 Q
    total_postpone_count: int    # 全线工序累计后移次数总和
    num_postponed_tasks: int     # 发生过至少一次后移的独立工序数
    num_started_tasks: int       # 实际已确认开工工序数
    total_tasks: int             # 批次开始时固定的工序数 N
    revision_time: float         # 相邻正式安排时间位置变化（已归一化）
    revision_team: float         # 相邻正式安排团队变化（已归一化）
    j_revision: float            # 时间/团队相邻修订加权费用
    num_revisions: int           # 正式修订次数

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def calculate_postpone_penalty(n: int, lambda_1: float = 0.10, lambda_2: float = 0.20) -> float:
    """计算单个工序改派 n 次的递增惩罚 g(n) = lambda_1 * n + lambda_2 * max(0, n - 1)。"""
    if n <= 0:
        return 0.0
    return float(lambda_1 * n + lambda_2 * max(0, n - 1))


def build_objective_report(
    breakdown: ObjectiveBreakdown,
    weights: ObjectiveWeights,
    *,
    h0: float,
) -> dict[str, Any]:
    """将未加权指标、加权费用和归一化元数据分开序列化。"""
    if not math.isfinite(h0) or h0 <= 0.0:
        raise ValueError("H0必须是有限正数")

    revision_time_weight = (
        weights.w_t if weights.w_revision_time is None else weights.w_revision_time
    )
    revision_team_weight = (
        weights.w_w if weights.w_revision_team is None else weights.w_revision_team
    )
    metrics_unweighted = {
        "j_h": float(breakdown.j_takt),
        "d_t": float(breakdown.d_time),
        "d_w": float(breakdown.d_team),
        "g_postpone": float(breakdown.j_postpone),
        "r_t": float(breakdown.revision_time),
        "r_w": float(breakdown.revision_team),
    }
    cost_components_weighted = {
        "j_takt": weights.w_h * metrics_unweighted["j_h"],
        "d_time": weights.w_t * metrics_unweighted["d_t"],
        "d_team": weights.w_w * metrics_unweighted["d_w"],
        "j_postpone": weights.w_p * metrics_unweighted["g_postpone"],
        "j_revision": (
            revision_time_weight * metrics_unweighted["r_t"]
            + revision_team_weight * metrics_unweighted["r_w"]
        ),
    }
    component_total = sum(cost_components_weighted.values())
    if not math.isclose(
        component_total,
        breakdown.j_total,
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ValueError(
            "ObjectiveBreakdown.j_total与加权分项之和不一致: "
            f"{breakdown.j_total} != {component_total}"
        )

    return {
        "metrics_unweighted": metrics_unweighted,
        "cost_components_weighted": cost_components_weighted,
        "revision_weighted_components": {
            "time": revision_time_weight * metrics_unweighted["r_t"],
            "team": revision_team_weight * metrics_unweighted["r_w"],
        },
        "objective_weights": asdict(weights),
        "metric_coverage": {
            "status": (
                "all_tasks_started"
                if breakdown.num_started_tasks == breakdown.total_tasks
                else "partial"
            ),
            "all_tasks_started": (
                breakdown.num_started_tasks == breakdown.total_tasks
            ),
            "started_task_count": int(breakdown.num_started_tasks),
            "fixed_task_count": int(breakdown.total_tasks),
        },
        "normalization": {
            "h0_hours": float(h0),
            "normalize_by_n": bool(weights.normalize_by_n),
            "fixed_task_count": int(breakdown.total_tasks),
            "metric_rules": {
                "j_h": "每次全线同步转站超期小时除以H0后求和",
                "d_t": "周期内开工偏差除以H0；normalize_by_n时再除以固定N",
                "d_w": "团队替换比例求和；normalize_by_n时再除以固定N",
                "g_postpone": "累计g(n)，不除以N",
                "r_t": "相邻正式版本时间位置差除以H0；normalize_by_n时再除以固定N",
                "r_w": "相邻正式版本团队替换比例求和；normalize_by_n时再除以固定N",
            },
        },
        "j_total": float(breakdown.j_total),
    }


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
    total_tasks = int(
        getattr(
            state,
            "normalization_task_count",
            getattr(state_or_env, "total_tasks", len(state.tasks)),
        )
    )
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
        cycle_start_time = getattr(task, "cycle_start_time", None)
        if cycle_start_time is not None:
            p_cycle_start = cycle_start_time
        else:
            # 回退保护：从 transfer_history 中查找最后一个 <= S_{ki} 的转站时刻
            p_cycle_start = 0.0
            for p in transfer_history:
                if p <= s_ki + 1e-5:
                    p_cycle_start = p
                else:
                    break

        b_ki = s_ki - p_cycle_start
        b_i0 = getattr(task, "in_station_offset", 0.0)
        d_time_sum += abs(b_ki - b_i0) / h0

        # 团队替换率 d(W, W^0) = 1 - |W cap W^0| / m_i
        w_actual = set(getattr(task, "assigned_team", ()))
        w_base = set(getattr(task, "base_team", ()))
        demand = getattr(task, "demand", 0)
        if demand > 0:
            overlap = len(w_actual & w_base)
            d_team_sum += 1.0 - (overlap / demand)

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

    # 4. 相邻正式安排修订费用。站位变更的累计次数费用已由 j_postpone 计入，
    # 这里仅统计相邻安排的时间位置与团队变化，避免重复收取同一份改站代价。
    revision_time_sum = 0.0
    revision_team_sum = 0.0
    num_revisions = 0
    for task in state.tasks.values():
        for revision in getattr(task, "revision_history", ()):
            revision_time_sum += float(revision.get("time_change", 0.0))
            revision_team_sum += float(revision.get("team_change", 0.0))
            num_revisions += 1

    revision_time = revision_time_sum * scale
    revision_team = revision_team_sum * scale
    revision_time_weight = (
        weights.w_t if weights.w_revision_time is None else weights.w_revision_time
    )
    revision_team_weight = (
        weights.w_w if weights.w_revision_team is None else weights.w_revision_team
    )
    j_revision = revision_time_weight * revision_time + revision_team_weight * revision_team

    # 5. 加权综合总成本；修订权重缺省分别继承w_t与w_w。
    j_total = (
        weights.w_h * j_takt
        + weights.w_t * d_time
        + weights.w_w * d_team
        + weights.w_p * j_postpone
        + j_revision
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
        revision_time=revision_time,
        revision_team=revision_team,
        j_revision=j_revision,
        num_revisions=num_revisions,
    )
