"""工作三 动态势函数奖励塑形器 (Task 6.4)。

核心数学模型与技术规范：
依据《方法设计确认稿》第 8 节（势函数与密集奖励）：
1. 势函数数学定义：
   Φ(s) = -a · (H^_q(s) / H_0) - b · ([H^_q(s) - H_0]_+ / H_0),   a, b >= 0
   - a 项: 引导周期内作业平滑推进 (未超期时的细分时间信号)；
   - b 项: 重点惩罚预计超期脉动节拍；
   - H^_q(s) = P^_q(s) - P_{q-1} 为时间修正头预测的本周期总耗时。
2. 步步塑形奖励计算：
   r_n^step = r_n^actual + β · [ γ · Φ(s_{n+1}) - Φ(s_n) ]
3. 物理约束与策略不变性定理保障：
   - 生产轨迹内预测器版本固定 (Frozen Snapshot)：每条生产流水线固定一份只读预测器副本，
     禁止在线策略反向传播修改该轨迹的势场定义；
   - 终局势归零 (Terminal Zero Potential)：当全线生产批次实际全部完成时，Φ(s_terminal) = 0.0；
   - 脉动周期中间转站不归零：中间转站是连续物理演化，势函数自然延续，绝不中途截断；
   - 当 γ = 1 时，全 episode 塑形奖励之和严格满足伸缩求和定理：
     Σ r_n^step = Σ r_n^actual + β · [Φ(s_N) - Φ(s_0)] = Σ r_n^actual - β · Φ(s_0)。
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn

from models.work3.time_head import TimeResidualHead


class PotentialRewardShaper:
    """基于时间修正头预测节拍的动态势函数奖励塑形器。"""

    def __init__(
        self,
        time_head: TimeResidualHead,
        actor_critic: Any | None = None,
        a: float = 0.5,
        b: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.99,
    ) -> None:
        """初始化势函数塑形器。

        Args:
            time_head: 时间残差预测头模型
            a: 周期内推进引导系数 (a >= 0)
            b: 预计超期惩罚系数 (b >= 0)
            beta: 势函数差分奖励缩放权重 (beta >= 0)
            gamma: 折扣因子 (0 < gamma <= 1.0)
        """
        self.a = float(a)
        self.b = float(b)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.snapshot_version = 0

        # 保持只读冻结副本
        self.frozen_head = copy.deepcopy(time_head)
        self.frozen_actor = copy.deepcopy(actor_critic) if actor_critic is not None else None
        self.frozen_head.eval()
        for p in self.frozen_head.parameters():
            p.requires_grad = False
        if self.frozen_actor is not None:
            self.frozen_actor.eval()
            for p in self.frozen_actor.parameters():
                p.requires_grad = False

    def update_snapshot(self, new_head: TimeResidualHead, new_actor: Any | None = None) -> None:
        """在新的训练 episode/轨迹开始前，同步最新的在线预测头权重副本。"""
        self.frozen_head = copy.deepcopy(new_head)
        self.frozen_head.eval()
        for p in self.frozen_head.parameters():
            p.requires_grad = False
        if new_actor is not None:
            self.frozen_actor = copy.deepcopy(new_actor)
            self.frozen_actor.eval()
            for p in self.frozen_actor.parameters():
                p.requires_grad = False
        self.snapshot_version += 1

    def compute_potential(
        self,
        state_feat: torch.Tensor,
        estimated_cmax: float,
        current_time: float,
        h0: float,
        last_transfer_time: float,
        is_terminal: bool = False,
        graph_data: Any | None = None,
    ) -> float:
        """计算指定状态下的势能值 Φ(s)。

        Args:
            state_feat: 状态特征张量 (in_dim,)
            estimated_cmax: 启发式转站估计 P_q^h
            current_time: 现场当前时刻 t
            h0: 基准节拍时长 H_0
            last_transfer_time: 上次脉动转站时刻 P_{q-1}
            is_terminal: 是否为生产批次完全完工的终局状态

        Returns:
            标量势能值 Φ(s)。终局状态下严格返回 0.0。
        """
        # 1. 终局状态势严格置零
        if is_terminal:
            return 0.0

        device = next(self.frozen_head.parameters()).device
        if not isinstance(state_feat, torch.Tensor):
            feat_tensor = torch.tensor(state_feat, dtype=torch.float, device=device)
        else:
            feat_tensor = state_feat.to(device)

        if feat_tensor.ndim == 1:
            feat_tensor = feat_tensor.unsqueeze(0)

        with torch.no_grad():
            if self.frozen_actor is not None:
                if graph_data is None:
                    raise ValueError("图预测器塑形必须提供当前状态图快照")
                shared_feat = self.frozen_actor.encode_shared_representation(
                    feat_tensor.squeeze(0),
                    graph_data,
                ).unsqueeze(0)
            else:
                shared_feat = feat_tensor
            _, _, h_est_tensor = self.frozen_head.predict_corrected_time(
                state_feat=shared_feat,
                estimated_cmax=estimated_cmax,
                current_time=current_time,
                h0=h0,
                last_transfer_time=last_transfer_time,
            )
            h_est = float(h_est_tensor.item())

        h0_f = float(h0)
        # H^_q / H_0
        h_ratio = h_est / h0_f
        # [H^_q - H_0]_+ / H_0
        overdue_ratio = max(0.0, h_est - h0_f) / h0_f

        # Φ(s) = -a · (H^_q / H_0) - b · ([H^_q - H_0]_+ / H_0)
        phi = -self.a * h_ratio - self.b * overdue_ratio
        return float(phi)

    def shape_reward(
        self,
        actual_reward: float,
        phi_current: float,
        phi_next: float,
    ) -> float:
        """计算加入势函数差分后的单步塑形奖励。

        r_n^step = r_n^actual + β · (γ · Φ(s_{n+1}) - Φ(s_n))
        """
        potential_diff = (self.gamma * float(phi_next)) - float(phi_current)
        shaped = float(actual_reward) + (self.beta * potential_diff)
        return shaped
