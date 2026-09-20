"""工作三 动作决策上下文显式时间特征融合层 (Task 6.3)。

核心数学模型与技术规范：
依据《方法设计确认稿》第 7.3 节（同一预测服务于动作、表征与塑形）：
1. 提取 2 维核心脉动紧迫度标量向量 u(s)：
   u(s) = [ R^(s) / H_0, (H_0 - (t - P_{q-1})) / H_0 ] ∈ R^2
   - 维度 0: 预测剩余耗时与基准节拍比值 (预计完工紧迫度)；
   - 维度 1: 周期名义窗口剩余预算比值 (名义窗口紧迫度，若已超期则为负值)。
2. 融合网络架构 TimeContextFusion：
   - 基础动作上下文输入: e_base ∈ R^(..., d)
   - 时间紧迫度输入: u(s) ∈ R^(..., 2)
   - 升维与非线性映射: e_time = LeakyReLU(W_u · u(s) + b_u) ∈ R^(..., d)
   - 残差门控融合: e_fused = LayerNorm(e_base + W_gate · e_time)
3. 梯度阻断机制 (stop_time_gradient):
   - 默认开启梯度阻断 (detach)，确保 PPO 策略梯度不反向污染时间预测头参数，
     保证时间修正量具备纯粹、无偏的物理时间预测解释性。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_time_urgency_vector(
    estimated_r: torch.Tensor | float,
    current_time: torch.Tensor | float,
    last_transfer_time: torch.Tensor | float,
    h0: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """计算 2 维显式脉动紧迫度向量 u(s) = [R^ / H_0, (H_0 - (t - P_{q-1})) / H_0]。

    Args:
        estimated_r: 预测剩余完工耗时 R^ (标量或张量)
        current_time: 当前物理时间 t
        last_transfer_time: 上一次脉动转站时刻 P_{q-1}
        h0: 基准脉动节拍时长 H_0
        device: 张量目标设备

    Returns:
        形状为 (..., 2) 的紧迫度张量
    """
    if not isinstance(estimated_r, torch.Tensor):
        r_tensor = torch.tensor(estimated_r, dtype=torch.float, device=device)
    else:
        r_tensor = estimated_r.to(device)

    if not isinstance(current_time, torch.Tensor):
        t_tensor = torch.tensor(current_time, dtype=torch.float, device=device)
    else:
        t_tensor = current_time.to(device)

    if not isinstance(last_transfer_time, torch.Tensor):
        p_last_tensor = torch.tensor(last_transfer_time, dtype=torch.float, device=device)
    else:
        p_last_tensor = last_transfer_time.to(device)

    h0_f = float(h0)
    # 维度 0: 预计剩余耗时比例
    dim0 = r_tensor / h0_f

    # 维度 1: 名义窗口剩余比例 (H_0 - (t - P_{q-1})) / H_0
    nominal_elapsed = t_tensor - p_last_tensor
    dim1 = (h0_f - nominal_elapsed) / h0_f

    return torch.stack([dim0, dim1], dim=-1)


class TimeContextFusion(nn.Module):
    """动作上下文与显式时间紧迫度特征融合网络。"""

    def __init__(
        self,
        context_dim: int = 64,
        time_dim: int = 2,
        hidden_dim: int = 64,
        stop_time_gradient: bool = True,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.time_dim = time_dim
        self.stop_time_gradient = stop_time_gradient

        # 时间特征升维映射
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, context_dim),
        )

        self.ln = nn.LayerNorm(context_dim) if use_layer_norm else nn.Identity()

    def forward(
        self,
        base_context: torch.Tensor,
        time_urgency: torch.Tensor,
    ) -> torch.Tensor:
        """执行上下文与时间特征的融合。

        Args:
            base_context: 基础动作上下文特征，形状 (..., context_dim)
            time_urgency: 显式时间紧迫度向量，形状 (..., time_dim)

        Returns:
            融合后的决策上下文张量，形状 (..., context_dim)
        """
        u = time_urgency
        if self.stop_time_gradient:
            u = u.detach()

        e_time = self.time_mlp(u)
        e_fused = self.ln(base_context + e_time)
        return e_fused
