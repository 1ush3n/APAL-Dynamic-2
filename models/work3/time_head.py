"""工作三 时间残差预测头 (Task 6.1)。

核心数学模型与技术规范：
依据《方法设计确认稿》第 7 节（启发式时间估计的学习修正）：
1. 挂载在状态表征之上的轻量级 2 层 MLP 时间修正头 δ_ψ(s)：
   输入: 状态向量 s_feat (默认 32 维精炼物理特征，或通用隐藏嵌入 z)
   输出: 归一化时间残差预测值 δ_ψ(s) ∈ R
2. 修正后脉动转站时刻：
   P^_q = max{t, P^_q^h + H_0 · δ_ψ(s)}
3. 衍生估计值：
   - 剩余生产耗时: R^ = P^_q - t
   - 周期持续耗时: H^_q = P^_q - P_{q-1}
4. 物理保障：
   - 时间不倒退：使用 torch.maximum / clamp 保证 P^_q >= t；
   - 零残差退化：当 δ_ψ = 0 时，结果退化为 max{t, P^_q^h}；
   - 保留计算图梯度流，支持端到端反向传播。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeResidualHead(nn.Module):
    """用于预测周期脉动完工时间归一化残差的轻量级门控神经网络模块。
    
    采用两阶段门控回归架构 (Hurdle / Gated Residual Regression)：
    1. gate_fc: 学习扰动与瓶颈识别门控，对名义无延误状态输出 0，杜绝假阳性时间漂移；
    2. reg_fc: 对存在延误风险的状态精确回归时间残差量值；
    3. 输出: δ_ψ(s) = Gate(s) · Reg(s) >= 0。
    """

    def __init__(
        self,
        in_dim: int = 32,
        hidden_dim: int = 64,
        use_layer_norm: bool = True,
        gate_threshold: float = 0.4,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.gate_threshold = float(gate_threshold)

        # 1. 扰动/瓶颈门控网络
        self.gate_fc = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, 1),
        )

        # 2. 残差量值回归网络
        self.reg_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32) if use_layer_norm else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(32, 1),
        )

        # 初始化回归偏置为微正值，防止死 ReLU
        self.reg_fc[-1].bias.data.fill_(0.05)
        # 初始化门控偏置为负值，使初始先验偏向于无扰动基准
        self.gate_fc[-1].bias.data.fill_(-1.0)

    def forward(self, state_feat: torch.Tensor) -> torch.Tensor:
        """前向传播计算归一化时间残差 δ_ψ(s)。"""
        gate_logits = self.gate_fc(state_feat).squeeze(-1)
        prob = torch.sigmoid(gate_logits)
        active = (prob > self.gate_threshold).float()
        mag = F.relu(self.reg_fc(state_feat).squeeze(-1))
        return active * mag

    def forward_with_logits(self, state_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """训练时专用前向接口，同时返回门控 logits、回归未激活输出与最终残差。"""
        gate_logits = self.gate_fc(state_feat).squeeze(-1)
        prob = torch.sigmoid(gate_logits)
        active = (prob > self.gate_threshold).float()
        mag = self.reg_fc(state_feat).squeeze(-1)
        delta = active * F.relu(mag)
        return gate_logits, mag, delta

    def predict_residual(self, state_feat: torch.Tensor) -> torch.Tensor:
        """调用前向传播计算残差。"""
        return self.forward(state_feat)

    def predict_corrected_time(
        self,
        state_feat: torch.Tensor,
        estimated_cmax: torch.Tensor | float,
        current_time: torch.Tensor | float,
        h0: float,
        last_transfer_time: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算物理约束下的修正转站时间 P^_q、剩余时间 R^ 与预计节拍 H^_q。

        Args:
            state_feat: 状态特征张量 (..., in_dim)
            estimated_cmax: 启发式估计完成时刻 P_q^h (标量或张量)
            current_time: 当前环境现场物理时刻 t
            h0: 基准脉动节拍时长 H_0
            last_transfer_time: 上一次脉动转站时刻 P_{q-1}，若未提供则默认取 current_time

        Returns:
            (p_corrected, r_estimated, h_estimated) 均为张量
        """
        device = state_feat.device
        delta = self.forward(state_feat)  # (...)

        if not isinstance(estimated_cmax, torch.Tensor):
            est_tensor = torch.tensor(estimated_cmax, dtype=torch.float, device=device)
        else:
            est_tensor = estimated_cmax.to(device)

        if not isinstance(current_time, torch.Tensor):
            t_tensor = torch.tensor(current_time, dtype=torch.float, device=device)
        else:
            t_tensor = current_time.to(device)

        # 1. 修正完工时间：P^_q = max{t, P^_q^h + H_0 · δ_ψ(s)}
        p_raw = est_tensor + (float(h0) * delta)
        p_corrected = torch.maximum(t_tensor, p_raw)

        # 2. 估计剩余耗时：R^ = P^_q - t
        r_estimated = torch.clamp(p_corrected - t_tensor, min=0.0)

        # 3. 估计周期节拍：H^_q = P^_q - P_{q-1}
        if last_transfer_time is not None:
            if not isinstance(last_transfer_time, torch.Tensor):
                p_last_tensor = torch.tensor(last_transfer_time, dtype=torch.float, device=device)
            else:
                p_last_tensor = last_transfer_time.to(device)
            h_estimated = torch.clamp(p_corrected - p_last_tensor, min=0.0)
        else:
            h_estimated = r_estimated

        return p_corrected, r_estimated, h_estimated
