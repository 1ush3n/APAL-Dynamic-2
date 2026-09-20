"""Task 6.1 时间修正头模型专项单元测试。

验证点：
1. 模型结构与输入输出维度：支持 1D 单样本与 2D Batch 张量；
2. 零残差退化性：当残差为 0 时输出与静态估计一致；
3. 时间单调性与物理下界：P^_q >= t 且 R^ >= 0，严禁时间倒流；
4. 梯度流与端到端反向传播：计算图完整可微，支持损失反传更新权重。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.work3.time_head import TimeResidualHead


def test_time_residual_head_shapes() -> None:
    """测试单样本与批处理输入的输出形状。"""
    head = TimeResidualHead(in_dim=32, hidden_dim=64)

    # 单样本 (32,)
    x_single = torch.randn(32)
    delta_single = head(x_single)
    assert delta_single.ndim == 0 or delta_single.shape == (), f"单样本输出应为标量，收到 {delta_single.shape}"

    # Batch 样本 (16, 32)
    x_batch = torch.randn(16, 32)
    delta_batch = head(x_batch)
    assert delta_batch.shape == (16,), f"Batch 输出应为 (16,)，收到 {delta_batch.shape}"


def test_zero_residual_fallback() -> None:
    """测试零残差退化特性：当模型权重置零时，修正时间退化为启发式基准。"""
    head = TimeResidualHead(in_dim=32, hidden_dim=64)
    # 手动将最后一层权重与偏置清零，确保输出 delta = 0
    head.fc2.weight.data.zero_()
    head.fc2.bias.data.zero_()

    x = torch.randn(5, 32)
    t = 100.0
    p_h = 250.0
    h0 = 200.0

    p_corr, r_est, h_est = head.predict_corrected_time(
        state_feat=x,
        estimated_cmax=p_h,
        current_time=t,
        h0=h0,
        last_transfer_time=50.0,
    )

    assert torch.allclose(p_corr, torch.tensor(p_h, dtype=torch.float)), "零残差下 P^_q 应严格等于 P_q^h"
    assert torch.allclose(r_est, torch.tensor(p_h - t, dtype=torch.float)), "零残差下 R^ 应严格等于 P_q^h - t"
    assert torch.allclose(h_est, torch.tensor(p_h - 50.0, dtype=torch.float)), "零残差下 H^_q 应严格等于 P_q^h - P_{q-1}"


def test_time_monotonicity_and_physical_bound() -> None:
    """测试当网络预测大幅负向残差时，物理时间下界严格生效，绝不早于当前时刻 t。"""
    head = TimeResidualHead(in_dim=32, hidden_dim=64)

    # 构造极端负残差场景：t = 500, p_h = 510, delta 被强制设为极度负值 -10.0
    x = torch.randn(32)
    t = 500.0
    p_h = 510.0
    h0 = 100.0

    # 假定 delta 计算后大幅跌穿 t (如 510 - 1000 = -490 < 500)
    with torch.no_grad():
        head.fc2.bias.data.fill_(-10.0)

    p_corr, r_est, h_est = head.predict_corrected_time(
        state_feat=x,
        estimated_cmax=p_h,
        current_time=t,
        h0=h0,
        last_transfer_time=400.0,
    )

    assert p_corr.item() >= t, f"修正时间 {p_corr.item()} 不得早于当前时刻 {t}"
    assert r_est.item() >= 0.0, f"剩余耗时 {r_est.item()} 不得小于 0"
    assert h_est.item() >= t - 400.0, f"节拍耗时 {h_est.item()} 应满足下界"


def test_gradient_backpropagation() -> None:
    """测试计算图完整可微，支持损失反向传播与权重更新。"""
    head = TimeResidualHead(in_dim=32, hidden_dim=64)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    x = torch.randn(8, 32)
    y_target = torch.randn(8)

    delta = head(x)
    loss = F.smooth_l1_loss(delta, y_target)

    optimizer.zero_grad()
    loss.backward()

    assert head.fc1.weight.grad is not None, "fc1 梯度未正常生成"
    assert head.fc2.weight.grad is not None, "fc2 梯度未正常生成"
    assert not torch.isnan(head.fc1.weight.grad).any(), "梯度中含有 NaN"

    optimizer.step()
