"""Task 6.3 动作头显式时间特征融合专项单元测试。

验证点：
1. compute_time_urgency_vector 时间紧迫度标量计算精度与正负号物理语义；
2. TimeContextFusion 融合层输入输出形状保持与多维 Batch 支持；
3. stop_time_gradient=True 时严格阻断时间头梯度回传；
4. stop_time_gradient=False 时支持端到端联合反向传播。
"""

from __future__ import annotations

import pytest
import torch

from models.work3.action_fusion import TimeContextFusion, compute_time_urgency_vector


def test_time_urgency_vector_calculation() -> None:
    """测试时间紧迫度向量计算与物理含义。"""
    # 场景 1: 周期刚开始 t = 100, P_{q-1} = 100, H_0 = 100, R^ = 80
    u1 = compute_time_urgency_vector(estimated_r=80.0, current_time=100.0, last_transfer_time=100.0, h0=100.0)
    assert torch.allclose(u1, torch.tensor([0.8, 1.0])), f"期望 [0.8, 1.0]，收到 {u1}"

    # 场景 2: 周期已过半 t = 150, P_{q-1} = 100, H_0 = 100, R^ = 30
    u2 = compute_time_urgency_vector(estimated_r=30.0, current_time=150.0, last_transfer_time=100.0, h0=100.0)
    assert torch.allclose(u2, torch.tensor([0.3, 0.5])), f"期望 [0.3, 0.5]，收到 {u2}"

    # 场景 3: 周期已超期 t = 220, P_{q-1} = 100, H_0 = 100, R^ = 10
    u3 = compute_time_urgency_vector(estimated_r=10.0, current_time=220.0, last_transfer_time=100.0, h0=100.0)
    assert torch.allclose(u3, torch.tensor([0.1, -0.2])), f"期望 [0.1, -0.2]，收到 {u3}"


def test_fusion_shapes_and_batching() -> None:
    """测试融合层在单样本与批处理场景下的张量形状一致性。"""
    fusion = TimeContextFusion(context_dim=64, time_dim=2, hidden_dim=32)

    # 2D Batch: (16, 64) 与 (16, 2)
    base_batch = torch.randn(16, 64)
    u_batch = torch.randn(16, 2)
    fused_batch = fusion(base_batch, u_batch)
    assert fused_batch.shape == (16, 64), f"期望形状 (16, 64)，收到 {fused_batch.shape}"

    # 3D Batch (序列/候选多任务): (8, 10, 64) 与 (8, 10, 2)
    base_3d = torch.randn(8, 10, 64)
    u_3d = torch.randn(8, 10, 2)
    fused_3d = fusion(base_3d, u_3d)
    assert fused_3d.shape == (8, 10, 64), f"期望形状 (8, 10, 64)，收到 {fused_3d.shape}"


def test_stop_gradient_mechanism() -> None:
    """测试 stop_time_gradient 梯度阻断行为。"""
    # 1. 默认 stop_time_gradient = True：时间向量梯度应为 None
    fusion_stop = TimeContextFusion(context_dim=32, time_dim=2, stop_time_gradient=True)
    base = torch.randn(4, 32, requires_grad=True)
    u = torch.randn(4, 2, requires_grad=True)

    out = fusion_stop(base, u)
    loss = out.sum()
    loss.backward()

    assert base.grad is not None, "基础上下文梯度应正常存在"
    assert u.grad is None, "stop_time_gradient=True 时，时间紧迫度向量绝不得产生梯度反传！"

    # 2. 开启 stop_time_gradient = False：时间向量应有梯度流经
    fusion_flow = TimeContextFusion(context_dim=32, time_dim=2, stop_time_gradient=False)
    base2 = torch.randn(4, 32, requires_grad=True)
    u2 = torch.randn(4, 2, requires_grad=True)

    out2 = fusion_flow(base2, u2)
    loss2 = out2.sum()
    loss2.backward()

    assert u2.grad is not None, "stop_time_gradient=False 时，时间向量应接收梯度"
    assert not torch.isnan(u2.grad).any(), "反向传播梯度中含有 NaN"
