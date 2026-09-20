"""Task 6.4 动态势函数奖励塑形器专项单元测试。

验证点：
1. 势函数数值单调性：预计节拍越长/超期越严重，Φ(s) 越负；
2. 终局势严格归零：is_terminal=True 时 Φ(s) = 0.0；
3. 折扣与无折扣下的伸缩求和定理 (Telescoping Sum & Policy Invariance)；
4. 轨迹内预测器版本冻结隔离：修改外部网络权重不影响当前塑形器内部副本。
"""

from __future__ import annotations

import pytest
import torch

from models.work3.potential_shaping import PotentialRewardShaper
from models.work3.time_head import TimeResidualHead


def test_potential_monotonicity() -> None:
    """测试预计节拍耗时与超期严重度对势能的负向单调影响。"""
    head = TimeResidualHead(in_dim=32, hidden_dim=64)
    # 将门控置为极小值，使修正时间严格退化为启发式基准值
    head.gate_fc[-1].bias.data.fill_(-100.0)

    shaper = PotentialRewardShaper(time_head=head, a=0.5, b=1.0, beta=1.0)
    feat = torch.randn(32)
    h0 = 100.0

    # 状态 A: 预计节拍正好为 100.0 (未超期)
    phi_a = shaper.compute_potential(
        state_feat=feat,
        estimated_cmax=100.0,
        current_time=0.0,
        h0=h0,
        last_transfer_time=0.0,
    )
    # H^ / H_0 = 1.0, 超期 0.0 => phi_a = -0.5 * 1.0 = -0.5
    assert abs(phi_a - (-0.5)) < 1e-4

    # 状态 B: 预计节拍为 150.0 (超期 50.0)
    phi_b = shaper.compute_potential(
        state_feat=feat,
        estimated_cmax=150.0,
        current_time=0.0,
        h0=h0,
        last_transfer_time=0.0,
    )
    # H^ / H_0 = 1.5, 超期 0.5 => phi_b = -0.5 * 1.5 - 1.0 * 0.5 = -0.75 - 0.5 = -1.25
    assert abs(phi_b - (-1.25)) < 1e-4
    assert phi_b < phi_a, "超期更严重时势能应更低"


def test_terminal_zero_potential() -> None:
    """测试终局状态势能严格归零。"""
    head = TimeResidualHead(in_dim=32, hidden_dim=64)
    shaper = PotentialRewardShaper(time_head=head)
    feat = torch.randn(32)

    phi_term = shaper.compute_potential(
        state_feat=feat,
        estimated_cmax=500.0,
        current_time=450.0,
        h0=100.0,
        last_transfer_time=400.0,
        is_terminal=True,
    )
    assert phi_term == 0.0, f"终局状态势能必须严格为 0.0，收到 {phi_term}"


def test_telescoping_sum_policy_invariance() -> None:
    """测试 γ = 1 时全轨迹塑形奖励与实际奖励的伸缩求和数学等价性。"""
    head = TimeResidualHead(in_dim=32, hidden_dim=64)
    beta = 1.5
    shaper = PotentialRewardShaper(time_head=head, a=0.5, b=1.0, beta=beta, gamma=1.0)

    # 模拟 5 步状态转移链
    h0 = 100.0
    phis = []
    actual_rewards = [-2.0, -0.5, -3.0, 0.0, -1.0]

    for i in range(5):
        feat = torch.randn(32)
        p_est = 100.0 + i * 10.0
        phi = shaper.compute_potential(
            state_feat=feat,
            estimated_cmax=p_est,
            current_time=float(i * 20),
            h0=h0,
            last_transfer_time=0.0,
            is_terminal=False,
        )
        phis.append(phi)

    # 终局状态 5
    phi_term = shaper.compute_potential(
        state_feat=torch.randn(32),
        estimated_cmax=200.0,
        current_time=100.0,
        h0=h0,
        last_transfer_time=0.0,
        is_terminal=True,
    )
    assert phi_term == 0.0

    # 逐步计算塑形奖励
    all_phis = phis + [phi_term]
    shaped_rewards = []
    for i in range(5):
        r_step = shaper.shape_reward(
            actual_reward=actual_rewards[i],
            phi_current=all_phis[i],
            phi_next=all_phis[i + 1],
        )
        shaped_rewards.append(r_step)

    sum_actual = sum(actual_rewards)
    sum_shaped = sum(shaped_rewards)

    # 数学理论值: sum_shaped = sum_actual + beta * (phi_term - phi_0) = sum_actual - beta * phi_0
    expected_sum_shaped = sum_actual + beta * (phi_term - all_phis[0])
    assert abs(sum_shaped - expected_sum_shaped) < 1e-12, "塑形奖励总和违反伸缩求和定理！"


def test_frozen_head_isolation() -> None:
    """测试轨迹内预测器版本冻结：外部网络权重修改不影响当前塑形器副本。"""
    head = TimeResidualHead(in_dim=32, hidden_dim=64)
    head.gate_fc[-1].bias.data.fill_(10.0)
    head.reg_fc[-1].bias.data.fill_(1.0)

    shaper = PotentialRewardShaper(time_head=head)
    feat = torch.ones(32)

    phi_before = shaper.compute_potential(
        state_feat=feat,
        estimated_cmax=100.0,
        current_time=0.0,
        h0=100.0,
        last_transfer_time=0.0,
    )

    # 外部模型发生剧烈权重更新
    head.reg_fc[-1].bias.data.fill_(-999.0)

    phi_after = shaper.compute_potential(
        state_feat=feat,
        estimated_cmax=100.0,
        current_time=0.0,
        h0=100.0,
        last_transfer_time=0.0,
    )

    assert phi_before == phi_after, "外部模型更新意外污染了冻结的势函数计算！"

    # 显式同步 snapshot 后，势能发生变化
    shaper.update_snapshot(head)
    phi_synced = shaper.compute_potential(
        state_feat=feat,
        estimated_cmax=100.0,
        current_time=0.0,
        h0=100.0,
        last_transfer_time=0.0,
    )
    assert phi_synced != phi_before, "显式 update_snapshot 后势函数应更新"
