"""W3-07/F06：有符号时间残差与共享图表征辅助损失反例。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from envs.work3.environment import AirLineEnvWork3
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from models.work3.ppo_buffer import PPOTransition, RolloutBufferWork3
from models.work3.ppo_trainer import PPOTrainerWork3
from models.work3.time_head import TimeResidualHead


@pytest.fixture
def baseline_path() -> Path:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return path


@pytest.fixture
def env(baseline_path: Path) -> AirLineEnvWork3:
    environment = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    environment.reset()
    return environment


def _state_inputs(
    env: AirLineEnvWork3,
) -> tuple[torch.Tensor, torch.Tensor]:
    estimated_cmax = compute_cycle_heuristic_cmax(env.state)
    state_feat = extract_compact_state_features(env.state, estimated_cmax)
    urgency = compute_time_urgency_vector(
        estimated_r=max(0.0, estimated_cmax - env.state.current_time),
        current_time=env.state.current_time,
        last_transfer_time=env.state.last_transfer_time,
        h0=env.state.h0,
    )
    return state_feat, urgency


def test_time_head_can_represent_both_signed_residuals() -> None:
    """时间头不能把高估纠正量截断为零。"""
    head = TimeResidualHead(in_dim=1, hidden_dim=8)
    with torch.no_grad():
        for module in head.reg_fc:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        head.reg_fc[-1].bias.fill_(-0.5)
    assert head(torch.zeros(1, 1)).item() < 0.0

    with torch.no_grad():
        head.reg_fc[-1].bias.fill_(0.5)
    assert head(torch.zeros(1, 1)).item() > 0.0


def test_signed_labels_are_all_used_by_offline_regression() -> None:
    """离线时间头训练应同时拟合负、正监督标签。"""
    torch.manual_seed(7)
    head = TimeResidualHead(in_dim=1, hidden_dim=16)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.03)
    x = torch.tensor([[-1.0], [1.0]]).repeat(32, 1)
    y = torch.tensor([-0.5, 0.5]).repeat(32)

    for _ in range(120):
        pred = head(x)
        loss = torch.nn.functional.smooth_l1_loss(pred, y, beta=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    fitted = head(torch.tensor([[-1.0], [1.0]])).detach()
    assert fitted[0].item() < -0.2
    assert fitted[1].item() > 0.2


@pytest.mark.parametrize(
    ("estimated_cmax", "current_time", "delta", "expected_time"),
    [
        pytest.param(101.0, 100.0, -2.0, 100.0, id="strong-negative-correction"),
        pytest.param(12.0, 5.0, -1.0, 5.0, id="H03-exact-boundary"),
    ],
)
def test_physical_time_lower_bound_remains_after_negative_correction(
    estimated_cmax: float,
    current_time: float,
    delta: float,
    expected_time: float,
) -> None:
    """有符号残差不能突破实际当前时刻下界。"""
    head = TimeResidualHead(in_dim=2, hidden_dim=8)
    with torch.no_grad():
        for module in head.reg_fc:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        head.reg_fc[-1].bias.fill_(delta)

    corrected, remaining, _ = head.predict_corrected_time(
        state_feat=torch.zeros(1, 2),
        estimated_cmax=estimated_cmax,
        current_time=current_time,
        h0=10.0,
    )
    assert corrected.item() == pytest.approx(expected_time)
    assert remaining.item() == pytest.approx(0.0)


def test_negative_residual_applies_exact_heuristic_time_correction() -> None:
    """真实时间头输出δ=-0.3时，P_h=12、t=5、H0=10应修正为9。"""
    head = TimeResidualHead(in_dim=2, hidden_dim=8)
    with torch.no_grad():
        for module in head.reg_fc:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        head.reg_fc[-1].bias.fill_(-0.3)

    corrected, remaining, _ = head.predict_corrected_time(
        state_feat=torch.zeros(1, 2),
        estimated_cmax=12.0,
        current_time=5.0,
        h0=10.0,
    )

    assert corrected.item() == pytest.approx(9.0)
    assert remaining.item() == pytest.approx(4.0)


def test_positive_residual_applies_exact_heuristic_time_correction() -> None:
    """时间头输出δ=0.2时，P_h=12、t=5、H0=10应修正为14。"""
    head = TimeResidualHead(in_dim=2, hidden_dim=8)
    with torch.no_grad():
        for module in head.reg_fc:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        head.reg_fc[-1].bias.fill_(0.2)

    corrected, remaining, _ = head.predict_corrected_time(
        state_feat=torch.zeros(1, 2),
        estimated_cmax=12.0,
        current_time=5.0,
        h0=10.0,
    )

    assert corrected.item() == pytest.approx(14.0)
    assert remaining.item() == pytest.approx(9.0)


def test_time_auxiliary_loss_reaches_shared_graph_encoder(env: AirLineEnvWork3) -> None:
    """时间辅助损失更新共享编码器与时间头，但不更新Critic。"""
    torch.manual_seed(23)
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    time_head = TimeResidualHead(in_dim=16, hidden_dim=16)
    trainer = PPOTrainerWork3(
        actor_critic=actor,
        time_head=time_head,
        time_loss_coef=0.5,
    )
    state_feat, urgency = _state_inputs(env)
    graph_snapshot = actor.build_graph_snapshot(env)
    aux_batch = {
        "state_feats": state_feat.unsqueeze(0),
        "time_urgencies": urgency.unsqueeze(0),
        "graph_snapshots": [graph_snapshot],
        "target_residuals": torch.tensor([-0.25]),
        "episode_ids": [1],
        "cycle_ids": [1],
        "decision_ids": [1],
    }

    trainer.optimizer.zero_grad()
    loss = trainer.compute_time_auxiliary_loss(aux_batch)
    assert loss.requires_grad
    loss.backward()

    graph_parameters = [p for p in actor.graph_encoder.parameters() if p.requires_grad]
    time_parameters = [p for p in time_head.parameters() if p.requires_grad]
    graph_gradients = [p.grad for p in graph_parameters if p.grad is not None]
    time_gradients = [p.grad for p in time_parameters if p.grad is not None]
    assert graph_gradients and all(torch.isfinite(grad).all() for grad in graph_gradients)
    assert time_gradients and all(torch.isfinite(grad).all() for grad in time_gradients)
    assert any(torch.count_nonzero(grad).item() > 0 for grad in graph_gradients)
    assert any(torch.count_nonzero(grad).item() > 0 for grad in time_gradients)
    assert all(parameter.grad is None for parameter in actor.critic.parameters())


def test_time_auxiliary_update_is_reported_separately_from_ppo_loss(
    env: AirLineEnvWork3,
) -> None:
    """时间辅助监督独立更新，并报告标签数、监督步数和时间损失。"""
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    time_head = TimeResidualHead(in_dim=16, hidden_dim=16)
    trainer = PPOTrainerWork3(
        actor_critic=actor,
        time_head=time_head,
        time_loss_coef=1.0,
    )
    state_feat, urgency = _state_inputs(env)
    action, log_prob, value, record = actor.select_action(env, state_feat, urgency, deterministic=True)
    assert action is not None

    buffer = RolloutBufferWork3(gamma=0.99, gae_lambda=0.95)
    buffer.add(
        PPOTransition(
            state_feat=state_feat,
            time_urgency=urgency,
            sample_record=record,
            reward=0.0,
            raw_reward=0.0,
            value=value,
            log_prob=log_prob,
            action_dict=action,
        )
    )
    buffer.finish_trajectory(last_value=0.0)
    metrics = trainer.train_step(
        buffer,
        ppo_epochs=1,
        batch_size=1,
        time_auxiliary_batch={
            "state_feats": state_feat.unsqueeze(0),
            "graph_snapshots": [record["graph_snapshot"]],
            "target_residuals": torch.tensor([-0.25]),
            "episode_ids": [1],
            "cycle_ids": [1],
            "decision_ids": [1],
        },
    )
    assert metrics["time_loss"] > 0.0
    assert metrics["time_label_count"] == 1
    assert metrics["time_supervision_steps"] == 1
