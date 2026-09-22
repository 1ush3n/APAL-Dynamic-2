"""W3-08/F07：修正时间、待补标签与塑形预测器快照闭环反例。"""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest
import torch

from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from models.work3.potential_shaping import PotentialRewardShaper
from models.work3.ppo_buffer import PendingTimeLabelCache
from models.work3.ppo_trainer import PPOTrainerWork3
from models.work3.time_head import TimeResidualHead
from scripts.work3.train_ppo_work3 import compute_online_time_inputs


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


def _zero_head(head: TimeResidualHead, bias: float) -> None:
    with torch.no_grad():
        for module in head.reg_fc:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
        head.reg_fc[-1].bias.fill_(bias)


def test_corrected_time_reaches_actor_urgency_input(env: AirLineEnvWork3) -> None:
    """改变时间头输出时，Actor接收的显式时间输入必须随之改变。"""
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    head = TimeResidualHead(in_dim=16, hidden_dim=16)
    cmax = compute_cycle_heuristic_cmax(env.state)
    state_feat = extract_compact_state_features(env.state, cmax)

    _zero_head(head, -0.2)
    _, early_urgency, early_prediction = compute_online_time_inputs(
        actor, head, env, state_feat, cmax
    )
    _zero_head(head, 0.2)
    _, late_urgency, late_prediction = compute_online_time_inputs(
        actor, head, env, state_feat, cmax
    )

    assert late_prediction.item() > early_prediction.item()
    assert not torch.equal(early_urgency, late_urgency)


def test_pending_time_labels_survive_rollout_buffer_boundary() -> None:
    """待补标签不能随一次PPO采样缓冲清空，真实转站后才生成监督样本。"""
    cache = PendingTimeLabelCache()
    cache.add(
        episode_id=3,
        cycle_id=2,
        state_feat=torch.ones(32),
        graph_snapshot="graph-snapshot",
        estimated_cmax=10.0,
        current_time=5.0,
        h0=5.0,
        predictor_version=7,
    )

    assert cache.pending_count == 1
    assert cache.drain_ready() is None

    cache.attach_transfer(
        episode_id=3,
        cycle_id=2,
        actual_transfer_time=15.0,
    )
    batch = cache.drain_ready()

    assert batch is not None
    assert batch["state_feats"].shape == (1, 32)
    assert batch["graph_snapshots"] == ["graph-snapshot"]
    assert batch["target_residuals"].tolist() == pytest.approx([1.0])
    assert batch["predictor_versions"] == [7]
    assert cache.pending_count == 0


def test_shaping_snapshot_freezes_graph_and_time_head_until_update(env: AirLineEnvWork3) -> None:
    """同一生产轨迹内塑形副本冻结，显式刷新后才读取新模型。"""
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    head = TimeResidualHead(in_dim=16, hidden_dim=16)
    _zero_head(head, 0.0)
    shaper = PotentialRewardShaper(time_head=head, actor_critic=actor)
    graph = actor.build_graph_snapshot(env)
    cmax = compute_cycle_heuristic_cmax(env.state)
    state_feat = extract_compact_state_features(env.state, cmax)
    kwargs = {
        "state_feat": state_feat,
        "estimated_cmax": cmax,
        "current_time": float(env.state.current_time),
        "h0": float(env.state.h0),
        "last_transfer_time": float(env.state.last_transfer_time),
        "graph_data": graph,
    }

    version_before = shaper.snapshot_version
    phi_before = shaper.compute_potential(**kwargs)
    _zero_head(head, 0.5)
    with torch.no_grad():
        next(iter(actor.graph_encoder.parameters())).add_(1.0)
    phi_during = shaper.compute_potential(**kwargs)

    assert phi_during == phi_before
    assert shaper.snapshot_version == version_before
    assert shaper.frozen_actor is not None
    assert all(not parameter.requires_grad for parameter in shaper.frozen_actor.parameters())

    shaper.update_snapshot(head, actor)
    phi_after = shaper.compute_potential(**kwargs)
    assert shaper.snapshot_version == version_before + 1
    assert phi_after != phi_before


def test_checkpoint_contains_actor_and_shared_time_head() -> None:
    """完整检查点必须同时恢复图Actor和共享时间头。"""
    actor_1 = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    head_1 = TimeResidualHead(in_dim=16, hidden_dim=16)
    trainer_1 = PPOTrainerWork3(actor_critic=actor_1, time_head=head_1)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir) / "work3_actor_time.pt"
        trainer_1.save_checkpoint(str(checkpoint))

        actor_2 = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
        head_2 = TimeResidualHead(in_dim=16, hidden_dim=16)
        trainer_2 = PPOTrainerWork3(actor_critic=actor_2, time_head=head_2)
        trainer_2.load_checkpoint(str(checkpoint))

        for name, parameter in head_1.state_dict().items():
            assert torch.equal(parameter, head_2.state_dict()[name])
