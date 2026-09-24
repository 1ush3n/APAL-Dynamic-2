"""W3-06 图策略接入与 PPO 重放快照反例。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from envs.work3.environment import AirLineEnvWork3
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features
from models.work3.graph_builder import GRAPH_FEATURE_VERSION
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax


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


def _state_inputs(env: AirLineEnvWork3) -> tuple[torch.Tensor, torch.Tensor]:
    cmax = compute_cycle_heuristic_cmax(env.state)
    state_feat = extract_compact_state_features(env.state, cmax)
    urgency = compute_time_urgency_vector(
        estimated_r=max(0.0, cmax - env.state.current_time),
        current_time=env.state.current_time,
        last_transfer_time=env.state.last_transfer_time,
        h0=env.state.h0,
    )
    return state_feat, urgency


def test_sampling_record_contains_immutable_graph_snapshot(env: AirLineEnvWork3) -> None:
    """采样时必须保存图快照，不能在现场变化后用新状态重构旧动作。"""
    net = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=32)
    state_feat, urgency = _state_inputs(env)

    action, _, _, record = net.select_action(env, state_feat, urgency)

    assert action is not None
    assert isinstance(record["graph_snapshot"], HeteroData)
    assert record["graph_version"] == GRAPH_FEATURE_VERSION
    assert len(record["candidate_task_node_indices"]) == len(record["cand_feats"])
    assert record["worker_node_indices"]


def test_graph_encoder_receives_policy_gradient(env: AirLineEnvWork3) -> None:
    """策略重放损失必须能反传到图编码器，而不是只更新汇总 MLP。"""
    net = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=32)
    state_feat, urgency = _state_inputs(env)
    action, _, _, record = net.select_action(env, state_feat, urgency, deterministic=True)
    assert action is not None

    _, log_probs, _ = net.evaluate_action_log_probs(
        state_feat.unsqueeze(0), urgency.unsqueeze(0), [record]
    )
    (-log_probs.mean()).backward()

    graph_parameters = [p for p in net.graph_encoder.parameters() if p.requires_grad]
    assert graph_parameters
    assert any(parameter.grad is not None for parameter in graph_parameters)
    assert any(parameter.grad is not None for parameter in net.encoder.parameters())


def test_worker_calendar_changes_graph_policy_input(env: AirLineEnvWork3) -> None:
    """同一物理状态下改变具体工人日历，图输入必须发生变化。"""
    net = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=32)
    before = net.build_graph_snapshot(env)
    worker_id = next(iter(env.state.workers))
    env.state.workers[worker_id].add_interval(0.0, 10.0, "calendar_probe")
    after = net.build_graph_snapshot(env)

    worker_index = net.graph_builder.worker_id_to_idx[worker_id]
    assert not torch.equal(
        before["worker"].x[worker_index],
        after["worker"].x[worker_index],
    )


def test_legacy_actor_checkpoint_can_be_loaded(baseline_path: Path) -> None:
    """已有工作三旧 Actor 检查点应显式兼容，不阻塞图策略接入。"""
    checkpoint = Path("models/work3/checkpoints/method_d_model.pt")
    if not checkpoint.is_file():
        pytest.skip(f"{checkpoint} 不存在")
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = data.get("actor_critic_state", data)

    net = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    net.load_state_dict(state_dict)
    assert net.graph_policy_enabled is False
