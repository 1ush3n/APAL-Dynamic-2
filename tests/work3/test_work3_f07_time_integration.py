"""W3-08/F07：修正时间、待补标签与塑形预测器快照闭环反例。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import pytest
import torch

from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from models.work3.potential_shaping import PotentialRewardShaper
from models.work3.ppo_buffer import PendingTimeLabelCache, PPOTransition, RolloutBufferWork3
from models.work3.ppo_trainer import PPOTrainerWork3
from models.work3.time_head import TimeResidualHead
from models.work3.action_fusion import compute_time_urgency_vector
from scripts.work3.train_ppo_work3 import (
    compute_online_snapshot_time_inputs,
    compute_online_time_inputs,
)


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
        decision_id=0,
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
    assert batch["decision_ids"] == [0]
    assert cache.pending_count == 0


def test_time_label_uses_heuristic_snapshot_captured_at_decision() -> None:
    """实际转站标签使用决策时P_h=12，而非由后续状态替换的数值。"""
    cache = PendingTimeLabelCache()
    cache.add(
        episode_id=3,
        cycle_id=2,
        decision_id=0,
        state_feat=torch.ones(32),
        graph_snapshot="decision-time-graph",
        estimated_cmax=12.0,
        current_time=5.0,
        h0=10.0,
        predictor_version=7,
    )

    assert cache.attach_transfer(
        episode_id=3,
        cycle_id=2,
        actual_transfer_time=9.0,
    )
    batch = cache.drain_ready()

    assert batch is not None
    assert batch["estimated_cmax"].tolist() == pytest.approx([12.0])
    assert batch["actual_transfer_times"].tolist() == pytest.approx([9.0])
    assert batch["target_residuals"].tolist() == pytest.approx([-0.3])


def test_all_decision_states_receive_their_own_cycle_label_without_episode_mixing() -> None:
    cache = PendingTimeLabelCache()
    snapshots = [
        {"feature": torch.tensor([float(index)])}
        for index in range(4)
    ]
    for index, estimated_cmax in enumerate((12.0, 15.0, 18.0)):
        cache.add(
            episode_id=3,
            cycle_id=2,
            decision_id=index,
            state_feat=torch.full((32,), float(index)),
            graph_snapshot=snapshots[index],
            estimated_cmax=estimated_cmax,
            current_time=float(index),
            h0=10.0,
            predictor_version=7,
        )
        snapshots[index]["feature"].fill_(-1.0)
    cache.add(
        episode_id=4,
        cycle_id=2,
        decision_id=3,
        state_feat=torch.full((32,), 3.0),
        graph_snapshot=snapshots[3],
        estimated_cmax=8.0,
        current_time=4.0,
        h0=10.0,
        predictor_version=8,
    )
    snapshots[3]["feature"].fill_(-1.0)

    assert cache.pending_count == 4
    assert cache.drain_ready() is None
    assert cache.attach_transfer(
        episode_id=3,
        cycle_id=2,
        actual_transfer_time=20.0,
    )
    assert cache.attach_transfer(
        episode_id=4,
        cycle_id=2,
        actual_transfer_time=25.0,
    )
    batch = cache.drain_ready()

    assert batch is not None
    assert batch["episode_ids"] == [3, 3, 3, 4]
    assert batch["cycle_ids"] == [2, 2, 2, 2]
    assert batch["decision_ids"] == [0, 1, 2, 3]
    assert batch["target_residuals"].tolist() == pytest.approx([0.8, 0.5, 0.2, 1.7])
    assert [float(graph["feature"][0]) for graph in batch["graph_snapshots"]] == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]


def test_discard_episode_drops_only_unlabelled_samples() -> None:
    cache = PendingTimeLabelCache()
    for decision_id, cycle_id in enumerate((1, 2)):
        cache.add(
            episode_id=8,
            cycle_id=cycle_id,
            decision_id=decision_id,
            state_feat=torch.zeros(32),
            graph_snapshot=f"graph-{cycle_id}",
            estimated_cmax=10.0,
            current_time=0.0,
            h0=10.0,
            predictor_version=1,
        )
    cache.attach_transfer(episode_id=8, cycle_id=1, actual_transfer_time=12.0)
    cache.discard_episode(8)

    assert cache.pending_count == 0
    batch = cache.drain_ready()
    assert batch is not None
    assert batch["decision_ids"] == [0]
    assert batch["target_residuals"].tolist() == pytest.approx([0.2])


def test_pending_cache_copies_pyg_graph_snapshot(env: AirLineEnvWork3) -> None:
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    graph = actor.build_graph_snapshot(env)
    original_task_feature = graph["task"].x[0, 0].clone()
    cache = PendingTimeLabelCache()
    cache.add(
        episode_id=1,
        cycle_id=1,
        decision_id=0,
        state_feat=torch.zeros(32),
        graph_snapshot=graph,
        estimated_cmax=10.0,
        current_time=0.0,
        h0=10.0,
        predictor_version=1,
    )
    graph["task"].x[0, 0] = original_task_feature + 1.0
    cache.attach_transfer(episode_id=1, cycle_id=1, actual_transfer_time=10.0)

    batch = cache.drain_ready()
    assert batch is not None
    stored_graph = batch["graph_snapshots"][0]
    assert stored_graph is not graph
    assert torch.equal(stored_graph["task"].x[0, 0], original_task_feature)


def test_auxiliary_supervision_calls_do_not_scale_with_ppo_batch_size(
    env: AirLineEnvWork3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_counts: list[int] = []
    cmax = compute_cycle_heuristic_cmax(env.state)
    state_feat = extract_compact_state_features(env.state, cmax)
    urgency = compute_time_urgency_vector(
        estimated_r=max(0.0, cmax - float(env.state.current_time)),
        current_time=env.state.current_time,
        last_transfer_time=env.state.last_transfer_time,
        h0=env.state.h0,
    )

    for ppo_batch_size in (1, 4):
        actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
        head = TimeResidualHead(in_dim=16, hidden_dim=16)
        trainer = PPOTrainerWork3(actor_critic=actor, time_head=head, time_loss_coef=1.0)
        action, log_prob, value, record = actor.select_action(
            env, state_feat, urgency, deterministic=True
        )
        buffer = RolloutBufferWork3(normalize_advantages=False)
        for index in range(4):
            buffer.add(PPOTransition(
                state_feat=state_feat,
                time_urgency=urgency,
                sample_record=record,
                reward=float(index),
                raw_reward=float(index),
                value=value,
                log_prob=log_prob,
                action_dict=action,
                terminated=index == 3,
            ))
        buffer.finish_trajectory()
        auxiliary_batch = {
            "state_feats": torch.stack([state_feat] * 3),
            "graph_snapshots": [record["graph_snapshot"]] * 3,
            "target_residuals": torch.tensor([0.8, 0.5, 0.2]),
            "episode_ids": [1, 1, 1],
            "cycle_ids": [1, 1, 1],
            "decision_ids": [10, 11, 12],
        }
        count = 0

        def counted_loss(_: dict[str, object]) -> torch.Tensor:
            nonlocal count
            count += 1
            return head.reg_fc[-1].bias.square().mean()

        monkeypatch.setattr(trainer, "compute_time_auxiliary_loss", counted_loss)
        metrics = trainer.train_step(
            buffer,
            ppo_epochs=1,
            batch_size=ppo_batch_size,
            time_auxiliary_batch=auxiliary_batch,
            time_auxiliary_epochs=1,
            time_auxiliary_batch_size=2,
        )
        assert metrics["time_label_count"] == 3
        assert metrics["time_supervision_steps"] == 2
        call_counts.append(count)

    assert call_counts == [2, 2]


def test_ppo_policy_loss_does_not_backpropagate_through_online_time_input(
    env: AirLineEnvWork3,
) -> None:
    """在线显式时间输入停止梯度，单独PPO损失不更新时间头。"""
    torch.manual_seed(17)
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    actor.eval()
    head = TimeResidualHead(in_dim=16, hidden_dim=16)
    trainer = PPOTrainerWork3(actor_critic=actor, time_head=head, time_loss_coef=1.0)
    cmax = compute_cycle_heuristic_cmax(env.state)
    state_feat = extract_compact_state_features(env.state, cmax)
    base_snapshot = actor.make_decision_snapshot(
        env,
        state_feat,
        torch.zeros(2),
        estimated_cmax=cmax,
    )
    _, time_input, _ = compute_online_snapshot_time_inputs(
        actor, head, base_snapshot
    )
    assert not time_input.requires_grad

    action, old_log_prob, value, record = actor.select_snapshot(
        replace(base_snapshot, time_features=time_input),
        deterministic=True,
    )
    assert action is not None
    minibatch = {
        "state_feats": state_feat.unsqueeze(0),  # [32] -> [1, 32]
        "time_urgencies": time_input.unsqueeze(0),  # [2] -> [1, 2]
        "old_log_probs": torch.tensor([old_log_prob]),  # [] -> [1]
        "advantages": torch.ones(1),
        "target_values": torch.tensor([value]),  # [] -> [1]
        "sample_records": [record],
    }

    trainer.optimizer.zero_grad()
    losses = trainer.compute_ppo_minibatch_loss(minibatch)
    losses["total_loss"].backward()

    assert all(parameter.grad is None for parameter in head.parameters())


def test_ppo_replay_uses_saved_time_input_after_online_head_changes(
    env: AirLineEnvWork3,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PPO重放保留采样时显式时间输入，不按更新后的时间头重算。"""
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    actor.eval()
    head = TimeResidualHead(in_dim=16, hidden_dim=16)
    _zero_head(head, 0.0)
    cmax = compute_cycle_heuristic_cmax(env.state)
    state_feat = extract_compact_state_features(env.state, cmax)
    base_snapshot = actor.make_decision_snapshot(
        env,
        state_feat,
        torch.zeros(2),
        estimated_cmax=cmax,
    )
    graph_snapshot, sampled_time_input, _ = compute_online_snapshot_time_inputs(
        actor, head, base_snapshot
    )
    action_snapshot = replace(base_snapshot, time_features=sampled_time_input)
    action, old_log_prob, value, record = actor.select_snapshot(
        action_snapshot, deterministic=True
    )
    assert action is not None

    buffer = RolloutBufferWork3(normalize_advantages=False)
    buffer.add(PPOTransition(
        state_feat=state_feat,
        time_urgency=sampled_time_input,
        sample_record=record,
        reward=1.0,
        raw_reward=1.0,
        value=value,
        log_prob=old_log_prob,
        done=True,
        terminated=True,
        action_dict=action,
    ))
    buffer.finish_trajectory()
    saved_time_input = buffer.transitions[0].time_urgency.clone()

    with torch.no_grad():
        head.reg_fc[-1].bias.add_(0.5)
    _, changed_time_input, _ = compute_online_snapshot_time_inputs(
        actor, head, base_snapshot
    )
    assert not torch.equal(sampled_time_input, changed_time_input)

    trainer = PPOTrainerWork3(
        actor_critic=actor,
        time_head=head,
        time_loss_coef=1.0,
    )
    observed_time_inputs: list[torch.Tensor] = []
    original_evaluate = actor.evaluate_action_log_probs

    def capture_time_inputs(
        *,
        state_feats: torch.Tensor,
        time_urgencies: torch.Tensor,
        sample_records: list[dict[str, object]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observed_time_inputs.append(time_urgencies.detach().cpu().clone())
        return original_evaluate(
            state_feats=state_feats,
            time_urgencies=time_urgencies,
            sample_records=sample_records,
        )

    monkeypatch.setattr(actor, "evaluate_action_log_probs", capture_time_inputs)
    head_before_update = {
        name: parameter.detach().clone()
        for name, parameter in head.named_parameters()
    }
    metrics = trainer.train_step(
        buffer,
        ppo_epochs=1,
        batch_size=1,
        time_auxiliary_batch={
            "state_feats": state_feat.unsqueeze(0),
            "graph_snapshots": [graph_snapshot],
            "target_residuals": torch.tensor([-0.3]),
            "worker_ids": [0],
            "episode_ids": [1],
            "cycle_ids": [base_snapshot.cycle_id],
            "decision_ids": [0],
        },
    )

    assert len(observed_time_inputs) == 1
    assert torch.equal(observed_time_inputs[0], saved_time_input.unsqueeze(0))
    assert torch.equal(buffer.transitions[0].time_urgency, saved_time_input)
    assert metrics["time_supervision_steps"] == 1
    assert any(
        not torch.equal(head_before_update[name], parameter.detach())
        for name, parameter in head.named_parameters()
    )


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
