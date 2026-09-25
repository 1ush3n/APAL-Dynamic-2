"""Task 7.2 PPO 经验回放缓存与条件分支训练器专项单元测试。

验证点：
1. RolloutBufferWork3 GAE 计算与优势值标准化数学正确性；
2. RolloutBufferWork3 批处理生成器张量形状与 sample_record 字典对齐；
3. PPOTrainerWork3 策略与价值多 Epoch 梯度反向传播、无 NaN/Inf、参数实际更新；
4. 训练器模型权重检查点保存与恢复闭环。
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import pytest
import torch

from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import (
    ActorCriticWork3,
    extract_compact_state_features,
)
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from models.work3.ppo_buffer import PPOTransition, RolloutBufferWork3
from models.work3.ppo_trainer import PPOTrainerWork3


@pytest.fixture
def env() -> AirLineEnvWork3:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    env = AirLineEnvWork3(baseline_json_path=str(path))
    env.reset()
    return env


def test_buffer_gae_calculation() -> None:
    """测试 GAE (广义优势估计) 的数学递推与标准化。"""
    buffer = RolloutBufferWork3(gamma=0.9, gae_lambda=0.8, normalize_advantages=False)

    # 构造 3 个人工步骤:
    # R_0 = 1.0, V_0 = 2.0
    # R_1 = 2.0, V_1 = 3.0
    # R_2 = 3.0, V_2 = 4.0
    # V_terminal = 0.0, done_2 = True
    dummy_feat = torch.zeros(32)
    dummy_urgency = torch.zeros(2)

    buffer.add(PPOTransition(
        state_feat=dummy_feat,
        time_urgency=dummy_urgency,
        sample_record={},
        reward=1.0,
        raw_reward=1.0,
        value=2.0,
        log_prob=-0.5,
        done=False,
    ))
    buffer.add(PPOTransition(
        state_feat=dummy_feat,
        time_urgency=dummy_urgency,
        sample_record={},
        reward=2.0,
        raw_reward=2.0,
        value=3.0,
        log_prob=-0.6,
        done=False,
    ))
    buffer.add(PPOTransition(
        state_feat=dummy_feat,
        time_urgency=dummy_urgency,
        sample_record={},
        reward=3.0,
        raw_reward=3.0,
        value=4.0,
        log_prob=-0.7,
        done=True,
    ))

    buffer.finish_trajectory(last_value=0.0)

    # 理论推导:
    # t=2: done=True, delta_2 = 3.0 + 0 - 4.0 = -1.0, gae_2 = -1.0
    # t=1: non_terminal=1.0, delta_1 = 2.0 + 0.9 * 4.0 - 3.0 = 2.6
    #      gae_1 = 2.6 + 0.9 * 0.8 * (-1.0) = 2.6 - 0.72 = 1.88
    # t=0: non_terminal=1.0, delta_0 = 1.0 + 0.9 * 3.0 - 2.0 = 1.7
    #      gae_0 = 1.7 + 0.9 * 0.8 * 1.88 = 1.7 + 1.3536 = 3.0536
    assert abs(buffer.advantages[2].item() - (-1.0)) < 1e-4
    assert abs(buffer.advantages[1].item() - 1.88) < 1e-4
    assert abs(buffer.advantages[0].item() - 3.0536) < 1e-4

    # 目标价值: V_target = A + V
    assert abs(buffer.target_values[2].item() - 3.0) < 1e-4
    assert abs(buffer.target_values[1].item() - 4.88) < 1e-4
    assert abs(buffer.target_values[0].item() - 5.0536) < 1e-4


def test_buffer_gae_two_nonterminal_steps_match_hand_calculation() -> None:
    """非终止两步样本保留段尾bootstrap并按GAE递推。"""
    buffer = RolloutBufferWork3(gamma=0.9, gae_lambda=0.8, normalize_advantages=False)
    dummy_feat = torch.zeros(32)
    dummy_urgency = torch.zeros(2)

    for reward, value in ((1.0, 0.5), (2.0, 1.0)):
        buffer.add(PPOTransition(
            state_feat=dummy_feat,
            time_urgency=dummy_urgency,
            sample_record={},
            reward=reward,
            raw_reward=reward,
            value=value,
            log_prob=0.0,
            done=False,
            terminated=False,
        ))

    buffer.finish_trajectory(last_value=3.0)

    # delta=[1.4, 3.7]；末步A1=delta1，A0=delta0+0.9*0.8*A1。
    assert buffer.advantages.tolist() == pytest.approx([4.064, 3.7], abs=1e-6)
    deltas = [
        buffer.advantages[0].item() - 0.9 * 0.8 * buffer.advantages[1].item(),
        buffer.advantages[1].item(),
    ]
    assert deltas == pytest.approx([1.4, 3.7], abs=1e-6)


def test_buffer_gae_second_step_termination_stops_recursion() -> None:
    """两步样本第二步真实终止时不使用段尾bootstrap或后续优势。"""
    buffer = RolloutBufferWork3(gamma=0.9, gae_lambda=0.8, normalize_advantages=False)
    dummy_feat = torch.zeros(32)
    dummy_urgency = torch.zeros(2)

    for reward, value, terminated in ((1.0, 0.5, False), (2.0, 1.0, True)):
        buffer.add(PPOTransition(
            state_feat=dummy_feat,
            time_urgency=dummy_urgency,
            sample_record={},
            reward=reward,
            raw_reward=reward,
            value=value,
            log_prob=0.0,
            done=terminated,
            terminated=terminated,
        ))

    buffer.finish_trajectory(last_value=3.0)

    # delta1=A1=2-1=1；delta0=1+0.9*1-0.5=1.4；A0=1.4+0.72=2.12。
    assert buffer.advantages.tolist() == pytest.approx([2.12, 1.0], abs=1e-6)
    assert buffer.advantages[1].item() == pytest.approx(1.0, abs=1e-6)
    assert (
        buffer.advantages[0].item() - 0.9 * 0.8 * buffer.advantages[1].item()
        == pytest.approx(1.4, abs=1e-6)
    )


def test_buffer_batches_generator() -> None:
    """测试 Mini-batch 生成器输出维度与样本字典对齐。"""
    buffer = RolloutBufferWork3(gamma=0.99, gae_lambda=0.95, normalize_advantages=True)

    for i in range(10):
        buffer.add(PPOTransition(
            state_feat=torch.full((32,), float(i)),
            time_urgency=torch.tensor([float(i), -float(i)]),
            sample_record={"step_id": i},
            reward=1.0,
            raw_reward=1.0,
            value=float(i),
            log_prob=-0.5,
            done=(i == 9),
        ))

    with pytest.raises(RuntimeError):
        # 未调用 finish_trajectory 时抛出异常
        list(buffer.get_batches(batch_size=4))

    buffer.finish_trajectory(last_value=0.0)

    batches = list(buffer.get_batches(batch_size=4, shuffle=False))
    assert len(batches) == 3  # 4 + 4 + 2

    first_batch = batches[0]
    assert first_batch["state_feats"].shape == (4, 32)
    assert first_batch["time_urgencies"].shape == (4, 2)
    assert first_batch["advantages"].shape == (4,)
    assert first_batch["target_values"].shape == (4,)
    assert len(first_batch["sample_records"]) == 4
    # 验证字典对齐
    for k in range(4):
        assert first_batch["sample_records"][k]["step_id"] == k


def test_ppo_clipped_surrogate_matches_hand_calculation_for_both_advantage_signs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ratio=1.5、epsilon=0.2时正负优势分别按PPO最小代理项裁剪。"""
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=8)
    new_log_probs = torch.full((2,), math.log(1.5), dtype=torch.float32)

    def evaluate_fixed_policy(
        state_feats: torch.Tensor,
        time_urgencies: torch.Tensor,
        sample_records: list[dict[str, object]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert state_feats.shape == (2, 32)
        assert time_urgencies.shape == (2, 2)
        assert len(sample_records) == 2
        return torch.zeros(2), new_log_probs, torch.zeros(2)

    monkeypatch.setattr(actor, "evaluate_action_log_probs", evaluate_fixed_policy)
    trainer = PPOTrainerWork3(
        actor_critic=actor,
        clip_eps=0.2,
        vf_coef=0.0,
        ent_coef=0.0,
        create_optimizer=False,
    )
    batch = {
        "state_feats": torch.zeros((2, 32)),
        "time_urgencies": torch.zeros((2, 2)),
        "old_log_probs": torch.zeros(2),
        "advantages": torch.tensor([2.0, -2.0]),
        "target_values": torch.zeros(2),
        "sample_records": [{"case": "positive"}, {"case": "negative"}],
    }

    losses = trainer.compute_ppo_minibatch_loss(batch)

    assert torch.allclose(losses["ratio"], torch.tensor([1.5, 1.5]), atol=1e-6)
    # A=+2: min(1.5*2, 1.2*2)=2.4; A=-2: min(1.5*-2, 1.2*-2)=-3.
    # PPO loss为两项最小代理目标的负均值：-((2.4 + -3.0) / 2)=0.3。
    assert float(losses["policy_loss"]) == pytest.approx(0.3, abs=1e-6)
    assert float(losses["clip_fraction"]) == pytest.approx(1.0)


def test_ppo_trainer_train_step(env: AirLineEnvWork3) -> None:
    """测试 PPOTrainerWork3 在真实仿真交互数据上的端到端多步梯度更新。"""
    net = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    trainer = PPOTrainerWork3(actor_critic=net, lr=1e-3, clip_eps=0.2)
    buffer = RolloutBufferWork3(gamma=0.99, gae_lambda=0.95)

    # 记录初始参数快照
    initial_param = next(net.encoder.parameters()).clone()

    # 交互采集 8 个真实动作样本
    for step in range(8):
        cmax_est = compute_cycle_heuristic_cmax(env.state)
        s_feat = extract_compact_state_features(env.state, cmax_est)
        u_time = compute_time_urgency_vector(
            estimated_r=max(0.0, cmax_est - env.state.current_time),
            current_time=env.state.current_time,
            last_transfer_time=env.state.last_transfer_time,
            h0=env.state.h0,
        )

        act, lp, v, rec = net.select_action(env, s_feat, u_time, deterministic=False)
        if act is None:
            break

        obs, reward, terminated, truncated, info = env.step(act)
        done = terminated or truncated

        buffer.add(PPOTransition(
            state_feat=s_feat,
            time_urgency=u_time,
            sample_record=rec,
            reward=reward,
            raw_reward=reward,
            value=v,
            log_prob=lp,
            done=done,
            action_dict=act,
        ))

        if done:
            break

    assert len(buffer) > 0
    buffer.finish_trajectory(last_value=0.0)

    # 执行 PPO 训练更新
    metrics = trainer.train_step(buffer, ppo_epochs=3, batch_size=4)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "total_loss" in metrics
    assert "grad_norm" in metrics

    assert not math.isnan(metrics["total_loss"])
    assert not math.isnan(metrics["policy_loss"])
    assert not math.isnan(metrics["value_loss"])

    # 确认参数发生更新且无 NaN
    updated_param = next(net.encoder.parameters())
    assert not torch.equal(initial_param, updated_param)
    assert not torch.isnan(updated_param).any()


def test_ppo_trainer_checkpoint_save_load() -> None:
    """测试检查点保存与加载一致性。"""
    net1 = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    trainer1 = PPOTrainerWork3(actor_critic=net1, lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ppo.pt"
        trainer1.save_checkpoint(str(ckpt_path))
        assert ckpt_path.is_file()

        net2 = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
        trainer2 = PPOTrainerWork3(actor_critic=net2, lr=1e-3)
        trainer2.load_checkpoint(str(ckpt_path))

        p1 = next(net1.encoder.parameters())
        p2 = next(net2.encoder.parameters())
        assert torch.equal(p1, p2)
