"""Task 7.1 条件分支自回归解码网络专项单元测试。

验证点：
1. select_action 动作采样与环境可执行动作格式闭环；
2. 条件分支截断 (Conditional Branch Truncation)：
   - POSTPONE 后移分支动作截断，仅累加工序与站位概率；
   - STAY 留站分支完整累加工序、站位、选人序列与二元对齐概率；
3. 末站物理硬约束：末站任务绝对禁止后移 (POSTPONE 被严格掩码)；
4. PPO 概率重放契约：在采样时刻 evaluate_action_log_probs 计算的对数概率与采样对数概率严格相等，
   保证重要性采样比率 r_0(θ) 恒等于 1.000000。
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch

from envs.work3.core_types import ActionBranch
from envs.work3.environment import AirLineEnvWork3
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.actor_critic import ActorCriticWork3, extract_candidate_task_features


@pytest.fixture
def env() -> AirLineEnvWork3:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    env = AirLineEnvWork3(baseline_json_path=str(path))
    env.reset()
    return env


def test_candidate_task_feature_extraction(env: AirLineEnvWork3) -> None:
    """测试候选就绪工序特征提取形状与数值范围。"""
    ready = env.get_ready_tasks()
    assert len(ready) > 0

    feats = extract_candidate_task_features(env.state, ready)
    assert feats.shape == (len(ready), 8)
    assert not torch.isnan(feats).any()
    assert not torch.isinf(feats).any()


def test_select_action_and_ppo_replay_contract(env: AirLineEnvWork3) -> None:
    """测试动作采样与 PPO evaluate_action_log_probs 的数学严格一致性 (r_0 == 1.0)。"""
    net = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    net.eval()

    state_feat = torch.randn(32)
    time_urgency = torch.tensor([0.8, 0.5])

    # 采样 5 个不同动作样本
    samples = []
    actions = []
    sample_log_probs = []

    for _ in range(5):
        act, lp, v, rec = net.select_action(env, state_feat, time_urgency, deterministic=False)
        assert act is not None
        assert "task_key" in act
        assert "branch" in act
        samples.append(rec)
        actions.append(act)
        sample_log_probs.append(lp)

    # 批量执行重放
    state_feats_batch = state_feat.unsqueeze(0).expand(len(samples), -1)
    time_urgencies_batch = time_urgency.unsqueeze(0).expand(len(samples), -1)

    vals, replay_lps, ents = net.evaluate_action_log_probs(
        state_feats_batch,
        time_urgencies_batch,
        samples,
    )

    assert vals.shape == (len(samples),)
    assert replay_lps.shape == (len(samples),)
    assert ents.shape == (len(samples),)

    for i in range(len(samples)):
        sampled_lp = sample_log_probs[i]
        replay_lp = float(replay_lps[i].item())
        ratio = math.exp(replay_lp - sampled_lp)
        assert abs(ratio - 1.0) < 1e-5, f"PPO 重放比率偏离 1.0: 收到 {ratio} (采样={sampled_lp}, 重放={replay_lp})"


def test_conditional_branch_truncation(env: AirLineEnvWork3) -> None:
    """测试条件分支截断逻辑：POSTPONE 时动作被严格截断且不含选人和对齐。"""
    net = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)

    # 强制将 branch_head 偏置设为强烈偏向 POSTPONE (branch=1)
    with torch.no_grad():
        net.branch_head[-1].bias.data[0] = -100.0
        net.branch_head[-1].bias.data[1] = 100.0

    # 清空就绪工序的站内后继，确保符合 Rule 5 合法后移准则
    for t in env.get_ready_tasks():
        env._successors_map[t.aircraft_id][t.task_id] = []

    state_feat = torch.randn(32)
    time_urgency = torch.tensor([0.2, -0.5])

    act, lp, v, rec = net.select_action(env, state_feat, time_urgency, deterministic=True)
    assert act is not None

    candidate_tasks = env.get_action_candidates()
    chosen_task = candidate_tasks[rec["task_idx"]]

    if env.validate_postpone(chosen_task) is None:
        assert act["branch"] == ActionBranch.POSTPONE
        assert rec["branch"] == 1
        assert "team" not in act or act.get("team") == ()
        assert "align" not in act or act.get("align") == 0
    else:
        assert act["branch"] == ActionBranch.STATION_EXECUTE
        assert rec["branch"] == 0


def test_last_station_cannot_postpone(env: AirLineEnvWork3) -> None:
    """测试末站物理硬约束：末站工序绝对禁止后移，掩码必须屏蔽 POSTPONE。"""
    net = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)

    # 将首架飞机及相关活动工序直接移至末站 4
    k0 = 0
    env.state.aircraft[k0].current_station = 4
    for t in env.state.tasks.values():
        if t.aircraft_id == k0 and t.status.name == "READY":
            t.current_station = 4

    ready = env.get_ready_tasks()
    assert len(ready) > 0
    assert all(t.current_station == 4 for t in ready)

    state_feat = torch.randn(32)
    time_urgency = torch.tensor([0.1, 0.1])

    # 即使网络极其偏向后移，末站掩码仍必须强制其选择 STAY (branch=0)
    with torch.no_grad():
        net.branch_head[-1].bias.data[0] = -100.0
        net.branch_head[-1].bias.data[1] = 100.0

    act, lp, v, rec = net.select_action(env, state_feat, time_urgency, deterministic=True)
    assert act is not None
    assert act["branch"] == ActionBranch.STATION_EXECUTE
    assert rec["branch"] == 0
