"""工作三 条件分支 PPO 经验回放缓存 (Task 7.2)。

核心数学模型与技术规范：
依据《方法设计确认稿》第 4 节（条件分支强化学习）与标准 PPO 算法规范：
1. 存储多架次脉动节拍仿真中的步步决策转移 (Transition)：
   - state_feat: 32 维精炼物理状态特征张量；
   - time_urgency: 2 维显式时间紧迫度张量 [u_1, u_2]；
   - sample_record: 包含工序索引、分支决策、工人选择、二元对齐等重放元数据的字典；
   - reward: 经势函数塑形后的标量回报 R_t = r_t^raw + F_t；
   - value: Critic 估计的状态价值 V(s_t)；
   - log_prob: 采样时 Actor 输出的动作全量对数概率 log π_old(a_t|s_t)；
   - done: 是否为批次生产终局或周期截断。
2. 广义优势估计 (Generalized Advantage Estimation, GAE)：
   - δ_t = R_t + γ · V(s_{t+1}) · (1 - d_t) - V(s_t)
   - A_t = δ_t + (γ · λ) · (1 - d_t) · A_{t+1}
   - V_target(s_t) = A_t + V(s_t)
   - 批次内优势值标准化: A^_t = (A_t - μ_A) / (σ_A + 1e-8)
3. 随机 Mini-batch 生成器：
   - 严格保持张量特征与 sample_record 字典的对应关系，支持条件分支重放。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import torch


@dataclass
class PPOTransition:
    """单步 PPO 交互样本数据类。"""
    state_feat: torch.Tensor          # (32,)
    time_urgency: torch.Tensor        # (2,)
    sample_record: dict[str, Any]     # 重放字典 (含 task_idx, branch, cand_feats, etc.)
    reward: float                     # 塑形后步步奖励
    raw_reward: float                 # 原始物理步步奖励
    value: float                      # Critic 估计的 V(s_t)
    log_prob: float                   # 采样时刻的动作对数概率 log π_old
    done: bool = False                # 是否结束
    action_dict: dict[str, Any] = field(default_factory=dict)


class RolloutBufferWork3:
    """工作三 PPO 经验回放缓存。"""

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
    ) -> None:
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.normalize_advantages = normalize_advantages

        self.transitions: list[PPOTransition] = []
        self.advantages: torch.Tensor = torch.empty(0, dtype=torch.float)
        self.target_values: torch.Tensor = torch.empty(0, dtype=torch.float)
        self.is_finalized: bool = False

    def add(self, transition: PPOTransition) -> None:
        """向缓冲区追加一个决策转移样本。"""
        self.transitions.append(transition)
        self.is_finalized = False

    def finish_trajectory(self, last_value: float = 0.0) -> None:
        """计算全轨迹的 GAE 优势值与目标价值 V_target。

        Args:
            last_value: 轨迹末尾状态的估计价值 V(s_T)。若终局 done=True 则为 0.0。
        """
        n = len(self.transitions)
        if n == 0:
            return

        advantages = torch.zeros(n, dtype=torch.float)
        target_values = torch.zeros(n, dtype=torch.float)

        gae = 0.0
        next_value = float(last_value)

        # 逆序计算 GAE
        for t in reversed(range(n)):
            trans = self.transitions[t]
            non_terminal = 1.0 - float(trans.done)
            delta = trans.reward + (self.gamma * next_value * non_terminal) - trans.value
            gae = delta + (self.gamma * self.gae_lambda * non_terminal * gae)
            advantages[t] = gae
            target_values[t] = gae + trans.value
            next_value = trans.value

        # 优势值标准化
        if self.normalize_advantages and n > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        self.advantages = advantages
        self.target_values = target_values
        self.is_finalized = True

    def get_batches(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """生成 PPO 训练所需的 mini-batch 迭代器。

        Yields:
            字典包含:
              - "state_feats": torch.Tensor (B, 32)
              - "time_urgencies": torch.Tensor (B, 2)
              - "old_log_probs": torch.Tensor (B,)
              - "advantages": torch.Tensor (B,)
              - "target_values": torch.Tensor (B,)
              - "sample_records": list[dict] of length B
        """
        if not self.is_finalized:
            raise RuntimeError("请先调用 finish_trajectory 计算 GAE 优势值后再生成 mini-batch！")

        n = len(self.transitions)
        if n == 0:
            return

        state_feats = torch.stack([t.state_feat for t in self.transitions])
        time_urgencies = torch.stack([t.time_urgency for t in self.transitions])
        old_log_probs = torch.tensor([t.log_prob for t in self.transitions], dtype=torch.float)

        if shuffle:
            indices = torch.randperm(n)
        else:
            indices = torch.arange(n)

        for start_idx in range(0, n, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            batch_records = [self.transitions[idx].sample_record for idx in batch_idx]

            yield {
                "state_feats": state_feats[batch_idx],
                "time_urgencies": time_urgencies[batch_idx],
                "old_log_probs": old_log_probs[batch_idx],
                "advantages": self.advantages[batch_idx],
                "target_values": self.target_values[batch_idx],
                "sample_records": batch_records,
            }

    def clear(self) -> None:
        """清空缓冲区。"""
        self.transitions.clear()
        self.advantages = torch.empty(0, dtype=torch.float)
        self.target_values = torch.empty(0, dtype=torch.float)
        self.is_finalized = False

    def __len__(self) -> int:
        return len(self.transitions)
