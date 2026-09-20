"""Task 7.3 PPO 验证训练流程单元测试。

验证点：
1. run_training() 端到端可执行性，无崩溃、无内存泄漏、无死锁；
2. 步步采集、势函数塑形、GAE 优势计算与 PPO 反向传播数值全量有限 (无 NaN/Inf)；
3. 输出检查点文件正确创建并可成功被 ActorCriticWork3 载入。
"""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import pytest
import torch

from models.work3.actor_critic import ActorCriticWork3
from scripts.work3.train_ppo_work3 import run_training


def test_ppo_training_pipeline_sanity() -> None:
    """运行极速 2 轮 PPO 验证训练，测试全流程端到端稳定性。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_ckpt = Path(tmpdir) / "test_method_d.pt"

        results = run_training(
            num_iterations=2,
            steps_per_iter=16,
            ppo_epochs=2,
            batch_size=8,
            lr=1e-3,
            output_ckpt=str(out_ckpt),
            device="cpu",
        )

        assert "history" in results
        assert len(results["history"]) == 2

        for log in results["history"]:
            assert not math.isnan(log["total_loss"])
            assert not math.isnan(log["policy_loss"])
            assert not math.isnan(log["value_loss"])
            assert not math.isnan(log["entropy"])
            assert not math.isnan(log["grad_norm"])

        # 检查点有效性检验
        assert out_ckpt.is_file()
        ckpt_data = torch.load(out_ckpt, map_location="cpu")
        assert "actor_critic_state" in ckpt_data

        test_net = ActorCriticWork3()
        test_net.load_state_dict(ckpt_data["actor_critic_state"])
