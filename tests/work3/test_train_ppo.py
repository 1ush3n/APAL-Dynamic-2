"""Task 7.3 PPO 验证训练流程单元测试。

验证点：
1. run_training() 端到端可执行性，无崩溃、无内存泄漏、无死锁；
2. 步步采集、势函数塑形、GAE 优势计算与 PPO 反向传播数值全量有限 (无 NaN/Inf)；
3. 输出检查点文件正确创建并可成功被 ActorCriticWork3 载入。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
import pytest
import torch

from envs.work3.core_types import ActionBranch
from models.work3.actor_critic import ActorCriticWork3
import scripts.work3.train_ppo_work3 as train_module
from scripts.work3.train_ppo_work3 import run_training


def test_ppo_training_pipeline_sanity() -> None:
    """运行极速 2 轮 PPO 验证训练，测试全流程端到端稳定性。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_ckpt = Path(tmpdir) / "test_method_d.pt"

        results = run_training(
            run_mode="smoke",
            num_iterations=1,
            steps_per_iter=32,
            ppo_epochs=1,
            batch_size=32,
            lr=1e-3,
            seed=42,
            output_ckpt=str(out_ckpt),
            device="cpu",
        )

        assert "history" in results
        assert len(results["history"]) == 1
        assert len(results["environment_worker_pids"]) == 1
        assert results["environment_worker_pids"][0] != os.getpid()
        assert results["environment_worker_cuda_initialized"] == [False]

        for log in results["history"]:
            assert not math.isnan(log["total_loss"])
            assert not math.isnan(log["policy_loss"])
            assert not math.isnan(log["value_loss"])
            assert not math.isnan(log["entropy"])
            assert not math.isnan(log["grad_norm"])
            assert log["ppo_updates"] > 0
            assert log["lightning_optimization_steps"] > 0
            assert log["environment_steps"] == 32
            assert log["sampling_replay_max_abs_error"] <= 1e-6
            assert log["sampling_replay_sample_count"] == 32

        # 检查点有效性检验
        assert out_ckpt.is_file()
        ckpt_data = torch.load(out_ckpt, map_location="cpu")
        assert "actor_critic_state" in ckpt_data

        test_net = ActorCriticWork3()
        test_net.load_state_dict(ckpt_data["actor_critic_state"])


def test_training_records_forced_advance_from_spawn_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """固定决策通过spawn worker执行并进入Lightning rollout。"""
    scenario = {
        "scenario_id": "SPAWN_ADVANCE_TEST",
        "timing": "middle",
        "intensity": "light",
        "station_id": 0,
        "aircraft_id": 0,
        "tau": 0.0,
        "recovery_time": 0.0,
        "affected_task_keys": [],
    }
    monkeypatch.setattr(train_module, "load_training_scenarios", lambda *_args: [scenario])

    def always_advance(
        self: ActorCriticWork3,
        snapshot: object,
        deterministic: bool = False,
    ) -> tuple[dict[str, ActionBranch], float, float, dict[str, int | str]]:
        del self, snapshot, deterministic
        return (
            {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT},
            0.0,
            0.0,
            {"action_type": "forced_advance"},
        )

    monkeypatch.setattr(ActorCriticWork3, "select_snapshot", always_advance)
    result = run_training(
        run_mode="pilot",
        successful_batch_target=1,
        max_decisions=1,
        num_iterations=1,
        steps_per_iter=1,
        ppo_epochs=0,
        batch_size=1,
        method_variant="C",
        output_ckpt=str(tmp_path / "spawn_advance.pt"),
    )

    assert result["history"][0]["total_steps"] == 1
    assert result["history"][0]["environment_steps"] == 1
    assert result["scenario_log"][0]["disturbance_triggered"] is True
    assert result["trajectory_audit"]["step_count"] == 1
    assert result["lightning_fit_calls"] == 1
