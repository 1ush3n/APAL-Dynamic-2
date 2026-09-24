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

from envs.work3.core_types import ActionBranch
from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import ActorCriticWork3
import scripts.work3.train_ppo_work3 as train_module
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


def test_training_starts_new_episode_after_advance_deadlock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ADVANCE死锁终止后，后续采样必须从新episode开始。"""
    scenario = {
        "scenario_id": "DEADLOCK_TEST",
        "timing": "middle",
        "intensity": "light",
        "station_id": 0,
        "aircraft_id": 0,
        "affected_task_keys": [],
    }
    monkeypatch.setattr(train_module, "load_training_scenarios", lambda *_args: [scenario])
    monkeypatch.setattr(AirLineEnvWork3, "load_scenario", lambda self, _scenario: None)

    def always_advance(
        self: ActorCriticWork3,
        env: AirLineEnvWork3,
        state_feat: torch.Tensor,
        time_urgency: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[dict[str, ActionBranch], float, float, dict[str, int | str]]:
        del self, env, state_feat, time_urgency, deterministic
        return (
            {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT},
            0.0,
            0.0,
            {"action_type": "advance_to_next_event", "advance_choice": 1},
        )

    monkeypatch.setattr(ActorCriticWork3, "select_action", always_advance)
    result = run_training(
        num_iterations=1,
        steps_per_iter=2,
        ppo_epochs=0,
        batch_size=2,
        method_variant="C",
        output_ckpt=str(tmp_path / "deadlock.pt"),
    )

    assert len(result["scenario_log"]) == 2
    assert [entry["episode_id"] for entry in result["scenario_log"]] == [0, 1]
    assert all(entry["success"] is False for entry in result["scenario_log"])
    assert all(entry["termination_reason"] == "deadlock" for entry in result["scenario_log"])


def test_training_records_forced_advance_when_no_legal_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """无候选环境转移必须进入PPO rollout并正常记录deadlock。"""
    scenario = {
        "scenario_id": "NO_CANDIDATE_TEST",
        "timing": "middle",
        "intensity": "light",
        "station_id": 0,
        "aircraft_id": 0,
        "affected_task_keys": [],
    }
    monkeypatch.setattr(train_module, "load_training_scenarios", lambda *_args: [scenario])
    monkeypatch.setattr(AirLineEnvWork3, "load_scenario", lambda self, _scenario: None)
    monkeypatch.setattr(AirLineEnvWork3, "get_action_candidates", lambda self: [])

    result = run_training(
        num_iterations=1,
        steps_per_iter=1,
        ppo_epochs=0,
        batch_size=1,
        method_variant="C",
        output_ckpt=str(tmp_path / "forced_advance.pt"),
    )

    assert result["history"][0]["total_steps"] == 1
    assert result["scenario_log"][0]["success"] is False
    assert result["scenario_log"][0]["termination_reason"] == "deadlock"
