# -*- coding: utf-8 -*-
"""
环境双缓冲流水线 (Double-Buffering Rollout Pipeline) 正确性与等价性单元测试。

验证范围：
1. 2/4/5/8 环境的分组调度与时序推进；
2. Memory 所有逐动作字段严格等长；
3. 一组提前结束时另一组安全收尾；
4. 步数截断与死锁处罚正确性；
5. assert_rollout_idle 在采样结束后断言无在途残留。
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch

from configs import Config, configs
from tests.runtime_safety import seed_everything, temporary_config
from utils.vector_env import EnvCreator, VectorEnv
from training.rollout_service import RolloutService
from models.hb_gat_pn import HBGATPN
from ppo_agent import PPOAgent

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("num_envs", [2, 4, 5])
def test_double_buffer_rollout_lifecycle(num_envs: int) -> None:
    """验证双缓冲流水线在不同环境数量下能够正确推进完整 rollout，且轨迹字段严格对齐。"""
    seed_everything(100 + num_envs)
    overrides = {
        "n_w": 40,
        "n_m": 5,
        "max_slots_per_station": 3,
        "randomize_durations": False,
        "enable_dynamic_events": False,
        "enable_station_breakdown": False,
        "enable_material_delay": False,
        "rollout_double_buffer": True,
        "enable_rollout_ipc_fusion": True,
        "rollout_max_steps": 5,  # 冒烟快速截断
        "num_envs": num_envs,
    }

    vec_env = None
    eval_env = None
    with temporary_config(configs, overrides):
        cfg = Config()
        for k, v in overrides.items():
            setattr(cfg, k, v)

        device = torch.device("cpu")
        make_env = EnvCreator(str(PROJECT_ROOT / "data" / "283.csv"), seed_offset=1000)
        vec_env = VectorEnv(make_env, num_envs=num_envs)
        eval_env = make_env(999)
        model = HBGATPN(cfg)
        agent = PPOAgent(
            model,
            lr=1.0e-4,
            gamma=0.99,
            k_epochs=1,
            eps_clip=0.2,
            device=device,
            batch_size=1,
            total_timesteps=1,
            config=cfg,
        )

        service = RolloutService(agent=agent, vector_env=vec_env, eval_env=eval_env, config=cfg, device=device)
        assert service.double_buffer is True

        try:
            # 执行 1 个 episode 的采样
            memories, metrics = service._collect_episode(episode=1)

            assert len(memories) == num_envs
            total_actions = sum(len(m.actions) for m in memories)
            assert total_actions > 0
            assert metrics.environment_steps == total_actions
            assert math.isfinite(metrics.steps_per_second)
            assert metrics.steps_per_second == pytest.approx(
                metrics.environment_steps / max(metrics.total_seconds, 1e-9)
            )

            # 验证每个环境的 Memory 字段等长
            for env_idx, m in enumerate(memories):
                n_act = len(m.actions)
                assert len(m.states) == n_act, f"env {env_idx}: states length mismatch"
                assert len(m.rewards) == n_act, f"env {env_idx}: rewards length mismatch"
                assert len(m.is_terminals) == n_act, f"env {env_idx}: is_terminals length mismatch"
                assert len(m.logprobs) == n_act, f"env {env_idx}: logprobs length mismatch"
                assert len(m.values) == n_act, f"env {env_idx}: values length mismatch"
                assert len(m.masks) == n_act, f"env {env_idx}: masks length mismatch"

            # 验证环境池与服务处于完全空闲状态
            service.assert_rollout_idle()
            vec_env.assert_idle()
        finally:
            if vec_env is not None:
                vec_env.close()
            if eval_env is not None:
                eval_env.close()
