from __future__ import annotations

from typing import Any

import pytest

from configs import Config
from runtime.configuration import validate_runtime_config


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    (
        ("update_every_episodes", 2, "update_every_episodes=1"),
        ("n_m", 8, "n_m.*最大为 7"),
        ("sample_temperature", 0.8, "sample_temperature=1"),
        ("ablation_no_mask", True, "ablation_no_mask 已禁用"),
    ),
)
def test_runtime_config_rejects_unsupported_training_values(
    field_name: str,
    invalid_value: Any,
    message: str,
) -> None:
    config = Config()
    setattr(config, field_name, invalid_value)

    with pytest.raises(ValueError, match=message):
        validate_runtime_config(config)


def test_runtime_config_double_buffer_guards() -> None:
    # 1. 默认关闭时合法
    config = Config()
    assert getattr(config, "rollout_double_buffer", False) is False
    validate_runtime_config(config)

    # 2. 开启且 num_envs < 2 报错
    bad_env_cfg = Config()
    bad_env_cfg.rollout_double_buffer = True
    bad_env_cfg.num_envs = 1
    bad_env_cfg.enable_rollout_ipc_fusion = True
    with pytest.raises(ValueError, match="rollout_double_buffer 要求 num_envs >= 2"):
        validate_runtime_config(bad_env_cfg)

    # 3. 开启且 enable_rollout_ipc_fusion=False 报错
    bad_fusion_cfg = Config()
    bad_fusion_cfg.rollout_double_buffer = True
    bad_fusion_cfg.num_envs = 4
    bad_fusion_cfg.enable_rollout_ipc_fusion = False
    with pytest.raises(ValueError, match="rollout_double_buffer 要求 enable_rollout_ipc_fusion=True"):
        validate_runtime_config(bad_fusion_cfg)

    # 4. 4/5/8 环境可正常配置通过
    for n in (2, 4, 5, 8):
        valid_cfg = Config()
        valid_cfg.rollout_double_buffer = True
        valid_cfg.num_envs = n
        valid_cfg.enable_rollout_ipc_fusion = True
        validate_runtime_config(valid_cfg)

