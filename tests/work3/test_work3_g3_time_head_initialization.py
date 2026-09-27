"""G3时间头初始化来源必须显式、确定且不得静默回退。"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.work3_runtime_config import (
    load_work3_runtime_config,
    validate_time_head_initialization,
    validate_work3_runtime_config,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "config_path",
    [
        ROOT / "conf/work3/train_pilot.yaml",
        ROOT / "conf/work3/pilot_trial_20260925.yaml",
        ROOT / "conf/work3/pilot_warmup_preflight_20260926.yaml",
    ],
)
def test_training_configs_use_explicit_random_initialization(config_path: Path) -> None:
    config = load_work3_runtime_config(config_path)
    assert config.runtime.time_head_initialization == "random_no_pretraining"
    assert config.paths.time_head_checkpoint is None


def test_random_initialization_rejects_any_checkpoint_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "independent_time_head.pt"
    checkpoint.touch()
    with pytest.raises(ValueError, match="随机初始化.*检查点路径必须为空"):
        validate_time_head_initialization("random_no_pretraining", checkpoint)


def test_explicit_paired_load_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="成对检查点不存在"):
        validate_time_head_initialization(
            "paired_checkpoint",
            tmp_path / "missing_pair.pt",
        )


def test_explicit_paired_load_rejects_checkpoint_until_protocol_exists(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "independent_or_unpaired.pt"
    checkpoint.write_bytes(b"not a paired checkpoint")
    with pytest.raises(ValueError, match="成对检查点协议尚未实现"):
        validate_time_head_initialization("paired_checkpoint", checkpoint)


def test_runtime_config_rejects_legacy_independent_head_path() -> None:
    config = load_work3_runtime_config(ROOT / "conf/work3/train_pilot.yaml")
    config.paths.time_head_checkpoint = "models/work3/checkpoints/time_head_best.pt"
    with pytest.raises(ValueError, match="随机初始化.*检查点路径必须为空"):
        validate_work3_runtime_config(config)
