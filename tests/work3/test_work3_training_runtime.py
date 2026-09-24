"""W3-13运行时的新配置、种子和并行训练边界测试。"""

from __future__ import annotations

import importlib
import importlib.util
import random
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT_DIR / "conf" / "work3" / "train_pilot.yaml"


@pytest.fixture
def restore_global_random_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def _runtime_config_module() -> ModuleType:
    """将缺少的新运行时模块呈现为清晰的测试失败，而不是导入错误。"""
    if importlib.util.find_spec("training.work3_runtime_config") is None:
        pytest.fail("工作三运行时配置加载器尚未实现", pytrace=False)
    return importlib.import_module("training.work3_runtime_config")


def test_runtime_config_loads_smoke_defaults_and_hashes_resolved_yaml() -> None:
    module = _runtime_config_module()

    config = module.load_work3_runtime_config(DEFAULT_CONFIG)
    resolved_yaml, fingerprint = module.resolved_config_fingerprint(config)

    assert config.runtime.run_mode == "smoke"
    assert config.runtime.method_profile == "C"
    assert config.runtime.num_envs == 1
    assert config.runtime.amp_dtype == "fp32"
    assert config.runtime.start_method == "spawn"
    assert config.runtime.dataloader_num_workers == 0
    assert config.runtime.total_env_steps == 64
    assert "num_envs: 1" in resolved_yaml
    assert len(fingerprint) == 64
    assert set(fingerprint) <= set("0123456789abcdef")


def test_runtime_config_overrides_change_resolved_values_and_fingerprint() -> None:
    module = _runtime_config_module()

    default = module.load_work3_runtime_config(DEFAULT_CONFIG)
    overridden = module.load_work3_runtime_config(
        DEFAULT_CONFIG,
        overrides=("runtime.seed=31415", "runtime.num_envs=2"),
    )
    _, default_fingerprint = module.resolved_config_fingerprint(default)
    resolved_yaml, override_fingerprint = module.resolved_config_fingerprint(overridden)

    assert overridden.runtime.seed == 31415
    assert overridden.runtime.num_envs == 2
    assert "seed: 31415" in resolved_yaml
    assert override_fingerprint != default_fingerprint


def test_runtime_config_rejects_missing_required_fields(tmp_path: Path) -> None:
    module = _runtime_config_module()
    incomplete_config = tmp_path / "incomplete.yaml"
    incomplete_config.write_text("runtime:\n  run_mode: smoke\n", encoding="utf-8")

    with pytest.raises(ValueError, match="缺少必需配置"):
        module.load_work3_runtime_config(incomplete_config)


@pytest.mark.parametrize(
    "override",
    (
        "runtime.num_envs=0",
        "runtime.amp_dtype=fp8",
        "runtime.total_env_steps=0",
        "runtime.dataloader_num_workers=1",
        "runtime.start_method=fork",
        "ppo.gamma=1.2",
        "paths.baseline=",
        "paths.baseline=' '",
    ),
)
def test_runtime_config_rejects_unsupported_or_out_of_range_values(override: str) -> None:
    module = _runtime_config_module()

    with pytest.raises((ValueError, TypeError)):
        module.load_work3_runtime_config(DEFAULT_CONFIG, overrides=(override,))


def test_runtime_seed_and_worker_episode_seed_are_reproducible(
    restore_global_random_state: None,
) -> None:
    module = _runtime_config_module()

    worker_seed = module.derive_worker_seed(42, worker_id=1, episode_index=3)
    assert worker_seed == module.derive_worker_seed(42, worker_id=1, episode_index=3)
    assert worker_seed != module.derive_worker_seed(42, worker_id=0, episode_index=3)
    assert worker_seed != module.derive_worker_seed(42, worker_id=1, episode_index=4)

    module.seed_work3_runtime(1234, deterministic=True)
    first = (random.random(), float(np.random.random()), torch.rand(3))
    module.seed_work3_runtime(1234, deterministic=True)
    second = (random.random(), float(np.random.random()), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.are_deterministic_algorithms_enabled()


def test_pilot_config_requires_a_positive_successful_batch_target() -> None:
    module = _runtime_config_module()

    with pytest.raises(ValueError, match="successful_batch_target"):
        module.load_work3_runtime_config(
            DEFAULT_CONFIG,
            overrides=("runtime.run_mode=pilot",),
        )

    pilot_config = module.load_work3_runtime_config(
        DEFAULT_CONFIG,
        overrides=(
            "runtime.run_mode=pilot",
            "runtime.successful_batch_target=1",
        ),
    )
    assert pilot_config.runtime.successful_batch_target == 1
