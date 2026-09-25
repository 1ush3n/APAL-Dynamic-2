"""工作三完整批次训练试点协议锁定测试。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from omegaconf import OmegaConf

from models.work3.actor_critic import ActorCriticWork3
from scripts.work3.train_ppo_work3 import (
    ROOT_DIR,
    _json_fingerprint,
    _module_fingerprint,
    build_episode_scenario_plan,
    build_worker_scenario_plan,
    load_training_scenarios,
)
from training.work3_runtime_config import (
    load_work3_runtime_config,
    resolved_config_fingerprint,
)


PILOT_CONFIG = ROOT_DIR / "conf" / "work3" / "pilot_trial_20260925.yaml"


def test_frozen_trial_config_records_protected_runtime_and_training_only_inputs() -> None:
    """试点预算、GPU精度、训练划分及D随机时间头初始化均明确锁定。"""
    assert PILOT_CONFIG.is_file(), "完整批次试点配置尚未冻结"
    config = load_work3_runtime_config(PILOT_CONFIG)

    assert config.runtime.run_mode == "pilot"
    assert config.runtime.method_profile == "C"
    assert config.runtime.seed == 42
    assert config.runtime.deterministic is True
    assert config.runtime.num_envs == 2
    assert config.runtime.start_method == "spawn"
    assert config.runtime.total_env_steps == 12000
    assert config.runtime.successful_batch_target == 1
    assert config.runtime.max_wall_seconds == 5400.0
    assert config.runtime.settle_timeout_seconds == 2.0
    assert config.runtime.device == "cuda:0"
    assert config.runtime.amp_dtype == "bf16"
    assert config.runtime.main_num_threads == 1
    assert config.runtime.env_num_threads == 1
    assert config.runtime.dataloader_num_workers == 0
    assert config.ppo.steps_per_iter == 64
    assert config.ppo.epochs == 1
    assert config.ppo.batch_size == 64
    assert config.protocol.run_order == ["C", "D"]
    assert config.protocol.training_split == "train"
    assert config.protocol.time_head_initialization == "random_no_pretraining"
    assert config.protocol.implementation_commit == "13ad3c9"
    assert config.protocol.research_result_eligible is False
    assert config.protocol.hardware.device_name == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert config.protocol.hardware.cuda_bf16_supported is True

    pretrained_path = ROOT_DIR / Path(config.paths.time_head_checkpoint)
    assert not pretrained_path.exists()
    assert hashlib.sha256(
        (ROOT_DIR / Path(config.paths.baseline)).read_bytes()
    ).hexdigest() == config.protocol.baseline_sha256
    assert hashlib.sha256(
        (ROOT_DIR / Path(config.runtime.scenario_split_path)).read_bytes()
    ).hexdigest() == config.protocol.scenario_split_sha256
    resolved_yaml, resolved_sha256 = resolved_config_fingerprint(config)
    assert "total_env_steps: 12000" in resolved_yaml
    assert len(resolved_sha256) == 64


def test_frozen_trial_plan_and_actor_fingerprint_match_recorded_values() -> None:
    """固定训练事件清单、worker映射和初始Actor可由配置重新生成。"""
    assert PILOT_CONFIG.is_file(), "完整批次试点配置尚未冻结"
    config = load_work3_runtime_config(PILOT_CONFIG)
    scenarios_path = ROOT_DIR / Path(config.runtime.scenario_pool_path)
    split_path = ROOT_DIR / Path(config.runtime.scenario_split_path)
    scenarios = load_training_scenarios(scenarios_path, split_path)
    episode_plan = build_episode_scenario_plan(
        scenarios,
        len(scenarios),
        seed=int(config.runtime.seed),
    )
    worker_plan = build_worker_scenario_plan(
        scenarios,
        num_workers=int(config.runtime.num_envs),
        episodes_per_worker=len(episode_plan),
        seed=int(config.runtime.seed),
    )
    protocol = config.protocol

    assert len(scenarios) == protocol.training_scenario_count == 25
    assert [item["scenario_id"] for item in episode_plan] == list(
        protocol.episode_plan_scenario_ids
    )
    assert _json_fingerprint(scenarios) == protocol.scenario_pool_sha256
    assert _json_fingerprint(episode_plan) == protocol.episode_plan_sha256
    assert _json_fingerprint(worker_plan) == protocol.worker_event_plan_sha256

    original_rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(int(config.runtime.seed))
        actor = ActorCriticWork3(
            state_dim=32,
            task_feat_dim=8,
            hidden_dim=64,
            max_station_workers=16,
        )
        assert _module_fingerprint({"actor_critic": actor}) == (
            protocol.initial_actor_fingerprint
        )
    finally:
        torch.set_rng_state(original_rng_state)


def test_c_and_d_trial_configs_differ_only_by_method_profile() -> None:
    """C/D以同一冻结配置运行，唯一resolved配置差异是方法profile。"""
    assert PILOT_CONFIG.is_file(), "完整批次试点配置尚未冻结"
    config_c = load_work3_runtime_config(PILOT_CONFIG)
    config_d = load_work3_runtime_config(
        PILOT_CONFIG,
        overrides=("runtime.method_profile=D",),
    )
    c_values = OmegaConf.to_container(config_c, resolve=True)
    d_values = OmegaConf.to_container(config_d, resolve=True)
    c_values["runtime"]["method_profile"] = "same-profile"
    d_values["runtime"]["method_profile"] = "same-profile"

    assert c_values == d_values
    _, c_sha256 = resolved_config_fingerprint(config_c)
    _, d_sha256 = resolved_config_fingerprint(config_d)
    assert c_sha256 != d_sha256
