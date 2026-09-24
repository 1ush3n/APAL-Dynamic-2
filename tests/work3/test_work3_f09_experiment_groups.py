"""W3-10/F09：训练扰动、正式 C/D 分组与检查点边界反例。"""

from __future__ import annotations

import json
from dataclasses import asdict
import hashlib
from pathlib import Path

import pytest

from scripts.work3.evaluate_c_vs_d import build_formal_evaluation_agent
from scripts.work3.experiment_protocol import build_method_profile
from scripts.work3.train_ppo_work3 import (
    build_episode_scenario_plan,
    load_training_scenarios,
    run_training,
)


def _save_profile_checkpoint(
    path: Path,
    method_variant: str,
    *,
    time_supervision_optimizer_updates: int = 0,
    config_hash_valid: bool = True,
) -> Path:
    import torch

    from models.work3.actor_critic import ActorCriticWork3
    from models.work3.ppo_trainer import PPOTrainerWork3
    from models.work3.time_head import TimeResidualHead

    profile = build_method_profile(method_variant)
    profile_data = asdict(profile)
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    time_head = (
        TimeResidualHead(in_dim=64, hidden_dim=64)
        if profile.use_time_auxiliary
        else None
    )
    config_yaml = f"runtime:\n  method_profile: {profile.name}\n"
    config_hash = hashlib.sha256(config_yaml.encode("utf-8")).hexdigest()
    metadata = {
        "run_mode": "pilot",
        "method_variant": profile.name,
        "method_profile": profile_data,
        "training_config": {
            "method_variant": profile.name,
            "method_profile": profile_data,
            "amp_dtype": "fp32",
            "lightning_precision": "32-true",
            "resolved_config_sha256": config_hash,
        },
        "resolved_runtime_config_yaml": config_yaml,
        "resolved_runtime_config_sha256": (
            config_hash if config_hash_valid else "0" * 64
        ),
        "source_sha": "9" * 40,
        "initial_actor_fingerprint": "a" * 64,
        "data_fingerprint": {
            "scenario_pool_sha256": "b" * 64,
            "scenario_split_sha256": "c" * 64,
            "baseline_sha256": "d" * 64,
            "event_plan_sha256": "e" * 64,
            "worker_event_plan_sha256": "f" * 64,
        },
        "successful_batch_count": 1,
        "lightning_optimization_steps": 1,
        "checkpoint_evaluation_eligible": True,
        "time_head_training_status": (
            "trained_online" if time_supervision_optimizer_updates else "untrained"
        ),
        "time_label_count": int(time_supervision_optimizer_updates > 0),
        "time_supervision_optimizer_updates": time_supervision_optimizer_updates,
        "potential_predictor_snapshot": (
            {
                "version": 2,
                "graph_feature_version": "work3_graph_v2",
                "actor_state": actor.state_dict(),
                "time_head_state": time_head.state_dict(),
                "time_head_model_version": "signed_residual_v1",
                "time_head_in_dim": 64,
            }
            if profile.use_time_auxiliary
            else None
        ),
    }
    trainer = PPOTrainerWork3(actor_critic=actor, time_head=time_head)
    trainer.save_checkpoint(
        str(path),
        metadata=metadata,
        training_state={
            "optimizer_state": torch.optim.AdamW(actor.parameters()).state_dict(),
            "precision_state": {
                "precision": "32-true",
                "grad_scaler_state": None,
            },
            "rng_state": {
                "python": None,
                "numpy": None,
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": None,
            },
            "event_plan_position": {"0": 1},
        },
    )
    return path


def test_formal_c_and_d_share_graph_policy_but_only_d_enables_time_learning() -> None:
    """正式 C/D 只能在确认的时间学习机制上不同。"""
    profile_c = build_method_profile("C")
    profile_d = build_method_profile("D")

    assert profile_c.graph_policy is True
    assert profile_d.graph_policy is True
    assert profile_c.allow_postpone is True
    assert profile_d.allow_postpone is True
    assert profile_c.use_time_auxiliary is False
    assert profile_d.use_time_auxiliary is True
    assert profile_c.use_corrected_time_input is False
    assert profile_d.use_corrected_time_input is True
    assert profile_c.use_learned_time_shaping is False
    assert profile_d.use_learned_time_shaping is True


def test_training_loads_explicit_split_and_builds_mixed_episode_plan(tmp_path: Path) -> None:
    """训练池来自固定清单，不从当前算法状态重新选择事件。"""
    scenarios = [
        {
            "scenario_id": "EARLY_LOW_S0",
            "timing": "EARLY",
            "intensity": "LOW",
            "station_id": 0,
            "aircraft_id": 5,
            "tau": 10.0,
            "delta": 1.0,
            "recovery_time": 11.0,
            "affected_task_keys": ["5_16"],
            "valid": True,
        },
        {
            "scenario_id": "LATE_HIGH_S4",
            "timing": "LATE",
            "intensity": "HIGH",
            "station_id": 4,
            "aircraft_id": 1,
            "tau": 20.0,
            "delta": 2.0,
            "recovery_time": 22.0,
            "affected_task_keys": ["1_16"],
            "valid": True,
        },
    ]
    scenarios_path = tmp_path / "scenarios.json"
    split_path = tmp_path / "train.json"
    scenarios_path.write_text(json.dumps(scenarios), encoding="utf-8")
    split_path.write_text(json.dumps([scenarios[1]]), encoding="utf-8")

    loaded = load_training_scenarios(str(scenarios_path), str(split_path))
    plan = build_episode_scenario_plan(loaded, num_episodes=4, seed=42)

    assert [item["scenario_id"] for item in loaded] == ["LATE_HIGH_S4"]
    assert len(plan) == 4
    assert {item["scenario_id"] for item in plan} == {"LATE_HIGH_S4"}
    assert all(item["timing"] == "LATE" for item in plan)
    assert all(item["intensity"] == "HIGH" for item in plan)


def test_formal_method_d_requires_checkpoint_unless_debug_mode(tmp_path: Path) -> None:
    """正式 D 不得把随机权重输出成训练后方法；调试随机权重必须显式开启。"""
    missing = tmp_path / "missing_method_d.pt"
    with pytest.raises(FileNotFoundError, match="正式方法 D"):
        build_formal_evaluation_agent(
            method_variant="D",
            checkpoint_path=missing,
            device="cpu",
            debug_random=False,
        )

    debug_agent = build_formal_evaluation_agent(
        method_variant="D",
        checkpoint_path=missing,
        device="cpu",
        debug_random=True,
    )
    assert debug_agent.debug_random is True
    assert debug_agent.profile.name == "D"


def test_formal_c_checkpoint_without_time_head_is_accepted(tmp_path: Path) -> None:
    checkpoint = _save_profile_checkpoint(tmp_path / "c.pt", "C")

    agent = build_formal_evaluation_agent("C", checkpoint, device="cpu")

    assert agent.debug_random is False
    assert agent.profile.name == "C"
    assert agent.time_head is None


def test_formal_d_rejects_checkpoint_with_untrained_time_head(tmp_path: Path) -> None:
    checkpoint = _save_profile_checkpoint(tmp_path / "d_untrained.pt", "D")

    with pytest.raises(ValueError, match="真实转站标签"):
        build_formal_evaluation_agent("D", checkpoint, device="cpu")


def test_formal_evaluation_rejects_checkpoint_from_other_profile(tmp_path: Path) -> None:
    checkpoint = _save_profile_checkpoint(
        tmp_path / "d_trained.pt",
        "D",
        time_supervision_optimizer_updates=1,
    )

    with pytest.raises(ValueError, match="profile不匹配"):
        build_formal_evaluation_agent("C", checkpoint, device="cpu")


def test_formal_evaluation_rejects_corrupt_resolved_config_hash(tmp_path: Path) -> None:
    checkpoint = _save_profile_checkpoint(
        tmp_path / "c_bad_hash.pt",
        "C",
        config_hash_valid=False,
    )

    with pytest.raises(ValueError, match="配置.*SHA256"):
        build_formal_evaluation_agent("C", checkpoint, device="cpu")


@pytest.mark.parametrize(
    ("metadata_path", "invalid_value"),
    [
        (("initial_actor_fingerprint",), "g" * 64),
        (("data_fingerprint", "event_plan_sha256"), None),
        (("data_fingerprint", "worker_event_plan_sha256"), "not-a-sha"),
    ],
)
def test_formal_evaluation_rejects_incomplete_or_malformed_run_fingerprints(
    tmp_path: Path,
    metadata_path: tuple[str, ...],
    invalid_value: str | None,
) -> None:
    import torch

    checkpoint_path = _save_profile_checkpoint(tmp_path / "c.pt", "C")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["run_metadata"]
    target = metadata
    for key in metadata_path[:-1]:
        target = target[key]
    target[metadata_path[-1]] = invalid_value
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="源码、初始化或数据指纹"):
        build_formal_evaluation_agent("C", checkpoint_path, device="cpu")


def test_formal_d_accepts_profiled_checkpoint_with_real_label_updates(
    tmp_path: Path,
) -> None:
    checkpoint = _save_profile_checkpoint(
        tmp_path / "d_trained.pt",
        "D",
        time_supervision_optimizer_updates=2,
    )

    agent = build_formal_evaluation_agent("D", checkpoint, device="cpu")

    assert agent.debug_random is False
    assert agent.profile.name == "D"
    assert agent.time_head is not None


def test_short_formal_c_training_loads_disturbance_and_records_episode(tmp_path: Path) -> None:
    """最小训练烟测必须真的加载扰动，而不是退回无扰动生产。"""
    result = run_training(
        run_mode="pilot",
        successful_batch_target=1,
        max_decisions=2,
        num_iterations=1,
        steps_per_iter=2,
        ppo_epochs=1,
        batch_size=2,
        method_variant="C",
        output_ckpt=str(tmp_path / "method_c.pt"),
        device="cpu",
    )

    assert result["method_variant"] == "C"
    assert result["scenario_log"]
    assert result["scenario_log"][0]["scenario_id"]
    assert result["scenario_log"][0]["timing"] in {"EARLY", "MID", "LATE"}
    assert isinstance(result["scenario_log"][0]["actual_hit_count"], int)
    entry = result["scenario_log"][0]
    train_pool = load_training_scenarios(
        "data/work3/scenarios_9class.json",
        "data/work3/experiment_splits/train.json",
    )
    assert entry["scenario_id"] in {item["scenario_id"] for item in train_pool}
    assert Path(result["training_config"]["scenario_split_path"]) == Path(
        "data/work3/experiment_splits/train.json"
    )
    assert entry["scheduled_target_count"] == len(entry["scheduled_affected_task_keys"])
    assert entry["scenario_id"] in result["planned_scenario_ids"]
    assert entry["disturbance_triggered"] is False
    assert entry["actual_hit_count"] == 0
    assert entry["actual_hit_task_keys"] == []


def test_training_log_separates_fixed_target_count_from_actual_hit_count(
    tmp_path: Path,
) -> None:
    """零时刻已揭示事件应记录预设目标、已触发状态和真实命中工序。"""
    scenario = {
        "scenario_id": "FIXED_TRAINING_LOG_HIT",
        "timing": "EARLY",
        "intensity": "LOW",
        "station_id": 0,
        "aircraft_id": 0,
        "tau": 0.0,
        "delta": 1.0,
        "recovery_time": 1.0,
        "affected_task_keys": ["0_15"],
        "valid": True,
    }
    scenarios_path = tmp_path / "scenarios.json"
    split_path = tmp_path / "train.json"
    scenarios_path.write_text(json.dumps([scenario]), encoding="utf-8")
    split_path.write_text(json.dumps([scenario]), encoding="utf-8")

    result = run_training(
        run_mode="pilot",
        successful_batch_target=1,
        max_decisions=1,
        num_iterations=1,
        steps_per_iter=1,
        ppo_epochs=1,
        batch_size=1,
        seed=42,
        method_variant="C",
        baseline_path="data/work3/real_283_k10_baseline.json",
        scenarios_path=scenarios_path,
        scenario_split_path=split_path,
        output_ckpt=tmp_path / "pilot.pt",
        report_path=tmp_path / "pilot.json",
        device="cpu",
    )

    entry = result["scenario_log"][0]
    assert entry["scheduled_target_count"] == 1
    assert entry["scheduled_affected_task_keys"] == ["0_15"]
    assert entry["disturbance_triggered"] is True
    assert entry["actual_hit_count"] == 1
    assert entry["actual_hit_task_keys"] == ["0_15"]
