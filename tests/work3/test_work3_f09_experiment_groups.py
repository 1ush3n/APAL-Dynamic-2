"""W3-10/F09：训练扰动、正式 C/D 分组与检查点边界反例。"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.work3.evaluate_c_vs_d import build_formal_evaluation_agent
from scripts.work3.experiment_protocol import build_method_profile
from scripts.work3.train_ppo_work3 import (
    build_episode_scenario_plan,
    load_training_scenarios,
    run_training,
)
from training.work3_runtime_config import load_work3_runtime_config


def _save_profile_checkpoint(
    path: Path,
    method_variant: str,
    *,
    time_supervision_optimizer_updates: int = 0,
    time_head_in_dim: int = 64,
    config_hash_valid: bool = True,
    include_potential_snapshot: bool | None = None,
) -> Path:
    import torch

    from models.work3.actor_critic import ActorCriticWork3
    from models.work3.ppo_trainer import PPOTrainerWork3
    from models.work3.time_head import TimeResidualHead

    profile = build_method_profile(method_variant)
    profile_data = asdict(profile)
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    time_head = (
        TimeResidualHead(in_dim=time_head_in_dim, hidden_dim=64)
        if profile.use_time_auxiliary
        else None
    )
    config_yaml = f"runtime:\n  method_profile: {profile.name}\n"
    config_hash = hashlib.sha256(config_yaml.encode("utf-8")).hexdigest()
    has_potential_snapshot = (
        profile.use_learned_time_shaping
        if include_potential_snapshot is None
        else include_potential_snapshot
    )
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
                "time_head_in_dim": time_head_in_dim,
            }
            if has_potential_snapshot
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


@pytest.mark.parametrize(
    ("variant", "auxiliary", "corrected_input", "learned_shaping"),
    [
        ("C", False, False, False),
        ("E", True, False, False),
        ("F", True, True, False),
        ("G", True, False, True),
        ("D", True, True, True),
    ],
)
def test_all_confirmed_profiles_keep_graph_and_postpone_and_only_toggle_time_modules(
    variant: str,
    auxiliary: bool,
    corrected_input: bool,
    learned_shaping: bool,
) -> None:
    """C/D/E/F/G共用图策略与后移，只按确认稿切换时间机制。"""
    profile = build_method_profile(variant)

    assert profile.name == variant
    assert profile.graph_policy is True
    assert profile.allow_postpone is True
    assert profile.use_time_auxiliary is auxiliary
    assert profile.use_corrected_time_input is corrected_input
    assert profile.use_learned_time_shaping is learned_shaping


@pytest.mark.parametrize("variant", ["E", "F", "G"])
def test_yaml_runtime_config_accepts_confirmed_ablation_profiles(variant: str) -> None:
    config = load_work3_runtime_config(
        Path("conf/work3/train_pilot.yaml"),
        overrides=(f"runtime.method_profile={variant}",),
    )

    assert config.runtime.method_profile == variant


def test_training_cli_accepts_ablation_profile_override() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/work3/train_ppo_work3.py",
            "--method",
            "E",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("variant", ["E", "G"])
def test_heuristic_input_profiles_do_not_call_corrected_time_head(
    variant: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E/G仍训练时间头，但Actor动作输入保持启发式值。"""
    from envs.work3.environment import AirLineEnvWork3
    from models.work3.actor_critic import ActorCriticWork3
    from models.work3.time_head import TimeResidualHead
    from scripts.work3.evaluate_c_vs_d import FormalEvaluationAgent

    profile = build_method_profile(variant)
    agent = FormalEvaluationAgent(
        profile,
        ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64),
        TimeResidualHead(in_dim=64, hidden_dim=64),
    )
    env = AirLineEnvWork3(
        baseline_json_path="data/work3/real_283_k10_baseline.json"
    )
    env.reset()

    def reject_prediction(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        pytest.fail("E/G不得用学习时间头修正Actor动作输入")

    monkeypatch.setattr(TimeResidualHead, "predict_corrected_time", reject_prediction)
    agent.select_action(env)

    assert agent.last_time_prediction is not None
    assert agent.last_time_prediction[1] == agent.last_time_prediction[2]


@pytest.mark.parametrize(
    ("variant", "corrected_input", "learned_shaping", "snapshot_version"),
    [
        ("E", False, False, None),
        ("F", True, False, None),
        ("G", False, True, 0),
    ],
)
def test_ablation_training_caches_time_labels_independent_of_shaping_mode(
    tmp_path: Path,
    variant: str,
    corrected_input: bool,
    learned_shaping: bool,
    snapshot_version: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E/F无学习塑形也缓存标签；G的塑形版本绑定到episode。"""
    import scripts.work3.train_ppo_work3 as training_module

    corrected_input_calls = 0
    original_compute = training_module.compute_online_snapshot_time_inputs

    def track_corrected_input(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        nonlocal corrected_input_calls
        corrected_input_calls += 1
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(
        training_module,
        "compute_online_snapshot_time_inputs",
        track_corrected_input,
    )
    scenario = {
        "scenario_id": f"{variant}_AUXILIARY_PENDING_LABEL",
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
    scenario_path = tmp_path / f"{variant}_scenarios.json"
    scenario_path.write_text(json.dumps([scenario]), encoding="utf-8")
    split_path = tmp_path / f"{variant}_train.json"
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
        method_variant=variant,
        baseline_path="data/work3/real_283_k10_baseline.json",
        scenarios_path=scenario_path,
        scenario_split_path=split_path,
        output_ckpt=tmp_path / f"{variant}.pt",
        report_path=tmp_path / f"{variant}.json",
        device="cpu",
        num_envs=1,
    )

    assert result["method_profile"]["use_time_auxiliary"] is True
    assert result["method_profile"]["use_corrected_time_input"] is corrected_input
    assert result["method_profile"]["use_learned_time_shaping"] is learned_shaping
    assert result["scenario_log"][0]["potential_snapshot_version"] == snapshot_version
    assert (corrected_input_calls > 0) is corrected_input
    cycle_samples = sum(
        item["label_count"] + item["pending_decision_count"]
        for item in result["cycle_time_labels"]
    )
    assert cycle_samples > 0


@pytest.mark.parametrize("variant", ["E", "F", "G"])
def test_yaml_runtime_config_accepts_confirmed_ablation_profiles(variant: str) -> None:
    config = load_work3_runtime_config(
        Path("conf/work3/train_pilot.yaml"),
        overrides=(f"runtime.method_profile={variant}",),
    )

    assert config.runtime.method_profile == variant


def test_training_cli_accepts_ablation_profile_override() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/work3/train_ppo_work3.py",
            "--method",
            "E",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("variant", ["E", "F"])
def test_auxiliary_ablation_collects_cycle_samples_without_learned_shaper(
    tmp_path: Path,
    variant: str,
) -> None:
    """E/F不启用学习塑形，仍须缓存决策图以等待实际转站标签。"""
    scenario = {
        "scenario_id": f"{variant}_AUXILIARY_PENDING_LABEL",
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
    scenario_path = tmp_path / f"{variant}_scenarios.json"
    scenario_path.write_text(json.dumps([scenario]), encoding="utf-8")
    split_path = tmp_path / f"{variant}_train.json"
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
        method_variant=variant,
        baseline_path="data/work3/real_283_k10_baseline.json",
        scenarios_path=scenario_path,
        scenario_split_path=split_path,
        output_ckpt=tmp_path / f"{variant}.pt",
        report_path=tmp_path / f"{variant}.json",
        device="cpu",
        num_envs=1,
    )

    assert result["method_profile"]["use_time_auxiliary"] is True
    assert result["method_profile"]["use_learned_time_shaping"] is False
    cycle_samples = sum(
        item["label_count"] + item["pending_decision_count"]
        for item in result["cycle_time_labels"]
    )
    assert cycle_samples > 0


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


def test_formal_d_rejects_offline_m4_time_head_dimension(tmp_path: Path) -> None:
    """离线M4的32维输入权重不能作为正式图表征D的时间头。"""
    checkpoint = _save_profile_checkpoint(
        tmp_path / "d_with_offline_m4_head.pt",
        "D",
        time_supervision_optimizer_updates=1,
        time_head_in_dim=32,
    )

    with pytest.raises(ValueError, match="缺少兼容的有符号时间头"):
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


@pytest.mark.parametrize("variant", ["E", "F"])
def test_formal_ef_accept_trained_time_head_without_shaping_snapshot(
    tmp_path: Path,
    variant: str,
) -> None:
    checkpoint = _save_profile_checkpoint(
        tmp_path / f"{variant.lower()}_trained.pt",
        variant,
        time_supervision_optimizer_updates=1,
    )

    agent = build_formal_evaluation_agent(variant, checkpoint, device="cpu")

    assert agent.profile.name == variant
    assert agent.time_head is not None


def test_formal_g_accepts_trained_time_head_with_frozen_shaping_snapshot(
    tmp_path: Path,
) -> None:
    checkpoint = _save_profile_checkpoint(
        tmp_path / "g_trained.pt",
        "G",
        time_supervision_optimizer_updates=1,
    )

    agent = build_formal_evaluation_agent("G", checkpoint, device="cpu")

    assert agent.profile.name == "G"
    assert agent.time_head is not None


@pytest.mark.parametrize("variant", ["D", "G"])
def test_learned_shaping_profiles_reject_checkpoint_without_frozen_snapshot(
    tmp_path: Path,
    variant: str,
) -> None:
    checkpoint = _save_profile_checkpoint(
        tmp_path / f"{variant.lower()}_no_snapshot.pt",
        variant,
        time_supervision_optimizer_updates=1,
        include_potential_snapshot=False,
    )

    with pytest.raises(ValueError, match="势函数预测器快照"):
        build_formal_evaluation_agent(variant, checkpoint, device="cpu")


@pytest.mark.parametrize("variant", ["E", "F"])
def test_non_shaping_profiles_reject_hidden_learned_shaping_snapshot(
    tmp_path: Path,
    variant: str,
) -> None:
    checkpoint = _save_profile_checkpoint(
        tmp_path / f"{variant.lower()}_unexpected_snapshot.pt",
        variant,
        time_supervision_optimizer_updates=1,
        include_potential_snapshot=True,
    )

    with pytest.raises(ValueError, match="不得包含学习势函数预测器快照"):
        build_formal_evaluation_agent(variant, checkpoint, device="cpu")


def test_formal_d_checkpoint_reload_preserves_prediction_and_policy_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一评测状态上保存重载前后的时间预测与策略输出一致。"""
    import torch

    from envs.work3.environment import AirLineEnvWork3
    from models.work3.actor_critic import ActorCriticWork3

    checkpoint = _save_profile_checkpoint(
        tmp_path / "formal_d.pt",
        "D",
        time_supervision_optimizer_updates=2,
    )
    first = build_formal_evaluation_agent("D", checkpoint, device="cpu")
    reloaded = build_formal_evaluation_agent("D", checkpoint, device="cpu")
    assert first.time_head is not None and reloaded.time_head is not None

    captured: dict[int, tuple[torch.Tensor, torch.Tensor, tuple[Any, ...]]] = {}
    original_select = ActorCriticWork3.select_action

    def capture_select(
        actor: ActorCriticWork3,
        env: Any,
        state_feat: torch.Tensor,
        time_urgency: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[Any, ...]:
        output = original_select(
            actor,
            env,
            state_feat,
            time_urgency,
            deterministic=deterministic,
        )
        captured[id(actor)] = (
            state_feat.detach().cpu().clone(),
            time_urgency.detach().cpu().clone(),
            output,
        )
        return output

    monkeypatch.setattr(ActorCriticWork3, "select_action", capture_select)
    env = AirLineEnvWork3(
        baseline_json_path="data/work3/real_283_k10_baseline.json"
    )
    env.reset()
    first_action = first.select_action(env)
    first_prediction = first.last_time_prediction
    reloaded_action = reloaded.select_action(env)
    reloaded_prediction = reloaded.last_time_prediction

    assert first_prediction is not None and reloaded_prediction is not None
    assert first_prediction == pytest.approx(reloaded_prediction, abs=1e-6)
    assert first_action == reloaded_action
    first_input = captured[id(first.actor_critic)]
    reloaded_input = captured[id(reloaded.actor_critic)]
    assert torch.equal(first_input[0], reloaded_input[0])
    assert torch.equal(first_input[1], reloaded_input[1])
    first_output = first_input[2]
    reloaded_output = reloaded_input[2]
    assert first_output[0] == reloaded_output[0]
    assert first_output[1] == pytest.approx(reloaded_output[1], abs=1e-6)
    assert first_output[2] == pytest.approx(reloaded_output[2], abs=1e-6)

    first_value, first_log_prob, first_entropy = (
        first.actor_critic.evaluate_action_log_probs(
            first_input[0].unsqueeze(0),
            first_input[1].unsqueeze(0),
            [first_output[3]],
        )
    )
    reload_value, reload_log_prob, reload_entropy = (
        reloaded.actor_critic.evaluate_action_log_probs(
            reloaded_input[0].unsqueeze(0),
            reloaded_input[1].unsqueeze(0),
            [reloaded_output[3]],
        )
    )
    assert torch.allclose(first_value, reload_value, atol=1e-6, rtol=1e-6)
    assert torch.allclose(first_log_prob, reload_log_prob, atol=1e-6, rtol=1e-6)
    assert torch.allclose(first_entropy, reload_entropy, atol=1e-6, rtol=1e-6)


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


def test_m08_repeated_benchmark_uses_same_checkpoints_and_manifest(
    tmp_path: Path,
) -> None:
    """测试用profile检查点只验复现链路，不代表训练后模型或科研评测结果。"""
    import torch

    from scripts.work3.evaluate_c_vs_d import run_benchmark_evaluation
    from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline

    baseline_source_path = Path("data/work3/real_283_k10_baseline.json")
    if not baseline_source_path.is_file():
        pytest.skip(f"基准实例不存在: {baseline_source_path}")
    source = MultiAircraftBaseline.load_from_json(baseline_source_path)
    fixed_tasks = ((1, 15), (2, 18), (3, 12), (4, 20), (5, 24))
    tasks: dict[str, Any] = {}
    for station_id, task_id in fixed_tasks:
        task = source.get_task(0, task_id)
        assert task.station_id == station_id
        tasks[task.task_key] = replace(
            task,
            in_station_offset=0.0,
            baseline_start=0.0,
            baseline_end=task.duration,
        )
    small_baseline_path = tmp_path / "m08_baseline.json"
    MultiAircraftBaseline(
        num_aircraft=1,
        num_stations=5,
        h0=source.h0,
        tasks=tasks,
        station_workers=source.station_workers,
    ).save_to_json(small_baseline_path)

    scenario = {
        "scenario_id": "M08_FIXED_EVENT",
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
    scenario_pool_path = tmp_path / "scenario_pool.json"
    manifest_path = tmp_path / "test_manifest.json"
    scenario_pool_path.write_text(
        json.dumps([scenario], sort_keys=True), encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps([{"scenario_id": scenario["scenario_id"]}], sort_keys=True),
        encoding="utf-8",
    )

    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(42)
        c_checkpoint = _save_profile_checkpoint(tmp_path / "method_c.pt", "C")
        torch.manual_seed(42)
        d_checkpoint = _save_profile_checkpoint(
            tmp_path / "method_d.pt",
            "D",
            time_supervision_optimizer_updates=1,
        )
    finally:
        torch.set_rng_state(rng_state)

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    input_fingerprints = {
        "baseline_sha256": sha256(small_baseline_path),
        "scenario_pool_sha256": sha256(scenario_pool_path),
        "scenario_manifest_sha256": sha256(manifest_path),
        "method_c_checkpoint_sha256": sha256(c_checkpoint),
        "method_d_checkpoint_sha256": sha256(d_checkpoint),
    }
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        reports: list[dict[str, Any]] = []
        for run_name in ("first", "second"):
            reports.append(
                run_benchmark_evaluation(
                    baseline_path=str(small_baseline_path),
                    scenarios_path=str(scenario_pool_path),
                    scenario_split_path=manifest_path,
                    method_c_ckpt=c_checkpoint,
                    method_d_ckpt=d_checkpoint,
                    output_json=tmp_path / f"m08_{run_name}.json",
                    device="cpu",
                    max_decisions=32,
                    warmup_mode="none",
                )
            )
    finally:
        torch.use_deterministic_algorithms(deterministic_before)

    for run_name, report in zip(("first", "second"), reports, strict=True):
        assert report["evaluation_inputs"] == input_fingerprints
        persisted = json.loads(
            (tmp_path / f"m08_{run_name}.json").read_text(encoding="utf-8")
        )
        assert persisted["evaluation_inputs"] == input_fingerprints
        protocol = report["evaluation_protocol"]
        assert protocol["device"] == "cpu"
        assert protocol["torch_deterministic_algorithms"] is True
        assert protocol["policy_action_selection"] == "greedy_argmax"
        assert "cudnn_deterministic" in protocol
        assert "cudnn_benchmark" in protocol
        assert [item["scenario_id"] for item in report["scenario_results"]] == [
            scenario["scenario_id"]
        ]

    def assert_semantically_equal(actual: Any, expected: Any, path: str = "report") -> None:
        ignored_timing_fields = {"evaluation_time_s", "time_c_s", "time_d_s"}
        if isinstance(actual, dict) and isinstance(expected, dict):
            assert actual.keys() == expected.keys(), path
            for key in actual.keys() - ignored_timing_fields:
                assert_semantically_equal(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(actual, list) and isinstance(expected, list):
            assert len(actual) == len(expected), path
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                assert_semantically_equal(left, right, f"{path}[{index}]")
        elif isinstance(actual, float) and isinstance(expected, float):
            assert actual == pytest.approx(expected, rel=0.0, abs=1e-6), path
        else:
            assert actual == expected, path

    assert_semantically_equal(reports[0], reports[1])
