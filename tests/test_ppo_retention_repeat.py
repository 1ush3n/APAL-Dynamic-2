from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from models.work3.actor_critic import ActorCriticWork3
from work3_4060.run_ppo_ablation_arms import apply_detach_value_hook
from work3_4060.run_ppo_retention_repeat import (
    build_arm_configs,
    delta_j,
    historical_interaction_budget,
    retention_pass,
    resolve_output_root,
    verify_first_rollout,
)


def test_seed_and_output_root_are_explicit_and_isolated() -> None:
    expected = Path(__file__).resolve().parents[1] / "work3_4060" / "ppo_retention_repeat_v1" / "ppo_seed43"
    assert resolve_output_root(expected) == expected.resolve()
    with pytest.raises(ValueError):
        resolve_output_root(expected.parent.parent / "ppo_ablation_arms_v1" / "seed42")


def test_r1_1024_config_changes_only_lr_seed_and_explicit_budget() -> None:
    from work3_4060.run_r1_retention_1024 import OUTPUT_ROOT, build_resolved_config, resolve_output_root

    source = {
        "method_variant": "C",
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "beta_shaping": 1.0,
        "learning_rate": 3e-4,
        "clip_epsilon": 0.2,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "ppo_epochs": 1,
        "batch_size": 2,
        "rollout_steps": 8,
        "num_envs": 1,
        "max_grad_norm": 0.5,
        "optimizer": "AdamW",
        "optimizer_eps": 1e-5,
        "weight_decay": 1e-4,
        "normalize_advantages": True,
        "amp_dtype": "fp32",
        "disturbance_enabled": False,
        "scenario": "NOMINAL",
        "warmup_mode": "none",
        "total_env_steps": 512,
        "max_decisions": 512,
    }
    resolved = build_resolved_config(source)
    assert resolved["effective_hyperparameters"]["learning_rate"] == pytest.approx(3e-5)
    assert resolved["effective_hyperparameters"]["seed"] == 43
    assert resolved["effective_hyperparameters"]["total_env_steps"] == 1024
    assert resolved["effective_hyperparameters"]["max_decisions"] == 1024
    assert resolved["points"] == [0, 128, 256, 512, 768, 1024]
    for key, value in source.items():
        if key not in {"learning_rate", "total_env_steps", "max_decisions"}:
            assert resolved["effective_hyperparameters"][key] == value
    assert resolve_output_root(OUTPUT_ROOT) == OUTPUT_ROOT.resolve()
    with pytest.raises(ValueError):
        resolve_output_root(OUTPUT_ROOT.parents[1] / "ppo_retention_repeat_v1" / "ppo_seed43")


def test_missing_training_dependency_is_reported_as_missing(tmp_path: Path) -> None:
    from work3_4060.run_r1_retention_1024 import inventory_dependency_paths

    rows, missing = inventory_dependency_paths({"calibrated_model": tmp_path / "absent.pt"})
    assert missing == ["calibrated_model"]
    assert rows[0]["exists"] is False
    assert rows[0]["sha256"] is None


def test_r1_report_gates_distinguish_failed_protocol_from_incomplete_evidence() -> None:
    from work3_4060.run_r1_retention_1024 import (
        _classify_protocol_status,
        _fixture_inputs_match,
        _pytest_summary_counts,
    )

    assert _fixture_inputs_match({"baseline": "same", "current": "same"})
    assert not _fixture_inputs_match({"baseline": "a", "current": "b"})
    assert _pytest_summary_counts("16 passed, 57 deselected in 1.2s") == {
        "passed": 16,
        "failed": 0,
        "skipped": 0,
    }
    assert _pytest_summary_counts("3 failed, 59 deselected in 1.2s") == {
        "passed": 0,
        "failed": 3,
        "skipped": 0,
    }
    checks = {
        "fixed_source_and_dependencies_unchanged": True,
        "regression_suite_passed": True,
        "fp32_equivalence_passed": True,
        "pretraining_archive_verified": True,
        "step0000_matches_bc_and_verifier": True,
        "first_rollout_paired": False,
        "first_minibatch_identity_matches": True,
        "r1_adamw_lr_contract": True,
        "run_json_effective_lr_and_budget_contract": True,
        "initial_parameters_complete_and_r1_has_no_r4_hook": True,
        "checkpoint_initial_fingerprints_match_source": True,
        "actual_training_budget_1024": True,
        "all_six_fixed_evaluations_present": False,
    }
    statuses = {"raw_cost": "INCOMPLETE"}
    assert _classify_protocol_status(
        checks,
        shaping_status="INCOMPLETE",
        objective_statuses=statuses,
        physical_status="INCOMPLETE",
    ) == "FAIL"
    checks["first_rollout_paired"] = True
    assert _classify_protocol_status(
        checks,
        shaping_status="INCOMPLETE",
        objective_statuses=statuses,
        physical_status="INCOMPLETE",
    ) == "INCOMPLETE"


def test_legacy_writer_paths_are_rebound_inside_new_output_root(monkeypatch: pytest.MonkeyPatch) -> None:
    from work3_4060.run_ppo_retention_repeat import _configure_runner
    import work3_4060.run_ppo_ablation_arms as runner

    expected = Path(__file__).resolve().parents[1] / "work3_4060" / "ppo_retention_repeat_v1" / "ppo_seed43"
    names = ("CAL_ROUND", "OUTPUT_DIR", "RUNS_DIR", "CAL_MODEL", "CAL_FIT", "REFERENCE_PPO",
             "REFERENCE_SUMMARY", "REFERENCE_HISTORY", "REFERENCE_PAIRING", "REFERENCE_FIRST_ROLLOUT",
             "BASELINE_PATH", "ARMS", "POINTS", "BUDGET", "RUN_SEED", "ARM_SPECS", "MANIFEST",
             "ARM_REGISTRY", "COMPARISON_CSV", "CYCLE_CSV", "DIAGNOSTICS_CSV", "VERIFICATION", "SUMMARY_MD")
    for name in names:
        monkeypatch.setattr(runner, name, getattr(runner, name))
    _configure_runner(expected)
    for arm in ("REF", "R1", "R4"):
        paths = runner.arm_paths(arm)
        assert paths["dir"].resolve().is_relative_to(expected.resolve())
        assert paths["run_json"].resolve() != (expected.parent.parent / "ppo_ablation_arms_v1" / "seed42" / arm / "runs" / "ppo_run.json").resolve()
    assert runner.REFERENCE_SUMMARY == expected.parents[1] / "critic_calibration_retention_v1" / "seed42" / "ppo" / "training_summary.json"


def test_r1_update_pairing_can_bind_prior_same_arm_minibatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import work3_4060.run_ppo_ablation_arms as runner

    source = tmp_path / "old_R1_first_update.json"
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path / "new_run")
    monkeypatch.setattr(runner, "REFERENCE_MINIBATCH_UPDATE", source)
    assert runner.minibatch_reference_path("R1") == source
    monkeypatch.setattr(runner, "REFERENCE_MINIBATCH_UPDATE", None)
    assert runner.minibatch_reference_path("R1") == tmp_path / "new_run" / "REF" / "first_update.json"


def test_arm_configs_only_scale_r1_lr_and_enable_r4_detach() -> None:
    source = {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "optimizer": "AdamW",
        "optimizer_eps": 1e-5,
        "weight_decay": 1e-4,
        "seed": 42,
    }
    arms = build_arm_configs(source, seed=43)
    assert set(arms) == {"REF", "R1", "R4"}
    assert {config["seed"] for config in arms.values()} == {43}
    assert arms["REF"]["learning_rate"] == 3e-4
    assert arms["R1"]["learning_rate"] == pytest.approx(3e-5, rel=0, abs=1e-20)
    assert arms["R4"]["learning_rate"] == 3e-4
    assert arms["R4"]["detach_value"] is True
    comparable = [
        {key: value for key, value in config.items() if key not in {"learning_rate", "detach_value"}}
        for config in arms.values()
    ]
    assert comparable[0] == comparable[1] == comparable[2]


def _rollout() -> dict[str, object]:
    return {
        "state_feats": torch.tensor([[1.0, 2.0]]),
        "snapshot_facts": [{"keys": ["0_1"], "branch_masks": [[True, False]], "worker_indices": [[0, 1]]}],
        "actions": [{"branch": 0, "task_key": "0_1", "team": [2, 3], "align": 1}],
        "old_log_probs": torch.tensor([-0.25]),
        "old_values": torch.tensor([0.5]),
        "raw_rewards": torch.tensor([0.0]),
        "shaped_rewards": torch.tensor([0.01]),
        "bootstrap_value": 0.4,
        "raw_advantages": torch.tensor([-0.1]),
        "advantages": torch.tensor([-1.0]),
        "return_targets": torch.tensor([0.4]),
        "pre_sampling_rng": {"torch_cpu": torch.tensor([7, 8], dtype=torch.uint8)},
        "pre_update_rng": {"torch_cpu": torch.tensor([9, 10], dtype=torch.uint8)},
        "minibatch_indices": [[2, 1, 0]],
    }


def test_pairing_uses_rollout_facts_and_rng_not_only_seed() -> None:
    reference = _rollout()
    paired = verify_first_rollout(reference, _rollout())
    assert paired["paired"] is True
    assert paired["steps_compared"] == 1

    candidate = _rollout()
    candidate["actions"] = [{"branch": 0, "task_key": "0_1", "team": [3, 2], "align": 1}]
    mismatch = verify_first_rollout(reference, candidate)
    assert mismatch["paired"] is False
    assert mismatch["first_difference"] == "actions"


def test_pairing_compares_ordered_sample_records_and_graph_facts() -> None:
    graph = HeteroData()
    graph["task"].x = torch.tensor([[1.0, 2.0]])
    graph["task", "precedes", "task"].edge_index = torch.tensor([[0], [0]])
    reference, candidate = _rollout(), _rollout()
    reference["sample_records"] = [{"graph_snapshot": graph.clone(), "worker_valid_masks": ((True, False),)}]
    candidate["sample_records"] = [{"graph_snapshot": graph.clone(), "worker_valid_masks": ((True, False),)}]
    assert verify_first_rollout(reference, candidate)["paired"] is True

    candidate["sample_records"][0]["worker_valid_masks"] = ((False, True),)
    mismatch = verify_first_rollout(reference, candidate)
    assert mismatch["paired"] is False
    assert mismatch["first_difference"] == "sample_records"


def test_retention_and_paired_delta_use_dynamic_evaluation_values() -> None:
    assert delta_j(0.7, 0.2) == pytest.approx(0.5)
    assert retention_pass({
        "completed_tasks": 210,
        "completed_aircraft": 7,
        "objective_j_total": 1e-4,
        "cost_components_weighted": {"j_takt": 1e-7},
        "postponed_count": 0,
    })
    assert not retention_pass({
        "completed_tasks": 210,
        "completed_aircraft": 7,
        "objective_j_total": 2e-3,
        "cost_components_weighted": {"j_takt": 1e-7},
        "postponed_count": 0,
    })
    assert historical_interaction_budget(210, updating_arms=4, per_arm_budget=512) == 2258


def test_value_detach_hook_cannot_be_installed_twice_and_can_be_removed() -> None:
    class TinyActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(2, 2)
            self.critic = nn.Linear(2, 1)

    model = TinyActorCritic()
    handle = apply_detach_value_hook(model)
    assert len(model.critic._forward_pre_hooks) == 1
    with pytest.raises(RuntimeError):
        apply_detach_value_hook(model)
    handle.remove()
    assert len(model.critic._forward_pre_hooks) == 0


def test_explicit_action_generator_is_reproducible_without_consuming_global_rng() -> None:
    distribution = Categorical(logits=torch.tensor([0.0, 1.0, -0.5]))
    before = torch.get_rng_state().clone()
    first = torch.Generator().manual_seed(43)
    second = torch.Generator().manual_seed(43)
    left = [int(ActorCriticWork3._sample_categorical(distribution, first).item()) for _ in range(12)]
    right = [int(ActorCriticWork3._sample_categorical(distribution, second).item()) for _ in range(12)]
    assert left == right
    assert torch.equal(before, torch.get_rng_state())
