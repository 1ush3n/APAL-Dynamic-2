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
