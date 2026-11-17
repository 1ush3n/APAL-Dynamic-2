"""同结构QUERY行为克隆的损失、真实更新与检查点身份。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from apal.training.work3.behavior_cloning import (
    BC_RECORD_VERSION, BC_RECORD_SEMANTICS_VERSION, behavior_cloning_loss,
    load_demonstrations, run_behavior_cloning,
)


def test_bc_scope_and_instance_weighting_reject_missing_layers() -> None:
    parts = [{"task": torch.tensor(-2.0, requires_grad=True), "team": torch.tensor(-4.0, requires_grad=True)},
             {"task": torch.tensor(-6.0, requires_grad=True)}]
    samples = [{"instance_id": "A", "label_scope": ("task", "team")},
               {"instance_id": "B", "label_scope": ("task",)}]
    loss, _ = behavior_cloning_loss(parts, samples, {"A": 2, "B": 1})
    assert loss.item() == pytest.approx((3.0 * 0.75 + 6.0 * 1.5) / 2)
    loss.backward()
    assert parts[0]["task"].grad.item() == pytest.approx(-0.1875)
    with pytest.raises(ValueError, match="监督层"):
        behavior_cloning_loss(parts[:1], [{"instance_id": "A", "label_scope": ("align",)}], {"A": 1})


def test_demand_sampler_preserves_non_team_records_and_validates_weights() -> None:
    from torch.utils.data import WeightedRandomSampler
    from apal.training.work3.behavior_cloning import _demand_sampling_weights

    rows = [{"label_scope": ("team",), "record": {"chosen_team": tuple(range(demand))}}
            for demand in (1, 2, 3, 5)]
    rows.append({"label_scope": ("branch",), "record": {"chosen_team": (0, 1, 2, 3, 4)}})
    assert _demand_sampling_weights(rows, None) is None
    weights = _demand_sampling_weights(rows, {1: 0.62, 2: 3.0, 3: 0.18, 5: 1.89})
    assert weights.tolist() == [0.62, 3.0, 0.18, 1.89, 1.0]
    first = list(WeightedRandomSampler(weights, num_samples=len(rows), replacement=True,
                                       generator=torch.Generator().manual_seed(42)))
    second = list(WeightedRandomSampler(weights, num_samples=len(rows), replacement=True,
                                        generator=torch.Generator().manual_seed(42)))
    assert first == second and 1 in first
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="有限正数"):
            _demand_sampling_weights(rows, {2: invalid})


def test_bc_lightning_updates_and_preserves_query_checkpoint_contract(tmp_path: Path) -> None:
    from apal.envs.work3.environment import AirLineEnvWork3
    from apal.models.work3.actor_critic import ActorCriticWork3
    from apal.models.work3.ppo_trainer import PPOTrainerWork3

    torch.set_num_threads(1)
    torch.manual_seed(42)
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=16)
    env = AirLineEnvWork3()
    env.reset()
    snapshot = actor.make_decision_snapshot(env, torch.zeros(32), torch.zeros(2))
    _, _, _, record = actor.select_snapshot(snapshot, deterministic=True)
    entry = {"instance_id": "A", "parent_group": "A", "state_features": snapshot.state_features,
             "time_features": snapshot.time_features, "record": record,
             "label_scope": ("task",), "choice_layers": ("task",)}
    initial = tmp_path / "initial.pt"
    PPOTrainerWork3(actor, create_optimizer=False).save_checkpoint(str(initial),
        metadata={"stage": "initial", "method_profile": "QUERY"})
    for split in ("train", "validation"):
        sample = deepcopy(entry)
        if split == "validation":
            sample["instance_id"] = sample["parent_group"] = "B"
        torch.save({"version": BC_RECORD_VERSION, "record_semantics_version": BC_RECORD_SEMANTICS_VERSION,
                    "split": split, "policy_input_semantics": actor.input_semantics(), "records": [sample]},
                   tmp_path / f"{split}.pt")
    config = OmegaConf.create({"runtime": {"seed": 42, "deterministic": True, "device": "cpu",
                               "amp_dtype": "fp32", "main_num_threads": 1, "max_wall_seconds": 60.0},
                              "paths": {"train_records": str(tmp_path / "train.pt"),
                                        "validation_records": str(tmp_path / "validation.pt"),
                                        "initial_checkpoint": str(initial)},
                              "bc": {"hidden_dim": 16, "max_station_workers": 16, "batch_size": 1,
                                     "team_objective": "sequence",
                                     "max_updates": 3, "learning_rate": 0.003, "max_grad_norm": 0.5,
                                     "validate_every": 1, "patience": 4}})
    report = run_behavior_cloning(config, output=tmp_path / "bc.pt", report=tmp_path / "bc.json")
    assert report["optimization_steps"] == report["optimizer_step_attempts"] == 3
    assert report["amp_skipped_steps"] == 0 and report["parameters_changed"] > 0
    assert report["final_validation_nll"] < report["initial_validation_nll"]
    assert report["shared_encoder_has_gradient"] and not report["critic_has_gradient"]
    best = torch.load(tmp_path / "bc.pt", map_location="cpu", weights_only=False)
    assert best["run_metadata"]["stage"] == "bc"
    assert best["run_metadata"]["checkpoint_evaluation_eligible"] is False
    assert best["bc_training_state"]["optimizer_state"]
    assert "lightning_training_state" not in best
    assert all(torch.equal(value, best["actor_critic_state"][key])
               for key, value in actor.state_dict().items() if key.startswith("critic."))
    loaded = ActorCriticWork3(condition_queries=True, hidden_dim=16)
    PPOTrainerWork3(loaded, create_optimizer=False).load_checkpoint(str(tmp_path / "bc.pt"))
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in best["actor_critic_state"].items())
    broken = deepcopy(torch.load(tmp_path / "train.pt", weights_only=False))
    broken["policy_input_semantics"]["condition_normalization"] = "错误归一化"
    torch.save(broken, tmp_path / "broken.pt")
    with pytest.raises(ValueError, match="输入语义"):
        load_demonstrations(tmp_path / "broken.pt", actor, split="train")
