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


def test_actual_batch_sampling_audit_preserves_order_and_scope_counts() -> None:
    from apal.training.work3.behavior_cloning import _sampling_audit

    rows = [{"instance_id": "train_024", "label_scope": ("task", "team", "align"),
             "record": {"task_key": "0_2", "branch": 0, "chosen_team": (3, 5), "align": 1}},
            {"instance_id": "train_001", "label_scope": ("branch",),
             "record": {"task_key": "0_4", "branch": 1}}]
    rng = torch.get_rng_state().clone()
    first = _sampling_audit(rows, [7, 3])
    assert first["scope_counts"] == {"task": 1, "team": 1, "align": 1, "branch": 1}
    assert first["train_024_0_2_count"] == 1
    assert first["sample_indices"] == [7, 3]
    assert first["sample_index_audit_complete"]
    assert not _sampling_audit(rows, [None, 3])["sample_index_audit_complete"]
    assert len(first["sample_sequence_sha256"]) == 64
    assert _sampling_audit(deepcopy(rows), [7, 3]) == first
    assert _sampling_audit(rows, [8, 3])["sample_sequence_sha256"] != first["sample_sequence_sha256"]
    assert _sampling_audit(list(reversed(rows)), [3, 7])["sample_sequence_sha256"] != first["sample_sequence_sha256"]
    changed = deepcopy(rows)
    changed[0]["record"]["chosen_team"] = (5, 3)
    assert _sampling_audit(changed, [7, 3])["sample_sequence_sha256"] != first["sample_sequence_sha256"]
    changed = deepcopy(rows)
    changed[0]["record"]["graph_snapshot"] = object()
    assert _sampling_audit(changed, [7, 3]) == first
    assert torch.equal(torch.get_rng_state(), rng)
    from apal.models.work3.actor_critic import ActorCriticWork3
    from apal.training.work3.behavior_cloning import Work3BCModule
    repeated = Work3BCModule(ActorCriticWork3(condition_queries=True, hidden_dim=8), [rows[0], rows[0]],
        OmegaConf.create({"learning_rate": 0.0003, "max_grad_norm": 0.5, "batch_size": 1}))
    assert not repeated.sample_index_audit_complete
    assert repeated._sample_indices[id(rows[0])] is None


@pytest.mark.parametrize("training", [False, True])
def test_diagnostic_checkpoint_preserves_rng_mode_steps_and_outputs(tmp_path: Path, training: bool) -> None:
    import random
    import numpy as np
    from apal.models.work3.actor_critic import ActorCriticWork3
    from apal.models.work3.ppo_trainer import PPOTrainerWork3
    from apal.training.work3.behavior_cloning import Work3BCModule, _Selection
    from .test_behavior_cloning import _teacher_state

    actor, snapshot, teacher = _teacher_state(tmp_path)
    _, _, _, record = actor.select_snapshot(snapshot, teacher_action=teacher)
    config = OmegaConf.create({"runtime": {"max_wall_seconds": 60.0},
        "bc": {"learning_rate": 0.0003, "max_grad_norm": 0.5, "batch_size": 1,
               "validate_every": 25, "max_updates": 200, "patience": 4,
               "diagnostic_checkpoint_dir": str(tmp_path / "diagnostics")}})
    module = Work3BCModule(actor, [{"instance_id": "A"}], config.bc)
    actor.train(training)
    state, times = snapshot.state_features.unsqueeze(0), snapshot.time_features.unsqueeze(0)
    with torch.no_grad():
        expected = actor.evaluate_action_log_probs(state, times, [record])
    original = {key: value.clone() for key, value in actor.state_dict().items()}
    explicit = torch.Generator().manual_seed(42)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng, explicit_rng = torch.get_rng_state().clone(), explicit.get_state().clone()
    cuda_initialized = torch.cuda.is_initialized()
    cuda_rng = torch.cuda.get_rng_state_all() if cuda_initialized else []
    counts = (module.optimization_steps, module.optimizer_step_attempts, module.amp_skipped_steps)
    selection = _Selection([], config, tmp_path / "best.pt", {}, 0.25)
    selection._save_diagnostic(module, 0, 0.25)
    assert random.getstate() == python_rng
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_rng[0] and np.array_equal(current_numpy[1], numpy_rng[1])
    assert current_numpy[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng) and torch.equal(explicit.get_state(), explicit_rng)
    assert torch.cuda.is_initialized() == cuda_initialized
    if cuda_initialized:
        assert all(torch.equal(left, right) for left, right in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    assert actor.training == training
    assert (module.optimization_steps, module.optimizer_step_attempts, module.amp_skipped_steps) == counts
    path = Path(selection.diagnostic_checkpoints[0]["path"])
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["run_metadata"]["stage"] == "bc"
    assert saved["run_metadata"]["diagnostic_update"] == 0
    assert saved["run_metadata"]["diagnostic_validation_nll"] == 0.25
    assert all(torch.equal(original[key], value) for key, value in saved["actor_critic_state"].items())
    loaded = ActorCriticWork3(condition_queries=True, hidden_dim=16)
    PPOTrainerWork3(loaded, create_optimizer=False).load_checkpoint(str(path))
    loaded.train(training)
    with torch.no_grad():
        actual = loaded.evaluate_action_log_probs(state, times, [record])
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))
    with pytest.raises(FileExistsError):
        selection._save_diagnostic(module, 0, 0.25)
    config.bc.max_updates = 225
    with pytest.raises(ValueError, match="八个"):
        _Selection([], config, tmp_path / "best.pt", {}, 0.25)


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
    assert not report.get("diagnostic_checkpoints")
    assert {path.name for path in tmp_path.glob("bc*.pt")} == {"bc.pt", "bc_initial.pt", "bc_final.pt"}
    assert report["optimization_steps"] == report["optimizer_step_attempts"] == 3
    assert report["sampled_scope_counts"] == {"task": 3}
    assert report["train_024_0_2_count"] == 0
    assert all(len(row["sample_sequence_sha256"]) == 64 for row in report["history"])
    assert all(row["sample_indices"] == [0] for row in report["history"])
    assert report["sample_index_audit_complete"]
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
    config.bc.diagnostic_checkpoint_dir = str(tmp_path / "diagnostics")
    diagnostic = run_behavior_cloning(config, output=tmp_path / "diagnostic.pt", report=tmp_path / "diagnostic.json")
    assert [row["update"] for row in diagnostic["diagnostic_checkpoints"]] == [0, 1, 2, 3]
    assert diagnostic["optimization_steps"] == report["optimization_steps"]
    assert [{key: value for key, value in row.items() if key != "elapsed_seconds"} for row in diagnostic["history"]] == [
        {key: value for key, value in row.items() if key != "elapsed_seconds"} for row in report["history"]]
    for suffix in (".pt", "_final.pt"):
        ordinary = torch.load(tmp_path / f"bc{suffix}", map_location="cpu", weights_only=False)
        saved_diagnostic = torch.load(tmp_path / f"diagnostic{suffix}", map_location="cpu", weights_only=False)
        assert all(torch.equal(value, saved_diagnostic["actor_critic_state"][key])
                   for key, value in ordinary["actor_critic_state"].items())
        left, right = ordinary["bc_training_state"], saved_diagnostic["bc_training_state"]
        assert left.keys() == right.keys()
        assert left["optimization_steps"] == right["optimization_steps"]
        assert left["optimizer_state"]["param_groups"] == right["optimizer_state"]["param_groups"]
        assert left["optimizer_state"]["state"].keys() == right["optimizer_state"]["state"].keys()
        for parameter, values in left["optimizer_state"]["state"].items():
            for key, value in values.items():
                other = right["optimizer_state"]["state"][parameter][key]
                assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
