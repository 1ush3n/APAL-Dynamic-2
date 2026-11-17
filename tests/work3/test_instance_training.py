"""公开实例切换与同结构QUERY权重继承回归。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from apal.models.work3.actor_critic import ActorCriticWork3
from apal.models.work3.ppo_trainer import PPOTrainerWork3


def _manifest(tmp_path: Path) -> Path:
    instances = []
    for name, size in (("A", 283), ("B", 680)):
        paths = {"baseline": Path("data") / "work3" / f"real_{size}_k10_baseline.json",
                 "raw_data": Path("data") / f"{size}.csv",
                 "worker_pool": Path("data") / "worker_pool_fixed.csv"}
        instances.append({"instance_id": name, "split": "train", "family": "regression",
                          "parent_group": name, **{key: str(path.resolve()) for key, path in paths.items()},
                          "public_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest()
                                            for key, path in paths.items()},
                          "scenario": None, "env_config": {"max_slots_per_station": 3}})
    path = tmp_path / "instances.json"
    path.write_text(json.dumps({"version": "work3_episode_instances_v1", "instances": instances,
                                "episode_sequence": ["A", "B", "A"]}), encoding="utf-8")
    return path


def test_public_manifest_and_spawn_switch_a_b_a(tmp_path: Path) -> None:
    from apal.training.work3.runtime_config import load_work3_episode_manifest
    from apal.training.work3.vector_env import Work3VectorEnv

    torch.set_num_threads(1)
    path = _manifest(tmp_path)
    instances = load_work3_episode_manifest(path)
    assert [item["instance_id"] for item in instances] == ["A", "B", "A"]
    instances[1]["scenario"] = {"scenario_id": "B_PUBLIC", "tau": 0.0, "recovery_time": 100.0,
                                 "affected_task_keys": ["0_317"], "valid": True}
    with Work3VectorEnv(condition_queries=True, seed=42) as vector:
        snapshots = []
        actor = ActorCriticWork3(condition_queries=True)
        for index, instance in enumerate(instances):
            result = vector.reset_all(scenarios=[instance["scenario"]], episode_ids=[index], episode_indices=[index],
                                      instances=[instance])[0]
            assert result.instance_id == instance["instance_id"]
            assert result.scenario_status["scenario_id"] == (instance["scenario"] or {}).get("scenario_id")
            snapshot = vector.snapshot()
            from apal.envs.work3.environment import AirLineEnvWork3
            from apal.models.work3.actor_critic import extract_compact_state_features
            fresh = AirLineEnvWork3(baseline_json_path=instance["baseline"], raw_data_path=instance["raw_data"],
                                    worker_pool_path=instance["worker_pool"], **instance["env_config"])
            fresh.reset()
            if instance["scenario"] is not None:
                fresh.load_scenario(instance["scenario"])
            reference = actor.make_decision_snapshot(fresh, extract_compact_state_features(fresh.state, None), torch.zeros(12))
            assert snapshot.candidate_task_keys == reference.candidate_task_keys
            assert snapshot.team_contexts == reference.team_contexts
            assert snapshot.query_tables == reference.query_tables
            for store in snapshot.graph_snapshot.stores:
                other = (reference.graph_snapshot[store._key] if store._key is not None
                         else reference.graph_snapshot._global_store)
                for key, value in store.items():
                    if isinstance(value, torch.Tensor):
                        assert torch.equal(value, other[key])
            snapshots.append(snapshot)
            action, _, _, _ = actor.select_snapshot(snapshot, deterministic=True)
            vector.step(action)
        first, middle, last = snapshots
        assert first.graph_snapshot["task"].x.shape[0] != middle.graph_snapshot["task"].x.shape[0]
        assert first.graph_snapshot["worker"].x.shape[0] != middle.graph_snapshot["worker"].x.shape[0]
        assert first.candidate_task_keys == last.candidate_task_keys
        assert first.branch_masks == last.branch_masks
        assert first.team_contexts == last.team_contexts
        assert first.query_tables == last.query_tables
    changed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    changed["instances"][0]["public_sha256"]["raw_data"] = "0" * 64
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_work3_episode_manifest(path)


def test_query_initialization_is_strict_and_does_not_restore_bc_optimizer(tmp_path: Path) -> None:
    from apal.training.work3.train import load_query_initial_weights

    torch.manual_seed(71)
    source = ActorCriticWork3(condition_queries=True)
    checkpoint = tmp_path / "bc.pt"
    trainer = PPOTrainerWork3(source)
    trainer.save_checkpoint(str(checkpoint), metadata={"stage": "bc", "optimizer_steps": 3})
    destination = ActorCriticWork3(condition_queries=True)
    report = load_query_initial_weights(destination, checkpoint)
    assert report["stage"] == "bc"
    assert report["optimizer_restored"] is False
    assert report["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert all(torch.equal(source.state_dict()[key], value)
               for key, value in destination.state_dict().items())
    damaged = torch.load(checkpoint, weights_only=False)
    damaged["policy_input_semantics"]["condition_normalization"] = "wrong"
    torch.save(damaged, checkpoint)
    with pytest.raises(ValueError, match="输入语义"):
        load_query_initial_weights(destination, checkpoint)


def test_pilot_uses_real_nominal_instance_plan_and_fresh_optimizer(tmp_path: Path) -> None:
    from apal.training.work3.train import run_training

    torch.set_num_threads(1)
    source = ActorCriticWork3(condition_queries=True)
    initial = tmp_path / "initial.pt"
    source_trainer = PPOTrainerWork3(source)
    for _ in range(2):
        source_trainer.optimizer.zero_grad()
        sum(parameter.sum() for parameter in source.parameters()).backward()
        source_trainer.optimizer.step()
    source_trainer.save_checkpoint(str(initial), metadata={"stage": "bc"})
    result = run_training(run_mode="pilot", method_variant="QUERY", successful_batch_target=1000,
                          max_decisions=8, max_wall_seconds=120.0, steps_per_iter=8,
                          num_envs=2, ppo_epochs=1, batch_size=8, gamma=1.0,
                          episode_manifest_path=_manifest(tmp_path), initial_checkpoint=initial,
                          output_ckpt=tmp_path / "ppo.pt", report_path=tmp_path / "ppo.json",
                          objective_weights={"w_makespan": 1.0, "w_h": 0.0, "w_t": 0.0,
                                             "w_w": 0.0, "w_p": 0.0, "w_revision_time": 0.0,
                                             "w_revision_team": 0.0})
    assert result["total_steps"] == 8
    assert result["lightning_optimization_steps"] == 1
    assert result["lightning_optimizer_step_attempts"] == 1
    assert result["lightning_amp_skipped_steps"] == 0
    assert result["initial_actor_fingerprint"] == result["initialization"]["actor_fingerprint"]
    assert result["initialization"]["optimizer_restored"] is False
    assert not result["disturbance_enabled"]
    assert [item["instance_id"] for item in result["scenario_log"]] == ["A", "B"]
    assert all(item["scenario_id"] is None and not item["disturbance_scheduled"] for item in result["scenario_log"])
    assert result["history"][0]["sampling_replay_sample_count"] > 0
    assert result["history"][0]["sampling_replay_max_abs_error"] < 1e-5
    checkpoint = torch.load(tmp_path / "ppo.pt", weights_only=False)
    assert any(not torch.equal(value, checkpoint["actor_critic_state"][key])
               for key, value in source.state_dict().items())
    assert checkpoint["lightning_training_state"]["optimizer_step_counts"]["successful"] == 1
    assert all(int(state["step"].item()) == 1
               for state in checkpoint["lightning_training_state"]["optimizer_state"]["state"].values())
