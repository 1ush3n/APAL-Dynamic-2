"""教师条件记录、分层监督与原PPO概率契约。"""
from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch

from apal.envs.work3.core_types import ActionBranch
from apal.envs.work3.environment import AirLineEnvWork3
from apal.models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features

from .test_work3_m02_disturbed_pulse_trace import TEAMS, _two_aircraft_baseline


def _teacher_state(tmp_path: Path) -> tuple:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    baseline = tmp_path / "teacher.json"
    _two_aircraft_baseline(baseline)
    env = AirLineEnvWork3(baseline_json_path=baseline)
    env.reset()
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=16)
    snapshot = actor.make_decision_snapshot(env, extract_compact_state_features(env.state, None), torch.zeros(2))
    action = {"task_key": "0_15", "branch": ActionBranch.STATION_EXECUTE,
              "team": tuple(reversed(TEAMS[15])), "align": 1}
    return actor, snapshot, action


def test_teacher_record_uses_actual_prefix_without_consuming_rng(tmp_path: Path) -> None:
    actor, snapshot, teacher = _teacher_state(tmp_path)
    rng = torch.get_rng_state().clone()
    action, sampled_lp, _, record = actor.select_snapshot(snapshot, teacher_action=teacher)
    assert torch.equal(torch.get_rng_state(), rng)
    assert action == teacher
    assert record["chosen_team"] == teacher["team"]
    assert record["worker_prefixes"] == tuple(teacher["team"][:i] for i in range(len(teacher["team"])))
    state, time = snapshot.state_features.unsqueeze(0), snapshot.time_features.unsqueeze(0)
    old_values, old_lp, old_entropy = actor.evaluate_action_log_probs(state, time, [record])
    values, replay, entropy, parts = actor.evaluate_action_log_probs(state, time, [record], return_components=True)
    assert torch.equal(values, old_values) and torch.equal(replay, old_lp) and torch.equal(entropy, old_entropy)
    terms = parts[0]
    assert set(terms) == ({"advance"} if snapshot.advance_available else set()) | {"task", "branch", "team", "align"}
    total = sum((terms.get(key, torch.zeros(())) for key in ("advance", "task", "branch", "team", "align")))
    assert torch.equal(total, replay[0])
    assert sampled_lp == pytest.approx(float(replay[0].detach()), abs=1e-5)
    loss = -terms["team"]
    loss.backward()
    assert actor.query_worker_score.weight.grad is not None
    assert any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in actor.graph_encoder.parameters())
    assert all(p.grad is None for p in actor.critic.parameters())


@pytest.mark.parametrize("change", [
    {"task_key": "不存在"}, {"team": (999, 998)}, {"team": ()}, {"align": 2},
    {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT, "task_key": "0_15"},
])
def test_illegal_teacher_action_is_rejected(tmp_path: Path, change: dict) -> None:
    actor, snapshot, teacher = _teacher_state(tmp_path)
    with pytest.raises(ValueError):
        actor.select_snapshot(snapshot, teacher_action={**teacher, **change})


def test_export_replays_current_teacher_and_keeps_private_answers_out_of_inputs(tmp_path: Path) -> None:
    from apal.data.work3.certified_examples import generate_dataset
    from apal.data.work3.demonstrations import export_demonstrations

    manifests = generate_dataset(tmp_path / "cases", train_count=1, validation_count=1, test_count=1, seed=4021)
    destination = tmp_path / "records.pt"
    summary = export_demonstrations(manifests["train"], destination, split="train")
    payload = torch.load(destination, weights_only=False)
    assert payload["version"] == "work3_bc_records_v1"
    assert payload["record_semantics_version"] == "work3_autoregressive_layers_v1"
    assert summary["instances"] == 1 and summary["records"] > 0
    assert summary["choice_records"] > 0
    for row in payload["records"]:
        assert row["state_features"].shape == (32,) and row["time_features"].shape == (2,)
        assert "certificate" not in row["record"] and "optimal_team" not in row["record"]
        if row["record"]["action_type"] == "forced_advance":
            assert not row["label_scope"]
    with pytest.raises(FileExistsError):
        export_demonstrations(manifests["train"], destination, split="train")


def test_manifest_evaluation_uses_its_worker_pool_and_keeps_stage_identity(tmp_path: Path) -> None:
    from apal.data.work3.certified_examples import generate_dataset
    from apal.evaluation.query import evaluate_query_instances
    from apal.utils.work3.objective_evaluator import ObjectiveWeights

    manifests = generate_dataset(tmp_path / "evaluation", train_count=1, validation_count=1, test_count=1, seed=5041)
    actor = ActorCriticWork3(condition_queries=True, hidden_dim=16)
    result = evaluate_query_instances(manifests["val"], actor, split="val",
        weights=ObjectiveWeights(w_makespan=1, w_h=0, w_t=0, w_w=0, w_p=0), max_decisions=500)
    assert len(result["instances"]) == 1
    row = result["instances"][0]
    assert row["instance_id"] == "val_000" and row["parent_group"]
    assert row["success"] and row["cost_ledger_matches"] and row["independent_feasible"]
    assert row["C_task"] <= row["makespan"]
    assert "certified_C_exit" not in row and "gap" not in row


def test_export_rejects_manifest_conditions_outside_certificate(tmp_path: Path) -> None:
    from apal.data.work3.certified_examples import generate_dataset
    from apal.data.work3.demonstrations import export_demonstrations

    manifests = generate_dataset(tmp_path / "overrides", train_count=1, validation_count=1, test_count=1, seed=6041)
    path = manifests["train"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["instances"][0]["env_config"]["transfer_policy"] = "nominal_anchored_v1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="证书|现场"):
        export_demonstrations(path, tmp_path / "wrong.pt", split="train")


def test_public_manifest_accepts_portable_windows_relative_paths(tmp_path: Path) -> None:
    from pathlib import PureWindowsPath
    from apal.data.work3.certified_examples import generate_dataset
    from apal.training.work3.runtime_config import load_work3_episode_manifest

    manifests = generate_dataset(tmp_path / "portable", train_count=1, validation_count=1, test_count=1, seed=7041)
    path = manifests["train"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {}
    for key in ("baseline", "raw_data", "worker_pool"):
        relative = payload["instances"][0][key]
        expected[key] = (path.parent / relative).resolve()
        payload["instances"][0][key] = str(PureWindowsPath(relative))
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_work3_episode_manifest(path)
    assert all(Path(loaded[0][key]) == target for key, target in expected.items())
