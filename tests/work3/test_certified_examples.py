"""有证书示例迁移的物理、语义和数据隔离回归。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("family", ["structured", "task_order", "five_person", "postpone"])
def test_certified_family_replays_current_engine(tmp_path: Path, family: str) -> None:
    from apal.data.work3.certified_examples import generate_instance, load_witness_actions, verify_instance

    case = tmp_path / family
    generate_instance(case, family=family, seed=1901)
    result = verify_instance(case)
    assert result["certificate_valid"] and result["native_verified"]
    assert result["independent_feasible"] and result["cost_ledger_matches"]
    assert result["C_exit"] == pytest.approx(result["certified_C_exit"], abs=1e-9)
    actions = load_witness_actions(case)
    assert actions and all("action" in row and "label_scope" in row for row in actions)
    if family == "postpone":
        assert any(row["action"]["branch"] == 1 for row in actions)
        assert all(row["label_scope"] in ([], ["branch"]) for row in actions)


@pytest.mark.parametrize("family,seed", [("alignment", 1902), ("revealed", 1901)])
def test_conditional_development_examples_keep_proof_scope(tmp_path: Path, family: str, seed: int) -> None:
    from apal.data.work3.certified_examples import generate_instance, load_witness_actions, verify_instance

    case = tmp_path / family
    generate_instance(case, family=family, seed=seed)
    result = verify_instance(case)
    assert result["certificate_valid"] and result["native_verified"]
    actions = load_witness_actions(case)
    if family == "alignment":
        assert any(row["action"].get("align") == 1 for row in actions)
        assert all(row["label_scope"] in ([], ["align"]) for row in actions)
    else:
        assert result["transfer_policy"] == "nominal_anchored_v1"
        assert result["actual_disturbance_hits"] > 0
        assert result["revealed_release_increase"] > 0
        assert sum(row["label_scope"] == ["branch"] for row in actions) == 1


def test_certificate_rejects_changed_public_semantics_and_illegal_teacher(tmp_path: Path) -> None:
    from apal.data.work3.certified_examples import generate_instance, replay_witness, verify_certificate

    case = tmp_path / "changed"
    generate_instance(case, family="five_person", seed=1902)
    tasks = case / "public" / "tasks.csv"
    with tasks.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["加工时间/h"] = str(float(row["加工时间/h"]) * 2)
    with tasks.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    certificate_path = case / "private" / "certificate.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["public_inputs_sha256"]["public/tasks.csv"] = hashlib.sha256(tasks.read_bytes()).hexdigest()
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    with pytest.raises((ValueError, AssertionError)):
        verify_certificate(case)
    clean = tmp_path / "illegal"
    generate_instance(clean, family="five_person", seed=1903)
    with pytest.raises(ValueError):
        replay_witness(clean, first_action_override={"task_key": "0_0", "branch": 0, "team": [999], "align": 0})


def test_dataset_manifest_contains_only_public_paths_and_disjoint_groups(tmp_path: Path) -> None:
    from apal.data.work3.certified_examples import generate_dataset

    manifests = generate_dataset(tmp_path / "dataset", train_count=3, validation_count=2, test_count=2, seed=1910)
    groups: dict[str, set[str]] = {}
    for split, path in manifests.items():
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        assert manifest["version"] == "work3_episode_instances_v1"
        assert len(manifest["instances"]) == {"train": 3, "val": 2, "test": 2}[split]
        groups[split] = {row["parent_group"] for row in manifest["instances"]}
        assert "private" not in json.dumps(manifest)
        for row in manifest["instances"]:
            for key in ("baseline", "raw_data", "worker_pool"):
                source = Path(path).parent / row[key]
                assert source.is_file()
                assert hashlib.sha256(source.read_bytes()).hexdigest() == row["public_sha256"][key]
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]


def test_unrevealed_future_release_is_absent_from_policy_inputs(tmp_path: Path) -> None:
    import torch
    from apal.data.work3.certified_examples import generate_instance, make_instance_env
    from apal.models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features

    torch.set_num_threads(1)
    actor = ActorCriticWork3(condition_queries=True)
    environments = []
    for name, release in (("earlier", 17), ("later", 23)):
        case = tmp_path / name
        generate_instance(case, family="revealed", seed=1901, release=release, reveal_time=1)
        environments.append(make_instance_env(case))

    def snapshots() -> list:
        return [actor.make_decision_snapshot(env, extract_compact_state_features(env.state, None), torch.zeros(2))
                for env in environments]

    before = snapshots()
    assert before[0].candidate_task_keys == before[1].candidate_task_keys
    assert before[0].branch_masks == before[1].branch_masks
    assert before[0].query_tables == before[1].query_tables
    assert torch.equal(before[0].state_features, before[1].state_features)
    for left, right in zip(before[0].graph_snapshot.stores, before[1].graph_snapshot.stores, strict=True):
        for key in left:
            if isinstance(left[key], torch.Tensor):
                assert torch.equal(left[key], right[key])
    for env in environments:
        env.step({"branch": 2})
        assert env.state.current_time == 1
        assert env.disturbance_event_triggered
    after = snapshots()
    assert any(not torch.equal(left.x, right.x) for left, right in zip(
        after[0].graph_snapshot.node_stores, after[1].graph_snapshot.node_stores, strict=True))
