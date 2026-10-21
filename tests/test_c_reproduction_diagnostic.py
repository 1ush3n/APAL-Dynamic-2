"""C 基准复现诊断的最小必要验证（第 9 节 1–6 项）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import ActorCriticWork3
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from scripts.work3.baseline_repair_agent import BaselinePlanRepairAgent
from scripts.work3.evaluate_c_vs_d import build_formal_evaluation_agent
import work3_4060.c_reproduction_protocol as protocol
from work3_4060.c_reproduction_protocol import (
    BASELINE_PATH,
    actor_fingerprint,
    build_c_state_features,
    build_c_time_features,
    build_diagnostic_checkpoint,
    build_teacher_sample,
    clone_actor_state,
    load_diagnostic_actor,
    sample_records,
    sample_snapshot,
    sha256_file,
    state_dict_fingerprint,
    teacher_label_from_action,
    teacher_marginal_log_prob,
)

DECISIONS_FOR_TESTS = 14


def _build_model() -> ActorCriticWork3:
    torch.manual_seed(7)
    model = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    model.eval()
    return model


@pytest.fixture(scope="module")
def teacher_bundle() -> dict:
    """在真实 Nominal 实例上采集前若干条 A 决策快照（CPU、单线程）。"""
    torch.set_num_threads(1)
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    agent = BaselinePlanRepairAgent()
    builder = _build_model()
    samples: list[dict] = []
    while len(samples) < DECISIONS_FOR_TESTS and not env._check_terminated():
        action = agent.select_action(env)
        assert action is not None
        cmax = compute_cycle_heuristic_cmax(env.state)
        state_features = build_c_state_features(env.state, cmax)
        time_features = build_c_time_features(env.state, cmax)
        snapshot = builder.make_decision_snapshot(
            env,
            state_features,
            time_features,
            episode_index=len(samples) + 1,
            estimated_cmax=cmax,
            allow_postpone=True,
        )
        label = teacher_label_from_action(snapshot, action)
        env.step(action)
        samples.append(
            build_teacher_sample(
                sample_id=len(samples),
                decision_index=len(samples) + 1,
                snapshot=snapshot,
                label=label,
                teacher_action=action,
                outcome={"scheduled_status": "RESERVED"},
            )
        )
    assert len(samples) == DECISIONS_FOR_TESTS
    return {"samples": samples}


def test_teacher_action_legal_in_formal_snapshot(teacher_bundle: dict) -> None:
    """第 9 节 1：A 动作在正式 C 快照中合法，且正式重放接受该记录。"""
    model = _build_model()
    for sample in teacher_bundle["samples"]:
        snapshot = sample_snapshot(sample)
        label = protocol.sample_label(sample)
        assert label.decision_kind == "stay"
        assert label.task_key in snapshot.candidate_task_keys
        assert snapshot.branch_masks[int(label.task_idx)][0] is True
        assert label.valid_permutations
        with torch.no_grad():
            log_prob = teacher_marginal_log_prob(model, sample)
        assert torch.isfinite(log_prob).all()


def test_illegal_branch_and_worker_index_rejected(teacher_bundle: dict) -> None:
    """第 9 节 1：非法分支与错误工人索引必须被正式重放拒绝。"""
    model = _build_model()
    sample = teacher_bundle["samples"][0]
    snapshot = sample_snapshot(sample)
    label = protocol.sample_label(sample)
    assert label.task_idx is not None
    record = sample_records(sample, model.max_station_workers)[0]
    state = snapshot.state_features.unsqueeze(0)
    time_feature = snapshot.time_features.unsqueeze(0)

    can_postpone = snapshot.branch_masks[int(label.task_idx)][1]
    if not can_postpone:
        illegal_branch = dict(record)
        illegal_branch["branch"] = 1
        illegal_branch["chosen_worker_indices"] = ()
        illegal_branch["worker_valid_masks"] = ()
        illegal_branch["align"] = 0
        with pytest.raises(ValueError):
            model.evaluate_action_log_probs(state, time_feature, [illegal_branch])

    illegal_worker = dict(record)
    num_workers = record["num_st_workers"]
    illegal_worker["chosen_worker_indices"] = (num_workers,)
    with pytest.raises(ValueError):
        model.evaluate_action_log_probs(state, time_feature, [illegal_worker])


def test_replay_log_prob_consistent_across_calls(teacher_bundle: dict) -> None:
    """第 9 节 2：固定样本重复前向一致，且共享编码不改变数值。"""
    model = _build_model()
    sample = teacher_bundle["samples"][0]
    with torch.no_grad():
        first = teacher_marginal_log_prob(model, sample, share_encoding=True)
        second = teacher_marginal_log_prob(model, sample, share_encoding=True)
        unshared = teacher_marginal_log_prob(model, sample, share_encoding=False)
    assert torch.equal(first, second)
    assert torch.allclose(first, unshared, rtol=0.0, atol=1e-6)


def test_unordered_team_marginal_equals_logsumexp(teacher_bundle: dict) -> None:
    """第 9 节 3：无序团队按合法排列 logsumexp，且与集合准确率比较口径一致。"""
    model = _build_model()
    samples = teacher_bundle["samples"]
    ambiguous = [
        sample
        for sample in samples
        if len(protocol.sample_label(sample).valid_permutations) == 2
    ]
    assert ambiguous, "测试样本中应至少存在一个团队顺序存在别名的状态"
    sample = ambiguous[0]
    snapshot = sample_snapshot(sample)
    label = protocol.sample_label(sample)
    assert len({frozenset(order) for order in label.valid_permutations}) == 1
    assert tuple(sorted(label.valid_permutations[0])) == tuple(sorted(label.valid_permutations[1]))

    records = sample_records(sample, model.max_station_workers)
    assert len(records) == len(label.valid_permutations)
    state = snapshot.state_features.unsqueeze(0).expand(len(records), -1).contiguous()
    time_feature = snapshot.time_features.unsqueeze(0).expand(len(records), -1).contiguous()
    with torch.no_grad():
        _values, log_probs, _entropy = model.evaluate_action_log_probs(state, time_feature, records)
        expected = torch.logsumexp(log_probs, dim=0)
        observed = teacher_marginal_log_prob(model, sample)
    assert torch.allclose(observed, expected, rtol=0.0, atol=1e-6)
    # 每个排列的逐人掩码必须随前缀变化，而不是复制同一张掩码
    first_masks = records[0]["worker_valid_masks"]
    second_masks = records[1]["worker_valid_masks"]
    assert first_masks[0] == second_masks[0]
    if len(first_masks) > 1:
        assert first_masks[1] != second_masks[1]


def test_bc_loss_differentiable_and_updates_policy_weights(teacher_bundle: dict) -> None:
    """第 9 节 2：BC 损失可微，策略参数获得有限非零梯度并真实更新。"""
    model = _build_model()
    value_names = set(protocol.value_parameter_names(model))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name not in value_names
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=3e-4, weight_decay=0.0)
    before = clone_actor_state(model)

    samples = teacher_bundle["samples"][:2]
    optimizer.zero_grad(set_to_none=True)
    loss = torch.zeros((), requires_grad=True)
    for sample in samples:
        log_prob = teacher_marginal_log_prob(model, sample)
        assert log_prob.requires_grad, "正式重放接口必须可微"
        loss = loss + (-log_prob / len(samples))
    loss.backward()

    policy_grad_norms = {
        name: float(parameter.grad.norm())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    assert policy_grad_norms, "策略相关参数没有获得梯度"
    assert any(norm > 0.0 for norm in policy_grad_norms.values())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in trainable
        if parameter.grad is not None
    )
    for name, parameter in model.named_parameters():
        if name in value_names:
            assert parameter.grad is None, f"价值专属参数不应获得梯度: {name}"

    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    assert torch.isfinite(grad_norm)
    optimizer.step()
    after = clone_actor_state(model)
    changed = [name for name in after if not torch.equal(before[name], after[name])]
    assert changed, "一次真实更新必须改变相关权重"
    assert state_dict_fingerprint(after) != state_dict_fingerprint(before)
    assert not any(name.startswith("critic.") for name in changed), "价值头参数不应被更新"


def test_cpu_snapshot_has_no_env_reference_and_no_future_labels(teacher_bundle: dict) -> None:
    """第 9 节 5：快照不携带环境可变引用，也不含未来标签。"""
    forbidden = {
        "actual_start",
        "actual_end",
        "actual_start_station",
        "transfer_history",
        "final_makespan",
        "future_transfer_time",
    }
    sample = teacher_bundle["samples"][0]
    snapshot = sample_snapshot(sample)
    assert not hasattr(snapshot.graph_snapshot, "env")
    for name in protocol.SNAPSHOT_FIELDS:
        value = getattr(snapshot, name)
        assert not isinstance(value, AirLineEnvWork3)
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu"
            assert value.requires_grad is False
            assert value.grad_fn is None
    for node_type in snapshot.graph_snapshot.node_types:
        for _key, value in snapshot.graph_snapshot[node_type].items():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"
    serialised = repr(sample["snapshot"])
    for key in forbidden:
        assert key not in serialised
    # 快照必须能独立往返磁盘，不依赖环境对象
    restored = protocol.snapshot_from_fields(sample["snapshot"])
    assert restored.candidate_task_keys == snapshot.candidate_task_keys
    assert restored.current_time == snapshot.current_time
    assert restored.branch_masks == snapshot.branch_masks


def test_diagnostic_checkpoint_roundtrip_and_strict_rejection(tmp_path: Path) -> None:
    """第 9 节 4：诊断格式可严格往返；非诊断格式与篡改内容被拒绝。"""
    model = _build_model()
    checkpoint = build_diagnostic_checkpoint(
        model,
        source_weights_path=Path("synthetic/weights.pt"),
        source_weights_sha256="0" * 64,
        source_actor_fingerprint=actor_fingerprint(model),
        teacher_manifest_sha256="1" * 64,
        training_summary={"optimizer_steps": 0, "stop_reason": "unit_test"},
        snapshot_contract={"graph_version": "test"},
    )
    path = tmp_path / "diagnostic_actor.pt"
    torch.save(checkpoint, path)

    loaded, metadata = load_diagnostic_actor(
        path, expected_teacher_manifest_sha256="1" * 64
    )
    assert metadata["diagnostic_format"] == protocol.DIAGNOSTIC_FORMAT
    assert actor_fingerprint(loaded) == metadata["actor_fingerprint"]

    with pytest.raises(ValueError):
        load_diagnostic_actor(path, expected_teacher_manifest_sha256="2" * 64)

    foreign = tmp_path / "foreign.pt"
    torch.save({"actor_critic_state": checkpoint["actor_critic_state"]}, foreign)
    with pytest.raises(ValueError):
        load_diagnostic_actor(foreign)

    tampered = dict(checkpoint)
    tampered_state = {
        name: tensor.clone() for name, tensor in checkpoint["actor_critic_state"].items()
    }
    first_name = sorted(tampered_state)[0]
    tampered_state[first_name] = tampered_state[first_name] + 1.0
    tampered["actor_critic_state"] = tampered_state
    tampered_path = tmp_path / "tampered.pt"
    torch.save(tampered, tampered_path)
    with pytest.raises(ValueError):
        load_diagnostic_actor(tampered_path)

    # 诊断检查点不得通过正式 PPO 加载合同
    with pytest.raises(ValueError):
        build_formal_evaluation_agent("C", path, device="cpu")


def test_original_checkpoint_is_untouched_by_diagnostic_copy() -> None:
    """第 9 节 4：诊断副本训练前指纹等于原 C，原文件字节未改写。"""
    if not protocol.ORIGINAL_C_WEIGHTS.is_file():
        pytest.skip("本机缺少原 C 检查点")
    digest_before = sha256_file(protocol.ORIGINAL_C_WEIGHTS)
    agent = build_formal_evaluation_agent("C", protocol.ORIGINAL_C_WEIGHTS, device="cpu")
    copy = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    copy.load_state_dict(clone_actor_state(agent.actor_critic), strict=True)
    assert actor_fingerprint(copy) == actor_fingerprint(agent.actor_critic)
    assert sha256_file(protocol.ORIGINAL_C_WEIGHTS) == digest_before


def test_input_visibility_report_covers_four_items(teacher_bundle: dict) -> None:
    """Phase A：清册必须覆盖四个信息项，并给出状态与证据。"""
    from work3_4060.run_c_reproduction_diagnostic import build_input_visibility_report

    baseline = protocol.load_json(BASELINE_PATH)
    samples = teacher_bundle["samples"]
    report = build_input_visibility_report(samples, baseline)
    assert len(report["items"]) == 4
    for item in report["items"]:
        assert item["entries"]
        for entry in item["entries"]:
            assert entry["status"].startswith(("direct", "derivable", "not_found"))
    assert report["tensor_contract"]["state_features"] == [32]
    assert report["tensor_contract"]["time_features"] == [12]
    assert report["probe_context"]["sample_count"] == len(samples)
    edge_keys = report["tensor_contract"]["graph_edges"]
    assert edge_keys["('task', 'baseline_team', 'worker')"] > 0
    assert edge_keys["('task', 'last_published_team', 'worker')"] > 0


def test_outcome_tier_mapping() -> None:
    """第 8 节：描述性等级必须由绝对费用与可行性判定，不用 A 作百分比分母。"""
    from work3_4060.run_c_reproduction_diagnostic import classify_outcome

    base = {
        "success": True,
        "feasible": True,
        "constraint_violation_count": 0,
        "postponed_count": 0,
        "objective_j_total": 0.0,
        "cost_components_weighted": {"j_takt": 0.0},
    }
    tier, _ = classify_outcome(base, 3.8)
    assert tier == "NEAR_BASELINE_COST"

    improved = dict(base)
    improved["objective_j_total"] = 0.05
    improved["cost_components_weighted"] = {"j_takt": 0.05}
    tier, _ = classify_outcome(improved, 3.8)
    assert tier == "SUBSTANTIAL_IMPROVEMENT_NOT_BASELINE"

    worse = dict(base)
    worse["objective_j_total"] = 3.9
    worse["cost_components_weighted"] = {"j_takt": 3.9}
    tier, _ = classify_outcome(worse, 3.8)
    assert tier == "NO_EFFECTIVE_CLOSURE_REPRODUCTION"

    illegal = dict(base)
    illegal["feasible"] = False
    tier, _ = classify_outcome(illegal, 3.8)
    assert tier == "CLOSURE_NOT_ACCEPTED"
