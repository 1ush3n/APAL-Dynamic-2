"""W3-06 图策略接入与 PPO 重放快照反例。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch_geometric.data import HeteroData

from envs.work3.core_types import ActionBranch
from envs.work3.decision_snapshot import DecisionSnapshot, TeamCompletionContext, WorkerSnapshot
from envs.work3.environment import AirLineEnvWork3
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features
from models.work3.graph_builder import GRAPH_FEATURE_DIMS, GRAPH_FEATURE_VERSION
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax


@pytest.fixture
def baseline_path() -> Path:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return path


@pytest.fixture
def env(baseline_path: Path) -> AirLineEnvWork3:
    environment = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    environment.reset()
    return environment


def _state_inputs(env: AirLineEnvWork3) -> tuple[torch.Tensor, torch.Tensor]:
    cmax = compute_cycle_heuristic_cmax(env.state)
    state_feat = extract_compact_state_features(env.state, cmax)
    urgency = compute_time_urgency_vector(
        estimated_r=max(0.0, cmax - env.state.current_time),
        current_time=env.state.current_time,
        last_transfer_time=env.state.last_transfer_time,
        h0=env.state.h0,
    )
    return state_feat, urgency


def _small_graph(task_count: int, worker_count: int, marker: float) -> HeteroData:
    graph = HeteroData()
    graph["task"].x = torch.full((task_count, GRAPH_FEATURE_DIMS["task"]), marker)
    graph["worker"].x = torch.zeros((worker_count, GRAPH_FEATURE_DIMS["worker"]))
    graph["worker"].x[:, 0] = torch.arange(worker_count, dtype=torch.float) + marker
    graph["station"].x = torch.zeros((1, GRAPH_FEATURE_DIMS["station"]))
    graph["skill"].x = torch.zeros((5, GRAPH_FEATURE_DIMS["skill"]))
    for edge_type, edge in {
        ("task", "precedes", "task"): torch.empty((2, 0), dtype=torch.long),
        ("task", "assigned_to", "station"): torch.tensor(
            [list(range(task_count)), [0] * task_count], dtype=torch.long
        ),
        ("station", "has_task", "task"): torch.tensor(
            [[0] * task_count, list(range(task_count))], dtype=torch.long
        ),
        ("worker", "has_skill", "skill"): torch.tensor(
            [list(range(worker_count)), [0] * worker_count], dtype=torch.long
        ),
        ("skill", "required_by", "task"): torch.tensor(
            [[0] * task_count, list(range(task_count))], dtype=torch.long
        ),
        ("task", "requires", "skill"): torch.tensor(
            [list(range(task_count)), [0] * task_count], dtype=torch.long
        ),
        ("skill", "provided_by", "worker"): torch.tensor(
            [[0] * worker_count, list(range(worker_count))], dtype=torch.long
        ),
        ("task", "done_by", "worker"): torch.empty((2, 0), dtype=torch.long),
        ("task", "baseline_team", "worker"): torch.tensor(
            [[0] * min(task_count, worker_count), list(range(min(task_count, worker_count)))],
            dtype=torch.long,
        ),
        ("task", "last_published_team", "worker"): torch.empty(
            (2, 0), dtype=torch.long
        ),
    }.items():
        graph[edge_type].edge_index = edge
    return graph


def _small_snapshot(
    *,
    worker_id: int,
    task_count: int,
    worker_ids: tuple[int, ...],
    candidate_indices: tuple[int, ...],
    demands: tuple[int, ...],
    marker: float,
) -> DecisionSnapshot:
    contexts = tuple(
        TeamCompletionContext(
            task_key=f"env{worker_id}_task{task_index}",
            station_id=worker_id,
            required_skill=0,
            demand=demand,
            workers=tuple(
                WorkerSnapshot(
                    worker_id=identity,
                    skills=(0,),
                    efficiency=1.0,
                    calendar_intervals=(),
                )
                for identity in worker_ids
            ),
        )
        for task_index, demand in zip(candidate_indices, demands, strict=True)
    )
    keys = tuple(context.task_key for context in contexts)
    return DecisionSnapshot(
        worker_id=worker_id,
        episode_id=worker_id,
        episode_index=0,
        state_features=torch.full((32,), marker),
        time_features=torch.tensor([marker, marker + 0.5]),
        graph_snapshot=_small_graph(task_count, len(worker_ids), marker),
        candidate_task_keys=keys,
        candidate_task_features=torch.full((len(keys), 8), marker),
        branch_masks=tuple((True, False) for _ in keys),
        team_contexts=contexts,
        candidate_task_node_indices=candidate_indices,
        worker_node_indices=tuple(
            tuple(range(len(worker_ids))) for _ in candidate_indices
        ),
        reserved_flags=tuple(False for _ in keys),
        advance_available=False,
        current_time=0.0,
        cycle_id=0,
        estimated_cmax=1.0,
        h0=1.0,
        last_transfer_time=0.0,
        graph_version=GRAPH_FEATURE_VERSION,
    )


def test_sampling_record_contains_immutable_graph_snapshot(env: AirLineEnvWork3) -> None:
    """采样时必须保存图快照，不能在现场变化后用新状态重构旧动作。"""
    net = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=32)
    state_feat, urgency = _state_inputs(env)

    action, _, _, record = net.select_action(env, state_feat, urgency)

    assert action is not None
    assert isinstance(record["graph_snapshot"], HeteroData)
    assert record["graph_version"] == GRAPH_FEATURE_VERSION
    assert len(record["candidate_task_node_indices"]) == len(record["cand_feats"])
    assert record["candidate_task_node_indices"]
    if (
        record["action_type"] == "schedule"
        and record["branch"] == int(ActionBranch.STATION_EXECUTE)
    ):
        assert record["worker_node_indices"]
    elif record["action_type"] == "schedule":
        assert record["branch"] == int(ActionBranch.POSTPONE)
        assert record["worker_node_indices"] == ()
    else:
        assert record["action_type"] == "advance_to_next_event"
        assert record["worker_node_indices"] == ()


def test_minibatch_keeps_local_candidate_and_worker_indices_across_graph_sizes() -> None:
    """异构图样本尺寸不同时，候选节点及工人身份仍按各自图的本地索引解释。"""
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16).eval()
    snapshots = (
        _small_snapshot(
            worker_id=0,
            task_count=3,
            worker_ids=(10, 11),
            candidate_indices=(2, 0),
            demands=(1, 2),
            marker=0.1,
        ),
        _small_snapshot(
            worker_id=1,
            task_count=5,
            worker_ids=(100, 101, 102, 103, 104),
            candidate_indices=(4,),
            demands=(3,),
            marker=0.9,
        ),
    )
    with torch.no_grad():
        actor.branch_head[-1].bias[0] = 10.0
        actor.branch_head[-1].bias[1] = -10.0

    sampled = [actor.select_snapshot(snapshot, deterministic=True) for snapshot in snapshots]
    actions = [result[0] for result in sampled]
    old_log_probs = torch.tensor([result[1] for result in sampled], dtype=torch.float32)
    records = [result[3] for result in sampled]

    assert [snapshot.graph_snapshot["task"].num_nodes for snapshot in snapshots] == [3, 5]
    assert [snapshot.graph_snapshot["worker"].num_nodes for snapshot in snapshots] == [2, 5]
    assert [len(snapshot.candidate_task_keys) for snapshot in snapshots] == [2, 1]
    assert [record["candidate_task_node_indices"] for record in records] == [(2, 0), (4,)]
    for snapshot, action, record in zip(snapshots, actions, records, strict=True):
        chosen_index = record["task_idx"]
        assert action["task_key"] == snapshot.candidate_task_keys[chosen_index]
        assert record["candidate_task_node_indices"][chosen_index] == (
            snapshot.candidate_task_node_indices[chosen_index]
        )
        context = snapshot.team_contexts[chosen_index]
        expected_team = tuple(
            context.workers[index].worker_id
            for index in record["chosen_worker_indices"]
        )
        assert action["team"] == expected_team
        assert set(action["team"]) <= {worker.worker_id for worker in context.workers}

    task_graph_sizes: list[int] = []
    worker_graph_sizes: list[int] = []
    task_hook = actor.task_graph_proj.register_forward_pre_hook(
        lambda _module, inputs: task_graph_sizes.append(int(inputs[0].shape[0]))
    )
    worker_hook = actor.worker_graph_score.register_forward_pre_hook(
        lambda _module, inputs: worker_graph_sizes.append(int(inputs[0].shape[0]))
    )
    try:
        batched_values, batched_log_probs, batched_entropies = actor.evaluate_action_log_probs(
            torch.stack([snapshot.state_features for snapshot in snapshots]),
            torch.stack([snapshot.time_features for snapshot in snapshots]),
            records,
        )
    finally:
        task_hook.remove()
        worker_hook.remove()

    assert task_graph_sizes == [2, 1]
    assert worker_graph_sizes == [2, 5]
    assert batched_values.shape == batched_log_probs.shape == batched_entropies.shape == (2,)
    assert torch.isfinite(batched_log_probs).all()
    assert torch.allclose(torch.exp(batched_log_probs - old_log_probs), torch.ones(2), atol=1e-5)

    for index in range(2):
        value, log_prob, entropy = actor.evaluate_action_log_probs(
            snapshots[index].state_features.unsqueeze(0),
            snapshots[index].time_features.unsqueeze(0),
            [records[index]],
        )
        assert torch.allclose(batched_values[index : index + 1], value, atol=1e-6)
        assert torch.allclose(batched_log_probs[index : index + 1], log_prob, atol=1e-6)
        assert torch.allclose(batched_entropies[index : index + 1], entropy, atol=1e-6)


def test_graph_node_relabel_and_candidate_reorder_preserve_identity_policy() -> None:
    """同步重编号图节点和候选顺序后，策略按实体身份保持等变。"""
    snapshot = _small_snapshot(
        worker_id=7,
        task_count=3,
        worker_ids=(70, 71),
        candidate_indices=(2, 0),
        demands=(1, 2),
        marker=0.2,
    )
    graph = snapshot.graph_snapshot.clone()
    graph["task"].x[:, 0] = torch.tensor([0.1, 0.4, 0.9])
    candidate_features = torch.tensor(
        [
            [2.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    snapshot = replace(
        snapshot,
        graph_snapshot=graph,
        candidate_task_features=candidate_features,
    )

    node_orders = {
        "task": torch.tensor([1, 2, 0]),
        "worker": torch.tensor([1, 0]),
        "station": torch.tensor([0]),
        "skill": torch.tensor([3, 0, 4, 2, 1]),
    }
    old_to_new: dict[str, torch.Tensor] = {}
    relabeled_graph = snapshot.graph_snapshot.clone()
    for node_type, new_to_old in node_orders.items():
        mapping = torch.empty_like(new_to_old)
        mapping[new_to_old] = torch.arange(new_to_old.numel())
        old_to_new[node_type] = mapping
        # 节点特征形状：[N, F] -> [N, F]，只按节点编号重排。
        relabeled_graph[node_type].x = relabeled_graph[node_type].x[new_to_old].clone()
    for edge_type in relabeled_graph.edge_types:
        edge_index = relabeled_graph[edge_type].edge_index
        source_type, _, target_type = edge_type
        # 边索引形状：[2, E] -> [2, E]，端点改写到对应的新节点编号。
        relabeled_graph[edge_type].edge_index = torch.stack(
            (
                old_to_new[source_type][edge_index[0]],
                old_to_new[target_type][edge_index[1]],
            )
        )

    candidate_order = (1, 0)
    relabeled_snapshot = replace(
        snapshot,
        graph_snapshot=relabeled_graph,
        candidate_task_keys=tuple(snapshot.candidate_task_keys[index] for index in candidate_order),
        candidate_task_features=snapshot.candidate_task_features[list(candidate_order)],
        branch_masks=tuple(snapshot.branch_masks[index] for index in candidate_order),
        team_contexts=tuple(snapshot.team_contexts[index] for index in candidate_order),
        candidate_task_node_indices=tuple(
            int(old_to_new["task"][snapshot.candidate_task_node_indices[index]])
            for index in candidate_order
        ),
        worker_node_indices=tuple(
            tuple(
                int(old_to_new["worker"][worker_index])
                for worker_index in snapshot.worker_node_indices[index]
            )
            for index in candidate_order
        ),
        reserved_flags=tuple(snapshot.reserved_flags[index] for index in candidate_order),
    )

    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16).eval()
    with torch.no_grad():
        context, task_nodes, worker_nodes = actor.graph_encoder(snapshot.graph_snapshot)
        relabeled_context, relabeled_task_nodes, relabeled_worker_nodes = actor.graph_encoder(
            relabeled_snapshot.graph_snapshot
        )
    assert torch.allclose(context, relabeled_context, atol=1e-5, rtol=1e-5)
    for old_index, new_index in enumerate(old_to_new["task"].tolist()):
        assert torch.allclose(
            task_nodes[old_index], relabeled_task_nodes[new_index], atol=1e-5, rtol=1e-5
        )
    for old_index, new_index in enumerate(old_to_new["worker"].tolist()):
        assert torch.allclose(
            worker_nodes[old_index],
            relabeled_worker_nodes[new_index],
            atol=1e-5,
            rtol=1e-5,
        )

    action, sampled_log_prob, _, record = actor.select_snapshot(
        snapshot, deterministic=True
    )
    relabeled_action, relabeled_log_prob, _, relabeled_record = actor.select_snapshot(
        relabeled_snapshot, deterministic=True
    )
    assert action is not None and relabeled_action is not None
    assert action["task_key"] == relabeled_action["task_key"]
    assert action["team"] == relabeled_action["team"]
    assert action["branch"] == relabeled_action["branch"]
    assert sampled_log_prob == pytest.approx(relabeled_log_prob, abs=1e-5)

    states = torch.stack((snapshot.state_features, relabeled_snapshot.state_features))
    times = torch.stack((snapshot.time_features, relabeled_snapshot.time_features))
    _, replay_log_probs, _ = actor.evaluate_action_log_probs(
        states, times, [record, relabeled_record]
    )
    assert torch.allclose(replay_log_probs, torch.full((2,), sampled_log_prob), atol=1e-5)


def test_graph_encoder_receives_policy_gradient(env: AirLineEnvWork3) -> None:
    """策略重放损失必须能反传到图编码器，而不是只更新汇总 MLP。"""
    net = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=32)
    state_feat, urgency = _state_inputs(env)
    action, _, _, record = net.select_action(env, state_feat, urgency, deterministic=True)
    assert action is not None

    _, log_probs, _ = net.evaluate_action_log_probs(
        state_feat.unsqueeze(0), urgency.unsqueeze(0), [record]
    )
    advantages = torch.ones_like(log_probs)
    policy_loss = -(log_probs * advantages).mean()
    assert torch.isfinite(policy_loss)
    policy_loss.backward()

    graph_parameters = [p for p in net.graph_encoder.parameters() if p.requires_grad]
    assert graph_parameters
    graph_gradients = [
        parameter.grad for parameter in graph_parameters if parameter.grad is not None
    ]
    assert graph_gradients
    assert all(torch.isfinite(gradient).all() for gradient in graph_gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in graph_gradients)
    assert any(parameter.grad is not None for parameter in net.encoder.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA验证图快照设备隔离")
def test_graph_encoder_does_not_move_replay_snapshot_to_cuda(
    env: AirLineEnvWork3,
) -> None:
    """图编码器应把CPU回放快照作为只读输入，避免整批历史图滞留显存。"""
    actor = ActorCriticWork3(
        state_dim=32,
        task_feat_dim=8,
        hidden_dim=32,
    ).cuda().eval()
    graph = actor.build_graph_snapshot(env)
    assert all(
        value.device.type == "cpu"
        for store in graph.stores
        for value in store.values()
        if isinstance(value, torch.Tensor)
    )

    with torch.no_grad():
        context, task_embeddings, worker_embeddings = actor.graph_encoder(graph)

    assert context.device.type == "cuda"
    assert task_embeddings.device.type == "cuda"
    assert worker_embeddings.device.type == "cuda"
    assert all(
        value.device.type == "cpu"
        for store in graph.stores
        for value in store.values()
        if isinstance(value, torch.Tensor)
    )


def test_worker_calendar_changes_graph_policy_input(env: AirLineEnvWork3) -> None:
    """同一物理状态下改变具体工人日历，图输入必须发生变化。"""
    net = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=32)
    before = net.build_graph_snapshot(env)
    worker_id = next(iter(env.state.workers))
    env.state.workers[worker_id].add_interval(0.0, 10.0, "calendar_probe")
    after = net.build_graph_snapshot(env)

    worker_index = net.graph_builder.worker_id_to_idx[worker_id]
    assert not torch.equal(
        before["worker"].x[worker_index],
        after["worker"].x[worker_index],
    )


@pytest.mark.parametrize("changed_input", ["calendar", "skill", "baseline_team"])
def test_worker_context_changes_reach_worker_selection_head(
    env: AirLineEnvWork3,
    changed_input: str,
) -> None:
    """单独改变工人资源/关系信息后，变化应送达工人选人评分头。"""
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=32).eval()
    task = next(
        item
        for item in env.state.tasks.values()
        if item.current_station >= 0 and item.base_team
    )
    builder = actor._get_graph_builder(env)
    graph_before = actor.build_graph_snapshot(env)

    if changed_input == "calendar":
        worker_id = task.base_team[0]
        intervals = env.state.workers[worker_id].intervals
        start = max(
            (interval.end for interval in intervals),
            default=env.state.current_time,
        ) + 1.0
        env.state.workers[worker_id].add_interval(
            start,
            start + 1.0,
            "calendar_probe",
        )
        graph_after = actor.build_graph_snapshot(env)
        assert not torch.equal(
            graph_before["worker"].x[builder.worker_id_to_idx[worker_id]],
            graph_after["worker"].x[builder.worker_id_to_idx[worker_id]],
        )
        changed_worker_id = worker_id
    elif changed_input == "skill":
        worker_id = next(
            worker
            for worker in env.state.station_worker_bindings[task.current_station]
            if len(env.worker_skills[worker]) < 5
        )
        available_skills = set(env.worker_skills[worker_id])
        added_skill = next(skill for skill in range(5) if skill not in available_skills)
        env.worker_skills[worker_id] = frozenset((*available_skills, added_skill))
        graph_after = actor.build_graph_snapshot(env)
        assert not torch.equal(
            graph_before["worker"].x[builder.worker_id_to_idx[worker_id]],
            graph_after["worker"].x[builder.worker_id_to_idx[worker_id]],
        )
        changed_worker_id = worker_id
    else:
        team = tuple(task.base_team)
        changed_worker_id = next(
            worker
            for worker in env.state.station_worker_bindings[task.current_station]
            if worker not in team
        )
        task.base_team = (changed_worker_id, *team[1:])
        graph_after = actor.build_graph_snapshot(env)
        assert not torch.equal(
            graph_before["task", "baseline_team", "worker"].edge_index,
            graph_after["task", "baseline_team", "worker"].edge_index,
        )

    with torch.no_grad():
        worker_nodes_before = actor.graph_encoder(graph_before)[2]
        worker_nodes_after = actor.graph_encoder(graph_after)[2]
    worker_index = builder.worker_id_to_idx[changed_worker_id]
    observed_head_inputs: list[torch.Tensor] = []
    hook = actor.worker_graph_score.register_forward_pre_hook(
        lambda _module, inputs: observed_head_inputs.append(
            inputs[0].detach().cpu().clone()
        )
    )
    try:
        actor.worker_graph_score(worker_nodes_before[worker_index].unsqueeze(0))
        actor.worker_graph_score(worker_nodes_after[worker_index].unsqueeze(0))
    finally:
        hook.remove()

    assert len(observed_head_inputs) == 2
    assert not torch.equal(observed_head_inputs[0], observed_head_inputs[1])


def test_training_entry_traces_graph_policy_and_updates_graph_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实训练入口应构图、使用图表征完成动作，并更新图编码器。"""
    from models.work3.actor_critic import ActorCriticWork3
    from scripts.work3.train_ppo_work3 import run_training

    trace: dict[str, Any] = {
        "sampled_actions": [],
        "graph_encoder": 0,
        "task_graph_proj": 0,
        "worker_graph_score": 0,
        "task_score_fc": 0,
        "worker_score_fc": 0,
    }
    initial_graph_state: dict[str, torch.Tensor] = {}
    original_actor_init = ActorCriticWork3.__init__
    original_select_snapshot = ActorCriticWork3.select_snapshot

    def capture_actor_init(actor: ActorCriticWork3, *args: Any, **kwargs: Any) -> None:
        original_actor_init(actor, *args, **kwargs)
        if initial_graph_state:
            return
        initial_graph_state.update({
            name: parameter.detach().cpu().clone()
            for name, parameter in actor.graph_encoder.named_parameters()
        })
        for module_name in (
            "graph_encoder",
            "task_graph_proj",
            "worker_graph_score",
            "task_score_fc",
            "worker_score_fc",
        ):
            getattr(actor, module_name).register_forward_hook(
                lambda _module, _inputs, _output, name=module_name: trace.__setitem__(
                    name,
                    trace[name] + 1,
                )
            )

    def capture_sampled_action(
        actor: ActorCriticWork3,
        snapshot: Any,
        deterministic: bool = False,
    ) -> Any:
        result = original_select_snapshot(actor, snapshot, deterministic=deterministic)
        assert isinstance(snapshot.graph_snapshot, HeteroData)
        trace["sampled_actions"].append((result[0], result[3]))
        return result

    monkeypatch.setattr(ActorCriticWork3, "__init__", capture_actor_init)
    monkeypatch.setattr(ActorCriticWork3, "select_snapshot", capture_sampled_action)

    report = run_training(
        run_mode="smoke",
        num_iterations=1,
        steps_per_iter=4,
        max_decisions=4,
        ppo_epochs=1,
        batch_size=4,
        seed=42,
        method_variant="C",
        device="cpu",
        num_envs=1,
        output_ckpt=tmp_path / "graph_policy_smoke.pt",
        report_path=tmp_path / "graph_policy_smoke.run.json",
    )

    assert report["history"][0]["environment_steps"] == 4
    assert report["lightning_optimization_steps"] == 1
    assert trace["graph_encoder"] > 0
    assert trace["task_graph_proj"] > 0
    assert trace["task_score_fc"] > 0
    assert trace["sampled_actions"]
    assert any(
        sample_record["worker_node_indices"]
        for _action, sample_record in trace["sampled_actions"]
    )
    assert trace["worker_graph_score"] > 0
    assert trace["worker_score_fc"] > 0

    checkpoint = torch.load(report["checkpoint_path"], map_location="cpu", weights_only=False)
    trained_graph_state = checkpoint["actor_critic_state"]
    changed_graph_parameters = [
        name
        for name, initial_value in initial_graph_state.items()
        if not torch.equal(
            initial_value,
            trained_graph_state[f"graph_encoder.{name}"].cpu(),
        )
    ]
    assert changed_graph_parameters


def test_legacy_actor_checkpoint_can_be_loaded(baseline_path: Path) -> None:
    """已有工作三旧 Actor 检查点应显式兼容，不阻塞图策略接入。"""
    checkpoint = Path("models/work3/checkpoints/method_d_model.pt")
    if not checkpoint.is_file():
        pytest.skip(f"{checkpoint} 不存在")
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = data.get("actor_critic_state", data)

    net = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    net.load_state_dict(state_dict)
    assert net.graph_policy_enabled is False
