"""W3-13运行时的新配置、种子和并行训练边界测试。"""

from __future__ import annotations

import importlib
import importlib.util
import random
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT_DIR / "conf" / "work3" / "train_pilot.yaml"


@pytest.fixture
def restore_global_random_state() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def _runtime_config_module() -> ModuleType:
    """将缺少的新运行时模块呈现为清晰的测试失败，而不是导入错误。"""
    if importlib.util.find_spec("training.work3_runtime_config") is None:
        pytest.fail("工作三运行时配置加载器尚未实现", pytrace=False)
    return importlib.import_module("training.work3_runtime_config")


def _decision_snapshot_module() -> ModuleType:
    if importlib.util.find_spec("envs.work3.decision_snapshot") is None:
        pytest.fail("工作三决策快照与纯团队掩码尚未实现", pytrace=False)
    return importlib.import_module("envs.work3.decision_snapshot")


def test_runtime_config_loads_smoke_defaults_and_hashes_resolved_yaml() -> None:
    module = _runtime_config_module()

    config = module.load_work3_runtime_config(DEFAULT_CONFIG)
    resolved_yaml, fingerprint = module.resolved_config_fingerprint(config)

    assert config.runtime.run_mode == "smoke"
    assert config.runtime.method_profile == "C"
    assert config.runtime.num_envs == 1
    assert config.runtime.amp_dtype == "fp32"
    assert config.runtime.start_method == "spawn"
    assert config.runtime.dataloader_num_workers == 0
    assert config.runtime.total_env_steps == 64
    assert "num_envs: 1" in resolved_yaml
    assert len(fingerprint) == 64
    assert set(fingerprint) <= set("0123456789abcdef")


def test_runtime_config_overrides_change_resolved_values_and_fingerprint() -> None:
    module = _runtime_config_module()

    default = module.load_work3_runtime_config(DEFAULT_CONFIG)
    overridden = module.load_work3_runtime_config(
        DEFAULT_CONFIG,
        overrides=("runtime.seed=31415", "runtime.num_envs=2"),
    )
    _, default_fingerprint = module.resolved_config_fingerprint(default)
    resolved_yaml, override_fingerprint = module.resolved_config_fingerprint(overridden)

    assert overridden.runtime.seed == 31415
    assert overridden.runtime.num_envs == 2
    assert "seed: 31415" in resolved_yaml
    assert override_fingerprint != default_fingerprint


def test_runtime_config_rejects_missing_required_fields(tmp_path: Path) -> None:
    module = _runtime_config_module()
    incomplete_config = tmp_path / "incomplete.yaml"
    incomplete_config.write_text("runtime:\n  run_mode: smoke\n", encoding="utf-8")

    with pytest.raises(ValueError, match="缺少必需配置"):
        module.load_work3_runtime_config(incomplete_config)


@pytest.mark.parametrize(
    "override",
    (
        "runtime.num_envs=0",
        "runtime.amp_dtype=fp8",
        "runtime.total_env_steps=0",
        "runtime.dataloader_num_workers=1",
        "runtime.start_method=fork",
        "ppo.gamma=1.2",
        "paths.baseline=",
        "paths.baseline=' '",
    ),
)
def test_runtime_config_rejects_unsupported_or_out_of_range_values(override: str) -> None:
    module = _runtime_config_module()

    with pytest.raises((ValueError, TypeError)):
        module.load_work3_runtime_config(DEFAULT_CONFIG, overrides=(override,))


def test_runtime_seed_and_worker_episode_seed_are_reproducible(
    restore_global_random_state: None,
) -> None:
    module = _runtime_config_module()

    worker_seed = module.derive_worker_seed(42, worker_id=1, episode_index=3)
    assert worker_seed == module.derive_worker_seed(42, worker_id=1, episode_index=3)
    assert worker_seed != module.derive_worker_seed(42, worker_id=0, episode_index=3)
    assert worker_seed != module.derive_worker_seed(42, worker_id=1, episode_index=4)

    module.seed_work3_runtime(1234, deterministic=True)
    first = (random.random(), float(np.random.random()), torch.rand(3))
    module.seed_work3_runtime(1234, deterministic=True)
    second = (random.random(), float(np.random.random()), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.are_deterministic_algorithms_enabled()


def test_pilot_config_requires_a_positive_successful_batch_target() -> None:
    module = _runtime_config_module()

    with pytest.raises(ValueError, match="successful_batch_target"):
        module.load_work3_runtime_config(
            DEFAULT_CONFIG,
            overrides=("runtime.run_mode=pilot",),
        )

    pilot_config = module.load_work3_runtime_config(
        DEFAULT_CONFIG,
        overrides=(
            "runtime.run_mode=pilot",
            "runtime.successful_batch_target=1",
        ),
    )
    assert pilot_config.runtime.successful_batch_target == 1


def test_snapshot_worker_mask_preserves_legal_team_completion_and_order() -> None:
    module = _decision_snapshot_module()
    worker = module.WorkerSnapshot
    context_type = module.TeamCompletionContext
    context = context_type(
        task_key="7_12",
        station_id=1,
        required_skill=3,
        demand=2,
        workers=(
            worker(18, (3,), 1.0, ((20.0, 22.0, "other"),)),
            worker(65, (3,), 0.9, ()),
            worker(91, (1,), 1.1, ()),
        ),
    )

    first_pointer = module.worker_completion_mask(context, (), max_station_workers=5)
    second_pointer = module.worker_completion_mask(context, (18,), max_station_workers=5)
    impossible_prefix = module.worker_completion_mask(
        context,
        (18, 65),
        max_station_workers=5,
    )

    assert first_pointer == (True, True, False, False, False)
    assert second_pointer == (False, True, False, False, False)
    assert impossible_prefix == (False, False, False, False, False)


def test_snapshot_worker_mask_disables_task_without_a_full_skill_team() -> None:
    module = _decision_snapshot_module()
    context = module.TeamCompletionContext(
        task_key="2_9",
        station_id=2,
        required_skill=4,
        demand=2,
        workers=(
            module.WorkerSnapshot(31, (4,), 1.0, ()),
            module.WorkerSnapshot(32, (1, 2), 1.0, ()),
        ),
    )

    assert module.worker_completion_mask(context, (), max_station_workers=3) == (
        False,
        False,
        False,
    )


def test_environment_team_context_is_an_independent_cpu_snapshot() -> None:
    from envs.work3.environment import AirLineEnvWork3

    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()
    task = env.get_action_candidates()[0]
    context = env.get_team_completion_context(task)

    station_workers = tuple(env.state.station_worker_bindings[task.current_station])
    assert tuple(worker.worker_id for worker in context.workers) == station_workers
    assert all(isinstance(worker.calendar_intervals, tuple) for worker in context.workers)
    original_skills = context.workers[0].skills
    original_intervals = context.workers[0].calendar_intervals

    env.worker_skills[station_workers[0]] = frozenset({99})
    env.state.workers[station_workers[0]].add_interval(100.0, 101.0, "later")

    assert context.workers[0].skills == original_skills
    assert context.workers[0].calendar_intervals == original_intervals


def test_decision_snapshot_clones_tensor_and_graph_payloads_to_cpu() -> None:
    from torch_geometric.data import HeteroData

    module = _decision_snapshot_module()
    context = module.TeamCompletionContext(
        task_key="0_0",
        station_id=0,
        required_skill=-1,
        demand=1,
        workers=(module.WorkerSnapshot(4, (0,), 1.0, ()),),
    )
    state_features = torch.ones(32)
    time_features = torch.tensor([0.25, 0.5])
    task_features = torch.ones((1, 8))
    graph = HeteroData()
    graph["task"].x = torch.ones((1, 3))

    snapshot = module.DecisionSnapshot(
        worker_id=0,
        episode_id=12,
        episode_index=3,
        state_features=state_features,
        time_features=time_features,
        graph_snapshot=graph,
        candidate_task_keys=("0_0",),
        candidate_task_features=task_features,
        branch_masks=((True, False),),
        team_contexts=(context,),
        candidate_task_node_indices=(0,),
        worker_node_indices=((0,),),
        reserved_flags=(False,),
        advance_available=False,
        current_time=5.0,
        cycle_id=1,
        estimated_cmax=10.0,
        h0=10.0,
        last_transfer_time=0.0,
        graph_version="work3_graph_v2",
    )

    state_features[0] = 99.0
    time_features[0] = 99.0
    task_features[0, 0] = 99.0
    graph["task"].x[0, 0] = 99.0

    assert snapshot.state_features.device.type == "cpu"
    assert snapshot.time_features.device.type == "cpu"
    assert snapshot.candidate_task_features.device.type == "cpu"
    assert snapshot.state_features[0].item() == 1.0
    assert snapshot.time_features[0].item() == 0.25
    assert snapshot.candidate_task_features[0, 0].item() == 1.0
    assert snapshot.graph_snapshot["task"].x[0, 0].item() == 1.0


def test_snapshot_masks_match_environment_and_actor_replay_probability() -> None:
    from dataclasses import replace
    from itertools import permutations

    from envs.work3.environment import AirLineEnvWork3
    from models.work3.actor_critic import ActorCriticWork3

    env = AirLineEnvWork3(
        baseline_json_path=ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    )
    env.reset()
    actor = ActorCriticWork3(hidden_dim=32)
    actor.eval()
    with torch.no_grad():
        actor.branch_head[-1].bias[0] = 100.0
        actor.branch_head[-1].bias[1] = -100.0
    state_features = torch.zeros(32)
    time_features = torch.zeros(2)

    snapshot = actor.make_decision_snapshot(
        env,
        state_features,
        time_features,
        worker_id=2,
        episode_id=7,
        episode_index=3,
    )
    live_candidates = env.get_action_candidates()
    assert snapshot.candidate_task_keys == tuple(task.task_key for task in live_candidates)
    assert snapshot.branch_masks == tuple(
        env.get_action_branch_mask(task) for task in live_candidates
    )
    with pytest.raises(ValueError, match="图版本"):
        actor.select_snapshot(replace(snapshot, graph_version="stale_graph"))

    for task, context in zip(live_candidates, snapshot.team_contexts, strict=True):
        assert tuple(worker.worker_id for worker in context.workers) == tuple(
            env.state.station_worker_bindings[task.current_station]
        )
        prefixes = [()]
        for prefix_length in range(1, task.demand):
            prefixes.extend(
                tuple(prefix)
                for prefix in permutations(
                    env.state.station_worker_bindings[task.current_station],
                    prefix_length,
                )
            )
        for prefix in prefixes:
            expected = tuple(
                env.valid_team_completion_workers(task, prefix)
            )
            actual = _decision_snapshot_module().team_completion_worker_ids(
                context, prefix
            )
            assert actual == expected

    snapshot_result = actor.select_snapshot(snapshot, deterministic=True)
    live_result = actor.select_action(
        env, state_features, time_features, deterministic=True
    )
    assert snapshot_result[0] == live_result[0]
    assert snapshot_result[1] == pytest.approx(live_result[1], abs=1e-6)
    record = snapshot_result[3]
    assert record["candidate_branch_masks"] == snapshot.branch_masks
    assert snapshot_result[0]["branch"] == 0
    _, replay_log_probs, _ = actor.evaluate_action_log_probs(
        state_features.unsqueeze(0), time_features.unsqueeze(0), [record]
    )
    assert float(replay_log_probs[0]) == pytest.approx(snapshot_result[1], abs=1e-6)
