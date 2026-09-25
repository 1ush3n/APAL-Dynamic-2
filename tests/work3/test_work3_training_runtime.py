"""W3-13运行时的新配置、种子和并行训练边界测试。"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

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


def _work3_vector_env_module() -> ModuleType:
    if importlib.util.find_spec("training.work3_vector_env") is None:
        pytest.fail("工作三spawn环境运行时尚未实现", pytrace=False)
    return importlib.import_module("training.work3_vector_env")


def _work3_lightning_module() -> ModuleType:
    if importlib.util.find_spec("training.work3_lightning") is None:
        pytest.fail("工作三Lightning训练生命周期尚未实现", pytrace=False)
    return importlib.import_module("training.work3_lightning")


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


def test_training_cli_uses_yaml_config_and_passes_resolved_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from scripts.work3 import train_ppo_work3

    runtime_config_module = _runtime_config_module()
    overrides = (
        "runtime.seed=31415",
        "runtime.num_envs=2",
        "ppo.steps_per_iter=8",
        "runtime.device=cuda",
        "runtime.amp_dtype=bf16",
    )
    resolved_config = runtime_config_module.load_work3_runtime_config(
        DEFAULT_CONFIG,
        overrides=overrides,
    )
    resolved_yaml, fingerprint = runtime_config_module.resolved_config_fingerprint(
        resolved_config
    )
    captured: dict[str, object] = {}

    def capture_training_call(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(train_ppo_work3, "run_training", capture_training_call)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo_work3",
            "--config",
            DEFAULT_CONFIG.as_posix(),
            *[argument for value in overrides for argument in ("--set", value)],
        ],
    )

    train_ppo_work3.main()

    assert captured["seed"] == 31415
    assert captured["num_envs"] == 2
    assert captured["steps_per_iter"] == 8
    assert captured["run_mode"] == "smoke"
    assert captured["method_variant"] == "C"
    assert captured["max_decisions"] == 8
    assert captured["deterministic"] is True
    assert captured["main_num_threads"] == 1
    assert captured["env_num_threads"] == 1
    assert captured["settle_timeout_seconds"] == 2.0
    assert captured["amp_dtype"] == "bf16"
    assert captured["resolved_config_yaml"] == resolved_yaml
    assert captured["resolved_config_sha256"] == fingerprint


def test_amp_precision_resolver_rejects_mixed_precision_on_cpu() -> None:
    runtime = _runtime_config_module()

    with pytest.raises(ValueError, match="CUDA"):
        runtime.resolve_work3_precision("fp16", torch.device("cpu"))


@pytest.mark.parametrize("amp_dtype", ["fp16", "bf16"])
def test_actor_sampling_and_replay_keep_probability_math_fp32_under_amp(
    amp_dtype: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA AMP验收需要CUDA设备")
    if amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("当前CUDA设备不支持bfloat16")

    from envs.work3.environment import AirLineEnvWork3
    from models.work3.actor_critic import ActorCriticWork3
    from scripts.work3.train_ppo_work3 import ROOT_DIR

    environment = AirLineEnvWork3(
        baseline_json_path=ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    )
    environment.reset()
    actor = ActorCriticWork3(hidden_dim=16).to("cuda").eval()
    state_features = torch.zeros(32)
    time_features = torch.zeros(2)
    snapshot = actor.make_decision_snapshot(
        environment,
        state_features,
        time_features,
        worker_id=0,
        episode_id=0,
        episode_index=0,
    )
    amp_torch_dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16

    with torch.amp.autocast(device_type="cuda", dtype=amp_torch_dtype):
        action, sampled_log_prob, _value, sample_record = actor.select_snapshot(
            snapshot,
            deterministic=True,
        )
        assert action is not None
        _values, replay_log_probs, entropies = actor.evaluate_action_log_probs(
            state_features.unsqueeze(0).to("cuda"),
            time_features.unsqueeze(0).to("cuda"),
            [sample_record],
        )
        from models.work3.ppo_trainer import PPOTrainerWork3

        objective = PPOTrainerWork3(
            actor_critic=actor,
            device="cuda",
            create_optimizer=False,
        )
        losses = objective.compute_ppo_minibatch_loss(
            {
                "state_feats": state_features.unsqueeze(0),
                "time_urgencies": time_features.unsqueeze(0),
                "old_log_probs": torch.tensor([sampled_log_prob]),
                "advantages": torch.tensor([1.0]),
                "target_values": torch.tensor([0.0]),
                "sample_records": [sample_record],
            }
        )
        from models.work3.time_head import TimeResidualHead

        time_head = TimeResidualHead(in_dim=16, hidden_dim=16).to("cuda")
        auxiliary_objective = PPOTrainerWork3(
            actor_critic=actor,
            time_head=time_head,
            device="cuda",
            create_optimizer=False,
        )
        time_loss = auxiliary_objective.compute_time_auxiliary_loss(
            {
                "state_feats": state_features.unsqueeze(0),
                "graph_snapshots": [snapshot.graph_snapshot],
                "target_residuals": torch.tensor([-0.2]),
                "worker_ids": [0],
                "episode_ids": [0],
                "cycle_ids": [0],
                "decision_ids": [0],
            }
        )

    assert torch.isfinite(torch.tensor(sampled_log_prob))
    assert replay_log_probs.dtype == torch.float32
    assert entropies.dtype == torch.float32
    assert torch.isfinite(replay_log_probs).all()
    assert torch.isfinite(entropies).all()
    assert float(replay_log_probs[0]) == pytest.approx(sampled_log_prob, abs=1e-6)
    for key in (
        "values",
        "new_log_probs",
        "entropies",
        "ratio",
        "log_ratio",
        "advantages",
        "target_values",
        "total_loss",
    ):
        if key in losses:
            assert losses[key].dtype == torch.float32
            assert torch.isfinite(losses[key]).all()
    losses["total_loss"].backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in actor.parameters()
    )
    actor.zero_grad(set_to_none=True)
    assert time_loss.dtype == torch.float32
    assert torch.isfinite(time_loss)
    time_loss.backward()
    shared_gradient = next(actor.graph_encoder.parameters()).grad
    time_head_gradient = next(time_head.parameters()).grad
    assert shared_gradient is not None and torch.isfinite(shared_gradient).all()
    assert time_head_gradient is not None and torch.isfinite(time_head_gradient).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="训练入口AMP验收需要CUDA设备")
@pytest.mark.parametrize(
    ("amp_dtype", "lightning_precision"),
    [("fp16", "16-mixed"), ("bf16", "bf16-mixed")],
)
def test_training_entry_uses_configured_amp_for_rollout_and_lightning(
    tmp_path: Path,
    amp_dtype: str,
    lightning_precision: str,
) -> None:
    if amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("当前CUDA设备不支持bfloat16")

    from scripts.work3.train_ppo_work3 import run_training

    report = run_training(
        run_mode="smoke",
        num_iterations=1,
        steps_per_iter=1,
        max_decisions=1,
        ppo_epochs=1,
        batch_size=1,
        method_variant="D",
        amp_dtype=amp_dtype,
        device="cuda",
        output_ckpt=tmp_path / "amp_bf16_smoke.pt",
        report_path=tmp_path / "amp_bf16_smoke.run.json",
    )

    assert report["amp_dtype"] == amp_dtype
    assert report["lightning_precision"] == lightning_precision
    assert report["method_variant"] == "D"
    assert report["memory_peak_kind"] == "cuda_max_memory_allocated"
    assert report["lightning_grad_scaler_enabled"] is (amp_dtype == "fp16")
    assert report["history"][0]["environment_steps"] == 1
    assert report["history"][0]["lightning_optimizer_step_attempts"] == 1
    assert (
        report["history"][0]["lightning_optimization_steps"]
        + report["history"][0]["lightning_amp_skipped_steps"]
        == 1
    )
    assert report["history"][0]["sampling_replay_max_abs_error"] <= 1e-6
    for key in ("total_loss", "policy_loss", "value_loss", "entropy", "grad_norm"):
        assert torch.isfinite(torch.tensor(report["history"][0][key]))
    checkpoint = torch.load(
        report["checkpoint_path"],
        map_location="cpu",
        weights_only=False,
    )
    precision_state = checkpoint["lightning_training_state"]["precision_state"]
    assert precision_state["precision"] == lightning_precision
    assert isinstance(precision_state["grad_scaler_state"], dict) is (
        amp_dtype == "fp16"
    )


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


def test_live_team_eligibility_does_not_construct_full_calendar_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from envs.work3.environment import AirLineEnvWork3

    env = AirLineEnvWork3(
        baseline_json_path=ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    )
    env.reset()
    task = env.get_ready_tasks()[0]

    def reject_snapshot_construction(*args: object, **kwargs: object) -> None:
        raise AssertionError("普通团队资格查询不应构造日历快照DTO")

    monkeypatch.setattr(env, "_team_completion_context", reject_snapshot_construction)
    valid_workers = env.valid_team_completion_workers(task, [])
    assert len(valid_workers) >= task.demand


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
    selected_task_index = snapshot.candidate_task_keys.index(
        snapshot_result[0]["task_key"]
    )
    selected_branch = int(snapshot_result[0]["branch"])
    assert snapshot.branch_masks[selected_task_index][selected_branch]
    _, replay_log_probs, _ = actor.evaluate_action_log_probs(
        state_features.unsqueeze(0), time_features.unsqueeze(0), [record]
    )
    assert float(replay_log_probs[0]) == pytest.approx(snapshot_result[1], abs=1e-6)


def test_single_env_worker_matches_direct_environment_for_frozen_actions() -> None:
    from copy import deepcopy

    from envs.work3.core_types import ActionBranch, TaskRuntimeState
    from envs.work3.environment import AirLineEnvWork3
    from models.work3.heuristic_agent import HeuristicAgentWork3
    from scripts.work3.train_ppo_work3 import create_work3_single_env_runtime
    from scripts.work3.evaluate_c_vs_d import _check_completed_trajectory_feasibility
    from utils.work3.trajectory_feasibility import (
        TaskConstraintRecord,
        TrajectoryExecutionRecord,
        validate_trajectory,
    )

    _work3_vector_env_module()
    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    if not baseline_path.is_file():
        pytest.skip(f"固定动作对照实例不存在: {baseline_path}")

    planning_env = AirLineEnvWork3(baseline_json_path=baseline_path)
    planning_env.reset()
    agent = HeuristicAgentWork3(name="SingleEnvParityFixture")
    frozen_actions: list[dict[str, object]] = []
    for _ in range(12000):
        if planning_env._check_terminated():
            break
        action = agent.select_action(planning_env)
        if action is None:
            action = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
        frozen_actions.append(deepcopy(action))
        _obs, _reward, terminated, truncated, info = planning_env.step(action)
        assert not truncated
        assert not (terminated and info.get("termination_reason") == "deadlock")
    assert planning_env._check_terminated()
    assert len(frozen_actions) < 12000
    action_branches = tuple(
        ActionBranch(action.get("branch", ActionBranch.STATION_EXECUTE))
        for action in frozen_actions
    )
    assert ActionBranch.ADVANCE_TO_NEXT_EVENT in action_branches
    assert ActionBranch.STATION_EXECUTE in action_branches

    direct_env = AirLineEnvWork3(baseline_json_path=baseline_path)
    direct_env.reset()
    direct_event_trace: list[tuple[float, str, int, str | None, int]] = []
    direct_start_station_by_task: dict[str, int] = {}
    original_pop = direct_env.event_queue.pop

    def traced_pop() -> object:
        event = original_pop()
        if event is not None:
            direct_event_trace.append(
                (
                    float(event.timestamp),
                    event.event_type.name,
                    int(event.event_id),
                    event.task_key,
                    int(event.generation),
                )
            )
        return event

    direct_env.event_queue.pop = traced_pop  # type: ignore[method-assign]
    original_started = direct_env._on_task_started

    def traced_started(task: TaskRuntimeState, start_time: float) -> None:
        original_started(task, start_time)
        if task.actual_start is not None:
            direct_start_station_by_task[task.task_key] = int(
                direct_env.state.aircraft[task.aircraft_id].current_station
            )

    direct_env._on_task_started = traced_started  # type: ignore[method-assign]
    direct_completed: set[str] = set()
    direct_steps = 0
    worker = create_work3_single_env_runtime(baseline_path)
    try:
        initial_observation = worker.reset(scenario=None, episode_id=21, episode_index=0)
        snapshot = worker.snapshot()
        assert snapshot.current_time == direct_env.state.current_time
        assert snapshot.state_features.device.type == "cpu"
        assert all(value.device.type == "cpu" for value in snapshot.graph_snapshot.x_dict.values())

        for action in frozen_actions:
            trace_start = len(direct_event_trace)
            _obs, expected_reward, expected_terminated, expected_truncated, expected_info = (
                direct_env.step(action)
            )
            expected_trace = tuple(direct_event_trace[trace_start:])
            expected_new_completions = []
            for task in direct_env.state.tasks.values():
                if task.actual_end is None or task.task_key in direct_completed:
                    continue
                direct_completed.add(task.task_key)
                station_id = int(task.last_published_assignment["station"])
                expected_new_completions.append(
                    (
                        task.task_key,
                        int(task.aircraft_id),
                        int(task.task_id),
                        station_id,
                        tuple(task.assigned_team),
                        float(task.actual_start),
                        float(task.actual_end),
                        direct_start_station_by_task[task.task_key],
                    )
                )

            actual = worker.step(action)
            direct_steps += 1
            assert actual.step_count == direct_steps
            assert actual.observation == _obs
            assert actual.raw_reward == pytest.approx(expected_reward, abs=1e-9)
            assert actual.terminated is expected_terminated
            assert actual.truncated is expected_truncated
            assert actual.info == expected_info
            assert actual.processed_events == expected_trace
            actual_new_completions = tuple(
                (
                    item["task_key"],
                    item["aircraft_id"],
                    item["task_id"],
                    item["station_id"],
                    tuple(item["team"]),
                    item["start"],
                    item["end"],
                    item["aircraft_station_at_start"],
                )
                for item in actual.completed_execution_records
            )
            assert actual_new_completions == tuple(expected_new_completions)
            if actual.terminated:
                break

        assert direct_steps == len(frozen_actions)
        assert direct_env._check_terminated()
        audit = worker.close()
        assert not worker.is_alive
    finally:
        if worker.is_alive:
            worker.close()

    assert initial_observation["current_time"] == 0.0
    assert audit["success"] is True
    assert audit["completed_tasks"] == audit["total_tasks"] == len(direct_env.state.tasks)
    assert audit["step_count"] == direct_env.step_count == direct_steps
    assert list(audit["transfer_history"]) == direct_env.state.transfer_history
    assert audit["cost_breakdown"] == {
        "cost_takt": direct_env.cost_takt,
        "cost_time": direct_env.cost_time,
        "cost_team": direct_env.cost_team,
        "cost_postpone": direct_env.cost_postpone,
        "cost_revision": direct_env.cost_revision,
    }

    def independently_check(serialized_audit: dict[str, object]) -> object:
        constraints = {
            int(task_id): TaskConstraintRecord(
                demand=int(value["demand"]),
                required_skill=int(value["required_skill"]),
                predecessors=tuple(value["predecessors"]),
                fixed_station=value["fixed_station"],
                max_allowed_station=value["max_allowed_station"],
            )
            for task_id, value in serialized_audit["task_constraints"].items()
        }
        records = [
            TrajectoryExecutionRecord(
                aircraft_id=int(item["aircraft_id"]),
                task_id=int(item["task_id"]),
                station_id=int(item["station_id"]),
                team=tuple(item["team"]),
                start=float(item["start"]),
                end=float(item["end"]),
                material_ready_time=float(item["material_ready_time"]),
                station_entry_time=item["station_entry_time"],
                aircraft_station_at_start=int(item["aircraft_station_at_start"]),
            )
            for item in serialized_audit["execution_records"]
        ]
        return validate_trajectory(
            records,
            task_constraints=constraints,
            worker_skills=serialized_audit["worker_skills"],
            worker_station_bindings=serialized_audit["worker_station_bindings"],
            station_capacities=serialized_audit["station_capacities"],
        )

    report = independently_check(audit)
    assert report.is_feasible
    assert not report.violations
    direct_feasible, direct_violations = _check_completed_trajectory_feasibility(direct_env)
    assert direct_feasible == report.is_feasible
    assert direct_violations == report.violations
    assert len(direct_completed) == len(direct_env.state.tasks)


def test_worker_scenario_plan_is_stable_for_worker_episode_coordinates() -> None:
    from scripts.work3.train_ppo_work3 import build_worker_scenario_plan

    scenarios = [
        {"scenario_id": f"scenario-{index}", "tau": float(index)}
        for index in range(5)
    ]
    first = build_worker_scenario_plan(
        scenarios,
        num_workers=2,
        episodes_per_worker=3,
        seed=41,
    )
    second = build_worker_scenario_plan(
        scenarios,
        num_workers=2,
        episodes_per_worker=3,
        seed=41,
    )

    assert first == second
    assert [(item["worker_id"], item["episode_index"]) for item in first] == [
        (worker_id, episode_index)
        for episode_index in range(3)
        for worker_id in range(2)
    ]
    assert all(item["scenario"]["scenario_id"] in {
        scenario["scenario_id"] for scenario in scenarios
    } for item in first)


def test_two_spawn_workers_share_one_main_actor_and_enforce_aggregate_step_budget() -> None:
    from envs.work3.core_types import ActionBranch
    from envs.work3.environment import AirLineEnvWork3
    from models.work3.actor_critic import ActorCriticWork3
    from models.work3.heuristic_agent import HeuristicAgentWork3
    from training.work3_vector_env import Work3VectorEnv

    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    planning_env = AirLineEnvWork3(baseline_json_path=baseline_path)
    planning_env.reset()
    agent = HeuristicAgentWork3(name="VectorBudgetFixture")
    frozen_actions: list[dict[str, object]] = []
    for _ in range(32):
        action = agent.select_action(planning_env)
        if action is None:
            action = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
        frozen_actions.append(action)
        planning_env.step(action)
        if action.get("branch") == ActionBranch.ADVANCE_TO_NEXT_EVENT:
            break
    assert frozen_actions[-1]["branch"] == ActionBranch.ADVANCE_TO_NEXT_EVENT

    vector = Work3VectorEnv(
        env_kwargs={"baseline_json_path": str(baseline_path)},
        num_envs=2,
        worker_torch_num_threads=2,
        start_method="spawn",
    )
    actor = ActorCriticWork3(hidden_dim=32)
    try:
        assert len(vector.worker_pids) == 2
        assert len(set(vector.worker_pids)) == 2
        assert all(pid != os.getpid() for pid in vector.worker_pids)
        assert vector.workers_alive == (True, True)
        assert vector.worker_cuda_initialized == (False, False)
        assert vector.worker_torch_num_threads == (2, 2)

        resets = vector.reset_all(
            scenarios=(None, None),
            episode_ids=(11, 12),
            episode_indices=(0, 0),
        )
        assert len(resets) == 2
        snapshots = vector.snapshots()
        assert snapshots[0].current_time == snapshots[1].current_time == 0.0
        assert all(snapshot.graph_snapshot["task"].x.device.type == "cpu" for snapshot in snapshots)
        assert actor.select_snapshot(snapshots[0], deterministic=True)[0] is not None

        observed: list[object] = []
        last_batch = None
        for action in frozen_actions:
            last_batch = vector.step_all(
                actions=(action, action),
                max_total_steps=17,
            )
            observed.extend(result for result in last_batch.results if result is not None)
        assert last_batch is not None
        assert last_batch.results[0] is not None
        assert last_batch.results[1] is None
        assert vector.total_env_steps == 17
        assert vector.budget_reserved_steps == 17
        assert len(observed) == 17
        assert vector.worker_step_counts == (9, 8)

        snapshots = vector.snapshots()
        unbounded_actions = tuple(
            actor.select_snapshot(snapshot, deterministic=True)[0]
            for snapshot in snapshots
        )
        assert all(action is not None for action in unbounded_actions)
        unbounded_batch = vector.step_all(
            actions=unbounded_actions,
            max_total_steps=None,
        )
        assert unbounded_batch.dispatched_worker_ids == (0, 1)
        assert vector.total_env_steps == 19
        assert vector.budget_reserved_steps == 19
    finally:
        vector.close()
    assert vector.workers_alive == (False, False)


def test_vector_reset_reports_fixed_scenario_and_actual_hit_status() -> None:
    from scripts.work3.train_ppo_work3 import build_worker_scenario_plan
    from training.work3_vector_env import Work3VectorEnv

    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    scenarios = [
        {
            "scenario_id": "VECTOR_T0_HIT",
            "tau": 0.0,
            "recovery_time": 1.0,
            "affected_task_keys": ["0_15"],
        },
        {
            "scenario_id": "VECTOR_NOT_YET_TRIGGERED",
            "tau": 1000.0,
            "recovery_time": 1001.0,
            "affected_task_keys": ["missing-task"],
        },
    ]
    plan = build_worker_scenario_plan(
        scenarios,
        num_workers=2,
        episodes_per_worker=1,
        seed=41,
    )
    with Work3VectorEnv(
        env_kwargs={"baseline_json_path": str(baseline_path)},
        num_envs=2,
        start_method="spawn",
    ) as vector:
        resets = vector.reset_from_plan(plan)

    for reset, planned in zip(resets, plan, strict=True):
        status = reset.scenario_status
        assert status["scenario_id"] == planned["scenario"]["scenario_id"]
        assert status["started"] is True
        assert status["scheduled_target_count"] == 1
        assert status["in_flight"] is True
        if status["scenario_id"] == "VECTOR_T0_HIT":
            assert status["event_triggered"] is True
            assert status["actual_hit_task_keys"] == ("0_15",)
            assert status["actual_hit_count"] == 1
        else:
            assert status["event_triggered"] is False
            assert status["actual_hit_count"] == 0
            assert status["unhit_reasons"] == {"missing-task": "event_not_triggered"}


def test_vector_worker_keeps_started_disturbance_target_fixed_and_unhit() -> None:
    from envs.work3.core_types import ActionBranch
    from training.work3_vector_env import Work3VectorEnv
    from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline

    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    baseline = MultiAircraftBaseline.load_from_json(baseline_path)
    target_key = "0_15"
    target = baseline.get_task(0, 15)
    tau = target.duration / 2.0
    scenario = {
        "scenario_id": "VECTOR_FIXED_TARGET_ALREADY_RUNNING",
        "tau": tau,
        "recovery_time": tau + 1.0,
        "affected_task_keys": [target_key],
    }

    with Work3VectorEnv(
        env_kwargs={"baseline_json_path": str(baseline_path)},
        num_envs=1,
        start_method="spawn",
    ) as vector:
        reset = vector.reset_all(
            scenarios=(scenario,),
            episode_ids=(70,),
            episode_indices=(0,),
        )[0]
        assert reset.scenario_status["event_triggered"] is False
        snapshot = vector.snapshot()
        assert target_key in snapshot.candidate_task_keys

        started = vector.step(
            {
                "task_key": target_key,
                "branch": ActionBranch.STATION_EXECUTE,
                "team": target.team,
                "align": 0,
            }
        )
        assert started.info["scheduled_status"] == "RUNNING"
        assert started.info["scheduled_start"] == pytest.approx(0.0)
        assert started.scenario_status["actual_hit_count"] == 0

        after_disturbance = vector.step(
            {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
        )

    assert any(
        event[1] == "DISTURBANCE" for event in after_disturbance.processed_events
    )
    assert after_disturbance.scenario_status["scenario_id"] == scenario["scenario_id"]
    assert after_disturbance.scenario_status["scheduled_target_count"] == 1
    assert after_disturbance.scenario_status["event_triggered"] is True
    assert after_disturbance.scenario_status["actual_hit_task_keys"] == ()
    assert after_disturbance.scenario_status["actual_hit_count"] == 0
    assert after_disturbance.scenario_status["unhit_reasons"] == {
        target_key: "already_started_or_completed_at_event"
    }


def test_disturbance_hit_with_later_existing_release_has_zero_added_wait() -> None:
    from types import SimpleNamespace

    from envs.work3.core_types import TaskStatus
    from envs.work3.environment import AirLineEnvWork3
    from scripts.work3.evaluate_c_vs_d import summarize_disturbance_effects
    from scripts.work3.train_ppo_work3 import count_actual_scenario_hits
    from training.work3_vector_env import _scenario_status

    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    env.reset()

    target_key = "0_15"
    target = env.state.tasks[target_key]
    target.status = TaskStatus.UNREADY
    target.material_ready_time = 10.0
    scenario = {
        "scenario_id": "RECOVERY_BEFORE_EXISTING_RELEASE",
        "aircraft_id": 0,
        "tau": 0.0,
        "recovery_time": 5.0,
        "affected_task_keys": [target_key],
    }
    env.load_scenario(scenario)

    assert env.disturbance_event_triggered is True
    assert target.material_ready_time == pytest.approx(10.0)
    event_result = env.disturbance_event_results[scenario["scenario_id"]]
    assert event_result["actual_hit_task_keys"] == (target_key,)
    assert event_result["unhit_reasons"] == {}
    assert count_actual_scenario_hits(env, scenario) == 1

    worker_status = _scenario_status(
        env,
        scenario,
        baseline_material_ready_by_task={target_key: 10.0},
        in_flight=True,
    )
    evaluation_status = summarize_disturbance_effects(
        {
            target_key: SimpleNamespace(
                aircraft_id=0,
                material_ready_time=10.0,
                actual_start=10.0,
            )
        },
        scenario,
        baseline_start_by_key={target_key: 10.0},
        baseline_material_ready_by_key={target_key: 10.0},
        event_triggered=env.disturbance_event_triggered,
        event_result=event_result,
    )

    assert worker_status["actual_hit_task_keys"] == (target_key,)
    assert worker_status["actual_hit_count"] == 1
    assert worker_status["material_ready_advanced_task_keys"] == ()
    assert worker_status["material_ready_advanced_count"] == 0
    assert worker_status["unhit_reasons"] == {}
    assert evaluation_status["actual_hit_task_keys"] == [target_key]
    assert evaluation_status["actual_hit_count"] == 1
    assert evaluation_status["material_ready_advanced_task_keys"] == []
    assert evaluation_status["material_ready_advanced_count"] == 0
    assert evaluation_status["unhit_reasons"] == {}
    assert evaluation_status["observed_added_wait_hours"] == pytest.approx(0.0)


def test_vector_worker_errors_and_incomplete_step_timeouts_are_explicit_and_cleaned() -> None:
    from envs.work3.core_types import ActionBranch
    from models.work3.actor_critic import ActorCriticWork3
    from training.work3_vector_env import Work3VectorEnv

    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    failed_vector = Work3VectorEnv(
        env_kwargs={"baseline_json_path": str(baseline_path)},
        num_envs=2,
        start_method="spawn",
    )
    try:
        failed_vector.reset_all(
            scenarios=(None, None),
            episode_ids=(30, 31),
            episode_indices=(0, 0),
        )
        failed_batch = failed_vector.step_all(
            actions=(
                {"branch": ActionBranch.STATION_EXECUTE},
                {"branch": ActionBranch.STATION_EXECUTE},
            ),
            max_total_steps=2,
        )
        assert failed_batch.worker_errors
        assert all("step" in message for _, message in failed_batch.worker_errors)
    finally:
        failed_vector.close()
    assert failed_vector.workers_alive == (False, False)

    timed_vector = Work3VectorEnv(
        env_kwargs={"baseline_json_path": str(baseline_path)},
        num_envs=2,
        start_method="spawn",
    )
    try:
        timed_vector.reset_all(
            scenarios=(None, None),
            episode_ids=(40, 41),
            episode_indices=(0, 0),
        )
        snapshots = timed_vector.snapshots()
        actor = ActorCriticWork3(hidden_dim=32)
        actions = tuple(
            actor.select_snapshot(snapshot, deterministic=True)[0]
            for snapshot in snapshots
        )
        assert all(action is not None for action in actions)
        batch = timed_vector.step_all(
            actions=tuple(action for action in actions if action is not None),
            max_total_steps=2,
            settle_timeout_seconds=1e-9,
        )
        assert batch.interrupted_worker_ids
        assert all(batch.results[index] is None for index in batch.interrupted_worker_ids)
        assert batch.total_env_steps <= 2
        assert batch.budget_reserved_steps == len(batch.dispatched_worker_ids)
    finally:
        timed_vector.close()
    assert timed_vector.workers_alive == (False, False)


def test_vector_expired_wall_clock_deadline_stops_dispatch_without_fake_result() -> None:
    import time

    from envs.work3.core_types import ActionBranch
    from training.work3_vector_env import Work3VectorEnv

    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    vector = Work3VectorEnv(
        env_kwargs={"baseline_json_path": str(baseline_path)},
        num_envs=1,
        start_method="spawn",
    )
    try:
        vector.reset_all(
            scenarios=(None,),
            episode_ids=(50,),
            episode_indices=(0,),
        )
        deadline = time.monotonic() - 1.0
        stopped = vector.step_all(
            actions=({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT},),
            max_total_steps=2,
            wall_clock_deadline=deadline,
        )
        assert stopped.wall_clock_expired is True
        assert stopped.dispatched_worker_ids == ()
        assert stopped.results == (None,)
        assert stopped.total_env_steps == 0
    finally:
        vector.close()
    assert vector.workers_alive == (False,)


def test_gae_is_partitioned_by_worker_episode_and_segment() -> None:
    from models.work3.ppo_buffer import PPOTransition, RolloutBufferWork3

    buffer = RolloutBufferWork3(gamma=1.0, gae_lambda=1.0, normalize_advantages=False)
    dummy_feat = torch.zeros(32)
    dummy_time = torch.zeros(2)

    def add(
        reward: float,
        *,
        worker_id: int,
        episode_id: int,
        segment_id: int,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        buffer.add(
            PPOTransition(
                state_feat=dummy_feat,
                time_urgency=dummy_time,
                sample_record={},
                reward=reward,
                raw_reward=reward,
                value=0.0,
                log_prob=0.0,
                worker_id=worker_id,
                episode_id=episode_id,
                segment_id=segment_id,
                terminated=terminated,
                truncated=truncated,
            )
        )

    add(1.0, worker_id=0, episode_id=10, segment_id=0)
    add(2.0, worker_id=1, episode_id=10, segment_id=0, terminated=True)
    add(3.0, worker_id=0, episode_id=11, segment_id=0, terminated=True)
    add(1.0, worker_id=0, episode_id=10, segment_id=0)
    add(4.0, worker_id=0, episode_id=10, segment_id=1, truncated=True)
    add(5.0, worker_id=1, episode_id=11, segment_id=1, truncated=True)

    buffer.finish_trajectories(
        last_values_by_segment={
            (0, 10, 0): 5.0,
            (0, 10, 1): 6.0,
            (1, 11, 1): 7.0,
        }
    )

    assert buffer.advantages.tolist() == pytest.approx([7.0, 2.0, 3.0, 6.0, 10.0, 12.0])
    assert buffer.target_values.tolist() == pytest.approx([7.0, 2.0, 3.0, 6.0, 10.0, 12.0])


def test_time_label_cache_is_worker_scoped_and_outlives_rollout_clear() -> None:
    from models.work3.ppo_buffer import PendingTimeLabelCache, PPOTransition, RolloutBufferWork3

    cache = PendingTimeLabelCache()
    rollout = RolloutBufferWork3(normalize_advantages=False)
    for worker_id, estimated_cmax in ((0, 10.0), (1, 20.0)):
        cache.add(
            worker_id=worker_id,
            episode_id=3,
            cycle_id=2,
            decision_id=0,
            state_feat=torch.full((32,), float(worker_id)),
            graph_snapshot={"worker": worker_id},
            estimated_cmax=estimated_cmax,
            current_time=5.0,
            h0=5.0,
            predictor_version=worker_id,
        )

    rollout.add(
        PPOTransition(
            state_feat=torch.zeros(32),
            time_urgency=torch.zeros(2),
            sample_record={},
            reward=0.0,
            raw_reward=0.0,
            value=0.0,
            log_prob=0.0,
        )
    )
    rollout.clear()
    assert cache.pending_cycle_counts(worker_id=0, episode_id=3) == {2: 1}
    assert cache.pending_cycle_counts(worker_id=1, episode_id=3) == {2: 1}
    assert cache.drain_ready() is None

    next_rollout = RolloutBufferWork3(normalize_advantages=False)
    next_rollout.add(
        PPOTransition(
            state_feat=torch.ones(32),
            time_urgency=torch.zeros(2),
            sample_record={},
            reward=0.0,
            raw_reward=0.0,
            value=0.0,
            log_prob=0.0,
        )
    )
    assert len(next_rollout) == 1
    assert cache.pending_cycle_counts(worker_id=0, episode_id=3) == {2: 1}

    assert cache.attach_transfer(
        worker_id=0,
        episode_id=3,
        cycle_id=2,
        actual_transfer_time=15.0,
    )
    ready = cache.drain_ready()
    assert ready is not None
    assert ready["worker_ids"] == [0]
    assert ready["episode_ids"] == [3]
    assert ready["target_residuals"].tolist() == pytest.approx([1.0])
    assert ready["estimated_cmax"].tolist() == pytest.approx([10.0])
    assert ready["graph_snapshots"] == [{"worker": 0}]
    assert len(next_rollout) == 1
    assert cache.pending_cycle_counts(worker_id=1, episode_id=3) == {2: 1}
    assert cache.drain_ready() is None
    cache.discard_episode(worker_id=1, episode_id=3)
    assert cache.pending_cycle_counts(worker_id=1, episode_id=3) == {}
    assert cache.drain_ready() is None


def test_episode_potential_versions_survive_overlapping_worker_episodes() -> None:
    from models.work3.potential_shaping import PotentialRewardShaper
    from models.work3.time_head import TimeResidualHead

    initial_head = TimeResidualHead(in_dim=32, hidden_dim=16, use_layer_norm=False)
    for parameter in initial_head.parameters():
        torch.nn.init.zeros_(parameter)
    initial_head.reg_fc[-1].bias.data.fill_(0.0)
    shaper = PotentialRewardShaper(time_head=initial_head)

    worker_a_version = shaper.begin_episode(worker_id=0, episode_id=7)
    worker_b_version = shaper.begin_episode(worker_id=1, episode_id=7)
    assert worker_a_version == worker_b_version == 0
    potential_kwargs = {
        "state_feat": torch.zeros(32),
        "estimated_cmax": 100.0,
        "current_time": 0.0,
        "h0": 10.0,
        "last_transfer_time": 0.0,
        "episode_id": 7,
    }
    phi_before = shaper.compute_potential(**potential_kwargs, worker_id=0)

    updated_head = TimeResidualHead(in_dim=32, hidden_dim=16, use_layer_norm=False)
    for parameter in updated_head.parameters():
        torch.nn.init.zeros_(parameter)
    updated_head.reg_fc[-1].bias.data.fill_(-0.5)
    new_version = shaper.update_snapshot(updated_head)

    assert new_version == 1
    assert shaper.compute_potential(**potential_kwargs, worker_id=0) == phi_before
    assert shaper.compute_potential(**potential_kwargs, worker_id=1) == phi_before
    assert shaper.retained_snapshot_versions == (0, 1)
    assert shaper.end_episode(worker_id=0, episode_id=7) is True
    assert shaper.retained_snapshot_versions == (0, 1)
    assert shaper.end_episode(worker_id=1, episode_id=7) is True
    assert shaper.retained_snapshot_versions == (1,)

    new_episode_version = shaper.begin_episode(worker_id=0, episode_id=8)
    assert new_episode_version == 1
    assert shaper.compute_potential(
        **{**potential_kwargs, "episode_id": 8}, worker_id=0
    ) != phi_before
    assert shaper.end_episode(worker_id=0, episode_id=8) is True


def test_training_preserves_rollout_boundary_potential_and_refreshes_episodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """末步沿用旧版本结算；下一条完整轨迹才绑定刷新后的版本。"""
    import importlib
    from types import SimpleNamespace

    from envs.work3.decision_snapshot import DecisionSnapshot
    from models.work3.graph_builder import GRAPH_FEATURE_VERSION
    from models.work3.potential_shaping import PotentialRewardShaper
    from torch_geometric.data import HeteroData
    from training.work3_vector_env import Work3StepResult, Work3VectorStepBatch

    training = importlib.import_module("scripts.work3.train_ppo_work3")
    events: list[tuple[Any, ...]] = []
    last_potential_context: tuple[int, int] | None = None

    class CompletedEpisodeEnv:
        def __init__(self) -> None:
            self.episode_ids: tuple[int, ...] = ()
            self.episode_indices: tuple[int, ...] = ()
            self.worker_step_counts = [0]
            self.episode_step_counts = [0]
            self.step_settlement_requests = 0

        @property
        def worker_pids(self) -> tuple[int]:
            return (0,)

        @property
        def worker_cuda_initialized(self) -> tuple[bool]:
            return (False,)

        @property
        def worker_torch_num_threads(self) -> tuple[int]:
            return (1,)

        def reset_all(
            self,
            *,
            scenarios: list[dict[str, Any] | None],
            episode_ids: list[int],
            episode_indices: list[int],
        ) -> tuple[SimpleNamespace, ...]:
            self.episode_ids = tuple(episode_ids)
            self.episode_indices = tuple(episode_indices)
            self.episode_step_counts = [0 for _ in episode_ids]
            return tuple(
                SimpleNamespace(scenario_status={}) for _ in episode_ids
            )

        def snapshots(self) -> tuple[DecisionSnapshot, ...]:
            return tuple(
                DecisionSnapshot(
                    worker_id=worker_id,
                    episode_id=episode_id,
                    episode_index=self.episode_indices[worker_id],
                    state_features=torch.zeros(32),
                    time_features=torch.zeros(2),
                    graph_snapshot=HeteroData(),
                    candidate_task_keys=(),
                    candidate_task_features=torch.zeros((0, 8)),
                    branch_masks=(),
                    team_contexts=(),
                    candidate_task_node_indices=(),
                    worker_node_indices=(),
                    reserved_flags=(),
                    advance_available=True,
                    current_time=float(self.episode_step_counts[worker_id]),
                    cycle_id=0,
                    estimated_cmax=20.0,
                    h0=10.0,
                    last_transfer_time=0.0,
                    graph_version=GRAPH_FEATURE_VERSION,
                )
                for worker_id, episode_id in enumerate(self.episode_ids)
            )

        def step_all(
            self,
            *,
            actions: list[dict[str, Any] | None],
            max_total_steps: int | None,
            settle_timeout_seconds: float | None,
            wall_clock_deadline: float | None,
        ) -> Work3VectorStepBatch:
            dispatched = tuple(
                worker_id
                for worker_id, action in enumerate(actions)
                if action is not None
            )
            results: list[Work3StepResult | None] = [None] * len(actions)
            for worker_id in dispatched:
                self.worker_step_counts[worker_id] += 1
                self.episode_step_counts[worker_id] += 1
                episode_finished = self.episode_step_counts[worker_id] == 2
                results[worker_id] = Work3StepResult(
                    observation={},
                    raw_reward=0.0,
                    terminated=episode_finished,
                    truncated=False,
                    info={
                        "success": episode_finished,
                        "termination_reason": "completed" if episode_finished else None,
                    },
                    processed_events=(),
                    completed_execution_records=(),
                    step_count=1,
                    scenario_status={},
                )
            return Work3VectorStepBatch(
                results=tuple(results),
                dispatched_worker_ids=dispatched,
                invoked_worker_ids=dispatched,
                interrupted_worker_ids=(),
                worker_errors=(),
                total_env_steps=len(dispatched),
                budget_reserved_steps=len(dispatched),
                wall_clock_expired=False,
            )

        def close(self) -> tuple[None, ...]:
            return (None,)

    class NoOpTrainer:
        def __init__(self, **kwargs: Any) -> None:
            self.precision_plugin = SimpleNamespace(scaler=None)

        def fit(self, *args: Any, **kwargs: Any) -> None:
            return None

    def select_advance(
        actor: Any,
        snapshot: DecisionSnapshot,
        *,
        deterministic: bool,
    ) -> tuple[dict[str, str], torch.Tensor, torch.Tensor, dict[str, Any]]:
        return (
            {"action": "advance"},
            torch.tensor(0.0),
            torch.tensor(0.0),
            {"graph_snapshot": snapshot.graph_snapshot},
        )

    original_begin = PotentialRewardShaper.begin_episode
    original_end = PotentialRewardShaper.end_episode
    original_update = PotentialRewardShaper.update_snapshot
    original_shape = PotentialRewardShaper.shape_reward

    def record_begin(
        shaper: PotentialRewardShaper,
        *,
        worker_id: int,
        episode_id: int,
    ) -> int:
        version = original_begin(shaper, worker_id=worker_id, episode_id=episode_id)
        events.append(("begin", episode_id, version))
        return version

    def record_end(
        shaper: PotentialRewardShaper,
        *,
        worker_id: int,
        episode_id: int,
    ) -> bool:
        version = shaper._episode_versions[(worker_id, episode_id)]
        events.append(("end", episode_id, version))
        return original_end(shaper, worker_id=worker_id, episode_id=episode_id)

    def record_update(
        shaper: PotentialRewardShaper,
        new_head: Any,
        new_actor: Any | None = None,
    ) -> int:
        version = original_update(shaper, new_head, new_actor)
        events.append(("update", version))
        return version

    def record_potential(
        shaper: PotentialRewardShaper,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        nonlocal last_potential_context
        episode_id = int(kwargs["episode_id"])
        worker_id = int(kwargs["worker_id"])
        version = shaper._episode_versions[(worker_id, episode_id)]
        last_potential_context = (episode_id, version)
        events.append(("potential", episode_id, version))
        current_time = float(kwargs["current_time"])
        return {0.0: -3.0, 1.0: -2.0}[current_time]

    def record_shape(
        shaper: PotentialRewardShaper,
        actual_reward: float,
        phi_current: float,
        phi_next: float,
    ) -> float:
        assert last_potential_context is not None
        shaped_reward = original_shape(shaper, actual_reward, phi_current, phi_next)
        events.append((
            "shape",
            *last_potential_context,
            float(phi_current),
            float(phi_next),
            float(shaped_reward),
        ))
        return shaped_reward

    scenarios = [
        {
            "scenario_id": f"tiny-{index}",
            "timing": "mid",
            "intensity": "light",
            "station_id": 1,
            "aircraft_id": 0,
            "affected_task_keys": [],
            "valid": True,
            "tau": 0.0,
            "recovery_time": 0.0,
        }
        for index in range(2)
    ]
    monkeypatch.setattr(training, "load_training_scenarios", lambda *args: scenarios)
    monkeypatch.setattr(
        training,
        "create_work3_single_env_runtime",
        lambda *args, **kwargs: CompletedEpisodeEnv(),
    )
    monkeypatch.setattr(training.ActorCriticWork3, "select_snapshot", select_advance)
    monkeypatch.setattr(
        training,
        "compute_online_snapshot_time_inputs",
        lambda *, actor_critic, time_head, snapshot: (
            snapshot.graph_snapshot,
            snapshot.time_features,
            0.0,
        ),
    )
    monkeypatch.setattr(
        training.ActorCriticWork3,
        "encode_state",
        lambda self, *args, **kwargs: (torch.tensor([0.0]), None),
    )
    monkeypatch.setattr(PotentialRewardShaper, "begin_episode", record_begin)
    monkeypatch.setattr(PotentialRewardShaper, "end_episode", record_end)
    monkeypatch.setattr(PotentialRewardShaper, "update_snapshot", record_update)
    monkeypatch.setattr(PotentialRewardShaper, "compute_potential", record_potential)
    monkeypatch.setattr(PotentialRewardShaper, "shape_reward", record_shape)
    monkeypatch.setattr(training.pl, "Trainer", NoOpTrainer)

    report = training.run_training(
        run_mode="pilot",
        successful_batch_target=2,
        max_decisions=4,
        max_wall_seconds=30.0,
        num_iterations=4,
        steps_per_iter=1,
        ppo_epochs=1,
        batch_size=1,
        num_envs=1,
        gamma=0.9,
        baseline_path=Path("data/work3/real_283_k10_baseline.json"),
        scenarios_path=Path("unused-scenarios.json"),
        scenario_split_path=Path("data/work3/experiment_splits/train.json"),
        time_head_ckpt=tmp_path / "missing_time_head.pt",
        output_ckpt=tmp_path / "pilot.pt",
        report_path=tmp_path / "pilot.json",
        device="cpu",
    )

    assert report["successful_batch_count"] == 2
    shaped_events = [event for event in events if event[0] == "shape"]
    assert shaped_events == [
        ("shape", 0, 0, -3.0, -2.0, 1.2),
        ("shape", 0, 0, -2.0, 0.0, 2.0),
        ("shape", 1, 1, -3.0, -2.0, 1.2),
        ("shape", 1, 1, -2.0, 0.0, 2.0),
    ]
    segment_rewards = [event[-1] for event in shaped_events[:2]]
    assert segment_rewards[0] + 0.9 * segment_rewards[1] == pytest.approx(3.0)
    assert [event for event in events if event[0] == "begin"] == [
        ("begin", 0, 0),
        ("begin", 1, 1),
    ]
    first_tail = events.index(("shape", 0, 0, -2.0, 0.0, 2.0))
    first_end = events.index(("end", 0, 0))
    refresh = events.index(("update", 1))
    second_begin = events.index(("begin", 1, 1))
    assert first_tail < first_end < refresh < second_begin
    assert ("potential", 0, 0) in events
    assert ("potential", 1, 1) in events


def test_online_predictor_update_preserves_all_active_snapshot_state() -> None:
    """在线头更新不得改写活动episode的图编码器、时间头或归一化状态。"""
    import copy

    from models.work3.actor_critic import ActorCriticWork3
    from models.work3.potential_shaping import PotentialRewardShaper
    from models.work3.time_head import TimeResidualHead

    actor = ActorCriticWork3(hidden_dim=16)
    head = TimeResidualHead(in_dim=16, hidden_dim=16)
    shaper = PotentialRewardShaper(time_head=head, actor_critic=actor)
    assert shaper.begin_episode(worker_id=0, episode_id=11) == 0

    active_actor = shaper.frozen_actor
    active_head = shaper.frozen_head
    assert active_actor is not None
    actor_state = {
        name: value.detach().clone()
        for name, value in active_actor.state_dict().items()
    }
    head_state = {
        name: value.detach().clone()
        for name, value in active_head.state_dict().items()
    }
    updated_actor = copy.deepcopy(actor)
    updated_head = copy.deepcopy(head)
    with torch.no_grad():
        next(updated_actor.parameters()).add_(0.01)
        next(updated_head.parameters()).add_(0.01)
    assert any(
        not torch.equal(actor_state[name], value)
        for name, value in updated_actor.state_dict().items()
    )
    assert any(
        not torch.equal(head_state[name], value)
        for name, value in updated_head.state_dict().items()
    )

    assert shaper.update_snapshot(updated_head, new_actor=updated_actor) == 1
    assert shaper.retained_snapshot_versions == (0, 1)
    assert all(
        torch.equal(actor_state[name], value)
        for name, value in active_actor.state_dict().items()
    )
    assert all(
        torch.equal(head_state[name], value)
        for name, value in active_head.state_dict().items()
    )


def test_rollout_buffer_stores_immutable_cpu_feature_mask_and_graph_snapshots() -> None:
    from torch_geometric.data import HeteroData

    from models.work3.ppo_buffer import PPOTransition, RolloutBufferWork3

    state_feat = torch.arange(32, dtype=torch.float32)
    time_urgency = torch.tensor([1.0, 2.0])
    worker_mask = torch.tensor([True, False])
    graph = HeteroData()
    graph["task"].x = torch.ones((1, 3))
    record = {
        "worker_valid_mask": worker_mask,
        "graph_snapshot": graph,
        "target_residual": torch.tensor(-0.25),
    }
    buffer = RolloutBufferWork3(normalize_advantages=False)
    buffer.add(
        PPOTransition(
            state_feat=state_feat,
            time_urgency=time_urgency,
            sample_record=record,
            reward=0.0,
            raw_reward=0.0,
            value=0.0,
            log_prob=0.0,
        )
    )

    state_feat.zero_()
    time_urgency.zero_()
    worker_mask.fill_(False)
    graph["task"].x.zero_()
    record["target_residual"].fill_(9.0)

    stored = buffer.transitions[0]
    assert stored.state_feat.device.type == "cpu"
    assert torch.equal(stored.state_feat, torch.arange(32, dtype=torch.float32))
    assert torch.equal(stored.time_urgency, torch.tensor([1.0, 2.0]))
    assert torch.equal(stored.sample_record["worker_valid_mask"], torch.tensor([True, False]))
    assert stored.sample_record["graph_snapshot"] is not graph
    assert torch.equal(stored.sample_record["graph_snapshot"]["task"].x, torch.ones((1, 3)))
    assert stored.sample_record["target_residual"].item() == pytest.approx(-0.25)


def _make_lightning_training_fixture() -> tuple[object, object, object]:
    from envs.work3.environment import AirLineEnvWork3
    from models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features
    from models.work3.action_fusion import compute_time_urgency_vector
    from models.work3.ppo_buffer import PPOTransition, RolloutBufferWork3
    from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
    from models.work3.time_head import TimeResidualHead
    lightning_runtime = _work3_lightning_module()

    environment = AirLineEnvWork3(
        baseline_json_path=ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
    )
    environment.reset()
    actor = ActorCriticWork3(hidden_dim=16)
    time_head = TimeResidualHead(in_dim=16, hidden_dim=16)
    cmax = compute_cycle_heuristic_cmax(environment.state)
    state_features = extract_compact_state_features(environment.state, cmax)
    time_features = compute_time_urgency_vector(
        estimated_r=max(0.0, cmax - environment.state.current_time),
        current_time=environment.state.current_time,
        last_transfer_time=environment.state.last_transfer_time,
        h0=environment.state.h0,
    )
    action, log_prob, value, sample_record = actor.select_action(
        environment,
        state_features,
        time_features,
        deterministic=True,
    )
    assert action is not None
    buffer = RolloutBufferWork3(normalize_advantages=False)
    buffer.add(
        PPOTransition(
            state_feat=state_features,
            time_urgency=time_features,
            sample_record=sample_record,
            reward=1.0,
            raw_reward=1.0,
            value=value,
            log_prob=log_prob,
            worker_id=0,
            episode_id=2,
            segment_id=0,
            done=True,
            terminated=True,
            action_dict=action,
        )
    )
    buffer.finish_trajectories(last_values_by_segment={})
    update = lightning_runtime.Work3TrainingUpdate(buffer=buffer, environment_steps=3)
    return actor, time_head, update


def test_lightning_trainer_fit_runs_ppo_update_and_separates_step_counts() -> None:
    import lightning.pytorch as pl

    lightning_runtime = _work3_lightning_module()

    actor, time_head, update = _make_lightning_training_fixture()
    parameters_before = [parameter.detach().clone() for parameter in actor.parameters()]
    module = lightning_runtime.Work3LightningModule(
        actor_critic=actor,
        time_head=time_head,
        ppo_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
    )
    data_module = lightning_runtime.Work3PPODataModule(update_factory=lambda: iter((update,)))
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=ROOT_DIR / "outputs" / "work3_lightning_test",
    )

    trainer.fit(module, datamodule=data_module)

    assert trainer.global_step == 1
    assert module.optimization_steps == 1
    assert module.environment_steps == 3
    assert module.last_metrics["sampling_replay_max_abs_error"] <= 1e-6
    assert module.last_metrics["sampling_replay_sample_count"] == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, actor.parameters(), strict=True)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Lightning AMP验收需要CUDA设备")
@pytest.mark.parametrize(
    ("amp_dtype", "precision"),
    [("fp16", "16-mixed"), ("bf16", "bf16-mixed")],
)
def test_lightning_amp_uses_only_precision_plugin_scaler(
    tmp_path: Path,
    amp_dtype: str,
    precision: str,
) -> None:
    if amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("当前CUDA设备不支持bfloat16")

    import lightning.pytorch as pl

    lightning_runtime = _work3_lightning_module()
    actor, time_head, update = _make_lightning_training_fixture()
    module = lightning_runtime.Work3LightningModule(
        actor_critic=actor,
        time_head=time_head,
        ppo_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=precision,
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(
        module,
        datamodule=lightning_runtime.Work3PPODataModule(
            update_factory=lambda: iter((update,))
        ),
    )

    scaler = getattr(trainer.precision_plugin, "scaler", None)
    assert (scaler is not None and scaler.is_enabled()) is (amp_dtype == "fp16")
    assert not hasattr(module, "scaler")
    assert module.optimizer_step_attempts == 1
    assert module.optimization_steps + module.amp_skipped_steps == 1
    assert module.last_metrics["optimizer_step_attempts"] == 1
    assert module.last_metrics["amp_skipped_steps"] == module.amp_skipped_steps
    grad_norm = module.last_metrics.get("grad_norm")
    if grad_norm is not None:
        assert torch.isfinite(torch.tensor(grad_norm))
    if amp_dtype == "bf16":
        assert module.optimization_steps == 1
        assert module.amp_skipped_steps == 0


def test_lightning_configures_one_deduplicated_optimizer_for_all_trainable_models() -> None:
    lightning_runtime = _work3_lightning_module()

    actor, time_head, _update = _make_lightning_training_fixture()
    module = lightning_runtime.Work3LightningModule(actor_critic=actor, time_head=time_head)
    optimizer = module.configure_optimizers()
    optimizer_parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    expected_parameter_ids = {
        id(parameter)
        for parameter in (*actor.parameters(), *time_head.parameters())
        if parameter.requires_grad
    }

    assert len(optimizer_parameter_ids) == len(set(optimizer_parameter_ids))
    assert set(optimizer_parameter_ids) == expected_parameter_ids
    assert len(module.optimizers_configured_parameter_ids) == len(expected_parameter_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="设备保留验收需要CUDA设备")
def test_lightning_module_construction_preserves_actor_cuda_device() -> None:
    from models.work3.actor_critic import ActorCriticWork3

    lightning_runtime = _work3_lightning_module()
    actor = ActorCriticWork3(hidden_dim=16).to("cuda")

    module = lightning_runtime.Work3LightningModule(actor_critic=actor)

    assert next(actor.parameters()).device.type == "cuda"
    assert next(module.actor_critic.parameters()).device.type == "cuda"


def test_lightning_ppo_loss_computation_is_pure_and_datamodule_uses_no_loader_workers() -> None:
    from models.work3.ppo_trainer import PPOTrainerWork3
    lightning_runtime = _work3_lightning_module()

    actor, time_head, update = _make_lightning_training_fixture()
    objective = PPOTrainerWork3(
        actor_critic=actor,
        time_head=time_head,
        device="cpu",
        create_optimizer=False,
    )
    minibatch = next(update.buffer.get_batches(batch_size=1, shuffle=False))
    parameters_before = [parameter.detach().clone() for parameter in actor.parameters()]

    losses = objective.compute_ppo_minibatch_loss(minibatch)

    assert set(losses) >= {"total_loss", "policy_loss", "value_loss", "entropy"}
    assert objective.optimizer is None
    assert torch.isfinite(losses["total_loss"])
    assert all(parameter.grad is None for parameter in actor.parameters())
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, actor.parameters(), strict=True)
    )
    data_module = lightning_runtime.Work3PPODataModule(update_factory=lambda: iter((update,)))
    data_module.setup("fit")
    assert data_module.train_dataloader().num_workers == 0


def test_lightning_time_auxiliary_batch_scopes_decision_ids_by_worker_episode() -> None:
    from models.work3.ppo_trainer import PPOTrainerWork3

    actor, time_head, update = _make_lightning_training_fixture()
    trainer = PPOTrainerWork3(
        actor_critic=actor,
        time_head=time_head,
        device="cpu",
        create_optimizer=False,
    )
    graph = update.buffer.transitions[0].sample_record["graph_snapshot"]
    batch = {
        "state_feats": torch.zeros((2, 32)),
        "graph_snapshots": [graph, graph],
        "target_residuals": torch.tensor([-0.2, 0.3]),
        "worker_ids": [0, 1],
        "episode_ids": [7, 7],
        "cycle_ids": [2, 2],
        "decision_ids": [11, 11],
    }

    loss = trainer.compute_time_auxiliary_loss(batch)

    assert torch.isfinite(loss)
    batch["worker_ids"] = [0, 0]
    with pytest.raises(ValueError, match="worker_id, episode_id, decision_id"):
        trainer.compute_time_auxiliary_loss(batch)


def test_lightning_fit_calls_preserve_the_single_optimizer_between_rollouts() -> None:
    import lightning.pytorch as pl

    lightning_runtime = _work3_lightning_module()
    actor, time_head, update = _make_lightning_training_fixture()
    module = lightning_runtime.Work3LightningModule(
        actor_critic=actor,
        time_head=time_head,
        ppo_epochs=1,
        batch_size=1,
        learning_rate=1e-3,
    )
    optimizer = module.configure_optimizers()
    parameter = next(actor.parameters())
    for environment_steps in (3, 5):
        single_update = lightning_runtime.Work3TrainingUpdate(
            buffer=update.buffer,
            environment_steps=environment_steps,
        )
        data_module = lightning_runtime.Work3PPODataModule(
            update_factory=lambda update=single_update: iter((update,))
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=1,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            default_root_dir=ROOT_DIR / "outputs" / "work3_lightning_sequential_test",
        )
        trainer.fit(module, datamodule=data_module)
        assert module.configure_optimizers() is optimizer

    assert module.optimization_steps == 2
    assert module.environment_steps == 8
    assert int(optimizer.state[parameter]["step"]) == 2


def test_lightning_counts_only_successful_real_time_label_updates(tmp_path: Path) -> None:
    import lightning.pytorch as pl

    lightning_runtime = _work3_lightning_module()
    actor, time_head, update = _make_lightning_training_fixture()
    graph = update.buffer.transitions[0].sample_record["graph_snapshot"]
    labeled_update = lightning_runtime.Work3TrainingUpdate(
        buffer=update.buffer,
        environment_steps=update.environment_steps,
        time_auxiliary_batch={
            "state_feats": torch.zeros((1, 32)),
            "graph_snapshots": [graph],
            "target_residuals": torch.tensor([-0.2]),
            "worker_ids": [0],
            "episode_ids": [2],
            "cycle_ids": [1],
            "decision_ids": [0],
        },
    )
    module = lightning_runtime.Work3LightningModule(
        actor_critic=actor,
        time_head=time_head,
        time_loss_coef=1.0,
        ppo_epochs=1,
        time_auxiliary_epochs=1,
        time_auxiliary_batch_size=1,
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(
        module,
        datamodule=lightning_runtime.Work3PPODataModule(
            update_factory=lambda: iter((labeled_update,))
        ),
    )

    assert module.last_metrics["time_supervision_steps"] == 1
    assert module.last_metrics["time_supervision_optimizer_updates"] == 1
    assert module.time_supervision_optimizer_updates == 1
    assert module.optimization_steps == 2
    optimizer = module.configure_optimizers()
    graph_parameter = next(actor.graph_encoder.parameters())
    time_parameter = next(time_head.parameters())
    optimizer_parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert optimizer_parameter_ids.count(id(graph_parameter)) == 1
    assert optimizer_parameter_ids.count(id(time_parameter)) == 1
    assert optimizer.state[graph_parameter]["step"].item() == 2
    assert optimizer.state[time_parameter]["step"].item() == 1


def test_work3_pilot_c_d_pair_reports_fixed_hit_and_runtime_gate_truthfully(
    tmp_path: Path,
) -> None:
    """有限C/D配对保留同一命中事件，并如实标记截断及缺失时间标签。"""
    from scripts.work3.train_ppo_work3 import run_training

    scenario = {
        "scenario_id": "TASK8_FIXED_TAU_ZERO_HIT",
        "timing": "EARLY",
        "intensity": "LOW",
        "station_id": 0,
        "aircraft_id": 0,
        "tau": 0.0,
        "delta": 1.0,
        "recovery_time": 1.0,
        "affected_task_keys": ["0_15"],
        "valid": True,
    }
    scenarios_path = tmp_path / "task8_scenarios.json"
    split_path = tmp_path / "task8_train.json"
    scenarios_path.write_text(json.dumps([scenario]), encoding="utf-8")
    split_path.write_text(json.dumps([scenario]), encoding="utf-8")

    common = {
        "run_mode": "pilot",
        "successful_batch_target": 1,
        "max_decisions": 1,
        "num_iterations": 1,
        "steps_per_iter": 1,
        "ppo_epochs": 1,
        "batch_size": 1,
        "seed": 42,
        "baseline_path": "data/work3/real_283_k10_baseline.json",
        "scenarios_path": scenarios_path,
        "scenario_split_path": split_path,
        "device": "cpu",
        "num_envs": 1,
    }
    c_report_path = tmp_path / "task8_c.json"
    c_report = run_training(
        **common,
        method_variant="C",
        output_ckpt=tmp_path / "task8_c.pt",
        report_path=c_report_path,
    )
    d_report = run_training(
        **common,
        method_variant="D",
        output_ckpt=tmp_path / "task8_d.pt",
        report_path=tmp_path / "task8_d.json",
        paired_report_path=c_report_path,
    )

    assert c_report["initial_actor_fingerprint"] == d_report["initial_actor_fingerprint"]
    assert d_report["paired_run_check"]["same_event_plan"] is True
    assert d_report["paired_run_check"]["same_interaction_budget"] is True
    assert d_report["paired_run_check"]["same_actual_hit_pattern"] is True
    event_fields = {
        "scenario_id": "TASK8_FIXED_TAU_ZERO_HIT",
        "scheduled_tau": 0.0,
        "scheduled_recovery_time": 1.0,
        "scheduled_affected_task_keys": ["0_15"],
        "scheduled_target_count": 1,
    }
    c_event = c_report["scenario_log"][0]
    d_event = d_report["scenario_log"][0]
    for field, expected in event_fields.items():
        assert c_event[field] == expected
        assert d_event[field] == expected
        assert c_event[field] == d_event[field]
    for report in (c_report, d_report):
        assert report["research_result_eligible"] is False
        assert report["total_decisions"] == 1
        assert report["scenario_log"][0]["disturbance_triggered"] is True
        assert report["scenario_log"][0]["actual_hit_task_keys"] == ["0_15"]
        assert report["scenario_log"][0]["actual_hit_count"] == 1
        assert report["scenario_log"][0]["material_ready_advanced_task_keys"] == ["0_15"]
        assert report["scenario_log"][0]["material_ready_advanced_count"] == 1
        assert report["scenario_log"][0]["success"] is False
        assert report["scenario_log"][0]["truncated"] is True
        expected_snapshot_version = 0 if report["method_variant"] == "D" else None
        assert report["scenario_log"][0]["potential_snapshot_version"] == expected_snapshot_version
        assert report["training_config"]["main_num_threads"] == 1
        assert report["training_config"]["env_num_threads"] == 1
        assert report["training_config"]["num_envs"] == 1
        assert report["training_config"]["rollout_steps"] == 1
        assert report["training_config"]["batch_size"] == 1
        assert report["amp_dtype"] == "fp32"
        assert report["device"] == "cpu"
        assert report["memory_peak_kind"]
        assert report["gpu_memory_peak_bytes"] is None
        assert report["host_memory_peak_bytes"] is None or report["host_memory_peak_bytes"] > 0
        assert report["memory_peak_bytes"] == report["host_memory_peak_bytes"]
        assert report["elapsed_seconds"] > 0.0
        assert report["worker_step_settlement_requests"] == sum(report["worker_step_counts"])
        audit = report["trajectory_audit"]
        assert audit["cost_breakdown"] == {
            "cost_takt": 0.0,
            "cost_time": 0.0,
            "cost_team": 0.0,
            "cost_postpone": 0.0,
            "cost_revision": 0.0,
        }
        assert audit["success"] is False
        assert audit["termination_reason"] == "incomplete"
        feasibility = report["independent_feasibility"][0]
        assert feasibility["status"] == "incomplete_not_assessed"
        assert feasibility["feasible"] is None

    assert d_report["cycle_time_labels"]
    assert all(label["label_available"] is False for label in d_report["cycle_time_labels"])
    assert d_report["time_head_training_status"] == "untrained_no_successful_online_update"
    assert d_report["time_supervision_optimizer_updates"] == 0
    assert d_report["checkpoint_evaluation_eligible"] is False


def test_work3_d_pilot_records_real_transfer_labels_and_nonterminal_potential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实周期转站补齐两种符号残差，并触发D的辅助优化更新。"""
    from models.work3.ppo_buffer import PendingTimeLabelCache
    from models.work3.potential_shaping import PotentialRewardShaper
    from scripts.work3.train_ppo_work3 import run_training

    scenario = {
        "scenario_id": "TASK8_SIGNED_RESIDUAL_TRANSFER",
        "timing": "EARLY",
        "intensity": "LOW",
        "station_id": 0,
        "aircraft_id": 0,
        "tau": 0.0,
        "delta": 1.0,
        "recovery_time": 1.0,
        "affected_task_keys": ["0_15"],
        "valid": True,
    }
    scenarios_path = tmp_path / "signed_scenarios.json"
    split_path = tmp_path / "signed_train.json"
    scenarios_path.write_text(json.dumps([scenario]), encoding="utf-8")
    split_path.write_text(json.dumps([scenario]), encoding="utf-8")

    observed_label_batches: list[dict[str, torch.Tensor]] = []
    observed_potentials: list[tuple[float, bool, float]] = []
    original_drain_ready = PendingTimeLabelCache.drain_ready
    original_compute_potential = PotentialRewardShaper.compute_potential

    def capture_residuals(cache: PendingTimeLabelCache) -> dict[str, Any] | None:
        batch = original_drain_ready(cache)
        if batch is not None:
            observed_label_batches.append({
                name: batch[name].clone()
                for name in (
                    "estimated_cmax",
                    "actual_transfer_times",
                    "h0",
                    "target_residuals",
                )
            })
        return batch

    def capture_potential(
        shaper: PotentialRewardShaper,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        potential = original_compute_potential(shaper, *args, **kwargs)
        observed_potentials.append((
            float(kwargs["last_transfer_time"]),
            bool(kwargs.get("is_terminal", False)),
            potential,
        ))
        return potential

    monkeypatch.setattr(PendingTimeLabelCache, "drain_ready", capture_residuals)
    monkeypatch.setattr(PotentialRewardShaper, "compute_potential", capture_potential)
    report = run_training(
        run_mode="pilot",
        successful_batch_target=1,
        max_decisions=64,
        max_wall_seconds=300.0,
        num_iterations=1,
        steps_per_iter=64,
        ppo_epochs=1,
        batch_size=64,
        seed=42,
        method_variant="D",
        baseline_path="data/work3/real_283_k10_baseline.json",
        scenarios_path=scenarios_path,
        scenario_split_path=split_path,
        device="cpu",
        num_envs=1,
        output_ckpt=tmp_path / "signed_d.pt",
    )

    assert report["actual_disturbance_hit_count"] > 0
    assert report["time_label_count"] > 0
    assert observed_label_batches
    for batch in observed_label_batches:
        expected_residuals = (
            batch["actual_transfer_times"] - batch["estimated_cmax"]
        ) / batch["h0"]
        assert torch.isfinite(batch["target_residuals"]).all()
        assert torch.allclose(
            batch["target_residuals"],
            expected_residuals,
            atol=1e-6,
            rtol=1e-6,
        )
    assert report["time_supervision_optimizer_updates"] > 0
    assert report["lightning_optimization_steps"] > 0
    assert report["time_head_training_status"] == "trained_online"
    available_labels = [
        item for item in report["cycle_time_labels"] if item["label_available"]
    ]
    unavailable_labels = [
        item for item in report["cycle_time_labels"] if not item["label_available"]
    ]
    assert available_labels
    assert all(
        item["label_count"] > 0 and item["actual_transfer_time"] is not None
        for item in available_labels
    )
    assert all(
        item["label_count"] == 0 and item["actual_transfer_time"] is None
        for item in unavailable_labels
    )
    assert {item["potential_snapshot_version"] for item in report["scenario_log"]} == {0}
    assert any(
        last_transfer_time > 0.0 and not is_terminal and abs(potential) > 1e-6
        for last_transfer_time, is_terminal, potential in observed_potentials
    )
    assert report["research_result_eligible"] is False


def test_training_audit_feasibility_uses_independent_execution_records() -> None:
    from scripts.work3.train_ppo_work3 import _independent_feasibility_from_audit

    audit = {
        "success": True,
        "completed_tasks": 1,
        "total_tasks": 1,
        "execution_records": [{
            "aircraft_id": 0,
            "task_id": 0,
            "station_id": 0,
            "team": [1],
            "start": 1.0,
            "end": 2.0,
            "material_ready_time": 0.0,
            "station_entry_time": 0.0,
            "aircraft_station_at_start": 0,
        }],
        "task_constraints": {
            0: {
                "demand": 1,
                "required_skill": 0,
                "predecessors": (),
                "fixed_station": 0,
                "max_allowed_station": 0,
            },
        },
        "worker_skills": {1: (0,)},
        "worker_station_bindings": {1: 0},
        "station_capacities": {0: 1},
    }

    valid = _independent_feasibility_from_audit(audit)[0]
    invalid = _independent_feasibility_from_audit({
        **audit,
        "worker_skills": {1: ()},
    })[0]
    incomplete = _independent_feasibility_from_audit({
        **audit,
        "success": False,
        "completed_tasks": 0,
        "execution_records": [],
    })[0]

    assert valid["status"] == "feasible" and valid["feasible"] is True
    assert invalid["status"] == "violations" and invalid["feasible"] is False
    assert invalid["violations"]["skill"] == 1
    assert incomplete["status"] == "incomplete_not_assessed"
    assert incomplete["feasible"] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="工作三CUDA试点验收需要GPU")
def test_work3_cuda_bf16_two_worker_pilot_records_real_hits_and_resource_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """从YAML覆写入口验证CUDA BF16、spawn环境、真实命中和预算记账。"""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("当前CUDA设备不支持bfloat16")

    from scripts.work3 import train_ppo_work3

    scenario = {
        "scenario_id": "TASK8_CUDA_TWO_WORKER_HIT",
        "timing": "EARLY",
        "intensity": "LOW",
        "station_id": 0,
        "aircraft_id": 0,
        "tau": 0.0,
        "delta": 1.0,
        "recovery_time": 1.0,
        "affected_task_keys": ["0_15"],
        "valid": True,
    }
    scenarios_path = tmp_path / "cuda_scenarios.json"
    split_path = tmp_path / "cuda_train.json"
    scenarios_path.write_text(json.dumps([scenario]), encoding="utf-8")
    split_path.write_text(json.dumps([scenario]), encoding="utf-8")

    checkpoint_path = tmp_path / "cuda_d_bf16.pt"
    report_path = tmp_path / "cuda_d_bf16.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ppo_work3.py",
            "--mode", "pilot",
            "--seed", "20260925",
            "--steps", "2",
            "--num-envs", "2",
            "--epochs", "1",
            "--batch-size", "2",
            "--successful-batch-target", "1",
            "--max-decisions", "2",
            "--method", "D",
            "--baseline", str(ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"),
            "--scenarios", str(scenarios_path),
            "--scenario-split", str(split_path),
            "--device", "cuda",
            "--output", str(checkpoint_path),
            "--report", str(report_path),
            "--set", "runtime.amp_dtype=bf16",
            "--set", "runtime.max_wall_seconds=120.0",
        ],
    )
    train_ppo_work3.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["amp_dtype"] == "bf16"
    assert report["lightning_precision"] == "bf16-mixed"
    assert report["lightning_grad_scaler_enabled"] is False
    assert report["device"] == "cuda"
    assert report["memory_peak_kind"] == "cuda_max_memory_allocated"
    assert report["memory_peak_bytes"] > 0
    assert report["gpu_memory_peak_bytes"] == report["memory_peak_bytes"]
    assert report["host_memory_peak_bytes"] is None or report["host_memory_peak_bytes"] > 0
    assert report["training_config"]["num_envs"] == 2
    assert report["environment_worker_cuda_initialized"] == [False, False]
    assert report["total_decisions"] == 2
    assert report["worker_step_counts"] == [1, 1]
    assert report["worker_step_settlement_requests"] == 2
    assert report["actual_disturbance_hit_count"] == 2
    assert all(
        entry["disturbance_triggered"] is True
        and entry["actual_hit_task_keys"] == ["0_15"]
        and entry["truncated"] is True
        and entry["success"] is False
        for entry in report["scenario_log"]
    )
    assert report["time_head_training_status"] == "untrained_no_successful_online_update"
    assert report["checkpoint_evaluation_eligible"] is False
    assert report["research_result_eligible"] is False
    assert report["history"][0]["lightning_optimization_steps"] == 1
    assert all(
        item["status"] == "incomplete_not_assessed" and item["feasible"] is None
        for item in report["independent_feasibility"]
    )


def test_pilot_success_target_waits_for_started_episodes_to_finish() -> None:
    """达成成功批次目标后，只要仍有已启动轨迹就不得结束试点。"""
    from scripts.work3.train_ppo_work3 import _should_stop_after_success_target

    assert not _should_stop_after_success_target(
        run_mode="pilot",
        successful_batch_target=1,
        successful_batch_count=1,
        active_episode_count=1,
    )
    assert _should_stop_after_success_target(
        run_mode="pilot",
        successful_batch_target=1,
        successful_batch_count=1,
        active_episode_count=0,
    )
    assert not _should_stop_after_success_target(
        run_mode="pilot",
        successful_batch_target=1,
        successful_batch_count=0,
        active_episode_count=0,
    )
