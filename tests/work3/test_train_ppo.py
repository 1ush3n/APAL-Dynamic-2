"""Task 7.3 PPO 验证训练流程单元测试。

验证点：
1. run_training() 端到端可执行性，无崩溃、无内存泄漏、无死锁；
2. 步步采集、势函数塑形、GAE 优势计算与 PPO 反向传播数值全量有限 (无 NaN/Inf)；
3. 输出检查点文件正确创建并可成功被 ActorCriticWork3 载入。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import tempfile
from typing import Any
import pytest
import torch

from envs.work3.core_types import ActionBranch
from models.work3.actor_critic import ActorCriticWork3
import scripts.work3.train_ppo_work3 as train_module
from scripts.work3.train_ppo_work3 import run_training


@pytest.mark.parametrize("environment_truncated", [False, True])
def test_d_time_prediction_runs_for_decisions_and_required_bootstrap_only(
    environment_truncated: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """正常转移不预先计算下一状态时间特征；真实截断仍计算bootstrap特征。"""
    from dataclasses import replace

    from training.work3_vector_env import Work3VectorEnv

    prediction_times: list[float] = []
    original_prediction = train_module.compute_online_snapshot_time_inputs
    original_step_all = Work3VectorEnv.step_all

    def record_prediction(
        actor_critic: ActorCriticWork3,
        time_head: train_module.TimeResidualHead,
        snapshot: train_module.DecisionSnapshot,
    ) -> tuple[Any, torch.Tensor, torch.Tensor]:
        prediction_times.append(snapshot.current_time)
        return original_prediction(actor_critic, time_head, snapshot)

    monkeypatch.setattr(
        train_module,
        "compute_online_snapshot_time_inputs",
        record_prediction,
    )

    if environment_truncated:
        def truncate_worker_result(
            environment: Work3VectorEnv,
            **kwargs: Any,
        ) -> Any:
            batch = original_step_all(environment, **kwargs)
            results = list(batch.results)
            assert results[0] is not None
            results[0] = replace(results[0], truncated=True)
            return replace(batch, results=tuple(results))

        monkeypatch.setattr(Work3VectorEnv, "step_all", truncate_worker_result)

    result = run_training(
        run_mode="pilot",
        successful_batch_target=1,
        max_decisions=1,
        max_wall_seconds=600.0,
        steps_per_iter=1,
        ppo_epochs=1,
        batch_size=1,
        seed=43,
        method_variant="D",
        num_envs=1,
        time_head_ckpt=str(tmp_path / "missing_time_head.pt"),
        output_ckpt=str(tmp_path / "time_prediction_calls.pt"),
        device="cpu",
    )

    assert result["total_decisions"] == 1
    assert len(prediction_times) == 2


def test_ppo_training_pipeline_sanity() -> None:
    """运行极速 2 轮 PPO 验证训练，测试全流程端到端稳定性。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_ckpt = Path(tmpdir) / "test_method_d.pt"

        results = run_training(
            run_mode="smoke",
            num_iterations=1,
            steps_per_iter=32,
            ppo_epochs=1,
            batch_size=32,
            lr=1e-3,
            seed=42,
            output_ckpt=str(out_ckpt),
            device="cpu",
        )

        assert "history" in results
        assert len(results["history"]) == 1
        assert len(results["environment_worker_pids"]) == 1
        assert results["environment_worker_pids"][0] != os.getpid()
        assert results["environment_worker_cuda_initialized"] == [False]

        for log in results["history"]:
            assert not math.isnan(log["total_loss"])
            assert not math.isnan(log["policy_loss"])
            assert not math.isnan(log["value_loss"])
            assert not math.isnan(log["entropy"])
            assert not math.isnan(log["grad_norm"])
            assert log["ppo_updates"] > 0
            assert log["lightning_optimization_steps"] > 0
            assert log["environment_steps"] == 32
            assert log["sampling_replay_max_abs_error"] <= 1e-6
            assert log["sampling_replay_sample_count"] == 32

        # 检查点有效性检验
        assert out_ckpt.is_file()
        ckpt_data = torch.load(out_ckpt, map_location="cpu")
        assert "actor_critic_state" in ckpt_data

        test_net = ActorCriticWork3()
        test_net.load_state_dict(ckpt_data["actor_critic_state"])


def test_training_records_forced_advance_from_spawn_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """固定决策通过spawn worker执行并进入Lightning rollout。"""
    scenario = {
        "scenario_id": "SPAWN_ADVANCE_TEST",
        "timing": "middle",
        "intensity": "light",
        "station_id": 0,
        "aircraft_id": 0,
        "tau": 0.0,
        "recovery_time": 0.0,
        "affected_task_keys": [],
    }
    monkeypatch.setattr(train_module, "load_training_scenarios", lambda *_args: [scenario])

    def always_advance(
        self: ActorCriticWork3,
        snapshot: object,
        deterministic: bool = False,
    ) -> tuple[dict[str, ActionBranch], float, float, dict[str, int | str]]:
        del self, snapshot, deterministic
        return (
            {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT},
            0.0,
            0.0,
            {"action_type": "forced_advance"},
        )

    monkeypatch.setattr(ActorCriticWork3, "select_snapshot", always_advance)
    result = run_training(
        run_mode="pilot",
        successful_batch_target=1,
        max_decisions=1,
        num_iterations=1,
        steps_per_iter=1,
        ppo_epochs=0,
        batch_size=1,
        method_variant="C",
        output_ckpt=str(tmp_path / "spawn_advance.pt"),
    )

    assert result["history"][0]["total_steps"] == 1
    assert result["history"][0]["environment_steps"] == 1
    assert result["scenario_log"][0]["disturbance_triggered"] is True
    assert result["trajectory_audit"]["step_count"] == 1
    assert result["lightning_fit_calls"] == 1


def test_resolved_yaml_and_fingerprint_are_saved_in_report_and_checkpoint(
    tmp_path: Path,
) -> None:
    import hashlib

    resolved_yaml = "runtime:\n  seed: 31\n  method_profile: C\n"
    fingerprint = hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()
    result = run_training(
        run_mode="smoke",
        num_iterations=1,
        steps_per_iter=1,
        max_decisions=1,
        ppo_epochs=1,
        batch_size=1,
        method_variant="C",
        output_ckpt=tmp_path / "resolved_config.pt",
        resolved_config_yaml=resolved_yaml,
        resolved_config_sha256=fingerprint,
    )

    assert result["resolved_runtime_config_yaml"] == resolved_yaml
    assert result["resolved_runtime_config_sha256"] == fingerprint
    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=False)
    assert checkpoint["run_metadata"]["resolved_runtime_config_yaml"] == resolved_yaml
    assert checkpoint["run_metadata"]["resolved_runtime_config_sha256"] == fingerprint


def test_training_entry_uses_two_spawn_workers_with_aggregate_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """双环境训练入口按总交互步预算采样并在同一Lightning模块更新。"""
    labels: list[tuple[int, int, int]] = []
    original_add = train_module.PendingTimeLabelCache.add

    def record_label(
        cache: train_module.PendingTimeLabelCache,
        **kwargs: Any,
    ) -> bool:
        labels.append((
            int(kwargs["worker_id"]),
            int(kwargs["episode_id"]),
            int(kwargs["decision_id"]),
        ))
        return original_add(cache, **kwargs)

    monkeypatch.setattr(
        train_module.PendingTimeLabelCache,
        "add",
        record_label,
    )
    result = run_training(
        run_mode="smoke",
        num_iterations=1,
        steps_per_iter=3,
        max_decisions=3,
        ppo_epochs=1,
        batch_size=3,
        seed=17,
        method_variant="D",
        num_envs=2,
        time_head_ckpt=str(tmp_path / "missing_time_head.pt"),
        output_ckpt=str(tmp_path / "two_workers.pt"),
    )

    assert result["training_config"]["num_envs"] == 2
    assert len(result["environment_worker_pids"]) == 2
    assert all(pid != os.getpid() for pid in result["environment_worker_pids"])
    assert result["environment_worker_cuda_initialized"] == [False, False]
    assert result["worker_step_counts"] == [2, 1]
    assert sum(result["worker_step_counts"]) == 3
    assert result["worker_step_settlement_requests"] == 3
    assert result["history"][0]["environment_steps"] == 3
    assert result["history"][0]["total_steps"] == 3
    assert result["history"][0]["lightning_optimization_steps"] > 0
    assert result["history"][0]["sampling_replay_sample_count"] == 3
    assert len(result["scenario_log"]) == 2
    assert {item["worker_id"] for item in result["scenario_log"]} == {0, 1}
    assert {item["potential_snapshot_version"] for item in result["scenario_log"]} == {0}
    assert {
        (item["worker_id"], item["episode_id"])
        for item in result["scenario_log"]
    } == {(item["worker_id"], item["episode_id"]) for item in result["worker_event_plan"]}
    assert {worker_id for worker_id, _episode_id, _decision_id in labels} == {0, 1}
    assert len(labels) == len(set(labels)) == 3


def test_training_entry_closes_other_workers_after_worker_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """一个worker报告错误时，训练入口也必须回收同批其他worker。"""
    from dataclasses import replace

    from training.work3_vector_env import Work3VectorEnv

    created_envs: list[Work3VectorEnv] = []
    original_create = train_module.create_work3_single_env_runtime
    original_step_all = Work3VectorEnv.step_all

    def capture_environment(*args: Any, **kwargs: Any) -> Work3VectorEnv:
        environment = original_create(*args, **kwargs)
        created_envs.append(environment)
        return environment

    def report_worker_error(
        environment: Work3VectorEnv,
        **kwargs: Any,
    ) -> Any:
        batch = original_step_all(environment, **kwargs)
        return replace(
            batch,
            worker_errors=((0, "simulated worker failure"),),
            interrupted_worker_ids=(0,),
        )

    monkeypatch.setattr(
        train_module,
        "create_work3_single_env_runtime",
        capture_environment,
    )
    monkeypatch.setattr(Work3VectorEnv, "step_all", report_worker_error)

    with pytest.raises(RuntimeError, match="step未完整结算"):
        run_training(
            run_mode="smoke",
            num_iterations=1,
            steps_per_iter=1,
            max_decisions=1,
            ppo_epochs=1,
            batch_size=1,
            seed=23,
            method_variant="C",
            num_envs=2,
            output_ckpt=str(tmp_path / "worker_error.pt"),
        )

    assert len(created_envs) == 1
    assert created_envs[0].workers_alive == (False, False)


def test_training_entry_closes_workers_after_actor_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """主进程Actor抛出异常时，训练入口也必须回收已启动的环境worker。"""
    from training.work3_vector_env import Work3VectorEnv

    created_envs: list[Work3VectorEnv] = []
    original_create = train_module.create_work3_single_env_runtime

    def capture_environment(*args: Any, **kwargs: Any) -> Work3VectorEnv:
        environment = original_create(*args, **kwargs)
        created_envs.append(environment)
        return environment

    def fail_actor(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated main-process actor failure")

    monkeypatch.setattr(
        train_module,
        "create_work3_single_env_runtime",
        capture_environment,
    )
    monkeypatch.setattr(ActorCriticWork3, "select_snapshot", fail_actor)

    try:
        with pytest.raises(RuntimeError, match="simulated main-process actor failure"):
            run_training(
                run_mode="smoke",
                num_iterations=1,
                steps_per_iter=1,
                max_decisions=1,
                ppo_epochs=1,
                batch_size=1,
                seed=29,
                method_variant="C",
                num_envs=2,
                output_ckpt=str(tmp_path / "actor_error.pt"),
            )

        assert len(created_envs) == 1
        assert created_envs[0].workers_alive == (False, False)
    finally:
        for environment in created_envs:
            if any(environment.workers_alive):
                environment.close()
