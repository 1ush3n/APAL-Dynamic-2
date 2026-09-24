"""R08：训练预算、可复现运行清单及非精确续训检查点反例。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import scripts.work3.train_ppo_work3 as train_module
from models.work3.ppo_buffer import PendingTimeLabelCache
from scripts.work3.train_ppo_work3 import ROOT_DIR, _compare_training_reports, run_training


def test_seeded_smoke_runs_repeat_initialization_and_event_plan(tmp_path: Path) -> None:
    """相同种子的32步无扰动烟测应复现初始化与场景序列并保存可读报告。"""
    results = []
    for run_name in ("first", "second"):
        results.append(run_training(
            run_mode="smoke",
            num_iterations=1,
            steps_per_iter=32,
            ppo_epochs=1,
            batch_size=32,
            seed=42,
            method_variant="C",
            output_ckpt=str(tmp_path / f"{run_name}.pt"),
            report_path=str(tmp_path / f"{run_name}.json"),
        ))

    first, second = results
    assert first["initial_parameter_fingerprint"] == second["initial_parameter_fingerprint"]
    assert first["event_plan_fingerprint"] == second["event_plan_fingerprint"]
    assert first["planned_scenario_ids"] == second["planned_scenario_ids"]
    assert first["disturbance_enabled"] is False
    assert first["scenario_log"][0]["actual_hit_count"] == 0
    assert first["scenario_log"][0]["actual_transfer_times"]
    assert first["scenario_log"][0]["actual_transfer_times"] == second["scenario_log"][0]["actual_transfer_times"]
    assert first["first_actual_transfer"]["transfer_time"] == first["scenario_log"][0]["actual_transfer_times"][0]
    assert first["scenario_log"][0]["success"] is False
    assert first["sampling_replay_max_abs_error"] <= 1e-5
    assert first["sampling_replay_scope"] == "first_pre_update_minibatch_per_rollout"
    assert first["history"][0]["ppo_updates"] == 1

    checkpoint = torch.load(first["checkpoint_path"], map_location="cpu", weights_only=False)
    assert checkpoint["resume_capability"] == "non_exact"
    assert checkpoint["checkpoint_role"] == "model_weights"
    assert checkpoint["checkpoint_version"]
    report = json.loads(Path(first["report_path"]).read_text(encoding="utf-8"))
    assert report["run_mode"] == "smoke"
    assert report["research_result_eligible"] is False
    assert report["source_sha"]
    assert report["data_fingerprint"]["scenario_pool_sha256"]
    assert report["device"] == "cpu"
    assert report["memory_peak_bytes"] is None or report["memory_peak_bytes"] > 0


def test_pilot_decision_cap_marks_incomplete_batch_truncated(tmp_path: Path) -> None:
    """决策预算耗尽时未完成批次必须失败/截断，不得登记为生产成功。"""
    result = run_training(
        run_mode="pilot",
        successful_batch_target=1,
        max_decisions=2,
        num_iterations=1,
        steps_per_iter=2,
        ppo_epochs=0,
        batch_size=2,
        seed=42,
        method_variant="C",
        output_ckpt=str(tmp_path / "pilot.pt"),
        report_path=str(tmp_path / "pilot.json"),
    )

    assert result["total_steps"] == 2
    assert result["successful_batch_count"] == 0
    assert result["scenario_log"][0]["success"] is False
    assert result["scenario_log"][0]["truncated"] is True
    assert result["scenario_log"][0]["termination_reason"] == "decision_limit"
    assert result["termination_reason"] == "decision_limit"
    assert result["terminated"] is False
    assert result["truncated"] is True
    assert result["first_scheduled_disturbance"]["scenario_id"] == result["scenario_log"][0]["scenario_id"]
    assert isinstance(result["scenario_log"][0]["actual_hit_count"], int)
    assert "first_actual_hit" in result
    assert "first_actual_transfer" in result


def test_pilot_wall_clock_cap_marks_episode_truncated(tmp_path: Path) -> None:
    """墙钟预算必须能在首个动作前结束并如实标记未完成episode。"""
    result = run_training(
        run_mode="pilot",
        successful_batch_target=1,
        max_wall_seconds=1e-9,
        steps_per_iter=1,
        ppo_epochs=0,
        seed=42,
        method_variant="C",
        output_ckpt=str(tmp_path / "wall_cap.pt"),
        report_path=str(tmp_path / "wall_cap.json"),
    )

    assert result["total_steps"] == 0
    assert result["successful_batch_count"] == 0
    assert result["termination_reason"] == "wall_time_limit"
    assert result["scenario_log"][0]["success"] is False
    assert result["scenario_log"][0]["truncated"] is True


def test_pilot_rejects_nonfinite_wall_clock_cap(tmp_path: Path) -> None:
    """NaN/无穷墙钟上限不能绕过训练退出条件。"""
    with pytest.raises(ValueError, match="有限正数"):
        run_training(
            run_mode="pilot",
            successful_batch_target=1,
            max_wall_seconds=float("nan"),
            steps_per_iter=1,
            method_variant="C",
            output_ckpt=str(tmp_path / "invalid_cap.pt"),
        )


def test_method_d_without_transfer_labels_reports_skipped_supervision(tmp_path: Path) -> None:
    """没有真实转站标签时，D组显式报告跳过而不是零损失收敛。"""
    result = run_training(
        run_mode="pilot",
        successful_batch_target=1,
        max_decisions=1,
        num_iterations=1,
        steps_per_iter=1,
        ppo_epochs=1,
        batch_size=1,
        seed=42,
        method_variant="D",
        output_ckpt=str(tmp_path / "pilot_d.pt"),
        report_path=str(tmp_path / "pilot_d.json"),
    )

    assert result["cycle_time_labels"]
    assert all(label["label_available"] is False for label in result["cycle_time_labels"])
    assert all(label["actual_transfer_time"] is None for label in result["cycle_time_labels"])
    assert result["history"][0]["time_label_count"] == 0
    assert result["history"][0]["time_supervision_status"] == "skipped_no_real_transfer_labels"
    assert result["scenario_log"][0]["truncated"] is True


def test_training_cli_exposes_seed_and_pilot_budget_arguments() -> None:
    """训练命令行必须能显式设置种子、批次目标和预算边界。"""
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "work3" / "train_ppo_work3.py"),
            "--help",
        ],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )

    for option in (
        "--seed",
        "--mode",
        "--successful-batch-target",
        "--max-decisions",
        "--max-wall-seconds",
        "--scenario-split",
    ):
        assert option in completed.stdout


def test_training_cli_forwards_seed_budget_and_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI解析到的种子、预算和场景划分必须传给训练入口。"""
    captured: dict[str, object] = {}
    monkeypatch.setattr(train_module.sys, "argv", [
        "train_ppo_work3.py",
        "--mode", "pilot",
        "--seed", "42",
        "--successful-batch-target", "1",
        "--max-decisions", "64",
        "--scenario-split", "custom_split.json",
        "--output", str(tmp_path / "cli.pt"),
    ])
    monkeypatch.setattr(
        train_module,
        "run_training",
        lambda **kwargs: captured.update(kwargs),
    )

    train_module.main()

    assert captured["seed"] == 42
    assert captured["run_mode"] == "pilot"
    assert captured["successful_batch_target"] == 1
    assert captured["max_decisions"] == 64
    assert str(captured["scenario_split_path"]) == "custom_split.json"


def test_pending_label_cache_reports_missing_cycle_counts() -> None:
    """跨采样段未补齐的周期标签可按episode/cycle如实盘点。"""
    cache = PendingTimeLabelCache()
    for decision_id, cycle_id in enumerate((1, 1, 2)):
        cache.add(
            episode_id=3,
            cycle_id=cycle_id,
            decision_id=decision_id,
            state_feat=torch.zeros(32),
            graph_snapshot=None,
            estimated_cmax=10.0,
            current_time=0.0,
            h0=10.0,
            predictor_version=0,
        )

    assert cache.pending_cycle_counts(3) == {1: 2, 2: 1}


def test_paired_c_d_reports_distinguish_schedule_from_actual_hit_pattern() -> None:
    """相同外生事件计划不代表策略实际命中相同，报告必须分开比较。"""
    common = {
        "method_variant": "C",
        "seed": 42,
        "data_fingerprint": {"scenario_pool_sha256": "pool"},
        "event_plan_fingerprint": "events",
        "training_config": {
            "max_decisions": 100,
            "max_wall_seconds": None,
            "successful_batch_target": 1,
            "rollout_steps": 32,
        },
        "scenario_log": [{"episode_id": 0, "scenario_id": "S1", "actual_hit_count": 1}],
    }
    paired = {
        **common,
        "method_variant": "D",
        "scenario_log": [{"episode_id": 0, "scenario_id": "S1", "actual_hit_count": 0}],
    }

    comparison = _compare_training_reports(common, paired)

    assert comparison["same_event_plan"] is True
    assert comparison["same_seed"] is True
    assert comparison["same_interaction_budget"] is True
    assert comparison["same_actual_hit_pattern"] is False
