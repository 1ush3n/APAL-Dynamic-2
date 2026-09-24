"""W3-10/F09：训练扰动、正式 C/D 分组与检查点边界反例。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.work3.evaluate_c_vs_d import build_formal_evaluation_agent
from scripts.work3.experiment_protocol import build_method_profile
from scripts.work3.train_ppo_work3 import (
    build_episode_scenario_plan,
    load_training_scenarios,
    run_training,
)


def test_formal_c_and_d_share_graph_policy_but_only_d_enables_time_learning() -> None:
    """正式 C/D 只能在确认的时间学习机制上不同。"""
    profile_c = build_method_profile("C")
    profile_d = build_method_profile("D")

    assert profile_c.graph_policy is True
    assert profile_d.graph_policy is True
    assert profile_c.allow_postpone is True
    assert profile_d.allow_postpone is True
    assert profile_c.use_time_auxiliary is False
    assert profile_d.use_time_auxiliary is True
    assert profile_c.use_corrected_time_input is False
    assert profile_d.use_corrected_time_input is True
    assert profile_c.use_learned_time_shaping is False
    assert profile_d.use_learned_time_shaping is True


def test_training_loads_explicit_split_and_builds_mixed_episode_plan(tmp_path: Path) -> None:
    """训练池来自固定清单，不从当前算法状态重新选择事件。"""
    scenarios = [
        {
            "scenario_id": "EARLY_LOW_S0",
            "timing": "EARLY",
            "intensity": "LOW",
            "station_id": 0,
            "aircraft_id": 5,
            "tau": 10.0,
            "delta": 1.0,
            "recovery_time": 11.0,
            "affected_task_keys": ["5_16"],
            "valid": True,
        },
        {
            "scenario_id": "LATE_HIGH_S4",
            "timing": "LATE",
            "intensity": "HIGH",
            "station_id": 4,
            "aircraft_id": 1,
            "tau": 20.0,
            "delta": 2.0,
            "recovery_time": 22.0,
            "affected_task_keys": ["1_16"],
            "valid": True,
        },
    ]
    scenarios_path = tmp_path / "scenarios.json"
    split_path = tmp_path / "train.json"
    scenarios_path.write_text(json.dumps(scenarios), encoding="utf-8")
    split_path.write_text(json.dumps([scenarios[1]]), encoding="utf-8")

    loaded = load_training_scenarios(str(scenarios_path), str(split_path))
    plan = build_episode_scenario_plan(loaded, num_episodes=4, seed=42)

    assert [item["scenario_id"] for item in loaded] == ["LATE_HIGH_S4"]
    assert len(plan) == 4
    assert {item["scenario_id"] for item in plan} == {"LATE_HIGH_S4"}
    assert all(item["timing"] == "LATE" for item in plan)
    assert all(item["intensity"] == "HIGH" for item in plan)


def test_formal_method_d_requires_checkpoint_unless_debug_mode(tmp_path: Path) -> None:
    """正式 D 不得把随机权重输出成训练后方法；调试随机权重必须显式开启。"""
    missing = tmp_path / "missing_method_d.pt"
    with pytest.raises(FileNotFoundError, match="正式方法 D"):
        build_formal_evaluation_agent(
            method_variant="D",
            checkpoint_path=missing,
            device="cpu",
            debug_random=False,
        )

    debug_agent = build_formal_evaluation_agent(
        method_variant="D",
        checkpoint_path=missing,
        device="cpu",
        debug_random=True,
    )
    assert debug_agent.debug_random is True
    assert debug_agent.profile.name == "D"


def test_short_formal_c_training_loads_disturbance_and_records_episode(tmp_path: Path) -> None:
    """最小训练烟测必须真的加载扰动，而不是退回无扰动生产。"""
    result = run_training(
        run_mode="pilot",
        successful_batch_target=1,
        max_decisions=2,
        num_iterations=1,
        steps_per_iter=2,
        ppo_epochs=1,
        batch_size=2,
        method_variant="C",
        output_ckpt=str(tmp_path / "method_c.pt"),
        device="cpu",
    )

    assert result["method_variant"] == "C"
    assert result["scenario_log"]
    assert result["scenario_log"][0]["scenario_id"]
    assert result["scenario_log"][0]["timing"] in {"EARLY", "MID", "LATE"}
    assert isinstance(result["scenario_log"][0]["actual_hit_count"], int)
