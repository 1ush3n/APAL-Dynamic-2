"""R06 专项测试：扰动场景指纹唯一性、训练/验证/测试零重叠划分与真实比例统计契约。"""

from __future__ import annotations

import json
import sys
from itertools import product
from math import isclose
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.work3.evaluate_c_vs_d as evaluation
import utils.work3.disturbance_generator as dg
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline
from utils.work3.disturbance_generator import (
    generate_9class_scenarios,
    split_scenarios,
)


def _event_fp(scenario: dict) -> tuple[int, tuple[str, ...], float, float]:
    fn = getattr(dg, "compute_event_fingerprint", None)
    if callable(fn):
        return fn(scenario)
    return (
        int(scenario["aircraft_id"]),
        tuple(sorted(str(k) for k in scenario["affected_task_keys"])),
        round(float(scenario["tau"]), 4),
        round(float(scenario["recovery_time"]), 4),
    )


def _group_key(scenario: dict) -> tuple[int, int, tuple[str, ...]]:
    fn = getattr(dg, "compute_target_group_key", None)
    if callable(fn):
        return fn(scenario)
    return (
        int(scenario["station_id"]),
        int(scenario["aircraft_id"]),
        tuple(sorted(str(k) for k in scenario["affected_task_keys"])),
    )


def test_r06_scenarios_report_honest_fraction_and_degeneracy(tmp_path):
    """45 场景回归集必须诚实报告 candidate_count、intended_count、actual_fraction 与小站位离散饱和原因。"""
    out_file = tmp_path / "scenarios_9class.json"
    scenarios = generate_9class_scenarios(output_json_path=str(out_file))
    assert len(scenarios) == 45

    assert hasattr(dg, "summarize_scenario_diagnostics")
    diag = dg.summarize_scenario_diagnostics([s.to_dict() for s in scenarios])
    assert len(diag["rows"]) == 45

    for item in diag["rows"]:
        assert "candidate_count" in item
        assert "intended_count" in item
        assert "actual_hit_count" in item
        assert "actual_fraction" in item
        assert "delta_h0_ratio" in item
        if item["candidate_count"] > 0:
            expected_frac = round(item["actual_hit_count"] / item["candidate_count"], 6)
            assert abs(item["actual_fraction"] - expected_frac) < 1e-6


def test_r06_split_scenarios_guarantees_zero_fingerprint_and_group_overlap():
    """候选扰动池按场景指纹与目标工序组拆分后，train/validation/test 之间必须零指纹重叠、零目标组泄漏。"""
    assert hasattr(dg, "generate_candidate_scenario_pool")
    pool = dg.generate_candidate_scenario_pool(seeds=(2026, 2027, 2028, 2029, 2030, 2031))
    assert len(pool) >= 45

    splits = split_scenarios(
        pool,
        seed=2026,
        train_ratio=0.6,
        validation_ratio=0.2,
        strict_group_isolation=True,
    )

    train_fp = {_event_fp(s) for s in splits["train"]}
    val_fp = {_event_fp(s) for s in splits["validation"]}
    test_fp = {_event_fp(s) for s in splits["test"]}

    assert train_fp.isdisjoint(val_fp)
    assert train_fp.isdisjoint(test_fp)
    assert val_fp.isdisjoint(test_fp)

    train_groups = {_group_key(s) for s in splits["train"]}
    val_groups = {_group_key(s) for s in splits["validation"]}
    test_groups = {_group_key(s) for s in splits["test"]}

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)


def test_r06_duplicate_event_fingerprint_never_leaks_across_splits(tmp_path):
    """即使输入包含相同事件指纹或重叠目标组的条目，split_scenarios 也绝不能把同一事件指纹分进两个不同子集。"""
    out_file = tmp_path / "scenarios_9class.json"
    scenarios = [s.to_dict() for s in generate_9class_scenarios(output_json_path=str(out_file))]
    duplicated = list(scenarios) + [dict(scenarios[0], scenario_id="ZZZ_DUP_CLONE_0")]

    splits = split_scenarios(duplicated, seed=2026)
    train_fp = {_event_fp(s) for s in splits["train"]}
    val_fp = {_event_fp(s) for s in splits["validation"]}
    test_fp = {_event_fp(s) for s in splits["test"]}

    assert train_fp.isdisjoint(val_fp)
    assert train_fp.isdisjoint(test_fp)
    assert val_fp.isdisjoint(test_fp)


def test_default_formal_test_manifest_covers_all_conditions_and_one_aircraft() -> None:
    """默认正式测试清单覆盖45个类别×站位格，且单场景目标不跨飞机或泄漏训练验证。"""
    test_path = Path("data/work3/experiment_splits/test.json")
    train_path = Path("data/work3/experiment_splits/train.json")
    validation_path = Path("data/work3/experiment_splits/validation.json")
    test_scenarios = json.loads(test_path.read_text(encoding="utf-8"))
    metadata = json.loads(
        Path("data/work3/experiment_splits/metadata.json").read_text(
            encoding="utf-8"
        )
    )
    train_validation = [
        *json.loads(train_path.read_text(encoding="utf-8")),
        *json.loads(validation_path.read_text(encoding="utf-8")),
    ]
    baseline = MultiAircraftBaseline.load_from_json(
        "data/work3/real_283_k10_baseline.json"
    )

    expected_cells = set(
        product(("EARLY", "MID", "LATE"), ("LOW", "MID", "HIGH"), range(5))
    )
    actual_cells = {
        (item["timing"], item["intensity"], int(item["station_id"]))
        for item in test_scenarios
    }
    assert len(test_scenarios) == 45
    assert actual_cells == expected_cells
    assert metadata["scenario_counts"]["test"] == 45
    assert metadata["test_generation"]["cycle_offset"] == 1
    assert metadata["test_generation"]["minimum_coverage_cells"] == 45

    timing_ratios = {"EARLY": 0.225, "MID": 0.400, "LATE": 0.575}
    for scenario in test_scenarios:
        station_id = int(scenario["station_id"])
        expected_aircraft = dg.CYCLE_6_STATION_AIRCRAFT[station_id] + 1
        assert int(scenario["aircraft_id"]) == expected_aircraft
        target_tasks = [baseline.tasks[key] for key in scenario["affected_task_keys"]]
        assert target_tasks
        assert {task.aircraft_id for task in target_tasks} == {
            expected_aircraft
        }
        assert {task.station_id for task in target_tasks} == {station_id + 1}
        tau = float(scenario["tau"])
        assert all(float(task.baseline_start) >= tau - 1e-4 for task in target_tasks)
        previous_station_tasks = [
            task
            for task in baseline.tasks.values()
            if task.aircraft_id == expected_aircraft
            and task.station_id < station_id + 1
        ]
        assert all(
            float(task.baseline_end) <= tau + 1e-4
            for task in previous_station_tasks
        )
        station_tasks = [
            task
            for task in baseline.tasks.values()
            if task.aircraft_id == expected_aircraft
            and task.station_id == station_id + 1
        ]
        station_span = max(float(task.in_station_offset) for task in station_tasks)
        expected_tau = (
            6.0 * baseline.h0
            + timing_ratios[scenario["timing"]] * station_span
        )
        assert isclose(float(scenario["tau"]), expected_tau, abs_tol=1e-3)

    test_fingerprints = {_event_fp(item) for item in test_scenarios}
    prior_fingerprints = {_event_fp(item) for item in train_validation}
    test_groups = {_group_key(item) for item in test_scenarios}
    prior_groups = {_group_key(item) for item in train_validation}
    assert test_fingerprints.isdisjoint(prior_fingerprints)
    assert test_groups.isdisjoint(prior_groups)


def test_default_test_manifest_is_reproducible_from_next_cycle_scenarios(
    tmp_path: Path,
) -> None:
    """测试split逐项等于下一脉动周期的固定生成结果，避免手工场景漂移。"""
    generated_path = tmp_path / "cycle7_scenarios.json"
    generate_9class_scenarios(
        output_json_path=str(generated_path),
        cycle_offset=1,
    )
    generated = json.loads(generated_path.read_text(encoding="utf-8"))
    actual = json.loads(
        Path("data/work3/experiment_splits/test.json").read_text(encoding="utf-8")
    )
    generated_by_id = {item["scenario_id"]: item for item in generated}
    actual_by_id = {item["scenario_id"]: item for item in actual}

    assert actual_by_id == generated_by_id


def test_later_cycle_generation_does_not_overwrite_the_cycle_six_regression_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未显式指定输出路径时，后续周期场景写入独立文件而保留原回归文件名。"""
    baseline_path = Path("data/work3/real_283_k10_baseline.json").resolve()
    monkeypatch.chdir(tmp_path)

    generate_9class_scenarios(
        baseline_json_path=str(baseline_path),
        cycle_offset=1,
    )

    assert Path("data/work3/scenarios_9class_cycle7.json").is_file()
    assert not Path("data/work3/scenarios_9class.json").exists()


def test_scenario_generator_rejects_non_integer_cycle_offset(tmp_path: Path) -> None:
    """周期偏移在公共生成器入口拒绝浮点值，而不延迟到任务查找时报错。"""
    with pytest.raises(TypeError, match="cycle_offset必须为整数"):
        generate_9class_scenarios(
            output_json_path=str(tmp_path / "invalid.json"),
            cycle_offset=1.5,  # type: ignore[arg-type]
        )


def test_formal_evaluation_cli_defaults_to_the_independent_test_manifest() -> None:
    """默认正式评测将独立test清单完整交给评测入口，而非回归场景池。"""
    captured: dict[str, object] = {}
    with patch.object(sys, "argv", ["evaluate_c_vs_d"]):
        with patch.object(
            evaluation,
            "run_benchmark_evaluation",
            side_effect=lambda **kwargs: captured.update(kwargs),
        ) as run_evaluation:
            evaluation.main()

    run_evaluation.assert_called_once()
    assert captured["scenario_split_path"] == "data/work3/experiment_splits/test.json"
