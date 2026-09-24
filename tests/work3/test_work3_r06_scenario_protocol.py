"""R06 专项测试：扰动场景指纹唯一性、训练/验证/测试零重叠划分与真实比例统计契约。"""

from __future__ import annotations

import utils.work3.disturbance_generator as dg
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
