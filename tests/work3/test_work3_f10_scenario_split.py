"""W3-11/F10：独立事件划分、扰动标定和实际影响统计反例。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.work3.collect_validation_trajectories import load_scenarios_for_split
from scripts.work3.evaluate_c_vs_d import (
    evaluate_single_trajectory,
    summarize_disturbance_effects,
)
from utils.work3.disturbance_generator import (
    CYCLE_6_STATION_AIRCRAFT,
    INTENSITY_SPECS,
    TIMING_OFFSETS,
    generate_9class_scenarios,
    select_unstarted_candidates,
    split_scenarios,
    write_experiment_splits,
)
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


@pytest.fixture
def baseline_path() -> Path:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return path


def test_scenario_time_scale_and_candidate_generation_are_consistent(
    baseline_path: Path,
    tmp_path: Path,
) -> None:
    """扰动时机统一使用站内相对位置，空候选不得偷偷回退到全部工序。"""
    output_path = tmp_path / "scenarios.json"
    scenarios = generate_9class_scenarios(str(baseline_path), str(output_path))
    baseline = MultiAircraftBaseline.load_from_json(str(baseline_path))

    assert len(scenarios) == 45
    assert {scenario.station_id for scenario in scenarios} == set(CYCLE_6_STATION_AIRCRAFT)
    for scenario in scenarios:
        station_tasks = [
            task
            for task in baseline.tasks.values()
            if task.aircraft_id == scenario.aircraft_id
            and task.station_id == scenario.station_id + 1
        ]
        station_span = max(task.in_station_offset for task in station_tasks)
        expected_tau = 5.0 * baseline.h0 + TIMING_OFFSETS[scenario.timing] * station_span
        assert scenario.tau == pytest.approx(expected_tau, abs=1e-3)
        assert scenario.timing_reference == "station_task_span"
        assert scenario.valid is True
        assert scenario.candidate_count > 0
        assert scenario.affected_count > 0
        assert scenario.delta == pytest.approx(
            INTENSITY_SPECS[scenario.intensity]["delta_ratio"] * baseline.h0,
            abs=1e-3,
        )

    fake_tasks = [
        SimpleNamespace(
            baseline_start=1.0,
            in_station_offset=0.0,
            task_id=1,
            task_key="1",
        )
    ]
    candidates, reason = select_unstarted_candidates(fake_tasks, tau=2.0)
    assert candidates == []
    assert reason == "no_unstarted_candidate"


def test_scenario_splits_are_deterministic_disjoint_and_cover_all_stations(
    baseline_path: Path,
    tmp_path: Path,
) -> None:
    """先按事件种子划分，再采集轨迹；三组不能共享事件。"""
    scenarios = [
        scenario.to_dict()
        for scenario in generate_9class_scenarios(
            str(baseline_path),
            str(tmp_path / "scenarios.json"),
        )
    ]
    first = split_scenarios(scenarios, seed=2026)
    second = split_scenarios(scenarios, seed=2026)

    assert first == second
    train_ids = {item["scenario_id"] for item in first["train"]}
    val_ids = {item["scenario_id"] for item in first["validation"]}
    test_ids = {item["scenario_id"] for item in first["test"]}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == {item["scenario_id"] for item in scenarios}
    assert {item["station_id"] for item in first["test"]} == set(CYCLE_6_STATION_AIRCRAFT)

    output_dir = tmp_path / "splits"
    write_experiment_splits(first, output_dir)
    loaded = json.loads((output_dir / "test.json").read_text(encoding="utf-8"))
    assert [item["scenario_id"] for item in loaded] == [item["scenario_id"] for item in first["test"]]


def test_collector_uses_explicit_split_manifest(tmp_path: Path, baseline_path: Path) -> None:
    """轨迹采集器按清单读取事件，不从默认45场景池重新挑选。"""
    scenarios_path = tmp_path / "scenarios.json"
    split_path = tmp_path / "split.json"
    scenarios = generate_9class_scenarios(str(baseline_path), str(scenarios_path))
    selected = [scenarios[0].to_dict(), scenarios[10].to_dict()]
    split_path.write_text(json.dumps(selected), encoding="utf-8")

    loaded = load_scenarios_for_split(str(scenarios_path), str(split_path))
    assert [item["scenario_id"] for item in loaded] == [item["scenario_id"] for item in selected]


def test_effect_report_distinguishes_hit_wait_and_cross_aircraft_impact() -> None:
    """评测必须报告实际命中、观察到的等待和其他飞机传播。"""
    tasks = {
        "target": SimpleNamespace(
            aircraft_id=1,
            material_ready_time=20.0,
            actual_start=25.0,
        ),
        "other": SimpleNamespace(
            aircraft_id=2,
            material_ready_time=0.0,
            actual_start=15.0,
        ),
        "same_aircraft": SimpleNamespace(
            aircraft_id=1,
            material_ready_time=0.0,
            actual_start=11.0,
        ),
    }
    report = summarize_disturbance_effects(
        tasks,
        {
            "affected_task_keys": ["target"],
            "aircraft_id": 1,
        },
        baseline_start_by_key={"target": 10.0, "other": 10.0, "same_aircraft": 10.0},
    )

    assert report["target_count"] == 1
    assert report["actual_hit_count"] == 1
    assert report["actual_hit_rate"] == pytest.approx(1.0)
    assert report["observed_added_wait_hours"] == pytest.approx(15.0)
    assert report["cross_aircraft_affected_task_count"] == 1
    assert report["cross_aircraft_affected_aircraft_ids"] == [2]


def test_effect_report_explains_each_unhit_target() -> None:
    tasks = {
        "started-before-event": SimpleNamespace(
            aircraft_id=1,
            material_ready_time=0.0,
            actual_start=5.0,
        ),
        "not-delayed": SimpleNamespace(
            aircraft_id=1,
            material_ready_time=0.0,
            actual_start=12.0,
        ),
    }

    report = summarize_disturbance_effects(
        tasks,
        {
            "affected_task_keys": [
                "started-before-event",
                "not-delayed",
                "missing-target",
            ],
            "aircraft_id": 1,
            "tau": 10.0,
        },
        baseline_start_by_key={},
    )

    assert report["unhit_reasons"] == {
        "started-before-event": "already_started_or_completed_at_event",
        "not-delayed": "no_material_delay_recorded",
        "missing-target": "target_not_in_instance",
    }

    untriggered_report = summarize_disturbance_effects(
        tasks,
        {
            "affected_task_keys": ["started-before-event", "missing-target"],
            "aircraft_id": 1,
            "tau": 10.0,
        },
        baseline_start_by_key={},
        event_triggered=False,
    )
    assert untriggered_report["unhit_reasons"] == {
        "started-before-event": "event_not_triggered",
        "missing-target": "event_not_triggered",
    }


def test_environment_tracks_whether_a_scenario_disturbance_was_processed(
    baseline_path: Path,
) -> None:
    from envs.work3.environment import AirLineEnvWork3

    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()
    assert env.disturbance_event_triggered is False

    env.load_scenario({
        "scenario_id": "IMMEDIATE_EVENT",
        "tau": 0.0,
        "recovery_time": 1.0,
        "affected_task_keys": ["missing-target"],
    })
    assert env.disturbance_event_triggered is True

    env.reset()
    env.load_scenario({
        "scenario_id": "FUTURE_EVENT",
        "tau": 100.0,
        "recovery_time": 101.0,
        "affected_task_keys": ["missing-target"],
    })
    assert env.disturbance_event_triggered is False
    assert env._advance_to_next_event() is True
    assert env.disturbance_event_triggered is True


def test_formal_evaluation_reports_an_event_that_has_not_triggered(
    baseline_path: Path,
) -> None:
    from envs.work3.environment import AirLineEnvWork3

    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    result = evaluate_single_trajectory(
        env,
        "Baseline-C",
        None,
        scenario={
            "scenario_id": "NOT_YET_TRIGGERED",
            "aircraft_id": 0,
            "tau": 100.0,
            "recovery_time": 101.0,
            "affected_task_keys": ["missing-target"],
        },
        max_decisions=0,
    )

    assert result["disturbance_effects"]["event_triggered"] is False
    assert result["disturbance_effects"]["unhit_reasons"] == {
        "missing-target": "event_not_triggered"
    }


def test_renamed_semantic_duplicate_is_not_counted_as_an_independent_event() -> None:
    """只改scenario_id的重复扰动在三子集切分前应合并为一个事件。"""
    scenarios = [
        {
            "scenario_id": f"S{station_id}_E{event_index}",
            "station_id": station_id,
            "aircraft_id": station_id,
            "affected_task_keys": [f"{station_id}_{event_index}"],
            "tau": float(event_index + 1),
            "recovery_time": float(event_index + 2),
            "valid": True,
        }
        for station_id in range(5)
        for event_index in range(3)
    ]
    renamed_duplicate = {
        **scenarios[0],
        "scenario_id": "RENAMED_SAME_EVENT",
    }

    splits = split_scenarios(
        scenarios + [renamed_duplicate],
        seed=2026,
    )
    output = [item for split in splits.values() for item in split]
    fingerprints = [
        (
            int(item["aircraft_id"]),
            tuple(sorted(str(key) for key in item["affected_task_keys"])),
            round(float(item["tau"]), 4),
            round(float(item["recovery_time"]), 4),
        )
        for item in output
    ]

    assert len(output) == 15
    assert len(fingerprints) == len(set(fingerprints))
    assert sum(
        item["scenario_id"] in {scenarios[0]["scenario_id"], "RENAMED_SAME_EVENT"}
        for item in output
    ) == 1
