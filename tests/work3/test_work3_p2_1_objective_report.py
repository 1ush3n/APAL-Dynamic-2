"""P2-1目标分项、加权费用与最终站位指标的口径验收。"""

from types import SimpleNamespace
from pathlib import Path

import pytest

from envs.work3.environment import AirLineEnvWork3
from scripts.work3.evaluate_c_vs_d import (
    evaluate_single_trajectory,
    summarize_final_station_deviation,
)
from utils.work3.objective_evaluator import (
    ObjectiveWeights,
    build_objective_report,
    evaluate_trajectory_objective,
)


def _task(
    *,
    actual_start: float,
    cycle_start_time: float,
    in_station_offset: float,
    assigned_team: tuple[int, ...],
    base_team: tuple[int, ...],
    postpone_count: int = 0,
    revision_history: list[dict[str, float]] | None = None,
    base_station: int = 0,
    actual_start_station: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        actual_start=actual_start,
        cycle_start_time=cycle_start_time,
        in_station_offset=in_station_offset,
        assigned_team=list(assigned_team),
        base_team=base_team,
        demand=len(base_team),
        postpone_count=postpone_count,
        revision_history=revision_history or [],
        base_station=base_station,
        actual_start_station=actual_start_station,
    )


def test_objective_report_separates_hand_calculated_metrics_and_weighted_costs() -> None:
    state = SimpleNamespace(
        h0=10.0,
        normalization_task_count=2,
        transfer_history=[14.0, 25.0, 35.0],
        tasks={
            "a": _task(
                actual_start=113.0,
                cycle_start_time=110.0,
                in_station_offset=3.0,
                assigned_team=(1, 2, 3, 5),
                base_team=(1, 2, 3, 4),
                postpone_count=1,
                base_station=0,
                actual_start_station=1,
            ),
            "b": _task(
                actual_start=112.0,
                cycle_start_time=110.0,
                in_station_offset=3.0,
                assigned_team=(6, 7, 8, 9),
                base_team=(6, 7, 8, 9),
                postpone_count=2,
                revision_history=[{"time_change": 0.2, "team_change": 0.25}],
            ),
        },
    )
    weights = ObjectiveWeights(
        w_h=2.0,
        w_t=3.0,
        w_w=4.0,
        w_p=5.0,
        lambda_1=0.1,
        lambda_2=0.2,
        normalize_by_n=True,
        w_revision_time=7.0,
        w_revision_team=8.0,
    )

    breakdown = evaluate_trajectory_objective(state, weights)
    report = build_objective_report(breakdown, weights, h0=10.0)

    assert breakdown.takt_violation_hours == pytest.approx(5.0)
    assert report["metrics_unweighted"] == pytest.approx(
        {
            "j_h": 0.5,
            "d_t": 0.05,
            "d_w": 0.125,
            "g_postpone": 0.5,
            "r_t": 0.1,
            "r_w": 0.125,
        }
    )
    assert report["cost_components_weighted"] == pytest.approx(
        {
            "j_takt": 1.0,
            "d_time": 0.15,
            "d_team": 0.5,
            "j_postpone": 2.5,
            "j_revision": 1.7,
        }
    )
    assert sum(report["cost_components_weighted"].values()) == pytest.approx(5.85)
    assert report["j_total"] == pytest.approx(5.85)
    assert report["normalization"]["h0_hours"] == 10.0
    assert report["normalization"]["normalize_by_n"] is True
    assert report["normalization"]["fixed_task_count"] == 2
    assert "不除以N" in report["normalization"]["metric_rules"]["g_postpone"]
    assert report["metric_coverage"]["status"] == "all_tasks_started"


def test_final_station_deviation_uses_started_tasks_and_withholds_partial_final_ratio() -> None:
    tasks = {
        "moved": SimpleNamespace(
            actual_start=12.0,
            actual_start_station=1,
            base_station=0,
        ),
        "unchanged": SimpleNamespace(
            actual_start=15.0,
            actual_start_station=2,
            base_station=2,
        ),
        "not_started": SimpleNamespace(
            actual_start=None,
            actual_start_station=None,
            base_station=4,
        ),
    }

    completed = summarize_final_station_deviation(
        tasks,
        total_tasks=2,
        success=True,
        termination_reason="completed",
    )
    partial = summarize_final_station_deviation(
        tasks,
        total_tasks=3,
        success=False,
        termination_reason="rollout_truncated",
    )

    assert completed["changed_started_task_count"] == 1
    assert completed["started_task_count"] == 2
    assert completed["final_deviation_ratio"] == pytest.approx(0.5)
    assert completed["valid_for_final_comparison"] is True
    assert partial["changed_started_task_count"] == 1
    assert partial["started_task_count"] == 2
    assert partial["final_deviation_ratio"] is None
    assert partial["observed_started_ratio"] == pytest.approx(0.5)
    assert partial["valid_for_final_comparison"] is False


def test_evaluation_result_names_weighted_legacy_fields_and_marks_partial_station_ratio() -> None:
    baseline_path = Path("data/work3/real_283_k10_baseline.json")
    if not baseline_path.is_file():
        pytest.skip(f"{baseline_path}不存在")
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))

    result = evaluate_single_trajectory(
        env,
        agent_type="Heuristic-Debug",
        agent=None,
        max_decisions=0,
    )

    assert result["cost_schema_version"] == "work3_trajectory_cost_v2"
    assert result["cost_components_weighted"] == result["raw_cost_components"]
    assert "已加权分项" in result["legacy_field_semantics"]["raw_cost_components"]
    assert set(result["metrics_unweighted"]) == {
        "j_h",
        "d_t",
        "d_w",
        "g_postpone",
        "r_t",
        "r_w",
    }
    assert result["ledger_reconciliation"]["consistent"] is True
    assert result["ledger_reconciliation"]["reward_sum_matches"] is True
    assert result["final_station_deviation"]["final_deviation_ratio"] is None
    assert result["final_station_deviation"]["status"] == "partial_or_invalid"
    assert result["metric_coverage"]["status"] == "partial"
    assert result["objective_normalization"]["normalize_by_n"] is True


def test_objective_report_respects_disabled_task_count_normalization() -> None:
    state = SimpleNamespace(
        h0=10.0,
        normalization_task_count=2,
        transfer_history=[],
        tasks={
            "one": _task(
                actual_start=12.0,
                cycle_start_time=10.0,
                in_station_offset=3.0,
                assigned_team=(1, 2, 3, 4),
                base_team=(1, 2, 3, 4),
                revision_history=[{"time_change": 0.5, "team_change": 0.25}],
            )
        },
    )
    weights = ObjectiveWeights(
        w_h=0.0,
        w_t=2.0,
        w_w=4.0,
        w_p=0.0,
        normalize_by_n=False,
    )

    breakdown = evaluate_trajectory_objective(state, weights)
    report = build_objective_report(breakdown, weights, h0=10.0)

    assert report["metrics_unweighted"]["d_t"] == pytest.approx(0.1)
    assert report["metrics_unweighted"]["r_t"] == pytest.approx(0.5)
    assert report["metrics_unweighted"]["r_w"] == pytest.approx(0.25)
    assert report["cost_components_weighted"]["j_revision"] == pytest.approx(2.0)
    assert report["normalization"]["normalize_by_n"] is False


@pytest.mark.parametrize(
    ("actual_team", "expected_team_deviation"),
    [((1, 2, 3, 5), 0.25), ((1, 5, 6, 7), 0.75)],
)
def test_team_deviation_is_replaced_member_fraction(
    actual_team: tuple[int, ...],
    expected_team_deviation: float,
) -> None:
    state = SimpleNamespace(
        h0=10.0,
        normalization_task_count=1,
        transfer_history=[],
        tasks={
            "task": _task(
                actual_start=5.0,
                cycle_start_time=0.0,
                in_station_offset=5.0,
                assigned_team=actual_team,
                base_team=(1, 2, 3, 4),
            )
        },
    )
    weights = ObjectiveWeights(
        w_h=0.0,
        w_t=0.0,
        w_w=1.0,
        w_p=0.0,
        normalize_by_n=False,
    )

    breakdown = evaluate_trajectory_objective(state, weights)

    assert breakdown.d_team == pytest.approx(expected_team_deviation)
