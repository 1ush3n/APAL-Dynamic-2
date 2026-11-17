"""闭环选模保留初始化、失败及并列规则。"""
from __future__ import annotations

import pytest


def test_resumed_evaluation_rejects_configuration_drift() -> None:
    from omegaconf import OmegaConf
    from apal.evaluation.bc_diagnostics import BCDiagnosis
    diagnosis = object.__new__(BCDiagnosis)
    diagnosis.config = OmegaConf.create({"evaluation": {"max_decisions": 1000}})
    diagnosis.plan = {"config": OmegaConf.to_container(diagnosis.config), "source_identities": {}}
    diagnosis.validate_plan()
    diagnosis.config.evaluation.max_decisions = 2
    with pytest.raises(ValueError, match="配置已冻结"):
        diagnosis.validate_plan()


def test_selection_includes_initial_and_does_not_hide_failures() -> None:
    from apal.evaluation.bc_diagnostics import select_checkpoint
    points = [
        {"update": 0, "validation_nll": 2.0, "legal_complete": 8, "total": 8, "mean_gap": 0.2},
        {"update": 25, "validation_nll": 0.1, "legal_complete": 7, "total": 8, "mean_gap": 0.01},
        {"update": 50, "validation_nll": 0.2, "legal_complete": 8, "total": 8, "mean_gap": 0.3},
        {"update": 75, "validation_nll": 0.3, "legal_complete": 8, "total": 8, "mean_gap": 0.2 - 1e-12},
    ]
    assert select_checkpoint(points, "validation_nll")["update"] == 25
    assert select_checkpoint(points, "closure")["update"] == 0
    failed = [{"update": step, "validation_nll": 1.0, "legal_complete": 0, "total": 8,
               "mean_gap": None} for step in (25, 0)]
    assert select_checkpoint(failed, "closure")["update"] == 0
    with pytest.raises(ValueError):
        select_checkpoint([{**points[0], "total": 7}, points[1]], "closure")


def test_truncation_has_no_terminal_cost_or_gap() -> None:
    from apal.evaluation.bc_diagnostics import result_row
    case = {"instance_id": "A", "parent_group": "parent", "family": "structured", "split": "train"}
    model = {"model_id": "W0", "seed": 42, "update": 0, "path": "initial.pt", "rule": "initial"}
    report = {"success": False, "terminated": False, "truncated": True, "termination_reason": "decision_limit",
              "makespan": 99.0, "C_task": 98.0, "independent_feasible": True,
              "cost_ledger_matches": True, "decisions": 3, "elapsed_seconds": 0.1}
    row = result_row(model, case, report, optimum=10.0, phase="A")
    assert row["C_exit"] is None and row["C_task"] is None and row["gap"] is None
    assert row["truncated"] and not row["success"]
    complete = {**report, "success": True, "terminated": True, "truncated": False,
                "termination_reason": "completed", "makespan": 9.0}
    with pytest.raises(ValueError, match="证书"):
        result_row(model, case, complete, optimum=10.0, phase="A")
