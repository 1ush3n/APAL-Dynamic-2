"""Regression tests for Trajectory Facts Verifier loopholes and counterexamples.

Verifies that the 8 classes of malicious/corrupted inputs are strictly REJECTED (FAIL / INCOMPLETE),
while the genuine 6 archived execution facts strictly PASS.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import pytest
import shutil

from work3_4060.verify_trajectory_facts import (
    CANONICAL_MANIFEST,
    COST_TOLERANCE,
    PHYSICAL_TIME_TOLERANCE,
    load_baseline_tasks,
    load_scenarios_pool,
    verify_all_trajectories,
    verify_single_trajectory,
)
from utils.work3.objective_evaluator import calculate_postpone_penalty

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "work3_4060/rule_baseline_ac_v1/mini210_rule_baseline.json"
SCENARIOS_PATH = REPO_ROOT / "work3_4060/rule_baseline_ac_v1/scenarios.json"
TRAJ_DIR = REPO_ROOT / "work3_4060/rule_baseline_ac_v1/trajectories"


@pytest.fixture(scope="module")
def baseline_data():
    h0, tasks, meta = load_baseline_tasks(BASELINE_PATH)
    scenarios = load_scenarios_pool(SCENARIOS_PATH)
    return h0, tasks, meta, scenarios


def test_genuine_six_trajectories_all_pass(baseline_data):
    """验证原始归档的 6 条真实物理事实全部通过独立核验，退出码为 0。"""
    summary = verify_all_trajectories(
        traj_dir=TRAJ_DIR,
        baseline_path=BASELINE_PATH,
        scenarios_path=SCENARIOS_PATH,
    )
    assert summary["overall_status"] == "PASS"
    assert summary["exit_code"] == 0
    assert summary["all_pass"] is True
    assert summary["verified_canonical_count"] == 6
    for t_res in summary["trajectories"]:
        assert t_res["status"] == "PASS"
        assert t_res["total_violations"] == 0


def test_loophole_1_missing_task_rejected(baseline_data, tmp_path: Path):
    """反例 1: 从 A_NOMINAL 中删除任务 6_125，核验器必须拦截并判定 FAIL。"""
    h0, tasks, _, scenarios = baseline_data
    with (TRAJ_DIR / "A_NOMINAL.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    del data["execution_facts"]["tasks_execution"]["6_125"]
    bad_file = tmp_path / "A_NOMINAL_missing_task.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    res = verify_single_trajectory(
        traj_path=bad_file,
        baseline_h0=h0,
        baseline_tasks=tasks,
        scenario_spec={"scenario_id": "NOMINAL"},
    )
    assert res["status"] == "FAIL"
    assert len(res["violations"]["task_set_mismatch"]) > 0


def test_loophole_2_missing_aircraft_execution_rejected(baseline_data, tmp_path: Path):
    """反例 2: 删除全部飞机进出站事实，核验器必须拦截并判定 FAIL。"""
    h0, tasks, _, _ = baseline_data
    with (TRAJ_DIR / "A_NOMINAL.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    data["execution_facts"]["aircraft_execution"] = {}
    bad_file = tmp_path / "A_NOMINAL_no_aircraft.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    res = verify_single_trajectory(
        traj_path=bad_file,
        baseline_h0=h0,
        baseline_tasks=tasks,
        scenario_spec={"scenario_id": "NOMINAL"},
    )
    assert res["status"] == "FAIL"
    assert len(res["violations"]["aircraft_set_mismatch"]) > 0


def test_loophole_3_missing_cycle_records_rejected(baseline_data, tmp_path: Path):
    """反例 3: 删除全部周期记录，核验器必须拦截并判定 FAIL。"""
    h0, tasks, _, _ = baseline_data
    with (TRAJ_DIR / "A_NOMINAL.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    data["execution_facts"]["cycle_execution_records"] = {}
    bad_file = tmp_path / "A_NOMINAL_no_cycles.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    res = verify_single_trajectory(
        traj_path=bad_file,
        baseline_h0=h0,
        baseline_tasks=tasks,
        scenario_spec={"scenario_id": "NOMINAL"},
    )
    assert res["status"] == "FAIL"
    assert len(res["violations"]["cycle_set_mismatch"]) > 0


def test_loophole_4_tampered_step_rewards_rejected(baseline_data, tmp_path: Path):
    """反例 4: 将单步奖励列表篡改为 [123.0]，核验器必须拦截并判定 FAIL。"""
    h0, tasks, _, _ = baseline_data
    with (TRAJ_DIR / "A_NOMINAL.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    data["execution_facts"]["step_rewards"] = [123.0]
    bad_file = tmp_path / "A_NOMINAL_bad_rewards.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    res = verify_single_trajectory(
        traj_path=bad_file,
        baseline_h0=h0,
        baseline_tasks=tasks,
        scenario_spec={"scenario_id": "NOMINAL"},
    )
    assert res["status"] == "FAIL"
    assert len(res["violations"]["step_rewards_violation"]) > 0


def test_loophole_5_nan_timestamp_rejected(baseline_data, tmp_path: Path):
    """反例 5: 某工序开工时间注入 NaN，核验器必须拦截并判定 FAIL。"""
    h0, tasks, _, _ = baseline_data
    with (TRAJ_DIR / "A_NOMINAL.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    data["execution_facts"]["tasks_execution"]["0_2"]["actual_start"] = None
    bad_file = tmp_path / "A_NOMINAL_nan_time.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    res = verify_single_trajectory(
        traj_path=bad_file,
        baseline_h0=h0,
        baseline_tasks=tasks,
        scenario_spec={"scenario_id": "NOMINAL"},
    )
    assert res["status"] == "FAIL"
    assert len(res["violations"]["non_finite_value"]) > 0


def test_loophole_6_fake_revision_cost_rejected(baseline_data, tmp_path: Path):
    """反例 6: 给无修订任务凭空添加 revision_cost=1 并篡改总额，核验器必须拦截并判定 FAIL。"""
    h0, tasks, _, _ = baseline_data
    with (TRAJ_DIR / "A_NOMINAL.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 凭空注入未发生物理变化的假修订记录
    data["execution_facts"]["tasks_execution"]["0_2"]["revision_history"] = [
        {"before": {"position": 0.0, "team": [12, 72]}, "after": {"position": 0.0, "team": [12, 72]}, "revision_cost": 1.0}
    ]
    data["j_revision"] = 1.0
    data["objective_j_total"] = float(data.get("objective_j_total", 0.0)) + 1.0
    bad_file = tmp_path / "A_NOMINAL_fake_revision.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    res = verify_single_trajectory(
        traj_path=bad_file,
        baseline_h0=h0,
        baseline_tasks=tasks,
        scenario_spec={"scenario_id": "NOMINAL"},
    )
    assert res["status"] == "FAIL"
    # 重算 revision 仍为 0，与 reported 1.0 产生 diff > cost_tolerance
    assert len(res["violations"]["cost_reconciliation_mismatch"]) > 0


def test_loophole_7_disguised_nominal_as_disturbance_rejected(baseline_data, tmp_path: Path):
    """反例 7: 将 Nominal 事实伪装成强扰动场景并宣称命中，核验器必须因开工时刻低于恢复下界而拒绝。"""
    h0, tasks, _, scenarios = baseline_data
    with (TRAJ_DIR / "A_NOMINAL.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 伪装 scenario_id 为强扰动且宣称命中 1 次
    data["scenario_id"] = "RULE_VAL_S2_STRONG"
    data["actual_disturbance_hits"] = 1
    bad_file = tmp_path / "A_VAL_S2_disguised.json"
    bad_file.write_text(json.dumps(data), encoding="utf-8")

    val_s2_spec = scenarios["RULE_VAL_S2_STRONG"]
    res = verify_single_trajectory(
        traj_path=bad_file,
        baseline_h0=h0,
        baseline_tasks=tasks,
        scenario_spec=val_s2_spec,
    )
    assert res["status"] == "FAIL"
    assert len(res["violations"]["scenario_binding_violation"]) > 0


def test_loophole_8_missing_files_in_manifest_not_pass(baseline_data, tmp_path: Path):
    """反例 8: 目录下只提供 1 份文件 (A_NOMINAL.json)，核验器必须报 INCOMPLETE，退出码为 3，绝不能 PASS。"""
    shutil.copy2(TRAJ_DIR / "A_NOMINAL.json", tmp_path / "A_NOMINAL.json")

    summary = verify_all_trajectories(
        traj_dir=tmp_path,
        baseline_path=BASELINE_PATH,
        scenarios_path=SCENARIOS_PATH,
    )
    assert summary["overall_status"] == "INCOMPLETE"
    assert summary["exit_code"] == 3
    assert summary["all_pass"] is False
    assert len(summary["missing_manifest_files"]) == 5


def test_postpone_penalty_formula_progressive():
    """验证后移费用的累进惩罚公式 g(n) = lambda_1 * n + lambda_2 * max(0, n - 1)。"""
    # 单次后移: 0.10 * 1 = 0.10
    assert math.isclose(calculate_postpone_penalty(1, lambda_1=0.10, lambda_2=0.20), 0.10, abs_tol=1e-9)
    # 两次后移: 0.10 * 2 + 0.20 * 1 = 0.40
    assert math.isclose(calculate_postpone_penalty(2, lambda_1=0.10, lambda_2=0.20), 0.40, abs_tol=1e-9)
    # 三次后移: 0.10 * 3 + 0.20 * 2 = 0.70
    assert math.isclose(calculate_postpone_penalty(3, lambda_1=0.10, lambda_2=0.20), 0.70, abs_tol=1e-9)
