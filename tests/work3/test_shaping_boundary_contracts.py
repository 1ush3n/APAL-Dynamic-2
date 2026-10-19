"""单元验收测试：核验 verify_shaping_trace.py 的段尾边界与账本全覆盖语义契约 (Unit B)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_checker(events_file: Path, run_json: Path | None = None) -> tuple[int, dict]:
    output_summary = events_file.parent / "summary.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "work3_4060/verify_shaping_trace.py"),
        "--events",
        str(events_file),
        "--output",
        str(output_summary),
    ]
    if run_json:
        cmd.extend(["--run-json", str(run_json)])

    sub_env = os.environ.copy()
    sub_env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=sub_env
    )
    summary = {}
    if output_summary.is_file():
        summary = json.loads(output_summary.read_text(encoding="utf-8"))
    return res.returncode, summary


@pytest.fixture
def base_scenario(tmp_path: Path):
    """构造一个两步完工的标准正确基准事件流及对应 run.json。"""
    # Step 0: worker 0, episode 0, rollout 0, step 0
    # raw_reward = -0.05, phi_curr = -0.1, phi_next = -0.08, beta=0.1, gamma=0.99
    # shaped = -0.05 + 0.1 * (0.99 * (-0.08) - (-0.1)) = -0.05 + 0.1 * (-0.0792 + 0.1) = -0.05 + 0.00208 = -0.04792
    ev0 = {
        "record_type": "transition",
        "run_id": "test_run",
        "worker_id": 0,
        "episode_id": 0,
        "rollout_id": 0,
        "episode_step": 0,
        "raw_reward": -0.05,
        "shaped_reward": -0.05 + 0.1 * (0.99 * (-0.08) - (-0.10)),
        "phi_current": -0.10,
        "phi_next": -0.08,
        "beta": 0.1,
        "gamma": 0.99,
        "terminated": False,
        "truncated": False,
        "predictor_version": 0,
        "phi_source": "learned_phi",
    }
    # Step 1: worker 0, episode 0, rollout 0, step 1 (natural termination)
    # raw_reward = -0.05, phi_curr = -0.08, phi_next = 0.0 (terminal)
    # shaped = -0.05 + 0.1 * (0.99 * 0.0 - (-0.08)) = -0.05 + 0.008 = -0.042
    ev1 = {
        "record_type": "transition",
        "run_id": "test_run",
        "worker_id": 0,
        "episode_id": 0,
        "rollout_id": 0,
        "episode_step": 1,
        "raw_reward": -0.05,
        "shaped_reward": -0.05 + 0.1 * (0.99 * 0.0 - (-0.08)),
        "phi_current": -0.08,
        "phi_next": 0.0,
        "beta": 0.1,
        "gamma": 0.99,
        "terminated": True,
        "truncated": False,
        "predictor_version": 0,
        "phi_source": "learned_phi",
    }
    seg0 = {
        "record_type": "segment_end",
        "run_id": "test_run",
        "worker_id": 0,
        "episode_id": 0,
        "rollout_id": 0,
        "bootstrap_value": 0.0,
        "boundary_type": "natural_termination",
        "termination_reason": "completed",
    }
    ledger0 = {
        "record_type": "episode_cost_summary",
        "run_id": "test_run",
        "worker_id": 0,
        "episode_id": 0,
        "cumulative_cost": 0.10,
        "cost_takt": 0.0,
        "cost_time": 0.05,
        "cost_team": 0.05,
        "cost_postpone": 0.0,
        "cost_revision": 0.0,
    }
    run_json_content = {
        "training_config": {"gamma": 0.99, "beta_shaping": 0.1},
        "resolved_runtime_config_yaml": "ppo:\n  gamma: 0.99\n  shaping_coefficient: 0.1\n",
    }
    return [ev0, ev1, seg0, ledger0], run_json_content


def test_baseline_valid_trace_passes(tmp_path: Path, base_scenario):
    """用例 1: 完整、正确的基准事件流 -> PASS / exit 0。"""
    events, run_cfg = base_scenario
    ev_file = tmp_path / "valid_events.jsonl"
    ev_file.write_text("\n".join(json.dumps(x) for x in events) + "\n", encoding="utf-8")
    rj_file = tmp_path / "run.json"
    rj_file.write_text(json.dumps(run_cfg), encoding="utf-8")

    code, summary = run_checker(ev_file, rj_file)
    assert code == 0
    assert summary.get("overall_status") == "PASS"
    assert summary["contracts"]["boundary_contract"] == "PASS"
    assert summary["contracts"]["ledger_reconciliation"] == "PASS"


def test_terminated_step_misclassified_as_continuation_fails(tmp_path: Path, base_scenario):
    """用例 2 (B1): terminated=True，却将该段标为 rollout_continuation 并给非零 bootstrap -> FAIL / exit 2。"""
    events, run_cfg = base_scenario
    # 破坏 seg0
    bad_events = list(events)
    bad_seg = {
        "record_type": "segment_end",
        "run_id": "test_run",
        "worker_id": 0,
        "episode_id": 0,
        "rollout_id": 0,
        "bootstrap_value": 0.85,  # 错误非零 bootstrap
        "boundary_type": "rollout_continuation",  # 错误分类为 continuation
        "termination_reason": "none",
    }
    bad_events[2] = bad_seg

    ev_file = tmp_path / "conflict_boundary.jsonl"
    ev_file.write_text("\n".join(json.dumps(x) for x in bad_events) + "\n", encoding="utf-8")
    rj_file = tmp_path / "run.json"
    rj_file.write_text(json.dumps(run_cfg), encoding="utf-8")

    code, summary = run_checker(ev_file, rj_file)
    assert code == 2
    assert summary.get("overall_status") == "FAIL"
    assert summary["contracts"]["boundary_contract"] == "FAIL"


def test_truncated_episode_missing_ledger_is_incomplete(tmp_path: Path, base_scenario):
    """用例 3 (B2): 已有完整 episode，另一个预算截断 episode 缺少账本 -> INCOMPLETE / exit 3。"""
    events, run_cfg = base_scenario
    # 增加第二个未完工的截断 episode: worker 0, episode 1, rollout 1
    ev_trunc = {
        "record_type": "transition",
        "run_id": "test_run",
        "worker_id": 0,
        "episode_id": 1,
        "rollout_id": 1,
        "episode_step": 0,
        "raw_reward": -0.05,
        "shaped_reward": -0.05 + 0.1 * (0.99 * (-0.05) - (-0.02)),
        "phi_current": -0.02,
        "phi_next": -0.05,
        "beta": 0.1,
        "gamma": 0.99,
        "terminated": False,
        "truncated": True,
        "predictor_version": 0,
        "phi_source": "learned_phi",
    }
    seg_trunc = {
        "record_type": "segment_end",
        "run_id": "test_run",
        "worker_id": 0,
        "episode_id": 1,
        "rollout_id": 1,
        "bootstrap_value": 0.12,
        "boundary_type": "budget_truncation",
        "termination_reason": "max_decisions_reached",
    }
    # 故意不写入 episode 1 的 episode_cost_summary
    augmented_events = list(events) + [ev_trunc, seg_trunc]

    ev_file = tmp_path / "truncated_missing_ledger.jsonl"
    ev_file.write_text("\n".join(json.dumps(x) for x in augmented_events) + "\n", encoding="utf-8")
    rj_file = tmp_path / "run.json"
    rj_file.write_text(json.dumps(run_cfg), encoding="utf-8")

    code, summary = run_checker(ev_file, rj_file)
    assert code == 3
    assert summary.get("overall_status") == "INCOMPLETE"
    assert summary["contracts"]["ledger_reconciliation"] == "INCOMPLETE"


def test_previous_adversarial_counter_examples_no_regression(tmp_path: Path, base_scenario):
    """用例 4: 上一轮缺失段尾、损坏 YAML、非有限数等已有反例保持对应 FAIL 或 INCOMPLETE，不发生回归。"""
    events, _ = base_scenario
    ev_file = tmp_path / "broken_yaml.jsonl"
    ev_file.write_text("\n".join(json.dumps(x) for x in events) + "\n", encoding="utf-8")

    # 4.1 损坏 YAML -> FAIL
    bad_rj = tmp_path / "broken_run.json"
    bad_rj.write_text(
        json.dumps({
            "training_config": {"gamma": 0.99, "beta_shaping": 0.1},
            "resolved_runtime_config_yaml": "ppo: [invalid syntax",
        }),
        encoding="utf-8",
    )
    code_yaml, sum_yaml = run_checker(ev_file, bad_rj)
    assert code_yaml == 2
    assert sum_yaml["contracts"]["config_consistency"] == "FAIL"

    # 4.2 缺失段尾 -> INCOMPLETE
    no_seg_events = [x for x in events if x.get("record_type") != "segment_end"]
    no_seg_file = tmp_path / "no_seg.jsonl"
    no_seg_file.write_text("\n".join(json.dumps(x) for x in no_seg_events) + "\n", encoding="utf-8")
    good_rj = tmp_path / "good_run.json"
    good_rj.write_text(
        json.dumps({
            "training_config": {"gamma": 0.99, "beta_shaping": 0.1},
            "resolved_runtime_config_yaml": "ppo:\n  gamma: 0.99\n  shaping_coefficient: 0.1\n",
        }),
        encoding="utf-8",
    )
    code_seg, sum_seg = run_checker(no_seg_file, good_rj)
    assert code_seg == 3
    assert sum_seg["contracts"]["boundary_contract"] == "INCOMPLETE"
