"""BC 初始化后 PPO 保持性诊断的针对性验证（本轮第 9 节 1–6 项）。"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.work3.actor_critic import ActorCriticWork3
from scripts.work3.evaluate_c_vs_d import build_formal_evaluation_agent
from training.work3_lightning import Work3LightningModule
import work3_4060.run_bc_ppo_retention as diag
from work3_4060.c_reproduction_protocol import load_diagnostic_actor, sha256_file


# 1. 受控初始化：来源绑定严格，诊断格式不走正式加载旁路


def test_bc_initialization_is_strictly_bound(tmp_path: Path) -> None:
    if not diag.BC_WEIGHTS.is_file():
        pytest.skip("本机缺少 BC 诊断副本")
    previous = json.loads(diag.BC_MANIFEST.read_text(encoding="utf-8"))
    model, payload = load_diagnostic_actor(
        diag.BC_WEIGHTS,
        expected_teacher_manifest_sha256=previous["teacher"]["manifest_sha256"],
    )
    assert payload["diagnostic_format"] == "diagnostic_behavior_clone"
    assert diag.actor_fingerprint(model) == previous["diagnostic_checkpoint"]["actor_fingerprint"]

    with pytest.raises(ValueError):
        load_diagnostic_actor(diag.BC_WEIGHTS, expected_teacher_manifest_sha256="0" * 64)
    with pytest.raises(ValueError):
        build_formal_evaluation_agent("C", diag.BC_WEIGHTS, device="cpu")

    fake = tmp_path / "fake.pt"
    torch.save({"actor_critic_state": {}}, fake)
    with pytest.raises(ValueError):
        diag.load_checkpoint_actor(fake)


def test_round_checkpoint_loader_rejects_foreign_format(tmp_path: Path) -> None:
    fake = tmp_path / "run_checkpoint.pt"
    torch.save({"diagnostic_format": "something_else", "diagnostic_format_version": 1}, fake)
    with pytest.raises(ValueError):
        diag.load_checkpoint_actor(fake)


# 2. 原 BC 不被修改；参数组完整；critic 可训练；优化器从零开始


def test_source_checkpoint_untouched_and_parameter_groups_complete() -> None:
    if not diag.BC_WEIGHTS.is_file():
        pytest.skip("本机缺少 BC 诊断副本")
    before = sha256_file(diag.BC_WEIGHTS)
    model, _payload = load_diagnostic_actor(
        diag.BC_WEIGHTS,
        expected_teacher_manifest_sha256=json.loads(
            diag.BC_MANIFEST.read_text(encoding="utf-8")
        )["teacher"]["manifest_sha256"],
    )
    groups = diag.parameter_groups(model)

    assert groups["counts"] == {"shared": 102, "policy": 32, "value": 4}
    assert groups["total"] == len(list(model.named_parameters())) == 138
    all_names = [name for names in groups["groups"].values() for name in names]
    assert len(all_names) == len(set(all_names)), "参数组之间不得重复"
    assert set(all_names) == {name for name, _ in model.named_parameters()}, "参数组必须全覆盖"
    assert all(name.startswith("critic.") for name in groups["groups"]["value"])
    assert all(parameter.requires_grad for parameter in model.parameters())

    lightning = Work3LightningModule(
        actor_critic=model, learning_rate=3e-4, clip_eps=0.2, vf_coef=0.5,
        ent_coef=0.01, max_grad_norm=0.5, ppo_epochs=1, batch_size=2,
    )
    optimizer = lightning.configure_optimizers()
    optimizer_ids = {id(p) for p in lightning.objective.optimized_parameters}
    assert optimizer_ids == {id(p) for p in model.parameters()}, "优化器参数集合必须完整"
    assert all(not state for state in optimizer.state.values()), "新建优化器不得携带历史状态"
    assert sha256_file(diag.BC_WEIGHTS) == before, "源文件不得被改写"


# 3. Nominal-only、随机采样训练、greedy 评估、无对齐强制


def test_nominal_only_and_no_alignment_wrapper() -> None:
    assert not hasattr(diag, "AlignForcedAgent"), "本轮不得复用对齐强制包装器"
    env_source = inspect.getsource(diag.build_env)
    assert "scenario" not in env_source, "训练环境不得携带任何扰动场景"
    train_source = inspect.getsource(diag.stage_train)
    assert "deterministic=False" in train_source, "训练必须是随机采样"
    evaluate_source = inspect.getsource(diag.stage_evaluate_one)
    assert "FormalEvaluationAgent" in evaluate_source, "评估必须走正式贪心入口"
    assert "scenario=None" in evaluate_source, "评估只能是 Nominal"


def test_recorded_config_is_nominal_only() -> None:
    if not diag.RESOLVED_CONFIG.is_file():
        pytest.skip("缺少 resolved_config.yaml")
    text = diag.RESOLVED_CONFIG.read_text(encoding="utf-8")
    assert "disturbance_enabled: false" in text
    assert "scenario: NOMINAL" in text
    assert "warmup_mode: none" in text


# 4. 重放合同：同一自回归事件、容差内、非有限数阻塞


def test_replay_contract_and_non_finite_blocking() -> None:
    assert diag.replay_contract_ok(3.0e-5) is True
    assert diag.replay_contract_ok(diag.REPLAY_TOLERANCE) is True
    assert diag.replay_contract_ok(1e-3) is False
    assert diag.replay_contract_ok(float("nan")) is False
    assert diag.replay_contract_ok(float("inf")) is False
    assert diag.replay_contract_ok(None) is False


def test_recorded_history_respects_replay_contract() -> None:
    if not diag.TRAINING_HISTORY.is_file():
        pytest.skip("缺少训练历史")
    rows = [json.loads(line) for line in diag.TRAINING_HISTORY.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "训练历史为空"
    for row in rows:
        assert diag.replay_contract_ok(row["replay_max_abs_error"]), row["replay_max_abs_error"]
        assert row["advantage_raw_stats"]["all_finite"]
        assert row["td_residual_stats"]["all_finite"]


# 5. 评估隔离与终止/截断区分


def test_evaluation_runs_in_fresh_processes() -> None:
    source = inspect.getsource(diag.stage_evaluate)
    assert "subprocess.run" in source, "每个评估点必须在独立子进程执行"
    assert "evaluate-one" in source


def test_truncation_differs_from_natural_termination() -> None:
    completed = diag.build_episode_cost_record(
        episode_id=0,
        info={"cumulative_cost": 0.5, "cost_breakdown": {"cost_takt": 0.5}},
        termination_reason="completed",
        episode_steps=210,
        raw_reward_sum=-0.5,
    )
    truncated = diag.build_episode_cost_record(
        episode_id=1,
        info={"cumulative_cost": 0.2, "cost_breakdown": {"cost_takt": 0.1, "cost_time": 0.1}},
        termination_reason="budget_truncation",
        episode_steps=92,
        raw_reward_sum=-0.2,
    )
    assert completed["success"] is True
    assert truncated["success"] is False
    assert completed["termination_reason"] == "completed"
    assert truncated["termination_reason"] == "budget_truncation"
    # 五分项必须来自环境累计计数器，不能被重新分摊成 0
    assert completed["cost_takt"] == pytest.approx(0.5)
    assert truncated["cost_time"] == pytest.approx(0.1)


def test_recorded_segments_mark_truncation_explicitly() -> None:
    if not diag.TRAINING_HISTORY.is_file():
        pytest.skip("缺少训练历史")
    rows = [json.loads(line) for line in diag.TRAINING_HISTORY.read_text(encoding="utf-8").splitlines() if line.strip()]
    boundary_types = {row["boundary_type"] for row in rows}
    assert boundary_types <= {"rollout_continuation", "budget_truncation", "natural_termination"}
    last = rows[-1]
    assert last["boundary_type"] == "budget_truncation"
    assert last["terminated"] is False


# 6. 协议状态、保持标准与动态计算


def test_protocol_status_precedence() -> None:
    expected = [{"path": "a"}, {"path": "b"}]
    assert diag.classify_protocol_status(expected, [{"path": "a", "status": "PASS"}, {"path": "b", "status": "PASS"}]) == "PASS"
    assert diag.classify_protocol_status(expected, [{"path": "a", "status": "PASS"}]) == "INCOMPLETE"
    assert diag.classify_protocol_status(expected, [{"path": "a", "status": "FAIL"}]) == "FAIL"


def test_retention_thresholds_are_exact() -> None:
    def payload(j_total, j_takt, postpone=0, violations=0, success=True):
        return {
            "success": success,
            "completed_tasks": 210,
            "constraint_violation_count": violations,
            "objective_j_total": j_total,
            "postponed_count": postpone,
            "cost_components_weighted": {"j_takt": j_takt},
        }

    assert diag.retention_pass(payload(1e-3, 1e-6)) is True
    assert diag.retention_pass(payload(1.0001e-3, 0.0)) is False
    assert diag.retention_pass(payload(0.0, 1.01e-6)) is False
    assert diag.retention_pass(payload(0.0, 0.0, postpone=1)) is False
    assert diag.retention_pass(payload(0.0, 0.0, violations=1)) is False
    assert diag.retention_pass(payload(0.0, 0.0, success=False)) is False


def test_statistics_are_computed_from_values() -> None:
    result = diag.stats([1.0, 2.0, 3.0])
    assert result["count"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert result["min"] == 1.0 and result["max"] == 3.0
    assert diag.stats([])["count"] == 0
    assert diag.stats([1.0, float("inf")])["all_finite"] is False


def test_evaluation_rows_are_dynamic_and_signed() -> None:
    """ΔJ 是“相对 BC 起点”的带符号差值（正值=更差），必须由原始事实算出。"""
    if not all(path.is_file() for path in (diag.TRAJ_DIR / "step0000_NOMINAL.json", diag.TRAJ_DIR / "step0512_NOMINAL.json")):
        pytest.skip("缺少本轮评估轨迹")
    baseline_bc = json.loads(diag.BC_NOMINAL_TRAJ.read_text(encoding="utf-8"))
    evaluations = []
    for step in (0, 512):
        path = diag.TRAJ_DIR / f"step{step:04d}_NOMINAL.json"
        evaluations.append({"step": step, "payload": json.loads(path.read_text(encoding="utf-8")), "teacher": None})
    rows = diag.build_evaluation_rows(evaluations, baseline_bc, float(json.loads(diag.BASELINE_PATH.read_text(encoding="utf-8"))["metadata"]["h0"]))
    zero = next(row for row in rows if row["step"] == 0)
    train = next(row for row in rows if row["step"] == 512)
    assert zero["j_total"] == pytest.approx(float(baseline_bc["objective_j_total"]), abs=1e-12)
    assert zero["delta_j_vs_bc"] == pytest.approx(0.0, abs=1e-12)
    assert train["delta_j_vs_bc"] == pytest.approx(
        train["j_total"] - float(baseline_bc["objective_j_total"]), abs=1e-12
    )
    assert train["retention_pass"] is False
