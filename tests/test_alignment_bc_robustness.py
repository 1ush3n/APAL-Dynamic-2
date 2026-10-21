"""对齐单变量干预与冻结 BC 受扰诊断的针对性验证（本轮第 7 节）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import ActionBranch
from envs.work3.environment import AirLineEnvWork3
from scripts.work3.evaluate_c_vs_d import FormalEvaluationAgent, build_formal_evaluation_agent
import work3_4060.run_alignment_bc_robustness as diag
from work3_4060.c_reproduction_protocol import load_diagnostic_actor

BASELINE = diag.BASELINE_PATH


def _advance_action() -> dict:
    return {"branch": int(ActionBranch.ADVANCE_TO_NEXT_EVENT)}


def _postpone_action() -> dict:
    return {"branch": int(ActionBranch.POSTPONE), "task_key": "0_2"}


def _stay_action(task_key: str = "0_2", team=(12, 72), align: int = 0) -> dict:
    return {
        "branch": int(ActionBranch.STATION_EXECUTE),
        "task_key": task_key,
        "team": list(team),
        "align": align,
    }


# 1. 干预只改变合法执行动作的 align；原字典不被原地污染


def test_intervention_only_changes_align_and_keeps_other_fields() -> None:
    original = _stay_action(task_key="1_4", team=(12, 72), align=0)
    snapshot = dict(original)
    forced, meta = diag.apply_align_intervention(original)

    assert forced is not None
    assert forced["align"] == 1
    assert forced["task_key"] == original["task_key"]
    assert forced["team"] == original["team"]
    assert forced["branch"] == original["branch"]
    assert original == snapshot, "入参动作字典不得被原地修改"
    assert forced is not original
    assert meta["changed"] is True and meta["align_before"] == 0


def test_intervention_noop_when_already_aligned() -> None:
    action = _stay_action(align=1)
    forced, meta = diag.apply_align_intervention(action)
    assert forced is None
    assert meta["applicable"] is True
    assert meta["already_aligned"] is True
    assert meta["changed"] is False


# 2. ADVANCE / 后移不被改写；非法 align=1 显式阻塞


def test_advance_and_postpone_are_not_rewritten() -> None:
    for action, reason in (
        (_advance_action(), "advance_to_next_event"),
        (_postpone_action(), "postpone"),
    ):
        snapshot = dict(action)
        forced, meta = diag.apply_align_intervention(action)
        assert forced is None
        assert meta["applicable"] is False
        assert meta["reason"] == reason
        assert action == snapshot


def test_illegal_align1_is_blocked_not_silently_allowed() -> None:
    env = AirLineEnvWork3(baseline_json_path=BASELINE)
    env.reset()
    candidates = env.get_action_candidates()
    assert candidates, "reset 之后应存在候选工序"
    task = candidates[0]
    valid_team = tuple(env.valid_team_completion_workers(task, []))[: task.demand]
    assert len(valid_team) == task.demand

    ok_action = {
        "branch": int(ActionBranch.STATION_EXECUTE),
        "task_key": task.task_key,
        "team": list(valid_team),
        "align": 1,
    }
    checked = diag.validate_aligned_action(env, ok_action)
    assert checked["checked"] is True
    assert checked["team_size"] == checked["demand"]

    # align 未被改写为 1 → 阻塞
    with pytest.raises(diag.AlignInterventionBlocked):
        diag.validate_aligned_action(env, {**ok_action, "align": 0})
    # 推进分支不得进入对齐校验
    with pytest.raises(diag.AlignInterventionBlocked):
        diag.validate_aligned_action(env, {"branch": int(ActionBranch.ADVANCE_TO_NEXT_EVENT)})
    # 团队人数不足 → 阻塞
    with pytest.raises(diag.AlignInterventionBlocked):
        diag.validate_aligned_action(env, {**ok_action, "team": list(valid_team)[: task.demand - 1]})
    # 不存在的工序 → 阻塞
    with pytest.raises(diag.AlignInterventionBlocked):
        diag.validate_aligned_action(env, {**ok_action, "task_key": "999_999"})


# 3. 普通 BC 评测不使用对齐包装器；原 C 正式入口默认行为不变


def test_bc_agent_is_plain_and_original_entry_is_unwrapped() -> None:
    if not diag.BC_WEIGHTS.is_file():
        pytest.skip("本机缺少 BC 诊断副本")
    bc_agent, _checkpoint, _model = diag.load_bc_agent()
    assert isinstance(bc_agent, FormalEvaluationAgent)
    assert not isinstance(bc_agent, diag.AlignForcedAgent)

    if diag.ORIGINAL_C_WEIGHTS.is_file():
        original_agent = build_formal_evaluation_agent(
            "C", diag.ORIGINAL_C_WEIGHTS, device="cpu"
        )
        assert type(original_agent) is FormalEvaluationAgent
        assert not isinstance(original_agent, diag.AlignForcedAgent)


# 4. 指纹/来源不符、v1/v2 场景混用、缺少检查点被拒绝


def test_diagnostic_loader_rejects_wrong_teacher_binding(tmp_path: Path) -> None:
    if not diag.BC_WEIGHTS.is_file():
        pytest.skip("本机缺少 BC 诊断副本")
    with pytest.raises(ValueError):
        load_diagnostic_actor(diag.BC_WEIGHTS, expected_teacher_manifest_sha256="0" * 64)
    with pytest.raises(FileNotFoundError):
        load_diagnostic_actor(tmp_path / "missing.pt")


def test_scenario_v1_v2_mixing_is_rejected() -> None:
    scenarios = diag.load_scenarios()
    good = scenarios["RULE_VAL_S1_MODERATE"]
    assert diag.require_v1_scenario(good, "RULE_VAL_S1_MODERATE")["scenario_id"] == good["scenario_id"]

    # 旧口径标签（v2/legacy 命名）必须被拒绝
    legacy = {**good, "timing_reference": "nominal_anchored_v1_relative"}
    with pytest.raises(ValueError):
        diag.require_v1_scenario(legacy, "RULE_VAL_S1_MODERATE")
    # 场景 id 与期望不符必须被拒绝
    with pytest.raises(ValueError):
        diag.require_v1_scenario(good, "RULE_VAL_S2_STRONG")
    # 缺字段必须被拒绝
    broken = dict(good)
    broken.pop("recovery_time")
    with pytest.raises(ValueError):
        diag.require_v1_scenario(broken, "RULE_VAL_S1_MODERATE")


# 5. 清单缺轨迹 → INCOMPLETE；损坏/不合法 → FAIL；效果恶化不被伪装成缺项


def test_protocol_status_precedence() -> None:
    expected = [
        {"path": "a", "method": "M", "scenario_id": "NOMINAL"},
        {"path": "b", "method": "M", "scenario_id": "S1"},
    ]
    passing = [{"path": "a", "status": "PASS"}, {"path": "b", "status": "PASS"}]
    assert diag.classify_protocol_status(expected, passing) == "PASS"

    missing = [{"path": "a", "status": "PASS"}]
    assert diag.classify_protocol_status(expected, missing) == "INCOMPLETE"

    failing = [{"path": "a", "status": "FAIL"}, {"path": "b", "status": "PASS"}]
    assert diag.classify_protocol_status(expected, failing) == "FAIL"

    both = [{"path": "a", "status": "FAIL"}]
    assert diag.classify_protocol_status(expected, both) == "FAIL", "FAIL 优先于 INCOMPLETE"

    assert diag.protocol_exit_code("PASS") == 0
    assert diag.protocol_exit_code("FAIL") == 2
    assert diag.protocol_exit_code("INCOMPLETE") == 3


def test_effect_worsening_is_not_a_protocol_gap() -> None:
    assert diag.classify_effect(-0.18) == "费用恶化"
    assert diag.classify_effect(0.0) == "近似持平"
    assert diag.classify_effect(0.5) == "费用改善"
    # 效果判定与协议状态彼此独立：恶化不影响 PASS/INCOMPLETE 的判定
    expected = [{"path": "a", "method": "M", "scenario_id": "NOMINAL"}]
    assert diag.classify_protocol_status(expected, [{"path": "a", "status": "PASS"}]) == "PASS"
    assert diag.classify_effect(-1.0) == "费用恶化"


# 6. 差值方向、Nominal 增量、近零分母与逐周期汇总


def test_delta_direction_and_near_zero_denominator() -> None:
    assert diag.delta(10.0, 7.0) == 3.0, "正值表示新值更低（更好）"
    assert diag.delta(7.0, 10.0) == -3.0
    assert diag.relative_reduction(0.0, 0.5) is None, "近零分母必须返回 None（报告写 N/A）"
    assert diag.relative_reduction(1e-12, 0.5) is None
    assert diag.relative_reduction(4.0, 3.0) == pytest.approx(0.25)
    assert diag.relative_reduction(4.0, 5.0) == pytest.approx(-0.25)
    assert diag.format_relative(None) == "N/A"


def test_cycle_summary_is_computed_from_rows() -> None:
    rows = [
        {"cycle": 1, "delay_hours": 0.0},
        {"cycle": 2, "delay_hours": 0.0},
        {"cycle": 3, "delay_hours": 1.5},
        {"cycle": 4, "delay_hours": 2.5},
    ]
    summary = diag.summarize_cycle_delay(rows)
    assert summary["delayed_cycle_count"] == 2
    assert summary["max_cycle_delay"] == pytest.approx(2.5)
    assert summary["sum_cycle_delay"] == pytest.approx(4.0)
    assert summary["first_delayed_cycle"] == 3
    assert summary["cycle_count"] == 4


def test_nominal_and_disturbance_tables_are_derived_from_facts() -> None:
    """汇总必须由原始轨迹动态计算，并与直接重算一致。"""
    required = [diag.ALIGN_TRAJ, diag.BC_S1_TRAJ, diag.BC_S2_TRAJ, diag.PREVIOUS_C_NOMINAL]
    if not all(path.is_file() for path in required):
        pytest.skip("本轮产物或上一轮对照缺失，跳过动态汇总核对")
    from work3_4060.c_reproduction_protocol import load_json

    align = load_json(diag.ALIGN_TRAJ)
    original = load_json(diag.PREVIOUS_C_NOMINAL)
    rows = diag.build_nominal_rows(
        [
            ("C_ORIGINAL", diag.ORIGINAL_METHOD, "原 C", original),
            (diag.ALIGN_METHOD, diag.ALIGN_METHOD, "对齐干预", align),
        ],
        diag.baseline_index(diag.baseline_bundle()[1]),
    )
    reference = next(row for row in rows if row["run_id"] == "C_ORIGINAL")
    forced = next(row for row in rows if row["run_id"] == diag.ALIGN_METHOD)
    assert reference["j_total"] == pytest.approx(float(original["objective_j_total"]))
    assert forced["j_total"] == pytest.approx(float(align["objective_j_total"]))
    assert forced["delta_j_vs_original_c"] == pytest.approx(
        reference["j_total"] - forced["j_total"]
    )
    assert forced["relative_j_reduction_vs_original_c"] == pytest.approx(
        (reference["j_total"] - forced["j_total"]) / reference["j_total"]
    )
