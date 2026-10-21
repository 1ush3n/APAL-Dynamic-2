"""Work-3 C 对齐单变量干预与冻结 BC 副本受扰诊断。

本轮只回答两个问题，全部为 CPU 推理，零参数更新：
1. 对原 C 只把执行分支的对齐选择改成 align=1，Nominal 排程改善多少；
2. 已复现 A Nominal 的 BC 诊断副本，在冻结状态下面对两个 v1 验证扰动能否继续合法出线。

阶段（每阶段独立子进程，保证建模前可显式设定 CPU 线程数）：

    inventory    权重/来源/历史对照清册
    align        对齐单变量干预一条完整 Nominal 轨迹
    disturbance  冻结 BC 副本两条完整受扰轨迹
    verify       严格轨迹事实核验、权重与来源完整性、协议状态
    report       CSV 与 summary.md
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3, NoFeasibleSlotError
from scripts.work3.evaluate_c_vs_d import (
    FormalEvaluationAgent,
    build_formal_evaluation_agent,
    evaluate_single_trajectory,
)
from scripts.work3.experiment_protocol import build_method_profile
from work3_4060.c_reproduction_protocol import (
    actor_fingerprint,
    compare_execution_facts,
    dump_json,
    dump_jsonl,
    enforce_single_thread_cpu,
    extract_execution_facts,
    load_diagnostic_actor,
    load_json,
    sha256_file,
    sha256_text,
)

BASE_DIR = REPO_ROOT / "work3_4060"
RULE_BASELINE_DIR = BASE_DIR / "rule_baseline_ac_v1"
PREVIOUS_DIR = BASE_DIR / "c_reproduction_diagnostic_v1" / "seed42"
OUTPUT_DIR = BASE_DIR / "c_alignment_bc_robustness_v1" / "seed42"

BASELINE_PATH = RULE_BASELINE_DIR / "mini210_rule_baseline.json"
SCENARIOS_PATH = RULE_BASELINE_DIR / "scenarios.json"
VALIDATION_SPLIT_PATH = RULE_BASELINE_DIR / "validation.json"

ORIGINAL_C_WEIGHTS = RULE_BASELINE_DIR / "seed42" / "C" / "weights.pt"
PREVIOUS_MANIFEST = PREVIOUS_DIR / "manifest.json"
BC_WEIGHTS = PREVIOUS_DIR / "diagnostic_actor.pt"

PREVIOUS_A_NOMINAL = RULE_BASELINE_DIR / "trajectories" / "A_NOMINAL.json"
PREVIOUS_A_S1 = RULE_BASELINE_DIR / "trajectories" / "A_RULE_VAL_S1_MODERATE.json"
PREVIOUS_A_S2 = RULE_BASELINE_DIR / "trajectories" / "A_RULE_VAL_S2_STRONG.json"
PREVIOUS_C_NOMINAL = PREVIOUS_DIR / "trajectories" / "C_ORIGINAL_NOMINAL.json"
PREVIOUS_BC_NOMINAL = PREVIOUS_DIR / "trajectories" / "C_BC_DIAGNOSTIC_NOMINAL.json"
PREVIOUS_C_S1 = RULE_BASELINE_DIR / "trajectories" / "C_RULE_VAL_S1_MODERATE.json"
PREVIOUS_C_S2 = RULE_BASELINE_DIR / "trajectories" / "C_RULE_VAL_S2_STRONG.json"

TRAJ_DIR = OUTPUT_DIR / "trajectories"
ALIGN_TRAJ = TRAJ_DIR / "C_ORIGINAL_ALIGN1_DIAGNOSTIC_NOMINAL.json"
BC_S1_TRAJ = TRAJ_DIR / "C_BC_RULE_VAL_S1_MODERATE.json"
BC_S2_TRAJ = TRAJ_DIR / "C_BC_RULE_VAL_S2_STRONG.json"

MANIFEST = OUTPUT_DIR / "manifest.json"
REUSE_INVENTORY = OUTPUT_DIR / "reuse_inventory.json"
INTERVENTION_EVENTS = OUTPUT_DIR / "intervention_events.jsonl"
TRAJECTORY_VERIFICATION = OUTPUT_DIR / "trajectory_verification.json"
WEIGHT_INTEGRITY = OUTPUT_DIR / "weight_integrity.json"
VERIFICATION = OUTPUT_DIR / "verification.json"
NOMINAL_COMPARISON = OUTPUT_DIR / "nominal_alignment_comparison.csv"
DISTURBANCE_COMPARISON = OUTPUT_DIR / "bc_disturbance_comparison.csv"
CYCLE_COMPARISON = OUTPUT_DIR / "cycle_comparison.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
RUNS_DIR = OUTPUT_DIR / "runs"
BLOCKED_INTERFACE = OUTPUT_DIR / "blocked_interface.json"
DISCARDED_RUNS = OUTPUT_DIR / "runs" / "discarded_runs.json"

STAGES: tuple[str, ...] = ("inventory", "align", "disturbance", "verify", "report")

DECISION_LIMIT = 4096
PHYSICAL_TOLERANCE = 1e-7
COST_TOLERANCE = 1e-9
NEAR_ZERO_DENOMINATOR = 1e-9

ALIGN_METHOD = "C_ORIGINAL_ALIGN1_DIAGNOSTIC"
BC_METHOD = "C_BC_DIAGNOSTIC"
ORIGINAL_METHOD = "C_ORIGINAL"
A_METHOD = "A_BASELINE_REPAIR"

EXPECTED_NEW_TRAJECTORIES: tuple[dict[str, str], ...] = (
    {
        "method": ALIGN_METHOD,
        "scenario_id": "NOMINAL",
        "path": str(ALIGN_TRAJ),
        "role": "对齐单变量干预（仅把执行分支 align 改为 1）",
    },
    {
        "method": BC_METHOD,
        "scenario_id": "RULE_VAL_S1_MODERATE",
        "path": str(BC_S1_TRAJ),
        "role": "冻结 BC 副本，v1 验证扰动 S1",
    },
    {
        "method": BC_METHOD,
        "scenario_id": "RULE_VAL_S2_STRONG",
        "path": str(BC_S2_TRAJ),
        "role": "冻结 BC 副本，v1 验证扰动 S2",
    },
)

VALIDATION_SCENARIO_IDS: tuple[str, ...] = ("RULE_VAL_S1_MODERATE", "RULE_VAL_S2_STRONG")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def format_float(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return format(float(value), ".17g")
    return str(value)


def baseline_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = load_json(BASELINE_PATH)
    return baseline, baseline["tasks"]


def baseline_index(tasks: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """把基准工序整理成本轮需要的只读索引（站位字段保留原始 1-based 语义）。"""
    index: dict[str, dict[str, Any]] = {}
    for key, task in tasks.items():
        index[str(key)] = {
            "baseline_station_field": int(task["station_id"]),
            "base_station": int(task["station_id"]) - 1,
            "in_station_offset": float(task["in_station_offset"]),
            "baseline_team": tuple(int(w) for w in task.get("team", ())),
            "demand": int(task.get("demand", len(task.get("team", ())))),
            "predecessors": tuple(int(p) for p in task.get("predecessors", ())),
            "duration": float(task.get("duration", 0.0)),
            "skill": int(task.get("skill", -1)),
        }
    return index


def git_state() -> dict[str, Any]:
    def _run(args: Sequence[str]) -> str | None:
        try:
            return subprocess.check_output(
                args, cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    head = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain=v1"])
    return {
        "head": head,
        "branch": branch,
        "worktree_dirty": bool(status),
        "worktree_entries": [] if not status else status.splitlines()[:40],
    }


def scenario_fingerprint(scenario: Mapping[str, Any]) -> str:
    return sha256_text(json.dumps(dict(scenario), sort_keys=True, ensure_ascii=False))


def load_scenarios() -> dict[str, dict[str, Any]]:
    payload = load_json(SCENARIOS_PATH)
    return {str(item["scenario_id"]): item for item in payload}


def require_v1_scenario(scenario: dict[str, Any], expected_id: str) -> dict[str, Any]:
    """严格绑定 v1 场景：id、timing_reference、目标工序必须与冻结文件一致。"""
    if str(scenario.get("scenario_id")) != expected_id:
        raise ValueError(
            f"场景绑定失败: 期望 {expected_id}, 实际 {scenario.get('scenario_id')}"
        )
    if scenario.get("timing_reference") != "nominal_entry_referenced_preentry_v1":
        raise ValueError(
            f"{expected_id} 的 timing_reference 不是 v1 口径: {scenario.get('timing_reference')!r}"
        )
    for key in ("aircraft_id", "station_id", "target_task_key", "tau", "recovery_time"):
        if key not in scenario:
            raise ValueError(f"{expected_id} 缺少必填字段 {key}")
    if float(scenario["recovery_time"]) <= float(scenario["tau"]):
        raise ValueError(f"{expected_id} 的恢复时刻不晚于事件时刻")
    return dict(scenario)


def decision_totals(payload: Mapping[str, Any]) -> dict[str, Any]:
    causes = payload.get("advance_cause_counts", {}) or {}
    return {
        "explicit_advances": int(causes.get("explicit", 0)),
        "forced_advances": int(causes.get("forced_no_candidate", 0)),
        "auto_advances": int(causes.get("auto_after_action", 0)),
        "identical_resubmissions": int(causes.get("identical_resubmission", 0)),
        "advance_total": int(sum(int(v) for v in causes.values())),
    }


def cycle_rows(payload: Mapping[str, Any], run_id: str, method: str) -> list[dict[str, Any]]:
    facts = payload["execution_facts"]
    records = facts.get("cycle_execution_records", {})
    rows: list[dict[str, Any]] = []
    for cycle_key in sorted(records, key=int):
        record = records[cycle_key]
        nominal = float(record["nominal_transfer_time"])
        actual = float(record["transfer_time"])
        rows.append(
            {
                "run_id": run_id,
                "method": method,
                "scenario_id": payload.get("scenario_id"),
                "cycle": int(record["cycle_id"]),
                "cycle_start_time": float(record["cycle_start_time"]),
                "nominal_transfer_time": nominal,
                "actual_transfer_time": actual,
                "delay_hours": actual - nominal,
            }
        )
    return rows


def summarize_cycle_delay(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    delays = [float(row["delay_hours"]) for row in rows]
    delayed = [row for row in rows if float(row["delay_hours"]) > PHYSICAL_TOLERANCE]
    return {
        "max_cycle_delay": max(delays) if delays else 0.0,
        "sum_cycle_delay": sum(delays),
        "delayed_cycle_count": len(delayed),
        "first_delayed_cycle": int(delayed[0]["cycle"]) if delayed else None,
        "cycle_count": len(delays),
    }


def relative_reduction(reference: float, new: float) -> float | None:
    """以 reference 为分母的相对降幅；近零分母返回 None（报告中写 N/A）。"""
    if abs(float(reference)) <= NEAR_ZERO_DENOMINATOR:
        return None
    return (float(reference) - float(new)) / float(reference)


def format_relative(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.4f}%"


def delta(reference: float, new: float) -> float:
    """正值表示 new 比 reference 更好（费用/工期更低）。"""
    return float(reference) - float(new)


def classify_effect(delta_j: float, tolerance: float = NEAR_ZERO_DENOMINATOR) -> str:
    if delta_j > tolerance:
        return "费用改善"
    if delta_j < -tolerance:
        return "费用恶化"
    return "近似持平"


# ---------------------------------------------------------------------------
# Phase B：对齐单变量干预
# ---------------------------------------------------------------------------


class AlignInterventionBlocked(RuntimeError):
    """align=1 在当前正式动作接口下不合法。"""


def action_snapshot(action: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        "branch": int(action["branch"]),
        "branch_name": ActionBranch(int(action["branch"])).name,
        "task_key": None if action.get("task_key") is None else str(action["task_key"]),
        "team": [int(w) for w in action.get("team", ())],
        "align": int(action.get("align", 0)),
    }


def validate_aligned_action(env: AirLineEnvWork3, action: Mapping[str, Any]) -> dict[str, Any]:
    """用正式动作合同核实改写后的动作；任何不合法都必须显式阻塞。"""
    branch = ActionBranch(int(action["branch"]))
    if branch != ActionBranch.STATION_EXECUTE:
        raise AlignInterventionBlocked(f"非留站执行分支不允许施加对齐干预: {branch.name}")
    task_key = str(action.get("task_key"))
    if task_key not in env.state.tasks:
        raise AlignInterventionBlocked(f"待干预工序不存在: {task_key}")
    if int(action.get("align", 0)) != 1:
        raise AlignInterventionBlocked("待干预动作的 align 未被改写为 1")
    task = env.state.tasks[task_key]
    can_reserve, _can_postpone = env.get_action_branch_mask(task)
    if not can_reserve:
        raise AlignInterventionBlocked(f"工序 {task_key} 当前不具备留站预约资格")
    team = tuple(int(w) for w in action.get("team", ()))
    if len(team) != int(task.demand):
        raise AlignInterventionBlocked(
            f"工序 {task_key} 团队人数{len(team)}与需求{int(task.demand)}不一致"
        )
    try:
        env._validate_team_for_task(task, team)
    except ValueError as error:
        raise AlignInterventionBlocked(
            f"工序 {task_key} 团队{team}未通过正式团队合同: {error}"
        ) from error
    aircraft = env.state.aircraft[task.aircraft_id]
    if int(aircraft.current_station) != int(task.current_station):
        raise AlignInterventionBlocked(
            f"工序 {task_key} 的飞机不在该站位: {aircraft.current_station} != {task.current_station}"
        )
    # 按 env.step 的同一套计算预演一次预约搜索，确保 align=1 之后仍存在合法区间。
    predecessor_release = env._predecessor_release_time(task)
    if predecessor_release is None:
        raise AlignInterventionBlocked(f"工序 {task_key} 的前驱尚未释放，无法核对 align=1 预约")
    align_target = float(env.state.last_transfer_time + task.in_station_offset)
    search_start = max(
        float(env.state.current_time),
        float(task.material_ready_time),
        float(predecessor_release),
        align_target,
    )
    try:
        duration = env.duration_for_team(task, team, start_time=search_start)
        slot = env._find_team_earliest_slot(
            station_id=task.current_station,
            team=team,
            search_start=search_start,
            duration=duration,
        )
    except NoFeasibleSlotError as error:
        raise AlignInterventionBlocked(
            f"工序 {task_key} 在 align=1 下无可行的团队/站位预约区间: {error}"
        ) from error
    return {
        "checked": True,
        "branch_mask": [bool(can_reserve), bool(_can_postpone)],
        "team_size": len(team),
        "demand": int(task.demand),
        "align_bound_target": align_target,
        "expected_scheduled_start": float(slot),
        "expected_duration": float(duration),
    }


def apply_align_intervention(
    action: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """只改写执行分支的 align；返回 (新动作或 None, 元信息)。

    绝不原地修改入参字典；推进与后移分支原样透传（返回 None 表示“不适用”）。
    """
    if action is None:
        return None, {"applicable": False, "reason": "no_action"}
    branch = ActionBranch(int(action["branch"]))
    if branch == ActionBranch.ADVANCE_TO_NEXT_EVENT:
        return None, {"applicable": False, "reason": "advance_to_next_event"}
    if branch == ActionBranch.POSTPONE:
        return None, {"applicable": False, "reason": "postpone"}
    align_before = int(action.get("align", 0))
    if align_before == 1:
        return None, {
            "applicable": True,
            "already_aligned": True,
            "changed": False,
            "align_before": 1,
            "align_after": 1,
            "reason": "already_aligned",
        }
    forced = dict(action)
    forced["align"] = 1
    return forced, {
        "applicable": True,
        "already_aligned": False,
        "changed": True,
        "align_before": align_before,
        "align_after": 1,
        "reason": "forced_0_to_1",
    }


@dataclass
class AlignForcedAgent(FormalEvaluationAgent):
    """在正式 C 贪心动作之后，仅把合法执行分支的 align 改成 1 的诊断包装器。

    只在 ``select_action`` 返回后复制一份动作字典并改写 ``align``；工序、团队、
    推进与后移分支一律原样透传，绝不注入教师动作、不改掩码、不按教师排程推时间。
    """

    events: list[dict[str, Any]] = field(default_factory=list)
    decision_index: int = 0

    def select_action(self, env: AirLineEnvWork3, *, return_diagnostics: bool = False):
        result = super().select_action(env, return_diagnostics=return_diagnostics)
        diagnostics: dict[str, Any] | None = None
        if return_diagnostics:
            action, diagnostics = result  # type: ignore[misc]
        else:
            action = result  # type: ignore[assignment]

        self.decision_index += 1
        event: dict[str, Any] = {
            "decision_index": self.decision_index,
            "cycle": int(env.state.current_cycle),
            "sim_time": float(env.state.current_time),
            "original_action": action_snapshot(action),
            "executed_action": action_snapshot(action),
            "align_before": None,
            "align_after": None,
            "applicable": False,
            "changed": False,
            "already_aligned": False,
            "reason": None,
            "legality": None,
            "execution_result": None,
        }

        if action is None:
            event["reason"] = "no_action"
            self.events.append(event)
            return result

        forced, meta = apply_align_intervention(action)
        if not meta["applicable"]:
            event["reason"] = meta["reason"]
            self.events.append(event)
            return result
        if meta["already_aligned"]:
            event["applicable"] = True
            event["align_before"] = 1
            event["align_after"] = 1
            event["already_aligned"] = True
            event["reason"] = meta["reason"]
            self.events.append(event)
            return result

        event["applicable"] = True
        event["align_before"] = int(meta["align_before"])
        assert forced is not None
        try:
            event["legality"] = validate_aligned_action(env, forced)
        except AlignInterventionBlocked as error:
            event["reason"] = "blocked_illegal_align1"
            event["legality"] = {"checked": False, "error": str(error)}
            self.events.append(event)
            dump_json(
                BLOCKED_INTERFACE,
                {
                    "status": "BLOCKED_INTERFACE",
                    "detail": str(error),
                    "event": event,
                    "candidate_task_keys": [t.task_key for t in env.get_action_candidates()],
                    "advance_available": any(
                        t.status == TaskStatus.RESERVED for t in env.state.tasks.values()
                    ),
                },
            )
            raise
        event["align_after"] = 1
        event["changed"] = True
        event["reason"] = meta["reason"]
        event["executed_action"] = action_snapshot(forced)
        self.events.append(event)
        if return_diagnostics:
            return forced, diagnostics
        return forced

    def summarise_events(self) -> dict[str, Any]:
        applicable = [e for e in self.events if e["applicable"]]
        return {
            "decision_count": len(self.events),
            "applicable_count": len(applicable),
            "already_aligned_count": sum(1 for e in applicable if e["already_aligned"]),
            "changed_0_to_1_count": sum(1 for e in applicable if e["changed"]),
            "not_applicable_count": len(self.events) - len(applicable),
            "advance_count": sum(
                1 for e in self.events if e["reason"] == "advance_to_next_event"
            ),
            "postpone_count": sum(1 for e in self.events if e["reason"] == "postpone"),
        }


def enrich_events_with_facts(
    events: list[dict[str, Any]],
    payload: Mapping[str, Any],
    index: Mapping[str, Mapping[str, Any]],
    control_payload: Mapping[str, Any] | None = None,
) -> None:
    """用执行事实回填每条干预动作的实际执行结果（不依赖 step 内部 hook）。

    “对齐下界是否真的改变该工序”不能只看本轨迹的开工时刻（它必然 ≥ 下界），
    必须与未干预原 C 中同一 task_key 的实际开工做对照。
    """
    tasks = payload["execution_facts"]["tasks_execution"]
    control_tasks = (
        control_payload["execution_facts"]["tasks_execution"]
        if control_payload is not None
        else {}
    )
    for event in events:
        executed = event.get("executed_action")
        if not executed or executed.get("task_key") is None:
            continue
        task_key = str(executed["task_key"])
        fact = tasks.get(task_key)
        if fact is None:
            continue
        meta = index[task_key]
        actual_start = fact.get("actual_start")
        cycle_start = fact.get("cycle_start_time")
        align_target = (
            None
            if cycle_start is None
            else float(cycle_start) + float(meta["in_station_offset"])
        )
        control = control_tasks.get(task_key, {})
        control_start = control.get("actual_start")
        shift = (
            None
            if actual_start is None or control_start is None
            else float(actual_start) - float(control_start)
        )
        event["execution_result"] = {
            "status": fact.get("status"),
            "actual_start": actual_start,
            "actual_end": fact.get("actual_end"),
            "cycle_start_time": cycle_start,
            "align_bound_target_from_facts": align_target,
            "satisfies_align_bound": None
            if actual_start is None or align_target is None
            else bool(float(actual_start) >= align_target - PHYSICAL_TOLERANCE),
            "control_task_key": task_key if control_tasks else None,
            "control_actual_start": control_start,
            "start_shift_vs_unintervened": shift,
            "start_moved_by_intervention": None
            if shift is None
            else bool(abs(shift) > PHYSICAL_TOLERANCE),
            "team": [int(w) for w in fact.get("team", ())],
            "postpone_count": int(fact.get("postpone_count", 0)),
        }


# ---------------------------------------------------------------------------
# Phase A：清册
# ---------------------------------------------------------------------------


def file_record(path: Path, comment: str = "") -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": bool(exists),
        "sha256": sha256_file(path) if exists else None,
        "comment": comment,
    }


def trajectory_record(
    path: Path,
    *,
    method: str,
    scenario_id: str,
    role: str,
    source: str,
) -> dict[str, Any]:
    exists = path.is_file()
    record: dict[str, Any] = {
        "role": role,
        "source": source,
        "method": method,
        "scenario_id": scenario_id,
        **file_record(path),
    }
    if exists:
        payload = load_json(path)
        record.update(
            {
                "declared_agent": payload.get("agent"),
                "declared_scenario_id": payload.get("scenario_id"),
                "decisions": int(payload.get("decisions", -1)),
                "completed_tasks": int(payload.get("completed_tasks", -1)),
                "success": bool(payload.get("success", False)),
                "feasible": bool(payload.get("feasible", False)),
                "objective_j_total": float(payload.get("objective_j_total", float("nan"))),
                "makespan": float(payload.get("makespan", float("nan"))),
                "actual_disturbance_hits": int(payload.get("actual_disturbance_hits", -1)),
                "has_execution_facts": bool(payload.get("execution_facts")),
            }
        )
    return record


def stage_inventory() -> int:
    threads = enforce_single_thread_cpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    baseline, tasks = baseline_bundle()
    metadata = baseline["metadata"]
    log(
        "基准: h0={:.15f} 任务={} 架={} 站={} 时钟={}".format(
            float(metadata["h0"]),
            len(tasks),
            metadata.get("num_aircraft"),
            metadata.get("num_stations"),
            metadata.get("transfer_policy"),
        )
    )

    if not ORIGINAL_C_WEIGHTS.is_file():
        raise FileNotFoundError(f"缺少原 C 检查点: {ORIGINAL_C_WEIGHTS}")
    if not BC_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"缺少 BC 诊断副本（二进制未入 Git，需本地原件）: {BC_WEIGHTS}"
        )
    if not PREVIOUS_MANIFEST.is_file():
        raise FileNotFoundError(f"缺少上一轮清单，无法绑定 BC 来源: {PREVIOUS_MANIFEST}")

    previous_manifest = load_json(PREVIOUS_MANIFEST)
    bc_expected = previous_manifest["diagnostic_checkpoint"]
    bc_sha = sha256_file(BC_WEIGHTS)
    if bc_sha != bc_expected["sha256"]:
        raise ValueError(
            f"BC 检查点 SHA256 与上一轮清单不一致: {bc_sha} != {bc_expected['sha256']}"
        )
    bc_model, bc_checkpoint = load_diagnostic_actor(
        BC_WEIGHTS,
        expected_teacher_manifest_sha256=previous_manifest["teacher"]["manifest_sha256"],
    )
    bc_fingerprint = actor_fingerprint(bc_model)
    if bc_fingerprint != bc_expected["actor_fingerprint"]:
        raise ValueError(
            f"BC Actor 指纹与上一轮清单不一致: {bc_fingerprint} != {bc_expected['actor_fingerprint']}"
        )
    if bc_checkpoint["source_checkpoint"]["sha256"] != sha256_file(ORIGINAL_C_WEIGHTS):
        raise ValueError("BC 记录的原 C 权重 SHA256 与当前原 C 文件不一致")

    original_agent, original_checkpoint = load_original_c()
    original_fingerprint = actor_fingerprint(original_agent.actor_critic)
    original_metadata = original_checkpoint.get("run_metadata", {})
    bc_source_fingerprint = bc_checkpoint["source_checkpoint"]["actor_fingerprint"]
    if original_fingerprint != bc_source_fingerprint:
        raise ValueError(
            "BC 绑定的源 Actor 指纹与当前原 C 不一致: "
            f"{bc_source_fingerprint} != {original_fingerprint}"
        )

    scenarios = load_scenarios()
    scenario_records: dict[str, Any] = {}
    for scenario_id in VALIDATION_SCENARIO_IDS:
        if scenario_id not in scenarios:
            raise ValueError(f"v1 场景文件缺少 {scenario_id}")
        scenario = require_v1_scenario(scenarios[scenario_id], scenario_id)
        target_key = str(scenario["target_task_key"])
        meta = baseline_index(tasks)[target_key]
        if int(scenario["station_id"]) != int(meta["baseline_station_field"]):
            raise ValueError(
                f"{scenario_id} 的站位字段与基准不一致: "
                f"scenario={scenario['station_id']} baseline={meta['baseline_station_field']}"
            )
        scenario_records[scenario_id] = {
            "scenario": scenario,
            "fingerprint": scenario_fingerprint(scenario),
            "target_task": target_key,
            "target_base_station_0based": int(meta["base_station"]),
            "target_in_station_offset": float(meta["in_station_offset"]),
            "target_baseline_team": list(meta["baseline_team"]),
            "target_demand": int(meta["demand"]),
            "split_membership": validation_split_membership(scenario_id),
        }

    reused = [
        trajectory_record(
            PREVIOUS_A_NOMINAL,
            method=A_METHOD,
            scenario_id="NOMINAL",
            role="历史复用：A 的 Nominal 完整物理事实",
            source="c_reproduction_diagnostic_v1/seed42（上一轮新增，已严格核验）",
        ),
        trajectory_record(
            PREVIOUS_C_NOMINAL,
            method=ORIGINAL_METHOD,
            scenario_id="NOMINAL",
            role="历史复用：未干预原 C 的 Nominal CPU 对照",
            source="c_reproduction_diagnostic_v1/seed42（上一轮新增，已严格核验）",
        ),
        trajectory_record(
            PREVIOUS_BC_NOMINAL,
            method=BC_METHOD,
            scenario_id="NOMINAL",
            role="历史复用：冻结 BC 副本自己的 Nominal 参照",
            source="c_reproduction_diagnostic_v1/seed42（上一轮新增，已严格核验）",
        ),
        trajectory_record(
            PREVIOUS_A_S1,
            method=A_METHOD,
            scenario_id="RULE_VAL_S1_MODERATE",
            role="历史复用：A 在 v1 S1 的完整物理事实",
            source="rule_baseline_ac_v1（已归档，六文件严格核验 PASS）",
        ),
        trajectory_record(
            PREVIOUS_A_S2,
            method=A_METHOD,
            scenario_id="RULE_VAL_S2_STRONG",
            role="历史复用：A 在 v1 S2 的完整物理事实",
            source="rule_baseline_ac_v1（已归档，六文件严格核验 PASS）",
        ),
    ]
    background = [
        trajectory_record(
            PREVIOUS_C_S1,
            method=ORIGINAL_METHOD,
            scenario_id="RULE_VAL_S1_MODERATE",
            role="背景参照：不做正式配对（本轮无原 C 受扰臂）",
            source="rule_baseline_ac_v1",
        ),
        trajectory_record(
            PREVIOUS_C_S2,
            method=ORIGINAL_METHOD,
            scenario_id="RULE_VAL_S2_STRONG",
            role="背景参照：不做正式配对（本轮无原 C 受扰臂）",
            source="rule_baseline_ac_v1",
        ),
    ]

    expected = list(EXPECTED_NEW_TRAJECTORIES)
    missing_reuse = [r["path"] for r in reused if not r["exists"]]

    inventory = {
        "schema_version": "c_alignment_bc_robustness_reuse_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_state(),
        "threads": threads,
        "device": "cpu",
        "precision": "fp32",
        "greedy": True,
        "warmup": "none",
        "decision_limit": DECISION_LIMIT,
        "optimizer_updates": 0,
        "baseline": {**file_record(BASELINE_PATH), "h0": float(metadata["h0"]),
                     "transfer_policy": metadata.get("transfer_policy")},
        "scenarios_file": file_record(SCENARIOS_PATH),
        "validation_split": {
            **file_record(VALIDATION_SPLIT_PATH),
            "membership": validation_split_membership(None),
        },
        "validation_scenarios": scenario_records,
        "original_c": {
            **file_record(ORIGINAL_C_WEIGHTS),
            "actor_fingerprint": original_fingerprint,
            "method_variant": original_metadata.get("method_variant"),
            "seed": original_metadata.get("seed"),
            "source_sha": original_metadata.get("source_sha"),
            "lightning_optimization_steps": original_metadata.get("lightning_optimization_steps"),
            "data_fingerprint": original_metadata.get("data_fingerprint"),
        },
        "diagnostic_copy": {
            **file_record(BC_WEIGHTS),
            "format": bc_checkpoint["diagnostic_format"],
            "actor_fingerprint": bc_fingerprint,
            "source_checkpoint_sha256": bc_checkpoint["source_checkpoint"]["sha256"],
            "source_actor_fingerprint": bc_checkpoint["source_checkpoint"]["actor_fingerprint"],
            "teacher_manifest_sha256": bc_checkpoint["teacher_manifest_sha256"],
            "training_steps": bc_checkpoint["training"]["optimizer_steps"],
            "local_only_note": (
                "*.pt 未入 Git（仓库既有 .gitignore）；本文件为用户本地原件，"
                "本轮以其 SHA256 与指纹做来源绑定。"
            ),
        },
        "reused_trajectories": reused,
        "background_only": background,
        "new_trajectories_expected": expected,
        "missing": {
            "reused": missing_reuse,
            "required_new": [item["path"] for item in expected],
        },
    }
    dump_json(REUSE_INVENTORY, inventory)

    dump_json(
        MANIFEST,
        {
            "task": "Work-3 C 对齐单变量干预与冻结 BC 副本受扰诊断",
            "schema_version": "c_alignment_bc_robustness_manifest_v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "reference_commit": "c3a1a3ea9680c3a83d13fcbc735a74bfcad9c19b",
            "run_context": git_state(),
            "output_dir": str(OUTPUT_DIR),
            "instance": {
                "baseline_path": str(BASELINE_PATH),
                "baseline_sha256": sha256_file(BASELINE_PATH),
                "dataset": "mini210",
                "aircraft": 7,
                "stations": 5,
                "tasks_per_aircraft": 30,
                "h0": float(metadata["h0"]),
                "transfer_policy": metadata.get("transfer_policy"),
            },
            "execution_settings": {
                "device": "cpu",
                "precision": "fp32",
                "greedy": True,
                "warmup": "none",
                "max_decisions": DECISION_LIMIT,
                "threads": threads,
                "optimizer_updates": 0,
            },
            "weights": {
                "original_c_sha256": sha256_file(ORIGINAL_C_WEIGHTS),
                "original_c_actor_fingerprint": original_fingerprint,
                "diagnostic_copy_sha256": bc_sha,
                "diagnostic_copy_actor_fingerprint": bc_fingerprint,
            },
            "scope": {
                "new_complete_trajectories": 3,
                "extra_control_runs": 0,
                "note": (
                    "原 C Nominal 对照复用上一轮同入口单线程 CPU 轨迹，因此本轮新增恰为三条；"
                    "未干预、未训练、未改场景、未改权重。"
                ),
            },
            "frozen_contracts": [
                "环境物理规则、费用权重、H0、同步转站模式、基准团队",
                "正式 C profile、候选集与所有原有掩码",
                "v1 场景文件与验证 split",
            ],
            "artifacts_expected": [item["path"] for item in expected],
        },
    )
    log("inventory 完成：来源、指纹、场景与复用清单已写入")
    return 0 if not missing_reuse else 3


def validation_split_membership(scenario_id: str | None) -> Any:
    if not VALIDATION_SPLIT_PATH.is_file():
        return None
    payload = load_json(VALIDATION_SPLIT_PATH)
    ids = payload if isinstance(payload, list) else payload.get("scenario_ids", payload)
    if scenario_id is None:
        return ids
    return scenario_id in (ids if isinstance(ids, list) else [])


def load_original_c() -> tuple[FormalEvaluationAgent, dict[str, Any]]:
    agent = build_formal_evaluation_agent("C", ORIGINAL_C_WEIGHTS, device="cpu")
    checkpoint = torch.load(ORIGINAL_C_WEIGHTS, map_location="cpu", weights_only=False)
    return agent, checkpoint


def load_bc_agent() -> tuple[FormalEvaluationAgent, dict[str, Any], torch.nn.Module]:
    model, checkpoint = load_diagnostic_actor(
        BC_WEIGHTS,
        expected_teacher_manifest_sha256=load_json(PREVIOUS_MANIFEST)["teacher"]["manifest_sha256"],
    )
    model.eval()
    agent = FormalEvaluationAgent(
        profile=build_method_profile("C"), actor_critic=model, time_head=None
    )
    return agent, checkpoint, model


# ---------------------------------------------------------------------------
# Phase B 执行
# ---------------------------------------------------------------------------


def stage_align() -> int:
    threads = enforce_single_thread_cpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)

    _, tasks = baseline_bundle()
    index = baseline_index(tasks)
    agent, _checkpoint = load_original_c()
    before_sha = sha256_file(ORIGINAL_C_WEIGHTS)
    before_fingerprint = actor_fingerprint(agent.actor_critic)

    forced = AlignForcedAgent(
        profile=agent.profile,
        actor_critic=agent.actor_critic,
        time_head=agent.time_head,
    )
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    log("运行对齐干预 Nominal 轨迹（仅改写执行分支 align，最多 4096 决策）...")
    try:
        result = evaluate_single_trajectory(
            env,
            "Method-C",
            forced,
            scenario=None,
            max_decisions=DECISION_LIMIT,
            device="cpu",
        )
    except AlignInterventionBlocked:
        log("对齐干预遇到非法 align=1，已写 blocked_interface.json 并停机")
        return 2

    facts = extract_execution_facts(env)
    payload = {
        **result,
        "agent": ALIGN_METHOD,
        "wrapped_agent_type": "Method-C",
        "diagnostic_banner": (
            "人为修改决策规则的诊断策略：仅把正式 C 贪心动作的 align 改为 1，"
            "不代表原 C 的正式评测成绩。"
        ),
        "intervention": {
            "policy": "force_align_1_on_legal_station_execute",
            "weights_changed": False,
            "log_prob_used": False,
            "log_prob_note": (
                "本轮不保存也不使用任何 log-prob：改写动作的概率未被重算，"
                "因此不以原动作 log-prob 冒充改写后动作的概率。"
            ),
            "original_c_sha256": before_sha,
            "original_c_actor_fingerprint": before_fingerprint,
        },
        "execution_facts": facts,
    }
    dump_json(ALIGN_TRAJ, payload)

    events = forced.events
    enrich_events_with_facts(events, payload, index, load_json(PREVIOUS_C_NOMINAL))
    dump_jsonl(INTERVENTION_EVENTS, events)
    summary = forced.summarise_events()
    summary["run_seconds_note"] = "见 runs/align_console.log"

    after_sha = sha256_file(ORIGINAL_C_WEIGHTS)
    after_fingerprint = actor_fingerprint(forced.actor_critic)
    dump_json(
        OUTPUT_DIR / "align_intervention_summary.json",
        {
            "method": ALIGN_METHOD,
            "counts": summary,
            "weights_checked": {
                "sha256_before": before_sha,
                "sha256_after": after_sha,
                "actor_fingerprint_before": before_fingerprint,
                "actor_fingerprint_after": after_fingerprint,
                "weights_unchanged": before_sha == after_sha
                and before_fingerprint == after_fingerprint,
            },
            "trajectory": {
                "decisions": int(result["decisions"]),
                "objective_j_total": float(result["objective_j_total"]),
                "makespan": float(result["makespan"]),
                "completed_tasks": int(result["completed_tasks"]),
                "success": bool(result["success"]),
                "feasible": bool(result["feasible"]),
                "postponed_count": int(result["postponed_count"]),
                "advance_cause_counts": result["advance_cause_counts"],
            },
            "threads": threads,
        },
    )
    log(
        "对齐干预: decisions={} J={:.10f} makespan={:.6f} 改写 0→1 {} 次 / 适用 {} 次".format(
            result["decisions"],
            result["objective_j_total"],
            result["makespan"],
            summary["changed_0_to_1_count"],
            summary["applicable_count"],
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Phase C 执行
# ---------------------------------------------------------------------------


def stage_disturbance() -> int:
    threads = enforce_single_thread_cpu()
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios()

    targets = {
        "RULE_VAL_S1_MODERATE": BC_S1_TRAJ,
        "RULE_VAL_S2_STRONG": BC_S2_TRAJ,
    }
    for scenario_id, path in targets.items():
        scenario = require_v1_scenario(scenarios[scenario_id], scenario_id)
        agent, checkpoint, model = load_bc_agent()
        before = actor_fingerprint(model)
        env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
        log(
            "运行冻结 BC 副本于 {}（目标 {}，tau={:.6f}h，R={:.6f}h）...".format(
                scenario_id,
                scenario["target_task_key"],
                float(scenario["tau"]),
                float(scenario["recovery_time"]),
            )
        )
        result = evaluate_single_trajectory(
            env,
            "Method-C",
            agent,
            scenario=scenario,
            max_decisions=DECISION_LIMIT,
            device="cpu",
        )
        after = actor_fingerprint(model)
        facts = extract_execution_facts(env)
        dump_json(
            path,
            {
                **result,
                "agent": BC_METHOD,
                "wrapped_agent_type": "Method-C",
                "diagnostic_banner": (
                    "冻结的 BC 诊断副本（diagnostic_behavior_clone）：本轮不训练、"
                    "不注入教师动作、不施加对齐干预。"
                ),
                "checkpoint_binding": {
                    "path": str(BC_WEIGHTS),
                    "sha256": sha256_file(BC_WEIGHTS),
                    "actor_fingerprint_before": before,
                    "actor_fingerprint_after": after,
                    "weights_unchanged": before == after,
                },
                "scenario_binding": {
                    "scenario_id": scenario_id,
                    "fingerprint": scenario_fingerprint(scenario),
                    "target_task_key": scenario["target_task_key"],
                    "target_aircraft_id": int(scenario["aircraft_id"]),
                    "target_station_field": int(scenario["station_id"]),
                    "tau": float(scenario["tau"]),
                    "recovery_time": float(scenario["recovery_time"]),
                    "timing_reference": scenario["timing_reference"],
                },
                "execution_facts": facts,
            },
        )
        log(
            "BC@{}: decisions={} J={:.10f} makespan={:.6f} completed={} success={} hits={}".format(
                scenario_id,
                result["decisions"],
                result["objective_j_total"],
                result["makespan"],
                result["completed_tasks"],
                result["success"],
                result["actual_disturbance_hits"],
            )
        )
    return 0


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------


def load_verifier():
    path = REPO_ROOT / "work3_4060" / "verify_trajectory_facts.py"
    spec = importlib.util.spec_from_file_location("work3_verify_trajectory_facts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载轨迹校验器: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_protocol_status(
    expected: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> str:
    """FAIL 优先于 INCOMPLETE；全部存在并通过才是 PASS。"""
    if any(row.get("status") == "FAIL" for row in results):
        return "FAIL"
    verified = {str(row.get("path")) for row in results}
    missing = [item for item in expected if str(item["path"]) not in verified]
    if missing or len(results) < len(expected):
        return "INCOMPLETE"
    return "PASS"


def protocol_exit_code(status: str) -> int:
    return {"PASS": 0, "FAIL": 2, "INCOMPLETE": 3}[status]


def stage_verify() -> int:
    threads = enforce_single_thread_cpu()
    verifier = load_verifier()
    baseline, tasks = baseline_bundle()
    h0 = float(baseline["metadata"]["h0"])
    scenarios = load_scenarios()
    index = baseline_index(tasks)

    rows: list[dict[str, Any]] = []
    for item in EXPECTED_NEW_TRAJECTORIES:
        path = Path(item["path"])
        if not path.is_file():
            continue
        scenario_spec = (
            {"scenario_id": "NOMINAL"}
            if item["scenario_id"] == "NOMINAL"
            else require_v1_scenario(scenarios[item["scenario_id"]], item["scenario_id"])
        )
        result = verifier.verify_single_trajectory(
            traj_path=path,
            baseline_h0=h0,
            baseline_tasks=tasks,
            scenario_spec=scenario_spec,
        )
        result["path"] = str(path)
        result["method"] = item["method"]
        result["scenario_id"] = item["scenario_id"]
        result["role"] = item["role"]
        rows.append(result)
        log(
            "轨迹核验[{} @ {}]: {} violations={} max_diff={:.3e}".format(
                item["method"],
                item["scenario_id"],
                result["status"],
                result["total_violations"],
                max(result["reconciliation_differences"].values()),
            )
        )

    protocol_status = classify_protocol_status(EXPECTED_NEW_TRAJECTORIES, rows)
    dump_json(
        TRAJECTORY_VERIFICATION,
        {
            "scope": (
                "本轮显式方法×场景清单；复用 work3_4060/verify_trajectory_facts.py::"
                "verify_single_trajectory 的单轨迹严格检查逻辑，六文件默认合同未被改动。"
            ),
            "expected_manifest": [dict(item) for item in EXPECTED_NEW_TRAJECTORIES],
            "verified_count": len(rows),
            "expected_count": len(EXPECTED_NEW_TRAJECTORIES),
            "protocol_status": protocol_status,
            "exit_code": protocol_exit_code(protocol_status),
            "tolerances": {
                "physical_time_hours": PHYSICAL_TOLERANCE,
                "cost_reconciliation": COST_TOLERANCE,
                "raw_reward_sum": 1e-9,
            },
            "baseline_sha256": sha256_file(BASELINE_PATH),
            "trajectories": rows,
        },
    )

    integrity = build_weight_integrity()
    dump_json(WEIGHT_INTEGRITY, integrity)

    production = build_production_status(rows)
    checks = {
        "protocol_status_pass": protocol_status == "PASS",
        "original_c_weights_unchanged": integrity["original_c"]["unchanged_across_round"],
        "diagnostic_copy_unchanged": integrity["diagnostic_copy"]["unchanged_across_round"],
        "diagnostic_copy_binding_valid": integrity["diagnostic_copy"]["binding_valid"],
        "scenario_binding_v1": integrity["scenarios"]["all_v1"],
        "align_intervention_executed": Path(ALIGN_TRAJ).is_file(),
        "bc_runs_used_no_alignment_wrapper": integrity["bc_runs"]["no_alignment_wrapper"],
    }
    overall = "PASS" if all(checks.values()) else "FAIL"
    dump_json(
        VERIFICATION,
        {
            "protocol_status": protocol_status,
            "overall_status": overall,
            "checks": checks,
            "production_status": production,
            "weight_integrity_summary": {
                "original_c_unchanged": integrity["original_c"]["unchanged_across_round"],
                "diagnostic_copy_unchanged": integrity["diagnostic_copy"]["unchanged_across_round"],
            },
            "threads": threads,
        },
    )
    log(f"verify 完成: protocol={protocol_status} overall={overall}")
    if overall != "PASS":
        return 2
    return protocol_exit_code(protocol_status)


def build_weight_integrity() -> dict[str, Any]:
    previous = load_json(PREVIOUS_MANIFEST)
    current_sha = sha256_file(ORIGINAL_C_WEIGHTS)
    agent, _checkpoint = load_original_c()
    current_fingerprint = actor_fingerprint(agent.actor_critic)
    bc_sha = sha256_file(BC_WEIGHTS)
    bc_expected = previous["diagnostic_checkpoint"]
    _, bc_checkpoint = load_diagnostic_actor(
        BC_WEIGHTS,
        expected_teacher_manifest_sha256=previous["teacher"]["manifest_sha256"],
    )
    bc_fingerprint = bc_checkpoint["actor_fingerprint"]

    align_summary_path = OUTPUT_DIR / "align_intervention_summary.json"
    align_summary = load_json(align_summary_path) if align_summary_path.is_file() else {}
    align_checked = align_summary.get("weights_checked", {})

    bc_run_bindings: list[dict[str, Any]] = []
    no_wrapper = True
    for path in (BC_S1_TRAJ, BC_S2_TRAJ):
        if not path.is_file():
            continue
        payload = load_json(path)
        binding = payload.get("checkpoint_binding", {})
        bc_run_bindings.append(
            {
                "path": str(path),
                "actor_fingerprint_before": binding.get("actor_fingerprint_before"),
                "actor_fingerprint_after": binding.get("actor_fingerprint_after"),
                "weights_unchanged": bool(binding.get("weights_unchanged", False)),
            }
        )
        if payload.get("agent") != BC_METHOD:
            no_wrapper = False
        if "align1" in str(payload.get("diagnostic_banner", "")):
            no_wrapper = False

    scenarios = load_scenarios()
    scenario_checks = {
        scenario_id: {
            "timing_reference": scenarios[scenario_id].get("timing_reference"),
            "is_v1": scenarios[scenario_id].get("timing_reference")
            == "nominal_entry_referenced_preentry_v1",
            "recovery_time": float(scenarios[scenario_id]["recovery_time"]),
        }
        for scenario_id in VALIDATION_SCENARIO_IDS
        if scenario_id in scenarios
    }

    return {
        "schema_version": "c_alignment_bc_weight_integrity_v1",
        "original_c": {
            "path": str(ORIGINAL_C_WEIGHTS),
            "sha256": current_sha,
            "actor_fingerprint": current_fingerprint,
            "sha256_before_align_run": align_checked.get("sha256_before"),
            "sha256_after_align_run": align_checked.get("sha256_after"),
            "actor_fingerprint_before_align_run": align_checked.get("actor_fingerprint_before"),
            "actor_fingerprint_after_align_run": align_checked.get("actor_fingerprint_after"),
            "unchanged_across_round": bool(
                align_checked.get("weights_unchanged", False)
                and align_checked.get("sha256_before") == current_sha
                and align_checked.get("sha256_after") == current_sha
            ),
        },
        "diagnostic_copy": {
            "path": str(BC_WEIGHTS),
            "sha256": bc_sha,
            "expected_sha256": bc_expected["sha256"],
            "actor_fingerprint": bc_fingerprint,
            "expected_actor_fingerprint": bc_expected["actor_fingerprint"],
            "source_checkpoint_sha256": bc_checkpoint["source_checkpoint"]["sha256"],
            "current_original_c_sha256": current_sha,
            "binding_valid": bc_sha == bc_expected["sha256"]
            and bc_fingerprint == bc_expected["actor_fingerprint"]
            and bc_checkpoint["source_checkpoint"]["sha256"] == current_sha,
            "unchanged_across_round": all(
                row["weights_unchanged"] and row["actor_fingerprint_before"] == bc_fingerprint
                for row in bc_run_bindings
            )
            if bc_run_bindings
            else False,
        },
        "bc_runs": {
            "runs": bc_run_bindings,
            "no_alignment_wrapper": no_wrapper,
            "note": "两条受扰轨迹均由正式 C profile 的普通 FormalEvaluationAgent 产生，未叠加对齐干预。",
        },
        "scenarios": {
            "detail": scenario_checks,
            "all_v1": all(item["is_v1"] for item in scenario_checks.values()),
            "scenarios_file_sha256": sha256_file(SCENARIOS_PATH),
        },
    }


def build_production_status(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    for item in EXPECTED_NEW_TRAJECTORIES:
        path = Path(item["path"])
        label = f"{item['method']}@{item['scenario_id']}"
        if not path.is_file():
            statuses[label] = {"production_status": "MISSING", "production_valid": False}
            continue
        payload = load_json(path)
        termination = str(payload.get("termination_reason"))
        if bool(payload.get("success")) and termination == "completed":
            status = "自然完工"
        elif termination == "decision_limit":
            status = "预算截止"
        elif int(payload.get("constraint_violation_count", 0)) > 0:
            status = "物理失败"
        else:
            status = termination
        statuses[label] = {
            "production_status": status,
            "termination_reason": termination,
            "decisions": int(payload["decisions"]),
            "completed_tasks": int(payload["completed_tasks"]),
            "constraint_violation_count": int(payload.get("constraint_violation_count", 0)),
            "actual_disturbance_hits": int(payload.get("actual_disturbance_hits", 0)),
            "makespan": float(payload["makespan"]),
            "objective_j_total": float(payload["objective_j_total"]),
            "production_valid": status == "自然完工",
        }
    return statuses


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def stage_report() -> int:
    baseline, tasks = baseline_bundle()
    h0 = float(baseline["metadata"]["h0"])
    index = baseline_index(tasks)
    scenarios = load_scenarios()
    reuse = load_json(REUSE_INVENTORY)
    integrity = load_json(WEIGHT_INTEGRITY)
    trajectory_verification = load_json(TRAJECTORY_VERIFICATION)
    verification = load_json(VERIFICATION)

    align_payload = load_json(ALIGN_TRAJ)
    bc_s1 = load_json(BC_S1_TRAJ)
    bc_s2 = load_json(BC_S2_TRAJ)
    a_nominal = load_json(PREVIOUS_A_NOMINAL)
    c_nominal = load_json(PREVIOUS_C_NOMINAL)
    bc_nominal = load_json(PREVIOUS_BC_NOMINAL)
    a_s1 = load_json(PREVIOUS_A_S1)
    a_s2 = load_json(PREVIOUS_A_S2)

    nominal_runs = [
        ("A", A_METHOD, "A（规则基准教师）", a_nominal),
        (ORIGINAL_METHOD, ORIGINAL_METHOD, "原 C（未干预）", c_nominal),
        (ALIGN_METHOD, ALIGN_METHOD, "C 对齐干预（仅 align=1）", align_payload),
    ]
    nominal_rows = build_nominal_rows(nominal_runs, index)
    write_csv(NOMINAL_COMPARISON, nominal_rows)

    disturbance_rows = build_disturbance_rows(
        [a_s1, a_s2], [bc_s1, bc_s2], a_nominal, bc_nominal
    )
    write_csv(DISTURBANCE_COMPARISON, disturbance_rows)

    cycle_rows_all: list[dict[str, Any]] = []
    for run_id, method, _label, payload in nominal_runs:
        cycle_rows_all.extend(cycle_rows(payload, run_id, method))
    for payload, run_id in (
        (a_s1, "A_S1"),
        (bc_s1, "BC_S1"),
        (a_s2, "A_S2"),
        (bc_s2, "BC_S2"),
    ):
        cycle_rows_all.extend(cycle_rows(payload, run_id, payload.get("agent", "")))
    write_csv(CYCLE_COMPARISON, cycle_rows_all)

    effect = analyse_effects(nominal_rows, disturbance_rows, index, scenarios)
    effect["disturbance"]["target_evidence"] = {
        "A@RULE_VAL_S1_MODERATE": target_task_evidence(
            a_s1, scenarios["RULE_VAL_S1_MODERATE"], index, a_nominal
        ),
        "BC@RULE_VAL_S1_MODERATE": target_task_evidence(
            bc_s1, scenarios["RULE_VAL_S1_MODERATE"], index, bc_nominal
        ),
        "A@RULE_VAL_S2_STRONG": target_task_evidence(
            a_s2, scenarios["RULE_VAL_S2_STRONG"], index, a_nominal
        ),
        "BC@RULE_VAL_S2_STRONG": target_task_evidence(
            bc_s2, scenarios["RULE_VAL_S2_STRONG"], index, bc_nominal
        ),
    }
    effect["schedule_equivalence"] = compare_schedules(
        {"A": a_nominal, "C": c_nominal, "BC": bc_nominal},
        {"A_S1": a_s1, "BC_S1": bc_s1, "A_S2": a_s2, "BC_S2": bc_s2},
        align_payload,
    )
    summary = build_summary(
        baseline=baseline,
        h0=h0,
        reuse=reuse,
        integrity=integrity,
        trajectory_verification=trajectory_verification,
        verification=verification,
        nominal_rows=nominal_rows,
        disturbance_rows=disturbance_rows,
        cycle_rows_all=cycle_rows_all,
        effect=effect,
        scenarios=scenarios,
        align_payload=align_payload,
        nominal_payloads={
            "A": a_nominal,
            "C": c_nominal,
            "BC": bc_nominal,
        },
        disturbed_payloads={
            "A_S1": a_s1,
            "BC_S1": bc_s1,
            "A_S2": a_s2,
            "BC_S2": bc_s2,
        },
    )
    SUMMARY_MD.write_text(summary, encoding="utf-8")

    dump_json(
        OUTPUT_DIR / "effect_analysis.json",
        {
            "schema_version": "c_alignment_bc_effect_v1",
            "nominal_alignment": effect["alignment"],
            "bc_disturbance": effect["disturbance"],
            "action_different_but_schedule_same": effect["schedule_equivalence"],
        },
    )
    log(
        "report 完成: ΔJ_align={} ΔM_align={} 协议={}".format(
            format_float(effect["alignment"]["delta_j"]),
            format_float(effect["alignment"]["delta_makespan"]),
            trajectory_verification["protocol_status"],
        )
    )
    return 0


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_float(row[key]) if key in row else "" for key in fieldnames})


def build_nominal_rows(
    runs: Sequence[tuple[str, str, str, Mapping[str, Any]]],
    index: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference: Mapping[str, Any] | None = None
    for run_id, method, label, payload in runs:
        components = payload["cost_components_weighted"]
        cycle = summarize_cycle_delay(cycle_rows(payload, run_id, method))
        item = {
            "run_id": run_id,
            "method": method,
            "label": label,
            "scenario_id": payload.get("scenario_id"),
            "decisions": int(payload["decisions"]),
            "completed_tasks": int(payload["completed_tasks"]),
            "success": bool(payload["success"]),
            "feasible": bool(payload["feasible"]),
            "constraint_violations": int(payload.get("constraint_violation_count", 0)),
            "explicit_advances": decision_totals(payload)["explicit_advances"],
            "auto_advances": decision_totals(payload)["auto_advances"],
            "postponed_count": int(payload["postponed_count"]),
            "j_total": float(payload["objective_j_total"]),
            "j_takt": float(components["j_takt"]),
            "d_time": float(components["d_time"]),
            "d_team": float(components["d_team"]),
            "j_postpone": float(components["j_postpone"]),
            "j_revision": float(components["j_revision"]),
            "makespan": float(payload["makespan"]),
            "transfers": int(payload["transfers"]),
            "max_cycle_delay": cycle["max_cycle_delay"],
            "sum_cycle_delay": cycle["sum_cycle_delay"],
            "delayed_cycle_count": cycle["delayed_cycle_count"],
            "first_delayed_cycle": cycle["first_delayed_cycle"],
        }
        if run_id == ORIGINAL_METHOD:
            reference = item
        rows.append(item)
    if reference is not None:
        for item in rows:
            item["delta_j_vs_original_c"] = delta(reference["j_total"], item["j_total"])
            item["delta_makespan_vs_original_c"] = delta(
                reference["makespan"], item["makespan"]
            )
            item["relative_j_reduction_vs_original_c"] = (
                None
                if item is reference
                else relative_reduction(reference["j_total"], item["j_total"])
            )
    return rows


def build_disturbance_rows(
    a_payloads: Sequence[Mapping[str, Any]],
    bc_payloads: Sequence[Mapping[str, Any]],
    a_nominal: Mapping[str, Any],
    bc_nominal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nominal_ref = {"A": a_nominal, "BC": bc_nominal}
    by_policy = {"A": list(a_payloads), "BC": list(bc_payloads)}
    # 按场景分组，便于在同一扰动下直接对照 A 与 BC
    for index_in_group in range(len(a_payloads)):
        for policy in ("A", "BC"):
            payload = by_policy[policy][index_in_group]
            base = nominal_ref[policy]
            components = payload["cost_components_weighted"]
            base_components = base["cost_components_weighted"]
            totals = decision_totals(payload)
            rows.append(
                {
                    "policy": policy,
                    "scenario_id": payload["scenario_id"],
                    "decisions": int(payload["decisions"]),
                    "completed_tasks": int(payload["completed_tasks"]),
                    "success": bool(payload["success"]),
                    "feasible": bool(payload["feasible"]),
                    "constraint_violations": int(payload.get("constraint_violation_count", 0)),
                    "actual_disturbance_hits": int(payload.get("actual_disturbance_hits", 0)),
                    "explicit_advances": totals["explicit_advances"],
                    "auto_advances": totals["auto_advances"],
                    "postponed_count": int(payload["postponed_count"]),
                    "j_total": float(payload["objective_j_total"]),
                    "j_takt": float(components["j_takt"]),
                    "d_time": float(components["d_time"]),
                    "d_team": float(components["d_team"]),
                    "j_postpone": float(components["j_postpone"]),
                    "j_revision": float(components["j_revision"]),
                    "makespan": float(payload["makespan"]),
                    "delta_j_vs_own_nominal": float(payload["objective_j_total"])
                    - float(base["objective_j_total"]),
                    "delta_makespan_vs_own_nominal": float(payload["makespan"])
                    - float(base["makespan"]),
                    "delta_j_takt_vs_own_nominal": float(components["j_takt"])
                    - float(base_components["j_takt"]),
                    "nominal_j_total": float(base["objective_j_total"]),
                    "nominal_makespan": float(base["makespan"]),
                }
            )
    return rows


def target_task_evidence(
    payload: Mapping[str, Any],
    scenario: Mapping[str, Any],
    index: Mapping[str, Mapping[str, Any]],
    nominal_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """用完整物理事实判断事件是否真正作用、是否传播到转站与工期。"""
    target = str(scenario["target_task_key"])
    meta = index[target]
    tasks = payload["execution_facts"]["tasks_execution"]
    fact = tasks.get(target, {})
    nominal_fact = (
        nominal_payload["execution_facts"]["tasks_execution"].get(target, {})
        if nominal_payload is not None
        else {}
    )
    predecessor_ends = []
    for pred_id in meta["predecessors"]:
        pred_key = f"{scenario['aircraft_id']}_{pred_id}"
        pred_fact = tasks.get(pred_key)
        if pred_fact and pred_fact.get("actual_end") is not None:
            predecessor_ends.append(float(pred_fact["actual_end"]))
    tau = float(scenario["tau"])
    recovery = float(scenario["recovery_time"])
    actual_start = fact.get("actual_start")
    nominal_start = nominal_fact.get("actual_start")
    cycles = payload["execution_facts"]["cycle_execution_records"]
    delayed_cycles = [
        int(rec["cycle_id"])
        for rec in cycles.values()
        if float(rec["delay_relative_to_nominal"]) > PHYSICAL_TOLERANCE
    ]
    same_team_before: list[dict[str, Any]] = []
    if actual_start is not None:
        target_team = set(meta["baseline_team"])
        for key, other in tasks.items():
            if key == target:
                continue
            if int(other.get("aircraft_id", -1)) != int(scenario["aircraft_id"]):
                continue
            if not (set(int(w) for w in other.get("team", ())) & target_team):
                continue
            other_start = other.get("actual_start")
            if other_start is None:
                continue
            if (
                float(other_start) >= recovery - PHYSICAL_TOLERANCE
                and float(other_start) < float(actual_start) - PHYSICAL_TOLERANCE
            ):
                same_team_before.append(
                    {"task_key": key, "actual_start": float(other_start)}
                )
    same_team_before.sort(key=lambda item: item["actual_start"])
    return {
        "target_task_key": target,
        "target_base_station_0based": int(meta["base_station"]),
        "target_baseline_team": list(meta["baseline_team"]),
        "target_in_station_offset": float(meta["in_station_offset"]),
        "tau": tau,
        "recovery_time": recovery,
        "actual_hit_keys": list(payload.get("disturbance_effects", {}).get("actual_hit_task_keys", [])),
        "target_actual_start": actual_start,
        "target_actual_end": fact.get("actual_end"),
        "target_team": [int(w) for w in fact.get("team", ())],
        "target_nominal_start": nominal_start,
        "start_shift_vs_own_nominal": None
        if actual_start is None or nominal_start is None
        else float(actual_start) - float(nominal_start),
        "target_started_before_event": None
        if actual_start is None
        else bool(float(actual_start) < tau - PHYSICAL_TOLERANCE),
        "target_start_satisfies_recovery": None
        if actual_start is None
        else bool(float(actual_start) >= recovery - PHYSICAL_TOLERANCE),
        "target_predecessor_end_times": predecessor_ends,
        "target_predecessor_released_before_recovery": None
        if not predecessor_ends
        else bool(max(predecessor_ends) < recovery - PHYSICAL_TOLERANCE),
        "same_team_tasks_started_after_recovery_before_target": same_team_before,
        "delayed_cycles": delayed_cycles,
        "makespan_shift_vs_own_nominal": None
        if nominal_payload is None
        else float(payload["makespan"]) - float(nominal_payload["makespan"]),
    }


def analyse_effects(
    nominal_rows: Sequence[Mapping[str, Any]],
    disturbance_rows: Sequence[Mapping[str, Any]],
    index: Mapping[str, Mapping[str, Any]],
    scenarios: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reference = next(row for row in nominal_rows if row["run_id"] == ORIGINAL_METHOD)
    align_row = next(row for row in nominal_rows if row["run_id"] == ALIGN_METHOD)
    a_row = next(row for row in nominal_rows if row["run_id"] == "A")

    delta_j = delta(reference["j_total"], align_row["j_total"])
    delta_m = delta(reference["makespan"], align_row["makespan"])
    relative = relative_reduction(reference["j_total"], align_row["j_total"])

    alignment = {
        "j_original": reference["j_total"],
        "j_align1": align_row["j_total"],
        "delta_j": delta_j,
        "delta_makespan": delta_m,
        "delta_j_takt": delta(reference["j_takt"], align_row["j_takt"]),
        "delta_d_time": delta(reference["d_time"], align_row["d_time"]),
        "delta_d_team": delta(reference["d_team"], align_row["d_team"]),
        "delta_j_postpone": delta(reference["j_postpone"], align_row["j_postpone"]),
        "delta_j_revision": delta(reference["j_revision"], align_row["j_revision"]),
        "relative_j_reduction": relative,
        "effect_status": classify_effect(delta_j),
        "gap_to_a": align_row["j_total"] - a_row["j_total"],
        "gap_to_a_original_c": reference["j_total"] - a_row["j_total"],
        "decisions_original": reference["decisions"],
        "decisions_align1": align_row["decisions"],
    }

    disturbed: dict[str, Any] = {}
    for scenario_id in VALIDATION_SCENARIO_IDS:
        scenario = scenarios[scenario_id]
        for policy in ("A", "BC"):
            row = next(
                r
                for r in disturbance_rows
                if r["policy"] == policy and r["scenario_id"] == scenario_id
            )
            disturbed[f"{policy}@{scenario_id}"] = {
                key: row[key]
                for key in (
                    "j_total",
                    "makespan",
                    "delta_j_vs_own_nominal",
                    "delta_makespan_vs_own_nominal",
                    "delta_j_takt_vs_own_nominal",
                    "actual_disturbance_hits",
                    "success",
                    "feasible",
                    "constraint_violations",
                    "decisions",
                )
            }
        disturbed[scenario_id] = {
            "target_task": scenario["target_task_key"],
            "tau": float(scenario["tau"]),
            "recovery_time": float(scenario["recovery_time"]),
        }
    return {
        "alignment": alignment,
        "disturbance": disturbed,
        "schedule_equivalence": {},
    }


# ---------------------------------------------------------------------------
# 汇总报告
# ---------------------------------------------------------------------------


def fmt(value: Any, digits: int = 10) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int,)):
        return str(value)
    number = float(value)
    if number != 0.0 and abs(number) < 1e-6:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def build_summary(
    *,
    baseline: Mapping[str, Any],
    h0: float,
    reuse: Mapping[str, Any],
    integrity: Mapping[str, Any],
    trajectory_verification: Mapping[str, Any],
    verification: Mapping[str, Any],
    nominal_rows: Sequence[Mapping[str, Any]],
    disturbance_rows: Sequence[Mapping[str, Any]],
    cycle_rows_all: Sequence[Mapping[str, Any]],
    effect: Mapping[str, Any],
    scenarios: Mapping[str, Mapping[str, Any]],
    align_payload: Mapping[str, Any],
    nominal_payloads: Mapping[str, Mapping[str, Any]],
    disturbed_payloads: Mapping[str, Mapping[str, Any]],
) -> str:
    alignment = effect["alignment"]
    lines: list[str] = []
    lines.append("# Work-3 对齐单变量干预与冻结 BC 副本受扰诊断报告")
    lines.append("")
    lines.append(
        "> 本轮只回答两问：(1) 只把原 C 执行分支的 `align` 改成 1，Nominal 改善多少；"
        "(2) 冻结的 BC 副本在两个 v1 验证扰动下能否继续合法出线、费用与周期延迟如何。"
        "全程 CPU 推理、零参数更新，未改环境物理、费用权重、H0、同步转站、基准团队、"
        "正式 C profile、候选集与掩码。"
    )
    lines.append("")
    lines.append(
        f"**协议状态**：`{trajectory_verification['protocol_status']}`"
        f"（退出码 {trajectory_verification['exit_code']}）；"
        f"诊断合同自检 `{verification['overall_status']}`。"
    )
    lines.append("")

    lines.append("## 1. align 的真实语义")
    lines.append("")
    lines.append(
        "`align` 是执行分支的**对齐选项**，不是独立动作：`ActionBranch.STATION_EXECUTE`（留站执行）"
        "带 `align∈{0,1}`，`ADVANCE_TO_NEXT_EVENT` 与 `POSTPONE` 没有该字段。"
    )
    lines.append("")
    lines.append(
        "`envs/work3/environment.py::AirLineEnvWork3.step` 中，`align == 1` 会把开工搜索下界抬高到"
        " `last_transfer_time + task.in_station_offset`（基准站内偏移对应的名义时刻）："
    )
    lines.append("")
    lines.append("```python")
    lines.append("if align == 1:")
    lines.append("    align_target = self.state.last_transfer_time + task.in_station_offset")
    lines.append("    search_start = max(search_start, align_target)")
    lines.append("```")
    lines.append("")
    lines.append(
        "`align = 0` 只用最早可行安排（`max(当前时刻, 物料下界, 前序释放)`）。"
        "因此 **align=1 只是加了一条下界，并不强制改变开工时刻**："
        "若该下界本来就已经不紧（任务自然排在偏移之后），align=0/1 得到的实际开工时间相同。"
        "所以本轮把两类都记：标签被改写的次数，与实际生效（下界起作用）的次数。"
    )
    lines.append("")
    lines.append(
        "另外，`align` 不进入任何费用项：`J_revision` 比较的是发布后的 `position = scheduled_start - "
        "last_transfer_time` 与团队，而不是 align 标志位。因此干预只能通过改变实际开工/团队来间接影响费用。"
    )
    lines.append("")
    lines.append(
        "**与上一轮的关系（先校正结论）**：上一轮的 0% 对齐准确率是**教师状态、教师前缀条件下**的"
        "单头指标，不能扩写为“原 C 在所有自主访问状态上从未选过 align=1”；"
        "也不能据此说对齐头是全部费用差距的唯一原因——原 C 在教师状态上的决策类型、工序、团队也都不同，"
        "且 BC 更新了共享编码器与多个策略头。本轮只测“把 align 改成 1 的闭环效果”。"
    )
    lines.append("")

    lines.append("## 2. 权重、来源与复用清册")
    lines.append("")
    original = reuse["original_c"]
    bc = reuse["diagnostic_copy"]
    lines.append(
        f"- 原 C：`{Path(original['path']).name}`，SHA256 `{original['sha256']}`，"
        f"Actor 指纹 `{original['actor_fingerprint']}`，方法 `{original['method_variant']}`，"
        f"seed {original['seed']}，来源提交 `{original['source_sha']}`。"
    )
    lines.append(
        f"- BC 诊断副本：`{Path(bc['path']).name}`，SHA256 `{bc['sha256']}`，"
        f"格式 `{bc['format']}`，Actor 指纹 `{bc['actor_fingerprint']}`，"
        f"绑定上一轮教师清单 `{bc['teacher_manifest_sha256']}`，训练步 {bc['training_steps']}。"
    )
    lines.append(
        "- 两者均与上一轮 `manifest.json` 逐项一致；BC 的 `source_checkpoint.sha256` 等于当前原 C 文件哈希。"
    )
    lines.append("")
    lines.append("复用与新增（完整路径与哈希见 `reuse_inventory.json`）：")
    lines.append("")
    lines.append("| 轨迹 | 方法 | 场景 | 本轮角色 | 新增/复用 |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for item in reuse["reused_trajectories"]:
        lines.append(
            f"| `{Path(item['path']).name}` | {item['method']} | {item['scenario_id']} | "
            f"{item['role']} | 复用（{item['sha256'][:16]}…） |"
        )
    for item in reuse["new_trajectories_expected"]:
        lines.append(
            f"| `{Path(item['path']).name}` | {item['method']} | {item['scenario_id']} | "
            f"{item['role']} | **新增** |"
        )
    lines.append("")
    lines.append(
        f"- 新增完整轨迹 {len(reuse['new_trajectories_expected'])} 条，额外对照 0 条"
        "（原 C Nominal 对照复用上一轮同入口单线程 CPU 轨迹）。"
    )
    lines.append("")

    lines.append("## 3. 对齐干预：Nominal 三行表")
    lines.append("")
    lines.append(
        "| 实体 | 决策 | 完工 | 可行 | 违规 | 显式推进 | 后移 | $J_{total}$ | $J_{takt}$ | "
        "$D_{time}$ | $D_{team}$ | $J_{postpone}$ | $J_{revision}$ | Makespan (h) | 最大周期延迟 (h) | 延迟周期数 |"
    )
    lines.append(
        "| :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in nominal_rows:
        lines.append(
            "| {label} | {decisions} | {completed} | {feasible} | {vio} | {exp} | {post} | {j} | {jt} | {dt} | {dw} | {jp} | {jr} | {ms} | {maxd} | {dc} |".format(
                label=row["label"],
                decisions=row["decisions"],
                completed=row["completed_tasks"],
                feasible="PASS" if row["feasible"] and row["success"] else "FAIL",
                vio=row["constraint_violations"],
                exp=row["explicit_advances"],
                post=row["postponed_count"],
                j=fmt(row["j_total"]),
                jt=fmt(row["j_takt"]),
                dt=fmt(row["d_time"]),
                dw=fmt(row["d_team"]),
                jp=fmt(row["j_postpone"]),
                jr=fmt(row["j_revision"]),
                ms=fmt(row["makespan"], 6),
                maxd=fmt(row["max_cycle_delay"], 6),
                dc=row["delayed_cycle_count"],
            )
        )
    lines.append("")
    lines.append(
        f"- $\\Delta J_{{align}} = J_{{C,original}} - J_{{C,align1}}$ = "
        f"{fmt(alignment['j_original'])} − {fmt(alignment['j_align1'])} = **{fmt(alignment['delta_j'])}**"
        f"（以原 C 为分母的相对变化 {format_relative(alignment['relative_j_reduction'])}，负值表示恶化）"
    )
    lines.append(
        f"- $\\Delta M_{{align}} = M_{{C,original}} - M_{{C,align1}}$ = "
        f"{fmt(alignment['delta_makespan'], 6)}（正值表示工期缩短）"
    )
    lines.append(
        f"- 分项变化：$J_{{takt}}$ {fmt(alignment['delta_j_takt'])}，$D_{{time}}$ {fmt(alignment['delta_d_time'])}，"
        f"$D_{{team}}$ {fmt(alignment['delta_d_team'])}，$J_{{postpone}}$ {fmt(alignment['delta_j_postpone'])}，"
        f"$J_{{revision}}$ {fmt(alignment['delta_j_revision'])}（均为原 C − 干预）"
    )
    lines.append(
        f"- 与 A 的剩余差距：干预后 {fmt(alignment['gap_to_a'])}，原 C 为 {fmt(alignment['gap_to_a_original_c'])}；"
        f"决策数 {alignment['decisions_original']} → {alignment['decisions_align1']}。"
    )
    lines.append(
        f"- 效果裁决（绝对容差 1e-9）：**{alignment['effect_status']}**。"
    )
    lines.append("")
    summary_path = OUTPUT_DIR / "align_intervention_summary.json"
    align_counts = load_json(summary_path)["counts"] if summary_path.is_file() else {}
    if align_counts:
        lines.append("干预局部统计（逐决策明细见 `intervention_events.jsonl`）：")
        lines.append("")
        lines.append(
            f"- 决策 {align_counts['decision_count']} 次；适用（留站执行分支）{align_counts['applicable_count']} 次；"
            f"原本已是 align=1 {align_counts['already_aligned_count']} 次；"
            f"真正 0→1 改写 {align_counts['changed_0_to_1_count']} 次；"
            f"不适用（显式推进 {align_counts['advance_count']}、后移 {align_counts['postpone_count']}）"
            f"{align_counts['not_applicable_count']} 次。"
        )
        lines.append("")
    align_events = load_events(INTERVENTION_EVENTS)
    changed = [e for e in align_events if e.get("changed")]
    moved = [
        e
        for e in changed
        if (e.get("execution_result") or {}).get("start_moved_by_intervention") is True
    ]
    unmoved = len(changed) - len(moved)
    if changed:
        lines.append(
            f"- 对照未干预原 C 的同一 `task_key`：被改写的 {len(changed)} 次里，"
            f"该工序实际开工**被推后** {len(moved)} 次，开工时刻**未变** {unmoved} 次"
            "（后者说明该下界本来就不紧，标签改了但没有物理效果）。"
        )
        shifts = [
            float(e["execution_result"]["start_shift_vs_unintervened"])
            for e in moved
            if e["execution_result"].get("start_shift_vs_unintervened") is not None
        ]
        if shifts:
            lines.append(
                f"- 被推后的幅度：最小 {fmt(min(shifts), 6)}h，中位 "
                f"{fmt(sorted(shifts)[len(shifts) // 2], 6)}h，最大 {fmt(max(shifts), 6)}h。"
            )
        lines.append(
            "- 干预只改 align：工序、团队、推进/后移分支保持原样；"
            "状态改变后后续决策自然可能不同，这不算同时手工改了其他策略头。"
        )
        lines.append("")

    lines.append("## 4. 冻结 BC 副本的两个 v1 受扰测试")
    lines.append("")
    lines.append(
        "| 策略 | 场景 | 决策 | 完工 | 可行 | 违规 | 命中 | 显式推进 | 后移 | $J_{total}$ | "
        "相对自身 Nominal 的 $\\Delta J$ | Makespan (h) | 相对自身 Nominal 的 $\\Delta M$ (h) |"
    )
    lines.append(
        "| :--- | :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in disturbance_rows:
        lines.append(
            "| {policy} | `{scenario}` | {decisions} | {completed} | {feasible} | {vio} | {hits} | "
            "{exp} | {post} | {j} | {dj} | {ms} | {dm} |".format(
                policy=row["policy"],
                scenario=row["scenario_id"],
                decisions=row["decisions"],
                completed=row["completed_tasks"],
                feasible="PASS" if row["feasible"] and row["success"] else "FAIL",
                vio=row["constraint_violations"],
                hits=row["actual_disturbance_hits"],
                exp=row["explicit_advances"],
                post=row["postponed_count"],
                j=fmt(row["j_total"]),
                dj=fmt(row["delta_j_vs_own_nominal"]),
                ms=fmt(row["makespan"], 6),
                dm=fmt(row["delta_makespan_vs_own_nominal"], 6),
            )
        )
    lines.append("")
    lines.append(
        "> $\\Delta J_M(s) = J_M(s) - J_M(\\text{nominal})$、$\\Delta M_M(s) = M_M(s) - M_M(\\text{nominal})$，"
        "每种策略用自己的 Nominal 作参照；总费用比较与扰动增量比较分开。"
    )
    lines.append("")
    for scenario_id in VALIDATION_SCENARIO_IDS:
        a_item = effect["disturbance"].get(f"A@{scenario_id}", {})
        bc_item = effect["disturbance"].get(f"BC@{scenario_id}", {})
        if not a_item or not bc_item:
            continue
        gap = float(bc_item["delta_j_vs_own_nominal"]) - float(
            a_item["delta_j_vs_own_nominal"]
        )
        lines.append(
            f"- {scenario_id}：BC 的扰动增量比 A "
            + ("低" if gap < 0 else ("高" if gap > 0 else "相同"))
            + f" {abs(gap):.3e}。**两者量级都远小于其总费用口径，"
            "且每个场景只有一条轨迹，本轮不据此判定优劣。**"
        )
    lines.append("")

    lines.append("### 4.1 事件是否真正造成约束影响")
    lines.append("")
    for scenario_id in VALIDATION_SCENARIO_IDS:
        evidence = effect["disturbance"].get(scenario_id, {})
        if not evidence:
            continue
        scenario_record = scenarios[scenario_id]
        lines.append(
            f"**{scenario_id}**：目标 `{evidence['target_task']}`，"
            f"$\\tau$={fmt(evidence['tau'], 6)}h，$R$={fmt(evidence['recovery_time'], 6)}h，"
            f"timing_reference=`{scenario_record['timing_reference']}`，"
            f"事件指纹 `{scenario_fingerprint(scenario_record)[:16]}…`。"
        )
        lines.append("")
        for policy in ("A", "BC"):
            item = effect["disturbance"].get(f"{policy}@{scenario_id}", {})
            lines.append(
                f"- {policy}：$\\Delta J$={fmt(item.get('delta_j_vs_own_nominal'))}，"
                f"$\\Delta M$={fmt(item.get('delta_makespan_vs_own_nominal'), 6)}h，"
                f"命中 {item.get('actual_disturbance_hits')}，"
                f"完工 {item.get('success')}，违规 {item.get('constraint_violations')}。"
            )
            evidence_target = effect["disturbance"].get("target_evidence", {}).get(
                f"{policy}@{scenario_id}"
            )
            if not evidence_target:
                continue
            lines.append(
                "  - 目标工序实际开工 "
                f"{fmt(evidence_target.get('target_actual_start'), 6)}h"
                f"（自身 Nominal 为 {fmt(evidence_target.get('target_nominal_start'), 6)}h，"
                f"偏移 {fmt(evidence_target.get('start_shift_vs_own_nominal'), 6)}h）；"
                f"是否早于事件发生：{fmt(evidence_target.get('target_started_before_event'))}；"
                f"是否满足 $start \\ge R$：{fmt(evidence_target.get('target_start_satisfies_recovery'))}。"
            )
            lines.append(
                "  - 目标前序完工时刻 "
                + (
                    "无（该工序在站内/机内无前序约束）"
                    if not evidence_target.get("target_predecessor_end_times")
                    else str(evidence_target.get("target_predecessor_end_times"))
                )
                + f"；延迟周期 "
                + (
                    "无（没有任何周期延迟）"
                    if not evidence_target.get("delayed_cycles")
                    else str(evidence_target.get("delayed_cycles"))
                )
                + f"；终局工期偏移 {fmt(evidence_target.get('makespan_shift_vs_own_nominal'), 6)}h。"
            )
            same_team = evidence_target.get(
                "same_team_tasks_started_after_recovery_before_target"
            ) or []
            lines.append(
                "  - 在 $R$ 之后、目标开工之前，同机同团队已开工的工序："
                + (
                    "无"
                    if not same_team
                    else "、".join(
                        f"`{item['task_key']}`@{item['actual_start']:.4f}h" for item in same_team
                    )
                )
                + ("（说明目标被排到这些工序之后，属于预约顺序差异）" if same_team else "")
            )
            lines.append("  - 判读：" + classify_event_effect(evidence_target))
        lines.append("")

    lines.append("## 5. 逐周期转站与“动作不同但排程相同”")
    lines.append("")
    lines.append(
        "逐周期（名义/实际转站、延迟）见 `cycle_comparison.csv`。摘要："
    )
    lines.append("")
    lines.append("| 运行 | 场景 | 延迟周期数 | 最大延迟 (h) | 累计延迟 (h) | 首个延迟周期 |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: |")
    seen: set[tuple[str, str]] = set()
    for row in cycle_rows_all:
        key = (str(row["run_id"]), str(row["scenario_id"]))
        if key in seen:
            continue
        seen.add(key)
        rows = [r for r in cycle_rows_all if (str(r["run_id"]), str(r["scenario_id"])) == key]
        summary = summarize_cycle_delay(rows)
        lines.append(
            f"| {row['run_id']} | `{row['scenario_id']}` | {summary['delayed_cycle_count']} | "
            f"{fmt(summary['max_cycle_delay'], 6)} | {fmt(summary['sum_cycle_delay'], 6)} | "
            f"{summary['first_delayed_cycle']} |"
        )
    lines.append("")
    equivalence = effect["schedule_equivalence"]
    lines.append(
        f"- **动作不同但排程相同**：对齐干预在 {len(changed)} 个决策上改写了 align 标签，"
        f"但执行事实与未干预原 C 相比，开工时刻偏离 {equivalence['align_start_deviation_count']} 道工序、"
        f"团队偏离 {equivalence['align_team_deviation_count']} 道、转站偏离 "
        f"{equivalence['align_transfer_deviation_count']} 个周期——"
        + ("即标签改写未形成物理干预效果。" if equivalence["align_schedule_identical"] else "形成了实质排程差异。")
    )
    lines.append(
        f"- BC 在 S1/S2 与自身 Nominal 相比：开工偏离 "
        f"{equivalence['bc_s1_start_deviation_count']}/{equivalence['bc_s2_start_deviation_count']} 道，"
        f"团队偏离 {equivalence['bc_s1_team_deviation_count']}/{equivalence['bc_s2_team_deviation_count']} 道，"
        f"转站偏离 {equivalence['bc_s1_transfer_deviation_count']}/{equivalence['bc_s2_transfer_deviation_count']} 个周期。"
    )
    for run_id, label in (("BC_S1", "BC@S1"), ("BC_S2", "BC@S2")):
        rows = [r for r in cycle_rows_all if str(r["run_id"]) == run_id]
        if not rows:
            continue
        cycle_summary = summarize_cycle_delay(rows)
        payload = disturbed_payloads.get(run_id, {})
        makespan_shift = float(payload.get("makespan", 0.0)) - float(
            nominal_payloads["BC"]["makespan"]
        )
        if cycle_summary["delayed_cycle_count"] > 0 and abs(makespan_shift) <= PHYSICAL_TOLERANCE:
            verdict = "中间周期延迟被后续周期回补（catch-up）吸收，终点工期未变。"
        elif cycle_summary["delayed_cycle_count"] > 0:
            verdict = "中间周期延迟未完全回补，终点工期被推后。"
        else:
            verdict = "没有任何周期出现延迟。"
        lines.append(
            f"- **{label}**：{cycle_summary['delayed_cycle_count']} 个周期出现延迟，"
            f"最大 {fmt(cycle_summary['max_cycle_delay'], 6)}h、累计 {fmt(cycle_summary['sum_cycle_delay'], 6)}h，"
            f"首个延迟周期 {cycle_summary['first_delayed_cycle']}，终局工期偏移 "
            f"{fmt(makespan_shift, 6)}h → {verdict}"
        )
    first_deviations = {
        "BC@S1": equivalence.get("bc_s1_first_start_deviation"),
        "BC@S2": equivalence.get("bc_s2_first_start_deviation"),
        "align": equivalence.get("align_first_start_deviation"),
    }
    for label, deviation in first_deviations.items():
        if deviation is None:
            lines.append(f"- {label}：首次实际开工偏离为空（该任务未发生开工时刻变化或不可比）。")
        else:
            lines.append(
                f"- {label}：首次实际开工偏离在 `{deviation['task_key']}`，"
                f"{fmt(deviation['left_start'], 6)}h → {fmt(deviation['right_start'], 6)}h"
                f"（{fmt(deviation['delta'], 6)}h）。"
            )
    lines.append(
        "- **工期不变不等于中间无延迟**：上面的逐周期结论直接由 `cycle_comparison.csv` 的实际转站时刻算出，"
        "不能只看终点工期就判定“扰动没有影响”。"
    )
    lines.append("")

    lines.append("## 6. 结论与边界")
    lines.append("")
    lines.append("### 6.1 得到本轮证据支持的判断")
    lines.append("")
    for text in supported_claims(
        alignment=alignment,
        effect=effect,
        trajectory_verification=trajectory_verification,
        nominal_rows=nominal_rows,
        changed_count=len(changed),
        moved_count=len(moved),
    ):
        lines.append(f"- {text}")
    lines.append("")
    lines.append("### 6.2 未被隔离、不能宣称的内容")
    lines.append("")
    for text in unsupported_claims():
        lines.append(f"- {text}")
    lines.append("")
    lines.append("### 6.3 是否满足进入下一步讨论的证据条件")
    lines.append("")
    lines.append(
        "满足**讨论**条件：`protocol_status="
        f"{trajectory_verification['protocol_status']}`，三条新增轨迹的覆盖/物理约束/账本均由既有严格路径独立核验；"
        "权重与来源绑定有效。不满足**判定优劣**条件：单实例、单场景组、每场景一条轨迹，"
        "两个验证场景不是两个独立训练种子。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def classify_event_effect(evidence: Mapping[str, Any]) -> str:
    """把“命中”拆成：约束是否生效、是否传播到周期、是否传播到最终工期。"""
    parts: list[str] = []
    shift = evidence.get("start_shift_vs_own_nominal")
    satisfies = evidence.get("target_start_satisfies_recovery")
    if shift is None:
        parts.append("目标工序无可比开工事实")
    elif abs(float(shift)) <= PHYSICAL_TOLERANCE:
        parts.append("目标开工时刻未变，物料下界未形成可观测影响")
    elif satisfies:
        parts.append(
            f"目标开工被推后 {float(shift):.6f}h 且满足 $start \\ge R$，与物料下界生效一致"
        )
    else:
        parts.append(
            f"目标开工被推后 {float(shift):.6f}h 但仍早于 $R$，需另找原因"
        )
    delayed = evidence.get("delayed_cycles") or []
    if delayed:
        parts.append(f"延迟传播到周期 {delayed} 的转站")
    else:
        parts.append("未传播到任何周期的转站")
    makespan_shift = evidence.get("makespan_shift_vs_own_nominal")
    if makespan_shift is None:
        parts.append("终局工期不可比")
    elif abs(float(makespan_shift)) <= PHYSICAL_TOLERANCE:
        parts.append("未传播到最终工期（终点工期不变）")
    else:
        parts.append(f"最终工期被推后 {float(makespan_shift):.6f}h")
    if evidence.get("target_started_before_event"):
        parts.append("注意：目标工序在事件发生前已开工")
    return "；".join(parts) + "。"


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compare_schedules(
    nominal_payloads: Mapping[str, Mapping[str, Any]],
    disturbed_payloads: Mapping[str, Mapping[str, Any]],
    align_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """按 task_key / cycle_id 对齐比较执行事实，不做行号配对。"""
    align_vs_c = compare_execution_facts(
        nominal_payloads["C"]["execution_facts"], align_payload["execution_facts"]
    )
    bc_s1_vs_nominal = compare_execution_facts(
        nominal_payloads["BC"]["execution_facts"],
        disturbed_payloads["BC_S1"]["execution_facts"],
    )
    bc_s2_vs_nominal = compare_execution_facts(
        nominal_payloads["BC"]["execution_facts"],
        disturbed_payloads["BC_S2"]["execution_facts"],
    )
    return {
        "align_start_deviation_count": align_vs_c["start_deviation_count"],
        "align_team_deviation_count": align_vs_c["team_deviation_count"],
        "align_transfer_deviation_count": align_vs_c["transfer_deviation_count"],
        "align_schedule_identical": align_vs_c["schedule_identical"],
        "align_first_start_deviation": align_vs_c["first_start_deviation"],
        "align_first_transfer_deviation": align_vs_c["first_transfer_deviation"],
        "bc_s1_start_deviation_count": bc_s1_vs_nominal["start_deviation_count"],
        "bc_s1_team_deviation_count": bc_s1_vs_nominal["team_deviation_count"],
        "bc_s1_transfer_deviation_count": bc_s1_vs_nominal["transfer_deviation_count"],
        "bc_s1_first_start_deviation": bc_s1_vs_nominal["first_start_deviation"],
        "bc_s2_start_deviation_count": bc_s2_vs_nominal["start_deviation_count"],
        "bc_s2_team_deviation_count": bc_s2_vs_nominal["team_deviation_count"],
        "bc_s2_transfer_deviation_count": bc_s2_vs_nominal["transfer_deviation_count"],
        "bc_s2_first_start_deviation": bc_s2_vs_nominal["first_start_deviation"],
        "comparison_basis": "按 task_key 与 cycle_id 对齐；不按不同步数上的动作行号硬配对。",
    }


def supported_claims(
    *,
    alignment: Mapping[str, Any],
    effect: Mapping[str, Any],
    trajectory_verification: Mapping[str, Any],
    nominal_rows: Sequence[Mapping[str, Any]],
    changed_count: int,
    moved_count: int,
) -> list[str]:
    claims: list[str] = []
    align_row = next(
        (row for row in nominal_rows if row["run_id"] == ALIGN_METHOD), None
    )
    claims.append(
        f"`align` 只是执行分支的开工下界选项；本轮在 {changed_count} 个留站执行决策上把 0 改成 1，"
        f"其中 {moved_count} 个决策对应工序的实际开工被推后，"
        f"{changed_count - moved_count} 个没有变化（下界不紧）。"
    )
    claims.append(
        "对齐干预在原 C 权重与其余决策规则不变的前提下完成了一条合法 Nominal 轨迹："
        f"$J$ {fmt(alignment['j_original'])} → {fmt(alignment['j_align1'])}，"
        f"$\\Delta J_{{align}}$={fmt(alignment['delta_j'])}，"
        f"$\\Delta M_{{align}}$={fmt(alignment['delta_makespan'], 6)}h，"
        f"裁决为**{alignment['effect_status']}**；"
        f"其中 $J_{{takt}}$ 变化 {fmt(alignment['delta_j_takt'])}，$D_{{time}}$ 变化 {fmt(alignment['delta_d_time'])}。"
    )
    if align_row is not None:
        claims.append(
            "对齐干预后决策数 "
            f"{alignment['decisions_original']} → {alignment['decisions_align1']}，"
            f"后移 {align_row['postponed_count']} 次，物理违规 {align_row['constraint_violations']}。"
        )
    disturbed = effect["disturbance"]
    for scenario_id in VALIDATION_SCENARIO_IDS:
        bc = disturbed.get(f"BC@{scenario_id}", {})
        a = disturbed.get(f"A@{scenario_id}", {})
        if not bc:
            continue
        claims.append(
            f"冻结 BC 副本在 {scenario_id} 上从 reset 独立运行，"
            f"自然完工={fmt(bc.get('success'))}、违规={bc.get('constraint_violations')}、"
            f"$\\Delta J$={fmt(bc.get('delta_j_vs_own_nominal'))}（同场景 A 为 {fmt(a.get('delta_j_vs_own_nominal'))}）；"
            f"该次运行前后 Actor 指纹一致，未训练、未注入教师动作、未叠加对齐干预。"
        )
    claims.append(
        f"三条新增轨迹的协议核验状态为 `{trajectory_verification['protocol_status']}`"
        f"（退出码 {trajectory_verification['exit_code']}），覆盖/物理约束/费用账本由既有严格路径独立复算。"
    )
    gaps: list[str] = []
    for scenario_id in VALIDATION_SCENARIO_IDS:
        a_item = disturbed.get(f"A@{scenario_id}") or {}
        bc_item = disturbed.get(f"BC@{scenario_id}") or {}
        if not a_item or not bc_item:
            continue
        gaps.append(
            f"{scenario_id} 上 {float(bc_item['delta_j_vs_own_nominal']) - float(a_item['delta_j_vs_own_nominal']):.3e}"
        )
    if gaps:
        claims.append(
            "两条受扰运行均自然完工、0 违规；BC 与 A 的扰动增量差距分别为 "
            + "、".join(gaps)
            + "。在这两个**既有**验证事件上，冻结副本表现出与 A 接近的有限保持能力；"
            "该判断不外推到未知扰动、恢复最优性或统计显著性。"
        )
    return claims


def unsupported_claims() -> list[str]:
    return [
        "不能把费用变化解释成“对齐头的独立贡献占比”：干预后访问的状态、后续工序与团队选择都会变化。",
        "不能宣称原 C 其他策略头没有问题，也不能宣称 PPO 的根因已唯一定位。",
        "不能由两个既有验证扰动推断未知扰动泛化、恢复最优性或强化学习优越性。",
        "两个验证场景不是两个独立训练种子，不能当作统计重复。",
        "不能把本轮结论外推到其他实例、其他基准版本或其他时钟策略。",
    ]


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def run_all_stages() -> int:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}
    for stage in STAGES:
        log(f"=== 子进程执行阶段: {stage} ===")
        log_path = RUNS_DIR / f"{stage}.log"
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--stage", stage],
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        results[stage] = process.returncode
        log(f"阶段 {stage} 退出码 {process.returncode}")
        if process.returncode != 0:
            dump_json(RUNS_DIR / "stage_results.json", results)
            return process.returncode
    dump_json(RUNS_DIR / "stage_results.json", results)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Work-3 对齐干预与冻结 BC 受扰诊断")
    parser.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    args = parser.parse_args()
    if args.stage == "all":
        return run_all_stages()
    handlers = {
        "inventory": stage_inventory,
        "align": stage_align,
        "disturbance": stage_disturbance,
        "verify": stage_verify,
        "report": stage_report,
    }
    started = time.perf_counter()
    code = handlers[args.stage]()
    log(f"阶段 {args.stage} 用时 {time.perf_counter() - started:.1f}s，退出码 {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
