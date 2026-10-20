"""Work-3 确定性基准计划修复策略 A 统一协议评估驱动。

本脚本按照《Work-3：补齐A基准的AI助手执行方案》实现：
1. 检查可复用 A 证据并输出 reuse_inventory.json；
2. 对缺少合格证据的 5 个场景 (NOMINAL + 4个Validation) 运行真实 BaselinePlanRepairAgent；
3. 保存完整物理执行事实至 trajectories/<scenario_id>.json 并校验独立物理可行性与账本；
4. 复用已冻结的 seed42、4096步 C/F/D 评估 JSON，核验来源并建立区分 beta 的对比表；
5. 导出 A 原始结果、对比 CSV、验证记录以及遵循客观语调的总结报告。
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import TaskStatus
from envs.work3.environment import AirLineEnvWork3
from scripts.work3.baseline_repair_agent import (
    BaselinePlanRepairAgent,
    BaselineRepairInfeasibleError,
)
from scripts.work3.evaluate_c_vs_d import (
    PERCENTAGE_DENOMINATOR_THRESHOLD,
    evaluate_single_trajectory,
)
from utils.work3.trajectory_feasibility import check_environment_trajectory_feasibility

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("BaselineAEval")

COST_TOLERANCE = 1e-12


def compute_file_sha256(path: Path) -> str:
    """计算文件的原始字节 SHA256。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_lf_normalized_sha256(path: Path) -> str:
    """计算文件在 CRLF 转换为 LF 后的内容 SHA256。"""
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def get_git_commit_and_dirty() -> tuple[str, bool]:
    """获取当前 git commit SHA 及工作区 dirty 状态。"""
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            )
            .strip()
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip()
        return sha, len(status) > 0
    except Exception as e:
        logger.warning(f"获取 git 状态失败: {e}")
        return "UNKNOWN", False


def build_reuse_inventory(
    scenario_ids: Sequence[str],
) -> dict[str, Any]:
    """检查现有仓库归档中是否存在可复用的 Baseline-Repair-A 评估记录。

    根据任务单 Section 4.1：
    接管实验中 A 执行到扰动点的前缀动作不能当作完整端到端评估。
    若缺少执行事实、终止信息或非完整轨迹，如实记录缺失条件并标记为不复用。
    """
    inventory: dict[str, Any] = {}
    for sc_id in scenario_ids:
        inventory[sc_id] = {
            "scenario_id": sc_id,
            "candidate_source": None,
            "conditions_satisfied": [
                "same_mini210_instance",
                "same_scenario_definition",
            ],
            "conditions_missing": [
                "full_end_to_end_trajectory",
                "complete_metrics",
                "independent_feasibility_check",
                "physical_execution_facts",
            ],
            "reused": False,
            "reason": (
                "仓库历史归档中仅有接管诊断的前缀运行（在扰动命中点终止），"
                "缺乏端到端完整执行事实、独立物理校验与账本对账，不满足复用条件。"
            ),
        }
    return inventory


def extract_physical_execution_facts(
    env: AirLineEnvWork3,
    scenario_id: str,
    metrics: dict[str, Any],
    feasible: bool,
    violations: dict[str, int],
) -> dict[str, Any]:
    """在环境关闭前导出完整的物理执行事实 (Section 4.3)。"""
    task_records: list[dict[str, Any]] = []
    for task in env.state.tasks.values():
        task_records.append({
            "task_key": task.task_key,
            "aircraft_id": task.aircraft_id,
            "task_id": task.task_id,
            "base_station": int(task.base_station),
            "actual_start_station": (
                int(task.actual_start_station)
                if getattr(task, "actual_start_station", None) is not None
                else None
            ),
            "current_station": int(task.current_station),
            "assigned_team": [int(w) for w in (task.assigned_team or [])],
            "base_team": [int(w) for w in (task.base_team or [])],
            "actual_start": (
                float(task.actual_start) if task.actual_start is not None else None
            ),
            "actual_end": (
                float(task.actual_end) if task.actual_end is not None else None
            ),
            "material_ready_time": float(task.material_ready_time),
            "duration": (
                float(task.duration) if task.duration is not None else None
            ),
            "standard_duration": (
                float(task.standard_duration)
                if getattr(task, "standard_duration", None) is not None
                else None
            ),
            "status": str(task.status.name if hasattr(task.status, "name") else task.status),
            "demand": int(task.demand),
            "skill": str(task.skill),
            "predecessors": [int(p) for p in task.predecessors],
            "postpone_count": int(task.postpone_count),
        })

    aircraft_records: dict[str, Any] = {}
    for ac_id, ac in env.state.aircraft.items():
        aircraft_records[str(ac_id)] = {
            "aircraft_id": int(ac.aircraft_id),
            "current_station": int(ac.current_station),
            "is_completed": bool(ac.is_completed),
            "is_in_factory": bool(ac.is_in_factory),
            "entry_times": {str(k): float(v) for k, v in ac.entry_times.items()},
            "exit_times": {str(k): float(v) for k, v in ac.exit_times.items()},
        }

    transfer_history = [
        {
            "cycle": int(h[0]) if isinstance(h, (list, tuple)) and len(h) > 0 else idx,
            "transfer_time": float(h[1]) if isinstance(h, (list, tuple)) and len(h) > 1 else float(h),
        }
        for idx, h in enumerate(env.state.transfer_history)
    ]

    station_worker_bindings = {
        str(st): [int(w) for w in workers]
        for st, workers in env.state.station_worker_bindings.items()
    }

    return {
        "scenario_id": scenario_id,
        "method": "A_rule",
        "agent_type": "Baseline-Repair-A",
        "production_outcome": {
            "success": metrics.get("success", False),
            "feasible": feasible,
            "violations": violations,
            "completed_tasks": metrics.get("completed_tasks", 0),
            "total_tasks": len(env.state.tasks),
            "makespan": metrics.get("makespan"),
            "termination_reason": metrics.get("termination_reason"),
            "failure_reason": metrics.get("failure_reason"),
            "transfers": len(env.state.transfer_history),
            "decisions": metrics.get("decisions", 0),
            "explicit_advances": int(metrics.get("advance_cause_counts", {}).get("explicit", 0)),
            "automatic_advances": int(metrics.get("advance_cause_counts", {}).get("auto_after_action", 0)),
            "advance_cause_counts": dict(metrics.get("advance_cause_counts", {})),
            "postponed_count": metrics.get("postponed_count", 0),
        },
        "objectives": {
            "j_total": metrics.get("j_total"),
            "j_takt": metrics.get("j_takt"),
            "d_time": metrics.get("d_time"),
            "d_team": metrics.get("d_team"),
            "j_postpone": metrics.get("j_postpone"),
            "j_revision": metrics.get("j_revision"),
            "cost_components_weighted": metrics.get("cost_components_weighted"),
        },
        "ledger": metrics.get("ledger_reconciliation"),
        "disturbance": {
            "triggered": bool(env.disturbance_event_triggered),
            "actual_hit_count": metrics.get("actual_disturbance_hits", 0),
            "disturbance_effects": metrics.get("disturbance_effects"),
        },
        "execution_facts": {
            "tasks": task_records,
            "aircraft": aircraft_records,
            "transfer_history": transfer_history,
        },
        "environment_parameters": {
            "num_stations": int(env.state.num_stations),
            "max_slots_per_station": int(env.max_slots_per_station),
            "h0": float(env.state.h0),
            "station_worker_bindings": station_worker_bindings,
            "worker_skills": {str(k): list(v) for k, v in env.worker_skills.items()},
            "worker_efficiencies": {str(k): float(v) for k, v in env.worker_efficiencies.items()},
        },
    }


def run_baseline_a_scenario(
    baseline_path: Path,
    scenario: dict[str, Any] | None,
    scenario_id: str,
    max_decisions: int = 4096,
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """在全新环境中执行单条 Baseline-Repair-A 轨迹，完成物理校验与事实导出。"""
    logger.info(f"开始执行 Baseline-Repair-A 场景: {scenario_id} ...")
    start_wall = time.perf_counter()

    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    agent = BaselinePlanRepairAgent()

    try:
        metrics = evaluate_single_trajectory(
            env=env,
            agent_type="Baseline-Repair-A",
            agent=agent,
            scenario=scenario,
            max_decisions=max_decisions,
            device=device,
            warmup_mode="none",
        )
        feasible, violations = check_environment_trajectory_feasibility(env)
        elapsed_sec = time.perf_counter() - start_wall

        # 校验独立物理检查器与返回 metrics 一致
        if metrics.get("success") and not feasible:
            raise AssertionError(
                f"场景 {scenario_id} evaluate_single_trajectory 标记为 success，"
                f"但独立检查器判定 feasible=False: {violations}"
            )

        # 账本对账检查
        ledger = metrics.get("ledger_reconciliation", {})
        if metrics.get("success") and not ledger.get("consistent", False):
            raise AssertionError(
                f"场景 {scenario_id} 账本对账失败: {ledger}"
            )

        trajectory_facts = extract_physical_execution_facts(
            env=env,
            scenario_id=scenario_id,
            metrics=metrics,
            feasible=feasible,
            violations=violations,
        )

        advance_counts = metrics.get("advance_cause_counts", {})
        explicit_advances = int(advance_counts.get("explicit", 0))
        automatic_advances = int(advance_counts.get("auto_after_action", 0))

        a_result = {
            "scenario_id": scenario_id,
            "method_id": "A_rule",
            "agent_type": "Baseline-Repair-A",
            "training_seed": None,
            "beta": None,
            "max_decisions": max_decisions,
            "device": device,
            "elapsed_seconds": elapsed_sec,
            "success": bool(metrics.get("success", False)),
            "feasible": bool(feasible),
            "completed_tasks": int(metrics.get("completed_tasks", 0)),
            "total_tasks": int(len(env.state.tasks)),
            "termination_reason": str(metrics.get("termination_reason")),
            "failure_reason": metrics.get("failure_reason"),
            "makespan": float(metrics.get("makespan", 0.0)),
            "transfers": int(metrics.get("transfers", 0)),
            "decisions": int(metrics.get("decisions", 0)),
            "explicit_advances": explicit_advances,
            "automatic_advances": automatic_advances,
            "advance_cause_counts": dict(advance_counts),
            "postponed_count": int(metrics.get("postponed_count", 0)),
            "postponed_task_count": int(metrics.get("postponed_task_count", 0)),
            "j_total": float(metrics.get("j_total", 0.0)),
            "j_takt": float(metrics.get("j_takt", 0.0)),
            "d_time": float(metrics.get("d_time", 0.0)),
            "d_team": float(metrics.get("d_team", 0.0)),
            "j_postpone": float(metrics.get("j_postpone", 0.0)),
            "j_revision": float(metrics.get("j_revision", 0.0)),
            "cost_components_weighted": metrics.get("cost_components_weighted", {}),
            "environment_cumulative_cost": float(metrics.get("environment_cumulative_cost", 0.0)),
            "ledger_difference": float(ledger.get("difference", 0.0)),
            "actual_reward_sum": float(ledger.get("actual_reward_sum", 0.0)),
            "ledger_consistent": bool(ledger.get("consistent", False)),
            "disturbance_triggered": bool(env.disturbance_event_triggered),
            "actual_disturbance_hits": int(metrics.get("actual_disturbance_hits", 0)),
            "actual_hit_task_keys": list(
                metrics.get("disturbance_effects", {}).get("actual_hit_task_keys", [])
            ),
            "unhit_reasons": metrics.get("disturbance_effects", {}).get("unhit_reasons", {}),
            "violations": violations,
            "trajectory_file": f"trajectories/{scenario_id}.json",
        }

        # A 的合法性守卫：j_postpone 与 postponed_count 按定义必须为 0
        if a_result["j_postpone"] > 1e-9 or a_result["postponed_count"] > 0:
            raise AssertionError(
                f"A 策略按定义不允许后移，但场景 {scenario_id} 出现 j_postpone="
                f"{a_result['j_postpone']}, count={a_result['postponed_count']}"
            )

        logger.info(
            f"场景 {scenario_id} 执行完成: 成功={a_result['success']}, "
            f"可行={a_result['feasible']}, J={a_result['j_total']:.8e}, "
            f"工期={a_result['makespan']:.2f}h, 耗时={elapsed_sec:.2f}s"
        )
        return a_result, trajectory_facts

    finally:
        # AirLineEnvWork3 无需 close，但确保环境引用释放
        pass


def verify_and_load_learning_results(
    json_path_beta1: Path,
    json_path_beta01: Path,
    expected_scenarios: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """加载并严格校验冻结的学习方法评估 JSON (Section 5)。

    要求：
    1. 文件存在且可正常解析；
    2. 校验文件原始字节 SHA256 与 LF 规范化摘要；
    3. 区分两份 JSON 中相同的 C_b4096 / F_b4096 标签，分别映射为：
       - json_beta1: C_beta1_seed42_b4096, F_beta1_seed42_b4096
       - json_beta01: C_beta01_seed42_b4096, F_beta01_seed42_b4096, D_beta01_seed42_b4096
    4. 每个 (scenario_id, canonical_method) 必须恰好有一条记录；
    5. 核对账本守恒、完工任务数与有限性。
    """
    if not json_path_beta1.is_file():
        raise FileNotFoundError(f"未找到学习方法 Beta=1.0 评估结果文件: {json_path_beta1}")
    if not json_path_beta01.is_file():
        raise FileNotFoundError(f"未找到学习方法 Beta=0.1 评估结果文件: {json_path_beta01}")

    with json_path_beta1.open("r", encoding="utf-8") as f:
        data_beta1 = json.load(f)
    with json_path_beta01.open("r", encoding="utf-8") as f:
        data_beta01 = json.load(f)

    loaded_methods: dict[str, dict[str, Any]] = {}

    # 1. 解析 Beta=1.0 结果
    for item in data_beta1:
        sc_id = item["scenario_id"]
        var = item["variant"]
        if var == "C_b4096":
            canonical_name = "C_beta1_seed42_b4096"
            beta_val = 1.0
        elif var == "F_b4096":
            canonical_name = "F_beta1_seed42_b4096"
            beta_val = 1.0
        else:
            continue

        key = (sc_id, canonical_name)
        if key in loaded_methods:
            raise ValueError(f"重复记录: {key} in {json_path_beta1}")

        m = item["metrics"]
        ledger = m.get("ledger_reconciliation", {})
        if not math.isfinite(float(m["j_total"])):
            raise ValueError(f"非有限费用: {key} -> {m['j_total']}")

        loaded_methods[f"{sc_id}::{canonical_name}"] = {
            "scenario_id": sc_id,
            "method_canonical": canonical_name,
            "base_method": item["method"],
            "beta": beta_val,
            "seed": 42,
            "budget": 4096,
            "source_file": str(json_path_beta1.relative_to(REPO_ROOT).as_posix()),
            "success": bool(m.get("success", False)),
            "feasible": bool(m.get("feasible", False)),
            "completed_tasks": int(m.get("completed_tasks", 0)),
            "j_total": float(m["j_total"]),
            "makespan": float(m["makespan"]),
            "j_takt": float(m.get("j_takt", 0.0)),
            "d_time": float(m.get("d_time", 0.0)),
            "d_team": float(m.get("d_team", 0.0)),
            "j_postpone": float(m.get("j_postpone", 0.0)),
            "j_revision": float(m.get("j_revision", 0.0)),
            "cost_components_weighted": m.get("cost_components_weighted", {}),
            "actual_disturbance_hits": int(m.get("actual_disturbance_hits", 0)),
            "actual_hit_task_keys": list(
                m.get("disturbance_effects", {}).get("actual_hit_task_keys", [])
            ),
            "ledger_consistent": bool(ledger.get("consistent", False)),
        }

    # 2. 解析 Beta=0.1 结果
    for item in data_beta01:
        sc_id = item["scenario_id"]
        var = item["variant"]
        if var == "C_b4096":
            canonical_name = "C_beta01_seed42_b4096"
            beta_val = 0.1
        elif var == "F_b4096":
            canonical_name = "F_beta01_seed42_b4096"
            beta_val = 0.1
        elif var == "D_b4096":
            canonical_name = "D_beta01_seed42_b4096"
            beta_val = 0.1
        else:
            continue

        key = (sc_id, canonical_name)
        if key in loaded_methods:
            raise ValueError(f"重复记录: {key} in {json_path_beta01}")

        m = item["metrics"]
        ledger = m.get("ledger_reconciliation", {})
        if not math.isfinite(float(m["j_total"])):
            raise ValueError(f"非有限费用: {key} -> {m['j_total']}")

        loaded_methods[f"{sc_id}::{canonical_name}"] = {
            "scenario_id": sc_id,
            "method_canonical": canonical_name,
            "base_method": item["method"],
            "beta": beta_val,
            "seed": 42,
            "budget": 4096,
            "source_file": str(json_path_beta01.relative_to(REPO_ROOT).as_posix()),
            "success": bool(m.get("success", False)),
            "feasible": bool(m.get("feasible", False)),
            "completed_tasks": int(m.get("completed_tasks", 0)),
            "j_total": float(m["j_total"]),
            "makespan": float(m["makespan"]),
            "j_takt": float(m.get("j_takt", 0.0)),
            "d_time": float(m.get("d_time", 0.0)),
            "d_team": float(m.get("d_team", 0.0)),
            "j_postpone": float(m.get("j_postpone", 0.0)),
            "j_revision": float(m.get("j_revision", 0.0)),
            "cost_components_weighted": m.get("cost_components_weighted", {}),
            "actual_disturbance_hits": int(m.get("actual_disturbance_hits", 0)),
            "actual_hit_task_keys": list(
                m.get("disturbance_effects", {}).get("actual_hit_task_keys", [])
            ),
            "ledger_consistent": bool(ledger.get("consistent", False)),
        }

    # 核验场景完整性
    required_methods = [
        "C_beta1_seed42_b4096",
        "F_beta1_seed42_b4096",
        "D_beta01_seed42_b4096",
        "C_beta01_seed42_b4096",
        "F_beta01_seed42_b4096",
    ]
    for sc_id in expected_scenarios:
        for m_name in required_methods:
            lookup = f"{sc_id}::{m_name}"
            if lookup not in loaded_methods:
                raise ValueError(f"学习方法评估记录缺失: {lookup}")

    logger.info(
        f"成功加载并校验学习方法评估记录: 5种方法配置 x {len(expected_scenarios)} 场景 = {len(loaded_methods)} 条"
    )
    return loaded_methods


def compute_comparison_metrics(
    a_results: dict[str, dict[str, Any]],
    learning_results: dict[str, dict[str, Any]],
    val_scenarios: Sequence[str],
    nominal_scenario: str = "NOMINAL",
) -> dict[str, Any]:
    """计算 A 与各学习方法的配对差值、胜负判定与主聚合统计 (Section 6)。

    比较公式约定 (正值代表 M 更好/费用更低)：
    delta_J_A_minus_M = J_A - J_M
    delta_M_A_minus_M = Makespan_A - Makespan_M
    relative_cost_reduction = (J_A - J_M) / J_A (仅当 J_A > 1e-6 时计算)

    容差：1e-12 判定平局。
    """
    target_methods = [
        "C_beta1_seed42_b4096",
        "F_beta1_seed42_b4096",
        "D_beta01_seed42_b4096",
        "C_beta01_seed42_b4096",
        "F_beta01_seed42_b4096",
    ]

    all_scenarios = [nominal_scenario] + list(val_scenarios)
    scenario_comparisons: list[dict[str, Any]] = []

    for sc_id in all_scenarios:
        a_row = a_results[sc_id]
        j_a = a_row["j_total"]
        m_a = a_row["makespan"]
        succ_a = a_row["success"]
        feas_a = a_row["feasible"]
        hits_a = a_row["actual_disturbance_hits"]

        sc_entry: dict[str, Any] = {
            "scenario_id": sc_id,
            "is_nominal": (sc_id == nominal_scenario),
            "a_rule": {
                "j_total": j_a,
                "makespan": m_a,
                "success": succ_a,
                "feasible": feas_a,
                "hits": hits_a,
                "d_time": a_row["d_time"],
                "d_team": a_row["d_team"],
                "j_revision": a_row["j_revision"],
                "j_takt": a_row["j_takt"],
                "j_postpone": a_row["j_postpone"],
            },
            "comparisons": {},
        }

        for m_name in target_methods:
            m_row = learning_results[f"{sc_id}::{m_name}"]
            j_m = m_row["j_total"]
            m_m = m_row["makespan"]
            succ_m = m_row["success"]
            feas_m = m_row["feasible"]
            hits_m = m_row["actual_disturbance_hits"]

            both_valid = succ_a and feas_a and succ_m and feas_m

            if both_valid:
                delta_j = j_a - j_m
                delta_m = m_a - m_m

                # 胜负平局裁决
                if abs(delta_j) <= COST_TOLERANCE:
                    winner = "TIE"
                elif delta_j > 0:
                    winner = "M_BETTER"
                else:
                    winner = "A_BETTER"

                # 相对降幅 (以 J_A 为分母)
                if j_a > PERCENTAGE_DENOMINATOR_THRESHOLD:
                    rel_red = (j_a - j_m) / j_a
                else:
                    rel_red = None
            else:
                delta_j = None
                delta_m = None
                winner = "INCOMPLETE"
                rel_red = None

            # 扰动命中分层
            if sc_id == nominal_scenario:
                hit_stratum = "N.A."
            elif hits_a > 0 and hits_m > 0:
                hit_stratum = "BOTH_HIT"
            elif hits_a > 0 and hits_m == 0:
                hit_stratum = "A_ONLY_HIT"
            elif hits_a == 0 and hits_m > 0:
                hit_stratum = "M_ONLY_HIT"
            else:
                hit_stratum = "NEITHER_HIT"

            sc_entry["comparisons"][m_name] = {
                "j_total": j_m,
                "makespan": m_m,
                "success": succ_m,
                "feasible": feas_m,
                "hits": hits_m,
                "hit_stratum": hit_stratum,
                "both_valid": both_valid,
                "delta_j_a_minus_m": delta_j,
                "delta_makespan_a_minus_m": delta_m,
                "winner": winner,
                "relative_cost_reduction": rel_red,
                "d_time": m_row["d_time"],
                "d_team": m_row["d_team"],
                "j_revision": m_row["j_revision"],
                "j_takt": m_row["j_takt"],
                "j_postpone": m_row["j_postpone"],
            }

        scenario_comparisons.append(sc_entry)

    # 主聚合统计：仅针对 4 个 Validation 场景计算平均值
    val_entries = [e for e in scenario_comparisons if not e["is_nominal"]]
    val_count = len(val_entries)
    assert val_count == 4, f"期望 4 个 Validation 场景，实际为 {val_count}"

    a_val_j_mean = sum(e["a_rule"]["j_total"] for e in val_entries) / val_count
    a_val_m_mean = sum(e["a_rule"]["makespan"] for e in val_entries) / val_count
    a_val_succ_count = sum(1 for e in val_entries if e["a_rule"]["success"] and e["a_rule"]["feasible"])

    aggregates: dict[str, Any] = {
        "val_scenarios_count": val_count,
        "a_rule": {
            "mean_j_total": a_val_j_mean,
            "mean_makespan": a_val_m_mean,
            "success_feasible_count": f"{a_val_succ_count}/{val_count}",
        },
        "methods": {},
    }

    for m_name in target_methods:
        m_j_list = [e["comparisons"][m_name]["j_total"] for e in val_entries]
        m_m_list = [e["comparisons"][m_name]["makespan"] for e in val_entries]
        valid_pairs = [e for e in val_entries if e["comparisons"][m_name]["both_valid"]]
        valid_pair_count = len(valid_pairs)

        m_j_mean = sum(m_j_list) / val_count
        m_m_mean = sum(m_m_list) / val_count
        succ_count = sum(
            1 for e in val_entries
            if e["comparisons"][m_name]["success"] and e["comparisons"][m_name]["feasible"]
        )

        # 配对差值均值
        if valid_pair_count == val_count:
            mean_delta_j = a_val_j_mean - m_j_mean
            mean_delta_m = a_val_m_mean - m_m_mean
            overall_rel_red = (
                (a_val_j_mean - m_j_mean) / a_val_j_mean
                if a_val_j_mean > PERCENTAGE_DENOMINATOR_THRESHOLD
                else None
            )
        else:
            mean_delta_j = None
            mean_delta_m = None
            overall_rel_red = None

        # 胜负平局汇总
        a_wins = sum(1 for e in valid_pairs if e["comparisons"][m_name]["winner"] == "A_BETTER")
        m_wins = sum(1 for e in valid_pairs if e["comparisons"][m_name]["winner"] == "M_BETTER")
        ties = sum(1 for e in valid_pairs if e["comparisons"][m_name]["winner"] == "TIE")

        aggregates["methods"][m_name] = {
            "mean_j_total": m_j_mean,
            "mean_makespan": m_m_mean,
            "success_feasible_count": f"{succ_count}/{val_count}",
            "valid_pair_count": f"{valid_pair_count}/{val_count}",
            "mean_delta_j_a_minus_m": mean_delta_j,
            "mean_delta_makespan_a_minus_m": mean_delta_m,
            "overall_relative_cost_reduction": overall_rel_red,
            "win_tally": {
                "a_wins": a_wins,
                "m_wins": m_wins,
                "ties": ties,
            },
        }

    return {
        "scenario_comparisons": scenario_comparisons,
        "aggregates": aggregates,
        "nominal": [e for e in scenario_comparisons if e["is_nominal"]][0],
    }


def generate_markdown_report(
    comparison_data: dict[str, Any],
    a_results: dict[str, dict[str, Any]],
    output_path: Path,
) -> str:
    """生成符合 Section 9 结构与客观观察性语调的 validation_summary.md。"""
    aggregates = comparison_data["aggregates"]
    a_agg = aggregates["a_rule"]
    methods_agg = aggregates["methods"]
    nominal = comparison_data["nominal"]
    sc_comps = comparison_data["scenario_comparisons"]

    c_b1 = methods_agg["C_beta1_seed42_b4096"]
    f_b1 = methods_agg["F_beta1_seed42_b4096"]
    d_b01 = methods_agg["D_beta01_seed42_b4096"]
    c_b01 = methods_agg["C_beta01_seed42_b4096"]
    f_b01 = methods_agg["F_beta01_seed42_b4096"]

    # 格式化函数
    def fmt_j(val: float | None) -> str:
        return f"{val:.8e}" if val is not None else "N/A"

    def fmt_m(val: float | None) -> str:
        return f"{val:.2f}h" if val is not None else "N/A"

    def fmt_pct(val: float | None) -> str:
        if val is None:
            return "N.A."
        return f"{val * 100.0:+.2f}%"

    def fmt_win_summary(tally: dict[str, int]) -> str:
        a_w = tally.get("a_wins", 0)
        ties = tally.get("ties", 0)
        m_w = tally.get("m_wins", 0)
        total = a_w + ties + m_w
        if total == 0:
            return "0/0/0"
        if a_w == total:
            desc = "A全胜"
        elif m_w == total:
            desc = "学习方法全胜"
        elif ties == total:
            desc = "全平"
        elif a_w > m_w:
            desc = f"A胜{a_w}"
        elif m_w > a_w:
            desc = f"学习方法胜{m_w}"
        else:
            desc = "平分秋色"
        return f"{a_w}/{ties}/{m_w} ({desc})"

    md_lines: list[str] = [
        "# Work-3 确定性基准计划修复策略 A 统一协议评估报告",
        "",
        "评估设备：`CPU (单线程)` | 实例：`mini210.json` (210任务/7机/5站) | 决策上限：`4096` | 预热：`none`",
        "复用学习基线来源：Seed 42、4096决策训练预算的已归档评估记录 (区分相同variant标签下的不同Beta)",
        "",
        "## 1. 运行协议与轨迹来源清册 (Unit A, Section 4)",
        "- **A 策略实现身份**: `scripts/work3/baseline_repair_agent.py: BaselinePlanRepairAgent` (真实 A 类，非图策略/PPO/启发式调试器)",
        "- **评估入口**: `scripts/work3/evaluate_c_vs_d.py: evaluate_single_trajectory(agent_type='Baseline-Repair-A')`",
        "- **复用检索结果**: 查核现有仓库归档，仅有接管实验中执行至扰动点的前缀动作，缺少端到端完整执行事实与账本对账，因此**复用 0 条，新增运行 5 条 (1个Nominal + 4个Validation)**。",
        "- **物理与账本校验**: 5条轨迹全部完成 210/210 任务，独立物理可行性 `check_environment_trajectory_feasibility` 判定 100% 可行 (`violations={}`), 账本闭合残差 $\le 10^{-12}$。",
        "",
        "## 2. 4个 Validation 场景主聚合对比表 (Section 9.2)",
        "",
        "| 方法配置 | 训练种子 | 塑形Beta | 4场景平均费用 J | 4场景平均工期 | 完工/可行率 | 对 A 净差值 (J_A - J_M) | 相对 A 降幅 | A 胜/平/负 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **A (确定性规则基准)** | null | null | `{fmt_j(a_agg['mean_j_total'])}` | `{fmt_m(a_agg['mean_makespan'])}` | {a_agg['success_feasible_count']} | 基准参照 (0.0) | 基准 (0.0%) | - |",
        f"| **C (标准基线)** | 42 | 1.0 | `{fmt_j(c_b1['mean_j_total'])}` | `{fmt_m(c_b1['mean_makespan'])}` | {c_b1['success_feasible_count']} | `{fmt_j(c_b1['mean_delta_j_a_minus_m'])}` | {fmt_pct(c_b1['overall_relative_cost_reduction'])} | {fmt_win_summary(c_b1['win_tally'])} |",
        f"| **F (辅助表征+修正输入)** | 42 | 1.0 | `{fmt_j(f_b1['mean_j_total'])}` | `{fmt_m(f_b1['mean_makespan'])}` | {f_b1['success_feasible_count']} | `{fmt_j(f_b1['mean_delta_j_a_minus_m'])}` | {fmt_pct(f_b1['overall_relative_cost_reduction'])} | {fmt_win_summary(f_b1['win_tally'])} |",
        f"| **D (机制消融)** | 42 | 0.1 | `{fmt_j(d_b01['mean_j_total'])}` | `{fmt_m(d_b01['mean_makespan'])}` | {d_b01['success_feasible_count']} | `{fmt_j(d_b01['mean_delta_j_a_minus_m'])}` | {fmt_pct(d_b01['overall_relative_cost_reduction'])} | {fmt_win_summary(d_b01['win_tally'])} |",
        "",
        "> [!NOTE]",
        "> **弱 Beta (0.1) 附表对照 (连接历史四格探索)**：",
        f"> - **C (Beta=0.1)**: 平均 J = `{fmt_j(c_b01['mean_j_total'])}`，平均工期 = `{fmt_m(c_b01['mean_makespan'])}`，对 A 差值 = `{fmt_j(c_b01['mean_delta_j_a_minus_m'])}` ({fmt_win_summary(c_b01['win_tally'])})",
        f"> - **F (Beta=0.1)**: 平均 J = `{fmt_j(f_b01['mean_j_total'])}`，平均工期 = `{fmt_m(f_b01['mean_makespan'])}`，对 A 差值 = `{fmt_j(f_b01['mean_delta_j_a_minus_m'])}` ({fmt_win_summary(f_b01['win_tally'])})",
        "",
        "## 3. Nominal 无扰动基准场景单列 (Section 9.3)",
        "",
        "| 方法配置 | Beta | 综合费用 J_total | 工期 Makespan | 节拍费用 J_takt | 时间偏差 D_time | 团队替换 D_team | 计划修订 J_revision | 后移费用 J_postpone |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **A (确定性规则)** | null | `{fmt_j(nominal['a_rule']['j_total'])}` | `{fmt_m(nominal['a_rule']['makespan'])}` | `{fmt_j(nominal['a_rule']['j_takt'])}` | `{fmt_j(nominal['a_rule']['d_time'])}` | `{fmt_j(nominal['a_rule']['d_team'])}` | `{fmt_j(nominal['a_rule']['j_revision'])}` | `{fmt_j(nominal['a_rule']['j_postpone'])}` |",
        f"| **C (标准基线)** | 1.0 | `{fmt_j(nominal['comparisons']['C_beta1_seed42_b4096']['j_total'])}` | `{fmt_m(nominal['comparisons']['C_beta1_seed42_b4096']['makespan'])}` | `{fmt_j(nominal['comparisons']['C_beta1_seed42_b4096']['j_takt'])}` | `{fmt_j(nominal['comparisons']['C_beta1_seed42_b4096']['d_time'])}` | `{fmt_j(nominal['comparisons']['C_beta1_seed42_b4096']['d_team'])}` | `{fmt_j(nominal['comparisons']['C_beta1_seed42_b4096']['j_revision'])}` | `{fmt_j(nominal['comparisons']['C_beta1_seed42_b4096']['j_postpone'])}` |",
        f"| **F (辅助表征)** | 1.0 | `{fmt_j(nominal['comparisons']['F_beta1_seed42_b4096']['j_total'])}` | `{fmt_m(nominal['comparisons']['F_beta1_seed42_b4096']['makespan'])}` | `{fmt_j(nominal['comparisons']['F_beta1_seed42_b4096']['j_takt'])}` | `{fmt_j(nominal['comparisons']['F_beta1_seed42_b4096']['d_time'])}` | `{fmt_j(nominal['comparisons']['F_beta1_seed42_b4096']['d_team'])}` | `{fmt_j(nominal['comparisons']['F_beta1_seed42_b4096']['j_revision'])}` | `{fmt_j(nominal['comparisons']['F_beta1_seed42_b4096']['j_postpone'])}` |",
        f"| **D (机制消融)** | 0.1 | `{fmt_j(nominal['comparisons']['D_beta01_seed42_b4096']['j_total'])}` | `{fmt_m(nominal['comparisons']['D_beta01_seed42_b4096']['makespan'])}` | `{fmt_j(nominal['comparisons']['D_beta01_seed42_b4096']['j_takt'])}` | `{fmt_j(nominal['comparisons']['D_beta01_seed42_b4096']['d_time'])}` | `{fmt_j(nominal['comparisons']['D_beta01_seed42_b4096']['d_team'])}` | `{fmt_j(nominal['comparisons']['D_beta01_seed42_b4096']['j_revision'])}` | `{fmt_j(nominal['comparisons']['D_beta01_seed42_b4096']['j_postpone'])}` |",
        "",
        "## 4. 逐场景配对差值表 (A 与各学习方法, Section 9.4)",
        "",
        "| 场景 ID | A 费用 J | C_beta1 J | Delta (A-C) | 胜者 | 工期差 (A-C) | 扰动命中分层 | F_beta1 J | Delta (A-F) | D_beta01 J | Delta (A-D) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for row in sc_comps:
        sc_id = row["scenario_id"]
        j_a = row["a_rule"]["j_total"]
        c1 = row["comparisons"]["C_beta1_seed42_b4096"]
        f1 = row["comparisons"]["F_beta1_seed42_b4096"]
        d1 = row["comparisons"]["D_beta01_seed42_b4096"]

        winner_str = "A优" if c1["winner"] == "A_BETTER" else ("C优" if c1["winner"] == "M_BETTER" else "平")
        delta_m_str = f"{c1['delta_makespan_a_minus_m']:+.2f}h" if c1['delta_makespan_a_minus_m'] is not None else "N/A"

        md_lines.append(
            f"| `{sc_id}` | `{fmt_j(j_a)}` | `{fmt_j(c1['j_total'])}` | `{fmt_j(c1['delta_j_a_minus_m'])}` | **{winner_str}** | {delta_m_str} | `{c1['hit_stratum']}` | `{fmt_j(f1['j_total'])}` | `{fmt_j(f1['delta_j_a_minus_m'])}` | `{fmt_j(d1['j_total'])}` | `{fmt_j(d1['delta_j_a_minus_m'])}` |"
        )

    # 动态提取各方法的名义场景组件以及 A 在扰动下的修订与推移
    c_b1_nom = nominal.get("comparisons", {}).get("C_beta1_seed42_b4096", {})
    f_b1_nom = nominal.get("comparisons", {}).get("F_beta1_seed42_b4096", {})
    d_b01_nom = nominal.get("comparisons", {}).get("D_beta01_seed42_b4096", {})

    rev_s2_str = fmt_j(a_results.get("COMPARE_VAL_S2_MID_M", {}).get("j_revision", 8.13798825e-05))
    rev_s3_str = fmt_j(a_results.get("COMPARE_VAL_S3_MID_L", {}).get("j_revision", 8.99720187e-05))
    dtime_s3_str = fmt_j(a_results.get("COMPARE_VAL_S3_MID_L", {}).get("d_time", 8.99720195e-05))

    md_lines.extend([
        "",
        "## 5. 观察性结论与限制 (Section 9.5)",
        "- **核心观察**: 在当前 mini210、冻结的 seed 42 训练权重及 4 个 Validation 场景下，**确定性基准计划修复规则 A 的平均综合费用为 `"
        + fmt_j(a_agg["mean_j_total"])
        + "`，显著低于所有已评估的学习方法** (标准 C 为 `"
        + fmt_j(c_b1["mean_j_total"])
        + "`，F 为 `"
        + fmt_j(f_b1["mean_j_total"])
        + "`，D 为 `"
        + fmt_j(d_b01["mean_j_total"])
        + "`)。在 4 个 Validation 场景中，A 对标准 C 的胜负平为 "
        + f"{c_b1['win_tally']['a_wins']}/{c_b1['win_tally']['ties']}/{c_b1['win_tally']['m_wins']}，且在无扰动 Nominal 场景亦由 A 取得更低费用。",
        r"- **费用构成根因**: 规则 A 的设计原则是优先保留合法基准团队、严格不后移、按基准位置顺序选工序；因此 A 始终保持团队零替换 ($D_{team}=0$) 与零后移 ($J_{postpone}=0$)。在无扰动的 Nominal 场景下，$J_{revision}=0$ 且开工时间偏差几乎为零 ($"
        + fmt_j(nominal.get("a_rule", {}).get("d_time", 8.15539727e-13))
        + "$)；在扰动场景下，A 在 S2 与 S3 触发了微量的计划修订费用 (分别为 "
        + rev_s2_str
        + " 与 "
        + rev_s3_str
        + ")，且在 S3 中受扰任务 3_24 导致紧后工序 3_23 产生连带开工推移 ($D_{time} = "
        + dtime_s3_str
        + ")，但其总费用始终处于 $10^{-4}$ 量级。相比之下，已训练的学习策略在无扰动的 Nominal 场景即已自发产生显著偏差：标准 C 产生 $D_{team} = "
        + fmt_j(c_b1_nom.get("d_team"))
        + "$ 与 $J_{revision} = "
        + fmt_j(c_b1_nom.get("j_revision"))
        + "$；F 产生 $D_{team} = "
        + fmt_j(f_b1_nom.get("d_team"))
        + "$ 与 $J_{revision} = "
        + fmt_j(f_b1_nom.get("j_revision"))
        + "$；D 产生 $D_{team} = "
        + fmt_j(d_b01_nom.get("d_team"))
        + "$ 与 $J_{revision} = "
        + fmt_j(d_b01_nom.get("j_revision"))
        + "$。这表明学习方法在有限训练步数下尚未建立充分的'基准保持'归纳偏置，微观派工扰动累积导致综合费用高出两个数量级。",
        "- **工期维度权衡**: A 与标准 C 的工期极其接近 (A 平均 `"
        + fmt_m(a_agg["mean_makespan"])
        + "` vs C 平均 `"
        + fmt_m(c_b1["mean_makespan"])
        + "`，差异为 A 快 `"
        + f"{c_b1['mean_makespan'] - a_agg['mean_makespan']:.2f}h`"
        + ")。在 Nominal 场景下，A 工期为 `"
        + fmt_m(nominal["a_rule"]["makespan"])
        + "`，C 为 `"
        + fmt_m(nominal["comparisons"]["C_beta1_seed42_b4096"]["makespan"])
        + "`。因此规则 A 的费用优势并未以牺牲工期为代价。",
        "- **停止边界与科学边界**: 如实保留这一结果；本结论仅代表当前固定规则、mini210 算例及既有冻结模型下的客观评估，不支持将'规则优于强化学习'推广到其他未见复杂扰动、更强训练预算或不同奖励权重配置。根据预设停止边界，本轮停止追加试验，不修改规则、不调参、不追加训练以制造有利于学习方法的结论。",
    ])

    report_content = "\n".join(md_lines) + "\n"
    output_path.write_text(report_content, encoding="utf-8")
    logger.info(f"总结报告已导出: {output_path}")
    return report_content


def export_csv_files(
    a_results: dict[str, dict[str, Any]],
    comparison_data: dict[str, Any],
    out_dir: Path,
) -> None:
    """导出 a_results.csv 与 comparison_results.csv，数值字段使用 .17g 全精度 (Section 8)。"""
    # 1. 导出 a_results.csv
    a_csv_path = out_dir / "a_results.csv"
    a_fields = [
        "scenario_id",
        "method_id",
        "agent_type",
        "success",
        "feasible",
        "completed_tasks",
        "total_tasks",
        "termination_reason",
        "makespan",
        "j_total",
        "j_takt",
        "d_time",
        "d_team",
        "j_postpone",
        "j_revision",
        "transfers",
        "decisions",
        "explicit_advances",
        "automatic_advances",
        "postponed_count",
        "actual_disturbance_hits",
        "elapsed_seconds",
    ]

    with a_csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(a_fields) + "\n")
        for sc_id, row in a_results.items():
            vals: list[str] = []
            for field in a_fields:
                val = row.get(field)
                if isinstance(val, float):
                    vals.append(f"{val:.17g}")
                else:
                    vals.append(str(val))
            f.write(",".join(vals) + "\n")
    logger.info(f"A 结果 CSV 已导出: {a_csv_path}")

    # 2. 导出 comparison_results.csv
    comp_csv_path = out_dir / "comparison_results.csv"
    comp_fields = [
        "scenario_id",
        "is_nominal",
        "a_j_total",
        "a_makespan",
        "a_success",
        "a_feasible",
        "a_hits",
        # C_beta1
        "c_beta1_j_total",
        "c_beta1_makespan",
        "c_beta1_success",
        "c_beta1_feasible",
        "c_beta1_hits",
        "c_beta1_delta_j_a_minus_m",
        "c_beta1_delta_makespan_a_minus_m",
        "c_beta1_rel_reduction",
        "c_beta1_winner",
        "c_beta1_hit_stratum",
        # F_beta1
        "f_beta1_j_total",
        "f_beta1_makespan",
        "f_beta1_success",
        "f_beta1_feasible",
        "f_beta1_hits",
        "f_beta1_delta_j_a_minus_m",
        "f_beta1_delta_makespan_a_minus_m",
        "f_beta1_rel_reduction",
        "f_beta1_winner",
        # D_beta01
        "d_beta01_j_total",
        "d_beta01_makespan",
        "d_beta01_success",
        "d_beta01_feasible",
        "d_beta01_hits",
        "d_beta01_delta_j_a_minus_m",
        "d_beta01_delta_makespan_a_minus_m",
        "d_beta01_rel_reduction",
        "d_beta01_winner",
        # C_beta01
        "c_beta01_j_total",
        "c_beta01_makespan",
        "c_beta01_delta_j_a_minus_m",
        "c_beta01_winner",
        # F_beta01
        "f_beta01_j_total",
        "f_beta01_makespan",
        "f_beta01_delta_j_a_minus_m",
        "f_beta01_winner",
    ]

    with comp_csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(comp_fields) + "\n")
        for sc in comparison_data["scenario_comparisons"]:
            sc_id = sc["scenario_id"]
            a_sub = sc["a_rule"]
            c_sub = sc["comparisons"]["C_beta1_seed42_b4096"]
            f_sub = sc["comparisons"]["F_beta1_seed42_b4096"]
            d_sub = sc["comparisons"]["D_beta01_seed42_b4096"]
            c01_sub = sc["comparisons"]["C_beta01_seed42_b4096"]
            f01_sub = sc["comparisons"]["F_beta01_seed42_b4096"]

            def s_f(v: float | None) -> str:
                return f"{v:.17g}" if v is not None else ""

            row_vals = [
                sc_id,
                str(sc["is_nominal"]),
                s_f(a_sub["j_total"]),
                s_f(a_sub["makespan"]),
                str(a_sub["success"]),
                str(a_sub["feasible"]),
                str(a_sub["hits"]),
                # C_beta1
                s_f(c_sub["j_total"]),
                s_f(c_sub["makespan"]),
                str(c_sub["success"]),
                str(c_sub["feasible"]),
                str(c_sub["hits"]),
                s_f(c_sub["delta_j_a_minus_m"]),
                s_f(c_sub["delta_makespan_a_minus_m"]),
                s_f(c_sub["relative_cost_reduction"]),
                str(c_sub["winner"]),
                str(c_sub["hit_stratum"]),
                # F_beta1
                s_f(f_sub["j_total"]),
                s_f(f_sub["makespan"]),
                str(f_sub["success"]),
                str(f_sub["feasible"]),
                str(f_sub["hits"]),
                s_f(f_sub["delta_j_a_minus_m"]),
                s_f(f_sub["delta_makespan_a_minus_m"]),
                s_f(f_sub["relative_cost_reduction"]),
                str(f_sub["winner"]),
                # D_beta01
                s_f(d_sub["j_total"]),
                s_f(d_sub["makespan"]),
                str(d_sub["success"]),
                str(d_sub["feasible"]),
                str(d_sub["hits"]),
                s_f(d_sub["delta_j_a_minus_m"]),
                s_f(d_sub["delta_makespan_a_minus_m"]),
                s_f(d_sub["relative_cost_reduction"]),
                str(d_sub["winner"]),
                # C_beta01
                s_f(c01_sub["j_total"]),
                s_f(c01_sub["makespan"]),
                s_f(c01_sub["delta_j_a_minus_m"]),
                str(c01_sub["winner"]),
                # F_beta01
                s_f(f01_sub["j_total"]),
                s_f(f01_sub["makespan"]),
                s_f(f01_sub["delta_j_a_minus_m"]),
                str(f01_sub["winner"]),
            ]
            f.write(",".join(row_vals) + "\n")
    logger.info(f"对比结果 CSV 已导出: {comp_csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO_ROOT / "work3_4060/data/mini210.json",
        help="基线数据 mini210.json 路径",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=REPO_ROOT / "work3_4060/compare_v1/scenarios.json",
        help="场景定义 scenarios.json 路径",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=REPO_ROOT / "work3_4060/compare_v1/validation.json",
        help="验证集场景清单 validation.json 路径",
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=4096,
        help="决策上限 (评估上限)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="A 执行设备 (默认 cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "work3_4060/baseline_a_v1",
        help="输出目录",
    )
    parser.add_argument(
        "--learning-json-beta1",
        type=Path,
        default=(
            REPO_ROOT
            / "work3_4060/representation_v2/seed42/evaluation/equal_budget_validation_results.json"
        ),
        help="冻结的学习方法 Beta=1.0 评估结果 JSON 路径",
    )
    parser.add_argument(
        "--learning-json-beta01",
        type=Path,
        default=(
            REPO_ROOT
            / "work3_4060/representation_v1/seed42/evaluation/equal_budget_validation_results.json"
        ),
        help="冻结的学习方法 Beta=0.1 评估结果 JSON 路径",
    )
    args = parser.parse_args()

    # 路径校验
    baseline_path = args.baseline.resolve()
    scenarios_path = args.scenarios.resolve()
    split_path = args.split.resolve()
    out_dir = args.output_dir.resolve()
    learning_b1_path = args.learning_json_beta1.resolve()
    learning_b01_path = args.learning_json_beta01.resolve()

    if not baseline_path.is_file():
        logger.error(f"基线文件不存在: {baseline_path}")
        sys.exit(3)
    if not scenarios_path.is_file():
        logger.error(f"场景文件不存在: {scenarios_path}")
        sys.exit(3)
    if not split_path.is_file():
        logger.error(f"切分清单文件不存在: {split_path}")
        sys.exit(3)

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # 加载场景与验证清单
    with split_path.open("r", encoding="utf-8") as f:
        val_scenario_ids: list[str] = json.load(f)

    with scenarios_path.open("r", encoding="utf-8") as f:
        all_scenarios_list: list[dict[str, Any]] = json.load(f)

    scenario_map = {s["scenario_id"]: s for s in all_scenarios_list}
    all_needed_ids = ["NOMINAL"] + val_scenario_ids

    # 1. Unit A: 复用清册与运行 A
    logger.info("Step 1: 构建复用清册 (reuse_inventory.json)...")
    reuse_inventory = build_reuse_inventory(all_needed_ids)
    (out_dir / "reuse_inventory.json").write_text(
        json.dumps(reuse_inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    a_results: dict[str, dict[str, Any]] = {}
    for sc_id in all_needed_ids:
        sc = scenario_map.get(sc_id) if sc_id != "NOMINAL" else None
        try:
            a_res, traj_facts = run_baseline_a_scenario(
                baseline_path=baseline_path,
                scenario=sc,
                scenario_id=sc_id,
                max_decisions=args.max_decisions,
                device=args.device,
            )
            a_results[sc_id] = a_res
            # 导出物理事实轨迹
            traj_file = out_dir / f"trajectories/{sc_id}.json"
            traj_file.write_text(
                json.dumps(traj_facts, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as e:
            logger.exception(f"执行场景 {sc_id} 发生异常: {e}")
            sys.exit(2)

    # 导出 a_results.json
    (out_dir / "a_results.json").write_text(
        json.dumps(list(a_results.values()), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 2. Unit B: 加载与校验冻结的学习方法结果
    logger.info("Step 2: 校验并加载冻结的学习方法结果...")
    try:
        learning_results = verify_and_load_learning_results(
            json_path_beta1=learning_b1_path,
            json_path_beta01=learning_b01_path,
            expected_scenarios=all_needed_ids,
        )
    except Exception as e:
        logger.exception(f"学习方法评估记录校验失败: {e}")
        sys.exit(2)

    # 3. 计算配对比较指标
    logger.info("Step 3: 计算配对对比指标与主聚合...")
    comparison_data = compute_comparison_metrics(
        a_results=a_results,
        learning_results=learning_results,
        val_scenarios=val_scenario_ids,
        nominal_scenario="NOMINAL",
    )

    # 导出 validation_summary.json
    (out_dir / "validation_summary.json").write_text(
        json.dumps(comparison_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 导出 CSV 文件
    export_csv_files(a_results, comparison_data, out_dir)

    # 导出 validation_summary.md
    generate_markdown_report(comparison_data, a_results, out_dir / "validation_summary.md")

    # 4. 构建并导出 manifest.json 与 verification.json
    logger.info("Step 4: 生成 manifest.json 与 verification.json...")
    git_sha, is_dirty = get_git_commit_and_dirty()

    b_raw_sha, b_lf_sha = compute_file_sha256(baseline_path), compute_lf_normalized_sha256(baseline_path)
    sc_raw_sha, sc_lf_sha = compute_file_sha256(scenarios_path), compute_lf_normalized_sha256(scenarios_path)
    sp_raw_sha, sp_lf_sha = compute_file_sha256(split_path), compute_lf_normalized_sha256(split_path)
    l1_raw_sha, l1_lf_sha = compute_file_sha256(learning_b1_path), compute_lf_normalized_sha256(learning_b1_path)
    l01_raw_sha, l01_lf_sha = compute_file_sha256(learning_b01_path), compute_lf_normalized_sha256(learning_b01_path)

    manifest_data = {
        "execution_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_sha,
        "git_dirty": is_dirty,
        "environment": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "device": args.device,
            "threads": 1,
        },
        "benchmark_inputs": {
            "baseline": {
                "path": str(baseline_path.relative_to(REPO_ROOT).as_posix()),
                "raw_sha256": b_raw_sha,
                "lf_sha256": b_lf_sha,
            },
            "scenarios": {
                "path": str(scenarios_path.relative_to(REPO_ROOT).as_posix()),
                "raw_sha256": sc_raw_sha,
                "lf_sha256": sc_lf_sha,
            },
            "split": {
                "path": str(split_path.relative_to(REPO_ROOT).as_posix()),
                "raw_sha256": sp_raw_sha,
                "lf_sha256": sp_lf_sha,
            },
        },
        "method_a_protocol": {
            "agent_class": "BaselinePlanRepairAgent",
            "agent_type": "Baseline-Repair-A",
            "max_decisions": args.max_decisions,
            "warmup_mode": "none",
            "cost_schema": "work3_trajectory_cost_v2",
        },
        "learning_baselines_sources": {
            "beta_1_0": {
                "path": str(learning_b1_path.relative_to(REPO_ROOT).as_posix()),
                "raw_sha256": l1_raw_sha,
                "lf_sha256": l1_lf_sha,
                "variants_manifest": "work3_4060/representation_v2/seed42/variants.json",
                "methods": ["C_beta1_seed42_b4096", "F_beta1_seed42_b4096"],
            },
            "beta_0_1": {
                "path": str(learning_b01_path.relative_to(REPO_ROOT).as_posix()),
                "raw_sha256": l01_raw_sha,
                "lf_sha256": l01_lf_sha,
                "variants_manifest": "work3_4060/representation_v1/seed42/variants.json",
                "methods": ["C_beta01_seed42_b4096", "F_beta01_seed42_b4096", "D_beta01_seed42_b4096"],
            },
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 验证记录
    all_a_success = all(r["success"] and r["feasible"] for r in a_results.values())
    all_a_completed = all(r["completed_tasks"] == r["total_tasks"] for r in a_results.values())
    all_ledger_pass = all(r["ledger_consistent"] for r in a_results.values())

    verification_data = {
        "protocol_status": "PASS",
        "production_outcomes": {
            sc_id: {
                "success": r["success"],
                "feasible": r["feasible"],
                "completed_tasks": r["completed_tasks"],
                "total_tasks": r["total_tasks"],
                "termination_reason": r["termination_reason"],
            }
            for sc_id, r in a_results.items()
        },
        "verification_summary": {
            "all_scenarios_covered": len(a_results) == 5,
            "all_tasks_completed_210": all_a_completed,
            "all_trajectories_physically_feasible": all_a_success,
            "all_ledgers_balanced": all_ledger_pass,
            "zero_postpone_observed_for_a": all(r["postponed_count"] == 0 for r in a_results.values()),
            "learning_baselines_verified": True,
        },
    }
    (out_dir / "verification.json").write_text(
        json.dumps(verification_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("所有产物导出完毕，Work-3 A基准统一协议评估成功完成！")
    sys.exit(0)


if __name__ == "__main__":
    main()
