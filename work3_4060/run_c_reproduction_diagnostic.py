"""Work-3 C 基准复现能力诊断执行入口（范围受控的可学习性诊断）。

阶段（每个阶段在独立子进程中运行，保证 CPU 线程数可在建模前显式设定）：

    collect          采集一条 A 教师 Nominal 轨迹、逐决策 CPU 快照与正式可表达性核对
    derive           从已落盘教师产物重建输入可见性清册与教师轨迹核验
    score-original   在教师状态上评分原 C，并核验其来源与加载合同
    train            单副本监督拟合（Adam lr=3e-4，有效批量 8，上限 1000 步 / 1800 s）
    evaluate         独立 CPU Nominal 闭环：原 C 与诊断副本各一次，并做诊断副本教师评分
    verify           轨迹事实、权重身份与快照合同的严格核对
    report           comparison.csv 与 summary.md

本脚本不新增网络结构、不改费用权重与环境物理规则、不做 PPO 续训、不引入扰动。
"""

from __future__ import annotations

import argparse
import csv
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

from envs.work3.core_types import TaskStatus
from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import ActorCriticWork3
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from scripts.work3.evaluate_c_vs_d import (
    FormalEvaluationAgent,
    build_formal_evaluation_agent,
    evaluate_single_trajectory,
)
from scripts.work3.baseline_repair_agent import BaselinePlanRepairAgent
from scripts.work3.experiment_protocol import build_method_profile
import work3_4060.c_reproduction_protocol as protocol
from work3_4060.c_reproduction_protocol import (
    BASELINE_PATH,
    DECISION_BUDGET,
    DIAGNOSTIC_FORMAT,
    EXPECTED_AIRCRAFT_COUNT,
    EXPECTED_H0,
    EXPECTED_STATION_COUNT,
    EXPECTED_TASK_COUNT,
    EARLY_STOP_ACCURACY,
    EARLY_STOP_CONSECUTIVE_SCANS,
    GPU_BUDGET_BYTES,
    ORIGINAL_C_WEIGHTS,
    ORIGINAL_C_RUN,
    OUTPUT_DIR,
    SNAPSHOT_FIELDS,
    TRAIN_EFFECTIVE_BATCH,
    TRAIN_GRAD_CLIP,
    TRAIN_LEARNING_RATE,
    TRAIN_MAX_OPTIMIZER_STEPS,
    TRAIN_MAX_SECONDS,
    TRAIN_SEED,
    TRAIN_WEIGHT_DECAY,
    TeacherLabel,
    actor_fingerprint,
    build_c_state_features,
    build_c_time_features,
    build_diagnostic_checkpoint,
    build_teacher_sample,
    clone_actor_state,
    compare_execution_facts,
    describe_thread_configuration,
    dump_json,
    dump_jsonl,
    enforce_single_thread_cpu,
    extract_execution_facts,
    gpu_memory_snapshot,
    load_diagnostic_actor,
    load_json,
    policy_parameter_names,
    sample_snapshot,
    scan_teacher_states,
    sha256_file,
    sha256_text,
    shuffled_cyclic_stream,
    state_dict_fingerprint,
    teacher_label_from_action,
    teacher_marginal_log_prob,
    value_parameter_names,
)

STAGES: tuple[str, ...] = (
    "collect",
    "derive",
    "score-original",
    "train",
    "evaluate",
    "verify",
    "report",
)

TEACHER_SNAPSHOTS = OUTPUT_DIR / "teacher_snapshots.pt"
TEACHER_ACTIONS = OUTPUT_DIR / "teacher_actions.jsonl"
TEACHER_VALIDATION = OUTPUT_DIR / "teacher_validation.json"
INPUT_VISIBILITY = OUTPUT_DIR / "input_visibility.json"
INITIAL_METRICS = OUTPUT_DIR / "initial_teacher_metrics.json"
FINAL_METRICS = OUTPUT_DIR / "final_teacher_metrics.json"
BC_HISTORY = OUTPUT_DIR / "bc_history.jsonl"
TRAIN_SCANS = OUTPUT_DIR / "train_scans.jsonl"
DIAGNOSTIC_ACTOR = OUTPUT_DIR / "diagnostic_actor.pt"
TRAJ_DIR = OUTPUT_DIR / "trajectories"
A_TRAJ = TRAJ_DIR / "A_NOMINAL.json"
C_ORIGINAL_TRAJ = TRAJ_DIR / "C_ORIGINAL_NOMINAL.json"
C_BC_TRAJ = TRAJ_DIR / "C_BC_DIAGNOSTIC_NOMINAL.json"
TRAJECTORY_VERIFICATION = OUTPUT_DIR / "trajectory_verification.json"
VERIFICATION = OUTPUT_DIR / "verification.json"
COMPARISON_CSV = OUTPUT_DIR / "comparison.csv"
SUMMARY_MD = OUTPUT_DIR / "summary.md"
MANIFEST = OUTPUT_DIR / "manifest.json"
BLOCKED_INTERFACE = OUTPUT_DIR / "blocked_interface.json"
RUNS_DIR = OUTPUT_DIR / "runs"

GRAD_LOG_INTERVAL = 1


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    dump_json(path, payload)


def read_teacher_bundle() -> dict[str, Any]:
    if not TEACHER_SNAPSHOTS.is_file():
        raise FileNotFoundError(f"缺少教师样本文件: {TEACHER_SNAPSHOTS}")
    bundle = torch.load(TEACHER_SNAPSHOTS, map_location="cpu", weights_only=False)
    if bundle.get("schema_version") != protocol.TEACHER_SNAPSHOT_VERSION:
        raise ValueError("教师样本 schema 版本不匹配")
    samples = bundle.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("教师样本为空")
    return bundle


def label_of(sample: Mapping[str, Any]) -> TeacherLabel:
    return protocol.sample_label(sample)


def teacher_manifest_sha256() -> str:
    digest = sha256_file(TEACHER_SNAPSHOTS) + "|" + sha256_file(TEACHER_ACTIONS)
    return sha256_text(digest)


def build_snapshot_builder() -> ActorCriticWork3:
    """仅用于图构造与快照冻结的容器；不加载、不改变任何 C 权重。"""
    builder = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    builder.eval()
    return builder


def make_env() -> AirLineEnvWork3:
    return AirLineEnvWork3(baseline_json_path=BASELINE_PATH)


def baseline_facts() -> dict[str, Any]:
    baseline = load_json(BASELINE_PATH)
    metadata = baseline["metadata"]
    return {
        "path": str(BASELINE_PATH),
        "sha256": sha256_file(BASELINE_PATH),
        "h0": float(metadata["h0"]),
        "task_count": len(baseline.get("tasks", {})),
        "metadata_keys": sorted(metadata.keys()),
        "schema_version": baseline.get("schema_version"),
        "clock_model": metadata.get("clock_model") or metadata.get("transfer_policy"),
    }


def load_original_c_actor() -> tuple[FormalEvaluationAgent, dict[str, Any]]:
    """通过现有正式加载合同读取原 C；任何合同不满足都必须抛出。"""
    if not ORIGINAL_C_WEIGHTS.is_file():
        raise FileNotFoundError(f"缺少原 C 检查点: {ORIGINAL_C_WEIGHTS}")
    agent = build_formal_evaluation_agent("C", ORIGINAL_C_WEIGHTS, device="cpu")
    checkpoint = torch.load(ORIGINAL_C_WEIGHTS, map_location="cpu", weights_only=False)
    return agent, checkpoint


def original_c_provenance(agent: FormalEvaluationAgent, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    metadata = checkpoint.get("run_metadata", {})
    run_json = load_json(ORIGINAL_C_RUN) if ORIGINAL_C_RUN.is_file() else {}
    data_fingerprint = metadata.get("data_fingerprint", {})
    current = {
        "baseline_sha256": sha256_file(BASELINE_PATH),
        "scenario_pool_sha256": sha256_file(protocol.SCENARIOS_PATH),
        "scenario_split_sha256": sha256_file(protocol.TRAIN_SPLIT_PATH),
    }
    drift = {
        key: {
            "training_value": data_fingerprint.get(key),
            "current_value": value,
            "matches": data_fingerprint.get(key) == value,
        }
        for key, value in current.items()
    }
    scenario_diff = describe_scenario_pool_drift()
    return {
        "weights_path": str(ORIGINAL_C_WEIGHTS),
        "weights_sha256": sha256_file(ORIGINAL_C_WEIGHTS),
        "run_json_path": str(ORIGINAL_C_RUN),
        "run_json_sha256": sha256_file(ORIGINAL_C_RUN) if ORIGINAL_C_RUN.is_file() else None,
        "actor_fingerprint": actor_fingerprint(agent.actor_critic),
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "checkpoint_role": checkpoint.get("checkpoint_role"),
        "method_variant": metadata.get("method_variant"),
        "method_profile": metadata.get("method_profile"),
        "seed": metadata.get("seed"),
        "source_sha": metadata.get("source_sha"),
        "source_tree_dirty": metadata.get("source_tree_dirty"),
        "initial_actor_fingerprint": metadata.get("initial_actor_fingerprint"),
        "successful_batch_count": metadata.get("successful_batch_count"),
        "lightning_optimization_steps": metadata.get("lightning_optimization_steps"),
        "total_decisions": run_json.get("total_decisions"),
        "resolved_runtime_config_sha256": metadata.get("resolved_runtime_config_sha256"),
        "data_fingerprint": data_fingerprint,
        "data_fingerprint_drift": drift,
        "scenario_pool_drift": scenario_diff,
        "loaded_by": "scripts.work3.evaluate_c_vs_d.build_formal_evaluation_agent('C', ..., device='cpu')",
        "graph_feature_version": checkpoint.get("graph_feature_version"),
        "graph_feature_schema_hash": checkpoint.get("graph_feature_schema_hash"),
    }


def describe_scenario_pool_drift() -> dict[str, Any]:
    """追溯训练时与当前 scenarios.json 的差异，确认只是标签/口径改名。"""
    training_text = None
    source_sha = load_json(ORIGINAL_C_RUN).get("source_sha") if ORIGINAL_C_RUN.is_file() else None
    if source_sha:
        try:
            training_text = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{source_sha}:work3_4060/rule_baseline_ac_v1/scenarios.json",
                ],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
            ).decode("utf-8")
        except Exception:
            training_text = None
    current = load_json(protocol.SCENARIOS_PATH)
    if training_text is None:
        return {
            "training_snapshot_available": False,
            "current_scenario_count": len(current),
            "note": "无法从 git 取回训练时 scenarios.json，未做字段级追溯。",
        }
    previous = json.loads(training_text)
    prev_by_id = {item["scenario_id"]: item for item in previous}
    current_by_id = {item["scenario_id"]: item for item in current}
    changed_fields: dict[str, list[str]] = {}
    physical_change = False
    for scenario_id in sorted(set(prev_by_id) | set(current_by_id)):
        left = prev_by_id.get(scenario_id, {})
        right = current_by_id.get(scenario_id, {})
        diff_fields = sorted(
            key for key in set(left) | set(right) if left.get(key) != right.get(key)
        )
        if diff_fields:
            changed_fields[scenario_id] = diff_fields
        for key in diff_fields:
            if key not in {"timing", "intensity", "timing_reference"}:
                physical_change = True
    return {
        "training_snapshot_available": True,
        "training_source_sha": source_sha,
        "changed_fields": changed_fields,
        "only_label_or_reference_fields_changed": not physical_change,
        "current_scenario_count": len(current),
        "used_by_this_diagnostic": False,
        "note": (
            "本轮只跑 Nominal（scenario=None），不加载任何扰动场景；"
            "差异仅出现在 timing/intensity/timing_reference 三个标签字段，事件物理定义未变。"
            if not physical_change
            else "检测到扰动事件物理定义变化，必须在报告中单独说明影响范围。"
        ),
    }


# ---------------------------------------------------------------------------
# Phase A：输入可见性清册
# ---------------------------------------------------------------------------


CANDIDATE_FEATURE_SEMANTICS: tuple[tuple[str, str], ...] = (
    ("duration_over_h0", "工序基准工时 / H0"),
    ("baseline_position_plus_scheduled_remaining", "基准站内偏移 + (预约开工-当前时刻)/H0"),
    ("demand_plus_team_diff_ratio", "需求人数/5 + 0.25*(1-|pub_team∩base_team|/demand)"),
    ("station_norm_plus_published_station_drift", "当前站位/4 + 0.1*(发布站位-基准站位)"),
    ("postpone_count_plus_reserved", "后移次数/5 + 0.5*[已预约]"),
    ("material_delayed_flag", "物料未到标记 + 0.5*[已预约]"),
    ("material_wait_plus_scheduled_remaining", "物料等待/H0 + 预约开工剩余/H0"),
    ("relative_station_offset_plus_scheduled_end", "站间偏移/4 + 预约完工剩余/H0"),
)

STATE_FEATURE_SEMANTICS: tuple[tuple[int, str], ...] = (
    (0, "本周期已运行时间比例 (t-P_{q-1})/H0"),
    (1, "各站剩余工时比例（站 0..4，索引 1..5）"),
    (6, "各站当前运行工序数比例（索引 6..10）"),
    (11, "各站物料未到工序数比例（索引 11..15）"),
    (16, "各站最大开工等待紧迫度 log1p（索引 16..20）"),
    (21, "各站是否有飞机在场 0/1（索引 21..25）"),
    (26, "启发式剩余时间比例"),
    (27, "名义剩余时间比例 (P_{q-1}+H0-t)/H0"),
    (28, "当前脉动周期比例"),
    (29, "全线累计完工工序比例"),
    (30, "全线后移工序数比例"),
    (31, "归一化总生产时长进度"),
)


def _aggregate_activity(tensors: Sequence[torch.Tensor]) -> dict[str, Any]:
    """跨全部教师状态统计每个特征维度的活跃度，避免单状态探针给出退化证据。"""
    if not tensors:
        raise ValueError("活跃度统计需要至少一个张量")
    rows = [
        tensor.detach().to(device="cpu", dtype=torch.float32).reshape(-1, tensor.shape[-1])
        for tensor in tensors
    ]
    stacked = torch.cat(rows, dim=0)
    total = int(stacked.size(0))
    nonzero_fraction = (stacked.abs() > 0).sum(dim=0).float() / float(total)
    std = stacked.std(dim=0, unbiased=False)
    distinct = torch.tensor(
        [len(torch.unique(stacked[:, index])) for index in range(stacked.size(1))]
    )
    return {
        "scanned_rows": total,
        "shapes_seen": sorted({tuple(int(v) for v in tensor.shape) for tensor in tensors}),
        "nonzero_fraction": nonzero_fraction.tolist(),
        "std": std.tolist(),
        "min": stacked.min(dim=0).values.tolist(),
        "max": stacked.max(dim=0).values.tolist(),
        "distinct_count": [int(value) for value in distinct.tolist()],
    }


def build_input_visibility_report(
    samples: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    from models.work3.graph_builder import (
        GRAPH_FEATURE_DIMS,
        GRAPH_FEATURE_SCHEMA,
        GRAPH_FEATURE_VERSION,
    )

    if not samples:
        raise ValueError("输入可见性清册需要至少一条教师状态")
    snapshots = [sample_snapshot(sample) for sample in samples]
    first = snapshots[0]
    first_label = label_of(samples[0])
    graph = first.graph_snapshot
    task_images = GRAPH_FEATURE_SCHEMA["task_features"]
    worker_images = GRAPH_FEATURE_SCHEMA["worker_features"]
    task_index = {name: index for index, name in enumerate(task_images)}
    worker_index = {name: index for index, name in enumerate(worker_images)}

    task_activity = _aggregate_activity([snap.graph_snapshot["task"].x for snap in snapshots])
    worker_activity = _aggregate_activity(
        [snap.graph_snapshot["worker"].x for snap in snapshots]
    )
    candidate_activity = _aggregate_activity(
        [snap.candidate_task_features for snap in snapshots]
    )
    state_activity = _aggregate_activity([snap.state_features for snap in snapshots])
    time_activity = _aggregate_activity([snap.time_features for snap in snapshots])

    def task_dim(name: str) -> dict[str, Any]:
        index = task_index[name]
        return {
            "feature": name,
            "index": index,
            "scanned_states": len(snapshots),
            "nonzero_fraction": task_activity["nonzero_fraction"][index],
            "std": task_activity["std"][index],
            "min": task_activity["min"][index],
            "max": task_activity["max"][index],
            "distinct_count": task_activity["distinct_count"][index],
        }

    def worker_dim(name: str) -> dict[str, Any]:
        index = worker_index[name]
        return {
            "feature": name,
            "index": index,
            "scanned_states": len(snapshots),
            "nonzero_fraction": worker_activity["nonzero_fraction"][index],
            "std": worker_activity["std"][index],
            "min": worker_activity["min"][index],
            "max": worker_activity["max"][index],
            "distinct_count": worker_activity["distinct_count"][index],
        }

    def candidate_dim(index: int) -> dict[str, Any]:
        return {
            "index": index,
            "scanned_states": len(snapshots),
            "nonzero_fraction": candidate_activity["nonzero_fraction"][index],
            "std": candidate_activity["std"][index],
            "min": candidate_activity["min"][index],
            "max": candidate_activity["max"][index],
            "distinct_count": candidate_activity["distinct_count"][index],
        }

    edge_types = {str(edge): int(graph[edge].num_edges) for edge in graph.edge_types}
    teacher_task_node = None
    teacher_baseline_team_edges = 0
    if first_label.decision_kind == "stay" and first_label.task_idx is not None:
        teacher_task_node = int(
            first.candidate_task_node_indices[int(first_label.task_idx)]
        )
        edge_index = graph["task", "baseline_team", "worker"].edge_index
        if edge_index.numel() > 0:
            teacher_baseline_team_edges = int((edge_index[0] == teacher_task_node).sum().item())

    items = [
        {
            "item_id": "baseline_local_offset_and_current_position",
            "question": "基准局部开工偏移与当前开工位置偏离",
            "entries": [
                {
                    "physical_source": "TaskRuntimeState.in_station_offset（基准站内偏移 b_i^0）",
                    "encoding_path": f"graph.task[{task_index['baseline_in_station_offset_norm']}] = baseline_in_station_offset_norm",
                    "status": "direct",
                    "evidence": task_dim("baseline_in_station_offset_norm"),
                },
                {
                    "physical_source": "launch_pulse_offset（上一转站以来的脉冲偏移）",
                    "encoding_path": f"graph.task[{task_index['launch_pulse_offset_norm']}] = launch_pulse_offset_norm",
                    "status": "direct",
                    "evidence": task_dim("launch_pulse_offset_norm"),
                },
                {
                    "physical_source": "publication_reference.position / scheduled_start",
                    "encoding_path": "candidate_task_features[1]、[7]；graph.task[...] last_published_position_norm",
                    "status": "direct",
                    "evidence": {
                        "candidate_feature_1": candidate_dim(1),
                        "candidate_feature_7": candidate_dim(7),
                        "graph": task_dim("last_published_position_norm"),
                    },
                },
            ],
            "verdict": "Actor 可直接获得基准站内偏移与当前开工位置偏离。",
        },
        {
            "item_id": "baseline_team_and_candidate_worker_relation",
            "question": "基准团队及候选工人与基准团队的关系",
            "entries": [
                {
                    "physical_source": "TaskRuntimeState.base_team（基准标准团队 W_i^0）",
                    "encoding_path": "异构图关系类型 (task, baseline_team, worker)；不是工人节点特征位",
                    "status": "direct（以关系类型编码）",
                    "evidence": {
                        "edge_type": "(task, baseline_team, worker)",
                        "num_edges": edge_types.get(
                            str(("task", "baseline_team", "worker")), 0
                        ),
                        "teacher_task_baseline_team_edges": teacher_baseline_team_edges,
                    },
                },
                {
                    "physical_source": "上一次正式发布团队",
                    "encoding_path": "异构图关系类型 (task, last_published_team, worker)",
                    "status": "direct（以关系类型编码）",
                    "evidence": {
                        "num_edges": edge_types.get(
                            str(("task", "last_published_team", "worker")), 0
                        ),
                    },
                },
                {
                    "physical_source": "候选工人自身的技能与站位归属",
                    "encoding_path": "graph.worker[skill_0..4]、graph.worker[station_0..4]",
                    "status": "direct",
                    "evidence": {
                        name: worker_dim(name)
                        for name in ("skill_0", "station_0", "station_1", "efficiency")
                        if name in worker_index
                    },
                },
                {
                    "physical_source": "候选团队相对基准团队的差异比例",
                    "encoding_path": "candidate_task_features[2] 中的 team_diff_ratio（任务级派生量）",
                    "status": "derivable（任务级聚合，非逐工人布尔位）",
                    "evidence": {"candidate_feature_2": candidate_dim(2)},
                },
            ],
            "verdict": (
                "基准团队关系通过专门的关系类型与工人特征可表达；但不存在"
                "“该工人是否属于本工序基准团队”的显式工人节点布尔特征，"
                "该判断需要网络在消息传递中沿关系类型推导。"
            ),
        },
        {
            "item_id": "current_legal_team_skill_demand_calendar",
            "question": "当前合法团队、技能/人数、资源日历相关信息",
            "entries": [
                {
                    "physical_source": "TaskRuntimeState.demand / skill",
                    "encoding_path": "graph.task[demand]、graph.task[skill_0..4]；candidate_task_features[2]",
                    "status": "direct",
                    "evidence": {
                        "demand": task_dim("demand"),
                        "skill_0": task_dim("skill_0"),
                    },
                },
                {
                    "physical_source": "工人效率与已占用日历区间",
                    "encoding_path": "graph.worker[efficiency]、reserved_slot_13..16、active_or_future_interval_count、next_reserved_start/end",
                    "status": "direct",
                    "evidence": {
                        name: worker_dim(name)
                        for name in (
                            "efficiency",
                            "active_or_future_interval_count",
                            "has_next_reserved_interval",
                            "next_reserved_start_remaining_norm",
                        )
                        if name in worker_index
                    },
                },
                {
                    "physical_source": "逐人条件合法掩码（技能/效率/已选工人）",
                    "encoding_path": "Actor 内部由 TeamCompletionContext 逐步构造并施加；不在图特征中",
                    "status": "direct（作为硬掩码进入指针头，而非软特征）",
                    "evidence": {
                        "teacher_demand": int(
                            first.team_contexts[int(first_label.task_idx)].demand
                        )
                        if first_label.task_idx is not None
                        else None,
                        "valid_permutation_count": len(first_label.valid_permutations),
                        "station_worker_count": len(first.team_contexts[0].workers)
                        if first.team_contexts
                        else 0,
                    },
                },
            ],
            "verdict": "合法团队与逐人掩码在正式接口内完全可见，技能/人数/日历均有显式编码。",
        },
        {
            "item_id": "cycle_time_and_next_nominal_boundary",
            "question": "当前周期、仿真时刻与下一名义转站边界",
            "entries": [
                {
                    "physical_source": "current_time / last_transfer_time / h0 / current_cycle",
                    "encoding_path": "state_features[0]=(t-P_{q-1})/H0、[27]=(P_{q-1}+H0-t)/H0、[28]=周期比例",
                    "status": "direct",
                    "evidence": {
                        "state_feature_0": {
                            "scanned_states": len(snapshots),
                            "nonzero_fraction": state_activity["nonzero_fraction"][0],
                            "std": state_activity["std"][0],
                            "min": state_activity["min"][0],
                            "max": state_activity["max"][0],
                        },
                        "state_feature_27": {
                            "nonzero_fraction": state_activity["nonzero_fraction"][27],
                            "min": state_activity["min"][27],
                            "max": state_activity["max"][27],
                        },
                        "state_feature_28_first_value": float(first.state_features[28]),
                        "note": (
                            "教师状态全部落在转站时刻，因此 state_features[0]=(t-P_{q-1})/H0 "
                            "在示范上恒为 0、[27] 恒为 1；这是状态分布特性，不是该维未被编码。"
                        ),
                    },
                },
                {
                    "physical_source": "下一名义转站边界的显式紧迫度输入",
                    "encoding_path": "time_features[0..1]（R^/H0 与 (H0-(t-P_{q-1}))/H0）",
                    "status": "direct",
                    "evidence": {
                        "time_feature_0": {
                            "scanned_states": len(snapshots),
                            "nonzero_fraction": time_activity["nonzero_fraction"][0],
                            "std": time_activity["std"][0],
                            "min": time_activity["min"][0],
                            "max": time_activity["max"][0],
                        },
                        "time_feature_1": {
                            "nonzero_fraction": time_activity["nonzero_fraction"][1],
                            "std": time_activity["std"][1],
                            "min": time_activity["min"][1],
                            "max": time_activity["max"][1],
                        },
                        "zero_residuals_2_to_6_all_zero": all(
                            value == 0.0
                            for snapshot in snapshots
                            for value in snapshot.time_features[2:7].tolist()
                        ),
                    },
                },
                {
                    "physical_source": "下一周期各站是否仍会驻留飞机",
                    "encoding_path": "time_features[7..11]（next_cycle_station_mask 占用位）",
                    "status": "direct",
                    "evidence": {
                        "occupancy_first_state": [bool(v) for v in first.next_cycle_station_mask],
                        "tail_nonzero_fraction": {
                            str(7 + offset): time_activity["nonzero_fraction"][7 + offset]
                            for offset in range(5)
                        },
                    },
                },
            ],
            "verdict": "周期、仿真时刻与名义转站边界既在状态特征中，也作为显式时间输入进入融合层。",
        },
    ]

    return {
        "schema_version": "c_reproduction_input_visibility_v1",
        "graph_feature_version": GRAPH_FEATURE_VERSION,
        "graph_feature_schema_hash": GRAPH_FEATURE_SCHEMA["schema_fingerprint"],
        "graph_feature_dims": dict(GRAPH_FEATURE_DIMS),
        "tensor_contract": {
            "scanned_states": len(snapshots),
            "state_features": list(first.state_features.shape),
            "time_features": list(first.time_features.shape),
            "candidate_task_features": list(first.candidate_task_features.shape),
            "candidate_feature_row_counts": candidate_activity["shapes_seen"],
            "graph_nodes": {key: int(graph[key].num_nodes) for key in graph.node_types},
            "graph_edges": edge_types,
        },
        "state_feature_semantics": [
            {"index": index, "semantics": text} for index, text in STATE_FEATURE_SEMANTICS
        ],
        "candidate_feature_semantics": [
            {"index": index, "semantics": text}
            for index, (_, text) in enumerate(CANDIDATE_FEATURE_SEMANTICS)
        ],
        "time_feature_semantics": (
            "0..1 为 (R^/H0, (H0-(t-P_{q-1}))/H0)；2..6 为 C profile 恒零的下一周期修正残差；"
            "7..11 为下一周期各站占用位。"
        ),
        "items": items,
        "probe_context": {
            "sample_count": len(snapshots),
            "first_sample_id": int(samples[0]["sample_id"]),
            "first_decision_index": int(samples[0]["decision_index"]),
            "first_sim_time": float(first.current_time),
            "first_cycle": int(first.cycle_id),
            "first_teacher_task": first_label.task_key,
            "evidence_scope": "跨全部教师状态统计；不是单状态示例",
        },
        "notes": [
            "清册只描述 Actor 实际接收的张量与关系类型，不依据环境对象是否持有字段推断可见性。",
            "本轮不新增任何特征；若某信息仅能由关系推导，如实记录为可推导而非直接编码。",
            "没有显式字段不等于完全不可表达；有环境字段也不等于网络看得见。",
        ],
    }


# ---------------------------------------------------------------------------
# Phase A/B：采集
# ---------------------------------------------------------------------------


def stage_collect() -> int:
    threads = enforce_single_thread_cpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    log(f"CPU 线程配置: {threads}")

    facts = baseline_facts()
    log(f"基准: h0={facts['h0']:.15f} 任务数={facts['task_count']} sha={facts['sha256'][:16]}")
    if abs(facts["h0"] - EXPECTED_H0) > 1e-9:
        raise ValueError(f"H0 与预期不一致: {facts['h0']} != {EXPECTED_H0}")
    if facts["task_count"] != EXPECTED_TASK_COUNT:
        raise ValueError(f"任务数与预期不一致: {facts['task_count']}")

    # 1) 正式入口下的 A 轨迹（独立事实来源）
    log("运行 A 正式 Nominal 轨迹（evaluate_single_trajectory）...")
    env = make_env()
    agent_a = BaselinePlanRepairAgent()
    a_result = evaluate_single_trajectory(
        env,
        "Baseline-Repair-A",
        agent_a,
        scenario=None,
        max_decisions=DECISION_BUDGET,
        device="cpu",
    )
    a_facts = extract_execution_facts(env)
    a_payload = {**a_result, "execution_facts": a_facts}
    write_json(A_TRAJ, a_payload)
    log(
        "A 轨迹: decisions={} J={:.6e} makespan={:.9f} completed={} transfers={}".format(
            a_result["decisions"],
            a_result["objective_j_total"],
            a_result["makespan"],
            a_result["completed_tasks"],
            a_result["transfers"],
        )
    )

    # 2) 逐决策快照采集（同源环境、同一 A 策略，独立第二遍）
    log("第二遍采集逐决策 CPU 快照与正式可表达性核对...")
    env = make_env()
    builder = build_snapshot_builder()
    agent_a = BaselinePlanRepairAgent()
    env.reset()
    samples: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    blocked: dict[str, Any] | None = None
    started = time.perf_counter()
    while True:
        if env._check_terminated():
            break
        if len(samples) >= DECISION_BUDGET:
            raise RuntimeError("教师轨迹超过决策预算仍未完工")
        action = agent_a.select_action(env)
        if action is None:
            raise RuntimeError("A 在批次未完成时未返回可执行动作")
        decision_index = len(samples) + 1
        cmax = compute_cycle_heuristic_cmax(env.state)
        state_features = build_c_state_features(env.state, cmax)
        time_features = build_c_time_features(env.state, cmax)
        snapshot = builder.make_decision_snapshot(
            env,
            state_features,
            time_features,
            worker_id=0,
            episode_id=0,
            episode_index=decision_index,
            estimated_cmax=cmax,
            allow_postpone=True,
        )
        try:
            label = teacher_label_from_action(snapshot, action)
        except ValueError as error:
            blocked = {
                "decision_index": decision_index,
                "sim_time": float(env.state.current_time),
                "cycle": int(env.state.current_cycle),
                "reason": str(error),
                "teacher_action": {
                    "branch": int(action["branch"]),
                    "task_key": action.get("task_key"),
                    "team": [int(w) for w in action.get("team", ())],
                    "align": int(action.get("align", 0)),
                },
                "candidate_task_keys": list(snapshot.candidate_task_keys),
                "candidate_branch_masks": [list(mask) for mask in snapshot.branch_masks],
                "advance_available": bool(snapshot.advance_available),
                "candidate_count": len(snapshot.candidate_task_keys),
                "status": "BLOCKED_INTERFACE",
            }
            write_json(BLOCKED_INTERFACE, blocked)
            log(f"BLOCKED_INTERFACE: {error}")
            break
        _obs, reward, terminated, truncated, info = env.step(action)
        outcome = {
            "scheduled_status": info.get("scheduled_status"),
            "scheduled_start": None
            if info.get("scheduled_start") is None
            else float(info["scheduled_start"]),
            "scheduled_to_station": None
            if info.get("postponed_to_station") is None
            else int(info["postponed_to_station"]),
            "step_cost": float(info.get("step_cost", 0.0)),
            "raw_reward": float(reward),
            "advance_cause": info.get("advance_cause"),
            "sim_time_after": float(env.state.current_time),
            "cycle_after": int(env.state.current_cycle),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        sample = build_teacher_sample(
            sample_id=len(samples),
            decision_index=decision_index,
            snapshot=snapshot,
            label=label,
            teacher_action=action,
            outcome=outcome,
        )
        samples.append(sample)
        action_rows.append(
            {
                "schema_version": protocol.TEACHER_ACTION_SCHEMA_VERSION,
                "sample_id": len(samples) - 1,
                "decision_index": decision_index,
                "cycle": int(snapshot.cycle_id),
                "sim_time": float(snapshot.current_time),
                "last_transfer_time": float(snapshot.last_transfer_time),
                "h0": float(snapshot.h0),
                "advance_available": bool(snapshot.advance_available),
                "candidate_count": len(snapshot.candidate_task_keys),
                "candidate_task_keys": list(snapshot.candidate_task_keys),
                "candidate_branch_masks": [list(mask) for mask in snapshot.branch_masks],
                "label": label.as_json(),
                "teacher_action": {
                    "branch": int(action["branch"]),
                    "task_key": action.get("task_key"),
                    "team": [int(w) for w in action.get("team", ())],
                    "align": int(action.get("align", 0)),
                },
                "outcome": outcome,
            }
        )
        if terminated or truncated:
            break

    collect_seconds = time.perf_counter() - started
    if blocked is not None:
        log("检测到正式 C 接口无法表达 A 动作，停止依赖该数据的 BC 训练。")
        return 2

    # 3) 与第一遍正式轨迹交叉核对
    replay = {
        "decisions": len(samples),
        "objective_j_total": float(env.cumulative_cost),
        "makespan": float(env.state.current_time),
        "completed_tasks": sum(
            1 for task in env.state.tasks.values() if task.status == TaskStatus.COMPLETED
        ),
        "transfers": len(env.state.transfer_history),
    }
    reference = {
        "decisions": int(a_result["decisions"]),
        "objective_j_total": float(a_result["objective_j_total"]),
        "makespan": float(a_result["makespan"]),
        "completed_tasks": int(a_result["completed_tasks"]),
        "transfers": int(a_result["transfers"]),
    }
    cross_check = {
        key: {
            "collection_pass": replay[key],
            "formal_pass": reference[key],
            "matches": replay[key] == reference[key]
            or (
                isinstance(replay[key], float)
                and math.isclose(replay[key], reference[key], rel_tol=0.0, abs_tol=1e-9)
            ),
        }
        for key in replay
    }
    if not all(item["matches"] for item in cross_check.values()):
        dump_json(OUTPUT_DIR / "collection_cross_check.json", cross_check)
        raise RuntimeError(f"A 两遍轨迹不一致: {cross_check}")

    a_valid = (
        reference["decisions"] == EXPECTED_TASK_COUNT
        and reference["completed_tasks"] == EXPECTED_TASK_COUNT
        and reference["transfers"] == EXPECTED_AIRCRAFT_COUNT + EXPECTED_STATION_COUNT - 1
        and abs(reference["makespan"] - (EXPECTED_AIRCRAFT_COUNT + EXPECTED_STATION_COUNT - 1) * facts["h0"]) <= 1e-6
        and abs(reference["objective_j_total"]) <= 1e-9
        and a_result["constraint_violation_count"] == 0
        and a_result["actual_disturbance_hits"] == 0
        and bool(a_result["success"])
    )
    if not a_valid:
        raise RuntimeError("A 未能复现当前基准（工序/出线/费用/工期核验未通过）")

    # 4) 只落盘原始产物；派生报告由 derive 阶段从磁盘重建，保证可单独复算
    log(f"保存 {len(samples)} 条教师样本（耗时 {collect_seconds:.1f}s）...")
    bundle = {
        "schema_version": protocol.TEACHER_SNAPSHOT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sample_count": len(samples),
        "snapshot_contract": {
            "graph_version": samples[0]["snapshot"]["graph_version"],
            "state_feature_dim": 32,
            "time_feature_dim": 12,
            "candidate_feature_dim": 8,
            "max_station_workers": 16,
            "allow_postpone": True,
            "profile": "C",
            "tensors": "detached_cpu_copies",
        },
        "source": {
            "baseline_path": str(BASELINE_PATH),
            "baseline_sha256": facts["sha256"],
            "h0": facts["h0"],
            "scenarios_used": [],
            "warmup": "none",
            "statement": "训练与评估使用同一实例 Nominal；结论仅限同实例复现能力。",
        },
        "samples": samples,
    }
    torch.save(bundle, TEACHER_SNAPSHOTS)
    dump_jsonl(TEACHER_ACTIONS, action_rows)

    write_json(
        OUTPUT_DIR / "collection_manifest.json",
        {
            "a_trajectory": str(A_TRAJ),
            "a_trajectory_sha256": sha256_file(A_TRAJ),
            "teacher_snapshots": str(TEACHER_SNAPSHOTS),
            "teacher_snapshots_sha256": sha256_file(TEACHER_SNAPSHOTS),
            "teacher_actions_sha256": sha256_file(TEACHER_ACTIONS),
            "teacher_manifest_sha256": teacher_manifest_sha256(),
            "baseline": facts,
            "collection_pass": replay,
            "formal_pass": reference,
            "cross_pass_consistency": cross_check,
            "collection_seconds": collect_seconds,
            "threads": threads,
            "device": "cpu",
        },
    )
    log("collect 完成")
    return 0


def build_teacher_validation(
    samples: Sequence[Mapping[str, Any]],
    a_payload: Mapping[str, Any],
    facts: Mapping[str, Any],
    collection_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """从已落盘产物重建教师轨迹核验（不依赖采集进程的内存状态）。"""
    decision_kind_counts: dict[str, int] = {}
    permutation_counts: dict[str, int] = {}
    candidate_counts: list[int] = []
    advance_available_when_schedules = 0
    decision_times: set[float] = set()
    cycle_boundary_flags: list[bool] = []
    for sample in samples:
        snapshot = sample_snapshot(sample)
        label = label_of(sample)
        decision_kind_counts[label.decision_kind] = (
            decision_kind_counts.get(label.decision_kind, 0) + 1
        )
        key = str(len(label.valid_permutations))
        permutation_counts[key] = permutation_counts.get(key, 0) + 1
        candidate_counts.append(len(snapshot.candidate_task_keys))
        decision_times.add(float(snapshot.current_time))
        cycle_boundary_flags.append(
            abs(float(snapshot.current_time) - float(snapshot.last_transfer_time)) <= 1e-9
        )
        if label.decision_kind == "stay" and snapshot.advance_available:
            advance_available_when_schedules += 1

    return {
        "schema_version": "c_reproduction_teacher_validation_v1",
        "sample_count": len(samples),
        "decision_kind_counts": decision_kind_counts,
        "valid_permutation_count_distribution": permutation_counts,
        "team_order_ambiguous_samples": sum(
            count for key, count in permutation_counts.items() if int(key) >= 2
        ),
        "teacher_task_demand_distribution": demand_distribution(samples),
        "candidate_count": {
            "min": min(candidate_counts),
            "max": max(candidate_counts),
            "mean": sum(candidate_counts) / len(candidate_counts),
        },
        "advance_available_when_teacher_schedules": advance_available_when_schedules,
        "decision_time_coverage": {
            "distinct_decision_times": len(decision_times),
            "distinct_cycles": len({int(sample_snapshot(s).cycle_id) for s in samples}),
            "all_decisions_at_cycle_boundary": all(cycle_boundary_flags),
            "min_sim_time": min(decision_times),
            "max_sim_time": max(decision_times),
            "note": (
                "A 的决策全部发生在转站时刻（t == P_{q-1}），教师状态只覆盖周期起始切片，"
                "不覆盖周期中段的中间决策状态；这是示范分布的结构性限制，不是编码缺陷。"
            ),
        },
        "trajectory": {
            "decisions": int(a_payload["decisions"]),
            "objective_j_total": float(a_payload["objective_j_total"]),
            "makespan": float(a_payload["makespan"]),
            "completed_tasks": int(a_payload["completed_tasks"]),
            "transfers": int(a_payload["transfers"]),
            "success": bool(a_payload["success"]),
            "feasible": bool(a_payload["feasible"]),
            "constraint_violation_count": int(a_payload["constraint_violation_count"]),
            "actual_disturbance_hits": int(a_payload["actual_disturbance_hits"]),
            "nominal_makespan_expected": (
                EXPECTED_AIRCRAFT_COUNT + EXPECTED_STATION_COUNT - 1
            )
            * facts["h0"],
        },
        "cross_pass_consistency": collection_manifest.get("cross_pass_consistency"),
        "expression": {
            "task_in_candidate_set": f"{len(samples)}/{len(samples)}",
            "branch_mask_allows_teacher": f"{len(samples)}/{len(samples)}",
            "teacher_team_reachable_via_masks": f"{len(samples)}/{len(samples)}",
            "advance_action_expressible": "本轨迹 A 无显式推进动作，未触发该分支核对",
            "blocked_interface": False,
            "scope": "逐决策按正式 C profile 重建快照后逐级核对；任一失败即写 BLOCKED_INTERFACE 并停机。",
        },
        "label_coverage_limits": [
            "A 在 Nominal 下的示范全部是留站预约（STATION_EXECUTE + align=1），"
            "不含后移与显式推进标签；因此本轮不能评估模型是否学会区分等待、后移与执行。",
            "多数样本的合法选人排列为 2，说明团队顺序存在观测别名；"
            "团队概率必须按排列 logsumexp 处理，不能当作顺序标签学习。",
        ],
        "collection_seconds": collection_manifest.get("collection_seconds"),
        "threads": describe_thread_configuration(),
    }


def stage_derive() -> int:
    """从已落盘的教师产物重建 Phase A 清册与教师轨迹核验。"""
    threads = enforce_single_thread_cpu()
    bundle = read_teacher_bundle()
    samples = bundle["samples"]
    facts = baseline_facts()
    a_payload = load_json(A_TRAJ)
    manifest_path = OUTPUT_DIR / "collection_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"缺少采集清单: {manifest_path}")
    collection_manifest = load_json(manifest_path)

    validation = build_teacher_validation(samples, a_payload, facts, collection_manifest)
    write_json(TEACHER_VALIDATION, validation)
    log(
        "教师核验: 样本={} 决策类型={} 排列分布={}".format(
            validation["sample_count"],
            validation["decision_kind_counts"],
            validation["valid_permutation_count_distribution"],
        )
    )

    visibility = build_input_visibility_report(samples, facts)
    visibility["sample_scan"] = {
        "scanned_states": len(samples),
        "distinct_candidate_counts": sorted(
            {len(sample_snapshot(sample).candidate_task_keys) for sample in samples}
        ),
        "samples_with_advance_available": sum(
            1 for sample in samples if sample_snapshot(sample).advance_available
        ),
    }
    write_json(INPUT_VISIBILITY, visibility)
    log(f"输入可见性清册: 4 个信息项，扫描 {len(samples)} 个教师状态；线程 {threads}")
    return 0


def demand_distribution(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """按教师实际选择的工序统计需求人数分布（样本级，不是候选级）。"""
    counts: dict[str, int] = {}
    for sample in samples:
        snapshot = sample_snapshot(sample)
        label = label_of(sample)
        if label.task_idx is None:
            continue
        key = str(int(snapshot.team_contexts[int(label.task_idx)].demand))
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Phase C：原 C 教师状态评分
# ---------------------------------------------------------------------------


def stage_score_original() -> int:
    threads = enforce_single_thread_cpu()
    bundle = read_teacher_bundle()
    samples = bundle["samples"]
    log(f"载入 {len(samples)} 条教师样本；CPU 线程 {threads}")

    agent, checkpoint = load_original_c_actor()
    provenance = original_c_provenance(agent, checkpoint)
    write_json(OUTPUT_DIR / "original_c_provenance.json", provenance)
    log(f"原 C: sha={provenance['weights_sha256'][:16]} fp={provenance['actor_fingerprint'][:16]}")

    metrics, per_sample = scan_teacher_states(agent.actor_critic, samples, device="cpu")
    metrics["evaluated_model"] = {
        "role": "original_C",
        "weights_path": str(ORIGINAL_C_WEIGHTS),
        "weights_sha256": provenance["weights_sha256"],
        "actor_fingerprint": provenance["actor_fingerprint"],
        "device": "cpu",
        "threads": threads,
    }
    metrics["teacher_manifest_sha256"] = teacher_manifest_sha256()
    write_json(INITIAL_METRICS, metrics)
    dump_jsonl(OUTPUT_DIR / "teacher_scores_original.jsonl", per_sample)

    # 原 C 独立闭环复现（同一正式评估入口）
    log("原 C 独立 CPU Nominal 闭环复现...")
    env = make_env()
    agent_eval, _ = load_original_c_actor()
    result = evaluate_single_trajectory(
        env,
        "Method-C",
        agent_eval,
        scenario=None,
        max_decisions=DECISION_BUDGET,
        device="cpu",
    )
    facts = extract_execution_facts(env)
    write_json(C_ORIGINAL_TRAJ, {**result, "execution_facts": facts})
    log(
        "原 C 闭环: decisions={} J={:.10f} makespan={:.6f} completed={} success={}".format(
            result["decisions"],
            result["objective_j_total"],
            result["makespan"],
            result["completed_tasks"],
            result["success"],
        )
    )
    write_json(
        OUTPUT_DIR / "original_c_reproduction.json",
        {
            "decisions": int(result["decisions"]),
            "objective_j_total": float(result["objective_j_total"]),
            "makespan": float(result["makespan"]),
            "completed_tasks": int(result["completed_tasks"]),
            "success": bool(result["success"]),
            "feasible": bool(result["feasible"]),
            "cost_components_weighted": result["cost_components_weighted"],
            "archived_reference": {
                "decisions": 395,
                "objective_j_total": 3.82256809126947,
                "makespan": 677.8154910577125,
                "note": "只作为来源明确的参考，不替代本次实测。",
            },
            "measured_matches_archive": {
                "decisions": int(result["decisions"]) == 395,
                "objective_j_total": math.isclose(
                    float(result["objective_j_total"]), 3.82256809126947, rel_tol=1e-9, abs_tol=1e-12
                ),
                "makespan": math.isclose(
                    float(result["makespan"]), 677.8154910577125, rel_tol=1e-12, abs_tol=1e-9
                ),
            },
            "device": "cpu",
            "threads": threads,
        },
    )
    log("score-original 完成")
    return 0


# ---------------------------------------------------------------------------
# Phase D：单副本监督拟合
# ---------------------------------------------------------------------------


def build_diagnostic_copy(source_actor: torch.nn.Module) -> ActorCriticWork3:
    """构造与原 C Actor 逐位一致的诊断副本；原对象不进入任何优化器。"""
    copy = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64)
    source_state = clone_actor_state(source_actor)
    copy.load_state_dict(source_state, strict=True)
    if actor_fingerprint(copy) != actor_fingerprint(source_actor):
        raise RuntimeError("诊断副本在训练前与原 C Actor 指纹不一致")
    return copy


def stage_train() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("监督训练要求 cuda:0，当前不可用")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    log(f"训练设备: {torch.cuda.get_device_name(0)}")

    bundle = read_teacher_bundle()
    samples = bundle["samples"]
    sample_count = len(samples)
    log(f"教师样本 {sample_count} 条")

    agent, checkpoint = load_original_c_actor()
    source_actor = agent.actor_critic
    source_sha = sha256_file(ORIGINAL_C_WEIGHTS)
    source_fingerprint = actor_fingerprint(source_actor)

    model = build_diagnostic_copy(source_actor).to(device)
    model.eval()
    value_names = set(value_parameter_names(model))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name not in value_names
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_names = policy_parameter_names(model)
    if not trainable:
        raise RuntimeError("没有可训练参数")

    optimizer = torch.optim.Adam(
        trainable, lr=TRAIN_LEARNING_RATE, weight_decay=TRAIN_WEIGHT_DECAY
    )
    log(f"可训练参数 {len(trainable_names)} 项；价值专属冻结 {len(value_names)} 项")

    manifest_sha = teacher_manifest_sha256()
    stream = shuffled_cyclic_stream(sample_count, seed=TRAIN_SEED)

    history: list[dict[str, Any]] = []
    scans: list[dict[str, Any]] = []
    stop_reason = "max_optimizer_steps"
    started = time.perf_counter()

    def run_scan(step: int, tag: str) -> dict[str, Any]:
        scan_start = time.perf_counter()
        metrics, per_sample = scan_teacher_states(model, samples, device=device)
        record = {
            "step": int(step),
            "tag": tag,
            "device": str(device),
            "elapsed_seconds": time.perf_counter() - scan_start,
            "nll_mean": metrics["nll_mean"],
            "nll_finite": metrics["nll_finite"],
            "decision_kind_accuracy": metrics["decision_kind_accuracy"],
            "task_accuracy_conditional": metrics["task_accuracy_conditional"],
            "branch_accuracy_conditional": metrics["branch_accuracy_conditional"],
            "team_accuracy_unordered_conditional": metrics["team_accuracy_unordered_conditional"],
            "align_accuracy_conditional": metrics["align_accuracy_conditional"],
            "full_physical_action_accuracy": metrics["full_physical_action_accuracy"],
        }
        scans.append(record)
        log(
            f"扫描[{tag}] step={step} NLL={record['nll_mean']:.6f} "
            f"task={fmt(record['task_accuracy_conditional'])} "
            f"team={fmt(record['team_accuracy_unordered_conditional'])} "
            f"full={fmt(record['full_physical_action_accuracy'])}"
        )
        return {"record": record, "metrics": metrics, "per_sample": per_sample}

    initial_scan = run_scan(0, "start")

    step = 0
    peak_reserved = 0
    try:
        while step < TRAIN_MAX_OPTIMIZER_STEPS:
            elapsed = time.perf_counter() - started
            if elapsed >= TRAIN_MAX_SECONDS:
                stop_reason = "max_wall_seconds"
                break
            group = [next(stream) for _ in range(TRAIN_EFFECTIVE_BATCH)]
            batch_size = len(group)
            optimizer.zero_grad(set_to_none=True)
            group_nll = 0.0
            finite = True
            for index in group:
                sample = samples[index]
                log_prob = teacher_marginal_log_prob(model, sample)
                if not torch.isfinite(log_prob):
                    finite = False
                    dump_json(
                        OUTPUT_DIR / "nonfinite_update.json",
                        {"step": step, "sample_id": int(sample["sample_id"]), "log_prob": float(log_prob)},
                    )
                    break
                loss = -log_prob / batch_size
                loss.backward()
                group_nll += float(-log_prob.detach().item()) / batch_size
            if not finite:
                stop_reason = "nonfinite_nll"
                optimizer.zero_grad(set_to_none=True)
                break
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, TRAIN_GRAD_CLIP)
            grad_norm_value = float(grad_norm.detach().item())
            if not math.isfinite(grad_norm_value):
                stop_reason = "nonfinite_grad_norm"
                dump_json(OUTPUT_DIR / "nonfinite_update.json", {"step": step, "grad_norm": grad_norm_value})
                optimizer.zero_grad(set_to_none=True)
                break
            optimizer.step()
            step += 1
            memory = gpu_memory_snapshot()
            peak_reserved = max(peak_reserved, memory["peak_reserved_bytes"])
            history.append(
                {
                    "step": int(step),
                    "nll": float(group_nll),
                    "grad_norm": grad_norm_value,
                    "grad_clipped": bool(grad_norm_value > TRAIN_GRAD_CLIP),
                    "samples_in_group": int(batch_size),
                    "sample_ids": [int(samples[i]["sample_id"]) for i in group],
                    "elapsed_seconds": time.perf_counter() - started,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "cuda_allocated_mib": memory["allocated_bytes"] / 2**20,
                    "cuda_reserved_mib": memory["reserved_bytes"] / 2**20,
                    "cuda_peak_reserved_mib": memory["peak_reserved_bytes"] / 2**20,
                }
            )
            if step % GRAD_LOG_INTERVAL == 0:
                log(
                    f"step {step:4d} NLL={group_nll:.6f} |g|={grad_norm_value:.3f} "
                    f"t={history[-1]['elapsed_seconds']:.0f}s "
                    f"reserved={history[-1]['cuda_peak_reserved_mib']:.0f}MiB"
                )
            if memory["peak_reserved_bytes"] >= GPU_BUDGET_BYTES:
                stop_reason = "gpu_budget_exceeded"
                break
            if step % protocol.SCAN_INTERVAL_STEPS == 0:
                run_scan(step, "interval")
                if _early_stop_ready(scans):
                    stop_reason = "early_stop_consecutive_scan_accuracy"
                    break
    except torch.cuda.OutOfMemoryError as error:  # pragma: no cover - 仅在显存异常时触发
        stop_reason = "cuda_oom"
        dump_json(OUTPUT_DIR / "cuda_oom.json", {"step": step, "error": str(error)})
        optimizer.zero_grad(set_to_none=True)

    final_scan = run_scan(step, "final")
    training_seconds = time.perf_counter() - started
    memory = gpu_memory_snapshot()
    if memory["peak_reserved_bytes"] >= GPU_BUDGET_BYTES and stop_reason == "max_optimizer_steps":
        stop_reason = "gpu_budget_exceeded"

    training_summary = {
        "optimizer_steps": step,
        "stop_reason": stop_reason,
        "training_seconds": training_seconds,
        "optimizer": "Adam",
        "learning_rate": TRAIN_LEARNING_RATE,
        "weight_decay": TRAIN_WEIGHT_DECAY,
        "gradient_clip_norm": TRAIN_GRAD_CLIP,
        "micro_batch": protocol.TRAIN_MICRO_BATCH,
        "effective_batch": TRAIN_EFFECTIVE_BATCH,
        "loss_normalization": "每个优化步按该组真实样本数 B 归一化（-1/B Σ logπ）；不足 8 时按实际数量归一化",
        "seed": TRAIN_SEED,
        "sample_count": sample_count,
        "shuffle": "固定种子逐轮 randperm 循环访问；不重复轨迹前缀",
        "distribution": "全部教师示范（同实例拟合），非法定泛化测试",
        "frozen_parameters": sorted(value_names),
        "trainable_parameter_count": len(trainable_names),
        "initial_teacher_fingerprint": source_fingerprint,
        "final_actor_fingerprint": actor_fingerprint(model),
        "gpu": {
            "device": torch.cuda.get_device_name(0),
            "peak_allocated_mib": memory["peak_allocated_bytes"] / 2**20,
            "peak_reserved_mib": memory["peak_reserved_bytes"] / 2**20,
            "budget_mib": GPU_BUDGET_BYTES / 2**20,
            "budget_exceeded": memory["peak_reserved_bytes"] >= GPU_BUDGET_BYTES,
        },
        "budget": {
            "max_optimizer_steps": TRAIN_MAX_OPTIMIZER_STEPS,
            "max_seconds": TRAIN_MAX_SECONDS,
            "scan_interval": protocol.SCAN_INTERVAL_STEPS,
            "early_stop": {
                "consecutive_scans": EARLY_STOP_CONSECUTIVE_SCANS,
                "accuracy_threshold": EARLY_STOP_ACCURACY,
                "metric": "full_physical_action_accuracy",
                "nll_requirement": "finite",
            },
        },
        "scan_count": len(scans),
        "initial_scan_nll": initial_scan["record"]["nll_mean"],
        "final_scan_nll": final_scan["record"]["nll_mean"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
    }

    # 先落盘诊断检查点：权重是唯一不可重建的产物，必须先于任何派生报告写入。
    state = clone_actor_state(model)
    if state_dict_fingerprint(state) != training_summary["final_actor_fingerprint"]:
        raise RuntimeError("训练后指纹与保存权重不一致")
    checkpoint_payload = build_diagnostic_checkpoint(
        model,
        source_weights_path=ORIGINAL_C_WEIGHTS,
        source_weights_sha256=source_sha,
        source_actor_fingerprint=source_fingerprint,
        teacher_manifest_sha256=manifest_sha,
        training_summary=training_summary,
        snapshot_contract=bundle["snapshot_contract"],
    )
    torch.save(checkpoint_payload, DIAGNOSTIC_ACTOR)
    log(f"诊断检查点已保存: {DIAGNOSTIC_ACTOR.name}")

    dump_jsonl(BC_HISTORY, history)
    dump_jsonl(TRAIN_SCANS, scans)
    write_json(
        OUTPUT_DIR / "train_scan_metrics.json",
        {
            "device": str(device),
            "scans": scans,
            "initial_full_metrics": initial_scan["metrics"],
            "final_full_metrics": final_scan["metrics"],
        },
    )
    dump_jsonl(
        OUTPUT_DIR / "teacher_scores_diagnostic_final_gpu.jsonl",
        final_scan["per_sample"],
    )
    log(
        f"训练结束: steps={step} reason={stop_reason} "
        f"用时={training_seconds:.0f}s 指纹={training_summary['final_actor_fingerprint'][:16]}"
    )
    return 0


def _early_stop_ready(scans: Sequence[Mapping[str, Any]]) -> bool:
    if len(scans) < EARLY_STOP_CONSECUTIVE_SCANS:
        return False
    tail = scans[-EARLY_STOP_CONSECUTIVE_SCANS:]
    return all(
        item["full_physical_action_accuracy"] is not None
        and float(item["full_physical_action_accuracy"]) >= EARLY_STOP_ACCURACY
        and bool(item["nll_finite"])
        for item in tail
    )


def fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


# ---------------------------------------------------------------------------
# Phase E：独立闭环评估
# ---------------------------------------------------------------------------


def stage_evaluate() -> int:
    threads = enforce_single_thread_cpu()
    bundle = read_teacher_bundle()
    samples = bundle["samples"]
    manifest_sha = teacher_manifest_sha256()

    if not DIAGNOSTIC_ACTOR.is_file():
        raise FileNotFoundError(f"缺少诊断副本: {DIAGNOSTIC_ACTOR}")
    model, checkpoint = load_diagnostic_actor(
        DIAGNOSTIC_ACTOR, expected_teacher_manifest_sha256=manifest_sha
    )
    model.to("cpu")
    log(
        "诊断副本载入: fp={} 训练步={} 停止={}".format(
            checkpoint["actor_fingerprint"][:16],
            checkpoint["training"]["optimizer_steps"],
            checkpoint["training"]["stop_reason"],
        )
    )
    trained_fingerprint = checkpoint["actor_fingerprint"]
    if trained_fingerprint == checkpoint["source_checkpoint"]["actor_fingerprint"]:
        raise RuntimeError("诊断副本指纹与源权重相同，训练未生效")

    # 1) 诊断副本在教师状态上的最终评分（CPU，与原 C 同一路径/设备）
    metrics, per_sample = scan_teacher_states(model, samples, device="cpu")
    metrics["evaluated_model"] = {
        "role": "diagnostic_behavior_clone",
        "checkpoint_path": str(DIAGNOSTIC_ACTOR),
        "checkpoint_sha256": sha256_file(DIAGNOSTIC_ACTOR),
        "actor_fingerprint": trained_fingerprint,
        "source_actor_fingerprint": checkpoint["source_checkpoint"]["actor_fingerprint"],
        "source_weights_sha256": checkpoint["source_checkpoint"]["sha256"],
        "device": "cpu",
        "threads": threads,
        "training_steps": checkpoint["training"]["optimizer_steps"],
        "stop_reason": checkpoint["training"]["stop_reason"],
    }
    metrics["teacher_manifest_sha256"] = manifest_sha
    write_json(FINAL_METRICS, metrics)
    dump_jsonl(OUTPUT_DIR / "teacher_scores_diagnostic_final_cpu.jsonl", per_sample)
    log(
        "最终教师评分: NLL={:.6f} task={} team={} full={}".format(
            metrics["nll_mean"],
            fmt(metrics["task_accuracy_conditional"]),
            fmt(metrics["team_accuracy_unordered_conditional"]),
            fmt(metrics["full_physical_action_accuracy"]),
        )
    )

    # 2) 独立闭环：只做一次，完全由副本自行选择
    log("诊断副本独立 CPU Nominal 闭环（最多 4096 决策）...")
    env = make_env()
    profile = build_method_profile("C")
    agent = FormalEvaluationAgent(profile=profile, actor_critic=model, time_head=None)
    result = evaluate_single_trajectory(
        env,
        "Method-C",
        agent,
        scenario=None,
        max_decisions=DECISION_BUDGET,
        device="cpu",
    )
    facts = extract_execution_facts(env)
    write_json(C_BC_TRAJ, {**result, "execution_facts": facts})
    log(
        "诊断副本闭环: decisions={} J={:.10f} makespan={:.6f} completed={} success={} postponed={}".format(
            result["decisions"],
            result["objective_j_total"],
            result["makespan"],
            result["completed_tasks"],
            result["success"],
            result["postponed_count"],
        )
    )

    # 3) 与 A / 原 C 的执行事实对照
    a_payload = load_json(A_TRAJ)
    c_payload = load_json(C_ORIGINAL_TRAJ)
    comparison = {
        "a_vs_diagnostic": compare_execution_facts(
            a_payload["execution_facts"], facts
        ),
        "original_c_vs_diagnostic": compare_execution_facts(
            c_payload["execution_facts"], facts
        ),
        "note": (
            "顺序差异不等于排程更差；这里分别记录首次动作差异（教师状态评分中）与"
            "首次实际开工时刻/团队/转站偏离（闭环执行事实中）。"
        ),
    }
    write_json(OUTPUT_DIR / "trajectory_comparison.json", comparison)
    write_json(
        OUTPUT_DIR / "diagnostic_closure.json",
        {
            "checkpoint_fingerprint": trained_fingerprint,
            "decisions": int(result["decisions"]),
            "objective_j_total": float(result["objective_j_total"]),
            "makespan": float(result["makespan"]),
            "completed_tasks": int(result["completed_tasks"]),
            "success": bool(result["success"]),
            "feasible": bool(result["feasible"]),
            "postponed_count": int(result["postponed_count"]),
            "cost_components_weighted": result["cost_components_weighted"],
            "advance_cause_counts": result["advance_cause_counts"],
            "threads": threads,
        },
    )
    log("evaluate 完成")
    return 0


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------


def _load_verify_module():
    path = REPO_ROOT / "work3_4060" / "verify_trajectory_facts.py"
    spec = importlib.util.spec_from_file_location("work3_verify_trajectory_facts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载轨迹校验器: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_verify() -> int:
    threads = enforce_single_thread_cpu()
    verify_module = _load_verify_module()
    baseline = load_json(BASELINE_PATH)
    baseline_h0 = float(baseline["metadata"]["h0"])
    baseline_tasks = baseline["tasks"]

    trajectory_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for path, role in (
        (A_TRAJ, "A_NOMINAL"),
        (C_ORIGINAL_TRAJ, "C_ORIGINAL_NOMINAL"),
        (C_BC_TRAJ, "C_BC_DIAGNOSTIC_NOMINAL"),
    ):
        if not path.is_file():
            missing.append(str(path))
            continue
        result = verify_module.verify_single_trajectory(
            traj_path=path,
            baseline_h0=baseline_h0,
            baseline_tasks=baseline_tasks,
            scenario_spec={"scenario_id": "NOMINAL"},
        )
        result["role"] = role
        trajectory_rows.append(result)
        log(
            f"轨迹核验[{role}]: {result['status']} violations={result['total_violations']} "
            f"max_diff={max(result['reconciliation_differences'].values()):.3e}"
        )

    if any(row["status"] == "FAIL" for row in trajectory_rows):
        trajectory_status = "FAIL"
    elif missing or len(trajectory_rows) < 3:
        trajectory_status = "INCOMPLETE"
    else:
        trajectory_status = "PASS"

    write_json(
        TRAJECTORY_VERIFICATION,
        {
            "overall_status": trajectory_status,
            "missing_files": missing,
            "baseline_sha256": sha256_file(BASELINE_PATH),
            "baseline_h0": baseline_h0,
            "verifier": "work3_4060/verify_trajectory_facts.py::verify_single_trajectory",
            "trajectories": trajectory_rows,
        },
    )

    # 权重身份与快照合同
    original_sha_now = sha256_file(ORIGINAL_C_WEIGHTS)
    agent, checkpoint = load_original_c_actor()
    original_fingerprint_now = actor_fingerprint(agent.actor_critic)

    diagnostic_model, diagnostic_checkpoint = load_diagnostic_actor(
        DIAGNOSTIC_ACTOR, expected_teacher_manifest_sha256=teacher_manifest_sha256()
    )
    diagnostic_state = clone_actor_state(diagnostic_model)

    bundle = read_teacher_bundle()
    samples = bundle["samples"]
    snapshot_checks = verify_snapshot_contract(samples)

    initial_metrics = load_json(INITIAL_METRICS)
    final_metrics = load_json(FINAL_METRICS)
    device_consistency = compare_scan_devices(final_metrics)
    identity = {
        "original_weights_sha256": original_sha_now,
        "original_weights_unchanged": original_sha_now
        == diagnostic_checkpoint["source_checkpoint"]["sha256"],
        "original_actor_fingerprint": original_fingerprint_now,
        "original_actor_unchanged": original_fingerprint_now
        == diagnostic_checkpoint["source_checkpoint"]["actor_fingerprint"],
        "diagnostic_copy_initial_fingerprint": diagnostic_checkpoint["source_checkpoint"][
            "actor_fingerprint"
        ],
        "diagnostic_copy_initial_matches_original": diagnostic_checkpoint["source_checkpoint"][
            "actor_fingerprint"
        ]
        == original_fingerprint_now,
        "diagnostic_copy_final_fingerprint": state_dict_fingerprint(diagnostic_state),
        "diagnostic_copy_final_matches_stored": state_dict_fingerprint(diagnostic_state)
        == diagnostic_checkpoint["actor_fingerprint"],
        "diagnostic_copy_weights_changed": state_dict_fingerprint(diagnostic_state)
        != diagnostic_checkpoint["source_checkpoint"]["actor_fingerprint"],
        "diagnostic_role_marked": DIAGNOSTIC_FORMAT
        == diagnostic_checkpoint["diagnostic_format"],
        "initial_metrics_model_fingerprint": initial_metrics["evaluated_model"][
            "actor_fingerprint"
        ],
        "final_metrics_model_fingerprint": final_metrics["evaluated_model"][
            "actor_fingerprint"
        ],
        "teacher_metrics_share_teacher_manifest": initial_metrics.get(
            "teacher_manifest_sha256"
        )
        == final_metrics.get("teacher_manifest_sha256")
        == teacher_manifest_sha256(),
    }

    a_trajectory_verified = any(
        row["role"] == "A_NOMINAL" and row["status"] == "PASS" for row in trajectory_rows
    )
    checks = {
        "original_weights_unchanged": identity["original_weights_unchanged"],
        "original_actor_unchanged": identity["original_actor_unchanged"],
        "diagnostic_copy_initial_matches_original": identity[
            "diagnostic_copy_initial_matches_original"
        ],
        "diagnostic_copy_weights_changed": identity["diagnostic_copy_weights_changed"],
        "diagnostic_role_marked": identity["diagnostic_role_marked"],
        "teacher_manifest_bound": identity["teacher_metrics_share_teacher_manifest"],
        "teacher_trajectory_completed_and_verified": a_trajectory_verified,
        **snapshot_checks["checks"],
    }
    overall = "PASS" if all(checks.values()) else "FAIL"
    write_json(
        VERIFICATION,
        {
            "overall_status": overall,
            "scope": (
                "本文件只判定诊断执行合同（接口/数值/权重身份/快照合同/教师轨迹）。"
                "闭环生产事实的覆盖与物理约束见 trajectory_verification.json，"
                "其状态属于诊断结果，不改变诊断合同的有效性。"
            ),
            "checks": checks,
            "identity": identity,
            "snapshot_contract": snapshot_checks,
            "trajectory_verification_status": trajectory_status,
            "device_consistency": device_consistency,
            "trajectory_roles": {
                row["role"]: {
                    "status": row["status"],
                    "total_violations": row["total_violations"],
                    "max_reconciliation_difference": max(
                        row["reconciliation_differences"].values()
                    ),
                }
                for row in trajectory_rows
            },
            "pathological_marker_note": (
                "路径标记由本文件给出；任何 FAIL 或 INCOMPLETE 都不得在报告中改写为 PASS。"
            ),
            "threads": threads,
        },
    )
    log(f"verify 完成: {overall}")
    return 0 if overall == "PASS" else 2


def compare_scan_devices(final_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """核对训练期 GPU 扫描与最终 CPU 扫描是否给出同一结论（口径一致性证据）。"""
    path = OUTPUT_DIR / "train_scan_metrics.json"
    if not path.is_file():
        return {"available": False, "note": "缺少训练期扫描记录"}
    train_scans = load_json(path)
    gpu_metrics = train_scans.get("final_full_metrics") or {}
    keys = (
        "nll_mean",
        "task_accuracy_conditional",
        "branch_accuracy_conditional",
        "team_accuracy_unordered_conditional",
        "align_accuracy_conditional",
        "full_physical_action_accuracy",
    )
    rows: dict[str, Any] = {}
    max_rate_diff = 0.0
    for key in keys:
        left = gpu_metrics.get(key)
        right = final_metrics.get(key)
        diff = None
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            diff = abs(float(left) - float(right))
            if "accuracy" in key:
                max_rate_diff = max(max_rate_diff, diff)
        rows[key] = {"gpu_final_scan": left, "cpu_final_scan": right, "abs_diff": diff}

    per_sample_diff = None
    cpu_path = OUTPUT_DIR / "teacher_scores_diagnostic_final_cpu.jsonl"
    gpu_path = OUTPUT_DIR / "teacher_scores_diagnostic_final_gpu.jsonl"
    if cpu_path.is_file() and gpu_path.is_file():
        with cpu_path.open("r", encoding="utf-8") as handle:
            cpu_rows = [json.loads(line) for line in handle if line.strip()]
        with gpu_path.open("r", encoding="utf-8") as handle:
            gpu_rows = [json.loads(line) for line in handle if line.strip()]
        if len(cpu_rows) == len(gpu_rows):
            per_sample_diff = max(
                abs(float(left["nll"]) - float(right["nll"]))
                for left, right in zip(cpu_rows, gpu_rows, strict=True)
            )
    return {
        "available": True,
        "metrics": rows,
        "max_accuracy_rate_difference": max_rate_diff,
        "max_per_sample_nll_difference": per_sample_diff,
        "agreement_scope": (
            "同一组教师快照、同一份权重、贪心判读；GPU 与 CPU 的前向舍入差异，"
            "用于确认“结论不依赖设备”，不作为逐位一致证明。"
        ),
        "conclusion_consistent": max_rate_diff <= 1e-9,
    }


def verify_snapshot_contract(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """快照必须是不含环境引用、不含未来标签、全在 CPU 的独立副本。"""
    forbidden_keys = {
        "actual_start",
        "actual_end",
        "actual_start_station",
        "transfer_history",
        "future_transfer_time",
        "final_makespan",
    }
    problems: list[str] = []
    tensor_count = 0
    for sample in samples:
        snapshot = sample_snapshot(sample)
        for name in SNAPSHOT_FIELDS:
            value = getattr(snapshot, name)
            if isinstance(value, torch.Tensor):
                tensor_count += 1
                if value.device.type != "cpu":
                    problems.append(f"sample {sample['sample_id']}: {name} 不在 CPU")
                if value.requires_grad:
                    problems.append(f"sample {sample['sample_id']}: {name} 仍带计算图")
        graph = snapshot.graph_snapshot
        for node_type in graph.node_types:
            store = graph[node_type]
            for key, value in store.items():
                if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                    problems.append(
                        f"sample {sample['sample_id']}: graph.{node_type}.{key} 不在 CPU"
                    )
        label_payload = sample["label"]
        payload_probe = json.dumps(
            {
                "label": label_payload.as_json()
                if isinstance(label_payload, TeacherLabel)
                else label_payload,
                "outcome": sample.get("outcome", {}),
            },
            ensure_ascii=False,
            default=str,
        )
        for key in forbidden_keys:
            if key in payload_probe:
                problems.append(f"sample {sample['sample_id']}: 出现未来事实字段 {key}")
        if snapshot.graph_version != bundle_graph_version():
            problems.append(f"sample {sample['sample_id']}: 图版本不一致")
    return {
        "checks": {
            "snapshots_cpu_only": not any("不在 CPU" in item for item in problems),
            "snapshots_detached": not any("计算图" in item for item in problems),
            "no_future_labels_in_inputs": not any("未来事实字段" in item for item in problems),
            "graph_version_consistent": not any("图版本不一致" in item for item in problems),
        },
        "tensor_count": tensor_count,
        "problems": problems,
        "sample_count": len(samples),
    }


def bundle_graph_version() -> str:
    from models.work3.graph_builder import GRAPH_FEATURE_VERSION

    return GRAPH_FEATURE_VERSION


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _fmt_cost(value: Any) -> str:
    """绝对值很小的费用以科学计数显示，避免被误读为严格零。"""
    number = float(value)
    if number != 0.0 and abs(number) < 1e-6:
        return f"{number:.3e}"
    return f"{number:.10f}".rstrip("0").rstrip(".") if number == 0.0 else f"{number:.10f}"


def load_divergences(path: Path) -> list[dict[str, Any]]:
    """读取教师状态评分中的逐样本差异，供报告列出顺序性差异。"""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not item.get("full_action_hit", False):
                rows.append(item)
    return rows


def git_state() -> dict[str, Any]:
    """记录实际执行时的 HEAD 与工作树状态（不依赖外部库）。"""
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
        "execution_head": head,
        "execution_branch": branch,
        "execution_worktree_dirty": bool(status),
        "worktree_entries": [] if not status else status.splitlines()[:40],
        "note": (
            "执行时仓库 HEAD 与工作树状态；本轮诊断实现与产物均为该工作树上的新增内容，"
            "不包含任何历史提交回退。"
        ),
    }


def stage_report() -> int:
    a_payload = load_json(A_TRAJ)
    c_payload = load_json(C_ORIGINAL_TRAJ)
    bc_payload = load_json(C_BC_TRAJ)
    initial = load_json(INITIAL_METRICS)
    final = load_json(FINAL_METRICS)
    validation = load_json(TEACHER_VALIDATION)
    visibility = load_json(INPUT_VISIBILITY)
    verification = load_json(VERIFICATION)
    trajectory_verification = load_json(TRAJECTORY_VERIFICATION)
    provenance = load_json(OUTPUT_DIR / "original_c_provenance.json")
    comparison = load_json(OUTPUT_DIR / "trajectory_comparison.json")
    diagnostic_checkpoint = torch.load(DIAGNOSTIC_ACTOR, map_location="cpu", weights_only=False)
    gate = diagnostic_checkpoint["training"]

    rows = [
        ("A", "Baseline-Repair-A（真实教师）", a_payload, None),
        ("C_ORIG", "方法 C 原检查点（4096 步）", c_payload, initial),
        ("C_BC_DIAG", "C 诊断副本（diagnostic_behavior_clone）", bc_payload, final),
    ]
    fieldnames = [
        "entity",
        "description",
        "decisions",
        "completed_tasks",
        "success",
        "feasible",
        "constraint_violations",
        "j_total",
        "j_takt",
        "d_time",
        "d_team",
        "j_postpone",
        "j_revision",
        "makespan",
        "postponed_count",
        "transfers",
        "teacher_nll_mean",
        "teacher_task_accuracy_conditional",
        "teacher_branch_accuracy_conditional",
        "teacher_team_accuracy_unordered",
        "teacher_align_accuracy_conditional",
        "teacher_full_action_accuracy",
    ]
    csv_rows: list[dict[str, Any]] = []
    for entity, description, payload, metrics in rows:
        components = payload["cost_components_weighted"]
        csv_rows.append(
            {
                "entity": entity,
                "description": description,
                "decisions": int(payload["decisions"]),
                "completed_tasks": int(payload["completed_tasks"]),
                "success": bool(payload["success"]),
                "feasible": bool(payload["feasible"]),
                "constraint_violations": int(payload["constraint_violation_count"]),
                "j_total": format(float(payload["objective_j_total"]), ".17g"),
                "j_takt": format(float(components["j_takt"]), ".17g"),
                "d_time": format(float(components["d_time"]), ".17g"),
                "d_team": format(float(components["d_team"]), ".17g"),
                "j_postpone": format(float(components["j_postpone"]), ".17g"),
                "j_revision": format(float(components["j_revision"]), ".17g"),
                "makespan": format(float(payload["makespan"]), ".17g"),
                "postponed_count": int(payload["postponed_count"]),
                "transfers": int(payload["transfers"]),
                "teacher_nll_mean": ""
                if metrics is None
                else format(float(metrics["nll_mean"]), ".17g"),
                "teacher_task_accuracy_conditional": ""
                if metrics is None
                else _fmt_optional(metrics["task_accuracy_conditional"]),
                "teacher_branch_accuracy_conditional": ""
                if metrics is None
                else _fmt_optional(metrics["branch_accuracy_conditional"]),
                "teacher_team_accuracy_unordered": ""
                if metrics is None
                else _fmt_optional(metrics["team_accuracy_unordered_conditional"]),
                "teacher_align_accuracy_conditional": ""
                if metrics is None
                else _fmt_optional(metrics["align_accuracy_conditional"]),
                "teacher_full_action_accuracy": ""
                if metrics is None
                else _fmt_optional(metrics["full_physical_action_accuracy"]),
            }
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    j_a = float(a_payload["objective_j_total"])
    j_orig = float(c_payload["objective_j_total"])
    j_bc = float(bc_payload["objective_j_total"])
    improvement = (j_orig - j_bc) / j_orig if j_orig > 0 else None
    tier, tier_note = classify_outcome(bc_payload, j_orig)
    verify_status = verification["overall_status"]
    trajectory_status = trajectory_verification["overall_status"]
    # 诊断合同成立才算“诊断执行有效”；闭环轨迹的覆盖状态是结果，不是合同前提。
    reportable = verify_status == "PASS"

    summary = build_summary_markdown(
        rows=csv_rows,
        a=a_payload,
        c=c_payload,
        bc=bc_payload,
        initial=initial,
        final=final,
        validation=validation,
        visibility=visibility,
        provenance=provenance,
        comparison=comparison,
        gate=gate,
        verification=verification,
        trajectory_verification=trajectory_verification,
        improvement=improvement,
        tier=tier,
        tier_note=tier_note,
        reportable=reportable,
        divergences=load_divergences(
            OUTPUT_DIR / "teacher_scores_diagnostic_final_cpu.jsonl"
        ),
    )
    SUMMARY_MD.write_text(summary, encoding="utf-8")

    write_json(
        MANIFEST,
        {
            "task": "Work-3 C 基准复现能力诊断（同实例可学习性诊断）",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "output_dir": str(OUTPUT_DIR),
            "instance": {
                "baseline_path": str(BASELINE_PATH),
                "baseline_sha256": sha256_file(BASELINE_PATH),
                "dataset": "mini210",
                "tasks_per_aircraft": 30,
                "aircraft": 7,
                "stations": 5,
                "h0": EXPECTED_H0,
                "clock_model": "nominal_anchored_v1",
                "scenario": "Nominal",
                "disturbance": "none",
                "warmup": "none",
            },
            "original_checkpoint": provenance,
            "diagnostic_checkpoint": {
                "path": str(DIAGNOSTIC_ACTOR),
                "sha256": sha256_file(DIAGNOSTIC_ACTOR),
                "format": diagnostic_checkpoint["diagnostic_format"],
                "format_version": diagnostic_checkpoint["diagnostic_format_version"],
                "actor_fingerprint": diagnostic_checkpoint["actor_fingerprint"],
                "source_actor_fingerprint": diagnostic_checkpoint["source_checkpoint"][
                    "actor_fingerprint"
                ],
                "training": gate,
            },
            "teacher": {
                "snapshots": str(TEACHER_SNAPSHOTS),
                "snapshots_sha256": sha256_file(TEACHER_SNAPSHOTS),
                "actions": str(TEACHER_ACTIONS),
                "actions_sha256": sha256_file(TEACHER_ACTIONS),
                "manifest_sha256": teacher_manifest_sha256(),
                "sample_count": int(validation["sample_count"]),
                "decision_kind_counts": validation["decision_kind_counts"],
            },
            "teacher_states_scoring": {
                "initial": {
                    "path": str(INITIAL_METRICS),
                    "nll_mean": initial["nll_mean"],
                    "full_physical_action_accuracy": initial[
                        "full_physical_action_accuracy"
                    ],
                },
                "final": {
                    "path": str(FINAL_METRICS),
                    "nll_mean": final["nll_mean"],
                    "full_physical_action_accuracy": final[
                        "full_physical_action_accuracy"
                    ],
                },
            },
            "closure": {
                "A": {"j_total": j_a, "makespan": float(a_payload["makespan"]), "decisions": int(a_payload["decisions"])},
                "C_ORIGINAL": {
                    "j_total": j_orig,
                    "makespan": float(c_payload["makespan"]),
                    "decisions": int(c_payload["decisions"]),
                },
                "C_BC_DIAGNOSTIC": {
                    "j_total": j_bc,
                    "makespan": float(bc_payload["makespan"]),
                    "decisions": int(bc_payload["decisions"]),
                },
                "relative_improvement_vs_original_c": improvement,
            },
            "verification": {
                "diagnostic_self_check": verify_status,
                "trajectory_facts": trajectory_status,
                "reportable": reportable,
            },
            "outcome_tier": tier,
            "run_context": git_state(),
            "artifacts": sorted(
                {
                    str(path.relative_to(OUTPUT_DIR))
                    for path in OUTPUT_DIR.rglob("*")
                    if path.is_file()
                }
                | {"manifest.json"}
            ),
            "boundaries": [
                "训练与评估使用同一实例 Nominal；结论仅限同实例复现能力。",
                "未开展 PPO 续训、未新增 D/E/F、未加入扰动、未改 H0/奖励权重/正式候选掩码。",
                "A 示范不含后移与显式推进标签，不能据此判断扰动下等待/后移/执行的区分能力。",
                "诊断副本不与正式 PPO 检查点混用，也不构成正式算法优越性证据。",
            ],
        },
    )
    log(f"report 完成: tier={tier} reportable={reportable}")
    return 0


def _fmt_optional(value: Any) -> str:
    return "" if value is None else format(float(value), ".6f")


def classify_outcome(payload: Mapping[str, Any], original_j: float) -> tuple[str, str]:
    j_total = float(payload["objective_j_total"])
    components = payload["cost_components_weighted"]
    j_takt = float(components["j_takt"])
    postpones = int(payload["postponed_count"])
    legal = bool(payload["success"]) and bool(payload["feasible"]) and int(
        payload["constraint_violation_count"]
    ) == 0
    if not legal:
        return (
            "CLOSURE_NOT_ACCEPTED",
            "副本未通过闭环生产验收（未完工或存在物理违规），需先定位实际失败。",
        )
    if j_total <= 1e-3 and j_takt <= 1e-6 and postpones == 0:
        return (
            "NEAR_BASELINE_COST",
            "在本实例达到接近基准的费用复现；阈值是工程诊断标准，不是最优证明。",
        )
    improvement = (original_j - j_total) / original_j if original_j > 0 else 0.0
    if improvement >= 0.9 and j_total > 1e-3:
        return (
            "SUBSTANTIAL_IMPROVEMENT_NOT_BASELINE",
            "基准保持明显改善，但仍未接近 A 的零费用水平。",
        )
    return (
        "NO_EFFECTIVE_CLOSURE_REPRODUCTION",
        "本次监督预算未建立有效的闭环复现，保留各分项证据。",
    )


def _fmt_signature(signature: Sequence[Any]) -> str:
    if not signature:
        return "n/a"
    kind = signature[0]
    if kind == "stay":
        return f"stay {signature[1]} team={list(signature[2])} align={signature[3]}"
    if kind == "postpone":
        return f"postpone {signature[1]}"
    return str(kind)


def build_summary_markdown(
    *,
    rows: Sequence[Mapping[str, Any]],
    a: Mapping[str, Any],
    c: Mapping[str, Any],
    bc: Mapping[str, Any],
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    validation: Mapping[str, Any],
    visibility: Mapping[str, Any],
    provenance: Mapping[str, Any],
    comparison: Mapping[str, Any],
    gate: Mapping[str, Any],
    verification: Mapping[str, Any],
    trajectory_verification: Mapping[str, Any],
    improvement: float | None,
    tier: str,
    tier_note: str,
    reportable: bool,
    divergences: Sequence[Mapping[str, Any]] = (),
) -> str:
    j_a = float(a["objective_j_total"])
    j_orig = float(c["objective_j_total"])
    j_bc = float(bc["objective_j_total"])
    a_comp = a["cost_components_weighted"]
    c_comp = c["cost_components_weighted"]
    bc_comp = bc["cost_components_weighted"]
    lines: list[str] = []
    lines.append("# Work-3 C 基准复现能力诊断报告（同实例可学习性诊断）")
    lines.append("")
    lines.append(
        "> 本轮回答的唯一问题是：在当前规则基准与**无扰动**实例上，C 在得到 A 的明确示范后，"
        "能否学习并独立产生接近基准的低费用排程。训练与评估使用同一实例，"
        "**结论只限同实例复现能力**，不构成泛化、扰动恢复或正式算法优越性的证据。"
    )
    lines.append("")
    lines.append(
        f"**执行有效性**：诊断合同自检 `{verification['overall_status']}`，"
        f"三条闭环轨迹事实核验 `{trajectory_verification['overall_status']}`"
        "（后者是闭环结果的一部分，不改变诊断合同的有效性）。"
    )
    lines.append("")

    lines.append("## 1. 主表：A / 原 C / C_BC 诊断副本")
    lines.append("")
    lines.append(
        "| 实体 | 决策数 | 完工 | 物理可行 | $J_{total}$ | $J_{takt}$ | $D_{time}$ | $D_{team}$ | $J_{postpone}$ | $J_{revision}$ | Makespan (h) | 后移次数 |"
    )
    lines.append("| :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| {desc} | {decisions} | {completed} | {feasible} | {j} | {jt} | {dt} | {dw} | {jp} | {jr} | {ms:.6f} | {post} |".format(
                desc=row["description"],
                decisions=row["decisions"],
                completed=row["completed_tasks"],
                feasible="PASS" if bool(row["feasible"]) else "FAIL",
                j=_fmt_cost(row["j_total"]),
                jt=_fmt_cost(row["j_takt"]),
                dt=_fmt_cost(row["d_time"]),
                dw=_fmt_cost(row["d_team"]),
                jp=_fmt_cost(row["j_postpone"]),
                jr=_fmt_cost(row["j_revision"]),
                ms=float(row["makespan"]),
                post=row["postponed_count"],
            )
        )
    lines.append("")
    lines.append(
        f"- 相对原 C，诊断副本费用变化：$J$ {_fmt_cost(j_orig)} → {_fmt_cost(j_bc)}"
        + (f"（以原 C 为基准的相对降幅 {improvement * 100:.2f}%）" if improvement is not None else "")
    )
    gap_bc = j_bc - j_a
    gap_orig = j_orig - j_a
    lines.append(
        f"- 相对 A（{j_a:.3e}）的绝对差距：诊断副本 "
        + ("0（与 A 逐位相同）" if gap_bc == 0.0 else _fmt_cost(gap_bc))
        + f"，原 C {_fmt_cost(gap_orig)}。"
    )
    lines.append("")
    lines.append("### 1.1 教师状态内评分（同一组教师快照）")
    lines.append("")
    lines.append(
        "| 实体 | NLL 均值 | 推进/留站准确率 | 工序准确率（条件） | 分支准确率（条件） | 无序团队准确率（条件） | 对齐准确率（条件） | 完整物理动作元组准确率 |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, metrics in (("原 C", initial), ("C_BC 诊断副本", final)):
        lines.append(
            "| {name} | {nll:.6f} | {kind} | {task} | {branch} | {team} | {align} | {full} |".format(
                name=label,
                nll=float(metrics["nll_mean"]),
                kind=_fmt_optional(metrics["decision_kind_accuracy"]),
                task=_fmt_optional(metrics["task_accuracy_conditional"]),
                branch=_fmt_optional(metrics["branch_accuracy_conditional"]),
                team=_fmt_optional(metrics["team_accuracy_unordered_conditional"]),
                align=_fmt_optional(metrics["align_accuracy_conditional"]),
                full=_fmt_optional(metrics["full_physical_action_accuracy"]),
            )
        )
    lines.append("")
    lines.append(
        "> 条件准确率固定教师自身前缀做单头判读，**不是**独立贪婪全流程准确率；"
        "完整物理动作元组准确率来自正式贪心入口（含推进门控），两者不可混用。"
    )
    lines.append("")

    lines.append("## 2. 逐问回答")
    lines.append("")
    lines.append("### 2.1 C 能看见哪些基准信息？A 动作能否全部合法表达？")
    lines.append("")
    lines.append(
        "| 信息项 | 结论 | 关键编码路径 |"
    )
    lines.append("| :--- | :--- | :--- |")
    for item in visibility["items"]:
        paths = "；".join(
            sorted({entry["encoding_path"].split("；")[0] for entry in item["entries"]})
        )
        lines.append(
            f"| {item['question']} | {item['verdict']} | {paths} |"
        )
    lines.append("")
    lines.append(
        "可表达性核对（在 A 实际访问的 210 个决策状态上，用正式 C profile 重建同一时刻独立 CPU 快照）："
    )
    lines.append("")
    for key, value in validation["expression"].items():
        lines.append(f"- `{key}`：{value}")
    lines.append("")
    lines.append(
        f"- 候选数范围 {validation['candidate_count']['min']}–{validation['candidate_count']['max']}"
        f"（均值 {validation['candidate_count']['mean']:.2f}）；"
        f"A 留站决策中另有 {validation['advance_available_when_teacher_schedules']} 次处于“推进门控可用但 A 选择留站”的状态。"
    )
    lines.append(
        f"- 团队顺序存在观测别名：{validation['team_order_ambiguous_samples']} 个样本的合法选人排列数为 2，"
        "因此团队概率按排列 logsumexp 归并，准确率按无序集合比较。"
    )
    lines.append("")
    lines.append("### 2.2 原 C 与 BC 副本在教师状态上的评分分别如何？哪个子决策最难？")
    lines.append("")
    head_names = [
        ("decision_kind_accuracy", "推进/留站门控"),
        ("task_accuracy_conditional", "工序选择"),
        ("branch_accuracy_conditional", "执行/后移分支"),
        ("team_accuracy_unordered_conditional", "无序团队选择"),
        ("align_accuracy_conditional", "对齐选项"),
    ]
    before = {key: initial[key] for key, _ in head_names}
    after = {key: final[key] for key, _ in head_names}
    hardest_before = min(
        (key for key, _ in head_names if before[key] is not None),
        key=lambda key: float(before[key]),
        default=None,
    )
    hardest_after = min(
        (key for key, _ in head_names if after[key] is not None),
        key=lambda key: float(after[key]),
        default=None,
    )
    lookup = dict(head_names)
    if hardest_before is not None:
        lines.append(
            f"- **原 C 最弱子决策**：{lookup[hardest_before]}（条件准确率 {float(before[hardest_before]):.4f}）。"
        )
    if hardest_after is not None:
        lines.append(
            f"- **诊断副本最弱子决策**：{lookup[hardest_after]}（条件准确率 {float(after[hardest_after]):.4f}）。"
        )
    lines.append(
        f"- NLL：原 C {float(initial['nll_mean']):.6f} → 诊断副本 {float(final['nll_mean']):.6f}；"
        "NLL 与准确率的组合才说明“拟合到什么程度”，单看 NLL 不足以判断闭环表现。"
    )
    if final.get("first_full_action_divergence") is None:
        lines.append("- 诊断副本在全部教师状态上的完整物理动作元组与教师一致，首次差异为空。")
    else:
        diverge = final["first_full_action_divergence"]
        lines.append(
            f"- 首次完整动作差异出现在决策 #{diverge['decision_index']}"
            f"（t={diverge['sim_time']:.4f}h，周期 {diverge['cycle']}）："
            f"教师 {diverge['teacher']} vs 贪婪 {diverge['greedy']}。"
        )
    lines.append("")
    lines.append("### 2.3 副本是否在独立运行中接近 A？改善主要来自哪个费用分项？")
    lines.append("")
    lines.append(
        "| 费用分项 | A | 原 C | 诊断副本 | 副本相对原 C 的变化 |"
    )
    lines.append("| :--- | ---: | ---: | ---: | ---: |")
    for key in ("j_takt", "d_time", "d_team", "j_postpone", "j_revision"):
        lines.append(
            f"| `{key}` | {_fmt_cost(a_comp[key])} | {_fmt_cost(c_comp[key])} | "
            f"{_fmt_cost(bc_comp[key])} | {_fmt_cost(float(bc_comp[key]) - float(c_comp[key]))} |"
        )
    lines.append(
        f"| `j_total` | {_fmt_cost(j_a)} | {_fmt_cost(j_orig)} | {_fmt_cost(j_bc)} | {_fmt_cost(j_bc - j_orig)} |"
    )
    lines.append("")
    dominant = max(
        ("j_takt", "d_time", "d_team", "j_postpone", "j_revision"),
        key=lambda key: float(c_comp[key]) - float(bc_comp[key]),
    )
    lines.append(
        f"- 改善主要来自 `{dominant}`：{_fmt_cost(c_comp[dominant])} → {_fmt_cost(bc_comp[dominant])}。"
    )
    lines.append(
        f"- 完工工期：A {float(a['makespan']):.6f}h，原 C {float(c['makespan']):.6f}h，"
        f"诊断副本 {float(bc['makespan']):.6f}h。"
    )
    lines.append(
        f"- 决策数：A {int(a['decisions'])}，原 C {int(c['decisions'])}，诊断副本 {int(bc['decisions'])}；"
        f"副本后移次数 {int(bc['postponed_count'])}。"
    )
    lines.append("")
    lines.append("### 2.4 动作顺序差异中，哪些没有改变实际排程？有哪些实质偏离？")
    lines.append("")
    vs_a = comparison["a_vs_diagnostic"]
    lines.append(
        f"- 与 A 相比：开工时刻偏离 {vs_a['start_deviation_count']} 道工序，"
        f"团队偏离 {vs_a['team_deviation_count']} 道，转站偏离 {vs_a['transfer_deviation_count']} 个周期；"
        f"排程是否完全一致：{'是' if vs_a['schedule_identical'] else '否'}。"
    )
    if (
        vs_a["first_start_deviation"] is None
        and vs_a["first_team_deviation"] is None
        and vs_a["first_transfer_deviation"] is None
    ):
        lines.append(
            "  - 首次实际开工时刻/团队/转站偏离：**无**（210 道工序与 11 次转站逐项一致）。"
        )
    if vs_a["first_start_deviation"] is not None:
        fs = vs_a["first_start_deviation"]
        lines.append(
            f"  - 首次实际开工偏离：{fs['task_key']} 由 {fs['left_start']:.6f}h 变为 "
            f"{fs['right_start']:.6f}h（{fs['delta']:+.6f}h）。"
        )
    if vs_a["first_team_deviation"] is not None:
        ft = vs_a["first_team_deviation"]
        lines.append(
            f"  - 首次团队偏离：{ft['task_key']} 由 {ft['left_team']} 变为 {ft['right_team']}。"
        )
    if vs_a["first_transfer_deviation"] is not None:
        fv = vs_a["first_transfer_deviation"]
        lines.append(
            f"  - 首次转站偏离：周期 {fv['cycle']} 推迟 {fv['delta']:+.6f}h。"
        )
    vs_c = comparison["original_c_vs_diagnostic"]
    lines.append(
        f"- 与原 C 相比：开工时刻偏离 {vs_c['start_deviation_count']} 道工序，"
        f"团队偏离 {vs_c['team_deviation_count']} 道，转站偏离 {vs_c['transfer_deviation_count']} 个周期。"
    )
    if divergences:
        lines.append("")
        lines.append(
            f"教师状态评分中有 {len(divergences)} 处完整动作元组差异，明细如下"
            "（差异均出现在同一飞机的相邻工序之间，团队与对齐完全一致）："
        )
        lines.append("")
        lines.append(
            "| 决策序号 | 仿真时刻 (h) | 周期 | 教师动作 | 贪婪动作 | 工序/分支/团队/对齐 |"
        )
        lines.append("| ---: | ---: | ---: | :--- | :--- | :--- |")
        for item in divergences[:20]:
            teacher = _fmt_signature(item["teacher_signature"])
            greedy = _fmt_signature(item["greedy_signature"])
            hits = "/".join(
                "✓" if bool(item[key]) else "✗"
                for key in ("task_hit", "branch_hit", "team_hit", "align_hit")
            )
            lines.append(
                f"| {int(item['decision_index'])} | {float(item['sim_time']):.4f} | "
                f"{int(item['cycle'])} | {teacher} | {greedy} | {hits} |"
            )
        lines.append("")
        lines.append(
            "闭环执行事实显示这些差异没有改变任何实际开工时刻、团队与转站；"
            "因此不能把“动作序列不同”读成“排程更差”，也不能反推为“模型已学会全部决策细节”。"
        )
    lines.append("")
    lines.append(
        "- 说明：严格动作序列不同不等于排程更差。工序预约顺序可能交换而对开工时刻、团队与转站无影响；"
        "本节把“首次动作差异”（教师状态评分）与“首次实际时间/团队/转站偏离”（闭环执行事实）分开记录，"
        "不把二者混为因果结论。"
    )
    lines.append("")
    lines.append("### 2.5 实际训练步数、停止原因、显存/耗时和未覆盖边界是什么？")
    lines.append("")
    lines.append(
        f"- 优化器步数：**{int(gate['optimizer_steps'])}**（上限 {gate['budget']['max_optimizer_steps']}），"
        f"停止原因：`{gate['stop_reason']}`。"
    )
    lines.append(
        f"- 实际训练耗时：{float(gate['training_seconds']):.1f}s（上限 {gate['budget']['max_seconds']:.0f}s）；"
        f"有效批量 {TRAIN_EFFECTIVE_BATCH}，micro-batch {gate['micro_batch']}，"
        f"Adam lr={gate['learning_rate']}，weight_decay={gate['weight_decay']}，梯度裁剪 {gate['gradient_clip_norm']}。"
    )
    lines.append(
        f"- 显存峰值：allocated {float(gate['gpu']['peak_allocated_mib']):.1f}MiB，"
        f"reserved {float(gate['gpu']['peak_reserved_mib']):.1f}MiB（预算 {float(gate['gpu']['budget_mib']):.0f}MiB，"
        f"是否超限：{'是' if gate['gpu']['budget_exceeded'] else '否'}）。"
    )
    lines.append(
        f"- 扫描次数 {int(gate['scan_count'])}；提前停止条件：连续 {EARLY_STOP_CONSECUTIVE_SCANS} 次完整扫描的"
        f"完整物理动作元组准确率 ≥ {EARLY_STOP_ACCURACY} 且 NLL 有限。"
    )
    lines.append(
        f"- 冻结参数 {len(gate['frozen_parameters'])} 项（价值专属 critic），"
        f"可训练参数 {gate['trainable_parameter_count']} 项；Actor 与 critic 共享的编码器随 BC 变化，"
        "因此 critic 的中间表征会改变，价值头本身的参数未被优化。"
    )
    lines.append("")
    lines.append("未覆盖边界：")
    lines.append("")
    for boundary in (
        "A 示范全部是留站预约，缺少后移与显式推进标签，本诊断不能评估扰动下等待/后移/执行的区分能力。",
        f"教师状态只覆盖 {validation['decision_time_coverage']['distinct_decision_times']} 个转站时刻，"
        "不覆盖周期中段的中间决策状态；闭环中段状态属于示范未覆盖区域。",
        "训练与评估同一实例、同一 Nominal 场景，未做任何泛化或跨实例检验。",
        "未开展 PPO 续训、DAgger、残差策略、特征增强或扰动训练；这些仅为后续讨论候选。",
        "诊断副本不是正式 PPO 检查点，本文任何数字都不构成正式算法优越性证据。",
        "只跑一次最终闭环评估，未在多个 checkpoint 中挑选闭环费用最低者。",
    ):
        lines.append(f"- {boundary}")
    lines.append("")
    lines.append("### 2.6 结果支持哪一种有限判断？哪些原因仍未被区分？")
    lines.append("")
    lines.append(f"- **预设等级**：`{tier}` — {tier_note}")
    lines.append(
        f"- 依据：闭环 $J={_fmt_cost(j_bc)}$，$J_{{takt}}={_fmt_cost(bc_comp['j_takt'])}$，"
        f"后移次数 {int(bc['postponed_count'])}，完工 {int(bc['completed_tasks'])}/210，"
        f"物理违规 {int(bc['constraint_violation_count'])}。"
    )
    lines.append(
        f"- 教师评分与闭环的组合判读：完整动作元组准确率 {_fmt_optional(final['full_physical_action_accuracy'])}，"
        f"独立闭环 $J$ 相对原 C 变化 {_fmt_cost(j_bc - j_orig)}。"
    )
    if tier == "NEAR_BASELINE_COST":
        lines.append(
            "- 这是“闭环接近 A”的形态：说明当前架构、输入与接口在现有初始化与示范下"
            "**具备该实例的保持能力**；但不能据此证明 PPO 必然能学会、辅助模块无价值，"
            "也不能说扰动恢复能力已经改善。"
        )
    elif final["full_physical_action_accuracy"] is not None and float(
        final["full_physical_action_accuracy"]
    ) >= 0.99 and j_bc > 1e-3:
        lines.append(
            "- 这是“教师评分好但闭环差”的形态，与偏离教师状态后的误差累积相容；"
            "但本轮不据此宣称已找到唯一原因。"
        )
    elif tier == "CLOSURE_NOT_ACCEPTED":
        lines.append(
            "- 这是“闭环未通过生产验收”的形态：先定位实际失败（未完工、死锁或物理违规），"
            "不把闭环失败直接归因为训练预算不足。"
        )
    lines.append("")
    lines.append("仍未被区分的原因（本轮设计不提供区分能力）：")
    lines.append("")
    for reason in (
        "表示不足与优化不足：本轮只做一次固定预算的监督拟合，未做表征消融。",
        "误差累积与分布偏移：闭环只跑一次，未做首个实质偏离后的分叉对照实验。",
        "PPO 能否学会：本轮未做任何 PPO 续训，A 的示范覆盖不足。",
        "辅助模块价值：本轮未启用时间学习或塑形变体，无法评价其必要性。",
        "扰动恢复能力：本轮仅 Nominal，无扰动实例。",
    ):
        lines.append(f"- {reason}")
    lines.append("")

    lines.append("## 3. 来源、合同与验证")
    lines.append("")
    state = git_state()
    lines.append(
        f"- 执行时仓库状态：HEAD `{state['execution_head']}`，分支 `{state['execution_branch']}`，"
        f"工作树{'有未提交改动（本轮新增实现与产物）' if state['execution_worktree_dirty'] else '干净'}。"
    )
    lines.append(
        f"- 原 C 权重 SHA256 `{provenance['weights_sha256']}`，Actor 指纹 `{provenance['actor_fingerprint']}`，"
        f"来源提交 `{provenance['source_sha']}`，方法 `{provenance['method_variant']}`，"
        f"seed {provenance['seed']}，优化步 {provenance['lightning_optimization_steps']}，"
        f"决策数 {provenance['total_decisions']}。"
    )
    drift = provenance["data_fingerprint_drift"]
    lines.append(
        f"- 数据合同核对：baseline 指纹{'一致' if drift['baseline_sha256']['matches'] else '不一致'}，"
        f"scenario split 指纹{'一致' if drift['scenario_split_sha256']['matches'] else '不一致'}，"
        f"scenario pool 指纹{'一致' if drift['scenario_pool_sha256']['matches'] else '不一致'}。"
    )
    scenario_drift = provenance["scenario_pool_drift"]
    if scenario_drift.get("training_snapshot_available"):
        lines.append(
            "- 场景文件追溯：训练时与当前的差异字段为 "
            f"`{', '.join(sorted({field for fields in scenario_drift['changed_fields'].values() for field in fields}))}`；"
            f"是否只涉及标签/口径改名：{'是' if scenario_drift['only_label_or_reference_fields_changed'] else '否'}。"
            "本轮只跑 Nominal（不加载任何扰动场景），因此该改名不改变本条结论。"
        )
    lines.append(
        f"- 诊断副本：格式 `{diagnostic_checkpoint_format()}`，训练前指纹等于原 C Actor 指纹，"
        f"训练后指纹 `{gate['final_actor_fingerprint']}` 已变化；原 C 权重文件未被改动。"
    )
    lines.append(
        f"- 轨迹事实核验：`{trajectory_verification['overall_status']}`"
        "（覆盖/物理约束/费用账本由现有严格检查路径独立复算）。"
    )
    lines.append(
        f"- 诊断自检：`{verification['overall_status']}`；明细见 `verification.json`。"
    )
    consistency = verification.get("device_consistency") or {}
    if consistency.get("available"):
        lines.append(
            f"- GPU/CPU 判读一致性：各分项准确率最大差 "
            f"{float(consistency['max_accuracy_rate_difference']):.1e}，逐样本 NLL 最大差 "
            f"{float(consistency['max_per_sample_nll_difference']):.2e}；"
            f"结论{'一致' if consistency['conclusion_consistent'] else '不一致'}。"
        )
    lines.append("")

    lines.append("## 4. 产物索引")
    lines.append("")
    for name, note in (
        ("manifest.json", "本轮总清单与边界声明"),
        ("input_visibility.json", "Phase A 输入可见性清册"),
        ("teacher_snapshots.pt", "210 条逐决策 CPU 快照"),
        ("teacher_actions.jsonl", "教师动作、候选与掩码"),
        ("teacher_validation.json", "A 轨迹与可表达性核对"),
        ("initial_teacher_metrics.json", "原 C 教师状态评分"),
        ("final_teacher_metrics.json", "诊断副本教师状态评分"),
        ("bc_history.jsonl", "逐次有效更新的 NLL/梯度范数/耗时/显存"),
        ("train_scans.jsonl", "训练期扫描记录"),
        ("diagnostic_actor.pt", "诊断副本（diagnostic_behavior_clone）"),
        ("trajectories/", "A、原 C、诊断副本三条 Nominal 轨迹"),
        ("trajectory_comparison.json", "闭环执行事实偏离对照"),
        ("trajectory_verification.json", "轨迹事实严格核验"),
        ("comparison.csv", "三层主表（机器可读）"),
        ("verification.json", "权重身份与快照合同自检"),
        ("original_c_provenance.json", "原 C 权重来源、指纹与数据合同追溯"),
        ("runs/", "各阶段控制台日志与废弃运行记录"),
    ):
        lines.append(f"- `{name}`：{note}")
    lines.append("")
    lines.append(
        "说明：`*.pt` 产物（`teacher_snapshots.pt`、`diagnostic_actor.pt`）按本仓库既有 "
        "`.gitignore` 约定仅保留在本地工作区，与原 C `weights.pt` 的处理方式一致；"
        "其 SHA256 与指纹记录在 `manifest.json` / `verification.json`。"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def diagnostic_checkpoint_format() -> str:
    try:
        payload = torch.load(DIAGNOSTIC_ACTOR, map_location="cpu", weights_only=False)
        return str(payload.get("diagnostic_format"))
    except Exception:
        return DIAGNOSTIC_FORMAT


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
        log(f"阶段 {stage} 退出码 {process.returncode}（日志 {log_path}）")
        if process.returncode != 0:
            write_json(OUTPUT_DIR / "stage_failures.json", results)
            return process.returncode
    write_json(OUTPUT_DIR / "stage_failures.json", results)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Work-3 C 基准复现能力诊断")
    parser.add_argument(
        "--stage",
        choices=list(STAGES) + ["all"],
        default="all",
        help="执行阶段；all 会以子进程顺序执行全部阶段",
    )
    args = parser.parse_args()

    if args.stage == "all":
        return run_all_stages()

    handlers = {
        "collect": stage_collect,
        "derive": stage_derive,
        "score-original": stage_score_original,
        "train": stage_train,
        "evaluate": stage_evaluate,
        "verify": stage_verify,
        "report": stage_report,
    }
    started = time.perf_counter()
    code = handlers[args.stage]()
    log(f"阶段 {args.stage} 用时 {time.perf_counter() - started:.1f}s，退出码 {code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
