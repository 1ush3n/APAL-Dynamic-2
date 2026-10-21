"""Work-3 单臂 R1 seed43、总预算1024步保持复核。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
import zipfile

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work3_4060"
OUTPUT_ROOT = BASE / "engineering_closeout_r1_1024_v1" / "ppo_seed43"
ARCHIVE_ROOT = ROOT.parent / "work3_local_archives"
CAL_ROOT = BASE / "critic_calibration_retention_v1" / "seed42"
CAL_DIR = CAL_ROOT / "calibration"
CAL_PPO = CAL_ROOT / "ppo"
OLD_R1 = BASE / "ppo_retention_repeat_v1" / "ppo_seed43" / "R1"
OLD_RUN = BASE / "ppo_retention_repeat_v1" / "ppo_seed43"
OLD_ARTIFACTS = OLD_RUN / "binary_artifact_inventory.json"
ABLATION = BASE / "ppo_ablation_arms_v1" / "seed42"
BC_SOURCE = BASE / "c_reproduction_diagnostic_v1" / "seed42"
BASELINE = BASE / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"
MODEL = CAL_DIR / "calibrated_model.pt"
SEED = 43
BUDGET = 1024
POINTS = (0, 128, 256, 512, 768, 1024)
EXPECTED_MODEL_SHA256 = "b19c9e8acc79de28eaadc8414413f2901d7d8d94ffbfb3b889d1e4a4f5ca68c6"
EXPECTED_FULL_FINGERPRINT = "cc1b350c3332d24d1e6d334cb5eda46c413e880718e6f2ab914612f772ba59c6"
EXPECTED_POLICY_FINGERPRINT = "0989e9bf8e2fed71ee630bc1d2d9da251d7d9cae5dec6c0e502ec54b8bcd1aee"
EXPECTED_BASELINE_SHA256 = "f461b887335b76bb67b85817f0f0f30b91a308f535c3379f9d1a2b73d8f20414"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: format(value, ".17g") if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def resolve_output_root(path: Path) -> Path:
    target = path.expanduser().resolve()
    expected = OUTPUT_ROOT.resolve()
    if target != expected or not target.is_relative_to(BASE.resolve()):
        raise ValueError(f"本轮输出根目录必须是 {expected}，拒绝写入 {target}")
    return target


def build_resolved_config(source: Mapping[str, Any]) -> dict[str, Any]:
    hyper = dict(source)
    hyper.update(
        {
            "seed": SEED,
            "learning_rate": float(source["learning_rate"]) * 0.1,
            "total_env_steps": BUDGET,
            "max_decisions": BUDGET,
        }
    )
    return {
        "name": "C_BC_PPO_R1_RETENTION_1024_DIAGNOSTIC",
        "arm": "R1",
        "ppo_sampling_seed": SEED,
        "initialization_source_seed": 42,
        "bc_source_seed": 42,
        "calibration_source_seed": 42,
        "source_hyperparameters": dict(source),
        "effective_hyperparameters": hyper,
        "points": list(POINTS),
        "evaluation_decision_limit": 4096,
        "training_decision_limit": BUDGET,
        "training_interactions_total": BUDGET,
        "training_contract": {
            "method_variant": "C", "allow_postpone": True, "allow_advance": True,
            "forced_align": False, "teacher_actions": False, "auxiliary_updates": 0,
            "optimizer_state_restored": False, "scheduler": None,
        },
        "change": "全部 PPO 参数组 learning_rate 乘 0.1",
    }


def _fixture_inputs_match(hashes: Mapping[str, str]) -> bool:
    return bool(hashes) and len(set(hashes.values())) == 1


def _pytest_summary_counts(summary: str) -> dict[str, int]:
    return {
        name: int(match.group(1)) if (match := re.search(rf"\b(\d+) {name}\b", summary)) else 0
        for name in ("passed", "failed", "skipped")
    }


def _classify_protocol_status(
    checks: Mapping[str, bool],
    *,
    shaping_status: str,
    objective_statuses: Mapping[str, str],
    physical_status: str,
) -> str:
    failed_checks = {
        "fixed_source_and_dependencies_unchanged",
        "regression_suite_passed",
        "fp32_equivalence_passed",
        "pretraining_archive_verified",
        "step0000_matches_bc_and_verifier",
        "first_rollout_paired",
        "first_minibatch_identity_matches",
        "r1_adamw_lr_contract",
        "run_json_effective_lr_and_budget_contract",
        "initial_parameters_complete_and_r1_has_no_r4_hook",
        "checkpoint_initial_fingerprints_match_source",
        "actual_training_budget_1024",
    }
    if physical_status == "FAIL" or shaping_status == "FAIL" or any(
        status == "FAIL" for status in objective_statuses.values()
    ) or any(not checks.get(key, False) for key in failed_checks):
        return "FAIL"
    complete = all(checks.values()) and physical_status == "PASS" and shaping_status == "PASS" \
        and all(status == "PASS" for status in objective_statuses.values())
    return "PASS" if complete else "INCOMPLETE"


def inventory_dependency_paths(
    paths: Mapping[str, Path],
    expected_hashes: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    expected_hashes = expected_hashes or {}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for name, path in paths.items():
        resolved = path.resolve()
        exists = resolved.is_file()
        actual_hash = sha256(resolved) if exists else None
        expected_hash = expected_hashes.get(name)
        matches = exists and (expected_hash is None or actual_hash == expected_hash)
        rows.append(
            {
                "name": name,
                "path": str(resolved),
                "exists": exists,
                "size_bytes": resolved.stat().st_size if exists else None,
                "sha256": actual_hash,
                "expected_sha256": expected_hash,
                "matches_expected": matches,
            }
        )
        if not exists or not matches:
            missing.append(name)
    return rows, missing


def _dependency_paths() -> tuple[dict[str, Path], dict[str, str]]:
    source_inventory = read_json(OLD_ARTIFACTS) if OLD_ARTIFACTS.is_file() else {"files": []}
    paths: dict[str, Path] = {
        "mini210_baseline": BASELINE,
        "regression_fixture_real_283": ROOT / "data" / "work3" / "real_283_k10_baseline.json",
        "raw_physical_process_data": ROOT / "data" / "283.csv",
        "worker_skill_calendar_data": ROOT / "data" / "worker_pool_fixed.csv",
        "calibrated_model": MODEL,
        "calibration_manifest": CAL_ROOT / "manifest.json",
        "calibration_weight_integrity": CAL_ROOT / "weight_integrity.json",
        "calibration_fit_metrics": CAL_DIR / "fit_metrics.json",
        "calibration_policy_invariance": CAL_DIR / "policy_invariance.json",
        "calibration_snapshot_targets": CAL_DIR / "snapshots_and_mc_targets.pt",
        "calibration_ppo_summary": CAL_PPO / "training_summary.json",
        "calibration_ppo_history": CAL_PPO / "training_history.jsonl",
        "calibration_ppo_first_rollout": CAL_PPO / "first_rollout.pt",
        "ablation_registry_seed42": ABLATION / "arm_registry.json",
        "ablation_comparison_seed42": ABLATION / "comparison_by_point.csv",
        "old_seed43_manifest": OLD_RUN / "manifest.json",
        "old_seed43_source_binding": OLD_RUN / "source_binding.json",
        "old_seed43_r1_rollout": OLD_R1 / "first_rollout.pt",
        "old_seed43_r1_first_update": OLD_R1 / "first_update.json",
        "old_seed43_r1_minibatches": OLD_R1 / "minibatch_indices.jsonl",
        "bc_diagnostic_actor": BC_SOURCE / "diagnostic_actor.pt",
        "bc_teacher_snapshots": BC_SOURCE / "teacher_snapshots.pt",
        "bc_nominal_trajectory": BC_SOURCE / "trajectories" / "C_BC_DIAGNOSTIC_NOMINAL.json",
        "prior_binary_inventory": OLD_ARTIFACTS,
    }
    expected = {
        "mini210_baseline": EXPECTED_BASELINE_SHA256,
        "calibrated_model": EXPECTED_MODEL_SHA256,
        "old_seed43_r1_rollout": "c8673791a3ed160bd4d806386397709a1f913504e8312bd455fa892718563e98",
        "old_seed43_r1_first_update": "84985c56948d7be334651465a1aab9aaee9abd2e507857b9c6117e5bf5deed1a",
        "old_seed43_r1_minibatches": "95648197ef589203ef7e0f0383529b5e4c2e065ee07c37c634ad459a4751de43",
        "bc_teacher_snapshots": "ac9855fa1c19eb906bd2bcbef6446acb80ade7fb8e0f4e94e9f930f8018bfa62",
    }
    for index, item in enumerate(source_inventory["files"]):
        paths[f"prior_binary_{index:02d}"] = OLD_RUN / Path(item["path"])
        expected[f"prior_binary_{index:02d}"] = item["sha256"]
    for index, path in enumerate(sorted(item for item in OLD_RUN.rglob("*") if item.is_file())):
        paths[f"old_seed43_artifact_{index:03d}"] = path
    return paths, expected


def _source_paths() -> dict[str, Path]:
    return {
        "work3_lightning": ROOT / "training" / "work3_lightning.py",
        "work3_vector_env": ROOT / "training" / "work3_vector_env.py",
        "work3_actor_critic": ROOT / "models" / "work3" / "actor_critic.py",
        "work3_graph_builder": ROOT / "models" / "work3" / "graph_builder.py",
        "work3_ppo_buffer": ROOT / "models" / "work3" / "ppo_buffer.py",
        "work3_ppo_trainer": ROOT / "models" / "work3" / "ppo_trainer.py",
        "work3_environment": ROOT / "envs" / "work3" / "environment.py",
        "work3_core_types": ROOT / "envs" / "work3" / "core_types.py",
        "ablation_runner": BASE / "run_ppo_ablation_arms.py",
        "calibration_loader": BASE / "run_critic_calibration_retention.py",
        "common_retention_runtime": BASE / "run_bc_ppo_retention.py",
        "seed43_pairing_runtime": BASE / "run_ppo_retention_repeat.py",
        "bc_reproduction_protocol": BASE / "c_reproduction_protocol.py",
        "trajectory_verifier": BASE / "verify_trajectory_facts.py",
        "shaping_verifier": BASE / "verify_shaping_trace.py",
        "objective_evaluator": ROOT / "utils" / "work3" / "objective_evaluator.py",
        "new_r1_runner": Path(__file__).resolve(),
        "training_runtime_tests": ROOT / "tests" / "work3" / "test_work3_training_runtime.py",
        "retention_contract_tests": ROOT / "tests" / "test_ppo_retention_repeat.py",
    }


def _git_info() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, text=True
    ).splitlines()
    return {"head": head, "branch": branch, "worktree_entries": status, "clean": not status}


def _configure_runner(out: Path) -> Any:
    import work3_4060.run_ppo_ablation_arms as runner

    runner.CAL_ROUND = CAL_ROOT
    runner.OUTPUT_DIR = out
    runner.RUNS_DIR = out / "runs"
    runner.CAL_MODEL = MODEL
    runner.CAL_FIT = CAL_DIR / "fit_metrics.json"
    runner.REFERENCE_PPO = CAL_PPO
    runner.REFERENCE_SUMMARY = CAL_PPO / "training_summary.json"
    runner.REFERENCE_HISTORY = OLD_R1 / "training_history.jsonl"
    runner.REFERENCE_PAIRING = OLD_R1 / "first_rollout_pairing.json"
    runner.REFERENCE_FIRST_ROLLOUT = OLD_R1 / "first_rollout.pt"
    runner.REFERENCE_MINIBATCH_UPDATE = OLD_R1 / "first_update.json"
    runner.BASELINE_PATH = BASELINE
    runner.ARMS = ("R1",)
    runner.POINTS = POINTS
    runner.BUDGET = BUDGET
    runner.RUN_SEED = SEED
    runner.ARM_SPECS = {"R1": {"change": "learning_rate_x0.1", "lr_scale": 0.1}}
    runner.MANIFEST = out / "manifest.json"
    runner.ARM_REGISTRY = out / "arm_registry.json"
    runner.COMPARISON_CSV = out / "comparison_by_point.csv"
    runner.CYCLE_CSV = out / "cycle_comparison.csv"
    runner.DIAGNOSTICS_CSV = out / "update_diagnostics.csv"
    runner.VERIFICATION = out / "verification.json"
    runner.SUMMARY_MD = out / "summary.md"
    resolved = out.resolve()
    for name, path in runner.arm_paths("R1").items():
        if not path.resolve().is_relative_to(resolved):
            raise ValueError(f"R1 输出路径逃逸出本轮输出根目录：{name}={path.resolve()}")
    if runner.arm_paths("R1")["run_json"].resolve() == (
        OLD_RUN / "REF" / "runs" / "ppo_run.json"
    ).resolve():
        raise ValueError("拒绝把 R1 run.json 写入历史 REF 路径")
    return runner


def _runtime_versions() -> dict[str, Any]:
    packages = ("numpy", "pandas", "torch", "torch-geometric", "lightning", "pytest")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "packages": versions,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count(),
    }


def _set_determinism() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _verify_fixed_source_config(source: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "beta_shaping": 1.0,
        "learning_rate": 3e-4,
        "clip_epsilon": 0.2,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.01,
        "ppo_epochs": 1,
        "batch_size": 2,
        "rollout_steps": 8,
        "num_envs": 1,
        "max_grad_norm": 0.5,
        "optimizer": "AdamW",
        "optimizer_eps": 1e-5,
        "weight_decay": 1e-4,
        "normalize_advantages": True,
        "amp_dtype": "fp32",
        "disturbance_enabled": False,
        "scenario": "NOMINAL",
        "warmup_mode": "none",
    }
    conflicts = {
        key: {"expected": value, "actual": source.get(key)}
        for key, value in expected.items()
        if source.get(key) != value
    }
    budget_fields = {
        "total_env_steps": source.get("total_env_steps"),
        "max_decisions": source.get("max_decisions"),
        "effective_total_env_steps": BUDGET,
        "effective_max_decisions": BUDGET,
    }
    if conflicts:
        raise ValueError(f"校准 PPO 来源超参与固定 R1 复核合同冲突：{conflicts}")
    return {"source_conflicts": conflicts, "budget_source_and_override": budget_fields}


def _verify_initialization() -> dict[str, Any]:
    import work3_4060.run_bc_ppo_retention as common
    import work3_4060.run_critic_calibration_retention as calibration

    model, payload = calibration.load_calibrated_actor(MODEL)
    if payload.get("actor_fingerprint") != EXPECTED_FULL_FINGERPRINT:
        raise ValueError("strict calibration loader 的完整模型指纹不符")
    groups = common.parameter_groups(model)
    state = model.state_dict()
    policy_fingerprint = calibration.subset_fingerprint(
        state, groups["groups"]["shared"] + groups["groups"]["policy"]
    )
    critic_fingerprint = calibration.subset_fingerprint(state, groups["groups"]["value"])
    if policy_fingerprint != EXPECTED_POLICY_FINGERPRINT:
        raise ValueError(f"校准策略子集指纹不符：{policy_fingerprint}")
    if len(model.critic._forward_pre_hooks) != 0:
        raise ValueError("共同校准起点存在 critic hook，R1 禁止带入 R4 机制")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_ids = [id(parameter) for parameter in trainable]
    if len(parameter_ids) != len(set(parameter_ids)) or len(parameter_ids) != groups["total"]:
        raise ValueError("R1 初始化参数未完整可训练或出现重复参数")
    return {
        "model_sha256": sha256(MODEL),
        "full_model_fingerprint": payload["actor_fingerprint"],
        "policy_subset_fingerprint": policy_fingerprint,
        "critic_subset_fingerprint": critic_fingerprint,
        "parameter_group_counts": groups["counts"],
        "all_parameters_trainable": len(trainable) == groups["total"],
        "trainable_parameter_count": len(trainable),
        "optimizer_state_loaded": False,
        "r4_hook_present": len(model.critic._forward_pre_hooks) != 0,
        "strict_loader": True,
    }


def _manifest(config: Mapping[str, Any], git: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "work3_r1_retention_seed43_1024_v1",
        "name": config["name"],
        "ppo_sampling_seed": SEED,
        "bc_source_seed": 42,
        "calibration_source_seed": 42,
        "arm": "R1",
        "initialization": "calibrated seed42 model; new model, environment, buffer, AdamW and counters",
        "source_model_sha256": EXPECTED_MODEL_SHA256,
        "source_full_fingerprint": EXPECTED_FULL_FINGERPRINT,
        "training_interactions_total": BUDGET,
        "training_interactions_includes_old_run": False,
        "evaluation_points": list(POINTS),
        "expected_evaluation_count": len(POINTS),
        "shared_step0000": "shared_step0000_NOMINAL.json",
        "evaluation_matrix": [
            {
                "arm": "R1",
                "point": point,
                "path": "shared_step0000_NOMINAL.json"
                if point == 0
                else f"R1/trajectories/step{point:04d}_NOMINAL.json",
                "shared": point == 0,
            }
            for point in POINTS
        ],
        "scenario": "NOMINAL",
        "disturbance": "none",
        "warmup": "none",
        "precision": "fp32",
        "device": "cuda:0",
        "vector_environment": {"count": 1, "start_method": "spawn"},
        "optimizer": "AdamW",
        "optimizer_schedulers": [],
        "training_decision_limit": BUDGET,
        "evaluation_decision_limit": 4096,
        "git_freeze": dict(git),
        "execution_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation_command": f"{Path(sys.executable).resolve()} {Path(__file__).resolve()}",
        "effect_thresholds": {
            "completed_tasks": 210,
            "completed_aircraft": 7,
            "j_total_max": 1e-3,
            "j_takt_max": 1e-6,
            "postponed_max": 0,
        },
    }


def prepare_run(out_arg: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    out = resolve_output_root(out_arg)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"拒绝覆盖已有实验产物：{out}")
    git = _git_info()
    if not git["clean"]:
        raise RuntimeError(f"训练源码冻结提交后工作树必须无变更：{git['worktree_entries']}")
    if Path(sys.prefix).name.casefold() != "rag_env":
        raise RuntimeError(f"必须使用 rag_env，当前环境为 {sys.prefix}")
    if not torch.cuda.is_available():
        raise RuntimeError("训练要求 cuda:0，当前 CUDA 不可用")
    out.mkdir(parents=True, exist_ok=True)
    for relative in ("engineering", "backup", "training/checkpoints", "training/trajectories", "runs"):
        (out / relative).mkdir(parents=True, exist_ok=True)

    dependency_paths, expected = _dependency_paths()
    rows, missing = inventory_dependency_paths(dependency_paths, expected)
    inventory = {"files": rows, "missing_or_mismatched": missing}
    write_json(out / "dependency_inventory.json", inventory)
    write_json(out / "environment_versions.json", _runtime_versions())
    if missing:
        write_json(
            out / "verification.json",
            {
                "protocol_status": "INCOMPLETE",
                "production_status": "INCOMPLETE",
                "retention_status": "INCOMPLETE",
                "stop_reason": "missing_or_mismatched_dependency",
                "missing_or_mismatched": missing,
            },
        )
        raise FileNotFoundError(f"训练依赖缺失或哈希不符：{missing}")

    baseline_sha = sha256(BASELINE)
    calibration_manifest = read_json(CAL_ROOT / "manifest.json")
    if calibration_manifest["instance"]["baseline_sha256"] != baseline_sha:
        raise ValueError("校准 manifest 的实例基准哈希与本地 mini210 文件不符")
    bc_manifest = read_json(BC_SOURCE / "manifest.json")
    if bc_manifest["instance"]["baseline_sha256"] != baseline_sha:
        raise ValueError("BC manifest 的实例基准哈希与本地 mini210 文件不符")
    if bc_manifest["diagnostic_checkpoint"]["sha256"] != sha256(BC_SOURCE / "diagnostic_actor.pt"):
        raise ValueError("BC 诊断 Actor 的文件哈希与来源清单不符")
    if bc_manifest["teacher"]["snapshots_sha256"] != sha256(BC_SOURCE / "teacher_snapshots.pt"):
        raise ValueError("BC 固定 teacher snapshots 文件哈希与来源清单不符")
    if sha256(MODEL) != EXPECTED_MODEL_SHA256:
        raise ValueError("校准权重 SHA256 不符合固定来源")
    source_registry = read_json(ABLATION / "arm_registry.json")
    if source_registry["shared_start"]["calibrated_model_sha256"] != EXPECTED_MODEL_SHA256:
        raise ValueError("seed42 ablation registry 的校准权重哈希不符")
    if source_registry["shared_start"]["calibrated_actor_fingerprint"] != EXPECTED_FULL_FINGERPRINT:
        raise ValueError("seed42 ablation registry 的全模型指纹不符")
    integrity = read_json(CAL_ROOT / "weight_integrity.json")
    if integrity["calibrated"]["policy_subset_fingerprint"] != EXPECTED_POLICY_FINGERPRINT:
        raise ValueError("校准 integrity 的策略子集指纹不符")
    if read_json(CAL_DIR / "policy_invariance.json")["full_model_fingerprint_after"] != EXPECTED_FULL_FINGERPRINT:
        raise ValueError("policy invariance 的全模型指纹不符")
    old_manifest = read_json(OLD_RUN / "manifest.json")
    if (old_manifest.get("source_model_sha256") != EXPECTED_MODEL_SHA256
            or int(old_manifest.get("ppo_sampling_seed", -1)) != SEED):
        raise ValueError("旧 seed43 R1 首段配对来源与本轮要求不符")
    calibration_summary = read_json(CAL_PPO / "training_summary.json")
    source = calibration_summary["hyperparameters"]
    source_validation = _verify_fixed_source_config(source)
    config = build_resolved_config(source)
    initialization = _verify_initialization()
    code_hashes = {
        name: {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}
        for name, path in _source_paths().items()
    }
    source_binding = {
        "git_freeze": git,
        "ppo_sampling_seed": SEED,
        "bc_and_calibration_source_seed": 42,
        "calibrated_model": {
            "path": str(MODEL.resolve()),
            "sha256": sha256(MODEL),
            "full_model_fingerprint": initialization["full_model_fingerprint"],
            "policy_subset_fingerprint": initialization["policy_subset_fingerprint"],
            "critic_subset_fingerprint": initialization["critic_subset_fingerprint"],
        },
        "old_seed43_r1_first_rollout": {
            "path": str((OLD_R1 / "first_rollout.pt").resolve()),
            "sha256": sha256(OLD_R1 / "first_rollout.pt"),
        },
        "old_seed43_r1_first_update": {
            "path": str((OLD_R1 / "first_update.json").resolve()),
            "sha256": sha256(OLD_R1 / "first_update.json"),
        },
        "source_hyperparameter_validation": source_validation,
        "source_code_hashes": code_hashes,
    }
    write_json(out / "training" / "resolved_config.yaml", config)
    write_json(out / "training" / "initialization_verification.json", initialization)
    write_json(out / "source_binding.json", source_binding)
    write_json(out / "manifest.json", _manifest(config, git))
    write_json(
        out / "arm_registry.json",
        {
            "schema_version": "work3_r1_retention_single_arm_v1",
            "source_calibration_manifest": str((CAL_ROOT / "manifest.json").resolve()),
            "shared_start": source_binding["calibrated_model"],
            "source_arm_registry": str((ABLATION / "arm_registry.json").resolve()),
            "source_arm_registry_sha256": sha256(ABLATION / "arm_registry.json"),
            "arm": config["effective_hyperparameters"],
            "points": list(POINTS),
            "ppo_sampling_seed": SEED,
            "initialization_seed": 42,
        },
    )
    write_json(
        out / "engineering" / "regression_test_results.json",
        {
            "status": "PENDING_PRETRAIN_REGRESSION",
            "baseline_commit": "0e2e0bbc19e93871a3b557ec132144440daa4bba",
            "current_pre_repair_commit": "2c434ff0592a28af8e9e3ad9d36cd7e6e4e1d3dd",
            "repaired_tests": [
                "test_single_env_worker_matches_direct_environment_for_frozen_actions",
                "test_two_spawn_workers_share_one_main_actor_and_enforce_aggregate_step_budget",
                "test_lightning_ppo_update_mixes_stay_and_postpone_samples",
            ],
        },
    )
    return config, source_binding, initialization


def _archive_items(out: Path) -> dict[str, Path]:
    paths, _expected = _dependency_paths()
    items: dict[str, Path] = {}
    for name, path in paths.items():
        if path.is_file():
            items[f"repo/{path.resolve().relative_to(ROOT.resolve()).as_posix()}"] = path.resolve()
    for path in _source_paths().values():
        if path.is_file():
            items[f"repo/{path.resolve().relative_to(ROOT.resolve()).as_posix()}"] = path.resolve()
    extra = (
        ROOT / "models" / "work3" / "graph_builder.py",
        ROOT / "models" / "work3" / "time_head.py",
        ROOT / "training" / "work3_r3_e04_audit.py",
        ROOT / "envs" / "work3" / "environment.py",
        ROOT / "envs" / "work3" / "core_types.py",
        ROOT / "utils" / "work3" / "objective_evaluator.py",
        ROOT / "pytest.ini",
        ROOT / "pyproject.toml",
    )
    for path in extra:
        if path.is_file():
            items[f"repo/{path.relative_to(ROOT).as_posix()}"] = path.resolve()
    if OLD_R1.is_dir():
        for path in OLD_R1.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".pt", ".json", ".jsonl", ".csv", ".yaml"}:
                items[f"repo/{path.relative_to(ROOT).as_posix()}"] = path.resolve()
    for path in out.rglob("*"):
        if path.is_file() and path.suffix.lower() != ".zip":
            items[f"repo/{path.relative_to(ROOT).as_posix()}"] = path.resolve()
    return dict(sorted(items.items()))


def create_verified_archive(out: Path, *, label: str) -> dict[str, Any]:
    if not ARCHIVE_ROOT.exists():
        ARCHIVE_ROOT.mkdir(parents=True)
    if not ARCHIVE_ROOT.is_dir():
        raise NotADirectoryError(ARCHIVE_ROOT)
    commit = _git_info()["head"][:12]
    archive = ARCHIVE_ROOT / f"work3_r1_1024_{label}_{commit}.zip"
    if archive.exists():
        raise FileExistsError(f"拒绝覆盖既有归档：{archive}")
    items = _archive_items(out)
    missing = [str(path) for path in items.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"归档源文件消失：{missing}")
    catalog = [
        {"member": member, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for member, path in items.items()
    ]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as bundle:
        bundle.writestr("ARCHIVE-CATALOG.json", json.dumps(catalog, ensure_ascii=False, indent=2) + "\n")
        for member, path in items.items():
            bundle.write(path, member)
    checks: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "r", allowZip64=True) as bundle:
        for item in catalog:
            content = bundle.read(item["member"])
            digest = hashlib.sha256(content).hexdigest()
            checks.append({"member": item["member"], "expected_sha256": item["sha256"],
                           "actual_sha256": digest, "size_bytes": len(content),
                           "passed": digest == item["sha256"] and len(content) == item["size_bytes"]})
    result = {
        "archive_path": str(archive.resolve()), "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive), "member_count": len(catalog),
        "pt_member_count": sum(Path(item["member"]).suffix.lower() == ".pt" for item in catalog),
        "all_members_verified_after_readback": all(item["passed"] for item in checks),
        "catalog": catalog, "member_checks": checks,
    }
    prior_inventory = read_json(OLD_ARTIFACTS)
    prior_paths = {f"repo/{(OLD_RUN / Path(row['path'])).resolve().relative_to(ROOT.resolve()).as_posix()}"
                   for row in prior_inventory["files"]}
    checked_members = {item["member"] for item in catalog}
    result["prior_registered_pt_count"] = len(prior_inventory["files"])
    result["prior_registered_pt_verified_count"] = len(prior_paths & checked_members)
    result["prior_registered_pt_all_verified"] = prior_paths <= checked_members and len(prior_paths) == len(prior_inventory["files"])
    write_json(out / "backup" / f"archive_manifest_{label}.json", result)
    write_json(out / "backup" / f"archive_verification_{label}.json", {
        "archive_path": result["archive_path"], "archive_sha256": result["archive_sha256"],
        "member_count": len(checks), "all_members_verified_after_readback": result["all_members_verified_after_readback"],
        "failures": [item for item in checks if not item["passed"]],
    })
    if not result["all_members_verified_after_readback"]:
        raise IOError("归档回读哈希校验失败")
    return result


def _run_child(out: Path, stage: str, *, point: int | None = None) -> tuple[int, Path, float]:
    log_path = out / "runs" / f"{stage}_{point if point is not None else 'main'}.log"
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(SEED)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", stage, "--output", str(out)]
    if point is not None:
        command += ["--point", str(point)]
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND=" + json.dumps(command, ensure_ascii=False) + "\n")
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return result.returncode, log_path, time.perf_counter() - started


def _write_engineering_evidence(out: Path) -> dict[str, Any]:
    temp_root = Path(os.environ.get("TEMP", Path.home() / "AppData" / "Local" / "Temp"))
    log_specs = {
        "baseline_0e2e": temp_root / "work3_baseline_0e2e_test.log",
        "current_pre_repair_2c434ff": temp_root / "work3_current_2c434ff_test.log",
        "fixed_current_targeted_regressions": temp_root / "work3_final_targeted_regressions_v3.log",
    }
    required_logs = list(log_specs.values())
    missing = [str(path) for path in required_logs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"基线/current/修复回归日志缺失：{missing}")
    logs_dir = out / "engineering" / "source_versions"
    logs_dir.mkdir(parents=True, exist_ok=True)
    baseline_root = ROOT.parent / f"{ROOT.name}_work3_baseline_0e2e"
    pre_repair_root = ROOT.parent / f"{ROOT.name}_work3_current_2c434ff"
    module_paths = {
        "baseline_0e2e": [baseline_root / "work3_4060" / "run_ppo_ablation_arms.py",
                          baseline_root / "work3_4060" / "run_bc_ppo_retention.py"],
        "current_pre_repair_2c434ff": [pre_repair_root / "work3_4060" / "run_ppo_ablation_arms.py",
                                        pre_repair_root / "work3_4060" / "run_bc_ppo_retention.py"],
        "fixed_current_targeted_regressions": [BASE / "run_ppo_ablation_arms.py", BASE / "run_bc_ppo_retention.py"],
    }
    missing_modules = [str(path) for paths_for_run in module_paths.values() for path in paths_for_run if not path.is_file()]
    if missing_modules:
        raise FileNotFoundError(f"测试导入源码路径缺失：{missing_modules}")
    rows: dict[str, Any] = {}
    for name, path in log_specs.items():
        content = path.read_text(encoding="utf-8", errors="replace")
        copied = logs_dir / f"{name}.log"
        shutil.copyfile(path, copied)
        summary = next((line.strip() for line in reversed(content.splitlines())
                        if re.search(r"\b(?:passed|failed|skipped)\b", line)), None)
        rows[name] = {"original_log": str(path.resolve()), "copied_log": str(copied.relative_to(out)),
                      "log_sha256": sha256(copied), "pytest_summary": summary,
                      "module_paths": [str(module.resolve()) for module in module_paths[name]],
                      "module_source_hashes": [{"path": str(module.resolve()), "sha256": sha256(module)}
                                               for module in module_paths[name]],
                      "log_reported_module_paths": [line.split("=", 1)[1].strip() for line in content.splitlines()
                                                     if line.startswith(("RUNNER_FILE=", "COMMON_FILE="))]}
    fixed_summary = rows["fixed_current_targeted_regressions"]["pytest_summary"] or ""
    fixed_counts = _pytest_summary_counts(fixed_summary)
    fixed_passed = fixed_counts["passed"] > 0 and fixed_counts["failed"] == 0 and fixed_counts["skipped"] == 0
    prior_failed_as_expected = all(
        _pytest_summary_counts(rows[name]["pytest_summary"] or "")
        == {"passed": 0, "failed": len((
            "test_single_env_worker_matches_direct_environment_for_frozen_actions",
            "test_two_spawn_workers_share_one_main_actor_and_enforce_aggregate_step_budget",
            "test_lightning_ppo_update_mixes_stay_and_postpone_samples",
        )), "skipped": 0}
        for name in ("baseline_0e2e", "current_pre_repair_2c434ff")
    )
    fixed_inputs = {}
    for root_name, root in (("baseline_0e2e", baseline_root), ("pre_repair_2c434ff", pre_repair_root),
                            ("fixed_current", ROOT)):
        baseline_data = root / "data" / "work3" / "real_283_k10_baseline.json"
        fixed_inputs[str(baseline_data.resolve())] = sha256(baseline_data)
    same_fixture_inputs = _fixture_inputs_match(fixed_inputs)
    result = {
        "classification": "基线/修复前均失败，修复覆盖通过" if same_fixture_inputs and prior_failed_as_expected and fixed_passed
        else "输入不一致，基线对照不成立" if not same_fixture_inputs
        else "修复后目标回归未通过" if fixed_counts["passed"] == 0 or fixed_counts["failed"] or fixed_counts["skipped"]
        else "基线与修复前未复现三项失败",
        "baseline_commit": "0e2e0bbc19e93871a3b557ec132144440daa4bba",
        "pre_repair_commit": "2c434ff0592a28af8e9e3ad9d36cd7e6e4e1d3dd",
        "same_python": str(Path(sys.executable).resolve()),
        "same_dependency_environment": str(Path(sys.prefix).resolve()),
        "test_selection": [
            "test_single_env_worker_matches_direct_environment_for_frozen_actions",
            "test_two_spawn_workers_share_one_main_actor_and_enforce_aggregate_step_budget",
            "test_lightning_ppo_update_mixes_stay_and_postpone_samples",
        ],
        "baseline_and_pre_repair_command": "历史日志未保存完整shell调用；从固定筛选记录可确认参数为 tests/work3/test_work3_training_runtime.py -q -k <三项修复目标>。",
        "fixed_targeted_command": "D:\\Conda\\envs\\rag_env\\python.exe -m pytest tests/test_ppo_retention_repeat.py tests/work3/test_work3_training_runtime.py -q -k \"test_single_env_worker_matches_direct_environment_for_frozen_actions or test_two_spawn_workers_share_one_main_actor_and_enforce_aggregate_step_budget or lightning_ppo_update_mixes_stay_and_postpone_samples or amp_precision_resolver_rejects_mixed_precision_on_cpu or actor_sampling_and_replay_keep_probability_math_fp32_under_amp or training_entry_uses_configured_amp_for_rollout_and_lightning or runtime_seed_and_worker_episode_seed_are_reproducible or lightning_trainer_fit_runs_ppo_update_and_separates_step_counts or lightning_amp_uses_only_precision_plugin_scaler or lightning_configures_one_deduplicated_optimizer_for_all_trainable_models or lightning_module_construction_preserves_actor_cuda_device or lightning_ppo_loss_computation_is_pure_and_datamodule_uses_no_loader_workers or lightning_fit_calls_preserve_the_single_optimizer_between_rollouts\"",
        "targeted_regression_selectors": [
            "test_single_env_worker_matches_direct_environment_for_frozen_actions",
            "test_two_spawn_workers_share_one_main_actor_and_enforce_aggregate_step_budget",
            "test_lightning_ppo_update_mixes_stay_and_postpone_samples",
            "amp_precision_resolver_rejects_mixed_precision_on_cpu",
            "actor_sampling_and_replay_keep_probability_math_fp32_under_amp",
            "training_entry_uses_configured_amp_for_rollout_and_lightning",
            "runtime_seed_and_worker_episode_seed_are_reproducible",
            "lightning_trainer_fit_runs_ppo_update_and_separates_step_counts",
            "lightning_amp_uses_only_precision_plugin_scaler",
            "lightning_configures_one_deduplicated_optimizer_for_all_trainable_models",
            "lightning_module_construction_preserves_actor_cuda_device",
            "lightning_ppo_loss_computation_is_pure_and_datamodule_uses_no_loader_workers",
            "lightning_fit_calls_preserve_the_single_optimizer_between_rollouts",
        ],
        "fixture_input_hashes": fixed_inputs,
        "same_fixture_input_hash_across_runs": same_fixture_inputs,
        "runs": rows,
        "current_repaired_regression_status": "PASS" if fixed_passed else "FAIL",
        "baseline_and_pre_repair_failure_confirmed": prior_failed_as_expected,
        "passed": fixed_passed and prior_failed_as_expected and same_fixture_inputs,
    }
    write_json(out / "engineering" / "baseline_vs_current_test_results.json", result)
    repair_text = """# 夹具修复记录

基线 `0e2e0bb` 与修复前当前版本 `2c434ff` 用同一 rag_env、同一测试筛选及同一 `data/work3/real_283_k10_baseline.json` 哈希复测，三项都失败。因此这是既有测试夹具与正式环境合同不一致，不是由本轮代码回归引入。

1. worker 对照原来要求启发式策略自然在最初32步产生 `ADVANCE_TO_NEXT_EVENT`；规则策略可通过真实预约及自动事件推进合法完成，从而不保证出现显式 ADVANCE。夹具现在先经正式接口预约一个合法的未来事件，随后显式推进，并把两项真实动作都在直接环境与worker中重放；仍核对状态、事件、奖励、任务执行事实和终止。双worker总预算17、共享Actor、CPU worker与派发上限仍保留。
2. Lightning 混合样本夹具先预约任务，再要求已预约任务仍可留站，并把它强行加回候选集。这违反预约不可撤回。现在分别从未预约合法快照实际采样留站/后移并各执行一次，同时核验留站任务预约后不再候选且分支掩码为 `(False, False)`。

没有修改环境候选掩码、物理规则或 heuristic 策略。基线、修复前与修复后完整日志及固定输入哈希见同目录 JSON 与 `source_versions/`。
"""
    (out / "engineering" / "test_fixture_repairs.md").write_text(repair_text, encoding="utf-8")
    snapshots = {
        "work3_lightning.py": ROOT / "training" / "work3_lightning.py",
        "run_ppo_ablation_arms.py": BASE / "run_ppo_ablation_arms.py",
        "run_r1_retention_1024.py": Path(__file__).resolve(),
        "test_work3_training_runtime.py": ROOT / "tests" / "work3" / "test_work3_training_runtime.py",
        "test_ppo_retention_repeat.py": ROOT / "tests" / "test_ppo_retention_repeat.py",
    }
    for name, source in snapshots.items():
        shutil.copyfile(source, logs_dir / name)
    (logs_dir / "autocast_compatibility_note.md").write_text(
        "# autocast 历史源码说明\n\n"
        "记录中的运行时 SHA256=86c0afddd0717f92b253f746dc85eb32a000f6d05bde5a85f4d0e9b9486bd7a3，最终源 SHA256=6221e53759c6e048b2296858c8571d68bb6f686a5d122b660bb3e1f5adb2821f。当前仓库保留尊重调用方 autocast 策略的最终实现。运行时临时文件未从本地历史恢复，因此本轮对照只重建“外层显式 autocast(enabled=False)”这一已记录差异，不称为旧源码逐字节复现。\n",
        encoding="utf-8",
    )
    return result


def _worker(stage: str, out: Path, point: int | None) -> int:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    _set_determinism()
    runner = _configure_runner(out)
    if stage == "train":
        code = runner.stage_arm("R1")
        summary_path = out / "R1" / "training_summary.json"
        if not summary_path.is_file():
            return code or 5
        summary = read_json(summary_path)
        return code if int(summary.get("training_decisions", -1)) == BUDGET and summary.get("stop_reason") == "budget_exhausted" else (code or 5)
    if stage == "evaluate":
        if point not in POINTS:
            raise ValueError(f"评估点不在固定清单：{point}")
        checkpoint = MODEL if point == 0 else out / "R1" / "checkpoints" / f"step{point:04d}.pt"
        return runner.stage_evaluate_one("R1", checkpoint, f"step{point:04d}")
    if stage == "probe":
        if point not in POINTS:
            raise ValueError(f"probe点不在固定清单：{point}")
        import work3_4060.run_ppo_retention_repeat as repeat

        repeat.OUTPUT_ROOT = out
        repeat.CAL_ROOT = CAL_ROOT
        repeat.CAL_DIR = CAL_DIR
        repeat.OLD_R1 = OLD_R1
        repeat.OLD_RUN = OLD_RUN
        repeat.ABLATION = ABLATION
        repeat.BC_SOURCE = BC_SOURCE
        repeat.BASELINE = BASELINE
        repeat.MODEL = MODEL
        return repeat._probe_checkpoint(out, "R1", point)
    raise ValueError(f"未知worker阶段：{stage}")


def run_fp32_equivalence(out: Path) -> dict[str, Any]:
    """用固定历史首个 mini-batch 比较显式 autocast-disabled 与调用方 FP32 路径。"""
    import work3_4060.run_bc_ppo_retention as common
    import work3_4060.run_critic_calibration_retention as calibration

    _set_determinism()
    rollout = torch.load(OLD_R1 / "first_rollout.pt", map_location="cpu", weights_only=False)
    prior_update = read_json(OLD_R1 / "first_update.json")
    indices = [int(index) for index in prior_update["first_minibatch_samples"]]
    if len(indices) != 2:
        raise ValueError(f"历史首个 mini-batch 记录不是2条样本：{indices}")
    device = torch.device("cuda:0")
    caller_torch_cpu = torch.get_rng_state().clone()
    caller_torch_cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
    caller_python_rng, caller_numpy_rng = random.getstate(), np.random.get_state()
    saved_rng = rollout["pre_update_rng"]
    saved_cpu_rng = saved_rng["torch_cpu"].clone()
    saved_cuda_rng = [state.clone() for state in saved_rng["torch_cuda"]]
    if len(saved_cuda_rng) != torch.cuda.device_count():
        raise ValueError("歷史 pre_update_rng CUDA 裝置數與當前設備不一致")
    states: dict[str, dict[str, Any]] = {}
    for mode in ("legacy_disabled_context_reconstructed", "current_fp32_caller_context"):
        model, payload = calibration.load_calibrated_actor(MODEL)
        if payload["actor_fingerprint"] != EXPECTED_FULL_FINGERPRINT:
            raise ValueError("FP32 对照初始化模型指纹不符")
        model.to(device)
        model.train()
        module = common.DiagnosticsLightningModule(
            actor_critic=model, learning_rate=3e-5, clip_eps=0.2, vf_coef=0.5,
            ent_coef=0.01, max_grad_norm=0.5, ppo_epochs=1, batch_size=2,
        )
        group_info = common.parameter_groups(model)
        module.parameter_group_names = group_info["groups"]
        optimizer = module.configure_optimizers()
        if any(optimizer.state.values()):
            raise RuntimeError("FP32诊断应使用全新 AdamW 状态")
        torch.set_rng_state(saved_cpu_rng)
        torch.cuda.set_rng_state_all(saved_cuda_rng)
        random.setstate(saved_rng["python"])
        np.random.set_state(saved_rng["numpy"])
        batch_index = torch.tensor(indices, dtype=torch.long)
        batch = {
            "state_feats": rollout["state_feats"][batch_index],
            "time_urgencies": rollout["time_urgencies"][batch_index],
            "old_log_probs": rollout["old_log_probs"][batch_index],
            "advantages": rollout["advantages"][batch_index],
            "target_values": rollout["return_targets"][batch_index],
            "sample_records": [rollout["sample_records"][index] for index in indices],
        }
        context = torch.amp.autocast(device_type="cuda", enabled=False) if mode.startswith("legacy") else nullcontext()
        optimizer.zero_grad(set_to_none=True)
        with context:
            losses = module.objective.compute_ppo_minibatch_loss(batch)
        metrics = {
            name: value.detach().float().cpu().clone()
            for name, value in losses.items()
            if isinstance(value, torch.Tensor)
        }
        losses["total_loss"].backward()
        pre_clip_gradients = {
            name: parameter.grad.detach().float().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.grad is not None
        }
        preclip_group_norms = common.group_grad_norms(model, group_info["groups"])
        total_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.grad is not None],
            max_norm=0.5, error_if_nonfinite=True,
        )
        clipped_group_norms = common.group_grad_norms(model, group_info["groups"])
        clipped_gradients = {
            name: parameter.grad.detach().float().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.grad is not None
        }
        optimizer.step()
        states[mode] = {
            "metrics": metrics, "pre_clip_gradients": pre_clip_gradients,
            "clipped_gradients": clipped_gradients,
            "preclip_group_norms": preclip_group_norms,
            "clipped_group_norms": clipped_group_norms,
            "total_preclip_norm": float(total_norm.detach().cpu()),
            "parameters_after_step": {name: value.detach().float().cpu().clone() for name, value in model.named_parameters()},
        }
    torch.set_rng_state(caller_torch_cpu)
    torch.cuda.set_rng_state_all(caller_torch_cuda)
    random.setstate(caller_python_rng)
    np.random.set_state(caller_numpy_rng)

    def compare_maps(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        rows = []
        for name in left:
            a, b = left[name].double(), right[name].double()
            diff = (a - b).abs()
            scale = torch.maximum(a.abs(), b.abs())
            active = scale > 1e-12
            max_relative = float((diff[active] / scale[active]).max()) if bool(active.any()) else None
            rows.append({"name": name, "max_abs": float(diff.max()) if diff.numel() else 0.0,
                         "max_relative_non_near_zero": max_relative,
                         "near_zero_elements": int((~active).sum()),
                         "allclose": bool(torch.allclose(a, b, atol=1e-6, rtol=1e-4))})
        return {"count": len(rows), "max_abs": max((r["max_abs"] for r in rows), default=0.0),
                "max_relative_non_near_zero": max((r["max_relative_non_near_zero"] or 0.0 for r in rows), default=0.0),
                "near_zero_elements": sum(r["near_zero_elements"] for r in rows),
                "allclose": all(r["allclose"] for r in rows)}

    reference = states["legacy_disabled_context_reconstructed"]
    current = states["current_fp32_caller_context"]
    result = {
        "status": "PASS" if all((
            compare_maps(reference["metrics"], current["metrics"])["allclose"],
            compare_maps(reference["pre_clip_gradients"], current["pre_clip_gradients"])["allclose"],
            compare_maps(reference["clipped_gradients"], current["clipped_gradients"])["allclose"],
            compare_maps(reference["parameters_after_step"], current["parameters_after_step"])["allclose"],
            abs(reference["total_preclip_norm"] - current["total_preclip_norm"]) <= 1e-6,
        )) else "FAIL",
        "historical_source_reconstruction": "RECONSTRUCTED_NOT_EXACT",
        "historical_exact_reproduction_status": "INCOMPLETE",
        "historical_runtime_sha256": "86c0afddd0717f92b253f746dc85eb32a000f6d05bde5a85f4d0e9b9486bd7a3",
        "recorded_final_source_sha256": "6221e53759c6e048b2296858c8571d68bb6f686a5d122b660bb3e1f5adb2821f",
        "reconstruction": "只按已记录差异在损失/前向调用外围显式关闭 CUDA autocast；未恢复旧运行时原文件字节。",
        "same_device": torch.cuda.get_device_name(0), "device": str(device),
        "sample_indices": indices, "optimizer": "fresh AdamW lr=3e-5 eps=1e-5 wd=1e-4",
        "rng_source": str((OLD_R1 / "first_rollout.pt").resolve()),
        "rng_matches_saved_pre_update_state": True,
        "loss_metrics": compare_maps(reference["metrics"], current["metrics"]),
        "pre_clip_gradients": compare_maps(reference["pre_clip_gradients"], current["pre_clip_gradients"]),
        "post_clip_gradients": compare_maps(reference["clipped_gradients"], current["clipped_gradients"]),
        "optimizer_step_parameters": compare_maps(reference["parameters_after_step"], current["parameters_after_step"]),
        "clip_norm_abs_difference": abs(reference["total_preclip_norm"] - current["total_preclip_norm"]),
        "preclip_group_norms": {key: {"reconstructed": reference["preclip_group_norms"][key],
                                       "current": current["preclip_group_norms"][key]}
                                for key in reference["preclip_group_norms"]},
        "clipped_group_norms": {key: {"reconstructed": reference["clipped_group_norms"][key],
                                      "current": current["clipped_group_norms"][key]}
                                 for key in reference["clipped_group_norms"]},
        "tolerances": {"atol": 1e-6, "rtol": 1e-4, "near_zero_scale_threshold": 1e-12},
        "historical_record_loss_difference": {
            key: float(reference["metrics"][key].item()) - float(prior_update[key])
            for key in ("total_loss", "policy_loss", "value_loss", "entropy")
            if key in prior_update and key in reference["metrics"]
        },
        "counted_as_environment_interactions": 0, "counted_as_ppo_updates": 0,
    }
    write_json(out / "engineering" / "fp32_equivalence.json", result)
    return result


def _verify_step0(out: Path) -> dict[str, Any]:
    import work3_4060.run_bc_ppo_retention as common

    path = out / "shared_step0000_NOMINAL.json"
    data = read_json(path)
    baseline = read_json(BASELINE)
    fact_result = common.load_verifier().verify_single_trajectory(
        traj_path=path, baseline_h0=float(baseline["metadata"]["h0"]),
        baseline_tasks=baseline["tasks"], scenario_spec={"scenario_id": "NOMINAL"},
    )
    bc = read_json(BC_SOURCE / "trajectories" / "C_BC_DIAGNOSTIC_NOMINAL.json")
    current_facts = data["execution_facts"]["tasks_execution"]
    bc_facts = bc["execution_facts"]["tasks_execution"]
    task_diffs = []
    for key in sorted(set(current_facts) | set(bc_facts)):
        left, right = current_facts.get(key), bc_facts.get(key)
        if left is None or right is None:
            task_diffs.append({"task_key": key, "reason": "task_missing"})
            continue
        for field in ("actual_start", "actual_end"):
            delta = float(left[field]) - float(right[field])
            if abs(delta) > 1e-7:
                task_diffs.append({"task_key": key, "field": field, "delta_hours": delta})
        if sorted(left.get("team", [])) != sorted(right.get("team", [])):
            task_diffs.append({"task_key": key, "field": "team_set", "current": left.get("team"), "bc": right.get("team")})
    result = {
        "trajectory_path": str(path.resolve()), "independent_fact_status": fact_result.get("status"),
        "completed_tasks": data.get("completed_tasks"), "success": data.get("success"),
        "feasible": data.get("feasible"), "j_total": data.get("objective_j_total"),
        "bc_j_total": bc.get("objective_j_total"),
        "matches_bc_physical_schedule": not task_diffs and set(current_facts) == set(bc_facts),
        "task_differences": task_diffs, "physical_time_tolerance_hours": 1e-7,
        "passed": fact_result.get("status") == "PASS" and bool(data.get("success"))
        and bool(data.get("feasible")) and int(data.get("completed_tasks", 0)) == 210
        and abs(float(data.get("objective_j_total", math.inf))) <= 1e-9
        and not task_diffs and set(current_facts) == set(bc_facts),
    }
    write_json(out / "training" / "step0000_contract.json", result)
    return result


def _verify_completed_evaluation(out: Path, point: int) -> dict[str, Any]:
    import work3_4060.run_bc_ppo_retention as common

    path = out / "shared_step0000_NOMINAL.json" if point == 0 else out / "R1" / "trajectories" / f"step{point:04d}_NOMINAL.json"
    data = read_json(path)
    baseline = read_json(BASELINE)
    facts = common.load_verifier().verify_single_trajectory(
        traj_path=path, baseline_h0=float(baseline["metadata"]["h0"]),
        baseline_tasks=baseline["tasks"], scenario_spec={"scenario_id": "NOMINAL"},
    )
    result = {
        "point": point, "path": str(path.resolve()), "success": bool(data.get("success")),
        "feasible": bool(data.get("feasible")), "termination_reason": data.get("termination_reason"),
        "decisions": int(data.get("decisions", 0)), "decision_limit": 4096,
        "completed_tasks": int(data.get("completed_tasks", 0)),
        "physical_verifier_status": facts.get("status"),
        "passed": bool(data.get("success")) and bool(data.get("feasible"))
        and facts.get("status") == "PASS" and int(data.get("completed_tasks", 0)) == 210
        and int(data.get("decisions", 0)) < 4096,
    }
    write_json(out / "R1" / "trajectories" / f"step{point:04d}_verification.json", result)
    if not result["passed"]:
        raise RuntimeError(f"step{point} 独立评估未合法完工或事实核验未通过：{result}")
    return result


def _trajectory_row(path: Path, point: int) -> dict[str, Any]:
    data = read_json(path)
    facts = data.get("execution_facts", {})
    cycles = facts.get("cycle_execution_records", {})
    aircraft = facts.get("aircraft_execution", {})
    costs = data.get("cost_components_weighted", {})
    trace = data.get("action_trace", [])
    completed_aircraft = sum(
        len(item.get("exit_times", {})) == 5 for item in aircraft.values()
    )
    baseline = read_json(BASELINE)
    return {
        "arm": "R1", "point": point, "path": str(path.resolve()),
        "decisions": int(data.get("decisions", 0)), "completed_tasks": int(data.get("completed_tasks", 0)),
        "completed_aircraft": int(completed_aircraft), "success": bool(data.get("success")),
        "feasible": bool(data.get("feasible")), "j_total": float(data.get("objective_j_total", math.inf)),
        "j_takt": float(costs.get("j_takt", math.inf)), "d_time": float(costs.get("d_time", math.inf)),
        "d_team": float(costs.get("d_team", math.inf)), "j_postpone": float(costs.get("j_postpone", math.inf)),
        "j_revision": float(costs.get("j_revision", math.inf)), "makespan": data.get("makespan"),
        "stay_actions": sum(item.get("branch") == 0 for item in trace),
        "postpone_actions": sum(item.get("branch") == 1 for item in trace),
        "advance_actions": sum(item.get("branch") == 2 for item in trace),
        "postponed_count": int(data.get("postponed_count", 0)),
        "cycle_count": len(cycles),
        "max_cycle_delay_hours": max((max(0.0, float(row["transfer_time"]) - float(row["nominal_transfer_time"]))
                                       for row in cycles.values()), default=0.0),
        "cumulative_cycle_delay_hours": sum(max(0.0, float(row["transfer_time"]) - float(row["nominal_transfer_time"]))
                                            for row in cycles.values()),
        "retention_pass": bool(data.get("success")) and bool(data.get("feasible"))
        and int(data.get("completed_tasks", 0)) == 210 and completed_aircraft == 7
        and float(data.get("objective_j_total", math.inf)) <= 1e-3
        and float(costs.get("j_takt", math.inf)) <= 1e-6 and int(data.get("postponed_count", 0)) == 0,
        "evaluation_episode_count": 1, "evaluation_decisions_are_separate_from_training": True,
        "aircraft_count_expected": int(baseline["metadata"]["num_aircraft"]),
    }


def _write_evaluation_reports(out: Path) -> dict[str, Any]:
    import work3_4060.run_bc_ppo_retention as common
    import work3_4060.run_critic_calibration_retention as calibration

    baseline = read_json(BASELINE)
    rows: list[dict[str, Any]] = []
    verifier_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    trajectories: dict[int, dict[str, Any]] = {}
    for point in POINTS:
        path = out / "shared_step0000_NOMINAL.json" if point == 0 else out / "R1" / "trajectories" / f"step{point:04d}_NOMINAL.json"
        if not path.is_file():
            continue
        data = read_json(path)
        trajectories[point] = data
        verified = common.load_verifier().verify_single_trajectory(
            traj_path=path, baseline_h0=float(baseline["metadata"]["h0"]),
            baseline_tasks=baseline["tasks"], scenario_spec={"scenario_id": "NOMINAL"},
        )
        verifier_rows.append({"arm": "R1", "point": point, "path": str(path.resolve()), **verified})
        trajectory_row = _trajectory_row(path, point)
        trajectory_row["retention_pass"] = bool(trajectory_row["retention_pass"]) and verified.get("status") == "PASS"
        rows.append(trajectory_row)
        facts = data.get("execution_facts", {})
        for cycle, record in sorted(facts.get("cycle_execution_records", {}).items(), key=lambda item: int(item[0])):
            cycle_rows.append({"arm": "R1", "point": point, "cycle_id": int(cycle),
                               "nominal_transfer_time": float(record["nominal_transfer_time"]),
                               "actual_transfer_time": float(record["transfer_time"]),
                               "delay_hours": float(record["transfer_time"]) - float(record["nominal_transfer_time"])})
        for task_key, task in sorted(facts.get("tasks_execution", {}).items()):
            task_rows.append({"point": point, "task_key": task_key, "actual_start": task.get("actual_start"),
                              "actual_end": task.get("actual_end"), "team": json.dumps(task.get("team", [])),
                              "revision_history": json.dumps(task.get("revision_history", []), ensure_ascii=False)})
    write_csv(out / "evaluation_results.csv", rows)
    initial_cycles = {int(cycle): record for cycle, record in trajectories.get(0, {}).get("execution_facts", {}).get("cycle_execution_records", {}).items()}
    for row in cycle_rows:
        start = initial_cycles.get(int(row["cycle_id"]))
        row["delay_vs_step0000_hours"] = None if start is None else float(row["delay_hours"]) - (
            float(start["transfer_time"]) - float(start["nominal_transfer_time"]))
    write_csv(out / "cycle_comparison.csv", cycle_rows)
    write_csv(out / "task_execution_comparison.csv", task_rows)

    baseline_facts = trajectories.get(0, {}).get("execution_facts", {}).get("tasks_execution", {})
    deviation_rows = []
    for point, data in sorted(trajectories.items()):
        now = data.get("execution_facts", {}).get("tasks_execution", {})
        for task_key in sorted(set(baseline_facts) | set(now)):
            initial, current = baseline_facts.get(task_key), now.get(task_key)
            if initial is None or current is None:
                deviation_rows.append({"point": point, "task_key": task_key, "missing": True})
                continue
            deviation_rows.append({"point": point, "task_key": task_key,
                                   "start_delta_hours": float(current["actual_start"]) - float(initial["actual_start"]),
                                   "end_delta_hours": float(current["actual_end"]) - float(initial["actual_end"]),
                                   "team_set_changed": sorted(current.get("team", [])) != sorted(initial.get("team", [])),
                                   "revision_count": len(current.get("revision_history", []))})
    write_csv(out / "task_deviations_vs_step0000.csv", deviation_rows)
    initial_actions = trajectories.get(0, {}).get("action_trace", [])
    action_rows = []
    for point, data in sorted(trajectories.items()):
        current_actions = data.get("action_trace", [])
        initial_branch_task = [(item.get("branch"), item.get("task_key")) for item in initial_actions]
        current_branch_task = [(item.get("branch"), item.get("task_key")) for item in current_actions]
        first_difference = next((index for index, pair in enumerate(zip(initial_branch_task, current_branch_task))
                                 if pair[0] != pair[1]), None)
        if first_difference is None and len(initial_branch_task) != len(current_branch_task):
            first_difference = min(len(initial_branch_task), len(current_branch_task))
        order_differences = sum(
            1 for before, after in zip(initial_actions, current_actions)
            if before.get("task_key") == after.get("task_key")
            and before.get("team_order") != after.get("team_order")
        )
        schedule_difference_count = sum(
            bool(item.get("team_set_changed")) or abs(float(item.get("start_delta_hours", 0.0))) > 1e-7
            or abs(float(item.get("end_delta_hours", 0.0))) > 1e-7
            for item in deviation_rows if int(item["point"]) == point
        )
        action_rows.append({"point": point, "initial_action_count": len(initial_actions),
                            "current_action_count": len(current_actions),
                            "branch_task_sequence_equal": initial_branch_task == current_branch_task,
                            "first_branch_task_difference_index": first_difference,
                            "worker_order_difference_count_at_aligned_task": order_differences,
                            "physical_task_schedule_difference_count": schedule_difference_count,
                            "dimensions_reported_separately": True})
    write_csv(out / "action_order_comparison.csv", action_rows)

    old_rows = []
    for point in POINTS:
        old_path = OLD_RUN / "shared_step0000_NOMINAL.json" if point == 0 else OLD_R1 / "trajectories" / f"step{point:04d}_NOMINAL.json"
        if old_path.is_file():
            old = read_json(old_path)
            old_rows.append({"point": point, "old_j_total": float(old["objective_j_total"]),
                             "old_j_takt": old.get("cost_components_weighted", {}).get("j_takt"),
                             "old_decisions": old.get("decisions"), "old_completed_tasks": old.get("completed_tasks"),
                             "new_j_total": next((row["j_total"] for row in rows if row["point"] == point), None),
                             "new_minus_old_j": next((row["j_total"] - float(old["objective_j_total"])
                                                       for row in rows if row["point"] == point), None),
                             "same_ppo_seed": True, "same_calibrated_start": True,
                             "independent_training_run": True})
    write_csv(out / "historical_r1_same_point.csv", old_rows)

    cross_seed_rows = []
    seed42_csv = ABLATION / "comparison_by_point.csv"
    seed42_values = {(row["arm"], int(row["point"])): float(row["j_total"])
                     for row in csv.DictReader(seed42_csv.open(encoding="utf-8-sig"))}
    for point in (128, 256, 512):
        cal_path = CAL_PPO / "trajectories" / f"step{point:04d}_NOMINAL.json"
        if cal_path.is_file():
            seed42_values["REF", point] = float(read_json(cal_path)["objective_j_total"])
    bc_j = float(read_json(BC_SOURCE / "trajectories" / "C_BC_DIAGNOSTIC_NOMINAL.json")["objective_j_total"])
    seed42_values["REF", 0] = bc_j
    for point in (0, 128, 256, 512):
        r1 = seed42_values.get(("R1", point))
        ref_j = seed42_values.get(("REF", point))
        if r1 is not None:
            cross_seed_rows.append({"ppo_seed": 42, "initialization_seed": 42, "point": point,
                                    "arm": "R1", "j_total": r1, "reference_j_total": ref_j,
                                    "delta_j_ref_minus_r1": None if ref_j is None else ref_j - r1,
                                    "source": "historical_seed42_ablation", "paired_with_reference_within_source_run": ref_j is not None,
                                    "independent_initialization": False})
    for point in POINTS:
        current = next((float(row["j_total"]) for row in rows if row["point"] == point), None)
        prior = next((float(row["old_j_total"]) for row in old_rows if row["point"] == point), None)
        old_ref_path = OLD_RUN / "shared_step0000_NOMINAL.json" if point == 0 else OLD_RUN / "REF" / "trajectories" / f"step{point:04d}_NOMINAL.json"
        old_ref = float(read_json(old_ref_path)["objective_j_total"]) if old_ref_path.is_file() else None
        cross_seed_rows.append({"ppo_seed": 43, "initialization_seed": 42, "point": point,
                                "arm": "R1", "j_total": current, "prior_same_seed_r1_j_total": prior,
                                "reference_j_total": old_ref,
                                "delta_j_ref_minus_r1": None if current is None or old_ref is None else old_ref - current,
                                "source": "new_r1_with_historical_seed43_context",
                                "paired_with_reference_within_source_run": False,
                                "reference_scope": "旧seed43 REF；仅作同seed历史背景，不是本次R1运行内同步训练对照",
                                "independent_initialization": False})
    write_csv(out / "cross_seed_summary.csv", cross_seed_rows)

    initial_model, _ = calibration.load_calibrated_actor(MODEL)
    initial = dict(initial_model.named_parameters())
    groups = common.parameter_groups(initial_model)["groups"]
    subsets = {"shared": groups["shared"], "policy": groups["policy"],
               "critic": groups["value"], "policy_plus_shared": groups["shared"] + groups["policy"]}
    param_rows = []
    for point in POINTS:
        checkpoint = MODEL if point == 0 else out / "R1" / "checkpoints" / f"step{point:04d}.pt"
        if not checkpoint.is_file():
            continue
        model, _payload = calibration.load_calibrated_actor(MODEL) if point == 0 else common.load_checkpoint_actor(checkpoint)
        current = dict(model.named_parameters())
        for subset, names in subsets.items():
            squared = sum(float((current[name].detach().cpu().double() - initial[name].detach().cpu().double()).square().sum())
                          for name in names)
            initial_squared = sum(float(initial[name].detach().cpu().double().square().sum()) for name in names)
            l2 = math.sqrt(squared)
            param_rows.append({"point": point, "subset": subset, "l2": l2,
                               "relative_l2": l2 / math.sqrt(initial_squared) if initial_squared > 0 else 0.0})
    write_csv(out / "parameter_change_metrics.csv", param_rows)

    probe_rows = []
    initial_logps = None
    for point in POINTS:
        probe_path = out / "R1" / "probe" / f"step{point:04d}.json"
        if not probe_path.is_file():
            continue
        probe = read_json(probe_path)
        if point == 0:
            initial_logps = probe.get("ordered_teacher_action_log_prob", [])
        logps = probe.get("ordered_teacher_action_log_prob", [])
        diffs = [abs(float(a) - float(b)) for a, b in zip(initial_logps or [], logps, strict=True)] if initial_logps else []
        metrics = probe["teacher_metrics"]
        probe_rows.append({"point": point, "teacher_snapshot_count": probe.get("teacher_snapshot_count"),
                           "full_action_accuracy": metrics.get("full_physical_action_accuracy"),
                           "decision_kind_accuracy": metrics.get("decision_kind_accuracy"),
                           "task_accuracy_conditional": metrics.get("task_accuracy_conditional"),
                           "branch_accuracy_conditional": metrics.get("branch_accuracy_conditional"),
                           "team_accuracy_unordered_conditional": metrics.get("team_accuracy_unordered_conditional"),
                           "align_accuracy_conditional": metrics.get("align_accuracy_conditional"),
                           "mean_teacher_nll": metrics.get("nll_mean"),
                           "mean_abs_fixed_action_logp_delta": sum(diffs) / len(diffs) if diffs else 0.0,
                           "max_abs_fixed_action_logp_delta": max(diffs, default=0.0),
                           "exact_joint_kl": None,
                           "coverage": "冻结BC教师状态；不代表策略自主访问的新状态"})
    write_csv(out / "policy_probe_metrics.csv", probe_rows)
    write_json(out / "trajectory_verification.json", {
        "rows": verifier_rows, "expected_points": list(POINTS), "verified_points": [row["point"] for row in verifier_rows],
        "physical_time_tolerance_hours": 1e-7, "cost_ledger_tolerance": 1e-9,
        "status": "INCOMPLETE" if len(verifier_rows) != len(POINTS) else (
            "FAIL" if any(row.get("status") == "FAIL" for row in verifier_rows) else
            "PASS" if all(row.get("status") == "PASS" for row in verifier_rows) else "INCOMPLETE"),
    })
    return {"trajectory_rows": rows, "verifier_rows": verifier_rows, "historical_rows": old_rows,
            "parameter_rows": param_rows, "probe_rows": probe_rows}


def _write_objective_alignment(out: Path) -> dict[str, Any]:
    import work3_4060.run_bc_ppo_retention as common
    events_path = out / "R1" / "shaping_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()] if events_path.is_file() else []
    transitions = [row for row in events if row.get("record_type", "transition") == "transition"]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in transitions:
        grouped.setdefault((int(row.get("worker_id", 0)), int(row.get("episode_id", 0))), []).append(row)
    audits = []
    nonzero_reward_locations = []
    gamma = 0.99
    for (worker, episode), rows in grouped.items():
        rows.sort(key=lambda row: int(row.get("episode_step", 0)))
        raw = sum(float(row["raw_reward"]) for row in rows)
        discounted_raw = sum((gamma ** index) * float(row["raw_reward"]) for index, row in enumerate(rows))
        discounted_gap = sum((gamma ** index) * (float(row["shaped_reward"]) - float(row["raw_reward"]))
                             for index, row in enumerate(rows))
        phi0 = float(rows[0]["phi_current"])
        terminal = bool(rows[-1].get("terminated"))
        phi_end = float(rows[-1]["phi_next"])
        expected_gap = -phi0 if terminal else (gamma ** len(rows)) * phi_end - phi0
        cost_increments = {
            key: sum(float(row.get("cost_component_increments", {}).get(key, 0.0)) for row in rows)
            for key in ("cost_takt", "cost_time", "cost_team", "cost_postpone", "cost_revision")
        }
        if any(abs(value) > 1e-15 for row in rows for value in row.get("cost_component_increments", {}).values()):
            for row in rows:
                increments = row.get("cost_component_increments", {})
                if any(abs(float(value)) > 1e-15 for value in increments.values()):
                    nonzero_reward_locations.append({
                        "worker_id": worker, "episode_id": episode,
                        "episode_step": int(row.get("episode_step", 0)),
                        "decision_time_hours": row.get("decision_time"), "cycle_id": row.get("cycle_id"),
                        "task_key": row.get("task_key"), "raw_reward": row.get("raw_reward"),
                        "discount_weight_gamma_0_99": gamma ** int(row.get("episode_step", 0)),
                        "cost_component_increments": increments,
                    })
        cost_increment_sum = sum(cost_increments.values())
        audits.append({"worker_id": worker, "episode_id": episode, "transitions": len(rows),
                       "natural_terminal": terminal, "raw_reward_sum": raw,
                       "cost_increment_sum": cost_increment_sum,
                       "raw_reward_cost_increment_residual": raw + cost_increment_sum,
                       "raw_cost_identity_pass_1e-9": abs(raw + cost_increment_sum) <= 1e-9,
                       "discounted_raw_return_gamma_0_99": discounted_raw,
                       "discounted_shaping_difference": discounted_gap,
                       "potential_identity_expected": expected_gap,
                       "potential_identity_residual": discounted_gap - expected_gap,
                       "identity_pass_1e-9": abs(discounted_gap - expected_gap) <= 1e-9,
                       "cost_component_increment_recorded": all("cost_component_increments" in row for row in rows)})
    accounting_rows = []
    for row in _read_eval_rows(out):
        path = Path(row["path"])
        data = read_json(path)
        raw_rewards = data.get("execution_facts", {}).get("step_rewards", [])
        raw_sum = sum(float(item) for item in raw_rewards)
        accounting_rows.append({"source": path.name, "episode_type": "greedy_evaluation_natural_episode",
                                "raw_reward_sum": raw_sum, "negative_j_total": -float(data["objective_j_total"]),
                                "raw_cost_identity_residual": raw_sum + float(data["objective_j_total"]),
                                "raw_cost_identity_pass_1e-9": abs(raw_sum + float(data["objective_j_total"])) <= 1e-9,
                                "gamma_discounted_raw_return": sum(gamma ** i * float(item) for i, item in enumerate(raw_rewards)),
                                "j_total_is_undiscounted": True})
    shaping_verifier = common.load_shaping_verifier()
    run_json = out / "R1" / "runs" / "ppo_run.json"
    shaping_audit = shaping_verifier.verify_shaping_trace(events_path, run_json, out / "R1" / "shaping_audit_summary.json", tolerance=1e-9) if events_path.is_file() and run_json.is_file() else {"overall_status": "INCOMPLETE"}
    result = {"gamma": gamma, "beta": 1.0, "reward_identity": "J_total = -sum_t r_raw_t", "discounted_objective": "G_gamma_raw=sum_t gamma^t r_raw_t",
              "training_episode_shaping_audits": audits, "greedy_evaluation_accounting": accounting_rows,
              "nonzero_training_cost_reward_locations": nonzero_reward_locations,
              "independent_shaping_verifier_status": shaping_audit.get("overall_status", "INCOMPLETE"),
              "truncated_episode_boundary": "截断保留 gamma^T * Phi(s_T) 与 bootstrap，不把预算截止设为自然终止。"}
    all_residuals_pass = all(row["identity_pass_1e-9"] for row in audits)
    if audits:
        result["training_potential_identity_status"] = "PASS" if all_residuals_pass else "FAIL"
        result["training_raw_cost_identity_status"] = "PASS" if all(row["raw_cost_identity_pass_1e-9"] for row in audits) else "FAIL"
    else:
        result["training_potential_identity_status"] = "INCOMPLETE"
        result["training_raw_cost_identity_status"] = "INCOMPLETE"
    result["evaluation_raw_cost_identity_status"] = (
        "PASS" if accounting_rows and all(row["raw_cost_identity_pass_1e-9"] for row in accounting_rows)
        else "FAIL" if accounting_rows else "INCOMPLETE")
    note = [
        "# 离线目标对齐核对", "",
        "原业务评价仍是未折扣的 J_total。训练中的 γ=0.99 按决策步作用于 raw return；它不是以小时计的物理时间折扣。",
        "逐条 greedy 完工 episode 用执行事实中的 step_rewards 核验 J_total=-Σ r_raw，并单独计算 G_gamma_raw。两者对不同决策位置赋权，因此不是同一目标。",
        "本轮记录每次 transition 的五类费用累计增量差分，可用于追踪奖励发生位置。折扣使靠后的费用在 G_gamma_raw 中权重更低，但本轮只确认数学权重差异；没有证据宣称策略已利用折扣套利。",
        "自然终止时核验 Σγ^t(r_shaped-r_raw)=-βΦ(s0)；非自然终止保留 γ^TΦ(sT)-Φ(s0) 并结合现有 bootstrap/边界审计。费用符号不作为优化是否改善的证据。",
        "当前采用决策步折扣的明确业务理由尚未在协议中找到。建议在正式恢复学习前单独讨论：若原 J_total 是有限任务全程总费用，γ<1可能改变事件时间权重；是否改为 γ=1 或物理时间折扣应先写清任务目标、终止/截断合同并由用户确认。本轮不改 γ、不追加对照。", "",
    ]
    (out / "objective_alignment_note.md").write_text("\n".join(note), encoding="utf-8")
    write_json(out / "objective_alignment_audit.json", result)
    return result


def _read_eval_rows(out: Path) -> list[dict[str, Any]]:
    path = out / "evaluation_results.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_recovery_draft(out: Path) -> None:
    text = """# 扰动恢复学习协议草案（待讨论，本轮未训练）

## 研究问题与初始化

主问题设为：扰动发生后，策略能否在物理可行约束内采取必要调整，并降低原始 `J_total`。Nominal 只作为退化监测，纯 Nominal 复现不算恢复贡献。暂以固定规则示范得到的 BC 模型作为 C 的初始化，并保留冻结副本；本轮 1024 步诊断终点既不晋升为正式起点，也不从检查点中挑选“最好”的模型。

## 对照与数据预算建议

建议固定三种比较：A（规则修复）、冻结 BC 初始化的 C、从同一冻结 BC 初始化恢复训练后的 C。建议正式恢复训练使用 **16 个预先生成的训练扰动实例、每实例最多 512 个训练决策，总交互上限 8192**；另设 4 个验证实例用于训练期间诊断，且验证决策不计入训练预算。该预算是待用户讨论的方案，不是本轮已批准或执行的训练。比较扩展 D/E/F 时，示范数据实例、可见状态、初始化权重、训练决策预算和评估预算应配平；任何额外先验需单列。

数据拆分需记录事件模板、目标工序、发生周期、强度、随机种子及版本。已经反复查看的 v1 两个验证场景只可称已见验证集，不能再称盲测。v1 与 v2 的事件定义、命中口径或物料边界不可静默混用。建议另留 8 个从未用于建模和调参的 v2 测试实例；测试只在协议、阈值和检查点规则冻结后运行一次。

## 可观测性、现场匹配与结果

扰动发生前事件不得对策略可见；发生后按预先声明的观测接口告知。实际命中目标不等于执行改变：逐个核对命中、物料可用下界、目标实际开工时刻、修订/后移动作及后续周期传播。

主要对照应在相同受扰现场和相同事件时刻比较冻结策略与恢复策略；另外报告端到端各自结果及相对各自 Nominal 的增量。仅有更小增量、无周期延迟或最终 makespan 相同均不足以判优。主要评价继续使用原五分项及 `J_total`，同时报告可行性、出线、后移、延迟周期和恢复任务偏差。

若需要新增 Nominal 警戒线，应在新协议运行前公开固定并与用户讨论；不得事后修改本轮 `J_total≤1e-3`、`J_takt≤1e-6`、后移为零的历史门槛。

## 建议门禁与待确认设计点

建议训练前冻结：实例清单及拆分、事件知晓时刻、三个对照的初始化和示范支持、各组交互/评估预算、独立轨迹核验、原始费用账本、检查点选择规则、Nominal 警戒线、失败停止条件。任何测试集泄漏、权重来源不一致、非法物理动作、账本不闭合或预算不匹配均阻止效果结论。

尚需讨论：① `γ=0.99` 的决策步折扣是否与有限任务全程费用目标一致；建议先明确目标，再考虑 `γ=1` 或按物理时间折扣，另立协议比较。② 验证集是否只监测而不选模型；建议固定最后一步作为主结果，另将任何早停规则预注册。③ “必要恢复”是否需要相对 A 的费用改善及非劣 Nominal 线；建议使用受扰现场配对的原始 `J_total` 差值，并在看新结果前定阈值。④ 训练种子与扰动实例如何配对；建议先固定实例清单，跨训练种子复用相同测试实例，但不把多个检查点当独立样本。

本文件是方案草案，不生成扰动数据、不运行恢复训练，也不授权扩大本轮范围。
"""
    (out / "recovery_learning_protocol_draft.md").write_text(text, encoding="utf-8")


def _source_hash_status(source_binding: Mapping[str, Any]) -> dict[str, bool]:
    return {name: Path(item["path"]).is_file() and item.get("sha256") == sha256(Path(item["path"]))
            for name, item in source_binding["source_code_hashes"].items()}


def _finalize(out: Path, source_binding: Mapping[str, Any], stage_results: Mapping[str, Any]) -> dict[str, Any]:
    import work3_4060.run_ppo_retention_repeat as repeat
    import work3_4060.run_bc_ppo_retention as common
    import work3_4060.run_critic_calibration_retention as calibration

    report = _write_evaluation_reports(out)
    objective = _write_objective_alignment(out)
    _write_recovery_draft(out)
    run_json = out / "R1" / "runs" / "ppo_run.json"
    shaping_path = out / "R1" / "shaping_events.jsonl"
    shaping = common.load_shaping_verifier().verify_shaping_trace(
        shaping_path, run_json, out / "R1" / "shaping_audit_summary.json", tolerance=1e-9
    ) if shaping_path.is_file() and run_json.is_file() else {"overall_status": "INCOMPLETE"}
    pairing = read_json(out / "R1" / "first_rollout_pairing.json") if (out / "R1" / "first_rollout_pairing.json").is_file() else {"paired": False}
    prior_update = read_json(OLD_R1 / "first_update.json")
    new_update = read_json(out / "R1" / "first_update.json") if (out / "R1" / "first_update.json").is_file() else {}
    minibatch_pairing = {
        "old_samples": prior_update.get("first_minibatch_samples"),
        "new_samples": new_update.get("first_minibatch_samples"),
        "old_permutation": prior_update.get("minibatch_permutation"),
        "new_permutation": new_update.get("minibatch_permutation"),
        "exact_match": prior_update.get("first_minibatch_samples") == new_update.get("first_minibatch_samples")
        and prior_update.get("minibatch_permutation") == new_update.get("minibatch_permutation"),
    }
    write_json(out / "training" / "first_rollout_pairing.json", {"rollout_pairing": pairing, "minibatch_identity": minibatch_pairing})

    summary_path = out / "R1" / "training_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    runtime_json_path = out / "R1" / "runs" / "ppo_run.json"
    runtime_json = read_json(runtime_json_path) if runtime_json_path.is_file() else {}
    history = [json.loads(line) for line in (out / "R1" / "training_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()] if (out / "R1" / "training_history.jsonl").is_file() else []
    optimizer_steps = int(summary.get("optimizer_steps", 0))
    all_lrs = [float(lr) for segment in history for update in segment.get("update_records", []) for lr in update.get("learning_rates", [])]
    expected_lr = float(read_json(CAL_PPO / "training_summary.json")["hyperparameters"]["learning_rate"]) * 0.1
    optimizer_groups = summary.get("optimizer_param_group_settings", [])
    optimizer_contract = bool(summary) and summary.get("optimizer_name") == "AdamW" and not summary.get("optimizer_schedulers") \
        and len(all_lrs) == optimizer_steps and all(abs(lr - expected_lr) <= 1e-15 for lr in all_lrs)
    optimizer_contract = optimizer_contract and bool(optimizer_groups) and all(
        abs(float(group.get("lr", math.inf)) - expected_lr) <= 1e-15
        and abs(float(group.get("eps", math.inf)) - 1e-5) <= 1e-15
        and abs(float(group.get("weight_decay", math.inf)) - 1e-4) <= 1e-15
        for group in optimizer_groups
    )
    runtime_training = runtime_json.get("training_config", {})
    runtime_budget_contract = runtime_training.get("total_env_steps") == BUDGET and runtime_training.get("max_decisions") == BUDGET
    runtime_lr_contract = abs(float(runtime_training.get("learning_rate", math.inf)) - expected_lr) <= 1e-15
    step0 = read_json(out / "training" / "step0000_contract.json") if (out / "training" / "step0000_contract.json").is_file() else {"passed": False}
    verification_rows = report["verifier_rows"]
    expected_points = list(POINTS)
    all_points_present = [row["point"] for row in report["trajectory_rows"]] == expected_points
    physical_status = "FAIL" if any(row.get("status") == "FAIL" for row in verification_rows) else (
        "PASS" if len(verification_rows) == len(POINTS) and all(row.get("status") == "PASS" for row in verification_rows) else "INCOMPLETE")
    source_checks = _source_hash_status(source_binding)
    source_unchanged = all(source_checks.values()) and bool(source_checks)
    old_input_hashes = {row["name"]: (row["sha256"] == sha256(Path(row["path"])) if row["exists"] else False)
                        for row in read_json(out / "dependency_inventory.json")["files"] if row["exists"]}
    old_sources_unchanged = all(old_input_hashes.values())
    archive_pre = read_json(out / "backup" / "archive_verification_pretraining.json") if (out / "backup" / "archive_verification_pretraining.json").is_file() else {}
    archive_post = {}
    init_record = read_json(out / "training" / "initialization_verification.json")
    expected_initial_fingerprints = {
        "full_model": init_record.get("full_model_fingerprint"),
        "policy_subset": init_record.get("policy_subset_fingerprint"),
        "critic_subset": init_record.get("critic_subset_fingerprint"),
    }
    protocol_checks = {
        "fixed_source_and_dependencies_unchanged": source_unchanged and old_sources_unchanged,
        "regression_suite_passed": read_json(out / "engineering" / "regression_test_results.json").get("status") == "PASS",
        "fp32_equivalence_passed": read_json(out / "engineering" / "fp32_equivalence.json").get("status") == "PASS",
        "pretraining_archive_verified": archive_pre.get("all_members_verified_after_readback") is True,
        "step0000_matches_bc_and_verifier": step0.get("passed") is True,
        "first_rollout_paired": pairing.get("paired") is True,
        "first_minibatch_identity_matches": minibatch_pairing["exact_match"],
        "r1_adamw_lr_contract": optimizer_contract,
        "run_json_effective_lr_and_budget_contract": runtime_lr_contract and runtime_budget_contract,
        "initial_parameters_complete_and_r1_has_no_r4_hook": bool(summary.get("optimized_parameter_count") == 138)
        and summary.get("detach_value") is False and not summary.get("detach_hook", {}).get("installed", True)
        and summary.get("detach_hook", {}).get("hooks_after_cleanup") == 0,
        "checkpoint_initial_fingerprints_match_source": summary.get("initial_fingerprints")
        == expected_initial_fingerprints,
        "actual_training_budget_1024": summary.get("training_decisions") == BUDGET and summary.get("stop_reason") == "budget_exhausted",
        "all_six_fixed_evaluations_present": all_points_present,
        "objective_raw_cost_accounting_passed": objective.get("training_raw_cost_identity_status") == "PASS"
        and objective.get("evaluation_raw_cost_identity_status") == "PASS",
        "potential_boundary_identity_passed": objective.get("training_potential_identity_status") == "PASS",
    }
    protocol = _classify_protocol_status(
        protocol_checks,
        shaping_status=shaping.get("overall_status", "INCOMPLETE"),
        objective_statuses={
            "training_raw_cost": objective.get("training_raw_cost_identity_status", "INCOMPLETE"),
            "evaluation_raw_cost": objective.get("evaluation_raw_cost_identity_status", "INCOMPLETE"),
            "training_potential": objective.get("training_potential_identity_status", "INCOMPLETE"),
        },
        physical_status=physical_status,
    )
    retention_by_point = {int(row["point"]): bool(row["retention_pass"]) for row in report["trajectory_rows"]}
    eval_complete = len(retention_by_point) == len(POINTS) and all(row.get("success") and row.get("feasible") for row in report["trajectory_rows"])
    retention = "INCOMPLETE" if not eval_complete else ("PASS" if all(retention_by_point.get(point, False) for point in POINTS) else "FAIL")
    if physical_status == "FAIL":
        production = "FAIL"
    elif physical_status != "PASS" or not eval_complete:
        production = "INCOMPLETE"
    else:
        production = "PASS"
    memory = summary.get("gpu", {})
    run_counts = {
        "training_environment_decisions": int(summary.get("training_decisions", 0)),
        "optimizer_steps": optimizer_steps,
        "offline_fp32_diagnostic_updates": 1,
        "evaluation_decisions": sum(int(row["decisions"]) for row in report["trajectory_rows"]),
        "evaluation_episodes": len(report["trajectory_rows"]),
        "discarded_environment_decisions": 0,
        "discarded_optimizer_steps": 0,
        "training_segments": len(history),
        "episodes_started": summary.get("episodes_started"),
        "episodes_completed": summary.get("episodes_completed"),
        "stop_reason": summary.get("stop_reason"),
        "elapsed_seconds": summary.get("elapsed_seconds"),
        "gpu_name": memory.get("name"),
        "peak_allocated_mib": memory.get("peak_allocated_mib"),
        "peak_reserved_mib": memory.get("peak_reserved_mib"),
    }
    verification = {
        "protocol_status": protocol, "production_status": production, "retention_status": retention,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_checks": protocol_checks, "physical_verifier_status": physical_status,
        "shaping_audit_status": shaping.get("overall_status", "INCOMPLETE"),
        "retention_by_point": retention_by_point, "strict_retention_thresholds": {
            "completed_tasks": 210, "completed_aircraft": 7, "j_total_max": 1e-3, "j_takt_max": 1e-6, "postponed_max": 0,
        },
        "optimizer_contract": {"expected_lr": expected_lr, "observed_lr_min": min(all_lrs, default=None),
                               "observed_lr_max": max(all_lrs, default=None), "observed_lr_count": len(all_lrs),
                               "optimizer_steps": optimizer_steps, "parameter_groups": optimizer_groups,
                               "scheduler_count": len(summary.get("optimizer_schedulers", [])), "passed": optimizer_contract},
        "source_unchanged": source_checks, "input_unchanged": old_input_hashes,
        "historical_autocast_runtime_exact_reproduction_status": "INCOMPLETE",
        "run_counts_and_resources": run_counts,
        "stage_results": dict(stage_results), "objective_alignment": objective,
    }
    checkpoint_integrity = {}
    for path in sorted((out / "R1" / "checkpoints").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_integrity[str(path.relative_to(out))] = {
            "sha256": sha256(path), "actor_fingerprint": payload.get("actor_fingerprint"),
            "training_decisions": int(payload.get("training_decisions", -1)),
            "optimizer_steps": int(payload.get("optimizer_steps", -1)),
        }
    write_json(out / "weight_integrity.json", {
        "source_model_sha256": sha256(MODEL), "source_full_fingerprint": EXPECTED_FULL_FINGERPRINT,
        "source_policy_subset_fingerprint": EXPECTED_POLICY_FINGERPRINT,
        "training_summary_initial_fingerprints": summary.get("initial_fingerprints"),
        "checkpoints": checkpoint_integrity,
        "source_model_unchanged": sha256(MODEL) == EXPECTED_MODEL_SHA256,
    })
    write_json(out / "verification.json", verification)
    _write_summary(out, verification, report, summary, run_counts)
    archive_post = create_verified_archive(out, label="postrun")
    verification["postrun_archive"] = {"path": archive_post["archive_path"], "sha256": archive_post["archive_sha256"],
                                        "verified": archive_post["all_members_verified_after_readback"]}
    write_json(out / "verification.json", verification)
    return verification


def _write_summary(out: Path, verification: Mapping[str, Any], report: Mapping[str, Any],
                   summary: Mapping[str, Any], run_counts: Mapping[str, Any]) -> None:
    rows = report["trajectory_rows"]
    lines = [
        "# Work-3 工程收口与 R1 seed43 / 1024 步保持复核", "",
        f"协议 / 生产 / 保持状态：{verification['protocol_status']} / {verification['production_status']} / {verification['retention_status']}。",
        "初始化来自 BC/价值校准 seed42；本轮 PPO 采样 seed43。R1 因 seed42 结果被选为主要复核，seed43 是后续复核，不是盲测。",
        "本轮只有一个 R1 训练运行；1024 步从共同校准权重重新开始，不续接旧 step512。", "",
        "## 固定评估点", "",
        "|决策|J_total|J_takt|D_time|D_team|J_postpone|J_revision|最大周期延迟(h)|出线架数|严格保持|", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|".format(
            row["point"], *[format(float(row[key]), ".10g") for key in ("j_total", "j_takt", "d_time", "d_team", "j_postpone", "j_revision", "max_cycle_delay_hours")],
            row["completed_aircraft"], "是" if row["retention_pass"] else "否"))
    lines += ["", "旧 seed43 R1 只用于同点历史复核；两个 PPO 运行共享 BC/校准初始化，不能视为两个独立模型或泛化样本。", ""]
    for row in report["historical_rows"]:
        lines.append(f"- {row['point']} 步：旧 J={row['old_j_total']:.17g}；新 J={float(row['new_j_total']):.17g}；新−旧={float(row['new_minus_old_j']):+.17g}。")
    by_point = {int(row["point"]): row for row in rows}
    if all(point in by_point for point in (512, 768, 1024)):
        j512, j768, j1024 = (float(by_point[point]["j_total"]) for point in (512, 768, 1024))
        d768, d1024 = j768 - j512, j1024 - j512
        if abs(d768) <= 1e-9 and abs(d1024) <= 1e-9:
            trend = "相近（差值均在 1e-9 费用账本容差内）"
        elif d768 > 0 and j1024 > j768:
            trend = "继续上升"
        elif d768 * d1024 < 0:
            trend = "波动（768 与 1024 位于 512 基线两侧）"
        elif d1024 < 0:
            trend = "相对 512 回落"
        else:
            trend = "波动或小幅上行"
        lines.append(f"- 512→768→1024 的 J_total={j512:.17g}→{j768:.17g}→{j1024:.17g}；相对512变化={d768:+.17g}/{d1024:+.17g}，描述为：{trend}。这只是固定评估点的描述，不是稳定性或统计趋势证明。")
        with (out / "task_deviations_vs_step0000.csv").open(encoding="utf-8-sig", newline="") as handle:
            changed_tasks = [row for row in csv.DictReader(handle) if int(row["point"]) == 1024
                             and (row.get("team_set_changed", "").lower() == "true"
                                  or abs(float(row.get("start_delta_hours") or 0.0)) > 1e-7
                                  or abs(float(row.get("end_delta_hours") or 0.0)) > 1e-7
                                  or int(float(row.get("revision_count") or 0)) > 0)]
        lines.append("- 1024 相对共同 step0 的任务偏离：" + ("、".join(row["task_key"] for row in changed_tasks[:20])
                      if changed_tasks else "未发现超过物理时间容差的开工/完工/团队/修订差异") + ("（最多列20项，完整清单见CSV）。" if len(changed_tasks) > 20 else "。"))
    lines += ["", "## 训练与工程证据", "",
              f"- 环境决策 {run_counts['training_environment_decisions']}；optimizer.step {run_counts['optimizer_steps']}；段数 {run_counts['training_segments']}；停止原因 `{run_counts['stop_reason']}`；耗时 {run_counts['elapsed_seconds']} 秒。",
              f"- 峰值显存 allocated/reserved：{run_counts['peak_allocated_mib']}/{run_counts['peak_reserved_mib']} MiB。评估另计 {run_counts['evaluation_episodes']} 条、{run_counts['evaluation_decisions']} 个决策；离线 FP32 更新不计入主训练。",
              f"- 首段动作/RNG/GAE 配对：{verification['protocol_checks']['first_rollout_paired']}；首个 mini-batch 身份配对：{verification['protocol_checks']['first_minibatch_identity_matches']}。",
              f"- R1 AdamW 所有实际学习率均核验为 {verification['optimizer_contract']['expected_lr']:.17g}；观测范围 {verification['optimizer_contract']['observed_lr_min']}–{verification['optimizer_contract']['observed_lr_max']}。",
              f"- 物理事实核验：{verification['physical_verifier_status']}；塑形审计：{verification['shaping_audit_status']}。", "",
              "- autocast 历史运行时源文件未能逐字节恢复；FP32检查只重建记录到的 disabled-context 差异并通过当前设备对照，所以精确历史复现状态为 INCOMPLETE。", "",
              "## 判读", "",
              "严格保持要求 210 道工序、7 架出线、J_total≤1e-3、J_takt≤1e-6、后移0。节拍保持与总费用保持分开报告；若 J_takt≈0 但其余费用超过阈值，严格保持仍未通过。",
              "位移、冻结 BC probe 动作一致率和固定有序动作 log-prob变化分别见 CSV；这些是策略功能变化观测，不等于长期有效或学习效率恢复。没有精确联合动作 KL 时不作此声称。",
              "本次结果只涉及 mini210、Nominal 和固定初始化。候选解释不等于因果结论；R1 全局学习率也同步改变 AdamW 衰减步幅。", "",
              "## 目标与下一步", "",
              "原始 `J_total` 为未折扣费用；PPO 的 `G_gamma_raw` 按决策步折扣，两者不同。决策步折扣的业务依据仍需在恢复学习前单独讨论。",
              "下一阶段只建议讨论并冻结扰动恢复协议草案：比较规则 A、冻结 BC-C 与恢复训练 C，在相同受扰现场对照，Nominal 作为退化监测。本轮不启动扰动训练。", "",
              "## 产物", "",
              "`manifest.json`、`dependency_inventory.json`、`source_binding.json`、`training/resolved_config.yaml`、`training/first_rollout_pairing.json`、`engineering/`、`backup/`、`R1/`、`evaluation_results.csv`、`historical_r1_same_point.csv`、`cycle_comparison.csv`、`task_deviations_vs_step0000.csv`、`policy_probe_metrics.csv`、`parameter_change_metrics.csv`、`trajectory_verification.json`、`objective_alignment_note.md`、`recovery_learning_protocol_draft.md`、`verification.json`。", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(out_arg: Path) -> int:
    out = resolve_output_root(out_arg)
    config, source_binding, _initialization = prepare_run(out)
    stage_results: dict[str, Any] = {}
    try:
        regression_evidence = _write_engineering_evidence(out)
        write_json(out / "engineering" / "regression_test_results.json", {
            "status": "PASS" if regression_evidence.get("passed") else "FAIL",
            "baseline_and_pre_repair_classification": regression_evidence.get("classification"),
            "fixed_current_targeted_regression_status": regression_evidence.get("current_repaired_regression_status"),
            "targeted_test_selection": regression_evidence.get("test_selection"),
            "test_logs": {name: row["copied_log"] for name, row in regression_evidence["runs"].items()},
            "fixed_shared_input_hashes": regression_evidence.get("fixture_input_hashes"),
        })
        if not regression_evidence.get("passed"):
            raise RuntimeError("受影响的针对性回归尚未记录为 PASS")
        fp32 = run_fp32_equivalence(out)
        if fp32["status"] != "PASS":
            raise RuntimeError("离线 FP32 等价检查失败")
        archive = create_verified_archive(out, label="pretraining")
        write_json(out / "backup" / "archive_manifest.json", archive)
        write_json(out / "backup" / "archive_verification.json", {
            "archive_path": archive["archive_path"], "archive_sha256": archive["archive_sha256"],
            "member_count": archive["member_count"], "pt_member_count": archive["pt_member_count"],
            "prior_registered_pt_count": archive["prior_registered_pt_count"],
            "prior_registered_pt_verified_count": archive["prior_registered_pt_verified_count"],
            "prior_registered_pt_all_verified": archive["prior_registered_pt_all_verified"],
            "all_members_verified_after_readback": archive["all_members_verified_after_readback"],
        })
        if not archive["all_members_verified_after_readback"]:
            raise RuntimeError("训练前归档回读验证失败")
        code, log_path, seconds = _run_child(out, "evaluate", point=0)
        stage_results["step0000_evaluation"] = {"return_code": code, "log": str(log_path), "seconds": seconds}
        if code != 0:
            raise RuntimeError(f"step0 独立评估失败，详见 {log_path}")
        initial = _verify_step0(out)
        if not initial["passed"]:
            raise RuntimeError(f"step0 未复现 BC 基准排程，训练前停止：{initial}")
        stage_results["step0000_verification"] = initial
        code, log_path, seconds = _run_child(out, "probe", point=0)
        stage_results["probe_0000"] = {"return_code": code, "log": str(log_path), "seconds": seconds}
        if code != 0:
            raise RuntimeError(f"step0 公共 BC probe 失败，详见 {log_path}")
        code, log_path, seconds = _run_child(out, "train")
        stage_results["R1_training"] = {"return_code": code, "log": str(log_path), "seconds": seconds}
        if code != 0:
            raise RuntimeError(f"R1 固定预算训练停止，详见 {log_path}")
        for point in POINTS[1:]:
            code, log_path, seconds = _run_child(out, "evaluate", point=point)
            stage_results[f"evaluation_{point}"] = {"return_code": code, "log": str(log_path), "seconds": seconds}
            if code != 0:
                raise RuntimeError(f"step{point} 独立评估失败，详见 {log_path}")
            stage_results[f"evaluation_verification_{point}"] = _verify_completed_evaluation(out, point)
            code, log_path, seconds = _run_child(out, "probe", point=point)
            stage_results[f"probe_{point}"] = {"return_code": code, "log": str(log_path), "seconds": seconds}
            if code != 0:
                raise RuntimeError(f"step{point} 固定 BC probe 失败，详见 {log_path}")
        _finalize(out, source_binding, stage_results)
        return 0
    except Exception as error:
        write_json(out / "runs" / "stage_results.json", stage_results)
        write_json(out / "verification.json", {
            "protocol_status": "INCOMPLETE", "production_status": "INCOMPLETE", "retention_status": "INCOMPLETE",
            "stop_reason": f"{type(error).__name__}: {error}", "stage_results": stage_results,
            "training_summary_exists": (out / "R1" / "training_summary.json").is_file(),
        })
        if (out / "summary.md").is_file():
            with (out / "summary.md").open("a", encoding="utf-8") as handle:
                handle.write(f"\n\n本轮停止：`{type(error).__name__}: {error}`。\n")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="R1 seed43、1024步 Work-3 保持性复核")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--worker", choices=("train", "evaluate", "probe"), default=None)
    parser.add_argument("--point", type=int, default=None)
    args = parser.parse_args()
    if args.worker is not None:
        return _worker(args.worker, resolve_output_root(args.output), args.point)
    return run_all(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
