"""Seed43 的 Work-3 三臂短程保持性复核入口。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "work3_4060"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = BASE / "ppo_retention_repeat_v1" / "ppo_seed43"
CAL_ROOT = BASE / "critic_calibration_retention_v1" / "seed42"
CAL_DIR = CAL_ROOT / "calibration"
PPO_SOURCE = CAL_ROOT / "ppo"
ABLATION_SOURCE = BASE / "ppo_ablation_arms_v1" / "seed42"
BC_SOURCE = BASE / "c_reproduction_diagnostic_v1" / "seed42"
BASELINE = BASE / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"
MODEL = CAL_DIR / "calibrated_model.pt"
SEED = 43
ARMS = ("REF", "R1", "R4")
POINTS = (0, 8, 128, 256, 512)
TOLERANCE = {"state_feats": 1e-7, "time_urgencies": 1e-7, "old_log_probs": 1e-4,
             "old_values": 1e-4, "raw_rewards": 1e-9, "shaped_rewards": 1e-9,
             "bootstrap_value": 1e-4, "raw_advantages": 1e-4, "advantages": 1e-4,
             "return_targets": 1e-4, "snapshot_facts": 1e-7, "bootstrap_input": 1e-7}
EXPECTED_MODEL_SHA256 = "b19c9e8acc79de28eaadc8414413f2901d7d8d94ffbfb3b889d1e4a4f5ca68c6"
EXPECTED_FULL_FINGERPRINT = "cc1b350c3332d24d1e6d334cb5eda46c413e880718e6f2ab914612f772ba59c6"
EXPECTED_POLICY_FINGERPRINT = "0989e9bf8e2fed71ee630bc1d2d9da251d7d9cae5dec6c0e502ec54b8bcd1aee"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    expected = OUTPUT_ROOT.resolve()
    if resolved != expected or resolved == BASE.resolve() or not resolved.is_relative_to(BASE.resolve()):
        raise ValueError(f"输出目录必须是本轮专用路径 {expected}，拒绝写入 {resolved}")
    return resolved


def build_arm_configs(source: Mapping[str, Any], seed: int = SEED) -> dict[str, dict[str, Any]]:
    configs = {arm: dict(source) for arm in ARMS}
    for config in configs.values():
        config["seed"] = int(seed)
    configs["R1"]["learning_rate"] = float(source["learning_rate"]) * 0.1
    configs["R4"]["detach_value"] = True
    return configs


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return value
    return value


def _equal(left: Any, right: Any, atol: float = 0.0) -> bool:
    left, right = _to_cpu(left), _to_cpu(right)
    if hasattr(left, "node_types") and hasattr(right, "node_types") and hasattr(left, "edge_types") and hasattr(right, "edge_types"):
        if list(left.node_types) != list(right.node_types) or list(left.edge_types) != list(right.edge_types):
            return False
        for kind in left.node_types:
            left_keys, right_keys = set(left[kind].keys()), set(right[kind].keys())
            if left_keys != right_keys or any(not _equal(left[kind][key], right[kind][key], atol) for key in left_keys):
                return False
        for edge in left.edge_types:
            left_keys, right_keys = set(left[edge].keys()), set(right[edge].keys())
            if left_keys != right_keys or any(not _equal(left[edge][key], right[edge][key], atol) for key in left_keys):
                return False
        return True
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            return False
        if left.is_floating_point() or right.is_floating_point():
            return bool(torch.allclose(left.double(), right.double(), atol=atol, rtol=0.0, equal_nan=False))
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape:
            return False
        if np.issubdtype(left.dtype, np.floating) or np.issubdtype(right.dtype, np.floating):
            return bool(np.allclose(left, right, atol=atol, rtol=0.0, equal_nan=False))
        return bool(np.array_equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_equal(left[key], right[key], atol) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_equal(a, b, atol) for a, b in zip(left, right))
    if isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= atol
    return left == right


def verify_first_rollout(reference: Mapping[str, Any], candidate: Mapping[str, Any], *,
                         arm: str | None = None, compared_to: str | None = None) -> dict[str, Any]:
    required = ("state_feats", "snapshot_facts", "actions", "old_log_probs", "old_values",
                "raw_rewards", "shaped_rewards", "bootstrap_value", "raw_advantages",
                "advantages", "return_targets", "pre_sampling_rng", "pre_update_rng",
                "predicted_minibatch_permutation")
    first_difference = None
    checks: dict[str, bool] = {}
    for key in required:
        left_key = "minibatch_indices" if key == "predicted_minibatch_permutation" and key not in reference else key
        right_key = "minibatch_indices" if key == "predicted_minibatch_permutation" and key not in candidate else key
        if left_key not in reference or right_key not in candidate:
            checks[key] = False
        else:
            checks[key] = _equal(reference[left_key], candidate[right_key], TOLERANCE.get(key, 0.0))
        if not checks[key] and first_difference is None:
            first_difference = key
    for key in ("time_urgencies", "sample_records", "bootstrap_input"):
        if key in reference or key in candidate:
            checks[key] = (key in reference and key in candidate
                           and _equal(reference[key], candidate[key], TOLERANCE.get(key, 0.0)))
            if not checks[key] and first_difference is None:
                first_difference = key
    return {
        "arm": arm,
        "compared_to": compared_to,
        "paired": all(checks.values()),
        "checks": checks,
        "first_difference": first_difference,
        "steps_compared": int(candidate.get("state_feats", torch.empty(0)).shape[0]),
        "tolerances": TOLERANCE,
        "action_fields_exact": checks.get("actions", False),
        "snapshot_masks_and_mapping_exact": checks.get("snapshot_facts", False),
        "rng_and_minibatch_identity_exact": checks.get("pre_sampling_rng", False)
        and checks.get("pre_update_rng", False)
        and checks.get("predicted_minibatch_permutation", False),
    }


def delta_j(reference_j: float, arm_j: float) -> float:
    return float(reference_j) - float(arm_j)


def retention_pass(trajectory: Mapping[str, Any]) -> bool:
    costs = trajectory.get("cost_components_weighted") or {}
    complete = int(trajectory.get("completed_tasks", 0)) == 210
    aircraft = int(trajectory.get("completed_aircraft", 0)) == 7
    no_violations = int(trajectory.get("constraint_violation_count", trajectory.get("violations", 0))) == 0
    j_total = float(trajectory.get("objective_j_total", math.inf))
    j_takt = float(costs.get("j_takt", math.inf))
    postponed = int(trajectory.get("postponed_count", math.inf))
    return (complete and aircraft and no_violations and math.isfinite(j_total) and math.isfinite(j_takt)
            and j_total <= 1e-3 and j_takt <= 1e-6 and postponed == 0)


def historical_interaction_budget(r0_decisions: int, updating_arms: int, per_arm_budget: int) -> int:
    return int(r0_decisions) + int(updating_arms) * int(per_arm_budget)


def _source_files() -> list[tuple[Path, str]]:
    return [
        (BASELINE, "mini210 确定性规则基准"), (MODEL, "价值头校准后的共享起点权重"),
        (CAL_ROOT / "manifest.json", "校准阶段清单"), (CAL_ROOT / "weight_integrity.json", "校准权重完整性记录"),
        (CAL_DIR / "fit_metrics.json", "价值校准拟合指标"), (CAL_DIR / "policy_invariance.json", "策略不变性及指纹"),
        (CAL_DIR / "snapshots_and_mc_targets.pt", "校准样本及MC目标"),
        (PPO_SOURCE / "training_summary.json", "校准后 PPO 实际超参与训练摘要"),
        (PPO_SOURCE / "resolved_config.yaml", "校准后 PPO 有效配置"),
        (PPO_SOURCE / "first_rollout.pt", "历史 PPO 首段事实格式参考"),
        (PPO_SOURCE / "first_rollout_pairing.json", "校准后 PPO 配对记录"),
        (PPO_SOURCE / "trajectories" / "step0000_NOMINAL.json", "校准后 PPO step0 轨迹"),
        (ABLATION_SOURCE / "arm_registry.json", "seed42 臂配置注册表"),
        (ABLATION_SOURCE / "comparison_by_point.csv", "seed42 原始逐点费用"),
        (ABLATION_SOURCE / "r0_frozen_sampling.json", "seed42 R0 实际采样预算"),
        (BC_SOURCE / "manifest.json", "BC 来源与实例清单"),
        (BC_SOURCE / "diagnostic_actor.pt", "BC 模型来源权重"),
        (BC_SOURCE / "teacher_snapshots.pt", "固定 BC 教师状态与动作 probe"),
        (BC_SOURCE / "trajectories" / "C_BC_DIAGNOSTIC_NOMINAL.json", "BC 名义轨迹"),
        (BASE / "run_ppo_ablation_arms.py", "正式 PPO 臂驱动"),
        (BASE / "run_bc_ppo_retention.py", "保持性采样、环境及评估驱动"),
        (BASE / "run_critic_calibration_retention.py", "严格校准副本加载器"),
        (BASE / "c_reproduction_protocol.py", "事实、probe 与指纹合同"),
        (BASE / "verify_trajectory_facts.py", "独立物理轨迹事实核验器"),
        (BASE / "verify_shaping_trace.py", "塑形及边界独立核验器"),
        (ROOT / "training" / "work3_lightning.py", "正式 PPO 更新与损失"),
        (ROOT / "training" / "work3_vector_env.py", "单环境 spawn 实现"),
        (ROOT / "models" / "work3" / "actor_critic.py", "图策略及条件动作实现"),
        (Path(__file__), "本轮复核入口"),
        (ROOT / "tests" / "test_ppo_retention_repeat.py", "本轮新增合同测试"),
    ]


def inventory_sources(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows, missing = [], []
    for path, purpose in _source_files():
        row = {"path": str(path.resolve()), "purpose": purpose, "exists": path.is_file()}
        if path.is_file():
            row.update({"size_bytes": path.stat().st_size, "sha256": sha256(path)})
        else:
            missing.append(str(path.resolve()))
        rows.append(row)
    write_json(out / "dependency_inventory.json", {"files": rows, "missing": missing})
    return rows, missing


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_reference_config() -> dict[str, Any]:
    summary = _read_json(PPO_SOURCE / "training_summary.json")
    config = dict(summary["hyperparameters"])
    required = {
        "method_variant": "C", "gamma": 0.99, "gae_lambda": 0.95, "beta_shaping": 1.0,
        "learning_rate": 3e-4, "clip_epsilon": 0.2, "value_coefficient": 0.5,
        "entropy_coefficient": 0.01, "ppo_epochs": 1, "batch_size": 2, "rollout_steps": 8,
        "num_envs": 1, "max_grad_norm": 0.5, "optimizer": "AdamW", "optimizer_eps": 1e-5,
        "weight_decay": 1e-4, "normalize_advantages": True, "amp_dtype": "fp32",
        "disturbance_enabled": False, "scenario": "NOMINAL", "warmup_mode": "none",
        "total_env_steps": 512, "max_decisions": 512,
    }
    conflicts = {key: {"expected": value, "actual": config.get(key)} for key, value in required.items()
                 if config.get(key) != value}
    if conflicts:
        raise ValueError(f"校准后 PPO 实际来源超参与固定复核合同冲突: {conflicts}")
    return config


def _write_configs(out: Path, source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    configs = build_arm_configs(source)
    changes = {"REF": "无参数变更", "R1": "所有 PPO 参数组 learning_rate 乘 0.1",
               "R4": "仅 critic 输入处 detach 共享特征"}
    for arm, config in configs.items():
        payload = {"seed": SEED, "arm": arm, "source_seed": 42, "source": "calibration/ppo/training_summary.json",
                   "effective_hyperparameters": config, "change": changes[arm]}
        # JSON 是 YAML 1.2 子集，避免依赖额外配置解析包。
        write_json(out / "resolved_configs" / f"{arm}.yaml", payload)
    return configs


def _configure_runner(out: Path) -> Any:
    import work3_4060.run_ppo_ablation_arms as runner

    runner.CAL_ROUND = CAL_ROOT
    runner.OUTPUT_DIR = out
    runner.RUNS_DIR = out / "runs"
    runner.CAL_MODEL = MODEL
    runner.CAL_FIT = CAL_DIR / "fit_metrics.json"
    runner.REFERENCE_PPO = PPO_SOURCE
    runner.REFERENCE_SUMMARY = PPO_SOURCE / "training_summary.json"
    runner.REFERENCE_HISTORY = PPO_SOURCE / "training_history.jsonl"
    runner.REFERENCE_PAIRING = PPO_SOURCE / "first_rollout_pairing.json"
    runner.REFERENCE_FIRST_ROLLOUT = out / "REF" / "first_rollout.pt"
    runner.BASELINE_PATH = BASELINE
    runner.ARMS = ARMS
    runner.POINTS = POINTS
    runner.BUDGET = 512
    runner.RUN_SEED = SEED
    runner.ARM_SPECS = {
        "REF": {"change": "original_ppo_configuration"},
        "R1": {"change": "learning_rate_x0.1", "lr_scale": 0.1},
        "R4": {"change": "value_loss_detached_at_shared_features", "detach_value": True},
    }
    runner.MANIFEST = out / "manifest.json"
    runner.ARM_REGISTRY = out / "arm_registry.json"
    runner.COMPARISON_CSV = out / "comparison_by_point.csv"
    runner.CYCLE_CSV = out / "cycle_comparison.csv"
    runner.DIAGNOSTICS_CSV = out / "update_diagnostics.csv"
    runner.VERIFICATION = out / "verification.json"
    runner.SUMMARY_MD = out / "summary.md"
    for arm in ARMS:
        destination = runner.arm_paths(arm)["dir"].resolve()
        if not destination.is_relative_to(out.resolve()):
            raise ValueError(f"{arm}输出路径逃逸出本轮目录: {destination}")
        if runner.arm_paths(arm)["run_json"].resolve() == (PPO_SOURCE / "runs" / "ppo_run.json").resolve():
            raise ValueError("拒绝将新臂 run.json 写入历史 REF 目录")
    return runner


def _set_determinism() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _run_child(out: Path, stage: str, *args: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(SEED)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", stage,
               "--output-root", str(out), *args]
    log_path = out / "runs" / f"{stage}_{'_'.join(args).replace(':', '_') or 'main'}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(ROOT), env=env, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    return result.returncode, f"{log_path} ({time.perf_counter() - started:.1f}s)"


def _write_stop_record(out: Path, stage_results: Mapping[str, Any], reason: str,
                       protocol: str = "INCOMPLETE") -> None:
    write_json(out / "runs" / "stage_results.json", dict(stage_results))
    write_json(out / "verification.json", {"protocol_status": protocol,
                "production_status": "INCOMPLETE", "retention_status": "INCOMPLETE",
                "stop_reason": reason, "completed_training_arms": [
                    arm for arm in ARMS if (out / arm / "training_summary.json").is_file()
                    and int(_read_json(out / arm / "training_summary.json").get("training_decisions", 0)) == 512],
                "stage_results": stage_results})
    (out / "summary.md").write_text(
        f"# Work-3 seed43 三臂短程保持性复核\n\n协议/生产/保持状态：{protocol} / INCOMPLETE / INCOMPLETE。\n\n停止原因：{reason}。\n",
        encoding="utf-8",
    )


def _worker(stage: str, out: Path, arm: str | None, point: int | None) -> int:
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _set_determinism()
    runner = _configure_runner(out)
    if stage == "train":
        if arm not in ARMS:
            raise ValueError("训练worker必须显式指定 REF、R1 或 R4")
        code = runner.stage_arm(arm)
        summary_path = out / arm / "training_summary.json"
        if summary_path.is_file():
            summary = _read_json(summary_path)
            if int(summary.get("training_decisions", 0)) != 512 or summary.get("stop_reason") != "budget_exhausted":
                return code or 5
        return code
    if stage == "evaluate":
        if arm not in ARMS or point not in POINTS:
            raise ValueError("评估worker参数无效")
        if point == 0:
            checkpoint = MODEL
        else:
            checkpoint = out / arm / "checkpoints" / f"step{point:04d}.pt"
        return runner.stage_evaluate_one(arm, checkpoint, f"step{point:04d}")
    if stage == "probe":
        if arm not in ARMS or point not in POINTS:
            raise ValueError("probe worker参数无效")
        return _probe_checkpoint(out, arm, point)
    if stage == "r4-preflight":
        return _r4_gradient_preflight(out)
    raise ValueError(f"未知worker阶段 {stage}")


def _r4_gradient_preflight(out: Path) -> int:
    import work3_4060.run_critic_calibration_retention as calibration
    import work3_4060.run_ppo_ablation_arms as runner
    import work3_4060.run_bc_ppo_retention as common

    model, _ = calibration.load_calibrated_actor(MODEL)
    model.to("cuda:0")
    model.eval()
    env = common.build_env(seed=SEED)
    report: dict[str, Any] = {"hook_target": "model.critic.forward_pre_hook", "installed_count": 0}
    try:
        env.reset(scenario=None, episode_id=0, episode_index=0)
        snapshot = env.snapshot()
        before = model.select_snapshot(snapshot, deterministic=True)
        handle = runner.apply_detach_value_hook(model)
        report["installed_count"] = len(model.critic._forward_pre_hooks)
        after = model.select_snapshot(snapshot, deterministic=True)
        report["forward_unchanged"] = (before[0] == after[0] and abs(before[1] - after[1]) <= 1e-4
                                        and abs(before[2] - after[2]) <= 1e-4)
        rec = before[3]
        # 正式 PPO 重放返回 [B] 的 value 与 ordered-action log-prob。
        state = snapshot.state_features.unsqueeze(0)  # [D] -> [1, D]
        temporal = snapshot.time_features.unsqueeze(0)  # [T] -> [1, T]
        values, log_probs, _entropy = model.evaluate_action_log_probs(state, temporal, [rec])
        groups = common.parameter_groups(model)["groups"]
        model.zero_grad(set_to_none=True)
        values.sum().backward(retain_graph=True)
        value_shared = common.group_grad_norms(model, {"shared": groups["shared"]})["shared"]
        value_critic = common.group_grad_norms(model, {"critic": groups["value"]})["critic"]
        model.zero_grad(set_to_none=True)
        log_probs.sum().backward()
        policy_shared = common.group_grad_norms(model, {"shared": groups["shared"]})["shared"]
        handle.remove()
        report.update({"value_shared_grad_norm": value_shared, "value_critic_grad_norm": value_critic,
                       "policy_shared_grad_norm": policy_shared,
                       "value_gradient_blocked": value_shared == 0.0,
                       "critic_gradient_present": value_critic > 0.0,
                       "policy_shared_gradient_present": policy_shared > 0.0,
                       "hook_cleared": len(model.critic._forward_pre_hooks) == 0})
    finally:
        env.close()
    report["passed"] = all(report.get(key) for key in ("forward_unchanged", "value_gradient_blocked",
                                                         "critic_gradient_present", "policy_shared_gradient_present",
                                                         "hook_cleared"))
    write_json(out / "r4_gradient_preflight.json", report)
    return 0 if report["passed"] else 4


def _probe_checkpoint(out: Path, arm: str, point: int) -> int:
    from work3_4060.c_reproduction_protocol import (
        aggregate_teacher_metrics, build_sample_record, evaluate_teacher_sample,
        greedy_action, sample_label, sample_snapshot,
    )
    import work3_4060.run_bc_ppo_retention as common
    import work3_4060.run_critic_calibration_retention as calibration

    common.enforce_single_thread_cpu()
    checkpoint = MODEL if point == 0 else out / arm / "checkpoints" / f"step{point:04d}.pt"
    model, _payload = calibration.load_calibrated_actor(checkpoint) if point == 0 else common.load_checkpoint_actor(checkpoint)
    model.to("cpu")
    model.eval()
    bundle = torch.load(BC_SOURCE / "teacher_snapshots.pt", map_location="cpu", weights_only=False)
    samples = bundle.get("samples") or bundle.get("teacher_samples")
    if not samples:
        raise ValueError("固定 BC teacher_snapshots.pt 缺少 samples")
    rows = [evaluate_teacher_sample(model, sample, device="cpu") for sample in samples]
    metrics = aggregate_teacher_metrics(rows)
    ordered_team_hits = []
    # 严格有序自回归 log-prob；不使用 BC 的团队边缘 logsumexp 作为 PPO ratio。
    ordered_logps = []
    for sample in samples:
        snapshot, label = sample_snapshot(sample), sample_label(sample)
        action = sample["teacher_action"]
        greedy, _greedy_record = greedy_action(model, snapshot)
        ordered_team_hits.append(
            tuple(int(w) for w in (greedy or {}).get("team", ()))
            == tuple(int(w) for w in action.get("team", ()))
        )
        record = build_sample_record(snapshot, label, tuple(action.get("team", ())), model.max_station_workers)
        with torch.no_grad():
            _v, logp, _entropy = model.evaluate_action_log_probs(
                snapshot.state_features.unsqueeze(0), snapshot.time_features.unsqueeze(0), [record]
            )
        ordered_logps.append(float(logp.item()))
    target = out / arm / "probe" / f"step{point:04d}.json"
    write_json(target, {"arm": arm, "point": point, "teacher_source": str(BC_SOURCE / "teacher_snapshots.pt"),
                        "teacher_source_sha256": sha256(BC_SOURCE / "teacher_snapshots.pt"),
                        "teacher_snapshot_count": len(samples), "teacher_metrics": metrics,
                        "ordered_teacher_action_log_prob": ordered_logps,
                        "greedy_team_order_accuracy": sum(ordered_team_hits) / len(ordered_team_hits),
                        "exact_joint_action_kl": None,
                        "log_prob_note": "固定教师状态上的有序自回归动作 log-prob；未将无序团队边缘 logsumexp 用作 PPO ratio。",
                        "coverage_note": "固定 BC 教师快照，不覆盖臂自主访问状态；top-1/top-2 间隔未由现有正式接口导出。"})
    return 0


def _write_errata(out: Path, historical_rows: Sequence[Mapping[str, str]], r0: Mapping[str, Any],
                  fit: Mapping[str, Any]) -> None:
    parsed = {(row["arm"], int(row["point"])): float(row["j_total"]) for row in historical_rows}
    lines = ["# seed42 历史结果判读勘误", "", "本文件只补充解释，不改写 seed42 原始日志、权重或轨迹。", "",
             "1. R1 是保持性正结果。降低学习率后参数位移减小是预期机制；“不能靠减速”不是预设验收条件。结果不证明长期保持或恢复学习效率。",
             "2. R2 的三个点均低于 REF。相对改善由原始费用动态计算："]
    for point in (128, 256, 512):
        ref_path = PPO_SOURCE / "trajectories" / f"step{point:04d}_NOMINAL.json"
        ref = float(_read_json(ref_path)["objective_j_total"]) if ref_path.is_file() else parsed.get(("REF", point))
        r2 = parsed.get(("R2", point))
        if ref is not None and r2 is not None:
            relative = (ref - r2) / ref * 100 if abs(ref) > 1e-9 else None
            lines.append(f"   - {point} 步：REF={ref:.17g}，R2={r2:.17g}，改善={ref-r2:.17g}，相对改善="
                         + (f"{relative:.6g}%" if relative is not None else "不计算近零基准百分比") + "。")
    lines += [
        "3. R0 直接冻结随机动作产生的费用，与随机样本影响 PPO 梯度方向是不同问题；其随机轨迹也不能与 greedy 评估构成同口径优势证明。",
        f"4. R0 实际采样 {int(r0['samples'])} 步；旧批总实际交互={historical_interaction_budget(int(r0['samples']), 4, 512)}（R1–R4 各 512），计划 2560 不等于实际；不补跑缺少步数。",
        "5. R3 只说明该改动在该配置下恶化；优势尺度还会改变 policy 相对 value/entropy 尺度及裁剪、优化器作用，不能据此证明方向已被噪声主导。",
        "6. 相近参数 L2 不等于相近策略变化；如果包含 critic，更不代表动作分布变化。R4 是候选，不因单个终点改善而成为唯一根因。",
    ]
    target_stats = fit.get("target_stats", {})
    const = fit.get("constant_baselines", {})
    lines.append(f"7. 校准 MC target 均值/标准差={target_stats.get('mean')}/{target_stats.get('std_unbiased')}；常数预测器训练/留出 MSE={const.get('train', {}).get('mse')}/{const.get('holdout', {}).get('mse')}，校准后误差不可据此宣称 critic 准确；本轮不再校准。")
    lines += [
        "8. 完工轨迹满足 G_shaped=G_raw−βΦ(s0)，前轮 +0.5 是固定平移，不能因回报正负不同称目标冲突；截断保留端点势函数项。",
        "",
    ]
    (out / "interpretation_errata.md").write_text("\n".join(lines), encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _trajectory(out: Path, arm: str, point: int) -> Path:
    if point == 0:
        return out / "shared_step0000_NOMINAL.json"
    return out / arm / "trajectories" / f"step{point:04d}_NOMINAL.json"


def _build_reports(out: Path, source_rows: Sequence[Mapping[str, Any]], initial_git: Mapping[str, Any],
                   runs: Mapping[str, Any]) -> dict[str, Any]:
    import work3_4060.run_bc_ppo_retention as common

    baseline = _read_json(BASELINE)
    verifier = common.load_verifier()
    shaping_verifier = common.load_shaping_verifier()
    trajectory_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for point in POINTS:
            path = _trajectory(out, arm, point)
            if not path.is_file():
                continue
            data = _read_json(path)
            result = verifier.verify_single_trajectory(
                traj_path=path, baseline_h0=float(baseline["metadata"]["h0"]),
                baseline_tasks=baseline["tasks"], scenario_spec={"scenario_id": "NOMINAL"},
            )
            result.update({"arm": arm, "point": point, "path": str(path)})
            verification_rows.append(result)
            aircraft = data.get("execution_facts", {}).get("aircraft_execution", {})
            completed_aircraft = sum(1 for item in aircraft.values()
                                     if len(item.get("exit_times", {})) == int(baseline["metadata"]["num_stations"]))
            costs = data.get("cost_components_weighted", {})
            cycles = data.get("execution_facts", {}).get("cycle_execution_records", {}).values()
            delays = [max(0.0, float(item["transfer_time"]) - float(item["nominal_transfer_time"]))
                      for item in cycles]
            from envs.work3.core_types import ActionBranch
            trace = data.get("action_trace", [])
            advance_count = sum(int(item.get("branch", -1)) == int(ActionBranch.ADVANCE_TO_NEXT_EVENT) for item in trace)
            action_postpone_count = sum(int(item.get("branch", -1)) == int(ActionBranch.POSTPONE) for item in trace)
            trajectory_rows.append({"arm": arm, "point": point, "path": str(path),
                                    "decisions": data.get("decisions"),
                                    "advance_actions": advance_count,
                                    "postpone_actions": action_postpone_count,
                                    "completed_tasks": data.get("completed_tasks"),
                                    "completed_aircraft": completed_aircraft,
                                    "objective_j_total": data.get("objective_j_total"),
                                    "j_takt": costs.get("j_takt"), "d_time": costs.get("d_time"),
                                    "d_team": costs.get("d_team"), "j_postpone": costs.get("j_postpone"),
                                    "j_revision": costs.get("j_revision"),
                                    "makespan": data.get("makespan"),
                                    "maximum_cycle_delay_hours": max(delays, default=0.0),
                                    "cumulative_cycle_delay_hours": sum(delays),
                                    "delayed_cycle_count": sum(delay > 1e-7 for delay in delays),
                                    "postponed_count": data.get("postponed_count"),
                                    "violations": result.get("total_violations"),
                                    "verification_status": result.get("status"),
                                    "retention_pass": retention_pass({**data, "completed_aircraft": completed_aircraft,
                                                                       "constraint_violation_count": result.get("total_violations", 0)})})
    ref_j = {int(row["point"]): float(row["objective_j_total"]) for row in trajectory_rows if row["arm"] == "REF"}
    comparison = []
    for row in trajectory_rows:
        reference = ref_j.get(int(row["point"]))
        comparison.append({**row, "delta_j_ref_minus_arm": None if reference is None else delta_j(reference, float(row["objective_j_total"]))})
    _csv(out / "comparison_by_point.csv", comparison)

    cycle_rows: list[dict[str, Any]] = []
    for row in trajectory_rows:
        facts = _read_json(Path(row["path"]))["execution_facts"]
        for cycle, record in sorted(facts.get("cycle_execution_records", {}).items(), key=lambda pair: int(pair[0])):
            cycle_rows.append({"arm": row["arm"], "point": row["point"], "cycle_id": int(cycle),
                               "nominal_transfer_time": record.get("nominal_transfer_time"),
                               "actual_transfer_time": record.get("transfer_time"),
                               "delay": record.get("transfer_time", 0) - record.get("nominal_transfer_time", 0)})
    _csv(out / "cycle_comparison.csv", cycle_rows)

    teacher_path = BC_SOURCE / "teacher_snapshots.pt"
    probe_rows = []
    probe_payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in ARMS:
        for point in POINTS:
            path = out / arm / "probe" / f"step{point:04d}.json"
            if path.is_file():
                payload = _read_json(path)
                probe_payloads[(arm, point)] = payload
                metrics = payload["teacher_metrics"]
                initial_logps = probe_payloads.get((arm, 0), {}).get("ordered_teacher_action_log_prob", [])
                current_logps = payload.get("ordered_teacher_action_log_prob", [])
                logp_diffs = [abs(float(a) - float(b)) for a, b in zip(initial_logps, current_logps, strict=True)] if initial_logps else []
                probe_rows.append({"arm": arm, "point": point,
                                   "teacher_snapshot_count": payload["teacher_snapshot_count"],
                                   "full_action_accuracy": metrics.get("full_physical_action_accuracy"),
                                   "decision_kind_accuracy": metrics.get("decision_kind_accuracy"),
                                   "task_accuracy_conditional": metrics.get("task_accuracy_conditional"),
                                   "branch_accuracy_conditional": metrics.get("branch_accuracy_conditional"),
                                   "team_accuracy_unordered_conditional": metrics.get("team_accuracy_unordered_conditional"),
                                   "align_accuracy_conditional": metrics.get("align_accuracy_conditional"),
                                   "mean_teacher_nll": metrics.get("nll_mean"),
                                   "greedy_team_order_accuracy": payload.get("greedy_team_order_accuracy"),
                                   "fixed_ordered_action_logp_mean_abs_delta_vs_init": sum(logp_diffs)/len(logp_diffs) if logp_diffs else 0.0,
                                   "fixed_ordered_action_logp_max_abs_delta_vs_init": max(logp_diffs) if logp_diffs else 0.0,
                                   "exact_joint_action_kl": None})
    _csv(out / "policy_probe_metrics.csv", probe_rows)

    parameter_rows = []
    from work3_4060.run_critic_calibration_retention import load_calibrated_actor, subset_fingerprint
    from work3_4060.run_bc_ppo_retention import load_checkpoint_actor, parameter_groups
    initial_model, _ = load_calibrated_actor(MODEL)
    initial_params = dict(initial_model.named_parameters())
    groups = parameter_groups(initial_model)["groups"]
    subsets = {"full": list(initial_params), "policy": groups["shared"] + groups["policy"],
               "critic": groups["value"]}
    for arm in ARMS:
        for point in POINTS:
            checkpoint_path = MODEL if point == 0 else out / arm / "checkpoints" / f"step{point:04d}.pt"
            if not checkpoint_path.is_file():
                continue
            current_model, checkpoint_payload = (load_calibrated_actor(MODEL) if point == 0
                                                  else load_checkpoint_actor(checkpoint_path))
            current_params = dict(current_model.named_parameters())
            current_state = current_model.state_dict()
            row: dict[str, Any] = {"arm": arm, "point": point,
                                   "actor_fingerprint": checkpoint_payload["actor_fingerprint"],
                                   "policy_fingerprint": subset_fingerprint(current_state, groups["shared"] + groups["policy"]),
                                   "critic_fingerprint": subset_fingerprint(current_state, groups["value"]),
                                   "checkpoint_sha256": sha256(checkpoint_path)}
            for group, names in subsets.items():
                # 参数位移对比同一校准初始化；policy 含共享编码器，critic 是专属价值头。
                delta2 = sum(float((current_params[name].detach().double().cpu() -
                                    initial_params[name].detach().double().cpu()).square().sum().item())
                             for name in names)
                initial2 = sum(float(initial_params[name].detach().double().cpu().square().sum().item()) for name in names)
                row[f"{group}_parameter_l2_from_init"] = math.sqrt(delta2)
                row[f"{group}_parameter_relative_l2_from_init"] = math.sqrt(delta2) / math.sqrt(initial2) if initial2 else None
            parameter_rows.append(row)
    _csv(out / "parameter_change_metrics.csv", parameter_rows)

    # 按 task_key 与共同起点配对，分别记录工序开工、团队集合/顺序和站位周期差异。
    shared = _read_json(out / "shared_step0000_NOMINAL.json")["execution_facts"]
    task_rows: list[dict[str, Any]] = []
    for row in trajectory_rows:
        current = _read_json(Path(row["path"]))["execution_facts"]
        ref_tasks, now_tasks = shared["tasks_execution"], current["tasks_execution"]
        for key in sorted(set(ref_tasks) | set(now_tasks)):
            ref, now = ref_tasks.get(key, {}), now_tasks.get(key, {})
            ref_team, now_team = [int(v) for v in ref.get("team", [])], [int(v) for v in now.get("team", [])]
            left, right = ref.get("actual_start"), now.get("actual_start")
            task_rows.append({"arm": row["arm"], "point": row["point"], "task_key": key,
                              "cycle_start_time": now.get("cycle_start_time"),
                              "actual_start_ref": left, "actual_start_arm": right,
                              "actual_start_delta_hours": None if left is None or right is None else float(right)-float(left),
                              "team_order_ref": json.dumps(ref_team), "team_order_arm": json.dumps(now_team),
                              "team_order_changed": ref_team != now_team,
                              "team_set_changed": sorted(ref_team) != sorted(now_team),
                              "physical_schedule_changed": (left is None) != (right is None) or
                                  (left is not None and right is not None and abs(float(left)-float(right)) > 1e-7) or
                                  sorted(ref_team) != sorted(now_team)})
    _csv(out / "trajectory_comparison.csv", task_rows)
    action_rows = []
    shared_trace = _read_json(out / "shared_step0000_NOMINAL.json").get("action_trace", [])
    for row in trajectory_rows:
        current_trace = _read_json(Path(row["path"])).get("action_trace", [])
        left = [(item.get("branch"), item.get("task_key")) for item in shared_trace]
        right = [(item.get("branch"), item.get("task_key")) for item in current_trace]
        first = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None)
        if first is None and len(left) != len(right):
            first = min(len(left), len(right))
        action_rows.append({"arm": row["arm"], "point": row["point"],
                            "base_action_count": len(left), "arm_action_count": len(right),
                            "branch_task_sequence_equal": left == right,
                            "first_branch_task_difference_index": first,
                            "team_order_difference_count": sum(
                                1 for a, b in zip(shared_trace, current_trace)
                                if a.get("task_key") == b.get("task_key") and a.get("team_order") != b.get("team_order")),
                            "team_order_note": "对齐到相同动作序列位置的 task_key 后比较；与 team 集合/排程差异分列。"})
    _csv(out / "action_order_comparison.csv", action_rows)

    cross_seed_rows = []
    historical = {(row["arm"], int(row["point"])): float(row["j_total"])
                  for row in csv.DictReader((ABLATION_SOURCE / "comparison_by_point.csv").open(encoding="utf-8-sig"))}
    historical["REF", 0] = float(_read_json(PPO_SOURCE / "trajectories" / "step0000_NOMINAL.json")["objective_j_total"])
    for point in POINTS:
        path = PPO_SOURCE / "trajectories" / f"step{point:04d}_NOMINAL.json"
        if path.is_file():
            historical["REF", point] = float(_read_json(path)["objective_j_total"])
    for seed_value, values in ((42, historical), (SEED, {(row["arm"], int(row["point"])): float(row["objective_j_total"])
                                                         for row in trajectory_rows})):
        reference_values = {point: value for (arm, point), value in values.items() if arm == "REF"}
        for arm in ARMS:
            for point in POINTS:
                value = values.get((arm, point))
                ref_value = reference_values.get(point)
                cross_seed_rows.append({"ppo_sampling_seed": seed_value, "source_initialization_seed": 42,
                                        "arm": arm, "point": point, "j_total": value,
                                        "within_seed_delta_j_ref_minus_arm": None if value is None or ref_value is None else delta_j(ref_value, value),
                                        "independent_model_seed": False})
    _csv(out / "cross_seed_summary.csv", cross_seed_rows)

    paired = {}
    for arm in ARMS:
        path = out / arm / "first_rollout_pairing.json"
        paired[arm] = _read_json(path) if path.is_file() else {"paired": False, "reason": "首段产物缺失"}

    optimizer_checks: dict[str, dict[str, Any]] = {}
    expected_lrs = {arm: float(_load_reference_config()["learning_rate"]) * (0.1 if arm == "R1" else 1.0)
                    for arm in ARMS}
    for arm in ARMS:
        summary_path = out / arm / "training_summary.json"
        history_path = out / arm / "training_history.jsonl"
        if not summary_path.is_file() or not history_path.is_file():
            optimizer_checks[arm] = {"complete": False, "passed": False}
            continue
        training_summary = _read_json(summary_path)
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        observed = [float(lr) for segment in history for update in segment.get("update_records", [])
                    for lr in update.get("learning_rates", [])]
        optimizer_checks[arm] = {
            "complete": True, "optimizer": training_summary.get("optimizer_name"),
            "expected_lr": expected_lrs[arm], "observed_lr_min": min(observed, default=None),
            "observed_lr_max": max(observed, default=None),
            "optimizer_steps": training_summary.get("optimizer_steps"),
            "scheduler_count": len(training_summary.get("optimizer_schedulers", [])),
            "passed": training_summary.get("optimizer_name") == "AdamW"
                and bool(observed) and all(abs(lr - expected_lrs[arm]) <= 1e-15 for lr in observed)
                and len(training_summary.get("optimizer_schedulers", [])) == 0,
        }

    shaping_status = {}
    for arm in ARMS:
        events, run_json = out / arm / "shaping_events.jsonl", out / arm / "runs" / "ppo_run.json"
        audit_path = out / arm / "shaping_audit_summary.json"
        if events.is_file() and run_json.is_file():
            audit = shaping_verifier.verify_shaping_trace(events, run_json, audit_path, tolerance=1e-9)
            shaping_status[arm] = audit.get("overall_status", "INCOMPLETE")
        else:
            shaping_status[arm] = "INCOMPLETE"
    expected = len(ARMS) * len(POINTS)
    actual = len(verification_rows)
    initial_fingerprints = {
        arm: _read_json(out / arm / "training_summary.json").get("initial_fingerprints")
        for arm in ARMS if (out / arm / "training_summary.json").is_file()
    }
    initial_fingerprints_match = len(initial_fingerprints) == len(ARMS) and len(
        {json.dumps(value, sort_keys=True) for value in initial_fingerprints.values()}) == 1
    r4_preflight = _read_json(out / "r4_gradient_preflight.json") if (out / "r4_gradient_preflight.json").is_file() else {}
    fail = any(row.get("status") == "FAIL" for row in verification_rows) or any(
        item.get("paired") is False for item in paired.values()) or any(value == "FAIL" for value in shaping_status.values())
    fail = fail or any(check.get("passed") is False and check.get("complete") for check in optimizer_checks.values())
    fail = fail or (bool(initial_fingerprints) and not initial_fingerprints_match)
    fail = fail or (bool(r4_preflight) and not bool(r4_preflight.get("passed")))
    protocol_status = "FAIL" if fail else ("PASS" if actual == expected and all(item.get("paired") for item in paired.values())
                                            and all(value == "PASS" for value in shaping_status.values())
                                            and all(check.get("passed") for check in optimizer_checks.values())
                                            and initial_fingerprints_match and bool(r4_preflight.get("passed")) else "INCOMPLETE")
    all_production = actual == expected and all(row.get("status") == "PASS" for row in verification_rows)
    production_status = "FAIL" if fail else ("PASS" if all_production else "INCOMPLETE")
    retention = {arm: {int(row["point"]): bool(row["retention_pass"]) for row in trajectory_rows if row["arm"] == arm}
                 for arm in ARMS}
    final_rows = {arm: next((row for row in trajectory_rows if row["arm"] == arm and int(row["point"]) == 512), None)
                  for arm in ARMS}
    final_incomplete = any(row is None or row.get("completed_tasks") != 210 or row.get("completed_aircraft") != 7
                           for row in final_rows.values())
    retention_status = ("INCOMPLETE" if final_incomplete else
                        "FAIL" if any(not result.get(512, False) for result in retention.values()) else "PASS")
    evaluation_complete = all(
        bool(_read_json(_trajectory(out, arm, point)).get("success"))
        and bool(_read_json(_trajectory(out, arm, point)).get("feasible"))
        for arm in ARMS for point in POINTS if _trajectory(out, arm, point).is_file()
    )
    production_status = "FAIL" if fail else ("PASS" if all_production and evaluation_complete else "INCOMPLETE")
    summary = {
        "protocol_status": protocol_status, "production_status": production_status,
        "retention_status": retention_status, "expected_trajectory_count": expected,
        "verified_trajectory_count": actual, "pairing": paired, "shaping_status": shaping_status,
        "optimizer_config_checks": optimizer_checks,
        "initial_fingerprints_by_arm": initial_fingerprints,
        "initial_fingerprints_match": initial_fingerprints_match,
        "r4_gradient_preflight": r4_preflight,
        "retention_by_arm_point": retention, "trajectory_verification": verification_rows,
        "source_hashes_unchanged": {row["path"]: sha256(Path(row["path"])) == row.get("sha256")
                                    for row in source_rows if row.get("exists")},
        "initial_git": initial_git, "stage_results": runs,
    }
    write_json(out / "trajectory_verification.json", {"rows": verification_rows,
                                                        "physical_time_tolerance_hours": 1e-7,
                                                        "ledger_tolerance": 1e-9})
    write_json(out / "verification.json", summary)
    _write_summary(out, trajectory_rows, comparison, summary)
    return summary


def _write_summary(out: Path, rows: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]],
                   verification: Mapping[str, Any]) -> None:
    by_arm = {arm: sorted((row for row in rows if row["arm"] == arm), key=lambda item: int(item["point"])) for arm in ARMS}
    lines = ["# Work-3 seed43 三臂短程保持性复核", "",
             f"- 实际来源：校准/BC seed42；本轮 PPO 采样 seed={SEED}。R1 因 seed42 结果被选作主要复核对象，因此 seed43 是后续复核，不是完全盲测。",
             f"- 协议 / 生产 / 保持状态：{verification['protocol_status']} / {verification['production_status']} / {verification['retention_status']}。",
             "- REF 为新的 seed43 训练对照；三臂各自从同一校准权重重载，实际训练预算以各臂 summary 为准。", "",
             "## 逐点费用与配对差值", "", "ΔJ=J_REF−J_arm，正值表示该臂费用较低。", "",
             "|臂|步数|决策数|ADVANCE|J_total|J_takt|D_time|D_team|J_postpone|J_revision|最大周期延迟(h)|累计延迟(h)|完工飞机|后移|保持|ΔJ|"]
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    delta_map = {(row["arm"], int(row["point"])): row.get("delta_j_ref_minus_arm") for row in paired}
    for row in sorted(rows, key=lambda item: (ARMS.index(item["arm"]), int(item["point"]))):
        fmt = lambda value: "—" if value is None else f"{float(value):.10g}"
        lines.append("|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|".format(
            row["arm"], row["point"], row.get("decisions"), row.get("advance_actions"), fmt(row.get("objective_j_total")),
            fmt(row.get("j_takt")), fmt(row.get("d_time")), fmt(row.get("d_team")),
            fmt(row.get("j_postpone")), fmt(row.get("j_revision")), fmt(row.get("maximum_cycle_delay_hours")),
            fmt(row.get("cumulative_cycle_delay_hours")), row.get("completed_aircraft"),
            row.get("postponed_count"), row.get("retention_pass"),
            fmt(delta_map.get((row["arm"], int(row["point"]))))))
    lines += ["", "## 首段配对与训练事实", ""]
    for arm in ARMS:
        result = verification["pairing"].get(arm, {})
        lines.append(f"- {arm} 首段配对：{result.get('paired')}，首个差异：{result.get('first_difference')}；包含状态/掩码/动作、old V/log-prob、回报/GAE、RNG 与 mini-batch 身份核对。")
    for arm in ARMS:
        path = out / arm / "training_summary.json"
        if path.is_file():
            item = _read_json(path)
            gpu = item.get("gpu", {})
            lines.append(f"- {arm}：交互 {item.get('training_decisions')}，optimizer.step {item.get('optimizer_steps')}，段 {item.get('segments')}，停止 {item.get('stop_reason')}，耗时 {item.get('elapsed_seconds')}s，峰值 allocated/reserved {gpu.get('peak_allocated_mib')}/{gpu.get('peak_reserved_mib')} MiB。")
    r4_path = out / "r4_gradient_preflight.json"
    if r4_path.is_file():
        r4 = _read_json(r4_path)
        lines.append(f"- R4 梯度路径预检：前向不变={r4.get('forward_unchanged')}，共享梯度阻断={r4.get('value_gradient_blocked')}，critic 专属梯度={r4.get('critic_gradient_present')}，策略共享梯度={r4.get('policy_shared_gradient_present')}，hook 清理={r4.get('hook_cleared')}。")
    lines += ["", "## 固定判读", "",
              "- R1 按 512 步终点保持结果判定；较小的参数位移或 KL 是较保守更新的功能证据，不是失败条件，也不证明长期有效或学习效率恢复。",
              "- REF seed43 是否退化及与 seed42 的方向，直接依据本报告的真实费用点和历史逐点 CSV；不同 seed 只在各自 REF 内配对，不做显著性或跨 seed 均值推断。",
              "- R4 只作探索性复核：512 方向须连同中间点、hook 梯度路径与保留阈值报告；不能从一个终点推断普遍因果。",
              "- 参数 L2、教师状态有序动作 log-prob 与 greedy/条件动作指标分开；未计算精确联合 KL 时明确留空，不用团队边缘 logsumexp 代替 PPO ratio。",
              "- 观察结果限于 mini210、Nominal-only 与这一个固定校准初始化；未区分的候选解释包括全局学习率同时改变 AdamW 衰减步幅、R4 梯度裁剪交互及自访问状态分布变化。",
              "- 后续至多建议先按 seed43 的配对结果决定是否对 R1 做更长但预先固定预算的保持复核；本轮不自动追加实验。", "",
              "## 产物索引", "",
              "`manifest.json`、`arm_registry.json`、`dependency_inventory.json`、`source_binding.json`、`resolved_configs/`、`interpretation_errata.md`、`pairing_verification.json`、`shared_step0000_NOMINAL.json`、三臂运行目录、`comparison_by_point.csv`、`cycle_comparison.csv`、`trajectory_comparison.csv`、`action_order_comparison.csv`、`policy_probe_metrics.csv`、`parameter_change_metrics.csv`、`cross_seed_summary.csv`、`trajectory_verification.json`、`weight_integrity.json`、`verification.json`、`summary.md`、`runs/discarded_runs.json`（如有）。", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(out_arg: Path) -> int:
    out = resolve_output_root(out_arg)
    if out.exists() and any(out.iterdir()):
        entries = list(out.iterdir())
        runs = out / "runs"
        allowed = (len(entries) == 1 and entries[0] == runs and runs.is_dir()
                   and (runs / "discarded_runs.json").is_file()
                   and all(child.name == "discarded_runs.json" or child.name.startswith("discarded_attempt_")
                           for child in runs.iterdir()))
        if not allowed:
            raise FileExistsError(f"输出目录含非归档文件，保留现有内容并拒绝覆盖：{out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    rows, missing = inventory_sources(out)
    if missing:
        write_json(out / "verification.json", {"protocol_status": "INCOMPLETE", "stop_reason": "missing_dependencies", "missing": missing})
        return 3
    if sha256(MODEL) != EXPECTED_MODEL_SHA256:
        raise ValueError("校准模型 SHA256 与授权输入标识不符，停止训练")
    calibration_manifest = _read_json(CAL_ROOT / "manifest.json")
    bc_manifest = _read_json(BC_SOURCE / "manifest.json")
    if calibration_manifest["instance"]["baseline_sha256"] != sha256(BASELINE):
        raise ValueError("校准 manifest 的 baseline SHA256 与当前实例不符")
    if bc_manifest["instance"]["baseline_sha256"] != sha256(BASELINE):
        raise ValueError("BC manifest 的 baseline SHA256 与当前实例不符")
    reference = _load_reference_config()
    configs = _write_configs(out, reference)
    _write_errata(out, list(csv.DictReader((ABLATION_SOURCE / "comparison_by_point.csv").open(encoding="utf-8-sig"))),
                  _read_json(ABLATION_SOURCE / "r0_frozen_sampling.json"), _read_json(CAL_DIR / "fit_metrics.json"))
    source_hashes = {row["path"]: row["sha256"] for row in rows}
    registry = _read_json(ABLATION_SOURCE / "arm_registry.json")
    cal_integrity = _read_json(CAL_ROOT / "weight_integrity.json")
    if registry["shared_start"]["calibrated_model_sha256"] != EXPECTED_MODEL_SHA256:
        raise ValueError("seed42 registry 指向的校准副本哈希不同")
    if registry["shared_start"]["calibrated_actor_fingerprint"] != EXPECTED_FULL_FINGERPRINT:
        raise ValueError("全模型指纹与 registry 不匹配")
    if cal_integrity["calibrated"]["policy_subset_fingerprint"] != EXPECTED_POLICY_FINGERPRINT:
        raise ValueError("策略子集指纹与校准 integrity 不匹配")
    if _read_json(CAL_DIR / "policy_invariance.json")["full_model_fingerprint_after"] != EXPECTED_FULL_FINGERPRINT:
        raise ValueError("校准模型全指纹与 policy invariance 清单不匹配")
    import work3_4060.run_critic_calibration_retention as calibration
    model, payload = calibration.load_calibrated_actor(MODEL)
    if payload.get("actor_fingerprint") != EXPECTED_FULL_FINGERPRINT:
        raise ValueError("strict calibration loader 返回的模型指纹不符")
    import work3_4060.run_bc_ppo_retention as common
    groups = common.parameter_groups(model)
    state = model.state_dict()
    policy_fp = calibration.subset_fingerprint(state, calibration.policy_names(model))
    if policy_fp != EXPECTED_POLICY_FINGERPRINT:
        raise ValueError(f"策略子集重算指纹不符: {policy_fp}")
    git = common.git_state()
    write_json(out / "source_binding.json", {"initial_git": git, "ppo_sampling_seed": SEED,
                "bc_and_calibration_source_seed": 42, "calibrated_model": {"path": str(MODEL.resolve()),
                "sha256": sha256(MODEL), "full_model_fingerprint": payload["actor_fingerprint"],
                "policy_subset_fingerprint": policy_fp}, "sources": source_hashes,
                "source_hyperparameters": reference, "configs": configs})
    write_json(out / "manifest.json", {"schema_version": "work3_ppo_retention_repeat_seed43_v1",
                "ppo_sampling_seed": SEED, "bc_source_seed": 42, "calibration_source_seed": 42,
                "source_model_sha256": sha256(MODEL), "source_full_fingerprint": payload["actor_fingerprint"],
                "arms": list(ARMS), "points": list(POINTS), "expected_evaluations": 13,
                "expected_training_interactions_per_arm": 512, "expected_total_training_interactions": 1536,
                "prior_discarded_runs": "runs/discarded_runs.json" if (out / "runs" / "discarded_runs.json").is_file() else None,
                "serial": True, "precision": "fp32", "vector_environment": {"count": 1, "start_method": "spawn"},
                "scenario": "NOMINAL", "disturbance": "none", "warmup": "none",
                "shared_step0000": "shared_step0000_NOMINAL.json", "evaluation_decision_limit": 4096,
                "evaluation_matrix": [{"arm": arm, "point": point,
                                       "trajectory_path": "shared_step0000_NOMINAL.json" if point == 0 else
                                           f"{arm}/trajectories/step{point:04d}_NOMINAL.json",
                                       "evaluation_id": "shared_step0000" if point == 0 else f"{arm}_step{point:04d}",
                                       "shared_trajectory": point == 0}
                                      for arm in ARMS for point in POINTS],
                "checks": ["protocol_status", "production_status", "retention_status"]})
    write_json(out / "arm_registry.json", {
        "schema_version": "work3_ppo_retention_repeat_arms_v1",
        "source_registry": str((ABLATION_SOURCE / "arm_registry.json").resolve()),
        "source_registry_sha256": sha256(ABLATION_SOURCE / "arm_registry.json"),
        "shared_start": {"path": str(MODEL.resolve()), "sha256": sha256(MODEL),
                         "full_model_fingerprint": payload["actor_fingerprint"],
                         "policy_subset_fingerprint": policy_fp, "source_seed": 42},
        "ppo_sampling_seed": SEED, "arms": configs,
        "points": list(POINTS), "evaluation_binding": {"step0000": "one shared CPU Nominal trajectory",
                                                         "other_points": "one independent CPU process per arm and point"},
        "actual_seed42_r0_budget": historical_interaction_budget(
            int(_read_json(ABLATION_SOURCE / "r0_frozen_sampling.json")["samples"]), 4, 512),
    })
    (out / "REF").mkdir()
    # shared step0000 首先独立运行；未复现 BC/A 名义排程则不启动训练。
    code, log = _run_child(out, "evaluate", "REF", "--point", "0")
    stage_results: dict[str, Any] = {"shared_step0000": {"exit_code": code, "log": log}}
    write_json(out / "runs" / "stage_results.json", stage_results)
    if code != 0:
        _write_stop_record(out, stage_results, "shared_step0000_evaluation_failed", "FAIL")
        return code
    shared_path = out / "shared_step0000_NOMINAL.json"
    shared = _read_json(shared_path)
    bc_nominal = _read_json(BC_SOURCE / "trajectories" / "C_BC_DIAGNOSTIC_NOMINAL.json")
    from work3_4060.c_reproduction_protocol import compare_execution_facts
    schedule_check = compare_execution_facts(shared["execution_facts"], bc_nominal["execution_facts"], tolerance=1e-7)
    initial_ok = (bool(shared.get("success")) and bool(shared.get("feasible"))
                  and int(shared.get("completed_tasks", 0)) == 210
                  and abs(float(shared.get("objective_j_total", math.inf))) <= 1e-9
                  and not schedule_check.get("start_diff") and not schedule_check.get("team_diff")
                  and int(schedule_check.get("end_deviation_count", 0)) == 0
                  and not schedule_check.get("transfer_diff"))
    write_json(out / "shared_step0000_verification.json", {"matches_bc_nominal": initial_ok,
                "objective_j_total": shared.get("objective_j_total"), "comparison": schedule_check,
                "trajectory_sha256": sha256(shared_path)})
    if not initial_ok:
        _write_stop_record(out, stage_results, "step0000_did_not_reproduce_bc_nominal", "FAIL")
        return 4
    for arm in ARMS:
        code, log = _run_child(out, "probe", arm, "--point", "0")
        stage_results[f"probe_{arm}_0"] = {"exit_code": code, "log": log}
        if code:
            _write_stop_record(out, stage_results, f"initial_probe_failed:{arm}", "FAIL")
            return code
    code, log = _run_child(out, "r4-preflight")
    stage_results["r4_gradient_preflight"] = {"exit_code": code, "log": log}
    write_json(out / "runs" / "stage_results.json", stage_results)
    if code:
        _write_stop_record(out, stage_results, "r4_gradient_preflight_failed", "FAIL")
        return code
    completed_training = []
    for arm in ARMS:
        code, log = _run_child(out, "train", arm)
        stage_results[f"train_{arm}"] = {"exit_code": code, "log": log}
        write_json(out / "runs" / "stage_results.json", stage_results)
        if code:
            break
        completed_training.append(arm)
    for arm in completed_training:
        for point in POINTS[1:]:
            for stage in ("evaluate", "probe"):
                code, log = _run_child(out, stage, arm, "--point", str(point))
                stage_results[f"{stage}_{arm}_{point}"] = {"exit_code": code, "log": log}
                write_json(out / "runs" / "stage_results.json", stage_results)
                if code:
                    break
            if code:
                break
        if code:
            break
    write_json(out / "runs" / "stage_results.json", stage_results)
    write_json(out / "pairing_verification.json", {arm: _read_json(out / arm / "first_rollout_pairing.json")
                if (out / arm / "first_rollout_pairing.json").is_file() else {"paired": False, "missing": True}
                for arm in ARMS})
    summary = _build_reports(out, rows, git, stage_results)
    source_end = {row["path"]: sha256(Path(row["path"])) == row["sha256"] for row in rows}
    parameter_rows = list(csv.DictReader((out / "parameter_change_metrics.csv").open(encoding="utf-8-sig")))
    initial_by_arm = {arm: _read_json(out / arm / "training_summary.json").get("initial_fingerprints")
                      for arm in ARMS if (out / arm / "training_summary.json").is_file()}
    write_json(out / "weight_integrity.json", {
        "source_model_sha256": sha256(MODEL),
        "source_model_unchanged": sha256(MODEL) == EXPECTED_MODEL_SHA256,
        "source_files_unchanged": source_end,
        "initial_fingerprints_by_arm": initial_by_arm,
        "initial_fingerprints_match": len(initial_by_arm) == len(ARMS) and
            len({json.dumps(value, sort_keys=True) for value in initial_by_arm.values()}) == 1,
        "checkpoint_fingerprints": parameter_rows,
    })
    return 0 if summary["protocol_status"] == "PASS" else 5


def main() -> int:
    parser = argparse.ArgumentParser(description="Work-3 seed43 REF/R1/R4 保持性复核")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--worker", choices=["train", "evaluate", "probe", "r4-preflight"])
    parser.add_argument("--point", type=int)
    parser.add_argument("arm", nargs="?")
    args = parser.parse_args()
    if Path(sys.prefix).name.casefold() != "rag_env":
        raise RuntimeError(f"本任务只允许在 rag_env 执行，当前 Python 环境为 {sys.prefix}")
    try:
        if args.worker:
            return _worker(args.worker, resolve_output_root(args.output_root), args.arm, args.point)
        return run_all(args.output_root)
    except Exception as error:
        out = resolve_output_root(args.output_root)
        if (out / "manifest.json").is_file() and (out / "runs").is_dir():
            write_json(out / "runs" / "fatal_error.json", {"type": type(error).__name__, "message": str(error)})
            write_json(out / "verification.json", {"protocol_status": "FAIL", "production_status": "INCOMPLETE",
                        "retention_status": "INCOMPLETE", "stop_reason": type(error).__name__,
                        "message": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
