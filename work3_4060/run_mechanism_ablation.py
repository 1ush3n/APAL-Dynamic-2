"""Work-3 机制消融实验驱动脚本 (E/F Ablation)。

实验设计与目的：
区分时间辅助损失 (Auxiliary Head)、修正时间观测输入 (Corrected Time Input) 与学习修正势函数引导 (Learned Potential Shaping vs Heuristic Potential Shaping) 各自的因果贡献：
- Method C: 无时间辅助学习、无修正时间输入、使用启发式势函数奖励塑形 (Φ_heur)
- Method E: 有时间辅助学习、无修正时间输入、使用启发式势函数奖励塑形 (Φ_heur)
- Method F: 有时间辅助学习、有修正时间输入、使用启发式势函数奖励塑形 (Φ_heur)
- Method D: 有时间辅助学习、有修正时间输入、使用学习修正势函数奖励塑形 (Φ_learned)

注：C、E、F 均包含启发式势函数奖励塑形；F→D 的干预是切换势函数的时间估计来源（启发式 vs 学习修正头），而非从“无势函数”到“有势函数”。
势函数数学定义为：Φ(s) = -a * (H^_q / H_0) - b * ([H^_q - H_0]_+ / H_0), a=0.5, b=1.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import ActorCriticWork3
from scripts.work3.experiment_protocol import build_method_profile
from scripts.work3.evaluate_c_vs_d import (
    build_formal_evaluation_agent,
    evaluate_single_trajectory,
    _validate_formal_checkpoint_contract,
)
from work3_4060.run_takeover_diagnostic import run_all_takeover_diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MechanismAblation")


def train_ablation_model(
    method: str,
    seed: int = 42,
    device: str = "cuda:0",
    max_decisions: int = 2048,
) -> tuple[Path, Path]:
    out_dir = REPO_ROOT / f"work3_4060/mechanism_v1/seed{seed}/{method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.pt"
    run_json_path = out_dir / "run.json"
    log_path = out_dir / "console.log"

    logger.info(f"=== 开始训练消融模型 Method-{method} (Seed {seed}) ===")
    cmd = [
        sys.executable,
        "scripts/work3/train_ppo_work3.py",
        "--config", "work3_4060/configs/pilot_d.yaml",
        "--method", method,
        "--seed", str(seed),
        "--device", device,
        "--max-decisions", str(max_decisions),
        "--successful-batch-target", "1000000",
        "--max-wall-seconds", "1800",
        "--scenarios", "work3_4060/compare_v1/scenarios.json",
        "--scenario-split", "work3_4060/compare_v1/train.json",
        "--output", str(weights_path),
        "--report", str(run_json_path),
    ]

    with log_path.open("w", encoding="utf-8") as f_log:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=f_log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if proc.returncode != 0:
        logger.error(f"Method-{method} 训练失败 (exit code {proc.returncode})！请查看日志: {log_path}")
        raise RuntimeError(f"Method-{method} 训练失败")

    logger.info(f"Method-{method} 训练完成！权重保存至: {weights_path}")
    return weights_path, run_json_path


def verify_contract(method: str, weights_path: Path, device: str = "cuda:0") -> None:
    logger.info(f"验证 Method-{method} 检查点契约...")
    profile = build_method_profile(method)
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64).to(device)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    _validate_formal_checkpoint_contract(profile, ckpt, actor)
    logger.info(f"Method-{method} 检查点契约严格校验通过！")


def evaluate_validation_scenarios(
    seed: int = 42,
    device: str = "cuda:0",
) -> Path:
    logger.info("=== 开始在 4 个验证集场景及 Nominal 上评估 C, E, F, D ===")
    baseline_path = str(REPO_ROOT / "work3_4060/data/mini210.json")

    with (REPO_ROOT / "work3_4060/compare_v1/scenarios.json").open("r", encoding="utf-8") as f:
        scenarios_by_id = {s["scenario_id"]: s for s in json.load(f)}

    with (REPO_ROOT / "work3_4060/compare_v1/validation.json").open("r", encoding="utf-8") as f:
        val_scenario_ids = json.load(f)

    # 包含 Nominal 与 4 个验证扰动场景
    all_eval_scenarios = [None] + [scenarios_by_id[sc_id] for sc_id in val_scenario_ids]

    method_ckpts = {
        "C": REPO_ROOT / f"work3_4060/compare_v1/seed{seed}/C/weights.pt",
        "E": REPO_ROOT / f"work3_4060/mechanism_v1/seed{seed}/E/weights.pt",
        "F": REPO_ROOT / f"work3_4060/mechanism_v1/seed{seed}/F/weights.pt",
        "D": REPO_ROOT / f"work3_4060/compare_v1/seed{seed}/D/weights.pt",
    }

    eval_results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for sc in all_eval_scenarios:
        sc_id = sc["scenario_id"] if sc is not None else "NOMINAL"
        logger.info(f"--- 评估场景 {sc_id} ---")

        row = {
            "seed": seed,
            "scenario_id": sc_id,
            "timing": sc.get("timing", "NONE") if sc else "NONE",
            "intensity": sc.get("intensity", "NONE") if sc else "NONE",
            "station": sc.get("station_id", sc.get("station", "NONE")) if sc else "NONE",
        }

        for method_name, ckpt_path in method_ckpts.items():
            agent = build_formal_evaluation_agent(method_name, str(ckpt_path), device=device)
            env = AirLineEnvWork3(baseline_json_path=baseline_path)
            res = evaluate_single_trajectory(
                env=env,
                agent_type=f"Method-{method_name}",
                agent=agent,
                scenario=sc,
                device=device,
                max_decisions=4096,
                warmup_mode="none",
            )

            m_prefix = method_name.lower()
            row[f"{m_prefix}_feasible"] = res["feasible"]
            row[f"{m_prefix}_makespan"] = round(float(res["makespan"]), 2)
            row[f"{m_prefix}_j_total"] = f"{float(res['j_total']):.6e}"
            row[f"{m_prefix}_j_takt"] = f"{float(res['j_takt']):.6e}"
            row[f"{m_prefix}_d_time"] = f"{float(res['d_time']):.6e}"
            row[f"{m_prefix}_d_team"] = f"{float(res['d_team']):.6e}"
            row[f"{m_prefix}_j_postpone"] = f"{float(res['j_postpone']):.6e}"
            row[f"{m_prefix}_j_revision"] = f"{float(res['j_revision']):.6e}"
            row[f"{m_prefix}_postponed_count"] = res["postponed_count"]
            row[f"{m_prefix}_decisions"] = res["decisions"]

            eval_results.append({
                "scenario_id": sc_id,
                "method": method_name,
                "result": res,
            })

        csv_rows.append(row)

    out_dir = REPO_ROOT / f"work3_4060/mechanism_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "validation_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    json_path = out_dir / "validation_results.json"
    json_path.write_text(json.dumps(eval_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    logger.info(f"验证集评估完成，结果已保存至: {csv_path}")
    return csv_path


def run_mechanism_takeover_diagnostics(
    seed: int = 42,
    device: str = "cuda:0",
) -> Path:
    logger.info("=== 开始在相同受扰现场接管下对比 C, E, F, D ===")
    out_dir = REPO_ROOT / "work3_4060/mechanism_v1/takeover"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_pattern = "work3_4060/compare_v1/seed{seed}/{method}/weights.pt"
    md_path = run_all_takeover_diagnostics(
        seeds=[seed],
        method_names=["C", "E", "F", "D"],
        ckpt_dir_pattern=ckpt_pattern,
        out_dir_path="work3_4060/mechanism_v1/takeover",
        device=device,
    )
    logger.info(f"接管诊断完成，报告已保存至: {md_path}")
    return md_path


def generate_mechanism_summary(
    seed: int = 42,
) -> Path:
    out_dir = REPO_ROOT / "work3_4060/mechanism_v1"
    val_csv_path = out_dir / "validation_results.csv"
    takeover_csv_path = out_dir / "takeover" / "takeover_results.csv"
    summary_md_path = out_dir / "mechanism_summary.md"

    val_rows: list[dict[str, str]] = []
    if val_csv_path.is_file():
        with val_csv_path.open("r", encoding="utf-8") as f:
            val_rows = list(csv.DictReader(f))

    takeover_rows: list[dict[str, str]] = []
    if takeover_csv_path.is_file():
        with takeover_csv_path.open("r", encoding="utf-8") as f:
            takeover_rows = list(csv.DictReader(f))

    lines = [
        f"# Work-3 机制消融实验诊断报告 (Seed {seed})",
        f"评估对象：Method C (基线), Method E (辅助头), Method F (辅助头+时间输入), Method D (完整模型)",
        "",
        "## 1. 验证集完整端到端对比 (Nominal + 4 Validation Scenarios)",
    ]

    if val_rows:
        header = "| " + " | ".join(val_rows[0].keys()) + " |"
        sep = "| " + " | ".join([":---"] * len(val_rows[0])) + " |"
        lines.append(header)
        lines.append(sep)
        for r in val_rows:
            lines.append("| " + " | ".join(str(v) for v in r.values()) + " |")

    lines.extend([
        "",
        "## 2. 相同受扰现场接管恢复对比 (Identical Disturbed State Takeover)",
    ])

    if takeover_rows:
        header = "| " + " | ".join(takeover_rows[0].keys()) + " |"
        sep = "| " + " | ".join([":---"] * len(takeover_rows[0])) + " |"
        lines.append(header)
        lines.append(sep)
        for r in takeover_rows:
            lines.append("| " + " | ".join(str(v) for v in r.values()) + " |")

    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"机制消融报告生成完成: {summary_md_path}")
    return summary_md_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-decisions", type=int, default=2048)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    if not args.skip_train:
        # 1. 训练 Method E
        w_e, _ = train_ablation_model("E", seed=args.seed, device=args.device, max_decisions=args.max_decisions)
        verify_contract("E", w_e, device=args.device)

        # 2. 训练 Method F
        w_f, _ = train_ablation_model("F", seed=args.seed, device=args.device, max_decisions=args.max_decisions)
        verify_contract("F", w_f, device=args.device)
    else:
        logger.info("跳过训练，直接执行评估")

    # 3. 验证集评估
    evaluate_validation_scenarios(seed=args.seed, device=args.device)

    # 4. 接管评估
    run_mechanism_takeover_diagnostics(seed=args.seed, device=args.device)

    # 5. 汇总报告
    generate_mechanism_summary(seed=args.seed)


if __name__ == "__main__":
    main()
