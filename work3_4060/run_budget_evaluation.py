"""Work-3 预算敏感性与效果检验评估驱动 (D_b2048 vs D_b4096 及基准对比)。

评估对象：
- D_b2048: Method D (beta=0.1, 2048决策)
- D_b4096: Method D (beta=0.1, 4096决策，从零训练)
- C: Method C 基线 (作为环境对照参考)
- F: Method F 机制消融 (作为环境对照参考)

评估场景：
- Nominal (mini210.json)
- 4个 Validation 场景 (COMPARE_VAL_S0_EARLY_M, S2_MID_M, S3_MID_L, S4_LATE_H)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import ActionBranch
from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import ActorCriticWork3
from scripts.work3.experiment_protocol import build_method_profile
from scripts.work3.evaluate_c_vs_d import (
    build_formal_evaluation_agent,
    evaluate_single_trajectory,
    _validate_formal_checkpoint_contract,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BudgetEval")


VARIANTS = [
    {
        "label": "D_b2048",
        "method": "D",
        "checkpoint": REPO_ROOT / "work3_4060/shaping_scale_v1/seed42/D_beta01/weights.pt",
        "expected_beta": 0.1,
    },
    {
        "label": "D_b4096",
        "method": "D",
        "checkpoint": REPO_ROOT / "work3_4060/budget_v1/seed42/D_beta01_b4096/weights.pt",
        "expected_beta": 0.1,
    },
    {
        "label": "C",
        "method": "C",
        "checkpoint": REPO_ROOT / "work3_4060/compare_v1/seed42/C/weights.pt",
        "expected_beta": None,
    },
    {
        "label": "F",
        "method": "F",
        "checkpoint": REPO_ROOT / "work3_4060/mechanism_v1/seed42/F/weights.pt",
        "expected_beta": None,
    },
]


def verify_checkpoint_contracts(device: str = "cuda:0") -> None:
    """校验各变体检查点契约。"""
    logger.info("开始校验各变体检查点契约...")
    for v in VARIANTS:
        ckpt_path = v["checkpoint"]
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"未找到检查点: {ckpt_path}")

        profile = build_method_profile(v["method"])
        actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64).to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _validate_formal_checkpoint_contract(profile, ckpt, actor)

        expected_beta = v.get("expected_beta")
        if expected_beta is not None:
            meta = ckpt.get("run_metadata", {})
            cfg = meta.get("training_config", {})
            beta_val = cfg.get("beta_shaping")
            if beta_val is None:
                beta_val = cfg.get("ppo", {}).get("shaping_coefficient")
            assert beta_val is not None and abs(float(beta_val) - expected_beta) < 1e-5, (
                f"{v['label']} 的 shaping_coefficient ({beta_val}) 不等于 {expected_beta}！"
            )
        logger.info(f"检查点 {v['label']} 契约校验通过: {ckpt_path.name}")


def evaluate_budget_scenarios(
    device: str = "cuda:0",
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    out_dir = output_dir or (REPO_ROOT / "work3_4060/budget_v1/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = str(REPO_ROOT / "work3_4060/data/mini210.json")

    with (REPO_ROOT / "work3_4060/compare_v1/scenarios.json").open("r", encoding="utf-8") as f:
        scenarios_by_id = {s["scenario_id"]: s for s in json.load(f)}

    with (REPO_ROOT / "work3_4060/compare_v1/validation.json").open("r", encoding="utf-8") as f:
        val_scenario_ids = json.load(f)

    all_eval_scenarios = [None] + [scenarios_by_id[sc_id] for sc_id in val_scenario_ids]

    csv_rows: list[dict[str, Any]] = []
    json_results: list[dict[str, Any]] = []

    for sc in all_eval_scenarios:
        sc_id = sc["scenario_id"] if sc is not None else "NOMINAL"
        logger.info(f"--- 评估场景: {sc_id} ---")

        row: dict[str, Any] = {
            "scenario_id": sc_id,
            "timing": sc.get("timing", "NONE") if sc else "NONE",
            "intensity": sc.get("intensity", "NONE") if sc else "NONE",
            "station": sc.get("station_id", sc.get("station", "NONE")) if sc else "NONE",
        }

        for v in VARIANTS:
            label = v["label"]
            method = v["method"]
            ckpt_path = v["checkpoint"]

            agent = build_formal_evaluation_agent(method, str(ckpt_path), device=device)
            env = AirLineEnvWork3(baseline_json_path=baseline_path)
            res = evaluate_single_trajectory(
                env=env,
                agent_type=f"Method-{method}",
                agent=agent,
                scenario=sc,
                device=device,
                max_decisions=4096,
                warmup_mode="none",
            )

            pfx = label.lower()
            actions = res.get("actions", [])
            total_dec = len(actions)
            explicit_adv_count = sum(
                1 for a in actions if a.get("branch") == ActionBranch.ADVANCE_TO_NEXT_EVENT
            )
            adv_pct = (explicit_adv_count / total_dec * 100.0) if total_dec > 0 else 0.0

            # 提取显式推进时间
            adv_durations = [
                float(a.get("dt", 0.0))
                for a in actions
                if a.get("branch") == ActionBranch.ADVANCE_TO_NEXT_EVENT
            ]
            mean_adv_dt = (sum(adv_durations) / len(adv_durations)) if adv_durations else 0.0

            row[f"{pfx}_j_total"] = f"{res['j_total']:.8e}"
            row[f"{pfx}_makespan"] = f"{res['makespan']:.4f}"
            row[f"{pfx}_cost_sum_error"] = f"{res['cost_sum_error']:.8e}"
            row[f"{pfx}_c_takt"] = f"{res['c_takt']:.8e}"
            row[f"{pfx}_c_time"] = f"{res['c_time']:.8e}"
            row[f"{pfx}_c_team"] = f"{res['c_team']:.8e}"
            row[f"{pfx}_c_postpone"] = f"{res['c_postpone']:.8e}"
            row[f"{pfx}_c_revision"] = f"{res['c_revision']:.8e}"
            row[f"{pfx}_total_decisions"] = total_dec
            row[f"{pfx}_explicit_advances"] = explicit_adv_count
            row[f"{pfx}_advance_pct"] = f"{adv_pct:.2f}%"
            row[f"{pfx}_mean_adv_dt"] = f"{mean_adv_dt:.4f}"
            row[f"{pfx}_feasible"] = res["is_feasible"]

            json_results.append({
                "scenario_id": sc_id,
                "variant": label,
                "method": method,
                "metrics": res,
                "explicit_advance_count": explicit_adv_count,
                "advance_percentage": adv_pct,
                "mean_advance_dt": mean_adv_dt,
            })

        # 计算 D_b4096 相对 D_b2048 的 Delta
        j_2048 = float(row["d_b2048_j_total"])
        j_4096 = float(row["d_b4096_j_total"])
        m_2048 = float(row["d_b2048_makespan"])
        m_4096 = float(row["d_b4096_makespan"])
        row["delta_j_4096_minus_2048"] = f"{j_4096 - j_2048:+.8e}"
        row["delta_makespan_4096_minus_2048"] = f"{m_4096 - m_2048:+.4f}"
        row["b4096_better_j"] = (j_4096 < j_2048)

        csv_rows.append(row)

    csv_path = out_dir / "validation_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info(f"验证集评估 CSV 已导出: {csv_path}")

    json_path = out_dir / "validation_results.json"
    json_path.write_text(json.dumps(json_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 生成一页 summary.md
    summary_path = out_dir / "summary.md"
    b4096_wins = sum(1 for r in csv_rows if r["b4096_better_j"])
    mean_j_2048 = sum(float(r["d_b2048_j_total"]) for r in csv_rows) / len(csv_rows)
    mean_j_4096 = sum(float(r["d_b4096_j_total"]) for r in csv_rows) / len(csv_rows)
    mean_m_2048 = sum(float(r["d_b2048_makespan"]) for r in csv_rows) / len(csv_rows)
    mean_m_4096 = sum(float(r["d_b4096_makespan"]) for r in csv_rows) / len(csv_rows)
    gate_pass = (mean_j_4096 < mean_j_2048) or (b4096_wins >= 3)

    lines = [
        "# Work-3 预算敏感性评估报告 (D_b2048 vs D_b4096)",
        f"评估设备：`{device}` | 场景集合：Nominal + 4个Validation场景 | 决策预算对比：2048 vs 4096 步",
        "",
        "## 1. 预算敏感性核心结论与决策门禁 (Section 4.3)",
        f"- **4096步综合费用优于2048步场景数**: **{b4096_wins} / {len(csv_rows)}**",
        f"- **平均综合费用 J_total**: D(2048步) = `{mean_j_2048:.8e}`, D(4096步) = `{mean_j_4096:.8e}` (变化: `{mean_j_4096 - mean_j_2048:+.8e}`)",
        f"- **平均工期 Makespan**: D(2048步) = `{mean_m_2048:.2f}h`, D(4096步) = `{mean_m_4096:.2f}h` (变化: `{mean_m_4096 - mean_m_2048:+.2f}h`)",
        f"- **扩展决策门禁判定**: **{'【通过】建议启动 Method C 4096步对照' if gate_pass else '【未通过】暂停扩大预算，转向机制层排查'}**",
        "",
        "## 2. 逐场景详细对比表",
        "| 场景 ID | D_2048 J | D_4096 J | Delta J (4096-2048) | 4096优? | D_2048 工期 | D_4096 工期 | Delta 工期 | D_2048 推进占比 | D_4096 推进占比 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in csv_rows:
        better_str = "是 (更优)" if r["b4096_better_j"] else "否 (持平/变劣)"
        lines.append(
            f"| `{r['scenario_id']}` | {r['d_b2048_j_total']} | {r['d_b4096_j_total']} | "
            f"**{r['delta_j_4096_minus_2048']}** | **{better_str}** | "
            f"{round(float(r['d_b2048_makespan']), 2)}h | {round(float(r['d_b4096_makespan']), 2)}h | "
            f"{float(r['delta_makespan_4096_minus_2048']):+.2f}h | "
            f"{r['d_b2048_advance_pct']} ({r['d_b2048_explicit_advances']}/{r['d_b2048_total_decisions']}) | "
            f"{r['d_b4096_advance_pct']} ({r['d_b4096_explicit_advances']}/{r['d_b4096_total_decisions']}) |"
        )

    lines.extend([
        "",
        "## 3. 基线与对照方法对比 (C, F, D_b2048, D_b4096)",
        "| 场景 ID | Method C J | Method F J | D_b2048 J | D_b4096 J | C 工期 | F 工期 | D_2048 工期 | D_4096 工期 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for r in csv_rows:
        lines.append(
            f"| `{r['scenario_id']}` | {r['c_j_total']} | {r['f_j_total']} | "
            f"{r['d_b2048_j_total']} | {r['d_b4096_j_total']} | "
            f"{round(float(r['c_makespan']), 2)}h | {round(float(r['f_makespan']), 2)}h | "
            f"{round(float(r['d_b2048_makespan']), 2)}h | {round(float(r['d_b4096_makespan']), 2)}h |"
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"综合对比报告 summary.md 已生成: {summary_path}")

    return csv_path, summary_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "work3_4060/budget_v1/evaluation")
    args = p.parse_args()

    verify_checkpoint_contracts(device=args.device)
    evaluate_budget_scenarios(device=args.device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
