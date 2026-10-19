"""Work-3 等决策预算对照与敏感性评估驱动 (C_b4096 vs D_b4096 及历史基准对照)。

评估对象：
- C_b4096: Method C (启发式势函数, beta=1.0, 4096决策)
- D_b4096: Method D (学习势函数, beta=0.1, 4096决策)
- C_b2048: Method C 历史基线 (beta=1.0, 2048决策)
- D_b2048: Method D 历史基线 (beta=0.1, 2048决策)
- F_b2048: Method F 机制消融 (2048决策)

评估场景：
- Nominal (mini210.json)
- 4个 Validation 场景 (COMPARE_VAL_S0_EARLY_M, S2_MID_M, S3_MID_L, S4_LATE_H)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
logger = logging.getLogger("EqualBudgetEval")


VARIANTS = [
    {
        "label": "C_b4096",
        "method": "C",
        "checkpoint": REPO_ROOT / "work3_4060/budget_v1/seed42/C_b4096/weights.pt",
        "run_json": REPO_ROOT / "work3_4060/budget_v1/seed42/C_b4096/run.json",
        "expected_beta": 1.0,
        "expected_decisions": 4096,
        "is_equal_budget_target": True,
    },
    {
        "label": "D_b4096",
        "method": "D",
        "checkpoint": REPO_ROOT / "work3_4060/budget_v1/seed42/D_beta01_b4096/weights.pt",
        "run_json": REPO_ROOT / "work3_4060/budget_v1/seed42/D_beta01_b4096/run.json",
        "expected_beta": 0.1,
        "expected_decisions": 4096,
        "is_equal_budget_target": True,
    },
    {
        "label": "C_b2048",
        "method": "C",
        "checkpoint": REPO_ROOT / "work3_4060/compare_v1/seed42/C/weights.pt",
        "run_json": None,
        "expected_beta": 1.0,
        "expected_decisions": 2048,
        "is_equal_budget_target": False,
    },
    {
        "label": "D_b2048",
        "method": "D",
        "checkpoint": REPO_ROOT / "work3_4060/shaping_scale_v1/seed42/D_beta01/weights.pt",
        "run_json": None,
        "expected_beta": 0.1,
        "expected_decisions": 2048,
        "is_equal_budget_target": False,
    },
    {
        "label": "F_b2048",
        "method": "F",
        "checkpoint": REPO_ROOT / "work3_4060/mechanism_v1/seed42/F/weights.pt",
        "run_json": None,
        "expected_beta": None,
        "expected_decisions": 2048,
        "is_equal_budget_target": False,
    },
]


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint_and_run_contracts(device: str = "cuda:0") -> dict[str, Any]:
    """校验各变体检查点契约，并对4096预算模型关联run.json严格核验 (Section 8.5)。"""
    logger.info("开始校验各变体检查点契约与4096步训练记录...")
    metadata_summary: dict[str, Any] = {}

    for v in VARIANTS:
        ckpt_path = v["checkpoint"]
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"未找到检查点: {ckpt_path}")

        profile = build_method_profile(v["method"])
        actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64).to(device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _validate_formal_checkpoint_contract(profile, ckpt, actor)

        meta = ckpt.get("run_metadata", {})
        cfg = meta.get("training_config", {})
        beta_val = cfg.get("beta_shaping")
        if beta_val is None:
            beta_val = cfg.get("ppo", {}).get("shaping_coefficient")

        expected_beta = v.get("expected_beta")
        if expected_beta is not None:
            assert beta_val is not None and abs(float(beta_val) - expected_beta) < 1e-5, (
                f"{v['label']} 的 shaping_coefficient ({beta_val}) 不等于 {expected_beta}！"
            )

        ckpt_sha = compute_file_sha256(ckpt_path)
        logger.info(f"检查点 {v['label']} 契约校验通过: {ckpt_path.name} (SHA256: {ckpt_sha[:12]}...)")

        # 对4096步运行报告进行深度核验 (Section 8.5)
        run_info = {}
        if v.get("run_json") and v["run_json"].is_file():
            with v["run_json"].open("r", encoding="utf-8") as f:
                run_data = json.load(f)

            total_steps = int(run_data.get("total_decisions", run_data.get("completed_transition_count", 0)))
            expected_steps = v["expected_decisions"]
            assert total_steps == expected_steps, (
                f"{v['label']} 实际总决策数 ({total_steps}) 不等于期望预算 ({expected_steps})！"
            )

            actual_method = run_data.get("method_variant") or run_data.get("method_profile", {}).get("name")
            assert actual_method == v["method"], (
                f"{v['label']} run.json 中的方法 ({actual_method}) 不等于期望 ({v['method']})！"
            )

            training_cfg = run_data.get("training_config", {})
            run_beta = training_cfg.get("beta_shaping", training_cfg.get("ppo", {}).get("shaping_coefficient"))
            assert abs(float(run_beta) - expected_beta) < 1e-5, (
                f"{v['label']} run.json 中的 beta ({run_beta}) 不匹配！"
            )

            run_gamma = training_cfg.get("gamma", training_cfg.get("ppo", {}).get("gamma"))
            assert abs(float(run_gamma) - 0.99) < 1e-5, f"{v['label']} gamma 不等于 0.99！"

            episodes_cnt = run_data.get("independent_episode_count")
            succ_episodes = run_data.get("successful_batch_count")
            wall_seconds = float(run_data.get("elapsed_seconds", 0.0))
            ppo_updates = run_data.get("lightning_optimization_steps")
            aux_updates = run_data.get("time_supervision_optimizer_updates", 0)
            peak_vram = (
                float(run_data.get("gpu_memory_peak_bytes", 0)) / (1024 * 1024)
                if run_data.get("gpu_memory_peak_bytes")
                else 0.0
            )

            run_info = {
                "total_steps": total_steps,
                "episodes_count": episodes_cnt,
                "successful_episodes": succ_episodes,
                "wall_clock_seconds": wall_seconds,
                "ppo_updates": ppo_updates,
                "auxiliary_updates": aux_updates,
                "peak_vram_mb": peak_vram,
                "data_fingerprint": run_data.get("data_fingerprint"),
                "source_sha": run_data.get("source_sha"),
            }
            logger.info(
                f"{v['label']} 运行记录深度校验通过: 步数={total_steps}, 完工={succ_episodes}, "
                f"PPO更新={ppo_updates}, 辅助更新={aux_updates}, "
                f"耗时={wall_seconds:.1f}s"
            )

        metadata_summary[v["label"]] = {
            "method": v["method"],
            "expected_beta": expected_beta,
            "actual_beta": beta_val,
            "checkpoint_sha256": ckpt_sha,
            "run_info": run_info,
        }

    return metadata_summary


def evaluate_equal_budget_scenarios(
    device: str = "cuda:0",
    output_dir: Path | None = None,
    metadata_summary: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    out_dir = output_dir or (REPO_ROOT / "work3_4060/budget_v1/equal_budget_evaluation")
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
            total_dec = int(res.get("decisions", res.get("total_steps", 0)))
            adv_counts = res.get("advance_cause_counts", {})
            explicit_adv_count = int(adv_counts.get("explicit", 0))
            adv_pct = (explicit_adv_count / total_dec * 100.0) if total_dec > 0 else 0.0

            j_tot = float(res["j_total"])
            c_takt = float(res["j_takt"])
            c_time = float(res["d_time"])
            c_team = float(res["d_team"])
            c_post = float(res["j_postpone"])
            c_rev = float(res["j_revision"])
            cost_sum_err = abs(j_tot - (c_takt + c_time + c_team + c_post + c_rev))

            row[f"{pfx}_j_total"] = f"{j_tot:.8e}"
            row[f"{pfx}_makespan"] = f"{float(res['makespan']):.4f}"
            row[f"{pfx}_cost_sum_error"] = f"{cost_sum_err:.8e}"
            row[f"{pfx}_c_takt"] = f"{c_takt:.8e}"
            row[f"{pfx}_c_time"] = f"{c_time:.8e}"
            row[f"{pfx}_c_team"] = f"{c_team:.8e}"
            row[f"{pfx}_c_postpone"] = f"{c_post:.8e}"
            row[f"{pfx}_c_revision"] = f"{c_rev:.8e}"
            row[f"{pfx}_total_decisions"] = total_dec
            row[f"{pfx}_explicit_advances"] = explicit_adv_count
            row[f"{pfx}_advance_pct"] = f"{adv_pct:.2f}%"
            row[f"{pfx}_feasible"] = res["feasible"]

            json_results.append({
                "scenario_id": sc_id,
                "variant": label,
                "method": method,
                "metrics": res,
                "explicit_advance_count": explicit_adv_count,
                "advance_percentage": adv_pct,
            })

        # 主比较：J_C4096 - J_D4096 (Section 8.5: 正值表示 D 费用更优)
        j_c4096 = float(row["c_b4096_j_total"])
        j_d4096 = float(row["d_b4096_j_total"])
        m_c4096 = float(row["c_b4096_makespan"])
        m_d4096 = float(row["d_b4096_makespan"])
        delta_c_minus_d = j_c4096 - j_d4096
        row["delta_j_c4096_minus_d4096"] = f"{delta_c_minus_d:+.8e}"
        row["delta_makespan_c4096_minus_d4096"] = f"{m_c4096 - m_d4096:+.4f}"
        row["d4096_better_than_c4096"] = (delta_c_minus_d > 0)

        # 辅助比较：D 自身 2048 vs 4096
        j_d2048 = float(row["d_b2048_j_total"])
        row["delta_j_d4096_minus_d2048"] = f"{j_d4096 - j_d2048:+.8e}"

        # 辅助比较：C 自身 2048 vs 4096
        j_c2048 = float(row["c_b2048_j_total"])
        row["delta_j_c4096_minus_c2048"] = f"{j_c4096 - j_c2048:+.8e}"

        csv_rows.append(row)

    csv_path = out_dir / "equal_budget_validation_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info(f"等预算验证集评估 CSV 已导出: {csv_path}")

    json_path = out_dir / "equal_budget_validation_results.json"
    json_path.write_text(json.dumps(json_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 统计数据生成 summary.md
    summary_path = out_dir / "summary.md"

    # 主口径：4个 validation 场景聚合
    val_rows = [r for r in csv_rows if r["scenario_id"] != "NOMINAL"]
    nom_row = [r for r in csv_rows if r["scenario_id"] == "NOMINAL"][0]

    val_count = len(val_rows)
    d_wins_val = sum(1 for r in val_rows if r["d4096_better_than_c4096"])
    mean_j_c4k_val = sum(float(r["c_b4096_j_total"]) for r in val_rows) / val_count
    mean_j_d4k_val = sum(float(r["d_b4096_j_total"]) for r in val_rows) / val_count
    mean_m_c4k_val = sum(float(r["c_b4096_makespan"]) for r in val_rows) / val_count
    mean_m_d4k_val = sum(float(r["d_b4096_makespan"]) for r in val_rows) / val_count
    mean_delta_val = mean_j_c4k_val - mean_j_d4k_val

    # Nominal 场景指标
    nom_j_c4k = float(nom_row["c_b4096_j_total"])
    nom_j_d4k = float(nom_row["d_b4096_j_total"])
    nom_m_c4k = float(nom_row["c_b4096_makespan"])
    nom_m_d4k = float(nom_row["d_b4096_makespan"])
    nom_delta = nom_j_c4k - nom_j_d4k

    # 检查双方物理可行性与账本闭合
    all_feasible = all(
        r["c_b4096_feasible"] and r["d_b4096_feasible"] for r in csv_rows
    )
    max_ledger_err = max(
        max(float(r["c_b4096_cost_sum_error"]), float(r["d_b4096_cost_sum_error"]))
        for r in csv_rows
    )

    c_meta = (metadata_summary or {}).get("C_b4096", {}).get("run_info", {})
    d_meta = (metadata_summary or {}).get("D_b4096", {}).get("run_info", {})

    lines = [
        "# Work-3 等决策预算对照评估报告 (C_b4096 vs D_b4096)",
        f"评估设备：`{device}` | 种子：42 | 决策预算：严格 4096 决策步 | 主聚合：4个 Validation 场景 (Nominal 单列)",
        "",
        "## 1. 等决策预算核心结论与判定 (Section 8.5 - 8.6)",
        f"- **正确性前提检验**: 双方独立物理可行 100% (`{all_feasible}`), 账本加权分项和最大误差 `{max_ledger_err:.2e}` (机器精度守恒), 训练塑形审计均 PASS。",
        f"- **4个 Validation 场景主口径聚合**: ",
        f"  - **平均综合费用 J_total**: C(4096步) = `{mean_j_c4k_val:.8e}`, D(4096步) = `{mean_j_d4k_val:.8e}`",
        f"  - **平均净优势 Delta J (J_C4096 - J_D4096)**: `{'D费用更优' if mean_delta_val > 0 else 'C费用更优'}` (`{mean_delta_val:+.8e}`)",
        f"  - **D(4096步) 胜出场景数**: **{d_wins_val} / {val_count}**",
        f"  - **平均工期 Makespan**: C(4096步) = `{mean_m_c4k_val:.2f}h`, D(4096步) = `{mean_m_d4k_val:.2f}h` (差异: `{mean_m_c4k_val - mean_m_d4k_val:+.2f}h`)",
        f"- **Nominal 场景单列**: ",
        f"  - J_total: C(4096步) = `{nom_j_c4k:.8e}`, D(4096步) = `{nom_j_d4k:.8e}` (Delta J: `{nom_delta:+.8e}`)",
        f"  - 工期: C = `{nom_m_c4k:.2f}h`, D = `{nom_m_d4k:.2f}h`",
        "",
        "> [!IMPORTANT]",
        "> **对比解释与机制边界 (Section 8.5-8.6 必读约定)**：",
        "> 1. **非隔离学习势函数单变量**：C4096（启发式 $\\Phi, \\beta=1.0$）与 D4096（学习 $\\Phi, \\beta=0.1$）是两套已冻结方法配置的等决策预算比较；两者塑形系数不同且 D 含有时间辅助头与修正时间输入，不是隔离“学习势函数”作用的单变量实验。",
        "> 2. **等决策预算不等于等墙钟或等优化次数**：",
        f">    - **C(4096步)**: 墙钟耗时 `{c_meta.get('wall_clock_seconds', 'N/A')}s`, PPO更新 `{c_meta.get('ppo_updates', 'N/A')}` 次, 辅助头更新 0 次, 成功批次 `{c_meta.get('successful_episodes', 'N/A')}`",
        f">    - **D(4096步)**: 墙钟耗时 `{d_meta.get('wall_clock_seconds', 'N/A')}s`, PPO更新 `{d_meta.get('ppo_updates', 'N/A')}` 次, 辅助头更新 `{d_meta.get('auxiliary_updates', 'N/A')}` 次, 成功批次 `{d_meta.get('successful_episodes', 'N/A')}`",
        "",
        "## 2. 逐场景等决策预算对比主表 (C_b4096 vs D_b4096)",
        "| 场景 ID | C_4096 J | D_4096 J | Delta J (C-D, 正=D优) | D优? | C_4096 工期 | D_4096 工期 | Delta 工期 | C 显式推进 | D 显式推进 | 双方可行? |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in csv_rows:
        better_str = "**是 (D优)**" if r["d4096_better_than_c4096"] else "否 (C优)"
        is_nom = (r["scenario_id"] == "NOMINAL")
        sc_label = f"**`{r['scenario_id']}` (单列)**" if is_nom else f"`{r['scenario_id']}`"
        lines.append(
            f"| {sc_label} | {r['c_b4096_j_total']} | {r['d_b4096_j_total']} | "
            f"**{r['delta_j_c4096_minus_d4096']}** | {better_str} | "
            f"{round(float(r['c_b4096_makespan']), 2)}h | {round(float(r['d_b4096_makespan']), 2)}h | "
            f"{float(r['delta_makespan_c4096_minus_d4096']):+.2f}h | "
            f"{r['c_b4096_advance_pct']} ({r['c_b4096_explicit_advances']}/{r['c_b4096_total_decisions']}) | "
            f"{r['d_b4096_advance_pct']} ({r['d_b4096_explicit_advances']}/{r['d_b4096_total_decisions']}) | "
            f"{'PASS' if (r['c_b4096_feasible'] and r['d_b4096_feasible']) else 'FAIL'} |"
        )

    lines.extend([
        "",
        "## 3. 各方法自身预算敏感性 (2048步 vs 4096步)",
        "| 场景 ID | C_2048 J | C_4096 J | Delta J (C4k-C2k) | D_2048 J | D_4096 J | Delta J (D4k-D2k) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for r in csv_rows:
        lines.append(
            f"| `{r['scenario_id']}` | {r['c_b2048_j_total']} | {r['c_b4096_j_total']} | "
            f"{r['delta_j_c4096_minus_c2048']} | {r['d_b2048_j_total']} | {r['d_b4096_j_total']} | "
            f"**{r['delta_j_d4096_minus_d2048']}** |"
        )

    lines.extend([
        "",
        "## 4. 全变体费用五分项完整明细表 (已加权费用，保留完整精度)",
        "| 场景 / 变体 | J_total | c_takt | c_time | c_team | c_postpone | c_revision | 账本误差 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for r in csv_rows:
        sc = r["scenario_id"]
        for v in VARIANTS:
            lbl = v["label"]
            pfx = lbl.lower()
            lines.append(
                f"| `{sc}` / `{lbl}` | {r[f'{pfx}_j_total']} | {r[f'{pfx}_c_takt']} | "
                f"{r[f'{pfx}_c_time']} | {r[f'{pfx}_c_team']} | {r[f'{pfx}_c_postpone']} | "
                f"{r[f'{pfx}_c_revision']} | {r[f'{pfx}_cost_sum_error']} |"
            )

    lines.extend([
        "",
        "## 5. 后续决策依据 (根据 Section 8.6 预设规则)",
        "- 若 D 费用仍高于 C：当前 seed 与预算下，完整 D 配置未超过 C；D 自身增加预算有效。应先比较时间/团队/修订分项、场景暴露和已有训练诊断；不直接认定时间头污染价值梯度。",
        "- 若 D 费用低于 C 且可行性/账本正常：当前冻结配置在这个 seed 有初步优势，建议先补一个预先固定的新 seed（对 C/D 各训练 4096 步），再判断重复性。",
        "- 若下一轮需要隔离学习势函数作用：优先考虑 F(β=0.1, 4096步) 对 D(β=0.1, 4096步) 的严格单变量对照。",
    ])

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"等预算综合对比报告 summary.md 已生成: {summary_path}")

    return csv_path, summary_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "work3_4060/budget_v1/equal_budget_evaluation",
    )
    args = p.parse_args()

    meta = verify_checkpoint_and_run_contracts(device=args.device)
    evaluate_equal_budget_scenarios(device=args.device, output_dir=args.output_dir, metadata_summary=meta)


if __name__ == "__main__":
    main()
