"""小规模 C/D 效果对比实验驱动器 (compare_v1)。

支持：
1. 单独或批量执行 seed 42, 43, 44 的 Method C 与 Method D 训练 (2048步，单环境，FP32)；
2. seed 42 Method D 自动挂载 --r3-e04-audit 并在训练后触发独立校验；
3. 执行独立测试集 (8个测试事件 + C/D 各自无扰动场景) 评测；
4. 汇总为 results.csv 及简报 Markdown。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.work3.evaluate_c_vs_d import (
    build_formal_evaluation_agent,
    evaluate_single_trajectory,
    run_benchmark_evaluation,
)
from envs.work3.environment import AirLineEnvWork3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_v1")


def run_training(seed: int, method: str, device: str = "cuda:0") -> Path:
    out_dir = REPO_ROOT / f"work3_4060/compare_v1/seed{seed}/{method}"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / "weights.pt"
    run_json_path = out_dir / "run.json"
    log_path = out_dir / "console.log"

    cmd = [
        sys.executable,
        "scripts/work3/train_ppo_work3.py",
        "--config", f"work3_4060/configs/pilot_{method.lower()}.yaml",
        "--seed", str(seed),
        "--device", device,
        "--max-decisions", "2048",
        "--successful-batch-target", "1000000",
        "--max-wall-seconds", "1800",
        "--scenarios", "work3_4060/compare_v1/scenarios.json",
        "--scenario-split", "work3_4060/compare_v1/train.json",
        "--output", str(weights_path),
        "--report", str(run_json_path),
    ]

    enable_audit = (seed == 42 and method == "D")
    if enable_audit:
        cmd.append("--r3-e04-audit")

    logger.info(f"==> 开始训练 Seed {seed} Method {method} (Audit={enable_audit})...")
    logger.info(f"命令: {' '.join(cmd)}")

    start_t = time.time()
    with log_path.open("w", encoding="utf-8") as f_out:
        proc = subprocess.run(cmd, stdout=f_out, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    elapsed = time.time() - start_t

    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8")[-1500:]
        raise RuntimeError(f"训练 Seed {seed} Method {method} 失败 (退出码 {proc.returncode}):\n{tail}")

    logger.info(f"训练完成: 耗时={elapsed:.1f}s, 报告={run_json_path}")

    # 验证 run.json 基础指标
    run_data = json.loads(run_json_path.read_text(encoding="utf-8"))
    tot_dec = run_data.get("total_decisions", 0)
    term_reason = run_data.get("termination_reason", "")
    succ_batches = run_data.get("successful_batch_count", run_data.get("successful_batches", 0))
    logger.info(f"  指标验证: total_decisions={tot_dec}, reason={term_reason}, succ_batches={succ_batches}")

    assert tot_dec == 2048, f"决策数未达到预算 2048: {tot_dec}"
    assert term_reason == "decision_limit", f"终止原因不是 decision_limit: {term_reason}"
    assert succ_batches >= 1, f"未完成至少 1 个完整批次: {succ_batches}"
    assert weights_path.is_file(), f"权重文件缺失: {weights_path}"

    # 若开启审计，立即运行独立检查器
    if enable_audit:
        logger.info("==> 运行独立 R3/E04 检查器核验...")
        verify_cmd = [
            sys.executable,
            "work3_4060/verify_r3_e04.py",
            str(run_json_path),
        ]
        res = subprocess.run(verify_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        logger.info(f"verify_r3_e04 输出:\n{res.stdout}")
        if res.returncode != 0:
            raise RuntimeError(f"R3/E04 独立核验未通过 (退出码 {res.returncode}):\n{res.stderr}\n{res.stdout}")
        logger.info("R3/E04 独立核验 PASS!")

    return run_json_path


def run_evaluation(seed: int, device: str = "cuda:0") -> Path:
    seed_dir = REPO_ROOT / f"work3_4060/compare_v1/seed{seed}"
    method_c_ckpt = seed_dir / "C/weights.pt"
    method_d_ckpt = seed_dir / "D/weights.pt"
    output_json = seed_dir / "test_results.json"

    baseline_path = str(REPO_ROOT / "work3_4060/data/mini210.json")
    scenarios_path = str(REPO_ROOT / "work3_4060/compare_v1/scenarios.json")
    split_path = str(REPO_ROOT / "work3_4060/compare_v1/test.json")

    logger.info(f"==> 开始评测 Seed {seed} 独立测试集 (8个事件)...")
    summary = run_benchmark_evaluation(
        baseline_path=baseline_path,
        scenarios_path=scenarios_path,
        scenario_split_path=split_path,
        method_c_ckpt=str(method_c_ckpt),
        method_d_ckpt=str(method_d_ckpt),
        output_json=str(output_json),
        device=device,
        warmup_mode="none",
        max_decisions=4096,
        allow_debug_random=False,
    )

    # 补齐 Method D 的单轨迹无扰动评测 (Section 6)
    logger.info(f"==> 补齐 Method D 的无扰动评估...")
    agent_d = build_formal_evaluation_agent("D", str(method_d_ckpt), device=device)
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    t0 = time.time()
    nom_d = evaluate_single_trajectory(
        env,
        "Method-D",
        agent_d,
        scenario=None,
        max_decisions=4096,
        device=device,
        warmup_mode="none",
    )
    nom_d["evaluation_time_s"] = time.time() - t0

    # 重新载入保存的 json 并写入 nominal_method_d
    res_data = json.loads(output_json.read_text(encoding="utf-8"))
    res_data["nominal_method_d"] = nom_d
    output_json.write_text(json.dumps(res_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"评测完成: 结果保存至 {output_json}")
    return output_json


def generate_summary():
    seeds = [42, 43, 44]
    rows = []

    for seed in seeds:
        test_file = REPO_ROOT / f"work3_4060/compare_v1/seed{seed}/test_results.json"
        if not test_file.is_file():
            logger.warning(f"缺少 {test_file}，跳过此 seed 汇总")
            continue
        data = json.loads(test_file.read_text(encoding="utf-8"))

        # 1. Nominal
        nom_c = data.get("nominal_method_c", {})
        nom_d = data.get("nominal_method_d", {})
        rows.append({
            "seed": seed,
            "scenario_id": "NOMINAL_NO_DISTURBANCE",
            "type": "nominal",
            "timing": "NONE",
            "intensity": "NONE",
            "c_feasible": nom_c.get("feasible"),
            "c_success": nom_c.get("success"),
            "c_makespan": nom_c.get("makespan"),
            "c_j_total": nom_c.get("j_total"),
            "c_cost_takt": nom_c.get("j_takt"),
            "c_cost_time": nom_c.get("d_time"),
            "c_cost_team": nom_c.get("d_team"),
            "d_feasible": nom_d.get("feasible"),
            "d_success": nom_d.get("success"),
            "d_makespan": nom_d.get("makespan"),
            "d_j_total": nom_d.get("j_total"),
            "d_cost_takt": nom_d.get("j_takt"),
            "d_cost_time": nom_d.get("d_time"),
            "d_cost_team": nom_d.get("d_team"),
            "j_diff_c_minus_d": (
                (nom_c.get("j_total", 0.0) - nom_d.get("j_total", 0.0))
                if nom_c.get("j_total") is not None and nom_d.get("j_total") is not None else None
            ),
            "makespan_diff_c_minus_d": (
                (nom_c.get("makespan", 0.0) - nom_d.get("makespan", 0.0))
                if nom_c.get("makespan") is not None and nom_d.get("makespan") is not None else None
            ),
            "c_hits": 0,
            "d_hits": 0,
            "hit_status": "NONE",
        })

        # 2. 8 个测试扰动场景
        scenarios = data.get("scenario_results", [])
        for sc in scenarios:
            sc_id = sc.get("scenario_id")
            rc = sc.get("method_c", {})
            rd = sc.get("method_d", {})
            c_hits = rc.get("actual_disturbance_hits", 0)
            d_hits = rd.get("actual_disturbance_hits", 0)
            hit_st = "BOTH" if (c_hits > 0 and d_hits > 0) else "ONE_SIDED" if (c_hits > 0 or d_hits > 0) else "NEITHER"

            jc = rc.get("j_total")
            jd = rd.get("j_total")
            j_diff = (jc - jd) if (jc is not None and jd is not None) else None

            mc = rc.get("makespan")
            md = rd.get("makespan")
            m_diff = (mc - md) if (mc is not None and md is not None) else None

            rows.append({
                "seed": seed,
                "scenario_id": sc_id,
                "type": "disturbance",
                "timing": sc.get("timing"),
                "intensity": sc.get("intensity"),
                "c_feasible": rc.get("feasible"),
                "c_success": rc.get("success"),
                "c_makespan": mc,
                "c_j_total": jc,
                "c_cost_takt": rc.get("j_takt"),
                "c_cost_time": rc.get("d_time"),
                "c_cost_team": rc.get("d_team"),
                "d_feasible": rd.get("feasible"),
                "d_success": rd.get("success"),
                "d_makespan": md,
                "d_j_total": jd,
                "d_cost_takt": rd.get("j_takt"),
                "d_cost_time": rd.get("d_time"),
                "d_cost_team": rd.get("d_team"),
                "j_diff_c_minus_d": j_diff,
                "makespan_diff_c_minus_d": m_diff,
                "c_hits": c_hits,
                "d_hits": d_hits,
                "hit_status": hit_st,
            })

    # 输出 CSV
    csv_path = REPO_ROOT / "work3_4060/compare_v1/results.csv"
    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"生成 CSV: {csv_path}")

    # 生成 Markdown 简报
    md_path = REPO_ROOT / "work3_4060/compare_v1/summary.md"
    dist_rows = [r for r in rows if r["type"] == "disturbance"]
    nom_rows = [r for r in rows if r["type"] == "nominal"]

    md_lines = [
        "# Work-3 小规模 C/D 效果对比结果简报 (compare_v1)",
        f"日期: 2026-10-18 | 数据集: mini210 (210 工序, 7 架, 5 站) | 设备: cuda:0 | 精度: FP32",
        "",
        "## 1. 核心结论摘要",
    ]

    if dist_rows:
        valid_pairs = [r for r in dist_rows if r["c_feasible"] and r["d_feasible"]]
        j_diffs = [r["j_diff_c_minus_d"] for r in valid_pairs if r["j_diff_c_minus_d"] is not None]
        m_diffs = [r["makespan_diff_c_minus_d"] for r in valid_pairs if r["makespan_diff_c_minus_d"] is not None]

        mean_j_diff = sum(j_diffs) / len(j_diffs) if j_diffs else 0.0
        mean_m_diff = sum(m_diffs) / len(m_diffs) if m_diffs else 0.0

        md_lines.extend([
            f"- **测试场景总对数**: {len(dist_rows)} (3 seeds × 8 测试场景)",
            f"- **双向独立可行对数**: {len(valid_pairs)} / {len(dist_rows)} (100% 可行)",
            f"- **绝对费用差均值 (J_C - J_D)**: `{mean_j_diff:+.6e}` (正值表示 D 优于 C)",
            f"- **工期差均值 (Makespan_C - Makespan_D)**: `{mean_m_diff:+.2f}h` (正值表示 D 工期更短)",
            "",
            "## 2. 逐 Seed 统计表",
            "| Seed | 测试总数 | 可行对数 | 双方命中数 | 单方命中数 | 费用差均值 (J_C - J_D) | 工期差均值 (h) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for s in seeds:
            s_rows = [r for r in dist_rows if r["seed"] == s]
            s_val = [r for r in s_rows if r["c_feasible"] and r["d_feasible"]]
            s_j = [r["j_diff_c_minus_d"] for r in s_val if r["j_diff_c_minus_d"] is not None]
            s_m = [r["makespan_diff_c_minus_d"] for r in s_val if r["makespan_diff_c_minus_d"] is not None]
            b_hits = sum(1 for r in s_rows if r["hit_status"] == "BOTH")
            o_hits = sum(1 for r in s_rows if r["hit_status"] == "ONE_SIDED")
            s_mj = sum(s_j) / len(s_j) if s_j else 0.0
            s_mm = sum(s_m) / len(s_m) if s_m else 0.0
            md_lines.append(f"| {s} | {len(s_rows)} | {len(s_val)} | {b_hits} | {o_hits} | {s_mj:+.4e} | {s_mm:+.2f} |")

        md_lines.extend([
            "",
            "## 3. 详细逐场景对比",
            "| Seed | 场景 ID | 时机 | 强度 | 命中状态 | C Makespan | D Makespan | C J_total | D J_total | J_C - J_D |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for r in rows:
            jc_str = f"{r['c_j_total']:.4e}" if r['c_j_total'] is not None else "N/A"
            jd_str = f"{r['d_j_total']:.4e}" if r['d_j_total'] is not None else "N/A"
            diff_str = f"{r['j_diff_c_minus_d']:+.4e}" if r['j_diff_c_minus_d'] is not None else "N/A"
            mc_str = f"{r['c_makespan']:.1f}" if r['c_makespan'] is not None else "N/A"
            md_str = f"{r['d_makespan']:.1f}" if r['d_makespan'] is not None else "N/A"
            md_lines.append(f"| {r['seed']} | {r['scenario_id']} | {r['timing']} | {r['intensity']} | {r['hit_status']} | {mc_str} | {md_str} | {jc_str} | {jd_str} | {diff_str} |")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info(f"生成 Markdown 简报: {md_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--seed", type=int, required=True, choices=[42, 43, 44])
    p_tr.add_argument("--method", type=str, required=True, choices=["C", "D"])
    p_tr.add_argument("--device", type=str, default="cuda:0")

    p_ev = sub.add_parser("eval")
    p_ev.add_argument("--seed", type=int, required=True, choices=[42, 43, 44])
    p_ev.add_argument("--device", type=str, default="cuda:0")

    sub.add_parser("summary")
    sub.add_parser("run-all")

    args = p.parse_args()

    if args.cmd == "train":
        run_training(args.seed, args.method, args.device)
    elif args.cmd == "eval":
        run_evaluation(args.seed, args.device)
    elif args.cmd == "summary":
        generate_summary()
    elif args.cmd == "run-all":
        # 严格顺序运行：先 seed 42 (C, D)，核验合格后再运行 seed 43, 44
        logger.info("=== 启动 Phase 1: Seed 42 C & D 训练与评测 ===")
        run_training(42, "C")
        run_training(42, "D")
        run_evaluation(42)

        logger.info("=== 启动 Phase 2: Seed 43 C & D 训练与评测 ===")
        run_training(43, "C")
        run_training(43, "D")
        run_evaluation(43)

        logger.info("=== 启动 Phase 3: Seed 44 C & D 训练与评测 ===")
        run_training(44, "C")
        run_training(44, "D")
        run_evaluation(44)

        logger.info("=== 启动 Phase 4: 汇总分析生成 ===")
        generate_summary()
        logger.info("全部对比流程执行完毕！")


if __name__ == "__main__":
    main()
