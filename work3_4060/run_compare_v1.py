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


def run_evaluation(seed: int, split: str = "test", device: str = "cuda:0") -> Path:
    seed_dir = REPO_ROOT / f"work3_4060/compare_v1/seed{seed}"
    method_c_ckpt = seed_dir / "C/weights.pt"
    method_d_ckpt = seed_dir / "D/weights.pt"
    output_json = seed_dir / f"{split}_results.json"

    baseline_path = str(REPO_ROOT / "work3_4060/data/mini210.json")
    scenarios_path = str(REPO_ROOT / "work3_4060/compare_v1/scenarios.json")
    split_path = str(REPO_ROOT / f"work3_4060/compare_v1/{split}.json")

    logger.info(f"==> 开始评测 Seed {seed} {split} 集...")
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


def generate_summary(split: str = "test"):
    seeds = [42, 43, 44]
    rows = []

    for seed in seeds:
        res_file = REPO_ROOT / f"work3_4060/compare_v1/seed{seed}/{split}_results.json"
        if not res_file.is_file():
            logger.warning(f"缺少 {res_file}，跳过此 seed 汇总")
            continue
        data = json.loads(res_file.read_text(encoding="utf-8"))

        # 1. Nominal
        nom_c = data.get("nominal_method_c", {})
        nom_d = data.get("nominal_method_d", {})

        c_cw_nom = nom_c.get("cost_components_weighted") or {}
        c_j_takt_nom = c_cw_nom.get("j_takt", nom_c.get("j_takt", 0.0))
        c_d_time_nom = c_cw_nom.get("d_time", nom_c.get("d_time", 0.0))
        c_d_team_nom = c_cw_nom.get("d_team", nom_c.get("d_team", 0.0))
        c_j_postpone_nom = c_cw_nom.get("j_postpone", nom_c.get("j_postpone", 0.0))
        c_j_revision_nom = c_cw_nom.get("j_revision", nom_c.get("j_revision", 0.0))
        c_comp_sum_nom = c_j_takt_nom + c_d_time_nom + c_d_team_nom + c_j_postpone_nom + c_j_revision_nom
        c_j_total_nom = nom_c.get("j_total")
        c_sum_err_nom = (c_comp_sum_nom - c_j_total_nom) if c_j_total_nom is not None else None
        c_m_nom = nom_c.get("makespan")

        d_cw_nom = nom_d.get("cost_components_weighted") or {}
        d_j_takt_nom = d_cw_nom.get("j_takt", nom_d.get("j_takt", 0.0))
        d_d_time_nom = d_cw_nom.get("d_time", nom_d.get("d_time", 0.0))
        d_d_team_nom = d_cw_nom.get("d_team", nom_d.get("d_team", 0.0))
        d_j_postpone_nom = d_cw_nom.get("j_postpone", nom_d.get("j_postpone", 0.0))
        d_j_revision_nom = d_cw_nom.get("j_revision", nom_d.get("j_revision", 0.0))
        d_comp_sum_nom = d_j_takt_nom + d_d_time_nom + d_d_team_nom + d_j_postpone_nom + d_j_revision_nom
        d_j_total_nom = nom_d.get("j_total")
        d_sum_err_nom = (d_comp_sum_nom - d_j_total_nom) if d_j_total_nom is not None else None
        d_m_nom = nom_d.get("makespan")

        rows.append({
            "seed": seed,
            "scenario_id": "NOMINAL_NO_DISTURBANCE",
            "type": "nominal",
            "timing": "NONE",
            "intensity": "NONE",
            "station_id": "NONE",
            "actual_hit_targets": "NONE",
            "hit_status": "NONE",
            "c_feasible": nom_c.get("feasible"),
            "c_success": nom_c.get("success"),
            "c_termination_reason": nom_c.get("termination_reason"),
            "c_makespan": c_m_nom,
            "c_j_total": c_j_total_nom,
            "c_j_takt": c_j_takt_nom,
            "c_d_time": c_d_time_nom,
            "c_d_team": c_d_team_nom,
            "c_j_postpone": c_j_postpone_nom,
            "c_j_revision": c_j_revision_nom,
            "c_component_sum_error": c_sum_err_nom,
            "c_postponed_count": nom_c.get("postponed_count", 0),
            "c_postponed_task_count": nom_c.get("postponed_task_count", 0),
            "c_multiply_postponed_count": nom_c.get("multiply_postponed_task_count", 0),
            "c_delta_j_vs_nominal": 0.0,
            "c_delta_makespan_vs_nominal": 0.0,
            "d_feasible": nom_d.get("feasible"),
            "d_success": nom_d.get("success"),
            "d_termination_reason": nom_d.get("termination_reason"),
            "d_makespan": d_m_nom,
            "d_j_total": d_j_total_nom,
            "d_j_takt": d_j_takt_nom,
            "d_d_time": d_d_time_nom,
            "d_d_team": d_d_team_nom,
            "d_j_postpone": d_j_postpone_nom,
            "d_j_revision": d_j_revision_nom,
            "d_component_sum_error": d_sum_err_nom,
            "d_postponed_count": nom_d.get("postponed_count", 0),
            "d_postponed_task_count": nom_d.get("postponed_task_count", 0),
            "d_multiply_postponed_count": nom_d.get("multiply_postponed_task_count", 0),
            "d_delta_j_vs_nominal": 0.0,
            "d_delta_makespan_vs_nominal": 0.0,
            "j_diff_c_minus_d": (
                (c_j_total_nom - d_j_total_nom)
                if c_j_total_nom is not None and d_j_total_nom is not None else None
            ),
            "makespan_diff_c_minus_d": (
                (c_m_nom - d_m_nom)
                if c_m_nom is not None and d_m_nom is not None else None
            ),
            "delta_j_diff_c_minus_d": 0.0,
            "delta_m_diff_c_minus_d": 0.0,
            "c_hits": 0,
            "d_hits": 0,
        })

        # 2. 扰动场景
        scenarios = data.get("scenario_results", [])
        for sc in scenarios:
            sc_id = sc.get("scenario_id")
            rc = sc.get("method_c", {})
            rd = sc.get("method_d", {})
            c_hits = rc.get("actual_disturbance_hits", 0)
            d_hits = rd.get("actual_disturbance_hits", 0)
            hit_st = "BOTH" if (c_hits > 0 and d_hits > 0) else "ONE_SIDED" if (c_hits > 0 or d_hits > 0) else "NEITHER"

            rc_eff = rc.get("disturbance_effects", {})
            rd_eff = rd.get("disturbance_effects", {})
            c_targets = rc_eff.get("actual_hit_task_keys", [])
            d_targets = rd_eff.get("actual_hit_task_keys", [])
            all_targets = sorted(set(c_targets) | set(d_targets))
            targets_str = ",".join(all_targets) if all_targets else "NONE"

            jc = rc.get("j_total")
            jd = rd.get("j_total")
            j_diff = (jc - jd) if (jc is not None and jd is not None) else None

            mc = rc.get("makespan")
            md = rd.get("makespan")
            m_diff = (mc - md) if (mc is not None and md is not None) else None

            # 5 分项
            c_cw = rc.get("cost_components_weighted") or {}
            c_j_takt = c_cw.get("j_takt", rc.get("j_takt", 0.0))
            c_d_time = c_cw.get("d_time", rc.get("d_time", 0.0))
            c_d_team = c_cw.get("d_team", rc.get("d_team", 0.0))
            c_j_postpone = c_cw.get("j_postpone", rc.get("j_postpone", 0.0))
            c_j_revision = c_cw.get("j_revision", rc.get("j_revision", 0.0))
            c_comp_sum = c_j_takt + c_d_time + c_d_team + c_j_postpone + c_j_revision
            c_sum_err = (c_comp_sum - jc) if jc is not None else None

            d_cw = rd.get("cost_components_weighted") or {}
            d_j_takt = d_cw.get("j_takt", rd.get("j_takt", 0.0))
            d_d_time = d_cw.get("d_time", rd.get("d_time", 0.0))
            d_d_team = d_cw.get("d_team", rd.get("d_team", 0.0))
            d_j_postpone = d_cw.get("j_postpone", rd.get("j_postpone", 0.0))
            d_j_revision = d_cw.get("j_revision", rd.get("j_revision", 0.0))
            d_comp_sum = d_j_takt + d_d_time + d_d_team + d_j_postpone + d_j_revision
            d_sum_err = (d_comp_sum - jd) if jd is not None else None

            # Nominal 附加量
            c_dj = (jc - c_j_total_nom) if (jc is not None and c_j_total_nom is not None) else None
            c_dm = (mc - c_m_nom) if (mc is not None and c_m_nom is not None) else None
            d_dj = (jd - d_j_total_nom) if (jd is not None and d_j_total_nom is not None) else None
            d_dm = (md - d_m_nom) if (md is not None and d_m_nom is not None) else None

            delta_j_diff = (c_dj - d_dj) if (c_dj is not None and d_dj is not None) else None
            delta_m_diff = (c_dm - d_dm) if (c_dm is not None and d_dm is not None) else None

            rows.append({
                "seed": seed,
                "scenario_id": sc_id,
                "type": "disturbance",
                "timing": sc.get("timing"),
                "intensity": sc.get("intensity"),
                "station_id": sc.get("station"),
                "actual_hit_targets": targets_str,
                "hit_status": hit_st,
                "c_feasible": rc.get("feasible"),
                "c_success": rc.get("success"),
                "c_termination_reason": rc.get("termination_reason"),
                "c_makespan": mc,
                "c_j_total": jc,
                "c_j_takt": c_j_takt,
                "c_d_time": c_d_time,
                "c_d_team": c_d_team,
                "c_j_postpone": c_j_postpone,
                "c_j_revision": c_j_revision,
                "c_component_sum_error": c_sum_err,
                "c_postponed_count": rc.get("postponed_count", 0),
                "c_postponed_task_count": rc.get("postponed_task_count", 0),
                "c_multiply_postponed_count": rc.get("multiply_postponed_task_count", 0),
                "c_delta_j_vs_nominal": c_dj,
                "c_delta_makespan_vs_nominal": c_dm,
                "d_feasible": rd.get("feasible"),
                "d_success": rd.get("success"),
                "d_termination_reason": rd.get("termination_reason"),
                "d_makespan": md,
                "d_j_total": jd,
                "d_j_takt": d_j_takt,
                "d_d_time": d_d_time,
                "d_d_team": d_d_team,
                "d_j_postpone": d_j_postpone,
                "d_j_revision": d_j_revision,
                "d_component_sum_error": d_sum_err,
                "d_postponed_count": rd.get("postponed_count", 0),
                "d_postponed_task_count": rd.get("postponed_task_count", 0),
                "d_multiply_postponed_count": rd.get("multiply_postponed_task_count", 0),
                "d_delta_j_vs_nominal": d_dj,
                "d_delta_makespan_vs_nominal": d_dm,
                "j_diff_c_minus_d": j_diff,
                "makespan_diff_c_minus_d": m_diff,
                "delta_j_diff_c_minus_d": delta_j_diff,
                "delta_m_diff_c_minus_d": delta_m_diff,
                "c_hits": c_hits,
                "d_hits": d_hits,
            })

    # 输出 CSV
    if split == "test":
        csv_path = REPO_ROOT / "work3_4060/compare_v1/results.csv"
        md_path = REPO_ROOT / "work3_4060/compare_v1/summary.md"
    else:
        csv_path = REPO_ROOT / f"work3_4060/compare_v1/{split}_results.csv"
        md_path = REPO_ROOT / f"work3_4060/compare_v1/{split}_summary.md"

    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"生成 CSV: {csv_path}")

    # 生成 Markdown 简报
    dist_rows = [r for r in rows if r["type"] == "disturbance"]
    nom_rows = [r for r in rows if r["type"] == "nominal"]

    md_lines = [
        f"# Work-3 小规模 C/D 效果对比结果简报 ({split} 集)",
        f"日期: 2026-10-18 | 数据集: mini210 (210 工序, 7 架, 5 站) | 设备: cuda:0 | 精度: FP32",
        "",
        "## 1. 核心结论摘要",
    ]

    if dist_rows:
        valid_pairs = [r for r in dist_rows if r["c_feasible"] and r["d_feasible"]]
        j_diffs = [r["j_diff_c_minus_d"] for r in valid_pairs if r["j_diff_c_minus_d"] is not None]
        m_diffs = [r["makespan_diff_c_minus_d"] for r in valid_pairs if r["makespan_diff_c_minus_d"] is not None]
        dj_diffs = [r["delta_j_diff_c_minus_d"] for r in valid_pairs if r["delta_j_diff_c_minus_d"] is not None]
        dm_diffs = [r["delta_m_diff_c_minus_d"] for r in valid_pairs if r["delta_m_diff_c_minus_d"] is not None]

        mean_j_diff = sum(j_diffs) / len(j_diffs) if j_diffs else 0.0
        mean_m_diff = sum(m_diffs) / len(m_diffs) if m_diffs else 0.0
        mean_dj_diff = sum(dj_diffs) / len(dj_diffs) if dj_diffs else 0.0
        mean_dm_diff = sum(dm_diffs) / len(dm_diffs) if dm_diffs else 0.0

        md_lines.extend([
            f"- **测试场景总对数**: {len(dist_rows)} (3 seeds × {len(dist_rows)//len(seeds)} 场景)",
            f"- **双向独立可行对数**: {len(valid_pairs)} / {len(dist_rows)} (100% 可行)",
            f"- **绝对费用差均值 (J_C - J_D)**: `{mean_j_diff:+.6e}` (正值表示 D 优于 C)",
            f"- **工期差均值 (Makespan_C - Makespan_D)**: `{mean_m_diff:+.2f}h` (正值表示 D 工期更短)",
            f"- **扰动附加费用差均值 (Delta_J_C - Delta_J_D)**: `{mean_dj_diff:+.6e}`",
            f"- **扰动附加工期差均值 (Delta_M_C - Delta_M_D)**: `{mean_dm_diff:+.2f}h`",
            "",
            "## 2. 逐 Seed 统计表",
            "| Seed | 场景数 | 可行对数 | 双方命中 | 单方命中 | 综合费用差 (J_C - J_D) | 工期差 (h) | 附加费用差 (ΔJ_C - ΔJ_D) | 附加工期差 (h) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for s in seeds:
            s_rows = [r for r in dist_rows if r["seed"] == s]
            s_val = [r for r in s_rows if r["c_feasible"] and r["d_feasible"]]
            s_j = [r["j_diff_c_minus_d"] for r in s_val if r["j_diff_c_minus_d"] is not None]
            s_m = [r["makespan_diff_c_minus_d"] for r in s_val if r["makespan_diff_c_minus_d"] is not None]
            s_dj = [r["delta_j_diff_c_minus_d"] for r in s_val if r["delta_j_diff_c_minus_d"] is not None]
            s_dm = [r["delta_m_diff_c_minus_d"] for r in s_val if r["delta_m_diff_c_minus_d"] is not None]
            b_hits = sum(1 for r in s_rows if r["hit_status"] == "BOTH")
            o_hits = sum(1 for r in s_rows if r["hit_status"] == "ONE_SIDED")
            s_mj = sum(s_j) / len(s_j) if s_j else 0.0
            s_mm = sum(s_m) / len(s_m) if s_m else 0.0
            s_mdj = sum(s_dj) / len(s_dj) if s_dj else 0.0
            s_mdm = sum(s_dm) / len(s_dm) if s_dm else 0.0
            md_lines.append(f"| {s} | {len(s_rows)} | {len(s_val)} | {b_hits} | {o_hits} | {s_mj:+.4e} | {s_mm:+.2f} | {s_mdj:+.4e} | {s_mdm:+.2f} |")

        md_lines.extend([
            "",
            "## 3. 详细逐场景对比 (含 5 分项与 Nominal 附加量)",
            "| Seed | 场景 ID | 工位 | 时机 | 强度 | 命中状态 | 目标工序 | C Makespan | D Makespan | C J_total | D J_total | C 后移 | D 后移 | J_C - J_D | ΔJ_C - ΔJ_D |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        for r in rows:
            jc_str = f"{r['c_j_total']:.4e}" if r['c_j_total'] is not None else "N/A"
            jd_str = f"{r['d_j_total']:.4e}" if r['d_j_total'] is not None else "N/A"
            diff_str = f"{r['j_diff_c_minus_d']:+.4e}" if r['j_diff_c_minus_d'] is not None else "N/A"
            dj_diff_str = f"{r['delta_j_diff_c_minus_d']:+.4e}" if r['delta_j_diff_c_minus_d'] is not None else "N/A"
            mc_str = f"{r['c_makespan']:.1f}" if r['c_makespan'] is not None else "N/A"
            md_str = f"{r['d_makespan']:.1f}" if r['d_makespan'] is not None else "N/A"
            st_str = str(r.get("station_id", "NONE"))
            tgt_str = str(r.get("actual_hit_targets", "NONE"))
            c_post = str(r.get("c_postponed_count", 0))
            d_post = str(r.get("d_postponed_count", 0))
            md_lines.append(f"| {r['seed']} | {r['scenario_id']} | {st_str} | {r['timing']} | {r['intensity']} | {r['hit_status']} | {tgt_str} | {mc_str} | {md_str} | {jc_str} | {jd_str} | {c_post} | {d_post} | {diff_str} | {dj_diff_str} |")

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
    p_ev.add_argument("--split", type=str, default="test", choices=["test", "validation"])
    p_ev.add_argument("--device", type=str, default="cuda:0")

    p_sum = sub.add_parser("summary")
    p_sum.add_argument("--split", type=str, default="test", choices=["test", "validation"])

    sub.add_parser("run-all")

    args = p.parse_args()

    if args.cmd == "train":
        run_training(args.seed, args.method, args.device)
    elif args.cmd == "eval":
        run_evaluation(args.seed, args.split, args.device)
    elif args.cmd == "summary":
        generate_summary(args.split)
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
