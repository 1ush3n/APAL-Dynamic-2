"""导出 mini210 上的详细时间预测指标 CSV (Task 2)。

覆盖 Seed 42, 43, 44，划分 test 与 validation，逐场景统计：
- seed, split, scenario_id
- checkpoint_hash
- sample_count (决策步数/评估样本数)
- labeled_cycle_count (有真实标签的周期数)
- heuristic_mae_hours, corrected_mae_hours (小时)
- heuristic_mae_h0, corrected_mae_h0 (归一化 H0 比例)
- error_p25, error_p50, error_p75, error_p90, error_max (误差分位数，单位：小时)
- mean_signed_bias (平均有符号偏差，单位：小时；正值表示预测偏大，负值表示偏小)
- next_cycle_5station_mask_mae (下一周期 5 个站位掩码有效维度的 MAE)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from envs.work3.environment import AirLineEnvWork3
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.actor_critic import extract_compact_state_features
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from scripts.work3.evaluate_c_vs_d import (
    build_formal_evaluation_agent,
    _check_completed_trajectory_feasibility,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PredMetrics")


def compute_prediction_metrics_for_trajectory(
    agent: Any,
    env: AirLineEnvWork3,
    scenario: dict[str, Any],
    device: str = "cuda:0",
    max_decisions: int = 4096,
) -> dict[str, Any]:
    env.reset()
    env.load_scenario(scenario)
    torch_device = torch.device(device)

    # 记录每步决策时的预测与状态
    samples: list[dict[str, Any]] = []

    decisions = 0
    with torch.inference_mode():
        while not env._check_terminated() and decisions < max_decisions:
            decisions += 1
            cycle_idx = int(env.state.current_cycle)
            cmax = compute_cycle_heuristic_cmax(env.state)
            state_feat = extract_compact_state_features(env.state, cmax)
            graph_snapshot = agent.actor_critic.build_graph_snapshot(env)

            shared_feature = agent.actor_critic.encode_shared_representation(
                state_feat.to(torch_device),
                graph_snapshot,
            ).unsqueeze(0)

            p_corrected, remaining, _ = agent.time_head.predict_corrected_time(
                state_feat=shared_feature,
                estimated_cmax=cmax,
                current_time=float(env.state.current_time),
                h0=float(env.state.h0),
                last_transfer_time=float(env.state.last_transfer_time),
            )
            corrected_cmax = float(p_corrected.detach().cpu().item())

            if hasattr(agent.time_head, "predict_next_cycle_residuals"):
                next_residuals = agent.time_head.predict_next_cycle_residuals(shared_feature).squeeze(0)
            else:
                next_residuals = torch.zeros(5, dtype=torch.float32, device=torch_device)

            next_mask = list(env.state.get_next_cycle_station_mask())

            # 记录样本
            samples.append({
                "cycle_idx": cycle_idx,
                "heuristic_cmax": float(cmax),
                "corrected_cmax": corrected_cmax,
                "h0": float(env.state.h0),
                "next_residuals": next_residuals.detach().cpu().numpy(),
                "next_mask": next_mask,
            })

            # 选择动作并推进
            action = agent.select_action(env)
            if action is None:
                break
            _obs, _r, term, trunc, _info = env.step(action)
            if term or trunc:
                break

    # 轨迹结束，提取真实事实
    transfer_history = list(env.state.transfer_history)
    actual_by_cycle = dict(enumerate(transfer_history, start=1))
    station_cleared = dict(env.state.station_first_cleared_times)

    labeled_samples = [s for s in samples if s["cycle_idx"] in actual_by_cycle]
    sample_count = len(labeled_samples)
    labeled_cycle_count = len({s["cycle_idx"] for s in labeled_samples})

    if sample_count == 0:
        return {
            "sample_count": 0,
            "labeled_cycle_count": 0,
            "heuristic_mae_hours": None,
            "corrected_mae_hours": None,
            "heuristic_mae_h0": None,
            "corrected_mae_h0": None,
            "error_p25": None,
            "error_p50": None,
            "error_p75": None,
            "error_p90": None,
            "error_max": None,
            "mean_signed_bias": None,
            "next_cycle_5station_mask_mae": None,
        }

    err_heur_hours = []
    err_corr_hours = []
    err_heur_h0 = []
    err_corr_h0 = []
    signed_biases = []

    for s in labeled_samples:
        act_t = actual_by_cycle[s["cycle_idx"]]
        h0 = s["h0"]
        e_h = abs(act_t - s["heuristic_cmax"])
        e_c = abs(act_t - s["corrected_cmax"])
        err_heur_hours.append(e_h)
        err_corr_hours.append(e_c)
        err_heur_h0.append(e_h / h0)
        err_corr_h0.append(e_c / h0)
        signed_biases.append(s["corrected_cmax"] - act_t)

    # 计算下一周期掩码有效维度的 MAE
    next_cycle_errors = []
    for s in labeled_samples:
        next_c = s["cycle_idx"] + 1
        if next_c in actual_by_cycle:
            p_prev = actual_by_cycle[s["cycle_idx"]]
            h0 = s["h0"]
            cleared_next = station_cleared.get(next_c, {})
            for m in range(5):
                if s["next_mask"][m] and m in cleared_next:
                    f_m = float(cleared_next[m])
                    target_next_m = (f_m - float(p_prev) - float(h0)) / float(h0)
                    pred_next_m = float(s["next_residuals"][m])
                    next_cycle_errors.append(abs(pred_next_m - target_next_m))

    err_corr_arr = np.array(err_corr_hours)
    quantiles = np.percentile(err_corr_arr, [25, 50, 75, 90, 100])

    return {
        "sample_count": sample_count,
        "labeled_cycle_count": labeled_cycle_count,
        "heuristic_mae_hours": float(np.mean(err_heur_hours)),
        "corrected_mae_hours": float(np.mean(err_corr_hours)),
        "heuristic_mae_h0": float(np.mean(err_heur_h0)),
        "corrected_mae_h0": float(np.mean(err_corr_h0)),
        "error_p25": float(quantiles[0]),
        "error_p50": float(quantiles[1]),
        "error_p75": float(quantiles[2]),
        "error_p90": float(quantiles[3]),
        "error_max": float(quantiles[4]),
        "mean_signed_bias": float(np.mean(signed_biases)),
        "next_cycle_5station_mask_mae": (
            float(np.mean(next_cycle_errors)) if next_cycle_errors else None
        ),
    }


def generate_all_prediction_metrics(
    seeds: list[int] = [42, 43, 44],
    splits: list[str] = ["test", "validation"],
    output_csv_path: str = "work3_4060/compare_v1/prediction_metrics.csv",
    device: str = "cuda:0",
) -> Path:
    out_csv = REPO_ROOT / output_csv_path
    baseline_path = str(REPO_ROOT / "work3_4060/data/mini210.json")

    with (REPO_ROOT / "work3_4060/compare_v1/scenarios.json").open("r", encoding="utf-8") as f:
        scenarios_by_id = {s["scenario_id"]: s for s in json.load(f)}

    rows: list[dict[str, Any]] = []

    for seed in seeds:
        ckpt_path = REPO_ROOT / f"work3_4060/compare_v1/seed{seed}/D/weights.pt"
        if not ckpt_path.is_file():
            logger.warning(f"种子 {seed} 缺少检查点 {ckpt_path}，跳过")
            continue

        ckpt_bytes = ckpt_path.read_bytes()
        ckpt_hash = hashlib.sha256(ckpt_bytes).hexdigest()

        logger.info(f"加载 Seed {seed} Method D 检查点 (SHA256: {ckpt_hash[:8]}...)")
        agent = build_formal_evaluation_agent("D", str(ckpt_path), device=device)

        for split in splits:
            split_file = REPO_ROOT / f"work3_4060/compare_v1/{split}.json"
            with split_file.open("r", encoding="utf-8") as f:
                sc_ids = json.load(f)

            for sc_id in sc_ids:
                sc = scenarios_by_id[sc_id]
                logger.info(f"  评估 Seed {seed} | Split: {split} | 场景: {sc_id}")
                env = AirLineEnvWork3(baseline_json_path=baseline_path)
                m = compute_prediction_metrics_for_trajectory(
                    agent=agent,
                    env=env,
                    scenario=sc,
                    device=device,
                )

                row = {
                    "seed": seed,
                    "split": split,
                    "scenario_id": sc_id,
                    "checkpoint_hash": ckpt_hash,
                    "sample_count": m["sample_count"],
                    "labeled_cycle_count": m["labeled_cycle_count"],
                    "heuristic_mae_hours": round(m["heuristic_mae_hours"], 4) if m["heuristic_mae_hours"] is not None else "",
                    "corrected_mae_hours": round(m["corrected_mae_hours"], 4) if m["corrected_mae_hours"] is not None else "",
                    "heuristic_mae_h0": round(m["heuristic_mae_h0"], 6) if m["heuristic_mae_h0"] is not None else "",
                    "corrected_mae_h0": round(m["corrected_mae_h0"], 6) if m["corrected_mae_h0"] is not None else "",
                    "error_p25": round(m["error_p25"], 4) if m["error_p25"] is not None else "",
                    "error_p50": round(m["error_p50"], 4) if m["error_p50"] is not None else "",
                    "error_p75": round(m["error_p75"], 4) if m["error_p75"] is not None else "",
                    "error_p90": round(m["error_p90"], 4) if m["error_p90"] is not None else "",
                    "error_max": round(m["error_max"], 4) if m["error_max"] is not None else "",
                    "mean_signed_bias": round(m["mean_signed_bias"], 4) if m["mean_signed_bias"] is not None else "",
                    "next_cycle_5station_mask_mae": (
                        round(m["next_cycle_5station_mask_mae"], 6)
                        if m["next_cycle_5station_mask_mae"] is not None
                        else ""
                    ),
                }
                rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"全部预测指标计算完成，已保存至: {out_csv}")
    return out_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="work3_4060/compare_v1/prediction_metrics.csv")
    args = parser.parse_args()

    generate_all_prediction_metrics(
        seeds=args.seeds,
        output_csv_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
