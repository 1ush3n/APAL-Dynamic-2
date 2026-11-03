"""Export comprehensive summary CSVs for all three diagnostic tasks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "work3_4060" / "diagnostics" / "summary_tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK1_JSON = REPO_ROOT / "work3_4060" / "diagnostics" / "task1_outputs" / "task1_reachability_report.json"
TASK2_COMPARE_JSON = REPO_ROOT / "work3_4060" / "diagnostics" / "task2_outputs" / "task2_compare_v1_evaluation.json"
TASK2_RULE_JSON = REPO_ROOT / "work3_4060" / "diagnostics" / "task2_outputs" / "task2_rule_baseline_evaluation.json"
TASK3_JSON = REPO_ROOT / "work3_4060" / "diagnostics" / "task3_outputs" / "task3_paired_arms_report.json"


def export_task1_csvs():
    with open(TASK1_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Five pairs witness table
    pairs_csv = OUTPUT_DIR / "task1_five_pairs_witnesses.csv"
    with open(pairs_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair", "aircraft_id", "predecessor_u", "successor_v",
            "v_postponed", "u_unlocked_and_postponed", "batch_completed",
            "makespan_h", "j_total", "cost_postpone", "cost_takt", "cost_team", "cost_time", "cost_revision",
            "steps"
        ])
        for p_name, w in data["pair_witnesses"].items():
            u, v = w["pair"]
            c = w["costs"]
            writer.writerow([
                p_name, w["aircraft_id"], u, v,
                w["v_postponed"], w["u_postponed"], w["all_completed"],
                f"{c['makespan']:.6f}", f"{c['cumulative_cost']:.6f}",
                f"{c['cost_postpone']:.4f}", f"{c['cost_takt']:.6e}",
                f"{c['cost_team']:.6f}", f"{c['cost_time']:.6f}", f"{c['cost_revision']:.6f}",
                w["final_step_count"]
            ])

    # 2. Joint capacity summary
    joint_csv = OUTPUT_DIR / "task1_joint_capacity_summary.csv"
    with open(joint_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "target_count", "postponed_count", "batch_completed",
            "makespan_h", "j_total", "cost_postpone", "cost_takt", "cost_team", "cost_time", "cost_revision", "steps"
        ])
        for exp_name, res in data["joint_postponement_capacity"].items():
            writer.writerow([
                exp_name,
                len(res.get("target_tasks", [])) if "target_tasks" in res else res.get("target_count"),
                res["postponed_count"], res["all_completed"],
                f"{res['makespan']:.6f}", f"{res['cumulative_cost']:.6f}",
                f"{res['cost_postpone']:.4f}", f"{res['cost_takt']:.6e}",
                f"{res['cost_team']:.6f}", f"{res['cost_time']:.6f}", f"{res['cost_revision']:.6f}",
                res["steps"]
            ])
    print(f"Task 1 CSVs exported to: {OUTPUT_DIR}")


def export_task2_csvs():
    # 1. Compare v1 20 scenarios review
    with open(TASK2_COMPARE_JSON, "r", encoding="utf-8") as f:
        data_cmp = json.load(f)

    cmp_csv = OUTPUT_DIR / "task2_compare_v1_20_scenarios_review.csv"
    with open(cmp_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_id", "category", "target_tasks", "target_count", "actual_hit_count",
            "tau_h", "recovery_time_R_h", "delta_param_h", "R_minus_tau_h",
            "target_start_shift_h", "max_transfer_delay_h", "delta_makespan_h",
            "delta_j_total", "batch_makespan_h", "j_total", "cost_takt", "cost_time", "cost_team", "cost_postpone", "cost_revision"
        ])
        for row in data_cmp["scenarios_summary_a"]:
            writer.writerow([
                row["scenario_id"], row["category"],
                ";".join(row["target_keys"]), row["target_count"], row["actual_hit_count"],
                f"{row['tau']:.4f}", f"{row['recovery_time']:.4f}",
                f"{row['delta_field']:.4f}" if row["delta_field"] is not None else "",
                f"{row['R_minus_tau']:.4f}",
                ";".join([f"{k}:{v:+.4f}" for k, v in row["start_shifts"].items() if v is not None]),
                f"{row['max_transfer_delay']:.4f}",
                f"{row['delta_makespan']:+.4f}",
                f"{row['delta_j_total']:+.6f}",
                f"{row['makespan']:.4f}",
                f"{row['cumulative_cost']:.6f}",
                f"{row['cost_takt']:.6e}",
                f"{row['cost_time']:.6f}",
                f"{row['cost_team']:.6f}",
                f"{row['cost_postpone']:.4f}",
                f"{row['cost_revision']:.6f}",
            ])

    # 2. Rule baseline 6 scenarios review
    with open(TASK2_RULE_JSON, "r", encoding="utf-8") as f:
        data_rule = json.load(f)

    rule_csv = OUTPUT_DIR / "task2_rule_baseline_scenarios_review.csv"
    with open(rule_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_id", "split", "category", "target_tasks", "target_count", "actual_hit_count",
            "tau_h", "recovery_time_R_h", "delta_field_h", "R_minus_tau_h", "R_minus_tau_minus_delta",
            "lower_bound_shift_h", "target_start_shift_h", "max_transfer_delay_h", "delta_makespan_h",
            "delta_j_total", "batch_makespan_h", "j_total", "cost_takt", "cost_time", "cost_team", "cost_postpone", "cost_revision"
        ])
        for row in data_rule["scenarios_summary"]:
            writer.writerow([
                row["scenario_id"], row["split"], row["category"],
                ";".join(row["target_keys"]), row["target_count"], row["actual_hit_count"],
                f"{row['tau']:.4f}", f"{row['recovery_time']:.4f}",
                f"{row['delta_field']:.4f}" if row["delta_field"] is not None else "",
                f"{row['R_minus_tau']:.4f}",
                f"{row['delta_vs_R_minus_tau_diff']:.4f}" if row["delta_vs_R_minus_tau_diff"] is not None else "",
                ";".join([f"{k}:{v:+.4f}" for k, v in row["lower_bound_shifts"].items()]),
                ";".join([f"{k}:{v:+.4f}" for k, v in row["start_shifts"].items() if v is not None]),
                f"{row['max_transfer_delay']:.4f}",
                f"{row['delta_makespan']:+.4f}",
                f"{row['delta_j_total']:+.6f}",
                f"{row['makespan']:.4f}",
                f"{row['cumulative_cost']:.6f}",
                f"{row['cost_takt']:.6f}",
                f"{row['cost_time']:.6f}",
                f"{row['cost_team']:.6f}",
                f"{row['cost_postpone']:.4f}",
                f"{row['cost_revision']:.6f}",
            ])

    # 3. RULE_VAL_S2_STRONG cycle-by-cycle breakdown
    cycle_csv = OUTPUT_DIR / "task2_rule_val_s2_strong_cycle_breakdown.csv"
    with open(cycle_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cycle", "nominal_transfer_time_h", "scenario_transfer_time_h", "cycle_delay_h", "status_note"])
        nom_trans = data_rule["nominal"]["transfer_history"]
        s2_trans = data_rule["detailed_results"]["RULE_VAL_S2_STRONG"]["transfer_history"]
        for c_idx in range(len(nom_trans)):
            p_nom = nom_trans[c_idx]
            p_sc = s2_trans[c_idx]
            diff = p_sc - p_nom
            note = ""
            if c_idx < 3:
                note = "Pre-disturbance: normal operation"
            elif 3 <= c_idx <= 8:
                note = f"Full-line bottleneck delay steady transfer (+{diff:.2f}h)"
            elif c_idx == 9:
                note = "Line draining catch-up: recovered to nominal"
            elif c_idx == 10:
                note = "Final completion: makespan unchanged"
            writer.writerow([c_idx + 1, f"{p_nom:.6f}", f"{p_sc:.6f}", f"{diff:+.6f}", note])

    print(f"Task 2 CSVs exported to: {OUTPUT_DIR}")


def export_task3_csvs():
    with open(TASK3_JSON, "r", encoding="utf-8") as f:
        data_t3 = json.load(f)

    t3_csv = OUTPUT_DIR / "task3_paired_arms_summary.csv"
    with open(t3_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "preset_site", "split_task", "action_ctrl", "action_treat",
            "ctrl_completed", "treat_completed",
            "ctrl_makespan_h", "treat_makespan_h", "delta_makespan_h",
            "ctrl_j_total", "treat_j_total", "delta_j_total",
            "delta_cost_postpone", "delta_cost_takt", "delta_cost_revision", "delta_cost_team", "delta_cost_time",
            "treat_postponed_task_start_station", "treat_postponed_task_team"
        ])
        for site_name, site in data_t3.items():
            ctrl = site["control"]
            treat = site["treatment"]
            comp = site["comparison"]
            audit = site["postponed_task_execution_audit"]
            writer.writerow([
                site_name, site["split_task_key"],
                site["split_info_ctrl"]["action"], site["split_info_treat"]["action"],
                ctrl["completed"], treat["completed"],
                f"{ctrl['makespan']:.6f}", f"{treat['makespan']:.6f}", f"{comp['delta_makespan_treatment_minus_control']:+.6f}",
                f"{ctrl['j_total']:.6f}", f"{treat['j_total']:.6f}", f"{comp['delta_j_total_treatment_minus_control']:+.6f}",
                f"{comp['delta_j_postpone']:+.4f}", f"{comp['delta_j_takt']:+.6e}", f"{comp['delta_j_revision']:+.6f}",
                f"{comp['delta_d_team']:+.6f}", f"{comp['delta_d_time']:+.6f}",
                audit["actual_start_station"], str(audit["assigned_team"])
            ])
    print(f"Task 3 CSV exported to: {OUTPUT_DIR}")


def main():
    export_task1_csvs()
    export_task2_csvs()
    export_task3_csvs()
    print("All summary tables exported successfully!")


if __name__ == "__main__":
    main()
