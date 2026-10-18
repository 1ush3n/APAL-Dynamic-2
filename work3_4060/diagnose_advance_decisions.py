"""Work-3 显式事件推进决策单变量物理与策略诊断 (Section 4)。

评估对象：
- Method C: 基线图策略 (work3_4060/compare_v1/seed42/C/weights.pt)
- Method F: 机制消融 (work3_4060/mechanism_v1/seed42/F/weights.pt)
- Method D(beta=0.1): 新训练 (work3_4060/shaping_scale_v1/seed42/D_beta01/weights.pt)

环境设置：
- mini210.json, Nominal (无扰动), greedy (deterministic=True), warmup=none.

核心诊断问题：
D(beta=0.1) 在 Nominal 下出现的 212 次显式推进（ADVANCE_TO_NEXT_EVENT=2）：
1. 是环境零时间无物理进展的异常转换 (Mask/推进Bug)？
2. 还是策略主动绕过有利的立即开工机会 (Value target/学习偏置)？
3. 抑或是资源/前序限制下的合理等待 (因不可撤回预约或工人占用而只能等待下一个离散事件)？
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3, NoFeasibleSlotError
from scripts.work3.evaluate_c_vs_d import build_formal_evaluation_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DiagnoseAdvance")


def analyze_candidate_opportunities(env: AirLineEnvWork3) -> dict[str, Any]:
    """严格利用既有物理约束计算候选工序的即时开工与未来预约机会。

    遵循规则：
    1. 必须考虑当前时刻、前序完工/预约释放时间、开工可用下界（物料时间）以及对齐目标。
    2. 必须考虑目标站内工人的日历占用区间。
    3. “立即可开工”：存在至少一个合法团队，其最早可行开工时刻 <= current_time + tolerance。
    4. “仅限未来预约”：所有候选工序的最早可行开工时刻均 > current_time + tolerance。
    """
    candidates = env.get_action_candidates()
    current_time = float(env.state.current_time)
    tol = env.tolerance

    if not candidates:
        return {
            "candidate_count": 0,
            "immediate_count": 0,
            "future_only_count": 0,
            "earliest_start_time": None,
            "immediate_candidates": [],
            "future_candidates": [],
        }

    immediate_tasks = []
    future_tasks = []
    all_earliest_starts = []

    for task in candidates:
        if not env.can_reserve(task):
            continue

        predecessor_release = env._predecessor_release_time(task)
        if predecessor_release is None:
            continue

        # 候选团队：优先尝试 base_team，其次尝试站内满足技能要求的工人群体
        candidate_teams: list[tuple[int, ...]] = []
        if task.base_team and len(task.base_team) == task.demand:
            candidate_teams.append(tuple(task.base_team))

        station_ctx = env.get_team_completion_context(task)
        skilled_workers = [w.worker_id for w in station_ctx.workers if task.skill in w.skills]
        if len(skilled_workers) >= task.demand:
            for team_comb in combinations(skilled_workers, task.demand):
                if team_comb not in candidate_teams:
                    candidate_teams.append(team_comb)
                if len(candidate_teams) >= 5:  # 限制最多检查5个团队组合，避免组合爆炸
                    break

        task_earliest_start = None

        for team in candidate_teams:
            for align in (0, 1):
                search_start = max(current_time, float(task.material_ready_time), float(predecessor_release))
                if align == 1:
                    align_target = float(env.state.last_transfer_time + task.in_station_offset)
                    search_start = max(search_start, align_target)

                duration = env.duration_for_team(task, team, start_time=search_start)
                try:
                    t_slot = env._find_team_earliest_slot(
                        station_id=task.current_station,
                        team=team,
                        search_start=search_start,
                        duration=duration,
                    )
                    if task_earliest_start is None or t_slot < task_earliest_start:
                        task_earliest_start = t_slot
                except NoFeasibleSlotError:
                    continue

        if task_earliest_start is not None:
            all_earliest_starts.append(task_earliest_start)
            if task_earliest_start <= current_time + tol:
                immediate_tasks.append((task.task_key, task_earliest_start))
            else:
                future_tasks.append((task.task_key, task_earliest_start))

    min_start = min(all_earliest_starts) if all_earliest_starts else None

    return {
        "candidate_count": len(candidates),
        "immediate_count": len(immediate_tasks),
        "future_only_count": len(future_tasks),
        "earliest_start_time": min_start,
        "immediate_candidates": immediate_tasks,
        "future_candidates": future_tasks,
    }


def run_detailed_diagnostic(
    variant_label: str,
    method_name: str,
    checkpoint_path: Path,
    device: str = "cuda:0",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logger.info(f"=== 运行变体决策诊断: {variant_label} ({method_name}) ===")
    baseline_path = str(REPO_ROOT / "work3_4060/data/mini210.json")

    agent = build_formal_evaluation_agent(method_name, str(checkpoint_path), device=device)
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    step_records: list[dict[str, Any]] = []
    step_idx = 0

    with torch.inference_mode():
        while not env._check_terminated() and step_idx < 4096:
            step_idx += 1
            t_before = float(env.state.current_time)
            cycle_before = int(env.state.current_cycle)
            cost_before = float(env.cumulative_cost)

            # 统计前置任务状态
            running_before = sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.RUNNING)
            reserved_before = sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.RESERVED)
            completed_before = sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.COMPLETED)

            # 诊断当前时刻物理开工机会
            opp = analyze_candidate_opportunities(env)

            # 策略决策
            act = agent.select_action(env)
            if act is None:
                break

            branch = int(act.get("branch", ActionBranch.STATION_EXECUTE))

            # 提取推进概率 (若模型支持 advance_logits)
            prob_advance = None
            if hasattr(agent, "actor_critic") and hasattr(agent.actor_critic, "_advance_logits"):
                try:
                    # 仅在具备 advance_available 时有 advance 概率
                    advance_available = any(
                        t.status == TaskStatus.RESERVED for t in env.state.tasks.values()
                    ) and len(env.get_action_candidates()) > 0
                    if advance_available:
                        # 从 agent 的最新 snapshot/特征推断
                        pass
                except Exception:
                    pass

            # 执行 step
            _obs, _r, term, trunc, info = env.step(act)

            t_after = float(env.state.current_time)
            dt = t_after - t_before
            cost_after = float(env.cumulative_cost)
            d_cost = cost_after - cost_before

            running_after = sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.RUNNING)
            reserved_after = sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.RESERVED)
            completed_after = sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.COMPLETED)

            advance_cause = str(info.get("advance_cause", "none"))
            is_explicit_advance = (branch == int(ActionBranch.ADVANCE_TO_NEXT_EVENT)) and (advance_cause == "explicit")

            # 零时间且无物理状态变化检测
            state_changed = (
                (running_after != running_before)
                or (reserved_after != reserved_before)
                or (completed_after != completed_before)
                or (abs(d_cost) > 1e-6)
            )
            zero_time_no_change = (abs(dt) <= env.tolerance) and (not state_changed)

            rec = {
                "variant": variant_label,
                "step": step_idx,
                "time_before": round(t_before, 2),
                "cycle_before": cycle_before,
                "branch": branch,
                "branch_name": ActionBranch(branch).name,
                "advance_cause": advance_cause,
                "is_explicit_advance": is_explicit_advance,
                "candidates_count": opp["candidate_count"],
                "immediate_count": opp["immediate_count"],
                "future_only_count": opp["future_only_count"],
                "earliest_start_time": round(opp["earliest_start_time"], 2) if opp["earliest_start_time"] is not None else None,
                "dt": round(dt, 2),
                "time_after": round(t_after, 2),
                "d_cost": f"{d_cost:.6e}",
                "running_before": running_before,
                "reserved_before": reserved_before,
                "completed_before": completed_before,
                "running_after": running_after,
                "reserved_after": reserved_after,
                "completed_after": completed_after,
                "zero_time_no_change": zero_time_no_change,
            }
            step_records.append(rec)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 聚合汇总统计
    total_decisions = len(step_records)
    explicit_advances = [r for r in step_records if r["is_explicit_advance"]]
    forced_advances = [r for r in step_records if r["advance_cause"] == "forced_no_candidate"]
    auto_advances = [r for r in step_records if r["advance_cause"] == "auto_after_action"]

    adv_with_immediate = [r for r in explicit_advances if r["immediate_count"] > 0]
    adv_with_future_only = [r for r in explicit_advances if r["immediate_count"] == 0 and r["future_only_count"] > 0]
    adv_with_no_cand = [r for r in explicit_advances if r["candidates_count"] == 0]
    zero_time_advances = [r for r in explicit_advances if r["zero_time_no_change"]]

    cum_waiting_time = sum(r["dt"] for r in explicit_advances)
    cum_advance_cost = sum(float(r["d_cost"]) for r in explicit_advances)

    summary = {
        "variant": variant_label,
        "method": method_name,
        "total_decisions": total_decisions,
        "explicit_advance_count": len(explicit_advances),
        "forced_advance_count": len(forced_advances),
        "auto_after_action_count": len(auto_advances),
        "adv_with_immediate_opportunity": len(adv_with_immediate),
        "adv_with_future_only_opportunity": len(adv_with_future_only),
        "adv_with_zero_candidates": len(adv_with_no_cand),
        "zero_time_no_state_change_count": len(zero_time_advances),
        "cumulative_waiting_time_hours": round(cum_waiting_time, 2),
        "cumulative_advance_cost_increment": f"{cum_advance_cost:.6e}",
        "final_makespan": round(float(env.state.current_time), 2),
        "final_j_total": f"{float(env.cumulative_cost):.6e}",
        "final_d_time": f"{float(env.cost_time):.6e}",
        "final_d_team": f"{float(env.cost_team):.6e}",
        "final_j_postpone": f"{float(env.cost_postpone):.6e}",
        "final_j_revision": f"{float(env.cost_revision):.6e}",
        "postponed_count": sum(t.postpone_count for t in env.state.tasks.values()),
    }

    logger.info(
        f"{variant_label}: 总决策={total_decisions}, 显式推进={len(explicit_advances)} "
        f"(有立即开工机会={len(adv_with_immediate)}, 仅未来预约={len(adv_with_future_only)}, 无候选={len(adv_with_no_cand)}, 零时间无变化={len(zero_time_advances)}), "
        f"累计等待={round(cum_waiting_time, 2)}h, J_total={summary['final_j_total']}"
    )

    return summary, step_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    logger.info(f"运行推进诊断，计算设备: {device}")

    variants = [
        {
            "label": "C",
            "method": "C",
            "checkpoint": REPO_ROOT / "work3_4060/compare_v1/seed42/C/weights.pt",
        },
        {
            "label": "F",
            "method": "F",
            "checkpoint": REPO_ROOT / "work3_4060/mechanism_v1/seed42/F/weights.pt",
        },
        {
            "label": "D_beta01",
            "method": "D",
            "checkpoint": REPO_ROOT / "work3_4060/shaping_scale_v1/seed42/D_beta01/weights.pt",
        },
    ]

    summaries = []
    all_d_records = []

    for v in variants:
        sum_dict, recs = run_detailed_diagnostic(
            variant_label=v["label"],
            method_name=v["method"],
            checkpoint_path=v["checkpoint"],
            device=device,
        )
        summaries.append(sum_dict)
        if v["label"] == "D_beta01":
            all_d_records = recs

    # 输出表格与报告
    out_dir = REPO_ROOT / "work3_4060/shaping_scale_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "advance_diagnosis_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    logger.info(f"诊断汇总表已保存: {csv_path}")

    # 保存 D(beta=0.1) 的详细推进步骤
    d_adv_records = [r for r in all_d_records if r["is_explicit_advance"]]
    json_path = out_dir / "advance_decisions_d_beta01.json"
    json_path.write_text(json.dumps(d_adv_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(f"D_beta01 显式推进详细记录保存至: {json_path}")

    # 生成 Markdown 诊断报告
    md_path = out_dir / "advance_diagnosis_summary.md"
    lines = [
        "# Work-3 显式事件推进决策单变量物理与策略诊断报告 (Section 4)",
        "评估场景：Nominal (mini210.json, 210 工序) | 种子：42 | 决策模式：Greedy (deterministic=True, warmup=none)",
        "",
        "## 1. 核心诊断汇总表 (Diagnosis Summary)",
        "| 变体方法 | 总决策数 | 显式推进总数 | 推进前有立即开工机会 | 推进前仅未来预约机会 | 无候选推进 | 零时间无状态变化 | 累计等待工期 (h) | 累计推进费用增量 | 最终工期 (h) | 综合费用 J_total |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for s in summaries:
        lines.append(
            f"| **{s['variant']}** ({s['method']}) | {s['total_decisions']} | {s['explicit_advance_count']} | "
            f"{s['adv_with_immediate_opportunity']} | {s['adv_with_future_only_opportunity']} | "
            f"{s['adv_with_zero_candidates']} | {s['zero_time_no_state_change_count']} | "
            f"{s['cumulative_waiting_time_hours']} | {s['cumulative_advance_cost_increment']} | "
            f"{s['final_makespan']} | {s['final_j_total']} |"
        )

    lines.extend([
        "",
        "## 2. D_beta01 前 15 次显式推进微观决策切片 (First 15 Explicit Advance Steps)",
        "| 步骤 | 当前时刻 | 推进等待 dt | 候选总数 | 立即开工候选 | 仅未来预约候选 | 最早开工时刻 | 运行中工序 | 预约中工序 | 当步费用增量 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for r in d_adv_records[:15]:
        lines.append(
            f"| Step {r['step']} | {r['time_before']}h | +{r['dt']}h | {r['candidates_count']} | "
            f"**{r['immediate_count']}** | {r['future_only_count']} | {r['earliest_start_time']}h | "
            f"{r['running_before']} | {r['reserved_before']} | {r['d_cost']} |"
        )

    lines.extend([
        "",
        "## 3. 诊断结论与机制定性 (Qualitative Conclusion)",
    ])

    s_d = summaries[-1]
    if s_d["adv_with_immediate_opportunity"] > 0:
        lines.append(
            f"- **发现策略主动等待偏差**: 在 {s_d['explicit_advance_count']} 次显式推进中，有 **{s_d['adv_with_immediate_opportunity']} 次** 推进前现场存在立即可开工的工序，但策略选择主动推进到下一个离散事件。"
        )
    if s_d["adv_with_future_only_opportunity"] > 0:
        lines.append(
            f"- **发现合理资源/时序等待**: 有 **{s_d['adv_with_future_only_opportunity']} 次** 推进前虽然有候选工序，但受前序完工、物料时间或工人日历占用限制，无法当场立即开工（最早开工时间晚于当前时刻），策略选择等待事件到达。"
        )
    if s_d["zero_time_no_state_change_count"] == 0:
        lines.append(
            "- **排除环境推进/死循环 Bug**: 零时间且无状态变化的推进次数为 **0**，每次显式推进均伴随时间真实前进或离散事件处理，排除了无进展空转的代码逻辑缺陷。"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Markdown 诊断报告已生成: {md_path}")


if __name__ == "__main__":
    main()
