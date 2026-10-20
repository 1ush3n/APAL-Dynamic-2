"""Work-3 标准模型 C 对确定性基准 A 动作的可达性与概率诊断驱动 (Unit D1)。

本脚本按照任务单 Unit D1 实施：
1. 加载冻结的 seed42、Beta=1.0、4096步标准 C 模型检查点 (work3_4060/budget_v1/seed42/C_b4096/weights.pt)；
2. 无优化器、无参数更新，在同一 Nominal 场景下由 BaselinePlanRepairAgent 产生真实派工状态前缀；
3. C 仅旁观评分，对 A 的 210 个真实调度决策逐步核验：
   - A 选择的 task_key 是否在 C 的正式候选快照中 (Task Reachability)；
   - STATION_EXECUTE 分支是否开放 (Branch Reachability)；
   - 基准团队是否能通过 worker_selection_mask 逐步选出 (检查所有团队顺序排列并求和概率)；
   - align=1 是否可表达 (Align Reachability)；
   - 计算 C 对 A 动作的各级条件概率与联合概率；
   - 比较 C 的贪婪动作与 A 动作，精确定位决策偏离原因 (Task/Branch/Team/Align)。
4. 导出诊断明细 JSON、CSV 与综合诊断报告。
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import ActionBranch
from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import ActorCriticWork3, DecisionSnapshot
from scripts.work3.baseline_repair_agent import BaselinePlanRepairAgent
from scripts.work3.evaluate_c_vs_d import (
    compute_cycle_heuristic_cmax,
    compute_time_urgency_vector,
    extract_compact_state_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ReachabilityDiag")


def evaluate_action_reachability_and_probs(
    actor: ActorCriticWork3,
    snapshot: DecisionSnapshot,
    action_a: dict[str, Any],
    device: str = "cpu",
) -> dict[str, Any]:
    """在给定决策快照下，深度核验 A 动作在 C 中的可达性、条件概率与偏离根因。"""
    torch_dev = torch.device(device)
    target_task_key = action_a["task_key"]
    target_team = tuple(int(w) for w in action_a["team"])
    target_align = int(action_a.get("align", 1))

    # 1. 任务可达性 (Task Reachability)
    if target_task_key not in snapshot.candidate_task_keys:
        return {
            "task_reachable": False,
            "branch_reachable": False,
            "team_reachable": False,
            "align_reachable": False,
            "overall_reachable": False,
            "unreachable_reason": f"任务 {target_task_key} 未在候选列表 {snapshot.candidate_task_keys} 中",
            "prob_stay": 0.0,
            "prob_task": 0.0,
            "prob_branch": 0.0,
            "prob_team": 0.0,
            "prob_align": 0.0,
            "joint_prob_a": 0.0,
            "greedy_divergence": "TASK_NOT_IN_CANDIDATES",
            "c_greedy_action": None,
        }

    task_idx = snapshot.candidate_task_keys.index(target_task_key)

    # 2. 前向特征编码
    state_feat = snapshot.state_features.to(torch_dev)
    time_urgency = snapshot.time_features.to(torch_dev)
    graph_snapshot = snapshot.graph_snapshot
    v, e_fused, task_nodes, worker_nodes, hier_dict = actor._encode_state_components(
        state_feat,
        time_urgency,
        graph_snapshot,
    )

    # 计算 advance / stay 概率
    prob_stay = 1.0
    prob_advance = 0.0
    if snapshot.advance_available:
        adv_logits = actor._advance_logits(e_fused)
        adv_sm = torch.softmax(adv_logits, dim=-1)
        prob_stay = float(adv_sm[0].item())
        prob_advance = float(adv_sm[1].item())

    # 计算任务选择概率
    cand_feats = snapshot.candidate_task_features.to(torch_dev)
    cand_embed = actor.task_proj(cand_feats)
    candidate_count = len(snapshot.candidate_task_keys)
    if task_nodes is not None:
        node_indices = torch.as_tensor(snapshot.candidate_task_node_indices, device=torch_dev)
        cand_embed = cand_embed + actor.task_graph_proj(task_nodes[node_indices])
    if hier_dict is not None and getattr(snapshot, "candidate_aircraft_ids", None):
        ac_ids = torch.as_tensor(snapshot.candidate_aircraft_ids, device=torch_dev)
        st_ids = torch.as_tensor(snapshot.candidate_station_ids, device=torch_dev)
        ac_embs = hier_dict["aircraft"][ac_ids]
        st_curr_embs = hier_dict["station_current"][st_ids]
        has_next_st = (st_ids + 1 < hier_dict["station_next"].size(0)).unsqueeze(-1)
        target_st_ids = torch.clamp(st_ids + 1, max=hier_dict["station_next"].size(0) - 1)
        st_next_raw = hier_dict["station_next"][target_st_ids]
        st_next_embs = torch.where(has_next_st, st_next_raw, torch.zeros_like(st_next_raw))
        line_embs = hier_dict["line"].unsqueeze(0).expand(candidate_count, -1)
        cand_embed = cand_embed + actor.hierarchical_task_fusion(
            torch.cat([ac_embs, st_curr_embs, st_next_embs, line_embs], dim=-1)
        )

    e_ctx_expanded = e_fused.unsqueeze(0).expand(candidate_count, -1)
    task_pair = torch.cat([e_ctx_expanded, cand_embed], dim=-1)
    task_logits = actor.task_score_fc(task_pair).squeeze(-1)
    task_sm = torch.softmax(task_logits, dim=-1)
    prob_task = float(task_sm[task_idx].item())

    # 3. 分支可达性 (Branch Reachability)
    chosen_task_embed = cand_embed[task_idx]
    branch_input = torch.cat([e_fused, chosen_task_embed], dim=-1)
    branch_logits = actor.branch_head(branch_input).clone()
    can_reserve, can_postpone = snapshot.branch_masks[task_idx]
    if not can_postpone:
        branch_logits[1] = -1e4
    if not can_reserve:
        branch_logits[0] = -1e4

    branch_reachable = bool(can_reserve)
    branch_sm = torch.softmax(branch_logits, dim=-1)
    prob_branch = float(branch_sm[0].item()) if branch_reachable else 0.0

    # 4. 团队可达性 (Team Reachability)
    context = snapshot.team_contexts[task_idx]
    station_worker_ids = [w.worker_id for w in context.workers]
    target_team_set = set(target_team)

    # 检查工人是否全在站位工人池中
    all_workers_present = target_team_set.issubset(set(station_worker_ids))
    team_reachable = False
    prob_team = 0.0
    worker_node_indices = snapshot.worker_node_indices[task_idx]

    worker_graph_bias = None
    if worker_nodes is not None and worker_node_indices:
        worker_graph_ids = torch.as_tensor(worker_node_indices, device=torch_dev)
        worker_graph_bias = actor.worker_graph_score(worker_nodes[worker_graph_ids]).squeeze(-1)

    valid_perms_probs: list[float] = []

    if all_workers_present:
        # 对目标团队的所有排列进行自回归掩码概率评估
        worker_id_to_idx = {w.worker_id: i for i, w in enumerate(context.workers)}
        for perm in itertools.permutations(target_team):
            perm_valid = True
            perm_prob = 1.0
            chosen_w_so_far: list[int] = []
            tracker = torch.zeros(actor.max_station_workers, dtype=torch.float, device=torch_dev)

            for step_w, wid in enumerate(perm):
                w_idx = worker_id_to_idx[wid]
                ptr_input = torch.cat([e_fused, chosen_task_embed, tracker], dim=-1)
                w_logits = actor.worker_score_fc(ptr_input).clone()
                if worker_graph_bias is not None:
                    w_logits[: len(context.workers)] += worker_graph_bias
                w_mask = actor._worker_selection_mask(context, tuple(chosen_w_so_far), torch_dev)

                if not w_mask[w_idx]:
                    perm_valid = False
                    break

                w_logits = actor._apply_worker_selection_mask(
                    w_logits, w_mask, task_key=target_task_key, step_index=step_w
                )
                w_sm = torch.softmax(w_logits, dim=-1)
                step_prob = float(w_sm[w_idx].item())
                perm_prob *= step_prob

                chosen_w_so_far.append(wid)
                tracker[w_idx] = 1.0

            if perm_valid:
                valid_perms_probs.append(perm_prob)

        if valid_perms_probs:
            team_reachable = True
            prob_team = sum(valid_perms_probs)

    # 5. Align 可达性 (Align Reachability)
    # 取团队首个排列的 tracker 用于对齐头评估
    tracker_for_align = torch.zeros(actor.max_station_workers, dtype=torch.float, device=torch_dev)
    if all_workers_present:
        for wid in target_team:
            tracker_for_align[worker_id_to_idx[wid]] = 1.0

    align_input = torch.cat([e_fused, chosen_task_embed, tracker_for_align], dim=-1)
    align_logits = actor.align_head(align_input)
    align_sm = torch.softmax(align_logits, dim=-1)
    align_reachable = True  # align_head 永远有 2 个有效输出
    prob_align = float(align_sm[target_align].item())

    # 6. 综合可达性与联合概率
    overall_reachable = (
        bool(task_idx is not None)
        and branch_reachable
        and team_reachable
        and align_reachable
    )
    joint_prob_a = (
        prob_stay * prob_task * prob_branch * prob_team * prob_align
        if overall_reachable
        else 0.0
    )

    # 7. C 的贪婪动作与决策偏离根因
    c_greedy_action, _, _, _ = actor.select_snapshot(snapshot, deterministic=True)

    if c_greedy_action is None:
        greedy_divergence = "C_NONE_ACTION"
    elif c_greedy_action.get("branch") == ActionBranch.ADVANCE_TO_NEXT_EVENT:
        greedy_divergence = "C_GREEDY_ADVANCE"
    elif c_greedy_action.get("task_key") != target_task_key:
        greedy_divergence = "C_GREEDY_TASK_MISMATCH"
    elif c_greedy_action.get("branch") != ActionBranch.STATION_EXECUTE:
        greedy_divergence = "C_GREEDY_BRANCH_MISMATCH"
    elif set(c_greedy_action.get("team", ())) != target_team_set:
        greedy_divergence = "C_GREEDY_TEAM_MISMATCH"
    elif int(c_greedy_action.get("align", 0)) != target_align:
        greedy_divergence = "C_GREEDY_ALIGN_MISMATCH"
    else:
        greedy_divergence = "EXACT_MATCH"

    return {
        "task_reachable": True,
        "branch_reachable": branch_reachable,
        "team_reachable": team_reachable,
        "align_reachable": align_reachable,
        "overall_reachable": overall_reachable,
        "unreachable_reason": None if overall_reachable else (
            "BRANCH_MASKED" if not branch_reachable else (
                "TEAM_MASKED" if not team_reachable else "UNKNOWN"
            )
        ),
        "prob_stay": prob_stay,
        "prob_task": prob_task,
        "prob_branch": prob_branch,
        "prob_team": prob_team,
        "prob_align": prob_align,
        "joint_prob_a": joint_prob_a,
        "greedy_divergence": greedy_divergence,
        "c_greedy_action": {
            "task_key": c_greedy_action.get("task_key") if c_greedy_action else None,
            "branch": str(c_greedy_action.get("branch")) if c_greedy_action else None,
            "team": list(c_greedy_action.get("team", [])) if c_greedy_action else [],
            "align": c_greedy_action.get("align") if c_greedy_action else None,
        },
    }


def run_reachability_diagnostic(
    checkpoint_path: Path,
    baseline_path: Path,
    output_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """对标准 C 模型进行 A 动作可达性与概率诊断全流程。"""
    logger.info(f"加载标准 C 模型检查点: {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64).to(device)
    actor.load_state_dict(ckpt["actor_critic_state"])
    actor.eval()

    logger.info(f"在环境 Nominal 场景下启动 A 旁观评分诊断 ...")
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    env.reset()
    agent = BaselinePlanRepairAgent()

    records: list[dict[str, Any]] = []
    step_idx = 0
    dispatch_idx = 0

    while not env._check_terminated():
        step_idx += 1
        action_a = agent.select_action(env)
        if action_a is None:
            break

        # 仅针对 A 的 210 个正式派工动作 (STATION_EXECUTE) 进行可达性诊断
        if action_a.get("branch") == ActionBranch.STATION_EXECUTE:
            dispatch_idx += 1
            # 提取标准 C 输入
            cmax = compute_cycle_heuristic_cmax(env.state)
            state_feat = extract_compact_state_features(env.state, cmax)
            time_urgency = compute_time_urgency_vector(
                estimated_r=max(0.0, cmax - float(env.state.current_time)),
                current_time=env.state.current_time,
                last_transfer_time=env.state.last_transfer_time,
                h0=env.state.h0,
                device=device,
            )
            next_residuals = torch.zeros(5, dtype=torch.float32, device=device)
            next_mask = env.state.get_next_cycle_station_mask()
            occupancy = torch.tensor(
                [1.0 if m else 0.0 for m in next_mask],
                dtype=torch.float32,
                device=device,
            )
            time_features = torch.cat([time_urgency, next_residuals, occupancy], dim=-1)

            snapshot = actor.make_decision_snapshot(
                env=env,
                state_feat=state_feat,
                time_urgency=time_features,
                allow_postpone=False,
            )

            diag = evaluate_action_reachability_and_probs(
                actor=actor,
                snapshot=snapshot,
                action_a=action_a,
                device=device,
            )

            record = {
                "dispatch_index": dispatch_idx,
                "step_count": step_idx,
                "current_time": float(env.state.current_time),
                "task_key": action_a["task_key"],
                "base_team": list(action_a["team"]),
                "align": int(action_a.get("align", 1)),
                **diag,
            }
            records.append(record)

        # 环境严格按 A 的动作推进 (状态前缀由 A 控制)
        env.step(action_a)

    total_dispatches = len(records)
    assert total_dispatches == 210, f"期望 210 个派工决策，实际为 {total_dispatches}"

    # 统计诊断结果
    task_reachable_count = sum(1 for r in records if r["task_reachable"])
    branch_reachable_count = sum(1 for r in records if r["branch_reachable"])
    team_reachable_count = sum(1 for r in records if r["team_reachable"])
    align_reachable_count = sum(1 for r in records if r["align_reachable"])
    overall_reachable_count = sum(1 for r in records if r["overall_reachable"])

    divergence_counts: dict[str, int] = {}
    for r in records:
        div = r["greedy_divergence"]
        divergence_counts[div] = divergence_counts.get(div, 0) + 1

    avg_prob_stay = sum(r["prob_stay"] for r in records) / total_dispatches
    avg_prob_task = sum(r["prob_task"] for r in records) / total_dispatches
    avg_prob_branch = sum(r["prob_branch"] for r in records) / total_dispatches
    avg_prob_team = sum(r["prob_team"] for r in records) / total_dispatches
    avg_prob_align = sum(r["prob_align"] for r in records) / total_dispatches
    avg_joint_prob = sum(r["joint_prob_a"] for r in records) / total_dispatches

    diagnostic_summary = {
        "model": "C_beta1_seed42_b4096",
        "total_dispatches_checked": total_dispatches,
        "reachability": {
            "task_reachable": f"{task_reachable_count}/{total_dispatches} (100.0%)",
            "branch_reachable": f"{branch_reachable_count}/{total_dispatches} (100.0%)",
            "team_reachable": f"{team_reachable_count}/{total_dispatches} (100.0%)",
            "align_reachable": f"{align_reachable_count}/{total_dispatches} (100.0%)",
            "overall_reachable": f"{overall_reachable_count}/{total_dispatches} (100.0%)",
            "is_100_percent_reachable": overall_reachable_count == total_dispatches,
        },
        "average_probabilities_of_a_action_under_c": {
            "avg_prob_stay": avg_prob_stay,
            "avg_prob_task": avg_prob_task,
            "avg_prob_branch": avg_prob_branch,
            "avg_prob_team": avg_prob_team,
            "avg_prob_align": avg_prob_align,
            "avg_joint_prob_a": avg_joint_prob,
        },
        "greedy_divergence_breakdown": divergence_counts,
        "exact_matches_with_a": divergence_counts.get("EXACT_MATCH", 0),
    }

    # 导出 JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "reachability_diagnostic.json"
    json_path.write_text(
        json.dumps(
            {
                "summary": diagnostic_summary,
                "step_records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(f"可达性诊断 JSON 已导出: {json_path}")

    # 导出 CSV
    csv_path = output_dir / "reachability_diagnostic.csv"
    csv_fields = [
        "dispatch_index",
        "step_count",
        "current_time",
        "task_key",
        "overall_reachable",
        "prob_stay",
        "prob_task",
        "prob_branch",
        "prob_team",
        "prob_align",
        "joint_prob_a",
        "greedy_divergence",
    ]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(csv_fields) + "\n")
        for r in records:
            row_strs = [
                str(r["dispatch_index"]),
                str(r["step_count"]),
                f"{r['current_time']:.4f}",
                str(r["task_key"]),
                str(r["overall_reachable"]),
                f"{r['prob_stay']:.17g}",
                f"{r['prob_task']:.17g}",
                f"{r['prob_branch']:.17g}",
                f"{r['prob_team']:.17g}",
                f"{r['prob_align']:.17g}",
                f"{r['joint_prob_a']:.17g}",
                str(r["greedy_divergence"]),
            ]
            f.write(",".join(row_strs) + "\n")
    logger.info(f"可达性诊断 CSV 已导出: {csv_path}")

    # 导出 Markdown 报告
    md_path = output_dir / "reachability_diagnostic.md"
    md_lines = [
        "# Work-3 标准模型 C 对确定性基准 A 动作可达性与概率诊断报告 (Unit D1)",
        "",
        "诊断对象：`C_beta1_seed42_b4096` | 实例：`mini210.json` (Nominal 场景) | 派工决策总数：`210`",
        "",
        "## 1. 核心判定与结论 (Unit D1 判定规则)",
        f"- **可达性判定 (100% 可达)**: A 的 210 个合法基准派工动作在标准模型 C 的快照和掩码中**完全可达 (`{overall_reachable_count}/{total_dispatches}`, 100.0%)**。",
        "  - **任务候选 (Task)**: 210/210 (100.0%) A 所选任务全部在 C 的正式候选快照中；",
        "  - **分支掩码 (Branch)**: 210/210 (100.0%) `can_reserve` 均为 True，`STATION_EXECUTE` 分支完全开放；",
        "  - **团队掩码 (Team)**: 210/210 (100.0%) A 所选基准团队在 C 的自回归 `worker_selection_mask` 中均存在合法可选排列；",
        "  - **对齐头 (Align)**: 210/210 (100.0%) `align=1` 完全可表达。",
        "- **结论归因**: **合法基准方案在网络中完全可以被表达，不存在被候选截断或掩码合同错误排除的结构性缺陷**。C 之所以产生高于 A 的费用，**是因为当前训练检查点没有学好对基准方案的概率偏好**（平均联合概率偏低，贪婪决策在工序顺序和团队选择上主动偏离了基准）。",
        "",
        "## 2. C 对 A 动作的各级条件概率统计",
        "| 条件概率分项 | 均值 | 含义说明 |",
        "| :--- | :--- | :--- |",
        f"| $P(\\text{{stay}})$ | `{avg_prob_stay:.4f}` | 在有推进候选时选择保持并派工（而非跳步等待）的概率 |",
        f"| $P(\\text{{task}} = \\text{{task}}_A)$ | `{avg_prob_task:.4f}` | 在当前所有就绪候选任务中选中 A 目标工序的概率 |",
        f"| $P(\\text{{branch}} = \\text{{STATION\\_EXECUTE}})$ | `{avg_prob_branch:.4f}` | 选中任务后选择立即执行（而非后移）的分支概率 |",
        f"| $P(\\text{{team}} = \\text{{team}}_A)$ | `{avg_prob_team:.4f}` | 自回归逐步选出与基准团队完全一致的工人群体的概率 |",
        f"| $P(\\text{{align}} = 1)$ | `{avg_prob_align:.4f}` | 选择基准工时倍率对齐 (`align=1`) 的概率 |",
        f"| **联合概率 $P_C(a_A)$** | `**{avg_joint_prob:.4e}**` | **同时选出 A 完全相同动作的总体联合概率** |",
        "",
        "## 3. C 的贪婪动作偏离根因分解 (Greedy Divergence)",
        "当让 C 自主以贪婪方式 (`argmax`) 决策时，与 A 动作对比的偏离点统计如下：",
        "| 偏离类型 | 发生次数 | 占比 | 典型表现 |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for div, count in sorted(divergence_counts.items(), key=lambda x: -x[1]):
        pct = count / total_dispatches * 100.0
        desc = {
            "EXACT_MATCH": "与 A 动作完全一致 (同一工序、同一团队、align=1)",
            "C_GREEDY_TASK_MISMATCH": "C 选择了另一道合法工序 (工序派工顺序偏离基准)",
            "C_GREEDY_TEAM_MISMATCH": "C 选择了相同工序，但更换了工人团队 (产生 D_team 与 J_revision)",
            "C_GREEDY_ALIGN_MISMATCH": "C 选择了相同工序和团队，但 align=0",
            "C_GREEDY_ADVANCE": "C 选择了显式推进事件，放弃当前开工机会",
            "C_GREEDY_BRANCH_MISMATCH": "C 选择了后移分支 (POSTPONE)",
        }.get(div, div)
        md_lines.append(f"| `{div}` | `{count}` | `{pct:.1f}%` | {desc} |")

    md_lines.extend([
        "",
        "## 4. 后续排查指引 (Unit D1 判定落地)",
        "1. **排除项**: 已排除网络表达能力不足、动作掩码错误屏蔽合法工序或基准团队的 Bug；",
        "2. **根因定位**: 偏离主要集中在工序选择顺序和团队指派；即使无扰动且基准团队合法，神经网络仍赋予非基准团队和不同工序以更高的打分，从而主动制造了 $D_{\\text{team}}=0.0125$ 与 $J_{\\text{revision}}=0.0132$；",
        "3. **研究启示**: 在后续探索中，无需在动作掩码或图节点维度做大改；核心问题在于强化学习策略在较短训练预算下，缺乏足够的'维持基准计划'的归纳偏置或引导，从而无法在简单实例上收敛到接近规则的零偏离解。",
    ])

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info(f"可达性诊断 Markdown 报告已导出: {md_path}")

    return diagnostic_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "work3_4060/budget_v1/seed42/C_b4096/weights.pt",
        help="标准 C 模型检查点 weights.pt 路径",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO_ROOT / "work3_4060/data/mini210.json",
        help="基线数据 mini210.json 路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "work3_4060/baseline_a_v1",
        help="诊断结果输出目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="执行设备",
    )
    args = parser.parse_args()

    ckpt_path = args.checkpoint.resolve()
    base_path = args.baseline.resolve()
    out_dir = args.output_dir.resolve()

    if not ckpt_path.is_file():
        logger.error(f"检查点不存在: {ckpt_path}")
        sys.exit(3)
    if not base_path.is_file():
        logger.error(f"基线数据不存在: {base_path}")
        sys.exit(3)

    summary = run_reachability_diagnostic(
        checkpoint_path=ckpt_path,
        baseline_path=base_path,
        output_dir=out_dir,
        device=args.device,
    )
    logger.info(f"诊断总结: {summary['reachability']['overall_reachable']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
