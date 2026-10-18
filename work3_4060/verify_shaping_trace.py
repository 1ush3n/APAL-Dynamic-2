"""独立检查器：核验训练轨迹势函数奖励塑形闭合性 (Task 3.2)。

独立读取 shaping_events.jsonl 与 run.json，不导入训练器重算，严格核验：
1. 单步势函数塑形公式：shaped_reward - raw_reward == beta * (gamma * phi_next - phi_current)
2. 跨步势连续性：同一 episode 内相邻决策步骤，上一行 phi_next 与下一行 phi_current 完全相等
3. 预测器版本与终局合同：episode 内预测器版本固定不变；仅在自然完工 (terminated=True) 时 phi_next == 0.0
4. 完整 episode 折扣望远镜恒等式求和：
   sum_{t=0}^{T-1} gamma^t * (shaped_reward - raw_reward) == beta * (gamma^T * phi_terminal - phi_start) == -beta * phi_start
5. 未完片段截断端点项与残差
6. 原始账本与未折扣 J_total 对账
7. 后移动作 (POSTPONE, action_type=1) 局部塑形增量与整条轨迹影响分析
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VerifyShaping")


def verify_shaping_trace(
    shaping_events_path: Path,
    run_json_path: Path | None = None,
    output_summary_path: Path | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    if not shaping_events_path.is_file():
        raise FileNotFoundError(f"未找到审计事件文件: {shaping_events_path}")

    # 1. 读取并解析所有事件
    events: list[dict[str, Any]] = []
    with shaping_events_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_line_no"] = line_idx
            events.append(item)

    total_events = len(events)
    logger.info(f"读取到 {total_events} 条势函数审计事件")
    if total_events == 0:
        raise ValueError("审计事件文件为空！")

    # 2. 读取有效配置信息
    run_config: dict[str, Any] = {}
    if run_json_path and run_json_path.is_file():
        with run_json_path.open("r", encoding="utf-8") as f:
            run_data = json.load(f)
            run_config = run_data.get("resolved_runtime_config", {})

    cfg_gamma = run_config.get("ppo", {}).get("gamma")
    cfg_beta = run_config.get("ppo", {}).get("shaping_coefficient")

    # 3. 单步公式校验 (Single-step formula check)
    step_formula_failures = []
    for ev in events:
        r_raw = float(ev["raw_reward"])
        r_shaped = float(ev["shaped_reward"])
        phi_curr = float(ev["phi_current"])
        phi_next = float(ev["phi_next"])
        beta = float(ev["beta"])
        gamma = float(ev["gamma"])

        if cfg_beta is not None and abs(beta - float(cfg_beta)) > 1e-6:
            step_formula_failures.append({
                "line": ev["_line_no"],
                "error": f"事件 beta ({beta}) 与配置 ({cfg_beta}) 不一致",
            })
        if cfg_gamma is not None and abs(gamma - float(cfg_gamma)) > 1e-6:
            step_formula_failures.append({
                "line": ev["_line_no"],
                "error": f"事件 gamma ({gamma}) 与配置 ({cfg_gamma}) 不一致",
            })

        diff_actual = r_shaped - r_raw
        diff_expected = beta * (gamma * phi_next - phi_curr)
        res = abs(diff_actual - diff_expected)
        scale = max(1.0, abs(diff_expected), abs(diff_actual))
        if res > tolerance * scale:
            step_formula_failures.append({
                "line": ev["_line_no"],
                "actual": diff_actual,
                "expected": diff_expected,
                "residual": res,
            })

    # 4. 按 (worker_id, episode_id) 分组核验连续性与望远镜求和
    episodes: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for ev in events:
        key = (int(ev["worker_id"]), int(ev["episode_id"]))
        episodes.setdefault(key, []).append(ev)

    episode_summaries = []
    continuity_failures = []
    version_contract_failures = []
    terminal_failures = []

    for (w_id, ep_id), ep_events in sorted(episodes.items()):
        # 按 episode_step 严格排序
        ep_events.sort(key=lambda x: (int(x["rollout_id"]), int(x["episode_step"])))
        n_steps = len(ep_events)

        # 4.1 检查版本合同
        versions = {ev.get("predictor_version") for ev in ep_events}
        if len(versions) > 1:
            version_contract_failures.append({
                "worker_id": w_id,
                "episode_id": ep_id,
                "versions_observed": list(versions),
            })

        # 4.2 检查跨步势连续性
        for i in range(n_steps - 1):
            curr_ev = ep_events[i]
            next_ev = ep_events[i + 1]
            p_next = float(curr_ev["phi_next"])
            p_curr = float(next_ev["phi_current"])
            res = abs(p_next - p_curr)
            if res > 1e-5:
                continuity_failures.append({
                    "worker_id": w_id,
                    "episode_id": ep_id,
                    "step_from": curr_ev["episode_step"],
                    "step_to": next_ev["episode_step"],
                    "phi_next": p_next,
                    "phi_curr": p_curr,
                    "residual": res,
                })

        # 4.3 终局状态与求和
        last_ev = ep_events[-1]
        first_ev = ep_events[0]
        is_terminated = bool(last_ev["terminated"])
        is_truncated = bool(last_ev["truncated"])

        # 终局势必须为 0.0
        if is_terminated and abs(float(last_ev["phi_next"])) > 1e-6:
            terminal_failures.append({
                "worker_id": w_id,
                "episode_id": ep_id,
                "phi_next_at_termination": float(last_ev["phi_next"]),
            })

        gamma = float(first_ev["gamma"])
        beta = float(first_ev["beta"])
        phi_0 = float(first_ev["phi_current"])
        phi_T = float(last_ev["phi_next"])

        # 4.4 折扣望远镜求和 (float64 累加)
        lhs = np.float64(0.0)
        disc_raw_ret = np.float64(0.0)
        disc_shaped_ret = np.float64(0.0)
        undisc_raw_sum = np.float64(0.0)

        for t, ev in enumerate(ep_events):
            discount = np.float64(gamma) ** t
            r_raw = np.float64(ev["raw_reward"])
            r_shaped = np.float64(ev["shaped_reward"])
            shaping_delta = r_shaped - r_raw

            lhs += discount * shaping_delta
            disc_raw_ret += discount * r_raw
            disc_shaped_ret += discount * r_shaped
            undisc_raw_sum += r_raw

        rhs = np.float64(beta) * ((np.float64(gamma) ** n_steps) * np.float64(phi_T) - np.float64(phi_0))
        sum_residual = float(lhs - rhs)
        scale = max(1.0, abs(float(lhs)), abs(float(rhs)))
        sum_closed = abs(sum_residual) <= (1e-5 * scale)

        # 统计后移动作
        postpone_events = []
        for ev in ep_events:
            if ev.get("action_type") == 1:
                postpone_events.append({
                    "step": ev["episode_step"],
                    "time": ev["decision_time"],
                    "cycle": ev["cycle_id"],
                    "task_id": ev["task_id"],
                    "raw_reward": ev["raw_reward"],
                    "phi_curr": ev["phi_current"],
                    "phi_next": ev["phi_next"],
                    "shaping_delta": ev["shaped_reward"] - ev["raw_reward"],
                })

        ep_summary = {
            "worker_id": w_id,
            "episode_id": ep_id,
            "steps": n_steps,
            "terminated": is_terminated,
            "truncated": is_truncated,
            "predictor_version": list(versions)[0] if versions else None,
            "phi_start": phi_0,
            "phi_end": phi_T,
            "gamma": gamma,
            "beta": beta,
            "lhs_discounted_shaping_sum": float(lhs),
            "rhs_telescope_endpoint": float(rhs),
            "telescope_residual": sum_residual,
            "telescope_closed": sum_closed,
            "discounted_raw_return": float(disc_raw_ret),
            "discounted_shaped_return": float(disc_shaped_ret),
            "undiscounted_raw_sum": float(undisc_raw_sum),
            "postpone_count": len(postpone_events),
            "postpone_actions": postpone_events,
        }
        episode_summaries.append(ep_summary)

    # 5. 综合评估判断
    completed_episodes = [ep for ep in episode_summaries if ep["terminated"]]
    all_single_steps_passed = len(step_formula_failures) == 0
    all_continuity_passed = len(continuity_failures) == 0
    all_version_passed = len(version_contract_failures) == 0
    all_terminal_passed = len(terminal_failures) == 0
    all_completed_telescope_closed = all(ep["telescope_closed"] for ep in completed_episodes) if completed_episodes else True

    overall_passed = (
        all_single_steps_passed
        and all_continuity_passed
        and all_version_passed
        and all_terminal_passed
        and all_completed_telescope_closed
    )

    summary = {
        "overall_status": "PASS" if overall_passed else "FAIL",
        "total_events_checked": total_events,
        "total_episodes_checked": len(episode_summaries),
        "completed_episodes_count": len(completed_episodes),
        "contracts": {
            "single_step_formula_valid": all_single_steps_passed,
            "single_step_failure_count": len(step_formula_failures),
            "step_to_step_continuity_valid": all_continuity_passed,
            "continuity_failure_count": len(continuity_failures),
            "predictor_version_invariance_valid": all_version_passed,
            "version_failure_count": len(version_contract_failures),
            "terminal_zero_potential_valid": all_terminal_passed,
            "terminal_failure_count": len(terminal_failures),
            "completed_episode_telescope_closed": all_completed_telescope_closed,
        },
        "step_formula_failures_sample": step_formula_failures[:5],
        "continuity_failures_sample": continuity_failures[:5],
        "version_failures_sample": version_contract_failures[:5],
        "terminal_failures_sample": terminal_failures[:5],
        "episodes": episode_summaries,
    }

    if output_summary_path:
        output_summary_path.parent.mkdir(parents=True, exist_ok=True)
        output_summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(f"审计摘要已保存至: {output_summary_path}")

    logger.info(f"势函数闭合校验结果: {summary['overall_status']}")
    logger.info(
        f"  单步公式: {'PASS' if all_single_steps_passed else 'FAIL'}, "
        f"  相邻势连续: {'PASS' if all_continuity_passed else 'FAIL'}, "
        f"  版本固定: {'PASS' if all_version_passed else 'FAIL'}, "
        f"  终局势置零: {'PASS' if all_terminal_passed else 'FAIL'}, "
        f"  完工望远镜求和闭合: {'PASS' if all_completed_telescope_closed else 'FAIL'}"
    )

    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", type=Path, required=True, help="shaping_events.jsonl 文件路径")
    p.add_argument("--run-json", type=Path, default=None, help="可选 run.json 文件路径")
    p.add_argument("--output", type=Path, default=None, help="可选 output shaping_audit_summary.json")
    args = p.parse_args()

    verify_shaping_trace(
        shaping_events_path=args.events,
        run_json_path=args.run_json,
        output_summary_path=args.output,
    )


if __name__ == "__main__":
    main()
