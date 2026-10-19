"""独立检查器：核验训练轨迹势函数奖励塑形闭合性与契约完整性 (Task 3.2 / 审计加固)。

独立读取 shaping_events.jsonl 与 run.json，不导入训练器重算，严格核验：
1. 单步势函数塑形公式：shaped_reward - raw_reward == beta * (gamma * phi_next - phi_current)
2. 跨步势连续性：同一 episode 内相邻决策步骤，上一行 phi_next 与下一行 phi_current 完全相等
3. 预测器版本与终局合同：episode 内预测器版本固定不变；仅在自然完工 (terminated=True) 时 phi_next == 0.0
4. 完整 episode 折扣望远镜恒等式求和：
   sum_{t=0}^{T-1} gamma^t * (shaped_reward - raw_reward) == beta * (gamma^T * phi_terminal - phi_start) == -beta * phi_start
5. 配置一致性：run.json 参数与事件流中 gamma/beta 严格一致，禁止中途篡改
6. 边界与 Bootstrap 合同：段尾结算记录存在且有限，预算截断与完工终局正确分离
7. 原始账本对账：sum(raw_reward) + J_total == 0 且 五分项之和 - J_total == 0
8. 状态与退出码合同：FAIL=2, INCOMPLETE=3, PASS=0；失败优先于不完整
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VerifyShaping")


def check_config_consistency(
    run_json_path: Path | None,
    events: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """核验 run.json 与事件流的 gamma / beta 配置一致性 (A1)。"""
    if run_json_path is None or not run_json_path.is_file():
        return (
            "INCOMPLETE",
            [{"error": f"未提供或未找到有效 run.json: {run_json_path}"}],
            {},
        )

    with run_json_path.open("r", encoding="utf-8") as f:
        run_data = json.load(f)

    training_cfg = run_data.get("training_config", {})
    cfg_gamma = training_cfg.get("gamma")
    cfg_beta = training_cfg.get("beta_shaping")
    failures: list[dict[str, Any]] = []

    # 交叉检查 resolved_runtime_config_yaml
    yaml_str = run_data.get("resolved_runtime_config_yaml")
    yaml_cfg = {}
    yaml_parse_error = False
    if yaml_str and isinstance(yaml_str, str):
        try:
            parsed = yaml.safe_load(yaml_str)
            if not isinstance(parsed, dict):
                failures.append({
                    "error": f"resolved_runtime_config_yaml 解析后不是字典: {type(parsed)}"
                })
                yaml_parse_error = True
            else:
                yaml_cfg = parsed
        except Exception as e:
            failures.append({
                "error": f"解析 resolved_runtime_config_yaml 失败: {e}"
            })
            yaml_parse_error = True

    yaml_ppo = yaml_cfg.get("ppo", {})
    yaml_gamma = yaml_ppo.get("gamma")
    yaml_beta = yaml_ppo.get("shaping_coefficient")

    if cfg_gamma is None or cfg_beta is None:
        failures.append({
            "error": f"training_config 中缺少参数: gamma={cfg_gamma}, beta_shaping={cfg_beta}"
        })
        return "INCOMPLETE", failures, {}

    if yaml_gamma is not None and abs(float(yaml_gamma) - float(cfg_gamma)) > 1e-6:
        failures.append({
            "error": f"training_config.gamma ({cfg_gamma}) 与 YAML ({yaml_gamma}) 不一致"
        })
    if yaml_beta is not None and abs(float(yaml_beta) - float(cfg_beta)) > 1e-6:
        failures.append({
            "error": f"training_config.beta_shaping ({cfg_beta}) 与 YAML ({yaml_beta}) 不一致"
        })

    # 核验每条 transition 事件
    ep_params: dict[tuple[int, int], tuple[float, float]] = {}
    for ev in events:
        if ev.get("record_type", "transition") != "transition":
            continue
        b = float(ev["beta"])
        g = float(ev["gamma"])
        if abs(b - float(cfg_beta)) > 1e-6:
            failures.append({
                "line": ev.get("_line_no"),
                "error": f"事件 beta ({b}) 与 run.json training_config.beta_shaping ({cfg_beta}) 不一致",
            })
        if abs(g - float(cfg_gamma)) > 1e-6:
            failures.append({
                "line": ev.get("_line_no"),
                "error": f"事件 gamma ({g}) 与 run.json training_config.gamma ({cfg_gamma}) 不一致",
            })
        ep_key = (int(ev["worker_id"]), int(ev["episode_id"]))
        if ep_key in ep_params:
            prev_g, prev_b = ep_params[ep_key]
            if abs(g - prev_g) > 1e-6 or abs(b - prev_b) > 1e-6:
                failures.append({
                    "line": ev.get("_line_no"),
                    "error": f"同一 episode {ep_key} 内 gamma/beta 发生变动: 前=({prev_g}, {prev_b}), 当前=({g}, {b})",
                })
        else:
            ep_params[ep_key] = (g, b)

    has_fail = (
        yaml_parse_error
        or any(
            "不一致" in f.get("error", "")
            or "发生变动" in f.get("error", "")
            or "不是字典" in f.get("error", "")
            or "失败" in f.get("error", "")
            for f in failures
        )
    )
    status = "FAIL" if has_fail else ("PASS" if not failures else "INCOMPLETE")
    resolved_params = {
        "cfg_gamma": float(cfg_gamma),
        "cfg_beta": float(cfg_beta),
        "yaml_gamma": float(yaml_gamma) if yaml_gamma is not None else None,
        "yaml_beta": float(yaml_beta) if yaml_beta is not None else None,
    }
    return status, failures, resolved_params


def verify_shaping_trace(
    shaping_events_path: Path,
    run_json_path: Path | None = None,
    output_summary_path: Path | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    if not shaping_events_path.is_file():
        raise FileNotFoundError(f"未找到审计事件文件: {shaping_events_path}")

    # 1. 读取并分类所有记录
    raw_lines: list[dict[str, Any]] = []
    transition_events: list[dict[str, Any]] = []
    segment_end_records: list[dict[str, Any]] = []
    episode_cost_summaries: list[dict[str, Any]] = []

    with shaping_events_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_line_no"] = line_idx
            raw_lines.append(item)
            rec_type = item.get("record_type", "transition")
            if rec_type == "segment_end":
                segment_end_records.append(item)
            elif rec_type == "episode_cost_summary":
                episode_cost_summaries.append(item)
            elif rec_type == "transition" or "raw_reward" in item:
                transition_events.append(item)

    total_transitions = len(transition_events)
    logger.info(
        f"读取到 {len(raw_lines)} 行记录: "
        f"{total_transitions} 条单步转换, "
        f"{len(segment_end_records)} 条段尾结算, "
        f"{len(episode_cost_summaries)} 条完工账本"
    )
    if total_transitions == 0:
        raise ValueError("审计事件文件中未包含任何单步转换事件！")

    # 2. 配置核验 (A1)
    config_status, config_failures, resolved_cfg = check_config_consistency(
        run_json_path=run_json_path,
        events=transition_events,
    )

    # 读取 run_data 以备账本核验
    run_data: dict[str, Any] = {}
    if run_json_path and run_json_path.is_file():
        with run_json_path.open("r", encoding="utf-8") as f:
            run_data = json.load(f)

    # 3. 单步转换公式与标量有限性校验
    step_formula_failures = []
    for ev in transition_events:
        line_no = ev["_line_no"]
        # 3.1 浮点数有限性核查 (A2)
        for k in ["raw_reward", "shaped_reward", "phi_current", "phi_next", "beta", "gamma"]:
            val = float(ev[k])
            if not math.isfinite(val):
                step_formula_failures.append({
                    "line": line_no,
                    "error": f"字段 {k} 非有限浮点数: {val}",
                })

        r_raw = float(ev["raw_reward"])
        r_shaped = float(ev["shaped_reward"])
        phi_curr = float(ev["phi_current"])
        phi_next = float(ev["phi_next"])
        beta = float(ev["beta"])
        gamma = float(ev["gamma"])

        diff_actual = r_shaped - r_raw
        diff_expected = beta * (gamma * phi_next - phi_curr)
        res = abs(diff_actual - diff_expected)
        scale = max(1.0, abs(diff_expected), abs(diff_actual))
        if res > tolerance * scale:
            step_formula_failures.append({
                "line": line_no,
                "actual": diff_actual,
                "expected": diff_expected,
                "residual": res,
            })

    # 4. 按 (worker_id, episode_id) 分组核验连续性、终局合同与望远镜求和
    episodes: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for ev in transition_events:
        key = (int(ev["worker_id"]), int(ev["episode_id"]))
        episodes.setdefault(key, []).append(ev)

    episode_summaries = []
    continuity_failures = []
    version_contract_failures = []
    terminal_failures = []

    for (w_id, ep_id), ep_events in sorted(episodes.items()):
        # 按照 (rollout_id, episode_step) 排序
        ep_events.sort(key=lambda x: (int(x.get("rollout_id", 0)), int(x.get("episode_step", 0))))
        n_steps = len(ep_events)

        # 4.1 检查步号严格连续 0, 1, ..., n-1 (A2)
        for idx, ev in enumerate(ep_events):
            actual_step = ev.get("episode_step")
            if actual_step != idx:
                continuity_failures.append({
                    "worker_id": w_id,
                    "episode_id": ep_id,
                    "line": ev.get("_line_no"),
                    "error": f"步号不连续: 期望索引 {idx}, 实际记录 episode_step={actual_step}",
                })

        # 4.2 检查版本合同与不可缺失性 (A2)
        versions = set()
        for ev in ep_events:
            ver = ev.get("predictor_version")
            src = ev.get("phi_source")
            if src == "learned_phi" and ver is None:
                version_contract_failures.append({
                    "worker_id": w_id,
                    "episode_id": ep_id,
                    "line": ev.get("_line_no"),
                    "error": "phi_source 为 learned_phi 但 predictor_version 为 null",
                })
            versions.add(ver)

        if len(versions) > 1:
            version_contract_failures.append({
                "worker_id": w_id,
                "episode_id": ep_id,
                "versions_observed": list(versions),
            })

        # 4.3 检查跨步势连续性
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
                    "step_from": curr_ev.get("episode_step"),
                    "step_to": next_ev.get("episode_step"),
                    "phi_next": p_next,
                    "phi_curr": p_curr,
                    "residual": res,
                })

        # 4.4 终局状态与位置核验 (A2)
        for idx in range(n_steps - 1):
            if ep_events[idx].get("terminated"):
                terminal_failures.append({
                    "worker_id": w_id,
                    "episode_id": ep_id,
                    "line": ep_events[idx].get("_line_no"),
                    "error": f"terminated=True 出现在非末尾步 (第 {idx}/{n_steps} 步)",
                })

        last_ev = ep_events[-1]
        first_ev = ep_events[0]
        is_terminated = bool(last_ev.get("terminated"))
        is_truncated = bool(last_ev.get("truncated"))

        if is_terminated and abs(float(last_ev["phi_next"])) > 1e-6:
            terminal_failures.append({
                "worker_id": w_id,
                "episode_id": ep_id,
                "line": last_ev.get("_line_no"),
                "error": f"完工步 phi_next 未归零: {float(last_ev['phi_next'])}",
            })

        gamma = float(first_ev["gamma"])
        beta = float(first_ev["beta"])
        phi_0 = float(first_ev["phi_current"])
        phi_T = float(last_ev["phi_next"])

        # 4.5 独立稳定望远镜累加 (float64 累加)
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
                    "step": ev.get("episode_step"),
                    "time": ev.get("decision_time"),
                    "cycle": ev.get("cycle_id"),
                    "task_id": ev.get("task_id"),
                    "raw_reward": ev.get("raw_reward"),
                    "phi_curr": ev.get("phi_current"),
                    "phi_next": ev.get("phi_next"),
                    "shaping_delta": ev.get("shaped_reward", 0.0) - ev.get("raw_reward", 0.0),
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

    completed_episodes = [ep for ep in episode_summaries if ep["terminated"]]
    completed_count = len(completed_episodes)

    # 4.6 望远镜求和子项状态判断 (A2: 零完工不可报 PASS)
    if completed_count == 0:
        telescope_status = "INCOMPLETE"
    else:
        all_closed = all(ep["telescope_closed"] for ep in completed_episodes)
        telescope_status = "PASS" if all_closed else "FAIL"

    expected_segments = {(int(ev["worker_id"]), int(ev["episode_id"]), int(ev.get("rollout_id", 0))) for ev in transition_events}
    expected_episodes = {(int(ev["worker_id"]), int(ev["episode_id"])) for ev in transition_events}
    expected_completed_episodes = {
        ep_key for ep_key, ep_evs in episodes.items()
        if any(bool(ev.get("terminated")) for ev in ep_evs)
    }

    # 5. 边界与 Bootstrap 合同校验 (A4)
    boundary_status = "INCOMPLETE"
    boundary_failures = []
    boundary_details: dict[str, Any] = {}

    if segment_end_records:
        seen_segments = set()
        duplicate_segments = set()
        for rec in segment_end_records:
            seg_key = (int(rec["worker_id"]), int(rec["episode_id"]), int(rec["rollout_id"]))
            if seg_key in seen_segments:
                duplicate_segments.add(seg_key)
            seen_segments.add(seg_key)

            b_val = rec.get("bootstrap_value")
            b_type = rec.get("boundary_type")
            if b_val is None:
                boundary_failures.append({"record": rec, "error": "缺少 bootstrap_value 字段"})
            else:
                try:
                    f_val = float(b_val)
                    if not math.isfinite(f_val):
                        boundary_failures.append({"record": rec, "error": f"bootstrap_value 非有限数: {b_val}"})
                    elif b_type == "natural_termination":
                        if abs(f_val) > 1e-6:
                            boundary_failures.append({
                                "record": rec,
                                "error": f"自然终止时 bootstrap_value ({b_val}) != 0.0",
                            })
                    elif b_type in ("budget_truncation", "rollout_continuation"):
                        pass
                    else:
                        boundary_failures.append({
                            "record": rec,
                            "error": f"未知 boundary_type: {b_type}",
                        })
                except (ValueError, TypeError):
                    boundary_failures.append({"record": rec, "error": f"bootstrap_value 无法解析为浮点数: {b_val}"})

        dangling_segments = seen_segments - expected_segments
        missing_segments = expected_segments - seen_segments

        if duplicate_segments:
            boundary_failures.append({
                "error": f"存在重复的 segment_end 记录: {sorted(list(duplicate_segments))}"
            })
        if dangling_segments:
            boundary_failures.append({
                "error": f"存在悬挂/未引用的 segment_end 记录: {sorted(list(dangling_segments))}"
            })

        boundary_details["expected_segments_count"] = len(expected_segments)
        boundary_details["observed_segments_count"] = len(segment_end_records)
        boundary_details["missing_segments"] = [list(k) for k in sorted(list(missing_segments))]
        boundary_details["dangling_segments"] = [list(k) for k in sorted(list(dangling_segments))]
        boundary_details["duplicate_segments"] = [list(k) for k in sorted(list(duplicate_segments))]

        if boundary_failures:
            boundary_status = "FAIL"
        elif missing_segments:
            boundary_status = "INCOMPLETE"
        else:
            boundary_status = "PASS"
    else:
        boundary_status = "INCOMPLETE"
        boundary_details["note"] = "未提供 segment_end 记录"
        boundary_details["expected_segments_count"] = len(expected_segments)
        boundary_details["observed_segments_count"] = 0
        boundary_details["missing_segments"] = [list(k) for k in sorted(list(expected_segments))]

    # 6. 原始账本对账 (A3)
    ledger_status = "INCOMPLETE"
    ledger_details: dict[str, Any] = {}
    if episode_cost_summaries:
        ledger_failures = []
        seen_ledgers = set()
        duplicate_ledgers = set()
        cost_fields = ["cost_takt", "cost_time", "cost_team", "cost_postpone", "cost_revision"]

        for ep_sum in episode_cost_summaries:
            ep_key = (int(ep_sum["worker_id"]), int(ep_sum["episode_id"]))
            if ep_key in seen_ledgers:
                duplicate_ledgers.add(ep_key)
            seen_ledgers.add(ep_key)

            # 五分项字段必须完整且为有限数
            for cf in cost_fields:
                if cf not in ep_sum:
                    ledger_failures.append({"episode": ep_key, "error": f"缺少五分项费用字段: {cf}"})
                else:
                    try:
                        val = float(ep_sum[cf])
                        if not math.isfinite(val):
                            ledger_failures.append({"episode": ep_key, "error": f"五分项字段 {cf} 非有限数: {ep_sum[cf]}"})
                    except (ValueError, TypeError):
                        ledger_failures.append({"episode": ep_key, "error": f"五分项字段 {cf} 无法解析为浮点数: {ep_sum[cf]}"})

            if "cumulative_cost" not in ep_sum:
                ledger_failures.append({"episode": ep_key, "error": "缺少 cumulative_cost 字段"})
            else:
                try:
                    val = float(ep_sum["cumulative_cost"])
                    if not math.isfinite(val):
                        ledger_failures.append({"episode": ep_key, "error": f"cumulative_cost 非有限数: {val}"})
                except (ValueError, TypeError):
                    ledger_failures.append({"episode": ep_key, "error": f"cumulative_cost 无法解析为浮点数: {ep_sum['cumulative_cost']}"})

            ep_evs = episodes.get(ep_key, [])
            if ep_evs:
                raw_sum = sum(float(ev["raw_reward"]) for ev in ep_evs)
                j_tot = float(ep_sum.get("cumulative_cost", 0.0))
                c_takt = float(ep_sum.get("cost_takt", 0.0))
                c_time = float(ep_sum.get("cost_time", 0.0))
                c_team = float(ep_sum.get("cost_team", 0.0))
                c_post = float(ep_sum.get("cost_postpone", 0.0))
                c_rev = float(ep_sum.get("cost_revision", 0.0))

                res_raw = abs(raw_sum + j_tot)
                if res_raw > 1e-5:
                    ledger_failures.append({
                        "episode": ep_key,
                        "error": f"raw_sum + J_total 残差 {res_raw} > 1e-5 (raw_sum={raw_sum}, J_total={j_tot})",
                    })
                comp_sum = c_takt + c_time + c_team + c_post + c_rev
                res_comp = abs(comp_sum - j_tot)
                if res_comp > 1e-5:
                    ledger_failures.append({
                        "episode": ep_key,
                        "error": f"五分项和 - J_total 残差 {res_comp} > 1e-5 (comp_sum={comp_sum}, J_total={j_tot})",
                    })

        dangling_ledgers = seen_ledgers - expected_episodes
        missing_completed_ledgers = expected_completed_episodes - seen_ledgers

        if duplicate_ledgers:
            ledger_failures.append({
                "error": f"存在重复的 episode_cost_summary 记录: {sorted(list(duplicate_ledgers))}"
            })
        if dangling_ledgers:
            ledger_failures.append({
                "error": f"存在悬挂/未引用的 episode_cost_summary 记录: {sorted(list(dangling_ledgers))}"
            })

        ledger_details["expected_completed_episodes_count"] = len(expected_completed_episodes)
        ledger_details["observed_ledger_count"] = len(episode_cost_summaries)
        ledger_details["missing_completed_episodes"] = [list(k) for k in sorted(list(missing_completed_ledgers))]
        ledger_details["dangling_episodes"] = [list(k) for k in sorted(list(dangling_ledgers))]
        ledger_details["duplicate_episodes"] = [list(k) for k in sorted(list(duplicate_ledgers))]

        if ledger_failures:
            ledger_status = "FAIL"
            ledger_details["failures"] = ledger_failures
        elif missing_completed_ledgers:
            ledger_status = "INCOMPLETE"
        else:
            ledger_status = "PASS"
            ledger_details["reconciled_episodes"] = len(episode_cost_summaries)
    elif run_data and "trajectory_audit" in run_data:
        traj_audit = run_data["trajectory_audit"]
        traj_j_tot = traj_audit.get("cumulative_cost")
        if traj_j_tot is not None and episodes:
            last_key = list(episodes.keys())[-1]
            last_ep_evs = episodes[last_key]
            raw_sum = sum(float(ev["raw_reward"]) for ev in last_ep_evs)
            diff = abs(raw_sum + float(traj_j_tot))
            if diff <= 1e-5:
                ledger_status = "INCOMPLETE"
                ledger_details["tail_episode_verified"] = {
                    "episode": last_key,
                    "raw_sum": raw_sum,
                    "trajectory_audit_cumulative_cost": traj_j_tot,
                    "diff": diff,
                    "note": "尾部片段与环境 cumulative_cost 对账完全吻合 (差值 <= 1e-5)；前序批次缺少独立账本，标记为 INCOMPLETE",
                }
            else:
                ledger_status = "FAIL"
                ledger_details["tail_episode_verified"] = {
                    "episode": last_key,
                    "raw_sum": raw_sum,
                    "trajectory_audit_cumulative_cost": traj_j_tot,
                    "diff": diff,
                    "error": "尾部片段 raw_sum + J_total 残差超出容差！",
                }
        else:
            ledger_status = "INCOMPLETE"
            ledger_details["note"] = "trajectory_audit 中缺少有效 cumulative_cost"
    else:
        ledger_status = "INCOMPLETE"
        ledger_details["note"] = "未提供完工账本记录，且 run.json 中无有效 trajectory_audit"

    # 7. 各子项状态汇总与总体裁定 (A2: FAIL优先于INCOMPLETE)
    single_step_status = "PASS" if len(step_formula_failures) == 0 else "FAIL"
    continuity_status = "PASS" if len(continuity_failures) == 0 else "FAIL"
    version_status = "PASS" if len(version_contract_failures) == 0 else "FAIL"
    terminal_status = "PASS" if len(terminal_failures) == 0 else "FAIL"

    sub_statuses = [
        single_step_status,
        continuity_status,
        version_status,
        terminal_status,
        telescope_status,
        config_status,
        boundary_status,
        ledger_status,
    ]

    if "FAIL" in sub_statuses:
        overall_status = "FAIL"
    elif "INCOMPLETE" in sub_statuses:
        overall_status = "INCOMPLETE"
    else:
        overall_status = "PASS"

    summary = {
        "overall_status": overall_status,
        "total_events_checked": total_transitions,
        "total_episodes_checked": len(episode_summaries),
        "completed_episodes_count": completed_count,
        "contracts": {
            "single_step_formula": single_step_status,
            "step_to_step_continuity": continuity_status,
            "predictor_version_invariance": version_status,
            "terminal_zero_potential": terminal_status,
            "completed_episode_telescope": telescope_status,
            "config_consistency": config_status,
            "boundary_contract": boundary_status,
            "ledger_reconciliation": ledger_status,
        },
        "resolved_config": resolved_cfg,
        "contract_counts": {
            "single_step_failures": len(step_formula_failures),
            "continuity_failures": len(continuity_failures),
            "version_failures": len(version_contract_failures),
            "terminal_failures": len(terminal_failures),
            "boundary_failures": len(boundary_failures),
        },
        "failures_sample": {
            "step_formula": step_formula_failures[:5],
            "continuity": continuity_failures[:5],
            "version": version_contract_failures[:5],
            "terminal": terminal_failures[:5],
            "config": config_failures[:5],
            "boundary": boundary_failures[:5],
        },
        "boundary_details": boundary_details,
        "ledger_details": ledger_details,
        "episodes": episode_summaries,
    }

    if output_summary_path:
        output_summary_path.parent.mkdir(parents=True, exist_ok=True)
        output_summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(f"审计摘要已保存至: {output_summary_path}")

    logger.info(f"势函数闭合校验裁定: {overall_status}")
    logger.info(
        f"  单步公式: {single_step_status}, "
        f"  相邻连续: {continuity_status}, "
        f"  版本固定: {version_status}, "
        f"  终局置零: {terminal_status}, "
        f"  望远镜求和: {telescope_status}, "
        f"  配置一致: {config_status}, "
        f"  边界截断: {boundary_status}, "
        f"  账本对账: {ledger_status}"
    )

    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", type=Path, required=True, help="shaping_events.jsonl 文件路径")
    p.add_argument("--run-json", type=Path, default=None, help="可选 run.json 文件路径")
    p.add_argument("--output", type=Path, default=None, help="可选 output shaping_audit_summary.json")
    args = p.parse_args()

    summary = verify_shaping_trace(
        shaping_events_path=args.events,
        run_json_path=args.run_json,
        output_summary_path=args.output,
    )

    status = summary.get("overall_status")
    if status == "FAIL":
        sys.exit(2)
    elif status == "INCOMPLETE":
        sys.exit(3)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
