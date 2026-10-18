"""独立复核 R3/E04 专项证据链检查器。

不依赖训练端直接写入的 passed 布尔值，而是从原始文件独立读取：
1. run.json：检查基础合同、批次完工与无非有限数；
2. r3_e04_events.jsonl：独立还原时序事件流，验证单步递增与各阶段事件顺序；
3. r3_e04_inputs.pt：检查原始输入特征；
4. 独立执行 (P_q - est_cmax)/H0 与 (F[m, q+1] - P_q - H0)/H0 理论数值重算并比对误差；
5. 验证冻结模型 SHA-256 不变性、在线模型参数变化以及重复推理最大误差 <= 1e-6。

退出码规范：
- 0: PASS (至少一个真实案例贯通 R3 与 E04 且各项数值检查通过)
- 2: FAIL (断言失败、数值超差、冻结状态改变、优化失败等)
- 3: INCOMPLETE (覆盖不足：未跨转站、未在转站间更新、未产生有效 next-only 等)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError, OSError):
                pass

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, nargs="?", help="Directory or summary JSON path")
    p.add_argument("--latest", help="Find latest run directory for stage, e.g. pilot-d")
    p.add_argument("--tol", type=float, default=1e-5, help="Tolerance for numerical recomputation")
    args = p.parse_args()

    if args.latest:
        found = list((Path(__file__).resolve().parent / "outputs").glob(args.latest + "_*/run.json"))
        if not found:
            raise SystemExit("No run.json found for --latest " + args.latest)
        run_file = max(found, key=lambda f: f.stat().st_mtime)
        run_dir = run_file.parent
    elif args.path is not None:
        p_in = args.path.resolve()
        run_dir = p_in if p_in.is_dir() else p_in.parent
    else:
        raise SystemExit("Provide a path or --latest STAGE")

    errors: list[str] = []
    incomplete: list[str] = []
    notes: list[str] = []

    # 1. 核查必要文件存在性
    run_json_path = run_dir / "run.json"
    events_path = run_dir / "r3_e04_events.jsonl"
    inputs_path = run_dir / "r3_e04_inputs.pt"
    summary_path = run_dir / "r3_e04_summary.json"

    if not run_json_path.is_file():
        errors.append("run.json 不存在")
    if not events_path.is_file():
        incomplete.append("r3_e04_events.jsonl 缺失（可能未开启 R3/E04 审计开关）")
    if not inputs_path.is_file():
        incomplete.append("r3_e04_inputs.pt 缺失")
    if not summary_path.is_file():
        incomplete.append("r3_e04_summary.json 缺失")

    if incomplete:
        result = {
            "status": "INCOMPLETE",
            "run_dir": str(run_dir),
            "errors": errors,
            "incomplete": incomplete,
            "notes": ["请使用开启审计开关的命令重新运行短试训"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(3)

    if errors:
        result = {
            "status": "FAIL",
            "run_dir": str(run_dir),
            "errors": errors,
            "incomplete": incomplete,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    # 2. 读取基础 run.json
    run_data = json.loads(run_json_path.read_text(encoding="utf-8"))
    if run_data.get("lightning_optimization_steps", 0) <= 0:
        errors.append("run.json 未记录优化步")
    if run_data.get("time_supervision_optimizer_updates", 0) <= 0:
        errors.append("run.json 未记录时间辅助优化")

    # 3. 读取事件流
    events: list[dict[str, Any]] = []
    with events_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                events.append(ev)
            except Exception as e:
                errors.append(f"events.jsonl 第 {idx} 行解析失败: {e}")

    # 检查序号单调性
    for i in range(1, len(events)):
        if events[i]["seq"] <= events[i - 1]["seq"]:
            errors.append(f"事件序号非严格递增: seq[{i}]={events[i]['seq']} <= seq[{i-1}]={events[i-1]['seq']}")

    # 4. 加载原始输入快照
    try:
        inputs_pt = torch.load(inputs_path, map_location="cpu", weights_only=False)
        tracked_inputs = inputs_pt.get("tracked_inputs", {})
    except Exception as e:
        errors.append(f"加载 r3_e04_inputs.pt 失败: {e}")
        tracked_inputs = {}

    # 5. 读取训练端生成的 summary 作为参照
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))

    # 6. 独立还原并验证 R3 证据链
    # 提取周期事实
    transfer_facts: dict[tuple[int, int, int], dict[str, Any]] = {}
    for ev in events:
        if ev["event_type"] == "ATTACH_TRANSFER":
            key = (ev["worker_id"], ev["episode_id"], ev["cycle_id"])
            transfer_facts[key] = ev

    # 提取所有在线参数更新事件序号
    online_update_seqs: list[int] = [
        ev["seq"] for ev in events
        if ev["event_type"] in ("PPO_OPTIMIZER_UPDATE", "AUX_MINIBATCH_OPTIMIZER_UPDATE")
    ]

    # 提取所有 drain 事件
    drain_records: list[dict[str, Any]] = []
    for ev in events:
        if ev["event_type"] == "DRAIN_READY_BATCH":
            ev_seq = ev["seq"]
            for item in ev.get("items", []):
                it = dict(item)
                it["seq"] = ev_seq
                drain_records.append(it)

    # 提取所有辅助 mini-batch 优化事件
    aux_opt_events: list[dict[str, Any]] = [
        ev for ev in events
        if ev["event_type"] == "AUX_MINIBATCH_OPTIMIZER_UPDATE"
    ]

    # 逐个独立核查 tracked_inputs 中的候选样本
    candidate_results: list[dict[str, Any]] = []
    for sample_id, inp in tracked_inputs.items():
        w = inp["worker_id"]
        ep = inp["episode_id"]
        cyc = inp["source_cycle_id"]
        dec = inp["decision_id"]
        est_cmax = inp["estimated_cmax"]
        h0 = inp["h0"]

        # 6.1 寻找登记事件
        reg_ev = next((ev for ev in events if ev["event_type"] == "REGISTER_DECISION" and ev.get("sample_id") == sample_id), None)
        if reg_ev is None:
            continue

        # 6.2 寻找首次取出（current-only）
        d_curr = next(
            (d for d in drain_records if d["sample_id"] == sample_id and d["current_target_mask"] is True),
            None,
        )
        # 6.3 寻找再次取出（next-only）
        d_next = next(
            (d for d in drain_records if d["sample_id"] == sample_id and d["current_target_mask"] is False and any(d["next_cycle_masks"])),
            None,
        )

        if d_curr is None or d_next is None:
            continue

        # 6.4 核查两次取出之间是否发生了在线更新
        updates_between = [
            s for s in online_update_seqs
            if d_curr["seq"] < s < d_next["seq"]
        ]

        fact_1 = transfer_facts.get((w, ep, cyc))
        fact_2 = transfer_facts.get((w, ep, cyc + 1))

        if not (fact_1 and fact_2):
            continue

        # 6.5 寻找包含该样本的 current-only 与 next-only 辅助优化步
        # 6.5.1 current-only: 优化发生在 d_curr 之后且在 fact_2 (下一周期转站) 到达之前
        curr_aux_opts: list[dict[str, Any]] = []
        for ev in aux_opt_events:
            s_ids = ev.get("sample_ids", [])
            if sample_id in s_ids:
                idx = s_ids.index(sample_id)
                c_mask = ev.get("current_masks", [])
                if idx < len(c_mask) and c_mask[idx] is True:
                    if d_curr["seq"] < ev["seq"] < fact_2["seq"] and ev.get("update_successful"):
                        curr_aux_opts.append(ev)

        # 6.5.2 next-only: 优化发生在 d_next 之后，行掩码 current_target_mask 为 False 且 any(next_cycle_masks)
        next_aux_opts: list[dict[str, Any]] = []
        for ev in aux_opt_events:
            s_ids = ev.get("sample_ids", [])
            if sample_id in s_ids:
                idx = s_ids.index(sample_id)
                c_mask = ev.get("current_masks", [])
                n_mask = ev.get("next_masks", [])
                is_curr_f = (idx < len(c_mask) and c_mask[idx] is False)
                is_next_t = (idx < len(n_mask) and any(n_mask[idx]))
                if is_curr_f and is_next_t and ev["seq"] > d_next["seq"] and ev.get("update_successful"):
                    # 检查行级误差或全局误差
                    row_errs = ev.get("row_next_max_errors")
                    if row_errs is not None and idx < len(row_errs) and row_errs[idx] is not None:
                        if row_errs[idx] > 1e-4:
                            next_aux_opts.append(ev)
                    else:
                        if ev.get("next_error_max", 0.0) > 1e-4:
                            next_aux_opts.append(ev)

        # 6.6 独立数值重算

        p1 = fact_1["actual_transfer_time"]
        theoretical_curr = (p1 - est_cmax) / h0
        actual_curr = d_curr["target_residual"]
        diff_curr = abs(actual_curr - theoretical_curr)
        tol_curr = args.tol + args.tol * abs(theoretical_curr)

        # 下一周期重算
        next_cleared = fact_2.get("station_first_cleared_times", {})
        next_mask = fact_2.get("next_cycle_station_mask", [False] * 5)
        diff_next_max = 0.0
        theoretical_next_list = [0.0] * 5

        for m in range(5):
            if m < len(next_mask) and next_mask[m]:
                # 键可能为 int 或 str
                f_m = next_cleared.get(m, next_cleared.get(str(m)))
                if f_m is not None:
                    ref_m = (float(f_m) - p1 - h0) / h0
                    theoretical_next_list[m] = ref_m
                    act_m = d_next["target_next_residuals"][m]
                    diff_m = abs(act_m - ref_m)
                    if diff_m > diff_next_max:
                        diff_next_max = diff_m

        is_valid_case = bool(
            len(updates_between) >= 1
            and len(curr_aux_opts) >= 1
            and len(next_aux_opts) >= 1
            and diff_curr <= tol_curr
            and diff_next_max <= (args.tol + args.tol * 10.0)
            and d_next["current_target_mask"] is False
            and any(d_next["next_cycle_masks"])
        )

        candidate_results.append({
            "sample_id": sample_id,
            "decision_id": dec,
            "valid": is_valid_case,
            "updates_between_count": len(updates_between),
            "curr_aux_opt_count": len(curr_aux_opts),
            "next_aux_opt_count": len(next_aux_opts),
            "diff_curr": diff_curr,
            "diff_next_max": diff_next_max,
            "next_masks": d_next["next_cycle_masks"],
        })

    # 7. 独立核查 E04 冻结预测器
    e04_checks = [
        ev for ev in events
        if ev["event_type"] == "POST_ROLLOUT_FIT_CHECK"
    ]
    if not e04_checks:
        incomplete.append("未记录 E04 冻结预测器更新前后核查事件")

    e04_valid = False
    if e04_checks:
        active_checks = [c for c in e04_checks if c.get("is_episode_active", True) is True]
        if not active_checks:
            incomplete.append("未在活跃绑定的 episode 中捕获到 E04 核查记录")

        frozen_matched = all(
            c.get("frozen_actor_sha_match") and c.get("frozen_head_sha_match") and c.get("frozen_eval_mode")
            for c in active_checks
        )
        online_changed = any(c.get("online_params_changed") for c in active_checks)

        # 检查 requires_grad 是否为 False（若字段存在）
        frozen_requires_grad_ok = all(
            c.get("frozen_requires_grad_false", True) for c in active_checks
        )

        max_inf_diff = 0.0
        has_inference_diffs = False
        for c in active_checks:
            diffs = c.get("inference_diffs", {})
            if diffs:
                has_inference_diffs = True
                for v in diffs.values():
                    if isinstance(v, (int, float)) and v > max_inf_diff:
                        max_inf_diff = v

        if not frozen_matched:
            errors.append("E04 失败：冻结预测器权重在在线更新前后发生变动或未处于 eval 模式！")
        if not online_changed:
            errors.append("E04 失败：在线模型参数未发生变化（无真实优化）！")
        if not frozen_requires_grad_ok:
            errors.append("E04 失败：冻结预测器参数 requires_grad 不为 False！")
        if not has_inference_diffs:
            incomplete.append("E04 不完整：未记录基准决策推理差异")
        elif max_inf_diff > 1e-6:
            errors.append(f"E04 失败：同一输入下冻结模型推理输出最大差异为 {max_inf_diff} > 1e-6！")

        # 若存在保存的代表性 CPU 模型状态，独立计算 SHA-256 并重算推理差异
        rep_e04 = inputs_pt.get("e04_representative")
        if rep_e04 is not None:
            from training.work3_r3_e04_audit import compute_state_dict_sha256
            head_state = rep_e04.get("frozen_head_state_dict")
            head_post = rep_e04.get("frozen_head_state_dict_post")
            if head_state is not None and head_post is not None:
                sha_pre = compute_state_dict_sha256(head_state)
                sha_post = compute_state_dict_sha256(head_post)
                if sha_pre != sha_post:
                    errors.append(f"E04 独立复核失败：代表性冻结 head state_dict SHA 不匹配 ({sha_pre} != {sha_post})")
            actor_state = rep_e04.get("frozen_actor_state_dict")
            actor_post = rep_e04.get("frozen_actor_state_dict_post")
            if actor_state is not None and actor_post is not None:
                sha_pre_a = compute_state_dict_sha256(actor_state)
                sha_post_a = compute_state_dict_sha256(actor_post)
                if sha_pre_a != sha_post_a:
                    errors.append(f"E04 独立复核失败：代表性冻结 actor state_dict SHA 不匹配 ({sha_pre_a} != {sha_post_a})")

            if rep_e04.get("frozen_head_requires_grad", False) or rep_e04.get("frozen_actor_requires_grad", False):
                errors.append("E04 独立复核失败：代表性冻结预测器 requires_grad 为 True")

        e04_valid = bool(frozen_matched and online_changed and frozen_requires_grad_ok and has_inference_diffs and max_inf_diff <= 1e-6 and not any("E04" in e for e in errors))

    # 8. 综合判定
    valid_r3_cases = [c for c in candidate_results if c["valid"]]
    if not valid_r3_cases:
        if not candidate_results:
            incomplete.append("未能捕获到跨周期的候选样本（可能试训步数不足以完成两周期）")
        else:
            errors.append("存在跨周期样本，但未完全满足 R3 证据链约束（请检查数值重算或在线更新介入）")

    status = "PASS" if (valid_r3_cases and e04_valid and not errors) else "INCOMPLETE" if incomplete else "FAIL"

    output = {
        "status": status,
        "run_dir": str(run_dir),
        "r3_valid_cases_count": len(valid_r3_cases),
        "r3_cases_detail": valid_r3_cases[:3],
        "e04_checks_count": len(e04_checks),
        "e04_valid": e04_valid,
        "errors": errors,
        "incomplete": incomplete,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    raise SystemExit(0 if status == "PASS" else 3 if status == "INCOMPLETE" else 2)


if __name__ == "__main__":
    main()
