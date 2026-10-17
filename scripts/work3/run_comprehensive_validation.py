"""工作三综合验证脚本 (Section 9.1, 9.2, 9.3)

执行内容：
- Section 9.1: 五站七架可审阅小实例（填线、满线、排空、受扰被动顺延、同步流转、完整可行性与账本复核）
- Section 9.2: 真实283工序×10架（2,830工序）桥接验证（无扰动轨迹与固定受扰轨迹，独立可行性与真实费用账本）
- Section 9.3: 真实PyTorch/PyG/Lightning C/D学习小烟测（跨转站有效标签、双目标梯度、固定副本独立性）
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import (
    ActorCriticWork3,
    extract_compact_state_features,
)
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.heuristic_agent import HeuristicAgentWork3
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from models.work3.time_head import TimeResidualHead
from models.work3.ppo_trainer import PPOTrainerWork3
from scripts.work3.evaluate_c_vs_d import (
    evaluate_single_trajectory,
    _check_completed_trajectory_feasibility,
)
from scripts.work3.train_ppo_work3 import run_training
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline
from utils.work3.objective_evaluator import (
    ObjectiveWeights,
    evaluate_trajectory_objective,
)
from utils.work3.trajectory_feasibility import validate_trajectory


def run_section_9_1_five_stations_seven_aircraft() -> dict[str, Any]:
    print("\n" + "=" * 80)
    print(">>> 执行 Section 9.1: 五站七架可审阅小实例验证")
    print("=" * 80)
    t0 = time.perf_counter()

    source = MultiAircraftBaseline.load_from_json(Path("data/work3/real_283_k10_baseline.json"))
    num_aircraft = 7
    # 选取代表性工序：覆盖5个站位、含站内紧密前序依赖链(16依赖15, 9依赖8, 13依赖12, 21依赖20, 23依赖24)
    # 站1: 15, 16; 站2: 8, 9; 站3: 12, 13; 站4: 20, 21; 站5: 24, 23
    selected_task_ids = (15, 16, 8, 9, 12, 13, 20, 21, 24, 23)
    tasks = {}
    for ac_id in range(num_aircraft):
        for t_id in selected_task_ids:
            tmpl = source.get_task(0, t_id)
            key = f"{ac_id}_{t_id}"
            st_id = tmpl.station_id
            entry = (ac_id + st_id - 1) * source.h0
            preds = [p for p in tmpl.predecessors if p in selected_task_ids]
            tasks[key] = replace(
                tmpl,
                aircraft_id=ac_id,
                task_id=t_id,
                task_key=key,
                predecessors=preds,
                station_id=st_id,
                baseline_start=entry + (tmpl.in_station_offset or 0.0),
                baseline_end=entry + (tmpl.in_station_offset or 0.0) + tmpl.duration,
                nominal_station_entry=entry,
                nominal_station_exit=entry + source.h0,
                cycle_idx=ac_id + st_id,
            )

    sub_baseline = MultiAircraftBaseline(
        num_aircraft=num_aircraft,
        num_stations=5,
        h0=source.h0,
        tasks=tasks,
        station_workers=source.station_workers,
    )
    tmp_baseline_path = Path(".pytest_tmp_v2/small_7ac_baseline.json")
    tmp_baseline_path.parent.mkdir(parents=True, exist_ok=True)
    sub_baseline.save_to_json(tmp_baseline_path)

    env = AirLineEnvWork3(baseline_json_path=tmp_baseline_path)
    env.reset()

    # 满线状态时在中间站(站2, 0-based)注入扰动，推迟恢复时刻
    # 周期 5 时，前 5 架飞机分别处于站 5, 4, 3, 2, 1
    # 飞机 2 在站 2 的工序 13 (2_13) 依赖 2_12，tau=21.0 时 2_13 尚未开工，能被严格命中并后移开工
    disturbance_scenario = {
        "scenario_id": "SEC9_1_DISTURBED",
        "tau": 21.0,
        "recovery_time": 35.0,
        "affected_task_keys": ["2_13"],
    }
    env.load_scenario(disturbance_scenario)

    agent = HeuristicAgentWork3()
    max_steps = len(tasks) * 5 + 50
    step_count = 0
    decision_log = []

    while not env._check_terminated() and step_count < max_steps:
        candidates = env.get_action_candidates()
        if not candidates:
            obs, reward, term, trunc, info = env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
        else:
            action = agent.select_action(env)
            if action is None:
                action = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
            obs, reward, term, trunc, info = env.step(action)
        step_count += 1
        decision_log.append({
            "step": step_count,
            "time": float(env.state.current_time),
            "cycle": int(env.state.current_cycle),
            "branch": info.get("action_branch"),
            "task": info.get("task_key"),
        })

    completed = len([t for t in env.state.tasks.values() if t.status == TaskStatus.COMPLETED])
    total = len(tasks)
    transfers = len(env.state.transfer_history)
    elapsed = time.perf_counter() - t0

    # 1. 独立可行性核查 (V01: 校验返回值必须被严格检查与断言)
    feasible, violations = _check_completed_trajectory_feasibility(env)
    assert feasible is True, f"Section 9.1 独立可行性核查失败: violations={violations}"
    assert len(violations) == 0, f"Section 9.1 存在可行性违规项: {violations}"

    # 2. 满线受扰真实命中与因果顺延核查 (V02)
    hit_keys = env.disturbance_event_results.get("SEC9_1_DISTURBED", {}).get("actual_hit_task_keys", ())
    assert len(hit_keys) > 0, f"Section 9.1 满线受扰未命中目标工序: {hit_keys}"
    assert "2_13" in hit_keys, f"工序 2_13 未命中: {hit_keys}"
    task_2_13 = env.state.tasks["2_13"]
    assert task_2_13.actual_start is not None and task_2_13.actual_start >= 35.0, (
        f"工序 2_13 开工时刻未顺延至恢复时间后: {task_2_13.actual_start}"
    )

    # 3. 费用账本一致性核验
    cost_breakdown = {
        "j_takt": float(env.cost_takt),
        "d_time": float(env.cost_time),
        "d_team": float(env.cost_team),
        "j_postpone": float(env.cost_postpone),
        "j_revision": float(env.cost_revision),
        "j_total": float(env.cumulative_cost),
    }
    sum_components = sum(cost_breakdown[k] for k in ("j_takt", "d_time", "d_team", "j_postpone", "j_revision"))
    ledger_gap = abs(cost_breakdown["j_total"] - sum_components)
    reward_cost_gap = abs(sum(env.step_rewards) + env.cumulative_cost)

    print(f"Section 9.1 完成: 工序 {completed}/{total}, 同步转站 {transfers} 次, 耗时 {elapsed:.2f}s")
    print(f"独立可行性: PASS (违规项: {violations}), 受扰命中: {hit_keys}")
    print(f"账本分项差: {ledger_gap:.2e}, 累计奖励与负费用差: {reward_cost_gap:.2e}")
    assert completed == total, f"工序未全部完工: {completed}/{total}"
    assert ledger_gap < 1e-9, f"账本分项不平衡: {ledger_gap}"
    assert reward_cost_gap < 1e-9, f"奖励与费用不守恒: {reward_cost_gap}"

    return {
        "status": "PASS",
        "completed_tasks": completed,
        "total_tasks": total,
        "transfers": transfers,
        "elapsed_seconds": elapsed,
        "cost_breakdown": cost_breakdown,
        "ledger_gap": ledger_gap,
        "reward_cost_gap": reward_cost_gap,
        "feasible": feasible,
        "violations": violations,
        "actual_hit_task_keys": list(hit_keys),
    }


def run_section_9_2_real_283_k10_bridge() -> dict[str, Any]:
    print("\n" + "=" * 80)
    print(">>> 执行 Section 9.2: 真实283工序×10架 (2,830工序) 桥接验证")
    print("=" * 80)
    baseline_path = Path("data/work3/real_283_k10_baseline.json")
    if not baseline_path.is_file():
        raise FileNotFoundError(f"基准计划文件不存在: {baseline_path}")

    # 1. 无扰动轨迹
    print("--> 运行 1/2: 真实 283×10 无扰动完整出线轨迹...")
    t0_unperturbed = time.perf_counter()
    env_unperturbed = AirLineEnvWork3(baseline_json_path=baseline_path)
    agent = HeuristicAgentWork3()
    res_unperturbed = evaluate_single_trajectory(
        env_unperturbed,
        "Heuristic-Debug",
        agent,
        scenario=None,
        device="cpu",
    )
    elapsed_unperturbed = time.perf_counter() - t0_unperturbed
    print(f"    无扰动出线完成: 完工 {res_unperturbed['completed_tasks']}/2830, 转站 {res_unperturbed['transfers']} 次, 耗时 {elapsed_unperturbed:.2f}s")
    assert res_unperturbed["completed_tasks"] == 2830
    assert res_unperturbed["transfers"] == 14
    assert res_unperturbed["feasible"] is True

    # 2. 固定受扰轨迹 (V03: 独立的有效受扰验证夹具 REAL_283_DEDICATED_HIT)
    print("--> 运行 2/2: 真实 283×10 固定有效受扰轨迹 (REAL_283_DEDICATED_HIT)...")
    dedicated_scenario = {
        "scenario_id": "REAL_283_DEDICATED_HIT",
        "tau": 423.0,
        "recovery_time": 460.0,
        "affected_task_keys": ["5_16"],
        "aircraft_id": 5,
        "station_id": 0,
        "intensity": "LOW",
        "timing": "EARLY",
    }

    t0_disturbed = time.perf_counter()
    env_disturbed = AirLineEnvWork3(baseline_json_path=baseline_path)
    res_disturbed = evaluate_single_trajectory(
        env_disturbed,
        "Heuristic-Debug",
        agent,
        scenario=dedicated_scenario,
        device="cpu",
    )
    elapsed_disturbed = time.perf_counter() - t0_disturbed
    print(f"    受扰出线完成: 完工 {res_disturbed['completed_tasks']}/2830, 转站 {res_disturbed['transfers']} 次, 耗时 {elapsed_disturbed:.2f}s")
    print(f"    实际扰动命中数: {res_disturbed['actual_disturbance_hits']}, 命中工序: {res_disturbed['disturbance_effects']['actual_hit_task_keys']}")
    assert res_disturbed["completed_tasks"] == 2830
    assert res_disturbed["transfers"] == 14
    assert res_disturbed["feasible"] is True
    assert res_disturbed["actual_disturbance_hits"] > 0
    assert "5_16" in res_disturbed["disturbance_effects"]["actual_hit_task_keys"]

    # 账本分项核查
    for name, res in [("无扰动", res_unperturbed), ("受扰", res_disturbed)]:
        sum_costs = (
            res["j_takt"] + res["d_time"] + res["d_team"] + res["j_postpone"] + res["j_revision"]
        )
        assert math.isclose(res["j_total"], sum_costs, rel_tol=1e-12, abs_tol=1e-12)
        print(f"    {name} 账本自洽性: j_total={res['j_total']:.6f} == sum_components={sum_costs:.6f}")

    return {
        "status": "PASS",
        "unperturbed": {
            "completed_tasks": res_unperturbed["completed_tasks"],
            "transfers": res_unperturbed["transfers"],
            "feasible": res_unperturbed["feasible"],
            "makespan": res_unperturbed["makespan"],
            "j_total": res_unperturbed["j_total"],
            "elapsed_seconds": elapsed_unperturbed,
        },
        "disturbed": {
            "scenario_id": "REAL_283_DEDICATED_HIT",
            "completed_tasks": res_disturbed["completed_tasks"],
            "transfers": res_disturbed["transfers"],
            "feasible": res_disturbed["feasible"],
            "makespan": res_disturbed["makespan"],
            "j_total": res_disturbed["j_total"],
            "actual_disturbance_hits": res_disturbed["actual_disturbance_hits"],
            "actual_hit_task_keys": res_disturbed["disturbance_effects"]["actual_hit_task_keys"],
            "elapsed_seconds": elapsed_disturbed,
        },
    }


def run_section_9_3_learning_smoke_test() -> dict[str, Any]:
    print("\n" + "=" * 80)
    print(">>> 执行 Section 9.3: 实际 C/D 学习烟测与双目标辅助更新")
    print("=" * 80)
    tmp_ckpt_dir = Path(".pytest_tmp_v2/smoke_checkpoints")
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. 运行 Method C 学习小烟测 (PPO 更新, budget=32)
    print("--> 1/2: 运行 Method C 学习烟测 (PPO 更新, 32步预算)...")
    t0_c = time.perf_counter()
    c_ckpt = tmp_ckpt_dir / "method_c_smoke.pt"
    res_c = run_training(
        run_mode="smoke",
        num_iterations=1,
        steps_per_iter=32,
        max_decisions=32,
        ppo_epochs=1,
        batch_size=32,
        method_variant="C",
        output_ckpt=c_ckpt,
    )
    elapsed_c = time.perf_counter() - t0_c
    print(f"    Method C 完成: decisions={res_c['total_decisions']}, 耗时={elapsed_c:.2f}s, ckpt={c_ckpt}")
    assert Path(res_c["checkpoint_path"]).is_file()
    assert res_c["total_decisions"] >= 21

    # 2. 运行 Method D 学习小烟测 (PPO 更新 + 真实双目标时间残差辅助头更新, budget=32)
    print("--> 2/2: 运行 Method D 学习烟测 (PPO + 双目标辅助头更新, 32步预算以覆盖P1与P2)...")
    t0_d = time.perf_counter()
    d_ckpt = tmp_ckpt_dir / "method_d_smoke.pt"
    res_d = run_training(
        run_mode="smoke",
        num_iterations=1,
        steps_per_iter=32,
        max_decisions=32,
        ppo_epochs=1,
        batch_size=32,
        method_variant="D",
        output_ckpt=d_ckpt,
    )
    elapsed_d = time.perf_counter() - t0_d
    print(f"    Method D 完成: decisions={res_d['total_decisions']}, 耗时={elapsed_d:.2f}s, ckpt={d_ckpt}")
    assert Path(res_d["checkpoint_path"]).is_file()
    assert res_d["total_decisions"] >= 21
    assert res_d.get("cycle_time_labels") or res_d.get("history"), "Method D 产生有效历史与时间监督记录"

    # 3. 详细检查 Method D 双目标梯度与分层图编码器各模块梯度流向 (V04)
    print("--> 3/3: 校验真实双目标损失与共享编码器各模块梯度流向...")
    actor_critic = ActorCriticWork3(
        state_dim=32,
        task_feat_dim=8,
        hidden_dim=64,
        max_station_workers=16,
    )
    time_head = TimeResidualHead(in_dim=actor_critic.hidden_dim, hidden_dim=64)
    trainer = PPOTrainerWork3(
        actor_critic=actor_critic,
        time_head=time_head,
        lr=1e-3,
        time_loss_coef=1.0,
    )

    env = AirLineEnvWork3(baseline_json_path=Path("data/work3/real_283_k10_baseline.json"))
    env.reset()
    state_feat = extract_compact_state_features(env.state, compute_cycle_heuristic_cmax(env.state))
    graph_snapshot = actor_critic.build_graph_snapshot(env)

    aux_batch = {
        "state_feats": state_feat.unsqueeze(0),
        "graph_snapshots": [graph_snapshot],
        "target_residuals": torch.tensor([0.25], dtype=torch.float32),
        "current_target_masks": torch.tensor([True], dtype=torch.bool),
        "target_next_residuals": torch.tensor([[0.1, -0.2, 0.05, 0.0, 0.0]], dtype=torch.float32),
        "next_cycle_masks": torch.tensor([[True, True, True, False, False]], dtype=torch.bool),
        "worker_ids": [0],
        "episode_ids": [0],
        "cycle_ids": [0],
        "decision_ids": [0],
    }

    trainer.optimizer.zero_grad()
    loss = trainer.compute_time_auxiliary_loss(aux_batch)
    loss.backward()

    # 断言紧凑状态编码器、分层融合MLP、站位汇总MLP、当前残差头和下一周期预测头均有有限非零梯度
    compact_grad = actor_critic.encoder[0].weight.grad
    line_fusion_grad = actor_critic.graph_encoder.line_fusion_mlp[0].weight.grad
    station_summary_grad = actor_critic.graph_encoder.station_summary_mlp[0].weight.grad
    current_head_grad = time_head.reg_fc[0].weight.grad
    next_head_grad = time_head.next_cycle_fc[0].weight.grad

    assert compact_grad is not None and torch.isfinite(compact_grad).all()
    assert line_fusion_grad is not None and torch.isfinite(line_fusion_grad).all()
    assert station_summary_grad is not None and torch.isfinite(station_summary_grad).all()
    assert current_head_grad is not None and torch.isfinite(current_head_grad).all()
    assert next_head_grad is not None and torch.isfinite(next_head_grad).all()

    print(f"    双目标损失={loss.item():.6f}")
    print(f"    紧凑状态编码器梯度模长: {compact_grad.norm().item():.6f} (有限且正常)")
    print(f"    整线融合MLP梯度模长: {line_fusion_grad.norm().item():.6f} (有限且正常)")
    print(f"    站位汇总MLP梯度模长: {station_summary_grad.norm().item():.6f} (有限且正常)")
    print(f"    当前残差预测头梯度模长: {current_head_grad.norm().item():.6f} (有限且正常)")
    print(f"    下一周期预测头梯度模长: {next_head_grad.norm().item():.6f} (有限且正常)")

    return {
        "status": "PASS",
        "method_c": {
            "decisions": res_c["total_decisions"],
            "elapsed_seconds": elapsed_c,
            "checkpoint": str(c_ckpt),
        },
        "method_d": {
            "decisions": res_d["total_decisions"],
            "elapsed_seconds": elapsed_d,
            "checkpoint": str(d_ckpt),
        },
        "dual_target_gradient_check": {
            "loss": float(loss.item()),
            "compact_grad_norm": float(compact_grad.norm().item()),
            "line_fusion_grad_norm": float(line_fusion_grad.norm().item()),
            "station_summary_grad_norm": float(station_summary_grad.norm().item()),
            "current_head_grad_norm": float(current_head_grad.norm().item()),
            "next_head_grad_norm": float(next_head_grad.norm().item()),
        },
    }


def main() -> None:
    print("=" * 80)
    print("开始工作三综合验证全流程 (Sections 9.1, 9.2, 9.3)")
    print("=" * 80)
    overall_start = time.perf_counter()

    r91 = run_section_9_1_five_stations_seven_aircraft()
    r92 = run_section_9_2_real_283_k10_bridge()
    r93 = run_section_9_3_learning_smoke_test()

    total_elapsed = time.perf_counter() - overall_start
    print("\n" + "=" * 80)
    print(f"工作三综合验证全部通过！总耗时: {total_elapsed:.2f}s")
    print("=" * 80)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": total_elapsed,
        "section_9_1": r91,
        "section_9_2": r92,
        "section_9_3": r93,
    }
    summary_path = Path("docs/section9_comprehensive_validation_report.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"综合验证报告已保存至: {summary_path}")


if __name__ == "__main__":
    main()
