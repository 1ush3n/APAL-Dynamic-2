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
import copy
from dataclasses import replace
import gc
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from envs.work3.core_types import ActionBranch, TaskStatus, CycleExecutionRecord
from envs.work3.environment import AirLineEnvWork3
from models.work3.actor_critic import (
    ActorCriticWork3,
    extract_compact_state_features,
)
from models.work3.ppo_buffer import (
    PendingTimeLabelCache,
    RolloutBufferWork3,
    PPOTransition,
)
from training.work3_lightning import Work3LightningModule, Work3TrainingUpdate
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

    source = MultiAircraftBaseline.load_from_json(ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json")
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
    baseline_path = ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json"
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
    cycle_time_labels = res_d.get("cycle_time_labels", [])
    assert cycle_time_labels and len(cycle_time_labels) > 0, "Method D 必须产生真实周期时间监督记录 (cycle_time_labels)！"
    available_labels = [l for l in cycle_time_labels if l.get("label_available")]
    assert len(available_labels) > 0, "Method D 必须产生至少一条实际可用的时间监督标签！"
    assert res_d.get("time_supervision_optimizer_updates", 0) > 0, "Method D 必须执行至少一次时间监督优化器更新！"
    replay_error = res_d.get("sampling_replay_max_abs_error")
    if replay_error is not None:
        assert replay_error < 1e-4, f"采样重放误差过大: {replay_error}"

    # 3. 校验双目标损失合成张量基准测试 (synthetic_dual_target_gradient_check)
    print("--> 3/4: 校验双目标损失合成张量基准测试 (synthetic_dual_target_gradient_check)...")
    actor_critic_synth = ActorCriticWork3(
        state_dim=32,
        task_feat_dim=8,
        hidden_dim=64,
        max_station_workers=16,
    )
    time_head_synth = TimeResidualHead(in_dim=actor_critic_synth.hidden_dim, hidden_dim=64)
    trainer_synth = PPOTrainerWork3(
        actor_critic=actor_critic_synth,
        time_head=time_head_synth,
        lr=1e-3,
        time_loss_coef=1.0,
    )

    env_synth = AirLineEnvWork3(baseline_json_path=ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json")
    env_synth.reset()
    state_feat_synth = extract_compact_state_features(env_synth.state, compute_cycle_heuristic_cmax(env_synth.state))
    graph_snapshot_synth = actor_critic_synth.build_graph_snapshot(env_synth)

    aux_batch_synth = {
        "state_feats": state_feat_synth.unsqueeze(0),
        "graph_snapshots": [graph_snapshot_synth],
        "target_residuals": torch.tensor([0.25], dtype=torch.float32),
        "current_target_masks": torch.tensor([True], dtype=torch.bool),
        "target_next_residuals": torch.tensor([[0.1, -0.2, 0.05, 0.0, 0.0]], dtype=torch.float32),
        "next_cycle_masks": torch.tensor([[True, True, True, False, False]], dtype=torch.bool),
        "worker_ids": [0],
        "episode_ids": [0],
        "cycle_ids": [0],
        "decision_ids": [0],
    }

    trainer_synth.optimizer.zero_grad()
    loss_synth = trainer_synth.compute_time_auxiliary_loss(aux_batch_synth)
    loss_synth.backward()

    compact_grad_s = actor_critic_synth.encoder[0].weight.grad
    line_fusion_grad_s = actor_critic_synth.graph_encoder.line_fusion_mlp[0].weight.grad
    station_summary_grad_s = actor_critic_synth.graph_encoder.station_summary_mlp[0].weight.grad
    aircraft_role_grad_s = actor_critic_synth.graph_encoder.aircraft_role_mlp[0].weight.grad
    current_head_grad_s = time_head_synth.reg_fc[0].weight.grad
    next_head_grad_s = time_head_synth.next_cycle_fc[0].weight.grad

    assert compact_grad_s is not None and torch.isfinite(compact_grad_s).all() and compact_grad_s.norm().item() > 1e-7
    assert line_fusion_grad_s is not None and torch.isfinite(line_fusion_grad_s).all() and line_fusion_grad_s.norm().item() > 1e-7
    assert station_summary_grad_s is not None and torch.isfinite(station_summary_grad_s).all() and station_summary_grad_s.norm().item() > 1e-7
    assert aircraft_role_grad_s is not None and torch.isfinite(aircraft_role_grad_s).all() and aircraft_role_grad_s.norm().item() > 1e-7
    assert current_head_grad_s is not None and torch.isfinite(current_head_grad_s).all() and current_head_grad_s.norm().item() > 1e-7
    assert next_head_grad_s is not None and torch.isfinite(next_head_grad_s).all() and next_head_grad_s.norm().item() > 1e-7

    print(f"    合成双目标损失={loss_synth.item():.6f}，6核心模块合成梯度检验通过。")
    del env_synth; gc.collect()

    # 4. 校验真实轨迹双目标闭环验证 (real_dual_target_closed_loop_check / R3)
    print("--> 4/4: 校验真实轨迹双目标闭环验证 (real_dual_target_closed_loop_check / R3)...")
    real_env = AirLineEnvWork3(baseline_json_path=ROOT_DIR / "data" / "work3" / "real_283_k10_baseline.json")
    real_env.reset()

    real_actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=64, max_station_workers=16)
    real_time_head = TimeResidualHead(in_dim=real_actor.hidden_dim, hidden_dim=64)
    frozen_actor = copy.deepcopy(real_actor)
    for p in frozen_actor.parameters():
        p.requires_grad = False

    real_lightning = Work3LightningModule(
        actor_critic=real_actor,
        time_head=real_time_head,
        learning_rate=1e-3,
        time_loss_coef=1.0,
        ppo_epochs=1,
    )

    real_cache = PendingTimeLabelCache()
    p1_drained = None
    p2_drained = None
    step_count = 0

    # --- 阶段 1: Rollout 轨迹段 1 直至到达 P1 转站并生成当前周期残差标签 ---
    buffer1 = RolloutBufferWork3()
    while p1_drained is None and step_count < 60:
        step_count += 1
        cands = real_env.get_action_candidates()
        cmax = compute_cycle_heuristic_cmax(real_env.state)
        state_feat = extract_compact_state_features(real_env.state, cmax)
        time_urg = torch.zeros(12)
        if not cands:
            act = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
            lp, val, srec = 0.0, 0.0, {"action_type": "advance_to_next_event"}
        else:
            act, lp, val, srec = real_actor.select_action(real_env, state_feat, time_urg)

        real_cache.add(
            worker_id=0,
            episode_id=0,
            cycle_id=int(real_env.state.current_cycle),
            decision_id=step_count,
            state_feat=state_feat,
            graph_snapshot=srec.get("graph_snapshot"),
            estimated_cmax=cmax,
            current_time=real_env.state.current_time,
            h0=real_env.state.h0,
            predictor_version=None,
        )
        buffer1.add(PPOTransition(
            state_feat=state_feat,
            time_urgency=time_urg,
            sample_record=srec,
            reward=0.0,
            raw_reward=0.0,
            value=val,
            log_prob=lp,
            worker_id=0,
            episode_id=0,
            segment_id=0,
        ))

        obs, r, term, trunc, info = real_env.step(act)
        if info.get("cycle_execution_records"):
            for cid, rec_dict in info["cycle_execution_records"].items():
                rec = CycleExecutionRecord.from_dict(rec_dict)
                real_cache.attach_transfer(
                    episode_id=0,
                    worker_id=0,
                    cycle_id=rec.cycle_id,
                    actual_transfer_time=rec.transfer_time,
                    cycle_record=rec,
                )
                if rec.cycle_id == 1 and p1_drained is None:
                    p1_drained = real_cache.drain_ready()

    assert p1_drained is not None, "真实轨迹未提取出 P1 当前残差标签！"
    assert p1_drained["current_target_masks"].any(), "P1 提取批次必须包含至少一个有效当前周期残差目标！"
    buffer1.finish_trajectories(last_values_by_segment={(0, 0, 0): 0.0})

    # P1 当前残差更新
    update1 = Work3TrainingUpdate(
        buffer=buffer1,
        environment_steps=len(buffer1),
        time_auxiliary_batch=p1_drained,
    )
    real_lightning.training_step(update1, 0)

    # 验证 real_actor 参数已更新
    actor_params_changed = any(
        not torch.equal(p_new, p_orig)
        for p_new, p_orig in zip(real_actor.parameters(), frozen_actor.parameters())
    )
    assert actor_params_changed, "P1 训练步必须使在线 real_actor 产生参数更新！"

    # 清空 buffer1，杜绝 PPO 轨迹跨更新段复用
    del buffer1
    gc.collect()

    # --- 阶段 2: 保持缓存跨段保留，使用已更新的在线 real_actor 继续 Rollout 段 2 直至到达 P2 转站 ---
    buffer2 = RolloutBufferWork3()
    while p2_drained is None and step_count < 100:
        step_count += 1
        cands = real_env.get_action_candidates()
        cmax = compute_cycle_heuristic_cmax(real_env.state)
        state_feat = extract_compact_state_features(real_env.state, cmax)
        time_urg = torch.zeros(12)
        if not cands:
            act = {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
            lp, val, srec = 0.0, 0.0, {"action_type": "advance_to_next_event"}
        else:
            act, lp, val, srec = real_actor.select_action(real_env, state_feat, time_urg)

        real_cache.add(
            worker_id=0,
            episode_id=0,
            cycle_id=int(real_env.state.current_cycle),
            decision_id=step_count,
            state_feat=state_feat,
            graph_snapshot=srec.get("graph_snapshot"),
            estimated_cmax=cmax,
            current_time=real_env.state.current_time,
            h0=real_env.state.h0,
            predictor_version=None,
        )
        buffer2.add(PPOTransition(
            state_feat=state_feat,
            time_urgency=time_urg,
            sample_record=srec,
            reward=0.0,
            raw_reward=0.0,
            value=val,
            log_prob=lp,
            worker_id=0,
            episode_id=0,
            segment_id=1,
        ))

        obs, r, term, trunc, info = real_env.step(act)
        if info.get("cycle_execution_records"):
            for cid, rec_dict in info["cycle_execution_records"].items():
                rec = CycleExecutionRecord.from_dict(rec_dict)
                real_cache.attach_transfer(
                    episode_id=0,
                    worker_id=0,
                    cycle_id=rec.cycle_id,
                    actual_transfer_time=rec.transfer_time,
                    cycle_record=rec,
                )
                if rec.cycle_id == 2 and p2_drained is None:
                    p2_drained = real_cache.drain_ready()

    assert p2_drained is not None, "真实轨迹未提取出 P2 跨周期下一周期预测标签！"
    assert p2_drained["next_cycle_masks"].any(), "P2 批次必须包含至少一个有效下一周期预测目标！"
    next_only_mask = (~p2_drained["current_target_masks"]) & p2_drained["next_cycle_masks"].any(dim=-1)
    assert next_only_mask.any(), "P2 批次必须包含跨段保留的 next-only 样本 (段1决策由段2转站结算)！"

    buffer2.finish_trajectories(last_values_by_segment={(0, 0, 1): 0.0})

    # P2 下一周期残差更新
    update2 = Work3TrainingUpdate(
        buffer=buffer2,
        environment_steps=len(buffer2),
        time_auxiliary_batch=p2_drained,
    )
    real_lightning.training_step(update2, 0)

    # 验证全部 6 个核心模块在真实轨迹监督下的梯度
    compact_grad = real_actor.encoder[0].weight.grad
    line_fusion_grad = real_actor.graph_encoder.line_fusion_mlp[0].weight.grad
    station_summary_grad = real_actor.graph_encoder.station_summary_mlp[0].weight.grad
    aircraft_role_grad = real_actor.graph_encoder.aircraft_role_mlp[0].weight.grad
    current_head_grad = real_time_head.reg_fc[0].weight.grad
    next_head_grad = real_time_head.next_cycle_fc[0].weight.grad

    assert compact_grad is not None and torch.isfinite(compact_grad).all() and compact_grad.norm().item() > 1e-7
    assert line_fusion_grad is not None and torch.isfinite(line_fusion_grad).all() and line_fusion_grad.norm().item() > 1e-7
    assert station_summary_grad is not None and torch.isfinite(station_summary_grad).all() and station_summary_grad.norm().item() > 1e-7
    assert aircraft_role_grad is not None and torch.isfinite(aircraft_role_grad).all() and aircraft_role_grad.norm().item() > 1e-7
    assert current_head_grad is not None and torch.isfinite(current_head_grad).all() and current_head_grad.norm().item() > 1e-7
    assert next_head_grad is not None and torch.isfinite(next_head_grad).all() and next_head_grad.norm().item() > 1e-7

    # 冻结副本梯度严格为 None
    assert all(p.grad is None for p in frozen_actor.parameters()), "冻结策略副本参数不得产生梯度！"

    print("    真实轨迹双目标更新成功，全部 6 核心模块梯度与冻结副本检验通过：")
    print(f"      紧凑状态编码器梯度模长: {compact_grad.norm().item():.6f}")
    print(f"      整线融合MLP梯度模长: {line_fusion_grad.norm().item():.6f}")
    print(f"      站位汇总MLP梯度模长: {station_summary_grad.norm().item():.6f}")
    print(f"      飞机角色MLP梯度模长: {aircraft_role_grad.norm().item():.6f}")
    print(f"      当前残差预测头梯度模长: {current_head_grad.norm().item():.6f}")
    print(f"      下一周期预测头梯度模长: {next_head_grad.norm().item():.6f}")

    del real_env, buffer2; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
            "cycle_time_labels_count": len(cycle_time_labels),
            "available_labels_count": len(available_labels),
            "actual_transfers": res_d.get("first_actual_transfer"),
        },
        "synthetic_dual_target_gradient_check": {
            "loss": float(loss_synth.item()),
            "compact_grad_norm": float(compact_grad_s.norm().item()),
            "line_fusion_grad_norm": float(line_fusion_grad_s.norm().item()),
            "station_summary_grad_norm": float(station_summary_grad_s.norm().item()),
            "aircraft_role_grad_norm": float(aircraft_role_grad_s.norm().item()),
            "current_head_grad_norm": float(current_head_grad_s.norm().item()),
            "next_head_grad_norm": float(next_head_grad_s.norm().item()),
        },
        "real_dual_target_closed_loop_check": {
            "p1_samples_drained": int(p1_drained["target_residuals"].numel()),
            "p2_samples_drained": int(p2_drained["target_residuals"].numel()),
            "compact_grad_norm": float(compact_grad.norm().item()),
            "line_fusion_grad_norm": float(line_fusion_grad.norm().item()),
            "station_summary_grad_norm": float(station_summary_grad.norm().item()),
            "aircraft_role_grad_norm": float(aircraft_role_grad.norm().item()),
            "current_head_grad_norm": float(current_head_grad.norm().item()),
            "next_head_grad_norm": float(next_head_grad.norm().item()),
            "frozen_actor_grad_is_none": True,
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
    summary_path = ROOT_DIR / "docs" / "section9_comprehensive_validation_report.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"综合验证报告已保存至: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)
