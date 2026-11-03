"""Task 1 Investigation:
1. Recompute the 4 counts: Static eligibility, Initial legal mask, Dynamic unlock, Actual postponement.
2. Formally step the 5 pairs (6->7, 15->16, 8->9, 12->13, 20->21):
   - Successor postpones first (v) while predecessor (u) is still pending;
   - Predecessor (u) postpone mask unlocks;
   - Predecessor (u) then postpones;
   - Rollout to batch completion with PostponeCompatibleRepairAgent;
   - Verify independent feasibility and ledger.
3. Compare single-task feasibility vs joint multi-task postponement capacity.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
import sys
from typing import Any

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from work3_4060.diagnostics.postpone_compatible_agent import PostponeCompatibleRepairAgent

BASELINE_PATH = repo_root / "work3_4060" / "rule_baseline_ac_v1" / "mini210_rule_baseline.json"
OUTPUT_DIR = repo_root / "work3_4060" / "diagnostics" / "task1_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_static_eligibility(env: AirLineEnvWork3) -> dict[int, list[int]]:
    """计算各架飞机的静态后移资格工序列表。"""
    eligible_by_ac: dict[int, list[int]] = {}
    for ac_id in range(env.state.num_aircraft):
        ac_tasks = [t for t in env.state.tasks.values() if t.aircraft_id == ac_id]
        eligible = []
        for t in ac_tasks:
            # 静态资格判定标准：
            # 1. 非末站 (current_station < num_stations - 1)
            # 2. 目标站有合格工人满足 demand
            # 3. 无固定站位/最晚站位/工艺违反 (在没有后继阻止的前提下)
            if t.current_station >= env.state.num_stations - 1:
                continue
            target_station = t.current_station + 1
            if len(env.valid_team_completion_workers(t, [], station_id=target_station)) < t.demand:
                continue
            if t.fixed_station is not None and target_station != t.fixed_station:
                continue
            if t.max_allowed_station is not None and target_station > t.max_allowed_station:
                continue
            station_map = {
                p_id: env.state.tasks[f"{t.aircraft_id}_{p_id}"].current_station
                for p_id in env.constraint_engine.physical_predecessors[t.task_id]
                if f"{t.aircraft_id}_{p_id}" in env.state.tasks
            }
            violation = env.constraint_engine.station_violation(t.task_id, target_station, station_map)
            if violation is not None:
                continue
            eligible.append(t.task_id)
        eligible_by_ac[ac_id] = sorted(eligible)
    return eligible_by_ac


def get_initial_postpone_mask(env: AirLineEnvWork3) -> dict[str, bool]:
    """计算初始时刻 (reset 后) 全局所有工序的 can_postpone 掩码。"""
    masks = {}
    for task_key, t in env.state.tasks.items():
        masks[task_key] = env.can_postpone(t)
    return masks


def run_witness_for_pair(
    pair: tuple[int, int],
    aircraft_id: int = 0,
) -> dict[str, Any]:
    """为一个特定 pair (u -> v) 执行合法动作见证：
    1. 正常运行至 aircraft_id 进驻该站位且 pair 处于候选/可决策状态；
    2. 核验 u 的 can_postpone 为 False (被 v 阻挡)；
    3. 核验 v 的 can_postpone 为 True；
    4. 执行 v 后移；
    5. 核验 u 的 can_postpone 立即解锁为 True；
    6. 执行 u 后移；
    7. 使用 PostponeCompatibleRepairAgent 续跑到完工；
    8. 核验全过程物理可行性与费用账本。
    """
    u, v = pair
    u_key = f"{aircraft_id}_{u}"
    v_key = f"{aircraft_id}_{v}"

    env = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH))
    env.reset()
    agent = PostponeCompatibleRepairAgent()

    witness_events = []
    step_records = []
    
    # 推进直到 v 可以后移，且 u 仍未决
    step = 0
    v_postponed = False
    u_postponed = False

    while not env._check_terminated() and step < 1000:
        t_u = env.state.tasks[u_key]
        t_v = env.state.tasks[v_key]
        ac = env.state.aircraft[aircraft_id]

        # 检查是否满足见证触发条件：飞机在所在站，v 可以后移
        if not v_postponed and ac.current_station == t_v.current_station:
            # 检查 u 与 v 状态
            u_can_p = env.can_postpone(t_u)
            v_can_p = env.can_postpone(t_v)
            u_err = env.validate_postpone(t_u)
            v_err = env.validate_postpone(t_v)

            witness_events.append({
                "phase": "pre_v_postpone",
                "time": float(env.state.current_time),
                "step": step,
                "u_status": t_u.status.name,
                "v_status": t_v.status.name,
                "u_can_postpone": u_can_p,
                "u_error": u_err,
                "v_can_postpone": v_can_p,
                "v_error": v_err,
            })

            if v_can_p and not u_can_p:
                # 执行 v 后移
                act_v = {"task_key": v_key, "branch": ActionBranch.POSTPONE}
                step_res = env.step(act_v)
                step += 1
                v_postponed = True

                # 立即检查 u 的掩码
                u_can_p_after = env.can_postpone(t_u)
                u_err_after = env.validate_postpone(t_u)
                witness_events.append({
                    "phase": "post_v_postpone_u_unlock_check",
                    "time": float(env.state.current_time),
                    "step": step,
                    "u_status": t_u.status.name,
                    "v_status": t_v.status.name,
                    "u_can_postpone": u_can_p_after,
                    "u_error": u_err_after,
                })

                if u_can_p_after:
                    # 执行 u 后移
                    act_u = {"task_key": u_key, "branch": ActionBranch.POSTPONE}
                    step_res = env.step(act_u)
                    step += 1
                    u_postponed = True
                    witness_events.append({
                        "phase": "post_u_postpone",
                        "time": float(env.state.current_time),
                        "step": step,
                        "u_status": t_u.status.name,
                        "v_status": t_v.status.name,
                        "u_station": int(t_u.current_station),
                        "v_station": int(t_v.current_station),
                    })
                    continue

        act = agent.select_action(env)
        if act is None:
            break
        step_res = env.step(act)
        step += 1

    terminated = env._check_terminated()
    
    # 独立可行性核验
    completed_tasks = sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.COMPLETED)
    all_completed = (completed_tasks == len(env.state.tasks))
    
    # 费用账本
    costs = {
        "cumulative_cost": float(env.cumulative_cost),
        "cost_takt": float(env.cost_takt),
        "cost_time": float(env.cost_time),
        "cost_team": float(env.cost_team),
        "cost_postpone": float(env.cost_postpone),
        "cost_revision": float(env.cost_revision),
        "makespan": float(env.state.current_time),
    }

    return {
        "pair": list(pair),
        "aircraft_id": aircraft_id,
        "u_postponed": u_postponed,
        "v_postponed": v_postponed,
        "terminated": terminated,
        "all_completed": all_completed,
        "completed_count": completed_tasks,
        "total_tasks": len(env.state.tasks),
        "witness_events": witness_events,
        "costs": costs,
        "final_step_count": step,
    }


def run_joint_postponement_test() -> dict[str, Any]:
    """测试在单条轨迹中共同后移多个工序的能力：
    1. 尝试在同一轨迹中共同后移所有 5 对工序 (10 道)；
    2. 尝试在同一轨迹中共同后移 Aircraft 0 的全部 17 道静态资格工序。
    """
    pairs = [(6, 7), (15, 16), (8, 9), (12, 13), (20, 21)]
    
    # --- 测试 A: 5 对工序 (10道) 共同后移 ---
    env = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH))
    env.reset()
    agent = PostponeCompatibleRepairAgent()

    pairs = [(6, 7), (15, 16), (8, 9), (12, 13), (20, 21)]
    postponed_set = set()
    step = 0
    witness_log_5pairs = []
    
    while not env._check_terminated() and step < 2000:
        ac0 = env.state.aircraft[0]
        action_taken = False
        for u, v in pairs:
            u_key = f"0_{u}"
            v_key = f"0_{v}"
            t_u = env.state.tasks[u_key]
            t_v = env.state.tasks[v_key]

            if ac0.current_station == t_v.current_station:
                if v_key not in postponed_set and env.can_postpone(t_v):
                    assert not env.can_postpone(t_u)
                    env.step({"task_key": v_key, "branch": ActionBranch.POSTPONE})
                    postponed_set.add(v_key)
                    step += 1
                    assert env.can_postpone(t_u)
                    env.step({"task_key": u_key, "branch": ActionBranch.POSTPONE})
                    postponed_set.add(u_key)
                    step += 1
                    action_taken = True
                    witness_log_5pairs.append({"u": u, "v": v, "time": float(env.state.current_time), "station": int(t_u.current_station)})
                    break
        if action_taken:
            continue

        act = agent.select_action(env)
        if act is None:
            break
        env.step(act)
        step += 1

    completed_tasks_5pairs = sum(1 for t in env.state.tasks.values() if t.status == TaskStatus.COMPLETED)
    results_5pairs = {
        "target_tasks": [f"0_{t}" for p in pairs for t in p],
        "postponed_count": len(postponed_set),
        "postponed_tasks": sorted(list(postponed_set)),
        "all_completed": completed_tasks_5pairs == len(env.state.tasks),
        "completed_count": completed_tasks_5pairs,
        "makespan": float(env.state.current_time),
        "cumulative_cost": float(env.cumulative_cost),
        "cost_postpone": float(env.cost_postpone),
        "cost_takt": float(env.cost_takt),
        "cost_team": float(env.cost_team),
        "cost_time": float(env.cost_time),
        "cost_revision": float(env.cost_revision),
        "witness_log": witness_log_5pairs,
        "steps": step,
    }

    # --- 测试 B: 尝试在单条轨迹中贪婪后移 Aircraft 0 的所有可后移工序 ---
    env_all = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH))
    env_all.reset()
    agent_all = PostponeCompatibleRepairAgent()

    all_17_eligible = [6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 20, 21, 25, 26, 28, 29]
    postponed_all = set()
    step = 0
    
    while not env_all._check_terminated() and step < 2000:
        # 检查 Aircraft 0 当前站位是否有任何可后移的工序
        ac = env_all.state.aircraft[0]
        can_p_tasks = [
            t for t in env_all.state.get_tasks_for_station(ac.current_station)
            if t.aircraft_id == 0 and t.task_id in all_17_eligible and t.task_key not in postponed_all and env_all.can_postpone(t)
        ]
        if can_p_tasks:
            # 优先后移有后继关系中的后继，或者任意合法者
            # 按照拓扑反序排序（即后继优先）
            t_to_postpone = sorted(can_p_tasks, key=lambda x: -len(x.predecessors))[0]
            env_all.step({"task_key": t_to_postpone.task_key, "branch": ActionBranch.POSTPONE})
            postponed_all.add(t_to_postpone.task_key)
            step += 1
            continue

        act = agent_all.select_action(env_all)
        if act is None:
            break
        env_all.step(act)
        step += 1

    completed_all = sum(1 for t in env_all.state.tasks.values() if t.status == TaskStatus.COMPLETED)
    results_all17 = {
        "target_count": len(all_17_eligible),
        "postponed_count": len(postponed_all),
        "postponed_tasks": sorted(list(postponed_all)),
        "all_completed": completed_all == len(env_all.state.tasks),
        "completed_count": completed_all,
        "makespan": float(env_all.state.current_time),
        "cumulative_cost": float(env_all.cumulative_cost),
        "cost_postpone": float(env_all.cost_postpone),
        "cost_takt": float(env_all.cost_takt),
        "cost_team": float(env_all.cost_team),
        "cost_time": float(env_all.cost_time),
        "cost_revision": float(env_all.cost_revision),
        "steps": step,
    }

    return {
        "five_pairs_joint": results_5pairs,
        "all_17_joint": results_all17,
    }


def main():
    print("=== Task 1 Investigation Starting ===")
    env = AirLineEnvWork3(baseline_json_path=str(BASELINE_PATH))
    env.reset()

    # 1. 静态资格
    static_eligibility = get_static_eligibility(env)
    print(f"\n1. 静态后移资格工序:")
    for ac_id, tasks in static_eligibility.items():
        print(f"  飞机 {ac_id}: {len(tasks)} 道工序 -> {tasks}")

    # 2. 初始合法掩码
    initial_masks = get_initial_postpone_mask(env)
    initial_true = [k for k, v in initial_masks.items() if v]
    print(f"\n2. 初始合法后移掩码 (can_postpone=True): {len(initial_true)} 道工序 -> {initial_true}")

    # 3. 逐个验证 5 对工序见证
    pairs = [(6, 7), (15, 16), (8, 9), (12, 13), (20, 21)]
    print("\n3. 验证 5 对工序动态见证:")
    witness_results = {}
    for p in pairs:
        print(f"  Running witness for pair {p[0]} -> {p[1]}...")
        res = run_witness_for_pair(p, aircraft_id=0)
        witness_results[f"{p[0]}->{p[1]}"] = res
        print(f"    v postponed: {res['v_postponed']}, u postponed: {res['u_postponed']}, all completed: {res['all_completed']}, J_total: {res['costs']['cumulative_cost']:.6f}, makespan: {res['costs']['makespan']:.2f}h")

    # 4. 共同后移容量测试
    print("\n4. 共同后移容量测试:")
    joint_results = run_joint_postponement_test()
    print(f"  5 对工序 (10道) 共同后移: 成功后移 {joint_results['five_pairs_joint']['postponed_count']}/10, 批次全完工={joint_results['five_pairs_joint']['all_completed']}, makespan={joint_results['five_pairs_joint']['makespan']:.2f}h")
    print(f"  全部17道候选共同后移: 成功后移 {joint_results['all_17_joint']['postponed_count']}/17, 批次全完工={joint_results['all_17_joint']['all_completed']}, makespan={joint_results['all_17_joint']['makespan']:.2f}h")

    # 汇总保存
    final_report = {
        "static_eligibility_per_aircraft": {str(k): v for k, v in static_eligibility.items()},
        "static_count_per_aircraft": len(static_eligibility[0]),
        "total_static_count": sum(len(v) for v in static_eligibility.values()),
        "initial_legal_count": len(initial_true),
        "initial_legal_tasks": initial_true,
        "pair_witnesses": witness_results,
        "joint_postponement_capacity": joint_results,
    }

    out_file = OUTPUT_DIR / "task1_reachability_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"\nReport written to: {out_file}")


if __name__ == "__main__":
    main()
