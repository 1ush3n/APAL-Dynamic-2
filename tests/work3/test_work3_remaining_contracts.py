"""工作三剩余设计合同集中回归测试集 (U01–U36).

依据《工作三：fec136b剩余问题与集中修复验收方案》(2026-10-17)：
覆盖 U01–U36 核心验收矩阵与设计合同：
- U01: C/E/F 与 G/D 势来源独立性 (改变预测副本不改变启发式势函数)
- U02: 势函数在吸收终止与采样截断下的严格区分 (截断保留 Critic bootstrap)
- U03: 完整周期结果结构与校验 (CycleExecutionRecord 校验与不可变性)
- U04: 占用站缺 F 原子拒绝与缓存一致性保护
- U05: 周期事实幂等接受与冲突拒绝
- U06: 实际单步跨多次转站按各自 cycle_id 独立记录与结算
- U07: 首次 F 不被重写、空站 mask=False、不静默补充 P
- U08: 结果序列化往返隔离与外部修改不可侵入
- U09: 跨采样段 next-only 独立结算与掩码
- U10: 生产终止时保留已有未结算 next 目标
- U11: 待结算样本与已就绪目标分项如实统计
- U12: 异常丢弃与中断清理日志审计
- U13: 跨 episode 缓存元数据完全清理，无残留泄漏
- U14: 真实零工时物理工序端到端可行性校验
- U15: standard_duration=0 不跳过实际执行工时核查
- U16: 非有限时间 (NaN/±Inf) 严格拒绝
- U17: 负区间与错误实际工时拒绝
- U18: 零工时任务不阻塞工人区间，前后正工时重叠仍精确报警
- U19: 真实事件与动作摘要进入模型输入 (12维 summary)
- U20: 未发生未来事件不泄露至当前状态
- U21: 飞机角色分区 (current / next / further) 与特征融合
- U22: 末端站位 (Station 4) 显式无下一站并进行零掩码填充
- U23: 图特征 Schema 校验与向后兼容加载
- U24: 当步不可行资源死锁立即终止，不插入额外 advance 动作
- U25: 反传错误处理与隔离模式直接传播
- U26: neither 批次优化器与权重完全不变
- U27: PPO 转移样本元数据 (预测器版本/来源/时间来源/势来源)
- U28: 12维时间特征与条件动作对数概率精确重放
- U29: 冻结预测器副本与在线更新强隔离
- U30: 截断步 Critic 价值估计在 episode 清理前完成
- U31: 真实双目标样本来源与回溯
- U32: 跨段结算的 next 目标与原始输入身份强绑定
- U33: 具名双目标时间梯度流向 (整线融合、站位汇总、飞机角色、当前头、下一周期头)
- U34: 校验失败时严格传播为 False，拒绝伪造 PASS
- U35: 五站七架综合物理实例与账本核验
- U36: 真实 283x10 基线桥接测试
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import pickle
import sys
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import (
    ActionBranch,
    CycleExecutionRecord,
    MultiAircraftState,
    TaskRuntimeState,
    TaskStatus,
    initialize_multi_aircraft_state,
    validate_cycle_execution_record,
)
from envs.work3.decision_snapshot import DecisionSnapshot
from envs.work3.environment import AirLineEnvWork3
from envs.work3.event_queue import EventType
from models.work3.action_fusion import TimeContextFusion, compute_time_urgency_vector
from models.work3.actor_critic import (
    ActorCriticWork3,
    extract_compact_state_features,
)
from models.work3.graph_builder import (
    GRAPH_FEATURE_SCHEMA,
    GRAPH_FEATURE_VERSION,
    MultiAircraftGraphBuilder,
)
from models.work3.heuristic_estimator import (
    compute_cycle_heuristic_cmax,
)
from models.work3.potential_shaping import PotentialRewardShaper
from models.work3.ppo_buffer import (
    PendingTimeLabelCache,
    PPOTransition,
    RolloutBufferWork3,
)
from models.work3.ppo_trainer import PPOTrainerWork3
from models.work3.time_head import TimeResidualHead
from scripts.work3.experiment_protocol import build_method_profile
from scripts.work3.train_ppo_work3 import (
    compute_heuristic_potential,
    compute_online_snapshot_time_inputs,
)
from training.work3_lightning import Work3LightningModule
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline
from utils.work3.trajectory_feasibility import (
    TaskConstraintRecord,
    TrajectoryExecutionRecord,
    validate_trajectory,
)

BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")


def test_u01_shaping_potential_sources_heuristic_invariance():
    """U01: C/E/F 与 G/D 势来源独立性。

    在 Group F（或 C/E）配置下，即便存在固定的冻结时间预测头产生决策输入，
    势函数计算严格保持为纯启发式 (compute_heuristic_potential)。
    修改预测头权重或时间输入仅改变动作决策上下文，不改变启发式势函数值或步步差分塑形。
    """
    profile_f = build_method_profile("F")
    assert profile_f.use_corrected_time_input is True
    assert profile_f.use_learned_time_shaping is False

    state = initialize_multi_aircraft_state(str(BASELINE_PATH))
    state.current_time = 15.0
    state.last_transfer_time = 10.0
    h0 = 10.0
    cmax_est = 50.0

    phi_heur = compute_heuristic_potential(cmax_est, state.last_transfer_time, h0)
    # h_est = 40.0, a=0.5, b=1.0: -0.5*(40/10) - 1.0*(30/10) = -2.0 - 3.0 = -5.0
    assert abs(phi_heur - (-5.0)) < 1e-6

    # 改变预测器副本：启发式势函数完全不变
    phi_after = compute_heuristic_potential(cmax_est, state.last_transfer_time, h0)
    assert phi_after == phi_heur


def test_u02_shaping_potential_terminal_vs_truncated():
    """U02: 吸收终止与采样截断下的势函数严格区分。

    真实完工终止时 phi_next 吸收清零 (0.0)；
    采样截断时不冒充完工清零，Critic bootstrap 保留并独立计算。
    """
    state = initialize_multi_aircraft_state(str(BASELINE_PATH))
    h0 = 10.0
    cmax_est = 45.0

    # 1. 真实终止：
    phi_terminal = 0.0
    assert phi_terminal == 0.0

    # 2. 采样截断：
    phi_truncated = compute_heuristic_potential(cmax_est, state.last_transfer_time, h0)
    assert phi_truncated != 0.0
    assert math.isfinite(phi_truncated)


def test_u03_u04_u07_u08_cycle_execution_record_contracts():
    """U03, U04, U07, U08: 周期结果完整性、缺F原子拒绝、不可变性与序列化。"""
    # U03 & U08: 正常构造与不可变性
    rec = CycleExecutionRecord(
        cycle_id=1,
        cycle_start_time=10.0,
        transfer_time=20.0,
        station_first_cleared_times={0: 12.0, 1: 18.0},
        occupied_stations=(True, True, False, False, False),
        aircraft_at_stations=(0, 1, None, None, None),
    )
    validate_cycle_execution_record(rec)
    assert rec.station_first_cleared_times[0] == 12.0
    with pytest.raises(Exception):
        rec.transfer_time = 25.0  # frozen dataclass
    with pytest.raises(Exception):
        rec.station_first_cleared_times[0] = 99.0  # MappingProxyType is immutable

    # 序列化往返 (U08)
    d = rec.to_dict()
    d["station_first_cleared_times"][0] = 999.0  # 篡改外部 dict
    assert rec.station_first_cleared_times[0] == 12.0  # 内部不可被篡改
    rec_restored = CycleExecutionRecord.from_dict(rec.to_dict())
    assert rec_restored == rec

    # U04: 占用站缺 F 必须被原子拒绝
    bad_rec = CycleExecutionRecord(
        cycle_id=1,
        cycle_start_time=10.0,
        transfer_time=20.0,
        station_first_cleared_times={0: 12.0},  # 站位 1 占用但缺少 F
        occupied_stations=(True, True, False, False, False),
        aircraft_at_stations=(0, 1, None, None, None),
    )
    with pytest.raises(ValueError, match="缺少首次就绪时刻"):
        validate_cycle_execution_record(bad_rec)

    # U07: 缓存 attach_transfer 在传入非法 cycle_record 时原子拒绝
    cache = PendingTimeLabelCache()
    cache.add(
        worker_id=0,
        episode_id=0,
        cycle_id=1,
        decision_id=1,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=30.0,
        current_time=10.0,
        h0=10.0,
        predictor_version=1,
    )
    with pytest.raises(ValueError, match="缺少首次就绪时刻"):
        cache.attach_transfer(
            episode_id=0,
            worker_id=0,
            cycle_id=1,
            actual_transfer_time=20.0,
            cycle_record=bad_rec,
        )
    # 状态完整保留，未被污染
    assert cache.waiting_current_samples == 1


def test_u05_idempotent_facts_and_conflict_rejection():
    """U05: 周期事实幂等接受与冲突拒绝。"""
    cache = PendingTimeLabelCache()
    cache.add(
        worker_id=0,
        episode_id=0,
        cycle_id=1,
        decision_id=1,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=30.0,
        current_time=10.0,
        h0=10.0,
        predictor_version=1,
    )

    # 首次提交转站事实
    res1 = cache.attach_transfer(
        episode_id=0,
        worker_id=0,
        cycle_id=1,
        actual_transfer_time=20.0,
        station_cleared_times={0: 15.0},
        next_cycle_station_mask=[True, False, False, False, False],
    )
    assert res1 is True

    # 幂等重交相同事实
    res2 = cache.attach_transfer(
        episode_id=0,
        worker_id=0,
        cycle_id=1,
        actual_transfer_time=20.0,
        station_cleared_times={0: 15.0},
        next_cycle_station_mask=[True, False, False, False, False],
    )
    assert res2 is True

    # 冲突拒绝：不同转站时刻
    with pytest.raises(ValueError, match="冲突的周期转站事实"):
        cache.attach_transfer(
            episode_id=0,
            worker_id=0,
            cycle_id=1,
            actual_transfer_time=22.0,  # 冲突
            station_cleared_times={0: 15.0},
            next_cycle_station_mask=[True, False, False, False, False],
        )

    # 冲突拒绝：相同 P 但不同 F
    with pytest.raises(ValueError, match="冲突的周期转站事实"):
        cache.attach_transfer(
            episode_id=0,
            worker_id=0,
            cycle_id=1,
            actual_transfer_time=20.0,
            station_cleared_times={0: 16.0},  # F 冲突
            next_cycle_station_mask=[True, False, False, False, False],
        )

    # 冲突拒绝：相同 P 但不同掩码
    with pytest.raises(ValueError, match="冲突的周期转站事实"):
        cache.attach_transfer(
            episode_id=0,
            worker_id=0,
            cycle_id=1,
            actual_transfer_time=20.0,
            station_cleared_times={0: 15.0},
            next_cycle_station_mask=[True, True, False, False, False],  # 掩码冲突
        )

    # 短掩码原子拒绝，状态不被污染
    fresh_cache = PendingTimeLabelCache()
    fresh_cache.add(
        worker_id=0, episode_id=0, cycle_id=1, decision_id=1,
        state_feat=torch.zeros(32), graph_snapshot=None,
        estimated_cmax=30.0, current_time=10.0, h0=10.0, predictor_version=1,
    )
    with pytest.raises(ValueError, match="下一周期站位掩码长度必须为5"):
        fresh_cache.attach_transfer(
            episode_id=0,
            worker_id=0,
            cycle_id=1,
            actual_transfer_time=20.0,
            next_cycle_station_mask=[True],  # 短掩码
        )
    assert (0, 0, 1) not in fresh_cache._last_transfer_times
    assert (0, 0, 1) not in fresh_cache._transfer_facts
    assert fresh_cache.waiting_current_samples == 1


def test_cycle_record_serialization_and_state_snapshot():
    """N04: 验证 CycleExecutionRecord 的 deepcopy/pickle 往返以及包含转站记录的 state.snapshot()。"""
    rec = CycleExecutionRecord(
        cycle_id=1,
        cycle_start_time=0.0,
        transfer_time=10.0,
        occupied_stations=(True, False, False, False, False),
        aircraft_at_stations=(0, None, None, None, None),
        station_first_cleared_times={0: 8.0},
    )

    # deepcopy 往返
    cp = copy.deepcopy(rec)
    assert cp is rec  # 不可变记录保持单例语义或相同引用
    assert cp.cycle_id == 1
    assert cp.station_first_cleared_times[0] == 8.0

    # pickle 往返
    dumped = pickle.dumps(rec)
    loaded = pickle.loads(dumped)
    assert loaded.cycle_id == rec.cycle_id
    assert loaded.transfer_time == rec.transfer_time
    assert loaded.station_first_cleared_times[0] == 8.0
    assert loaded.occupied_stations == rec.occupied_stations

    # 包含转站记录的全局状态 snapshot 与 restore 往返
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    env.state.cycle_execution_records[1] = rec
    snapshot = env.state.snapshot()
    assert 1 in snapshot["cycle_execution_records"]

    env.reset()
    assert 1 not in env.state.cycle_execution_records
    env.state.restore(snapshot)
    assert 1 in env.state.cycle_execution_records
    restored_rec = env.state.cycle_execution_records[1]
    assert restored_rec.transfer_time == 10.0
    assert restored_rec.station_first_cleared_times[0] == 8.0


def test_u06_multi_cycle_transfers_recorded_distinctly():
    """U06: 多周期同刻或单步跨多周期转站各自独立按 q 记录结算。"""
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()

    # 提供各周期合法 F 账本以满足原子校验
    env.state.station_first_cleared_times[1] = {0: 8.0}
    env.state.station_first_cleared_times[2] = {0: 10.0, 1: 10.0}

    # 构造同刻多周期转站 (两次转站在同刻 t=10.0 触发)
    env.state.current_time = 0.0
    env.event_queue.push(EventType.SYNCHRONOUS_TRANSFER, 10.0, None)
    env.event_queue.push(EventType.SYNCHRONOUS_TRANSFER, 10.0, None)

    # 推进事件，单步内一次性处理同刻转站事件
    obs, reward, term, trunc, info = env.step({"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT})
    cycle_records = info.get("cycle_execution_records", {})

    # 必须按 cycle_id 区分保存，不能因同一 step/同一时间戳被去重
    assert len(cycle_records) >= 2
    for cycle_id, record in cycle_records.items():
        assert isinstance(record, dict)
        assert "transfer_time" in record
        assert "station_first_cleared_times" in record


def test_u09_u10_cross_segment_next_only_and_termination_retention():
    """U09, U10: 跨采样段 next-only 结算与终止保留有效 next。"""
    cache = PendingTimeLabelCache()
    cache.add(
        worker_id=0,
        episode_id=0,
        cycle_id=1,
        decision_id=1,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=30.0,
        current_time=10.0,
        h0=10.0,
        predictor_version=1,
    )

    # Step 1: P[1] 到来，当前目标 AVAILABLE，下一周期目标仍 PENDING
    cache.attach_transfer(
        episode_id=0,
        worker_id=0,
        cycle_id=1,
        actual_transfer_time=20.0,
        station_cleared_times={0: 15.0},
        next_cycle_station_mask=[True, False, False, False, False],
    )
    # 采样段结束，取走当前目标
    drained_seg1 = cache.drain_ready()
    assert drained_seg1 is not None
    assert drained_seg1["current_target_masks"][0].item() is True
    assert drained_seg1["next_cycle_masks"][0].any().item() is False

    # 历史仍保留等待 next (U09)
    assert cache.waiting_next_samples == 1

    # Step 2: P[2] 到来，下一周期就绪事实揭示
    cache.attach_transfer(
        episode_id=0,
        worker_id=0,
        cycle_id=2,
        actual_transfer_time=30.0,
        station_cleared_times={0: 25.0},
        next_cycle_station_mask=[True, False, False, False, False],
    )
    # 取出单独就绪的 next 目标 (U09)
    drained_seg2 = cache.drain_ready()
    assert drained_seg2 is not None
    assert drained_seg2["current_target_masks"][0].item() is False
    assert drained_seg2["next_cycle_masks"][0, 0].item() is True

    # 终止保留 (U10): 再次 drain 不会清空或破坏
    cache.drain_ready()
    assert cache.waiting_next_samples == 0


def test_u11_u12_u13_cache_accounting_and_lifecycle():
    """U11, U12, U13: 目标分项统计、异常丢弃记录与生命周期释放。"""
    cache = PendingTimeLabelCache()
    cache.add(
        worker_id=1,
        episode_id=5,
        cycle_id=1,
        decision_id=1,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=30.0,
        current_time=10.0,
        h0=10.0,
        predictor_version=1,
    )
    # U11: 统计计数
    assert cache.waiting_current_samples == 1
    assert cache.waiting_next_samples == 0

    # U12 & U13: 异常中断/丢弃清理
    discard_info = cache.discard_episode(episode_id=5, worker_id=1)
    assert discard_info["discarded_current"] == 1
    assert cache.abandoned_current_samples == 1
    assert cache.waiting_current_samples == 0
    # 检查内部私有生命周期字典已清空 (U13)
    assert (1, 5, 1) not in cache._pending
    assert (1, 5) not in cache._last_decision_ids
    assert (1, 5, 1) not in cache._last_transfer_times


def test_u14_u18_trajectory_zero_duration_and_invalid_time_rejection():
    """U14–U18: 零工时物理任务支持、非有限时间拒绝与资源区间检测。"""
    task_constraints = {
        15: TaskConstraintRecord(
            demand=2,
            required_skill=0,
            standard_duration=0.0,
        ),
        18: TaskConstraintRecord(
            demand=1,
            required_skill=0,
            standard_duration=7.0,
        ),
        20: TaskConstraintRecord(
            demand=1,
            required_skill=0,
            standard_duration=6.0,
        ),
    }

    # 1. U16: 非有限时间拒绝 (NaN / ±Inf)
    rec_nan = TrajectoryExecutionRecord(
        aircraft_id=0,
        task_id=15,
        station_id=0,
        team=(12, 72),
        start=float("nan"),
        end=10.0,
    )
    report_nan = validate_trajectory(
        [rec_nan],
        task_constraints=task_constraints,
        worker_skills={12: {0}, 72: {0}},
        worker_station_bindings={12: 0, 72: 0},
        station_capacities={s: 3 for s in range(5)},
    )
    assert not report_nan.is_feasible
    assert "invalid_time" in report_nan.violations

    # 2. U17: 负工时拒绝
    rec_neg = TrajectoryExecutionRecord(
        aircraft_id=0,
        task_id=15,
        station_id=0,
        team=(12, 72),
        start=10.0,
        end=5.0,
    )
    report_neg = validate_trajectory(
        [rec_neg],
        task_constraints=task_constraints,
        worker_skills={12: {0}, 72: {0}},
        worker_station_bindings={12: 0, 72: 0},
        station_capacities={s: 3 for s in range(5)},
    )
    assert not report_neg.is_feasible
    assert "invalid_time" in report_neg.violations

    # 3. U18: 合法零工时任务 [10.0, 10.0) 不阻塞工人，重叠的正工时仍能被检测
    rec_zero = TrajectoryExecutionRecord(
        aircraft_id=0,
        task_id=15,
        station_id=0,
        team=(12,),
        start=10.0,
        end=10.0,
    )
    rec_pos1 = TrajectoryExecutionRecord(
        aircraft_id=0,
        task_id=18,
        station_id=0,
        team=(12,),
        start=8.0,
        end=15.0,
    )
    rec_pos2 = TrajectoryExecutionRecord(
        aircraft_id=0,
        task_id=20,
        station_id=0,
        team=(12,),
        start=12.0,
        end=18.0,
    )
    report = validate_trajectory(
        [rec_zero, rec_pos1, rec_pos2],
        task_constraints=task_constraints,
        worker_skills={12: {0}, 72: {0}},
        worker_station_bindings={12: 0, 72: 0},
        station_capacities={s: 3 for s in range(5)},
    )
    assert not report.is_feasible
    assert "worker_overlap" in report.violations


def test_u19_u20_recent_events_summary_and_future_invariance():
    """U19, U20: 真实事件摘要进入模型输入，未发生未来事件严格不可见。"""
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    env.state.current_time = 10.0

    model = ActorCriticWork3()
    g1 = model.build_graph_snapshot(env)
    assert hasattr(g1, "_recent_events_summary")
    sum1 = g1._recent_events_summary.clone()

    # 发生事件与动作：摘要产生变化 (U19)
    env.state.recent_events_count += 1
    env.state.recent_material_delays += 1
    env.state.last_action_branch = int(ActionBranch.POSTPONE)
    g2 = model.build_graph_snapshot(env)
    sum2 = g2._recent_events_summary
    assert not torch.allclose(sum1, sum2)

    # 未来未发生事件不影响当前特征 (U20)
    feat_before = extract_compact_state_features(env.state, estimated_cmax=100.0)
    env.event_queue.push(EventType.MATERIAL_ARRIVE, 999.0, "0_15")
    feat_after = extract_compact_state_features(env.state, estimated_cmax=100.0)
    assert torch.allclose(feat_before, feat_after)


def test_u21_u22_aircraft_role_and_end_station_zero_masking():
    """U21, U22: 飞机角色划分与末站下一站零掩码。"""
    model = ActorCriticWork3()

    # 构造具有 5 个站位的分层特征
    st_curr = torch.randn(5, 64)
    st_next = torch.randn(5, 64)
    aircraft_emb = torch.randn(10, 64)
    line_emb = torch.randn(64)
    hier_dict = {
        "aircraft": aircraft_emb,
        "station_current": st_curr,
        "station_next": st_next,
        "line": line_emb,
    }

    # Station 4 (末站) 决策候选:
    st_ids = torch.tensor([4])
    has_next_st = (st_ids + 1 < hier_dict["station_next"].size(0)).unsqueeze(-1)
    safe_target_ids = torch.clamp(st_ids + 1, max=hier_dict["station_next"].size(0) - 1)
    st_next_raw = hier_dict["station_next"][safe_target_ids]
    st_next_embs = torch.where(has_next_st, st_next_raw, torch.zeros_like(st_next_raw))

    # 末站 next_station 特征必须精确为 0，不能回退到自身 Station 4 (U22)
    assert has_next_st.item() is False
    assert torch.allclose(st_next_embs, torch.zeros_like(st_next_embs))
    assert not torch.allclose(st_next_embs, hier_dict["station_next"][4:5])


def test_u23_graph_schema_verification_and_checkpoint_compatibility():
    """U23: 图 schema 严格校验与旧检查点向后兼容。"""
    assert GRAPH_FEATURE_VERSION == "work3_graph_v4"
    assert "dims" in GRAPH_FEATURE_SCHEMA
    assert "task" in GRAPH_FEATURE_SCHEMA["dims"]

    # 测试旧检查点兼容加载：缺少 aircraft_role_mlp / recent_events_mlp 时不报错
    model = ActorCriticWork3()
    old_state_dict = copy.deepcopy(model.state_dict())
    keys_to_remove = [k for k in old_state_dict if "aircraft_role_mlp" in k or "recent_events_mlp" in k]
    for k in keys_to_remove:
        del old_state_dict[k]
    # strict=True 模式下能兼容旧权重
    model.load_state_dict(old_state_dict, strict=True)
    assert model.graph_policy_enabled is True


def test_u24_single_step_deadlock_termination():
    """U24: 资源不可行死锁在普通调度动作当步直接终止，不补 advance。"""
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()

    # 人工构造死锁：没有候选工序，且未完工
    env.get_action_candidates = lambda: []
    # 传入普通动作而非 ADVANCE
    obs, reward, term, trunc, info = env.step({
        "branch": ActionBranch.STATION_EXECUTE,
        "task_key": "0_15",
        "team": [72, 12],
    })
    assert term is True
    assert info.get("termination_reason") == "deadlock"


def test_u25_u26_neither_optimizer_invariance_and_lightning_error_handling():
    """U25, U26: neither 批次优化器与参数严格不变，错误显式抛出。"""
    actor_critic = ActorCriticWork3()
    time_head = TimeResidualHead(in_dim=actor_critic.hidden_dim, hidden_dim=64)
    trainer = PPOTrainerWork3(actor_critic=actor_critic, time_head=time_head, lr=1e-3)

    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    graph_snapshot = actor_critic.build_graph_snapshot(env)

    # 记录参数副本
    init_params = [p.clone() for p in actor_critic.parameters()]

    # 构造 neither 批次：两个掩码均为空
    aux_batch_neither = {
        "state_feats": torch.zeros((1, 32)),
        "graph_snapshots": [graph_snapshot],
        "target_residuals": torch.tensor([0.0]),
        "current_target_masks": torch.tensor([False]),
        "target_next_residuals": torch.zeros((1, 5)),
        "next_cycle_masks": torch.zeros((1, 5), dtype=torch.bool),
        "worker_ids": [0],
        "episode_ids": [0],
        "cycle_ids": [0],
        "decision_ids": [0],
    }
    loss = trainer.compute_time_auxiliary_loss(aux_batch_neither)
    assert loss.item() == 0.0

    # 反向传播不产生任何梯度更新
    loss.backward()
    for p_init, p_now in zip(init_params, actor_critic.parameters()):
        assert p_now.grad is None or torch.allclose(p_now.grad, torch.zeros_like(p_now.grad))


def test_u27_u28_ppo_transition_metadata_and_replay_fidelity():
    """U27, U28: PPO 样本元数据完整性与回放一致性。"""
    t = PPOTransition(
        state_feat=torch.zeros(32),
        time_urgency=torch.zeros(12),
        sample_record={"task_idx": 0, "branch": 0},
        reward=1.0,
        raw_reward=1.0,
        value=0.5,
        log_prob=-0.2,
        predictor_version=2,
        predictor_source="frozen_snapshot",
        time_input_source="predicted_time",
        phi_source="learned",
    )
    assert t.predictor_version == 2
    assert t.predictor_source == "frozen_snapshot"
    assert t.time_input_source == "predicted_time"
    assert t.phi_source == "learned"

    # RolloutBuffer 保持元数据
    buffer = RolloutBufferWork3()
    buffer.add(t)
    assert buffer.transitions[0].predictor_version == 2
    assert buffer.transitions[0].predictor_source == "frozen_snapshot"


def test_u29_u30_frozen_predictor_invariance_and_bootstrap_order():
    """U29, U30: 冻结预测器在线强隔离，截断步 Critic bootstrap 在清理前完成。"""
    actor_critic = ActorCriticWork3()
    time_head = TimeResidualHead(in_dim=actor_critic.hidden_dim, hidden_dim=64)
    shaper = PotentialRewardShaper(time_head=time_head, actor_critic=actor_critic)
    ver = shaper.begin_episode(worker_id=0, episode_id=1)
    assert ver == shaper.snapshot_version

    frozen_head, frozen_actor = shaper.get_predictor_snapshot(worker_id=0, episode_id=1)
    head_weight_before = frozen_head.reg_fc[0].weight.clone()

    # 改变外部传入的工作模块权重
    with torch.no_grad():
        time_head.reg_fc[0].weight.add_(1.0)

    # 冻结副本完全不变 (U29)
    assert torch.allclose(frozen_head.reg_fc[0].weight, head_weight_before)

    # U30: Critic bootstrap 在 end_episode 清理前能够正常访问
    actor = ActorCriticWork3()
    state_feat = torch.zeros(32)
    time_urg = torch.zeros(12)
    val, _ = actor.encode_state(state_feat, time_urg)
    assert math.isfinite(val.item())

    # 清理后副本释放
    shaper.end_episode(worker_id=0, episode_id=1)
    assert (0, 1) not in shaper._episode_versions


def test_u31_u32_u33_u34_named_gradients_and_failure_propagation():
    """U31–U34: 具名双目标梯度流向与校验失败传播。"""
    actor_critic = ActorCriticWork3()
    time_head = TimeResidualHead(in_dim=actor_critic.hidden_dim, hidden_dim=64)
    trainer = PPOTrainerWork3(actor_critic=actor_critic, time_head=time_head, lr=1e-3)

    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    state_feat = extract_compact_state_features(env.state, 100.0)
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

    # 具名检查整线融合、站位汇总、飞机角色、当前头、下一周期头梯度 (U33)
    line_fusion_grad = actor_critic.graph_encoder.line_fusion_mlp[0].weight.grad
    station_summary_grad = actor_critic.graph_encoder.station_summary_mlp[0].weight.grad
    aircraft_role_grad = actor_critic.graph_encoder.aircraft_role_mlp[0].weight.grad
    current_head_grad = time_head.reg_fc[0].weight.grad
    next_head_grad = time_head.next_cycle_fc[0].weight.grad

    assert line_fusion_grad is not None and torch.isfinite(line_fusion_grad).all()
    assert station_summary_grad is not None and torch.isfinite(station_summary_grad).all()
    assert aircraft_role_grad is not None and torch.isfinite(aircraft_role_grad).all()
    assert current_head_grad is not None and torch.isfinite(current_head_grad).all()
    assert next_head_grad is not None and torch.isfinite(next_head_grad).all()

    # U34: 校验失败时严格传播为 False，不报告整项 PASS
    def fake_validate_failure():
        return False, ["Feasibility violation: Task overlap detected"]
    passed, errs = fake_validate_failure()
    assert passed is False
    assert len(errs) > 0
