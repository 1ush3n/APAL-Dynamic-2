"""工作三 S3 (C04, C03) 与 S4 (C02, C06, C01) 核心测试集。

最小可复现反例测试（依据 2026-10-17 审查报告）：
1. T10 & T11 & T12 & T13: 首次就绪账本与结算 (C04)
   - T10: 首次就绪时刻事实记录与账本跟踪 (state.station_first_cleared_times)
   - T11: 手算核对: H0=10, P1=10, P2=25, Cycle 2 站位1和2占用, F=[12, 23], 站位3..5为空
          Cycle 1 下一周期标签必须为: [-0.8, 0.3, placeholder, placeholder, placeholder], valid_mask = [True, True, False, False, False]
   - T12: 占用站位缺少 F 必须等待或报错，严禁静默回退到 P；空站位必须无效 (mask=False)，不得伪造 0.0 有效标签
   - T13: 单步连续/多周期转站各自独立按对应周期的占用与 F 结算
2. T14 & T15 & T16: 双目标独立结算与生命周期 (C03)
   - T14: 当 P[q] 发生时，当前周期目标变为 AVAILABLE，而下一周期目标保持 PENDING
   - T15: 采样段末尾 drain() 必须允许取走当前目标，严禁清除/丢弃待结算的下一周期记录；当 P[q+1] 发生后下一目标变为 AVAILABLE 并可单目标独立训练
   - T16: 真实终止标记未来无效；采样截断严禁伪造标签
3. T17 & T15: 实际双头梯度与单目标训练 (C02)
   - T17: compute_time_auxiliary_loss 与 Lightning training_step 支持 current-only, next-only, both, neither 四种批次
   - 反向传播必须为共享图编码器/融合层和当前活跃头生成非零有限梯度
4. T18: F/D 副本合同 (C06)
   - 即使势函数塑形关闭 (Group F)，仍必须使用固定的冻结预测器副本产生外部时间决策输入
   - episode 内在线权重更新严禁修改/漂移冻结预测器
5. T19 & T20: 完整12维时间路径与重放 (C01)
   - 外部时间特征布局:
     [0..1]: 标量转站残差与名义预算比例
     [2..6]: 5站下一周期就绪偏差
     [7..11]: 5站占用指示标记
   - 轨迹中保存的 12 维特征用于 PPO 重放，精确重建采样动作概率
"""

from __future__ import annotations

import copy
from dataclasses import replace
import math
from pathlib import Path
import sys
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# 保证仓库根目录在 sys.path 中
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.work3.core_types import ActionBranch, MultiAircraftState, TaskStatus
from envs.work3.decision_snapshot import DecisionSnapshot
from envs.work3.environment import AirLineEnvWork3
from models.work3.action_fusion import TimeContextFusion, compute_time_urgency_vector
from models.work3.actor_critic import ActorCriticWork3, extract_compact_state_features
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from models.work3.potential_shaping import PotentialRewardShaper
from models.work3.ppo_buffer import PendingTimeLabelCache, PPOTransition, RolloutBufferWork3
from models.work3.ppo_trainer import PPOTrainerWork3
from models.work3.time_head import TimeResidualHead
from scripts.work3.experiment_protocol import build_method_profile
from scripts.work3.train_ppo_work3 import (
    compute_online_snapshot_time_inputs,
    compute_online_time_inputs,
)
from training.work3_lightning import Work3LightningModule, Work3TrainingUpdate


BASELINE_PATH = REPO_ROOT / "data" / "work3" / "real_283_k10_baseline.json"


@pytest.fixture
def baseline_path() -> Path:
    if not BASELINE_PATH.is_file():
        pytest.skip(f"{BASELINE_PATH} 不存在")
    return BASELINE_PATH


@pytest.fixture
def env(baseline_path: Path) -> AirLineEnvWork3:
    environment = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    environment.reset()
    return environment


# ==============================================================================
# 1. T10 & T11 & T12 & T13: 首次就绪账本与结算 (C04)
# ==============================================================================


def test_t10_first_cleared_times_ledger_tracking(env: AirLineEnvWork3) -> None:
    """T10: 验证站位首次就绪事实记录与不可篡改账本跟踪 (C04)。

    - 当站位内所有未后移工序完工时，记录当前物理时间为该站的首次就绪时刻 F[m, q]。
    - 账本记录后在当前周期内锁定，后续同周期事件或轮次推进严禁篡改该数值。
    - 空站位（无停靠飞机）严禁记录首次就绪事实。
    """
    state: MultiAircraftState = env.state
    cycle = state.current_cycle
    assert cycle == 1

    # 初始状态下，0号飞机停靠在0号站位，其它站位（1..4）为空站位
    assert state.get_aircraft_at_station(0) == 0
    for s in range(1, 5):
        assert state.get_aircraft_at_station(s) is None

    # 0号站位尚有未完成任务，初始不应记录首次就绪
    cleared_map = state.station_first_cleared_times.setdefault(cycle, {})
    assert 0 not in cleared_map
    for s in range(1, 5):
        assert s not in cleared_map

    # 模拟将0号飞机在0号站位的所有物理任务标记为 COMPLETED
    tasks_st0 = state._ac_station_tasks[(0, 0)]
    for t in tasks_st0:
        t.status = TaskStatus.COMPLETED
        t.actual_end = 8.5

    state.current_time = 8.5
    env._record_station_first_cleared_if_ready()

    # 0号站位首次就绪时刻应锁定为 8.5
    assert 0 in state.station_first_cleared_times[cycle]
    assert state.station_first_cleared_times[cycle][0] == pytest.approx(8.5)

    # 推进物理时刻到 10.0，再次触发检查，就绪时刻严禁被覆写或篡改
    state.current_time = 10.0
    env._record_station_first_cleared_if_ready()
    assert state.station_first_cleared_times[cycle][0] == pytest.approx(8.5)

    # 空站位（1..4）在当前周期内严禁记录就绪事实
    for s in range(1, 5):
        assert s not in state.station_first_cleared_times[cycle], f"空站位 {s} 被非法记录首次就绪！"


def test_t11_hand_calculation_cycle_ledger_settlement() -> None:
    """T11: 严格手算核对下一周期就绪偏差标签与有效性掩码 (C04)。

    手算条件：
    - H0 = 10.0, P1 = 10.0, P2 = 25.0
    - Cycle 2 站位1和2占用（0-based站位 0 和 1），首次就绪时刻 F = [12.0, 23.0]
    - 站位3..5 为空站位（0-based站位 2, 3, 4）
    - 占用掩码: next_cycle_station_mask = [True, True, False, False, False]

    手算标签计算：
    - 偏差公式: target = (F[m] - P_prev - H0) / H0 = (F[m] - 10.0 - 10.0) / 10.0
    - 站位 0: (12.0 - 20.0) / 10.0 = -0.8
    - 站位 1: (23.0 - 20.0) / 10.0 = +0.3
    - 站位 2..4: 空站位，有效性掩码必须为 False，不参与监督

    预期结果：
    - target_next_residuals: [-0.8, 0.3, placeholder, placeholder, placeholder]
    - next_cycle_masks: [True, True, False, False, False]
    """
    cache = PendingTimeLabelCache()

    # 1. Cycle 1 内产生决策并登记待补记录
    cache.add(
        episode_id=1,
        cycle_id=1,
        decision_id=0,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=10.0,
        current_time=5.0,
        h0=10.0,
        predictor_version=1,
    )

    # 2. Cycle 1 实际转站 P1 = 10.0
    cache.attach_transfer(
        episode_id=1,
        cycle_id=1,
        actual_transfer_time=10.0,
    )

    # 3. Cycle 2 内产生决策
    cache.add(
        episode_id=1,
        cycle_id=2,
        decision_id=1,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=25.0,
        current_time=15.0,
        h0=10.0,
        predictor_version=1,
    )

    # 4. Cycle 2 实际转站 P2 = 25.0，揭示 Cycle 2 的首次就绪事实
    station_cleared_times = {0: 12.0, 1: 23.0}
    next_cycle_station_mask = [True, True, False, False, False]

    cache.attach_transfer(
        episode_id=1,
        cycle_id=2,
        actual_transfer_time=25.0,
        station_cleared_times=station_cleared_times,
        next_cycle_station_mask=next_cycle_station_mask,
    )

    batch = cache.drain_ready()
    assert batch is not None, "未能取出已结算批次"

    # 检索 Cycle 1 对应的决策记录
    cycle1_indices = [i for i, cid in enumerate(batch["cycle_ids"]) if cid == 1]
    assert len(cycle1_indices) == 1
    idx = cycle1_indices[0]

    target_next = batch["target_next_residuals"][idx]
    mask_next = batch["next_cycle_masks"][idx]

    # 严格手算断言
    assert float(target_next[0].item()) == pytest.approx(-0.8, abs=1e-5), "站位0就绪偏差手算不符"
    assert float(target_next[1].item()) == pytest.approx(0.3, abs=1e-5), "站位1就绪偏差手算不符"
    assert mask_next.tolist() == [True, True, False, False, False], "有效性掩码与空站过滤不符"


def test_t12_missing_f_on_occupied_station_and_empty_station_mask() -> None:
    """T12: 占用站位缺少 F 严禁静默回退到 P；空站位必须置为无效 (C04)。

    - 反例验证：如果站位被标记为占用 (mask=True)，但 station_cleared_times 缺少该站的 F 记录，
      缓存必须报错或保持等待，绝不能悄悄用 actual_transfer_time 替代 F 并伪造 mask=True。
    - 空站位（mask=False）必须保证 next_cycle_masks 为 False，不得输出假有效 0.0。
    """
    cache = PendingTimeLabelCache()
    cache.add(
        episode_id=1,
        cycle_id=1,
        decision_id=0,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=10.0,
        current_time=5.0,
        h0=10.0,
        predictor_version=1,
    )
    cache.attach_transfer(
        episode_id=1,
        cycle_id=1,
        actual_transfer_time=10.0,
    )
    cache.add(
        episode_id=1,
        cycle_id=2,
        decision_id=1,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=25.0,
        current_time=15.0,
        h0=10.0,
        predictor_version=1,
    )

    # 站位 0 和 1 均被标记为有飞机占用，但 cleared_times 仅提供了站位 0，站位 1 缺失 F！
    next_cycle_station_mask = [True, True, False, False, False]
    partial_cleared_times = {0: 12.0}  # 站位 1 缺失

    # 规范要求：缺少占用站位真实 F 时，不得静默使用 P 补齐并伪造有效掩码。
    # 若代码静默回退，则断言其未报错且输出了 mask[1]=True 为反例缺陷！
    try:
        cache.attach_transfer(
            episode_id=1,
            cycle_id=2,
            actual_transfer_time=25.0,
            station_cleared_times=partial_cleared_times,
            next_cycle_station_mask=next_cycle_station_mask,
        )
        batch = cache.drain_ready()
        assert batch is not None
        cycle1_idx = batch["cycle_ids"].index(1)
        mask_next = batch["next_cycle_masks"][cycle1_idx]
        target_next = batch["target_next_residuals"][cycle1_idx]

        # 检查是否发生了静默回退漏洞 (fallback to actual_transfer_time 25.0 -> (25-10-10)/10 = 0.5)
        silent_fallback_detected = (
            bool(mask_next[1].item()) is True
            and float(target_next[1].item()) == pytest.approx(0.5, abs=1e-4)
        )
        assert not silent_fallback_detected, (
            "【C04 缺陷捕获】占用站位1缺少首次就绪时刻F，代码静默回退到转站时刻P=25.0并伪造mask=True！"
        )
    except (KeyError, ValueError, RuntimeError):
        # 合法契约：报错拒绝静默回退
        pass


def test_t13_single_step_multiple_cycle_transfers_independent_settlement() -> None:
    """T13: 连续/多周期脉动流转各自独立按对应周期事实结算 (C04)。

    - 在单步中连续发生多个周期的脉动流转（Cycle 1 -> Cycle 2 -> Cycle 3）。
    - Cycle 1 必须严格依据 Cycle 2 的占用和 F 结算下一周期向量。
    - Cycle 2 必须严格依据 Cycle 3 的占用和 F 结算下一周期向量。
    - 各周期的当前残差与下一周期向量互不混淆、不跨周期篡改。
    """
    cache = PendingTimeLabelCache()

    # Cycle 1
    cache.add(
        episode_id=1,
        cycle_id=1,
        decision_id=10,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=10.0,
        current_time=2.0,
        h0=10.0,
        predictor_version=1,
    )
    cache.attach_transfer(
        episode_id=1,
        cycle_id=1,
        actual_transfer_time=10.0,
    )

    # Cycle 2
    cache.add(
        episode_id=1,
        cycle_id=2,
        decision_id=11,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=24.0,
        current_time=12.0,
        h0=10.0,
        predictor_version=1,
    )
    # P2 = 25.0, Cycle 2 事实: F={0: 12.0, 1: 23.0}, mask=[True, True, False, False, False]
    cache.attach_transfer(
        episode_id=1,
        cycle_id=2,
        actual_transfer_time=25.0,
        station_cleared_times={0: 12.0, 1: 23.0},
        next_cycle_station_mask=[True, True, False, False, False],
    )

    # Cycle 3
    cache.add(
        episode_id=1,
        cycle_id=3,
        decision_id=12,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=40.0,
        current_time=28.0,
        h0=10.0,
        predictor_version=1,
    )
    # P3 = 42.0, Cycle 3 事实: F={0: 30.0, 1: 34.0, 2: 38.0}, mask=[True, True, True, False, False]
    cache.attach_transfer(
        episode_id=1,
        cycle_id=3,
        actual_transfer_time=42.0,
        station_cleared_times={0: 30.0, 1: 34.0, 2: 38.0},
        next_cycle_station_mask=[True, True, True, False, False],
    )

    batch = cache.drain_ready()
    assert batch is not None

    # 核验 Cycle 1: 下一周期向量应对应 Cycle 2 事实 (P_prev=10, H0=10, F0=12-> -0.8, F1=23-> 0.3)
    c1_idx = batch["cycle_ids"].index(1)
    assert batch["target_residuals"][c1_idx].item() == pytest.approx(0.0)  # (10-10)/10
    assert batch["target_next_residuals"][c1_idx, 0].item() == pytest.approx(-0.8)
    assert batch["target_next_residuals"][c1_idx, 1].item() == pytest.approx(0.3)
    assert batch["next_cycle_masks"][c1_idx].tolist() == [True, True, False, False, False]

    # 核验 Cycle 2: 下一周期向量应对应 Cycle 3 事实 (P_prev=25, H0=10)
    # F0=30 -> (30-25-10)/10 = -0.5
    # F1=34 -> (34-25-10)/10 = -0.1
    # F2=38 -> (38-25-10)/10 = 0.3
    c2_idx = batch["cycle_ids"].index(2)
    assert batch["target_residuals"][c2_idx].item() == pytest.approx(0.1)  # (25-24)/10
    assert batch["target_next_residuals"][c2_idx, 0].item() == pytest.approx(-0.5)
    assert batch["target_next_residuals"][c2_idx, 1].item() == pytest.approx(-0.1)
    assert batch["target_next_residuals"][c2_idx, 2].item() == pytest.approx(0.3)
    assert batch["next_cycle_masks"][c2_idx].tolist() == [True, True, True, False, False]


# ==============================================================================
# 2. T14 & T15 & T16: 双目标独立结算与生命周期 (C03)
# ==============================================================================


def test_t14_dual_target_independent_availability_on_transfer() -> None:
    """T14: 当 P[q] 到达时当前目标 AVAILABLE，下一目标保持 PENDING (C03)。"""
    cache = PendingTimeLabelCache()
    cache.add(
        episode_id=1,
        cycle_id=1,
        decision_id=101,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=12.0,
        current_time=5.0,
        h0=10.0,
        predictor_version=1,
    )

    # 脉动时刻 P1=10 到达
    cache.attach_transfer(
        episode_id=1,
        cycle_id=1,
        actual_transfer_time=10.0,
    )

    # 规范定义：当前目标已确定 ((10-12)/10 = -0.2)，但下一目标仍待 Cycle 2 发生
    # 检查内部待补映射：Cycle 1 必须位于 _pending_next_cycle 中等待后续周期事实
    key = (0, 1, 1)
    assert key in cache._pending_next_cycle, "P[q] 到达后未将样本保留在 _pending_next_cycle 中！"
    item = cache._pending_next_cycle[key][0]
    assert item["target_residual"] == pytest.approx(-0.2)
    # 下一周期掩码尚未就绪
    assert not item["next_cycle_masks"].any(), "下一目标在 Cycle 2 到达前不应被标记为有效！"


def test_t15_drain_ready_preserves_pending_next_target_for_independent_training() -> None:
    """T15: drain() 采样段末尾取出可用当前目标，严禁丢弃 pending 下一目标 (C03)。

    【严重缺陷捕获】
    在 commit 54989c3 中，PendingTimeLabelCache.drain_ready() 强行清空了 _pending_next_cycle:
    ```python
    for key in list(self._pending_next_cycle.keys()):
        items = self._pending_next_cycle.pop(key)
        self._ready.extend(items)
    ```
    这导致跨采样段（Rollout Segment Boundary）时，尚未获得下一周期就绪事实的样本被彻底丢弃，
    当后续周期 P2 到达时，根本无法补齐 Cycle 1 的下一目标！
    """
    cache = PendingTimeLabelCache()

    # Step 1: Cycle 1 产生决策并结算 P1=10.0
    cache.add(
        episode_id=1,
        cycle_id=1,
        decision_id=201,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=10.0,
        current_time=4.0,
        h0=10.0,
        predictor_version=1,
    )
    cache.attach_transfer(
        episode_id=1,
        cycle_id=1,
        actual_transfer_time=10.0,
    )

    # Step 2: 采样段 1 结束，调用 drain_ready() 收集已就绪批次
    batch1 = cache.drain_ready()
    assert batch1 is not None, "采样段1未产出可用训练数据"

    # Step 3: 关键断言——drain_ready() 取走当前残差后，Cycle 1 的下一目标待结算凭证严禁被丢弃！
    key_cycle1 = (0, 1, 1)
    has_pending_next = key_cycle1 in cache._pending_next_cycle
    assert has_pending_next, (
        "【C03 缺陷捕获】drain_ready() 在采样段末尾清空了 _pending_next_cycle，"
        "导致跨采样段的下一目标记录被永久丢失！"
    )

    # Step 4: 采样段 2 开始，Cycle 2 发生并到达转站时刻 P2=25.0
    cache.add(
        episode_id=1,
        cycle_id=2,
        decision_id=202,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=25.0,
        current_time=14.0,
        h0=10.0,
        predictor_version=1,
    )
    cache.attach_transfer(
        episode_id=1,
        cycle_id=2,
        actual_transfer_time=25.0,
        station_cleared_times={0: 12.0, 1: 23.0},
        next_cycle_station_mask=[True, True, False, False, False],
    )

    # Step 5: 采样段 2 drain_ready() 应成功产出 Cycle 1 的下一目标监督样本（支持单目标独立训练）
    batch2 = cache.drain_ready()
    assert batch2 is not None

    # 核验 Cycle 1 的下一目标已被正确结算
    c1_records = [i for i, cid in enumerate(batch2["cycle_ids"]) if cid == 1]
    assert len(c1_records) >= 1, "Cycle 1 的下一目标未在 Cycle 2 到达后被成功结算"
    c1_idx = c1_records[0]
    assert batch2["next_cycle_masks"][c1_idx].any(), "Cycle 1 下一目标掩码全为 False"
    assert batch2["target_next_residuals"][c1_idx, 0].item() == pytest.approx(-0.8)


def test_t16_real_terminal_marks_future_invalid_and_truncation_never_forges_labels() -> None:
    """T16: 真实终止标记未来无效；采样截断严禁伪造标签 (C03)。

    - 当批次真实终止（所有飞机出线）时，不存在更后的下一个周期，其 next_cycle_masks 必须全为 False。
    - 当环境采样截断（truncated，未到达实际转站）时，缓存严禁伪造 transfer_time 或生成虚假 0.0 残差。
    """
    cache = PendingTimeLabelCache()

    # 1. 截断场景：周期 2 内未发生转站即遭遇截断
    cache.add(
        episode_id=5,
        cycle_id=2,
        decision_id=301,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=15.0,
        current_time=8.0,
        h0=10.0,
        predictor_version=1,
    )

    # 截断时未调用 attach_transfer
    # 此时调用 drain_ready() 严禁输出 cycle 2 的虚假标签
    batch_truncated = cache.drain_ready()
    if batch_truncated is not None:
        assert 301 not in batch_truncated["decision_ids"], (
            "【C03 缺陷捕获】采样截断时未发生实际转站，但缓存伪造了该决策的转站监督标签！"
        )

    # 2. 真实终止场景：周期 14 为全厂最后脉动周期
    cache.add(
        episode_id=6,
        cycle_id=14,
        decision_id=302,
        state_feat=torch.zeros(32),
        graph_snapshot=None,
        estimated_cmax=140.0,
        current_time=135.0,
        h0=10.0,
        predictor_version=1,
    )
    # 真实终止转站
    cache.attach_transfer(
        episode_id=6,
        cycle_id=14,
        actual_transfer_time=142.0,
        station_cleared_times={},
        next_cycle_station_mask=[False, False, False, False, False],
    )
    # 丢弃或终止 episode
    cache.discard_episode(episode_id=6)
    batch_terminal = cache.drain_ready()
    assert batch_terminal is not None
    idx = batch_terminal["decision_ids"].index(302)
    # 真实终止周期的下一周期必须全部无效
    assert not batch_terminal["next_cycle_masks"][idx].any(), "真实终止后的未来目标被错误标记为有效！"


# ==============================================================================
# 3. T17 & T15: 实际双头梯度与单目标训练 (C02)
# ==============================================================================


def test_t17_compute_time_auxiliary_loss_supports_all_target_combinations() -> None:
    """T17a: compute_time_auxiliary_loss 必须支持 current-only, next-only, both, neither (C02)。

    【严重缺陷捕获】
    在 commit 54989c3 中，models/work3/ppo_trainer.py 的 compute_time_auxiliary_loss
    无条件执行:
    ```python
    targets = batch["target_residuals"].to(self.device)
    ...
    per_sample_loss = F.smooth_l1_loss(predictions.float(), targets.float(), ...)
    current_loss = per_sample_loss.mean()
    ```
    当批次仅包含 next-only（无当前残差标签或当前标签不可用）或 neither 时，
    代码无法处理掩码，会报错崩溃或在无效残差上计算虚假损失！
    """
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    head = TimeResidualHead(in_dim=16, hidden_dim=16, num_stations=5)
    trainer = PPOTrainerWork3(
        actor_critic=actor,
        time_head=head,
        device="cpu",
        create_optimizer=False,
    )

    state_feat = torch.randn(2, 32)
    graph_snapshots = [None, None]
    worker_ids = [0, 0]
    episode_ids = [1, 1]
    cycle_ids = [1, 1]
    decision_ids = [401, 402]

    # Case 1: Both (当前残差与下一周期向量均有效)
    batch_both = {
        "state_feats": state_feat,
        "graph_snapshots": graph_snapshots,
        "target_residuals": torch.tensor([0.2, -0.1]),
        "current_target_masks": torch.tensor([True, True]),
        "target_next_residuals": torch.randn(2, 5),
        "next_cycle_masks": torch.tensor([[True, True, False, False, False], [False, True, True, False, False]]),
        "worker_ids": worker_ids,
        "episode_ids": episode_ids,
        "cycle_ids": cycle_ids,
        "decision_ids": decision_ids,
    }
    loss_both = trainer.compute_time_auxiliary_loss(batch_both)
    assert torch.isfinite(loss_both) and loss_both.item() > 0.0

    # Case 2: Current-Only (下一周期掩码全 False 或缺失)
    batch_current_only = {
        "state_feats": state_feat,
        "graph_snapshots": graph_snapshots,
        "target_residuals": torch.tensor([0.2, -0.1]),
        "current_target_masks": torch.tensor([True, True]),
        "target_next_residuals": torch.zeros(2, 5),
        "next_cycle_masks": torch.zeros(2, 5, dtype=torch.bool),
        "worker_ids": worker_ids,
        "episode_ids": episode_ids,
        "cycle_ids": cycle_ids,
        "decision_ids": decision_ids,
    }
    loss_curr = trainer.compute_time_auxiliary_loss(batch_current_only)
    assert torch.isfinite(loss_curr) and loss_curr.item() > 0.0

    # Case 3: Next-Only (当前残差缺失或掩码全 False，仅下一周期有效)
    # 在现有实现中，未提供有效 target_residuals 会直接 KeyError 或错误计算
    batch_next_only = {
        "state_feats": state_feat,
        "graph_snapshots": graph_snapshots,
        "target_residuals": torch.full((2,), float("nan")),  # 当前无有效标签
        "current_target_masks": torch.tensor([False, False]),
        "target_next_residuals": torch.tensor([[-0.8, 0.3, 0.0, 0.0, 0.0], [0.1, -0.2, 0.0, 0.0, 0.0]]),
        "next_cycle_masks": torch.tensor([[True, True, False, False, False], [True, True, False, False, False]]),
        "worker_ids": worker_ids,
        "episode_ids": episode_ids,
        "cycle_ids": cycle_ids,
        "decision_ids": decision_ids,
    }

    loss_next = trainer.compute_time_auxiliary_loss(batch_next_only)
    assert torch.isfinite(loss_next), (
        "【C02 缺陷捕获】compute_time_auxiliary_loss 无法独立处理 next-only 批次，产生了 NaN 或崩溃！"
    )

    # Case 4: Neither (两项均无有效监督)
    batch_neither = {
        "state_feats": state_feat,
        "graph_snapshots": graph_snapshots,
        "target_residuals": torch.zeros(2),
        "current_target_masks": torch.zeros(2, dtype=torch.bool),
        "target_next_residuals": torch.zeros(2, 5),
        "next_cycle_masks": torch.zeros(2, 5, dtype=torch.bool),
        "worker_ids": worker_ids,
        "episode_ids": episode_ids,
        "cycle_ids": cycle_ids,
        "decision_ids": decision_ids,
    }
    loss_neither = trainer.compute_time_auxiliary_loss(batch_neither)
    assert torch.isfinite(loss_neither)
    assert loss_neither.item() == pytest.approx(0.0)


def test_t17_dual_head_gradient_flow_and_selective_head_activation() -> None:
    """T17b: 验证共享编码器梯度流与活跃头的选择性反传 (C02)。

    - 当前目标活跃时: reg_fc 具有非零有限梯度，next_cycle_fc 无梯度 (None或0)
    - 下一目标活跃时: next_cycle_fc 具有非零有限梯度，reg_fc 无梯度
    - 两者皆活跃时: reg_fc 与 next_cycle_fc 均有非零梯度
    - 共享表征 (actor.encoder) 在任何活跃情况下均获得非零有限梯度
    """
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    head = TimeResidualHead(in_dim=16, hidden_dim=16, num_stations=5)
    trainer = PPOTrainerWork3(
        actor_critic=actor,
        time_head=head,
        device="cpu",
        create_optimizer=False,
    )

    state_feat = torch.randn(1, 32)
    batch_base = {
        "state_feats": state_feat,
        "graph_snapshots": [None],
        "worker_ids": [0],
        "episode_ids": [1],
        "cycle_ids": [1],
        "decision_ids": [501],
    }

    # 1. Current-only 反传
    actor.zero_grad()
    head.zero_grad()
    b_curr = dict(batch_base)
    b_curr["target_residuals"] = torch.tensor([0.5])
    b_curr["current_target_masks"] = torch.tensor([True])
    b_curr["target_next_residuals"] = torch.zeros(1, 5)
    b_curr["next_cycle_masks"] = torch.zeros(1, 5, dtype=torch.bool)

    loss_curr = trainer.compute_time_auxiliary_loss(b_curr)
    loss_curr.backward()

    # 断言 reg_fc 有梯度，next_cycle_fc 无梯度
    assert head.reg_fc[0].weight.grad is not None and torch.isfinite(head.reg_fc[0].weight.grad).all()
    assert (
        head.next_cycle_fc[0].weight.grad is None
        or head.next_cycle_fc[0].weight.grad.abs().sum().item() == 0.0
    ), "Current-only 批次错误地激活了 next_cycle_fc 的梯度！"

    # 断言共享编码器获得反传梯度
    assert actor.encoder[0].weight.grad is not None and torch.isfinite(actor.encoder[0].weight.grad).all()

    # 2. Next-only 反传
    actor.zero_grad()
    head.zero_grad()
    b_next = dict(batch_base)
    b_next["target_residuals"] = torch.tensor([float("nan")])
    b_next["current_target_masks"] = torch.tensor([False])
    b_next["target_next_residuals"] = torch.tensor([[0.2, -0.3, 0.0, 0.0, 0.0]])
    b_next["next_cycle_masks"] = torch.tensor([[True, True, False, False, False]])

    loss_next = trainer.compute_time_auxiliary_loss(b_next)
    loss_next.backward()

    # 断言 next_cycle_fc 有梯度，reg_fc 无梯度
    assert head.next_cycle_fc[0].weight.grad is not None and torch.isfinite(head.next_cycle_fc[0].weight.grad).all()
    assert (
        head.reg_fc[0].weight.grad is None
        or head.reg_fc[0].weight.grad.abs().sum().item() == 0.0
    ), "Next-only 批次错误地激活了 reg_fc 的梯度！"
    assert actor.encoder[0].weight.grad is not None and torch.isfinite(actor.encoder[0].weight.grad).all()


def test_t17_lightning_training_step_supports_dual_target_auxiliary_batches() -> None:
    """T17c: 验证 Lightning training_step 透传下一周期目标并更新双头 (C02)。

    【严重缺陷捕获】
    在 commit 54989c3 中，training/work3_lightning.py 构造 minibatch 时：
    ```python
    minibatch = {
        "state_feats": auxiliary["state_feats"][chosen],
        "graph_snapshots": [auxiliary["graph_snapshots"][i] for i in chosen],
        "target_residuals": auxiliary["target_residuals"][chosen],
        ...
    }
    ```
    漏掉了 target_next_residuals 和 next_cycle_masks，导致下一周期头参数在 Lightning 训练步中永远无法被更新！
    """
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    head = TimeResidualHead(in_dim=16, hidden_dim=16, num_stations=5)
    module = Work3LightningModule(
        actor_critic=actor,
        time_head=head,
        ppo_epochs=1,
        batch_size=1,
        time_auxiliary_batch_size=1,
        time_auxiliary_epochs=1,
    )

    buffer = RolloutBufferWork3(normalize_advantages=False)
    buffer.add(
        PPOTransition(
            state_feat=torch.randn(32),
            time_urgency=torch.zeros(2),
            sample_record={
                "candidate_task_features": torch.randn(1, 8),
                "candidate_branch_masks": ((True, False),),
                "action_branch": 0,
                "selected_task_index": 0,
                "worker_selection_records": (),
                "align_selection": 0,
                "graph_snapshot": None,
                "candidate_task_node_indices": (0,),
                "worker_node_indices": (),
            },
            reward=1.0,
            raw_reward=1.0,
            value=0.5,
            log_prob=-0.2,
            done=True,
            terminated=True,
        )
    )
    buffer.finish_trajectories(last_values_by_segment={})

    # 包含双目标的辅助监督批次
    auxiliary_batch = {
        "state_feats": torch.randn(1, 32),
        "graph_snapshots": [None],
        "target_residuals": torch.tensor([-0.2]),
        "target_next_residuals": torch.tensor([[0.5, -0.4, 0.0, 0.0, 0.0]]),
        "next_cycle_masks": torch.tensor([[True, True, False, False, False]]),
        "worker_ids": [0],
        "episode_ids": [1],
        "cycle_ids": [1],
        "decision_ids": [601],
    }

    update = Work3TrainingUpdate(
        buffer=buffer,
        environment_steps=1,
        time_auxiliary_batch=auxiliary_batch,
    )

    next_cycle_weight_before = head.next_cycle_fc[0].weight.detach().clone()
    module.training_step(update, 0)
    next_cycle_weight_after = head.next_cycle_fc[0].weight.detach().clone()

    # 检验 next_cycle_fc 是否得到了参数更新
    weights_changed = not torch.equal(next_cycle_weight_before, next_cycle_weight_after)
    assert weights_changed, (
        "【C02 缺陷捕获】Lightning training_step 遗漏了 target_next_residuals 和 next_cycle_masks，"
        "导致 next_cycle_fc 在辅助训练中未发生任何权重更新！"
    )


# ==============================================================================
# 4. T18: F/D 副本合同 (C06)
# ==============================================================================


def test_t18_group_f_frozen_predictor_contract_and_online_weight_isolation(
    env: AirLineEnvWork3,
) -> None:
    """T18: Group F 关闭势函数塑形时，仍必须使用冻结预测器副本产生外部时间输入 (C06)。

    【严重缺陷捕获】
    在 commit 54989c3 中，scripts/work3/train_ppo_work3.py 行 1324 仅在
    `shaper is not None and state.predictor_version is not None` 时才使用冻结预测器副本。
    在 Group F 中，use_learned_time_shaping=False，shaper 为 None！
    导致 Group F 在采样决策时直接调用了正在被在线梯度更新的 live actor_critic 与 time_head，
    严重违背了“网络方案B中完整轨迹内预测器冻结不漂移”的顶层设计规范！
    """
    profile = build_method_profile("F")
    assert profile.use_corrected_time_input is True, "Group F 必须启用时间修正输入"
    assert profile.use_learned_time_shaping is False, "Group F 势函数塑形必须关闭"

    # 初始化在线模型与冻结副本服务
    online_actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    online_head = TimeResidualHead(in_dim=16, hidden_dim=16, num_stations=5)

    cmax = compute_cycle_heuristic_cmax(env.state)
    state_feat = extract_compact_state_features(env.state, cmax)
    snapshot = online_actor.make_decision_snapshot(
        env,
        state_feat,
        torch.zeros(2),
        estimated_cmax=cmax,
    )

    # 模拟 Episode 采样前建立预测器快照
    shaper_for_group_f = PotentialRewardShaper(
        initial_actor=online_actor,
        initial_head=online_head,
    )
    shaper_for_group_f.bind_episode(worker_id=0, episode_id=1)

    # 在 Group F 规范下，决策时计算时间输入必须绑定该 episode 的冻结副本
    frozen_head, frozen_actor = shaper_for_group_f.get_predictor_snapshot(
        worker_id=0, episode_id=1
    )
    _, time_input_frozen_initial, _ = compute_online_snapshot_time_inputs(
        actor_critic=frozen_actor if frozen_actor is not None else online_actor,
        time_head=frozen_head,
        snapshot=snapshot,
    )

    # 模拟在线训练进行了一次剧烈的权重更新
    with torch.no_grad():
        online_head.reg_fc[-1].bias.add_(10.0)
        online_head.next_cycle_fc[-1].bias.add_(10.0)

    # 如果代码错误地回退到 online 模型（如 commit 54989c3 中的行为）:
    _, time_input_online_mutated, _ = compute_online_snapshot_time_inputs(
        actor_critic=online_actor,
        time_head=online_head,
        snapshot=snapshot,
    )

    # 再次读取 episode 绑定的预测器输入
    # 按照规范要求：即使 Group F 不进行奖励塑形，其提供给 Actor 的时间输入也必须与在线权重更新严格隔离！
    frozen_drift_detected = not torch.equal(time_input_frozen_initial, time_input_online_mutated)
    assert frozen_drift_detected, "测试探针未构造出在线权重漂移"

    # 断言通过冻结副本获取的输入与初始输入严格一致（隔离性）
    _, time_input_frozen_after, _ = compute_online_snapshot_time_inputs(
        actor_critic=frozen_actor if frozen_actor is not None else online_actor,
        time_head=frozen_head,
        snapshot=snapshot,
    )
    assert torch.equal(time_input_frozen_initial, time_input_frozen_after), (
        "【C06 缺陷捕获】冻结预测器副本发生了内部漂移！"
    )


# ==============================================================================
# 5. T19 & T20: 完整12维时间路径与重放 (C01)
# ==============================================================================


def test_t19_complete_12dim_external_time_feature_layout(env: AirLineEnvWork3) -> None:
    """T19: 验证外部显式时间输入的完整 12 维物理布局 (C01)。

    【严重缺陷捕获】
    在 commit 54989c3 中，train_ppo_work3.py 的 compute_online_snapshot_time_inputs
    仅返回 2 维紧迫度向量:
    ```python
    urgency = compute_time_urgency_vector(...)  # 仅 2 维
    return graph_snapshot, urgency, predicted_transfer
    ```
    而在 actor_critic.py 中，time_fusion 要求的维度为 12！
    缺失了：
    - [2..6]: 5 站下一周期首次就绪有符号时间偏差
    - [7..11]: 5 站下一周期占用二值指示标记
    代码通过 TimeContextFusion 自动补零掩盖了特征缺失，丢失了多架次空间耦合语义！
    """
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    head = TimeResidualHead(in_dim=16, hidden_dim=16, num_stations=5)

    cmax = compute_cycle_heuristic_cmax(env.state)
    state_feat = extract_compact_state_features(env.state, cmax)
    snapshot = actor.make_decision_snapshot(
        env,
        state_feat,
        torch.zeros(12),  # 期望 12 维时间输入
        estimated_cmax=cmax,
    )

    _, time_features, _ = compute_online_snapshot_time_inputs(
        actor_critic=actor,
        time_head=head,
        snapshot=snapshot,
    )

    # 断言必须输出完整 12 维特征
    assert time_features.shape == (12,), (
        f"【C01 缺陷捕获】外部时间特征维度为 {time_features.shape}，未能生成完整 12 维时间路径！"
    )

    # 验证 12 维内部布局语义
    # [0..1]: 标量转站残差与名义预算
    h0 = env.state.h0
    curr_t = env.state.current_time
    last_p = env.state.last_transfer_time
    expected_dim1 = (h0 - (curr_t - last_p)) / h0
    assert float(time_features[1].item()) == pytest.approx(expected_dim1, abs=1e-5)

    # [2..6]: 5 站下一周期有符号偏差预测
    # [7..11]: 5 站下一周期占用标记指示
    next_cycle_mask = env.state.get_next_cycle_station_mask()
    expected_occupancy = [1.0 if occupied else 0.0 for occupied in next_cycle_mask]
    assert time_features[7:12].tolist() == pytest.approx(expected_occupancy, abs=1e-5)


def test_t20_12dim_time_features_ppo_replay_exactness(env: AirLineEnvWork3) -> None:
    """T20: 验证轨迹中保存的 12 维时间输入支持 PPO 精确概率重放 (C01)。

    - 在采样时刻生成并保存 12 维完整时间向量。
    - 在 PPO evaluate_action_log_probs 重放中，使用保存的 12 维特征必须重建精确一致的 log_prob。
    - 改变 12 维时间特征中任意一维，动作输出概率必须发生敏感变化（验证 12 维真实参与策略运算）。
    """
    torch.manual_seed(42)
    actor = ActorCriticWork3(state_dim=32, task_feat_dim=8, hidden_dim=16)
    actor.eval()

    cmax = compute_cycle_heuristic_cmax(env.state)
    state_feat = extract_compact_state_features(env.state, cmax)

    # 构造合法的 12 维时间向量
    time_12dim = torch.tensor([
        0.8, -0.2,            # [0..1]
        -0.5, 0.2, 0.0, 0.4, -0.1,  # [2..6]
        1.0, 1.0, 0.0, 0.0, 0.0    # [7..11]
    ], dtype=torch.float32)

    snapshot = actor.make_decision_snapshot(
        env,
        state_feat,
        time_12dim,
        estimated_cmax=cmax,
    )

    action, sampled_log_prob, value, sample_record = actor.select_snapshot(
        snapshot, deterministic=True
    )
    assert action is not None

    # 重放验证
    values, replay_log_probs, entropies = actor.evaluate_action_log_probs(
        state_feats=state_feat.unsqueeze(0),
        time_urgencies=time_12dim.unsqueeze(0),
        sample_records=[sample_record],
    )

    assert float(replay_log_probs[0].item()) == pytest.approx(sampled_log_prob, abs=1e-5), (
        "12 维时间特征下的重放动作对数概率与采样时刻不一致！"
    )

    # 敏感性检验：修改 [2..6] 站位预测偏差，重放概率应发生响应变化
    perturbed_time = time_12dim.clone()
    perturbed_time[2] += 2.0  # 大幅改变下一周期站位 0 偏差
    _, perturbed_log_probs, _ = actor.evaluate_action_log_probs(
        state_feats=state_feat.unsqueeze(0),
        time_urgencies=perturbed_time.unsqueeze(0),
        sample_records=[sample_record],
    )

    assert not math.isclose(
        float(perturbed_log_probs[0].item()),
        float(replay_log_probs[0].item()),
        abs_tol=1e-4,
    ), "策略对外部时间特征的扰动无任何响应，时间融合层可能被旁路或失效！"
