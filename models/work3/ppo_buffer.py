"""工作三 条件分支 PPO 经验回放缓存 (Task 7.2)。

核心数学模型与技术规范：
依据《方法设计确认稿》第 4 节（条件分支强化学习）与标准 PPO 算法规范：
1. 存储多架次脉动节拍仿真中的步步决策转移 (Transition)：
   - state_feat: 32 维精炼物理状态特征张量；
   - time_urgency: 2 维显式时间紧迫度张量 [u_1, u_2]；
   - sample_record: 包含工序索引、分支决策、工人选择、二元对齐等重放元数据的字典；
   - reward: 经势函数塑形后的标量回报 R_t = r_t^raw + F_t；
   - value: Critic 估计的状态价值 V(s_t)；
   - log_prob: 采样时 Actor 输出的动作全量对数概率 log π_old(a_t|s_t)；
   - terminated/truncated: 分开的真实批次终止与采样截断标记。
2. 广义优势估计 (Generalized Advantage Estimation, GAE)：
   - δ_t = R_t + γ · V(s_{t+1}) · (1 - d_t) - V(s_t)
   - A_t = δ_t + (γ · λ) · (1 - d_t) · A_{t+1}
   - V_target(s_t) = A_t + V(s_t)
   - 批次内优势值标准化: A^_t = (A_t - μ_A) / (σ_A + 1e-8)
3. 随机 Mini-batch 生成器：
   - 严格保持张量特征与 sample_record 字典的对应关系，支持条件分支重放。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

import torch

from envs.work3.core_types import CycleExecutionRecord, validate_cycle_execution_record


@dataclass
class PPOTransition:
    """单步 PPO 交互样本数据类。"""
    state_feat: torch.Tensor          # (32,)
    time_urgency: torch.Tensor        # (2,)
    sample_record: dict[str, Any]     # 重放字典 (含 task_idx, branch, cand_feats, etc.)
    reward: float                     # 塑形后步步奖励
    raw_reward: float                 # 原始物理步步奖励
    value: float                      # Critic 估计的 V(s_t)
    log_prob: float                   # 采样时刻的动作对数概率 log π_old
    worker_id: int = 0
    episode_id: int = 0
    segment_id: int = 0

    @property
    def is_terminated(self) -> bool:
        """兼容旧done字段；截断不是吸收终止，必须保留bootstrap。"""
        if self.terminated is not None:
            return bool(self.terminated)
        return bool(self.done and not self.truncated)
    done: bool = False                # 旧接口兼容字段
    action_dict: dict[str, Any] = field(default_factory=dict)
    terminated: bool | None = None   # 真实终止；None 时回退到旧 done 语义
    truncated: bool = False           # 采样截断，不代表生产终止
    predictor_version: int | None = None
    predictor_source: str = "none"
    time_input_source: str = "none"
    phi_source: str = "none"


@dataclass(frozen=True)
class PendingTimeLabel:
    """等待真实转站时刻揭示的决策级时间监督样本。"""

    worker_id: int
    episode_id: int
    cycle_id: int
    decision_id: int
    state_feat: torch.Tensor
    graph_snapshot: Any
    estimated_cmax: float
    current_time: float
    h0: float
    predictor_version: int | None
    time_urgency: torch.Tensor | None = None


class PendingTimeLabelCache:
    """跨PPO采样段保存周期样本，直到真实转站事件补齐标签。"""

    def __init__(self) -> None:
        self._pending: dict[tuple[int, int, int], list[PendingTimeLabel]] = {}
        self._pending_next_cycle: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        self._ready: list[dict[str, Any]] = []
        self._last_decision_ids: dict[tuple[int, int], int] = {}
        self._last_transfer_times: dict[tuple[int, int, int], float] = {}
        self._transfer_facts: dict[tuple[int, int, int], tuple[Any, ...]] = {}
        self.abandoned_current_samples: int = 0
        self.abandoned_next_samples: int = 0

    @property
    def waiting_current_samples(self) -> int:
        """等待当前周期转站时刻 P_q 结算的样本总数。"""
        return sum(len(samples) for samples in self._pending.values())

    @property
    def waiting_next_samples(self) -> int:
        """等待下一周期就绪事实 F[m, q+1] 结算的样本总数。"""
        return sum(
            sum(1 for item in items if not item.get("has_next_ready", False))
            for items in self._pending_next_cycle.values()
        )

    @property
    def available_current_samples(self) -> int:
        """当前残差已就绪且尚未被训练取出的样本数。"""
        return sum(
            sum(1 for item in items if item.get("has_current_ready", False) and not item.get("current_drained", False))
            for items in self._pending_next_cycle.values()
        )

    @property
    def available_next_samples(self) -> int:
        """下一周期残差已就绪且尚未被训练取出的样本数。"""
        return sum(
            sum(1 for item in items if item.get("has_next_ready", False) and not item.get("next_drained", False))
            for items in self._pending_next_cycle.values()
        )

    @property
    def available_next_station_targets(self) -> int:
        """下一周期残差已就绪且尚未取出的有效站位目标总数。"""
        count = 0
        for items in self._pending_next_cycle.values():
            for item in items:
                if item.get("has_next_ready", False) and not item.get("next_drained", False):
                    mask = item.get("next_cycle_masks")
                    if mask is not None and isinstance(mask, torch.Tensor):
                        count += int(mask.sum().item())
        return count

    @property
    def ready_rows(self) -> int:
        """暂存在 _ready 中的行数。"""
        return len(self._ready)

    @property
    def total_pending_count(self) -> int:
        """等待结算的样本总数 (当前 + 下一)。"""
        return self.waiting_current_samples + self.waiting_next_samples

    @property
    def pending_count(self) -> int:
        """保留旧接口：返回等待当前周期结算的样本数。"""
        return self.waiting_current_samples

    def pending_cycle_counts(
        self,
        episode_id: int,
        worker_id: int = 0,
    ) -> dict[int, int]:
        """返回指定episode中尚未获得当前转站标签的周期样本数。"""
        episode = int(episode_id)
        worker = int(worker_id)
        return {
            cycle_id: len(samples)
            for (pending_worker, pending_episode, cycle_id), samples in self._pending.items()
            if pending_worker == worker and pending_episode == episode
        }

    def pending_next_cycle_counts(
        self,
        episode_id: int,
        worker_id: int = 0,
    ) -> dict[int, int]:
        """返回指定episode中已获得P_q但仍在等待下一周期事实的周期样本数 (U11)。"""
        episode = int(episode_id)
        worker = int(worker_id)
        counts: dict[int, int] = {}
        for (w, ep, c), items in self._pending_next_cycle.items():
            if w == worker and ep == episode:
                unsettled = sum(1 for item in items if not item.get("has_next_ready", False))
                if unsettled > 0:
                    counts[c] = unsettled
        return counts

    def stats(self, episode_id: int | None = None, worker_id: int | None = None) -> dict[str, int]:
        """返回详细的样本与目标分项统计报告 (U11)。"""
        return {
            "waiting_current_samples": self.waiting_current_samples,
            "waiting_next_samples": self.waiting_next_samples,
            "available_current_samples": self.available_current_samples,
            "available_next_samples": self.available_next_samples,
            "available_next_station_targets": self.available_next_station_targets,
            "ready_rows": self.ready_rows,
            "total_pending_count": self.total_pending_count,
            "abandoned_current_samples": self.abandoned_current_samples,
            "abandoned_next_samples": self.abandoned_next_samples,
        }

    def add(
        self,
        *,
        episode_id: int,
        cycle_id: int,
        decision_id: int,
        state_feat: torch.Tensor,
        graph_snapshot: Any,
        estimated_cmax: float,
        current_time: float,
        h0: float,
        predictor_version: int | None,
        worker_id: int = 0,
        time_urgency: torch.Tensor | None = None,
    ) -> bool:
        """保存周期内每个决策状态；编号在对应worker/episode内递增。"""
        worker = int(worker_id)
        episode = int(episode_id)
        key = (worker, episode, int(cycle_id))
        decision_scope = (worker, episode)
        unique_decision_id = int(decision_id)
        previous_id = self._last_decision_ids.get(decision_scope, -1)
        if unique_decision_id <= previous_id:
            raise ValueError(
                f"decision_id必须唯一递增：收到{unique_decision_id}，"
                f"上一编号为{previous_id}"
            )
        sample = PendingTimeLabel(
            worker_id=worker,
            episode_id=episode,
            cycle_id=key[2],
            decision_id=unique_decision_id,
            state_feat=state_feat.detach().cpu().clone(),
            graph_snapshot=_copy_to_cpu(graph_snapshot),
            estimated_cmax=float(estimated_cmax),
            current_time=float(current_time),
            h0=float(h0),
            predictor_version=(
                None if predictor_version is None else int(predictor_version)
            ),
            time_urgency=None if time_urgency is None else time_urgency.detach().cpu().clone(),
        )
        self._pending.setdefault(key, []).append(sample)
        self._last_decision_ids[decision_scope] = unique_decision_id
        return True

    def attach_transfer(
        self,
        *,
        episode_id: int,
        cycle_id: int,
        actual_transfer_time: float,
        worker_id: int = 0,
        station_cleared_times: dict[int, float] | None = None,
        next_cycle_station_mask: Sequence[bool] | None = None,
        cycle_record: CycleExecutionRecord | None = None,
    ) -> bool:
        """用真实转站时刻补齐一个周期的归一化残差，并结算上一周期的下一周期预测向量 (R07/U03-U06/R1)。"""
        worker = int(worker_id)
        episode = int(episode_id)
        cycle = int(cycle_id)
        key = (worker, episode, cycle)

        if not math.isfinite(float(actual_transfer_time)):
            raise ValueError(f"实际转站时刻非有限数值: {actual_transfer_time}")

        # 1. 周期记录与参数一致性校验 (R1/N01/N03)
        if cycle_record is not None:
            validate_cycle_execution_record(cycle_record)
            if cycle_record.cycle_id != cycle:
                raise ValueError(f"cycle_record周期号 {cycle_record.cycle_id} 与参数 {cycle} 不一致")
            if abs(cycle_record.transfer_time - float(actual_transfer_time)) > 1e-4:
                raise ValueError(
                    f"cycle_record转站时刻 {cycle_record.transfer_time} 与参数 {actual_transfer_time} 不一致"
                )
            if station_cleared_times is not None:
                # 必须完全一致，严禁冲突或覆盖
                if dict(cycle_record.station_first_cleared_times) != {int(k): float(v) for k, v in station_cleared_times.items()}:
                    raise ValueError("传入的 station_cleared_times 与 cycle_record 不一致")
            else:
                station_cleared_times = dict(cycle_record.station_first_cleared_times)

            if next_cycle_station_mask is not None:
                if tuple(bool(x) for x in next_cycle_station_mask) != tuple(cycle_record.occupied_stations):
                    raise ValueError("传入的 next_cycle_station_mask 与 cycle_record 不一致")
            else:
                next_cycle_station_mask = cycle_record.occupied_stations

            aircraft_at_stations = tuple(cycle_record.aircraft_at_stations)
            cycle_start_time = float(cycle_record.cycle_start_time)
        else:
            # 兼容模式：必须严格校验 mask 维度和数值有限性 (N03)
            if next_cycle_station_mask is not None:
                if len(next_cycle_station_mask) != 5:
                    raise ValueError(f"下一周期站位掩码长度必须为5，收到: {len(next_cycle_station_mask)}")
            if station_cleared_times is not None:
                for s, f_val in station_cleared_times.items():
                    if not (0 <= int(s) < 5):
                        raise ValueError(f"站位索引 {s} 超出五站范围 [0, 4]")
                    if not math.isfinite(float(f_val)):
                        raise ValueError(f"站位 {s} 清空时刻非有限数值: {f_val}")
            aircraft_at_stations = None
            cycle_start_time = None

        norm_mask = tuple(bool(x) for x in next_cycle_station_mask) if next_cycle_station_mask is not None else None
        norm_cleared = tuple(sorted((int(k), round(float(v), 6)) for k, v in (station_cleared_times or {}).items()))
        fact_fingerprint = (
            cycle,
            round(cycle_start_time, 6) if cycle_start_time is not None else None,
            round(float(actual_transfer_time), 6),
            norm_mask,
            aircraft_at_stations,
            norm_cleared,
        )

        # 2. 幂等性与冲突校验 (N02):
        if key in self._transfer_facts:
            existing_fact = self._transfer_facts[key]
            if existing_fact != fact_fingerprint:
                raise ValueError(
                    f"冲突的周期转站事实：周期 {cycle} 已记录事实 {existing_fact}，收到冲突事实 {fact_fingerprint}"
                )
            if key in self._pending_next_cycle and key not in self._pending:
                # 相同事实重交幂等，已消费不重复提交
                return True
        elif key in self._last_transfer_times:
            recorded_t = self._last_transfer_times[key]
            if abs(recorded_t - float(actual_transfer_time)) > 1e-4:
                raise ValueError(
                    f"冲突的周期转站事实：周期 {cycle} 已记录转站时刻 {recorded_t}，收到冲突时刻 {actual_transfer_time}"
                )
            if key in self._pending_next_cycle and key not in self._pending:
                return True

        # 3. 预先计算所有增量与变更，校验通过前不修改内部任何状态 (原子性保证 N01/N03)
        # 3.1 预计算上一周期 (cycle - 1) 的 next_cycle 结算
        prev_key = (worker, episode, cycle - 1)
        prev_samples = self._pending_next_cycle.get(prev_key)
        computed_prev_updates: list[tuple[dict[str, Any], torch.Tensor, torch.Tensor]] = []
        if prev_samples:
            for item in prev_samples:
                p_prev = item["actual_transfer_time"]
                h0 = item["h0"]
                target_next = torch.zeros(5, dtype=torch.float)
                mask_next = torch.zeros(5, dtype=torch.bool)
                for m in range(5):
                    has_ac = (
                        bool(next_cycle_station_mask[m])
                        if next_cycle_station_mask is not None
                        else False
                    )
                    if has_ac:
                        if station_cleared_times is not None and m in station_cleared_times:
                            f_m = float(station_cleared_times[m])
                            if not math.isfinite(f_m):
                                raise ValueError(f"站位 {m} 清空时刻非有限数值: {f_m}")
                            target_next[m] = (f_m - float(p_prev) - float(h0)) / float(h0)
                            mask_next[m] = True
                        else:
                            # 占用站位缺少 F 记录，严禁静默回退到 P；保持 mask 为 False！
                            mask_next[m] = False
                    else:
                        mask_next[m] = False
                computed_prev_updates.append((item, target_next, mask_next))

        # 3.2 预计算当前周期 samples
        samples = self._pending.get(key)
        computed_cycle_items: list[dict[str, Any]] = []
        if samples:
            for sample in samples:
                label_y = (float(actual_transfer_time) - sample.estimated_cmax) / sample.h0
                if not math.isfinite(label_y):
                    raise ValueError(f"计算的当前残差标签非有限数值: {label_y}")
                default_next = torch.zeros(5, dtype=torch.float)
                default_mask = torch.zeros(5, dtype=torch.bool)
                item = {
                    "worker_id": sample.worker_id,
                    "episode_id": sample.episode_id,
                    "cycle_id": sample.cycle_id,
                    "decision_id": sample.decision_id,
                    "state_feat": sample.state_feat,
                    "graph_snapshot": sample.graph_snapshot,
                    "estimated_cmax": sample.estimated_cmax,
                    "current_time": sample.current_time,
                    "h0": sample.h0,
                    "predictor_version": sample.predictor_version,
                    "time_urgency": sample.time_urgency,
                    "actual_transfer_time": float(actual_transfer_time),
                    "target_residual": float(label_y),
                    "target_next_residuals": default_next,
                    "next_cycle_masks": default_mask,
                    "has_current_ready": True,
                    "current_drained": False,
                    "has_next_ready": False,
                    "next_drained": False,
                }
                computed_cycle_items.append(item)

        # 4. 全部校验与计算均成功，原子提交写入！
        self._last_transfer_times[key] = float(actual_transfer_time)
        self._transfer_facts[key] = fact_fingerprint

        for item, target_next, mask_next in computed_prev_updates:
            item["target_next_residuals"] = target_next
            item["next_cycle_masks"] = mask_next
            item["has_next_ready"] = True

        if samples:
            self._pending.pop(key, None)
            self._pending_next_cycle[key] = computed_cycle_items
            return True
        else:
            return False

    def drain_ready(self) -> dict[str, Any] | None:
        """取出已补齐标签的批次；跨采样段保留 _pending_next_cycle，直到获得下一周期事实。"""
        ready_items: list[dict[str, Any]] = []

        if self._ready:
            ready_items.extend(self._ready)
            self._ready = []

        keys_to_remove: list[tuple[int, int, int]] = []
        for key, items in self._pending_next_cycle.items():
            all_drained = True
            for item in items:
                # 若当前残差未被取出：
                if item["has_current_ready"] and not item["current_drained"]:
                    # 若此时下一周期也已结算，合并为单条双目标样本输出
                    if item["has_next_ready"] and not item["next_drained"]:
                        out_item = copy.deepcopy(item)
                        out_item["current_target_mask"] = True
                        ready_items.append(out_item)
                        item["current_drained"] = True
                        item["next_drained"] = True
                    else:
                        # 仅输出当前残差，下一周期掩码置空
                        out_item = copy.deepcopy(item)
                        out_item["next_cycle_masks"] = torch.zeros(5, dtype=torch.bool)
                        out_item["current_target_mask"] = True
                        ready_items.append(out_item)
                        item["current_drained"] = True

                # 若当前残差在之前采样段已取出，现在下一周期目标刚结算：
                elif item["has_next_ready"] and not item["next_drained"]:
                    out_item = copy.deepcopy(item)
                    out_item["current_target_mask"] = False
                    ready_items.append(out_item)
                    item["next_drained"] = True

                if not (item["current_drained"] and item["next_drained"]):
                    all_drained = False

            if all_drained:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._pending_next_cycle[key]

        if not ready_items:
            return None

        result: dict[str, Any] = {
            "worker_ids": [item["worker_id"] for item in ready_items],
            "episode_ids": [item["episode_id"] for item in ready_items],
            "cycle_ids": [item["cycle_id"] for item in ready_items],
            "decision_ids": [item["decision_id"] for item in ready_items],
            "state_feats": torch.stack([item["state_feat"] for item in ready_items]),
            "graph_snapshots": [item["graph_snapshot"] for item in ready_items],
            "estimated_cmax": torch.tensor([item["estimated_cmax"] for item in ready_items]),
            "current_times": torch.tensor([item["current_time"] for item in ready_items]),
            "h0": torch.tensor([item["h0"] for item in ready_items]),
            "predictor_versions": [item["predictor_version"] for item in ready_items],
            "actual_transfer_times": torch.tensor([item["actual_transfer_time"] for item in ready_items]),
            "target_residuals": torch.tensor([item["target_residual"] for item in ready_items]),
            "current_target_masks": torch.tensor(
                [item.get("current_target_mask", True) for item in ready_items],
                dtype=torch.bool,
            ),
            "target_next_residuals": torch.stack([item["target_next_residuals"] for item in ready_items]),
            "next_cycle_masks": torch.stack([item["next_cycle_masks"] for item in ready_items]),
        }
        if all(item.get("time_urgency") is not None for item in ready_items):
            result["time_urgencies"] = torch.stack([item["time_urgency"] for item in ready_items])
        return result

    def discard_episode(
        self,
        episode_id: int,
        worker_id: int | None = None,
        reason: str = "episode_ended",
    ) -> dict[str, int]:
        """清空指定worker/episode的待定样本与结果索引，审计丢弃数量并释放内存 (U12-U13)。"""
        episode = int(episode_id)
        w_filter = int(worker_id) if worker_id is not None else None

        discarded_current = 0
        discarded_next = 0

        # 1. 丢弃尚未收到本周期转站时刻 P_q 的样本
        retained_pending: dict[tuple[int, int, int], list[PendingTimeLabel]] = {}
        for key, samples in self._pending.items():
            if key[1] == episode and (w_filter is None or key[0] == w_filter):
                discarded_current += len(samples)
                discarded_next += len(samples)
            else:
                retained_pending[key] = samples
        self._pending = retained_pending

        # 2. 检查已结算当前残差但仍在等待下一周期的样本，转入 _ready
        retained_next = {}
        for key, items in self._pending_next_cycle.items():
            if key[1] == episode and (w_filter is None or key[0] == w_filter):
                for item in items:
                    if not item.get("has_next_ready", False):
                        discarded_next += 1
                    if not item["current_drained"] or not item["next_drained"]:
                        out_item = copy.deepcopy(item)
                        if not item.get("has_next_ready", False):
                            # 真实终止且未来未揭示，未来不再有下一个周期，掩码全设为 False
                            out_item["next_cycle_masks"] = torch.zeros(5, dtype=torch.bool)
                        out_item["current_target_mask"] = not item["current_drained"]
                        self._ready.append(out_item)
            else:
                retained_next[key] = items
        self._pending_next_cycle = retained_next

        # 3. 清理结果索引与缓存，避免跨 episode 内存泄漏 (U13)
        self._last_decision_ids = {
            key: decision_id
            for key, decision_id in self._last_decision_ids.items()
            if not (key[1] == episode and (w_filter is None or key[0] == w_filter))
        }
        self._last_transfer_times = {
            key: t_val
            for key, t_val in self._last_transfer_times.items()
            if not (key[1] == episode and (w_filter is None or key[0] == w_filter))
        }
        self._transfer_facts = {
            key: fact
            for key, fact in self._transfer_facts.items()
            if not (key[1] == episode and (w_filter is None or key[0] == w_filter))
        }

        self.abandoned_current_samples += discarded_current
        self.abandoned_next_samples += discarded_next

        return {
            "discarded_current": discarded_current,
            "discarded_next": discarded_next,
        }


def _copy_to_cpu(value: Any) -> Any:
    """递归复制回放状态；PyG图副本及其中张量都留在CPU。"""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_to_cpu(item) for item in value)
    if value.__class__.__module__.startswith("torch_geometric."):
        copied = value.clone()
        copied.to("cpu")
        return copied
    return copy.deepcopy(value)


class RolloutBufferWork3:
    """工作三 PPO 经验回放缓存。"""

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
    ) -> None:
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.normalize_advantages = normalize_advantages

        self.transitions: list[PPOTransition] = []
        self.advantages: torch.Tensor = torch.empty(0, dtype=torch.float)
        self.target_values: torch.Tensor = torch.empty(0, dtype=torch.float)
        self.is_finalized: bool = False

    def add(self, transition: PPOTransition) -> None:
        """以独立CPU副本追加决策样本，避免后续状态变更污染回放。"""
        self.transitions.append(
            PPOTransition(
                state_feat=_copy_to_cpu(transition.state_feat),
                time_urgency=_copy_to_cpu(transition.time_urgency),
                sample_record=_copy_to_cpu(transition.sample_record),
                reward=float(transition.reward),
                raw_reward=float(transition.raw_reward),
                value=float(transition.value),
                log_prob=float(transition.log_prob),
                worker_id=int(transition.worker_id),
                episode_id=int(transition.episode_id),
                segment_id=int(transition.segment_id),
                done=bool(transition.done),
                action_dict=_copy_to_cpu(transition.action_dict),
                terminated=transition.terminated,
                truncated=bool(transition.truncated),
                predictor_version=transition.predictor_version,
                predictor_source=str(transition.predictor_source),
                time_input_source=str(transition.time_input_source),
                phi_source=str(transition.phi_source),
            )
        )
        self.is_finalized = False

    def finish_trajectory(self, last_value: float = 0.0) -> None:
        """计算全轨迹的 GAE 优势值与目标价值 V_target。

        Args:
            last_value: 轨迹末尾状态的估计价值 V(s_T)。真实终止时应传 0.0。
        """
        keys = {
            (transition.worker_id, transition.episode_id, transition.segment_id)
            for transition in self.transitions
        }
        if len(keys) > 1:
            raise ValueError(
                "混合worker/episode/segment样本必须调用finish_trajectories"
            )
        last_values = {next(iter(keys)): float(last_value)} if keys else {}
        self.finish_trajectories(last_values_by_segment=last_values)

    def finish_trajectories(
        self,
        *,
        last_values_by_segment: Mapping[tuple[int, int, int], float],
    ) -> None:
        """按复合轨迹键计算GAE；段尾bootstrap值必须与对应worker/episode一致。"""
        n = len(self.transitions)
        if n == 0:
            return

        advantages = torch.zeros(n, dtype=torch.float)
        target_values = torch.zeros(n, dtype=torch.float)

        indices_by_segment: dict[tuple[int, int, int], list[int]] = {}
        for index, transition in enumerate(self.transitions):
            key = (transition.worker_id, transition.episode_id, transition.segment_id)
            indices_by_segment.setdefault(key, []).append(index)

        for key, indices in indices_by_segment.items():
            last_transition = self.transitions[indices[-1]]
            is_terminal = last_transition.is_terminated
            if not is_terminal and key not in last_values_by_segment:
                raise ValueError(f"非终止轨迹段缺少bootstrap value：{key}")
            next_value = 0.0 if is_terminal else float(last_values_by_segment[key])
            gae = 0.0

            # 每个复合键独立逆序；worker交错插入不改变其时间顺序。
            for index in reversed(indices):
                transition = self.transitions[index]
                is_step_terminal = transition.is_terminated
                non_terminal = 1.0 - float(is_step_terminal)
                delta = (
                    transition.reward
                    + self.gamma * next_value * non_terminal
                    - transition.value
                )
                gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
                advantages[index] = gae
                target_values[index] = gae + transition.value
                next_value = transition.value

        # 优势值标准化
        if self.normalize_advantages and n > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        self.advantages = advantages
        self.target_values = target_values
        self.is_finalized = True

    def get_batches(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """生成 PPO 训练所需的 mini-batch 迭代器。

        Yields:
            字典包含:
              - "state_feats": torch.Tensor (B, 32)
              - "time_urgencies": torch.Tensor (B, 2)
              - "old_log_probs": torch.Tensor (B,)
              - "advantages": torch.Tensor (B,)
              - "target_values": torch.Tensor (B,)
              - "sample_records": list[dict] of length B
        """
        if not self.is_finalized:
            raise RuntimeError("请先调用 finish_trajectory 计算 GAE 优势值后再生成 mini-batch！")

        n = len(self.transitions)
        if n == 0:
            return

        state_feats = torch.stack([t.state_feat for t in self.transitions])
        time_urgencies = torch.stack([t.time_urgency for t in self.transitions])
        old_log_probs = torch.tensor([t.log_prob for t in self.transitions], dtype=torch.float)

        if shuffle:
            indices = torch.randperm(n)
        else:
            indices = torch.arange(n)

        for start_idx in range(0, n, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            batch_records = [self.transitions[idx].sample_record for idx in batch_idx]

            yield {
                "state_feats": state_feats[batch_idx],
                "time_urgencies": time_urgencies[batch_idx],
                "old_log_probs": old_log_probs[batch_idx],
                "advantages": self.advantages[batch_idx],
                "target_values": self.target_values[batch_idx],
                "sample_records": batch_records,
            }

    def clear(self) -> None:
        """清空缓冲区。"""
        self.transitions.clear()
        self.advantages = torch.empty(0, dtype=torch.float)
        self.target_values = torch.empty(0, dtype=torch.float)
        self.is_finalized = False

    def __len__(self) -> int:
        return len(self.transitions)
