"""工作三策略决策所需的CPU快照与纯工人补全规则。"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import HeteroData


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    """一个站内工人的稳定身份、技能、效率和已知日历区间。"""

    worker_id: int
    skills: tuple[int, ...]
    efficiency: float | None
    calendar_intervals: tuple[tuple[float, float, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "skills", tuple(sorted({int(skill) for skill in self.skills})))
        copied_intervals = tuple(
            (float(start), float(end), str(task_key))
            for start, end, task_key in self.calendar_intervals
        )
        if any(
            not math.isfinite(start) or not math.isfinite(end) or end < start
            for start, end, _ in copied_intervals
        ):
            raise ValueError("工人快照包含非法日历区间")
        object.__setattr__(self, "calendar_intervals", copied_intervals)
        if self.efficiency is not None and (
            not math.isfinite(float(self.efficiency)) or float(self.efficiency) <= 0.0
        ):
            raise ValueError(f"工人 {self.worker_id} 的效率必须是正有限数值")
        if self.efficiency is not None:
            object.__setattr__(self, "efficiency", float(self.efficiency))


@dataclass(frozen=True, slots=True)
class TeamCompletionContext:
    """判断任务在一个固定站位能否逐人补全合法团队的纯数据。"""

    task_key: str
    station_id: int
    required_skill: int
    demand: int
    workers: tuple[WorkerSnapshot, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "workers", tuple(self.workers))
        worker_ids = tuple(worker.worker_id for worker in self.workers)
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError(f"任务 {self.task_key} 的团队上下文含重复工人")
        if self.demand < 1:
            raise ValueError(f"任务 {self.task_key} 的需求人数必须为正数")


@dataclass(frozen=True, slots=True)
class DecisionSnapshot:
    """策略与PPO重放使用的不可变CPU决策状态副本。"""

    worker_id: int
    episode_id: int
    episode_index: int
    state_features: Tensor
    time_features: Tensor
    graph_snapshot: HeteroData
    candidate_task_keys: tuple[str, ...]
    candidate_task_features: Tensor
    branch_masks: tuple[tuple[bool, bool], ...]
    team_contexts: tuple[TeamCompletionContext, ...]
    candidate_task_node_indices: tuple[int, ...]
    worker_node_indices: tuple[tuple[int, ...], ...]
    reserved_flags: tuple[bool, ...]
    advance_available: bool
    current_time: float
    cycle_id: int
    estimated_cmax: float | None
    h0: float
    last_transfer_time: float
    graph_version: str

    def __post_init__(self) -> None:
        # 输入形状：[32]、[2]、[N, 8]；快照拥有独立CPU副本。
        state_features = self._cpu_tensor_copy(self.state_features)
        time_features = self._cpu_tensor_copy(self.time_features)
        task_features = self._cpu_tensor_copy(self.candidate_task_features)
        assert state_features.ndim == 1 and state_features.shape == (32,)
        assert time_features.ndim == 1 and time_features.shape == (2,)
        assert task_features.ndim == 2 and task_features.shape[1] == 8
        object.__setattr__(self, "state_features", state_features)
        object.__setattr__(self, "time_features", time_features)
        object.__setattr__(self, "candidate_task_features", task_features)

        graph_copy = copy.deepcopy(self.graph_snapshot)
        for store in graph_copy.stores:
            for key, value in tuple(store.items()):
                if isinstance(value, Tensor):
                    store[key] = value.detach().to(device="cpu").clone()
        object.__setattr__(self, "graph_snapshot", graph_copy)

        keys = tuple(str(key) for key in self.candidate_task_keys)
        masks = tuple(tuple(bool(value) for value in mask) for mask in self.branch_masks)
        contexts = tuple(self.team_contexts)
        task_indices = tuple(int(index) for index in self.candidate_task_node_indices)
        worker_indices = tuple(
            tuple(int(index) for index in indices)
            for indices in self.worker_node_indices
        )
        reserved = tuple(bool(value) for value in self.reserved_flags)
        candidate_count = len(keys)
        assert len(set(keys)) == candidate_count
        assert task_features.shape[0] == candidate_count
        assert len(masks) == candidate_count
        assert len(contexts) == candidate_count
        assert len(task_indices) == candidate_count
        assert len(worker_indices) == candidate_count
        assert len(reserved) == candidate_count
        if any(len(mask) != 2 or not any(mask) for mask in masks):
            raise ValueError("每个快照候选任务必须至少有一个合法分支")
        for key, context, indices in zip(keys, contexts, worker_indices, strict=True):
            if context.task_key != key or len(indices) != len(context.workers):
                raise ValueError(f"任务 {key} 的团队上下文或工人图索引不匹配")
        object.__setattr__(self, "candidate_task_keys", keys)
        object.__setattr__(self, "branch_masks", masks)
        object.__setattr__(self, "team_contexts", contexts)
        object.__setattr__(self, "candidate_task_node_indices", task_indices)
        object.__setattr__(self, "worker_node_indices", worker_indices)
        object.__setattr__(self, "reserved_flags", reserved)

        scalar_values = (
            self.current_time,
            self.h0,
            self.last_transfer_time,
        )
        if not all(math.isfinite(float(value)) for value in scalar_values):
            raise ValueError("决策快照的时间字段必须是有限数值")
        if self.estimated_cmax is not None and not math.isfinite(float(self.estimated_cmax)):
            raise ValueError("决策快照的启发式完工预测必须是有限数值或缺失")
        if not isinstance(self.graph_version, str) or not self.graph_version:
            raise ValueError("决策快照必须记录图特征版本")

    @staticmethod
    def _cpu_tensor_copy(value: Tensor) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError("决策快照张量字段必须是torch.Tensor")
        return value.detach().to(device="cpu").clone()


def team_completion_worker_ids(
    context: TeamCompletionContext,
    selected_worker_ids: tuple[int, ...],
) -> tuple[int, ...]:
    """按环境团队资格规则返回当前前缀后仍可选的工人ID。"""
    selected = tuple(int(worker_id) for worker_id in selected_worker_ids)
    if len(selected) > context.demand or len(selected) != len(set(selected)):
        return ()

    workers_by_id = {worker.worker_id: worker for worker in context.workers}
    for worker_id in selected:
        worker = workers_by_id.get(worker_id)
        if worker is None or worker.efficiency is None:
            return ()
        if context.required_skill >= 0 and context.required_skill not in worker.skills:
            return ()

    candidates = tuple(
        worker.worker_id
        for worker in context.workers
        if worker.worker_id not in selected
        and worker.efficiency is not None
        and (
            context.required_skill < 0
            or context.required_skill in worker.skills
        )
    )
    remaining_demand = context.demand - len(selected)
    return candidates if len(candidates) >= remaining_demand else ()


def worker_completion_mask(
    context: TeamCompletionContext,
    selected_worker_ids: tuple[int, ...],
    max_station_workers: int,
) -> tuple[bool, ...]:
    """按快照候选顺序生成定长逐人掩码，末尾用False补齐。"""
    if max_station_workers < len(context.workers):
        raise ValueError(
            f"站内工人数量{len(context.workers)}超过Actor掩码上限{max_station_workers}"
        )
    valid_ids = set(team_completion_worker_ids(context, selected_worker_ids))
    values = [worker.worker_id in valid_ids for worker in context.workers]
    values.extend([False] * (max_station_workers - len(values)))
    assert len(values) == max_station_workers
    return tuple(values)
