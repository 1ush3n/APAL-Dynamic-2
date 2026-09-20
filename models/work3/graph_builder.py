"""工作三 多架次异构图构造器 (Task 5.1)。

核心技术规范：
1. 节点体系：
   - task: 2,830 道工序 (10 架飞机 x 283 道物理工序)；
   - worker: 80 名工人 (来源于初始调度内生绑定并固定于 5 个站位)；
   - station: 5 个物理脉动装配工位 (0 ~ 4)；
   - skill: 5 个工种技能中转节点 (0 ~ 4，经典 Skill Hub 模式)。
2. 特征排布：
   - 与已有代码 (environment.py / utils.resource_graph) 100% 对齐；
   - task_x: [工时, 状态独热(1:5), 5技能独热(5:10), 基准站内偏移, 架次归一化, 物理相对站位偏移(s_k - m_i^0), 改派次数, 物理工序, 周期索引, 需求人数, 到料等待时间]，共 18 维；
   - worker_x: [效率, 5技能资质(1:6), 等待时间, 空闲标记, 站位锁定(8:13), 疲劳度等]，共 17 维；
   - station_x: [剩余负荷, 在场飞机, 可用槽位, 槽位等待时间, 宏观特征]，共 15 维；
   - skill_x: 5 个技能节点的动态资源统计特征，共 11 维。
3. 边拓扑体系：
   - DAG 偏序边: ("task", "precedes", "task")；
   - 站位归属边: ("task", "assigned_to", "station"), ("station", "has_task", "task")；
   - 工人执行边: ("task", "done_by", "worker")；
   - 双向 Skill Hub: worker <-> skill <-> task。
4. 性能指标：预构建静态图骨架，单步动态更新耗时 < 2ms。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

from envs.work3.core_types import MultiAircraftState, TaskRuntimeState, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from utils.resource_graph import (
    SkillHubTopology,
    apply_resource_graph,
    build_skill_features,
    build_skill_hub_topology,
)
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


@dataclass
class Work3ResourceConfig:
    """符合 ResourceGraphConfig 协议的工作三轻量级配置。"""

    use_skill_hub: bool = True
    skill_hub_bidirectional: bool = True
    num_skill_types: int = 5
    worker_skill_feature_slots: int = 5
    worker_feat_dim: int = 17
    skill_feat_dim: int = 11


class MultiAircraftGraphBuilder:
    """工作三多架次装配线异构图构造器。"""

    def __init__(
        self,
        baseline: MultiAircraftBaseline,
        config: Work3ResourceConfig | None = None,
    ) -> None:
        self.baseline = baseline
        self.config = config if config is not None else Work3ResourceConfig()
        self.h0 = float(baseline.h0)

        # 建立有序映射
        # task_keys: 按照 (aircraft_id, task_id) 严格排序
        self.task_keys: list[str] = sorted(
            baseline.tasks.keys(),
            key=lambda k: (baseline.tasks[k].aircraft_id, baseline.tasks[k].task_id),
        )
        self.task_key_to_idx: dict[str, int] = {k: i for i, k in enumerate(self.task_keys)}
        self.num_tasks = len(self.task_keys)

        # 收集全线所有工人（来源于初始调度内生绑定）
        all_workers_set: set[int] = set()
        for workers in baseline.station_workers.values():
            all_workers_set.update(workers)
        self.sorted_workers: list[int] = sorted(all_workers_set)
        self.worker_id_to_idx: dict[int, int] = {w: i for i, w in enumerate(self.sorted_workers)}
        self.num_workers = len(self.sorted_workers)
        self.num_stations = 5

        # ------------------
        # 1. 预构建静态 Task 特征底座 (Shape: [num_tasks, 18])
        # ------------------
        self.base_task_x = torch.zeros((self.num_tasks, 18), dtype=torch.float)
        for i, key in enumerate(self.task_keys):
            t = baseline.tasks[key]
            # [0] 持续工时归一化
            self.base_task_x[i, 0] = float(t.duration) / self.h0
            # [5:10] 5 种技能独热编码 (0 ~ 4)
            if 0 <= t.skill < self.config.num_skill_types:
                self.base_task_x[i, 5 + t.skill] = 1.0
            # [10] 基准站内偏移
            self.base_task_x[i, 10] = float(t.in_station_offset) / self.h0
            # [11] 飞机架次归一化 (k / 10)
            self.base_task_x[i, 11] = float(t.aircraft_id) / 10.0
            # [14] 是否物理工序
            self.base_task_x[i, 14] = 1.0 if t.duration > 1e-5 else 0.0
            # [15] 归属基准周期编号归一化
            self.base_task_x[i, 15] = float(t.cycle_idx) / 14.0
            # [16] 需求人数
            self.base_task_x[i, 16] = float(t.demand)

        # ------------------
        # 2. 预构建静态 Worker 特征底座 (Shape: [num_workers, 17])
        # ------------------
        self.base_worker_x = torch.zeros((self.num_workers, 17), dtype=torch.float)
        # 统计各工人的技能能力（从基准任务中聚合该工人的技能）
        worker_skills: dict[int, set[int]] = {w: set() for w in self.sorted_workers}
        worker_station: dict[int, int] = {}
        for s_id, w_list in baseline.station_workers.items():
            for w in w_list:
                worker_station[w] = s_id - 1  # 0-based station
        for t in baseline.tasks.values():
            if 0 <= t.skill < self.config.num_skill_types:
                for w in t.team:
                    if w in worker_skills:
                        worker_skills[w].add(t.skill)

        for w_idx, w_id in enumerate(self.sorted_workers):
            # [0] 默认基准效率 1.0
            self.base_worker_x[w_idx, 0] = 1.0
            # [1:6] 5 种工种资质
            for sk in worker_skills.get(w_id, set()):
                if 0 <= sk < self.config.num_skill_types:
                    self.base_worker_x[w_idx, 1 + sk] = 1.0
            # [8:13] 站位锁定/归属 one-hot
            st = worker_station.get(w_id, 0)
            if 0 <= st < 5:
                self.base_worker_x[w_idx, 8 + st] = 1.0

        # ------------------
        # 3. 预构建静态边：同机工序工艺 DAG 偏序边 ("task", "precedes", "task")
        # ------------------
        precedence_src: list[int] = []
        precedence_dst: list[int] = []
        for dst_idx, key in enumerate(self.task_keys):
            t = baseline.tasks[key]
            ac_id = t.aircraft_id
            for p_id in t.predecessors:
                p_key = f"{ac_id}_{p_id}"
                if p_key in self.task_key_to_idx:
                    src_idx = self.task_key_to_idx[p_key]
                    precedence_src.append(src_idx)
                    precedence_dst.append(dst_idx)

        if precedence_src:
            self.precedence_edge_index = torch.tensor(
                [precedence_src, precedence_dst], dtype=torch.long
            )
        else:
            self.precedence_edge_index = torch.empty((2, 0), dtype=torch.long)

        # ------------------
        # 4. 预构建 Skill Hub 静态拓扑 与 NumPy 底座
        # ------------------
        self.skill_hub_topology: SkillHubTopology = build_skill_hub_topology(
            self.base_task_x,
            self.base_worker_x,
            self.config.num_skill_types,
        )
        self.base_task_x_np = self.base_task_x.numpy()
        self.base_worker_x_np = self.base_worker_x.numpy()

    def build_graph(self, env: AirLineEnvWork3) -> HeteroData:
        """从当前仿真环境状态构建完整的 PyG HeteroData 异构图。"""
        state: MultiAircraftState = env.state
        current_time = float(state.current_time)
        data = HeteroData()

        # 1. 刷新 Task 特征 (使用 NumPy 极速批量更新)
        task_x_np = self.base_task_x_np.copy()
        assigned_ts_src: list[int] = []
        assigned_ts_dst: list[int] = []
        done_by_src: list[int] = []
        done_by_dst: list[int] = []
        station_remain_workload = [0.0] * 5
        station_running_count = [0] * 5

        h0 = self.h0
        for idx, key in enumerate(self.task_keys):
            t_rt: TaskRuntimeState | None = state.tasks.get(key)
            if t_rt is None:
                continue

            # [1:5] 状态独热设置 (UNREADY=0, READY=1, RESERVED=2, RUNNING=3, POSTPONED=4, COMPLETED=5)
            st_val = int(t_rt.status)
            if 0 <= st_val <= 4:
                task_x_np[idx, 1:5] = 0.0
                if st_val > 0:
                    task_x_np[idx, st_val] = 1.0

            # [12] 物理相对站位偏移：(飞机当前所在站位 s_k - 工序基准基础站位 m_i^0)
            ac_state = state.aircraft.get(t_rt.aircraft_id)
            if ac_state is not None:
                curr_station = ac_state.current_station
                base_station = t_rt.base_station  # 0-based
                task_x_np[idx, 12] = float(curr_station - base_station) if curr_station >= 0 else 0.0

            # [13] 累计改派后移次数 n_{ki}
            task_x_np[idx, 13] = float(t_rt.postpone_count)

            # [17] 到料等待紧迫度 log1p(max(0, R - t) / H_0)
            wait_time = max(0.0, float(t_rt.material_ready_time) - current_time)
            task_x_np[idx, 17] = math.log1p(wait_time / h0)

            # 动态边收集与站位负荷累积
            st = t_rt.current_station
            if 0 <= st < 5:
                assigned_ts_src.append(idx)
                assigned_ts_dst.append(st)
                if t_rt.status != TaskStatus.COMPLETED:
                    station_remain_workload[st] += t_rt.duration
                    if t_rt.status == TaskStatus.RUNNING:
                        station_running_count[st] += 1

            if t_rt.status in (TaskStatus.RUNNING, TaskStatus.RESERVED):
                for w_id in t_rt.assigned_team:
                    if w_id in self.worker_id_to_idx:
                        w_idx = self.worker_id_to_idx[w_id]
                        done_by_src.append(idx)
                        done_by_dst.append(w_idx)

        task_x = torch.from_numpy(task_x_np)
        data["task"].x = task_x

        # 2. 刷新 Worker 特征 (NumPy 极速批量更新)
        worker_x_np = self.base_worker_x_np.copy()
        for w_idx, w_id in enumerate(self.sorted_workers):
            cal = state.workers.get(w_id)
            if cal is not None:
                is_free = True
                wait_w = 0.0
                for iv in cal.intervals:
                    if iv.start <= current_time < iv.end - 1e-5:
                        is_free = False
                        wait_w = max(wait_w, iv.end - current_time)
                # [6] 等待空闲时间 log1p(wait_w / H_0)
                worker_x_np[w_idx, 6] = math.log1p(wait_w / h0)
                # [7] 当前是否空闲
                worker_x_np[w_idx, 7] = 1.0 if is_free else 0.0

        worker_x = torch.from_numpy(worker_x_np)
        data["worker"].x = worker_x

        # 3. 刷新 Station 特征 (Shape: [5, 15])
        station_x_np = np.zeros((self.num_stations, 15), dtype=np.float32)
        cycle_elapsed = max(0.0, current_time - state.last_transfer_time)
        for s in range(self.num_stations):
            ac_id = state.get_aircraft_at_station(s)
            station_x_np[s, 0] = float(station_remain_workload[s]) / h0
            station_x_np[s, 1] = float(ac_id) / 10.0 if ac_id is not None else -1.0
            station_x_np[s, 2] = float(max(0, 3 - station_running_count[s])) / 3.0
            station_x_np[s, 3] = cycle_elapsed / h0

        data["station"].x = torch.from_numpy(station_x_np)

        # 4. 组装边拓扑
        data["task", "precedes", "task"].edge_index = self.precedence_edge_index

        if assigned_ts_src:
            t_s = torch.tensor([assigned_ts_src, assigned_ts_dst], dtype=torch.long)
            s_t = torch.tensor([assigned_ts_dst, assigned_ts_src], dtype=torch.long)
        else:
            t_s = torch.empty((2, 0), dtype=torch.long)
            s_t = torch.empty((2, 0), dtype=torch.long)
        data["task", "assigned_to", "station"].edge_index = t_s
        data["station", "has_task", "task"].edge_index = s_t

        if done_by_src:
            t_w = torch.tensor([done_by_src, done_by_dst], dtype=torch.long)
        else:
            t_w = torch.empty((2, 0), dtype=torch.long)
        data["task", "done_by", "worker"].edge_index = t_w

        # 5. 挂载 Skill Hub 特征与双向资源边
        apply_resource_graph(
            data,
            task_x,
            worker_x,
            self.config,
            skill_hub_topology=self.skill_hub_topology,
        )

        return data

    def get_ready_task_indices(self, env: AirLineEnvWork3) -> list[int]:
        """获取当前状态下处于 READY 状态的所有工序在图节点中的全局索引。"""
        ready_tasks = env.get_ready_tasks()
        return [self.task_key_to_idx[t.task_key] for t in ready_tasks if t.task_key in self.task_key_to_idx]

    def get_ready_task_mask(self, env: AirLineEnvWork3) -> torch.Tensor:
        """获取全图工序就绪布尔掩码 (Shape: [num_tasks], True 为就绪)。"""
        mask = torch.zeros(self.num_tasks, dtype=torch.bool)
        indices = self.get_ready_task_indices(env)
        if indices:
            mask[torch.tensor(indices, dtype=torch.long)] = True
        return mask

    def get_action_branch_mask(self, env: AirLineEnvWork3, task_idx: int) -> dict[str, bool]:
        """获取指定工序在当前环境下的动作分支合法性掩码。

        Returns:
            {"can_stay": bool, "can_postpone": bool}
        """
        if task_idx < 0 or task_idx >= self.num_tasks:
            return {"can_stay": False, "can_postpone": False}
        key = self.task_keys[task_idx]
        task = env.state.tasks.get(key)
        if task is None or task.status != TaskStatus.READY:
            return {"can_stay": False, "can_postpone": False}

        # 分支 A (留在当前站)：只要处于 READY 即合法
        can_stay = True

        # 分支 B (后移至下一站)：校验后移放行条件（非末站即可合法后移）
        can_postpone = bool(task.current_station < env.state.num_stations - 1)

        return {"can_stay": can_stay, "can_postpone": can_postpone}
