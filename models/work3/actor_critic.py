"""工作三 条件分支自回归 Actor-Critic 决策网络 (Task 7.1)。

核心数学模型与技术规范：
依据《方法设计确认稿》第 4.2 节（自回归动作）与第 7.3 节（动作上下文显式时间融合）：
1. 共享状态编码器与 Critic：
   - 编码器: z = Encoder(s_feat) ∈ R^d (默认 d=64)
   - Critic: V(s) = Critic(z) ∈ R (评估综合回报价值)
   - 时间融合: e_fused = TimeContextFusion(z, u(s)) (Task 6.3 成果)
2. 四头条件自回归解码架构：
   - Head 1 (工序选择): 在就绪工序候选池中输出 Categorical 分布，采样工序 task；
   - Head 2 (站位分支): 输出 [STAY, POSTPONE] 分布，采样分支 branch；
     - 物理硬掩码: 末站 (站位 4) 或达到后移上限时，强制禁止 POSTPONE！
   - 条件分支动作截断 (核心定理)：
     - 若 branch == POSTPONE:
       * 动作在此截断！不选工人，不选对齐；
       * log π(a|s) = log π_task + log π_branch。
     - 若 branch == STAY:
       * Head 3 (工人指针序列): 自回归挑选 m_i 名站内工人，避免重复选人；
       * Head 4 (二元对齐): 决策 α ∈ {0, 1}；
       * log π(a|s) = log π_task + log π_branch + Σ log π_worker + log π_align。
3. PPO 重放契约 (evaluate_actions)：
   - 严格根据实际采样的动作分支计算对数概率与策略熵；
   - 保证在采样时刻比率 r_0(θ) = π_θ(a) / π_old(a) 严格恒等于 1.0。
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from envs.work3.core_types import ActionBranch, MultiAircraftState, TaskRuntimeState, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from models.work3.action_fusion import TimeContextFusion, compute_time_urgency_vector
from models.hb_gat_pn import FeatureEmbedder, HeteroGATEncoder
from models.work3.graph_builder import MultiAircraftGraphBuilder, Work3ResourceConfig
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


class Work3GraphEncoder(nn.Module):
    """工作三轻量异构图编码器，复用仓库已有 HB-GAT 输入与消息传递。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        config = SimpleNamespace(
            hidden_dim=int(hidden_dim),
            num_gat_layers=1,
            num_heads=2,
            task_feat_dim=18,
            worker_feat_dim=17,
            station_feat_dim=15,
            skill_feat_dim=11,
            use_skill_hub=True,
            skill_hub_bidirectional=True,
            num_skill_types=5,
            worker_skill_feature_slots=5,
            graph_encoder_mode="hetero_gat",
            task_feature_scope="full",
            station_feature_scope="full",
            homogeneous_shared_input_projection=False,
        )
        self.embedder = FeatureEmbedder(config)
        self.message_passing = HeteroGATEncoder(config)
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(
        self,
        graph: HeteroData,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回全局上下文、任务节点嵌入和工人节点嵌入。"""
        device = next(self.parameters()).device
        graph = graph.to(device)
        raw_x = graph.x_dict
        assert set(("task", "worker", "station", "skill")) <= set(raw_x)
        x_dict = self.embedder(raw_x)
        edge_index_dict = graph.edge_index_dict
        encoded = self.message_passing(x_dict, edge_index_dict)

        def mean_pool(node_type: str) -> torch.Tensor:
            value = encoded.get(node_type)
            if value is None or value.numel() == 0:
                return torch.zeros(
                    (1, next(self.parameters()).size(0)),
                    dtype=next(self.parameters()).dtype,
                    device=device,
                )
            return value.mean(dim=0, keepdim=True)

        pooled = torch.cat(
            [mean_pool("task"), mean_pool("worker"), mean_pool("station"), mean_pool("skill")],
            dim=-1,
        )
        context = self.context_projection(pooled).squeeze(0)
        return context, encoded["task"], encoded["worker"]


def extract_candidate_task_features(
    state: MultiAircraftState,
    candidate_tasks: list[TaskRuntimeState],
) -> torch.Tensor:
    """提取就绪候选工序的 8 维标准化物理特征张量。

    维度定义:
      0: 标准工时比例 (duration / H_0)
      1: 周期内基准开工偏移 (in_station_offset / H_0)
      2: 人数需求比例 (demand / 5.0)
      3: 当前排定执行站位 (current_station / 4.0)
      4: 累计后移次数 (postpone_count / 5.0)
      5: 是否缺料等待 (material_ready_time > current_time)
      6: 物料缺料延误紧迫度 (max(0, R - t) / H_0)
      7: 物理相对站位偏移 ((s_k - m_i^0) / 4.0)
    """
    if not candidate_tasks:
        return torch.empty((0, 8), dtype=torch.float)

    h0 = float(state.h0)
    current_time = float(state.current_time)
    num_cand = len(candidate_tasks)
    feats = torch.zeros((num_cand, 8), dtype=torch.float)

    for idx, task in enumerate(candidate_tasks):
        ac = state.aircraft.get(task.aircraft_id)
        ac_station = ac.current_station if ac is not None else task.current_station
        rel_station_offset = float(ac_station - task.base_station)

        feats[idx, 0] = float(task.duration) / h0
        feats[idx, 1] = float(task.in_station_offset) / h0
        feats[idx, 2] = float(task.demand) / 5.0
        feats[idx, 3] = float(task.current_station) / 4.0
        feats[idx, 4] = float(task.postpone_count) / 5.0

        is_delayed = 1.0 if task.material_ready_time > current_time else 0.0
        feats[idx, 5] = is_delayed
        feats[idx, 6] = max(0.0, float(task.material_ready_time - current_time)) / h0
        feats[idx, 7] = rel_station_offset / 4.0

    return feats


def extract_compact_state_features(state: MultiAircraftState, estimated_cmax: float) -> torch.Tensor:
    """提取当前仿真状态的 32 维精炼物理特征向量。

    维度组成:
      [0]: 当前周期已运行时间比例 (t - P_{q-1}) / H_0
      [1:6]: 5 站剩余未完工工序标准工时比例
      [6:11]: 5 站当前正在执行任务数比例 (<= 3)
      [11:16]: 5 站延误到料任务数比例
      [16:21]: 5 站最大物料延误紧迫度 (log1p)
      [21:26]: 5 站在场飞机编号归一化
      [26]: 启发式估计剩余时间比例 (P_q^h - t) / H_0
      [27]: 名义剩余时间比例 (P_{q-1} + H_0 - t) / H_0
      [28]: 产线当前脉动周期比例
      [29]: 全线累计完工工序比例
      [30]: 全线累计后移工序数比例
      [31]: 归一化总生产时长进度
    """
    feat = torch.zeros(32, dtype=torch.float)
    h0 = float(state.h0)
    current_time = float(state.current_time)
    p_last = float(state.last_transfer_time)

    total_cycles = float(state.num_aircraft + state.num_stations - 1)
    total_tasks = float(len(state.tasks)) if len(state.tasks) > 0 else 2830.0

    feat[0] = max(0.0, current_time - p_last) / h0

    for s in range(state.num_stations):
        ac_id = state.get_aircraft_at_station(s)
        st_tasks = [
            t for t in state.tasks.values()
            if t.current_station == s and t.status != TaskStatus.COMPLETED
        ]
        w_remain = sum(t.duration for t in st_tasks)
        feat[1 + s] = float(w_remain) / h0

        running_count = sum(1 for t in st_tasks if t.status == TaskStatus.RUNNING)
        feat[6 + s] = float(running_count) / 3.0

        delayed_tasks = [t for t in st_tasks if t.material_ready_time > current_time]
        feat[11 + s] = float(len(delayed_tasks)) / 10.0

        if delayed_tasks:
            max_delay = max(t.material_ready_time - current_time for t in delayed_tasks)
            feat[16 + s] = math.log1p(float(max_delay) / h0)

        feat[21 + s] = float(ac_id) / float(state.num_aircraft) if ac_id is not None else -1.0

    feat[26] = max(0.0, estimated_cmax - current_time) / h0
    feat[27] = (p_last + h0 - current_time) / h0
    feat[28] = float(state.current_cycle) / total_cycles

    completed = sum(1 for t in state.tasks.values() if t.status == TaskStatus.COMPLETED)
    feat[29] = float(completed) / total_tasks

    postponed = sum(t.postpone_count for t in state.tasks.values())
    feat[30] = float(postponed) / 50.0

    feat[31] = current_time / (total_cycles * h0)

    return feat


class ActorCriticWork3(nn.Module):
    """支持条件分支自回归解码的 Actor-Critic 强化学习网络。"""

    def __init__(
        self,
        state_dim: int = 32,
        task_feat_dim: int = 8,
        hidden_dim: int = 64,
        max_station_workers: int = 16,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.task_feat_dim = task_feat_dim
        self.hidden_dim = hidden_dim
        self.max_station_workers = max_station_workers

        # 图骨干；encoder 保留为旧32维接口的补充状态分支和旧检查点兼容层。
        self.graph_encoder = Work3GraphEncoder(hidden_dim)
        self.task_graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.worker_graph_score = nn.Linear(hidden_dim, 1)
        self.graph_builder: MultiAircraftGraphBuilder | None = None
        self._graph_baseline_key: str | None = None
        self.graph_policy_enabled = True

        # 1. 状态编码器与 Critic 头
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, 1),
        )

        # 2. 显式时间特征融合模块 (Task 6.3)
        self.time_fusion = TimeContextFusion(
            context_dim=hidden_dim,
            time_dim=2,
            hidden_dim=hidden_dim,
            stop_time_gradient=True,
        )

        # 3. Head 1: 工序评分打分头 (Task Selection Head)
        self.task_proj = nn.Linear(task_feat_dim, hidden_dim)
        self.task_score_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, 1),
        )

        # 4. Head 2: 站位分支选择头 (Station Branch Head: STAY vs POSTPONE)
        self.branch_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, 2),  # 0: STAY, 1: POSTPONE
        )

        # 5. Head 3: 站内工人指针头 (Worker Pointer Head)
        # 输入: [e_fused, task_embed, cumulative_worker_embed]
        self.worker_score_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, max_station_workers),
        )

        # 6. Head 4: 二元对齐头 (Binary Alignment Head)
        self.align_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + max_station_workers, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, 2),  # 0: 对齐 α=0, 1: 对齐 α=1
        )

    def _get_graph_builder(self, env: AirLineEnvWork3) -> MultiAircraftGraphBuilder:
        """按环境基准实例缓存图构造器，动态节点特征仍每步重建。"""
        baseline_key = str(Path(env.baseline_json_path).resolve())
        if self.graph_builder is None or self._graph_baseline_key != baseline_key:
            baseline = MultiAircraftBaseline.load_from_json(env.baseline_json_path)
            self.graph_builder = MultiAircraftGraphBuilder(
                baseline,
                config=Work3ResourceConfig(),
            )
            self._graph_baseline_key = baseline_key
        return self.graph_builder

    def build_graph_snapshot(self, env: AirLineEnvWork3) -> HeteroData:
        """构造可保存的当前图快照，避免PPO重放读取变化后的现场状态。"""
        graph = self._get_graph_builder(env).build_graph(env)
        return graph.clone()

    def _encode_state_components(
        self,
        state_feat: torch.Tensor,
        time_urgency: torch.Tensor,
        graph_data: HeteroData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """编码状态并返回图节点嵌入；无图时走旧接口兼容路径。"""
        assert state_feat.ndim == 1 and state_feat.size(0) == self.state_dim
        assert time_urgency.ndim == 1 and time_urgency.size(0) == 2
        legacy_context = self.encoder(state_feat)
        task_nodes: torch.Tensor | None = None
        worker_nodes: torch.Tensor | None = None
        if graph_data is None or not self.graph_policy_enabled:
            context = legacy_context
        else:
            graph_context, task_nodes, worker_nodes = self.graph_encoder(graph_data)
            context = graph_context + legacy_context
        v = self.critic(context).squeeze(-1)
        e_fused = self.time_fusion(context, time_urgency)
        return v, e_fused, task_nodes, worker_nodes

    def encode_shared_representation(
        self,
        state_feat: torch.Tensor,
        graph_data: HeteroData | None = None,
    ) -> torch.Tensor:
        """返回供时间辅助头共享的图增强状态表征。"""
        assert state_feat.ndim == 1 and state_feat.size(0) == self.state_dim
        legacy_context = self.encoder(state_feat)
        if graph_data is None or not self.graph_policy_enabled:
            return legacy_context
        graph_context, _, _ = self.graph_encoder(graph_data)
        return legacy_context + graph_context

    def encode_state(
        self,
        state_feat: torch.Tensor,
        time_urgency: torch.Tensor,
        graph_data: HeteroData | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """编码状态，图快照存在时输出图增强的价值与策略上下文。"""
        if state_feat.ndim == 2:
            if graph_data is not None:
                raise ValueError("批量状态请通过 evaluate_action_log_probs 传入图快照列表")
            z = self.encoder(state_feat)
            v = self.critic(z).squeeze(-1)
            e_fused = self.time_fusion(z, time_urgency)
            return v, e_fused
        v, e_fused, _, _ = self._encode_state_components(
            state_feat,
            time_urgency,
            graph_data,
        )
        return v, e_fused

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False):
        """兼容没有图模块参数的旧工作三 Actor 检查点。"""
        has_graph_weights = any(key.startswith("graph_encoder.") for key in state_dict)
        if strict and not has_graph_weights:
            strict = False
        self.graph_policy_enabled = has_graph_weights
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _advance_logits(self, context: torch.Tensor) -> torch.Tensor:
        """复用既有分支头产生推进门控，保持旧Actor检查点可加载。"""
        zero_task = torch.zeros(self.hidden_dim, dtype=context.dtype, device=context.device)
        return self.branch_head(torch.cat([context, zero_task], dim=-1))

    @torch.no_grad()
    def select_action(
        self,
        env: AirLineEnvWork3,
        state_feat: torch.Tensor,
        time_urgency: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[dict[str, Any] | None, float, float, dict[str, Any]]:
        """与仿真环境交互采样单步条件动作。

        Returns:
            (action_dict, log_prob, state_value, sample_record)
        """
        device = next(self.parameters()).device
        state_feat = state_feat.to(device)
        time_urgency = time_urgency.to(device)

        candidate_tasks = env.get_action_candidates()
        if not candidate_tasks:
            return None, 0.0, 0.0, {}

        # 1. 状态编码
        graph_snapshot = self.build_graph_snapshot(env)
        graph_builder = self._get_graph_builder(env)
        v, e_fused, task_nodes, worker_nodes = self._encode_state_components(
            state_feat,
            time_urgency,
            graph_snapshot,
        )
        state_value = float(v.item())

        revision_available = any(
            task.status == TaskStatus.RESERVED
            and any(env.get_action_branch_mask(task))
            for task in candidate_tasks
        )
        advance_log_prob = torch.zeros((), device=device)
        advance_choice = 0
        if revision_available:
            advance_logits = self._advance_logits(e_fused)
            dist_advance = Categorical(logits=advance_logits)
            advance_choice = (
                int(torch.argmax(advance_logits).item())
                if deterministic
                else int(dist_advance.sample().item())
            )
            advance_log_prob = dist_advance.log_prob(
                torch.tensor(advance_choice, device=device)
            )
            if advance_choice == 1:
                return (
                    {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT},
                    float(advance_log_prob.item()),
                    state_value,
                    {
                        "action_type": "advance_to_next_event",
                        "advance_available": True,
                        "advance_choice": 1,
                        "branch": int(ActionBranch.ADVANCE_TO_NEXT_EVENT),
                        "graph_snapshot": graph_snapshot,
                        "graph_version": "work3_graph_v1",
                        "candidate_task_node_indices": (),
                        "worker_node_indices": (),
                    },
                )

        # 2. Head 1: 工序选择
        cand_feats = extract_candidate_task_features(env.state, candidate_tasks).to(device)
        cand_embed = self.task_proj(cand_feats)  # (N, hidden_dim)
        candidate_task_node_indices = [
            graph_builder.task_key_to_idx[task.task_key] for task in candidate_tasks
        ]
        if task_nodes is not None:
            node_indices = torch.as_tensor(candidate_task_node_indices, device=device)
            cand_embed = cand_embed + self.task_graph_proj(task_nodes[node_indices])

        e_ctx_expanded = e_fused.unsqueeze(0).expand(len(candidate_tasks), -1)  # (N, hidden_dim)
        task_pair = torch.cat([e_ctx_expanded, cand_embed], dim=-1)
        task_logits = self.task_score_fc(task_pair).squeeze(-1)  # (N,)

        dist_task = Categorical(logits=task_logits)
        if deterministic:
            task_idx = int(torch.argmax(task_logits).item())
        else:
            task_idx = int(dist_task.sample().item())

        chosen_task = candidate_tasks[task_idx]
        log_prob_task = dist_task.log_prob(torch.tensor(task_idx, device=device))
        chosen_task_embed = cand_embed[task_idx]

        # 3. Head 2: 站位分支二选一 (STAY vs POSTPONE)
        branch_input = torch.cat([e_fused, chosen_task_embed], dim=-1)
        branch_logits = self.branch_head(branch_input).clone()

        # 物理硬掩码：环境与Actor共用同一套后移合法性规则。
        can_reserve, can_postpone = env.get_action_branch_mask(chosen_task)

        if not can_postpone:
            branch_logits[1] = -1e4
        if not can_reserve:
            branch_logits[0] = -1e4

        dist_branch = Categorical(logits=branch_logits)
        if deterministic:
            branch_act = int(torch.argmax(branch_logits).item())
        else:
            branch_act = int(dist_branch.sample().item())

        log_prob_branch = dist_branch.log_prob(torch.tensor(branch_act, device=device))

        # -------------------------------------------------------------
        # 4. 条件分支判定与动作截断 (Conditional Branch Truncation)
        # -------------------------------------------------------------
        sample_record = {
            "action_type": "schedule",
            "advance_available": revision_available,
            "advance_choice": advance_choice,
            "task_idx": task_idx,
            "task_key": chosen_task.task_key,
            "branch": branch_act,
            "cand_feats": cand_feats.cpu(),
            "can_reserve": can_reserve,
            "can_postpone": can_postpone,
            "num_st_workers": self.max_station_workers,
            "chosen_team": (),
            "align": 0,
            "graph_snapshot": graph_snapshot,
            "graph_version": "work3_graph_v1",
            "candidate_task_node_indices": tuple(candidate_task_node_indices),
            "worker_node_indices": (),
        }

        if branch_act == 1:
            # 分支 B: POSTPONE 后移 —— 动作严格截断！
            total_log_prob = advance_log_prob + log_prob_task + log_prob_branch
            action_dict = {
                "task_key": chosen_task.task_key,
                "branch": ActionBranch.POSTPONE,
            }
            return action_dict, float(total_log_prob.item()), state_value, sample_record

        # 分支 A: STAY 留站执行 —— 继续解码站内团队与对齐
        st_workers = env.state.station_worker_bindings.get(chosen_task.current_station, [])
        num_st_workers = len(st_workers)
        sample_record["num_st_workers"] = num_st_workers
        worker_node_indices = [graph_builder.worker_id_to_idx[w] for w in st_workers]
        sample_record["worker_node_indices"] = tuple(worker_node_indices)
        demand = chosen_task.demand

        # 5. Head 3: 站内工人指针自回归选择
        chosen_worker_indices: list[int] = []
        log_prob_workers = torch.tensor(0.0, device=device)
        worker_mask_tracker = torch.zeros(self.max_station_workers, dtype=torch.float, device=device)

        for step_w in range(demand):
            ptr_input = torch.cat([e_fused, chosen_task_embed, worker_mask_tracker], dim=-1)
            w_logits = self.worker_score_fc(ptr_input).clone()

            # 掩码: 超出本站工人数量的位置置 -1e4
            if self.max_station_workers > num_st_workers:
                w_logits[num_st_workers:] = -1e4
            # 掩码: 已选取的工人置 -1e4
            for prev_idx in chosen_worker_indices:
                w_logits[prev_idx] = -1e4
            if worker_nodes is not None and worker_node_indices:
                worker_graph_ids = torch.as_tensor(worker_node_indices, device=device)
                worker_bias = self.worker_graph_score(worker_nodes[worker_graph_ids]).squeeze(-1)
                w_logits[:num_st_workers] = w_logits[:num_st_workers] + worker_bias
            valid_global_workers = set(
                env.valid_team_completion_workers(
                    chosen_task,
                    [st_workers[index] for index in chosen_worker_indices],
                )
            )
            for worker_index, worker_id in enumerate(st_workers[:num_st_workers]):
                if worker_id not in valid_global_workers:
                    w_logits[worker_index] = -1e4

            dist_w = Categorical(logits=w_logits)
            if deterministic:
                w_idx = int(torch.argmax(w_logits).item())
            else:
                w_idx = int(dist_w.sample().item())

            chosen_worker_indices.append(w_idx)
            log_prob_workers = log_prob_workers + dist_w.log_prob(torch.tensor(w_idx, device=device))
            worker_mask_tracker[w_idx] = 1.0

        chosen_team = tuple(st_workers[idx] for idx in chosen_worker_indices)

        # 6. Head 4: 二元对齐选择
        align_input = torch.cat([e_fused, chosen_task_embed, worker_mask_tracker], dim=-1)
        align_logits = self.align_head(align_input)
        dist_align = Categorical(logits=align_logits)

        if deterministic:
            align_act = int(torch.argmax(align_logits).item())
        else:
            align_act = int(dist_align.sample().item())

        log_prob_align = dist_align.log_prob(torch.tensor(align_act, device=device))

        # 7. 全量对数概率求和
        total_log_prob = (
            advance_log_prob + log_prob_task + log_prob_branch + log_prob_workers + log_prob_align
        )

        sample_record["chosen_team"] = chosen_team
        sample_record["chosen_worker_indices"] = tuple(chosen_worker_indices)
        sample_record["align"] = align_act

        action_dict = {
            "task_key": chosen_task.task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": chosen_team,
            "align": align_act,
        }
        return action_dict, float(total_log_prob.item()), state_value, sample_record

    def evaluate_action_log_probs(
        self,
        state_feats: torch.Tensor,
        time_urgencies: torch.Tensor,
        sample_records: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO 训练重放接口：严格按照条件分支结构计算重放对数概率、状态值与熵。

        Returns:
            (values, log_probs, entropies)
        """
        device = next(self.parameters()).device
        state_feats = state_feats.to(device)
        time_urgencies = time_urgencies.to(device)
        batch_size = len(sample_records)

        values_list: list[torch.Tensor] = []
        context_list: list[torch.Tensor] = []
        task_nodes_list: list[torch.Tensor | None] = []
        worker_nodes_list: list[torch.Tensor | None] = []
        for index, record in enumerate(sample_records):
            value_i, context_i, task_nodes_i, worker_nodes_i = self._encode_state_components(
                state_feats[index],
                time_urgencies[index],
                record.get("graph_snapshot"),
            )
            values_list.append(value_i)
            context_list.append(context_i)
            task_nodes_list.append(task_nodes_i)
            worker_nodes_list.append(worker_nodes_i)
        values = torch.stack(values_list)
        e_fused = torch.stack(context_list)

        log_probs = []
        entropies = []

        for i in range(batch_size):
            rec = sample_records[i]
            e_ctx = e_fused[i]

            if rec.get("action_type") == "advance_to_next_event":
                dist_advance = Categorical(logits=self._advance_logits(e_ctx))
                lp_advance = dist_advance.log_prob(torch.as_tensor(1, device=device))
                log_probs.append(lp_advance)
                entropies.append(dist_advance.entropy())
                continue

            lp_advance = torch.zeros((), device=device)
            ent_advance = torch.zeros((), device=device)
            if rec.get("advance_available", False):
                dist_advance = Categorical(logits=self._advance_logits(e_ctx))
                lp_advance = dist_advance.log_prob(torch.as_tensor(0, device=device))
                ent_advance = dist_advance.entropy()

            # 1. 重放 Head 1: 工序对数概率
            cand_feats = rec["cand_feats"].to(device)
            task_idx = rec["task_idx"]
            cand_embed = self.task_proj(cand_feats)
            replay_task_nodes = task_nodes_list[i]
            candidate_node_indices = rec.get("candidate_task_node_indices", ())
            if replay_task_nodes is not None and candidate_node_indices:
                node_indices = torch.as_tensor(candidate_node_indices, device=device)
                cand_embed = cand_embed + self.task_graph_proj(
                    replay_task_nodes[node_indices]
                )
            e_ctx_exp = e_ctx.unsqueeze(0).expand(len(cand_feats), -1)
            task_pair = torch.cat([e_ctx_exp, cand_embed], dim=-1)
            task_logits = self.task_score_fc(task_pair).squeeze(-1)
            dist_task = Categorical(logits=task_logits)
            lp_task = dist_task.log_prob(torch.as_tensor(task_idx, device=device))
            ent_task = dist_task.entropy()

            # 2. 重放 Head 2: 站位分支对数概率
            chosen_task_embed = cand_embed[task_idx]
            branch_input = torch.cat([e_ctx, chosen_task_embed], dim=-1)
            branch_logits = self.branch_head(branch_input).clone()
            can_reserve = rec.get("can_reserve", True)
            can_postpone = rec.get("can_postpone", True)
            if not can_reserve:
                branch_logits[0] = -1e4
            if not can_postpone:
                branch_logits[1] = -1e4
            dist_branch = Categorical(logits=branch_logits)
            branch_act = rec["branch"]
            lp_branch = dist_branch.log_prob(torch.as_tensor(branch_act, device=device))
            ent_branch = dist_branch.entropy()

            # 3. 条件分支截断判定
            if branch_act == 1:
                # POSTPONE 分支：仅累加工序与站位
                log_probs.append(lp_advance + lp_task + lp_branch)
                entropies.append(ent_advance + ent_task + ent_branch)
                continue

            # 4. STAY 分支：重放选人与对齐
            chosen_worker_indices = rec.get("chosen_worker_indices", ())
            num_st_workers = rec.get("num_st_workers", self.max_station_workers)
            lp_workers = torch.zeros((), device=device)
            ent_workers = torch.zeros((), device=device)
            worker_mask = torch.zeros(self.max_station_workers, dtype=torch.float, device=device)
            replay_worker_nodes = worker_nodes_list[i]
            worker_node_indices = rec.get("worker_node_indices", ())
            worker_bias = None
            if replay_worker_nodes is not None and worker_node_indices:
                node_indices = torch.as_tensor(worker_node_indices, device=device)
                worker_bias = self.worker_graph_score(
                    replay_worker_nodes[node_indices]
                ).squeeze(-1)

            for step_w, w_idx in enumerate(chosen_worker_indices):
                ptr_input = torch.cat([e_ctx, chosen_task_embed, worker_mask], dim=-1)
                w_logits = self.worker_score_fc(ptr_input).clone()
                # 掩码超出本站工人数的位置
                if self.max_station_workers > num_st_workers:
                    w_logits[num_st_workers:] = -1e4
                if worker_bias is not None:
                    w_logits[:num_st_workers] = w_logits[:num_st_workers] + worker_bias
                # 屏蔽已选
                for prev in chosen_worker_indices[:step_w]:
                    w_logits[prev] = -1e4
                dist_w = Categorical(logits=w_logits)
                lp_workers = lp_workers + dist_w.log_prob(torch.as_tensor(w_idx, device=device))
                ent_workers = ent_workers + dist_w.entropy()
                worker_mask[w_idx] = 1.0

            align_input = torch.cat([e_ctx, chosen_task_embed, worker_mask], dim=-1)
            align_logits = self.align_head(align_input)
            dist_align = Categorical(logits=align_logits)
            align_act = rec.get("align", 0)
            lp_align = dist_align.log_prob(torch.as_tensor(align_act, device=device))
            ent_align = dist_align.entropy()

            total_lp = lp_advance + lp_task + lp_branch + lp_workers + lp_align
            total_ent = ent_advance + ent_task + ent_branch + ent_workers + ent_align

            log_probs.append(total_lp)
            entropies.append(total_ent)

        return values, torch.stack(log_probs), torch.stack(entropies)
