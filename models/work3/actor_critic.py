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
from envs.work3.decision_snapshot import (
    DecisionSnapshot,
    TeamCompletionContext,
    build_decision_snapshot,
    worker_completion_mask,
)
from envs.work3.environment import AirLineEnvWork3
from models.work3.action_fusion import TimeContextFusion, compute_time_urgency_vector
from models.hb_gat_pn import FeatureEmbedder, HeteroGATEncoder
from models.work3.graph_builder import (
    GRAPH_FEATURE_DIMS,
    GRAPH_FEATURE_SCHEMA,
    GRAPH_FEATURE_VERSION,
    MultiAircraftGraphBuilder,
    Work3ResourceConfig,
)
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


class Work3GraphEncoder(nn.Module):
    """工作三轻量异构图编码器，复用仓库已有 HB-GAT 输入与消息传递。"""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        config = SimpleNamespace(
            hidden_dim=int(hidden_dim),
            num_gat_layers=1,
            num_heads=2,
            task_feat_dim=GRAPH_FEATURE_DIMS["task"],
            worker_feat_dim=GRAPH_FEATURE_DIMS["worker"],
            station_feat_dim=GRAPH_FEATURE_DIMS["station"],
            skill_feat_dim=GRAPH_FEATURE_DIMS["skill"],
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
        self.baseline_team_task_to_worker = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.published_team_task_to_worker = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.baseline_team_worker_to_task = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.published_team_worker_to_task = nn.Linear(hidden_dim, hidden_dim, bias=False)
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

        task_emb = encoded["task"]
        worker_emb = encoded["worker"]
        for edge_key, t2w_proj, w2t_proj in (
            (
                ("task", "baseline_team", "worker"),
                self.baseline_team_task_to_worker,
                self.baseline_team_worker_to_task,
            ),
            (
                ("task", "last_published_team", "worker"),
                self.published_team_task_to_worker,
                self.published_team_worker_to_task,
            ),
        ):
            edge_index = edge_index_dict.get(edge_key)
            if edge_index is not None and edge_index.numel() > 0:
                src_task, dst_worker = edge_index[0], edge_index[1]
                w_delta = torch.zeros_like(worker_emb)
                w_delta.index_add_(
                    0,
                    dst_worker,
                    t2w_proj(task_emb[src_task]).to(dtype=w_delta.dtype),
                )
                t_delta = torch.zeros_like(task_emb)
                t_delta.index_add_(
                    0,
                    src_task,
                    w2t_proj(worker_emb[dst_worker]).to(dtype=t_delta.dtype),
                )
                worker_emb = worker_emb + w_delta
                task_emb = task_emb + t_delta
        encoded["task"] = task_emb
        encoded["worker"] = worker_emb

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
    """提取候选工序的 8 维标准化物理特征张量（含未来预约时刻与上一版发布差异）。"""
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

        last_pub = task.last_published_assignment or task.baseline_assignment
        pub_pos = (
            float(last_pub["position"])
            if isinstance(last_pub, dict)
            and last_pub.get("position") is not None
            and math.isfinite(float(last_pub["position"]))
            else float(task.in_station_offset)
        )
        pub_station = (
            float(last_pub.get("station", task.base_station))
            if isinstance(last_pub, dict) and last_pub.get("station") is not None
            else float(task.base_station)
        )
        pub_team = last_pub.get("team") if isinstance(last_pub, dict) else None
        team_diff_ratio = 0.0
        if pub_team and task.demand > 0:
            overlap = len(set(pub_team) & set(task.base_team))
            team_diff_ratio = 1.0 - (float(overlap) / float(task.demand))

        is_reserved = task.status == TaskStatus.RESERVED
        has_sched = (
            task.status in (TaskStatus.RESERVED, TaskStatus.RUNNING)
            and task.scheduled_start is not None
            and math.isfinite(float(task.scheduled_start))
        )
        sched_start_rem = (
            max(0.0, float(task.scheduled_start) - current_time) / h0
            if has_sched
            else 0.0
        )
        exec_dur = (
            task.execution_duration
            if task.execution_duration is not None
            else task.duration
        )
        sched_end_rem = (
            max(0.0, float(task.scheduled_start) + float(exec_dur) - current_time) / h0
            if has_sched and exec_dur is not None and math.isfinite(float(exec_dur))
            else 0.0
        )

        feats[idx, 0] = float(task.duration) / h0
        feats[idx, 1] = (pub_pos / h0) + (sched_start_rem if is_reserved else 0.0)
        feats[idx, 2] = (float(task.demand) / 5.0) + (0.25 * team_diff_ratio)
        feats[idx, 3] = (float(task.current_station) / 4.0) + (0.1 * (pub_station - float(task.base_station)))
        feats[idx, 4] = (float(task.postpone_count) / 5.0) + (0.5 if is_reserved else 0.0)

        is_delayed = 1.0 if task.material_ready_time > current_time else 0.0
        feats[idx, 5] = is_delayed + (0.5 if is_reserved else 0.0)
        feats[idx, 6] = (max(0.0, float(task.material_ready_time - current_time)) / h0) + sched_start_rem
        feats[idx, 7] = (rel_station_offset / 4.0) + sched_end_rem

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

    def _worker_selection_mask(
        self,
        context: TeamCompletionContext,
        selected_worker_ids: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """构造当前指针步的固定长度布尔可选工人掩码。"""
        mask = torch.tensor(
            worker_completion_mask(context, selected_worker_ids, self.max_station_workers),
            dtype=torch.bool,
            device=device,
        )
        if not bool(mask.any().item()):
            raise ValueError(
                f"工序 {context.task_key} 的工人指针无合法补全："
                f"已选工人={selected_worker_ids}"
            )
        return mask

    @staticmethod
    def _apply_worker_selection_mask(
        logits: torch.Tensor,
        mask: torch.Tensor,
        *,
        task_key: str,
        step_index: int,
    ) -> torch.Tensor:
        """掩码在图偏置加入后应用；拒绝全屏蔽或形状不匹配的分布。"""
        if mask.dtype != torch.bool or mask.shape != logits.shape:
            raise ValueError(
                f"工序 {task_key} 第{step_index}步工人掩码形状/类型无效："
                f"mask={tuple(mask.shape)} {mask.dtype}, logits={tuple(logits.shape)}"
            )
        if not bool(mask.any().item()):
            raise ValueError(f"工序 {task_key} 第{step_index}步工人掩码全空")
        return logits.masked_fill(~mask, -torch.inf)

    def make_decision_snapshot(
        self,
        env: AirLineEnvWork3,
        state_feat: torch.Tensor,
        time_urgency: torch.Tensor,
        *,
        worker_id: int = 0,
        episode_id: int = -1,
        episode_index: int | None = None,
        estimated_cmax: float | None = None,
    ) -> DecisionSnapshot:
        """兼容单环境入口：从当前现场生成不含环境引用的CPU决策快照。"""
        return build_decision_snapshot(
            env,
            self._get_graph_builder(env),
            state_feat,
            time_urgency,
            worker_id=worker_id,
            episode_id=episode_id,
            episode_index=env.step_count if episode_index is None else episode_index,
            estimated_cmax=estimated_cmax,
        )

    @torch.no_grad()
    def select_action(
        self,
        env: AirLineEnvWork3,
        state_feat: torch.Tensor,
        time_urgency: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[dict[str, Any] | None, float, float, dict[str, Any]]:
        """兼容旧调用者，先冻结现场快照，再走纯快照策略入口。"""
        snapshot = self.make_decision_snapshot(env, state_feat, time_urgency)
        return self.select_snapshot(snapshot, deterministic=deterministic)

    @torch.no_grad()
    def select_snapshot(
        self,
        snapshot: DecisionSnapshot,
        deterministic: bool = False,
    ) -> tuple[dict[str, Any] | None, float, float, dict[str, Any]]:
        """只依据独立CPU快照采样动作；不读取环境对象或可变现场状态。"""
        if snapshot.graph_version != GRAPH_FEATURE_VERSION:
            raise ValueError(
                f"快照图版本{snapshot.graph_version!r}与Actor要求的"
                f"{GRAPH_FEATURE_VERSION!r}不匹配"
            )
        device = next(self.parameters()).device
        state_feat = snapshot.state_features.to(device)
        time_urgency = snapshot.time_features.to(device)
        graph_snapshot = snapshot.graph_snapshot
        v, e_fused, task_nodes, worker_nodes = self._encode_state_components(
            state_feat,
            time_urgency,
            graph_snapshot,
        )
        state_value = float(v.item())
        candidate_count = len(snapshot.candidate_task_keys)
        sample_common = {
            "sample_record_version": 2,
            "decision_snapshot_id": (
                snapshot.worker_id,
                snapshot.episode_id,
                snapshot.episode_index,
            ),
            "graph_snapshot": graph_snapshot.clone(),
            "graph_version": snapshot.graph_version,
            "candidate_task_keys": snapshot.candidate_task_keys,
            "candidate_branch_masks": snapshot.branch_masks,
            "candidate_task_node_indices": snapshot.candidate_task_node_indices,
            "worker_node_indices": (),
        }

        if candidate_count == 0:
            return (
                {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT},
                0.0,
                state_value,
                {
                    **sample_common,
                    "action_type": "forced_advance",
                    "advance_available": False,
                    "branch": int(ActionBranch.ADVANCE_TO_NEXT_EVENT),
                },
            )

        advance_log_prob = torch.zeros((), device=device)
        advance_choice = 0
        if snapshot.advance_available:
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
                        **sample_common,
                        "action_type": "advance_to_next_event",
                        "advance_available": True,
                        "advance_choice": 1,
                        "branch": int(ActionBranch.ADVANCE_TO_NEXT_EVENT),
                    },
                )

        cand_feats = snapshot.candidate_task_features.to(device)
        cand_embed = self.task_proj(cand_feats)
        candidate_task_node_indices = snapshot.candidate_task_node_indices
        if task_nodes is not None:
            node_indices = torch.as_tensor(candidate_task_node_indices, device=device)
            cand_embed = cand_embed + self.task_graph_proj(task_nodes[node_indices])
        e_ctx_expanded = e_fused.unsqueeze(0).expand(candidate_count, -1)
        task_pair = torch.cat([e_ctx_expanded, cand_embed], dim=-1)
        task_logits = self.task_score_fc(task_pair).squeeze(-1)
        dist_task = Categorical(logits=task_logits)
        task_idx = (
            int(torch.argmax(task_logits).item())
            if deterministic
            else int(dist_task.sample().item())
        )
        task_key = snapshot.candidate_task_keys[task_idx]
        log_prob_task = dist_task.log_prob(torch.tensor(task_idx, device=device))
        chosen_task_embed = cand_embed[task_idx]

        branch_input = torch.cat([e_fused, chosen_task_embed], dim=-1)
        branch_logits = self.branch_head(branch_input).clone()
        can_reserve, can_postpone = snapshot.branch_masks[task_idx]
        if not can_postpone:
            branch_logits[1] = -1e4
        if not can_reserve:
            branch_logits[0] = -1e4
        dist_branch = Categorical(logits=branch_logits)
        branch_act = (
            int(torch.argmax(branch_logits).item())
            if deterministic
            else int(dist_branch.sample().item())
        )
        log_prob_branch = dist_branch.log_prob(torch.tensor(branch_act, device=device))
        context = snapshot.team_contexts[task_idx]
        chosen_worker_ids: list[int] = []
        sample_record = {
            **sample_common,
            "action_type": "schedule",
            "advance_available": snapshot.advance_available,
            "advance_choice": advance_choice,
            "task_idx": task_idx,
            "task_key": task_key,
            "branch": branch_act,
            "cand_feats": cand_feats.detach().cpu().clone(),
            "can_reserve": can_reserve,
            "can_postpone": can_postpone,
            "num_st_workers": self.max_station_workers,
            "chosen_team": (),
            "align": 0,
        }

        if branch_act == int(ActionBranch.POSTPONE):
            total_log_prob = advance_log_prob + log_prob_task + log_prob_branch
            return (
                {"task_key": task_key, "branch": ActionBranch.POSTPONE},
                float(total_log_prob.item()),
                state_value,
                sample_record,
            )

        num_station_workers = len(context.workers)
        worker_node_indices = snapshot.worker_node_indices[task_idx]
        sample_record["num_st_workers"] = num_station_workers
        sample_record["worker_node_indices"] = worker_node_indices
        sample_record["station_worker_ids"] = tuple(
            worker.worker_id for worker in context.workers
        )
        worker_valid_masks: list[tuple[bool, ...]] = []
        chosen_worker_indices: list[int] = []
        log_prob_workers = torch.zeros((), device=device)
        worker_mask_tracker = torch.zeros(
            self.max_station_workers, dtype=torch.float, device=device
        )
        worker_graph_bias = None
        if worker_nodes is not None and worker_node_indices:
            worker_graph_ids = torch.as_tensor(worker_node_indices, device=device)
            worker_graph_bias = self.worker_graph_score(
                worker_nodes[worker_graph_ids]
            ).squeeze(-1)

        for step_w in range(context.demand):
            ptr_input = torch.cat([e_fused, chosen_task_embed, worker_mask_tracker], dim=-1)
            worker_logits = self.worker_score_fc(ptr_input).clone()
            if worker_graph_bias is not None:
                worker_logits[:num_station_workers] += worker_graph_bias
            worker_mask = self._worker_selection_mask(
                context,
                tuple(chosen_worker_ids),
                device,
            )
            worker_valid_masks.append(tuple(bool(value) for value in worker_mask.cpu().tolist()))
            worker_logits = self._apply_worker_selection_mask(
                worker_logits,
                worker_mask,
                task_key=task_key,
                step_index=step_w,
            )
            dist_worker = Categorical(logits=worker_logits)
            worker_index = (
                int(torch.argmax(worker_logits).item())
                if deterministic
                else int(dist_worker.sample().item())
            )
            chosen_worker_indices.append(worker_index)
            chosen_worker_ids.append(context.workers[worker_index].worker_id)
            log_prob_workers += dist_worker.log_prob(
                torch.tensor(worker_index, device=device)
            )
            worker_mask_tracker[worker_index] = 1.0

        align_input = torch.cat([e_fused, chosen_task_embed, worker_mask_tracker], dim=-1)
        align_logits = self.align_head(align_input)
        dist_align = Categorical(logits=align_logits)
        align_act = (
            int(torch.argmax(align_logits).item())
            if deterministic
            else int(dist_align.sample().item())
        )
        log_prob_align = dist_align.log_prob(torch.tensor(align_act, device=device))
        total_log_prob = (
            advance_log_prob + log_prob_task + log_prob_branch + log_prob_workers + log_prob_align
        )
        sample_record["chosen_team"] = tuple(chosen_worker_ids)
        sample_record["chosen_worker_indices"] = tuple(chosen_worker_indices)
        sample_record["worker_valid_masks"] = tuple(worker_valid_masks)
        sample_record["align"] = align_act
        action = {
            "task_key": task_key,
            "branch": ActionBranch.STATION_EXECUTE,
            "team": tuple(chosen_worker_ids),
            "align": align_act,
        }
        return action, float(total_log_prob.item()), state_value, sample_record

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
        for record in sample_records:
            graph_version = record.get("graph_version")
            if graph_version is not None and graph_version != GRAPH_FEATURE_VERSION:
                raise ValueError(
                    f"PPO样本图版本{graph_version!r}与Actor要求的"
                    f"{GRAPH_FEATURE_VERSION!r}不匹配"
                )

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

            if rec.get("action_type") == "forced_advance":
                log_probs.append(torch.zeros((), device=device))
                entropies.append(torch.zeros((), device=device))
                continue

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
            candidate_masks = rec.get("candidate_branch_masks")
            if candidate_masks is not None:
                if len(candidate_masks) != len(cand_feats) or not 0 <= task_idx < len(candidate_masks):
                    raise ValueError("PPO样本中的候选分支掩码与工序特征数量不匹配")
                candidate_keys = rec.get("candidate_task_keys", ())
                if candidate_keys and (
                    len(candidate_keys) != len(candidate_masks)
                    or candidate_keys[task_idx] != rec.get("task_key")
                ):
                    raise ValueError("PPO样本的工序键与快照候选顺序不匹配")
                can_reserve, can_postpone = candidate_masks[task_idx]
                branch = int(rec.get("branch", -1))
                if branch not in (0, 1) or not candidate_masks[task_idx][branch]:
                    raise ValueError("PPO样本选择了快照掩码禁止的动作分支")
                if (
                    bool(can_reserve) != bool(rec.get("can_reserve"))
                    or bool(can_postpone) != bool(rec.get("can_postpone"))
                ):
                    raise ValueError("PPO样本的分支掩码与所选工序掩码不一致")
            else:
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
            if rec.get("sample_record_version") != 2:
                raise ValueError(
                    f"工序 {rec.get('task_key', '<unknown>')} 的采样记录版本不支持正式PPO重放；"
                    "请重新采样，不允许静默按全体工人合法处理"
                )
            worker_valid_masks = rec.get("worker_valid_masks")
            if not isinstance(worker_valid_masks, (tuple, list)) or len(worker_valid_masks) != len(
                chosen_worker_indices
            ):
                raise ValueError(
                    f"工序 {rec.get('task_key', '<unknown>')} 缺少逐步工人合法掩码快照"
                )
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
                if worker_bias is not None:
                    w_logits[:num_st_workers] = w_logits[:num_st_workers] + worker_bias
                saved_mask = torch.as_tensor(
                    worker_valid_masks[step_w], dtype=torch.bool, device=device
                )
                if saved_mask.numel() != self.max_station_workers:
                    raise ValueError(
                        f"工序 {rec.get('task_key', '<unknown>')} 第{step_w}步掩码长度"
                        f"{saved_mask.numel()}与Actor输出长度{self.max_station_workers}不符"
                    )
                if not 0 <= int(w_idx) < self.max_station_workers or not bool(saved_mask[int(w_idx)]):
                    raise ValueError(
                        f"工序 {rec.get('task_key', '<unknown>')} 第{step_w}步所选工人"
                        f"索引{w_idx}不在采样时合法掩码中"
                    )
                w_logits = self._apply_worker_selection_mask(
                    w_logits,
                    saved_mask,
                    task_key=str(rec.get("task_key", "<unknown>")),
                    step_index=step_w,
                )
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
