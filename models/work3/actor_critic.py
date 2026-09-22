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
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from envs.work3.core_types import ActionBranch, MultiAircraftState, TaskRuntimeState, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from models.work3.action_fusion import TimeContextFusion, compute_time_urgency_vector


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

    def encode_state(
        self,
        state_feat: torch.Tensor,
        time_urgency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """编码状态表征，输出状态值 V(s) 与时间感知决策上下文 e_fused。"""
        z = self.encoder(state_feat)
        v = self.critic(z).squeeze(-1)
        e_fused = self.time_fusion(z, time_urgency)
        return v, e_fused

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
        v, e_fused = self.encode_state(state_feat, time_urgency)
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
                    },
                )

        # 2. Head 1: 工序选择
        cand_feats = extract_candidate_task_features(env.state, candidate_tasks).to(device)
        cand_embed = self.task_proj(cand_feats)  # (N, hidden_dim)

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

        values, e_fused = self.encode_state(state_feats, time_urgencies)

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

            for step_w, w_idx in enumerate(chosen_worker_indices):
                ptr_input = torch.cat([e_ctx, chosen_task_embed, worker_mask], dim=-1)
                w_logits = self.worker_score_fc(ptr_input).clone()
                # 掩码超出本站工人数的位置
                if self.max_station_workers > num_st_workers:
                    w_logits[num_st_workers:] = -1e4
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
