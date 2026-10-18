"""工作三 R3/E04 专项审计与独立证据收集模块。

默认关闭；仅在开启审计开关（WORK3_R3_E04_AUDIT=1 或 --r3-e04-audit）时激活。
本模块遵循严格只读/旁路监听原则：
1. 不修改任何动作采样、随机数种子、奖励计算、转站时机或优化控制流；
2. 捕获真实样本身份、真实周期转站事实、掩码演变及在线/冻结模型参数摘要；
3. 将必要的小型输入快照保存在 CPU 内存（上限数个样本），不长期占用 GPU 显存；
4. 运行结束时导出：
   - r3_e04_events.jsonl (逐事件审计流)
   - r3_e04_inputs.pt (CPU 原始快照支持离线重现)
   - r3_e04_summary.json (数值重算、断言结果与元数据)
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

_GLOBAL_AUDITOR: Work3R3E04Auditor | None = None


def is_r3_e04_audit_enabled() -> bool:
    """检查当前环境是否开启 R3/E04 专项审计。"""
    val = os.environ.get("WORK3_R3_E04_AUDIT", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def get_r3_e04_auditor() -> Work3R3E04Auditor | None:
    """获取全局审计器单例（未开启或未初始化时返回 None）。"""
    global _GLOBAL_AUDITOR
    return _GLOBAL_AUDITOR


def init_r3_e04_auditor(
    output_dir: Path | str,
    run_id: str,
    *,
    enabled: bool = False,
    max_tracked_candidates: int = 5,
) -> Work3R3E04Auditor | None:
    """初始化并激活全局审计器单例。"""
    global _GLOBAL_AUDITOR
    should_enable = enabled or is_r3_e04_audit_enabled()
    if not should_enable:
        _GLOBAL_AUDITOR = None
        return None

    _GLOBAL_AUDITOR = Work3R3E04Auditor(
        output_dir=Path(output_dir),
        run_id=run_id,
        max_tracked_candidates=max_tracked_candidates,
    )
    return _GLOBAL_AUDITOR


def compute_state_dict_sha256(module: nn.Module | None) -> str:
    """对 nn.Module 的 state_dict（按参数与持久/非持久缓冲区名称排序）计算 SHA-256。"""
    if module is None:
        return "none"
    h = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state.keys()):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            continue
        t_cpu = tensor.detach().cpu()
        h.update(name.encode("utf-8"))
        h.update(str(t_cpu.dtype).encode("utf-8"))
        h.update(str(tuple(t_cpu.shape)).encode("utf-8"))
        h.update(t_cpu.numpy().tobytes())
    return h.hexdigest()


def compute_param_l2_diff(mod1_state: dict[str, torch.Tensor], mod2: nn.Module | None) -> float:
    """计算两个状态字典间参数与缓冲区的 L2 范数差值。"""
    if mod2 is None:
        return 0.0
    mod2_state = mod2.state_dict()
    diff_sq = 0.0
    for name in sorted(mod1_state.keys()):
        if name in mod2_state:
            t1 = mod1_state[name].detach().cpu().float()
            t2 = mod2_state[name].detach().cpu().float()
            diff_sq += float(torch.sum((t1 - t2) ** 2).item())
    return math.sqrt(diff_sq)


class Work3R3E04Auditor:
    """R3/E04 专项闭环审计器。"""

    def __init__(
        self,
        output_dir: Path,
        run_id: str,
        max_tracked_candidates: int = 5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = str(run_id)
        self.max_tracked_candidates = int(max_tracked_candidates)

        self.event_seq = 0
        self.optimization_seq = 0
        self.events: list[dict[str, Any]] = []

        # 跟踪的样本身份与输入副本（仅限少量样本保存在 CPU）
        self.tracked_sample_ids: list[str] = []
        self.tracked_inputs: dict[str, dict[str, Any]] = {}
        self.tracked_history: dict[str, dict[str, Any]] = {}

        # 周期事实记录：key = (worker_id, episode_id, cycle_id)
        self.cycle_transfer_facts: dict[tuple[int, int, int], dict[str, Any]] = {}

        # E04 冻结预测器审计相关
        self.frozen_test_snapshot: Any | None = None
        self.frozen_eval_checks: list[dict[str, Any]] = []
        self._pre_fit_state: dict[str, Any] | None = None

    def log_event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """记录一条结构化审计事件并递增序号。"""
        self.event_seq += 1
        entry = {
            "seq": self.event_seq,
            "event_type": str(event_type),
            "run_id": self.run_id,
            **payload,
        }
        self.events.append(entry)
        return entry

    # -------------------------------------------------------------------------
    # Buffer 钩子：add、attach_transfer、drain_ready
    # -------------------------------------------------------------------------

    def on_buffer_add(
        self,
        *,
        worker_id: int,
        episode_id: int,
        cycle_id: int,
        decision_id: int,
        state_feat: torch.Tensor,
        graph_snapshot: Any,
        estimated_cmax: float,
        current_time: float,
        h0: float,
        time_urgency: torch.Tensor | None,
        predictor_version: int | None,
    ) -> str:
        """当决策步登记到时间缓存时调用。"""
        sample_id = f"{self.run_id}:{int(worker_id)}:{int(episode_id)}:{int(cycle_id)}:{int(decision_id)}"

        # 优先跟踪第一周期产生的首批候选样本
        is_candidate = False
        if cycle_id == 1 and len(self.tracked_sample_ids) < self.max_tracked_candidates:
            if sample_id not in self.tracked_sample_ids:
                self.tracked_sample_ids.append(sample_id)
                is_candidate = True

                # CPU 浅/深拷贝原始输入，不保留反传图
                feat_cpu = state_feat.detach().cpu().clone()
                self.tracked_inputs[sample_id] = {
                    "sample_id": sample_id,
                    "worker_id": int(worker_id),
                    "episode_id": int(episode_id),
                    "source_cycle_id": int(cycle_id),
                    "decision_id": int(decision_id),
                    "state_feat": feat_cpu,
                    "state_feat_sha256": hashlib.sha256(feat_cpu.numpy().tobytes()).hexdigest(),
                    "estimated_cmax": float(estimated_cmax),
                    "current_time": float(current_time),
                    "h0": float(h0),
                    "predictor_version": predictor_version,
                }
                self.tracked_history[sample_id] = {
                    "sample_id": sample_id,
                    "source_cycle_id": int(cycle_id),
                    "decision_id": int(decision_id),
                    "register_seq": self.event_seq + 1,
                    "drain_current": None,
                    "drain_next": None,
                    "online_updates_between": [],
                    "aux_optimizations": [],
                }

        self.log_event(
            "REGISTER_DECISION",
            {
                "worker_id": int(worker_id),
                "episode_id": int(episode_id),
                "source_cycle_id": int(cycle_id),
                "decision_id": int(decision_id),
                "sample_id": sample_id,
                "is_tracked_candidate": is_candidate,
                "estimated_cmax": float(estimated_cmax),
                "current_time": float(current_time),
                "h0": float(h0),
            },
        )
        return sample_id

    def on_buffer_attach_transfer(
        self,
        *,
        worker_id: int,
        episode_id: int,
        cycle_id: int,
        actual_transfer_time: float,
        cycle_start_time: float | None,
        station_first_cleared_times: Mapping[int, float] | None,
        next_cycle_station_mask: Any | None,
        aircraft_at_stations: Any | None,
    ) -> None:
        """当真实周期转站到达并写入事实时调用。"""
        key = (int(worker_id), int(episode_id), int(cycle_id))
        fact_dict = {
            "worker_id": int(worker_id),
            "episode_id": int(episode_id),
            "cycle_id": int(cycle_id),
            "actual_transfer_time": float(actual_transfer_time),
            "cycle_start_time": float(cycle_start_time) if cycle_start_time is not None else None,
            "station_first_cleared_times": {
                int(k): float(v) for k, v in (station_first_cleared_times or {}).items()
            },
            "next_cycle_station_mask": [
                bool(x) for x in (next_cycle_station_mask or [])
            ],
            "aircraft_at_stations": [
                (None if x is None else int(x)) for x in (aircraft_at_stations or [])
            ],
        }
        self.cycle_transfer_facts[key] = fact_dict

        self.log_event(
            "ATTACH_TRANSFER",
            {
                "worker_id": int(worker_id),
                "episode_id": int(episode_id),
                "cycle_id": int(cycle_id),
                "actual_transfer_time": float(actual_transfer_time),
                "cycle_start_time": fact_dict["cycle_start_time"],
                "station_first_cleared_times": fact_dict["station_first_cleared_times"],
                "next_cycle_station_mask": fact_dict["next_cycle_station_mask"],
            },
        )

    def on_buffer_drain_ready(self, ready_items: list[dict[str, Any]]) -> None:
        """当已就绪的标签样本被取出（drain）时调用。"""
        drained_records: list[dict[str, Any]] = []
        for item in ready_items:
            w = int(item["worker_id"])
            ep = int(item["episode_id"])
            cyc = int(item["cycle_id"])
            dec = int(item["decision_id"])
            sample_id = f"{self.run_id}:{w}:{ep}:{cyc}:{dec}"

            curr_mask = bool(item.get("current_target_mask", True))
            raw_next_masks = item.get("next_cycle_masks", torch.zeros(5, dtype=torch.bool))
            next_masks = [bool(x) for x in raw_next_masks.tolist()]

            target_curr = float(item["target_residual"])
            target_next = [float(x) for x in item["target_next_residuals"].tolist()]

            drain_kind = "both" if (curr_mask and any(next_masks)) else "current_only" if curr_mask else "next_only"

            rec = {
                "sample_id": sample_id,
                "worker_id": w,
                "episode_id": ep,
                "source_cycle_id": cyc,
                "decision_id": dec,
                "drain_kind": drain_kind,
                "current_target_mask": curr_mask,
                "next_cycle_masks": next_masks,
                "target_residual": target_curr,
                "target_next_residuals": target_next,
            }
            drained_records.append(rec)

            if sample_id in self.tracked_history:
                hist = self.tracked_history[sample_id]
                if curr_mask and hist["drain_current"] is None:
                    hist["drain_current"] = {
                        "seq": self.event_seq + 1,
                        "drain_kind": drain_kind,
                        "current_target_mask": curr_mask,
                        "next_cycle_masks": next_masks,
                        "target_residual": target_curr,
                    }
                elif (not curr_mask) and any(next_masks) and hist["drain_next"] is None:
                    hist["drain_next"] = {
                        "seq": self.event_seq + 1,
                        "drain_kind": drain_kind,
                        "current_target_mask": curr_mask,
                        "next_cycle_masks": next_masks,
                        "target_next_residuals": target_next,
                    }

        self.log_event(
            "DRAIN_READY_BATCH",
            {
                "item_count": len(ready_items),
                "items": drained_records,
            },
        )

    # -------------------------------------------------------------------------
    # 训练/优化钩子：rollout pre-fit、post-fit、aux mini-batch
    # -------------------------------------------------------------------------

    def on_pre_rollout_fit(
        self,
        *,
        rollout_id: int,
        online_actor: nn.Module,
        online_head: nn.Module,
        shaper: Any,
        episode_id: int = 0,
        worker_id: int = 0,
        test_snapshot: Any | None = None,
    ) -> None:
        """在单段 rollout 执行 Lightning.fit 前调用：记录冻结预测器与在线参数起点。"""
        if self.frozen_test_snapshot is None and test_snapshot is not None:
            # 捕获一份真实的 CPU 决策快照用于同一输入的重复推理对比
            self.frozen_test_snapshot = copy.deepcopy(test_snapshot)

        try:
            frozen_head, frozen_actor = shaper.get_predictor_snapshot(worker_id, episode_id)
        except (ValueError, KeyError):
            frozen_head, frozen_actor = shaper.get_predictor_snapshot(worker_id, None)
        bound_version = shaper.snapshot_version

        frozen_actor_sha = compute_state_dict_sha256(frozen_actor)
        frozen_head_sha = compute_state_dict_sha256(frozen_head)
        online_actor_sha = compute_state_dict_sha256(online_actor)
        online_head_sha = compute_state_dict_sha256(online_head)

        # 在冻结对象上执行基准推理（确保 eval 与 no_grad）
        ref_inference = self._run_frozen_inference(
            frozen_actor,
            frozen_head,
            self.frozen_test_snapshot,
            shaper,
        )

        self._pre_fit_state = {
            "rollout_id": int(rollout_id),
            "bound_version": int(bound_version),
            "frozen_actor_sha": frozen_actor_sha,
            "frozen_head_sha": frozen_head_sha,
            "online_actor_sha": online_actor_sha,
            "online_head_sha": online_head_sha,
            "online_actor_state": copy.deepcopy(online_actor.state_dict()),
            "online_head_state": copy.deepcopy(online_head.state_dict()),
            "ref_inference": ref_inference,
        }

        self.log_event(
            "PRE_ROLLOUT_FIT",
            {
                "rollout_id": int(rollout_id),
                "bound_version": int(bound_version),
                "frozen_actor_sha": frozen_actor_sha,
                "frozen_head_sha": frozen_head_sha,
                "online_actor_sha": online_actor_sha,
                "online_head_sha": online_head_sha,
                "frozen_actor_eval": not frozen_actor.training if frozen_actor is not None else True,
                "frozen_head_eval": not frozen_head.training if frozen_head is not None else True,
            },
        )

    def on_ppo_update_step(
        self,
        *,
        rollout_id: int,
        loss_dict: dict[str, float],
        grad_norm: float | None,
        online_actor: nn.Module | None,
    ) -> None:
        """PPO 更新步发生时调用。"""
        self.optimization_seq += 1
        opt_id = self.optimization_seq

        # 记录给处于 drain_current 与 drain_next 之间的候选样本
        for sample_id, hist in self.tracked_history.items():
            if hist["drain_current"] is not None and hist["drain_next"] is None:
                hist["online_updates_between"].append(opt_id)

        self.log_event(
            "PPO_OPTIMIZER_UPDATE",
            {
                "opt_seq": opt_id,
                "rollout_id": int(rollout_id),
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
                "losses": {k: float(v) for k, v in loss_dict.items()},
            },
        )

    def on_aux_minibatch_step(
        self,
        *,
        rollout_id: int,
        minibatch: dict[str, Any],
        loss: torch.Tensor,
        grad_norm: float | None,
        time_head: nn.Module,
    ) -> None:
        """时间辅助监督 mini-batch 优化步完成时调用。"""
        self.optimization_seq += 1
        opt_id = self.optimization_seq

        w_list = [int(x) for x in minibatch["worker_ids"]]
        ep_list = [int(x) for x in minibatch["episode_ids"]]
        cyc_list = [int(x) for x in minibatch["cycle_ids"]]
        dec_list = [int(x) for x in minibatch["decision_ids"]]

        minibatch_sample_ids = [
            f"{self.run_id}:{w}:{ep}:{cyc}:{dec}"
            for w, ep, cyc, dec in zip(w_list, ep_list, cyc_list, dec_list)
        ]

        curr_masks = (
            [bool(x) for x in minibatch["current_target_masks"].tolist()]
            if "current_target_masks" in minibatch
            else [True] * len(minibatch_sample_ids)
        )
        next_masks = (
            [ [bool(m) for m in row] for row in minibatch["next_cycle_masks"].tolist() ]
            if "next_cycle_masks" in minibatch
            else [ [False] * 5 for _ in minibatch_sample_ids ]
        )

        # 检查是否包含跟踪的候选样本
        for sample_id in minibatch_sample_ids:
            if sample_id in self.tracked_history:
                hist = self.tracked_history[sample_id]
                idx = minibatch_sample_ids.index(sample_id)
                hist["aux_optimizations"].append({
                    "opt_seq": opt_id,
                    "rollout_id": int(rollout_id),
                    "current_mask": curr_masks[idx],
                    "next_masks": next_masks[idx],
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "loss": float(loss.detach().float().item()),
                })
            # 若处于两次转站中间，该优化也算作一次有效的在线更新
            if sample_id in self.tracked_history:
                hist = self.tracked_history[sample_id]
                if hist["drain_current"] is not None and hist["drain_next"] is None:
                    hist["online_updates_between"].append(opt_id)

        # 检查是否包含下一周期有效掩码且损失非零
        has_next_mask = False
        if "next_cycle_masks" in minibatch:
            mask_t = minibatch["next_cycle_masks"]
            has_next_mask = bool(mask_t.any().item() if torch.is_tensor(mask_t) else any(any(r) for r in mask_t))

        loss_val = float(loss.detach().float().item())
        next_err_max = loss_val if has_next_mask else 0.0

        self.log_event(
            "AUX_MINIBATCH_OPTIMIZER_UPDATE",
            {
                "opt_seq": opt_id,
                "rollout_id": int(rollout_id),
                "batch_size": len(minibatch_sample_ids),
                "sample_ids": minibatch_sample_ids,
                "current_masks": curr_masks,
                "next_masks": next_masks,
                "loss": float(loss.detach().float().item()),
                "grad_norm": float(grad_norm) if grad_norm is not None else None,
                "update_successful": bool(grad_norm is not None and math.isfinite(grad_norm)),
                "next_error_max": next_err_max,
            },
        )

    def on_post_rollout_fit(
        self,
        *,
        rollout_id: int,
        online_actor: nn.Module,
        online_head: nn.Module,
        shaper: Any,
        episode_id: int = 0,
        worker_id: int = 0,
    ) -> None:
        """在单段 rollout 执行 Lightning.fit 后调用：核验冻结预测器不变性与在线参数更新。"""
        if self._pre_fit_state is None:
            return

        try:
            frozen_head, frozen_actor = shaper.get_predictor_snapshot(worker_id, episode_id)
        except (ValueError, KeyError):
            frozen_head, frozen_actor = shaper.get_predictor_snapshot(worker_id, None)
        bound_version = shaper.snapshot_version

        frozen_actor_sha_after = compute_state_dict_sha256(frozen_actor)
        frozen_head_sha_after = compute_state_dict_sha256(frozen_head)

        actor_delta = compute_param_l2_diff(self._pre_fit_state["online_actor_state"], online_actor)
        head_delta = compute_param_l2_diff(self._pre_fit_state["online_head_state"], online_head)

        # 重复测试推理对比
        curr_inference = self._run_frozen_inference(
            frozen_actor,
            frozen_head,
            self.frozen_test_snapshot,
            shaper,
        )

        diff_metrics: dict[str, float] = {}
        if self._pre_fit_state["ref_inference"] is not None and curr_inference is not None:
            ref = self._pre_fit_state["ref_inference"]
            diff_metrics["pred_transfer_diff"] = abs(ref["pred_transfer"] - curr_inference["pred_transfer"])
            diff_metrics["h_estimated_diff"] = abs(ref["h_estimated"] - curr_inference["h_estimated"])
            diff_metrics["phi_diff"] = abs(ref["potential_phi"] - curr_inference["potential_phi"])
            diff_metrics["urgency_diff_max"] = max(
                abs(a - b) for a, b in zip(ref["urgency_12d"], curr_inference["urgency_12d"])
            )
            diff_metrics["next_residuals_diff_max"] = max(
                abs(a - b) for a, b in zip(ref["next_residuals_5d"], curr_inference["next_residuals_5d"])
            )

        check_result = {
            "rollout_id": int(rollout_id),
            "bound_version": int(bound_version),
            "frozen_actor_sha_match": (frozen_actor_sha_after == self._pre_fit_state["frozen_actor_sha"]),
            "frozen_head_sha_match": (frozen_head_sha_after == self._pre_fit_state["frozen_head_sha"]),
            "online_actor_param_delta_l2": actor_delta,
            "online_head_param_delta_l2": head_delta,
            "online_params_changed": bool(actor_delta > 1e-7 or head_delta > 1e-7),
            "inference_diffs": diff_metrics,
            "frozen_eval_mode": (not frozen_head.training) and (frozen_actor is None or not frozen_actor.training),
        }
        self.frozen_eval_checks.append(check_result)

        self.log_event(
            "POST_ROLLOUT_FIT_CHECK",
            check_result,
        )
        self._pre_fit_state = None

    def _run_frozen_inference(
        self,
        frozen_actor: nn.Module | None,
        frozen_head: nn.Module | None,
        snapshot: Any,
        shaper: Any,
    ) -> dict[str, Any] | None:
        """在冻结模型上运行单次推理并返回标量与列表摘要。"""
        if frozen_head is None or snapshot is None:
            return None

        device = next(frozen_head.parameters()).device
        was_head_train = frozen_head.training
        was_actor_train = frozen_actor.training if frozen_actor is not None else False

        frozen_head.eval()
        if frozen_actor is not None:
            frozen_actor.eval()

        with torch.no_grad():
            state_feats = snapshot.state_features.to(device)
            graph_snapshot = snapshot.graph_snapshot
            estimated_cmax = snapshot.estimated_cmax

            if frozen_actor is not None:
                shared_feature = frozen_actor.encode_shared_representation(
                    state_feats,
                    graph_snapshot,
                ).unsqueeze(0)
            else:
                shared_feature = state_feats.unsqueeze(0)

            pred_transfer, remaining, h_estimated = frozen_head.predict_corrected_time(
                state_feat=shared_feature,
                estimated_cmax=estimated_cmax,
                current_time=float(snapshot.current_time),
                h0=float(snapshot.h0),
                last_transfer_time=float(snapshot.last_transfer_time),
            )

            from models.work3.action_fusion import compute_time_urgency_vector
            urgency = compute_time_urgency_vector(
                estimated_r=remaining.squeeze(0),
                current_time=snapshot.current_time,
                last_transfer_time=snapshot.last_transfer_time,
                h0=snapshot.h0,
                device=device,
            )

            if hasattr(frozen_head, "predict_next_cycle_residuals"):
                next_residuals = frozen_head.predict_next_cycle_residuals(shared_feature).squeeze(0)
            else:
                next_residuals = torch.zeros(5, dtype=torch.float32, device=device)

            next_mask = snapshot.next_cycle_station_mask or [False] * 5
            occupancy = torch.tensor(
                [1.0 if m else 0.0 for m in next_mask],
                dtype=torch.float32,
                device=device,
            )
            time_features = torch.cat([urgency, next_residuals, occupancy], dim=-1)

            from models.work3.potential_shaping import compute_potential_scalar_from_h_est
            phi = compute_potential_scalar_from_h_est(
                float(h_estimated.squeeze().item()),
                float(snapshot.h0),
                a=shaper.a,
                b=shaper.b,
            )

        # 恢复原有模式
        frozen_head.train(was_head_train)
        if frozen_actor is not None:
            frozen_actor.train(was_actor_train)

        return {
            "pred_transfer": float(pred_transfer.squeeze().item()),
            "h_estimated": float(h_estimated.squeeze().item()),
            "potential_phi": float(phi),
            "urgency_12d": [float(x) for x in time_features.squeeze().tolist()],
            "next_residuals_5d": [float(x) for x in next_residuals.squeeze().tolist()],
        }

    # -------------------------------------------------------------------------
    # 终结与导出
    # -------------------------------------------------------------------------

    def finalize_and_save(self, run_report: dict[str, Any]) -> dict[str, Any]:
        """训练结束时进行全量证据闭环校验，并写入文件系统。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 验证并构建 R3 证据案例链
        verified_cases: list[dict[str, Any]] = []
        for sample_id in self.tracked_sample_ids:
            hist = self.tracked_history.get(sample_id)
            if hist is None:
                continue

            inp = self.tracked_inputs.get(sample_id)
            if inp is None:
                continue

            # 必须包含完整步骤：
            # 步骤 1: 登记在 cycle 1
            # 步骤 2: 首次取出为 current_only
            # 步骤 3: 两次取出之间有在线参数更新
            # 步骤 4: 再次取出为 next_only
            # 步骤 5: next_cycle_masks.any() == True 且 current_target_mask == False
            # 步骤 6: 参与真实辅助优化且 next_error_max > 0
            d_curr = hist.get("drain_current")
            d_next = hist.get("drain_next")
            updates_between = hist.get("online_updates_between", [])
            aux_opts = hist.get("aux_optimizations", [])

            if not (d_curr and d_next and updates_between and aux_opts):
                continue

            # 数值独立重算
            w = inp["worker_id"]
            ep = inp["episode_id"]
            cyc = inp["source_cycle_id"]
            est_cmax = inp["estimated_cmax"]
            h0 = inp["h0"]

            # 周期 1 转站事实
            fact_1 = self.cycle_transfer_facts.get((w, ep, cyc))
            # 周期 2 转站事实
            fact_2 = self.cycle_transfer_facts.get((w, ep, cyc + 1))

            if not (fact_1 and fact_2):
                continue

            p1 = fact_1["actual_transfer_time"]
            ref_target_curr = (p1 - est_cmax) / h0
            act_target_curr = d_curr["target_residual"]
            curr_diff = abs(act_target_curr - ref_target_curr)
            curr_tol = 1e-5 + 1e-5 * abs(ref_target_curr)

            # 下一周期站位重算
            next_station_cleared = fact_2["station_first_cleared_times"]
            next_station_mask = fact_2["next_cycle_station_mask"]
            ref_next_targets: list[float] = [0.0] * 5
            next_diffs: list[float] = [0.0] * 5

            for m in range(5):
                if m < len(next_station_mask) and next_station_mask[m]:
                    if m in next_station_cleared:
                        f_m = next_station_cleared[m]
                        ref_val = (f_m - p1 - h0) / h0
                        ref_next_targets[m] = ref_val
                        act_val = d_next["target_next_residuals"][m]
                        next_diffs[m] = abs(act_val - ref_val)

            act_next_masks = d_next["next_cycle_masks"]
            max_next_diff = max(next_diffs) if any(act_next_masks) else 0.0

            # 检查是否有 next-only 参与了成功的辅助优化
            next_aux_opts = [
                opt for opt in aux_opts
                if not opt["current_mask"] and any(opt["next_masks"]) and opt["grad_norm"] is not None
            ]

            case_passed = bool(
                d_curr["current_target_mask"] is True
                and not any(d_curr["next_cycle_masks"])
                and d_next["current_target_mask"] is False
                and any(act_next_masks)
                and len(updates_between) >= 1
                and len(next_aux_opts) >= 1
                and curr_diff <= curr_tol
                and max_next_diff <= 1e-5
            )

            verified_cases.append({
                "sample_id": sample_id,
                "passed": case_passed,
                "source_cycle_id": cyc,
                "decision_id": inp["decision_id"],
                "register_seq": hist["register_seq"],
                "drain_current_seq": d_curr["seq"],
                "drain_next_seq": d_next["seq"],
                "online_updates_between_count": len(updates_between),
                "online_update_sample_seqs": updates_between[:5],
                "drained_current_target_mask": d_curr["current_target_mask"],
                "drained_next_cycle_masks": act_next_masks,
                "recomputed_curr_target": ref_target_curr,
                "actual_curr_target": act_target_curr,
                "curr_target_abs_diff": curr_diff,
                "recomputed_next_targets": ref_next_targets,
                "actual_next_targets": d_next["target_next_residuals"],
                "max_next_target_abs_diff": max_next_diff,
                "aux_optimization_count": len(next_aux_opts),
            })

        # 2. 汇总 E04 结果
        e04_passed = False
        if self.frozen_eval_checks:
            all_frozen_matched = all(
                c["frozen_actor_sha_match"] and c["frozen_head_sha_match"] and c["frozen_eval_mode"]
                for c in self.frozen_eval_checks
            )
            all_online_changed = any(c["online_params_changed"] for c in self.frozen_eval_checks)

            max_inf_diff = 0.0
            for c in self.frozen_eval_checks:
                diffs = c.get("inference_diffs", {})
                for v in diffs.values():
                    if isinstance(v, (int, float)) and v > max_inf_diff:
                        max_inf_diff = v

            e04_passed = bool(all_frozen_matched and all_online_changed and max_inf_diff <= 1e-6)

        # 3. 总体判定
        has_passing_r3 = any(c["passed"] for c in verified_cases)
        status = "PASS" if (has_passing_r3 and e04_passed) else "INCOMPLETE" if not verified_cases else "FAIL"

        summary = {
            "status": status,
            "r3_passed": has_passing_r3,
            "e04_passed": e04_passed,
            "run_id": self.run_id,
            "total_audit_events": len(self.events),
            "total_optimization_steps": self.optimization_seq,
            "tracked_candidates_count": len(self.tracked_sample_ids),
            "verified_r3_cases": verified_cases,
            "e04_check_count": len(self.frozen_eval_checks),
            "e04_eval_checks": self.frozen_eval_checks,
            "device": run_report.get("device"),
            "git_commit": run_report.get("git_head", {}).get("stdout", "").strip(),
        }

        # 4. 写入文件
        # 4.1 r3_e04_events.jsonl
        events_path = self.output_dir / "r3_e04_events.jsonl"
        with events_path.open("w", encoding="utf-8") as f:
            for ev in self.events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

        # 4.2 r3_e04_inputs.pt
        inputs_path = self.output_dir / "r3_e04_inputs.pt"
        torch.save(
            {
                "run_id": self.run_id,
                "tracked_inputs": self.tracked_inputs,
                "frozen_test_snapshot": self.frozen_test_snapshot,
            },
            inputs_path,
        )

        # 4.3 r3_e04_summary.json
        summary_path = self.output_dir / "r3_e04_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        return summary
