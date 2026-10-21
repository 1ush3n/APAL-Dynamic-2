"""Work-3 C 基准复现能力诊断：正式接口适配与诊断合同。

本模块只做三件事，不改变环境物理规则、费用权重、候选掩码或网络结构：
1. 把 A 教师决策现场冻结成与 C 完全同源的 CPU 决策快照与正式 sample_record；
2. 复用 ``ActorCriticWork3.evaluate_action_log_probs`` 计算教师动作在正式 C
   分布下的对数概率，并对无序团队按合法选人排列取 logsumexp；
3. 定义只属于本轮诊断的检查点格式（``diagnostic_behavior_clone``）及其严格校验。

团队语义：团队 [12, 72] 与 [72, 12] 是同一物理团队。指针头的自回归结构使两条
选人序列互斥，因此团队事件的概率是各合法排列对数概率的 logsumexp。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import permutations
from pathlib import Path
from typing import Any

import torch

from envs.work3.core_types import ActionBranch
from envs.work3.decision_snapshot import (
    DecisionSnapshot,
    team_completion_worker_ids,
    worker_completion_mask,
)

# ---------------------------------------------------------------------------
# 常量与路径
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = REPO_ROOT / "work3_4060"
RULE_BASELINE_DIR = BASE_DIR / "rule_baseline_ac_v1"
OUTPUT_DIR = BASE_DIR / "c_reproduction_diagnostic_v1" / "seed42"

BASELINE_PATH = RULE_BASELINE_DIR / "mini210_rule_baseline.json"
SCENARIOS_PATH = RULE_BASELINE_DIR / "scenarios.json"
TRAIN_SPLIT_PATH = RULE_BASELINE_DIR / "train.json"
ORIGINAL_C_DIR = RULE_BASELINE_DIR / "seed42" / "C"
ORIGINAL_C_WEIGHTS = ORIGINAL_C_DIR / "weights.pt"
ORIGINAL_C_RUN = ORIGINAL_C_DIR / "run.json"

TEACHER_SNAPSHOT_VERSION = "c_reproduction_teacher_snapshot_v1"
TEACHER_ACTION_SCHEMA_VERSION = "c_reproduction_teacher_action_v1"
DIAGNOSTIC_FORMAT = "diagnostic_behavior_clone"
DIAGNOSTIC_FORMAT_VERSION = 1
METRICS_SCHEMA_VERSION = "c_reproduction_teacher_metrics_v1"

EXPECTED_TASK_COUNT = 210
EXPECTED_AIRCRAFT_COUNT = 7
EXPECTED_STATION_COUNT = 5
EXPECTED_H0 = 52.037844126999545

GPU_BUDGET_BYTES = 6 * 1024**3
DECISION_BUDGET = 4096

# 诊断训练预算（第 2 节预先规定）
TRAIN_MAX_OPTIMIZER_STEPS = 1000
TRAIN_MAX_SECONDS = 1800.0
TRAIN_EFFECTIVE_BATCH = 8
TRAIN_MICRO_BATCH = 1
TRAIN_LEARNING_RATE = 3e-4
TRAIN_WEIGHT_DECAY = 0.0
TRAIN_GRAD_CLIP = 1.0
TRAIN_SEED = 42
SCAN_INTERVAL_STEPS = 50
EARLY_STOP_ACCURACY = 0.99
EARLY_STOP_CONSECUTIVE_SCANS = 2


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_dict_fingerprint(state_dict: Mapping[str, torch.Tensor]) -> str:
    """按名称排序的参数字节指纹；不依赖 pickle，跨进程稳定。"""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"指纹输入必须是张量: {name}")
        contiguous = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def actor_fingerprint(model: torch.nn.Module) -> str:
    return state_dict_fingerprint({k: v for k, v in model.state_dict().items()})


def policy_parameter_names(model: torch.nn.Module) -> tuple[str, ...]:
    """价值专属参数之外的 Actor 参数；共享编码器按设计会随 BC 变化。"""
    return tuple(
        name for name, _ in model.named_parameters() if not name.startswith("critic.")
    )


def value_parameter_names(model: torch.nn.Module) -> tuple[str, ...]:
    return tuple(
        name for name, _ in model.named_parameters() if name.startswith("critic.")
    )


def describe_thread_configuration() -> dict[str, int]:
    return {
        "intra_op_threads": int(torch.get_num_threads()),
        "inter_op_threads": int(torch.get_num_interop_threads()),
    }


def enforce_single_thread_cpu() -> dict[str, int]:
    """显式设置并回读 CPU 线程数；失败必须抛出而不是伪报成功。"""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = describe_thread_configuration()
    if config["intra_op_threads"] != 1 or config["inter_op_threads"] != 1:
        raise RuntimeError(f"CPU 线程数未能设置为 1: {config}")
    return config


def gpu_memory_snapshot() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {"allocated_bytes": 0, "reserved_bytes": 0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


# ---------------------------------------------------------------------------
# C profile 的决策输入（与 FormalEvaluationAgent 的 C 分支逐项一致）
# ---------------------------------------------------------------------------


def build_c_time_features(state: Any, cmax: float, device: str | torch.device = "cpu") -> torch.Tensor:
    """构造 C profile 的 12 维时间特征：2 维紧迫度 + 5 维零残差 + 5 维占用。"""
    from models.work3.action_fusion import compute_time_urgency_vector

    urgency = compute_time_urgency_vector(
        estimated_r=max(0.0, float(cmax) - float(state.current_time)),
        current_time=float(state.current_time),
        last_transfer_time=float(state.last_transfer_time),
        h0=float(state.h0),
        device=device,
    )
    next_residuals = torch.zeros(5, dtype=torch.float32, device=device)
    occupancy = torch.tensor(
        [1.0 if flag else 0.0 for flag in state.get_next_cycle_station_mask()],
        dtype=torch.float32,
        device=device,
    )
    return torch.cat([urgency, next_residuals, occupancy], dim=-1)


def build_c_state_features(state: Any, cmax: float) -> torch.Tensor:
    from models.work3.actor_critic import extract_compact_state_features

    return extract_compact_state_features(state, float(cmax))


# ---------------------------------------------------------------------------
# 教师样本：快照字段、标签与结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TeacherLabel:
    """A 决策在正式 C 动作结构中的标签拆解。"""

    decision_kind: str  # "advance" | "postpone" | "stay"
    task_idx: int | None
    task_key: str | None
    branch: int | None
    team: tuple[int, ...]  # 无序团队（升序）
    align: int | None
    valid_permutations: tuple[tuple[int, ...], ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "decision_kind": self.decision_kind,
            "task_idx": None if self.task_idx is None else int(self.task_idx),
            "task_key": self.task_key,
            "branch": None if self.branch is None else int(self.branch),
            "team": [int(w) for w in self.team],
            "align": None if self.align is None else int(self.align),
            "valid_permutations": [[int(w) for w in order] for order in self.valid_permutations],
        }


def valid_team_permutations(
    snapshot: DecisionSnapshot,
    task_idx: int,
    team: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """返回与正式逐人条件掩码相容的全部选人排列。

    掩码本身只排除“技能/效率不合格”与“已选工人”，因此排列是否合法必须逐步按
    正式规则判定，而不是只比较集合。
    """
    context = snapshot.team_contexts[task_idx]
    wanted = tuple(sorted({int(w) for w in team}))
    if len(wanted) != len(tuple(team)):
        raise ValueError(f"教师团队含重复工人: {tuple(team)}")
    if len(wanted) != int(context.demand):
        raise ValueError(
            f"教师团队人数{len(wanted)}与工序需求{context.demand}不一致: {context.task_key}"
        )
    orders: list[tuple[int, ...]] = []
    for order in permutations(wanted):
        selected: tuple[int, ...] = ()
        accepted = True
        for worker_id in order:
            if worker_id not in team_completion_worker_ids(context, selected):
                accepted = False
                break
            selected = selected + (int(worker_id),)
        if accepted:
            orders.append(order)
    return tuple(orders)


def teacher_label_from_action(
    snapshot: DecisionSnapshot,
    action: Mapping[str, Any],
) -> TeacherLabel:
    """把 A 的原始动作映射为正式 C 的标签结构，并做合法性判定。"""
    branch = ActionBranch(int(action["branch"]))
    if branch == ActionBranch.ADVANCE_TO_NEXT_EVENT:
        return TeacherLabel(
            decision_kind="advance",
            task_idx=None,
            task_key=None,
            branch=None,
            team=(),
            align=None,
            valid_permutations=(),
        )
    task_key = str(action["task_key"])
    if task_key not in snapshot.candidate_task_keys:
        raise ValueError(f"教师动作工序{task_key}不在 C 候选集中")
    task_idx = snapshot.candidate_task_keys.index(task_key)
    can_reserve, can_postpone = snapshot.branch_masks[task_idx]
    branch_value = int(branch)
    if branch_value not in (0, 1):
        raise ValueError(f"未知分支值: {branch_value}")
    allowed = can_reserve if branch_value == 0 else can_postpone
    if not allowed:
        raise ValueError(
            f"教师分支{branch_value}被 C 分支掩码禁止: {task_key} mask={(can_reserve, can_postpone)}"
        )
    if branch_value == int(ActionBranch.POSTPONE):
        return TeacherLabel(
            decision_kind="postpone",
            task_idx=task_idx,
            task_key=task_key,
            branch=branch_value,
            team=(),
            align=0,
            valid_permutations=(),
        )
    team = tuple(int(w) for w in action.get("team", ()))
    orders = valid_team_permutations(snapshot, task_idx, team)
    if not orders:
        raise ValueError(f"教师团队{team}在 C 逐人条件掩码下不可达: {task_key}")
    align = int(action.get("align", 0))
    if align not in (0, 1):
        raise ValueError(f"教师对齐选项越界: {align}")
    return TeacherLabel(
        decision_kind="stay",
        task_idx=task_idx,
        task_key=task_key,
        branch=branch_value,
        team=tuple(sorted(team)),
        align=align,
        valid_permutations=orders,
    )


SNAPSHOT_FIELDS: tuple[str, ...] = (
    "worker_id",
    "episode_id",
    "episode_index",
    "state_features",
    "time_features",
    "graph_snapshot",
    "candidate_task_keys",
    "candidate_task_features",
    "branch_masks",
    "team_contexts",
    "candidate_task_node_indices",
    "worker_node_indices",
    "reserved_flags",
    "advance_available",
    "current_time",
    "cycle_id",
    "estimated_cmax",
    "h0",
    "last_transfer_time",
    "graph_version",
    "candidate_aircraft_ids",
    "candidate_station_ids",
    "next_cycle_station_mask",
)


def snapshot_to_fields(snapshot: DecisionSnapshot) -> dict[str, Any]:
    return {name: getattr(snapshot, name) for name in SNAPSHOT_FIELDS}


def snapshot_from_fields(fields: Mapping[str, Any]) -> DecisionSnapshot:
    return DecisionSnapshot(**{name: fields[name] for name in SNAPSHOT_FIELDS})


def sample_label(sample: Mapping[str, Any]) -> TeacherLabel:
    """读取样本标签；磁盘上存为纯字典，便于 torch.save 与人工核对。"""
    label = sample["label"]
    if isinstance(label, TeacherLabel):
        return label
    return TeacherLabel(
        decision_kind=str(label["decision_kind"]),
        task_idx=None if label.get("task_idx") is None else int(label["task_idx"]),
        task_key=label.get("task_key"),
        branch=None if label.get("branch") is None else int(label["branch"]),
        team=tuple(int(w) for w in label.get("team", ())),
        align=None if label.get("align") is None else int(label["align"]),
        valid_permutations=tuple(
            tuple(int(w) for w in order) for order in label.get("valid_permutations", ())
        ),
    )


def build_teacher_sample(
    *,
    sample_id: int,
    decision_index: int,
    snapshot: DecisionSnapshot,
    label: TeacherLabel,
    teacher_action: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> dict[str, Any]:
    """组装一条 CPU 教师样本；张量均为独立副本，不含环境引用。"""
    return {
        "schema_version": TEACHER_SNAPSHOT_VERSION,
        "sample_id": int(sample_id),
        "decision_index": int(decision_index),
        "snapshot": snapshot_to_fields(snapshot),
        "label": label.as_json(),
        "teacher_action": {
            "branch": int(teacher_action["branch"]),
            "task_key": teacher_action.get("task_key"),
            "team": tuple(int(w) for w in teacher_action.get("team", ())),
            "align": int(teacher_action.get("align", 0)),
        },
        "outcome": dict(outcome),
    }


def sample_snapshot(sample: Mapping[str, Any]) -> DecisionSnapshot:
    return snapshot_from_fields(sample["snapshot"])


def build_sample_record(
    snapshot: DecisionSnapshot,
    label: TeacherLabel,
    worker_order: Sequence[int],
    max_station_workers: int,
) -> dict[str, Any]:
    """构造正式 sample_record；字段契约与 select_snapshot 的 v2 记录保持一致。"""
    common = {
        "sample_record_version": 2,
        "decision_snapshot_id": (
            snapshot.worker_id,
            snapshot.episode_id,
            snapshot.episode_index,
        ),
        "graph_snapshot": snapshot.graph_snapshot,
        "graph_version": snapshot.graph_version,
        "candidate_task_keys": snapshot.candidate_task_keys,
        "candidate_branch_masks": snapshot.branch_masks,
        "candidate_task_node_indices": snapshot.candidate_task_node_indices,
        "candidate_aircraft_ids": snapshot.candidate_aircraft_ids,
        "candidate_station_ids": snapshot.candidate_station_ids,
    }
    if label.decision_kind == "advance":
        return {
            **common,
            "action_type": "advance_to_next_event",
            "advance_available": snapshot.advance_available,
            "worker_node_indices": (),
        }
    assert label.task_idx is not None
    task_idx = int(label.task_idx)
    context = snapshot.team_contexts[task_idx]
    record = {
        **common,
        "action_type": "schedule",
        "advance_available": snapshot.advance_available,
        "task_idx": task_idx,
        "task_key": snapshot.candidate_task_keys[task_idx],
        "branch": int(label.branch),
        "cand_feats": snapshot.candidate_task_features,
        "can_reserve": bool(snapshot.branch_masks[task_idx][0]),
        "can_postpone": bool(snapshot.branch_masks[task_idx][1]),
        "num_st_workers": len(context.workers),
        "chosen_team": (),
        "chosen_worker_indices": (),
        "worker_valid_masks": (),
        "align": 0,
        "worker_node_indices": (),
    }
    if int(label.branch) == int(ActionBranch.POSTPONE):
        return record

    worker_id_to_index = {worker.worker_id: index for index, worker in enumerate(context.workers)}
    order = tuple(int(w) for w in worker_order)
    if order not in label.valid_permutations:
        raise ValueError(f"选人排列{order}不在已核验的合法排列集合中: {label.task_key}")
    indices: list[int] = []
    masks: list[tuple[bool, ...]] = []
    selected: tuple[int, ...] = ()
    for worker_id in order:
        masks.append(
            worker_completion_mask(context, selected, max_station_workers)
        )
        indices.append(int(worker_id_to_index[worker_id]))
        selected = selected + (worker_id,)
    record.update(
        {
            "chosen_team": order,
            "chosen_worker_indices": tuple(indices),
            "worker_valid_masks": tuple(masks),
            "align": int(label.align if label.align is not None else 0),
            "worker_node_indices": snapshot.worker_node_indices[task_idx],
        }
    )
    return record


def sample_records(
    sample: Mapping[str, Any],
    max_station_workers: int,
) -> list[dict[str, Any]]:
    """返回该样本全部合法选人排列对应的正式记录（同一条快照、同一图对象）。"""
    snapshot = sample_snapshot(sample)
    label = sample_label(sample)
    if label.decision_kind != "stay":
        return [build_sample_record(snapshot, label, (), max_station_workers)]
    return [
        build_sample_record(snapshot, label, order, max_station_workers)
        for order in label.valid_permutations
    ]


# ---------------------------------------------------------------------------
# 复用正式可微重放接口
# ---------------------------------------------------------------------------


class _EncodeOnce:
    """在一次重放调用内复用同一图快照的编码结果。

    ``evaluate_action_log_probs`` 逐条记录编码；无序团队的多个排列共享同一图对象，
    若重复编码只会得到逐位相同的张量。这里只消除重复计算，不改变任何数值：调用方
    必须保证同一次作用域内只喂入同一快照的记录。
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self._original: Any = None
        self._cache: dict[int, Any] = {}

    def __enter__(self) -> "_EncodeOnce":
        model = self.model
        self._original = model._encode_state_components
        original = self._original

        def wrapper(state_feat: torch.Tensor, time_urgency: torch.Tensor, graph_data: Any = None):
            if graph_data is None:
                return original(state_feat, time_urgency, graph_data)
            key = id(graph_data)
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            result = original(state_feat, time_urgency, graph_data)
            self._cache[key] = result
            return result

        object.__setattr__(model, "_encode_state_components", wrapper)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._cache.clear()
        if self._original is not None:
            object.__setattr__(self.model, "_encode_state_components", self._original)


def teacher_marginal_log_prob(
    model: torch.nn.Module,
    sample: Mapping[str, Any],
    *,
    share_encoding: bool = True,
) -> torch.Tensor:
    """教师物理动作在正式 C 分布下的对数概率（无序团队按排列取 logsumexp）。"""
    snapshot = sample_snapshot(sample)
    max_workers = int(getattr(model, "max_station_workers", 16))
    records = sample_records(sample, max_workers)
    state_feats = snapshot.state_features.unsqueeze(0).expand(len(records), -1).contiguous()
    time_features = snapshot.time_features.unsqueeze(0).expand(len(records), -1).contiguous()
    context = _EncodeOnce(model) if share_encoding else None
    if context is None:
        _values, log_probs, _entropies = model.evaluate_action_log_probs(
            state_feats, time_features, records
        )
    else:
        with context:
            _values, log_probs, _entropies = model.evaluate_action_log_probs(
                state_feats, time_features, records
            )
    if len(records) == 1:
        return log_probs[0]
    return torch.logsumexp(log_probs, dim=0)


# ---------------------------------------------------------------------------
# 判读：独立贪婪动作与教师前缀条件预测
# ---------------------------------------------------------------------------


def _project_snapshot(
    snapshot: DecisionSnapshot,
    *,
    task_idx: int | None = None,
    advance_available: bool = False,
) -> DecisionSnapshot:
    """产出一个诊断用投影快照：只用于“教师前缀条件判读”，不参与闭环执行。"""
    if task_idx is None:
        return replace(snapshot, advance_available=bool(advance_available))
    if task_idx < 0 or task_idx >= len(snapshot.candidate_task_keys):
        raise IndexError(f"候选索引越界: {task_idx}")
    return replace(
        snapshot,
        advance_available=bool(advance_available),
        candidate_task_keys=(snapshot.candidate_task_keys[task_idx],),
        candidate_task_features=snapshot.candidate_task_features[task_idx : task_idx + 1].clone(),
        branch_masks=(snapshot.branch_masks[task_idx],),
        team_contexts=(snapshot.team_contexts[task_idx],),
        candidate_task_node_indices=(snapshot.candidate_task_node_indices[task_idx],),
        worker_node_indices=(snapshot.worker_node_indices[task_idx],),
        reserved_flags=(snapshot.reserved_flags[task_idx],),
        candidate_aircraft_ids=(
            (snapshot.candidate_aircraft_ids[task_idx],)
            if snapshot.candidate_aircraft_ids
            else ()
        ),
        candidate_station_ids=(
            (snapshot.candidate_station_ids[task_idx],)
            if snapshot.candidate_station_ids
            else ()
        ),
    )


def greedy_action(model: torch.nn.Module, snapshot: DecisionSnapshot) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """正式贪心入口：返回 (动作, sample_record)。"""
    action, _log_prob, _value, record = model.select_snapshot(snapshot, deterministic=True)
    return action, record


def action_signature(action: Mapping[str, Any] | None) -> tuple[Any, ...]:
    """物理动作的规范化比较签名；团队按集合比较，顺序不参与判定。"""
    if action is None:
        return ("none",)
    branch = ActionBranch(int(action["branch"]))
    if branch == ActionBranch.ADVANCE_TO_NEXT_EVENT:
        return ("advance",)
    if branch == ActionBranch.POSTPONE:
        return ("postpone", str(action["task_key"]))
    return (
        "stay",
        str(action["task_key"]),
        tuple(sorted(int(w) for w in action.get("team", ()))),
        int(action.get("align", 0)),
    )


def teacher_signature(sample: Mapping[str, Any]) -> tuple[Any, ...]:
    label = sample_label(sample)
    if label.decision_kind == "advance":
        return ("advance",)
    if label.decision_kind == "postpone":
        return ("postpone", str(label.task_key))
    return ("stay", str(label.task_key), tuple(sorted(label.team)), int(label.align or 0))


def evaluate_teacher_sample(
    model: torch.nn.Module,
    sample: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """单样本判读：NLL、独立贪婪动作、教师前缀条件头预测。"""
    snapshot = sample_snapshot(sample)
    label = sample_label(sample)
    with torch.no_grad():
        log_prob = teacher_marginal_log_prob(model, sample)
        nll = float(-log_prob.item())
        if not math.isfinite(nll):
            raise FloatingPointError(f"样本{sample['sample_id']}的教师NLL非有限: {nll}")

        open_action, open_record = greedy_action(model, snapshot)
        open_signature = action_signature(open_action)
        teacher_sig = teacher_signature(sample)
        greedy_kind = open_signature[0]

        conditional: dict[str, Any] = {
            "advance_choice": _advance_choice_from(open_record),
        }
        task_hit: bool | None = None
        if label.decision_kind == "stay":
            if greedy_kind == "advance" or greedy_kind == "none":
                # 贪婪先推进时，工序头未被执行；用“禁用推进门控”的教师前缀投影判读。
                projected = _project_snapshot(snapshot, advance_available=False)
                projected_action, _ = greedy_action(model, projected)
                task_hit = (
                    projected_action is not None
                    and action_signature(projected_action)[0] != "advance"
                    and str(projected_action.get("task_key")) == str(label.task_key)
                )
            else:
                task_hit = str(open_action.get("task_key")) == str(label.task_key)
            # 教师工序前缀下的分支/团队/对齐条件判读
            projected_task = _project_snapshot(snapshot, task_idx=int(label.task_idx))
            head_action, _ = greedy_action(model, projected_task)
            conditional.update(
                {
                    "branch": int(head_action["branch"]) if head_action else None,
                    "team": tuple(sorted(int(w) for w in head_action.get("team", ())))
                    if head_action
                    else (),
                    "align": int(head_action.get("align", 0)) if head_action else None,
                    "branch_hit": int(head_action["branch"]) == int(label.branch)
                    if head_action
                    else False,
                    "team_hit": tuple(sorted(int(w) for w in head_action.get("team", ())))
                    == tuple(sorted(label.team))
                    if head_action
                    else False,
                    "align_hit": int(head_action.get("align", 0)) == int(label.align or 0)
                    if head_action
                    else False,
                }
            )
            conditional["task_hit"] = bool(task_hit)
        else:
            conditional["task_hit"] = None
            conditional["branch_hit"] = None
            conditional["team_hit"] = None
            conditional["align_hit"] = None

    return {
        "sample_id": int(sample["sample_id"]),
        "decision_index": int(sample["decision_index"]),
        "sim_time": float(snapshot.current_time),
        "cycle": int(snapshot.cycle_id),
        "teacher_signature": teacher_sig,
        "greedy_signature": open_signature,
        "greedy_task_key": None if open_action is None else open_action.get("task_key"),
        "greedy_branch": None if open_action is None else int(open_action["branch"]),
        "greedy_team": ()
        if open_action is None
        else tuple(sorted(int(w) for w in open_action.get("team", ()))),
        "greedy_align": None if open_action is None else int(open_action.get("align", 0)),
        "nll": nll,
        "decision_kind_hit": greedy_kind == teacher_sig[0],
        "full_action_hit": open_signature == teacher_sig,
        "task_hit": conditional.get("task_hit"),
        "branch_hit": conditional.get("branch_hit"),
        "team_hit": conditional.get("team_hit"),
        "align_hit": conditional.get("align_hit"),
        "advance_choice": conditional.get("advance_choice"),
        "teacher_branch": None if label.branch is None else int(label.branch),
        "teacher_team": tuple(sorted(label.team)),
        "teacher_task_key": label.task_key,
        "candidate_count": len(snapshot.candidate_task_keys),
        "advance_available": bool(snapshot.advance_available),
    }


def _advance_choice_from(record: Mapping[str, Any]) -> int | None:
    """从正式采样记录还原推进门控的贪婪选择；无门控时返回 None。"""
    action_type = record.get("action_type")
    if action_type in ("advance_to_next_event", "schedule", "postpone", "station_execute"):
        if not record.get("advance_available", False):
            return None
        return int(record.get("advance_choice", 0)) if action_type != "advance_to_next_event" else 1
    return None


def aggregate_teacher_metrics(
    per_sample: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """把逐样本判读汇总为教师状态内评分（条件准确率单独标注）。"""
    counts = len(per_sample)
    if counts == 0:
        raise ValueError("教师状态评分为空")

    def _rate(key: str) -> float | None:
        values = [item[key] for item in per_sample if item.get(key) is not None]
        if not values:
            return None
        return float(sum(bool(v) for v in values) / len(values))

    def _count(key: str) -> int:
        return sum(1 for item in per_sample if item.get(key) is not None)

    stay_samples = [item for item in per_sample if item["teacher_signature"][0] == "stay"]
    feature_counts: dict[str, int] = {}
    for item in per_sample:
        key = str(item["teacher_signature"][0])
        feature_counts[key] = feature_counts.get(key, 0) + 1

    first_full = next(
        (item for item in per_sample if not item["full_action_hit"]),
        None,
    )
    first_task = next(
        (item for item in stay_samples if item.get("task_hit") is False),
        None,
    )
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "sample_count": counts,
        "teacher_decision_kind_counts": feature_counts,
        "nll_mean": float(sum(item["nll"] for item in per_sample) / counts),
        "nll_finite": all(math.isfinite(item["nll"]) for item in per_sample),
        "decision_kind_accuracy": _rate("decision_kind_hit"),
        "task_accuracy_conditional": _rate("task_hit"),
        "branch_accuracy_conditional": _rate("branch_hit"),
        "team_accuracy_unordered_conditional": _rate("team_hit"),
        "align_accuracy_conditional": _rate("align_hit"),
        "full_physical_action_accuracy": _rate("full_action_hit"),
        "conditional_denominators": {
            "task": _count("task_hit"),
            "branch": _count("branch_hit"),
            "team": _count("team_hit"),
            "align": _count("align_hit"),
        },
        "accuracy_scopes": {
            "task_accuracy_conditional": "教师推进/不推进前缀固定为不推进后的工序头；非独立贪婪全流程",
            "branch_accuracy_conditional": "教师工序前缀固定的分支头；非独立贪婪全流程",
            "team_accuracy_unordered_conditional": "教师工序前缀固定的指针头，按无序团队集合比较",
            "align_accuracy_conditional": "教师工序与团队前缀固定的对齐头",
            "full_physical_action_accuracy": "正式贪心入口的完整物理动作元组（含推进门控）",
        },
        "first_full_action_divergence": None
        if first_full is None
        else {
            "sample_id": int(first_full["sample_id"]),
            "decision_index": int(first_full["decision_index"]),
            "sim_time": float(first_full["sim_time"]),
            "cycle": int(first_full["cycle"]),
            "teacher": _jsonable_signature(first_full["teacher_signature"]),
            "greedy": _jsonable_signature(first_full["greedy_signature"]),
        },
        "first_task_divergence": None
        if first_task is None
        else {
            "sample_id": int(first_task["sample_id"]),
            "decision_index": int(first_task["decision_index"]),
            "sim_time": float(first_task["sim_time"]),
            "cycle": int(first_task["cycle"]),
            "teacher_task": first_task["teacher_task_key"],
            "greedy_task": first_task["greedy_task_key"],
        },
    }


def _jsonable_signature(signature: Sequence[Any]) -> list[Any]:
    return [list(item) if isinstance(item, tuple) else item for item in signature]


def scan_teacher_states(
    model: torch.nn.Module,
    samples: Sequence[Mapping[str, Any]],
    *,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """在全部教师状态上做一次无梯度评分。"""
    model.eval()
    per_sample = [
        evaluate_teacher_sample(model, sample, device=device) for sample in samples
    ]
    return aggregate_teacher_metrics(per_sample), per_sample


# ---------------------------------------------------------------------------
# 诊断检查点：只属于本轮，不与正式 PPO 检查点混用
# ---------------------------------------------------------------------------


def build_diagnostic_checkpoint(
    model: torch.nn.Module,
    *,
    source_weights_path: Path,
    source_weights_sha256: str,
    source_actor_fingerprint: str,
    teacher_manifest_sha256: str,
    training_summary: Mapping[str, Any],
    snapshot_contract: Mapping[str, Any],
) -> dict[str, Any]:
    from models.work3.graph_builder import (
        GRAPH_FEATURE_DIMS,
        GRAPH_FEATURE_SCHEMA,
        GRAPH_FEATURE_VERSION,
    )
    from scripts.work3.experiment_protocol import build_method_profile

    state = {name: tensor.detach().to(device="cpu").clone() for name, tensor in model.state_dict().items()}
    return {
        "diagnostic_format": DIAGNOSTIC_FORMAT,
        "diagnostic_format_version": DIAGNOSTIC_FORMAT_VERSION,
        "format_role": "diagnostic_behavior_clone_weights",
        "actor_critic_state": state,
        "actor_fingerprint": state_dict_fingerprint(state),
        "method_profile": asdict(build_method_profile("C")),
        "method_variant": "C",
        "graph_feature_version": GRAPH_FEATURE_VERSION,
        "graph_feature_dims": dict(GRAPH_FEATURE_DIMS),
        "graph_feature_schema_hash": GRAPH_FEATURE_SCHEMA["schema_fingerprint"],
        "hyperparameters": {
            "state_dim": int(getattr(model, "state_dim", 32)),
            "task_feat_dim": int(getattr(model, "task_feat_dim", 8)),
            "hidden_dim": int(getattr(model, "hidden_dim", 64)),
            "max_station_workers": int(getattr(model, "max_station_workers", 16)),
        },
        "source_checkpoint": {
            "path": str(source_weights_path),
            "sha256": source_weights_sha256,
            "actor_fingerprint": source_actor_fingerprint,
        },
        "teacher_manifest_sha256": teacher_manifest_sha256,
        "snapshot_contract": dict(snapshot_contract),
        "training": dict(training_summary),
        "declaration": (
            "本文件是行为克隆诊断副本，不是标准 PPO 正式检查点；"
            "只能由诊断加载路径读取，不得冒充方法 C 的正式权重。"
        ),
    }


def load_diagnostic_actor(
    checkpoint_path: str | Path,
    *,
    expected_teacher_manifest_sha256: str | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """严格校验本轮自定义诊断格式后构造 Actor；任何偏差直接拒绝。"""
    from models.work3.actor_critic import ActorCriticWork3
    from models.work3.graph_builder import (
        GRAPH_FEATURE_DIMS,
        GRAPH_FEATURE_SCHEMA,
        GRAPH_FEATURE_VERSION,
    )
    from scripts.work3.experiment_protocol import build_method_profile

    path = Path(checkpoint_path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("诊断检查点必须是字典")
    if checkpoint.get("diagnostic_format") != DIAGNOSTIC_FORMAT:
        raise ValueError(
            f"诊断加载路径拒绝非诊断格式: {checkpoint.get('diagnostic_format')!r}"
        )
    if int(checkpoint.get("diagnostic_format_version", -1)) != DIAGNOSTIC_FORMAT_VERSION:
        raise ValueError("诊断检查点格式版本不匹配")
    if checkpoint.get("format_role") != "diagnostic_behavior_clone_weights":
        raise ValueError("诊断检查点角色标记不匹配")
    if checkpoint.get("method_profile") != asdict(build_method_profile("C")):
        raise ValueError("诊断检查点方法 profile 不是 C")
    if checkpoint.get("graph_feature_version") != GRAPH_FEATURE_VERSION:
        raise ValueError("诊断检查点图特征版本不匹配")
    if dict(checkpoint.get("graph_feature_dims") or {}) != dict(GRAPH_FEATURE_DIMS):
        raise ValueError("诊断检查点图特征维度不匹配")
    if checkpoint.get("graph_feature_schema_hash") != GRAPH_FEATURE_SCHEMA["schema_fingerprint"]:
        raise ValueError("诊断检查点图特征 schema 指纹不匹配")
    if (
        expected_teacher_manifest_sha256 is not None
        and checkpoint.get("teacher_manifest_sha256") != expected_teacher_manifest_sha256
    ):
        raise ValueError("诊断检查点绑定的教师样本指纹与本轮不一致")
    hyper = checkpoint.get("hyperparameters") or {}
    model = ActorCriticWork3(
        state_dim=int(hyper.get("state_dim", 32)),
        task_feat_dim=int(hyper.get("task_feat_dim", 8)),
        hidden_dim=int(hyper.get("hidden_dim", 64)),
        max_station_workers=int(hyper.get("max_station_workers", 16)),
    )
    state = checkpoint.get("actor_critic_state")
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise ValueError("诊断检查点 Actor 参数集合不匹配")
    for name, tensor in model.state_dict().items():
        if not isinstance(state[name], torch.Tensor) or state[name].shape != tensor.shape:
            raise ValueError(f"诊断检查点 Actor 参数维度不匹配: {name}")
    fingerprint = state_dict_fingerprint(state)
    if fingerprint != checkpoint.get("actor_fingerprint"):
        raise ValueError("诊断检查点 Actor 指纹与权重内容不一致")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# 轨迹事实的小工具
# ---------------------------------------------------------------------------


def extract_execution_facts(env: Any) -> dict[str, Any]:
    """与正式 A/C 导出完全一致的执行事实投影（字段逐一复刻）。"""
    tasks_execution = {
        task.task_key: {
            "task_key": task.task_key,
            "aircraft_id": task.aircraft_id,
            "task_id": task.task_id,
            "station_id": task.current_station,
            "team": list(task.assigned_team),
            "actual_start": float(task.actual_start) if task.actual_start is not None else None,
            "actual_end": float(task.actual_end) if task.actual_end is not None else None,
            "cycle_start_time": float(task.cycle_start_time)
            if getattr(task, "cycle_start_time", None) is not None
            else None,
            "duration": float(task.execution_duration)
            if task.execution_duration is not None
            else float(task.standard_duration),
            "status": task.status.name,
            "postpone_count": task.postpone_count,
            "revision_history": task.revision_history,
        }
        for task in env.state.tasks.values()
    }
    aircraft_execution = {
        str(key): {
            "aircraft_id": aircraft.aircraft_id,
            "entry_times": {str(station): float(t) for station, t in aircraft.entry_times.items()},
            "exit_times": {str(station): float(t) for station, t in aircraft.exit_times.items()},
        }
        for key, aircraft in env.state.aircraft.items()
    }
    cycle_execution_records = {
        str(cycle): record.to_dict()
        for cycle, record in env.state.cycle_execution_records.items()
    }
    return {
        "tasks_execution": tasks_execution,
        "aircraft_execution": aircraft_execution,
        "cycle_execution_records": cycle_execution_records,
        "step_rewards": [float(value) for value in env.step_rewards],
    }


def compare_execution_facts(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """比较两份执行事实，区分“动作顺序差异”与“真实排程偏离”。"""
    left_tasks = left.get("tasks_execution", {})
    right_tasks = right.get("tasks_execution", {})
    shared = sorted(set(left_tasks) & set(right_tasks))
    start_diff: list[dict[str, Any]] = []
    team_diff: list[dict[str, Any]] = []
    end_diff: list[dict[str, Any]] = []
    for key in shared:
        a = left_tasks[key]
        b = right_tasks[key]
        if a.get("actual_start") is None or b.get("actual_start") is None:
            continue
        if abs(float(a["actual_start"]) - float(b["actual_start"])) > tolerance:
            start_diff.append(
                {
                    "task_key": key,
                    "left_start": float(a["actual_start"]),
                    "right_start": float(b["actual_start"]),
                    "delta": float(b["actual_start"]) - float(a["actual_start"]),
                }
            )
        if abs(float(a["actual_end"]) - float(b["actual_end"])) > tolerance:
            end_diff.append(
                {
                    "task_key": key,
                    "left_end": float(a["actual_end"]),
                    "right_end": float(b["actual_end"]),
                }
            )
        if sorted(int(w) for w in a.get("team", ())) != sorted(int(w) for w in b.get("team", ())):
            team_diff.append(
                {
                    "task_key": key,
                    "left_team": [int(w) for w in a.get("team", ())],
                    "right_team": [int(w) for w in b.get("team", ())],
                }
            )
    left_cycles = left.get("cycle_execution_records", {})
    right_cycles = right.get("cycle_execution_records", {})
    transfer_diff: list[dict[str, Any]] = []
    for cycle in sorted(set(left_cycles) & set(right_cycles), key=int):
        lt = float(left_cycles[cycle]["transfer_time"])
        rt = float(right_cycles[cycle]["transfer_time"])
        if abs(lt - rt) > tolerance:
            transfer_diff.append(
                {"cycle": int(cycle), "left_transfer": lt, "right_transfer": rt, "delta": rt - lt}
            )
    return {
        "compared_task_count": len(shared),
        "start_deviation_count": len(start_diff),
        "end_deviation_count": len(end_diff),
        "team_deviation_count": len(team_diff),
        "transfer_deviation_count": len(transfer_diff),
        "first_start_deviation": start_diff[0] if start_diff else None,
        "first_team_deviation": team_diff[0] if team_diff else None,
        "first_transfer_deviation": transfer_diff[0] if transfer_diff else None,
        "task_start_deviations": start_diff[:40],
        "team_deviations": team_diff[:40],
        "transfer_deviations": transfer_diff,
        "schedule_identical": not start_diff and not team_diff,
    }


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def dump_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def clone_actor_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device="cpu").clone()
        for name, tensor in model.state_dict().items()
    }


def shuffled_cyclic_stream(
    total_samples: int,
    *,
    seed: int,
) -> Iterator[int]:
    """无限循环的固定种子洗牌访问流。"""
    generator = torch.Generator().manual_seed(int(seed))
    while True:
        for index in torch.randperm(total_samples, generator=generator).tolist():
            yield int(index)


__all__ = [
    "BASE_DIR",
    "BASELINE_PATH",
    "DECISION_BUDGET",
    "DIAGNOSTIC_FORMAT",
    "DIAGNOSTIC_FORMAT_VERSION",
    "EARLY_STOP_ACCURACY",
    "EARLY_STOP_CONSECUTIVE_SCANS",
    "EXPECTED_H0",
    "GPU_BUDGET_BYTES",
    "ORIGINAL_C_DIR",
    "ORIGINAL_C_RUN",
    "ORIGINAL_C_WEIGHTS",
    "OUTPUT_DIR",
    "REPO_ROOT",
    "RULE_BASELINE_DIR",
    "SCAN_INTERVAL_STEPS",
    "SCENARIOS_PATH",
    "SNAPSHOT_FIELDS",
    "TRAIN_EFFECTIVE_BATCH",
    "TRAIN_GRAD_CLIP",
    "TRAIN_LEARNING_RATE",
    "TRAIN_MAX_OPTIMIZER_STEPS",
    "TRAIN_MAX_SECONDS",
    "TRAIN_MICRO_BATCH",
    "TRAIN_SEED",
    "TRAIN_SPLIT_PATH",
    "TRAIN_WEIGHT_DECAY",
    "TEACHER_ACTION_SCHEMA_VERSION",
    "TEACHER_SNAPSHOT_VERSION",
    "TeacherLabel",
    "action_signature",
    "actor_fingerprint",
    "aggregate_teacher_metrics",
    "build_c_state_features",
    "build_c_time_features",
    "build_diagnostic_checkpoint",
    "build_sample_record",
    "build_teacher_sample",
    "clone_actor_state",
    "compare_execution_facts",
    "describe_thread_configuration",
    "dump_json",
    "dump_jsonl",
    "enforce_single_thread_cpu",
    "evaluate_teacher_sample",
    "extract_execution_facts",
    "greedy_action",
    "gpu_memory_snapshot",
    "load_diagnostic_actor",
    "load_json",
    "policy_parameter_names",
    "sample_label",
    "sample_records",
    "sample_snapshot",
    "scan_teacher_states",
    "sha256_file",
    "sha256_text",
    "shuffled_cyclic_stream",
    "snapshot_from_fields",
    "snapshot_to_fields",
    "state_dict_fingerprint",
    "teacher_label_from_action",
    "teacher_marginal_log_prob",
    "teacher_signature",
    "valid_team_permutations",
    "value_parameter_names",
]
