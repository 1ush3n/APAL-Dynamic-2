"""从当前原生见证导出CPU事实和真实教师前缀，不复用研究包中的旧隐向量。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from apal.data.work3.certified_examples import load_witness_actions, make_instance_env, verify_certificate
from apal.envs.work3.decision_snapshot import DecisionSnapshot, worker_completion_mask
from apal.models.work3.actor_critic import ActorCriticWork3, TEAM_SET_CACHE_VERSION
from apal.training.work3.runtime_config import load_work3_episode_manifest, seed_work3_runtime


def build_team_set_cache(snapshot: DecisionSnapshot, record: dict[str, Any], width: int) -> dict[str, Any]:
    """CPU采样阶段冻结全部真子集条件；不保存任何神经表示。"""
    if record.get("action_type") != "schedule" or int(record["branch"]) != 0:
        raise ValueError("团队子集缓存仅适用于实际留站团队")
    target = tuple(sorted(record["chosen_worker_indices"]))
    if not 1 <= len(target) <= 5 or len(set(target)) != len(target):
        raise ValueError("团队集合监督支持一至五人的合法团队")
    index = int(record["task_idx"])
    context, table = snapshot.team_contexts[index], snapshot.query_tables[index]
    worker_ids = tuple(record["station_worker_ids"])
    if worker_ids != tuple(worker.worker_id for worker in context.workers):
        raise ValueError("团队子集缓存的站内工人顺序不匹配")
    entries = []
    for mask in range((1 << len(target)) - 1):
        selected = tuple(target[bit] for bit in range(len(target)) if mask & (1 << bit))
        prefix = tuple(worker_ids[index] for index in selected)
        entries.append({"selected_worker_indices": selected,
                        "worker_valid_mask": worker_completion_mask(context, prefix, width),
                        "query_worker_features": table.worker_features(worker_ids, prefix,
                            snapshot.current_time, snapshot.h0, width)})
    return {"version": TEAM_SET_CACHE_VERSION, "target_worker_indices": target, "subsets": tuple(entries)}


def export_demonstrations(manifest: Path, output: Path, *, split: str,
                          actor: ActorCriticWork3 | None = None) -> dict[str, Any]:
    """按已冻结来源导出教师记录；私有答案仅用于选动作，不传给环境或图构建器。"""
    if output.exists():
        raise FileExistsError(output)
    actor = actor or ActorCriticWork3(condition_queries=True)
    if not actor.condition_queries or next(actor.parameters()).device.type != "cpu":
        raise ValueError("教师导出要求CPU QUERY网络；查询只能发生在采样前处理阶段")
    instances = load_work3_episode_manifest(manifest, split=split)
    records: list[dict[str, Any]] = []
    layers: Counter[str] = Counter()
    choices: Counter[str] = Counter()
    started = time.monotonic()
    instance_results = []
    actor.eval()
    for instance in instances:
        case = Path(instance["baseline"]).parent.parent
        expected_paths = {"baseline": "baseline.json", "raw_data": "tasks.csv", "worker_pool": "workers.csv"}
        if any(Path(instance[key]).resolve() != (case / "public" / filename).resolve()
               for key, filename in expected_paths.items()):
            raise ValueError("实例清单文件不属于同一证书现场")
        certificate = verify_certificate(case)
        env = make_instance_env(case)
        actual_config = {"max_slots_per_station": env.max_slots_per_station,
                         "transfer_policy": env.transfer_policy, "tolerance": env.tolerance}
        if any(value != actual_config[key] for key, value in instance["env_config"].items()):
            raise ValueError("实例清单条件与已核证书现场不一致")
        scenarios = []
        for name in ("scenario.json", "scenarios.json"):
            path = case / "public" / name
            if path.exists():
                value = json.loads(path.read_text(encoding="utf-8"))
                scenarios.extend(value if isinstance(value, list) else [value])
        if scenarios != ([] if instance["scenario"] is None else [instance["scenario"]]):
            raise ValueError("实例清单事件与已核证书现场不一致")
        witness = load_witness_actions(case)
        for row in witness:
            snapshot = actor.make_decision_snapshot(env, torch.zeros(32), torch.zeros(2))
            action, _, _, record = actor.select_snapshot(snapshot, teacher_action=row["action"])
            present: set[str] = set()
            available: set[str] = set()
            if record["action_type"] != "forced_advance":
                if snapshot.advance_available:
                    present.add("advance")
                    available.add("advance")
                if record["action_type"] == "schedule":
                    present.update(("task", "branch"))
                    if len(snapshot.candidate_task_keys) > 1:
                        available.add("task")
                    if sum(snapshot.branch_masks[record["task_idx"]]) > 1:
                        available.add("branch")
                    if int(record["branch"]) == 0:
                        record["team_set_cache"] = build_team_set_cache(snapshot, record, actor.max_station_workers)
                        present.update(("team", "align"))
                        available.add("align")
                        if len(snapshot.query_tables[record["task_idx"]].entries) > 1:
                            available.add("team")
            scope = tuple(layer for layer in row["label_scope"] if layer in present)
            choice = tuple(layer for layer in scope if layer in available)
            records.append({"instance_id": instance["instance_id"], "parent_group": instance["parent_group"],
                            "family": instance["family"], "state_features": snapshot.state_features,
                            "time_features": snapshot.time_features, "record": record,
                            "label_scope": scope, "choice_layers": choice})
            layers.update(scope)
            choices.update(choice)
            _, _, terminated, truncated, info = env.step(action)
            if truncated or (terminated and not info.get("success")):
                raise ValueError(f"教师记录未通过当前环境: {instance['instance_id']} {info}")
        if not env._check_terminated() or not math.isclose(
            env.state.current_time, certificate["certified_C_exit"], rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("教师导出未完整达到已核下界")
        instance_results.append({"instance_id": instance["instance_id"], "decisions": len(witness),
                                 "C_exit": env.state.current_time, "C_task": max(
                                     task.actual_end for task in env.state.tasks.values())})
    summary = {"instances": len(instances), "records": len(records),
               "supervised_records": sum(bool(row["label_scope"]) for row in records),
               "choice_records": sum(bool(row["choice_layers"]) for row in records),
               "supervised_layers": dict(layers), "choice_layers": dict(choices),
               "elapsed_seconds": time.monotonic() - started, "instance_results": instance_results,
               "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
               "export_process_seed": torch.initial_seed(),
               "team_supervision": "已证物理团队的全部合法人员顺序概率之和",
               "team_objective": "set"}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": "work3_bc_records_v1", "record_semantics_version": "work3_autoregressive_layers_v1",
                "split": "validation" if split == "val" else split,
                "team_objective": "set",
                "policy_input_semantics": actor.input_semantics(), "records": records, "summary": summary}, output)
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="在当前QUERY网络上导出原生教师事实记录")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    seed_work3_runtime(args.seed, deterministic=True)
    torch.set_num_threads(1)
    summary = export_demonstrations(args.manifest, args.output, split=args.split)
    print(json.dumps({key: summary[key] for key in ("instances", "records", "choice_records", "elapsed_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
