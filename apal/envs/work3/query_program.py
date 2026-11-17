"""离线已核验的只读调用计划；不执行模型生成的Python或任意网络请求。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
from apal.project import PROJECT_ROOT
from typing import Any, TYPE_CHECKING

from apal.envs.work3.scheduling_queries import QUERY_VERSION, CompletionTable, SchedulingQueries, QueryStatus, ScheduleCondition

if TYPE_CHECKING:
    from apal.envs.work3.environment import AirLineEnvWork3

PROGRAM_VERSION = "work3_query_plan_v2"


@dataclass(frozen=True, slots=True)
class QueryStep:
    api: str
    aircraft_ids: tuple[int, ...] = ()
    task_ids: tuple[int, ...] = ()
    relation: str = ""
    include_self: bool = False
    predicate: str = ""
    negate: bool = False


@dataclass(frozen=True, slots=True)
class ConditionProgram:
    description: str
    version: str
    fingerprint: str
    steps: tuple[QueryStep, ...]

    def execute(self, env: AirLineEnvWork3, candidate_key: str) -> CompletionTable:
        """每次绑定飞机—工序实例；只读最新事实，不保存编译时日历值。"""
        return self.execute_candidates(env, (candidate_key,))[0]

    def execute_candidates(self, env: AirLineEnvWork3, candidates: tuple[str, ...]) -> tuple[CompletionTable, ...]:
        """对象/关系/条件只决定检索范围；未检索候选返回未知，不修改动作合法掩码。"""
        for key in candidates:
            if key not in env.state.tasks:
                raise ValueError(f"查询对象不存在：{key}")
        selected: set[str] = set()
        for step in self.steps:
            if step.api == "bind_candidate":
                selected = set(candidates)
            elif step.api == "bind_tasks":
                selected = {f"{aircraft}_{task}" for aircraft in step.aircraft_ids for task in step.task_ids}
                if not selected.issubset(env.state.tasks):
                    raise ValueError("查询计划绑定了当前实例不存在的飞机—工序对象")
            elif step.api == "related_tasks":
                related: set[str] = set()
                for key in selected:
                    task = env.state.tasks[key]
                    if step.relation == "predecessors":
                        related.update(f"{task.aircraft_id}_{p}" for p in task.predecessors)
                    else:
                        related.update(other.task_key for other in env.state.tasks.values()
                                       if other.aircraft_id == task.aircraft_id and task.task_id in other.predecessors)
                if not related.issubset(env.state.tasks):
                    raise ValueError("查询关系引用了当前实例不存在的工序")
                selected = related | (selected if step.include_self else set())
            elif step.api == "where":
                def matches(key: str) -> bool:
                    task = env.state.tasks[key]
                    value = (task.material_ready_time > env.state.current_time + env.tolerance
                             if step.predicate == "material_wait" else env._predecessor_release_time(task) is not None)
                    return bool(value) != step.negate
                selected = {key for key in selected if matches(key)}
        query = SchedulingQueries(env)
        results: list[CompletionTable] = []
        for key in candidates:
            task = env.state.tasks[key]
            if task.task_key != f"{task.aircraft_id}_{task.task_id}":
                raise ValueError("查询对象的飞机—工序身份不一致")
            if key in selected:
                results.append(query.completions(key))
            else:
                results.append(CompletionTable(key, (), ScheduleCondition(QueryStatus.UNKNOWN, reasons=("outside_program_scope",)),
                    tuple(env.valid_team_completion_workers(task, [])), task.demand))
        return tuple(results)


def validate_condition_program(plan: dict[str, Any]) -> ConditionProgram:
    """本轮仅支持读取已有调度事实；新增限制必须先由环境实现。"""
    if not isinstance(plan, dict):
        raise ValueError("查询计划必须为JSON对象")
    if "steps" not in plan:
        if plan.get("api") != "team_completions":
            raise ValueError("调用不在调度查询白名单中")
        if set(plan) != {"description", "version", "scope", "api", "args", "review"}:
            raise ValueError("查询计划字段不完整或含不支持的条件/限制")
        if plan["scope"] != "current_candidate_instance" or plan["args"] != {"task_key": "$candidate.task_key"}:
            raise ValueError("仅支持绑定当前候选实例，不接受任意对象或参数")
        raw_steps = [{"api": "bind_candidate"}, {"api": "team_completions"}]
    else:
        if set(plan) != {"description", "version", "plan_version", "steps", "review"} or plan["plan_version"] != PROGRAM_VERSION:
            raise ValueError("查询计划结构版本或字段不支持")
        raw_steps = plan["steps"]
    if plan["version"] != QUERY_VERSION:
        raise ValueError("查询接口版本不一致，需要重新核验程序")
    if plan["review"] != "offline_verified_retrieval_only":
        raise ValueError("程序未经离线语义核验；语法正确不能替代核验")
    if not isinstance(plan["description"], str) or not plan["description"].strip():
        raise ValueError("程序缺少来源文字说明")
    if (not isinstance(raw_steps, list) or not 2 <= len(raw_steps) <= 6
        or not isinstance(raw_steps[0], dict) or raw_steps[0].get("api") not in {"bind_candidate", "bind_tasks"}
        or raw_steps[-1] != {"api": "team_completions"}):
        raise ValueError("计划必须先绑定对象并以团队窗口查询结束，步骤数为2到6")
    steps: list[QueryStep] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise ValueError("查询步骤必须为JSON对象")
        api = raw.get("api")
        if index == 0 and api == "bind_candidate" and set(raw) == {"api"}:
            step = QueryStep(api)
        elif index == 0 and api == "bind_tasks" and set(raw) == {"api", "aircraft_ids", "task_ids"}:
            for field in ("aircraft_ids", "task_ids"):
                values = raw[field]
                if not isinstance(values, list) or not values or any(type(value) is not int or value < 0 for value in values) or len(set(values)) != len(values):
                    raise ValueError("绑定对象编号必须为非空、不重复的非负整数列表")
            step = QueryStep(api, tuple(raw["aircraft_ids"]), tuple(raw["task_ids"]))
        elif 0 < index < len(raw_steps) - 1 and api == "related_tasks" and set(raw) == {"api", "relation", "include_self"}:
            if raw["relation"] not in {"predecessors", "successors"} or type(raw["include_self"]) is not bool:
                raise ValueError("仅支持同机直接前序/后继关系及明确的包含自身标记")
            step = QueryStep(api, relation=raw["relation"], include_self=raw["include_self"])
        elif 0 < index < len(raw_steps) - 1 and api == "where" and set(raw) == {"api", "predicate", "negate"}:
            if raw["predicate"] not in {"material_wait", "predecessor_known"} or type(raw["negate"]) is not bool:
                raise ValueError("仅支持已揭示物料等待/前序释放已知条件及明确否定标记")
            step = QueryStep(api, predicate=raw["predicate"], negate=raw["negate"])
        elif index == len(raw_steps) - 1 and raw == {"api": "team_completions"}:
            step = QueryStep(api)
        else:
            raise ValueError("查询步骤不在白名单或含非法参数/顺序")
        steps.append(step)
    fingerprint = hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return ConditionProgram(plan["description"], plan["version"], fingerprint, tuple(steps))


def load_condition_program(path: Path | None = None) -> ConditionProgram:
    if path is None:
        path = PROJECT_ROOT / "conf" / "work3" / "condition_program.json"
    return validate_condition_program(json.loads(Path(path).read_text(encoding="utf-8")))


def compile_condition_program(description: str, generate: Callable[[str, str], dict[str, Any]],
                              *, cache_dir: Path, attempts: int,
                              verified_examples: tuple[dict[str, Any], ...] = ()) -> Path:
    """低频编译并缓存候选计划；模型输出必须另经语义核验才能用于现场查询。"""
    if not description.strip() or type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError("说明不能为空，生成重试上限必须在1到3之间")
    if len(verified_examples) > 3:
        raise ValueError("生成示例最多使用3个已核验计划")
    for example in verified_examples:
        validate_condition_program(example)
    examples_context = ("已核验检索示例：" + json.dumps(verified_examples, ensure_ascii=False)) if verified_examples else ""
    key = hashlib.sha256((PROGRAM_VERSION + QUERY_VERSION + description).encode("utf-8")).hexdigest()
    output = Path(cache_dir) / f"{key}.json"
    if output.exists():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if cached.get("description") != description or cached.get("version") != QUERY_VERSION:
            raise ValueError("编译缓存身份不一致")
        validate_condition_program({**cached, "review": "offline_verified_retrieval_only"})
        return output
    feedback = ""
    for _ in range(attempts):
        try:
            plan = generate(description, examples_context + feedback)
            if not isinstance(plan, dict):
                raise ValueError("查询计划必须为JSON对象")
            if plan.get("description") != description:
                raise ValueError("生成计划丢失或改写了原始说明")
            # 此处只核验接口，不能将模型自评当作语义审核。
            plan["review"] = "offline_verified_retrieval_only"
            validate_condition_program(plan)
        except (ValueError, TypeError, KeyError) as error:
            feedback = str(error)
            continue
        plan["review"] = "pending_semantic_review"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2)
        return output
    raise ValueError(f"查询计划未通过接口核验：{feedback}")


def main() -> None:
    """显式预编译命令；训练和普通事件不会进入此服务调用入口。"""
    import argparse
    import os
    from urllib.request import Request, urlopen
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="低频生成查询计划；默认训练使用已核验离线计划")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--description-file", type=Path, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    settings = config.llm
    endpoint = os.environ[str(settings.endpoint_env)]
    api_key = os.environ[str(settings.api_key_env)]
    model = os.environ[str(settings.model_env)]
    if not endpoint.startswith("https://"):
        raise ValueError("外部LLM服务地址必须使用HTTPS")
    template = {"description": "保留说明原文", "version": QUERY_VERSION, "plan_version": PROGRAM_VERSION,
                "steps": [{"api": "bind_candidate"}, {"api": "team_completions"}], "review": "pending_semantic_review"}
    examples = tuple(json.loads(Path(path).read_text(encoding="utf-8")) for path in settings.get("example_program_paths", []))

    def generate(description: str, feedback: str) -> dict[str, Any]:
        instructions = ("输出JSON查询计划，结构严格参照模板。steps首步为bind_candidate或bind_tasks(aircraft_ids,task_ids非负整数列表)，"
                        "中间可用related_tasks(relation为predecessors或successors,include_self布尔值)，"
                        "where(predicate为material_wait或predecessor_known,negate布尔值)，末步为team_completions，无其它参数。"
                        "只检索已揭示事实，不改变资格/资源/后移硬约束；编号必须是指定飞机的具体工序实例。"
                        "保留说明原文；若含新增限制、例外或对象范围不能表示，返回unsupported_reason，禁止省略限制。"
                        + json.dumps(template, ensure_ascii=False))
        payload = {"model": model, "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": instructions},
                                {"role": "user", "content": description + "\n接口核验反馈：" + feedback}]}
        request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), method="POST",
                          headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        with urlopen(request, timeout=float(settings.timeout_seconds)) as response:
            result = json.load(response)
        return json.loads(result["choices"][0]["message"]["content"])

    output = compile_condition_program(args.description_file.read_text(encoding="utf-8"), generate,
                                       cache_dir=Path(settings.cache_dir), attempts=int(settings.attempts), verified_examples=examples)
    print(f"候选计划已保存，仍需语义核验：{output}")


if __name__ == "__main__":
    main()
