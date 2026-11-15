"""离线已核验的只读调用计划；不执行模型生成的Python或任意网络请求。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from envs.work3.scheduling_queries import QUERY_VERSION, CompletionTable, SchedulingQueries

if TYPE_CHECKING:
    from envs.work3.environment import AirLineEnvWork3


@dataclass(frozen=True, slots=True)
class ConditionProgram:
    description: str
    version: str
    fingerprint: str

    def execute(self, env: AirLineEnvWork3, candidate_key: str) -> CompletionTable:
        """每次绑定飞机—工序实例；只读最新事实，不保存编译时日历值。"""
        if candidate_key not in env.state.tasks:
            raise ValueError(f"查询对象不存在：{candidate_key}")
        task = env.state.tasks[candidate_key]
        if task.task_key != f"{task.aircraft_id}_{task.task_id}":
            raise ValueError("查询对象的飞机—工序身份不一致")
        return SchedulingQueries(env).completions(candidate_key)


def validate_condition_program(plan: dict[str, Any]) -> ConditionProgram:
    """本轮仅支持读取已有调度事实；新增限制必须先由环境实现。"""
    if plan.get("api") != "team_completions":
        raise ValueError("调用不在调度查询白名单中")
    if set(plan) != {"description", "version", "scope", "api", "args", "review"}:
        raise ValueError("查询计划字段不完整或含不支持的条件/限制")
    if plan["version"] != QUERY_VERSION:
        raise ValueError("查询接口版本不一致，需要重新核验程序")
    if plan["scope"] != "current_candidate_instance" or plan["args"] != {"task_key": "$candidate.task_key"}:
        raise ValueError("仅支持绑定当前候选实例，不接受任意对象或参数")
    if plan["review"] != "offline_verified_retrieval_only":
        raise ValueError("程序未经离线语义核验；语法正确不能替代核验")
    if not isinstance(plan["description"], str) or not plan["description"].strip():
        raise ValueError("程序缺少来源文字说明")
    fingerprint = hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return ConditionProgram(plan["description"], plan["version"], fingerprint)


def load_condition_program(path: Path | None = None) -> ConditionProgram:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "conf" / "work3" / "condition_program.json"
    return validate_condition_program(json.loads(Path(path).read_text(encoding="utf-8")))


def compile_condition_program(description: str, generate: Callable[[str, str], dict[str, Any]],
                              *, cache_dir: Path, attempts: int) -> Path:
    """低频编译并缓存候选计划；模型输出必须另经语义核验才能用于现场查询。"""
    if not description.strip() or type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError("说明不能为空，生成重试上限必须在1到3之间")
    key = hashlib.sha256((QUERY_VERSION + description).encode("utf-8")).hexdigest()
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
            plan = generate(description, feedback)
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
    template = json.loads((Path(__file__).resolve().parents[2] / "conf" / "work3" / "condition_program.json").read_text(encoding="utf-8"))

    def generate(description: str, feedback: str) -> dict[str, Any]:
        instructions = ("输出JSON查询计划，结构严格参照模板。仅支持读取已有调度条件；"
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
                                       cache_dir=Path(settings.cache_dir), attempts=int(settings.attempts))
    print(f"候选计划已保存，仍需语义核验：{output}")


if __name__ == "__main__":
    main()
