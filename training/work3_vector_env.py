"""工作三单环境spawn适配器；环境进程不持有策略或训练优化器。"""

from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import torch

from envs.work3.core_types import TaskRuntimeState
from envs.work3.decision_snapshot import DecisionSnapshot, build_decision_snapshot
from envs.work3.environment import AirLineEnvWork3
from models.work3.action_fusion import compute_time_urgency_vector
from models.work3.actor_critic import extract_compact_state_features
from models.work3.graph_builder import (
    MultiAircraftGraphBuilder,
    Work3ResourceConfig,
)
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


@dataclass(frozen=True, slots=True)
class Work3StepResult:
    """一次真实环境step的完整可序列化返回。"""

    observation: dict[str, Any]
    raw_reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    processed_events: tuple[tuple[float, str, int, str | None, int], ...]
    completed_execution_records: tuple[dict[str, Any], ...]
    step_count: int


def _execution_record(
    env: AirLineEnvWork3,
    task_key: str,
    start_station_by_task: dict[str, int],
) -> dict[str, Any]:
    task = env.state.tasks[task_key]
    assignment = task.last_published_assignment or task.baseline_assignment
    station_id = int(assignment.get("station", task.current_station))
    aircraft = env.state.aircraft[task.aircraft_id]
    if task_key not in start_station_by_task:
        raise RuntimeError(f"工序{task_key}缺少真实开工站位记录")
    return {
        "task_key": task.task_key,
        "aircraft_id": int(task.aircraft_id),
        "task_id": int(task.task_id),
        "station_id": station_id,
        "team": tuple(int(worker_id) for worker_id in task.assigned_team),
        "start": float(task.actual_start),
        "end": float(task.actual_end),
        "material_ready_time": float(task.material_ready_time),
        "station_entry_time": aircraft.entry_times.get(station_id),
        "aircraft_station_at_start": start_station_by_task[task_key],
    }


def _trajectory_audit(
    env: AirLineEnvWork3,
    last_info: dict[str, Any],
    start_station_by_task: dict[str, int],
) -> dict[str, Any]:
    execution_records = [
        _execution_record(env, task.task_key, start_station_by_task)
        for task in env.state.tasks.values()
        if task.actual_start is not None and task.actual_end is not None
    ]
    task_constraints = {
        int(task.task_id): {
            "demand": int(task.demand),
            "required_skill": int(task.skill),
            "predecessors": tuple(int(item) for item in task.predecessors),
            "fixed_station": task.fixed_station,
            "max_allowed_station": task.max_allowed_station,
        }
        for task in env.state.tasks.values()
    }
    worker_station_bindings = {
        int(worker_id): int(station_id)
        for station_id, worker_ids in env.state.station_worker_bindings.items()
        for worker_id in worker_ids
    }
    success = env._check_terminated()
    return {
        "success": success,
        "termination_reason": (
            "completed" if success else str(last_info.get("termination_reason", "incomplete"))
        ),
        "current_time": float(env.state.current_time),
        "current_cycle": int(env.state.current_cycle),
        "completed_tasks": sum(
            task.actual_end is not None for task in env.state.tasks.values()
        ),
        "total_tasks": len(env.state.tasks),
        "step_count": int(env.step_count),
        "transfer_history": tuple(float(item) for item in env.state.transfer_history),
        "cumulative_cost": float(env.cumulative_cost),
        "cost_breakdown": {
            "cost_takt": float(env.cost_takt),
            "cost_time": float(env.cost_time),
            "cost_team": float(env.cost_team),
            "cost_postpone": float(env.cost_postpone),
            "cost_revision": float(env.cost_revision),
        },
        "execution_records": execution_records,
        "task_constraints": task_constraints,
        "worker_skills": {
            int(worker_id): tuple(sorted(int(skill) for skill in skills))
            for worker_id, skills in env.worker_skills.items()
        },
        "worker_station_bindings": worker_station_bindings,
        "station_capacities": {
            int(station_id): int(env.max_slots_per_station)
            for station_id in range(env.state.num_stations)
        },
    }


def _worker_main(connection: Connection, env_kwargs: dict[str, Any]) -> None:
    """spawn入口：只初始化CPU环境与图构造器，不构造Actor或载入权重。"""
    env: AirLineEnvWork3 | None = None
    try:
        torch.set_num_threads(1)
        normalized_kwargs = {
            key: Path(value) if key.endswith("_path") else value
            for key, value in env_kwargs.items()
        }
        env = AirLineEnvWork3(**normalized_kwargs)
        baseline = MultiAircraftBaseline.load_from_json(env.baseline_json_path)
        graph_builder = MultiAircraftGraphBuilder(
            baseline,
            config=Work3ResourceConfig(),
        )
        processed_events: list[tuple[float, str, int, str | None, int]] = []
        original_pop = env.event_queue.pop

        def traced_pop() -> Any:
            event = original_pop()
            if event is not None:
                processed_events.append(
                    (
                        float(event.timestamp),
                        event.event_type.name,
                        int(event.event_id),
                        event.task_key,
                        int(event.generation),
                    )
                )
            return event

        env.event_queue.pop = traced_pop  # type: ignore[method-assign]
        start_station_by_task: dict[str, int] = {}
        original_started = env._on_task_started

        def traced_started(task: TaskRuntimeState, start_time: float) -> None:
            original_started(task, start_time)
            if task.actual_start is not None:
                start_station_by_task[task.task_key] = int(
                    env.state.aircraft[task.aircraft_id].current_station
                )

        env._on_task_started = traced_started  # type: ignore[method-assign]
        known_completed: set[str] = set()
        episode_id = 0
        episode_index = 0
        last_info: dict[str, Any] = {}
        connection.send({"request_id": 0, "ok": True, "result": {"pid": mp.current_process().pid}})

        while True:
            request = connection.recv()
            request_id = request.get("request_id")
            command = request.get("command")
            try:
                if not isinstance(request_id, int) or not isinstance(command, str):
                    raise ValueError("worker请求缺少整数request_id或command")
                if command == "reset":
                    episode_id = int(request.get("episode_id", 0))
                    episode_index = int(request.get("episode_index", 0))
                    observation = env.reset()
                    known_completed.clear()
                    start_station_by_task.clear()
                    last_info = {}
                    scenario = request.get("scenario")
                    if scenario is not None:
                        env.load_scenario(scenario)
                        observation = env._get_observation()
                    processed_events.clear()
                    result: Any = observation
                elif command == "snapshot":
                    estimated_cmax = compute_cycle_heuristic_cmax(env.state)
                    state_features = extract_compact_state_features(
                        env.state,
                        estimated_cmax,
                    )
                    time_features = compute_time_urgency_vector(
                        estimated_r=max(0.0, estimated_cmax - env.state.current_time),
                        current_time=env.state.current_time,
                        last_transfer_time=env.state.last_transfer_time,
                        h0=env.state.h0,
                    )
                    result = build_decision_snapshot(
                        env,
                        graph_builder,
                        state_features,
                        time_features,
                        worker_id=int(request.get("worker_id", 0)),
                        episode_id=episode_id,
                        episode_index=episode_index,
                        estimated_cmax=estimated_cmax,
                    )
                elif command == "step":
                    action = request.get("action")
                    if not isinstance(action, dict):
                        raise TypeError("step动作必须是字典")
                    event_offset = len(processed_events)
                    observation, raw_reward, terminated, truncated, info = env.step(action)
                    last_info = dict(info)
                    completed_records = []
                    for task in env.state.tasks.values():
                        if task.actual_end is None or task.task_key in known_completed:
                            continue
                        known_completed.add(task.task_key)
                        completed_records.append(
                            _execution_record(env, task.task_key, start_station_by_task)
                        )
                    episode_index += 1
                    result = Work3StepResult(
                        observation=observation,
                        raw_reward=float(raw_reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        info=dict(info),
                        processed_events=tuple(processed_events[event_offset:]),
                        completed_execution_records=tuple(completed_records),
                        step_count=int(env.step_count),
                    )
                elif command == "close":
                    result = _trajectory_audit(env, last_info, start_station_by_task)
                    connection.send({"request_id": request_id, "ok": True, "result": result})
                    break
                else:
                    raise ValueError(f"未知worker命令: {command}")
                connection.send({"request_id": request_id, "ok": True, "result": result})
            except Exception as exc:
                connection.send(
                    {
                        "request_id": request_id,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                break
    except EOFError:
        pass
    except Exception as exc:
        try:
            connection.send(
                {
                    "request_id": 0,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class Work3VectorEnv:
    """当前阶段限制为单个spawn环境；双worker在后续任务单独验收。"""

    def __init__(
        self,
        *,
        env_kwargs: dict[str, Any] | None = None,
        num_envs: int = 1,
        start_method: str = "spawn",
        startup_timeout_seconds: float = 60.0,
        request_timeout_seconds: float = 300.0,
    ) -> None:
        if num_envs != 1:
            raise ValueError("单环境FP32阶段只支持num_envs=1")
        if start_method != "spawn":
            raise ValueError("工作三环境worker必须使用spawn启动")
        if startup_timeout_seconds <= 0 or request_timeout_seconds <= 0:
            raise ValueError("worker超时必须为正数")
        context = mp.get_context(start_method)
        self._parent_connection, child_connection = context.Pipe(duplex=True)
        self._process = context.Process(
            target=_worker_main,
            args=(child_connection, dict(env_kwargs or {})),
            name="work3-env-0",
        )
        self._request_id = 0
        self._closed = False
        self._request_timeout_seconds = float(request_timeout_seconds)
        self._episode_id = 0
        self._episode_index = 0
        self._process.start()
        child_connection.close()
        try:
            if not self._parent_connection.poll(startup_timeout_seconds):
                raise TimeoutError("工作三环境worker启动超时")
            response = self._parent_connection.recv()
            if not response.get("ok"):
                raise RuntimeError(
                    f"工作三环境worker启动失败: {response.get('error')}\n"
                    f"{response.get('traceback', '')}"
                )
        except Exception:
            self._terminate()
            raise

    @property
    def pid(self) -> int | None:
        return self._process.pid

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()

    def _request(self, command: str, **payload: Any) -> Any:
        if self._closed:
            raise RuntimeError("工作三环境worker已经关闭")
        self._request_id += 1
        request_id = self._request_id
        try:
            self._parent_connection.send(
                {"request_id": request_id, "command": command, **payload}
            )
            if not self._parent_connection.poll(self._request_timeout_seconds):
                raise TimeoutError(f"工作三环境worker命令{command}超时")
            response = self._parent_connection.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError("工作三环境worker连接中断") from exc
        if response.get("request_id") != request_id:
            raise RuntimeError("工作三环境worker响应序号不匹配")
        if not response.get("ok"):
            raise RuntimeError(
                f"工作三环境worker命令{command}失败: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        return response.get("result")

    def reset(
        self,
        *,
        scenario: dict[str, Any] | None = None,
        episode_id: int = 0,
        episode_index: int = 0,
    ) -> dict[str, Any]:
        if episode_id < 0 or episode_index < 0:
            raise ValueError("episode_id和episode_index不得为负数")
        self._episode_id = int(episode_id)
        self._episode_index = int(episode_index)
        return self._request(
            "reset",
            scenario=scenario,
            episode_id=self._episode_id,
            episode_index=self._episode_index,
        )

    def snapshot(self) -> DecisionSnapshot:
        snapshot = self._request(
            "snapshot",
            worker_id=0,
            episode_id=self._episode_id,
            episode_index=self._episode_index,
        )
        if not isinstance(snapshot, DecisionSnapshot):
            raise TypeError("环境worker返回了非DecisionSnapshot结果")
        return snapshot

    def step(self, action: dict[str, Any]) -> Work3StepResult:
        result = self._request("step", action=action)
        if not isinstance(result, Work3StepResult):
            raise TypeError("环境worker返回了非Work3StepResult结果")
        self._episode_index += 1
        return result

    def close(self) -> dict[str, Any]:
        if self._closed:
            return {}
        audit: dict[str, Any] = {}
        try:
            if self._process.is_alive():
                audit = self._request("close")
        finally:
            self._closed = True
            self._parent_connection.close()
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)
        if self._process.is_alive():
            raise RuntimeError("工作三环境worker未能退出")
        return audit

    def _terminate(self) -> None:
        self._closed = True
        self._parent_connection.close()
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)

    def __enter__(self) -> Work3VectorEnv:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except (OSError, RuntimeError):
                pass
