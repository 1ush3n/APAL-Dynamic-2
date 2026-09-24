"""工作三单环境spawn适配器；环境进程不持有策略或训练优化器。"""

from __future__ import annotations

import multiprocessing as mp
import math
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Any

import torch

from envs.work3.core_types import TaskRuntimeState
from envs.work3.decision_snapshot import DecisionSnapshot, build_decision_snapshot
from envs.work3.environment import AirLineEnvWork3
from envs.work3.event_queue import EventType
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
    scenario_status: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Work3ResetResult:
    """一次episode启动结果及预先分配场景的执行状态。"""

    observation: dict[str, Any]
    scenario_status: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Work3VectorStepBatch:
    """同步向量step的完整/中断结果；缺失结果不得用于bootstrap。"""

    results: tuple[Work3StepResult | None, ...]
    dispatched_worker_ids: tuple[int, ...]
    invoked_worker_ids: tuple[int, ...]
    interrupted_worker_ids: tuple[int, ...]
    worker_errors: tuple[tuple[int, str], ...]
    total_env_steps: int
    budget_reserved_steps: int
    wall_clock_expired: bool


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


def _scenario_status(
    env: AirLineEnvWork3,
    scenario: dict[str, Any] | None,
    *,
    event_triggered: bool,
    unhit_reasons_at_event: dict[str, str],
    baseline_material_ready_by_task: dict[str, float],
    in_flight: bool,
) -> dict[str, Any]:
    target_keys = tuple(
        str(task_key) for task_key in (scenario or {}).get("affected_task_keys", ())
    )
    hit_keys = tuple(
        task_key
        for task_key in target_keys
        if event_triggered
        and task_key in env.state.tasks
        and env.state.tasks[task_key].material_ready_time
        > baseline_material_ready_by_task.get(task_key, 0.0) + env.tolerance
    )
    unhit_reasons = {
        task_key: (
            "event_not_triggered"
            if not event_triggered
            else unhit_reasons_at_event.get(
                task_key,
                "target_not_in_instance"
                if task_key not in env.state.tasks
                else "no_material_delay_recorded",
            )
        )
        for task_key in target_keys
        if task_key not in hit_keys
    }
    return {
        "scenario_id": (scenario or {}).get("scenario_id"),
        "started": scenario is not None,
        "event_triggered": bool(event_triggered),
        "scheduled_target_count": len(target_keys),
        "actual_hit_task_keys": hit_keys,
        "actual_hit_count": len(hit_keys),
        "unhit_reasons": unhit_reasons,
        "in_flight": bool(in_flight),
    }


def _worker_main(
    connection: Connection,
    env_kwargs: dict[str, Any],
    torch_num_threads: int,
) -> None:
    """spawn入口：只初始化CPU环境与图构造器，不构造Actor或载入权重。"""
    env: AirLineEnvWork3 | None = None
    try:
        torch.set_num_threads(torch_num_threads)
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
        current_scenario: dict[str, Any] | None = None
        scenario_event_triggered = False
        unhit_reasons_at_event: dict[str, str] = {}
        baseline_material_ready_by_task: dict[str, float] = {}
        original_pop = env.event_queue.pop

        def traced_pop() -> Any:
            nonlocal scenario_event_triggered
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
                if event.event_type == EventType.DISTURBANCE:
                    payload = event.payload
                    if (
                        current_scenario is not None
                        and payload.get("scenario_id") == current_scenario.get("scenario_id")
                    ):
                        scenario_event_triggered = True
                        for task_key_value in payload.get("affected_task_keys", ()):
                            task_key = str(task_key_value)
                            task = env.state.tasks.get(task_key)
                            if task is None:
                                unhit_reasons_at_event[task_key] = "target_not_in_instance"
                            elif task.status.name in {"RUNNING", "COMPLETED"}:
                                unhit_reasons_at_event[task_key] = (
                                    "already_started_or_completed_at_event"
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
        connection.send(
            {
                "request_id": 0,
                "ok": True,
                "result": {
                    "pid": mp.current_process().pid,
                    "cuda_initialized": torch.cuda.is_initialized(),
                    "torch_num_threads": torch.get_num_threads(),
                },
            }
        )

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
                    current_scenario = request.get("scenario")
                    if current_scenario is not None:
                        current_scenario = dict(current_scenario)
                    scenario_event_triggered = False
                    unhit_reasons_at_event.clear()
                    baseline_material_ready_by_task = {
                        str(task_key): float(env.state.tasks[task_key].material_ready_time)
                        for task_key in (current_scenario or {}).get("affected_task_keys", ())
                        if task_key in env.state.tasks
                    }
                    if current_scenario is not None:
                        env.load_scenario(current_scenario)
                        observation = env._get_observation()
                    processed_events.clear()
                    result: Any = Work3ResetResult(
                        observation=observation,
                        scenario_status=_scenario_status(
                            env,
                            current_scenario,
                            event_triggered=scenario_event_triggered,
                            unhit_reasons_at_event=unhit_reasons_at_event,
                            baseline_material_ready_by_task=baseline_material_ready_by_task,
                            in_flight=not env._check_terminated(),
                        ),
                    )
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
                    connection.send(
                        {"request_id": request_id, "ok": True, "phase": "step_started"}
                    )
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
                        scenario_status=_scenario_status(
                            env,
                            current_scenario,
                            event_triggered=scenario_event_triggered,
                            unhit_reasons_at_event=unhit_reasons_at_event,
                            baseline_material_ready_by_task=baseline_material_ready_by_task,
                            in_flight=not (terminated or truncated),
                        ),
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
    """主进程协调固定顺序的spawn CPU环境worker。"""

    def __init__(
        self,
        *,
        env_kwargs: dict[str, Any] | None = None,
        num_envs: int = 1,
        worker_torch_num_threads: int = 1,
        start_method: str = "spawn",
        startup_timeout_seconds: float = 60.0,
        request_timeout_seconds: float = 300.0,
    ) -> None:
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs必须为正整数")
        if type(worker_torch_num_threads) is not int or worker_torch_num_threads < 1:
            raise ValueError("worker_torch_num_threads必须为正整数")
        if start_method != "spawn":
            raise ValueError("工作三环境worker必须使用spawn启动")
        if startup_timeout_seconds <= 0 or request_timeout_seconds <= 0:
            raise ValueError("worker超时必须为正数")
        self._num_envs = num_envs
        self._request_id = 0
        self._closed = False
        self._request_timeout_seconds = float(request_timeout_seconds)
        self._processes: list[mp.Process] = []
        self._parent_connections: list[Connection] = []
        self._worker_metadata: list[dict[str, Any]] = []
        self._episode_ids = [0] * num_envs
        self._episode_indices = [0] * num_envs
        self._worker_step_counts = [0] * num_envs
        self._total_env_steps = 0
        self._budget_reserved_steps = 0
        self._interrupted_workers: set[int] = set()

        context = mp.get_context(start_method)
        try:
            for worker_id in range(num_envs):
                parent_connection, child_connection = context.Pipe(duplex=True)
                process = context.Process(
                    target=_worker_main,
                    args=(
                        child_connection,
                        dict(env_kwargs or {}),
                        worker_torch_num_threads,
                    ),
                    name=f"work3-env-{worker_id}",
                )
                try:
                    process.start()
                except Exception:
                    parent_connection.close()
                    child_connection.close()
                    raise
                child_connection.close()
                self._processes.append(process)
                self._parent_connections.append(parent_connection)

            pending = {
                connection: (worker_id, 0)
                for worker_id, connection in enumerate(self._parent_connections)
            }
            results, errors, timed_out, _invoked = self._collect_responses(
                pending,
                float(startup_timeout_seconds),
            )
            if timed_out or errors or len(results) != num_envs:
                raise RuntimeError(
                    f"工作三worker启动失败：errors={errors}, timed_out={sorted(timed_out)}"
                )
            self._worker_metadata = [results[index] for index in range(num_envs)]
        except Exception:
            self._terminate_all()
            raise

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def pid(self) -> int | None:
        return self._processes[0].pid if self._processes else None

    @property
    def worker_pids(self) -> tuple[int | None, ...]:
        return tuple(process.pid for process in self._processes)

    @property
    def worker_cuda_initialized(self) -> tuple[bool, ...]:
        return tuple(bool(item["cuda_initialized"]) for item in self._worker_metadata)

    @property
    def worker_torch_num_threads(self) -> tuple[int, ...]:
        return tuple(int(item["torch_num_threads"]) for item in self._worker_metadata)

    @property
    def workers_alive(self) -> tuple[bool, ...]:
        return tuple(process.is_alive() for process in self._processes)

    @property
    def is_alive(self) -> bool:
        return any(self.workers_alive)

    @property
    def worker_step_counts(self) -> tuple[int, ...]:
        return tuple(self._worker_step_counts)

    @property
    def total_env_steps(self) -> int:
        return self._total_env_steps

    @property
    def budget_reserved_steps(self) -> int:
        """已派发step数；在途/中断请求也占用预算，防止超发。"""
        return self._budget_reserved_steps

    @property
    def step_settlement_requests(self) -> int:
        """已发送并等待worker回执的step请求数，包含超时或中断请求。"""
        return self._budget_reserved_steps

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _validate_worker_id(self, worker_id: int) -> None:
        if type(worker_id) is not int or not 0 <= worker_id < self._num_envs:
            raise ValueError(f"worker_id必须位于[0, {self._num_envs})")

    def _send_command(
        self,
        worker_id: int,
        command: str,
        payload: dict[str, Any],
    ) -> tuple[Connection, tuple[int, int]]:
        self._validate_worker_id(worker_id)
        if self._closed:
            raise RuntimeError("工作三环境worker已经关闭")
        if worker_id in self._interrupted_workers or not self._processes[worker_id].is_alive():
            raise RuntimeError(f"工作三worker {worker_id}已退出或被中断")
        request_id = self._next_request_id()
        connection = self._parent_connections[worker_id]
        connection.send({"request_id": request_id, "command": command, **payload})
        if command == "step":
            self._budget_reserved_steps += 1
        return connection, (worker_id, request_id)

    def _collect_responses(
        self,
        pending: dict[Connection, tuple[int, int]],
        timeout_seconds: float,
    ) -> tuple[dict[int, Any], list[tuple[int, str]], set[int], tuple[int, ...]]:
        results: dict[int, Any] = {}
        errors: list[tuple[int, str]] = []
        invoked: list[int] = []
        acknowledged: set[int] = set()
        deadline = time.monotonic() + timeout_seconds
        while pending:
            remaining = max(0.0, deadline - time.monotonic())
            ready = wait(tuple(pending), timeout=remaining)
            if not ready:
                break
            for connection in ready:
                worker_id, expected_request_id = pending[connection]
                try:
                    response = connection.recv()
                except (EOFError, OSError) as exc:
                    pending.pop(connection, None)
                    errors.append((worker_id, f"worker连接中断: {exc}"))
                    continue
                if not isinstance(response, dict):
                    pending.pop(connection, None)
                    errors.append((worker_id, "worker响应不是字典"))
                    continue
                if response.get("request_id") != expected_request_id:
                    pending.pop(connection, None)
                    errors.append((worker_id, "worker响应序号不匹配"))
                    continue
                if response.get("phase") == "step_started":
                    if worker_id in acknowledged:
                        pending.pop(connection, None)
                        errors.append((worker_id, "worker重复确认同一step"))
                        continue
                    acknowledged.add(worker_id)
                    invoked.append(worker_id)
                    self._total_env_steps += 1
                    self._worker_step_counts[worker_id] += 1
                    continue
                pending.pop(connection, None)
                if not response.get("ok"):
                    errors.append(
                        (
                            worker_id,
                            f"{response.get('error')}\n{response.get('traceback', '')}",
                        )
                    )
                else:
                    results[worker_id] = response.get("result")
        timed_out = {worker_id for worker_id, _request_id in pending.values()}
        return results, errors, timed_out, tuple(invoked)

    def _terminate_worker(self, worker_id: int) -> None:
        self._interrupted_workers.add(worker_id)
        connection = self._parent_connections[worker_id]
        process = self._processes[worker_id]
        try:
            connection.close()
        except OSError:
            pass
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)

    def _terminate_all(self) -> None:
        self._closed = True
        for worker_id, process in enumerate(self._processes):
            try:
                self._parent_connections[worker_id].close()
            except (IndexError, OSError):
                pass
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

    def _request_many(
        self,
        commands: dict[int, tuple[str, dict[str, Any]]],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[int, Any]:
        pending: dict[Connection, tuple[int, int]] = {}
        send_errors: list[tuple[int, str]] = []
        for worker_id, (command, payload) in sorted(commands.items()):
            try:
                connection, request = self._send_command(worker_id, command, payload)
                pending[connection] = request
            except (BrokenPipeError, EOFError, OSError, RuntimeError) as exc:
                send_errors.append((worker_id, str(exc)))
        results, errors, timed_out, _invoked = self._collect_responses(
            pending,
            self._request_timeout_seconds if timeout_seconds is None else timeout_seconds,
        )
        all_errors = send_errors + errors
        for worker_id, _message in all_errors:
            self._terminate_worker(worker_id)
        for worker_id in timed_out:
            self._terminate_worker(worker_id)
        if all_errors:
            raise RuntimeError(f"工作三worker请求失败: {all_errors}")
        if timed_out:
            raise TimeoutError(f"工作三worker请求超时: {sorted(timed_out)}")
        return results

    def reset(
        self,
        *,
        scenario: dict[str, Any] | None = None,
        episode_id: int = 0,
        episode_index: int = 0,
        worker_id: int = 0,
    ) -> dict[str, Any]:
        result = self._reset_one(
            worker_id,
            scenario=scenario,
            episode_id=episode_id,
            episode_index=episode_index,
        )
        return result.observation

    def _reset_one(
        self,
        worker_id: int,
        *,
        scenario: dict[str, Any] | None,
        episode_id: int,
        episode_index: int,
    ) -> Work3ResetResult:
        self._validate_worker_id(worker_id)
        if (
            type(episode_id) is not int
            or type(episode_index) is not int
            or min(episode_id, episode_index) < 0
        ):
            raise ValueError("episode_id和episode_index必须是非负整数")
        self._episode_ids[worker_id] = episode_id
        self._episode_indices[worker_id] = episode_index
        result = self._request_many(
            {
                worker_id: (
                    "reset",
                    {
                        "scenario": scenario,
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                    },
                )
            }
        )[worker_id]
        if not isinstance(result, Work3ResetResult):
            raise TypeError("环境worker返回了非Work3ResetResult结果")
        return result

    def reset_all(
        self,
        *,
        scenarios: Sequence[dict[str, Any] | None],
        episode_ids: Sequence[int],
        episode_indices: Sequence[int],
    ) -> tuple[Work3ResetResult, ...]:
        if not (
            len(scenarios)
            == len(episode_ids)
            == len(episode_indices)
            == self._num_envs
        ):
            raise ValueError("reset_all必须为每个worker提供场景、episode_id和episode_index")
        commands: dict[int, tuple[str, dict[str, Any]]] = {}
        for worker_id, (scenario, episode_id, episode_index) in enumerate(
            zip(scenarios, episode_ids, episode_indices, strict=True)
        ):
            if (
                type(episode_id) is not int
                or type(episode_index) is not int
                or min(episode_id, episode_index) < 0
            ):
                raise ValueError("episode_id和episode_index必须是非负整数")
            self._episode_ids[worker_id] = episode_id
            self._episode_indices[worker_id] = episode_index
            commands[worker_id] = (
                "reset",
                {
                    "scenario": scenario,
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                },
            )
        results = self._request_many(commands)
        resets = tuple(results[index] for index in range(self._num_envs))
        if not all(isinstance(result, Work3ResetResult) for result in resets):
            raise TypeError("环境worker返回了非Work3ResetResult结果")
        return resets

    def reset_from_plan(
        self,
        plan: Sequence[dict[str, Any]],
    ) -> tuple[Work3ResetResult, ...]:
        """按冻结的worker/episode坐标装载一轮场景，不按完成先后重抽。"""
        if len(plan) != self._num_envs:
            raise ValueError("每轮计划必须恰好包含每个worker一个episode")
        by_worker: dict[int, dict[str, Any]] = {}
        for item in plan:
            worker_id = item.get("worker_id")
            if type(worker_id) is not int or not 0 <= worker_id < self._num_envs:
                raise ValueError("场景计划包含非法worker_id")
            if worker_id in by_worker:
                raise ValueError(f"场景计划重复包含worker {worker_id}")
            if (
                type(item.get("episode_id")) is not int
                or type(item.get("episode_index")) is not int
            ):
                raise ValueError("场景计划必须包含整数episode_id和episode_index")
            if not isinstance(item.get("scenario"), dict):
                raise ValueError("场景计划必须包含scenario字典")
            by_worker[worker_id] = item
        if set(by_worker) != set(range(self._num_envs)):
            raise ValueError("场景计划未覆盖所有worker")
        ordered = [by_worker[index] for index in range(self._num_envs)]
        return self.reset_all(
            scenarios=[item["scenario"] for item in ordered],
            episode_ids=[item["episode_id"] for item in ordered],
            episode_indices=[item["episode_index"] for item in ordered],
        )

    def snapshot(self, *, worker_id: int = 0) -> DecisionSnapshot:
        self._validate_worker_id(worker_id)
        snapshot = self._request_many(
            {
                worker_id: (
                    "snapshot",
                    {
                        "worker_id": worker_id,
                        "episode_id": self._episode_ids[worker_id],
                        "episode_index": self._episode_indices[worker_id],
                    },
                )
            }
        )[worker_id]
        if not isinstance(snapshot, DecisionSnapshot):
            raise TypeError("环境worker返回了非DecisionSnapshot结果")
        return snapshot

    def snapshots(self) -> tuple[DecisionSnapshot, ...]:
        results = self._request_many(
            {
                worker_id: (
                    "snapshot",
                    {
                        "worker_id": worker_id,
                        "episode_id": self._episode_ids[worker_id],
                        "episode_index": self._episode_indices[worker_id],
                    },
                )
                for worker_id in range(self._num_envs)
            }
        )
        snapshots = tuple(results[index] for index in range(self._num_envs))
        if not all(isinstance(snapshot, DecisionSnapshot) for snapshot in snapshots):
            raise TypeError("环境worker返回了非DecisionSnapshot结果")
        return snapshots

    def step(self, action: dict[str, Any], *, worker_id: int = 0) -> Work3StepResult:
        result = self._request_many(
            {worker_id: ("step", {"action": action})}
        )[worker_id]
        if not isinstance(result, Work3StepResult):
            raise TypeError("环境worker返回了非Work3StepResult结果")
        self._episode_indices[worker_id] += 1
        return result

    def step_all(
        self,
        *,
        actions: Sequence[dict[str, Any] | None],
        max_total_steps: int | None,
        settle_timeout_seconds: float | None = None,
        wall_clock_deadline: float | None = None,
    ) -> Work3VectorStepBatch:
        if len(actions) != self._num_envs:
            raise ValueError("step_all动作数必须与worker数相同")
        if max_total_steps is not None and (
            type(max_total_steps) is not int or max_total_steps < 1
        ):
            raise ValueError("max_total_steps必须为正整数或None")
        if settle_timeout_seconds is not None and settle_timeout_seconds <= 0:
            raise ValueError("settle_timeout_seconds必须为正数")
        if wall_clock_deadline is not None and not math.isfinite(wall_clock_deadline):
            raise ValueError("wall_clock_deadline必须是有限的monotonic时间戳")
        wall_clock_expired = (
            wall_clock_deadline is not None
            and time.monotonic() >= wall_clock_deadline
        )
        remaining = (
            self._num_envs
            if max_total_steps is None
            else max_total_steps - self._budget_reserved_steps
        )
        eligible = (
            [
                worker_id
                for worker_id, action in enumerate(actions)
                if action is not None
                and worker_id not in self._interrupted_workers
                and self._processes[worker_id].is_alive()
            ]
            if not wall_clock_expired
            else []
        )
        selected = eligible[: max(0, remaining)]
        results: dict[int, Any] = {}
        errors: list[tuple[int, str]] = []
        timed_out: set[int] = set()
        invoked: tuple[int, ...] = ()
        dispatched: list[int] = []
        if selected:
            pending: dict[Connection, tuple[int, int]] = {}
            for worker_id in selected:
                try:
                    connection, request = self._send_command(
                        worker_id,
                        "step",
                        {"action": actions[worker_id]},
                    )
                    pending[connection] = request
                    dispatched.append(worker_id)
                except (BrokenPipeError, EOFError, OSError, RuntimeError) as exc:
                    errors.append((worker_id, str(exc)))
            results, response_errors, timed_out, invoked = self._collect_responses(
                pending,
                self._request_timeout_seconds
                if settle_timeout_seconds is None
                else settle_timeout_seconds,
            )
            errors.extend(response_errors)
            for worker_id, _message in errors:
                self._terminate_worker(worker_id)
            for worker_id in timed_out:
                self._terminate_worker(worker_id)

        ordered_results: list[Work3StepResult | None] = [None] * self._num_envs
        for worker_id, result in results.items():
            if not isinstance(result, Work3StepResult):
                errors.append((worker_id, "环境worker返回了非Work3StepResult结果"))
                self._terminate_worker(worker_id)
                continue
            self._episode_indices[worker_id] += 1
            ordered_results[worker_id] = result
        return Work3VectorStepBatch(
            results=tuple(ordered_results),
            dispatched_worker_ids=tuple(dispatched),
            invoked_worker_ids=tuple(sorted(invoked)),
            interrupted_worker_ids=tuple(
                sorted(timed_out | {worker_id for worker_id, _ in errors})
            ),
            worker_errors=tuple(errors),
            total_env_steps=self._total_env_steps,
            budget_reserved_steps=self._budget_reserved_steps,
            wall_clock_expired=(
                wall_clock_expired
                or (
                    wall_clock_deadline is not None
                    and time.monotonic() >= wall_clock_deadline
                )
            ),
        )

    def close(self) -> dict[str, Any] | tuple[dict[str, Any] | None, ...]:
        if self._closed:
            return {} if self._num_envs == 1 else tuple(None for _ in self._processes)
        audits: list[dict[str, Any] | None] = [None] * self._num_envs
        close_error: Exception | None = None
        closed_workers: set[int] = set()
        active = {
            worker_id: ("close", {})
            for worker_id, process in enumerate(self._processes)
            if process.is_alive() and worker_id not in self._interrupted_workers
        }
        if active:
            try:
                results = self._request_many(active, timeout_seconds=30.0)
                for worker_id, result in results.items():
                    if isinstance(result, dict):
                        audits[worker_id] = result
                        closed_workers.add(worker_id)
            except (OSError, RuntimeError, TimeoutError) as exc:
                close_error = exc
        self._closed = True
        for worker_id, connection in enumerate(self._parent_connections):
            try:
                connection.close()
            except OSError:
                pass
            process = self._processes[worker_id]
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        if any(process.is_alive() for process in self._processes):
            raise RuntimeError("工作三环境worker未能退出")
        unexpectedly_exited = [
            worker_id
            for worker_id, process in enumerate(self._processes)
            if not process.is_alive()
            and worker_id not in self._interrupted_workers
            and worker_id not in closed_workers
        ]
        if unexpectedly_exited and close_error is None:
            close_error = RuntimeError(f"worker非正常退出: {unexpectedly_exited}")
        if close_error is not None:
            raise RuntimeError(f"工作三worker关闭失败: {close_error}") from close_error
        if self._num_envs == 1:
            return audits[0] or {}
        return tuple(audits)

    def __enter__(self) -> Work3VectorEnv:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except (OSError, RuntimeError, TimeoutError):
                pass
