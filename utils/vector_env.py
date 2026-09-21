import copy
import multiprocessing as mp
import os
import time
from multiprocessing.connection import wait
from pathlib import Path
from typing import Callable, List, Tuple, Any, Optional, Sequence
import numpy as np
import torch
from torch_geometric.data import HeteroData

class EnvCreator:
    """
    一个可序列化 (Picklable) 的环境创建器。
    用于规避 Windows 平台 spawn 模式下闭包 (Local Function) 无法被序列化传输至子进程的问题。
    """
    def __init__(self, data_path_or_dir: str | Sequence[str], seed_offset: int = 42, config_overrides: Optional[dict] = None):
        self.data_path_or_dir = tuple(data_path_or_dir) if not isinstance(data_path_or_dir, str) else data_path_or_dir
        self.seed_offset = seed_offset
        if config_overrides is None:
            try:
                from configs import configs
                config_overrides = configs.to_flat_dict()
            except Exception:
                config_overrides = {}
        self.config_overrides = dict(config_overrides)

    def __call__(self, index: int):
        if self.config_overrides:
            from configs import configs
            configs.update_from_dict(self.config_overrides)
        from environment import AirLineEnv_Graph, _fill_station_macro_features
        return AirLineEnv_Graph(data_path_or_dir=self.data_path_or_dir, seed=self.seed_offset + index)


class VectorEnvWorkerError(RuntimeError):
    """VectorEnv 子进程启动、通信或执行失败。"""


def _snapshot_to_ipc(snapshot: dict, include_static: bool = True) -> dict:
    """只转换 snapshot 中可能触发 Torch 共享内存传输的张量。
    当 include_static 为 False 时，剥离不变的基础特征张量 base_task_x 和 base_worker_x，
    减少 75%~81% 的跨进程 IPC 序列化与数据传输开销。由主进程 EnvProxy 本地影子缓存还原。
    """
    from configs import configs
    result = dict(snapshot)
    should_include = include_static or bool(getattr(configs, 'enable_online_duration_perturb', False))
    if not should_include:
        result.pop("base_task_x", None)
        result.pop("base_worker_x", None)
        return result

    for key in ("base_task_x", "base_worker_x"):
        value = result.get(key)
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().numpy().copy()
    return result


def _masks_to_ipc(masks: tuple[torch.Tensor, ...]) -> tuple[np.ndarray, ...]:
    return tuple(mask.detach().cpu().numpy() for mask in masks)


def _worker(
    conn,
    make_env_fn: Callable[[int], Any],
    index: int,
    worker_threads: int,
):
    """
    运行在独立子进程中的环境步进循环。
    通过 Pipe 与主进程的 EnvProxy 通信，避免主进程 GIL 争用。
    """
    worker_threads = max(1, int(worker_threads))
    os.environ["OMP_NUM_THREADS"] = str(worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(worker_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(worker_threads)
    torch.set_num_threads(worker_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, worker_threads)))
    except RuntimeError:
        pass

    try:
        env = make_env_fn(index)
    except Exception:
        import traceback
        tb_str = traceback.format_exc()
        conn.send(("INIT_ERROR", tb_str))
        conn.close()
        return

    # 1. 成功初始化后，打包发送初始静态属性，由主进程 Proxy 本地缓存
    init_info = {
        'num_tasks': getattr(env, 'num_tasks', None),
        'ideal_makespan': getattr(env, 'ideal_makespan', None),
        'mean_task_time': getattr(env, 'mean_task_time', None),
        'dataset_count': int(getattr(env, 'dataset_count', 1)),
        'active_dataset_idx': int(getattr(env, 'active_dataset_idx', 0)),
        'dataset_descriptor': env.get_dataset_descriptor(getattr(env, 'active_dataset_idx', 0)),
        'base_task_x': _snapshot_to_ipc({'base_task_x': getattr(env, 'base_task_x', None)}, include_static=True).get('base_task_x'),
        'base_worker_x': _snapshot_to_ipc({'base_worker_x': getattr(env, 'base_worker_x', None)}, include_static=True).get('base_worker_x'),
        'worker_audit': {
            'pid': os.getpid(),
            'torch_num_threads': torch.get_num_threads(),
            'omp_num_threads': os.environ.get("OMP_NUM_THREADS"),
            'cuda_available': torch.cuda.is_available(),
        },
    }
    conn.send(("INIT_OK", init_info))

    # 2. 持续循环监听主进程的物理驱动指令
    while True:
        try:
            cmd, data = conn.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            break
        try:
            
            if cmd == 'step':
                if data is None:
                    # 停滞或死锁处理，不步进，返回轻量级切片 snapshot 作为 obs 兜底
                    snap = _snapshot_to_ipc(env.get_state_snapshot(), include_static=False)
                    reward = 0.0
                    done = True
                    info = {}
                else:
                    _, reward, done, info = env.step(data)
                    snap = _snapshot_to_ipc(env.get_state_snapshot(), include_static=False)
                
                # 随每一次步进返回更新后的动态属性
                dynamic_info = {
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                info['dynamic_info'] = dynamic_info
                conn.send(("OK", (snap, reward, done, info)))
                
            elif cmd == 'reset':
                env.reset(**data)
                snap = _snapshot_to_ipc(env.get_state_snapshot(), include_static=True)
                
                # 域随机化后，很多静态和动态属性会改变，需回传主进程同步刷新缓存
                dynamic_info = {
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'num_tasks': getattr(env, 'num_tasks', None),
                    'ideal_makespan': getattr(env, 'ideal_makespan', None),
                    'mean_task_time': getattr(env, 'mean_task_time', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                conn.send(("OK", (snap, dynamic_info)))

            elif cmd == 'reset_rollout':
                old_skip_obs = getattr(env, 'skip_obs_building', False)
                try:
                    env.skip_obs_building = True
                    env.reset(**data)
                finally:
                    env.skip_obs_building = old_skip_obs
                masks = _masks_to_ipc(env.get_masks())
                snap = _snapshot_to_ipc(env.get_state_snapshot(), include_static=True)
                dynamic_info = {
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'num_tasks': getattr(env, 'num_tasks', None),
                    'ideal_makespan': getattr(env, 'ideal_makespan', None),
                    'mean_task_time': getattr(env, 'mean_task_time', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                conn.send(("OK", (masks, snap, dynamic_info)))
                
            elif cmd == 'get_masks':
                masks = env.get_masks()
                conn.send(("OK", _masks_to_ipc(masks)))
                
            elif cmd == 'get_rollout_state':
                masks = _masks_to_ipc(env.get_masks())
                snap = _snapshot_to_ipc(env.get_state_snapshot(), include_static=False)
                dynamic_info = {
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                conn.send(("OK", (masks, snap, dynamic_info)))
                
            elif cmd == 'step_snapshot':
                if data is None:
                    snap = _snapshot_to_ipc(env.get_state_snapshot(), include_static=False)
                    conn.send(("OK", (snap, 0.0, True, {})))
                else:
                    old_skip_obs = getattr(env, 'skip_obs_building', False)
                    try:
                        env.skip_obs_building = True
                        _, reward, done, info = env.step(data)
                    finally:
                        env.skip_obs_building = old_skip_obs
                    snap = _snapshot_to_ipc(env.get_state_snapshot(), include_static=False)
                    dynamic_info = {
                        'station_wall_clock': getattr(env, 'station_wall_clock', None),
                        'assigned_tasks': getattr(env, 'assigned_tasks', None),
                        'task_status': getattr(env, 'task_status', None),
                        'current_time': getattr(env, 'current_time', 0.0),
                    }
                    info['dynamic_info'] = dynamic_info
                    conn.send(("OK", (snap, reward, done, info)))

            elif cmd == 'step_rollout':
                if data is None:
                    reward = 0.0
                    done = True
                    info = {}
                else:
                    old_skip_obs = getattr(env, 'skip_obs_building', False)
                    try:
                        env.skip_obs_building = True
                        _, reward, done, info = env.step(data)
                    finally:
                        env.skip_obs_building = old_skip_obs
                masks = _masks_to_ipc(env.get_masks())
                snap = _snapshot_to_ipc(env.get_state_snapshot(), include_static=False)
                dynamic_info = {
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                info['dynamic_info'] = dynamic_info
                conn.send(("OK", (masks, snap, reward, done, info)))
                
            elif cmd == 'try_wait_for_resources':
                res = env.try_wait_for_resources()
                dynamic_info = {
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                conn.send(("OK", (res, dynamic_info)))

            elif cmd == 'wait_rollout':
                res = env.try_wait_for_resources()
                masks = _masks_to_ipc(env.get_masks())
                snap = _snapshot_to_ipc(env.get_state_snapshot(), include_static=False)
                dynamic_info = {
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                conn.send(("OK", (res, masks, snap, dynamic_info)))
                
            elif cmd == 'get_state_snapshot':
                snap = _snapshot_to_ipc(env.get_state_snapshot(), include_static=False)
                conn.send(("OK", snap))
                
            elif cmd == 'switch_dataset':
                env.switch_dataset(data)
                # 切图后，更新对应的所有静态/动态特征骨架
                dynamic_info = {
                    'num_tasks': getattr(env, 'num_tasks', None),
                    'ideal_makespan': getattr(env, 'ideal_makespan', None),
                    'mean_task_time': getattr(env, 'mean_task_time', None),
                    'dataset_count': int(getattr(env, 'dataset_count', 1)),
                    'active_dataset_idx': int(getattr(env, 'active_dataset_idx', 0)),
                    'dataset_descriptor': env.get_dataset_descriptor(data),
                    'station_wall_clock': getattr(env, 'station_wall_clock', None),
                    'assigned_tasks': getattr(env, 'assigned_tasks', None),
                    'task_status': getattr(env, 'task_status', None),
                    'current_time': getattr(env, 'current_time', 0.0),
                }
                conn.send(("OK", dynamic_info))
                
            elif cmd == 'rebuild_state_from_snapshot':
                # 兜底接口，通常只由本地 Proxy 直接执行，此处仅作向下兼容
                env.rebuild_state_from_snapshot(data)
                conn.send(("OK", None))
                
            elif cmd == 'initialize_dataset_context':
                idx = int(data)
                conn.send(("OK", env.get_dataset_descriptor(idx)))
                
            elif cmd == 'close':
                conn.send(("OK", None))
                conn.close()
                break
            else:
                conn.send(("ERROR", ValueError(f"Unknown command: {cmd}")))
        except Exception:
            import traceback
            conn.send(("ERROR", traceback.format_exc()))


class EnvProxy:
    def _load_dataset_context_locally(self, idx: int) -> dict:
        """在主进程按路径重建静态上下文，避免通过 Pipe 传递 Tensor。"""
        if not 0 <= idx < len(self.dataset_pool):
            raise IndexError(f"数据集索引越界: {idx}/{len(self.dataset_pool)}")
        descriptor = self.dataset_pool[idx]
        if not descriptor or not descriptor.get("file_path"):
            raise VectorEnvWorkerError(
                f"VectorEnv worker {self._idx} 缺少数据集 {idx} 的文件描述"
            )

        from environment import AirLineEnv_Graph

        file_path = Path(str(descriptor["file_path"])).resolve()
        loader = AirLineEnv_Graph(file_path, seed=0)
        source = loader.dataset_pool[loader.active_dataset_idx]
        required = (
            "file_path",
            "num_tasks",
            "base_data",
            "base_task_x",
            "base_worker_x",
            "base_station_x",
            "task_skill_edge_index",
            "mean_task_time",
            "ideal_station_load",
        )
        missing = [key for key in required if key not in source]
        if missing:
            raise VectorEnvWorkerError(
                f"数据集 {file_path} 的静态上下文缺少字段: {missing}"
            )
        context = {key: source[key] for key in required}
        context["file_path"] = str(file_path)
        self.dataset_pool[idx] = context
        return context
    """
    用于多进程 VectorEnv 的环境代理类。
    对主进程表现得与单环境一模一样，通过本地属性影子缓存实现超低延迟的同步属性查询。
    """
    def __init__(self, conn, process: mp.Process, idx: int, command_timeout_sec: float):
        self._conn = conn
        self._process = process
        self._idx = idx
        self._command_timeout_sec = float(command_timeout_sec)
        
        # 1. 静态属性缓存 (在初始化或切换数据集时写入)
        self.num_tasks: Optional[int] = None
        self.ideal_makespan: Optional[float] = None
        self.mean_task_time: Optional[float] = None
        self.dataset_count: int = 0
        self.active_dataset_idx: int = 0
        self.dataset_pool: List[Optional[dict]] = []
        self._cached_base_task_x: Optional[Any] = None
        self._cached_base_worker_x: Optional[Any] = None
        
        # 2. 动态属性缓存 (在 reset, step 和 try_wait_for_resources 之后同步更新)
        self.station_wall_clock: Optional[np.ndarray] = None
        self.assigned_tasks: List[Any] = []
        self.task_status: Optional[np.ndarray] = None
        self.current_time: float = 0.0
        self._worker_skill_topology_cache: dict[tuple[int, bytes], torch.Tensor] = {}

    def _enrich_snapshot(self, snapshot: dict) -> dict:
        """从本地影子缓存注入被剥离的静态特征张量，保证下游语义 100% 完整与零拷贝"""
        if snapshot is None:
            return snapshot
        if 'base_task_x' in snapshot and snapshot['base_task_x'] is not None:
            self._cached_base_task_x = snapshot['base_task_x']
        elif self._cached_base_task_x is not None:
            snapshot['base_task_x'] = self._cached_base_task_x

        if 'base_worker_x' in snapshot and snapshot['base_worker_x'] is not None:
            self._cached_base_worker_x = snapshot['base_worker_x']
        elif self._cached_base_worker_x is not None:
            snapshot['base_worker_x'] = self._cached_base_worker_x
        return snapshot

    def update_static_properties(self, info: dict):
        if 'num_tasks' in info: self.num_tasks = info['num_tasks']
        if 'ideal_makespan' in info: self.ideal_makespan = info['ideal_makespan']
        if 'mean_task_time' in info: self.mean_task_time = info['mean_task_time']
        if 'dataset_count' in info:
            self.dataset_count = int(info['dataset_count'])
            while len(self.dataset_pool) < self.dataset_count:
                self.dataset_pool.append(None)
        if 'active_dataset_idx' in info:
            self.active_dataset_idx = int(info['active_dataset_idx'])
        if 'base_task_x' in info and info['base_task_x'] is not None:
            self._cached_base_task_x = info['base_task_x']
        if 'base_worker_x' in info and info['base_worker_x'] is not None:
            self._cached_base_worker_x = info['base_worker_x']
        descriptor = info.get('dataset_descriptor')
        if descriptor is not None:
            idx = int(descriptor['dataset_idx'])
            while len(self.dataset_pool) <= idx:
                self.dataset_pool.append(None)
            cached = self.dataset_pool[idx] or {}
            cached.update(descriptor)
            self.dataset_pool[idx] = cached

    def _recv(self, operation: str):
        if not self._conn.poll(self._command_timeout_sec):
            exit_code = self._process.exitcode
            state = "alive" if self._process.is_alive() else f"exitcode={exit_code}"
            raise TimeoutError(
                f"VectorEnv worker {self._idx} 执行 {operation} 超时 "
                f"({self._command_timeout_sec:.1f}s, {state})"
            )
        status, value = self._conn.recv()
        if status not in {"OK", "INIT_OK"}:
            raise VectorEnvWorkerError(
                f"VectorEnv worker {self._idx} 执行 {operation} 失败:\n{value}"
            )
        return value

    def update_dynamic_properties(self, info: dict):
        if 'station_wall_clock' in info: self.station_wall_clock = info['station_wall_clock']
        if 'assigned_tasks' in info: self.assigned_tasks = info['assigned_tasks']
        if 'task_status' in info: self.task_status = info['task_status']
        if 'current_time' in info: self.current_time = info['current_time']

    def reset(self, randomize_duration: bool = False, randomize_workers: bool = False, seed: Optional[int] = None):
        self._conn.send(('reset', {'randomize_duration': randomize_duration, 'randomize_workers': randomize_workers, 'seed': seed}))
        snapshot, dynamic_info = self._recv("reset")
        snapshot = self._enrich_snapshot(snapshot)
        self.update_dynamic_properties(dynamic_info)
        self.update_static_properties(dynamic_info)
        return self.rebuild_state_from_snapshot(snapshot)

    def step(self, action: Any):
        self._conn.send(('step', action))
        snapshot, reward, done, info = self._recv("step")
        snapshot = self._enrich_snapshot(snapshot)
        if 'dynamic_info' in info:
            self.update_dynamic_properties(info.pop('dynamic_info'))
        return self.rebuild_state_from_snapshot(snapshot), reward, done, info

    def get_masks(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._conn.send(('get_masks', None))
        return tuple(torch.from_numpy(mask) for mask in self._recv("get_masks"))

    def get_rollout_state(self):
        self._conn.send(('get_rollout_state', None))
        masks, snap, dynamic_info = self._recv("get_rollout_state")
        snap = self._enrich_snapshot(snap)
        self.update_dynamic_properties(dynamic_info)
        return tuple(torch.from_numpy(mask) for mask in masks), snap

    def step_snapshot(self, action: Any):
        self._conn.send(('step_snapshot', action))
        snap, reward, done, info = self._recv("step_snapshot")
        snap = self._enrich_snapshot(snap)
        if 'dynamic_info' in info:
            self.update_dynamic_properties(info.pop('dynamic_info'))
        return snap, reward, done, info

    def try_wait_for_resources(self) -> bool:
        self._conn.send(('try_wait_for_resources', None))
        res, dynamic_info = self._recv("try_wait_for_resources")
        self.update_dynamic_properties(dynamic_info)
        return res

    def get_state_snapshot(self) -> dict:
        self._conn.send(('get_state_snapshot', None))
        return self._enrich_snapshot(self._recv("get_state_snapshot"))

    def switch_dataset(self, idx: int):
        self._conn.send(('switch_dataset', idx))
        val = self._recv("switch_dataset")
        self.update_static_properties(val)
        self.update_dynamic_properties(val)

    def rebuild_state_from_snapshot(
        self,
        snapshot: dict,
        *,
        reusable_state: Optional[HeteroData] = None,
        reuse_resource_topology: bool = False,
    ) -> HeteroData:
        """
        基于快照恢复成 PyG 图结构。
        核心设计：此方法完全通过本地影子缓存直接进行数学与张量计算，不需要向子进程发送 IPC 信号。
        消除了 PPO 训练更新阶段高频通信带来的带宽和延迟延迟。
        支持传入 reusable_state 复用静态拓扑结构，消除全图深拷贝开销。
        """
        import hashlib
        from configs import configs
        from environment import _fill_station_macro_features
        from worker_feature_layout import resolve_worker_feature_layout
        from utils.resource_graph import (
            SkillHubTopology,
            apply_resource_graph,
            build_skill_features,
            build_worker_skill_edges,
            worker_topology_key,
        )
        
        ctx_idx = snapshot.get('dataset_idx', 0)
        while len(self.dataset_pool) <= ctx_idx:
            self.dataset_pool.append(None)
        ctx = self.dataset_pool[ctx_idx]
        
        # 数据集上下文只在主进程按路径构造，禁止通过 Pipe 传递 Tensor。
        if ctx is None or 'base_data' not in ctx:
            ctx = self._load_dataset_context_locally(ctx_idx)

        raw_worker_topology_key = snapshot.get("worker_topology_key")
        topology_digest = hashlib.sha256(
            repr(raw_worker_topology_key).encode("utf-8")
        ).hexdigest()
        topology_key = (
            f"skill={int(bool(getattr(configs, 'use_skill_hub', False)))};"
            f"bidir={int(bool(getattr(configs, 'skill_hub_bidirectional', False)))};"
            f"tasks={int(ctx['num_tasks'])};workers={int(len(snapshot['worker_free_time']))};"
            f"worker_topology={topology_digest}"
        )
        if reusable_state is not None:
            if not reuse_resource_topology:
                raise ValueError("传入 reusable_state 时必须显式启用 reuse_resource_topology")
            cached_key = getattr(reusable_state, "apal_resource_topology_key", None)
            if cached_key != topology_key:
                raise ValueError(
                    "可复用观测的静态拓扑与当前快照不一致: "
                    f"cached={cached_key!r}, current={topology_key!r}"
                )
            data = reusable_state
        else:
            data = ctx['base_data'].clone()
        
        # 1. 重建任务节点特征
        snapshot_task_x = snapshot.get('base_task_x')
        task_x = (
            torch.as_tensor(snapshot_task_x).clone()
            if snapshot_task_x is not None
            else ctx['base_task_x'].clone()
        )
        if task_x.shape != ctx['base_task_x'].shape:
            raise ValueError(
                "向量环境快照任务特征形状与数据集上下文不一致: "
                f"snapshot={tuple(task_x.shape)}, context={tuple(ctx['base_task_x'].shape)}"
            )
        task_x[:, 1:5] = 0.0
        task_x[torch.arange(ctx['num_tasks']), snapshot['task_status'] + 1] = 1.0
        
        snap_mat = snapshot.get('task_material_ready', np.zeros(ctx['num_tasks']))
        wait_times_t = np.maximum(0, snap_mat - snapshot['current_time'])
        task_x[:, 17] = torch.log1p(torch.tensor(wait_times_t, dtype=torch.float) / ctx['mean_task_time'])
        if task_x.size(1) >= 24 and 'baseline_start' in snapshot:
            snap_num_workers_for_task = len(snapshot['worker_free_time'])
            takt = max(1e-6, float(snapshot.get('baseline_makespan', 1.0)))
            task_x[:, 18] = torch.tensor((snapshot['baseline_start'] - snapshot['current_time']) / takt, dtype=torch.float)
            task_x[:, 19] = torch.tensor((snapshot['baseline_station'] + 1) / max(1, self.dataset_pool[ctx_idx]['base_station_x'].shape[0]), dtype=torch.float)
            task_x[:, 20] = torch.tensor(snapshot['baseline_team_size'] / max(1, snap_num_workers_for_task), dtype=torch.float)
            task_x[:, 21] = torch.tensor(snapshot['baseline_frozen'], dtype=torch.float)
            task_x[:, 22] = torch.tensor((snap_mat > snapshot.get('reschedule_start_time', 0.0) + 1e-9).astype(float), dtype=torch.float)
            cur_t = float(snapshot['current_time'])
            snap_status = snapshot['task_status']
            for task_id in range(task_x.size(0)):
                base_start = float(snapshot['baseline_start'][task_id])
                if cur_t > base_start + 1e-9 and snap_status[task_id] != 2:
                    task_x[task_id, 23] = float(cur_t - base_start) / takt
        
        data['task'].x = task_x
        
        # 2. 重建工人节点特征
        snap_num_workers = len(snapshot['worker_free_time'])
        worker_x = torch.as_tensor(snapshot['base_worker_x']).clone()
        worker_layout = resolve_worker_feature_layout(configs)
        assert worker_x.size(1) == worker_layout.total_dim, (
            f"快照工人特征维度错误: {worker_x.size(1)} != {worker_layout.total_dim}"
        )
        
        wait_times_w = np.maximum(0, snapshot['worker_free_time'] - snapshot['current_time'])
        worker_x[:, worker_layout.wait_idx] = torch.log1p(
            torch.tensor(wait_times_w, dtype=torch.float) / ctx['mean_task_time']
        )
        
        is_free_bool = (snapshot['worker_free_time'] <= snapshot['current_time'])
        worker_x[:, worker_layout.free_idx] = torch.tensor(is_free_bool, dtype=torch.float)
        
        worker_x[:, worker_layout.lock_slice] = 0.0
        snap_locks = snapshot['worker_locks']
        lock_indices = torch.tensor(snap_locks, dtype=torch.long).clamp(max=7)
        worker_x[torch.arange(snap_num_workers), worker_layout.lock_start + lock_indices] = 1.0
        
        snap_cum = snapshot.get('worker_cumulative_work', np.zeros(snap_num_workers))
        snap_last = snapshot.get('worker_last_busy_end', np.zeros(snap_num_workers))
        for w in range(snap_num_workers):
            cum_work = snap_cum[w]
            last_end = snap_last[w]
            if last_end > 0 and snapshot['current_time'] > last_end:
                idle_time = snapshot['current_time'] - last_end
                recovery_ratio = getattr(configs, 'fatigue_recovery_ratio', 0.5)
                cum_work = max(0.0, cum_work - idle_time * recovery_ratio)
            
            alpha = getattr(configs, 'fatigue_threshold_hours', 4.0)
            beta = getattr(configs, 'fatigue_decay_slope', 0.05)
            f_min = getattr(configs, 'fatigue_efficiency_floor', 0.60)
            overtime = max(0.0, cum_work - alpha)
            fatigue_f = max(f_min, 1.0 - beta * overtime / (alpha * 2))
            worker_x[w, worker_layout.fatigue_idx] = fatigue_f
            
        data['worker'].x = worker_x
        if reusable_state is not None:
            if bool(getattr(configs, "use_skill_hub", False)):
                skill_x = build_skill_features(worker_x, int(configs.num_skill_types))
                if skill_x.size(1) != int(configs.skill_feat_dim):
                    raise ValueError(
                        f"skill_feat_dim 配置错误: {configs.skill_feat_dim}，"
                        f"实际需要 {skill_x.size(1)}"
                    )
                data["skill"].x = skill_x
        else:
            topology = None
            if bool(getattr(configs, "use_skill_hub", False)):
                topology_key_worker = snapshot.get("worker_topology_key") or worker_topology_key(
                    worker_x,
                    int(configs.num_skill_types),
                )
                worker_edges = self._worker_skill_topology_cache.get(topology_key_worker)
                if worker_edges is None:
                    worker_edges = build_worker_skill_edges(
                        worker_x,
                        int(configs.num_skill_types),
                    )
                    self._worker_skill_topology_cache[topology_key_worker] = worker_edges
                topology = SkillHubTopology(
                    worker_to_skill=worker_edges,
                    skill_to_task=ctx["task_skill_edge_index"],
                )
            apply_resource_graph(
                data,
                task_x,
                worker_x,
                configs,
                skill_hub_topology=topology,
            )
        
        # 3. 重建站位特征
        num_stations = len(snapshot['station_loads'])
        station_x = ctx['base_station_x'].clone()
        station_x[:, 0] = torch.tensor(snapshot['station_loads'], dtype=torch.float) / max(1.0, ctx['ideal_station_load'])
        
        # 重建 Relative Load Competition 特征 (station_x[:, 5] 和 [:, 6])
        snap_loads = snapshot['station_loads']
        sum_loads = np.sum(snap_loads)
        max_load = np.max(snap_loads)
        station_x[:, 5] = torch.tensor(snap_loads / (sum_loads + 1e-6), dtype=torch.float)
        station_x[:, 6] = torch.tensor(snap_loads / (max_load + 1e-6), dtype=torch.float)
        
        max_slots = getattr(configs, 'max_slots_per_station', 3)
        snap_slots = snapshot.get('station_available_slots', np.full(num_stations, max_slots))
        station_x[:, 7] = torch.tensor(snap_slots, dtype=torch.float) / max_slots
        
        _fill_station_macro_features(station_x, snap_locks, is_free_bool)
        
        # 重建站位槽位释放时间特征 (station_x[s, 4])
        station_finish_lists = [[] for _ in range(num_stations)]
        for t_id, s_id, team, start_t, finish_t in snapshot['assigned_tasks']:
            if s_id != -1 and finish_t > snapshot['current_time']:
                station_finish_lists[s_id].append(finish_t)
                
        import math
        for s in range(num_stations):
            # 计算槽位释放时间并应用对数归一化
            lst = station_finish_lists[s]
            lst.sort()
            allowed_slots = snap_slots[s]
            if len(lst) >= allowed_slots:
                wait_time_s = max(0.0, lst[0] - snapshot['current_time'])
            else:
                wait_time_s = 0.0
            station_x[s, 4] = math.log1p(wait_time_s / ctx['mean_task_time'])
            
        data['station'].x = station_x
        
        # 4. 重建指派关系边
        ts_src, ts_dst, tw_src, tw_dst = [], [], [], []
        for t_id, s_id, team, _, _ in snapshot['assigned_tasks']:
            if s_id != -1:
                ts_src.append(t_id)
                ts_dst.append(s_id)
                for w_id in team:
                    tw_src.append(t_id)
                    tw_dst.append(w_id)
                    
        if ts_src:
            t_s_edge = torch.tensor([ts_src, ts_dst], dtype=torch.long)
            s_t_edge = torch.stack([t_s_edge[1], t_s_edge[0]], dim=0)
        else:
            t_s_edge = torch.empty((2, 0), dtype=torch.long)
            s_t_edge = torch.empty((2, 0), dtype=torch.long)
            
        data['task', 'assigned_to', 'station'].edge_index = t_s_edge
        data['station', 'has_task', 'task'].edge_index = s_t_edge
        
        if tw_src:
            t_w_edge = torch.tensor([tw_src, tw_dst], dtype=torch.long)
        else:
            t_w_edge = torch.empty((2, 0), dtype=torch.long)
             
        data['task', 'done_by', 'worker'].edge_index = t_w_edge
        
        data.apal_resource_topology_key = topology_key
        return data


class VectorEnv:
    """
    多进程向量化环境封装器 (Subprocess Vectorized Environment Wrapper)。
    利用多进程双向管道实现 CPU 并行步进，以最大化 Ryzen 9 7945HX 多核 CPU 的并行优势。
    """
    def __init__(
        self,
        make_env_fn: Callable[[int], Any],
        num_envs: int = 4,
        max_workers: Any = None,
        start_method: Optional[str] = None,
        worker_threads: Any = "auto",
        init_timeout_sec: float = 120.0,
        command_timeout_sec: float = 120.0,
    ):
        self.num_envs = num_envs
        self.closed = False
        self.start_method = start_method
        self.init_timeout_sec = float(init_timeout_sec)
        self.command_timeout_sec = float(command_timeout_sec)
        self._mp_ctx = mp.get_context(start_method) if start_method else mp.get_context()
        self.worker_threads = self._resolve_worker_threads(worker_threads, num_envs)
        self.worker_audits: List[dict] = []
        
        # 创建双向管道
        self.parent_conns, self.child_conns = zip(*[self._mp_ctx.Pipe() for _ in range(num_envs)])
        
        # 启动守护子进程
        self.processes = []
        for i in range(num_envs):
            p = self._mp_ctx.Process(
                target=_worker,
                args=(self.child_conns[i], make_env_fn, i, self.worker_threads),
                daemon=True,
            )
            self.processes.append(p)
            p.start()
            
        # 收集子进程初始化和静态配置的反馈
        self.envs = []
        try:
            for i in range(num_envs):
                conn = self.parent_conns[i]
                process = self.processes[i]
                if not conn.poll(self.init_timeout_sec):
                    state = "alive" if process.is_alive() else f"exitcode={process.exitcode}"
                    raise TimeoutError(
                        f"VectorEnv worker {i} 初始化超时 "
                        f"({self.init_timeout_sec:.1f}s, {state})"
                    )
                status, info = conn.recv()
                if status != "INIT_OK":
                    raise VectorEnvWorkerError(f"VectorEnv worker {i} 初始化失败:\n{info}")
                proxy = EnvProxy(conn, process, i, self.command_timeout_sec)
                proxy.update_static_properties(info)
                self.envs.append(proxy)
                self.worker_audits.append(dict(info.get("worker_audit", {})))
        except Exception:
            self.close()
            raise

    @staticmethod
    def _resolve_worker_threads(worker_threads: Any, num_envs: int) -> int:
        if worker_threads is None or str(worker_threads).lower() == "auto":
            # 仿真环境步进主要为纯 Python 调度逻辑与小张量运算。
            # 在多核 CPU (如 16核/32线程) 上，若每个 worker 分配 (cpu_count // num_envs) 即 8 线程，
            # 4 个 worker 将占用全部 32 线程，导致线程剧烈抢占与上下文切换，并挤占主进程 PyTorch/Lightning 的计算资源。
            # 因此将 auto 默认上限设定为 1~2 线程。
            available_cores = os.cpu_count() or 1
            calculated = available_cores // max(1, int(num_envs))
            return max(1, min(2, calculated))
        try:
            return max(1, int(worker_threads))
        except (TypeError, ValueError):
            return 1

    def _send_worker(self, index: int, command: tuple[str, Any]) -> None:
        if self.closed:
            raise VectorEnvWorkerError(
                "VectorEnv 已关闭或因 worker 失败而失效，不可复用"
            )
        try:
            self.parent_conns[index].send(command)
        except Exception as exc:
            self.close()
            raise VectorEnvWorkerError(
                f"VectorEnv worker {index} 命令发送失败，环境已失效且不可复用"
            ) from exc

    def _recv_worker(self, index: int, operation: str) -> Any:
        conn = self.parent_conns[index]
        process = self.processes[index]
        if not conn.poll(self.command_timeout_sec):
            exit_code = process.exitcode
            state = "alive" if process.is_alive() else f"exitcode={exit_code}"
            self.close()
            raise TimeoutError(
                f"VectorEnv worker {index} 执行 {operation} 超时 "
                f"({self.command_timeout_sec:.1f}s, {state})"
            )
        try:
            status, value = conn.recv()
        except Exception as exc:
            self.close()
            raise VectorEnvWorkerError(
                f"VectorEnv worker {index} 连接异常，环境已失效"
            ) from exc
        if status != "OK":
            self.close()
            raise VectorEnvWorkerError(
                f"VectorEnv worker {index} 执行 {operation} 失败:\n{value}"
            )
        return value

    def _recv_workers_unordered(self, target_indices: list[int], operation: str) -> dict[int, Any]:
        """使用 wait() 乱序收集所有 worker 的返回结果，消除固定顺序遍历的伪队头阻塞。"""
        if not target_indices:
            return {}
        pending_conns = {self.parent_conns[idx]: idx for idx in target_indices}
        results: dict[int, Any] = {}
        deadline = time.monotonic() + self.command_timeout_sec

        while pending_conns:
            remaining_time = max(0.0, deadline - time.monotonic())
            ready_conns = wait(list(pending_conns.keys()), timeout=remaining_time)
            if not ready_conns:
                timed_out_indices = list(pending_conns.values())
                self.close()
                raise TimeoutError(
                    f"VectorEnv worker {timed_out_indices} 执行 {operation} 超时 "
                    f"({self.command_timeout_sec:.1f}s)"
                )
            for conn in ready_conns:
                idx = pending_conns.pop(conn)
                try:
                    status, value = conn.recv()
                except Exception as exc:
                    self.close()
                    raise VectorEnvWorkerError(
                        f"VectorEnv worker {idx} 连接异常，环境已失效"
                    ) from exc
                if status != "OK":
                    self.close()
                    raise VectorEnvWorkerError(
                        f"VectorEnv worker {idx} 执行 {operation} 失败:\n{value}"
                    )
                results[idx] = value
        return results

    def reset_all(self, randomize_duration: bool = False, randomize_workers: bool = False) -> List[Any]:
        """异步广播复位指令并重新构建各子环境的初始 HeteroData 异构图"""
        for i in range(self.num_envs):
            self._send_worker(i, ('reset', {'randomize_duration': randomize_duration, 'randomize_workers': randomize_workers, 'seed': None}))
        
        worker_results = self._recv_workers_unordered(list(range(self.num_envs)), "reset")
        results = []
        for i in range(self.num_envs):
            snapshot, dynamic_info = worker_results[i]
            snapshot = self.envs[i]._enrich_snapshot(snapshot)
            self.envs[i].update_dynamic_properties(dynamic_info)
            self.envs[i].update_static_properties(dynamic_info)
            results.append(self.envs[i].rebuild_state_from_snapshot(snapshot))
        return results

    def reset_rollout_all(
        self,
        randomize_duration: bool = False,
        randomize_workers: bool = False,
    ) -> tuple[
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        List[dict],
    ]:
        """复位后一次返回 rollout 所需 masks 与 snapshot，不构造或传输 HeteroData。"""
        request = {
            'randomize_duration': bool(randomize_duration),
            'randomize_workers': bool(randomize_workers),
        }
        for index in range(self.num_envs):
            self._send_worker(index, ('reset_rollout', request))
        worker_results = self._recv_workers_unordered(
            list(range(self.num_envs)),
            "reset_rollout",
        )
        masks_list = []
        snapshots = []
        for index in range(self.num_envs):
            masks, snapshot, dynamic_info = worker_results[index]
            snapshot = self.envs[index]._enrich_snapshot(snapshot)
            self.envs[index].update_dynamic_properties(dynamic_info)
            self.envs[index].update_static_properties(dynamic_info)
            masks_list.append(tuple(torch.from_numpy(mask) for mask in masks))
            snapshots.append(snapshot)
        return masks_list, snapshots

    def reset_indices(
        self,
        requests: dict[int, dict[str, Any]],
    ) -> dict[int, Any]:
        """按索引复位环境；每个环境可使用独立 seed。"""
        target_indices = sorted(int(index) for index in requests)
        if not target_indices:
            return {}
        for index in target_indices:
            self._send_worker(index, ("reset", dict(requests[index])))
        worker_results = self._recv_workers_unordered(target_indices, "reset")
        results: dict[int, Any] = {}
        for index in target_indices:
            snapshot, dynamic_info = worker_results[index]
            snapshot = self.envs[index]._enrich_snapshot(snapshot)
            self.envs[index].update_dynamic_properties(dynamic_info)
            self.envs[index].update_static_properties(dynamic_info)
            results[index] = self.envs[index].rebuild_state_from_snapshot(snapshot)
        return results

    def step_all(self, actions: List[Any]) -> Tuple[List[Any], List[float], List[bool], List[dict]]:
        """异步发送物理步进指令并收集步进后的全部观测及奖励数据"""
        for i in range(self.num_envs):
            self._send_worker(i, ('step', actions[i]))
                
        worker_results = self._recv_workers_unordered(list(range(self.num_envs)), "step")
        results = []
        for i in range(self.num_envs):
            snapshot, reward, done, info = worker_results[i]
            snapshot = self.envs[i]._enrich_snapshot(snapshot)
            if 'dynamic_info' in info:
                self.envs[i].update_dynamic_properties(info.pop('dynamic_info'))
            results.append((self.envs[i].rebuild_state_from_snapshot(snapshot), reward, done, info))
                
        next_states = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        return next_states, rewards, dones, infos

    def step_snapshot_all(self, actions: List[Any]) -> Tuple[List[dict], List[float], List[bool], List[dict]]:
        """异步步进并返回轻量 snapshot，主进程本地 rebuild 以消除 HeteroData 跨进程序列化开销"""
        for i in range(self.num_envs):
            self._send_worker(i, ('step_snapshot', actions[i]))
                
        worker_results = self._recv_workers_unordered(list(range(self.num_envs)), "step_snapshot")
        results = []
        for i in range(self.num_envs):
            snap, reward, done, info = worker_results[i]
            snap = self.envs[i]._enrich_snapshot(snap)
            if 'dynamic_info' in info:
                self.envs[i].update_dynamic_properties(info.pop('dynamic_info'))
            results.append((snap, reward, done, info))
                
        snapshots = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        return snapshots, rewards, dones, infos

    def step_snapshot_indices(
        self,
        actions: dict[int, Any],
    ) -> dict[int, tuple[dict, float, bool, dict]]:
        """只步进指定环境，返回值按环境索引组织。"""
        target_indices = sorted(int(index) for index in actions)
        if not target_indices:
            return {}
        for index in target_indices:
            self._send_worker(index, ("step_snapshot", actions[index]))
        worker_results = self._recv_workers_unordered(target_indices, "step_snapshot")
        results: dict[int, tuple[dict, float, bool, dict]] = {}
        for index in target_indices:
            snapshot, reward, done, info = worker_results[index]
            snapshot = self.envs[index]._enrich_snapshot(snapshot)
            if "dynamic_info" in info:
                self.envs[index].update_dynamic_properties(info.pop("dynamic_info"))
            results[index] = (snapshot, float(reward), bool(done), info)
        return results

    def step_rollout_indices(
        self,
        actions: dict[int, Any],
    ) -> dict[
        int,
        tuple[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            dict,
            float,
            bool,
            dict,
        ],
    ]:
        """只步进活跃环境，并在同一次 IPC 中返回下一 masks 与 snapshot。"""
        target_indices = sorted(int(index) for index in actions)
        if not target_indices:
            return {}
        for index in target_indices:
            self._send_worker(index, ('step_rollout', actions[index]))
        worker_results = self._recv_workers_unordered(target_indices, "step_rollout")
        results = {}
        for index in target_indices:
            masks, snapshot, reward, done, info = worker_results[index]
            snapshot = self.envs[index]._enrich_snapshot(snapshot)
            if 'dynamic_info' in info:
                self.envs[index].update_dynamic_properties(info.pop('dynamic_info'))
            results[index] = (
                tuple(torch.from_numpy(mask) for mask in masks),
                snapshot,
                float(reward),
                bool(done),
                info,
            )
        return results

    def get_masks_all(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """异步收集各进程环境的动作空间掩码"""
        for i in range(self.num_envs):
            self._send_worker(i, ('get_masks', None))
        
        worker_results = self._recv_workers_unordered(list(range(self.num_envs)), "get_masks")
        results = []
        for i in range(self.num_envs):
            masks = worker_results[i]
            results.append(tuple(torch.from_numpy(mask) for mask in masks))
        return results

    def get_masks_and_snapshots_all(self):
        """合并 IPC：一次命令返回 masks + snapshot + dynamic_info，替代 get_masks_all() + 逐环境 get_state_snapshot()"""
        for i in range(self.num_envs):
            self._send_worker(i, ('get_rollout_state', None))
        
        worker_results = self._recv_workers_unordered(list(range(self.num_envs)), "get_rollout_state")
        masks_list = []
        snapshots = []
        for i in range(self.num_envs):
            masks, snap, dynamic_info = worker_results[i]
            snap = self.envs[i]._enrich_snapshot(snap)
            self.envs[i].update_dynamic_properties(dynamic_info)
            masks_list.append(tuple(torch.from_numpy(mask) for mask in masks))
            snapshots.append(snap)
        return masks_list, snapshots

    def get_rollout_state_indices(self, indices: List[int]) -> dict[int, tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict]]:
        """批量获取指定环境的 masks 与 snapshot。"""
        target_indices = [int(index) for index in indices]
        if not target_indices:
            return {}
        for index in target_indices:
            self._send_worker(index, ('get_rollout_state', None))

        worker_results = self._recv_workers_unordered(target_indices, "get_rollout_state")
        results = {}
        for index in target_indices:
            masks, snap, dynamic_info = worker_results[index]
            snap = self.envs[index]._enrich_snapshot(snap)
            self.envs[index].update_dynamic_properties(dynamic_info)
            results[index] = (tuple(torch.from_numpy(mask) for mask in masks), snap)
        return results

    def try_wait_for_resources_all(self) -> List[bool]:
        """异步收集并推送时间推移操作"""
        indexed = self.try_wait_for_resources_indices(list(range(self.num_envs)))
        return [indexed[i] for i in range(self.num_envs)]

    def try_wait_for_resources_indices(self, indices: List[int]) -> dict[int, bool]:
        """批量推进指定环境的资源等待。"""
        target_indices = [int(index) for index in indices]
        if not target_indices:
            return {}
        for index in target_indices:
            self._send_worker(index, ('try_wait_for_resources', None))

        worker_results = self._recv_workers_unordered(target_indices, "try_wait_for_resources")
        results = {}
        for index in target_indices:
            res, dynamic_info = worker_results[index]
            self.envs[index].update_dynamic_properties(dynamic_info)
            results[index] = bool(res)
        return results

    def wait_rollout_indices(
        self,
        indices: List[int],
    ) -> dict[
        int,
        tuple[
            bool,
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            dict,
        ],
    ]:
        """等待资源并在同一次 IPC 中返回推进后的 masks 与 snapshot。"""
        target_indices = [int(index) for index in indices]
        if not target_indices:
            return {}
        for index in target_indices:
            self._send_worker(index, ('wait_rollout', None))
        worker_results = self._recv_workers_unordered(target_indices, "wait_rollout")
        results = {}
        for index in target_indices:
            waited, masks, snapshot, dynamic_info = worker_results[index]
            snapshot = self.envs[index]._enrich_snapshot(snapshot)
            self.envs[index].update_dynamic_properties(dynamic_info)
            results[index] = (
                bool(waited),
                tuple(torch.from_numpy(mask) for mask in masks),
                snapshot,
            )
        return results

    def get_state_snapshot_all(self) -> List[dict]:
        """获取所有环境当前的静态状态缓存"""
        return [env.get_state_snapshot() for env in self.envs]

    def switch_dataset_all(self, idx: int):
        """同步广播命令给所有并行子进程切换相同的数据集"""
        for i in range(self.num_envs):
            self._send_worker(i, ('switch_dataset', idx))
            
        worker_results = self._recv_workers_unordered(list(range(self.num_envs)), "switch_dataset")
        for i in range(self.num_envs):
            val = worker_results[i]
            self.envs[i].update_static_properties(val)
            self.envs[i].update_dynamic_properties(val)

    def switch_dataset_indices(self, dataset_indices: dict[int, int]) -> None:
        """按环境索引切换不同数据集，支持多规模并行 rollout。"""
        target_indices = sorted(int(index) for index in dataset_indices)
        if not target_indices:
            return
        for index in target_indices:
            self._send_worker(
                index,
                ("switch_dataset", int(dataset_indices[index])),
            )
        worker_results = self._recv_workers_unordered(
            target_indices,
            "switch_dataset",
        )
        for index in target_indices:
            value = worker_results[index]
            self.envs[index].update_static_properties(value)
            self.envs[index].update_dynamic_properties(value)

    def close(self):
        if self.closed:
            return
        for conn, process in zip(
            getattr(self, "parent_conns", ()),
            getattr(self, "processes", ()),
        ):
            if process.is_alive():
                try:
                    conn.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        deadline = time.monotonic() + 2.0
        for p in getattr(self, "processes", ()):
            if p.is_alive():
                p.join(timeout=max(0.0, deadline - time.monotonic()))
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
        for conn in getattr(self, "parent_conns", ()):
            try:
                conn.close()
            except Exception:
                pass
        self.closed = True
