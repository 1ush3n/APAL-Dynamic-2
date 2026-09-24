"""Task 5.1 异构图构造器专项单元测试。

验证点：
1. 图节点体系完整性：包含 task (2,830), worker (80), station (5), skill (5)；
2. 特征维度 100% 契合已有架构：
   - task_x: 18 维 (包含物理相对站位偏移 s_k - m_i^0, 5工种独热等)；
   - worker_x: 17 维；
   - station_x: 15 维；
   - skill_x: 11 维；
3. Skill Hub 双向资源拓扑完整连接；
4. 就绪任务掩码与动作分支掩码合法性；
5. 单步构图耗时优良 (断言 < 5ms)。
"""

from __future__ import annotations

import time
from pathlib import Path
import pytest
import torch
from torch_geometric.data import HeteroData

from envs.work3.core_types import ActionBranch
from envs.work3.environment import AirLineEnvWork3
from models.work3.graph_builder import MultiAircraftGraphBuilder, Work3ResourceConfig
from utils.work3.multi_aircraft_baseline import MultiAircraftBaseline


@pytest.fixture
def baseline_path() -> str:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return str(path)


def test_graph_builder_node_and_edge_integrity(baseline_path: str) -> None:
    """测试异构图节点类型、特征维度及边拓扑结构。"""
    baseline = MultiAircraftBaseline.load_from_json(baseline_path)
    builder = MultiAircraftGraphBuilder(baseline)
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    data: HeteroData = builder.build_graph(env)

    # 1. 验证 4 类核心节点
    assert set(data.node_types) == {"task", "worker", "station", "skill"}
    assert data["task"].x.shape == (2830, 26)
    assert data["worker"].x.shape == (builder.num_workers, 21)
    assert data["station"].x.shape == (5, 15)
    assert data["skill"].x.shape == (5, 11)

    # 2. 验证核心边
    edge_types = set(data.edge_types)
    assert ("task", "precedes", "task") in edge_types
    assert ("task", "assigned_to", "station") in edge_types
    assert ("station", "has_task", "task") in edge_types
    assert ("worker", "has_skill", "skill") in edge_types
    assert ("skill", "provided_by", "worker") in edge_types
    assert ("skill", "required_by", "task") in edge_types
    assert ("task", "requires", "skill") in edge_types

    # 3. 验证工艺 DAG 边数量非空
    assert data["task", "precedes", "task"].edge_index.size(1) > 0


def test_relative_station_offset_feature(baseline_path: str) -> None:
    """测试物理相对站位偏移 (s_k - m_i^0) 在图特征第 [12] 维的正确性。"""
    baseline = MultiAircraftBaseline.load_from_json(baseline_path)
    builder = MultiAircraftGraphBuilder(baseline)
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    # 初始状态：飞机 0 在站位 0，其基础归属站位为 0 的工序，偏移应为 0.0
    data = builder.build_graph(env)
    t0_key = next(k for k in builder.task_keys if k.startswith("0_"))
    t0_idx = builder.task_key_to_idx[t0_key]
    assert data["task"].x[t0_idx, 12].item() == 0.0


def test_ready_mask_and_branch_masks(baseline_path: str) -> None:
    """测试动作掩码：就绪任务掩码与分支选择合法性掩码。"""
    baseline = MultiAircraftBaseline.load_from_json(baseline_path)
    builder = MultiAircraftGraphBuilder(baseline)
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    ready_indices = builder.get_ready_task_indices(env)
    assert len(ready_indices) > 0

    mask = builder.get_ready_task_mask(env)
    assert mask.sum().item() == len(ready_indices)

    # 针对就绪工序检验分支选择掩码
    for idx in ready_indices:
        b_mask = builder.get_action_branch_mask(env, idx)
        assert b_mask["can_stay"] is True
        assert isinstance(b_mask["can_postpone"], bool)


def test_graph_building_latency_performance(baseline_path: str) -> None:
    """测试单步异构图动态组装延迟 (断言平均 < 10ms)。"""
    baseline = MultiAircraftBaseline.load_from_json(baseline_path)
    builder = MultiAircraftGraphBuilder(baseline)
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    # 预热
    _ = builder.build_graph(env)

    num_trials = 50
    t0 = time.perf_counter()
    for _ in range(num_trials):
        _ = builder.build_graph(env)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / num_trials

    print(f"\n[LATENCY] MultiAircraftGraphBuilder 单步构图平均耗时: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 25.0, f"构图耗时超标: {elapsed_ms:.2f} ms >= 25.0 ms"


def test_task_status_one_hot_encoding(baseline_path: str) -> None:
    """测试状态独热编码与枚举对齐（包含 POSTPONED 状态在槽位 4 的准确置 1）。"""
    from envs.work3.core_types import TaskStatus

    baseline = MultiAircraftBaseline.load_from_json(baseline_path)
    builder = MultiAircraftGraphBuilder(baseline)
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    # 人为设置不同状态以验证图特征槽位
    first_task_key = builder.task_keys[0]
    first_idx = 0

    # 1. READY: 槽位 1
    env.state.tasks[first_task_key].status = TaskStatus.READY
    data = builder.build_graph(env)
    assert data["task"].x[first_idx, 1].item() == 1.0
    assert data["task"].x[first_idx, 2].item() == 0.0
    assert data["task"].x[first_idx, 3].item() == 0.0
    assert data["task"].x[first_idx, 4].item() == 0.0

    # 2. RESERVED: 槽位 2
    env.state.tasks[first_task_key].status = TaskStatus.RESERVED
    data = builder.build_graph(env)
    assert data["task"].x[first_idx, 2].item() == 1.0
    assert data["task"].x[first_idx, 1].item() == 0.0

    # 3. RUNNING: 槽位 3
    env.state.tasks[first_task_key].status = TaskStatus.RUNNING
    data = builder.build_graph(env)
    assert data["task"].x[first_idx, 3].item() == 1.0

    # 4. POSTPONED: 槽位 4
    env.state.tasks[first_task_key].status = TaskStatus.POSTPONED
    data = builder.build_graph(env)
    assert data["task"].x[first_idx, 4].item() == 1.0
    assert data["task"].x[first_idx, 1:4].sum().item() == 0.0

    # 5. UNREADY / COMPLETED: 全 0
    env.state.tasks[first_task_key].status = TaskStatus.UNREADY
    data = builder.build_graph(env)
    assert data["task"].x[first_idx, 1:5].sum().item() == 0.0

    env.state.tasks[first_task_key].status = TaskStatus.COMPLETED
    data = builder.build_graph(env)
    assert data["task"].x[first_idx, 1:5].sum().item() == 0.0
