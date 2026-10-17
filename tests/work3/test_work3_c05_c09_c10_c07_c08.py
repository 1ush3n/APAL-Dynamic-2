"""S1 (Execution & Data Contract: C05, C09, C10) & S2 (Representation Input Contract: C07, C08)

Minimal reproducible tests as defined in the 2026-10-17 review report:
- T01: 无冲突预约保持 (C05)
- T02: 必要顺延传播 (C05)
- T03: 完整快照等价 (C09)
- T05: 零工时物理节点支持 (C10)
- T07 & T08: 目标站读出与 next 集合范围 (C08)
- T09: 最近摘要进入模型输入 (C07)
"""

from __future__ import annotations

import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from core.constraints import ConstraintEngine
from envs.work3.core_types import (
    ActionBranch,
    MultiAircraftState,
    TaskRuntimeState,
    TaskStatus,
    initialize_multi_aircraft_state,
)
from envs.work3.decision_snapshot import DecisionSnapshot, TeamCompletionContext, WorkerSnapshot
from envs.work3.environment import AirLineEnvWork3
from envs.work3.event_queue import EventType
from models.work3.actor_critic import (
    ActorCriticWork3,
    extract_compact_state_features,
)
from runtime.five_skill_schema import validate_explicit_five_skill_frame
from torch_geometric.data import HeteroData

BASELINE_PATH = Path("data/work3/real_283_k10_baseline.json")


def test_t01_no_conflict_reservation_retention_c05():
    """T01: 无冲突预约保持 (C05)

    When worker has task A=[20, 25] (align=1) and task B=[25, 50] (align=0, dur=25).
    A is disturbed and deferred to [60, 65].
    B does not overlap [60, 65], so B must remain at [25, 50] and NOT be prematurely moved to [1, 26]!
    """
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    env.state.current_time = 1.0

    w = 12
    task_a = env.state.tasks["0_15"]
    task_b = env.state.tasks["1_15"]

    task_a.demand = 1
    task_a.skill = 0
    task_a.current_station = 0
    task_a.status = TaskStatus.RESERVED
    task_a.assigned_team = [w]
    task_a.scheduled_start = 20.0
    task_a.execution_duration = 5.0
    task_a.duration = 5.0
    task_a.alignment = 1
    task_a.in_station_offset = 20.0

    task_b.demand = 1
    task_b.skill = 0
    task_b.current_station = 0
    task_b.status = TaskStatus.RESERVED
    task_b.assigned_team = [w]
    task_b.scheduled_start = 25.0
    task_b.execution_duration = 25.0
    task_b.duration = 25.0
    task_b.alignment = 0
    task_b.predecessors = ()

    env.state.workers[w].intervals = []
    env.state.workers[w].add_interval(20.0, 25.0, task_a.task_key)
    env.state.workers[w].add_interval(25.0, 50.0, task_b.task_key)

    # Disturb A: material ready time becomes 60.0
    task_a.material_ready_time = 60.0
    task_a.in_station_offset = 60.0
    env._passive_defer_reserved_task(task_a, cause="disturbance")

    # A is deferred to [60, 65]
    assert task_a.scheduled_start >= 60.0

    # B does not overlap [60, 65], so B must remain at [25, 50]
    assert task_b.scheduled_start == 25.0, (
        f"C05 violation: Task B was prematurely moved to {task_b.scheduled_start} "
        f"instead of remaining at 25.0!"
    )
    assert task_b.scheduled_start + task_b.execution_duration == 50.0
    assert task_b.assigned_team == [w]
    assert getattr(task_b, "alignment", 0) == 0


def test_t02_necessary_deferral_propagation_c05():
    """T02: 必要顺延传播 (C05)

    When task A is deferred and truly overlaps task B, B is postponed only as much as necessary.
    Team and alignment remain locked, and upstream task A is not displaced by backward cascade.
    """
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    env.state.current_time = 1.0

    w = 12
    task_a = env.state.tasks["0_15"]
    task_b = env.state.tasks["1_15"]

    task_a.demand = 1
    task_a.skill = 0
    task_a.current_station = 0
    task_a.status = TaskStatus.RESERVED
    task_a.assigned_team = [w]
    task_a.scheduled_start = 22.0
    task_a.execution_duration = 6.0  # [22.0, 28.0]
    task_a.duration = 6.0
    task_a.alignment = 0

    task_b.demand = 1
    task_b.skill = 0
    task_b.current_station = 0
    task_b.status = TaskStatus.RESERVED
    task_b.assigned_team = [w]
    task_b.scheduled_start = 25.0  # Truly overlaps A's interval [22.0, 28.0] on worker w
    task_b.execution_duration = 10.0
    task_b.duration = 10.0
    task_b.alignment = 1
    task_b.in_station_offset = 25.0
    task_b.predecessors = ()

    env.state.workers[w].intervals = []
    env.state.workers[w].add_interval(22.0, 28.0, task_a.task_key)

    # Task B is postponed due to true resource conflict with A [22, 28]
    env._passive_defer_reserved_task(task_b, cause="resource_conflict")

    # 1. B must be postponed to after A completes (>= 28.0)
    assert task_b.scheduled_start >= 28.0 - 1e-5, (
        f"C05 violation: Task B starts at {task_b.scheduled_start} before worker is freed at 28.0!"
    )
    # 2. B must NOT be moved backward in time
    assert task_b.scheduled_start >= 25.0, (
        f"C05 violation: Task B was moved backward in time to {task_b.scheduled_start}!"
    )
    # 3. Task A must remain at [22.0, 28.0] and NOT be displaced to 1.0 by a backward cascade!
    assert task_a.scheduled_start == 22.0, (
        f"C05 violation: Upstream Task A was displaced to {task_a.scheduled_start} by backward deferral cascade!"
    )
    # 4. Team and alignment must remain locked
    assert task_b.assigned_team == [w], "C05 violation: Task B assigned team was altered!"
    assert getattr(task_b, "alignment", 0) == 1, "C05 violation: Task B alignment was altered!"


def test_t03_complete_snapshot_equivalence_c09():
    """T03: 完整快照等价 (C09)

    Create a MultiAircraftState where a task has `alignment=1`, `committed=True`,
    and state has `station_first_cleared_times`.
    Call `snapshot()` and `restore()`.
    Assert `alignment == 1`, `committed == True`, `_state_ref` points to the restored state,
    and moving tasks after restore does not alter the snapshot.
    """
    state = initialize_multi_aircraft_state(str(BASELINE_PATH))
    task = state.tasks["0_15"]
    task.reserve(team=[12], scheduled_start=20.0, alignment=1)
    assert getattr(task, "alignment", 0) == 1
    assert getattr(task, "committed", False) is True

    state.station_first_cleared_times = {1: {0: 15.0}}

    snap = state.snapshot()

    # Check snapshot preserves fields
    snap_task = snap["tasks"]["0_15"]
    assert getattr(snap_task, "alignment", 0) == 1, (
        "C09 violation: snapshot() dropped task.alignment (became 0 or None)!"
    )
    assert getattr(snap_task, "committed", False) is True, (
        "C09 violation: snapshot() dropped task.committed (became False or None)!"
    )

    # Restore into fresh state
    restored_state = initialize_multi_aircraft_state(str(BASELINE_PATH))
    restored_state.restore(snap)

    res_task = restored_state.tasks["0_15"]
    assert getattr(res_task, "alignment", 0) == 1, (
        "C09 violation: restore() dropped task.alignment (became 0 or None)!"
    )
    assert getattr(res_task, "committed", False) is True, (
        "C09 violation: restore() dropped task.committed (became False or None)!"
    )
    assert res_task._state_ref is restored_state, (
        "C09 violation: restored task._state_ref does not reference restored_state!"
    )
    assert restored_state.station_first_cleared_times == {1: {0: 15.0}}, (
        "C09 violation: station_first_cleared_times not restored properly!"
    )

    # Moving tasks on restored state does not alter snapshot
    old_station = snap["tasks"]["0_15"].current_station
    res_task.postpone_to_next_station()
    assert res_task.current_station == old_station + 1
    assert snap["tasks"]["0_15"].current_station == old_station, (
        "C09 violation: mutating restored state mutated snapshot tasks!"
    )


def test_t05_zero_duration_physical_task_support_c10():
    """T05: 零工时物理节点支持 (C10)

    Test that physical tasks with duration == 0 are preserved (not dropped or rejected by schema/loader),
    virtual summary nodes are distinguished, and topological reachability holds.
    """
    # 1. Schema check: Physical task with duration == 0 must be allowed and distinguished from virtual nodes
    df = pd.DataFrame([
        {"序号": 0, "AO号": "VR01", "类型": 1, "专业编码": "", "工种": -1, "紧前工序AO号": "", "需求人数": 0, "加工时间/h": 0.0, "限定站位": -1, "部位容量": 0},
        {"序号": 1, "AO号": "TA01", "类型": 2, "专业编码": "A", "工种": 0, "紧前工序AO号": "", "需求人数": 1, "加工时间/h": 0.0, "限定站位": -1, "部位容量": 0},
        {"序号": 2, "AO号": "TD01", "类型": 2, "专业编码": "D", "工种": 1, "紧前工序AO号": "TA01", "需求人数": 1, "加工时间/h": 5.0, "限定站位": -1, "部位容量": 0},
        {"序号": 3, "AO号": "TB01", "类型": 2, "专业编码": "B", "工种": 2, "紧前工序AO号": "", "需求人数": 1, "加工时间/h": 5.0, "限定站位": -1, "部位容量": 0},
        {"序号": 4, "AO号": "TN01", "类型": 2, "专业编码": "N", "工种": 3, "紧前工序AO号": "", "需求人数": 1, "加工时间/h": 5.0, "限定站位": -1, "部位容量": 0},
        {"序号": 5, "AO号": "TC01", "类型": 2, "专业编码": "C", "工种": 4, "紧前工序AO号": "", "需求人数": 1, "加工时间/h": 5.0, "限定站位": -1, "部位容量": 0},
    ])
    counts = validate_explicit_five_skill_frame(df, source="test_zero_duration", require_all_skills=False)
    assert counts["0"] == 1, "C10 violation: Physical zero-duration task was not counted in skill counts!"

    # 2. ConstraintEngine check: Node 1 (physical, dur=0) must be physical, Node 0 (virtual, dur=0) must NOT be physical
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64).T
    durations = [0.0, 0.0, 5.0]
    fixed_stations = [-1, -1, -1]

    engine = ConstraintEngine.build(
        num_tasks=3,
        num_stations=5,
        edges=edges,
        durations=durations,
        fixed_stations=fixed_stations,
        node_types=[1, 2, 2],
    )
    # Virtual node 0 is not physical, physical task 1 is physical
    assert not engine.physical_mask[0], "C10 violation: Virtual node 0 misclassified as physical!"
    assert engine.physical_mask[1], "C10 violation: Zero-duration physical task 1 dropped from physical_mask!"

    # Topological reachability: Task 2 must have Task 1 as a physical predecessor
    assert 1 in engine.physical_predecessors[2], (
        "C10 violation: Topological predecessor relationship across zero-duration physical task was dropped!"
    )


def test_t07_t08_target_station_readout_and_next_set_bounds_c08():
    """T07 & T08: 目标站读出与 next 集合范围 (C08)

    Test that when an action considers staying at station s, it reads station s current.
    When considering postponing from s to s+1, it reads station s current and station s+1 next cycle summary!
    Station 4 (last station) cannot postpone out of bounds.
    """
    model = ActorCriticWork3()

    # Construct hier_dict with 5 stations: 0..4
    st_curr = torch.randn(5, 64)
    st_next = torch.randn(5, 64)
    aircraft_emb = torch.randn(10, 64)
    line_emb = torch.randn(64)

    hier_dict = {
        "aircraft": aircraft_emb,
        "station_current": st_curr,
        "station_next": st_next,
        "line": line_emb,
    }

    # Case 1: Candidate at station s = 1 considering postponing to station s+1 = 2
    # The postpone representation MUST read station 2's next cycle summary (st_next[2]),
    # NOT station 1's next cycle summary (st_next[1])!
    s = 1
    target_s = s + 1  # 2

    ac_ids = torch.tensor([0])
    st_ids = torch.tensor([s])

    # Under C08 requirement: postpone candidate representation must fuse target station (s+1) next summary:
    # readout_c08 = [ac_emb, station_current[s], station_next[s+1], line_emb]
    # In commit 54989c3, it wrongly reads station_next[s]:
    # readout_54989c3 = [ac_emb, station_current[s], station_next[s], line_emb]
    readout_c08 = torch.cat([
        aircraft_emb[ac_ids],
        hier_dict["station_current"][st_ids],
        hier_dict["station_next"][torch.tensor([target_s])],
        line_emb.unsqueeze(0),
    ], dim=-1)

    readout_54989c3 = torch.cat([
        aircraft_emb[ac_ids],
        hier_dict["station_current"][st_ids],
        hier_dict["station_next"][st_ids],
        line_emb.unsqueeze(0),
    ], dim=-1)

    embed_c08 = model.hierarchical_task_fusion(readout_c08)
    embed_54989c3 = model.hierarchical_task_fusion(readout_54989c3)

    assert not torch.allclose(embed_c08, embed_54989c3), (
        "C08 violation: Target station s+1 next summary was not distinguished from source station s next summary!"
    )

    # In 54989c3 lines 739-740:
    # st_curr_embs = hier_dict["station_current"][st_ids]
    # st_next_embs = hier_dict["station_next"][st_ids]
    # Assert that 54989c3 wrongly indexed source station s instead of target_s:
    assert st_ids.item() == s, "Candidate station ID must be source station s"
    read_index = st_ids.item()
    assert read_index == target_s, (
        f"C08 violation: Model reads station_next[{read_index}] (source station) "
        f"instead of station_next[{target_s}] (target station s+1)!"
    )

    # Case 2: Station 4 (last station, index 4) bounds
    # Station 4 cannot postpone out of bounds: target station 4 + 1 = 5 does not exist.
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    st4_tasks = [t for t in env.state.tasks.values() if t.current_station == 4]
    assert len(st4_tasks) > 0, "No task at station 4 found!"
    task_st4 = st4_tasks[0]
    can_reserve, can_postpone = env.get_action_branch_mask(task_st4)
    assert can_postpone is False, "C08 violation: Station 4 task is allowed to postpone out of bounds!"


def test_t09_recent_summary_enters_model_input_c07():
    """T09: 最近摘要进入模型输入 (C07)

    Test that recent changes/events are reflected in `compact_state_features` / decision snapshot,
    and future unrevealed events do not alter current input.
    """
    state = initialize_multi_aircraft_state(str(BASELINE_PATH))
    state.current_time = 10.0
    feat_base = extract_compact_state_features(state, estimated_cmax=100.0)

    # 1. Recent changes/events must be reflected in compact_state_features / decision snapshot
    state.last_action_branch = int(ActionBranch.POSTPONE)
    state.recent_postponed_count += 1
    state.micro_steps_in_timestamp += 1

    feat_after_recent = extract_compact_state_features(state, estimated_cmax=100.0)
    diff = torch.norm(feat_after_recent - feat_base).item()
    assert diff > 1e-4, (
        f"C07 violation: Recent state changes (branch={state.last_action_branch}, "
        f"postponed_count={state.recent_postponed_count}) produced zero change in model input features!"
    )

    # 2. Future unrevealed events must NOT alter current decision input
    env = AirLineEnvWork3(baseline_json_path=BASELINE_PATH)
    env.reset()
    env.state.current_time = 10.0
    feat_before_future = extract_compact_state_features(env.state, estimated_cmax=100.0)

    # Add future event far in future (t=100.0)
    env.event_queue.push(
        event_type=EventType.MATERIAL_READY,
        timestamp=100.0,
        task_key="0_15",
    )
    feat_after_future = extract_compact_state_features(env.state, estimated_cmax=100.0)
    assert torch.allclose(feat_before_future, feat_after_future), (
        "C07 violation: Future unrevealed event altered current decision input features!"
    )
