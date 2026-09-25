"""Task 5.4 训练与验证轨迹数据集收集专项单元测试。

验证点：
1. 32 维紧凑物理特征提取：向量维度无误，无 NaN / Inf；
2. 单轨迹数据采集完整性：决策步数、估计时间与实际脉动转站时刻闭环；
3. 监督残差标签计算精度：断言 label_y == (actual_transfer_time - estimated_cmax) / H_0；
4. 数据集文件持久化与加载恢复 (torch.save / torch.load)。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch

from envs.work3.environment import AirLineEnvWork3
from models.work3.heuristic_agent import HeuristicAgentWork3
from models.work3.heuristic_estimator import compute_cycle_heuristic_cmax
import scripts.work3.collect_validation_trajectories as collection_module
from scripts.work3.collect_validation_trajectories import (
    collect_single_trajectory,
    extract_compact_state_features,
)
from utils.work3.uniform_baseline_warmup import UniformBaselineWarmupResult


@pytest.fixture
def baseline_path() -> str:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return str(path)


def test_compact_state_feature_extraction(baseline_path: str) -> None:
    """测试 32 维精炼宏观与微观物理特征提取。"""
    env = AirLineEnvWork3(baseline_json_path=baseline_path)
    env.reset()

    est = compute_cycle_heuristic_cmax(env.state)
    feat = extract_compact_state_features(env.state, est)

    assert feat.shape == (32,)
    assert not torch.isnan(feat).any(), "特征向量不得含有 NaN"
    assert not torch.isinf(feat).any(), "特征向量不得含有 Inf"


def test_trajectory_collection_and_label_consistency(baseline_path: str) -> None:
    """测试单轨迹数据采集及其监督标签 y = (P_q - P_q^h)/H_0 的严格一致性。"""
    agent = HeuristicAgentWork3()
    traj = collect_single_trajectory(agent, baseline_path=baseline_path, scenario=None, trajectory_id=1)

    assert traj["trajectory_id"] == 1
    assert traj["total_steps"] > 0
    assert traj["transfer_count"] == 14
    h0 = traj["h0"]

    # 验证每步样本字段及残差标签
    for step in traj["steps"]:
        assert "state_feat" in step
        assert step["state_feat"].shape == (32,)
        p_actual = step["actual_transfer_time"]
        p_est = step["estimated_cmax"]
        expected_y = (p_actual - p_est) / h0
        assert abs(step["label_y"] - expected_y) < 1e-5, "监督残差标签计算不一致"


def test_collection_loads_fixed_scenario_only_after_uniform_warmup(
    baseline_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopAtFirstPolicyDecision(Exception):
        pass

    scenario = {
        "scenario_id": "COLLECTION_AFTER_WARMUP",
        "tau": 1000.0,
        "recovery_time": 1001.0,
        "affected_task_keys": ["missing-task"],
    }
    warmup_results: list[UniformBaselineWarmupResult] = []
    scenario_loads: list[tuple[float, float]] = []
    policy_observations: list[tuple[float, bool]] = []
    original_warmup = collection_module.run_uniform_baseline_warmup
    original_load_scenario = AirLineEnvWork3.load_scenario

    def record_warmup(
        env: AirLineEnvWork3,
        **kwargs: object,
    ) -> UniformBaselineWarmupResult:
        result = original_warmup(env, **kwargs)
        warmup_results.append(result)
        return result

    def record_scenario_load(
        env: AirLineEnvWork3,
        candidate: dict[str, object],
    ) -> None:
        scenario_loads.append((float(env.state.current_time), float(candidate["tau"])))
        original_load_scenario(env, candidate)

    class ProbeAgent:
        def select_action(self, env: AirLineEnvWork3) -> None:
            policy_observations.append(
                (float(env.state.current_time), bool(env.disturbance_event_triggered))
            )
            raise StopAtFirstPolicyDecision

    monkeypatch.setattr(collection_module, "run_uniform_baseline_warmup", record_warmup)
    monkeypatch.setattr(AirLineEnvWork3, "load_scenario", record_scenario_load)
    with pytest.raises(StopAtFirstPolicyDecision):
        collect_single_trajectory(
            ProbeAgent(),  # type: ignore[arg-type]
            baseline_path=baseline_path,
            scenario=scenario,
            trajectory_id=2,
            warmup_mode="uniform_baseline",
        )

    assert len(warmup_results) == 1
    warmup = warmup_results[0]
    assert warmup.completed is True
    assert scenario_loads == [(warmup.completion_time, scenario["tau"])]
    assert policy_observations == [(warmup.completion_time, False)]


def test_dataset_serialization_roundtrip(baseline_path: str) -> None:
    """测试轨迹数据集序列化保存与加载恢复。"""
    agent = HeuristicAgentWork3()
    traj = collect_single_trajectory(agent, baseline_path=baseline_path, scenario=None, trajectory_id=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_traj.pt"
        torch.save([traj], save_path)

        loaded = torch.load(save_path, weights_only=False)
        assert len(loaded) == 1
        assert loaded[0]["trajectory_id"] == 0
        assert loaded[0]["total_steps"] == traj["total_steps"]
