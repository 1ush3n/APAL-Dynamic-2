"""W3-12 真实终止、截断、标签与评测失败报告反例。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from envs.work3.environment import AirLineEnvWork3
from models.work3.heuristic_agent import HeuristicAgentWork3
from models.work3.ppo_buffer import PPOTransition, RolloutBufferWork3
from models.work3.train_time_head import StepResidualDataset
from scripts.work3.collect_validation_trajectories import attach_transfer_labels
from scripts.work3.evaluate_c_vs_d import evaluate_single_trajectory


@pytest.fixture
def baseline_path() -> Path:
    path = Path("data/work3/real_283_k10_baseline.json")
    if not path.is_file():
        pytest.skip(f"{path} 不存在")
    return path


def _transition(*, terminated: bool | None = None, truncated: bool = False, done: bool = False) -> PPOTransition:
    return PPOTransition(
        state_feat=torch.zeros(32),
        time_urgency=torch.zeros(2),
        sample_record={},
        reward=0.0,
        raw_reward=0.0,
        value=1.0,
        log_prob=0.0,
        done=done,
        terminated=terminated,
        truncated=truncated,
    )


def test_truncated_rollout_bootstraps_instead_of_zeroing() -> None:
    """采样截断不是生产终止，必须使用截断后的状态价值 bootstrap。"""
    buffer = RolloutBufferWork3(gamma=0.9, gae_lambda=0.95, normalize_advantages=False)
    buffer.add(_transition(terminated=False, truncated=True, done=True))

    buffer.finish_trajectory(last_value=5.0)

    assert buffer.advantages.tolist() == [3.5]
    assert buffer.target_values.tolist() == [4.5]


def test_missing_transfer_label_is_explicitly_unavailable() -> None:
    """没有真实转站时刻时不得使用当前时刻伪造标签。"""
    records = [
        {"cycle_idx": 1, "estimated_cmax": 12.0},
        {"cycle_idx": 2, "estimated_cmax": 18.0},
    ]

    attach_transfer_labels(records, transfer_history=[10.0], h0=10.0)

    assert records[0]["label_available"] is True
    assert records[0]["actual_transfer_time"] == 10.0
    assert records[1]["label_available"] is False
    assert records[1]["actual_transfer_time"] is None
    assert records[1]["label_y"] is None


def test_time_dataset_drops_unobserved_labels() -> None:
    """离线时间头训练只能接收已有真实转站标签的样本。"""
    trajectories = [{
        "h0": 10.0,
        "steps": [
            {
                "state_feat": torch.zeros(32),
                "label_y": 0.2,
                "estimated_cmax": 12.0,
                "actual_transfer_time": 14.0,
                "current_time": 1.0,
            },
            {
                "state_feat": torch.ones(32),
                "label_y": None,
                "estimated_cmax": 20.0,
                "actual_transfer_time": None,
                "current_time": 2.0,
            },
        ],
    }]

    dataset = StepResidualDataset(trajectories)

    assert len(dataset) == 1
    assert dataset[0]["label_y"] == 0.2


def test_evaluation_decision_limit_is_not_success(baseline_path: Path) -> None:
    """达到决策上限时必须报告不完整，而不能按成功轨迹返回。"""
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    result = evaluate_single_trajectory(
        env,
        "Baseline-C",
        HeuristicAgentWork3(name="Baseline-C"),
        max_decisions=0,
    )

    assert result["success"] is False
    assert result["feasible"] is False
    assert result["termination_reason"] == "decision_limit"
    assert result["completed_tasks"] == 0


def test_heuristic_limit_has_no_fake_transfer_label(baseline_path: Path) -> None:
    """启发式轨迹被截断时，未观测周期必须保留缺失标签。"""
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    result = HeuristicAgentWork3().run_trajectory(env, max_decisions=1)

    assert result["success"] is False
    assert result["termination_reason"] == "decision_limit"
    assert any(
        record["actual_transfer_time"] is None
        for record in result["step_records"]
    )
