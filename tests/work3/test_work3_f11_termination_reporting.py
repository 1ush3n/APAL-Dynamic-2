"""W3-12 真实终止、截断、标签与评测失败报告反例。"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from envs.work3.core_types import ActionBranch, TaskStatus
from envs.work3.environment import AirLineEnvWork3
from envs.work3.event_queue import EventType
from models.work3.heuristic_agent import HeuristicAgentWork3
from models.work3.ppo_buffer import PPOTransition, RolloutBufferWork3
from models.work3.train_time_head import StepResidualDataset
from scripts.work3.collect_validation_trajectories import attach_transfer_labels
import scripts.work3.evaluate_c_vs_d as evaluation_module
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


def test_legacy_truncated_transition_does_not_treat_done_as_termination() -> None:
    """旧调用未显式传terminated时，truncated优先于done完成bootstrap。"""
    buffer = RolloutBufferWork3(gamma=0.9, gae_lambda=0.95, normalize_advantages=False)
    buffer.add(_transition(terminated=None, truncated=True, done=True))

    buffer.finish_trajectory(last_value=5.0)

    assert buffer.advantages.tolist() == [3.5]
    assert buffer.target_values.tolist() == [4.5]


def test_failed_terminal_transition_does_not_bootstrap_across_episode() -> None:
    """失败终止与自然终止一样切断bootstrap，不能接到下一episode。"""
    buffer = RolloutBufferWork3(gamma=0.9, gae_lambda=0.95, normalize_advantages=False)
    buffer.add(_transition(terminated=True, truncated=False, done=True))

    buffer.finish_trajectory(last_value=99.0)

    assert buffer.advantages.tolist() == [-1.0]
    assert buffer.target_values.tolist() == [0.0]


def test_l01_single_success_terminal_step_uses_zero_bootstrap() -> None:
    """真实终止单步样本用零后继值，TD残差为r−V。"""
    buffer = RolloutBufferWork3(gamma=0.9, gae_lambda=0.95, normalize_advantages=False)
    buffer.add(
        PPOTransition(
            state_feat=torch.zeros(32),
            time_urgency=torch.zeros(2),
            sample_record={},
            reward=2.0,
            raw_reward=2.0,
            value=1.0,
            log_prob=0.0,
            done=True,
            terminated=True,
        )
    )

    buffer.finish_trajectory(last_value=99.0)

    assert buffer.advantages.tolist() == pytest.approx([1.0])
    assert buffer.target_values.tolist() == pytest.approx([2.0])


def test_l02_single_nonterminal_step_uses_segment_bootstrap() -> None:
    """非终止采样段末步使用V_next=4，TD残差为4.6。"""
    buffer = RolloutBufferWork3(gamma=0.9, gae_lambda=0.95, normalize_advantages=False)
    buffer.add(
        PPOTransition(
            state_feat=torch.zeros(32),
            time_urgency=torch.zeros(2),
            sample_record={},
            reward=2.0,
            raw_reward=2.0,
            value=1.0,
            log_prob=0.0,
            done=False,
            terminated=False,
        )
    )

    buffer.finish_trajectory(last_value=4.0)

    assert buffer.advantages.tolist() == pytest.approx([4.6])
    assert buffer.target_values.tolist() == pytest.approx([5.6])


def test_gae_does_not_cross_adjacent_episodes_in_one_worker_segment() -> None:
    """同一worker同一采样段内，真实终止必须切断相邻episode的GAE。"""
    buffer = RolloutBufferWork3(gamma=0.9, gae_lambda=0.8, normalize_advantages=False)
    dummy_feat = torch.zeros(32)
    dummy_time = torch.zeros(2)

    for episode_id, reward, value, terminated in (
        (7, 2.0, 1.0, True),
        (8, 10.0, 3.0, False),
        (8, 20.0, 4.0, True),
    ):
        buffer.add(
            PPOTransition(
                state_feat=dummy_feat,
                time_urgency=dummy_time,
                sample_record={},
                reward=reward,
                raw_reward=reward,
                value=value,
                log_prob=0.0,
                done=terminated,
                terminated=terminated,
                worker_id=0,
                episode_id=episode_id,
                segment_id=0,
            )
        )

    buffer.finish_trajectories(last_values_by_segment={})

    assert buffer.advantages.tolist() == pytest.approx([1.0, 22.12, 16.0])
    assert buffer.target_values.tolist() == pytest.approx([2.0, 25.12, 20.0])


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


@pytest.mark.parametrize("trajectory_feasible", [True, False])
def test_evaluation_completion_on_exact_decision_limit_requires_feasibility(
    baseline_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trajectory_feasible: bool,
) -> None:
    """最后一个额度内动作真实完成批次时，不能误报decision_limit。"""
    task = SimpleNamespace(
        status=TaskStatus.READY,
        material_ready_time=0.0,
        postpone_count=0,
    )

    class OneStepCompletionEnv:
        baseline_json_path = str(baseline_path)
        tolerance = 1e-5

        def __init__(self) -> None:
            self.reset()

        def reset(self) -> None:
            task.status = TaskStatus.READY
            self.finished = False
            self.step_count = 0
            self.cumulative_cost = 0.0
            self.cost_takt = 0.0
            self.cost_time = 0.0
            self.cost_team = 0.0
            self.cost_postpone = 0.0
            self.cost_revision = 0.0
            self.disturbance_event_triggered = False
            self.disturbance_event_results = {}
            self.state = SimpleNamespace(
                tasks={"one": task},
                transfer_history=[],
                current_time=1.0,
            )

        def get_action_candidates(self) -> list[SimpleNamespace]:
            return [] if self.finished else [task]

        def step(self, action: dict[str, object]) -> tuple[dict[str, object], float, bool, bool, dict[str, str]]:
            assert action == {"test_action": True}
            self.finished = True
            self.step_count += 1
            task.status = TaskStatus.COMPLETED
            return {}, 0.0, True, False, {"termination_reason": "completed"}

        def _check_terminated(self) -> bool:
            return self.finished

    class CompleteInOneActionAgent:
        def select_action(self, _env: OneStepCompletionEnv) -> dict[str, bool]:
            return {"test_action": True}

    monkeypatch.setattr(
        evaluation_module,
        "_check_completed_trajectory_feasibility",
        lambda _env: (trajectory_feasible, {}),
    )
    env = OneStepCompletionEnv()

    result = evaluate_single_trajectory(
        env,
        "Baseline-C",
        CompleteInOneActionAgent(),
        max_decisions=1,
    )

    assert result["decisions"] == 1
    assert result["completed_tasks"] == 1
    assert result["success"] is trajectory_feasible
    assert result["feasible"] is trajectory_feasible
    assert result["termination_reason"] == "completed"


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


def test_advance_without_valid_future_event_is_failed_termination(baseline_path: Path) -> None:
    """仅有过期事件时显式推进应以deadlock失败终止，而不是停留在非终止状态。"""
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    env.reset()
    task = env.get_ready_tasks()[0]
    env.event_queue.push(
        event_type=EventType.TASK_START,
        timestamp=20.0,
        task_key=task.task_key,
        generation=task.generation,
    )
    env.event_queue.invalidate_task_events(task.task_key, task.generation + 1)

    _obs, _reward, terminated, truncated, info = env.step(
        {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    )

    assert terminated is True
    assert truncated is False
    assert env._check_terminated() is False
    assert env.state.current_time == pytest.approx(0.0)
    assert info["advanced"] is False
    assert info["success"] is False
    assert info["termination_reason"] == "deadlock"


def test_l07_deadlock_is_failed_terminal_without_bootstrap_or_transfer_label(
    baseline_path: Path,
) -> None:
    """deadlock失败终止切断价值bootstrap，未发生转站的周期不生成标签。"""
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    env.reset()
    env.event_queue.reset(start_time=env.state.current_time)
    cycle_id = env.state.current_cycle
    _obs, reward, terminated, truncated, info = env.step(
        {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}
    )

    assert terminated is True
    assert truncated is False
    assert info["success"] is False
    assert info["termination_reason"] == "deadlock"

    buffer = RolloutBufferWork3(gamma=0.9, gae_lambda=0.95, normalize_advantages=False)
    buffer.add(
        PPOTransition(
            state_feat=torch.zeros(32),
            time_urgency=torch.zeros(2),
            sample_record={},
            reward=reward,
            raw_reward=reward,
            value=1.0,
            log_prob=0.0,
            done=True,
            terminated=True,
            truncated=False,
        )
    )
    buffer.finish_trajectory(last_value=99.0)
    assert buffer.advantages.tolist() == pytest.approx([reward - 1.0])
    assert buffer.target_values.tolist() == pytest.approx([reward])

    records = [{"cycle_idx": cycle_id, "estimated_cmax": 10.0}]
    attach_transfer_labels(records, list(env.state.transfer_history), env.state.h0)
    assert records[0]["label_available"] is False
    assert records[0]["actual_transfer_time"] is None
    assert records[0]["label_y"] is None


class _AlwaysAdvanceAgent:
    def select_action(self, env: AirLineEnvWork3) -> dict[str, ActionBranch]:
        return {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}


class _AlwaysAdvanceHeuristic(HeuristicAgentWork3):
    def select_action(self, env: AirLineEnvWork3) -> dict[str, ActionBranch]:
        return {"branch": ActionBranch.ADVANCE_TO_NEXT_EVENT}


def test_l08_evaluation_propagates_software_exception_without_success_report(
    baseline_path: Path,
) -> None:
    """评测中的软件异常必须显式失败，不能被包装为成功轨迹。"""
    class FailingAgent:
        def select_action(self, _env: AirLineEnvWork3) -> dict[str, object]:
            raise RuntimeError("simulated policy failure")

    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))

    with pytest.raises(RuntimeError, match="simulated policy failure"):
        evaluate_single_trajectory(
            env,
            "Baseline-C",
            FailingAgent(),
            max_decisions=1,
        )


@pytest.mark.parametrize("entrypoint", ["evaluation", "heuristic"])
def test_trajectory_entrypoints_report_advance_deadlock_as_failure(
    baseline_path: Path,
    entrypoint: str,
) -> None:
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    if entrypoint == "evaluation":
        result = evaluate_single_trajectory(
            env,
            "Baseline-C",
            _AlwaysAdvanceAgent(),
            max_decisions=1,
        )
    else:
        result = _AlwaysAdvanceHeuristic().run_trajectory(env, max_decisions=1)

    assert result["success"] is False
    assert result["termination_reason"] == "deadlock"
    assert result["completed_tasks"] == 0


def test_l09_completed_tasks_do_not_terminate_before_aircraft_exit(
    baseline_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """所有工序状态完成但飞机仍停在末站时，评测必须判为未成功。"""
    env = AirLineEnvWork3(baseline_json_path=str(baseline_path))
    original_reset = env.reset

    def reset_with_aircraft_still_at_final_station() -> None:
        original_reset()
        for task in env.state.tasks.values():
            task.status = TaskStatus.COMPLETED
        for aircraft in env.state.aircraft.values():
            aircraft.current_station = env.state.num_stations - 1
        env.event_queue.reset(start_time=env.state.current_time)
        assert env._check_terminated() is False

    monkeypatch.setattr(env, "reset", reset_with_aircraft_still_at_final_station)

    result = evaluate_single_trajectory(
        env,
        "Baseline-C",
        HeuristicAgentWork3(name="Baseline-C"),
        max_decisions=1,
    )

    assert result["success"] is False
    assert result["termination_reason"] == "completed"
    assert result["completed_tasks"] == len(env.state.tasks)
    assert env._check_terminated() is True
    assert all(aircraft.is_completed for aircraft in env.state.aircraft.values())
    assert all(
        env.state.num_stations - 1 in aircraft.exit_times
        for aircraft in env.state.aircraft.values()
    )
