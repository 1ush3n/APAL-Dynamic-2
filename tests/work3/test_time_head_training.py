"""Task 6.2 时间修正头辅助回归训练与离线验证单元测试 (里程碑 M4)。

验证点：
1. StepResidualDataset 数据加载与 Batch 整理维度一致性；
2. 轨迹级无时序泄漏切分 (split_trajectories_by_scenario)；
3. 模型离线训练收敛性与检查点持久化恢复；
4. 【里程碑 M4 验收测试】：在含扰动的验证集上，修正后 MAE 相对原始启发式下降 15% 以上。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch

from models.work3.time_head import TimeResidualHead
from models.work3.train_time_head import (
    StepResidualDataset,
    collate_step_batch,
    evaluate_time_head,
    split_trajectories_by_scenario,
    train_time_head,
)


def _create_synthetic_trajectory(
    traj_id: int,
    scenario_id: str,
    num_steps: int = 50,
    h0: float = 292.5,
    delay_hours: float = 20.0,
) -> dict:
    """生成包含确定性残差物理特征的合成生产轨迹。"""
    steps = []
    # 假设特征第 0 维和第 11 维分别代表时间进度和缺料特征
    for i in range(num_steps):
        feat = torch.zeros(32, dtype=torch.float)
        t_ratio = i / float(num_steps)
        feat[0] = t_ratio
        feat[11] = 0.5  # 模拟缺料存在

        # 真实转站时刻有 delay_hours 延误
        p_actual = h0 + delay_hours
        p_est = h0  # 原始启发式低估了该延误
        label_y = (p_actual - p_est) / h0

        steps.append({
            "step_idx": i,
            "cycle_idx": 1,
            "current_time": t_ratio * h0,
            "estimated_cmax": p_est,
            "actual_transfer_time": p_actual,
            "label_y": label_y,
            "state_feat": feat,
        })

    return {
        "trajectory_id": traj_id,
        "scenario_id": scenario_id,
        "h0": h0,
        "total_steps": num_steps,
        "steps": steps,
    }


def test_step_residual_dataset_and_collate() -> None:
    """测试数据集封装与批整合。"""
    trajs = [_create_synthetic_trajectory(1, "SC1", num_steps=20)]
    dataset = StepResidualDataset(trajs)
    assert len(dataset) == 20

    loader = torch.utils.data.DataLoader(dataset, batch_size=5, collate_fn=collate_step_batch)
    batch = next(iter(loader))

    assert "state_feat" in batch
    assert batch["state_feat"].shape == (5, 32)
    assert batch["label_y"].shape == (5,)
    assert batch["estimated_cmax"].shape == (5,)
    assert batch["actual_transfer_time"].shape == (5,)


def test_disjoint_trajectory_splitting() -> None:
    """测试轨迹切分的场景级独立性（无时序步跨集泄漏）。"""
    trajs = [_create_synthetic_trajectory(i, f"SC_{i}", num_steps=10) for i in range(10)]
    train_t, val_t = split_trajectories_by_scenario(trajs, val_ratio=0.3, seed=123)

    assert len(train_t) == 7
    assert len(val_t) == 3

    train_ids = {t["trajectory_id"] for t in train_t}
    val_ids = {t["trajectory_id"] for t in val_t}
    assert train_ids.isdisjoint(val_ids), "训练集与验证集轨迹 ID 存在重叠，发生数据泄漏！"


def test_identical_nominal_trajectories_are_deduplicated_and_scenario_grouped() -> None:
    """五份相同名义轨迹只能作为一个样本，且scenario不能跨训练/验证集。"""
    trajectories = [
        _create_synthetic_trajectory(index, "NOMINAL_BASELINE", num_steps=10)
        for index in range(5)
    ]
    trajectories.extend(
        _create_synthetic_trajectory(index + 5, f"SC_{index}", num_steps=10)
        for index in range(3)
    )

    train_trajs, val_trajs = split_trajectories_by_scenario(
        trajectories,
        val_ratio=0.25,
        seed=123,
    )

    train_scenarios = {item["scenario_id"] for item in train_trajs}
    val_scenarios = {item["scenario_id"] for item in val_trajs}
    nominal_copies = [
        item
        for item in train_trajs + val_trajs
        if item["scenario_id"] == "NOMINAL_BASELINE"
    ]
    assert train_scenarios.isdisjoint(val_scenarios)
    assert len(nominal_copies) == 1
    assert len(train_trajs) + len(val_trajs) == 4


def test_split_rejects_five_copies_of_only_one_scenario() -> None:
    """仅有同一确定性场景的重复轨迹时不得伪造独立验证集。"""
    trajectories = [
        _create_synthetic_trajectory(index, "NOMINAL_BASELINE", num_steps=10)
        for index in range(5)
    ]

    with pytest.raises(ValueError, match="至少需要两个独立scenario"):
        split_trajectories_by_scenario(trajectories, val_ratio=0.25, seed=123)


def test_distinct_states_from_one_training_scenario_stay_in_training() -> None:
    """同一训练事件生成的多条不同状态轨迹和全部step保持同一训练归属。"""
    trajectories = [
        _create_synthetic_trajectory(
            index,
            "TRAIN_EVENT_A",
            num_steps=3,
            delay_hours=10.0 + index,
        )
        for index in range(3)
    ]
    trajectories.extend(
        _create_synthetic_trajectory(index + 3, f"SC_{index}", num_steps=3)
        for index in range(3)
    )

    train_trajs, val_trajs = split_trajectories_by_scenario(
        trajectories,
        val_ratio=0.25,
        seed=42,
    )

    event_train = [item for item in train_trajs if item["scenario_id"] == "TRAIN_EVENT_A"]
    event_val = [item for item in val_trajs if item["scenario_id"] == "TRAIN_EVENT_A"]
    train_scenarios = {item["scenario_id"] for item in train_trajs}
    val_scenarios = {item["scenario_id"] for item in val_trajs}
    assert len(event_train) == 3
    assert sum(len(item["steps"]) for item in event_train) == 9
    assert event_val == []
    assert train_scenarios.isdisjoint(val_scenarios)


def test_collector_defaults_to_one_nominal_trajectory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """确定性基准采集默认只写一条名义轨迹，不制造重复样本数。"""
    from scripts.work3 import collect_validation_trajectories

    def fake_collect_single_trajectory(
        agent: object,
        baseline_path: str,
        scenario: dict[str, object] | None = None,
        trajectory_id: int = 0,
        warmup_mode: str = "none",
        warmup_max_steps: int | None = None,
    ) -> dict[str, object]:
        del agent, baseline_path, warmup_mode, warmup_max_steps
        return {
            "trajectory_id": trajectory_id,
            "scenario_id": "NOMINAL_BASELINE" if scenario is None else "SCENARIO",
            "h0": 1.0,
            "total_steps": 0,
            "makespan": 0.0,
            "transfer_count": 0,
            "steps": [],
        }

    monkeypatch.setattr(
        collect_validation_trajectories,
        "collect_single_trajectory",
        fake_collect_single_trajectory,
    )
    trajectories = collect_validation_trajectories.collect_all_trajectories(
        baseline_path=str(tmp_path / "baseline.json"),
        scenarios_path=str(tmp_path / "absent_scenarios.json"),
        output_path=str(tmp_path / "trajectories.pt"),
    )

    assert len(trajectories) == 1
    assert trajectories[0]["scenario_id"] == "NOMINAL_BASELINE"


def test_train_time_head_convergence_and_checkpoint() -> None:
    """测试训练循环的数值收敛与最佳检查点读写。"""
    train_trajs = [_create_synthetic_trajectory(i, f"TRAIN_{i}", num_steps=40, delay_hours=15.0) for i in range(4)]
    val_trajs = [_create_synthetic_trajectory(i + 10, f"VAL_{i}", num_steps=40, delay_hours=15.0) for i in range(2)]

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = str(Path(tmpdir) / "time_head_test.pt")
        model, metrics = train_time_head(
            train_trajectories=train_trajs,
            val_trajectories=val_trajs,
            epochs=15,
            batch_size=32,
            lr=5e-3,
            checkpoint_path=ckpt_path,
        )

        assert Path(ckpt_path).is_file(), "检查点文件未成功保存"
        assert "mae_improvement_pct" in metrics
        # 由于合成数据中原始估计有固定 15h 误差，模型训练后应显著消除该误差
        assert metrics["mae_improvement_pct"] > 50.0, f"合成数据下改善度应极其显著，得到 {metrics['mae_improvement_pct']:.2f}%"

        # 验证检查点加载恢复
        ckpt = torch.load(ckpt_path, weights_only=False)
        loaded_model = TimeResidualHead(in_dim=32, hidden_dim=64)
        loaded_model.load_state_dict(ckpt["model_state_dict"])
        loaded_metrics = evaluate_time_head(loaded_model, val_trajs)
        assert abs(loaded_metrics["mae_improvement_pct"] - metrics["mae_improvement_pct"]) < 1e-4


def test_milestone_m4_checkpoint_acceptance() -> None:
    """【里程碑 M4 硬性验收测试】：断言真实最佳检查点在验证集上 MAE 降低幅度 >= 15.0%。"""
    ckpt_path = Path("models/work3/checkpoints/time_head_best.pt")
    traj_path = Path("data/work3/val_trajectories.pt")

    if not ckpt_path.is_file() or not traj_path.is_file():
        pytest.skip("检查点或轨迹数据集尚未就绪，跳过 M4 验收测试")

    ckpt = torch.load(ckpt_path, weights_only=False)
    if ckpt.get("model_version") != "signed_residual_v1":
        pytest.skip("现有检查点属于旧门控非负残差版本")
    metrics = ckpt.get("metrics", {})
    impr = metrics.get("mae_improvement_pct", 0.0)

    assert impr >= 15.0, f"里程碑 M4 验收未通过：验证集误差改善幅度 {impr:.2f}% 低于 15.0% 门槛！"

    # 在验证集上重新评估断言
    trajectories = torch.load(traj_path, weights_only=False)
    _, val_trajs = split_trajectories_by_scenario(trajectories, val_ratio=0.25, seed=2026)

    model = TimeResidualHead(in_dim=32, hidden_dim=64)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_metrics = evaluate_time_head(model, val_trajs)
    assert eval_metrics["mae_improvement_pct"] >= 15.0, (
        f"重新评估不通过: 原始 MAE={eval_metrics['mae_raw_hours']:.3f}h, "
        f"修正 MAE={eval_metrics['mae_corrected_hours']:.3f}h, "
        f"改善幅度={eval_metrics['mae_improvement_pct']:.2f}% < 15.0%"
    )
