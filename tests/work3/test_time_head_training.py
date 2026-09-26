"""Task 6.2 时间修正头辅助回归训练与离线验证单元测试 (里程碑 M4)。

验证点：
1. StepResidualDataset 数据加载与 Batch 整理维度一致性；
2. 轨迹级无时序泄漏切分 (split_trajectories_by_scenario)；
3. 模型离线训练收敛性与检查点持久化恢复；
4. 【里程碑 M4 验收测试】：在含扰动的验证集上，修正后 MAE 相对原始启发式下降 15% 以上。
"""

from __future__ import annotations

import json
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


def _write_official_split_inputs(
    tmp_path: Path,
    train_ids: list[str],
    validation_ids: list[str],
    test_ids: list[str],
    train_artifact_ids: list[str] | None = None,
) -> dict[str, Path]:
    """构造独立事件清单和对应小型监督轨迹文件。"""
    manifests = {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids,
    }
    paths: dict[str, Path] = {}
    for split_name, scenario_ids in manifests.items():
        manifest_path = tmp_path / f"{split_name}.json"
        manifest_path.write_text(
            json.dumps({"scenarios": [{"scenario_id": item} for item in scenario_ids]}),
            encoding="utf-8",
        )
        paths[f"{split_name}_split"] = manifest_path

    for split_name, scenario_ids in (
        ("train", train_ids if train_artifact_ids is None else train_artifact_ids),
        ("validation", validation_ids),
    ):
        artifact_path = tmp_path / f"{split_name}_trajectories.pt"
        torch.save(
            [
                _create_synthetic_trajectory(index, scenario_id)
                for index, scenario_id in enumerate(scenario_ids)
            ],
            artifact_path,
        )
        paths[f"{split_name}_trajectories"] = artifact_path
    return paths


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


def test_time_predictor_rejects_legacy_pooled_trajectory_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """时间头入口不得再对采集后混合轨迹执行随机训练/验证切分。"""
    import sys

    from scripts.work3 import train_time_predictor

    pooled_path = tmp_path / "pooled.pt"
    pooled_path.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_time_predictor.py",
            "--trajectories",
            str(pooled_path),
            "--checkpoint",
            str(tmp_path / "unused.pt"),
        ],
    )
    monkeypatch.setattr(
        train_time_predictor.torch,
        "load",
        lambda *_args, **_kwargs: [
            _create_synthetic_trajectory(1, "TRAIN_EVENT"),
            _create_synthetic_trajectory(2, "VALIDATION_EVENT"),
            _create_synthetic_trajectory(3, "TEST_EVENT"),
        ],
    )
    monkeypatch.setattr(
        train_time_predictor,
        "train_time_head",
        lambda **_kwargs: (
            None,
            {
                "mae_raw_hours": 1.0,
                "mae_corrected_hours": 0.5,
                "mae_improvement_pct": 50.0,
                "total_val_samples": 50,
            },
        ),
    )

    with pytest.raises(SystemExit) as error:
        train_time_predictor.main()

    assert error.value.code == 2


def test_time_predictor_trains_only_on_exact_official_split_manifests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """训练入口按预先划分的事件清单加载训练与验证轨迹。"""
    import sys

    from scripts.work3 import train_time_predictor

    paths = _write_official_split_inputs(
        tmp_path,
        train_ids=["TRAIN_A", "TRAIN_B"],
        validation_ids=["VALIDATION_A"],
        test_ids=["TEST_A"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_time_predictor.py",
            "--train-trajectories",
            str(paths["train_trajectories"]),
            "--validation-trajectories",
            str(paths["validation_trajectories"]),
            "--train-split",
            str(paths["train_split"]),
            "--validation-split",
            str(paths["validation_split"]),
            "--test-split",
            str(paths["test_split"]),
            "--checkpoint",
            str(tmp_path / "time_head.pt"),
        ],
    )
    captured: dict[str, object] = {}

    def capture_training(**kwargs: object) -> tuple[None, dict[str, float]]:
        captured.update(kwargs)
        return None, {
            "mae_raw_hours": 1.0,
            "mae_corrected_hours": 0.5,
            "mae_improvement_pct": 50.0,
            "total_val_samples": 50,
        }

    monkeypatch.setattr(train_time_predictor, "train_time_head", capture_training)

    train_time_predictor.main()

    train_ids = {item["scenario_id"] for item in captured["train_trajectories"]}
    validation_ids = {
        item["scenario_id"] for item in captured["val_trajectories"]
    }
    assert train_ids == {"TRAIN_A", "TRAIN_B"}
    assert validation_ids == {"VALIDATION_A"}
    provenance = captured["data_provenance"]
    assert provenance["scenario_ids"]["test"] == ["TEST_A"]


def test_time_predictor_rejects_trajectory_from_another_official_split(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """train轨迹文件包含validation/test事件时，训练必须在拟合前失败。"""
    import sys

    from scripts.work3 import train_time_predictor

    paths = _write_official_split_inputs(
        tmp_path,
        train_ids=["TRAIN_A"],
        validation_ids=["VALIDATION_A"],
        test_ids=["TEST_A"],
        train_artifact_ids=["TRAIN_A", "TEST_A"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_time_predictor.py",
            "--train-trajectories",
            str(paths["train_trajectories"]),
            "--validation-trajectories",
            str(paths["validation_trajectories"]),
            "--train-split",
            str(paths["train_split"]),
            "--validation-split",
            str(paths["validation_split"]),
            "--test-split",
            str(paths["test_split"]),
            "--checkpoint",
            str(tmp_path / "time_head.pt"),
        ],
    )

    with pytest.raises(ValueError, match="不属于train场景清单"):
        train_time_predictor.main()


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
        provenance = {
            "protocol": "official_event_scenario_split_v1",
            "scenario_ids": {
                "train": ["TRAIN_0", "TRAIN_1", "TRAIN_2", "TRAIN_3"],
                "validation": ["VAL_0", "VAL_1"],
                "test": ["TEST_0"],
            },
        }
        model, metrics = train_time_head(
            train_trajectories=train_trajs,
            val_trajectories=val_trajs,
            epochs=15,
            batch_size=32,
            lr=5e-3,
            checkpoint_path=ckpt_path,
            data_provenance=provenance,
        )

        assert Path(ckpt_path).is_file(), "检查点文件未成功保存"
        assert "mae_improvement_pct" in metrics
        # 由于合成数据中原始估计有固定 15h 误差，模型训练后应显著消除该误差
        assert metrics["mae_improvement_pct"] > 50.0, f"合成数据下改善度应极其显著，得到 {metrics['mae_improvement_pct']:.2f}%"

        # 验证检查点加载恢复
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert ckpt["data_provenance"] == provenance
        loaded_model = TimeResidualHead(in_dim=32, hidden_dim=64)
        loaded_model.load_state_dict(ckpt["model_state_dict"])
        loaded_metrics = evaluate_time_head(loaded_model, val_trajs)
        assert abs(loaded_metrics["mae_improvement_pct"] - metrics["mae_improvement_pct"]) < 1e-4


def test_milestone_m4_checkpoint_acceptance() -> None:
    """【里程碑 M4 硬性验收测试】：断言真实最佳检查点在验证集上 MAE 降低幅度 >= 15.0%。"""
    from scripts.work3.train_time_predictor import load_official_trajectory_splits

    ckpt_path = Path("models/work3/checkpoints/time_head_m4_official.pt")
    train_path = Path("data/work3/m4_train_trajectories.pt")
    validation_path = Path("data/work3/m4_validation_trajectories.pt")

    if not ckpt_path.is_file() or not train_path.is_file() or not validation_path.is_file():
        pytest.skip("当前官方事件划分的M4轨迹或时间头检查点尚未生成")

    _, val_trajectories, provenance = load_official_trajectory_splits(
        train_trajectories_path=train_path,
        validation_trajectories_path=validation_path,
        train_split_path=Path("data/work3/experiment_splits/train.json"),
        validation_split_path=Path("data/work3/experiment_splits/validation.json"),
        test_split_path=Path("data/work3/experiment_splits/test.json"),
    )

    ckpt = torch.load(ckpt_path, weights_only=False)
    if ckpt.get("model_version") != "signed_residual_v1":
        pytest.fail("M4正式检查点不是signed_residual_v1版本")
    saved_provenance = ckpt.get("data_provenance", {})
    assert saved_provenance.get("protocol") == provenance["protocol"]
    assert saved_provenance.get("scenario_ids") == provenance["scenario_ids"]
    for artifact_group in ("manifests", "trajectory_artifacts"):
        assert {
            split: item["sha256"]
            for split, item in saved_provenance.get(artifact_group, {}).items()
        } == {
            split: item["sha256"]
            for split, item in provenance[artifact_group].items()
        }, f"M4检查点的{artifact_group}指纹与当前官方划分不一致"
    metrics = ckpt.get("metrics", {})
    impr = metrics.get("mae_improvement_pct", 0.0)

    assert impr >= 15.0, f"里程碑 M4 验收未通过：验证集误差改善幅度 {impr:.2f}% 低于 15.0% 门槛！"

    # 直接在预先隔离的官方validation轨迹上重新评估，不做事后随机切分。
    model = TimeResidualHead(in_dim=32, hidden_dim=64)
    model.load_state_dict(ckpt["model_state_dict"])

    eval_metrics = evaluate_time_head(model, val_trajectories)
    assert eval_metrics["mae_improvement_pct"] >= 15.0, (
        f"重新评估不通过: 原始 MAE={eval_metrics['mae_raw_hours']:.3f}h, "
        f"修正 MAE={eval_metrics['mae_corrected_hours']:.3f}h, "
        f"改善幅度={eval_metrics['mae_improvement_pct']:.2f}% < 15.0%"
    )
