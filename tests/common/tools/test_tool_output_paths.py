from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _isolate_working_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize(
    "module_name, experiment",
    [
        ("scripts.work1.audit_initial_team_opportunity", "initial_team_opportunity"),
        ("scripts.work1.audit_initial_team_opportunity_full", "initial_team_opportunity_full"),
        ("scripts.work2.audit_reschedule_team_opportunity", "reschedule_team_opportunity"),
    ],
)
def test_audit_output_defaults_to_new_run_without_creating_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module_name: str, experiment: str
) -> None:
    target = importlib.import_module(module_name)
    monkeypatch.setattr(target, "PROJECT_ROOT", tmp_path)
    default = target.parse_args([]).output_dir
    assert default.relative_to(tmp_path).parts[:2] == ("runs", experiment)
    assert default.parts[-2:] == ("artifacts", "reports")
    assert not default.exists()
    explicit = tmp_path / "explicit"
    assert target.parse_args(["--output-dir", str(explicit)]).output_dir == explicit


@pytest.mark.parametrize(
    "module_name, evaluator_name, arguments_name",
    [
        ("scripts.work2.evaluate_reschedule_manifest", "evaluate_manifest_instances", "MANIFEST_EVAL_ARGS"),
        ("scripts.work2.evaluate_literature_reschedule_manifest", "evaluate_literature_r5_manifest", "EXTRA_ARGS"),
        ("scripts.work2.evaluate_graph_ddqn_r5_manifest", "evaluate_graph_ddqn_r5_manifest", "EXTRA_ARGS"),
        ("scripts.work2.evaluate_l2d_ppo_r5_manifest", "evaluate_l2d_ppo_r5_manifest", "EXTRA_ARGS"),
    ],
)
def test_manifest_cli_uses_run_eval_and_respects_output_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module_name: str,
    evaluator_name: str,
    arguments_name: str,
) -> None:
    target = importlib.import_module(module_name)
    monkeypatch.setattr(target, "PROJECT_ROOT", tmp_path)
    for key, value in {"artifact_layout": "runs", "runs_root": str(tmp_path / "runs"),
                       "experiment_name": "manifest_check", "run_id": "fixed"}.items():
        monkeypatch.setattr(target.configs, key, value)
    received: list[Path] = []

    def evaluate(**kwargs: object) -> dict[str, object]:
        received.append(Path(kwargs["output_dir"]))
        return {"summaries": {}, "rows": []}

    current_output: list[str | None] = [None]

    def initialize(*args: object, **kwargs: object) -> SimpleNamespace:
        options = getattr(target, arguments_name)
        return SimpleNamespace(**{
            **{key: option.default for key, option in options.items()},
            "model_path": "model.pt", "manifest_path": "manifest.json",
            "output_dir": current_output[0],
        })

    monkeypatch.setattr(target, "initialize_hydra_runtime", initialize)
    monkeypatch.setattr(target, evaluator_name, evaluate)
    assert target.main([]) == 0
    assert received[-1].is_relative_to(tmp_path / "runs" / "manifest_check" / "fixed" / "eval")
    current_output[0] = str(tmp_path / "explicit")
    assert target.main([]) == 0
    assert received[-1] == tmp_path / "explicit"


def test_joint_plan_default_and_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = importlib.import_module("scripts.work1.run_joint_variant_function_screen")
    monkeypatch.setattr(target, "PROJECT_ROOT", tmp_path)
    assert target.main(["mode=plan"]) == 0
    plans = list((tmp_path / "runs" / "joint_variant_function_screen").glob("*/artifacts/reports/screen_plan.json"))
    assert len(plans) == 1
    assert len(json.loads(plans[0].read_text(encoding="utf-8"))) == len(target.VARIANTS)
    explicit = tmp_path / "explicit"
    assert target.main(["mode=plan", f"output_dir={explicit}"]) == 0
    assert (explicit / "screen_plan.json").is_file()


def test_ablation_plan_uses_run_reports_and_child_tensorboard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = importlib.import_module("scripts.work2.reschedule_ablation_suite")
    monkeypatch.setattr(target, "PROJECT_ROOT", tmp_path)
    assert target.main(["mode=plan", "variants=[full]", "validate_paths=false", f"runs_root={tmp_path / 'runs'}"]) == 0
    plans = list((tmp_path / "runs" / "reschedule_ablation_suite").glob("*/artifacts/reports/reschedule_ablation_command_plan.json"))
    assert len(plans) == 1
    row = json.loads(plans[0].read_text(encoding="utf-8"))[0]
    assert str(tmp_path / "runs" / "reschedule_task_delay" / row["run_id"] / "logs" / "tensorboard") in row["command"]


def test_joint_run_routes_console_logs_to_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = importlib.import_module("scripts.work1.run_joint_variant_function_screen")
    monkeypatch.setattr(target, "PROJECT_ROOT", tmp_path)
    received: list[Path] = []

    class TrainingBoundary(Exception):
        pass

    def stop_before_training(**kwargs: object) -> None:
        received.append(Path(kwargs["logs_dir"]))
        raise TrainingBoundary

    monkeypatch.setattr(target, "_run_one", stop_before_training)
    with pytest.raises(TrainingBoundary):
        target.main(["mode=run", "variants=[full_joint]"])
    assert received[-1].relative_to(tmp_path).parts[:2] == ("runs", "joint_variant_function_screen")
    assert received[-1].name == "logs"


def test_max_fill_summary_default_uses_run_report_without_running_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = importlib.import_module("scripts.common.performance.sweep_gpu_max_fill")
    monkeypatch.setattr(target, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["sweep_gpu_max_fill"])
    monkeypatch.setattr(target, "run_adaptive_max_fill_sweep", lambda **kwargs: {"test_only": True})
    assert target.main() == 0
    summaries = list((tmp_path / "runs" / "gpu_max_fill").glob("*/artifacts/reports/max_fill_summary.json"))
    assert len(summaries) == 1
    assert json.loads(summaries[0].read_text(encoding="utf-8")) == {"test_only": True}
    explicit = tmp_path / "explicit.json"
    monkeypatch.setattr("sys.argv", ["sweep_gpu_max_fill", "--output-summary-json", str(explicit)])
    assert target.main() == 0
    assert json.loads(explicit.read_text(encoding="utf-8")) == {"test_only": True}


def test_sensitivity_dispatch_writes_reports_without_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = importlib.import_module("scripts.common.sensitivity_analysis")
    monkeypatch.setattr(target, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["sensitivity_analysis", "--param", "lr", "--values", "0.001"])
    monkeypatch.setattr(target, "run_single_param_sweep", lambda *args, **kwargs: target.pd.DataFrame({"value": [0.001]}))
    monkeypatch.setattr(target, "plot_sensitivity", lambda *args, **kwargs: None)
    received: list[Path] = []
    monkeypatch.setattr(target, "generate_summary_report", lambda results, output: received.append(Path(output)))
    target.main()
    assert received[-1].relative_to(tmp_path).parts[:2] == ("runs", "sensitivity_analysis")
    assert (received[-1] / "raw_lr.csv").is_file()
    explicit = tmp_path / "explicit"
    monkeypatch.setattr("sys.argv", ["sensitivity_analysis", "--param", "lr", "--values", "0.001", "--output", str(explicit)])
    target.main()
    assert received[-1] == explicit
    assert (explicit / "raw_lr.csv").is_file()


def test_replay_benchmark_dispatch_resolves_report_dir_before_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = importlib.import_module("scripts.common.performance.benchmark_graph_ddqn_replay")
    monkeypatch.setattr(target, "PROJECT_ROOT", tmp_path)
    for key, value in {"artifact_layout": "runs", "runs_root": str(tmp_path / "runs"),
                       "experiment_name": "replay_check", "run_id": "fixed"}.items():
        monkeypatch.setattr(target.configs, key, value)
    current_output: list[str | None] = [None]
    monkeypatch.setattr(target, "initialize_hydra_runtime", lambda *args, **kwargs: SimpleNamespace(
        benchmark_data_path="data.csv", benchmark_transitions=8, benchmark_updates=1,
        benchmark_repeats=1, output_dir=current_output[0],
    ))
    monkeypatch.setattr(target, "get_available_device", lambda: target.torch.device("cpu"))
    monkeypatch.setattr(target, "AirLineEnv_Graph", lambda *args, **kwargs: object())
    monkeypatch.setattr(target, "GraphDDQNAgent", lambda *args, **kwargs: object())
    monkeypatch.setattr(target, "_clone_args", lambda *args, **kwargs: SimpleNamespace())

    class SamplingBoundary(Exception):
        pass

    def stop_before_sampling(*args: object, **kwargs: object) -> None:
        raise SamplingBoundary

    monkeypatch.setattr(target, "_collect_transitions", stop_before_sampling)
    with pytest.raises(SamplingBoundary):
        target.main([])
    expected = tmp_path / "runs" / "replay_check" / "fixed" / "artifacts" / "reports" / "ddqn_replay_benchmark"
    assert expected.is_dir()
    current_output[0] = str(tmp_path / "explicit")
    with pytest.raises(SamplingBoundary):
        target.main([])
    assert (tmp_path / "explicit").is_dir()
