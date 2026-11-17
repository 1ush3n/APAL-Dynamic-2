"""运行当前 APAL 五技能训练前检查。

用法：
    python tests/run_preflight_tests.py
    python tests/run_preflight_tests.py --all
"""

from __future__ import annotations

from apal.project import PROJECT_ROOT

from pathlib import Path
import subprocess
import sys


PREFLIGHT_TESTS = (
    'tests/common/runtime/test_config_loader.py',
    'tests/common/data/test_five_skill_data.py',
    'tests/work1/test_initial_training_preflight.py',
    'tests/common/training/test_ppo_static_worker_mask.py',
    'tests/common/baselines/test_graph_rl_baselines.py',
    'tests/common/baselines/test_graph_ddqn_performance.py',
    'tests/common/runtime/test_checkpoint_metadata.py',
    'tests/common/training/test_lightning_architecture.py',
    'tests/common/data/test_training_data_manifest.py',
    'tests/common/policy/test_worker_pointer_v2.py',
    'tests/work1/test_verify_worker_pointer_v2_training_run.py',
    'tests/common/physics/test_skill_hub_graph.py',
    'tests/common/physics/test_heterogeneous_rebuild.py',
    'tests/common/physics/test_dynamic_events.py',
    'tests/common/training/test_vector_env_safety.py',
    'tests/work2/test_reschedule_ablation_suite.py',
    'tests/common/policy/test_policy_observation_scope.py',
)


def main(argv: list[str] | None = None) -> int:
    """以当前解释器执行严格前检查，避免生成任何正式实验产物。"""

    args = list(sys.argv[1:] if argv is None else argv)
    selected = ["tests"] if "--all" in args else list(PREFLIGHT_TESTS)
    base_temp = PROJECT_ROOT / ".pytest_tmp_v2" / "preflight"
    base_temp.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--basetemp",
        str(base_temp),
        *selected,
    ]
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
