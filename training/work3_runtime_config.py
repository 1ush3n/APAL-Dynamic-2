"""工作三运行配置、随机种子和可复现指纹。"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


_REQUIRED_CONFIG_KEYS = (
    "runtime.run_mode",
    "runtime.method_profile",
    "runtime.seed",
    "runtime.deterministic",
    "runtime.num_envs",
    "runtime.start_method",
    "runtime.total_env_steps",
    "runtime.max_wall_seconds",
    "runtime.settle_timeout_seconds",
    "runtime.device",
    "runtime.amp_dtype",
    "runtime.main_num_threads",
    "runtime.env_num_threads",
    "runtime.dataloader_num_workers",
    "runtime.scenario_pool_path",
    "runtime.scenario_split_path",
    "paths.baseline",
    "paths.time_head_checkpoint",
    "paths.checkpoint_dir",
    "paths.report_dir",
    "ppo.steps_per_iter",
    "ppo.epochs",
    "ppo.batch_size",
    "ppo.learning_rate",
    "ppo.clip_epsilon",
    "ppo.value_coefficient",
    "ppo.entropy_coefficient",
    "ppo.gamma",
    "ppo.gae_lambda",
    "ppo.shaping_coefficient",
    "ppo.time_loss_coefficient",
    "ppo.time_auxiliary_epochs",
    "ppo.time_auxiliary_batch_size",
)


def _required(config: DictConfig, key: str) -> object:
    value = OmegaConf.select(config, key)
    if value is None:
        raise ValueError(f"缺少必需配置: {key}")
    return value


def _positive_int(config: DictConfig, key: str, *, allow_zero: bool = False) -> int:
    value = _required(config, key)
    if type(value) is not int or value < (0 if allow_zero else 1):
        qualifier = "非负" if allow_zero else "正"
        raise ValueError(f"配置 {key} 必须是{qualifier}整数")
    return value


def _finite_number(config: DictConfig, key: str, *, minimum: float) -> float:
    value = _required(config, key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"配置 {key} 必须是有限数值")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < minimum:
        raise ValueError(f"配置 {key} 必须是大于等于 {minimum} 的有限数值")
    return numeric_value


def validate_work3_runtime_config(config: DictConfig) -> None:
    """检查工作三运行所需配置及跨平台运行边界。"""
    if not isinstance(config, DictConfig):
        raise TypeError("工作三运行配置必须是OmegaConf DictConfig")
    missing = [key for key in _REQUIRED_CONFIG_KEYS if OmegaConf.select(config, key) is None]
    if missing:
        raise ValueError(f"缺少必需配置: {', '.join(missing)}")

    if config.runtime.run_mode not in {"smoke", "pilot"}:
        raise ValueError("runtime.run_mode仅支持smoke或pilot")
    successful_batch_target = OmegaConf.select(
        config,
        "runtime.successful_batch_target",
    )
    if config.runtime.run_mode == "pilot":
        if type(successful_batch_target) is not int or successful_batch_target < 1:
            raise ValueError("pilot必须配置正整数runtime.successful_batch_target")
    elif successful_batch_target is not None:
        raise ValueError("smoke不得设置runtime.successful_batch_target")
    if config.runtime.method_profile not in {"C", "D"}:
        raise ValueError("runtime.method_profile仅支持C或D")
    if type(config.runtime.seed) is not int:
        raise ValueError("runtime.seed必须是整数")
    if type(config.runtime.deterministic) is not bool:
        raise ValueError("runtime.deterministic必须是布尔值")
    _positive_int(config, "runtime.num_envs")
    _positive_int(config, "runtime.total_env_steps")
    _positive_int(config, "runtime.main_num_threads")
    _positive_int(config, "runtime.env_num_threads")
    _positive_int(config, "runtime.dataloader_num_workers", allow_zero=True)
    if config.runtime.dataloader_num_workers != 0:
        raise ValueError("runtime.dataloader_num_workers必须为0；环境并行由spawn worker池承担")
    if config.runtime.start_method != "spawn":
        raise ValueError("runtime.start_method必须为spawn以保持Windows/Linux一致")
    if config.runtime.amp_dtype not in {"fp32", "bf16", "fp16"}:
        raise ValueError("runtime.amp_dtype仅支持fp32、bf16或fp16")
    if not isinstance(config.runtime.device, str) or not config.runtime.device.strip():
        raise ValueError("runtime.device必须是非空设备标识")
    if _finite_number(config, "runtime.max_wall_seconds", minimum=0.0) <= 0.0:
        raise ValueError("配置 runtime.max_wall_seconds必须是正有限数值")
    if _finite_number(config, "runtime.settle_timeout_seconds", minimum=0.0) <= 0.0:
        raise ValueError("配置 runtime.settle_timeout_seconds必须是正有限数值")

    for key in ("runtime.scenario_pool_path", "runtime.scenario_split_path"):
        value = _required(config, key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"配置 {key}必须是pathlib可解析的路径字符串")
    for key in ("paths.baseline", "paths.time_head_checkpoint", "paths.checkpoint_dir", "paths.report_dir"):
        value = _required(config, key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"配置 {key}必须是pathlib可解析的路径字符串")

    _positive_int(config, "ppo.steps_per_iter")
    _positive_int(config, "ppo.epochs", allow_zero=True)
    _positive_int(config, "ppo.batch_size")
    _positive_int(config, "ppo.time_auxiliary_epochs", allow_zero=True)
    _positive_int(config, "ppo.time_auxiliary_batch_size")
    for key in (
        "ppo.learning_rate",
        "ppo.clip_epsilon",
        "ppo.value_coefficient",
        "ppo.entropy_coefficient",
        "ppo.gamma",
        "ppo.gae_lambda",
        "ppo.shaping_coefficient",
        "ppo.time_loss_coefficient",
    ):
        _finite_number(config, key, minimum=0.0)
    for key in ("ppo.gamma", "ppo.gae_lambda"):
        if _finite_number(config, key, minimum=0.0) > 1.0:
            raise ValueError(f"配置 {key}必须位于[0, 1]")
    if _finite_number(config, "ppo.learning_rate", minimum=0.0) <= 0.0:
        raise ValueError("配置 ppo.learning_rate必须是正有限数值")
    if _finite_number(config, "ppo.clip_epsilon", minimum=0.0) <= 0.0:
        raise ValueError("配置 ppo.clip_epsilon必须是正有限数值")


def load_work3_runtime_config(
    path: Path,
    overrides: Sequence[str] = (),
) -> DictConfig:
    """载入YAML与OmegaConf点式覆盖，并在返回前校验完整配置。"""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"工作三运行配置不存在: {config_path}")
    base_config = OmegaConf.load(config_path)
    override_config = OmegaConf.from_dotlist(list(overrides))
    config = OmegaConf.merge(base_config, override_config)
    validate_work3_runtime_config(config)
    return config


def apply_work3_runtime_overrides(
    config: DictConfig,
    overrides: Mapping[str, object],
) -> DictConfig:
    """将显式CLI值写入配置副本并重新执行完整边界校验。"""
    if not isinstance(config, DictConfig):
        raise TypeError("工作三运行配置必须是OmegaConf DictConfig")
    resolved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    for key, value in overrides.items():
        OmegaConf.update(resolved, key, value, merge=False)
    validate_work3_runtime_config(resolved)
    return resolved


def resolved_config_fingerprint(config: DictConfig) -> tuple[str, str]:
    """返回可存档的解析后YAML文本及其SHA256。"""
    validate_work3_runtime_config(config)
    resolved_yaml = OmegaConf.to_yaml(config, resolve=True)
    fingerprint = hashlib.sha256(resolved_yaml.encode("utf-8")).hexdigest()
    return resolved_yaml, fingerprint


def derive_worker_seed(seed: int, worker_id: int, episode_index: int) -> int:
    """从主种子和稳定worker/episode坐标导出跨进程稳定种子。"""
    if type(seed) is not int or type(worker_id) is not int or type(episode_index) is not int:
        raise TypeError("主种子、worker_id和episode_index必须是整数")
    if worker_id < 0 or episode_index < 0:
        raise ValueError("worker_id和episode_index不能为负数")
    payload = f"work3:{seed}:{worker_id}:{episode_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def seed_work3_runtime(seed: int, *, deterministic: bool) -> None:
    """统一设置Python、NumPy、PyTorch及CUDA随机源与确定性后端。"""
    if type(seed) is not int or type(deterministic) is not bool:
        raise TypeError("seed必须是整数且deterministic必须是布尔值")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False
