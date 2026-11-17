"""工作三运行配置、随机种子和可复现指纹。"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from apal.runtime.artifacts import RunContext, create_run_context


_REQUIRED_CONFIG_KEYS = (
    'runtime.run_mode',
    'runtime.method_profile',
    'runtime.seed',
    'runtime.deterministic',
    'runtime.num_envs',
    'runtime.start_method',
    'runtime.total_env_steps',
    'runtime.max_wall_seconds',
    'runtime.settle_timeout_seconds',
    'runtime.device',
    'runtime.amp_dtype',
    'runtime.warmup_mode',
    'runtime.time_head_initialization',
    'runtime.main_num_threads',
    'runtime.env_num_threads',
    'runtime.dataloader_num_workers',
    'runtime.scenario_pool_path',
    'runtime.scenario_split_path',
    "paths.baseline",
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


def load_work3_episode_manifest(path: str | Path, *, split: str = "train") -> list[dict[str, Any]]:
    """只载入已核对哈希的公开实例，按冻结序列分配episode。"""
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != "work3_episode_instances_v1":
        raise ValueError("公开实例manifest版本错误")
    items = manifest.get("instances")
    if not isinstance(items, list) or not items:
        raise ValueError("公开实例manifest必须包含非空instances列表")
    allowed = {"instance_id", "split", "family", "parent_group", "baseline", "raw_data", "worker_pool",
               "public_sha256", "scenario", "env_config", "generator_version", "generation_seed", "source_ref"}
    by_id: dict[str, dict[str, Any]] = {}
    for raw in items:
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise ValueError("公开实例含未知字段或私有资料")
        instance_id = raw.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id or instance_id in by_id:
            raise ValueError("公开instance_id必须是唯一非空字符串")
        if not isinstance(raw.get("split"), str):
            raise ValueError("公开实例缺少split")
        instance = dict(raw)
        hashes = raw.get("public_sha256")
        if not isinstance(hashes, dict):
            raise ValueError("公开实例缺少public_sha256")
        for key in ("baseline", "raw_data", "worker_pool"):
            value = raw.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"公开实例缺少路径{key}")
            source = Path(value)
            # 接受Windows导出的相对路径；在Linux上仍按路径分量读取同一公开输入。
            source = source if source.is_absolute() else manifest_path.parent.joinpath(*PureWindowsPath(value).parts)
            source = source.resolve()
            if hashlib.sha256(source.read_bytes()).hexdigest() != hashes.get(key):
                raise ValueError(f"公开实例{instance_id}的{key} SHA256不匹配")
            instance[key] = str(source)
        if raw.get("scenario") is not None and not isinstance(raw["scenario"], dict):
            raise ValueError("公开场景必须是字典或None")
        env_config = raw.get("env_config", {})
        if not isinstance(env_config, dict) or set(env_config) - {"max_slots_per_station", "transfer_policy", "tolerance"}:
            raise ValueError("公开实例env_config包含未支持参数")
        if "max_slots_per_station" in env_config and (type(env_config["max_slots_per_station"]) is not int
                                                     or env_config["max_slots_per_station"] < 1):
            raise ValueError("公开实例站位容量必须为正整数")
        if "tolerance" in env_config and (not isinstance(env_config["tolerance"], (int, float))
                                          or not math.isfinite(env_config["tolerance"]) or env_config["tolerance"] <= 0):
            raise ValueError("公开实例tolerance必须为正有限数")
        instance["env_config"] = dict(env_config)
        by_id[instance_id] = instance
    sequence = manifest.get("episode_sequence", list(by_id))
    if not isinstance(sequence, list) or not sequence or any(not isinstance(key, str) or key not in by_id for key in sequence):
        raise ValueError("episode_sequence必须引用已声明的公开实例")
    selected = [dict(by_id[key]) for key in sequence if by_id[key]["split"] == split]
    if not selected:
        raise ValueError(f"公开实例manifest没有{split}实例")
    return selected


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


def validate_time_head_initialization(
    initialization: str,
    checkpoint_path: str | Path | None,
) -> None:
    """校验时间头来源；当前拒绝所有成对检查点加载请求，不回退随机权重。"""
    if initialization == "random_no_pretraining":
        if checkpoint_path is not None and str(checkpoint_path).strip():
            raise ValueError("随机初始化模式下检查点路径必须为空")
        return
    if initialization != "paired_checkpoint":
        raise ValueError(
            'runtime.time_head_initialization仅支持random_no_pretraining或paired_checkpoint'
        )
    if checkpoint_path is None or not str(checkpoint_path).strip():
        raise FileNotFoundError("显式成对检查点加载缺少检查点路径")
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"成对检查点不存在：{path}")
    raise ValueError(
        "成对检查点协议尚未实现；拒绝加载以避免版本、维度或配对关系不兼容后静默回退"
    )


def validate_work3_runtime_config(config: DictConfig) -> None:
    """检查工作三运行所需配置及跨平台运行边界。"""
    if not isinstance(config, DictConfig):
        raise TypeError("工作三运行配置必须是OmegaConf DictConfig")
    query_method = OmegaConf.select(config, 'runtime.method_profile') == "QUERY"
    episode_manifest = OmegaConf.select(config, "paths.episode_manifest")
    runs_layout = OmegaConf.select(config, "artifacts.experiment_name") is not None
    if runs_layout:
        for key in ("artifacts.experiment_name", "artifacts.runs_root"):
            value = _required(config, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"配置 {key}必须是非空字符串")
        run_id = OmegaConf.select(config, "artifacts.run_id")
        if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
            raise ValueError("artifacts.run_id必须为空或非空字符串")
    historical_keys = {'runtime.time_head_initialization', "ppo.shaping_coefficient",
                       "ppo.time_loss_coefficient", "ppo.time_auxiliary_epochs", "ppo.time_auxiliary_batch_size"}
    missing = [key for key in _REQUIRED_CONFIG_KEYS
               if not (query_method and key in historical_keys)
               and not (episode_manifest is not None and key in {"runtime.scenario_pool_path", "runtime.scenario_split_path", "paths.baseline"})
               and not (runs_layout and key in {"paths.checkpoint_dir", "paths.report_dir"})
               and OmegaConf.select(config, key) is None]
    if missing:
        raise ValueError(f"缺少必需配置: {', '.join(missing)}")

    if config.runtime.run_mode not in {"smoke", "pilot"}:
        raise ValueError('runtime.run_mode仅支持smoke或pilot')
    if config.runtime.warmup_mode not in {"none", "uniform_baseline"}:
        raise ValueError('runtime.warmup_mode仅支持none或uniform_baseline')
    if config.runtime.run_mode == "smoke" and config.runtime.warmup_mode != "none":
        raise ValueError("smoke必须使用runtime.warmup_mode=none")
    successful_batch_target = OmegaConf.select(
        config,
        'runtime.successful_batch_target',
    )
    if config.runtime.run_mode == "pilot":
        if type(successful_batch_target) is not int or successful_batch_target < 1:
            raise ValueError("pilot必须配置正整数runtime.successful_batch_target")
    elif successful_batch_target is not None:
        raise ValueError("smoke不得设置runtime.successful_batch_target")
    if config.runtime.method_profile not in {"QUERY", "B", "C", "D", "E", "F", "G"}:
        raise ValueError("runtime.method_profile仅支持QUERY或历史B/C/D/E/F/G")
    if not query_method:
        validate_time_head_initialization(str(config.runtime.time_head_initialization),
                                         OmegaConf.select(config, "paths.time_head_checkpoint"))
    elif OmegaConf.select(config, "paths.time_head_checkpoint") is not None:
        raise ValueError("QUERY不接受时间预测头权重")
    program_path = OmegaConf.select(config, "paths.query_program")
    if program_path is not None and (not query_method or not isinstance(program_path, str) or not program_path.strip()):
        raise ValueError("paths.query_program仅支持QUERY的非空程序文件路径")
    for key in ("paths.episode_manifest", "paths.initial_checkpoint"):
        value = OmegaConf.select(config, key)
        if value is not None and (not query_method or not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key}仅支持QUERY的非空文件路径")
    if episode_manifest is not None and config.runtime.warmup_mode != "none":
        raise ValueError("公开实例计划暂只支持warmup_mode=none")
    if type(config.runtime.seed) is not int:
        raise ValueError('runtime.seed必须是整数')
    if type(config.runtime.deterministic) is not bool:
        raise ValueError('runtime.deterministic必须是布尔值')
    _positive_int(config, 'runtime.num_envs')
    _positive_int(config, 'runtime.total_env_steps')
    _positive_int(config, 'runtime.main_num_threads')
    _positive_int(config, 'runtime.env_num_threads')
    _positive_int(config, 'runtime.dataloader_num_workers', allow_zero=True)
    if config.runtime.dataloader_num_workers != 0:
        raise ValueError("runtime.dataloader_num_workers必须为0；环境并行由spawn worker池承担")
    if config.runtime.start_method != "spawn":
        raise ValueError("runtime.start_method必须为spawn以保持Windows/Linux一致")
    if config.runtime.amp_dtype not in {"fp32", "bf16", "fp16"}:
        raise ValueError("runtime.amp_dtype仅支持fp32、bf16或fp16")
    if not isinstance(config.runtime.device, str) or not config.runtime.device.strip():
        raise ValueError('runtime.device必须是非空设备标识')
    if _finite_number(config, 'runtime.max_wall_seconds', minimum=0.0) <= 0.0:
        raise ValueError("配置 runtime.max_wall_seconds必须是正有限数值")
    if _finite_number(config, 'runtime.settle_timeout_seconds', minimum=0.0) <= 0.0:
        raise ValueError("配置 runtime.settle_timeout_seconds必须是正有限数值")

    for key in ('runtime.scenario_pool_path', 'runtime.scenario_split_path'):
        if episode_manifest is not None:
            continue
        value = _required(config, key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"配置 {key}必须是pathlib可解析的路径字符串")
    for key in ("paths.baseline", "paths.checkpoint_dir", "paths.report_dir"):
        if episode_manifest is not None and key == "paths.baseline":
            continue
        if runs_layout and key != "paths.baseline" and OmegaConf.select(config, key) is None:
            continue
        value = _required(config, key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"配置 {key}必须是pathlib可解析的路径字符串")

    _positive_int(config, "ppo.steps_per_iter")
    _positive_int(config, "ppo.epochs", allow_zero=True)
    _positive_int(config, "ppo.batch_size")
    if not query_method:
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
        if not (query_method and key in historical_keys):
            _finite_number(config, key, minimum=0.0)
    for key in ("ppo.gamma", "ppo.gae_lambda"):
        if _finite_number(config, key, minimum=0.0) > 1.0:
            raise ValueError(f"配置 {key}必须位于[0, 1]")
    if _finite_number(config, "ppo.learning_rate", minimum=0.0) <= 0.0:
        raise ValueError("配置 ppo.learning_rate必须是正有限数值")
    if _finite_number(config, "ppo.clip_epsilon", minimum=0.0) <= 0.0:
        raise ValueError("配置 ppo.clip_epsilon必须是正有限数值")


def resolve_work3_precision(
    amp_dtype: str,
    device: str | torch.device,
) -> tuple[str, torch.dtype | None]:
    """将配置精度映射为Lightning策略与rollout autocast dtype。"""
    if amp_dtype not in {"fp32", "fp16", "bf16"}:
        raise ValueError("amp_dtype仅支持fp32、fp16或bf16")
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError(f"工作三训练不支持设备类型: {target.type}")
    if amp_dtype == "fp32":
        return "32-true", None
    if target.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("fp16/bf16混合精度要求可用的CUDA设备")
    if target.index is not None and target.index >= torch.cuda.device_count():
        raise ValueError(f"配置的CUDA设备不存在: {target}")
    if amp_dtype == "fp16":
        return "16-mixed", torch.float16
    with torch.cuda.device(target):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError(f"配置的CUDA设备不支持硬件bfloat16: {target}")
    return "bf16-mixed", torch.bfloat16


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


def resolve_work3_output_paths(
    config: DictConfig,
    project_root: Path,
    *,
    output: Path | None = None,
    report: Path | None = None,
) -> tuple[RunContext | None, Path, Path]:
    """先落实运行身份和显式输出，再计算可复现配置指纹。"""
    context = None
    if OmegaConf.select(config, "artifacts.experiment_name") is not None:
        context = create_run_context(
            project_root,
            experiment_name=str(config.artifacts.experiment_name),
            run_id=config.artifacts.get("run_id"),
            runs_root=str(config.artifacts.runs_root),
        )
        config.artifacts.run_id = context.run_id
        config.artifacts.run_dir = str(context.run_dir)
    checkpoint_dir = config.paths.get("checkpoint_dir")
    report_dir = config.paths.get("report_dir")
    if checkpoint_dir is None:
        if context is None:
            raise ValueError("缺少检查点目录及runs运行配置")
        checkpoint_dir = context.checkpoint_dir
    if report_dir is None:
        if context is None:
            raise ValueError("缺少报告目录及runs运行配置")
        report_dir = context.reports_dir
    stem = f"{config.runtime.run_mode}_method_{str(config.runtime.method_profile).lower()}"
    output_path = Path(output) if output is not None else Path(checkpoint_dir) / f"{stem}_weights.pt"
    report_path = Path(report) if report is not None else Path(report_dir) / f"{stem}.run.json"
    config.paths.checkpoint_dir = str(output_path.parent)
    config.paths.report_dir = str(report_path.parent)
    config.paths.output_checkpoint = str(output_path)
    config.paths.run_report = str(report_path)
    return context, output_path, report_path


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
