import time
import logging
from pathlib import Path
from typing import Any
from apal.configs import configs
from apal.project import PROJECT_ROOT
from apal.runtime.artifacts import build_run_id, run_context, uses_runs_layout

def init_logger(args: Any, experiment_name: str) -> tuple[logging.Logger, Path]:
    output_root = (
        getattr(args, 'output_dir', None)
        or getattr(args, 'result_dir', None)
    )
    if not output_root and not uses_runs_layout(configs):
        output_root = configs.result_dir
    if output_root:
        exp_dir = Path(output_root) / build_run_id(configs)
        log_dir = exp_dir
    else:
        context = run_context(configs, PROJECT_ROOT, create_dirs=True)
        exp_dir, log_dir = context.eval_dir, context.logs_dir
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{experiment_name}.log"
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"========== {experiment_name} 实验开始 ==========")
    logger.info(f"实验参数: {vars(args) if hasattr(args, '__dict__') else args}")
    logger.info(f"结果将归档至: {exp_dir}")
    
    return logger, exp_dir

def record_experiment_time(logger, start_time):
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    logger.info(f"实验结束。总耗时: {hours}小时{minutes}分钟{seconds}秒")
    return elapsed
