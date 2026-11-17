"""轻量项目根定位；命令从仓库根运行，工具可显式指定另一仓库根。"""

from __future__ import annotations

from pathlib import Path


def resolve_project_root(project_root: str | Path | None = None) -> Path:
    """按包文件位置或显式路径定位，并核实仓库的配置和数据目录。"""
    root = Path(project_root).expanduser().resolve() if project_root is not None else Path(__file__).resolve().parents[1]
    if not (root / "conf").is_dir() or not (root / "data").is_dir():
        raise FileNotFoundError(f"项目根必须同时包含conf和data目录：{root}；请从仓库根运行或显式指定项目根")
    return root


PROJECT_ROOT = resolve_project_root()
