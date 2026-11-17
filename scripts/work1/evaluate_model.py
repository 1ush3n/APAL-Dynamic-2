"""兼容评估命令；不依赖同名文件的导入顺序。"""
from apal.evaluation.initial import cli_main

if __name__ == "__main__":
    raise SystemExit(cli_main())
