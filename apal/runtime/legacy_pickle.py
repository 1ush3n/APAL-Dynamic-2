"""只迁移已知历史工件的类路径；调用方仍负责确认pickle来源可信。"""

from __future__ import annotations

import io
import pickle
from collections.abc import Iterable
from typing import Any, BinaryIO


_LEGACY_CLASSES = {
    "baselines.literature_dqn.replay": (
        "apal.baselines.literature_dqn.replay", ("DDQNTransition",),
    ),
    "envs.work3.core_types": (
        "apal.envs.work3.core_types",
        ("TaskStatus", "ActionBranch", "TimeInterval", "WorkerCalendar", "TaskRuntimeState",
         "AircraftRuntimeState", "CycleExecutionRecord", "MultiAircraftState"),
    ),
    "envs.work3.decision_snapshot": (
        "apal.envs.work3.decision_snapshot", ("WorkerSnapshot", "TeamCompletionContext", "DecisionSnapshot"),
    ),
    "envs.work3.scheduling_queries": (
        "apal.envs.work3.scheduling_queries", ("QueryStatus", "ScheduleCondition", "CompletionTable"),
    ),
}


class Unpickler(pickle.Unpickler):
    """逐类映射有限旧路径，不修改模块别名或PyTorch全局加载器。"""

    def find_class(self, module: str, name: str) -> Any:
        if module in _LEGACY_CLASSES:
            target_module, names = _LEGACY_CLASSES[module]
            if name not in names:
                raise pickle.UnpicklingError(f"未列入历史兼容范围的类：{module}.{name}")
            module = target_module
        return super().find_class(module, name)


def load(
    file: BinaryIO, *, fix_imports: bool = True, encoding: str = "ASCII",
    errors: str = "strict", buffers: Iterable[Any] | None = None,
) -> Any:
    """供torch.load显式pickle_module适配及可信旧pickle文件读取。"""
    return Unpickler(file, fix_imports=fix_imports, encoding=encoding, errors=errors, buffers=buffers).load()


def loads(
    data: bytes | bytearray | memoryview, *, fix_imports: bool = True,
    encoding: str = "ASCII", errors: str = "strict", buffers: Iterable[Any] | None = None,
) -> Any:
    """从内存读取同一有限历史类集合。"""
    return load(io.BytesIO(data), fix_imports=fix_imports, encoding=encoding, errors=errors, buffers=buffers)


Pickler = pickle.Pickler
dump = pickle.dump
dumps = pickle.dumps
