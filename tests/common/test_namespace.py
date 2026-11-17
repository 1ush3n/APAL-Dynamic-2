"""包命名空间、项目根与有限历史类路径的集中回归。"""

from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path

import pytest


def test_project_root_uses_package_location_and_validates_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apal.project import PROJECT_ROOT, resolve_project_root

    expected = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    assert PROJECT_ROOT == expected
    assert resolve_project_root() == expected
    explicit = tmp_path / "explicit"
    (explicit / "conf").mkdir(parents=True)
    (explicit / "data").mkdir()
    assert resolve_project_root(explicit) == explicit.resolve()
    with pytest.raises(FileNotFoundError, match="conf.*data"):
        resolve_project_root(tmp_path / "missing")


@pytest.mark.parametrize("kind", ["action", "transition"])
def test_legacy_class_globals_resolve_to_canonical_types(kind: str) -> None:
    from apal.runtime import legacy_pickle
    import torch

    if kind == "action":
        from apal.envs.work3.core_types import ActionBranch

        value = ActionBranch.ADVANCE_TO_NEXT_EVENT
        old_module = "envs.work3.core_types"
    else:
        from apal.baselines.literature_dqn.replay import DDQNTransition

        value = DDQNTransition(7, 0, {}, (1, 0, [2]), 0.25, {}, False, ([], [], []), ([], [], []))
        old_module = "baselines.literature_dqn.replay"
    cls = type(value)
    current = pickle.dumps(value, protocol=0)
    current_global = f"c{cls.__module__}\n{cls.__name__}\n".encode("ascii")
    legacy_global = f"c{old_module}\n{cls.__name__}\n".encode("ascii")
    assert current.count(current_global) == 1
    legacy = current.replace(current_global, legacy_global)
    loaded = legacy_pickle.loads(legacy)
    assert type(loaded) is cls
    assert loaded == value
    assert legacy_pickle.load(io.BytesIO(legacy)) == value
    saved = io.BytesIO()
    torch.save(value, saved, pickle_protocol=0)
    legacy_checkpoint = io.BytesIO()
    with zipfile.ZipFile(saved) as source, zipfile.ZipFile(legacy_checkpoint, "w") as destination:
        for entry in source.infolist():
            content = source.read(entry.filename)
            if entry.filename.endswith("/data.pkl"):
                assert content.count(current_global) == 1
                content = content.replace(current_global, legacy_global)
            destination.writestr(entry, content)
    legacy_checkpoint.seek(0)
    loaded_checkpoint = torch.load(legacy_checkpoint, map_location="cpu", weights_only=False, pickle_module=legacy_pickle)
    assert type(loaded_checkpoint) is cls
    assert loaded_checkpoint == value
    with pytest.raises(pickle.UnpicklingError, match="未列入"):
        legacy_pickle.loads(b"cenvs.work3.core_types\nUnknownOldClass\n.")
