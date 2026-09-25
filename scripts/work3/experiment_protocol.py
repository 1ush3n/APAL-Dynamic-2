"""工作三正式实验的最小方法分组定义。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Work3MethodProfile:
    """C/D共享环境和图策略，只切换确认的时间学习机制。"""

    name: str
    graph_policy: bool
    allow_postpone: bool
    use_time_auxiliary: bool
    use_corrected_time_input: bool
    use_learned_time_shaping: bool


def build_method_profile(method_variant: str) -> Work3MethodProfile:
    """返回确认稿定义的正式时间机制profile。"""
    name = str(method_variant).strip().upper()
    time_profiles = {
        "C": (False, False, False),
        "E": (True, False, False),
        "F": (True, True, False),
        "G": (True, False, True),
        "D": (True, True, True),
    }
    try:
        use_auxiliary, use_corrected_input, use_learned_shaping = time_profiles[name]
    except KeyError as error:
        raise ValueError(
            f"正式方法只支持 C、D、E、F、G，不支持: {method_variant}"
        ) from error
    return Work3MethodProfile(
        name=name,
        graph_policy=True,
        allow_postpone=True,
        use_time_auxiliary=use_auxiliary,
        use_corrected_time_input=use_corrected_input,
        use_learned_time_shaping=use_learned_shaping,
    )
