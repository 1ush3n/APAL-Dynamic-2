"""工作三正式实验的最小方法分组定义。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Work3MethodProfile:
    """对照组共享环境和图策略，只切换确认的后移/时间学习机制。"""

    name: str
    graph_policy: bool
    allow_postpone: bool
    use_time_auxiliary: bool
    use_corrected_time_input: bool
    use_learned_time_shaping: bool


def build_method_profile(method_variant: str) -> Work3MethodProfile:
    """返回正式对照或消融profile。"""
    name = str(method_variant).strip().upper()
    time_profiles = {
        "B": (False, False, False),
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
            f"正式方法只支持 B、C、D、E、F、G，不支持: {method_variant}"
        ) from error
    return Work3MethodProfile(
        name=name,
        graph_policy=True,
        allow_postpone=name != "B",
        use_time_auxiliary=use_auxiliary,
        use_corrected_time_input=use_corrected_input,
        use_learned_time_shaping=use_learned_shaping,
    )
