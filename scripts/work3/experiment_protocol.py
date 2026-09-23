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
    """返回正式C或D配置；启发式运行器不属于正式C/D。"""
    name = str(method_variant).strip().upper()
    if name == "C":
        return Work3MethodProfile(
            name="C",
            graph_policy=True,
            allow_postpone=True,
            use_time_auxiliary=False,
            use_corrected_time_input=False,
            use_learned_time_shaping=False,
        )
    if name == "D":
        return Work3MethodProfile(
            name="D",
            graph_policy=True,
            allow_postpone=True,
            use_time_auxiliary=True,
            use_corrected_time_input=True,
            use_learned_time_shaping=True,
        )
    raise ValueError(f"正式方法只支持 C 或 D，不支持: {method_variant}")
