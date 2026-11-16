"""工具注册表。

模型的眼里只有 name / description / schema —— 它看不到实现。
所以这三样写得清不清楚，直接决定它会不会选错工具、传错参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict
    func: Callable[..., Any]
    group: str = "default"

    def to_openai_schema(self) -> dict:
        """转成 OpenAI function calling 需要的格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


TOOL_REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, description: str, schema: dict, group: str = "default"):
    """把一个普通函数注册成模型可调用的工具。

    用法：
        @tool(name="get_weather", description="...", schema={...})
        def get_weather(city: str, date: str | None = None) -> dict: ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in TOOL_REGISTRY:
            raise ValueError(f"工具重名：{name}")
        TOOL_REGISTRY[name] = ToolSpec(
            name=name, description=description, schema=schema, func=func, group=group
        )
        return func

    return decorator


def get(name: str) -> ToolSpec:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"未注册的工具：{name}（已注册：{sorted(TOOL_REGISTRY)}）")
    return TOOL_REGISTRY[name]


def tools_schema(groups: list[str] | None = None) -> list[dict]:
    """返回给模型的 tools 数组；groups 可只取某几组，避免一次给太多工具。"""
    specs = TOOL_REGISTRY.values()
    if groups:
        specs = [s for s in specs if s.group in groups]
    return [s.to_openai_schema() for s in specs]


def clear() -> None:
    """仅供测试使用。"""
    TOOL_REGISTRY.clear()
