"""工具执行：参数校验 + 异常兜底。

核心原则：**错误也是 observation**。
工具报错不中断对话，而是把错误信息回填给模型，让它自己修正参数重试——
这比直接抛异常中断体验好得多，也是 ReAct 循环能不能稳定跑的关键。
"""

from __future__ import annotations

import json
from typing import Any

from agent import registry

_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate(schema: dict, args: dict) -> list[str]:
    """按 JSON Schema 的必填项与基础类型做校验，返回错误列表（空表示通过）。"""
    errors: list[str] = []
    props: dict = schema.get("properties", {})

    for key in schema.get("required", []):
        if key not in args or args[key] in (None, ""):
            errors.append(f"缺少必填参数 {key}")

    for key, value in args.items():
        if key not in props:
            errors.append(f"未知参数 {key}（可用：{sorted(props)}）")
            continue
        expected = props[key].get("type")
        py_type = _TYPE_MAP.get(expected)
        if py_type and not isinstance(value, py_type):
            errors.append(f"参数 {key} 类型应为 {expected}，实际是 {type(value).__name__}")

    return errors


def safe_execute(name: str, arguments: str) -> dict:
    """执行工具，任何异常都转成结构化结果返回，不向上抛。"""
    try:
        spec = registry.get(name)
    except KeyError as e:
        return {"error": str(e)}

    # 模型偶尔会把参数拼成不合法的 JSON（多一个逗号、少一个引号）
    try:
        args = json.loads(arguments) if arguments and arguments.strip() else {}
    except json.JSONDecodeError as e:
        return {"error": f"参数不是合法 JSON：{e}"}

    if not isinstance(args, dict):
        return {"error": f"参数应为对象，实际收到 {type(args).__name__}"}

    errors = validate(spec.schema, args)
    if errors:
        return {"error": "参数校验未通过：" + "；".join(errors)}

    try:
        result: Any = spec.func(**args)
    except Exception as e:  # noqa: BLE001 工具内部任何异常都要兜住
        return {"error": f"{type(e).__name__}: {e}"}

    if not isinstance(result, (dict, list, str, int, float, bool)):
        result = str(result)
    return result  # type: ignore[return-value]
