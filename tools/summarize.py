"""文本摘要工具：内部再调一次模型；没有 API Key 时退化为按句截断。

工具内部调用模型是 Agent 里常见的"套娃"结构——模型调用一个工具，
工具内部又请求了一次模型，所以超时和成本都要单独考虑。
"""

from __future__ import annotations

import os
import re

from agent.registry import tool

_SENT_SPLIT = re.compile(r"(?<=[。！？；.!?;])")


def _truncate(text: str, max_len: int) -> str:
    """按句子累加到 max_len，避免从词中间切断。"""
    out = ""
    for sent in _SENT_SPLIT.split(text):
        if len(out) + len(sent) > max_len:
            break
        out += sent
    return out.strip() or text[:max_len]


@tool(
    name="summarize_text",
    description="把一段长文本压缩成不超过 max_len 字的摘要。只能用于压缩文本，不要用它改写或扩写内容。",
    schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "待摘要的原文"},
            "max_len": {"type": "integer", "description": "摘要最大字数，默认 80"},
        },
        "required": ["text"],
    },
    group="daily",
)
def summarize_text(text: str, max_len: int = 80) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("text 不能为空")
    if len(text) <= max_len:
        return {"summary": text, "mode": "noop"}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"summary": _truncate(text, max_len), "mode": "truncate", "note": "未配置 API，按句截断"}

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
    resp = client.chat.completions.create(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": f"把用户给的文本压缩成不超过 {max_len} 字的中文摘要，只输出摘要本身。"},
            {"role": "user", "content": text},
        ],
        timeout=20,
    )
    return {"summary": (resp.choices[0].message.content or "").strip(), "mode": "model"}
