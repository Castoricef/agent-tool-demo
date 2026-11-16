"""待办工具：写操作，落到 storage/todos.json。

写操作是 Agent 里最容易出事的一类（重复添加、误删），
所以这里把「新增」和「列出」严格分开，删改留到后续版本。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from agent.registry import tool

STORE = Path(os.getenv("TODO_STORE", "storage/todos.json"))


def _load() -> list[dict]:
    if not STORE.exists():
        return []
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save(items: list[dict]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


@tool(
    name="add_todo",
    description="新增一条待办事项。content 为要做的事，due 为截止时间（可留空）。",
    schema={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "待办内容，如 带伞 / 交实验报告"},
            "due": {"type": "string", "description": "截止时间，如 明天 / 2025-09-12"},
        },
        "required": ["content"],
    },
    group="daily",
)
def add_todo(content: str, due: str | None = None) -> dict:
    if not content.strip():
        raise ValueError("content 不能为空")

    items = _load()
    todo_id = max([i.get("id", 0) for i in items], default=0) + 1
    item = {
        "id": todo_id,
        "content": content.strip(),
        "due": due or "",
        "done": False,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    items.append(item)
    _save(items)
    return {"todo_id": todo_id, "status": "created", "content": item["content"]}


@tool(
    name="list_todos",
    description="列出待办事项。include_done 为 true 时包含已完成的。",
    schema={
        "type": "object",
        "properties": {
            "include_done": {"type": "boolean", "description": "是否包含已完成项，默认 false"},
        },
        "required": [],
    },
    group="daily",
)
def list_todos(include_done: bool = False) -> dict:
    items = _load()
    if not include_done:
        items = [i for i in items if not i.get("done")]
    return {"count": len(items), "items": items}
