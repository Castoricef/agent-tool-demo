"""ReAct 主循环。

三步收敛：
1. 把 messages + tools 交给模型；
2. 没有 tool_calls → 模型认为可以回答了，直接返回；
3. 有 tool_calls → 逐个执行，把结果作为 tool 消息回填，进入下一轮。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from agent import executor, registry

MAX_STEPS = 6
REQUEST_TIMEOUT = 30  # 秒，设在请求层，别设在工具内部


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ModelReply:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Step:
    index: int
    thought: str
    tool: str
    arguments: str
    observation: dict


@dataclass
class RunResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    stopped_by: str = "answer"  # answer | max_steps | error


class OpenAIClient:
    """OpenAI 兼容接口的薄封装，把响应压成 ModelReply。"""

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def create(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            timeout=REQUEST_TIMEOUT,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=c.function.arguments or "{}")
            for c in (msg.tool_calls or [])
        ]
        return ModelReply(content=msg.content, tool_calls=calls)


class ScriptedClient:
    """演示用：按预设脚本返回工具调用，不联网。

    只为在没有 API Key 的情况下展示循环怎么跑。真实使用请走 OpenAIClient。
    """

    def __init__(self) -> None:
        self._i = 0
        self._script = [
            ModelReply(
                content="需要先查天气",
                tool_calls=[ToolCall("call_1", "get_weather", '{"city": "广州", "date": "明天"}')],
            ),
            ModelReply(
                content="有雨，加个待办",
                tool_calls=[ToolCall("call_2", "add_todo", '{"content": "带伞", "due": "明天"}')],
            ),
            ModelReply(content="（演示模式）已查询明天广州天气，并把「带伞」加入待办。接真实模型后这里会是自然语言总结。", tool_calls=[]),
        ]

    def create(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        reply = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return reply


def build_client(demo: bool = False):
    if demo:
        return ScriptedClient()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未配置 OPENAI_API_KEY。想先看循环怎么跑，加 --demo 用脚本模拟模型。"
        )
    return OpenAIClient(
        model=os.getenv("MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )


def run(
    user_message: str,
    client=None,
    max_steps: int = MAX_STEPS,
    verbose: bool = False,
    tool_groups: list[str] | None = None,
) -> RunResult:
    client = client or build_client()
    tools = registry.tools_schema(tool_groups)
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "你是一个个人助理。需要外部信息或写操作时调用工具，"
                "不要凭猜测回答；信息足够后直接用中文给出简洁结论。"
            ),
        },
        {"role": "user", "content": user_message},
    ]

    steps: list[Step] = []

    for i in range(1, max_steps + 1):
        reply = client.create(messages, tools)
        messages.append(_assistant_message(reply))

        if not reply.tool_calls:
            return RunResult(answer=(reply.content or "").strip(), steps=steps)

        for call in reply.tool_calls:
            observation = executor.safe_execute(call.name, call.arguments)
            step = Step(i, reply.content or "", call.name, call.arguments, observation)
            steps.append(step)
            if verbose:
                _print_step(step)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(observation, ensure_ascii=False),
                }
            )

    return RunResult(
        answer=f"已达到最大轮数 {max_steps}，任务未收敛。最后一步：{steps[-1].tool if steps else '无'}",
        steps=steps,
        stopped_by="max_steps",
    )


def _assistant_message(reply: ModelReply) -> dict:
    """把 ModelReply 还原成 messages 里的 assistant 条目（含 tool_calls）。"""
    msg: dict = {"role": "assistant", "content": reply.content or ""}
    if reply.tool_calls:
        msg["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in reply.tool_calls
        ]
    return msg


def _print_step(step: Step) -> None:
    try:
        from rich.console import Console

        c = Console()
        c.print(f"[bold cyan]step {step.index}[/] [dim]thought:[/] {step.thought or '—'}")
        c.print(f"        [yellow]action[/] : {step.tool}")
        c.print(f"        [green]observ[/] : {json.dumps(step.observation, ensure_ascii=False)[:200]}")
    except ImportError:
        print(f"[step {step.index}] {step.thought or '—'} -> {step.tool} -> {step.observation}")
