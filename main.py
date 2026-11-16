"""入口。

    python main.py                                  # 交互模式
    python main.py "明天广州会下雨吗？下雨就加个待办"   # 单次执行
    python main.py --verbose "..."                  # 打印每轮 thought/action/observation
    python main.py --demo                           # 不联网，用脚本模拟模型，看循环怎么跑
"""

from __future__ import annotations

import argparse
import sys

try:  # .env 可选，没装 python-dotenv 也能跑
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agent import loop  # noqa: E402
import tools  # noqa: F401,E402  导入即注册工具


def ask(text: str, args) -> loop.RunResult:
    client = loop.build_client(demo=args.demo)
    result = loop.run(
        text,
        client=client,
        max_steps=args.max_steps,
        verbose=args.verbose or args.demo,
    )
    print("\n" + result.answer)
    if result.stopped_by != "answer":
        print(f"\n[停止原因] {result.stopped_by}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="工具调用型个人助理 Agent")
    ap.add_argument("question", nargs="*", help="要处理的需求；不传则进入交互模式")
    ap.add_argument("--verbose", action="store_true", help="打印每一轮调度轨迹")
    ap.add_argument("--demo", action="store_true", help="离线演示：脚本模拟模型，无需 API Key")
    ap.add_argument("--max-steps", type=int, default=loop.MAX_STEPS, help="最大轮数")
    args = ap.parse_args()

    text = " ".join(args.question).strip()

    if args.demo and not text:
        text = "明天广州会下雨吗？如果下雨就把「带伞」加到待办里"

    if text:
        try:
            ask(text, args)
        except RuntimeError as e:
            sys.exit(str(e))
        return

    print("个人助理已就绪。输入需求回车执行，输入 q 退出。\n")
    while True:
        try:
            line = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in {"q", "quit", "exit"}:
            break
        if not line:
            continue
        try:
            ask(line, args)
        except RuntimeError as e:
            sys.exit(str(e))


if __name__ == "__main__":
    main()
