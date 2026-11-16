"""天气工具。

默认返回**确定性演示数据**（同一城市+日期结果固定），不联网也能跑。
接真实服务只需替换 _fetch()，其余不用动。
"""

from __future__ import annotations

import hashlib
from datetime import date as Date, timedelta

from agent.registry import tool

_WEATHERS = ["晴", "多云", "阴", "小雨", "中雨", "大雨", "雷阵雨"]


def _normalize_date(value: str | None) -> str:
    """把「今天 / 明天 / 后天 / 2025-09-12」统一成 YYYY-MM-DD。"""
    today = Date.today()
    if not value or value in {"今天", "today"}:
        return today.isoformat()
    if value in {"明天", "tomorrow"}:
        return (today + timedelta(days=1)).isoformat()
    if value in {"后天", "the day after tomorrow"}:
        return (today + timedelta(days=2)).isoformat()
    return value  # 假定已是 ISO 日期，非法值由工具内部抛错 → 由 executor 兜底回填


def _fetch(city: str, day: str) -> dict:
    """演示数据：用 hash 生成稳定的伪天气。

    换成真实接口时，把这里替换成 HTTP 请求即可，例如：
        resp = requests.get("https://api.example.com/weather", params={...}, timeout=5)
        return resp.json()
    """
    seed = int(hashlib.md5(f"{city}{day}".encode()).hexdigest()[:6], 16)
    weather = _WEATHERS[seed % len(_WEATHERS)]
    low = 18 + seed % 10
    high = low + 5 + seed % 6
    return {
        "city": city,
        "date": day,
        "weather": weather,
        "temp": f"{low}~{high}℃",
        "source": "demo-data",
    }


@tool(
    name="get_weather",
    description="查询指定城市在指定日期的天气。日期支持「今天/明天/后天」或 YYYY-MM-DD，缺省为今天。",
    schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名，如 广州"},
            "date": {"type": "string", "description": "今天 / 明天 / 后天，或 YYYY-MM-DD"},
        },
        "required": ["city"],
    },
    group="daily",
)
def get_weather(city: str, date: str | None = None) -> dict:
    # 注意：参数名 date 会遮蔽 datetime.date，所以上面用 Date 别名导入
    day = _normalize_date(date)
    # 校验日期格式，非法就让 executor 把错误回填给模型
    try:
        Date.fromisoformat(day)
    except ValueError:
        raise ValueError(f"无法识别的日期：{day}，请用 YYYY-MM-DD 或「明天」") from None
    return _fetch(city, day)
