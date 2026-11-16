"""基础单测：注册表格式、参数校验、异常兜底、待办读写。

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import tempfile
import unittest

_TMP = tempfile.mkdtemp(prefix="agent-demo-test-")
os.environ["TODO_STORE"] = os.path.join(_TMP, "todos.json")

from agent import executor, registry  # noqa: E402
import tools  # noqa: F401,E402


class TestRegistry(unittest.TestCase):
    def test_tools_registered(self):
        names = set(registry.TOOL_REGISTRY)
        self.assertIn("get_weather", names)
        self.assertIn("add_todo", names)
        self.assertIn("list_todos", names)
        self.assertIn("summarize_text", names)

    def test_schema_shape(self):
        schemas = registry.tools_schema()
        for s in schemas:
            self.assertEqual(s["type"], "function")
            fn = s["function"]
            self.assertTrue(fn["name"])
            self.assertTrue(fn["description"])
            self.assertEqual(fn["parameters"]["type"], "object")

    def test_group_filter(self):
        schemas = registry.tools_schema(groups=["nonexistent"])
        self.assertEqual(schemas, [])

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ValueError):
            @registry.tool(name="get_weather", description="x", schema={"type": "object"})
            def _dup():
                pass


class TestValidator(unittest.TestCase):
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "day": {"type": "integer"}},
        "required": ["city"],
    }

    def test_missing_required(self):
        self.assertTrue(executor.validate(self.schema, {}))

    def test_unknown_param(self):
        errs = executor.validate(self.schema, {"city": "广州", "foo": 1})
        self.assertTrue(any("未知参数" in e for e in errs))

    def test_wrong_type(self):
        errs = executor.validate(self.schema, {"city": 123})
        self.assertTrue(any("类型" in e for e in errs))

    def test_ok(self):
        self.assertEqual(executor.validate(self.schema, {"city": "广州"}), [])


class TestExecutor(unittest.TestCase):
    def test_unknown_tool(self):
        out = executor.safe_execute("not_exist", "{}")
        self.assertIn("error", out)

    def test_bad_json(self):
        out = executor.safe_execute("get_weather", '{"city": 广州}')
        self.assertIn("error", out)

    def test_error_is_returned_not_raised(self):
        """工具内部抛错必须变成 observation，而不是把循环打断。"""
        out = executor.safe_execute("get_weather", '{"city": "广州", "date": "不存在的日期"}')
        self.assertIn("error", out)

    def test_weather_ok(self):
        out = executor.safe_execute("get_weather", '{"city": "广州", "date": "明天"}')
        self.assertEqual(out["city"], "广州")
        self.assertIn("weather", out)


class TestTodo(unittest.TestCase):
    def test_add_and_list(self):
        add = executor.safe_execute("add_todo", '{"content": "带伞", "due": "明天"}')
        self.assertEqual(add["status"], "created")
        listed = executor.safe_execute("list_todos", "{}")
        self.assertGreaterEqual(listed["count"], 1)
        self.assertTrue(any(i["content"] == "带伞" for i in listed["items"]))

    def test_empty_content_rejected(self):
        out = executor.safe_execute("add_todo", '{"content": "  "}')
        self.assertIn("error", out)


if __name__ == "__main__":
    unittest.main()
