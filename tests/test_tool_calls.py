import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.grok.tool_calls import (
    ToolCallPayloadError,
    ToolCallPlanner,
    parse_tool_call_payload,
)


class ToolCallParserTests(unittest.TestCase):
    def test_parse_single_tool_call_object(self):
        raw = (
            "<|tool_call|>"
            "{\"function\":{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Shanghai\"}}}"
        )
        calls = parse_tool_call_payload(raw, "<|tool_call|>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertIn("city", calls[0]["function"]["arguments"])

    def test_parse_tool_calls_array(self):
        raw = (
            "<|tool_call|>"
            "{\"tool_calls\":[{\"id\":\"call_1\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\\\"abc\\\"}\"}}]}"
        )
        calls = parse_tool_call_payload(raw, "<|tool_call|>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_1")
        self.assertEqual(calls[0]["function"]["name"], "search")

    def test_parse_invalid_payload_raises(self):
        with self.assertRaises(ToolCallPayloadError):
            parse_tool_call_payload("<|tool_call|>{\"a\":1}", "<|tool_call|>")


class ToolCallPlannerTests(unittest.TestCase):
    def test_planner_plain_text(self):
        planner = ToolCallPlanner(prefix="<|tool_call|>", strict=True)
        result = planner.plan("hello world", tools=[{"function": {"name": "search"}}])
        self.assertFalse(result.is_tool_call)
        self.assertEqual(result.tool_calls, [])

    def test_planner_tool_call_success(self):
        planner = ToolCallPlanner(prefix="<|tool_call|>", strict=True)
        result = planner.plan(
            "<|tool_call|>{\"function\":{\"name\":\"search\",\"arguments\":{\"q\":\"abc\"}}}",
            tools=[{"type": "function", "function": {"name": "search"}}],
            tool_choice="auto",
        )
        self.assertTrue(result.is_tool_call)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].function.name, "search")

    def test_planner_required_function_mismatch(self):
        planner = ToolCallPlanner(prefix="<|tool_call|>", strict=True)
        result = planner.plan(
            "<|tool_call|>{\"function\":{\"name\":\"search\",\"arguments\":{}}}",
            tools=[{"type": "function", "function": {"name": "search"}}],
            tool_choice={"type": "function", "function": {"name": "lookup"}},
        )
        self.assertTrue(result.is_tool_call)
        self.assertEqual(result.tool_calls, [])
        self.assertIsNotNone(result.error)

    def test_planner_non_strict_fallback(self):
        planner = ToolCallPlanner(prefix="<|tool_call|>", strict=False)
        result = planner.plan(
            "<|tool_call|>{\"bad\":1}",
            tools=[{"type": "function", "function": {"name": "search"}}],
            tool_choice="auto",
        )
        self.assertFalse(result.is_tool_call)
        self.assertEqual(result.tool_calls, [])


if __name__ == "__main__":
    unittest.main()
