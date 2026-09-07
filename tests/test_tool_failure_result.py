import io
import json
import sys
import types
import unittest
from contextlib import redirect_stdout

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

from agent_runtime.mcp_chat_agent import MCPChatAgent
from agent_runtime.mcp_client import result_for_model
from agent_runtime.skills import SkillRegistry


class FailingBridge:
    def list_tools(self):
        return []

    def call_tool(self, _name, _arguments):
        raise TimeoutError("analysis exceeded 180 seconds")


class DetectionBridge:
    def __init__(self, result):
        self.result = result

    def list_tools(self):
        return []

    def call_tool(self, _name, _arguments):
        return self.result


class ToolFailureResultTests(unittest.TestCase):
    def test_tool_exception_becomes_matching_tool_message(self):
        agent = MCPChatAgent.__new__(MCPChatAgent)
        agent.bridge = FailingBridge()
        agent.session_id = "session-1"
        agent.messages = [
            {"role": "system", "content": "system"},
            {"role": "system", "content": "summary"},
            {
                "role": "user",
                "content": MCPChatAgent._scoped_user_content(
                    "analyze", SkillRegistry.load_default().get("detection"), True
                ),
            },
        ]
        completions = iter([
            (
                "",
                "准备调用工具。",
                [{
                    "id": "call-1",
                    "name": "detect_eeg_events",
                    "arguments": "{}",
                }],
            ),
            ("工具执行失败，无法完成本次分析。", "根据错误生成答复。", []),
        ])
        agent._stream_completion = types.MethodType(
            lambda self, _tools, on_delta=None: next(completions),
            agent,
        )

        trace_events = []
        result = agent._run_stream_with_temporary_context(
            "analyze",
            [],
            SkillRegistry.load_default().get("detection"),
            on_trace=trace_events.append,
            trace_turn=3,
        )

        tool_message = next(message for message in agent.messages if message["role"] == "tool")
        error = json.loads(tool_message["content"])
        self.assertEqual(tool_message["tool_call_id"], "call-1")
        self.assertFalse(error["ok"])
        self.assertTrue(error["is_error"])
        self.assertEqual(error["error_type"], "TimeoutError")
        self.assertTrue(error["retryable"])
        self.assertEqual(result["response"], "工具执行失败，无法完成本次分析。")
        tool_updates = [
            event for event in trace_events
            if event.get("id") == "turn-3-tool-call-1"
        ]
        self.assertEqual([event["status"] for event in tool_updates], ["running", "error"])
        self.assertTrue(any(
            event.get("id") == "turn-3-end" and event.get("status") == "complete"
            for event in trace_events
        ))

    def test_mcp_error_is_explicitly_marked_for_the_model(self):
        content = result_for_model({
            "is_error": True,
            "structured_content": None,
            "content": ["invalid analysis window"],
        })

        error = json.loads(content)
        self.assertFalse(error["ok"])
        self.assertTrue(error["is_error"])
        self.assertEqual(error["message"], "invalid analysis window")

    def test_detection_result_sent_to_model_is_bounded_and_summarized(self):
        events = [
            {
                "event_type": "seizure_like_activity",
                "start_seconds": index,
                "end_seconds": index + 1,
                "channel": f"CH-{index % 4}",
                "brain_region": f"region-{index % 2}",
                "confidence": 0.7 + index / 1000,
            }
            for index in range(100)
        ]
        result = {
            "is_error": False,
            "structured_content": {
                "event_count": len(events),
                "events": events,
                "summary": "Detected 100 events.",
            },
            "content": [],
        }

        content = result_for_model(result, tool_name="detect_eeg_events")
        compact = json.loads(content)

        self.assertEqual(compact["event_count"], 100)
        self.assertEqual(len(compact["events"]), 24)
        self.assertTrue(compact["events_are_representative"])
        self.assertEqual(compact["events_omitted_from_model"], 76)
        self.assertEqual(compact["event_counts_by_channel"]["CH-0"], 25)
        self.assertEqual(compact["event_counts_by_brain_region"]["region-0"], 50)
        self.assertEqual(len(result["structured_content"]["events"]), 100)

    def test_detection_trace_output_matches_content_added_to_model_context(self):
        events = [
            {
                "event_type": "spike",
                "start_seconds": index,
                "end_seconds": index + 0.5,
                "channel": f"CH-{index % 4}",
                "brain_region": f"region-{index % 2}",
                "confidence": 0.9,
            }
            for index in range(30)
        ]
        tool_result = {
            "is_error": False,
            "structured_content": {
                "event_count": len(events),
                "events": events,
                "summary": "Detected 30 events.",
            },
            "content": [],
        }

        agent = MCPChatAgent.__new__(MCPChatAgent)
        agent.bridge = DetectionBridge(tool_result)
        agent.session_id = "session-1"
        agent.messages = [
            {"role": "system", "content": "system"},
            {"role": "system", "content": "summary"},
            {
                "role": "user",
                "content": MCPChatAgent._scoped_user_content(
                    "analyze", SkillRegistry.load_default().get("detection"), True
                ),
            },
        ]
        agent.session_summary = {
            "recording": None,
            "patient": {},
            "analyses": [],
            "findings": [],
            "reports": [],
            "conversation": [],
        }
        completions = iter([
            ((
                "",
                "准备调用检测工具。",
                [{
                    "id": "call-detection",
                    "name": "detect_eeg_events",
                    "arguments": '{"start": 0, "end": 60}',
                }],
            ), {
                "prompt_tokens": 1000,
                "prompt_cache_hit_tokens": 900,
                "prompt_cache_miss_tokens": 100,
            }),
            (("分析完成", "根据工具结果整理回答。", []), {
                "prompt_tokens": 5000,
                "prompt_cache_hit_tokens": 1300,
                "prompt_cache_miss_tokens": 3700,
            }),
        ])

        def fake_completion(self, _tools, on_delta=None):
            completion, usage = next(completions)
            self.last_completion_usage = usage
            return completion

        agent._stream_completion = types.MethodType(fake_completion, agent)

        trace_events = []
        console = io.StringIO()
        with redirect_stdout(console):
            run_result = agent._run_stream_with_temporary_context(
                "analyze",
                [],
                SkillRegistry.load_default().get("detection"),
                on_trace=trace_events.append,
                trace_turn=4,
            )

        tool_message = next(message for message in agent.messages if message["role"] == "tool")
        completed_tool_event = next(
            event
            for event in reversed(trace_events)
            if event.get("id") == "turn-4-tool-call-detection"
            and event.get("status") == "complete"
        )
        self.assertEqual(completed_tool_event["output"], tool_message["content"])
        self.assertEqual(len(json.loads(completed_tool_event["output"])["events"]), 24)
        completed_turn_event = next(
            event
            for event in trace_events
            if event.get("id") == "turn-4-end" and event.get("status") == "complete"
        )
        self.assertEqual(completed_turn_event["title"], "执行结果")
        self.assertNotIn("output", completed_turn_event)
        self.assertEqual(run_result["context_limit_tokens"], 43008)
        self.assertEqual(completed_turn_event["context_tokens"], run_result["context_tokens"])
        self.assertEqual(run_result["context_tokens"], 5000)
        self.assertEqual(run_result["cache_hit_tokens"], 2200)
        self.assertEqual(run_result["cache_miss_tokens"], 3800)
        self.assertAlmostEqual(run_result["cache_hit_rate"], 2200 / 6000)
        user_index = next(
            index
            for index, message in enumerate(agent.messages)
            if message.get("role") == "user"
        )
        scoped_user = agent.messages[user_index]
        self.assertEqual(scoped_user["role"], "user")
        self.assertIn('name="detection"', scoped_user["content"])
        self.assertIn(
            'applies_to="containing_user_message"',
            scoped_user["content"],
        )
        assistant_messages = [
            message for message in agent.messages if message.get("role") == "assistant"
        ]
        self.assertEqual(
            assistant_messages[0]["reasoning_content"],
            "准备调用检测工具。",
        )
        self.assertEqual(
            assistant_messages[1]["reasoning_content"],
            "根据工具结果整理回答。",
        )
        console_text = console.getvalue()
        self.assertIn(
            "[Agent model tokens] turn=4 step=1 input=1000tok "
            "cache_hit=900tok cache_miss=100tok context=1000/43008tok",
            console_text,
        )
        self.assertIn(
            "[Agent model tokens] turn=4 step=2 input=5000tok "
            "cache_hit=1300tok cache_miss=3700tok context=5000/43008tok",
            console_text,
        )


if __name__ == "__main__":
    unittest.main()
