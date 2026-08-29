import json
import sys
import types
import unittest

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


class ToolFailureResultTests(unittest.TestCase):
    def test_tool_exception_becomes_matching_tool_message(self):
        agent = MCPChatAgent.__new__(MCPChatAgent)
        agent.bridge = FailingBridge()
        agent.session_id = "session-1"
        agent.messages = [
            {"role": "system", "content": "system"},
            {"role": "system", "content": "summary"},
            {"role": "user", "content": "analyze"},
        ]
        completions = iter([
            (
                "",
                [{
                    "id": "call-1",
                    "name": "detect_eeg_events",
                    "arguments": "{}",
                }],
            ),
            ("工具执行失败，无法完成本次分析。", []),
        ])
        agent._stream_completion = types.MethodType(
            lambda self, _tools, on_delta=None, transient_system_messages=None: next(completions),
            agent,
        )

        result = agent._run_stream_with_temporary_context(
            "analyze",
            [],
            SkillRegistry.load_default().get("detection"),
        )

        tool_message = next(message for message in agent.messages if message["role"] == "tool")
        error = json.loads(tool_message["content"])
        self.assertEqual(tool_message["tool_call_id"], "call-1")
        self.assertFalse(error["ok"])
        self.assertTrue(error["is_error"])
        self.assertEqual(error["error_type"], "TimeoutError")
        self.assertTrue(error["retryable"])
        self.assertEqual(result["response"], "工具执行失败，无法完成本次分析。")

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


if __name__ == "__main__":
    unittest.main()
