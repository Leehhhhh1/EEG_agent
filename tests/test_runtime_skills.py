import sys
import types
import unittest

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

from agent_runtime.mcp_chat_agent import MCPChatAgent
from agent_runtime.skills import SkillRegistry
from agent_runtime.token_budget import DeepSeekV4TokenCounter


class RuntimeSkillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = SkillRegistry.load_default()

    def test_loads_expected_runtime_skills(self):
        self.assertEqual(
            {skill.name for skill in self.registry.all()},
            {"basic_information", "detection", "exploration", "general_eeg", "reporting"},
        )

    def test_selector_returns_exactly_one_best_skill(self):
        self.assertEqual(self.registry.select("检查前30秒有没有癫痫样活动").name, "detection")
        self.assertEqual(self.registry.select("分析背景节律和振幅").name, "exploration")
        self.assertEqual(self.registry.select("生成筛查报告").name, "reporting")
        self.assertEqual(self.registry.select("这是一条没有专用关键词的问题").name, "general_eeg")

    def test_keyword_match_does_not_call_description_selector(self):
        def unexpected_selector(query, skills):
            raise AssertionError("description selector should not be called")

        selected = self.registry.select("检查癫痫样放电", unexpected_selector)

        self.assertEqual(selected.name, "detection")

    def test_description_selector_is_used_after_keyword_miss(self):
        selected = self.registry.select(
            "帮我寻找可能存在的尖锐波形",
            lambda query, skills: "detection",
        )

        self.assertEqual(selected.name, "detection")

    def test_description_selector_can_choose_general_fallback(self):
        selected = self.registry.select(
            "我想了解一下这个问题",
            lambda query, skills: "general_eeg",
        )

        self.assertEqual(selected.name, "general_eeg")

    def test_invalid_or_failed_description_selection_falls_back(self):
        self.assertEqual(
            self.registry.select("没有关键词", lambda query, skills: "not_registered").name,
            "general_eeg",
        )

        def failed_selector(query, skills):
            raise TimeoutError("router timed out")

        self.assertEqual(
            self.registry.select("还是没有关键词", failed_selector).name,
            "general_eeg",
        )

    def test_deepseek_description_router_requires_strict_json(self):
        def agent_with_response(content):
            agent = MCPChatAgent.__new__(MCPChatAgent)
            agent.model = "test-model"
            response = types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content)
                    )
                ]
            )
            completions = types.SimpleNamespace(create=lambda **kwargs: response)
            agent.client = types.SimpleNamespace(
                chat=types.SimpleNamespace(completions=completions)
            )
            return agent

        skills = tuple(self.registry.all())
        valid_agent = agent_with_response('{"skill": "exploration"}')
        invalid_agent = agent_with_response("exploration")

        self.assertEqual(
            valid_agent._select_skill_by_description("分析一个片段", skills),
            "exploration",
        )
        self.assertIsNone(
            invalid_agent._select_skill_by_description("分析一个片段", skills)
        )

    def test_registry_rejects_unknown_mcp_tools(self):
        with self.assertRaisesRegex(ValueError, "unavailable MCP tools"):
            self.registry.validate_tools({"get_eeg_basic_information"})

    def test_tool_schema_uses_active_skill_whitelist(self):
        tools = [
            types.SimpleNamespace(
                name="get_eeg_basic_information",
                description="basic",
                inputSchema={
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "required": ["session_id"],
                },
            ),
            types.SimpleNamespace(
                name="detect_eeg_events",
                description="detect",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "start": {"type": "number"},
                    },
                    "required": ["session_id"],
                },
            ),
            types.SimpleNamespace(
                name="explore_eeg_segment",
                description="explore",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
        ]
        agent = MCPChatAgent.__new__(MCPChatAgent)
        agent.session_id = "session-test"
        agent.bridge = types.SimpleNamespace(list_tools=lambda: tools)

        schemas = agent._tool_schemas(self.registry.get("detection").allowed_tools)

        self.assertEqual(
            {schema["function"]["name"] for schema in schemas},
            {"get_eeg_basic_information", "detect_eeg_events"},
        )
        detect_schema = next(
            schema for schema in schemas
            if schema["function"]["name"] == "detect_eeg_events"
        )
        self.assertNotIn("session_id", detect_schema["function"]["parameters"]["properties"])
        self.assertNotIn("session_id", detect_schema["function"]["parameters"]["required"])

    def test_skill_instructions_are_request_scoped(self):
        agent = MCPChatAgent.__new__(MCPChatAgent)
        agent.system_message = {"role": "system", "content": "base"}
        agent.session_summary = {
            "recording": None,
            "patient": {},
            "analyses": [],
            "findings": [],
            "reports": [],
        }
        agent.messages = [
            agent.system_message,
            agent._session_summary_message(),
            {"role": "user", "content": "question"},
        ]
        agent.token_counter = DeepSeekV4TokenCounter(thinking_mode="thinking")
        agent.short_term_token_limit = 32_768
        skill_message = self.registry.get("detection").as_system_message()

        request_messages = agent._prepare_request_messages([], [skill_message])

        self.assertIs(request_messages[2], skill_message)
        self.assertFalse(any(message is skill_message for message in agent.messages))
        self.assertIn("active_eeg_skill", request_messages[2]["content"])


if __name__ == "__main__":
    unittest.main()
