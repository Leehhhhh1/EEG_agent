import sys
import types
import unittest

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

from agent_runtime.mcp_chat_agent import MCPChatAgent
from agent_runtime.skills import SemanticSkillSelector, SkillRegistry
from agent_runtime.skills.models import SemanticCandidate, SemanticSelection
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

    def test_keyword_selector_returns_specialized_skill_and_unmatched_returns_none(self):
        self.assertEqual(self.registry.select("检查前30秒有没有发作").name, "detection")
        self.assertEqual(self.registry.select("分析背景节律和振幅").name, "exploration")
        self.assertEqual(self.registry.select("生成筛查报告").name, "reporting")
        self.assertIsNone(self.registry.select("这是一条没有专用关键词的问题"))

    def test_specialized_skills_have_semantic_examples(self):
        for skill in self.registry.all():
            if skill.name != "general_eeg":
                self.assertGreaterEqual(len(skill.routing_examples), 3)

    def test_keyword_match_does_not_call_semantic_selector(self):
        def unexpected_selector(query, skills):
            raise AssertionError("semantic selector should not be called")

        selected = self.registry.select("检测癫痫样放电", unexpected_selector)

        self.assertEqual(selected.name, "detection")

    def test_semantic_selector_is_used_after_keyword_miss(self):
        semantic_result = SemanticSelection(
            accepted_name="detection",
            candidates=(SemanticCandidate("detection", 0.82, ("寻找尖锐波形",)),),
            top_score=0.82,
            margin=0.14,
        )
        selection = self.registry.select_with_details(
            "帮我寻找可能存在的尖锐波形",
            lambda query, skills: semantic_result,
        )

        self.assertEqual(selection.skill.name, "detection")
        self.assertEqual(selection.source, "embedding")
        self.assertAlmostEqual(selection.top_score, 0.82)

    def test_medium_confidence_semantic_result_uses_restricted_general_skill(self):
        semantic_result = SemanticSelection(
            accepted_name=None,
            candidates=(SemanticCandidate("exploration", 0.51, ("查看波形",)),),
            top_score=0.51,
            margin=0.01,
        )
        selection = self.registry.select_with_details(
            "我想了解一下这个问题", lambda query, skills: semantic_result
        )

        self.assertEqual(selection.skill.name, "general_eeg")
        self.assertEqual(selection.source, "general")
        self.assertAlmostEqual(selection.top_score, 0.51)

    def test_low_confidence_semantic_result_selects_no_skill(self):
        semantic_result = SemanticSelection(
            accepted_name=None,
            candidates=(SemanticCandidate("exploration", 0.44, ("查看波形",)),),
            top_score=0.44,
            margin=0.01,
        )
        selection = self.registry.select_with_details(
            "我想了解一下这个问题", lambda query, skills: semantic_result
        )

        self.assertIsNone(selection.skill)
        self.assertEqual(selection.source, "no_skill")

    def test_failed_semantic_selection_selects_no_skill(self):
        def failed_selector(query, skills):
            raise TimeoutError("router timed out")

        self.assertIsNone(self.registry.select("还是没有关键词", failed_selector))

    def test_bge_semantic_selector_uses_score_and_margin_thresholds(self):
        class FakeEmbedder:
            vectors = {
                "detection description": [1.0, 0.0],
                "detect example": [1.0, 0.0],
                "exploration description": [0.0, 1.0],
                "explore example": [0.0, 1.0],
                "query": [0.95, 0.05],
            }

            def encode(self, texts):
                return [self.vectors[text] for text in texts]

        detection = self.registry.get("detection")
        exploration = self.registry.get("exploration")
        skills = (
            type(detection)(
                **{**detection.__dict__, "description": "detection description", "routing_examples": ("detect example",)}
            ),
            type(exploration)(
                **{**exploration.__dict__, "description": "exploration description", "routing_examples": ("explore example",)}
            ),
        )
        result = SemanticSkillSelector(
            FakeEmbedder(), min_score=0.60, min_margin=0.10
        ).select("query", skills)

        self.assertEqual(result.accepted_name, "detection")
        self.assertGreater(result.margin, 0.10)

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
