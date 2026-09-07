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
            types.SimpleNamespace(
                name="generate_eeg_report",
                description="report",
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

        agent.skill_registry = self.registry
        detection_schemas = agent._model_tool_schemas(self.registry.get("detection"))
        basic_schemas = agent._model_tool_schemas(self.registry.get("basic_information"))
        expected_names = [
            "detect_eeg_events",
            "explore_eeg_segment",
            "generate_eeg_report",
            "get_eeg_basic_information",
        ]
        self.assertEqual(
            [schema["function"]["name"] for schema in detection_schemas],
            expected_names,
        )
        self.assertEqual(detection_schemas, basic_schemas)

    def test_skill_message_declares_its_runtime_tool_allowlist(self):
        instruction = self.registry.get("detection").as_instruction_block()

        self.assertIn("<allowed_tools>", instruction)
        self.assertIn("- detect_eeg_events", instruction)
        self.assertIn("- get_eeg_basic_information", instruction)
        self.assertNotIn("- explore_eeg_segment", instruction)
        self.assertIn('applies_to="containing_user_message"', instruction)

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
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
            {
                "role": "user",
                "content": agent._scoped_user_content(
                    "question", self.registry.get("detection"), True
                ),
            },
        ]
        agent.token_counter = DeepSeekV4TokenCounter(thinking_mode="thinking")
        agent.short_term_token_limit = 32_768

        request_messages = agent._prepare_request_messages([])

        self.assertEqual(
            [message["role"] for message in request_messages],
            ["system", "system", "user", "assistant", "user"],
        )
        self.assertIs(request_messages[-1], agent.messages[-1])
        self.assertIn("active_eeg_skill", request_messages[-1]["content"])
        self.assertEqual(
            agent._user_request_text(request_messages[-1]["content"]), "question"
        )
        self.assertEqual(
            sum(message["role"] == "system" for message in request_messages), 2
        )

    def test_skill_instructions_stay_before_current_user_during_tool_rounds(self):
        agent = MCPChatAgent.__new__(MCPChatAgent)
        agent.system_message = {"role": "system", "content": "base"}
        agent.session_summary = {
            "recording": None,
            "patient": {},
            "analyses": [],
            "findings": [],
            "reports": [],
        }
        current_user = {
            "role": "user",
            "content": agent._scoped_user_content(
                "question", self.registry.get("detection"), True
            ),
        }
        agent.messages = [
            agent.system_message,
            agent._session_summary_message(),
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
            current_user,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "detect_eeg_events", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ]
        agent.token_counter = DeepSeekV4TokenCounter(thinking_mode="thinking")
        agent.short_term_token_limit = 32_768

        request_messages = agent._prepare_request_messages([])

        user_index = request_messages.index(current_user)
        self.assertIn("active_eeg_skill", request_messages[user_index]["content"])
        self.assertEqual(
            sum(message["role"] == "system" for message in request_messages), 2
        )
        self.assertEqual(request_messages[-1]["role"], "tool")

    def test_tool_memory_does_not_rewrite_prompt_before_compaction(self):
        agent = MCPChatAgent.__new__(MCPChatAgent)
        agent.system_message = {"role": "system", "content": "base"}
        agent.session_summary = {
            "recording": None,
            "patient": {},
            "analyses": [],
            "findings": [],
            "reports": [],
            "conversation": [],
        }
        summary_message = agent._session_summary_message()
        agent.messages = [agent.system_message, summary_message]

        agent._remember_tool_result(
            "detect_eeg_events",
            {
                "is_error": False,
                "structured_content": {
                    "analysis_window": {"start_seconds": 0, "end_seconds": 30},
                    "event_count": 0,
                    "events": [],
                },
                "content": [],
            },
        )

        self.assertIs(agent.messages[1], summary_message)
        self.assertNotIn("已检测", agent.messages[1]["content"])
        self.assertIn("已检测", agent.session_summary["analyses"][0])

    def test_crossing_threshold_compacts_oldest_complete_turns(self):
        class CharacterCounter:
            @staticmethod
            def count_prompt(messages, tools):
                return sum(len(str(message)) for message in messages) + len(str(tools))

        agent = MCPChatAgent.__new__(MCPChatAgent)
        agent.system_message = {"role": "system", "content": "base"}
        agent.session_summary = {
            "recording": None,
            "patient": {},
            "analyses": ["已检测 0-30 秒，发现 0 个筛查事件"],
            "findings": [],
            "reports": [],
            "conversation": [],
        }
        original_summary = agent._session_summary_message()
        old_turn = [
            {"role": "user", "content": "分析第一段" + "旧" * 180},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "old-call",
                    "type": "function",
                    "function": {"name": "detect_eeg_events", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "old-call", "content": "详细结果" + "值" * 180},
            {"role": "assistant", "content": "第一段未发现筛查事件"},
        ]
        latest_turn = [
            {"role": "user", "content": "现在分析第二段"},
            {"role": "assistant", "content": "正在处理"},
        ]
        agent.messages = [agent.system_message, original_summary, *old_turn, *latest_turn]
        agent.token_counter = CharacterCounter()
        agent.short_term_token_limit = 900
        agent.compression_trigger_ratio = 0.80
        agent.compression_target_ratio = 0.55

        request_messages = agent._prepare_request_messages([])

        self.assertIsNot(agent.messages[1], original_summary)
        self.assertIn("已压缩对话", agent.messages[1]["content"])
        self.assertIn("分析第一段", agent.messages[1]["content"])
        self.assertFalse(any(message.get("tool_call_id") == "old-call" for message in agent.messages))
        self.assertTrue(any(message.get("content") == "现在分析第二段" for message in request_messages))


if __name__ == "__main__":
    unittest.main()
