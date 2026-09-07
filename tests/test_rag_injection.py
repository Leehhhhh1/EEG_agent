import types
import unittest
import sys
from pathlib import Path

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

from agent_runtime.mcp_chat_agent import GenerationCancelled, MCPChatAgent
from agent_runtime.skills.models import SkillSpec


class RAGInjectionTests(unittest.TestCase):
    def test_retrieval_context_is_current_turn_only(self):
        agent = MCPChatAgent.__new__(MCPChatAgent)
        agent.system_message = {"role": "system", "content": "fixed system prompt"}
        agent.session_summary = {
            "recording": None,
            "patient": {},
            "analyses": [],
            "findings": [],
            "reports": [],
        }
        agent.messages = [agent.system_message, agent._session_summary_message()]
        agent.session_id = None
        selected_skill = SkillSpec(
            name="general_eeg",
            description="fallback",
            priority=0,
            requires_session=False,
            trigger_keywords=(),
            routing_examples=(),
            allowed_tools=frozenset(),
            instructions="Answer the request.",
            path=Path("SKILL.md"),
        )
        def skill_route_must_not_run(*args, **kwargs):
            raise AssertionError("Skill routing must not run without an EDF session")

        agent.skill_registry = types.SimpleNamespace(
            select_with_details=skill_route_must_not_run
        )

        agent._retrieve_eeg_knowledge = types.MethodType(
            lambda self, query: (
                query + "\n<temporary_retrieved_eeg_knowledge>evidence</temporary_retrieved_eeg_knowledge>",
                [{"source": "guide.pdf"}],
            ),
            agent,
        )

        def fake_run(self, user_query, retrieval_results, skill, selection, **callbacks):
            self.assert_temporary = self.messages[-1]["content"]
            self.assert_skill = skill
            self.assert_selection = selection
            return {"response": "ok", "retrieved_sources": [retrieval_results[0]["source"]]}

        agent._run_stream_with_temporary_context = types.MethodType(fake_run, agent)
        original_system = agent.messages[0]["content"]

        result = agent.run_stream("question")

        self.assertIn("temporary_retrieved_eeg_knowledge", agent.assert_temporary)
        self.assertEqual(agent.messages[-1]["content"], "question")
        self.assertEqual(agent.messages[0]["content"], original_system)
        self.assertIsNone(agent.assert_skill)
        self.assertIsNone(agent.assert_selection)
        self.assertEqual(result["retrieved_sources"], ["guide.pdf"])

    def test_cancelled_generation_keeps_only_the_visible_partial_answer(self):
        agent = MCPChatAgent.__new__(MCPChatAgent)
        agent.system_message = {"role": "system", "content": "fixed system prompt"}
        agent.session_summary = {
            "recording": None,
            "patient": {},
            "analyses": [],
            "findings": [],
            "reports": [],
        }
        agent.messages = [agent.system_message, agent._session_summary_message()]
        agent.session_id = None
        agent.skill_registry = types.SimpleNamespace(
            select_with_details=lambda *args, **kwargs: None
        )
        agent._retrieve_eeg_knowledge = types.MethodType(
            lambda self, query: (query, []),
            agent,
        )

        def fake_run(self, user_query, retrieval_results, skill, selection, **callbacks):
            callbacks["on_delta"]("已经生成的内容")
            raise GenerationCancelled()

        agent._run_stream_with_temporary_context = types.MethodType(fake_run, agent)
        trace_events = []

        with self.assertRaises(GenerationCancelled) as raised:
            agent.run_stream("question", on_trace=trace_events.append)

        self.assertEqual(raised.exception.partial_response, "已经生成的内容")
        self.assertEqual(agent.messages[-2]["role"], "user")
        self.assertEqual(agent.messages[-1], {"role": "assistant", "content": "已经生成的内容"})
        self.assertEqual(trace_events[-1]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
