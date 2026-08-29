import types
import unittest
import sys
from pathlib import Path

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub

from agent_runtime.mcp_chat_agent import MCPChatAgent
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
        selected_skill = SkillSpec(
            name="general_eeg",
            description="fallback",
            priority=0,
            requires_session=False,
            trigger_keywords=(),
            allowed_tools=frozenset(),
            instructions="Answer the request.",
            path=Path("SKILL.md"),
        )
        agent.skill_registry = types.SimpleNamespace(
            select=lambda query, description_selector=None: selected_skill
        )

        agent._retrieve_eeg_knowledge = types.MethodType(
            lambda self, query: (
                query + "\n<temporary_retrieved_eeg_knowledge>evidence</temporary_retrieved_eeg_knowledge>",
                [{"source": "guide.pdf"}],
            ),
            agent,
        )

        def fake_run(self, user_query, retrieval_results, skill, **callbacks):
            self.assert_temporary = self.messages[-1]["content"]
            self.assert_skill = skill.name
            return {"response": "ok", "retrieved_sources": [retrieval_results[0]["source"]]}

        agent._run_stream_with_temporary_context = types.MethodType(fake_run, agent)
        original_system = agent.messages[0]["content"]

        result = agent.run_stream("question")

        self.assertIn("temporary_retrieved_eeg_knowledge", agent.assert_temporary)
        self.assertEqual(agent.messages[-1]["content"], "question")
        self.assertEqual(agent.messages[0]["content"], original_system)
        self.assertEqual(agent.assert_skill, "general_eeg")
        self.assertEqual(result["retrieved_sources"], ["guide.pdf"])


if __name__ == "__main__":
    unittest.main()
