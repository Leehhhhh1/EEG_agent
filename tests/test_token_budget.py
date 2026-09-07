import unittest
from types import SimpleNamespace

from agent_runtime.mcp_chat_agent import MCPChatAgent
from agent_runtime.token_budget import (
    DeepSeekV4TokenCounter,
    split_history_turns,
    trim_messages_to_token_limit,
)


class TokenBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counter = DeepSeekV4TokenCounter(thinking_mode="thinking")

    def test_count_includes_tools(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "system", "content": "summary"},
            {"role": "user", "content": "hello"},
        ]
        tools = [{
            "type": "function",
            "function": {
                "name": "demo",
                "description": "demo tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        self.assertGreater(
            self.counter.count_prompt(messages, tools),
            self.counter.count_prompt(messages, []),
        )

    def test_stream_collects_reasoning_without_emitting_it_as_visible_text(self):
        class FakeCompletions:
            def __init__(self):
                self.requests = []

            def create(self, **request):
                self.requests.append(request)
                return iter([
                    SimpleNamespace(
                        usage=None,
                        choices=[SimpleNamespace(delta=SimpleNamespace(
                            reasoning_content="内部推理",
                            content=None,
                            tool_calls=[],
                        ))],
                    ),
                    SimpleNamespace(
                        usage=None,
                        choices=[SimpleNamespace(delta=SimpleNamespace(
                            reasoning_content=None,
                            content="可见回答",
                            tool_calls=[],
                        ))],
                    ),
                    SimpleNamespace(
                        usage={"prompt_tokens": 10},
                        choices=[],
                    ),
                ])

        completions = FakeCompletions()
        agent = MCPChatAgent.__new__(MCPChatAgent)
        agent.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        agent.model = "deepseek-v4-flash"
        agent.messages = [
            {"role": "system", "content": "system"},
            {"role": "system", "content": "summary"},
            {"role": "user", "content": "question"},
        ]
        agent.token_counter = self.counter
        agent.short_term_token_limit = 32_768
        agent.compression_trigger_ratio = 0.8
        agent.last_prompt_token_count = 0
        agent._previous_prompt_token_ids = []

        visible_deltas = []
        content, reasoning, calls = agent._stream_completion(
            [], on_delta=visible_deltas.append
        )

        self.assertEqual(content, "可见回答")
        self.assertEqual(reasoning, "内部推理")
        self.assertEqual(calls, [])
        self.assertEqual(visible_deltas, ["可见回答"])
        self.assertEqual(
            completions.requests[0]["extra_body"],
            {"thinking": {"type": "enabled"}},
        )
        self.assertEqual(
            agent.last_prompt_diagnostics["local_tokens"],
            len(agent._previous_prompt_token_ids),
        )

    def test_scoped_skill_instruction_stays_with_following_user_turn(self):
        first_skill = {
            "role": "user",
            "content": (
                '<active_eeg_skill name="detection" '
                'applies_to="containing_user_message">instructions'
                "</active_eeg_skill><user_request>first question</user_request>"
            ),
        }
        second_skill = {
            "role": "user",
            "content": (
                '<active_eeg_skill name="reporting" '
                'applies_to="containing_user_message">instructions'
                "</active_eeg_skill><user_request>second question</user_request>"
            ),
        }
        messages = [
            first_skill,
            {"role": "assistant", "content": "first answer"},
            second_skill,
            {"role": "assistant", "content": "second answer"},
        ]

        turns = split_history_turns(messages)

        self.assertEqual(len(turns), 2)
        self.assertIs(turns[0][0], first_skill)
        self.assertIs(turns[1][0], second_skill)

    def test_trim_removes_oldest_complete_turn_only(self):
        fixed = [
            {"role": "system", "content": "system"},
            {"role": "system", "content": "summary"},
        ]
        first_turn = [
            {"role": "user", "content": "old " * 200},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-old",
                    "type": "function",
                    "function": {"name": "demo", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call-old", "content": "old result"},
            {"role": "assistant", "content": "old answer"},
        ]
        latest_turn = [
            {"role": "user", "content": "latest question"},
            {"role": "assistant", "content": "latest answer"},
        ]
        without_oldest = fixed + latest_turn
        limit = self.counter.count_prompt(without_oldest, [])

        trimmed, token_count = trim_messages_to_token_limit(
            fixed + first_turn + latest_turn,
            [],
            self.counter,
            token_limit=limit,
        )

        self.assertEqual(trimmed, without_oldest)
        self.assertLessEqual(token_count, limit)
        self.assertFalse(any(message.get("tool_call_id") == "call-old" for message in trimmed))

    def test_no_turn_count_limit_when_tokens_fit(self):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "system", "content": "summary"},
        ]
        for index in range(20):
            messages.extend([
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ])

        trimmed, _ = trim_messages_to_token_limit(
            messages,
            [],
            self.counter,
            token_limit=32_768,
        )

        self.assertEqual(trimmed, messages)


if __name__ == "__main__":
    unittest.main()
