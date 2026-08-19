import unittest

from agent_runtime.token_budget import (
    DeepSeekV4TokenCounter,
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
