"""DeepSeek-V4 prompt token counting and token-only history trimming."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from third_party.deepseek_v4.encoding_dsv4 import encode_messages


DEFAULT_SHORT_TERM_TOKEN_LIMIT = 32 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = PROJECT_ROOT / "third_party" / "deepseek_v4" / "tokenizer.json"


class DeepSeekV4TokenCounter:
    """Count the exact tokens produced by DeepSeek's official V4 encoder."""

    def __init__(self, tokenizer_path: Path = TOKENIZER_PATH, thinking_mode: str = "thinking"):
        if thinking_mode not in {"chat", "thinking"}:
            raise ValueError("thinking_mode must be 'chat' or 'thinking'.")
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"DeepSeek tokenizer not found: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.thinking_mode = thinking_mode

    def count_prompt(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
        """Render OpenAI-format messages/tools with the official V4 prompt encoder and count tokens."""
        encoded_messages = copy.deepcopy(messages)
        if tools:
            if not encoded_messages or encoded_messages[0].get("role") != "system":
                raise ValueError("DeepSeek tool schemas require a leading system message.")
            encoded_messages[0]["tools"] = copy.deepcopy(tools)
        prompt = encode_messages(encoded_messages, thinking_mode=self.thinking_mode)
        return len(self.tokenizer.encode(prompt, add_special_tokens=False).ids)


def _split_history_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split history at user-message boundaries while keeping tool exchanges intact."""
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


def trim_messages_to_token_limit(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    counter: DeepSeekV4TokenCounter,
    token_limit: int = DEFAULT_SHORT_TERM_TOKEN_LIMIT,
    preserved_message_count: int = 2,
) -> tuple[list[dict[str, Any]], int]:
    """Remove oldest complete turns until the rendered prompt fits the token limit."""
    if token_limit <= 0:
        raise ValueError("token_limit must be positive.")
    if len(messages) < preserved_message_count:
        raise ValueError("The message list does not contain the required preserved messages.")

    preserved = messages[:preserved_message_count]
    turns = _split_history_turns(messages[preserved_message_count:])

    while True:
        candidate = preserved + [message for turn in turns for message in turn]
        token_count = counter.count_prompt(candidate, tools)
        if token_count <= token_limit:
            return candidate, token_count
        if len(turns) <= 1:
            raise ValueError(
                f"The fixed prompt and latest conversation turn use {token_count} tokens, "
                f"which exceeds the {token_limit}-token short-term memory limit."
            )
        turns.pop(0)
