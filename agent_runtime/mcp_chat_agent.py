"""基于会话级 MCP EEG 工具的 DeepSeek 原生函数调用 agent。"""

import copy
import hashlib
import json
import os
import threading
import time
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI
from .mcp_client import MCPClientBridge, result_for_model
from .skills import SemanticSkillSelector, SkillRegistry, SkillSelection, SkillSpec
from .token_budget import (
    DEFAULT_SHORT_TERM_TOKEN_LIMIT,
    DeepSeekV4TokenCounter,
    split_history_turns,
)


MAX_MEMORY_ITEMS = 12
MAX_TOOL_ROUNDS = 8
DEFAULT_COMPRESSION_TRIGGER_RATIO = 0.80
DEFAULT_COMPRESSION_TARGET_RATIO = 0.55
MAX_MEMORY_TEXT_CHARS = 320
NO_SKILL_INSTRUCTION = (
    '<active_eeg_skill name="none" applies_to="containing_user_message">\n'
    "当前请求未匹配可安全执行的 EEG 分析任务。不要调用 EEG 工具，也不要对当前记录作出推断。"
    "若用户可能在询问当前记录，请要求其明确分析目标、时间范围，以及必要时的导联。"
    "若是一般 EEG 知识问题，可根据当前轮检索到的参考资料回答。"
    "\n</active_eeg_skill>"
)
USER_REQUEST_OPEN = "<user_request>"
USER_REQUEST_CLOSE = "</user_request>"


class GenerationCancelled(RuntimeError):
    """Raised when the user cancels the active generation."""

    def __init__(self, partial_response: str = ""):
        super().__init__("Generation cancelled by user.")
        self.partial_response = partial_response


def _as_structured_result(result: dict[str, Any]) -> Any:
    """从 MCP 返回结果中提取结构化内容。"""
    structured = result.get("structured_content")
    if structured is not None:
        return structured
    content = result.get("content") or []
    if len(content) == 1:
        try:
            return json.loads(content[0])
        except json.JSONDecodeError:
            return None
    return None


def _tool_exception_result(tool_name: str, exc: Exception) -> dict[str, Any]:
    """Convert a transport/runtime exception into a model-visible tool result."""
    message = str(exc).strip() or exc.__class__.__name__
    retryable = isinstance(exc, (TimeoutError, ConnectionError))
    details = {
        "ok": False,
        "error_type": exc.__class__.__name__,
        "message": f"工具 {tool_name} 调用失败：{message}",
        "retryable": retryable,
    }
    return {
        "is_error": True,
        "structured_content": details,
        "content": [details["message"]],
    }


def _trace_text(value: Any, limit: int = 24000) -> str:
    """Render bounded, readable trace details without changing model data."""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…（轨迹详情已截断，共 {len(text)} 个字符）"


def _emit_trace(callback, event: dict[str, Any]) -> None:
    """Publish optional UI telemetry without retaining it in model history."""
    if callback is not None:
        callback(event)


def _completion_usage_dict(usage: Any) -> dict[str, int]:
    """Normalize OpenAI-compatible and DeepSeek-specific usage fields."""
    if usage is None:
        return {}
    if isinstance(usage, dict):
        raw = usage
    elif hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    else:
        raw = {}
    fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
    )
    normalized: dict[str, int] = {}
    for field in fields:
        value = raw.get(field, getattr(usage, field, None))
        if isinstance(value, (int, float)):
            normalized[field] = int(value)
    details = raw.get("completion_tokens_details", getattr(usage, "completion_tokens_details", None))
    if details is not None:
        if isinstance(details, dict):
            reasoning_tokens = details.get("reasoning_tokens")
        else:
            reasoning_tokens = getattr(details, "reasoning_tokens", None)
        if isinstance(reasoning_tokens, (int, float)):
            normalized["reasoning_tokens"] = int(reasoning_tokens)
    return normalized


def _limit_items(items: list[Any], limit: int = MAX_MEMORY_ITEMS) -> list[Any]:
    """限制列表长度，避免摘要内容过长。"""
    return items[-limit:]


class MCPChatAgent:
    def __init__(self, bridge: MCPClientBridge, session_id: str | None = None, initial_basic_info: dict[str, Any] | None = None):
        """初始化对象状态。"""
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("api_key")
        if not api_key:
            raise ValueError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY in .env.")
        self.client = OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.short_term_token_limit = int(
            os.getenv("SHORT_TERM_MEMORY_TOKENS", str(DEFAULT_SHORT_TERM_TOKEN_LIMIT))
        )
        self.compression_trigger_ratio = float(
            os.getenv("MEMORY_COMPRESSION_TRIGGER_RATIO", str(DEFAULT_COMPRESSION_TRIGGER_RATIO))
        )
        self.compression_target_ratio = float(
            os.getenv("MEMORY_COMPRESSION_TARGET_RATIO", str(DEFAULT_COMPRESSION_TARGET_RATIO))
        )
        if not 0 < self.compression_target_ratio < self.compression_trigger_ratio <= 1:
            raise ValueError(
                "Memory compression ratios must satisfy 0 < target < trigger <= 1."
            )
        self.token_counter = DeepSeekV4TokenCounter(
            thinking_mode=os.getenv("DEEPSEEK_THINKING_MODE", "thinking")
        )
        self.last_prompt_token_count = 0
        self.last_prompt_diagnostics: dict[str, Any] = {}
        self._previous_prompt_token_ids: list[int] = []
        self.rag_retriever = None
        self.semantic_skill_selector = None
        self.trace_turn = 0
        self._cancel_event = threading.Event()
        self._stream_lock = threading.Lock()
        self._active_stream = None
        self.bridge = bridge
        self.skill_registry = SkillRegistry.load_default()
        self.skill_registry.validate_tools(tool.name for tool in self.bridge.list_tools())
        self.session_id = session_id
        self.system_message = {
            "role": "system",
            "content": (
                "你是 EEGAgent 脑电图辅助分析助手，请使用中文回答。未绑定 EEG 记录时，可以回答一般 EEG 知识问题，"
                "但必须明确说明无法分析具体记录。绑定 EEG 记录后，仅根据 MCP 工具结果描述该记录的 EEG 发现。"
                "自动筛查不是临床诊断；涉及异常、发作或癫痫样活动时，必须说明需要由合格脑电图专业人员复核。"
                "不要编造工具未提供的时间、导联、脑区、置信度或诊断结论。"
                "用户消息中的 <active_eeg_skill> 仅适用于同一条消息内的 <user_request>，以及该请求产生的工具调用循环；"
                "不得把旧轮次的 Skill 指令应用到后续用户请求。"
            ),
        }
        self.session_summary = {
            "recording": None,
            "patient": {},
            "analyses": [],
            "findings": [],
            "reports": [],
            "conversation": [],
        }
        if initial_basic_info:
            self._remember_tool_result(
                "get_eeg_basic_information",
                {"is_error": False, "structured_content": initial_basic_info, "content": []},
            )
        self.messages = [self.system_message, self._session_summary_message()]

    def cancel_current_run(self) -> None:
        """Request cooperative cancellation and close the active model stream."""
        self._ensure_cancel_state()
        self._cancel_event.set()
        with self._stream_lock:
            stream = self._active_stream
        if stream is not None:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _check_cancelled(self) -> None:
        self._ensure_cancel_state()
        if self._cancel_event.is_set():
            raise GenerationCancelled()

    def _ensure_cancel_state(self) -> None:
        """Initialize cancellation state for normal and test-only construction."""
        if not hasattr(self, "_cancel_event"):
            self._cancel_event = threading.Event()
        if not hasattr(self, "_stream_lock"):
            self._stream_lock = threading.Lock()
        if not hasattr(self, "_active_stream"):
            self._active_stream = None

    def set_rag_retriever(self, retriever) -> None:
        """Attach the shared RAG runtime and reuse its BGE-M3 embedder for Skill routing."""
        if self.rag_retriever is retriever and self.semantic_skill_selector is not None:
            return
        self.rag_retriever = retriever
        self.semantic_skill_selector = SemanticSkillSelector(retriever.embedder)

    def attach_session(self, session_id: str, basic_info: dict[str, Any] | None = None) -> None:
        """Attach an EEG recording without discarding conversation history."""
        self._discard_tool_messages()
        self._previous_prompt_token_ids = []
        self.session_id = session_id
        self.session_summary = {
            "recording": None,
            "patient": {},
            "analyses": [],
            "findings": [],
            "reports": [],
            "conversation": [],
        }
        if basic_info:
            self._remember_tool_result(
                "get_eeg_basic_information",
                {"is_error": False, "structured_content": basic_info, "content": []},
            )
        self._refresh_session_summary_message()

    def detach_session(self) -> None:
        """Remove EEG-specific state while preserving conversation history."""
        self._discard_tool_messages()
        self._previous_prompt_token_ids = []
        self.session_id = None
        self.session_summary = {
            "recording": None,
            "patient": {},
            "analyses": [],
            "findings": [],
            "reports": [],
            "conversation": [],
        }
        self._refresh_session_summary_message()

    def _discard_tool_messages(self) -> None:
        """Drop session-bound tool exchanges while retaining ordinary chat."""
        self.messages = [
            message for message in self.messages
            if message.get("role") != "tool" and not message.get("tool_calls")
        ]

    def reset(self) -> None:
        """重置当前对象的内部状态。"""
        self.messages = [self.system_message, self._session_summary_message()]
        self.trace_turn = 0
        self._previous_prompt_token_ids = []
        self.last_prompt_diagnostics = {}

    def _session_summary_message(self) -> dict[str, str]:
        """生成当前 EEG 会话摘要消息。"""
        return {
            "role": "system",
            "content": self._render_session_summary(),
        }

    def _render_session_summary(self, summary: dict[str, Any] | None = None) -> str:
        """把 EEG 会话摘要渲染为模型可读文本。"""
        summary = summary or self.session_summary
        lines = ["当前 EEG 会话摘要，用于保留被裁剪聊天历史中的关键事实。"]
        recording = summary["recording"]
        if recording:
            lines.append(
                "- 记录："
                f"{recording.get('name', 'unknown')}，"
                f"时长 {recording.get('duration_seconds', 'unknown')} 秒，"
                f"采样率 {recording.get('sampling_rate_hz', 'unknown')} Hz。"
            )
        patient = summary["patient"]
        if patient:
            lines.append(f"- 患者信息：{json.dumps(patient, ensure_ascii=False)}")
        analyses = summary["analyses"]
        if analyses:
            lines.append("- 已完成分析：" + "；".join(analyses[-MAX_MEMORY_ITEMS:]))
        findings = summary["findings"]
        if findings:
            lines.append("- 关键发现：" + "；".join(findings[-MAX_MEMORY_ITEMS:]))
        reports = summary["reports"]
        if reports:
            lines.append("- 已生成报告：" + "；".join(reports[-MAX_MEMORY_ITEMS:]))
        conversation = summary.get("conversation", [])
        if conversation:
            lines.append("- 已压缩对话：" + "；".join(conversation[-MAX_MEMORY_ITEMS:]))
        if len(lines) == 1:
            lines.append("- 当前未绑定 EEG 记录，不提供 EEG 分析工具。")
        return "\n".join(lines)

    def _refresh_session_summary_message(self) -> None:
        """刷新对话中的 EEG 会话摘要。"""
        if len(self.messages) < 2:
            self.messages = [self.system_message, self._session_summary_message()] + self.messages[1:]
        else:
            self.messages[1] = self._session_summary_message()

    def _prepare_request_messages(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compress old turns only after the prompt crosses the configured threshold."""
        request_messages = list(self.messages)
        prompt_tokens = self.token_counter.count_prompt(request_messages, tools)
        trigger_tokens = int(
            self.short_term_token_limit
            * getattr(self, "compression_trigger_ratio", DEFAULT_COMPRESSION_TRIGGER_RATIO)
        )
        if prompt_tokens <= trigger_tokens:
            self.last_prompt_token_count = prompt_tokens
            return request_messages
        return self._compact_old_history(tools, prompt_tokens)

    @staticmethod
    def _scoped_user_content(content: str, skill: SkillSpec | None, has_session: bool) -> str:
        """Keep dynamic Skill instructions at the tail inside the current user message."""
        if not has_session:
            return content
        instruction = (
            skill.as_instruction_block() if skill is not None else NO_SKILL_INSTRUCTION
        )
        return (
            f"{instruction}\n"
            f"{USER_REQUEST_OPEN}\n"
            f"{content}\n"
            f"{USER_REQUEST_CLOSE}"
        )

    @staticmethod
    def _user_request_text(content: str) -> str:
        """Extract the real question from a scoped user message for routing and memory."""
        if USER_REQUEST_OPEN not in content:
            return content
        request = content.split(USER_REQUEST_OPEN, 1)[1]
        if USER_REQUEST_CLOSE in request:
            request = request.split(USER_REQUEST_CLOSE, 1)[0]
        if "<temporary_retrieved_eeg_knowledge>" in request:
            request = request.split("<temporary_retrieved_eeg_knowledge>", 1)[0]
        return request.strip()

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        """Return bounded plain text without retaining raw tool payloads."""
        content = message.get("content")
        if not isinstance(content, str):
            return ""
        if message.get("role") == "user":
            content = MCPChatAgent._user_request_text(content)
        return " ".join(content.split())[:MAX_MEMORY_TEXT_CHARS]

    def _summarize_turns(self, turns: list[list[dict[str, Any]]]) -> list[str]:
        """Create bounded memory notes for complete turns selected for removal."""
        notes: list[str] = []
        for turn in turns:
            user_text = next(
                (self._message_text(message) for message in turn if message.get("role") == "user"),
                "",
            )
            assistant_texts = [
                self._message_text(message)
                for message in turn
                if message.get("role") == "assistant" and not message.get("tool_calls")
            ]
            assistant_text = next((text for text in reversed(assistant_texts) if text), "")
            if user_text and assistant_text:
                notes.append(f"用户请求：{user_text}；助手结论：{assistant_text}")
            elif user_text:
                notes.append(f"用户请求：{user_text}")
        return notes

    def _summary_with_compacted_turns(
        self,
        turns: list[list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Build a prospective summary without changing the live prompt prefix."""
        summary = copy.deepcopy(self.session_summary)
        conversation = list(summary.get("conversation", []))
        conversation.extend(self._summarize_turns(turns))
        summary["conversation"] = _limit_items(conversation)
        return summary

    def _compact_old_history(
        self,
        tools: list[dict[str, Any]],
        original_token_count: int,
    ) -> list[dict[str, Any]]:
        """Replace the oldest complete turns with one low-frequency memory checkpoint."""
        turns = split_history_turns(self.messages[2:])
        if len(turns) <= 1:
            if original_token_count > self.short_term_token_limit:
                raise ValueError(
                    "The fixed prompt and latest scoped user turn use "
                    f"{original_token_count} tokens, which exceeds "
                    f"the {self.short_term_token_limit}-token short-term memory limit."
                )
            self.last_prompt_token_count = original_token_count
            return list(self.messages)

        target_tokens = int(
            self.short_term_token_limit
            * getattr(self, "compression_target_ratio", DEFAULT_COMPRESSION_TARGET_RATIO)
        )
        fallback: tuple[dict[str, Any], list[dict[str, Any]], int] | None = None
        for remove_count in range(1, len(turns)):
            summary = self._summary_with_compacted_turns(turns[:remove_count])
            summary_message = {
                "role": "system",
                "content": self._render_session_summary(summary),
            }
            remaining = [message for turn in turns[remove_count:] for message in turn]
            candidate = [self.system_message, summary_message, *remaining]
            token_count = self.token_counter.count_prompt(candidate, tools)
            if token_count <= self.short_term_token_limit:
                fallback = (summary, remaining, token_count)
            if token_count <= target_tokens:
                break
        else:
            if fallback is None:
                raise ValueError(
                    "The fixed prompt, memory summary, and latest conversation turn "
                    f"exceed the {self.short_term_token_limit}-token short-term memory limit."
                )
            summary, remaining, token_count = fallback

        self.session_summary = summary
        self.messages = [
            self.system_message,
            {"role": "system", "content": self._render_session_summary()},
            *remaining,
        ]
        self.last_prompt_token_count = token_count
        return list(self.messages)

    def _remember_tool_result(self, tool_name: str, result: dict[str, Any]) -> None:
        """记录工具返回的摘要，供后续轮次复用。"""
        if result.get("is_error"):
            return
        data = _as_structured_result(result)
        if not isinstance(data, dict):
            return

        if tool_name == "get_eeg_basic_information":
            self.session_summary["recording"] = data.get("recording") or self.session_summary["recording"]
            self.session_summary["patient"] = data.get("patient") or self.session_summary["patient"]
            montage = data.get("montage", {})
            channel_count = montage.get("raw_channel_count")
            bipolar_count = len(montage.get("available_bipolar_channels", []))
            if channel_count is not None:
                self.session_summary["analyses"].append(
                    f"已读取基础信息：原始通道 {channel_count} 个，可用双极导联 {bipolar_count} 个"
                )
        elif tool_name == "explore_eeg_segment":
            window = data.get("window", {})
            start = window.get("start_seconds")
            end = window.get("end_seconds")
            focus = data.get("focus", "overview")
            self.session_summary["analyses"].append(f"已探索 {start}-{end} 秒，关注 {focus}")
            findings = data.get("abnormality_screen", {}).get("findings", [])
            if findings:
                self.session_summary["findings"].extend(findings)
        elif tool_name == "detect_eeg_events":
            window = data.get("analysis_window", {})
            start = window.get("start_seconds")
            end = window.get("end_seconds")
            event_count = data.get("event_count", 0)
            self.session_summary["analyses"].append(f"已检测 {start}-{end} 秒，发现 {event_count} 个筛查事件")
            for event in data.get("events", [])[:MAX_MEMORY_ITEMS]:
                self.session_summary["findings"].append(
                    f"{event.get('start_seconds')}-{event.get('end_seconds')} 秒 "
                    f"{event.get('channel')} {event.get('brain_region')} "
                    f"{event.get('event_type')}，置信度 {event.get('confidence')}"
                )
        elif tool_name == "generate_eeg_report":
            report_id = data.get("report_id")
            if report_id:
                self.session_summary["reports"].append(report_id)
            impression = data.get("impression")
            if impression:
                self.session_summary["findings"].append(f"报告印象：{impression}")

        self.session_summary["analyses"] = _limit_items(self.session_summary["analyses"])
        self.session_summary["findings"] = _limit_items(self.session_summary["findings"])
        self.session_summary["reports"] = _limit_items(self.session_summary["reports"])

    def _tool_schemas(self, allowed_tools: frozenset[str]) -> list[dict[str, Any]]:
        """Build stable OpenAI schemas for the requested MCP tool names."""
        if self.session_id is None:
            return []
        schemas = []
        for tool in sorted(self.bridge.list_tools(), key=lambda item: item.name):
            if tool.name not in allowed_tools:
                continue
            input_schema = copy.deepcopy(getattr(tool, "inputSchema", getattr(tool, "input_schema", {})))
            properties = input_schema.get("properties", {})
            properties.pop("session_id", None)
            required = [item for item in input_schema.get("required", []) if item != "session_id"]
            input_schema["required"] = required
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": input_schema,
                },
            })
        return schemas

    def _model_tool_schemas(self, skill: SkillSpec | None) -> list[dict[str, Any]]:
        """Expose one fixed union of runtime Skill tools throughout an EEG session."""
        if self.session_id is None:
            return []
        registry = getattr(self, "skill_registry", None)
        if registry is None:
            tool_names = skill.allowed_tools if skill is not None else frozenset()
        else:
            tool_names = frozenset(
                tool_name
                for registered_skill in registry.all()
                for tool_name in registered_skill.allowed_tools
            )
        return self._tool_schemas(tool_names)

    def _stream_completion(
        self,
        tools: list[dict[str, Any]],
        on_delta=None,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """流式调用大模型并收集文本与工具调用。"""
        self._check_cancelled()
        request_messages = self._prepare_request_messages(tools)
        prompt_token_ids = self.token_counter.prompt_token_ids(request_messages, tools)
        previous_token_ids = list(getattr(self, "_previous_prompt_token_ids", []) or [])
        common_prefix_tokens = 0
        for previous_token, current_token in zip(previous_token_ids, prompt_token_ids):
            if previous_token != current_token:
                break
            common_prefix_tokens += 1
        fingerprint_source = ",".join(str(token_id) for token_id in prompt_token_ids)
        self.last_prompt_diagnostics = {
            "fingerprint": hashlib.sha256(fingerprint_source.encode("ascii")).hexdigest()[:12],
            "local_tokens": len(prompt_token_ids),
            "previous_tokens": len(previous_token_ids),
            "common_prefix_tokens": common_prefix_tokens,
        }
        self._previous_prompt_token_ids = prompt_token_ids
        request = {
            "model": self.model,
            "messages": request_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "timeout": 90,
            "extra_body": {
                "thinking": {
                    "type": (
                        "enabled"
                        if self.token_counter.thinking_mode == "thinking"
                        else "disabled"
                    )
                }
            },
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        stream = self.client.chat.completions.create(**request)
        with self._stream_lock:
            self._active_stream = stream
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        self.last_completion_usage = {}
        try:
            for chunk in stream:
                self._check_cancelled()
                usage = _completion_usage_dict(getattr(chunk, "usage", None))
                if usage:
                    self.last_completion_usage = usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning_delta = getattr(delta, "reasoning_content", None)
                if reasoning_delta is None:
                    model_extra = getattr(delta, "model_extra", None) or {}
                    reasoning_delta = model_extra.get("reasoning_content")
                if isinstance(reasoning_delta, str) and reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                if delta.content:
                    text_parts.append(delta.content)
                    if on_delta:
                        on_delta(delta.content)
                for tool_call in delta.tool_calls or []:
                    entry = calls.setdefault(tool_call.index, {"id": None, "name": "", "arguments": ""})
                    if tool_call.id:
                        entry["id"] = tool_call.id
                    if tool_call.function and tool_call.function.name:
                        entry["name"] += tool_call.function.name
                    if tool_call.function and tool_call.function.arguments:
                        entry["arguments"] += tool_call.function.arguments
            self._check_cancelled()
        except GenerationCancelled:
            raise
        except Exception as exc:
            if self._cancel_event.is_set():
                raise GenerationCancelled() from exc
            raise
        finally:
            with self._stream_lock:
                if self._active_stream is stream:
                    self._active_stream = None
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        return (
            "".join(text_parts),
            "".join(reasoning_parts),
            [calls[index] for index in sorted(calls)],
        )

    def _retrieve_eeg_knowledge(self, user_query: str) -> tuple[str, list[dict[str, Any]]]:
        """Build current-turn-only RAG context without changing the system prompt."""
        from RAG.retriever import EEGRetriever, format_temporary_context
        from RAG.retrieval_policy import decide_retrieval

        previous_user_query = next(
            (
                self._user_request_text(str(message.get("content", "")))
                for message in reversed(self.messages)
                if message.get("role") == "user" and message.get("content")
            ),
            None,
        )
        decision = decide_retrieval(
            user_query,
            has_eeg_session=self.session_id is not None,
            previous_user_query=previous_user_query,
        )
        if decision.mode == "skip":
            return user_query, []

        if self.rag_retriever is None:
            self.set_rag_retriever(EEGRetriever())
        results = self.rag_retriever.retrieve(
            decision.retrieval_query,
            require_faiss_probe=decision.mode == "probe",
        )
        return format_temporary_context(user_query, results), results

    def run_stream(
        self,
        user_query: str,
        on_delta=None,
        on_tool_start=None,
        on_tool_end=None,
        on_tool_call_detected=None,
        on_trace=None,
    ) -> dict[str, Any]:
        """执行一轮流式React对话，必要时循环调用本地 EEG 工具。"""
        self._ensure_cancel_state()
        self._cancel_event.clear()
        message_checkpoint = len(self.messages)
        partial_response_parts: list[str] = []
        selection: SkillSelection | None = None
        skill: SkillSpec | None = None

        def forward_delta(delta: str) -> None:
            partial_response_parts.append(delta)
            if on_delta:
                on_delta(delta)

        def forward_tool_call_detected() -> None:
            partial_response_parts.clear()
            if on_tool_call_detected:
                on_tool_call_detected()

        self.trace_turn = getattr(self, "trace_turn", 0) + 1
        turn = self.trace_turn
        turn_started_at = time.time()
        _emit_trace(on_trace, {
            "id": f"turn-{turn}-user",
            "type": "user/message",
            "kind": "user",
            "lane": "input",
            "turn": turn,
            "status": "complete",
            "title": "用户请求",
            "summary": user_query,
            "input": user_query,
            "started_at": turn_started_at,
            "ended_at": turn_started_at,
            "duration": 0.0,
        })
        try:
            self._check_cancelled()
            route_started_at = time.time()
            if self.session_id is not None:
                def semantic_route(query: str, skills: tuple[SkillSpec, ...]):
                    if self.semantic_skill_selector is None:
                        if self.rag_retriever is None:
                            from RAG.retriever import EEGRetriever

                            self.set_rag_retriever(EEGRetriever())
                        else:
                            self.set_rag_retriever(self.rag_retriever)
                    return self.semantic_skill_selector.select(query, skills)

                selection = self.skill_registry.select_with_details(
                    user_query,
                    semantic_selector=semantic_route,
                )
                skill = selection.skill
            route_ended_at = time.time()
            self._check_cancelled()
            route_source = selection.source if selection is not None else "no_eeg_session"
            route_detail = {
                "skill": skill.name if skill is not None else None,
                "source": route_source,
                "keyword_matches": list(selection.keyword_matches) if selection else [],
                "top_score": selection.top_score if selection else 0.0,
                "margin": selection.margin if selection else 0.0,
                "candidates": [
                    {"name": candidate.name, "score": candidate.score}
                    for candidate in (selection.candidates[:2] if selection else ())
                ],
            }
            _emit_trace(on_trace, {
                "id": f"turn-{turn}-route",
                "type": "routing/decision",
                "kind": "context",
                "lane": "input",
                "turn": turn,
                "status": "complete",
                "title": "Skill 路由",
                "summary": skill.name if skill is not None else "普通 RAG 问答",
                "output": _trace_text(route_detail),
                "started_at": route_started_at,
                "ended_at": route_ended_at,
                "duration": route_ended_at - route_started_at,
            })

            retrieval_started_at = time.time()
            temporary_content, retrieval_results = self._retrieve_eeg_knowledge(user_query)
            retrieval_ended_at = time.time()
            self._check_cancelled()
            sources = [item.get("source", "unknown") for item in retrieval_results]
            _emit_trace(on_trace, {
                "id": f"turn-{turn}-rag",
                "type": "rag/retrieval",
                "kind": "context",
                "lane": "input",
                "turn": turn,
                "status": "complete",
                "title": "知识检索",
                "summary": f"检索到 {len(retrieval_results)} 条参考片段" if retrieval_results else "本轮未注入参考片段",
                "input": user_query,
                "output": _trace_text({
                    "sources": sources,
                    "passages": retrieval_results,
                }),
                "started_at": retrieval_started_at,
                "ended_at": retrieval_ended_at,
                "duration": retrieval_ended_at - retrieval_started_at,
            })

            user_message = {
                "role": "user",
                "content": self._scoped_user_content(
                    temporary_content,
                    skill,
                    has_session=self.session_id is not None,
                ),
            }
            self.messages.append(user_message)
            try:
                return self._run_stream_with_temporary_context(
                    user_query,
                    retrieval_results,
                    skill,
                    selection,
                    on_delta=forward_delta,
                    on_tool_start=on_tool_start,
                    on_tool_end=on_tool_end,
                    on_tool_call_detected=forward_tool_call_detected,
                    on_trace=on_trace,
                    trace_turn=turn,
                    trace_started_at=turn_started_at,
                )
            finally:
                # Retrieved passages are intentionally limited to this request.
                user_message["content"] = self._scoped_user_content(
                    user_query,
                    skill,
                    has_session=self.session_id is not None,
                )
        except GenerationCancelled:
            partial_response = "".join(partial_response_parts)
            del self.messages[message_checkpoint:]
            self.messages.append({
                "role": "user",
                "content": self._scoped_user_content(
                    user_query,
                    skill,
                    has_session=self.session_id is not None,
                ),
            })
            if partial_response:
                self.messages.append({"role": "assistant", "content": partial_response})
            cancelled_at = time.time()
            _emit_trace(on_trace, {
                "id": f"turn-{turn}-end",
                "type": "turn/end",
                "kind": "context",
                "lane": "input",
                "turn": turn,
                "status": "cancelled",
                "title": "用户已取消",
                "summary": "本轮生成已中止",
                "output": partial_response,
                "started_at": turn_started_at,
                "ended_at": cancelled_at,
                "duration": cancelled_at - turn_started_at,
            })
            raise GenerationCancelled(partial_response)
        except Exception as exc:
            failed_at = time.time()
            _emit_trace(on_trace, {
                "id": f"turn-{turn}-end",
                "type": "turn/end",
                "kind": "context",
                "lane": "input",
                "turn": turn,
                "status": "error",
                "title": "分析失败",
                "summary": str(exc) or exc.__class__.__name__,
                "output": _trace_text({"error_type": exc.__class__.__name__, "message": str(exc)}),
                "started_at": turn_started_at,
                "ended_at": failed_at,
                "duration": failed_at - turn_started_at,
            })
            raise

    def _run_stream_with_temporary_context(
        self,
        user_query: str,
        retrieval_results: list[dict[str, Any]],
        skill: SkillSpec | None,
        selection: SkillSelection | None = None,
        on_delta=None,
        on_tool_start=None,
        on_tool_end=None,
        on_tool_call_detected=None,
        on_trace=None,
        trace_turn: int | None = None,
        trace_started_at: float | None = None,
    ) -> dict[str, Any]:
        tools = self._model_tool_schemas(skill)
        started = time.time()
        turn_number = trace_turn if trace_turn is not None else getattr(self, "trace_turn", 1)
        turn_wall_started = trace_started_at if trace_started_at is not None else started
        model_name = getattr(self, "model", "deepseek")
        model_time = 0.0
        tool_time = 0.0
        rounds = 0
        tool_rounds = 0
        turn_cache_hit_tokens = 0
        turn_cache_miss_tokens = 0
        has_turn_cache_usage = False

        while True:
            self._check_cancelled()
            rounds += 1
            force_final_answer = tool_rounds >= MAX_TOOL_ROUNDS
            if force_final_answer:
                self.messages.append({
                    "role": "system",
                    "content": (
                        f"已达到 {MAX_TOOL_ROUNDS} 轮工具调用上限。不要再调用工具，"
                        "请仅根据现有对话和工具结果直接给出最终答案。"
                    ),
                })
            model_started = time.time()
            model_event_id = f"turn-{turn_number}-model-{rounds}"
            available_tools = [] if force_final_answer else tools
            _emit_trace(on_trace, {
                "id": model_event_id,
                "type": "model/request",
                "kind": "assistant",
                "lane": "model",
                "turn": turn_number,
                "step": rounds,
                "status": "running",
                "title": f"模型调用 {rounds}",
                "summary": model_name,
                "input": _trace_text({
                    "model": model_name,
                    "message_count": len(self.messages),
                    "tools": [schema["function"]["name"] for schema in available_tools],
                    "force_final_answer": force_final_answer,
                }),
                "started_at": model_started,
            })
            first_token_at = None

            def traced_delta(delta: str):
                nonlocal first_token_at
                if first_token_at is None:
                    first_token_at = time.time()
                    _emit_trace(on_trace, {
                        "id": model_event_id,
                        "first_token_at": first_token_at,
                        "ttft": first_token_at - model_started,
                    })
                if on_delta:
                    on_delta(delta)

            model_error = None
            model_cancelled = False
            self.last_completion_usage = {}
            try:
                response, reasoning_content, calls = self._stream_completion(
                    available_tools,
                    on_delta=traced_delta,
                )
            except GenerationCancelled:
                model_cancelled = True
                raise
            except Exception as exc:
                model_error = exc
                raise
            finally:
                model_ended = time.time()
                model_elapsed = model_ended - model_started
                model_time += model_elapsed
                if force_final_answer:
                    self.messages.pop()
                if model_cancelled:
                    _emit_trace(on_trace, {
                        "id": model_event_id,
                        "status": "cancelled",
                        "summary": "用户取消了模型调用",
                        "ended_at": model_ended,
                        "duration": model_elapsed,
                    })
                elif model_error is not None:
                    _emit_trace(on_trace, {
                        "id": model_event_id,
                        "status": "error",
                        "summary": str(model_error) or model_error.__class__.__name__,
                        "output": _trace_text({
                            "error_type": model_error.__class__.__name__,
                            "message": str(model_error),
                        }),
                        "ended_at": model_ended,
                        "duration": model_elapsed,
                    })
            usage = dict(getattr(self, "last_completion_usage", {}) or {})
            cache_hit_tokens = usage.get("prompt_cache_hit_tokens")
            cache_miss_tokens = usage.get("prompt_cache_miss_tokens")
            if isinstance(cache_hit_tokens, (int, float)):
                turn_cache_hit_tokens += int(cache_hit_tokens)
                has_turn_cache_usage = True
            if isinstance(cache_miss_tokens, (int, float)):
                turn_cache_miss_tokens += int(cache_miss_tokens)
                has_turn_cache_usage = True
            prompt_tokens = usage.get(
                "prompt_tokens", getattr(self, "last_prompt_token_count", None)
            )
            completion_tokens = usage.get("completion_tokens")
            context_limit_tokens = int(getattr(
                self, "short_term_token_limit", DEFAULT_SHORT_TERM_TOKEN_LIMIT
            ))
            prompt_token_count = int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else 0
            context_ratio = (
                prompt_token_count / context_limit_tokens
                if context_limit_tokens > 0 else 0.0
            )
            cache_hit_text = (
                str(int(cache_hit_tokens))
                if isinstance(cache_hit_tokens, (int, float)) else "n/a"
            )
            cache_miss_text = (
                str(int(cache_miss_tokens))
                if isinstance(cache_miss_tokens, (int, float)) else "n/a"
            )
            print(
                f"[Agent model tokens] turn={turn_number} step={rounds} "
                f"input={prompt_token_count}tok "
                f"cache_hit={cache_hit_text}tok cache_miss={cache_miss_text}tok "
                f"context={prompt_token_count}/{context_limit_tokens}tok "
                f"usage={context_ratio:.1%}",
                flush=True,
            )
            prompt_diagnostics = dict(
                getattr(self, "last_prompt_diagnostics", {}) or {}
            )
            if prompt_diagnostics:
                print(
                    f"[Agent prompt prefix] turn={turn_number} step={rounds} "
                    f"fingerprint={prompt_diagnostics.get('fingerprint', 'n/a')} "
                    f"local={prompt_diagnostics.get('local_tokens', 0)}tok "
                    f"previous={prompt_diagnostics.get('previous_tokens', 0)}tok "
                    f"common_prefix={prompt_diagnostics.get('common_prefix_tokens', 0)}tok",
                    flush=True,
                )
            _emit_trace(on_trace, {
                "id": model_event_id,
                "status": "complete",
                "summary": f"返回 {len(calls)} 个工具调用" if calls else "生成最终回答",
                "output": _trace_text({
                    "text": response,
                    "tool_calls": calls,
                }),
                "usage": usage,
                "prompt_tokens": prompt_tokens,
                "context_limit_tokens": context_limit_tokens,
                "completion_tokens": completion_tokens,
                "cache_hit_tokens": cache_hit_tokens,
                "cache_miss_tokens": cache_miss_tokens,
                "ended_at": model_ended,
                "duration": model_elapsed,
                "first_token_at": first_token_at,
                "ttft": None if first_token_at is None else first_token_at - model_started,
            })
            if force_final_answer:
                calls = []
            if not calls:
                assistant_response_message = {
                    "role": "assistant",
                    "content": response,
                }
                if reasoning_content:
                    assistant_response_message["reasoning_content"] = reasoning_content
                self.messages.append(assistant_response_message)
                turn_cache_total_tokens = (
                    turn_cache_hit_tokens + turn_cache_miss_tokens
                )
                turn_cache_hit_rate = (
                    turn_cache_hit_tokens / turn_cache_total_tokens
                    if has_turn_cache_usage and turn_cache_total_tokens > 0
                    else None
                )
                routing = {
                    "enabled": selection is not None,
                    "skill_selected": skill is not None,
                    "skill": skill.name if skill is not None else None,
                    "source": selection.source if selection is not None else "no_eeg_session",
                    "keyword_matches": list(selection.keyword_matches) if selection else [],
                    "top_score": selection.top_score if selection else 0.0,
                    "margin": selection.margin if selection else 0.0,
                    "candidates": [
                        {
                            "name": candidate.name,
                            "score": candidate.score,
                            "matched_examples": list(candidate.matched_examples),
                        }
                        for candidate in (selection.candidates[:2] if selection else ())
                    ],
                    "allowed_tools": sorted(skill.allowed_tools) if skill is not None else [],
                    "model_tool_schemas": [schema["function"]["name"] for schema in tools],
                }
                result = {
                    "response": response,
                    "rounds": rounds,
                    "model_time": model_time,
                    "local_tool_time": tool_time,
                    "total_time": time.time() - started,
                    "route": skill.name if skill is not None else "direct_rag",
                    "skill": skill.name if skill is not None else None,
                    "routing": routing,
                    "retrieved_sources": [item["source"] for item in retrieval_results],
                    "context_tokens": int(prompt_tokens or 0),
                    "context_limit_tokens": context_limit_tokens,
                    "cache_hit_tokens": turn_cache_hit_tokens,
                    "cache_miss_tokens": turn_cache_miss_tokens,
                    "cache_hit_rate": turn_cache_hit_rate,
                }
                turn_ended_at = time.time()
                context_ratio = (
                    result["context_tokens"] / context_limit_tokens
                    if context_limit_tokens > 0 else 0.0
                )
                print(
                    f"[Agent context] turn={turn_number} "
                    f"tokens={result['context_tokens']}/{context_limit_tokens} "
                    f"usage={context_ratio:.1%}",
                    flush=True,
                )
                cache_rate_text = (
                    f"{turn_cache_hit_rate:.1%}"
                    if turn_cache_hit_rate is not None else "n/a"
                )
                print(
                    f"[Agent cache] turn={turn_number} "
                    f"hit={turn_cache_hit_tokens} miss={turn_cache_miss_tokens} "
                    f"total={turn_cache_total_tokens} rate={cache_rate_text} "
                    "formula=hit/(hit+miss)",
                    flush=True,
                )
                _emit_trace(on_trace, {
                    "id": f"turn-{turn_number}-end",
                    "type": "turn/end",
                    "kind": "context",
                    "lane": "input",
                    "turn": turn_number,
                    "status": "complete",
                    "title": "执行结果",
                    "summary": f"{rounds} 次模型调用，{tool_rounds} 轮工具调用",
                    "context_tokens": result["context_tokens"],
                    "context_limit_tokens": context_limit_tokens,
                    "cache_hit_tokens": turn_cache_hit_tokens,
                    "cache_miss_tokens": turn_cache_miss_tokens,
                    "cache_hit_rate": turn_cache_hit_rate,
                    "started_at": turn_wall_started,
                    "ended_at": turn_ended_at,
                    "duration": turn_ended_at - turn_wall_started,
                })
                return result

            tool_rounds += 1
            if on_tool_call_detected:
                on_tool_call_detected()
            assistant_message = {
                "role": "assistant",
                "content": response or None,
                "tool_calls": [
                    {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
                    for call in calls
                ],
            }
            if reasoning_content:
                assistant_message["reasoning_content"] = reasoning_content
            self.messages.append(assistant_message)
            for call_index, call in enumerate(calls, start=1):
                self._check_cancelled()
                tool_name = call["name"] or "unknown_tool"
                call_id = str(call["id"] or f"round-{rounds}-call-{call_index}")
                tool_started = time.time()
                tool_event_id = f"turn-{turn_number}-tool-{call_id}"
                _emit_trace(on_trace, {
                    "id": tool_event_id,
                    "type": "tool/call",
                    "kind": "tool",
                    "lane": "tool",
                    "turn": turn_number,
                    "step": rounds,
                    "call_id": call_id,
                    "status": "running",
                    "title": tool_name,
                    "summary": f"正在调用 {tool_name}",
                    "input": _trace_text(call.get("arguments") or "{}"),
                    "started_at": tool_started,
                })
                if self.session_id is None:
                    failed_at = time.time()
                    _emit_trace(on_trace, {
                        "id": tool_event_id,
                        "status": "error",
                        "summary": "缺少 EEG 会话",
                        "ended_at": failed_at,
                        "duration": failed_at - tool_started,
                    })
                    raise RuntimeError("An EEG session is required before calling EEG tools.")
                if skill is None or tool_name not in skill.allowed_tools:
                    result = _tool_exception_result(
                        tool_name,
                        PermissionError(
                            f"Tool '{tool_name}' is not allowed by the active runtime Skill."
                        ),
                    )
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result_for_model(result, tool_name=tool_name),
                    })
                    failed_at = time.time()
                    _emit_trace(on_trace, {
                        "id": tool_event_id,
                        "status": "error",
                        "summary": f"{tool_name} 未获当前 Skill 授权",
                        "output": _trace_text(result),
                        "ended_at": failed_at,
                        "duration": failed_at - tool_started,
                    })
                    continue
                try:
                    arguments = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                arguments["session_id"] = self.session_id
                if on_tool_start:
                    on_tool_start(tool_name)
                try:
                    result = self.bridge.call_tool(tool_name, arguments)
                except Exception as exc:
                    result = _tool_exception_result(tool_name, exc)
                finally:
                    if on_tool_end:
                        on_tool_end(tool_name)
                self._check_cancelled()
                tool_ended = time.time()
                tool_elapsed = tool_ended - tool_started
                tool_time += tool_elapsed
                self._remember_tool_result(tool_name, result)
                model_context_content = result_for_model(result, tool_name=tool_name)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": model_context_content,
                })
                is_error = bool(result.get("is_error"))
                _emit_trace(on_trace, {
                    "id": tool_event_id,
                    "status": "error" if is_error else "complete",
                    "summary": f"{tool_name} 调用失败" if is_error else f"{tool_name} 调用完成",
                    "input": _trace_text(arguments),
                    "output": (
                        model_context_content
                        if tool_name == "detect_eeg_events"
                        else _trace_text(result)
                    ),
                    "ended_at": tool_ended,
                    "duration": tool_elapsed,
                })
