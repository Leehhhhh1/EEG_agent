"""基于会话级 MCP EEG 工具的 DeepSeek 原生函数调用 agent。"""

import copy
import json
import os
import time
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI
from .mcp_client import MCPClientBridge, result_for_model
from .skills import SemanticSkillSelector, SkillRegistry, SkillSelection, SkillSpec
from .token_budget import (
    DEFAULT_SHORT_TERM_TOKEN_LIMIT,
    DeepSeekV4TokenCounter,
    trim_messages_to_token_limit,
)


MAX_MEMORY_ITEMS = 12
MAX_TOOL_ROUNDS = 8
NO_SKILL_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "当前请求未匹配可安全执行的 EEG 分析任务。不要调用 EEG 工具，也不要对当前记录作出推断。"
        "若用户可能在询问当前记录，请要求其明确分析目标、时间范围，以及必要时的导联。"
        "若是一般 EEG 知识问题，可根据当前轮检索到的参考资料回答。"
    ),
}


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
        self.token_counter = DeepSeekV4TokenCounter(
            thinking_mode=os.getenv("DEEPSEEK_THINKING_MODE", "thinking")
        )
        self.last_prompt_token_count = 0
        self.rag_retriever = None
        self.semantic_skill_selector = None
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
            ),
        }
        self.session_summary = {
            "recording": None,
            "patient": {},
            "analyses": [],
            "findings": [],
            "reports": [],
        }
        self.messages = [self.system_message, self._session_summary_message()]
        if initial_basic_info:
            self._remember_tool_result(
                "get_eeg_basic_information",
                {"is_error": False, "structured_content": initial_basic_info, "content": []},
            )

    def set_rag_retriever(self, retriever) -> None:
        """Attach the shared RAG runtime and reuse its BGE-M3 embedder for Skill routing."""
        if self.rag_retriever is retriever and self.semantic_skill_selector is not None:
            return
        self.rag_retriever = retriever
        self.semantic_skill_selector = SemanticSkillSelector(retriever.embedder)

    def attach_session(self, session_id: str, basic_info: dict[str, Any] | None = None) -> None:
        """Attach an EEG recording without discarding conversation history."""
        self._discard_tool_messages()
        self.session_id = session_id
        self.session_summary = {
            "recording": None,
            "patient": {},
            "analyses": [],
            "findings": [],
            "reports": [],
        }
        if basic_info:
            self._remember_tool_result(
                "get_eeg_basic_information",
                {"is_error": False, "structured_content": basic_info, "content": []},
            )
        else:
            self._refresh_session_summary_message()

    def detach_session(self) -> None:
        """Remove EEG-specific state while preserving conversation history."""
        self._discard_tool_messages()
        self.session_id = None
        self.session_summary = {
            "recording": None,
            "patient": {},
            "analyses": [],
            "findings": [],
            "reports": [],
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

    def _session_summary_message(self) -> dict[str, str]:
        """生成当前 EEG 会话摘要消息。"""
        return {
            "role": "system",
            "content": self._render_session_summary(),
        }

    def _render_session_summary(self) -> str:
        """把 EEG 会话摘要渲染为模型可读文本。"""
        lines = ["当前 EEG 会话摘要，用于保留被裁剪聊天历史中的关键事实。"]
        recording = self.session_summary["recording"]
        if recording:
            lines.append(
                "- 记录："
                f"{recording.get('name', 'unknown')}，"
                f"时长 {recording.get('duration_seconds', 'unknown')} 秒，"
                f"采样率 {recording.get('sampling_rate_hz', 'unknown')} Hz。"
            )
        patient = self.session_summary["patient"]
        if patient:
            lines.append(f"- 患者信息：{json.dumps(patient, ensure_ascii=False)}")
        analyses = self.session_summary["analyses"]
        if analyses:
            lines.append("- 已完成分析：" + "；".join(analyses[-MAX_MEMORY_ITEMS:]))
        findings = self.session_summary["findings"]
        if findings:
            lines.append("- 关键发现：" + "；".join(findings[-MAX_MEMORY_ITEMS:]))
        reports = self.session_summary["reports"]
        if reports:
            lines.append("- 已生成报告：" + "；".join(reports[-MAX_MEMORY_ITEMS:]))
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
        transient_system_messages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Trim history while preserving request-scoped system instructions."""
        self._refresh_session_summary_message()
        transient = list(transient_system_messages or [])
        request_messages = self.messages[:2] + transient + self.messages[2:]
        trimmed, self.last_prompt_token_count = trim_messages_to_token_limit(
            request_messages,
            tools,
            self.token_counter,
            token_limit=self.short_term_token_limit,
            preserved_message_count=2 + len(transient),
        )
        transient_ids = {id(message) for message in transient}
        self.messages = [message for message in trimmed if id(message) not in transient_ids]
        return trimmed

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
        self._refresh_session_summary_message()

    def _tool_schemas(self, allowed_tools: frozenset[str]) -> list[dict[str, Any]]:
        """Expose only the MCP tools allowed by the active runtime skill."""
        if self.session_id is None:
            return []
        schemas = []
        for tool in self.bridge.list_tools():
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

    def _stream_completion(
        self,
        tools: list[dict[str, Any]],
        on_delta=None,
        transient_system_messages: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """流式调用大模型并收集文本与工具调用。"""
        request_messages = self._prepare_request_messages(tools, transient_system_messages)
        request = {
            "model": self.model,
            "messages": request_messages,
            "stream": True,
            "timeout": 90,
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        stream = self.client.chat.completions.create(**request)
        text_parts: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
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
        return "".join(text_parts), [calls[index] for index in sorted(calls)]

    def _retrieve_eeg_knowledge(self, user_query: str) -> tuple[str, list[dict[str, Any]]]:
        """Build current-turn-only RAG context without changing the system prompt."""
        from RAG.retriever import EEGRetriever, format_temporary_context
        from RAG.retrieval_policy import decide_retrieval

        previous_user_query = next(
            (
                str(message.get("content", ""))
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

    def run_stream(self, user_query: str, on_delta=None, on_tool_start=None, on_tool_end=None, on_tool_call_detected=None) -> dict[str, Any]:
        """执行一轮流式React对话，必要时循环调用本地 EEG 工具。"""
        self._refresh_session_summary_message()
        selection: SkillSelection | None = None
        skill: SkillSpec | None = None
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
        temporary_content, retrieval_results = self._retrieve_eeg_knowledge(user_query)
        user_message = {"role": "user", "content": temporary_content}
        self.messages.append(user_message)
        try:
            return self._run_stream_with_temporary_context(
                user_query,
                retrieval_results,
                skill,
                selection,
                on_delta=on_delta,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
                on_tool_call_detected=on_tool_call_detected,
            )
        finally:
            # Retrieved passages are intentionally limited to this request.
            user_message["content"] = user_query

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
    ) -> dict[str, Any]:
        tools = self._tool_schemas(skill.allowed_tools) if skill is not None else []
        transient_messages = (
            [skill.as_system_message()]
            if skill is not None
            else ([NO_SKILL_SYSTEM_MESSAGE] if self.session_id is not None else [])
        )
        started = time.time()
        model_time = 0.0
        tool_time = 0.0
        rounds = 0
        tool_rounds = 0

        while True:
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
            try:
                response, calls = self._stream_completion(
                    [] if force_final_answer else tools,
                    on_delta=on_delta,
                    transient_system_messages=transient_messages,
                )
            finally:
                model_time += time.time() - model_started
                if force_final_answer:
                    self.messages.pop()
            if force_final_answer:
                calls = []
            if not calls:
                self.messages.append({"role": "assistant", "content": response})
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
                    "allowed_tools": [schema["function"]["name"] for schema in tools],
                }
                return {
                    "response": response,
                    "rounds": rounds,
                    "model_time": model_time,
                    "local_tool_time": tool_time,
                    "total_time": time.time() - started,
                    "route": skill.name if skill is not None else "direct_rag",
                    "skill": skill.name if skill is not None else None,
                    "routing": routing,
                    "retrieved_sources": [item["source"] for item in retrieval_results],
                }

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
            self.messages.append(assistant_message)
            for call in calls:
                if self.session_id is None:
                    raise RuntimeError("An EEG session is required before calling EEG tools.")
                if skill is None or call["name"] not in skill.allowed_tools:
                    result = _tool_exception_result(
                        call["name"],
                        PermissionError(
                            f"Tool '{call['name']}' is not allowed by the active runtime Skill."
                        ),
                    )
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result_for_model(result),
                    })
                    continue
                try:
                    arguments = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                arguments["session_id"] = self.session_id
                tool_started = time.time()
                if on_tool_start:
                    on_tool_start(call["name"])
                try:
                    result = self.bridge.call_tool(call["name"], arguments)
                except Exception as exc:
                    result = _tool_exception_result(call["name"], exc)
                finally:
                    if on_tool_end:
                        on_tool_end(call["name"])
                tool_time += time.time() - tool_started
                self._remember_tool_result(call["name"], result)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result_for_model(result),
                })
