"""基于会话级 MCP EEG 工具的 DeepSeek 原生函数调用 agent。"""

import copy
import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .mcp_client import MCPClientBridge, result_for_model


ROUTES = {
    "basic_information": {"get_eeg_basic_information"},
    "exploration": {"explore_eeg_segment", "get_eeg_basic_information"},
    "detection": {"detect_eeg_events", "get_eeg_basic_information"},
    "reporting": {"generate_eeg_report", "get_eeg_basic_information"},
    "general_eeg": {
        "get_eeg_basic_information",
        "explore_eeg_segment",
        "detect_eeg_events",
        "generate_eeg_report",
    },
}


MAX_HISTORY_TURNS = 8
MAX_MEMORY_ITEMS = 12


def _select_route(question: str) -> str:
    """根据用户问题选择可用的 EEG 工具路由。"""
    text = question.lower()
    if any(word in text for word in ("report", "报告", "总结", "summary")):
        return "reporting"
    if any(word in text for word in ("seizure", "epilep", "discharge", "发作", "癫痫", "放电", "定位", "where")):
        return "detection"
    if any(word in text for word in ("rhythm", "amplitude", "symmetry", "background", "频率", "振幅", "节律", "对称", "背景")):
        return "exploration"
    return "general_eeg"


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


def _limit_items(items: list[Any], limit: int = MAX_MEMORY_ITEMS) -> list[Any]:
    """限制列表长度，避免摘要内容过长。"""
    return items[-limit:]


class MCPChatAgent:
    def __init__(self, bridge: MCPClientBridge, session_id: str, initial_basic_info: dict[str, Any] | None = None):
        """初始化对象状态。"""
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("api_key")
        if not api_key:
            raise ValueError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY in .env.")
        self.client = OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.bridge = bridge
        self.session_id = session_id
        self.system_message = {
            "role": "system",
            "content": (
                "你是 EEGAgent 脑电图辅助分析助手。请使用中文回答，并且仅根据 MCP 工具结果描述 EEG 发现。"
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
            lines.append("- 暂无工具结果摘要。")
        return "\n".join(lines)

    def _refresh_session_summary_message(self) -> None:
        """刷新对话中的 EEG 会话摘要。"""
        if len(self.messages) < 2:
            self.messages = [self.system_message, self._session_summary_message()] + self.messages[1:]
        else:
            self.messages[1] = self._session_summary_message()

    def _trim_messages(self) -> None:
        """裁剪对话历史，控制发送给模型的上下文长度。"""
        self._refresh_session_summary_message()
        preserved = self.messages[:2]
        history = self.messages[2:]
        user_indexes = [
            index for index, message in enumerate(history)
            if message.get("role") == "user"
        ]
        if len(user_indexes) <= MAX_HISTORY_TURNS:
            return
        keep_from = user_indexes[-MAX_HISTORY_TURNS]
        self.messages = preserved + history[keep_from:]

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

    def _tool_schemas(self, route: str) -> list[dict[str, Any]]:
        """按当前路由生成可暴露给模型的工具 schema。"""
        allowed = ROUTES[route]
        schemas = []
        for tool in self.bridge.list_tools():
            if tool.name not in allowed:
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

    def _stream_completion(self, tools: list[dict[str, Any]], on_delta=None) -> tuple[str, list[dict[str, Any]]]:
        """流式调用大模型并收集文本与工具调用。"""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
            timeout=90,
        )
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

    def run_stream(self, user_query: str, on_delta=None, on_tool_start=None, on_tool_end=None, on_tool_call_detected=None) -> dict[str, Any]:
        """执行一轮流式对话，必要时循环调用本地 EEG 工具。"""
        self._refresh_session_summary_message()
        self.messages.append({"role": "user", "content": user_query})
        route = _select_route(user_query)
        tools = self._tool_schemas(route)
        started = time.time()
        model_time = 0.0
        tool_time = 0.0
        rounds = 0

        while True:
            rounds += 1
            model_started = time.time()
            response, calls = self._stream_completion(tools, on_delta=on_delta)
            model_time += time.time() - model_started
            if not calls:
                self.messages.append({"role": "assistant", "content": response})
                self._trim_messages()
                return {
                    "response": response,
                    "rounds": rounds,
                    "model_time": model_time,
                    "local_tool_time": tool_time,
                    "total_time": time.time() - started,
                    "route": route,
                }

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
