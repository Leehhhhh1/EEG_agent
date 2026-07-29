"""DeepSeek native Function Calling agent backed by session-scoped MCP EEG skills."""

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


def _select_route(question: str) -> str:
    text = question.lower()
    if any(word in text for word in ("report", "报告", "总结", "summary")):
        return "reporting"
    if any(word in text for word in ("seizure", "epilep", "discharge", "发作", "癫痫", "放电", "定位", "where")):
        return "detection"
    if any(word in text for word in ("rhythm", "amplitude", "symmetry", "background", "频", "振幅", "节律", "对称", "背景")):
        return "exploration"
    return "general_eeg"


class MCPChatAgent:
    def __init__(self, bridge: MCPClientBridge, session_id: str):
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("api_key")
        if not api_key:
            raise ValueError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY in .env.")
        self.client = OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.bridge = bridge
        self.session_id = session_id
        self.messages = [{
            "role": "system",
            "content": (
                "你是 EEGAgent 脑电图辅助分析助手。使用中文回答，且仅根据 MCP 工具的结果描述 EEG 发现。"
                "自动筛查不是临床诊断；涉及异常或发作时必须说明需由合格脑电图专业人员复核。"
                "不要编造工具未提供的时间、导联、脑区或置信度。"
            ),
        }]

    def reset(self) -> None:
        self.messages = self.messages[:1]

    def _tool_schemas(self, route: str) -> list[dict[str, Any]]:
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
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result_for_model(result),
                })
