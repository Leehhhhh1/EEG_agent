"""A synchronous facade over one persistent local MCP stdio connection."""

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from threading import Event, Thread
from typing import Any


MAX_MODEL_EVENT_ITEMS = 24


def _event_rank(event: dict[str, Any]) -> tuple[float, float]:
    """Rank representative events by confidence and duration."""
    confidence = event.get("confidence")
    confidence_value = float(confidence) if isinstance(confidence, (int, float)) else 0.0
    start = event.get("start_seconds")
    end = event.get("end_seconds")
    duration = (
        max(0.0, float(end) - float(start))
        if isinstance(start, (int, float)) and isinstance(end, (int, float))
        else 0.0
    )
    return confidence_value, duration


def _representative_events(
    events: list[dict[str, Any]],
    limit: int = MAX_MODEL_EVENT_ITEMS,
) -> list[dict[str, Any]]:
    """Keep channel coverage first, then fill with globally strongest events."""
    if len(events) <= limit:
        return list(events)
    best_by_channel: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, event in enumerate(events):
        channel = str(event.get("channel") or "unknown")
        current = best_by_channel.get(channel)
        if current is None or _event_rank(event) > _event_rank(current[1]):
            best_by_channel[channel] = (index, event)
    selected_indexes = {
        index
        for index, _event in sorted(
            best_by_channel.values(), key=lambda item: _event_rank(item[1]), reverse=True
        )[:limit]
    }
    if len(selected_indexes) < limit:
        remaining = sorted(
            (
                (index, event) for index, event in enumerate(events)
                if index not in selected_indexes
            ),
            key=lambda item: _event_rank(item[1]),
            reverse=True,
        )
        selected_indexes.update(index for index, _event in remaining[:limit - len(selected_indexes)])
    selected = [events[index] for index in selected_indexes]
    return sorted(
        selected,
        key=lambda event: (
            float(event.get("start_seconds", 0)),
            float(event.get("end_seconds", 0)),
            str(event.get("channel", "")),
        ),
    )


def _event_counts(events: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        value = str(event.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _compact_events_for_model(
    structured: dict[str, Any],
    event_key: str,
) -> dict[str, Any]:
    """Return a model-facing copy with aggregate counts and bounded events."""
    events = structured.get(event_key)
    if not isinstance(events, list):
        return structured
    compact = dict(structured)
    representatives = _representative_events(events)
    compact[event_key] = representatives
    compact["events_are_representative"] = len(representatives) < len(events)
    compact["events_returned_to_model"] = len(representatives)
    compact["events_omitted_from_model"] = max(0, len(events) - len(representatives))
    compact["event_counts_by_channel"] = _event_counts(events, "channel")
    compact["event_counts_by_brain_region"] = _event_counts(events, "brain_region")
    return compact


def _structured_result_for_model(tool_name: str | None, structured: Any) -> Any:
    """Apply tool-specific size policy without mutating the local full result."""
    if not isinstance(structured, dict):
        return structured
    if tool_name == "detect_eeg_events":
        return _compact_events_for_model(structured, "events")
    if tool_name == "generate_eeg_report":
        return _compact_events_for_model(structured, "abnormal_findings")
    return structured


class MCPClientBridge:
    """Keep one EEG MCP server process alive for the desktop application's lifetime."""

    def __init__(self, project_root: Path):
        """初始化对象状态。"""
        self.project_root = project_root
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._ready = Event()
        self._startup_error: Exception | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self._tools: list[Any] = []

    def start(self) -> None:
        """启动当前流程。"""
        if self._thread is not None:
            if self._startup_error:
                raise RuntimeError("Unable to start the local MCP EEG server.") from self._startup_error
            return
        self._thread = Thread(target=self._run_loop, name="eeg-mcp-client", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=30)
        if not self._ready.is_set():
            raise TimeoutError("Timed out while starting the local MCP EEG server.")
        if self._startup_error:
            raise RuntimeError("Unable to start the local MCP EEG server.") from self._startup_error

    def _run_loop(self) -> None:
        """处理 run loop 相关逻辑。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connection_lifecycle())
        except Exception as exc:  # 保存中间状态。
            if not self._ready.is_set():
                self._startup_error = exc
                self._ready.set()
        finally:
            self._loop.close()

    async def _connection_lifecycle(self) -> None:
        """Enter and exit MCP's AnyIO contexts in the same asyncio task."""
        self._shutdown_event = asyncio.Event()
        try:
            await self._connect()
        except Exception as exc:
            await self._disconnect()
            self._startup_error = exc
            self._ready.set()
            return

        self._ready.set()
        try:
            await self._shutdown_event.wait()
        finally:
            await self._disconnect()

    async def _connect(self) -> None:
        """处理 connect 相关逻辑。"""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError("The MCP SDK is not installed. Run: pip install 'mcp>=1.26,<2'") from exc

        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            cwd=str(self.project_root),
        )
        self._stack = AsyncExitStack()
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(parameters))
        self._session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()
        self._tools = list((await self._session.list_tools()).tools)

    async def _disconnect(self) -> None:
        """处理 disconnect 相关逻辑。"""
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
        self._tools = []

    def _submit(self, coroutine):
        """处理 submit 相关逻辑。"""
        self.start()
        if self._loop is None:
            raise RuntimeError("The MCP event loop is unavailable.")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout=180)

    def list_tools(self) -> list[Any]:
        """处理 list tools 相关逻辑。"""
        self.start()
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """处理 call tool 相关逻辑。"""
        return self._submit(self._call_tool(name, arguments))

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """处理 call tool 相关逻辑。"""
        if self._session is None:
            raise RuntimeError("The MCP session is not connected.")
        result = await self._session.call_tool(name, arguments=arguments)
        return self._normalize_result(result)

    @staticmethod
    def _normalize_result(result: Any) -> dict[str, Any]:
        """处理 normalize result 相关逻辑。"""
        structured = getattr(result, "structuredContent", None)
        if structured is None:
            structured = getattr(result, "structured_content", None)
        content = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text is not None:
                content.append(text)
        # 兼容旧版 MCP 服务可能只在文本内容中返回 JSON 结果。
        if structured is None and len(content) == 1:
            try:
                structured = json.loads(content[0])
            except json.JSONDecodeError:
                pass
        return {
            "is_error": bool(getattr(result, "isError", getattr(result, "is_error", False))),
            "structured_content": structured,
            "content": content,
        }

    def close(self) -> None:
        """关闭并移除 EEG 会话。"""
        if self._loop is None or self._thread is None:
            return
        thread = self._thread
        shutdown_event = self._shutdown_event
        if shutdown_event is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(shutdown_event.set)
        thread.join(timeout=10)
        if thread.is_alive():
            return
        self._thread = None
        self._loop = None
        self._shutdown_event = None
        self._startup_error = None
        self._ready.clear()


def result_for_model(result: dict[str, Any], tool_name: str | None = None) -> str:
    """处理 result for model 相关逻辑。"""
    if result.get("is_error"):
        structured = result.get("structured_content")
        if isinstance(structured, dict):
            error = dict(structured)
        elif structured is not None:
            error = {"details": structured}
        else:
            error = {}
        error["ok"] = False
        error["is_error"] = True
        content = result.get("content") or []
        if content and not error.get("message"):
            error["message"] = "\n".join(content)
        if not error.get("message"):
            error["message"] = "The tool call failed without an error message."
        return json.dumps(error, ensure_ascii=False)
    if result["structured_content"] is not None:
        structured = _structured_result_for_model(tool_name, result["structured_content"])
        return json.dumps(structured, ensure_ascii=False)
    if result["content"]:
        return "\n".join(result["content"])
    return json.dumps({"error": "The tool returned no content."}, ensure_ascii=False)
