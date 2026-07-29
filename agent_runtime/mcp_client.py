"""A synchronous facade over one persistent local MCP stdio connection."""

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from threading import Event, Thread
from typing import Any


class MCPClientBridge:
    """Keep one EEG MCP server process alive for the desktop application's lifetime."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._ready = Event()
        self._startup_error: Exception | None = None
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self._tools: list[Any] = []

    def start(self) -> None:
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
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except Exception as exc:  # Stored so the UI can show the root cause.
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()
        self._loop.run_until_complete(self._disconnect())
        self._loop.close()

    async def _connect(self) -> None:
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
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
        self._tools = []

    def _submit(self, coroutine):
        self.start()
        if self._loop is None:
            raise RuntimeError("The MCP event loop is unavailable.")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout=180)

    def list_tools(self) -> list[Any]:
        self.start()
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return self._submit(self._call_tool(name, arguments))

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._session is None:
            raise RuntimeError("The MCP session is not connected.")
        result = await self._session.call_tool(name, arguments=arguments)
        return self._normalize_result(result)

    @staticmethod
    def _normalize_result(result: Any) -> dict[str, Any]:
        structured = getattr(result, "structuredContent", None)
        if structured is None:
            structured = getattr(result, "structured_content", None)
        content = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text is not None:
                content.append(text)
        # Older MCP 1.x servers may put a JSON tool result only in TextContent.
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
        if self._loop is None or self._thread is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        self._thread = None
        self._loop = None
        self._ready.clear()


def result_for_model(result: dict[str, Any]) -> str:
    """Use structured MCP output where available, preserving tool errors for model recovery."""
    if result["structured_content"] is not None:
        return json.dumps(result["structured_content"], ensure_ascii=False)
    if result["content"]:
        return "\n".join(result["content"])
    return json.dumps({"error": "The tool returned no content."}, ensure_ascii=False)
