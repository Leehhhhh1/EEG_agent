import asyncio
import unittest
from pathlib import Path

from agent_runtime.mcp_client import MCPClientBridge


class RecordingBridge(MCPClientBridge):
    def __init__(self):
        super().__init__(Path.cwd())
        self.connect_task = None
        self.disconnect_task = None

    async def _connect(self):
        self.connect_task = asyncio.current_task()

    async def _disconnect(self):
        self.disconnect_task = asyncio.current_task()


class MCPClientLifecycleTests(unittest.TestCase):
    def test_connect_and_disconnect_run_in_the_same_asyncio_task(self):
        bridge = RecordingBridge()
        bridge.start()
        bridge.close()

        self.assertIsNotNone(bridge.connect_task)
        self.assertIs(bridge.connect_task, bridge.disconnect_task)


if __name__ == "__main__":
    unittest.main()
