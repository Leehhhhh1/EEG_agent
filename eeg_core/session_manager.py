from threading import RLock
from uuid import uuid4

from .session import EEGSession


class SessionNotFoundError(KeyError):
    pass


class EEGSessionManager:
    """线程安全的已打开 EEG 记录内存注册表。"""

    def __init__(self):
        """初始化对象状态。"""
        self._sessions: dict[str, EEGSession] = {}
        self._lock = RLock()

    def create(self, file_path, raw, basic_info: dict, config: dict) -> EEGSession:
        """创建并登记新的 EEG 会话。"""
        session_id = f"eeg_{uuid4().hex[:12]}"
        session = EEGSession.now(session_id, file_path, raw, basic_info, config)
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> EEGSession:
        """按会话编号获取 EEG 会话。"""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Unknown EEG session: {session_id}")
        return session

    def close(self, session_id: str) -> bool:
        """关闭并移除 EEG 会话。"""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
