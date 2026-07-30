from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class EEGSession:
    """保存单个已打开 EEG 记录的内存状态。"""

    session_id: str
    file_path: Path
    raw: Any
    basic_info: dict[str, Any]
    config: dict[str, Any]
    created_at: datetime
    processed_data: Any | None = None
    bipolar_channels: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    reports: list[dict[str, Any]] = field(default_factory=list)

    @property
    def recording_name(self) -> str:
        """返回当前记录文件名。"""
        return self.file_path.name

    @classmethod
    def now(cls, session_id: str, file_path: Path, raw: Any, basic_info: dict[str, Any], config: dict[str, Any]):
        """使用当前时间创建 EEG 会话对象。"""
        return cls(
            session_id=session_id,
            file_path=file_path,
            raw=raw,
            basic_info=basic_info,
            config=config,
            created_at=datetime.now(timezone.utc),
        )
