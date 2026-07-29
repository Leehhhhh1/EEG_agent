from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class EEGSession:
    """In-memory state for one opened EEG recording."""

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
        return self.file_path.name

    @classmethod
    def now(cls, session_id: str, file_path: Path, raw: Any, basic_info: dict[str, Any], config: dict[str, Any]):
        return cls(
            session_id=session_id,
            file_path=file_path,
            raw=raw,
            basic_info=basic_info,
            config=config,
            created_at=datetime.now(timezone.utc),
        )
