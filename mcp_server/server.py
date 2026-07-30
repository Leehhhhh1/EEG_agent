import json
from pathlib import Path

import mne
from mcp.server.fastmcp import FastMCP

from eeg_core.basic_information import build_basic_information
from eeg_core.session_manager import EEGSessionManager, SessionNotFoundError
from eeg_skills.detection import detect_events
from eeg_skills.exploration import explore_segment
from eeg_skills.reporting import export_report, generate_report


mne.set_log_level("WARNING")
mcp = FastMCP("EEGAgent MCP Skills", json_response=True)
sessions = EEGSessionManager()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
with (PROJECT_ROOT / "config" / "config.json").open("r", encoding="utf-8") as config_file:
    CONFIG = json.load(config_file)


def _resolve_edf_path(file_path: str) -> Path:
    """解析并校验用户选择的 EDF 文件路径。"""
    path = Path(file_path).expanduser().resolve()
    if path.suffix.lower() != ".edf":
        raise ValueError("Only EDF files are supported.")
    if not path.is_file():
        raise FileNotFoundError(f"EDF file not found: {path}")
    return path


@mcp.tool()
def open_eeg_session(file_path: str) -> dict:
    """处理 open eeg session 相关逻辑。"""
    path = _resolve_edf_path(file_path)
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    basic_info = build_basic_information(path, raw)
    session = sessions.create(path, raw, basic_info, CONFIG)
    recording = basic_info["recording"]
    montage = basic_info["montage"]
    return {
        "session_id": session.session_id,
        "recording_name": recording["name"],
        "duration_seconds": recording["duration_seconds"],
        "channel_count": montage["raw_channel_count"],
    }


@mcp.tool()
def get_eeg_basic_information(session_id: str) -> dict:
    """获取 get eeg basic information 相关信息。"""
    try:
        return sessions.get(session_id).basic_info
    except SessionNotFoundError as exc:
        raise ValueError(str(exc)) from exc


@mcp.tool()
def explore_eeg_segment(
    session_id: str,
    start: float,
    end: float,
    focus: str = "overview",
    channels: list[str] | None = None,
) -> dict:
    """处理 explore eeg segment 相关逻辑。"""
    try:
        session = sessions.get(session_id)
    except SessionNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    return explore_segment(session, start, end, focus, channels)


@mcp.tool()
def detect_eeg_events(
    session_id: str,
    start: float = 0,
    end: float | None = None,
    event_types: list[str] | None = None,
    channels: list[str] | None = None,
    sensitivity: str = "balanced",
) -> dict:
    """处理 detect eeg events 相关逻辑。"""
    try:
        session = sessions.get(session_id)
    except SessionNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    return detect_events(session, start, end, event_types, channels, sensitivity)


@mcp.tool()
def generate_eeg_report(
    session_id: str,
    include_basic_information: bool = True,
    include_exploration: bool = True,
    include_detection: bool = True,
    language: str = "zh-CN",
) -> dict:
    """处理 generate eeg report 相关逻辑。"""
    try:
        session = sessions.get(session_id)
    except SessionNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    return generate_report(session, include_basic_information, include_exploration, include_detection, language)


@mcp.tool()
def export_eeg_report(
    session_id: str,
    report_id: str,
    output_path: str,
    format: str = "json",
) -> dict:
    """导出 export eeg report 相关结果。"""
    try:
        session = sessions.get(session_id)
    except SessionNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    return export_report(session, report_id, output_path, format)


@mcp.tool()
def close_eeg_session(session_id: str) -> dict:
    """处理 close eeg session 相关逻辑。"""
    if not sessions.close(session_id):
        raise ValueError(f"Unknown EEG session: {session_id}")
    return {"session_id": session_id, "closed": True}


if __name__ == "__main__":
    mcp.run(transport="stdio")
