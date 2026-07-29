"""Evidence-linked screening report generation and lightweight export."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from eeg_core.report_schema import DIAGNOSTIC_STATUS, REPORT_TYPE, validate_language


def _findings_for_skill(session: Any, skill: str) -> list[dict[str, Any]]:
    return [entry["result"] for entry in session.findings if entry.get("skill") == skill and "result" in entry]


def _exploration_section(results: list[dict[str, Any]], language: str) -> dict[str, Any]:
    screens = []
    for result in results:
        findings = result.get("abnormality_screen", {}).get("findings", [])
        window = result.get("window", {})
        screens.append({
            "window": {
                "start_seconds": window.get("start_seconds"),
                "end_seconds": window.get("end_seconds"),
            },
            "findings": findings,
            "source": {"skill": "exploration", "focus": result.get("focus")},
        })
    if language == "zh-CN":
        summary = "未执行背景活动探索。" if not screens else f"已汇总 {len(screens)} 个探索分析窗口。"
    else:
        summary = "No background exploration was run." if not screens else f"Summarized {len(screens)} exploration window(s)."
    return {"summary": summary, "screens": screens}


def _detection_section(results: list[dict[str, Any]], language: str) -> dict[str, Any]:
    events = []
    for result in results:
        for event in result.get("events", []):
            events.append({
                "event_type": event.get("event_type"),
                "start_seconds": event.get("start_seconds"),
                "end_seconds": event.get("end_seconds"),
                "channel": event.get("channel"),
                "brain_region": event.get("brain_region"),
                "confidence": event.get("confidence"),
                "evidence": event.get("evidence", {}),
                "source": {"skill": "detection", "session_id": result.get("session_id")},
            })
    if language == "zh-CN":
        summary = f"共汇总 {len(events)} 个需临床复核的发作样导联事件。"
    else:
        summary = f"Summarized {len(events)} seizure-like channel event(s) requiring clinical review."
    return {"summary": summary, "events": events}


def _impression(exploration: dict[str, Any], detection: dict[str, Any], language: str) -> str:
    event_count = len(detection["events"])
    screen_count = sum(len(screen["findings"]) for screen in exploration["screens"])
    if language == "zh-CN":
        if event_count:
            return f"自动筛查发现 {event_count} 个发作样导联事件，并存在 {screen_count} 项探索筛查提示；应结合原始波形由脑电图专业人员复核。"
        if screen_count:
            return f"未发现达到当前阈值的发作样导联事件，但存在 {screen_count} 项探索筛查提示；应结合原始波形复核。"
        return "当前已执行分析未发现达到阈值的发作样导联事件；该结果不能排除癫痫样放电或其他异常。"
    if event_count:
        return f"Automated screening found {event_count} seizure-like channel event(s) and {screen_count} exploration flag(s); review of the source EEG is required."
    if screen_count:
        return f"No seizure-like channel event met the current threshold, but {screen_count} exploration flag(s) were present; review of the source EEG is required."
    return "The completed analyses found no seizure-like channel events meeting the threshold; this does not exclude epileptiform or other abnormal activity."


def generate_report(
    session: Any,
    include_basic_information: bool = True,
    include_exploration: bool = True,
    include_detection: bool = True,
    language: str = "zh-CN",
) -> dict[str, Any]:
    """Create an evidence-linked report draft only from recorded session findings."""
    validate_language(language)
    exploration_results = _findings_for_skill(session, "exploration") if include_exploration else []
    detection_results = _findings_for_skill(session, "detection") if include_detection else []
    exploration = _exploration_section(exploration_results, language)
    detection = _detection_section(detection_results, language)
    report_id = f"report_{uuid4().hex[:12]}"
    limitations = (
        [
            "本报告由自动化筛查生成，不构成癫痫或其他疾病诊断。",
            "所有异常提示均需由具备资质的脑电图专业人员结合原始波形、临床病史和采集条件复核。",
            "未达到当前阈值不代表不存在癫痫样放电、发作或其他异常。",
        ]
        if language == "zh-CN"
        else [
            "This report is an automated screening draft and is not a diagnosis of epilepsy or another condition.",
            "All screening flags require review of the source EEG, clinical history, and acquisition context by a qualified EEG professional.",
            "No event meeting the current threshold does not exclude epileptiform activity, seizures, or other abnormalities.",
        ]
    )
    report = {
        "report_id": report_id,
        "session_id": session.session_id,
        "report_type": REPORT_TYPE,
        "language": language,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patient_information": session.basic_info.get("patient", {}) if include_basic_information else {},
        "recording_information": session.basic_info.get("recording", {}) if include_basic_information else {},
        "montage_information": session.basic_info.get("montage", {}) if include_basic_information else {},
        "background_summary": exploration,
        "abnormal_findings": detection["events"],
        "detection_summary": detection["summary"],
        "impression": _impression(exploration, detection, language),
        "limitations": limitations,
        "diagnostic_status": DIAGNOSTIC_STATUS,
        "provenance": {
            "included_skills": [skill for skill, enabled in (("basic_information", include_basic_information), ("exploration", include_exploration), ("detection", include_detection)) if enabled],
            "exploration_result_count": len(exploration_results),
            "detection_result_count": len(detection_results),
        },
    }
    session.reports.append(report)
    return report


def _get_report(session: Any, report_id: str) -> dict[str, Any]:
    for report in reversed(session.reports):
        if report.get("report_id") == report_id:
            return report
    raise ValueError(f"Unknown report '{report_id}' for session {session.session_id}.")


def _render_markdown(report: dict[str, Any]) -> str:
    recording = report.get("recording_information", {})
    patient = report.get("patient_information", {})
    lines = [
        "# EEG Screening Report Draft",
        "",
        f"- Report ID: {report['report_id']}",
        f"- Session ID: {report['session_id']}",
        f"- Generated: {report['generated_at']}",
        "",
        "## Patient Information",
        f"- Age: {patient.get('age', 'unavailable')}",
        f"- Sex: {patient.get('sex', 'unavailable')}",
        "",
        "## Recording Information",
        f"- Name: {recording.get('name', 'unavailable')}",
        f"- Duration (s): {recording.get('duration_seconds', 'unavailable')}",
        f"- Sampling rate (Hz): {recording.get('sampling_rate_hz', 'unavailable')}",
        "",
        "## Background Summary",
        report["background_summary"]["summary"],
        "",
        "## Detection Summary",
        report["detection_summary"],
        "",
        "## Abnormal Findings",
    ]
    events = report.get("abnormal_findings", [])
    if events:
        for event in events:
            lines.append(f"- {event['event_type']}: {event['start_seconds']}-{event['end_seconds']} s, {event['channel']}, {event['brain_region']}, confidence {event['confidence']}")
    else:
        lines.append("- None meeting the configured screening threshold.")
    lines.extend(["", "## Impression", report["impression"], "", "## Limitations"])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def export_report(session: Any, report_id: str, output_path: str, format: str = "json") -> dict[str, Any]:
    """Export a session report as JSON or Markdown after explicit user request."""
    if format not in {"json", "markdown"}:
        raise ValueError("Only json and markdown exports are available in this version.")
    report = _get_report(session, report_id)
    path = Path(output_path).expanduser().resolve()
    expected_suffix = ".json" if format == "json" else ".md"
    if path.suffix.lower() != expected_suffix:
        raise ValueError(f"A {format} export must use the '{expected_suffix}' file extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "json":
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        path.write_text(_render_markdown(report), encoding="utf-8")
    return {"report_id": report_id, "format": format, "output_path": str(path), "exported": True}
