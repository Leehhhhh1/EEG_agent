"""Session-isolated event screening built on the repository's pretrained models."""

from typing import Any

import numpy as np

from eeg_core.processing import ensure_processed_data
from eeg_core.region_mapping import region_for_channel


VALID_EVENT_TYPES = {"seizure"}
SENSITIVITY_THRESHOLDS = {
    "sensitive": (0.25, 0.55),
    "balanced": (0.45, 0.70),
    "specific": (0.60, 0.85),
}
MAX_ANALYSIS_SECONDS = 600


def _select_channels(available: list[str], requested: list[str] | None) -> list[str]:
    """处理 select channels 相关逻辑。"""
    if requested is None:
        return available
    invalid = sorted(set(requested) - set(available))
    if invalid:
        raise ValueError(f"Unsupported bipolar channels: {', '.join(invalid)}")
    return requested


def _merge_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并 merge candidates 相关结果。"""
    events: list[dict[str, Any]] = []
    for candidate in candidates:
        if events and events[-1]["channel"] == candidate["channel"] and abs(events[-1]["end_seconds"] - candidate["start_seconds"]) < 1e-6:
            event = events[-1]
            event["end_seconds"] = candidate["end_seconds"]
            event["confidence"] = max(event["confidence"], candidate["confidence"])
            event["evidence"]["fine_windows"] += 1
        else:
            events.append(candidate)
    return events


def detect_events(
    session: Any,
    start: float = 0,
    end: float | None = None,
    event_types: list[str] | None = None,
    channels: list[str] | None = None,
    sensitivity: str = "balanced",
) -> dict[str, Any]:
    """处理 detect events 相关逻辑。"""
    requested_types = set(event_types or ["seizure"])
    unsupported = requested_types - VALID_EVENT_TYPES
    if unsupported:
        raise ValueError(f"Unsupported event types: {', '.join(sorted(unsupported))}. Only seizure is available in this version.")
    if sensitivity not in SENSITIVITY_THRESHOLDS:
        raise ValueError(f"Unsupported sensitivity '{sensitivity}'. Choose sensitive, balanced, or specific.")

    data, available_channels, fs = ensure_processed_data(session)
    if fs != 256:
        raise ValueError("Detection models require preprocessing at 256 Hz.")
    duration = data.shape[1] / fs
    end = duration if end is None else end
    if start < 0 or end <= start or end > duration:
        raise ValueError(f"Require 0 <= start < end <= {duration:g}.")
    if end - start > MAX_ANALYSIS_SECONDS:
        raise ValueError(f"Detection windows cannot exceed {MAX_ANALYSIS_SECONDS} seconds.")

    selected_channels = _select_channels(available_channels, channels)
    selected_indexes = [available_channels.index(channel) for channel in selected_channels]
    start_second = int(np.floor(start))
    end_second = int(np.ceil(end))
    coarse_threshold, fine_threshold = SENSITIVITY_THRESHOLDS[sensitivity]

    # 延迟导入，避免基础信息工具强制触发模型推理。
    from tools.singleChannel import predict_seizure_artifact_background, predict_seizure_normal
    from tools.slowSeizBckg import predict_slow_seizure_background

    coarse_windows = []
    candidate_seconds: set[int] = set()
    full_ten_second_windows = len(available_channels) == 22
    for window_start in range(start_second, end_second, 10):
        window_end = min(window_start + 10, end_second)
        if full_ten_second_windows and window_end - window_start == 10:
            chunk = data[:, window_start * 256:window_end * 256]
            coarse = predict_slow_seizure_background(chunk)
            coarse_windows.append({"start_seconds": window_start, "end_seconds": window_end, "probabilities": coarse})
            if coarse["seizure"] >= coarse_threshold:
                candidate_seconds.update(range(window_start, window_end))
        else:
            candidate_seconds.update(range(window_start, window_end))

    raw_candidates: list[dict[str, Any]] = []
    for second in sorted(candidate_seconds):
        if second < start_second or second >= end_second or second + 1 > duration:
            continue
        chunk = data[selected_indexes, second * 256:(second + 1) * 256]
        seizure_probs = predict_seizure_normal(chunk)
        artifact_probs = predict_seizure_artifact_background(chunk)
        for channel, seizure, artifact in zip(selected_channels, seizure_probs, artifact_probs):
            confidence = min(seizure["seizure"], artifact["seizure"])
            if confidence >= fine_threshold and artifact["artifact"] < confidence:
                raw_candidates.append({
                    "event_type": "seizure_like_activity",
                    "start_seconds": second,
                    "end_seconds": second + 1,
                    "channel": channel,
                    "brain_region": region_for_channel(channel),
                    "confidence": round(confidence, 4),
                    "evidence": {
                        "seizure_vs_nonseizure": round(seizure["seizure"], 4),
                        "seizure_vs_artifact_background": round(artifact["seizure"], 4),
                        "artifact_probability": round(artifact["artifact"], 4),
                        "fine_windows": 1,
                    },
                })

    events = _merge_candidates(raw_candidates)
    result = {
        "session_id": session.session_id,
        "analysis_window": {"start_seconds": start, "end_seconds": end, "channels": selected_channels, "sampling_rate_hz": fs},
        "event_types": sorted(requested_types),
        "sensitivity": sensitivity,
        "coarse_screen": coarse_windows,
        "events": events,
        "event_count": len(events),
        "diagnostic_status": "screening_only",
        "summary": (
            f"Detected {len(events)} seizure-like channel event(s) requiring clinical review."
            if events else "No seizure-like channel events met the selected screening threshold; this does not exclude epileptiform activity."
        ),
    }
    session.findings.append({"skill": "detection", "result": result})
    return result
