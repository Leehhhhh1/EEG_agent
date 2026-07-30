from typing import Any

import numpy as np
from scipy.signal import welch

from eeg_core.processing import ensure_processed_data


VALID_FOCUSES = {
    "overview",
    "background_rhythm",
    "amplitude",
    "symmetry",
    "abnormality_screen",
}

SYMMETRY_PAIRS = [
    ("FP1-F7", "FP2-F8"),
    ("F7-T3", "F8-T4"),
    ("T3-T5", "T4-T6"),
    ("T5-O1", "T6-O2"),
    ("FP1-F3", "FP2-F4"),
    ("F3-C3", "F4-C4"),
    ("C3-P3", "C4-P4"),
    ("P3-O1", "P4-O2"),
]


def _select_channels(available: list[str], requested: list[str] | None) -> list[str]:
    """处理 select channels 相关逻辑。"""
    if requested is None:
        return available
    invalid = sorted(set(requested) - set(available))
    if invalid:
        raise ValueError(f"Unsupported bipolar channels: {', '.join(invalid)}")
    return requested


def _amplitude_features(data: np.ndarray, channels: list[str]) -> dict[str, dict[str, float]]:
    """处理 amplitude features 相关逻辑。"""
    microvolts = data * 1_000_000
    return {
        channel: {
            "mean_abs_uv": round(float(np.mean(np.abs(signal))), 3),
            "rms_uv": round(float(np.sqrt(np.mean(signal ** 2))), 3),
            "max_uv": round(float(np.max(signal)), 3),
            "min_uv": round(float(np.min(signal)), 3),
        }
        for channel, signal in zip(channels, microvolts)
    }


def _band_power(data: np.ndarray, channels: list[str], fs: float, bands: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """处理 band power 相关逻辑。"""
    result = {}
    for channel, signal in zip(channels, data):
        frequencies, power = welch(signal, fs=fs, nperseg=min(256, signal.size))
        powers = {}
        for name, (low, high) in bands.items():
            mask = (frequencies >= low) & (frequencies < high)
            powers[name] = round(float(np.trapz(power[mask], frequencies[mask]) * 1e12), 5) if np.any(mask) else 0.0
        result[channel] = powers
    return result


def _symmetry_features(data: np.ndarray, channels: list[str]) -> dict[str, float]:
    """处理 symmetry features 相关逻辑。"""
    indexes = {channel: index for index, channel in enumerate(channels)}
    result = {}
    for left, right in SYMMETRY_PAIRS:
        if left in indexes and right in indexes:
            correlation = np.corrcoef(data[indexes[left]], data[indexes[right]])[0, 1]
            result[f"{left}__{right}"] = round(float(correlation), 4) if np.isfinite(correlation) else 0.0
    return result


def _screen_findings(band_power: dict[str, dict[str, float]], symmetry: dict[str, float]) -> list[str]:
    """处理 screen findings 相关逻辑。"""
    findings = []
    for pair, correlation in symmetry.items():
        if correlation < 0.5:
            findings.append(f"Reduced left-right correlation in {pair} ({correlation:.2f}).")
    for channel, powers in band_power.items():
        slow = powers.get("delta", 0.0) + powers.get("theta", 0.0)
        fast = powers.get("alpha", 0.0) + powers.get("beta", 0.0) + powers.get("gamma", 0.0)
        if slow > 0 and slow > fast * 2:
            findings.append(f"Slow-band power is dominant in {channel}.")
    return findings


def _summary(focus: str, findings: list[str], start: float, end: float) -> str:
    """处理 summary 相关逻辑。"""
    prefix = f"Exploration of {start:g}-{end:g} seconds ({focus})."
    if findings:
        return f"{prefix} Screening findings: {' '.join(findings)} This is not an event diagnosis."
    return f"{prefix} No amplitude, spectral, or symmetry screening flags were identified. This is not an event diagnosis."


def explore_segment(session: Any, start: float, end: float, focus: str = "overview", channels: list[str] | None = None) -> dict[str, Any]:
    """处理 explore segment 相关逻辑。"""
    if focus not in VALID_FOCUSES:
        raise ValueError(f"Unsupported focus '{focus}'. Choose from: {', '.join(sorted(VALID_FOCUSES))}.")
    if start < 0 or end <= start:
        raise ValueError("Require 0 <= start < end.")
    if end - start > 60:
        raise ValueError("Exploration windows cannot exceed 60 seconds.")

    data, available_channels, fs = ensure_processed_data(session)
    duration = data.shape[1] / fs
    if end > duration:
        raise ValueError(f"Requested end time {end:g}s exceeds recording duration {duration:g}s.")

    selected_channels = _select_channels(available_channels, channels)
    indexes = [available_channels.index(channel) for channel in selected_channels]
    window = data[indexes, int(start * fs):int(end * fs)]
    if window.shape[1] < 2:
        raise ValueError("The selected EEG window contains insufficient samples.")

    include_amplitude = focus in {"overview", "background_rhythm", "amplitude", "abnormality_screen"}
    include_power = focus in {"overview", "background_rhythm", "abnormality_screen"}
    include_symmetry = focus in {"overview", "background_rhythm", "symmetry", "abnormality_screen"}
    amplitude = _amplitude_features(window, selected_channels) if include_amplitude else {}
    band_power = _band_power(window, selected_channels, fs, session.config["prior knowledge"]["freq_bands"]) if include_power else {}
    symmetry = _symmetry_features(window, selected_channels) if include_symmetry else {}
    findings = _screen_findings(band_power, symmetry) if focus in {"overview", "abnormality_screen"} else []

    result = {
        "session_id": session.session_id,
        "window": {
            "start_seconds": start,
            "end_seconds": end,
            "channels": selected_channels,
            "sampling_rate_hz": fs,
        },
        "focus": focus,
        "statistical_features": {
            "amplitude": amplitude,
            "band_power_uv_squared": band_power,
            "symmetry_correlation": symmetry,
        },
        "abnormality_screen": {
            "findings": findings,
            "diagnostic_status": "screening_only",
        },
        "summary": _summary(focus, findings, start, end),
    }
    session.findings.append({"skill": "exploration", "result": result})
    return result
