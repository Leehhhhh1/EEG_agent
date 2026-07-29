from pathlib import Path
from typing import Any

from tools.baseInfo import baseInfo
from tools.polar import bipolar_pairs

from .processing import find_raw_channel


def build_basic_information(file_path: Path, raw: Any) -> dict[str, Any]:
    """Build a JSON-safe summary without returning the raw signal."""
    header = baseInfo(str(file_path))
    channel_names = list(raw.ch_names)
    available_bipolar_channels = []
    for first, second in bipolar_pairs:
        if find_raw_channel(first, channel_names) and find_raw_channel(second, channel_names):
            available_bipolar_channels.append(f"{first}-{second}")

    duration_seconds = round(raw.n_times / raw.info["sfreq"], 3)
    patient = {
        key: header[key]
        for key in ("age", "sex")
        if key in header
    }
    warnings = []
    if not available_bipolar_channels:
        warnings.append("No supported bipolar channels could be constructed from this recording.")
    if "age" not in patient:
        warnings.append("Patient age is unavailable in the EDF header.")

    return {
        "recording": {
            "name": file_path.name,
            "format": "EDF",
            "duration_seconds": duration_seconds,
            "sampling_rate_hz": raw.info["sfreq"],
            "start_date": header.get("start_date"),
            "start_time": header.get("start_time"),
        },
        "patient": patient,
        "montage": {
            "raw_channel_count": len(channel_names),
            "raw_channels": channel_names,
            "available_bipolar_channels": available_bipolar_channels,
        },
        "analysis_constraints": {
            "valid_time_range_seconds": [0, duration_seconds],
            "supports_current_bipolar_pipeline": bool(available_bipolar_channels),
        },
        "warnings": warnings,
    }
