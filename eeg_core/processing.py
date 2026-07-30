from typing import Any

import numpy as np

from tools.polar import bipolar_pairs
from tools.preprocessing import preprocessing


def find_raw_channel(prefix: str, channel_names: list[str]) -> str | None:
    """查找 find raw channel 对应的数据。"""
    expected = f"EEG {prefix}".upper()
    for name in channel_names:
        if name.upper().startswith(expected):
            return name
    return None


def ensure_processed_data(session: Any) -> tuple[np.ndarray, list[str], float]:
    """处理 ensure processed data 相关逻辑。"""
    if session.processed_data is not None:
        return session.processed_data, session.bipolar_channels, session.config["fs"]

    raw = session.raw.copy().load_data()
    raw = preprocessing(raw, session.config)
    channel_names = list(raw.ch_names)
    bipolar_data = []
    bipolar_channels = []
    for first, second in bipolar_pairs:
        first_name = find_raw_channel(first, channel_names)
        second_name = find_raw_channel(second, channel_names)
        if first_name and second_name:
            bipolar_data.append(raw[first_name][0][0] - raw[second_name][0][0])
            bipolar_channels.append(f"{first}-{second}")

    if not bipolar_data:
        raise ValueError("No supported bipolar channels could be constructed from this recording.")

    session.processed_data = np.asarray(bipolar_data, dtype=np.float32)
    session.bipolar_channels = bipolar_channels
    return session.processed_data, session.bipolar_channels, raw.info["sfreq"]
