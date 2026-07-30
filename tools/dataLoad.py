import mne
import numpy as np
import regex as re
from .preprocessing import preprocessing
from .polar import bipolar_pairs, single_polar, sleep_polar
mne.set_log_level('WARNING')

# 双极
def match_channel(prefix, channel_list):
    """处理 match channel 相关逻辑。"""
    pattern = re.compile(rf'^{prefix}')
    for ch in channel_list:
        if pattern.match(ch):
            return ch
    return None

def get_data(raw):
    """获取 get data 相关信息。"""
    data = []
    raw_channels = raw.ch_names  # 获取原始通道名列表
    for ch1_prefix, ch2_prefix in bipolar_pairs:
        ch1 = match_channel('EEG '+ch1_prefix, raw_channels)
        ch2 = match_channel('EEG '+ch2_prefix, raw_channels)
        if ch1 and ch2:
            data.append(raw[ch1][0][0] - raw[ch2][0][0])
    return np.array(data, dtype=np.float32)


def dataLoad(file_path, config):
    """处理 data Load 相关逻辑。"""
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw = preprocessing(raw, config)
    data = get_data(raw)
    return data

def load_MDD_edf(file_path, config):
    """加载 load  M D D edf 所需的数据。"""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    names = ["EEG " + name + "-LE" for name in single_polar]
    raw.pick(names)
    data = raw.get_data()  # 张量形状说明。
    return np.array(data) 

def load_Sleep_edf(file_path, config):
    """加载 load  Sleep edf 所需的数据。"""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    names = ["EEG " + name for name in sleep_polar]
    raw.pick(names)
    data = raw.get_data()  # 张量形状说明。
    return np.array(data) 