"""
EEG 前處理函式。

對應原 EEG_DL/Utils/preprocess.py。
"""
import numpy as np
from scipy.signal import filtfilt, butter, detrend, decimate

import config


def detrend_epoch(epoch):
    """去趨勢（type='constant' 為去均值，預設為線性去趨勢）"""
    return detrend(epoch, axis=1)


def each_channel_demean(data, axis=0):
    """對 each channel 去均值"""
    return data - np.mean(data, axis=axis, keepdims=True)


def bandpass(data, axis=0, fs=config.SAMPLE_RATE,
             low=config.band_pass_low, high=config.band_pass_high):
    """帶通濾波"""
    b, a = butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data, axis=axis)


def down_sample(data, new_fs=config.SAMPLE_RATE):
    """降取樣"""
    decimation_factor = config.SAMPLE_RATE // new_fs
    return decimate(data, decimation_factor, axis=-1, zero_phase=True)


# ─── Debug ───
if __name__ == "__main__":
    print("=== eeg.preprocess standalone test ===")
    test_data = np.random.randn(13, 500)
    demeaned = each_channel_demean(test_data, axis=1)
    filtered = bandpass(demeaned, axis=-1)
    print(f"Input shape:    {test_data.shape}")
    print(f"Demeaned shape: {demeaned.shape}")
    print(f"Filtered shape: {filtered.shape}")
    print("✅ preprocess OK")
