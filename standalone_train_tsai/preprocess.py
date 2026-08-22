"""
EEG 前處理函式（standalone 複本）。
"""
import numpy as np
from scipy.signal import filtfilt, butter

import config


def each_channel_demean(data, axis=0):
    """對 each channel 去均值"""
    return data - np.mean(data, axis=axis, keepdims=True)


def bandpass(data, axis=0, fs=config.SAMPLE_RATE,
             low=config.band_pass_low, high=config.band_pass_high):
    """帶通濾波"""
    b, a = butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data, axis=axis)
