
import numpy as np
from scipy.signal import filtfilt, butter, detrend, decimate



import EEG_DL.Utils.config as config


# ----------------------------
# Preprocessing & features (unchanged)
# ----------------------------
def detrend_epoch(epoch):

    # 跟去平均很像， type='constant' 就是去平均，預設為會根據線性去平均
    return detrend(epoch, axis=1)

def each_channel_demean(data, axis=0):
    # data = (sample, channel)
    # 對 each channel 去平均
    return data - np.mean(data, axis=axis, keepdims=True)


def bandpass(data, axis=0, fs=config.SAMPLE_RATE, low=config.band_pass_low, high=config.band_pass_high):

    b, a = butter(4, [low / (0.5 * fs), high / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data, axis=axis)                                  # prediction axis = -1


def down_sample(data, new_fs=config.SAMPLE_RATE):                           # data: (n_samples, n_channels)

    decimation_factor = config.SAMPLE_RATE // new_fs                        # 500/125 = 4
    return decimate(data, decimation_factor, axis=-1, zero_phase=True)
