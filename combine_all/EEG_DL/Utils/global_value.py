"""
會變動的變數存放位置，global
"""
import numpy as np

import threading
import os


import EEG_DL.Utils.config as config
from collections import deque


# ----------------------------
# Global buffer & concurrency
# ----------------------------
eeg_buffer = np.empty((config.CHOOSE_CHANNELS, 500), dtype=float)       # 用在 EEG_Read 和 EEGPrediction
data = np.empty((config.CHOOSE_CHANNELS, 1), dtype=float)               # 用在 EEG_Read 和 EEGPrediction
ts = 0                                                                  # 用在 EEG_Read 和 EEGPrediction timestamp


# use in EEG_Read
buffer_lock = threading.Lock()


# use in EEG_Read # 32 channel
eeg_buffer_ori_data = np.empty((config.ALL_CHANNELS, 500), dtype=float) # 用在 EEG_Read
buffer_lock_ori_data = threading.Lock()





pred_buffer = deque(maxlen=200)

prob_buffer = deque(maxlen=200)

current_marker = -1
current_predict = -1
current_prob = [0.0, 0.0, 0.0, 0.0]
start_lsl_time = 0.0


REALTIME_BASE_FILE = os.path.join("account", "Guest", "real_time_data")
CSV_FILENAME = os.path.join(REALTIME_BASE_FILE, "eeg_record.csv")


RL_path = os.path.join("account", "Guest", "bandit.npy")

json_path = os.path.join("account", "Guest", "version.json")




save_dir_model = os.path.join("account", "Guest", "model")

user_base_model = os.path.join("account", "Guest", "model", "base_model", "for_VR_4class.pth")


EEG_CHECKPOINT_BASE_DIR = os.path.join("account", "Guest", "model", "base_model")

# 放 new check point "folder" ---> Calibed_Model_v001
EEG_CHECKPOINT_TMP_BASE_DIR = os.path.join("account", "Guest", "model", "base_model", "Calibed_Model_v001")


# 放要被 FT 的 model == now active model
NOW_FT_CHECKPOINT = os.path.join("EEG_DL", "EEG", "base_model", "for_VR_4class")

