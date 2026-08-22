"""
全域可變狀態（執行期間會被修改的變數）。

此檔案為整個 combine_all_new 專案的共用狀態，
涵蓋 EEG buffer、預測結果、帳號路徑等跨模組可變量。

原位於 eeg/global_state.py（歷史原因），現已提升至專案根目錄。
"""
import numpy as np
import threading
import os
from collections import deque

import config

# ─── EEG Buffer（預測用，已選通道） ───
eeg_buffer              = np.zeros((config.CHOOSE_CHANNELS, 500), dtype=float)
buffer_lock             = threading.Lock()

# ─── EEG Buffer（原始 32 通道） ───
eeg_buffer_ori_data     = np.zeros((config.ALL_CHANNELS, 500), dtype=float)
buffer_lock_ori_data    = threading.Lock()


# ─── 預測結果 Buffer ───
pred_buffer = deque(maxlen=200)
prob_buffer = deque(maxlen=200)

# ─── 即時狀態 ───
current_marker  = -1
current_predict = -1
current_prob    = [0.0, 0.0, 0.0, 0.0]
start_lsl_time  = 0.0

# ─── 帳號相關路徑（由 AccountManager 在登入後設定） ───
user = "Guest"
REALTIME_BASE_FILE  = os.path.join(config.account_dir, user, config.data_dir)
CSV_FILENAME        = os.path.join(REALTIME_BASE_FILE, "eeg_record.csv")

RL_path             = os.path.join(config.account_dir, user, config.bandit_RL)
json_path           = os.path.join(config.account_dir, user, config.ver_json)

save_dir_model      = os.path.join(config.account_dir, user, config.model_dir)
user_base_model     = os.path.join(config.account_dir, user, config.model_dir, config.base_model_dir, config.base_model_name)

EEG_CHECKPOINT_BASE_DIR     = os.path.join(config.account_dir, user, config.model_dir, config.base_model_dir)
EEG_CHECKPOINT_TMP_BASE_DIR = os.path.join(EEG_CHECKPOINT_BASE_DIR, "Calibed_Model_v001")

# 放要被 FT 的 model == now active model
NOW_FT_CHECKPOINT           = os.path.join(EEG_CHECKPOINT_BASE_DIR, config.base_model_name[:-4])


# ─── Debug ───
if __name__ == "__main__":
    print("=== global_state standalone test ===")
    print(f"eeg_buffer shape        = {eeg_buffer.shape}")
    print(f"eeg_buffer_ori shape    = {eeg_buffer_ori_data.shape}")
    print(f"pred_buffer maxlen      = {pred_buffer.maxlen}")
    print(f"REALTIME_BASE_FILE      = {REALTIME_BASE_FILE}")
    print(f"NOW_FT_CHECKPOINT       = {NOW_FT_CHECKPOINT}")
    print("✅ global_state OK")
