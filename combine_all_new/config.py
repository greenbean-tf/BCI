"""
全域不變常數與設定值。

此檔案為整個 combine_all_new 專案的共用設定，
涵蓋 EEG、TCP、遊戲、RL 和路徑等所有跨模組常數。

原位於 eeg/config.py（歷史原因），現已提升至專案根目錄。
"""
from enum import Enum
import os
import torch

# ─── LSL ───
RECEIVE_CYGNUS_LSL_STREAM = "Cygnus-329018-RawEEG"

# ─── 特殊譜面 ───
the_zero_song_name = "Hanami"

# ─── TCP ───
TCP_PORT        = 50007
TCP_HOST        = "0.0.0.0"
TCP_CLIENT      = "127.0.0.1"
TCP_STR_SPLIT   = "@@@"

# ─── 取樣率 ───
SAMPLE_RATE     = 500
SAMPLE_INTERVAL = 0.002

# ─── EEG 通道設定 ───
choose_channel_idx      = [7, 8, 9, 12, 13, 14, 17, 18, 19, 21, 22, 23, 28]
CHOOSE_CHANNELS         = 13
choose_channels_names   = [
    "F3",  "Fz",  "F4",
    "FC3", "FCz", "FC4",
    "C3",  "Cz",  "C4",
    "CP3", "CPz", "CP4",
           "Pz",
]

# Alpha 計算用的 channel idx（choose 後的 index）
compute_alpha_channel = [6, 7, 8]  # "C3", "Cz", "C4"


ALL_CHANNELS        = 32
all_channels_names  = [
            "Fp1",          "Fp2",
        "AF3",                 "AF4",
    "F7",   "F3",   "Fz",   "F4",   "F8",
    "FT7",  "FC3",  "FCz",  "FC4",  "FT8",
    "T7",   "C3",   "Cz",   "C4",   "T8",
    "TP7",  "CP3",  "CPz",  "CP4",  "TP8",
    "P7",   "P3",   "Pz",   "P4",   "P8",
        "O1",       "Oz",      "O2",
]


# ─── 分類數 ───
N_Class         = 4
choose_marker   = list(range(N_Class))
stay_marker     = -1

stay_prob       = [0.0] * N_Class
prob_classes    = [f"prob_class_{i}" for i in range(N_Class)]


# ─── Buffer ───
WINDOW_SECONDS      = 3.0
BUFFER_SIZE         = int(SAMPLE_RATE * WINDOW_SECONDS)
PREDICTION_INTERVAL = 0.01  # seconds between predictions

# ─── 帶通濾波 ───
band_pass_low   = 1
band_pass_high  = 40

# ─── 模擬模式 ───
is_simulated_eeg = False

# ─── 路徑 ───
account_dir     = "account"
bandit_RL       = "bandit.pt"
ver_json        = "version.json"
log_dir         = "log_dir"
model_dir       = "model"
base_model_dir  = "base_model"
data_dir        = "real_time_data"

# ─── CSV ───
SAVE_CSV    = True

# ─── RL Loop ───
USE_RL_LOOP = False

# ─── DL Base Model ───
EEG_CHECKPOINT_MAIN_BASE_FILE   = os.path.join("eeg", "base_model")

# 目前就只有 4 class 的 model
base_model_name                 = f"pretrain_DL_{N_Class}class.pth"
base_base_model                 = os.path.join(EEG_CHECKPOINT_MAIN_BASE_FILE, base_model_name)


# ─── RL Base Model ───
RL_CHECKPOINT_MAIN_BASE_FILE    = os.path.join("rl", "rl_base_model")
rl_base_base_model              = os.path.join(RL_CHECKPOINT_MAIN_BASE_FILE, "pretrained_bandit.pt")


# ─── Enums ───
class TAGS(Enum):
    INFO    = "[INFO]"
    MARKER  = "[MARKER]"
    WARNING = "[WARNING]"
    ERROR   = "[ERROR]"


# ─── Debug ───
if __name__ == "__main__":
    print("=== config standalone test ===")
    print(f"SAMPLE_RATE       = {SAMPLE_RATE}")
    print(f"CHOOSE_CHANNELS   = {CHOOSE_CHANNELS}")
    print(f"N_Class           = {N_Class}")
    print(f"BUFFER_SIZE       = {BUFFER_SIZE}")
    print(f"base_base_model   = {base_base_model}")
    print(f"choose_marker     = {choose_marker}")
    print(f"prob_classes      = {prob_classes}")
    print("✅ config OK")
