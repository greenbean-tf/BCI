"""
不會變動的變數存放位置，global
"""
from enum import Enum
import os
import torch

# use in main.py
RECEIVE_CYGNUS_LSL_STREAM = "Cygnus-329018-RawEEG" 


the_zero_song_name = "Hanami" # Hanami

TCP_PORT = 50007
TCP_HOST = "0.0.0.0"  # all
TCP_CLIENT = "127.0.0.1"

SAMPLE_RATE = 500
SAMPLE_INTERVAL = 0.002




# 主要針對資料讀取的 channel，但是在 CygnusEEGReader.py read_eeg 裡面要對應 channel 做額外處裡
choose_channel_idx = [7, 8, 9, 12, 13, 14, 17, 18, 19, 21, 22, 23, 28]
CHOOSE_CHANNELS = 13
choose_channels_names =[
                    "F3",   "Fz",   "F4",
                    "FC3",  "FCz",  "FC4",
                    "C3",   "Cz",   "C4",
                    "CP3",  "CPz",  "CP4",
                            "Pz", 
            ]

# 已經 choose 後的 idx --> "C3",   "Cz",   "C4",
compute_alpha_channel = [6, 7, 8]


ALL_CHANNELS = 32
all_channels_names =[
                            "Fp1",          "Fp2", 
                        "AF3",                  "AF4", 
                    "F7",   "F3",   "Fz",   "F4",   "F8",
                    "FT7",  "FC3",  "FCz",  "FC4",  "FT8", 
                    "T7",   "C3",   "Cz",   "C4",   "T8",
                    "TP7",  "CP3",  "CPz",  "CP4",  "TP8", 
                    "P7",   "P3",   "Pz",   "P4",   "P8",
                        "O1",       "Oz",       "O2"
            ]

N_Class = 4

stay_marker = -1
choose_marker = []

for i in range(N_Class):
    choose_marker.append(i)


stay_prob = [0.0] * N_Class
prob_classes = []

for i in range(N_Class):
    prob_classes.append(f"prob_class_{i}")



WINDOW_SECONDS = 1.0
BUFFER_SIZE = int(SAMPLE_RATE * WINDOW_SECONDS)
PREDICTION_INTERVAL = 0.01              # seconds between predictions  # EEG_Prediction.py 裡面 # 實際會比這個慢，大概 1 s 70 個



band_pass_low = 1
band_pass_high = 40

# use in EEG_Read True, 真正測試要改成 False
# 要 connect 才能測試
is_simulated_eeg = False             



BASE_FILE = "EEG_DL"


account_dir = "account"
bandit_RL = "bandit.pt"
ver_json = "version.json"
log_dir = "log_dir"
model_dir = "model"
base_model_dir = "base_model"
data_dir = "real_time_data"



# use in EEG_Read
SAVE_CSV = True

# use in main run_level
USE_RL_LOOP = True

EEG_CHECKPOINT_MAIN_BASE_FILE = os.path.join("EEG_DL", "EEG", "base_model")


base_model_name = f"for_VR_{N_Class}class.pth"
base_base_model = os.path.join(EEG_CHECKPOINT_MAIN_BASE_FILE, base_model_name)



TCP_STR_SPLIT = "@@@"



# tsai: 2026/1/16 append Osu = "Osu"
# 目前好像無用
class GameSTATE(Enum):
    Calibration = "EEG_Calibration"
    BeatSaber = "BeatSaber"
    Osu = "Osu"
    MI = "MI"
    LOBBY = "Lobby"
    TRAIN = "Training"


# tsai: 2026/1/16 append UP = "up", DOWN = "down"
# 目前好像無用
class MIClass(Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    NONE = "none"


class TAGS(Enum):
    INFO = "[INFO]"
    MARKER = "[MARKER]"
    WARNING = "[WARNING]"
    ERROR = "[ERROR]"
