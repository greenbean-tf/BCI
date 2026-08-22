"""
EEG 預測器（背景線程）。

對應原 EEG_DL/EEG/EEG_Predict.py。
"""
import time
import threading
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from braindecode.models import ShallowFBCSPNet

from eeg.eeg_model import load_shallowfbcsp_params, load_EEGNet_our_ver_params, EEGNet_our_ver

import config
import eeg.preprocess as pps
import global_state


class EEGPredictor:
    def __init__(self, tcp_client):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_check_point_path = ""

        # 使用 EEGNet
        self.model_class = EEGNet_our_ver
        self.model = None  # 由 init_model() 初始化
        self.load_func = load_EEGNet_our_ver_params

        self.model_arg = self.set_model_arg()

        self.predict_eeg_stop_event = threading.Event()
        self.predict_eeg_thread = None

        # 模型切換鎖：RL Loop 重載模型時，predict_loop 暫停推論
        self.modelLock = threading.Lock()

        self.tcp_client = tcp_client

    def start_predict_eeg_thread(self):
        """啟動預測線程"""
        if self.predict_eeg_thread and self.predict_eeg_thread.is_alive():
            print(f"{config.TAGS.WARNING.value} Marker thread is already running.")
            return

        # 載入最新 checkpoint
        self.load_check_point_path = global_state.NOW_FT_CHECKPOINT + ".pth"
        self.model = self.init_model()
        print(f"[DEBUG] 重新載入模型: {self.load_check_point_path}")

        print(f"{config.TAGS.INFO.value} Start predict loop...")
        self.predict_eeg_stop_event.clear()
        self.predict_eeg_thread = threading.Thread(target=self.predict_loop, daemon=True)
        self.predict_eeg_thread.start()

    def end_predict_eeg_thread(self):
        """停止預測線程"""
        print(f"{config.TAGS.INFO.value} End predict loop...")
        self.predict_eeg_stop_event.set()
        if self.predict_eeg_thread and self.predict_eeg_thread.is_alive():
            self.predict_eeg_thread.join()
            self.predict_eeg_thread = None

    def init_model(self):
        """載入模型權重"""
        model = self.model_class(**self.model_arg).to(self.device)
        checkpoint = torch.load(self.load_check_point_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def update_check_point(self, path):
        self.load_check_point_path = path

    def update_model(self):
        """重新載入模型（使用 Lock 防止與 predict_loop 並發）"""
        # 1. 先在背景載入模型 (耗時的 Disk I/O，不上鎖)
        new_model = self.init_model()
        
        # 2. 只有在替換指標的瞬間上鎖 (耗時 < 1 微秒)
        with self.modelLock:
            self.model = new_model
            print(f"[EEGPredictor] update_model: 模型已重載")

    def reload_model(self, checkpointPath: str):
        """
        指定路徑重載模型（供 RL Loop 在 DL_n → DL_{n+1} 時呼叫）。

        Args:
            checkpointPath: 新模型的 .pth 檔案路徑
        """
        oldPath = self.load_check_point_path
        self.load_check_point_path = checkpointPath
        
        # 1. 先在背景載入新模型 (耗時的 Disk I/O，不上鎖)
        new_model = self.init_model()
        
        # 2. 只有在替換指標的瞬間上鎖 (耗時 < 1 微秒)
        with self.modelLock:
            self.model = new_model
            print(f"[EEGPredictor] reload_model: {oldPath} → {checkpointPath}")

    def to_dataset(self, segment_list, label_list):
        data_x = np.array(segment_list)
        X = torch.tensor(data_x).unsqueeze(1)
        y = torch.tensor(label_list).long()
        return TensorDataset(X, y)

    def set_model_arg(self):
        simulate_x = np.zeros((1, config.CHOOSE_CHANNELS, config.SAMPLE_RATE))
        simulate_y = np.zeros((1, config.N_Class))
        dataset = self.to_dataset(simulate_x, simulate_y)
        return self.load_func(dataset)

    def predict_loop(self):
        """預測主迴圈"""
        # 等待 buffer 至少填滿一個完整視窗
        print("[EEGPredictor] 等待 EEG buffer 填滿...")
        while not self.predict_eeg_stop_event.is_set():
            if global_state.eeg_buffer.shape[1] >= config.SAMPLE_RATE:
                # 檢查 buffer 是否已被真實資料覆蓋（非 np.empty 遺留）
                if np.any(global_state.eeg_buffer[:, -1] != 0):
                    break
            time.sleep(0.1)
        print("[EEGPredictor] buffer 已就緒，開始推論")
        # ... 原有的 predict_loop 邏輯


        while not self.predict_eeg_stop_event.is_set():
            if global_state.eeg_buffer.shape[1] >= config.SAMPLE_RATE:
                data = global_state.eeg_buffer[:, -config.SAMPLE_RATE:]

                data = pps.each_channel_demean(data, axis=1)
                data = pps.bandpass(data, axis=-1)

                x_batch = torch.from_numpy(data.copy()).float().unsqueeze(0).to(self.device)

                self.model.eval()
                with torch.no_grad():
                    with self.modelLock:
                        outputs = self.model(x_batch)

                    prediction = int(torch.argmax(outputs).cpu().item())
                    probability = F.softmax(outputs, dim=1).cpu().detach().numpy()[0].tolist()

                    global_state.pred_buffer.append(prediction)
                    global_state.prob_buffer.append(probability)

                    # TCP 推送（目前關閉）
                    if False:
                        self.tcp_client.send_messages(
                            f"pred{config.TCP_STR_SPLIT}"
                            f"{prediction}{config.TCP_STR_SPLIT}"
                            f"{probability[0]}{config.TCP_STR_SPLIT}"
                            f"{probability[1]}{config.TCP_STR_SPLIT}"
                            f"{probability[2]}{config.TCP_STR_SPLIT}"
                            f"{probability[3]}"
                        )
                    time.sleep(config.PREDICTION_INTERVAL)


# ─── Debug ───
if __name__ == "__main__":
    print("=== eeg.eeg_predictor standalone test ===")
    print("EEGPredictor requires a trained model checkpoint to run.")
    print("✅ eeg_predictor import OK")
