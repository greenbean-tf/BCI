"""
EEG 資料讀取器（背景線程）。

對應原 EEG_DL/EEG/EEG_Read.py。
負責從 LSL 或模擬源讀取 EEG 資料，更新 global buffer，並寫入 CSV。
"""
import time
import csv
import os
import threading
import numpy as np
from datetime import datetime
import pylsl

import config
import global_state
from eeg.helpers import rename_file_with_time


class EEGReader:
    def __init__(self, inlet=None):
        self.inlet = inlet
        self.is_simulated = config.is_simulated_eeg

        self.save_csv = config.SAVE_CSV
        self.filename = global_state.CSV_FILENAME

        self.csv_writer = None
        self.csv_file = None
        self.read_eeg_stop_event = threading.Event()
        self.read_eeg_thread = None

    def start_read_eeg_thread(self):
        """啟動背景讀取線程"""
        if self.read_eeg_thread and self.read_eeg_thread.is_alive():
            print(f"{config.TAGS.WARNING.value} Marker thread is already running.")
            return

        print(f"{config.TAGS.INFO.value} Start read cygnus eeg...")
        self.read_eeg_stop_event.clear()
        self.read_eeg_thread = threading.Thread(target=self.update_predict_buffer, daemon=True)
        self.read_eeg_thread.start()

    def end_read_eeg_thread(self):
        """停止背景讀取線程"""
        print(f"{config.TAGS.INFO.value} End read cygnus eeg...")
        self.read_eeg_stop_event.set()
        if self.read_eeg_thread and self.read_eeg_thread.is_alive():
            self.read_eeg_thread.join()
            self.read_eeg_thread = None

    def build_csv_file(self):
        """建立新的 CSV 檔案"""
        self.filename = rename_file_with_time(global_state.CSV_FILENAME)

        if config.SAVE_CSV:
            self.csv_file = open(self.filename, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([f"Simulating Cygnus version: 0.28.0.7,File version: 2021.11"])
            self.csv_writer.writerow([f"Operative system: Windows"])
            self.csv_writer.writerow([f"Record datetime: {datetime.now()}"])
            self.csv_writer.writerow([f"Simulating Device ID: STEEG_DG329018"])
            self.csv_writer.writerow([f"Device verison: "])
            self.csv_writer.writerow([f"Device bandwidth: DC to 131 Hz"])
            self.csv_writer.writerow([f"Device sampling rate: 500 samples/second"])
            self.csv_writer.writerow([f"Data type / unit: EEG / micro-volt (uV)"])
            self.csv_writer.writerow([f"Bandpass filter: None"])
            self.csv_writer.writerow([f"Notch filter: None"])
            header = (["Timestamp"] + ["Beatmap_Marker"] + ["Predict_Marker"]
                      + config.all_channels_names)
            self.csv_writer.writerow(header)
            print("[CSV] Start write EEG data to", self.filename)

    def close_csv_file(self):
        """關閉 CSV 檔案（先切斷寫入器，再關閉檔案，防止 race condition）"""
        # 先將 writer 設為 None，讓背景線程的 `if self.csv_writer` 檢查立即失敗
        writer = self.csv_writer
        fileObj = self.csv_file
        self.csv_writer = None
        self.csv_file = None

        if fileObj:
            try:
                fileObj.close()
                print("[CSV] Saved EEG data to", self.filename)
            except Exception as closeErr:
                print(f"[CSV] Saved fail: {closeErr}")

    def filter_channels(self, sample_arr):
        """將原始 32 通道過濾為選定的通道"""
        np_s = np.array(sample_arr)
        return np_s[config.choose_channel_idx, :]

    def update_predict_buffer(self):
        """背景執行緒：更新 global eeg_buffer 供預測使用，並負責高頻資料記錄"""
        while not self.read_eeg_stop_event.is_set():

            # print(f"[EEG] update predict buffer self.is_simulated: {self.is_simulated}")
            
            if self.is_simulated:
                # 模擬 500Hz 採樣
                sample = np.random.randint(-10001, 10001, size=(config.ALL_CHANNELS, 1))
                time.sleep(1 / config.SAMPLE_RATE)

                if self.csv_writer and config.SAVE_CSV:
                    try:
                        marker = global_state.current_marker
                        pred = global_state.current_predict
                        record_time_s = pylsl.local_clock() - global_state.start_lsl_time
                        row = [f"{record_time_s:.3f}", marker, pred] + sample.flatten().tolist()
                        self.csv_writer.writerow(row)
                    except (ValueError, AttributeError):
                        pass  # 檔案已關閉或 writer 已清除，靜默忽略

            else:
                if self.inlet is None:
                    time.sleep(1)
                    continue

                samples, ts = self.inlet.pull_chunk(timeout=0.0)

                # print(f"[EEG] samples {samples}")
                # print(f"[EEG] ts {ts}")

                if not samples:
                    time.sleep(0.001)
                    continue

                sample = np.array(samples).T

                # print(f"[EEG] sample.T {sample}")

                # 紀錄資料
                if self.csv_writer and config.SAVE_CSV:
                    try:
                        marker = global_state.current_marker
                        pred = global_state.current_predict
                        start_ts = global_state.start_lsl_time

                        batch_rows = []
                        for i, s_pt in enumerate(samples):
                            record_time_s = ts[i] - start_ts
                            row = [f"{record_time_s:.3f}", marker, pred] + s_pt
                            batch_rows.append(row)

                        self.csv_writer.writerows(batch_rows)
                    except (ValueError, AttributeError):
                        pass  # 檔案已關閉或 writer 已清除，靜默忽略

            # 選擇通道供預測
            sample_pred = self.filter_channels(sample)

            with global_state.buffer_lock_ori_data:
                global_state.eeg_buffer_ori_data = np.hstack((global_state.eeg_buffer_ori_data, sample))
                if global_state.eeg_buffer_ori_data.shape[1] > config.BUFFER_SIZE:
                    global_state.eeg_buffer_ori_data = global_state.eeg_buffer_ori_data[:, -config.BUFFER_SIZE:]

            with global_state.buffer_lock:
                global_state.eeg_buffer = np.hstack((global_state.eeg_buffer, sample_pred))
                if global_state.eeg_buffer.shape[1] > config.BUFFER_SIZE:
                    global_state.eeg_buffer = global_state.eeg_buffer[:, -config.BUFFER_SIZE:]


# ─── Debug ───
if __name__ == "__main__":
    print("=== eeg.eeg_reader standalone test ===")
    reader = EEGReader(inlet=None)
    print(f"is_simulated = {reader.is_simulated}")
    print(f"save_csv     = {reader.save_csv}")
    print("✅ eeg_reader OK (no thread started)")
