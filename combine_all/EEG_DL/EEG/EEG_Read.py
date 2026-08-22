import time
import csv
import os
import threading
import numpy as np
from datetime import datetime
import pylsl




import EEG_DL.Utils.config as config
import EEG_DL.Utils.global_value as global_value
from EEG_DL.Utils.some_functions import rename_file_with_time



class EEGReader():
    def __init__(self, inlet=None):

        self.inlet = inlet

        self.is_simulated = config.is_simulated_eeg

    
        self.save_csv = config.SAVE_CSV
        self.filename = global_value.CSV_FILENAME
        
        self.csv_writer = None
        self.csv_file = None
        self.read_eeg_stop_event = threading.Event()
        self.read_eeg_thread = None

    def start_read_eeg_thread(self):

        # 避免重啟
        if self.read_eeg_thread and self.read_eeg_thread.is_alive():
            print(f"{config.TAGS.WARNING.value} Marker thread is already running.")
            return  


        print(f"{config.TAGS.INFO.value} Start read cygnus eeg...")

        self.read_eeg_stop_event.clear()                                            # 重置 stop_event，允許重新啟動
        self.read_eeg_thread = threading.Thread(target=self.update_predict_buffer, daemon=True)  # 每次都重設定 thread
        self.read_eeg_thread.start()


    def end_read_eeg_thread(self):

        print(f"{config.TAGS.INFO.value} End read cygnus eeg...")
        self.read_eeg_stop_event.set()                                              # 設置停止事件
        if self.read_eeg_thread and self.read_eeg_thread.is_alive():
            self.read_eeg_thread.join()
            self.read_eeg_thread = None


    def build_csv_file(self):

        self.filename = rename_file_with_time(global_value.CSV_FILENAME)

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
            header = ["Timestamp"] + ["Beatmap_Marker"] + ["Predict_Marker"] + config.prob_classes + config.all_channels_names
            self.csv_writer.writerow(header)
            print("[CSV] Start write EEG data to", self.filename)


    def close_csv_file(self):
        if self.csv_file:
            try:
                self.csv_file.close()
                print("[CSV] Saved EEG data to", self.filename)
                
            except Exception:
                print("[CSV] Saved fail")
            finally:
                self.csv_file = None
                self.csv_writer = None


    def filter_channels(self, sample_arr):
        """將原始 32 通道過濾為特定的 13 通道"""
        # sample_arr shape: (32, N)
        np_s = np.array(sample_arr)
        filtered = np_s[config.choose_channel_idx, :]
        return filtered
    

    def update_predict_buffer(self):
        """背景執行緒：僅負責更新 global_value.eeg_buffer 供預測使用並負責高頻資料記錄"""
        while not self.read_eeg_stop_event.is_set():
            if self.is_simulated:
                # 模擬 500Hz 採樣
                sample = np.random.randint(-10001, 10001, size=(config.ALL_CHANNELS, 1))
                time.sleep(1 / config.SAMPLE_RATE)
                
                if self.csv_writer and config.SAVE_CSV:
                    marker = global_value.current_marker
                    pred = global_value.current_predict
                    prob = global_value.current_prob
                    record_time_s = pylsl.local_clock() - global_value.start_lsl_time
                    row = [f"{record_time_s:.3f}", marker, pred] + prob + sample.flatten().tolist()
                    self.write_csv(row)
                
            else:
                if self.inlet is None:
                    time.sleep(1)
                    continue

                # 抓取所有累積樣本更新預測用 Buffer
                samples, ts = self.inlet.pull_chunk(timeout=0.0)
                
                if not samples:
                    time.sleep(0.001)
                    continue
                
                # pylsl 取得的 samples 為 list of lists，需轉為二維 (N, 32) 再轉置成 (32, N)
                sample = np.array(samples).T
                
                # 紀錄資料
                if self.csv_writer and config.SAVE_CSV:
                    marker = global_value.current_marker
                    pred = global_value.current_predict
                    prob = global_value.current_prob
                    start_ts = global_value.start_lsl_time
                    
                    batch_rows = []
                    for i, s_pt in enumerate(samples):
                        # samples 中的每個 s_pt 是一個長度為 32 的 list
                        record_time_s = ts[i] - start_ts
                        row = [f"{record_time_s:.3f}", marker, pred] + prob + s_pt
                        batch_rows.append(row)
                        
                    self.csv_writer.writerows(batch_rows)
            

            # 根據你的需求挑選 13 通道
            sample_pred = self.filter_channels(sample)

            # print(f"sample {sample}")

            with global_value.buffer_lock_ori_data:
                global_value.eeg_buffer_ori_data = np.hstack((global_value.eeg_buffer_ori_data, sample))
                if global_value.eeg_buffer_ori_data.shape[1] > config.BUFFER_SIZE:
                    global_value.eeg_buffer_ori_data = global_value.eeg_buffer_ori_data[:, -config.BUFFER_SIZE:]


            with global_value.buffer_lock:
                global_value.eeg_buffer = np.hstack((global_value.eeg_buffer, sample_pred))
                if global_value.eeg_buffer.shape[1] > config.BUFFER_SIZE:
                    global_value.eeg_buffer = global_value.eeg_buffer[:, -config.BUFFER_SIZE:]