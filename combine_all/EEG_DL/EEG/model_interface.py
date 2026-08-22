import time
import os
import glob
import json
import pylsl
import torch
from pathlib import Path

from EEG_DL.EEG.data_process_np import EEGDataLoader_v2
from EEG_DL.EEG.EEG_Read import EEGReader
from EEG_DL.EEG.EEG_Train import EEGTrainingPipeline
from EEG_DL.EEG.EEG_Predict import EEGPredictor

import EEG_DL.Utils.config as config
import EEG_DL.Utils.global_value as global_value
from EEG_DL.Utils.TCP_use import TCPClient



class model_interface:
    # mode 0 模擬, other 實際開始
    def __init__(self, inlet=None, valid_ratio=0.25):
    
        self.eeg_reader = EEGReader(inlet)               # cygnus use lsl
        self.eeg_reader.start_read_eeg_thread()

        # self.version train, calib 是 共用的
        self.version = 0
        self.active_model_name = config.base_model_name[:-4]

        self.tcp_client = TCPClient(host=config.TCP_CLIENT, port=config.TCP_PORT)
        self.eeg_predictor = EEGPredictor(tcp_client=self.tcp_client)


        self.eeg_dataloader = EEGDataLoader_v2(valid_ratio=valid_ratio)
        

        # train pipeline: 取當前所有資料，直接 train 一個新 model
        self.train_pipeline = EEGTrainingPipeline(epochs=1, freeze_layers=False, ft=False, tcp_client=self.tcp_client)

        # ft pipeline: 只取最新一首歌，對當前使用 model 做 calib，產出新 model
        self.ft_pipeline = EEGTrainingPipeline(epochs=10, freeze_layers=True, ft=True, tcp_client=self.tcp_client)

        # rl ft pipeline: 同 ft pipeline，改 epochs 而已
        self.rl_ft_pipeline = EEGTrainingPipeline(epochs=3, freeze_layers=True, ft=True, tcp_client=self.tcp_client)


        self.model_list = self.get_model_list()

        self.last_acc = 0.0
        self.lastValLoss = float('inf')  # 上一次的 validation loss

        # RL context 用
        self.delta_acc = 0.0
        self.deltaValLoss = 0.0       # ΔLoss = current - last（負=進步）


        # RL loop 用
        self.RL_loop_delta_acc = 0.0
        self.rlLoopDeltaValLoss = 0.0  # RL loop 專用 delta_val_loss
        self.RL_loop_highest_valid_acc = 0.0
        self.RL_loop_highest_valid_acc_model_name = ""
        


    def version_setting(self, version):
        self.version = version

    
    def start_read_eeg(self):
        """開始遊戲時呼叫：準備 CSV 並啟動背景緩衝執行緒"""
        self.eeg_reader.build_csv_file()

    def start_pred_eeg(self):
        self.eeg_predictor.start_predict_eeg_thread()



    def end_pred_eeg(self):
        self.eeg_predictor.end_predict_eeg_thread()
        

    def end_read_eeg(self):
        """結束遊戲時呼叫"""
        self.eeg_reader.close_csv_file()


    def fetch_raw_eeg(self):
        """回傳目前最新的 eeg 樣本 (供主迴圈 record 使用)"""
        with global_value.buffer_lock_ori_data:
            return global_value.eeg_buffer_ori_data[:, -1] if global_value.eeg_buffer_ori_data.size > 0 else None


    def write_csv(self, row):
        # 寫入 csv_writer
        if self.eeg_reader.csv_writer:
            self.eeg_reader.csv_writer.writerow(row)

    def write_csv_batch(self, rows):
        """
        rows: 一個包含多行資料的 List
        """
        try:
            if self.eeg_reader.csv_writer:
                self.eeg_reader.csv_writer.writerows(rows) # 使用 writerows 一次寫入多行
        except Exception as e:
            print(f"Batch write failed: {e}")





    # @@@@@@@@@@@@@
    # main.py 的 run_intermission 用
    # @@@@@@@@@@@@@
    def train_or_calib(self, mode, RL_open, RL_interface, RL_LOOP_USE=False, update_times=None):

        """
        mode: int
        0 -> train
        1 -> calib
        """

        """
        model 相關路徑 

        save_dir_model = os.path.join("account", "Guest", "model")

        EEG_CHECKPOINT_BASE_DIR = os.path.join("account", "Guest", "model", "base_model")

        # 放 new check point "folder" ---> Calibed_Model_v001
        EEG_CHECKPOINT_TMP_BASE_DIR = os.path.join("account", "Guest", "model", "base_model", "Calibed_Model_v001")


        # 放要被 FT 的 model == now active model
        NOW_FT_CHECKPOINT = os.path.join("EEG_DL", "EEG", "base_model", "for_VR_4class")
        
        
        """
        if mode not in [0, 1]:
            print(f"incorrect mode num {mode}")
            return False

        # data 處理
        if not RL_LOOP_USE:
            # 只有 全部完整 玩完一首歌才會 print 訊息
            print("Song end process data start")

            # RL Loop 取上一次的最後一首歌，所以不 update path
            self.eeg_dataloader.setup_the_data_path(mode)
        
        else:
            print("RL Loop process data start")

            print(f"[INFO] pt_file_paths: {self.eeg_dataloader.pt_file_paths}")


        dataset, dataset_val, session_records = self.eeg_dataloader.load_data_and_prepare_dataset(RL_open, RL_interface)
        
        if not RL_LOOP_USE:
            print("process data finish")

            # model name 處理
            self.version += 1

            if mode == 0:
                new_model_name = f"Trained_Model_v{self.version:03d}"
            else:
                new_model_name = f"Calibed_Model_v{self.version:03d}"

            save_dir = os.path.join(global_value.save_dir_model, new_model_name)
            os.makedirs(save_dir, exist_ok=True)
            print(f"save_dir {save_dir}")
            global_value.EEG_CHECKPOINT_TMP_BASE_DIR = save_dir


            # train
            save_path = os.path.join(global_value.save_dir_model, "base_model", new_model_name+".pth")
            if mode == 0:
                val_acc, val_loss = self.train_pipeline.run_training(dataset, dataset_val, save_path)
            else:
                val_acc, val_loss = self.ft_pipeline.run_training(dataset, dataset_val, save_path)


            self.delta_acc = val_acc - self.last_acc
            self.last_acc = val_acc

            # 計算 delta_val_loss（平滑化 Reward 信號）
            self.deltaValLoss = val_loss - self.lastValLoss
            self.lastValLoss = val_loss
            print(f"[DEBUG] delta_acc={self.delta_acc:.4f}, deltaValLoss={self.deltaValLoss:.4f}")

            time.sleep(1)

            self.active_model_name = new_model_name
            global_value.NOW_FT_CHECKPOINT = os.path.join(global_value.EEG_CHECKPOINT_BASE_DIR, new_model_name)

            print(f"[Backend] 訓練完成！新模型: {new_model_name}")


            # 更新 version 紀錄
            # json_data 紀錄的是次數
            with open(global_value.json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            if mode == 0:
                json_data["train_model"] += 1
            else:
                json_data["calib_model"] += 1

            with open(global_value.json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

        
        else:
            # RL Loop 用，目前是一律用 calib，不會更新 global_value.NOW_FT_CHECKPOINT

            # 好像會有 multi thread 搶同一個 model 的 問題

            new_model_name = f"RL_Loop_Model_v{self.version:03d}_v{update_times}"

            save_dir = os.path.join(global_value.save_dir_model, new_model_name)
            os.makedirs(save_dir, exist_ok=True)
            
            global_value.EEG_CHECKPOINT_TMP_BASE_DIR = save_dir


            # train

            RL_loop_model_dir = os.path.join(global_value.save_dir_model, "RL_loop_model")
            os.makedirs(RL_loop_model_dir, exist_ok=True)
            
            save_path = os.path.join(RL_loop_model_dir, new_model_name+".pth")

            if mode == 0:
                val_acc, val_loss = self.train_pipeline.run_training(dataset, dataset_val, save_path)
            else:
                val_acc, val_loss = self.ft_pipeline.run_training(dataset, dataset_val, save_path)


            self.RL_loop_delta_acc = val_acc - self.last_acc

            # 計算 RL loop 專用 delta_val_loss
            self.rlLoopDeltaValLoss = val_loss - self.lastValLoss
            print(f"[DEBUG] RL_loop_delta_acc={self.RL_loop_delta_acc:.4f}, rlLoopDeltaValLoss={self.rlLoopDeltaValLoss:.4f}")

            if val_acc > self.RL_loop_highest_valid_acc:
                self.RL_loop_highest_valid_acc = val_acc
                self.RL_loop_highest_valid_acc_model_name = new_model_name

            

            time.sleep(1)

        
        return session_records
    

    def get_model_list(self):
        
        model_list = glob.glob(os.path.join(global_value.EEG_CHECKPOINT_BASE_DIR, "**", "*.pth"), recursive=True)

        # 排序確保順序固定
        model_list.sort()

        self.model_list = []

        for p in model_list:
            self.model_list.append(Path(p).stem)

        return self.model_list


    def set_active_model(self, idx):
        if 0 <= idx < len(self.model_list):
            self.active_model_name = self.model_list[idx]
            global_value.NOW_FT_CHECKPOINT = os.path.join(global_value.EEG_CHECKPOINT_BASE_DIR, self.model_list[idx])
