"""
Model Interface — EEG 模型管理門面。

對應原 EEG_DL/EEG/model_interface.py。
提供 start/end read/predict、train_or_calib、model select 等統一介面。
"""
import time
import os
import glob
import json
import pylsl
import torch
import torch.nn.functional as F
from pathlib import Path

from eeg.data_loader import EEGDataLoader_v2
from eeg.eeg_reader import EEGReader
from eeg.eeg_trainer import EEGTrainingPipeline
from eeg.eeg_predictor import EEGPredictor

import config
import global_state
from eeg.tcp_utils import TCPClient


class model_interface:
    def __init__(self, inlet=None, valid_ratio=0.25):
        self.eeg_reader = EEGReader(inlet)
        self.eeg_reader.start_read_eeg_thread()

        self.version = 0
        self.active_model_name = config.base_model_name[:-4]

        self.tcp_client         = TCPClient(host=config.TCP_CLIENT, port=config.TCP_PORT)
        self.eeg_predictor      = EEGPredictor(tcp_client=self.tcp_client)
        self.eeg_dataloader     = EEGDataLoader_v2(valid_ratio=valid_ratio)

        # Train pipeline
        self.train_pipeline = EEGTrainingPipeline(epochs=60, freeze_layers=False, ft=False, tcp_client=self.tcp_client)
        # Fine-tune pipeline
        self.ft_pipeline    = EEGTrainingPipeline(epochs=10, freeze_layers=True, ft=True, tcp_client=self.tcp_client)
        # RL fine-tune pipeline
        self.rl_ft_pipeline = EEGTrainingPipeline(epochs=5, freeze_layers=True, ft=True, tcp_client=self.tcp_client)

        self.model_list = self.get_model_list()

        self.last_acc       = 0.0
        self.lastValLoss    = float('inf')

        # RL context 用
        self.delta_acc      = 0.0
        self.deltaValLoss   = 0.0

        # RL loop 用
        self.RL_loop_delta_acc                      = 0.0
        self.rlLoopDeltaValLoss                     = 0.0
        self.RL_loop_highest_valid_acc              = 0.0
        self.RL_loop_highest_valid_acc_model_name   = ""

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
        """回傳目前最新的 eeg 樣本"""
        with global_state.buffer_lock_ori_data:
            return global_state.eeg_buffer_ori_data[:, -1] if global_state.eeg_buffer_ori_data.size > 0 else None

    def write_csv(self, row):
        if self.eeg_reader.csv_writer:
            self.eeg_reader.csv_writer.writerow(row)

    def write_csv_batch(self, rows):
        try:
            if self.eeg_reader.csv_writer:
                self.eeg_reader.csv_writer.writerows(rows)
        except Exception as e:
            print(f"Batch write failed: {e}")

    def train_or_calib(self, mode, RL_open, RL_interface, RL_LOOP_USE=False, update_times=None):
        """
        mode: 0 -> train, 1 -> calib
        """
        if mode not in [0, 1]:
            print(f"incorrect mode num {mode}")
            return False

        # 資料處理
        if not RL_LOOP_USE:
            print("Song end process data start")
            self.eeg_dataloader.setup_the_data_path(mode)

            # CSV 已轉為 PT，用最新 DL 模型重跑所有 .pt 的 inference
            # （包含剛轉完的最新一首歌的 PT）
            self.rerun_inference_on_pt_files()
        else:
            print("RL Loop process data start")
            print(f"[INFO] pt_file_paths: {self.eeg_dataloader.pt_file_paths}")

        dataset, dataset_val, session_records = self.eeg_dataloader.load_data_and_prepare_dataset(
            RL_open, RL_interface)

        if not RL_LOOP_USE:
            print("process data finish")
            self.version += 1

            if mode == 0:
                new_model_name = f"Trained_Model_v{self.version:03d}"
            else:
                new_model_name = f"Calibed_Model_v{self.version:03d}"

            save_dir = os.path.join(global_state.save_dir_model, new_model_name)
            os.makedirs(save_dir, exist_ok=True)
            print(f"save_dir {save_dir}")
            global_state.EEG_CHECKPOINT_TMP_BASE_DIR = save_dir

            save_path = os.path.join(global_state.save_dir_model, "base_model", new_model_name + ".pth")
            if mode == 0:
                val_acc, val_loss = self.train_pipeline.run_training(dataset, dataset_val, save_path)
            else:
                val_acc, val_loss = self.ft_pipeline.run_training(dataset, dataset_val, save_path)

            self.delta_acc = val_acc - self.last_acc
            self.last_acc = val_acc

            self.deltaValLoss = val_loss - self.lastValLoss
            self.lastValLoss = val_loss
            print(f"[DEBUG] delta_acc={self.delta_acc:.4f}, deltaValLoss={self.deltaValLoss:.4f}")

            time.sleep(1)

            self.active_model_name = new_model_name
            global_state.NOW_FT_CHECKPOINT = os.path.join(
                global_state.EEG_CHECKPOINT_BASE_DIR, new_model_name)

            print(f"[Backend] 訓練完成！新模型: {new_model_name}")

            # 更新 version 紀錄
            with open(global_state.json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            if mode == 0:
                json_data["train_model"] += 1
            else:
                json_data["calib_model"] += 1

            with open(global_state.json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

        else:
            # RL Loop 用
            new_model_name = f"RL_Loop_Model_v{self.version:03d}_v{update_times}"

            save_dir = os.path.join(global_state.save_dir_model, new_model_name)
            os.makedirs(save_dir, exist_ok=True)
            global_state.EEG_CHECKPOINT_TMP_BASE_DIR = save_dir

            RL_loop_model_dir = os.path.join(global_state.save_dir_model, "RL_loop_model")
            os.makedirs(RL_loop_model_dir, exist_ok=True)

            save_path = os.path.join(RL_loop_model_dir, new_model_name + ".pth")
            if mode == 0:
                val_acc, val_loss = self.train_pipeline.run_training(dataset, dataset_val, save_path)
            else:
                val_acc, val_loss = self.ft_pipeline.run_training(dataset, dataset_val, save_path)

            self.RL_loop_delta_acc = val_acc - self.last_acc
            self.rlLoopDeltaValLoss = val_loss - self.lastValLoss
            print(f"[DEBUG] RL_loop_delta_acc={self.RL_loop_delta_acc:.4f}, "
                  f"rlLoopDeltaValLoss={self.rlLoopDeltaValLoss:.4f}")

            if val_acc > self.RL_loop_highest_valid_acc:
                self.RL_loop_highest_valid_acc = val_acc
                self.RL_loop_highest_valid_acc_model_name = new_model_name

            time.sleep(1)

        return session_records

    # ─── RL Loop Phase 1：純蒐集（不訓練 DL） ───
    def collect_rl_data(self, RL_interface):
        """
        RL Loop 用：只做 Bandit 決策 + UpdateScheduler 累積，不做 DL 訓練。
        歌曲進行中，DL 維持 DL_n 不動。

        Returns:
            session_records:  list of {'context', 'action', 'segment', 'label'}
            shouldUpdate:     UpdateScheduler 是否已觸發（累積分數 >= threshold）
        """
        print(f"[model_interface] collect_rl_data: 開始蒐集")
        print(f"[model_interface] pt_file_paths: {self.eeg_dataloader.pt_file_paths}")

        data_x_list, data_y_list, probs_list = [], [], []
        for name in self.eeg_dataloader.pt_file_paths:
            train_data = torch.load(name, map_location='cpu')
            data_x_list.append(train_data['x_data'].numpy())
            data_y_list.append(train_data['y_data'].numpy())
            probs_list.append(train_data['probs'].numpy())

        session_records, shouldUpdate = self.eeg_dataloader.collect_rl_decisions(
            data_x_list, data_y_list, probs_list, RL_interface
        )

        print(f"[model_interface] collect_rl_data 完成: "
              f"records={len(session_records)}, shouldUpdate={shouldUpdate}")
        return session_records, shouldUpdate

    # ─── RL Loop Phase 2：混合訓練 DL_n → DL_{n+1} ───
    def train_rl_loop_update(self, session_records, replay_segments, replay_labels,
                              update_times):
        """
        RL Loop 觸發更新時呼叫：將新資料 + replay 資料混合訓練 DL。

        Args:
            session_records:  collect_rl_data 回傳的記錄
            replay_segments:  ExperienceReplayMemory.sample() 的 segments
            replay_labels:    ExperienceReplayMemory.sample() 的 labels
            update_times:     目前的更新次數（用於命名）

        Returns:
            val_acc:             驗證準確度
            val_loss:            驗證 loss
            new_checkpoint_path: 新模型的 .pth 路徑
            new_train_segments:  被保留的新高品質 segments（供 Dynamic_Buffer）
            new_train_labels:    對應 labels
            new_train_actions:   對應 actions
        """
        print(f"[model_interface] train_rl_loop_update: 開始混合訓練 (update #{update_times})")

        # 使用 prepare_datasets_RL_v2 建構含 replay 的 dataset
        result = self.eeg_dataloader.prepare_datasets_RL_v2(
            session_records, replay_segments, replay_labels
        )
        dataset, dataset_val, new_train_segments, new_train_labels, new_train_actions = result

        if dataset is None or dataset_val is None:
            print(f"[model_interface] train_rl_loop_update: 資料不足，跳過訓練")
            return None, None, None, new_train_segments, new_train_labels, new_train_actions

        # 建構模型名稱和路徑
        new_model_name  = f"RL_Loop_Model_v{self.version:03d}_u{update_times}"

        save_dir        = os.path.join(global_state.save_dir_model, new_model_name)
        os.makedirs(save_dir, exist_ok=True)
        global_state.EEG_CHECKPOINT_TMP_BASE_DIR = save_dir

        rl_loop_model_dir = os.path.join(global_state.save_dir_model, "RL_loop_model")
        os.makedirs(rl_loop_model_dir, exist_ok=True)

        save_path = os.path.join(rl_loop_model_dir, new_model_name + ".pth")

        # 使用 rl_ft_pipeline 進行 fine-tune
        val_acc, val_loss = self.rl_ft_pipeline.run_training(dataset, dataset_val, save_path)

        # 更新 delta 值
        self.RL_loop_delta_acc = val_acc - self.last_acc
        self.rlLoopDeltaValLoss = val_loss - self.lastValLoss
        print(f"[model_interface] train_rl_loop_update: "
              f"val_acc={val_acc:.4f}, val_loss={val_loss:.4f}, "
              f"RL_loop_delta_acc={self.RL_loop_delta_acc:.4f}, "
              f"rlLoopDeltaValLoss={self.rlLoopDeltaValLoss:.4f}")

        if val_acc > self.RL_loop_highest_valid_acc:
            self.RL_loop_highest_valid_acc = val_acc
            self.RL_loop_highest_valid_acc_model_name = new_model_name
            print(f"[model_interface] 新最高 valid acc: {val_acc:.4f} ({new_model_name})")

        return val_acc, val_loss, save_path, new_train_segments, new_train_labels, new_train_actions

    # ─── Intermission 前：用最新 DL 模型重跑所有 .pt 的 inference ───
    def rerun_inference_on_pt_files(self):
        """
        遊玩結束後、進入 intermission 前呼叫。

        用當前 predictor 中已載入的最新 DL 模型，對
        eeg_dataloader.pt_file_paths 內所有 .pt 檔案的 x_data
        做 batch inference，將 softmax 機率寫回 .pt 的 'probs' 欄位。

        效果：
          - intermission 的 calib / collect_rl_data 都能拿到最新 DL probs
          - 不需要在遊玩期間即時記錄 probs（舊 probs 完全由此覆蓋）
        """
        self.eeg_dataloader.update_dir_path()
        self.eeg_dataloader.update_pt_file_paths(mode=0)   # 載入所有 .pt

        pt_paths = self.eeg_dataloader.pt_file_paths
        if not pt_paths:
            print("[model_interface] rerun_inference: 找不到 .pt 檔案，跳過")
            return

        model = self.eeg_predictor.model
        device = self.eeg_predictor.device

        if model is None:
            print("[model_interface] rerun_inference: predictor 模型尚未載入，跳過")
            return

        model.eval()
        total_trials = 0

        with self.eeg_predictor.modelLock:
            for pt_path in pt_paths:
                try:
                    saved = torch.load(pt_path, map_location='cpu')
                    x_data = saved['x_data']          # (N, C, T)  float32

                    # 模型訓練時使用 segment_len=500 (1s)，
                    # 但 .pt 存的是 trail_size=1500 (3s)，需截取最後 500 samples
                    T = x_data.shape[-1]
                    if T > config.SAMPLE_RATE:
                        x_data = x_data[:, :, -config.SAMPLE_RATE:]

                    # batch inference
                    with torch.no_grad():
                        # EEGNet / ShallowFBCSP 需要 (N, 1, C, T) 或 (N, C, T)
                        # predictor 在 predict_loop 中使用 unsqueeze(0)，
                        # 這裡對整個 batch 統一 unsqueeze(1)
                        x_batch = x_data.unsqueeze(1).to(device).float()
                        outputs = model(x_batch)          # (N, n_classes)
                        probs   = F.softmax(outputs, dim=1).cpu()  # (N, n_classes)

                    saved['probs'] = probs
                    torch.save(saved, pt_path)
                    total_trials += len(x_data)
                    print(f"[model_interface] rerun_inference: "
                          f"{os.path.basename(pt_path)} → {len(x_data)} trials 已更新")

                except Exception as e:
                    print(f"[model_interface] rerun_inference: "
                          f"{pt_path} 失敗 ({e})，跳過")

        print(f"[model_interface] rerun_inference 完成，共更新 {total_trials} 筆 trial")

    def get_model_list(self):
        model_list = glob.glob(
            os.path.join(global_state.EEG_CHECKPOINT_BASE_DIR, "**", "*.pth"),
            recursive=True)
        model_list.sort()
        self.model_list = [Path(p).stem for p in model_list]
        return self.model_list

    def set_active_model(self, idx):
        if 0 <= idx < len(self.model_list):
            self.active_model_name = self.model_list[idx]
            global_state.NOW_FT_CHECKPOINT = os.path.join(
                global_state.EEG_CHECKPOINT_BASE_DIR, self.model_list[idx])


# ─── Debug ───
if __name__ == "__main__":
    print("=== eeg.model_interface standalone test ===")
    print("model_interface requires LSL/TCP to initialize fully.")
    print("✅ model_interface import OK")
