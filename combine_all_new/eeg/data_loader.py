"""
EEG 資料載入器（CSV → PT → sliding window → Dataset）。

對應原 EEG_DL/EEG/data_process_np.py。
"""
import numpy as np
import datetime
import re
import time
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
import random

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import config
import global_state
import eeg.preprocess as pps

from rl.bandit import BanditAction


class EEGDataLoader_v2:
    def __init__(self, valid_ratio=0.25):
        self.csv_file_paths = []
        self.pt_file_paths = []

        self.csv_dir_path = os.path.join(global_state.REALTIME_BASE_FILE, "**", "*.csv")
        self.pt_dir_path = os.path.join(global_state.REALTIME_BASE_FILE, "**", "*.pt")

        self.valid_ratio = valid_ratio

        self.trail_size = 1500  # 3s @ 500Hz
        self.segment_len = 500
        self.train_stride = 20
        self.valid_stride = 50
        self.fs = config.SAMPLE_RATE

    def update_dir_path(self):
        self.csv_dir_path = os.path.join(global_state.REALTIME_BASE_FILE, "**", "*.csv")
        self.pt_dir_path = os.path.join(global_state.REALTIME_BASE_FILE, "**", "*.pt")

    def update_csv_file_paths(self, mode=0):
        self.csv_file_paths = []
        csv_file_paths = glob.glob(self.csv_dir_path, recursive=True)
        csv_file_paths.sort()
        if mode == 0:
            self.csv_file_paths = csv_file_paths
        else:
            self.csv_file_paths.append(csv_file_paths[-1])

    def update_pt_file_paths(self, mode=0):
        self.pt_file_paths = []
        pt_file_paths = glob.glob(self.pt_dir_path, recursive=True)
        pt_file_paths.sort()
        if mode == 0:
            self.pt_file_paths = pt_file_paths
        else:
            self.pt_file_paths.append(pt_file_paths[-1])

    # ─── CSV → numpy ───
    def _csv_to_np(self, csv_path):
        DATA_descriptions = pd.read_csv(csv_path, nrows=10, on_bad_lines='skip', encoding="utf-8")
        DATA_pd = pd.read_csv(csv_path, skiprows=10, encoding="utf-8")

        DATA_np_x = DATA_pd[config.choose_channels_names].to_numpy()
        DATA_np_y = DATA_pd['Beatmap_Marker'].to_numpy()
        # probs 已不從 CSV 讀取，將由 intermission 前的 rerun_inference 統一覆蓋
        DATA_np_prob = np.zeros((len(DATA_np_x), config.N_Class), dtype=np.float32)

        return {
            'DATA_np_x': DATA_np_x,
            'DATA_np_y': DATA_np_y,
            'DATA_np_prob': DATA_np_prob,
        }

    # ─── Cut trails ───
    def _cut_trail(self, DATA_np_dict):
        DATA_np_x = DATA_np_dict['DATA_np_x']
        DATA_np_y = DATA_np_dict['DATA_np_y']
        DATA_np_prob = DATA_np_dict['DATA_np_prob']

        change_points = np.where(np.diff(DATA_np_y.astype(int)) != 0)[0] + 1
        split_indices = np.concatenate((change_points, [len(DATA_np_y)]))

        X_record, y_record, prob_record = [], [], []

        for i in range(len(split_indices) - 1):
            ed = split_indices[i]
            st = ed - self.trail_size

            if DATA_np_y[ed - 1] in config.choose_marker:
                if st >= 0:
                    X_record.append(DATA_np_x[st:ed].T)
                    y_record.append(DATA_np_y[ed - 1])
                    prob_record.append(DATA_np_prob[ed - 1])
                else:
                    print(f"[Debug] 略過長度不足的 trial: ed={ed}, st={st}")

        return {
            'DATA_np_x': np.stack(X_record, axis=0),
            'DATA_np_y': np.stack(y_record, axis=0),
            'DATA_np_prob': np.stack(prob_record, axis=0),
        }

    # ─── Save as PT ───
    def _save_as_pt(self, DATA_np_dict, save_path):
        x_tensor = torch.from_numpy(DATA_np_dict['DATA_np_x']).float()
        y_tensor = torch.from_numpy(DATA_np_dict['DATA_np_y']).long()
        prob_tensor = torch.from_numpy(DATA_np_dict['DATA_np_prob']).float()

        save_dict = {
            'x_data': x_tensor,
            'y_data': y_tensor,
            'probs': prob_tensor,
            'metadata': {'fs': config.SAMPLE_RATE, 'channels': config.CHOOSE_CHANNELS}
        }

        torch.save(save_dict, save_path)
        print(f"檔案已成功儲存至: {save_path}")

    # ─── CSV → PT pipeline ───
    def _load_data_from_csv_to_pt(self, csv_path):
        pt_path = csv_path[:-4] + ".pt"
        if os.path.exists(pt_path):
            return

        DATA_np_dict = self._csv_to_np(csv_path)
        DATA_np_dict = self._cut_trail(DATA_np_dict)
        self._save_as_pt(DATA_np_dict, pt_path)

    # ─── Train/Valid split ───
    def _train_valid_idx(self, x_np_data, y_np_data):
        indices = np.arange(len(x_np_data))
        train_idx, valid_idx = train_test_split(
            indices,
            test_size=self.valid_ratio,
            stratify=y_np_data,
            random_state=42
        )
        return set(valid_idx)

    # ─── Sliding window ───
    def _sliding_window(self, stride, segment):
        if segment.shape[0] < self.segment_len:
            return None

        window_list = []
        for start in range(0, segment.shape[0] - self.segment_len + 1, stride):
            data = segment[start:start + self.segment_len]
            demeaned_data = pps.each_channel_demean(data, axis=0)
            window = pps.bandpass(demeaned_data)
            window_list.append(window)

        return window_list

    # ─── To TensorDataset ───
    def _to_dataset(self, segment_list, label_list):
        data_x = np.transpose(np.stack(segment_list), (0, 2, 1))
        X = torch.tensor(data_x).unsqueeze(1)
        y = F.one_hot(torch.tensor(label_list).long(), num_classes=config.N_Class)
        return TensorDataset(X, y)

    # ─── 標準 Dataset（train 用） ───
    def prepare_datasets_v2(self, data_x_list, data_y_list):
        print(f"valid_ratio: {self.valid_ratio}")
        if self.valid_ratio <= 0:
            print("[WARNING] self.valid_ratio should > 0")
            return False

        aug_train_seg, aug_train_lab = [], []
        aug_valid_seg, aug_valid_lab = [], []

        for idx in range(len(data_x_list)):
            x_np = np.transpose(data_x_list[idx], (0, 2, 1))
            y_np = data_y_list[idx]

            valid_idx_set = self._train_valid_idx(x_np, y_np)

            for i, segment in enumerate(x_np):
                if segment.shape[0] < self.segment_len:
                    continue

                window_list = self._sliding_window(self.train_stride, segment)
                label_list = [y_np[i]] * len(window_list)

                if window_list:
                    if i in valid_idx_set:
                        aug_valid_seg.extend(window_list)
                        aug_valid_lab.extend(label_list)
                    else:
                        aug_train_seg.extend(window_list)
                        aug_train_lab.extend(label_list)

        dataset = self._to_dataset(aug_train_seg, aug_train_lab)
        dataset_valid = self._to_dataset(aug_valid_seg, aug_valid_lab)
        return dataset, dataset_valid

    # ─── RL Dataset（calib 用） ───
    def prepare_datasets_RL_v1(self, data_x_list, data_y_list, probs_list,
                                RL_interface, session_records):
        print(f"valid_ratio: {self.valid_ratio}")
        if self.valid_ratio <= 0:
            print("self.valid_ratio should > 0")
            return False

        aug_train_seg, aug_train_lab = [], []
        aug_valid_seg, aug_valid_lab = [], []

        for idx in range(len(data_x_list)):
            x_np = np.transpose(data_x_list[idx], (0, 2, 1))
            y_np = data_y_list[idx]
            probs_data = probs_list[idx]

            valid_idx_set = self._train_valid_idx(x_np, y_np)

            for i, segment in enumerate(x_np):
                # Valid 不走 RL
                if i in valid_idx_set:
                    window_list = self._sliding_window(self.valid_stride, segment)
                    label_list = [y_np[i]] * len(window_list)
                    if window_list:
                        aug_valid_seg.extend(window_list)
                        aug_valid_lab.extend(label_list)
                    continue

                # Train 走 RL
                hit_data = {'eeg': segment.T, 'probs': probs_data[i]}

                hit_context_raw = RL_interface.extract_per_hit_context(hit_data)
                norm_context = RL_interface.normalizer.update_and_normalize(hit_context_raw)
                action = RL_interface.bandit.select_action(norm_context)

                record = {'context': norm_context, 'action': action}
                session_records.append(record)

                shouldTriggerUpdate = RL_interface.updateScheduler.record(action)
                print(f"[DataProcess] Hit#{i}: action={action.name}, shouldTriggerUpdate={shouldTriggerUpdate}")

                # 根據 action 調整 stride（所有 action 皆納入訓練）
                current_stride = self.train_stride
                if action == BanditAction.HIGH_WEIGHT:
                    current_stride = max(1, self.train_stride // 2)
                elif action == BanditAction.LOW_WEIGHT:
                    current_stride = self.train_stride * 2

                window_list = self._sliding_window(current_stride, segment)
                label_list = [y_np[i]] * len(window_list)

                if window_list:
                    aug_train_seg.extend(window_list)
                    aug_train_lab.extend(label_list)

        dataset = self._to_dataset(aug_train_seg, aug_train_lab)
        dataset_valid = self._to_dataset(aug_valid_seg, aug_valid_lab)
        return dataset, dataset_valid, session_records

    # ─── RL Loop 蒐集：只做 Bandit 決策，不做 sliding window ───
    def collect_rl_decisions(self, data_x_list, data_y_list, probs_list, RL_interface):
        """
        純蒐集模式：對每筆 trial 執行 Bandit 決策和 UpdateScheduler 累積，
        但不做 sliding window / dataset 建構（不訓練 DL）。

        Returns:
            session_records:    list of {'context', 'action', 'segment', 'label'}
            shouldUpdate:       UpdateScheduler 是否已觸發
        """
        session_records = []
        shouldUpdate = False

        for idx in range(len(data_x_list)):
            x_np = np.transpose(data_x_list[idx], (0, 2, 1))  # (N, samples, channels)
            y_np = data_y_list[idx]
            probs_data = probs_list[idx]

            valid_idx_set = self._train_valid_idx(x_np, y_np)

            for i, segment in enumerate(x_np):
                if i in valid_idx_set:
                    continue  # Valid 不參與 RL 決策

                if segment.shape[0] < self.segment_len:
                    continue

                # 提取 context → Bandit 選 action
                hit_data = {'eeg': segment.T, 'probs': probs_data[i]}
                hit_context_raw = RL_interface.extract_per_hit_context(hit_data)
                norm_context = RL_interface.normalizer.update_and_normalize(hit_context_raw)
                action = RL_interface.bandit.select_action(norm_context)

                # UpdateScheduler 累積
                updateResult = RL_interface.updateScheduler.record(action)
                if updateResult:
                    shouldUpdate = True

                record = {
                    'context': norm_context,
                    'action': action,
                    'segment': segment,  # shape: (samples, channels)
                    'label': int(y_np[i]),
                }
                session_records.append(record)

                print(f"[DataLoader] collect_rl: Hit#{i}: action={action.name}, "
                      f"shouldUpdate={shouldUpdate}")

        print(f"[DataLoader] collect_rl 完成: {len(session_records)} 筆記錄, shouldUpdate={shouldUpdate}")
        return session_records, shouldUpdate

    # ─── RL v2 Dataset（含 replay 資料注入）───
    def prepare_datasets_RL_v2(self, session_records, replay_segments, replay_labels):
        """
        從 session_records（已做完 Bandit 決策）+ replay 資料建構 dataset。

        **Trial-level split**：先在 trial 層級做 train/valid 分割，
        再對各自做 sliding window，避免同一 trial 的高度重疊 windows
        同時出現在 train 和 valid 中（防止 data leak）。

        Args:
            session_records:  collect_rl_decisions 回傳的記錄
            replay_segments:  ExperienceReplayMemory.sample() 回傳的 segments
            replay_labels:    ExperienceReplayMemory.sample() 回傳的 labels

        Returns:
            dataset:           TensorDataset (train)
            dataset_valid:     TensorDataset (valid) — 從新資料中分出
            new_train_segments: 被保留的新資料 segments（供存入 Dynamic_Buffer）
            new_train_labels:   對應的 labels
            new_train_actions:  對應的 actions
        """
        aug_train_seg, aug_train_lab = [], []
        aug_valid_seg, aug_valid_lab = [], []
        new_train_segments = []
        new_train_labels = []
        new_train_actions = []

        total_trials = len(session_records)
        if total_trials == 0:
            return None, None, [], [], []

        # ── 1. Trial-level train/valid split（防止 data leak）──
        trial_labels = np.array([rec['label'] for rec in session_records])
        trial_indices = np.arange(total_trials)

        # 嘗試 stratified split；若某 class 樣本不足則 fallback 到 random split
        valid_trial_count = max(1, int(total_trials * self.valid_ratio))
        if valid_trial_count >= total_trials:
            valid_trial_count = max(1, total_trials // 4)

        try:
            from sklearn.model_selection import train_test_split
            _, valid_indices = train_test_split(
                trial_indices, test_size=valid_trial_count,
                stratify=trial_labels, random_state=42
            )
            valid_trial_set = set(valid_indices)
        except ValueError:
            # stratify 失敗（某 class 數量太少），改用 random split
            np.random.seed(42)
            shuffled = trial_indices.copy()
            np.random.shuffle(shuffled)
            valid_trial_set = set(shuffled[:valid_trial_count])

        print(f"[DataLoader] prepare_datasets_RL_v2: trial-level split: "
              f"total_trials={total_trials}, valid_trials={len(valid_trial_set)}, "
              f"train_trials={total_trials - len(valid_trial_set)}")

        # ── 2. 對 train / valid trials 分別做 sliding window ──
        for idx, rec in enumerate(session_records):
            action = rec['action']
            segment = rec['segment']
            label = rec['label']

            if idx in valid_trial_set:
                # Valid trials：使用較大的 valid_stride 產生更獨立的評估樣本
                window_list = self._sliding_window(self.valid_stride, segment)
                if window_list:
                    aug_valid_seg.extend(window_list)
                    aug_valid_lab.extend([label] * len(window_list))
            else:
                # Train trials：根據 action 調整 stride
                current_stride = self.train_stride
                if action == BanditAction.HIGH_WEIGHT:
                    current_stride = max(1, self.train_stride // 2)
                elif action == BanditAction.LOW_WEIGHT:
                    current_stride = self.train_stride * 2

                window_list = self._sliding_window(current_stride, segment)
                if window_list:
                    aug_train_seg.extend(window_list)
                    aug_train_lab.extend([label] * len(window_list))

            # 記錄被保留的新資料（供 Dynamic_Buffer，所有 trial 皆記錄）
            new_train_segments.append(segment)
            new_train_labels.append(label)
            new_train_actions.append(action)

        # ── 3. 從 replay 資料做 sliding window，只加入訓練集 ──
        for seg, lab in zip(replay_segments, replay_labels):
            if seg.shape[0] < self.segment_len:
                continue
            window_list = self._sliding_window(self.train_stride, seg)
            if window_list:
                aug_train_seg.extend(window_list)
                aug_train_lab.extend([lab] * len(window_list))

        if len(aug_train_seg) < 2 or len(aug_valid_seg) < 2:
            print(f"[DataLoader] prepare_datasets_RL_v2: 資料不足 "
                  f"(train={len(aug_train_seg)}, valid={len(aug_valid_seg)})")
            return None, None, new_train_segments, new_train_labels, new_train_actions

        dataset = self._to_dataset(aug_train_seg, aug_train_lab)
        dataset_valid = self._to_dataset(aug_valid_seg, aug_valid_lab)

        print(f"[DataLoader] prepare_datasets_RL_v2: train={len(aug_train_seg)}, "
              f"valid={len(aug_valid_seg)}, replay={len(replay_segments)}, "
              f"new_hq={len(new_train_segments)}")

        return dataset, dataset_valid, new_train_segments, new_train_labels, new_train_actions

    # ─── model_interface 呼叫入口 ───
    def setup_the_data_path(self, mode=0):
        self.update_dir_path()
        self.update_csv_file_paths(mode)
        print(f"{config.TAGS.INFO.value} csv_file_paths: {self.csv_file_paths}")

        for csv_path in self.csv_file_paths:
            self._load_data_from_csv_to_pt(csv_path)

        self.update_pt_file_paths(mode)
        print(f"{config.TAGS.INFO.value} pt_file_paths: {self.pt_file_paths}")

    """
    def load_data_and_prepare_dataset(self, RL_open=False, RL_interface=None):
        data_x_list, data_y_list, probs_list = [], [], []
        print(f"RL_open: {RL_open}")

        for name in self.pt_file_paths:
            print(f"name {name}")
            train_data = torch.load(name, map_location='cpu')
            data_x_list.append(train_data['x_data'].numpy())
            data_y_list.append(train_data['y_data'].numpy())
            probs_list.append(train_data['probs'].numpy())

        session_records = []

        if not RL_open:
            dataset, dataset_val = self.prepare_datasets_v2(data_x_list, data_y_list)
        else:
            dataset, dataset_val, session_records = self.prepare_datasets_RL_v1(
                data_x_list, data_y_list, probs_list, RL_interface, session_records)

        return dataset, dataset_val, session_records
    """
    


    def load_data_and_prepare_dataset(self, RL_open=False, RL_interface=None, replay_segments=None, replay_labels=None):
        # 預設 Replay 資料為空陣列，相容尚未實作 Experience Replay 的呼叫端
        if replay_segments is None:
            replay_segments = []
        if replay_labels is None:
            replay_labels = []

        data_x_list, data_y_list, probs_list = [], [], []
        print(f"RL_open: {RL_open}")

        for name in self.pt_file_paths:
            print(f"name {name}")
            train_data = torch.load(name, map_location='cpu')
            data_x_list.append(train_data['x_data'].numpy())
            data_y_list.append(train_data['y_data'].numpy())
            probs_list.append(train_data['probs'].numpy())

        session_records = []

        if not RL_open:
            dataset, dataset_val = self.prepare_datasets_v2(data_x_list, data_y_list)
        else:
            # 1. 呼叫純蒐集模式，取得 Bandit 決策與 context
            session_records, shouldUpdate = self.collect_rl_decisions(
                data_x_list, data_y_list, probs_list, RL_interface
            )
            
            # 2. 呼叫 v2，結合決策結果與 replay 資料建構 DL 需要的 Dataset
            dataset, dataset_val, new_train_seg, new_train_lab, new_train_act = self.prepare_datasets_RL_v2(
                session_records, replay_segments, replay_labels
            )

            # 備註：若外部（如 train.py 或 model_interface）需要 new_train_seg 來存入 Dynamic Buffer，
            # 可以考慮更改此函式的 return 數量。目前維持回傳 3 個變數以確保向下相容。

        return dataset, dataset_val, session_records
    

# ─── Debug ───
if __name__ == "__main__":
    print("=== eeg.data_loader standalone test ===")
    loader = EEGDataLoader_v2(valid_ratio=0.25)
    print(f"trail_size   = {loader.trail_size}")
    print(f"segment_len  = {loader.segment_len}")
    print(f"train_stride = {loader.train_stride}")
    print("✅ data_loader OK")
