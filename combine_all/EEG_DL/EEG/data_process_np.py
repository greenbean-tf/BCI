import numpy as np
import datetime, re
import time
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset




import EEG_DL.Utils.config as config
import EEG_DL.Utils.global_value as global_value
import EEG_DL.Utils.preprocess as pps

from RL.bandit import BanditAction

class EEGDataLoader_v2:
    def __init__(self, valid_ratio=0.25):

        self.csv_file_paths = []
        self.pt_file_paths = []

        self.csv_dir_path = os.path.join(global_value.REALTIME_BASE_FILE, "**", "*.csv")
        self.pt_dir_path = os.path.join(global_value.REALTIME_BASE_FILE, "**", "*.pt")

        self.valid_ratio = valid_ratio
        
        

        self.trail_size = 1500 # 實際應該要是 1500 = 3s
        self.segment_len = 500

        self.train_stride = 20
        self.valid_stride = 50

        self.fs = config.SAMPLE_RATE



    def update_dir_path(self):
        self.csv_dir_path = os.path.join(global_value.REALTIME_BASE_FILE, "**", "*.csv")
        self.pt_dir_path = os.path.join(global_value.REALTIME_BASE_FILE, "**", "*.pt")
    
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




    def _csv_to_np(self, csv_path):
    
        ## for a songs
        DATA_descriptions = pd.read_csv(csv_path,  nrows=10, on_bad_lines='skip', encoding="utf-8")
        # print(DATA_descriptions)

        # csv to pd
        ## DATA_pd = (time, title)
        DATA_pd = pd.read_csv(csv_path, skiprows=10, encoding="utf-8")


        # pd_to_np
        ## DATA_np = (time, title)
        DATA_np_x = DATA_pd[config.choose_channels_names].to_numpy()
        DATA_np_y = DATA_pd['Beatmap_Marker'].to_numpy()
        DATA_np_prob = DATA_pd[config.prob_classes].to_numpy()

        DATA_np_dict = {
            'DATA_np_x': DATA_np_x,
            'DATA_np_y': DATA_np_y,
            'DATA_np_prob': DATA_np_prob,
        }

        return DATA_np_dict


    def _cut_trail(self, DATA_np_dict):
        """
        trail 長度取決於 marker 不為 -1 的長度

        input: DATA_np = (time, title)

        return  DATA_np = (trail, time, title)
        """

        DATA_np_x = DATA_np_dict['DATA_np_x']
        DATA_np_y = DATA_np_dict['DATA_np_y']
        DATA_np_prob = DATA_np_dict['DATA_np_prob']


        # change_points: list
        change_points = np.where(np.diff(DATA_np_y.astype(int)) != 0)[0] + 1

        # 補上開頭與結尾
        split_indices = np.concatenate((change_points, [len(DATA_np_y)]))

        X_record = []
        y_record = []
        prob_record = []

        for i in range(len(split_indices) - 1):

            ed = split_indices[i]
            st = ed - self.trail_size

            if DATA_np_y[ed-1] in config.choose_marker:
                X_record.append(DATA_np_x[st:ed].T)
                y_record.append(DATA_np_y[ed-1])
                prob_record.append(DATA_np_prob[ed-1])
                


        DATA_np_x = np.stack(X_record, axis=0)
        DATA_np_y = np.stack(y_record, axis=0)
        DATA_np_prob = np.stack(prob_record, axis=0)

        DATA_np_dict = {
            'DATA_np_x': DATA_np_x,
            'DATA_np_y': DATA_np_y,
            'DATA_np_prob': DATA_np_prob,
        }

        return DATA_np_dict
        
    
    def _save_as_pt(self, DATA_np_dict, save_path):

        DATA_np_x = DATA_np_dict['DATA_np_x']
        DATA_np_y = DATA_np_dict['DATA_np_y']
        DATA_np_prob = DATA_np_dict['DATA_np_prob']


        # 1. 建議先轉換為 Tensor (雖然 torch.save 也支援存 numpy，但存 Tensor 較標準)
        x_tensor = torch.from_numpy(DATA_np_x).float()
        y_tensor = torch.from_numpy(DATA_np_y).long()
        prob_tensor = torch.from_numpy(DATA_np_prob).float()


        # 2. 包裝成字典
        save_dict = {
            'x_data': x_tensor,
            'y_data': y_tensor,
            'probs': prob_tensor,
            'metadata': {'fs': config.SAMPLE_RATE, 'channels': config.CHOOSE_CHANNELS} # 也可以存入一些額外資訊
        }

        # 3. 儲存檔案
        torch.save(save_dict, save_path)

        print(f"檔案已成功儲存至: {save_path}")

    

    
 
    # 這裡做 cut_trail, choose channel, save_pt
    def _load_data_from_csv_to_pt(self, csv_path):

        # 如果已處理過，略過處理階段
        pt_path = csv_path[:-4] + ".pt"
        if os.path.exists(pt_path):
            return
        

        # choose channel in here
        # DATA_np_x = (time, channel)
        # DATA_np_y = (time)
        DATA_np_dict = self._csv_to_np(csv_path)

        # DATA_np_x = (trail, channel, time)
        # DATA_np_y = (trail,)
        DATA_np_dict = self._cut_trail(DATA_np_dict)

        self._save_as_pt(DATA_np_dict, pt_path)





    # prepare_datasets 用
    def _train_valid_idx(self, x_np_data, y_np_data):
        
        # --- 分層切分索引 ---
        indices = np.arange(len(x_np_data))
        
        # 使用 stratify=y_np_data 確保每個檔案內的類別比例被維持
        train_idx, valid_idx = train_test_split(
            indices, 
            test_size=self.valid_ratio, 
            stratify=y_np_data, 
            random_state=42
        )
        valid_idx_set = set(valid_idx)

        return valid_idx_set
        

    # prepare_datasets 用
    def _sliding_window(self, stride, segment):
        if segment.shape[0] < self.segment_len: 
            return None

        window_list = []

        for start in range(0, segment.shape[0] - self.segment_len + 1, stride):
            data = segment[start:start + self.segment_len]
            
            # 前處理：去均值(each channel) + 帶通濾波
            demeaned_data = pps.each_channel_demean(data, axis=0)
            window = pps.bandpass(demeaned_data)
            window_list.append(window)

        return window_list


    # prepare_datasets 用
    def _to_dataset(self, segment_list, label_list):
        data_x = np.transpose(np.stack(segment_list), (0, 2, 1))                            # (T, C, N) -> (trial, channel, sample)
        X = torch.tensor(data_x).unsqueeze(1)                                               # (T, 1, C, N)
        y = F.one_hot(torch.tensor(label_list).long(), num_classes=config.N_Class)                                      # trail, n class
        return TensorDataset(X, y)



    # 這段做 window, (demean, bandpass) -> 是對 window 做 , window, train vaild cut
    # 目前是 train 流程會用
    def prepare_datasets_v2(self, 
                            data_x_list, 
                            data_y_list,
                        ):

        print(f"valid_ratio: {self.valid_ratio}")

        if self.valid_ratio <= 0:
            print("[WARNING] self.valid_ratio should > 0")
            return False
        
        augmented_segments_train, augmented_labels_train = [], []
        augmented_segments_valid, augmented_labels_valid = [], []


        # 遍歷每個資料集，這裡 data_x_list 和 data_y_list 都是列表
        for idx in range(len(data_x_list)):
            x_np_data = data_x_list[idx]                                                        # 獲取每一個資料集的 x_data
            y_np_data = data_y_list[idx]                                                        # 獲取對應的 y_data

            # 假設 x_np_data 是 (T, C, S) 形式的資料，y_np_data 是對應的標籤
            x_np_data = np.transpose(x_np_data, (0, 2, 1))                                      # (T, C, S) -> (T, S, C)

            # 決定驗證集索引
            valid_idx_set = self._train_valid_idx(x_np_data, y_np_data)
            
            # 遍歷每一個 Trial
            for i, segment in enumerate(x_np_data):

                if segment.shape[0] < self.segment_len:
                    continue

                # 滑動窗口數據擴增
                window_list = self._sliding_window(self.train_stride, segment)
                label_list = [y_np_data[i]] * len(window_list)

                if window_list:
                    # 根據剛才切分的索引決定歸類
                    if i in valid_idx_set:
                        augmented_segments_valid.extend(window_list)
                        augmented_labels_valid.extend(label_list)
                    else:
                        augmented_segments_train.extend(window_list)
                        augmented_labels_train.extend(label_list)

        
        # 構建訓練集
        dataset = self._to_dataset(augmented_segments_train, augmented_labels_train)
        dataset_valid = self._to_dataset(augmented_segments_valid, augmented_labels_valid)
        return dataset, dataset_valid
        


    # RL 用 第一版
    # 目前是 calib 流程會用
    def prepare_datasets_RL_v1(self, 
                               data_x_list, 
                               data_y_list, 
                               probs_list,
                               RL_interface,
                               session_records,
                        ):

        print(f"valid_ratio: {self.valid_ratio}")

        if self.valid_ratio <= 0:
            print("self.valid_ratio should > 0")
            return False
        
        augmented_segments_train, augmented_labels_train = [], []
        augmented_segments_valid, augmented_labels_valid = [], []

        
        for idx in range(len(data_x_list)):
            x_np_data = data_x_list[idx] # (T, C, S)
            y_np_data = data_y_list[idx]
            probs_data = probs_list[idx] # (T, n_classes)
            
            # 轉換維度
            x_np_data = np.transpose(x_np_data, (0, 2, 1)) # (T, S, C)


            # 1. 決定驗證集索引
            valid_idx_set = self._train_valid_idx(x_np_data, y_np_data)


            # 2. RL 遍歷 each Trial (Hit)
            for i, segment in enumerate(x_np_data):

                # valid 不走 RL 
                if i in valid_idx_set:
                    window_list = self._sliding_window(self.valid_stride, segment)
                    label_list = [y_np_data[i]] * len(window_list)

                    if window_list:
                        augmented_segments_valid.extend(window_list)
                        augmented_labels_valid.extend(label_list)

                    continue
                

                # train 走 RL 
                else:
                    hit_data = {
                        'eeg': segment.T,
                        'probs': probs_data[i],
                    }
                    
                    # --- [核心接點] Bandit 決策流程 ---
                    # get context
                    hit_context_raw = RL_interface.extract_per_hit_context(hit_data)

                    # 正規化並拿取 Action
                    norm_context = RL_interface.normalizer.update_and_normalize(hit_context_raw)
                    action = RL_interface.bandit.select_action(norm_context)
                    
                    record = {
                        'context':norm_context,
                        'action':action,
                    }
                                    
                    session_records.append(record)

                    # UpdateScheduler 記錄此次 action 的品質分數
                    shouldTriggerUpdate = RL_interface.updateScheduler.record(action)
                    print(f"[DataProcess] Hit#{i}: action={action.name}, shouldTriggerUpdate={shouldTriggerUpdate}")


                    # -- 根據 Bandit Action 決定行為 --

                    # 不進入訓練集
                    if action == BanditAction.DISCARD: 
                        continue 
                    

                    # 設定權重（例如 HIGH_WEIGHT 就增加滑動窗口的重疊度）
                    current_stride = self.train_stride

                    # 縮小步長 = 產生更多重複樣本
                    if action == BanditAction.HIGH_WEIGHT:
                        current_stride = max(1, self.train_stride // 2)

                    # 加大步長 = 樣本變少
                    elif action == BanditAction.LOW_WEIGHT:
                        current_stride = self.train_stride * 2 


                    window_list = self._sliding_window(current_stride, segment)
                    label_list = [y_np_data[i]] * len(window_list)

                    if window_list:
                        augmented_segments_train.extend(window_list)
                        augmented_labels_train.extend(label_list)
                    


        # 構建訓練集
        dataset = self._to_dataset(augmented_segments_train, augmented_labels_train)

        # 構建驗證集
        dataset_valid = self._to_dataset(augmented_segments_valid, augmented_labels_valid)
        
        return dataset, dataset_valid, session_records
        






    # @@@@@@@@@@@@@@@@@@@@@
    # model interface 的 train_or_calib 用
    # @@@@@@@@@@@@@@@@@@@@@
    def setup_the_data_path(self, mode=0):
        """
        mode: int
        0 -> train
        1 -> calib

        預想 
        train 是取全部 data 處理
        calib 是只用最新一首
        """
        self.update_dir_path()

        self.update_csv_file_paths(mode)
        print(f"{config.TAGS.INFO.value} csv_file_paths: {self.csv_file_paths}")

        for csv_path in self.csv_file_paths:
            self._load_data_from_csv_to_pt(csv_path)

        self.update_pt_file_paths(mode)
        print(f"{config.TAGS.INFO.value} pt_file_paths: {self.pt_file_paths}")


    # @@@@@@@@@@@@@@@@@@@@@
    # model interface 的 train_or_calib 用
    # @@@@@@@@@@@@@@@@@@@@@      
    def load_data_and_prepare_dataset(self, RL_open=False, RL_interface=None):
        """
        name: String (str)檔案的完整路徑。例如："data/train_set_1.pt"。
        
        train_data: (dict)
        
        train_data['x_data'] torch.Tensor 或 np.ndarray (Trial, Channel, Sample)
        train_data['y_data'] torch.Tensor 或 np.ndarray (Trial, )
        train_data['probs'] torch.Tensor 或 np.ndarray (Trial, classes)
        """
        data_x_list, data_y_list, probs_list = [], [], []
        print(f"RL_open: {RL_open}")

        for name in self.pt_file_paths:
            print(f"name {name}")
            train_data = torch.load(name)
            data_x_list.append(train_data['x_data'].numpy())
            data_y_list.append(train_data['y_data'].numpy())
            probs_list.append(train_data['probs'].numpy())


        session_records = []

        if not RL_open:
            dataset, dataset_val = self.prepare_datasets_v2(data_x_list, 
                                                            data_y_list,
                                                            )
        
        else:
            dataset, dataset_val, session_records = self.prepare_datasets_RL_v1(data_x_list, 
                                                                                data_y_list, 
                                                                                probs_list, 
                                                                                RL_interface,
                                                                                session_records,
                                                                                )
            
        return dataset, dataset_val, session_records




def main():
    print("You are in data process, do nothing")
    pass

    


if __name__ == '__main__':
    main()
