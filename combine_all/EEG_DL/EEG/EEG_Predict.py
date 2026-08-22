import time
import threading


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset




from braindecode.models import ShallowFBCSPNet


from EEG_DL.EEG.MI_train import load_shallowfbcsp_params, load_EEGNet_our_ver_params, EEGNet_our_ver



import EEG_DL.Utils.config as config
import EEG_DL.Utils.preprocess as pps
import EEG_DL.Utils.global_value as global_value


class EEGPredictor:                                                         # 調整模型需要改 self.model，和 self.load_func 方法
    def __init__(self, tcp_client):                                         # old: lsl_outlet=None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_check_point_path = "" # global_value.NOW_FT_CHECKPOINT + ".pth"
        
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ shallownet
        
        """
        self.model = ShallowFBCSPNet
        self.load_func = load_shallowfbcsp_params                            # load shallow net parameter
        """

        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ EEGNet
        
        self.model_class = EEGNet_our_ver
        self.model = None                                                           # 實例，由 init_model() 初始化
        self.load_func = load_EEGNet_our_ver_params                          # load EEGNet parameter
        
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

        self.model_arg = self.set_model_arg()

        self.predict_eeg_stop_event = threading.Event()
        self.predict_eeg_thread = None

        # 用來卡更新模型，更新模型不預測 # 使用 bool 判斷如果 unity 那邊按太快會出錯，所以有在 True 的時候會卡不能再次更新
        self.is_updating = False

        # self.outlet = lsl_outlet
        self.tcp_client = tcp_client                                        # TCP Server



    def start_predict_eeg_thread(self):
        
        # 防止重啟 thread
        if self.predict_eeg_thread and self.predict_eeg_thread.is_alive():              
            print(f"{config.TAGS.WARNING.value} Marker thread is already running.")
            return  


         # ← 在這裡重新載入最新的 checkpoint
        self.load_check_point_path = global_value.NOW_FT_CHECKPOINT + ".pth"
        self.model = self.init_model()
        print(f"[DEBUG] 重新載入模型: {self.load_check_point_path}")


        print(f"{config.TAGS.INFO.value} Start predict loop...")
        self.predict_eeg_stop_event.clear()                                             # 重置 stop_event，允許重新啟動
        self.predict_eeg_thread = threading.Thread(target=self.predict_loop, daemon=True)
        self.predict_eeg_thread.start()


    def end_predict_eeg_thread(self):

        print(f"{config.TAGS.INFO.value} End predict loop...")
        self.predict_eeg_stop_event.set()                                               # 設置停止事件
        if self.predict_eeg_thread and self.predict_eeg_thread.is_alive():
            self.predict_eeg_thread.join()
            self.predict_eeg_thread = None


    def init_model(self):
        # ======== Model Setup ============
        model = self.model_class(**self.model_arg).to(self.device)                     # n channel, sample, n class
        checkpoint = torch.load(self.load_check_point_path, map_location=self.device)
        
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def update_check_point(self, path):
        self.load_check_point_path = path


    def update_model(self):
        if self.is_updating:                                                            # 防止一直進來，出現 crash 的問題
            return
        self.is_updating = True

        # model_class 已在 __init__ 設定，直接用 init_model 重新載入
        self.model = self.init_model()
        self.is_updating = False


    def to_dataset(self, segment_list, label_list):
        data_x = np.array(segment_list)                                                 # (T, C, N) # (trial, channel, sample)
        X = torch.tensor(data_x).unsqueeze(1)                                           # (T, 1, C, N)
        y = torch.tensor(label_list).long()
        return TensorDataset(X, y)


    def set_model_arg(self):
        simulate_x = np.zeros((1, config.CHOOSE_CHANNELS, config.SAMPLE_RATE))               # (sample, channel, trial)
        simulate_y = np.zeros((1, config.N_Class))                                      # (trail, n class)
        dataset = self.to_dataset(simulate_x, simulate_y)                               # 這邊是為了要對應 load_conformer_params 才使用這個 (懶得改 XD
        return self.load_func(dataset)                                                   # load_EEGNet_our_ver_params, load_shallowfbcsp_params

    
    def predict_loop(self):

        while not self.predict_eeg_stop_event.is_set():

            if global_value.eeg_buffer.shape[1] >= config.BUFFER_SIZE:


                # shape: (channel, BUFFER_SIZE) -> (channel, sample)


                # input_data = pps.down_sample(pps.bandpass(eeg_buffer[:, -BUFFER_SIZE:].copy()))

                # 需要選擇的 channel
                # indices = [5, 6, 7, 10, 11, 12, 15, 16, 17, 19, 20, 21, 25, 26, 27]

                # data = global_value.eeg_buffer[indices, :].copy()  # shape (15, N)

                data = global_value.eeg_buffer[:, -config.BUFFER_SIZE:] # shape (15, 500)

                data = pps.each_channel_demean(data, axis=1)
                data = pps.bandpass(data, axis=-1)
                

                x_batch = torch.from_numpy(data.copy()).float().unsqueeze(0).to(self.device)
                # print(f"shape: {x_batch.shape}") # (1,15,500) (trial, channel, sample)
                
                self.model.eval()
                with torch.no_grad():
                    if not self.is_updating:
                        outputs = self.model(x_batch)                       # (n_classes,) [BatchSize=1, n_classes]

                    prediction = int(torch.argmax(outputs).cpu().item())
                    # print(f"Prediction: {prediction}")
                    # print(f"outputs: {outputs}, type {type(outputs)}, shape {outputs.shape}")
                    # outputs: type <class 'torch.Tensor'>, shape torch.Size([1, 4])

                    probability = F.softmax(outputs, dim=1).cpu().detach().numpy()[0].tolist()
                    # print(f"prob: {probability}, type {type(probability)}, shape {len(probability)} prob[0] {type(probability[0])}")
                    # self.outlet.push_sample([prediction])


                    global_value.pred_buffer.append(prediction)
                    global_value.prob_buffer.append(probability)

                    # 推送到 TCP
                    # if self.tcp_client:
                    if False:
                        self.tcp_client.send_messages(f"pred{config.TCP_STR_SPLIT}"
                                                      f"{prediction}{config.TCP_STR_SPLIT}"
                                                      f"{probability[0]}{config.TCP_STR_SPLIT}"
                                                      f"{probability[1]}{config.TCP_STR_SPLIT}"
                                                      f"{probability[2]}{config.TCP_STR_SPLIT}"
                                                      f"{probability[3]}"
                                                      )
                    time.sleep(config.PREDICTION_INTERVAL)


# ======== Main ============
def main():
    print("You are in EEG Prediction, do nothing")
    pass
    


if __name__ == "__main__":
    main()
