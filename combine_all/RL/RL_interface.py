
import numpy as np

import EEG_DL.Utils.config as config
from RL.bandit import NeuralContextualBandit, ContextNormalizer, UpdateScheduler

class RL_interface:
    def __init__(
        self, 
        bandit: NeuralContextualBandit | None = None
    ):
        self.bandit = bandit or NeuralContextualBandit()
        self.normalizer = ContextNormalizer()
        self.updateScheduler = UpdateScheduler(scoreThreshold=15.0)

        # Reward 用：delta_loss（平滑化信號）
        self.deltaValLoss = 0.0
        self.rlLoopDeltaValLoss = 0.0

        # 保留 delta_acc 供 log 參考（不再參與 reward 計算）
        self.delta_acc = 0.0
        self.RL_loop_delta_acc = 0.0

        self.total_update_times = 0

        print(f"[RL_interface] 初始化完成")
        print()



    def _get_band_power(self, freqs, psd, low, high):
        """
        psd             : after channel ave
        low, high       : 頻帶範圍
        """
        idx = np.logical_and(freqs >= low, freqs <= high)
        return np.sum(psd[idx])
    
    def _get_alpha_ratio(self, eeg_data):
        """
        計算 Alpha Band (8-13Hz) 在訊號中的能量佔比
        eeg_data: (n_channels, n_samples) 或 (n_samples,)

        self.trail_size = 1500
        已做 前處理：去均值(each channel) + 帶通濾波
        """

        SAMPLE_RATE = config.SAMPLE_RATE
        BUFFER_SIZE = config.SAMPLE_RATE * 3  # 3s 

        # ======== FFT use ========
        # 頻率軸 (0, 1, 2, ..., 500 Hz)
        freqs = np.fft.rfftfreq(BUFFER_SIZE, d=1/SAMPLE_RATE)


        # 窗函數 (Hanning Window)，用於減少頻譜洩漏
        # fft_window (1, BUFFER_SIZE) 以便與 
        # eeg_choose_channel (n_channel, BUFFER_SIZE) 相乘
        fft_window = np.hanning(BUFFER_SIZE).reshape(1, -1)


        # ====== 1. preprocess ======
        # 選定特定通道 目前是 "C3",   "Cz",   "C4",
        eeg_choose_channel = eeg_data[config.compute_alpha_channel]

        # Windowing
        data_windowed = eeg_choose_channel * fft_window


        # ====== 2. FFT  ======
        # rfft: Real FFT (只計算正頻率部分)
        fft_vals = np.fft.rfft(data_windowed, axis=1)
        
        # 計算功率譜 (PSD)
        # 取絕對值(振幅) -> 平方 -> 除以長度(正規化)
        # 這裡簡單用 |FFT|^2 / N 即可代表相對能量強度
        psd = (np.abs(fft_vals) ** 2) / BUFFER_SIZE
        

        #  channel ave
        avg_psd = np.mean(psd, axis=0)


        # ====== 3. get_band_power ======
        alpha_power = self._get_band_power(freqs, avg_psd, 8, 13)
        
        return alpha_power 

    def extract_per_hit_context(self, hit_data):
        """
        含 
        eeg: (n_channels, n_samples), 
        probs: (n_classes,), 
        """
        
        # 1. Entropy (針對單一 Hit 的預測機率)
        p = hit_data['probs']
        entropy = -np.sum(p * np.log(p + 1e-9))
        
        # 2. Margin (最高與次高機率差)
        sorted_p = np.sort(p)
        margin = sorted_p[-1] - sorted_p[-2]
        
        # 3. Alpha Power (使用下方定義的函數)
        # hit['eeg'] 預期為  或 (n_samples,)
        alpha_p = self._get_alpha_ratio(hit_data['eeg'])


        # 組合 Context (4,)
        # 注意：第 3 維改為 delta_loss（由外部填入），這裡先放 0
        context = np.array([
            entropy, 
            margin, 
            self.deltaValLoss,  # 使用 delta_loss 取代 delta_acc
            alpha_p
        ])

        if False:
            print(f"[RL_interface] extract_per_hit_context: entropy={entropy:.4f}, "
                f"margin={margin:.4f}, deltaValLoss={self.deltaValLoss:.4f}, alpha={alpha_p:.4f}")
        
        return context
    
