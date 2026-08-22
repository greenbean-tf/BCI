"""
RL Interface — 強化學習介面。

對應原 RL/RL_interface.py，import 更新。
"""
import numpy as np

import config
from rl.bandit import NeuralContextualBandit, ContextNormalizer, UpdateScheduler


class RL_interface:
    def __init__(self, bandit: NeuralContextualBandit | None = None):
        self.bandit = bandit or NeuralContextualBandit()
        self.normalizer = ContextNormalizer()
        self.updateScheduler = UpdateScheduler(scoreThreshold=7.0)

        # Reward 用
        self.deltaValLoss = 0.0
        self.rlLoopDeltaValLoss = 0.0

        # 保留 delta_acc 供 log
        self.delta_acc = 0.0
        self.RL_loop_delta_acc = 0.0

        self.total_update_times = 0

        print(f"[RL_interface] 初始化完成")
        print()

    def _get_band_power(self, freqs, psd, low, high):
        idx = np.logical_and(freqs >= low, freqs <= high)
        return np.sum(psd[idx])

    def _get_alpha_ratio(self, eeg_data):
        """計算 Alpha Band (8-13Hz) 能量佔比"""
        SAMPLE_RATE = config.SAMPLE_RATE
        BUFFER_SIZE = config.SAMPLE_RATE * 3

        freqs = np.fft.rfftfreq(BUFFER_SIZE, d=1 / SAMPLE_RATE)
        fft_window = np.hanning(BUFFER_SIZE).reshape(1, -1)

        eeg_choose_channel = eeg_data[config.compute_alpha_channel]
        data_windowed = eeg_choose_channel * fft_window

        fft_vals = np.fft.rfft(data_windowed, axis=1)
        psd = (np.abs(fft_vals) ** 2) / BUFFER_SIZE
        avg_psd = np.mean(psd, axis=0)

        alpha_power = self._get_band_power(freqs, avg_psd, 8, 13)
        return alpha_power

    def extract_per_hit_context(self, hit_data):
        """
        提取 per-hit context 向量 (3維)
        context = [mean_entropy, mean_margin, alpha_power_normalized]
        所有維度皆為從 data 資料中濃縮出的狀態描述。

        hit_data: {'eeg': (n_channels, n_samples), 'probs': (n_classes,)}
        """
        p = hit_data['probs']
        entropy = -np.sum(p * np.log(p + 1e-9))

        sorted_p = np.sort(p)
        margin = sorted_p[-1] - sorted_p[-2]

        alpha_p = self._get_alpha_ratio(hit_data['eeg'])

        context = np.array([
            entropy,
            margin,
            alpha_p
        ])

        return context


# ─── Debug ───
if __name__ == "__main__":
    print("=== rl.rl_interface standalone test ===")
    rl = RL_interface()
    print(f"bandit type = {type(rl.bandit).__name__}")
    print(f"normalizer  = {type(rl.normalizer).__name__}")
    print("✅ rl_interface OK")
