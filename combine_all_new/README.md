# combine_all_new

VR-BCI 音遊校準系統 — 以 EEG 腦波訊號控制 OSU Mania 4K 音遊，搭配強化學習（NeuralUCB Contextual Bandit）自動校準 DL 分類模型。

## 系統特色

- **NeuralUCB Contextual Bandit**：根據 EEG 資料品質（entropy、margin、alpha power）自動決定資料篩選策略
- **緩衝型滾動進化 (Buffered Rolling Evolution)**：歌曲進行中拒絕高頻模型更新，累積足夠品質分數才觸發 DL 更新
- **雙重緩衝 Experience Replay**：混合穩定歷史資料與即時高品質資料，防止過擬合
- **即時模型切換**：RL Loop 觸發更新後，predict thread 透過 Lock 機制安全切換至新模型

## 目錄結構

```
combine_all_new/
├── main.py              # 入口
├── core/                # 遊戲流程控制
├── eeg/                 # EEG 讀取、預測、訓練
├── rl/                  # 強化學習（Bandit + Scheduler + Replay Memory）
├── osu/                 # OSU Mania 遊戲介面
└── account/             # 使用者帳號資料
```

## 環境需求

- Python 3.10+
- PyTorch, NumPy, SciPy, scikit-learn
- Braindecode, pylsl, Pygame

## 執行

```bash
python main.py
```

## RL Loop 工作流程

1. 歌曲播放中，背景 RL Loop 持續蒐集 EEG 資料
2. Bandit 對每筆資料做 INCLUDE/DISCARD/HIGH_WEIGHT/LOW_WEIGHT 決策
3. UpdateScheduler 累積品質分數，達到 7.0 時觸發 DL 模型更新
4. 更新時混合 Stable_Buffer(30%) + Dynamic_Buffer(70%) 進行 fine-tune
5. 新模型即時切換至推論線程
6. 歌曲結束時，DL 和 RL 進行一次無條件更新

## 更新紀錄 (Changelog)

- 修正 `eeg/data_loader.py` 在 `calib` 模式時因 `st < 0` 造成截取片段長度不一致，導致 `np.stack` 拋出 `ValueError` 的問題，並新增了 Debug log。
