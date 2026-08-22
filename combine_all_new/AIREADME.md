# combine_all_new — AIREADME

> 本文件僅供 Antigravity 內部參考，作為專案架構與設計決策的唯一可信來源。

## 專案概覽

VR-BCI 音遊校準系統：透過 EEG 腦波訊號控制 OSU Mania 4K 音遊，搭配 RL（Contextual Bandit）自動校準 DL 分類模型。

## 核心架構

```
combine_all_new/
├── main.py                    # 入口
├── config.py                  # 全域不變常數（EEG/TCP/遊戲/RL/路徑）
├── global_state.py            # 全域可變狀態（buffer/路徑/即時狀態）
├── log_class.py               # Tee stdout 至 log
├── rl_pretrain.py             # 離線 RL Pretrain 腳本
│
├── core/                      # 遊戲流程控制
│   ├── app_controller.py      # MainController 主控台
│   ├── level_runner.py        # 單首歌曲遊玩迴圈
│   ├── intermission.py        # 結算畫面 + Train/Calib
│   ├── rl_loop.py             # RL 閉環背景線程（緩衝型滾動進化）
│   ├── test_model_runner.py   # 模型測試
│   └── account_manager.py     # 帳號管理
│
├── eeg/                       # EEG 子系統
│   ├── config.py              # ⚠ shim → re-export root config.py
│   ├── global_state.py        # ⚠ shim → re-export root global_state.py
│   ├── eeg_reader.py          # LSL 讀取 + CSV 記錄
│   ├── eeg_predictor.py       # 背景推論線程（threading.Lock 保護）
│   ├── eeg_trainer.py         # 訓練 Pipeline
│   ├── braindecode_trainer.py # Braindecode 訓練器
│   ├── eeg_model.py           # EEGNet 模型定義
│   ├── data_loader.py         # 資料載入 + RL v1/v2 Dataset
│   ├── model_interface.py     # 模型管理門面
│   ├── preprocess.py          # 前處理 (demean, bandpass)
│   ├── lsl_utils.py           # LSL inlet 建立
│   └── tcp_utils.py           # TCP Server/Client
│
├── rl/                        # RL 子系統
│   ├── bandit.py              # NeuralUCB + UpdateScheduler + ExperienceReplayMemory
│   ├── rl_interface.py        # RL 介面 (Bandit + Normalizer + Scheduler)
│   └── rl_pretrain/           # 離線 pretrain 資料 + log
│       └── data/              # pretrain .pt 檔（Stable_Buffer 來源）
│
└── osu/                       # OSU Mania 遊戲介面
    ├── osu_interface.py
    ├── osu_parser.py
    └── game_view.py
```

## RL Loop 架構：緩衝型滾動進化

### 設計原則
- **拒絕高頻更新**：歌曲進行中 DL 維持 DL_n 不動
- **積分制推進**：UpdateScheduler(scoreThreshold=7.0) 累積滿分才觸發 DL_n → DL_{n+1}
- **Experience Replay 防護**：混合 Stable_Buffer(30%) + Dynamic_Buffer(70%) 防止過擬合
- **即時模型同步**：更新後使用 threading.Lock 安全切換 predictor 模型
- **即時資料來源**：從 global_state.eeg_buffer + prob_buffer 取 real-time data（不讀歷史 .pt）

### 兩階段架構 (rl_loop.py)

**Phase 1 — 即時資料蒐集（每 3 秒一次）**
- 從 global_state.eeg_buffer 擷取 EEG snapshot + prob_buffer 取最新預測機率
- 前處理（demean + bandpass）→ extract context → Bandit 決策 → UpdateScheduler 累積
- DL 不動，只暫存 pending records

**Phase 2 — 條件觸發更新（shouldUpdate 時）**
- ExperienceReplayMemory.sample(30% stable + 70% dynamic)
- model_interface.train_rl_loop_update() → DL_{n+1}
- eeg_predictor.reload_model() 🔒 Lock
- Bandit reward 更新

### 歌曲結束時
- intermission 中 DL + RL 無條件更新一次（不用 UpdateScheduler）
- save_hq_buffer() 持久化 Dynamic_Buffer → account/{user}/hq_buffer.pt

## ExperienceReplayMemory 雙重緩衝

| 緩衝區 | 來源 | 行為 | 抽樣比例 |
|---|---|---|---|
| Stable_Buffer | rl/rl_pretrain/data/*.pt | 啟動時一次性載入，只讀 | 30% |
| Dynamic_Buffer | 遊玩中 INCLUDE/HIGH_WEIGHT 資料 | 持續累積，deque(maxlen=2000) | 70% |

### 持久化
- save_buffer(): 取 Dynamic_Buffer 中 qualityScore 最高前 100 筆 → hq_buffer.pt
- load_buffer(): 啟動時嘗試載入 hq_buffer.pt 預填 Dynamic_Buffer

## 多線程安全

| 共享資源 | 保護機制 |
|---|---|
| eeg_predictor.model | threading.Lock (modelLock) |
| global_state.NOW_FT_CHECKPOINT | 先寫 path → acquire lock → reload |

## Bandit Action 品質分數

| Action | 分數 | 效果 |
|---|---|---|
| HIGH_WEIGHT | 2.0 | stride 減半 |
| INCLUDE | 1.0 | 標準 stride |
| LOW_WEIGHT | 0.7 | stride 加倍 |

Action 空間定義（n_actions=3）：
- A0 - INCLUDE: 納入訓練 buffer（預設行為）
- A1 - HIGH_WEIGHT: 高權重納入 buffer
- A2 - LOW_WEIGHT: 低權重納入 buffer

## Context 定義（3 維）

context 為從 EEG data 濃縮出的狀態描述，不包含 deltaLoss（deltaLoss 僅用於 reward 計算）。

| Index | 名稱 | 來源 |
|---|---|---|
| 0 | MEAN_ENTROPY | DL 預測的 class distribution entropy |
| 1 | MEAN_MARGIN | top-1 與 top-2 class probability 差值 |
| 2 | ALPHA_POWER_NORM | C3/Cz/C4 通道 Alpha Band (8-13Hz) 能量 |

## ContextNormalizer（EMA + sigmoid）

- 使用 EMA (α=0.05) 追蹤 mean 和 variance，替代 running min-max 避免離群值永久發散
- z-score → sigmoid 壓縮至 [0, 1]
- 第一筆回傳中性值 0.5

## 關鍵超參數

- `scoreThreshold = 7.0` — UpdateScheduler 觸發閾值
- `maxDynamicSize = 2000` — Dynamic_Buffer 容量
- `stableRatio = 0.3` — Stable_Buffer 抄樣比例
- `topK = 100` — save_buffer 保留筆數
- `rl_ft_pipeline.epochs = 3` — RL Loop fine-tune epoch 數
- `context_dim = 3` — context 維度（entropy, margin, alpha_power）
- `n_actions = 3` — BanditAction 數量（INCLUDE / HIGH_WEIGHT / LOW_WEIGHT）
