# AIREADME.md — Antigravity 專用專案設計文件

> 此文件僅供 Google Antigravity 閱讀，作為專案架構的唯一可信來源。

---

## 專案概述

VR-BCI 校正系統：透過 osu!mania 遊戲介面收集 EEG 腦波資料，使用深度學習模型即時預測 4-class MI（Motor Imagery），並以 RL Bandit 演算法自動優化線上校正流程。

## 核心模組

### 1. `main.py` — 主控制器 (MainController)

- 管理帳號系統、遊戲流程（歌曲選擇 → 遊玩 → 結算 → 訓練/校正）
- 啟動多執行緒：`data_logger_worker`（EEG 記錄）、`rl_loop_worker`（RL 閉環）
- 透過 `model_interface` 和 `RL_interface` 橋接 DL 與 RL 子系統

### 2. `RL/bandit.py` — NeuralUCB Contextual Bandit

**2026/04/09 升級：LinUCB → NeuralUCB**

#### 架構
- `BanditAction`：4 個動作（INCLUDE, DISCARD, HIGH_WEIGHT, LOW_WEIGHT）
  - **已移除 ONLINE_UPDATE**
- `RewardMLP`：輕量 MLP (context_dim → 64 → 32 → 1)，每個 action 獨立
- `NeuralContextualBandit`：
  - 選擇：MLP 預測 + 梯度 norm 探索項（NeuralUCB）
  - 更新：MSE loss + Experience Replay Buffer（最近 500 筆）
  - 裝置：由 `BANDIT_DEVICE` 變數控制 CPU/GPU
- `UpdateScheduler`：策略 B 加權品質分數排程
  - HIGH_WEIGHT=2.0, INCLUDE=1.0, LOW_WEIGHT=0.3, DISCARD=0.0
  - 累積分數 ≥ 15.0 觸發更新
- `ContextNormalizer`：running min-max 正規化
- `ContextualBandit`：Legacy alias → `NeuralContextualBandit`

#### Context 向量 (4 維)
```
[MEAN_ENTROPY, MEAN_MARGIN, DELTA_LOSS, ALPHA_POWER_NORM]
```
注意：第 3 維從 DELTA_ACC 改為 DELTA_LOSS

#### Reward 公式
```
r_t = -ΔLoss_t - λ * T_t - μ * γ_t * LQP_base_t
```
- ΔLoss = current_val_loss - last_val_loss（連續可導信號）
- clip 至 [-2.0, 2.0]

#### 存檔格式
- `neural_ucb_v1`：MLP state_dict + optimizer state_dict + replay buffers
- 載入時自動偵測舊 LinUCB 格式並丟棄重建

### 3. `RL/RL_interface.py` — RL 接口

- 持有 `NeuralContextualBandit`、`ContextNormalizer`、`UpdateScheduler`
- `extract_per_hit_context()`：從 EEG + probs 提取 context 向量
  - Entropy, Margin, deltaValLoss, Alpha Power
- 屬性：`deltaValLoss`、`rlLoopDeltaValLoss`（平滑化 reward 信號）

### 4. `EEG_DL/EEG/model_interface.py` — DL 模型接口

- 管理 EEGReader、EEGPredictor、訓練 pipeline
- `train_or_calib()`：訓練或校正，回傳 session_records
  - 現在計算 `deltaValLoss` 和 `rlLoopDeltaValLoss`
  - `lastValLoss` 追蹤上次 val_loss
- 模型版本管理：`Trained_Model_vXXX` / `Calibed_Model_vXXX` / `RL_Loop_Model_vXXX`

### 5. `EEG_DL/EEG/EEG_Train.py` — 訓練 Pipeline

- `EEGTrainingPipeline`：包裝 BraindecodeTrainer
- `_run_braindecode_training()` 回傳 `(val_acc, val_loss)` tuple
- `run_training()` 傳遞 tuple 回 model_interface

### 6. `EEG_DL/EEG/data_process_np.py` — 資料處理

- `prepare_datasets_RL_v1()`：RL 路徑中每次 hit 做 Bandit 決策
  - 現在整合 `UpdateScheduler.record(action)`
  - 不再有 ONLINE_UPDATE 分支

### 7. `osu/` — 遊戲介面
- `game_view.py`：Pygame 繪製
- `osu_parser.py`：.osu 譜面解析
- `osu_interface.py`：按鍵對照 (左=0, 右=1, 下=2, 上=3)

## 資料流

```
EEG 硬體 → LSL → EEGReader → CSV/Buffer
                                    ↓
osu! 遊戲 → Marker 標注 → data_process_np → .pt 檔
                                    ↓
                            Bandit (NeuralUCB)
                            ├─ INCLUDE → 納入訓練集
                            ├─ DISCARD → 丟棄
                            ├─ HIGH_WEIGHT → 縮小 stride
                            └─ LOW_WEIGHT → 加大 stride
                                    ↓
                            UpdateScheduler 判斷
                            (品質分數 ≥ 15.0?)
                                    ↓
                            EEGTrainingPipeline
                                    ↓
                            val_loss → compute_reward
                            → Bandit 更新 (Experience Replay)
```

## 設定檔

- `config.py`：全域常數（SAMPLE_RATE=500, N_Class=4, CHANNEL 設定等）
- `global_value.py`：執行時期動態路徑
- `BANDIT_DEVICE`（bandit.py 內）：MLP 裝置設定

## 按鍵對照規則
左=0, 右=1, 下=2, 上=3
