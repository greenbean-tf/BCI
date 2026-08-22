# combine_all_new — Implementation Document

## 概述

`combine_all_new/` 是 `combine_all/` 的模組化重建版本。  
核心目標：**解耦**、**可 debug**、**每個 .py 都能獨立執行**。

---

## 目錄結構

```
combine_all_new/
├── main.py                       # 入口（~20 行）
├── log_class.py                  # Tee stdout/file log
│
├── core/                         # 核心控制（拆自 main.py 1103 行）
│   ├── app_controller.py         # MainController 組裝 + 主選單迴圈
│   ├── account_manager.py        # 帳號初始化 / 登入
│   ├── level_runner.py           # run_level / run_zero_level / run_pause_menu
│   ├── intermission.py           # 結算畫面 + train/calib 觸發
│   ├── test_model_runner.py      # 即時預測測試頁面
│   └── rl_loop.py                # RL 閉環背景線程
│
├── eeg/                          # EEG 深度學習（原 EEG_DL/）
│   ├── config.py                 # 常數
│   ├── global_state.py           # 全域可變狀態
│   ├── lsl_utils.py              # LSL 初始化
│   ├── tcp_utils.py              # TCP Server/Client
│   ├── preprocess.py             # 帶通濾波、去均值
│   ├── helpers.py                # 工具函式
│   ├── eeg_model.py              # EEGNet_our_ver 網路定義（拆自 MI_train.py）
│   ├── braindecode_trainer.py    # 通用訓練器（拆自 MI_train.py）
│   ├── eeg_reader.py             # LSL 資料讀取（背景線程）
│   ├── eeg_predictor.py          # 即時預測（背景線程）
│   ├── eeg_trainer.py            # 訓練 pipeline
│   ├── data_loader.py            # CSV→PT→sliding window→Dataset
│   ├── model_interface.py        # 模型管理門面
│   └── base_model/               # .pth 權重檔（symlink）
│
├── osu/                          # 遊戲 UI
│   ├── osu_parser.py             # 譜面解析
│   ├── game_view.py              # Pygame 繪圖
│   ├── osu_interface.py          # 選單與遊玩事件
│   ├── settings.json
│   ├── songs/ → (symlink)
│   └── sound_effect/ → (symlink)
│
├── rl/                           # 強化學習
│   ├── bandit.py                 # NeuralContextualBandit + UpdateScheduler
│   └── rl_interface.py           # RL 介面（context 提取、alpha 計算）
│
├── account/ → (symlink)
├── test_rl.py                    # RL 獨立測試
└── rl_pretrain.py                # RL 離線預訓練
```

---

## 設計原則

### 1. 單一職責
原本 `main.py` (1103 行) 包含：帳號、遊戲、RL、暫停選單、模型選擇...  
現在拆為 6 個獨立模組，每個 < 300 行。

### 2. 可 Debug
每個 `.py` 都有 `if __name__ == "__main__":` 區塊。  
可以在任意檔案設斷點、單獨執行，不需要啟動整個系統。

### 3. Import 路徑統一
```
舊: import EEG_DL.Utils.config as config
新: import eeg.config as config

舊: from RL.bandit import ...
新: from rl.bandit import ...
```

### 4. 資料夾共用
`songs/`, `sound_effect/`, `base_model/`, `account/` 使用 symlink 指向原 `combine_all/`。

---

## 模組對照表

### main.py → core/

| 原 main.py 區塊 | 新模組 | 行數 |
|---|---|---|
| `class Tee` | `log_class.py` | 17 |
| `MainController.__init__` + `start()` + `before_start()` | `core/app_controller.py` | ~170 |
| `init_account` / `load_accounts` / `login_as` | `core/account_manager.py` | ~100 |
| `run_level` / `run_zero_level` / `run_pause_menu` | `core/level_runner.py` | ~300 |
| `run_intermission` | `core/intermission.py` | ~110 |
| `run_test_model` | `core/test_model_runner.py` | ~75 |
| `rl_loop_worker` | `core/rl_loop.py` | ~70 |

### MI_train.py → eeg/

| 原區塊 | 新模組 |
|---|---|
| `class EEGNet_our_ver` | `eeg/eeg_model.py` |
| `class BraindecodeTrainer` | `eeg/braindecode_trainer.py` |
| `load_*_params()` | `eeg/eeg_model.py` |

### EEG_DL/ → eeg/

| 原路徑 | 新路徑 |
|---|---|
| `EEG_DL/Utils/config.py` | `eeg/config.py` |
| `EEG_DL/Utils/global_value.py` | `eeg/global_state.py` |
| `EEG_DL/Utils/LSL.py` | `eeg/lsl_utils.py` |
| `EEG_DL/Utils/TCP_use.py` | `eeg/tcp_utils.py` |
| `EEG_DL/Utils/preprocess.py` | `eeg/preprocess.py` |
| `EEG_DL/Utils/some_functions.py` | `eeg/helpers.py` |
| `EEG_DL/EEG/EEG_Read.py` | `eeg/eeg_reader.py` |
| `EEG_DL/EEG/EEG_Predict.py` | `eeg/eeg_predictor.py` |
| `EEG_DL/EEG/EEG_Train.py` | `eeg/eeg_trainer.py` |
| `EEG_DL/EEG/data_process_np.py` | `eeg/data_loader.py` |
| `EEG_DL/EEG/model_interface.py` | `eeg/model_interface.py` |

---

## 執行方式

```bash
cd combine_all_new
python main.py
```

### 測試 RL 模組
```bash
python test_rl.py
```

### 離線 RL 預訓練
```bash
python rl_pretrain.py --dry-run
```

---

## 線程架構

```
main thread (pygame event loop)
├── eeg_reader thread      → 持續讀取 LSL 資料、更新 buffer、寫 CSV
├── eeg_predictor thread   → 持續從 buffer 預測 → 寫入 pred_buffer
└── rl_loop thread         → 背景 RL 閉環（calib + bandit update）
```

所有線程使用 `threading.Event` 控制生命週期，`threading.Lock` 保護共享資料。

---

## 驗證結果

- ✅ `test_rl.py`：9/9 測試通過
- ✅ 所有 32 個 .py 檔案已建立
- ✅ symlink 建立完成（songs, sound_effect, base_model, account）
- ✅ 原始演算法邏輯完整保留，僅做結構重組
