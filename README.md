# 2026_BCI_-
專題東西放這邊


---

## 一、Main 完整流程 (`main.py`)

### 啟動階段

```
MainController()
  ├─ 初始化 LSL（模擬 or 實體 EEG）
  ├─ 初始化 TCP Server / Client（跨程序通訊）
  ├─ 初始化 osu_interface（遊戲介面）
  ├─ 初始化 model_interface（DL 模型管理）
  └─ 初始化 RL_interface（Bandit 決策引擎）
```

### 登入 → 主迴圈

```
before_start()
  ├─ run_account_login()          ← 帳號選擇畫面
  ├─ init_account()               ← 建立帳號資料夾、載入 RL Bandit / version.json
  ├─ 設定全域路徑（model / data / log）
  └─ start()                      ← 進入主選單迴圈
```

### 主選單 (`start()`)

```
while True:
  action = osu_interface.run_main_menu()
  
  ┌─ "song_select"   → run_song_select() → run_level()
  ├─ "model_select"  → run_model_select()
  ├─ "test_model"    → run_test_model()      ← 持續預測 EEG 並顯示（按 ESC 結束）
  └─ "settings"      → run_settings_menu()
```

### 遊玩流程 (`run_level()`)

```
run_level(map_file)
  ├─ 判斷是 zero_level（模擬人類操作）或 normal_level
  │
  ├─ 啟動遊戲音樂
  ├─ 啟動 Thread: start_read_eeg()        ← EEG 背景緩衝
  ├─ 啟動 Thread: start_pred_eeg()        ← EEG 預測迴圈
  ├─ 啟動 Thread: data_logger_worker()    ← CSV 紀錄（SAVE_CSV=True 時）
  ├─ 啟動 Thread: rl_loop_worker()        ← RL 閉環（USE_RL_LOOP=True 時）
  │
  ├─ 遊玩迴圈（while playing）:
  │     ├─ 計算 game_time_ms（LSL 時間對齊）
  │     ├─ calculate_marker()              ← 取得當前音符的 ground truth
  │     ├─ 從 pred_buffer 取得 BCI 預測結果
  │     ├─ check_hit()                     ← 判定是否命中
  │     ├─ 監聽鍵盤（手動操作 / 暫停）
  │     ├─ 計算準確率
  │     └─ 繪製遊玩畫面
  │
  └─ 歌曲結束 → 停止所有 Thread → return (state, map_notes)
```

### 結算畫面 (`run_intermission()`)

```
run_intermission(map_notes)
  ├─ 顯示成績（正確數 / 每類準確率 / 使用模型）
  │
  ├─ 按 T → 全量訓練（mode=0, 不使用 RL）
  │         → run_model_select()
  │
  ├─ 按 C → 校正（mode=1, 使用 RL Bandit）
  │         → Bandit 逐 hit 決策 → 訓練 → compute_reward(deltaLoss) → bandit.update()
  │         → run_model_select()
  │
  └─ 按 S → 跳過，返回主選單
```

### RL 閉環 Thread (`rl_loop_worker()`)

```
while rl_loop_running:
  ├─ bandit.load()
  ├─ train_or_calib(mode=1, RL_LOOP_USE=True)
  │     └─ data_process 中 Bandit 逐 hit 決策
  ├─ 同步 delta_val_loss
  ├─ for each record:
  │     compute_reward(deltaLoss) → bandit.update()
  ├─ bandit.save()
  └─ sleep(3)
```

---

## 二、Data Process 流程 (`data_process_np.py`)

### 資料載入流程

```
setup_the_data_path(mode)
  ├─ update_csv_file_paths(mode)   ← mode=0: 全部 CSV / mode=1: 最新一首
  ├─ for each CSV:
  │     _load_data_from_csv_to_pt(csv_path)
  │       ├─ _csv_to_np()          ← CSV → numpy (time, channel)
  │       ├─ _cut_trail()          ← 根據 marker 切 trail (trail, channel, time)
  │       └─ _save_as_pt()         ← 存為 .pt 檔
  └─ update_pt_file_paths(mode)
```

### 資料集建構

```
load_data_and_prepare_dataset(RL_open, RL_interface)
  ├─ 從 .pt 載入所有 data_x, data_y, probs
  │
  ├─ if RL_open == False:
  │     prepare_datasets_v2()      ← 一般訓練路徑
  │       ├─ 分層切分 train / valid
  │       └─ sliding_window + demean + bandpass → TensorDataset
  │
  └─ if RL_open == True:
        prepare_datasets_RL_v1()   ← RL 路徑
          ├─ 分層切分 train / valid
          ├─ valid 不走 RL（直接用 valid_stride 滑窗）
          └─ train 每個 hit:
                ├─ extract_per_hit_context()    ← 提取 entropy, margin, alpha
                ├─ normalizer.update_and_normalize()
                ├─ bandit.select_action()       ← NeuralUCB 決策
                ├─ updateScheduler.record()     ← 記錄品質分數
                │
                ├─ DISCARD  → 跳過
                ├─ INCLUDE  → train_stride 滑窗
                ├─ HIGH_WEIGHT → stride // 2（更多重疊）
                └─ LOW_WEIGHT  → stride * 2（更少樣本）
```

### 滑動窗口 (`_sliding_window`)

```
對每個 trail (channel, 1500 samples):
  ├─ 以 stride 步長滑動，產生多個 500-sample 片段
  ├─ 每個片段做:
  │     ├─ each_channel_demean()   ← 去均值
  │     └─ bandpass()              ← 帶通濾波 (1-40Hz)
  └─ 回傳 window_list
```

---

## 三、DL Model 流程 (`model_interface.py` + `EEG_Train.py` + `EEG_Predict.py`)

### 模型架構

```
EEGNet_our_ver (自訂 EEGNet)
  ├─ 1. Multi-Kernel Temporal Conv (kernel: 32, 64, 128, 200, 256, 512)
  ├─ 2. Multi-Kernel Spatial Conv (kernel: n_chans, n_chans//2+1, 3)
  ├─ 3. AvgPool + Dropout
  ├─ 4. Separable Conv + AvgPool + Dropout
  └─ 5. Linear Classifier → 4 classes
```

### 訓練流程 (`EEGTrainingPipeline`)

```
run_training(dataset, dataset_val, save_path)
  ├─ 初始化 BraindecodeTrainer
  │     ├─ model = EEGNet_our_ver(**kwargs)
  │     ├─ criterion = CrossEntropyLoss
  │     └─ optimizer = Adam (lr=1e-5)
  │
  ├─ if ft (fine-tune):
  │     ├─ load_checkpoint() 載入現有模型
  │     └─ freeze 除 classifier 外的所有層
  │
  ├─ for epoch in range(num_epochs):
  │     ├─ train loop → loss.backward() → optimizer.step()
  │     ├─ _evaluate(val_loader) → val_loss, val_acc
  │     ├─ 儲存 checkpoint
  │     ├─ Early Stopping: val_loss 連續 patience 次未改善則停止
  │     └─ TCP 傳送訓練進度
  │
  └─ return (val_acc, val_loss)   ← 取 best_val_loss epoch 的結果
```

### 即時預測流程 (`EEGPredictor`)

```
predict_loop() [背景 Thread]:
  while not stopped:
    ├─ 從 global_value.eeg_buffer 取最新 500 samples
    ├─ 前處理: demean → bandpass
    ├─ model.eval() → forward(x_batch)
    ├─ prediction = argmax(outputs)
    ├─ probability = softmax(outputs)
    ├─ 存入 global_value.pred_buffer / prob_buffer
    └─ sleep(PREDICTION_INTERVAL = 0.01s)
```

### model_interface 管理

```
model_interface
  ├─ train_pipeline   ← 全量訓練 (epochs=1, freeze=False)
  ├─ ft_pipeline      ← 校正 (epochs=10, freeze=True, ft=True)
  ├─ rl_ft_pipeline   ← RL 閉環用 (epochs=3, freeze=True, ft=True)
  │
  ├─ train_or_calib(mode, RL_open, RL_interface)
  │     ├─ 資料處理: data_process → dataset
  │     ├─ 訓練: pipeline.run_training() → (val_acc, val_loss)
  │     ├─ 計算: delta_acc, deltaValLoss
  │     ├─ 儲存模型 + 更新 version.json
  │     └─ return session_records
  │
  └─ 版本命名: Trained_Model_vXXX / Calibed_Model_vXXX / RL_Loop_Model_vXXX
```

---

## 四、OSU 介面流程 (`osu/`)

### 檔案結構

```
osu/
  ├─ osu_parser.py      ← .osu 譜面解析
  ├─ osu_interface.py   ← 遊戲邏輯（按鍵判定、Marker 計算、歌曲管理）
  ├─ game_view.py       ← Pygame 繪製（所有 UI 畫面）
  ├─ settings.json      ← 使用者設定
  ├─ songs/             ← .osz 歌曲包（自動解壓）
  └─ sound_effect/      ← 音效檔（hit.mp3）
```

### osu_parser.py

```
parse_osu_map(filepath)
  ├─ 解析 .osu 檔案格式
  ├─ 提取 meta: Title, AudioFilename, 難度等
  └─ 提取 notes[]:
        ├─ lane: math.floor(x * key_count / 512)   ← 軌道 0-3
        ├─ time: 出現時間 (ms)
        ├─ type: 'tap' or 'hold'
        ├─ end: 長按結束時間
        └─ hit: False（初始值）
```

### osu_interface.py — 核心邏輯

```
calculate_marker(map_notes, current_note_index, game_time_ms)
  ├─ 掃描音符列表，找到當前「正在接近判定線」的音符
  ├─ 判斷條件:
  │     ├─ 音符出現: note.time <= game_time + st_to_ed_time
  │     └─ 判定窗口: |note.time - game_time| < 150ms
  └─ return (marker=lane, current_note_index)

check_hit(map_notes, note_index, key, game_time_ms, score, hits)
  ├─ 判定: |note.time - game_time| < 150ms
  ├─ 視覺效果: trigger_effect(key)
  ├─ if lane == key and not hit:
  │     ├─ hit = True
  │     ├─ score += 100, hits += 1
  │     └─ 播放打擊音效
  └─ return (map_notes, score, hits)
```

### game_view.py — UI 畫面

```
GameView (Pygame)
  ├─ draw_main_menu()         ← 主選單（開始/設定/選模型/測模型/退出）
  ├─ draw_account_select()    ← 帳號選擇
  ├─ draw_song_select()       ← 歌曲選擇（難度排序、預覽播放）
  ├─ draw_play_screen()       ← 遊玩畫面（4 軌道、音符下落、分數顯示）
  ├─ draw_pause_menu()        ← 暫停選單（繼續/退出/跳過）
  ├─ draw_intermission()      ← 結算畫面（命中率/每類準確率）
  ├─ draw_model_select()      ← 模型選擇
  ├─ draw_test_model()        ← 測試模型畫面（即時預測結果）
  ├─ draw_settings()          ← 設定畫面（音量/速度/offset/解析度）
  ├─ draw_offset_wizard()     ← Offset 校正精靈
  └─ draw_loading()           ← 讀取中畫面
```


### 按鍵對照規則
左=0, 右=1, 下=2, 上=3


### 設定項 (`settings.json`)

| 設定 | 說明 | 預設值 |
|------|------|--------|
| volume | 音樂音量 | 0.5 |
| sfx_volume | 打擊音效音量 | 0.5 |
| speed | 音符下落速度 | 0.8 |
| offset | 時間偏移 (ms) | 0 |
| fullscreen | 全螢幕 | false |
| resolution | 視窗解析度 | 800 |

---

## 五、RL Bandit 架構（2026/04/09 升級）

### 核心改動
1. **Reward 平滑化**：ΔAcc → ΔLoss（Cross-Entropy Loss 下降量），提供連續可導的學習信號
2. **NeuralUCB**：LinUCB → NeuralUCB（MLP + 梯度 norm 探索項 + Experience Replay）
3. **動作空間解耦**：移除 `ONLINE_UPDATE`，由 `UpdateScheduler`（加權品質分數策略）獨立排程模型更新

### 目錄結構
- `combine_all/RL/bandit.py`：NeuralUCB Bandit + UpdateScheduler
- `combine_all/RL/RL_interface.py`：RL 接口
- `combine_all/EEG_DL/EEG/model_interface.py`：DL 模型接口
- `combine_all/EEG_DL/EEG/EEG_Train.py`：訓練 Pipeline
- `combine_all/EEG_DL/EEG/data_process_np.py`：資料處理 + Bandit 決策

### 設備設定
`bandit.py` 中的 `BANDIT_DEVICE` 變數控制 MLP 使用 CPU 或 GPU。




## 六、RL Pretrain 完整流程 (`RL_pretrain.py`)