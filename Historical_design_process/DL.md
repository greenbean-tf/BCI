# DL（Deep Learning / EEG 模型）歷史設計流程

> 整理範圍：EEG 解碼模型架構、資料前處理、訓練 / Fine-tune / Calibration 流程、模型版本管理。
> 資料來源：git log、`record.py`、Notion 會議記錄、`combine_all/docs/EEG_DL_General.md`、`combine_all/DL/`、`combine_all_new/implementation.md`。

---

## 0. 任務定義

- **輸入**：EEG raw（多通道，500 Hz），sliding window。
- **輸出**：4 類 Motor Imagery（左 / 右 / 下 / 上 = 0/1/2/3）。
- **模型版本系列**：
  - `Trained_Model_vXXX`：離線訓練好的 base model。
  - `Calibed_Model_vXXX`：離線校正後的個人化模型。
  - `RL_Loop_Model_vXXX`：經 RL 線上 update 後的模型快照。

---

## 1. v0：原型期（2026/01/22 ~ 01/26）

### 1.1 起點
- `2026/01/22  Initial commit`：建立 DL 模組骨架。
- `2026/01/26  all in combine`：DL 與 osu 模擬器合併進主程式。

### 1.2 早期決定
- 使用 EEGNet 風格架構：時間卷積 + depthwise spatial conv + separable conv。
- sample rate：500 Hz（Notion 2026/01/23 osu! 會議定案）。
- 暫時用公開資料集 + 自製 demo session 測試前向傳遞。

---

## 2. v1：Predict / Read 改 thread + 時間對齊（2026/01/28）

git：`predict/read thread form`
record.py 2026/01/28：
- 把 **predict** 與 **read** 拆成獨立 thread，避免遊玩主迴圈被 EEG 推論阻塞。
- 浮現問題：thread 間 timestamp 不對齊 → window 取樣偏差。
- 暫解：以 LSL 提供的 `time_correction` 校時，後面 v6 才系統化。

---

## 3. v2：Data load 用 SKF + Calib 完成（2026/01/30 ~ 02/02）

### 3.1 git
- `2026/01/30  data load done, train測試完畢`：data loader 完成。
- `2026/02/02  SKF切, calib完成`：用 Stratified KFold 切 calib data，使 4 類別比例均衡。

### 3.2 record.py 2026/02/02
- calib 流程跑通：calib 歌（Notion 2026/01/30 議定 0/1/2/3 各 4 首）→ 切 SKF → 訓練 → 出 `Calibed_Model_v???`。

### 3.3 設計取捨
- 使用 SKF 而非隨機切：避免某類別在 val set 缺失 → 4 類 acc 評估失真。
- key 數設計為 4 的倍數，方便 SKF。

---

## 4. v3：Fine-tune 改抓「當前 active」模型（2026/02/04）

git：`fine tune model active`
record.py 2026/02/04：

- 之前 fine-tune 寫死讀某個 base model；改成讀「目前 active 的模型」（active 由 settings/UI 指定）。
- 影響：使用者切換到 `RL_Loop_Model_v??` 後再校正，會在那個版本上累積 fine-tune，而不是回到 base。

---

## 5. v4：帳號分離 + 多數決 + JSON 版本（2026/02/11 ~ 02/23）

### 5.1 record.py 2026/02/11
- **多數決**：同一段 EEG 滑動切多個 window，每個 window 推論一次，整段以多數決結果作為「按鍵」。
- 帳號功能上線後，每個帳號的模型 / 資料分開。

### 5.2 git 2026/02/23：`json紀錄version, 多數決`
- 模型 metadata 寫進 JSON：包含 `model_version`、`calib_version`、`channel_list`、`sample_rate`、`window_len`、`stride`。
- 之後 RL 系列也沿用同一格式。

---

## 6. v5：Data process 加入 prob + IMU（2026/02/26）

git：`data process加prob+imu`
record.py 2026/02/26：

- 推論時除了 argmax，也保留整個 softmax `prob`。
- 同時把 IMU 時序也存下，原本想做 motion 對齊的 reward shaping。
- 之後 2026/03/05 motion 部分被捨棄（仍保留紀錄）。

---

## 7. v6：Window 150ms + 棄 motion（2026/03/05）

record.py 2026/03/05：
- check window 改成 **150ms**，原本更大 → 推論延遲偏高、判定脫節。
- **已捨棄 motion 相關**：DL 只看 EEG。後面 RL 的 reward 也不再看 motion。

設計影響：
- 模型 input window 變短 → 必須提高時間卷積的 receptive field 利用率（用更多 kernel size 組合）。
- 為 EEGNet_our_ver 的 multi-kernel 鋪路。

---

## 8. v7：RL 閉環架構成形（影響 DL 部分）（2026/03/06）

record.py 2026/03/06（RL 閉環想法）：
```
song1 → train → use → calib delta acc reward
```
- DL 必須支援 **小批量 online update**：每首歌結束後可以拿 buffer 內的樣本做 mini fine-tune。
- 為此 DL 介面新增 `online_update(samples, weights)`，接受 RL 提供的樣本 + 權重。

---

## 9. v8：Multi-thread 對齊 + 維度修正（2026/03/08 ~ 03/11）

- `2026/03/08  multi-thread檢查`：DL 推論 thread、資料 logger thread、RL update thread 之間的 queue 對齊；統一以 `(t_start, t_end, eeg_window, label?, prob?)` 作為基本資料結構。
- `2026/03/11  EEG_Read維度問題`：EEG raw 維度 `(channel, sample)` vs 期望 `(batch, channel, sample)` 不一致；統一在 `eeg_reader` 出口 reshape，下游 DL / RL 都拿到 `(B, C, T)`。

---

## 10. v9：test_model 介面（2026/03/12）

git：`test_model介面, parser修正`

- 加入離線測試介面：給定 raw EEG + 譜面 → 輸出 acc / per-class acc / loss / prob 曲線。
- 是後面 RL pretrain（2026/04/15 ~ 17）能跑 episodic offline 模擬的基礎。

---

## 11. v10：File structure 大整理 + EEGNet_our_ver（2026/04/08 ~ 04/09）

### 11.1 git
- `2026/04/08  file structure更新`
- `2026/04/09  軌道標示修正, RL改神經網路架構`

### 11.2 EEGNet_our_ver
（依 `combine_all/docs/EEG_DL_General.md`）核心特徵：
- **Multi-kernel temporal conv**：多個 kernel size 並行（捕捉不同頻段節奏）。
- **Depthwise spatial conv**：每條時間 channel 單獨學空間 filter。
- **Separable conv**：減少參數量。
- **BatchNorm + ELU + Dropout**。
- 輸出 4 類 logits。

### 11.3 BraindecodeTrainer
- 統一訓練 pipeline：負責 fit / fine-tune / online_update / save / load。
- 接受 `weights` 參數，給 RL 後續做加權樣本訓練（HIGH_WEIGHT=2.0、INCLUDE=1.0、LOW_WEIGHT=0.3）。

---

## 12. v11：RL_pretrain DL 介面（2026/04/15 ~ 04/17）

### 12.1 record.md（`combine_all/RL/RL_pretrain/record.md`）
- DL 必須支援 **episodic** 形式：在 RL pretrain 時，DL 暴露 `forward_loss(window, label)` 給 bandit 計算 reward。
- DL 同時提供 `entropy / margin / alpha_power` 等中間量，給 bandit 取 context。

### 12.2 Notion 2026/04/17 wei 會議
- 「**DL acc > 85% 停止 update RL**」：DL 端要持續回報目前 rolling acc；當穩定超過 85%，主迴圈停止觸發 RL 的 update。
- 「update 從小 trial 累積」：DL 的 online_update 一次只吃少量 trial，逐步逼近，避免 catastrophic forgetting。

---

## 13. v12：EWMA Early Stopping（offline pretrain only）

record.py 2026/05/07：
- **EWMA early stopping**：以 reward 的指數加權移動平均判斷收斂，**僅 offline pretrain 用**。
- 線上 RL 不用 EWMA stop（線上要持續適應 distribution shift）。

DL 端的影響：
- offline 訓練 trainer 加入 `EarlyStopperEWMA(alpha, patience)` callback。
- 線上 fine-tune 改採固定步數 + RL 控制是否進行（DISCARD / INCLUDE / HIGH / LOW）。

---

## 14. v13：Refactor band_power（2026/05/04）

git：`refactor band_power`

- 把 EEG 頻段功率計算獨立成 `band_power.py`：給 DL 做特徵 + 給 RL 取 context（α power）。
- 統一 FFT / Welch 流程，避免兩邊不一致。

---

## 15. v14：Log_class + 刪 other/（2026/05/06）

git：`delete other/, log_class`

- DL 訓練 / 推論的 log（loss、acc、prob、wall time）寫進統一的 logger。
- `other/` 移除：早期 motion / skeleton 試驗模型徹底退場。

---

## 16. v15：combine_all_new 模組化（2026/05/06 ~ 05/07）

依 `combine_all_new/implementation.md`，DL 模組化拆分：

| 模組 | 對應功能 |
|------|----------|
| `dl/model.py` | EEGNet_our_ver 定義 |
| `dl/trainer.py` | BraindecodeTrainer（fit / fine_tune / online_update） |
| `dl/preprocess.py` | demean、bandpass 1–40Hz、sliding window |
| `dl/band_power.py` | α / β / θ band power |
| `dl/io.py` | 模型 save / load + JSON metadata |

主程式只透過介面呼叫，不直接碰實作。

---

## 17. 訓練 / 校正 / 線上 update 完整流程（v15 為止）

```
[Calibration session]
calib 歌 (16 首) → SKF 切 → BraindecodeTrainer.fit()
       ↓
   Trained_Model_vN (base model)
       ↓
   個人化 fine-tune (active model)
       ↓
   Calibed_Model_vN

[Gameplay session]
EEG raw → demean + bandpass(1–40Hz) → sliding window
       → EEGNet_our_ver.predict() → softmax prob
       → 多數決 → 按鍵
       (同時把 prob / entropy / margin / α power 給 RL)

[Per-song RL update]
Buffer (500 samples) + bandit 給的 weights →
BraindecodeTrainer.online_update()
       ↓
   RL_Loop_Model_vN
```

---

## 18. 重要設計取捨總表（DL）

| 取捨 | 結果 | 日期 |
|------|------|------|
| Sequential vs Threaded predict | Thread + LSL 校時 | 2026/01/28 |
| Random vs SKF split | SKF | 2026/02/02 |
| Fine-tune base vs active | active | 2026/02/04 |
| Window 大小 | 150ms | 2026/03/05 |
| Motion / IMU 是否進 DL | 不進 | 2026/03/05 |
| 模型架構 | EEGNet_our_ver（multi-kernel） | 2026/04/09 |
| Online update | 接受 RL 加權樣本 | 2026/04/15~17 |
| Early stopping | 僅 offline pretrain 用 EWMA | 2026/05/07 |
| 程式組織 | combine_all → combine_all_new 模組化 | 2026/05/06~07 |

---

## 19. 與 RL / UI 的介面整理

- **對 RL**：暴露 `forward_loss / prob / entropy / margin / band_power`，並接受 `online_update(samples, weights)`。
- **對 UI**：暴露 `current_model_version / current_acc / per_class_acc`，UI 顯示與選模型用。
- **對 logger**：所有 fit / fine-tune / online_update 都會寫一條紀錄（model 版本、樣本數、loss 變化）。

---

> 補充：詳細模型超參數請看 `combine_all/docs/EEG_DL_General.md`；訓練腳本與 callback 在 `combine_all/DL/` 與 `combine_all_new/dl/` 下。
