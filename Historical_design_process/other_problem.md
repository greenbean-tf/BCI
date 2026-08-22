# Other Problems：周邊與系統面歷史問題與設計流程

> 整理範圍：EEG 帽 / Cygnus 放大器、VR 頭盔、LSL 串流、TCP server-client、Multi-thread、時間對齊、IMU / motion、skeleton detection、帳號管理、log 系統等「不是 UI / DL / RL 主軸」但影響整個專題運作的問題。
> 資料來源：git log、`record.py`、Notion 會議記錄（含「流程預想」、「連結記錄」）、`combine_all/docs/`、`combine_all_new/implementation.md`、Google Drive 作品說明書 ver2。

---

## 1. EEG 帽 / Cygnus 放大器

### 1.1 角色（Notion 流程預想）
- 使用者戴 EEG 帽 → 接 **Cygnus** 放大器 → 經 LSL 串流 → 主機端 `eeg_reader` thread 收。
- sample rate 500 Hz（Notion 2026/01/23 osu! 會議）。

### 1.2 遇到的問題
- **電極阻抗 / 接觸**：在 calib 之前要量 Cygnus 上的阻抗，否則 EEG 全是 noise → 後來把「先量阻抗再進系統」寫進 SOP（record.py 2026/04/11）。
- **裝置開機順序**：EEG → 放大器 → 主程式，順序錯會導致 LSL 拿不到 stream。
- **連線掉包**：見 §3 LSL 區塊。

### 1.3 與 BCI 主流程的接口
- DL 端固定接受 `(channel, sample)` 的 raw EEG，channel list / 排序由 JSON metadata 紀錄。

---

## 2. VR 頭盔

### 2.1 構想（Notion 流程預想 / 連結記錄）
- 「VR-BCI」：把 osu!mania 移到 VR，使用者沉浸度更高、MI 想像更專注。
- 連結記錄裡有學長之前的 VR_BCI 參考連結、簡報、海報、Canva 流程圖。

### 2.2 為什麼最後 demo 沒走 VR
- 開發優先順序：EEG + DL + RL 閉環是必要功能；VR 是加分項。
- 同時 VR 會牽動 frame timing / motion sickness / EEG artifact 等大量未知。
- 結果：VR 元件保留在「下一階段 / 加值情境」，本專題版本以螢幕版 osu! 為主。

### 2.3 殘留的影響
- 譜面 / 流程設計都保留可移植性（`game_view` 可換成 VR scene），所以未來轉 VR 仍可重用。

---

## 3. LSL（Lab Streaming Layer）串流

### 3.1 用途
- 把 EEG（含 IMU）從採集端推到主機端 `eeg_reader`，自帶時間同步機制。

### 3.2 重大事件
- **2026/01/28**（record.py）：predict / read 改 thread 後，**時間不對齊** → 用 LSL `time_correction()` 補正。
- **2026/03/11**（`EEG_Read 維度問題`）：reader 出口維度修正後，下游才正確 reshape。
- **2026/05/07**（record.py）：lsl 連線問題：偶發 packet drop / 重連後 stream id 改變，主程式必須容忍 reconnect。修正後遊玩頁不再因 LSL 斷線而停頓。

### 3.3 設計取捨
- 主時鐘採 LSL clock，UI / 譜面時間皆透過 mapping 換算。
- 出現 drop 時 reader 會插入 NaN 段並通知 logger，DL 收到後跳過該 window。

---

## 4. TCP Server-Client（thread 間通訊）

### 4.1 用途
依 `combine_all/docs/`（Game.md / EEG_DL_General.md / RL.md）：
- `eeg_reader` ↔ `eeg_predictor` ↔ `data_logger_worker` ↔ `rl_loop_worker` ↔ `game_view` 之間的訊息傳遞。
- 多採 TCP local socket / queue 抽象。

### 4.2 問題
- **2026/03/08（multi-thread 檢查）**：thread 間 queue 沒對齊，導致某些 thread 餓死或重複拉資料。一次性梳理之後底定。
- **死鎖**：早期 lock 順序不一致，2026/03/08 重排後解決。

---

## 5. Multi-thread / 時間對齊

### 5.1 主要 thread
- `eeg_reader`：從 LSL 拉 raw。
- `eeg_predictor`：跑 DL 推論。
- `data_logger_worker`：寫 disk（raw、prob、IMU、按鍵、judgement、reward）。
- `rl_loop_worker`：跑 bandit 決策、update scheduler、online_update。
- `game_view`：UI 主迴圈。

### 5.2 時間對齊
- 所有事件以 LSL clock 為主軸，記錄 `(t_start, t_end)`，再對應到 osu marker 與 UI frame。
- 早期把推論時間和畫面時間混用 → judgement 偏差；2026/03/05 把 check window 改 150ms 並統一時間軸後解決。

### 5.3 Logging
- `2026/05/06  delete other/, log_class`：把所有 thread 的 log 接到一個 logger class，避免分散。

---

## 6. IMU / Motion 訊號

### 6.1 構想（2026/02/26）
- `data process加prob+imu`：當下想用 IMU 做：
  - 動作雜訊偵測（手抖 → EEG artifact）。
  - reward shaping（沒動作 = 純 MI）。

### 6.2 為什麼捨棄（2026/03/05）
- record.py 2026/03/05：「**已捨棄 motion 相關**」。
- 原因：
  - 動作分數雜訊大、不穩定。
  - 與 BCI 訴求衝突（理想是純想像，不應依賴肢體動作）。
  - reward 改用 -ΔLoss 後不再需要 motion。
- 結果：IMU 仍記錄但不進 DL / RL 主流程。

---

## 7. Skeleton Detection（2026/03/31）

git：`try skeleton detection`

- 嘗試在主程式接 webcam 跑骨架偵測，當 motion 訊號替代品。
- 結果：
  - 對 BCI 概念衝突（純想像）。
  - 環境光與 webcam 品質影響大。
- 處置：保留在 `other/` 子資料夾；2026/05/06 `delete other/, log_class` 一併移除。

---

## 8. 帳號管理（2026/02/11 ~ 02/23）

### 8.1 動機（record.py 2026/02/11）
- 多人測試需要：每人 calib data / 模型 / 紀錄分離；切換時要乾淨。

### 8.2 實作
- `user_data/<account_name>/` 下放：
  - `calib/` 校正 raw + label
  - `models/` 含 Trained / Calibed / RL_Loop 多版
  - `logs/` 遊玩紀錄、bandit 決策
  - `settings.json` 個人化設定（之後共用全域 + 帳號覆寫）
- 模型 metadata 寫進 JSON（`json紀錄version, 多數決`，2026/02/23）。

---

## 9. 模型版本管理（橫跨 DL / RL，但屬系統面）

- 命名約定：
  - `Trained_Model_v???`：base 訓練。
  - `Calibed_Model_v???`：個人化校正。
  - `RL_Loop_Model_v???`：RL 線上 update 後快照。
- 每個模型一個 JSON metadata：channel list / sample rate / window / stride / 父版本 / 訓練時樣本數。
- UI 模型選擇頁與大廳依此顯示。

---

## 10. File Structure 重組（2026/04/08 ~ 05/07）

### 10.1 2026/04/08（file structure 更新）
- 把零散檔案整理進 `combine_all/{osu, DL, RL, utils, docs}`。
- 文件入口：`combine_all/docs/Game.md`、`EEG_DL_General.md`、`RL.md`。

### 10.2 2026/05/06 ~ 05/07（combine_all_new）
- 切到模組化 package：依 `combine_all_new/implementation.md` 表格分模組。
- 主程式只做組裝，不直接動內部實作。

---

## 11. 鍵重疊（同 frame 多軌觸發）

- 2026/01/28（record.py）發現：predict / read thread 化後，同 frame 多軌觸發時畫面與判定錯亂。
- 暫解：同一 frame 只取一鍵（drop 其它）。
- 後續：靠多數決 + window 150ms（2026/03/05）緩解；判定改以時間視窗而非 per-frame。

---

## 12. 鍵消失動畫（2026/02/06）

- 之前 note 觸發後直接消失，使用者搞不清楚有沒有打到。
- 改成「擊中閃光 + 上飄淡出 + hit.mp3」。
- 同時為「sfx_volume」設定欄位鋪路。

---

## 13. EWMA Threshold 僅 offline 用（2026/05/07）

record.py 2026/05/07：
- EWMA-based early stopping 只用於 offline pretrain。
- 線上不能 early stop（會錯過 distribution shift 的更新時機）。
- 此規則同時影響 DL trainer 與 RL pretrain。

---

## 14. 資料 / 隱私 / 競賽相關（Google Drive）

- Google Drive `專題/` 含：經濟部競賽資料、`20260507_update_per_song_Liang.zip`（46.8 MB，per-song 更新試驗）、`專題.drawio`。
- `經濟部競賽/` 下 `作品說明書_ver2.docx` 已部分讀取：
  - 第 1 節 創意發想背景。
  - 第 5 節 醫療 B2B / B2C 應用。
  - 第 6 節 結語。
- 競賽相關內容對 BCI 系統設計沒有強制限制，只影響「對外簡報 / 商業敘事」這一層；未列入主程式設計流程。

---

## 15. 已捨棄項目總表

| 項目 | 何時捨棄 | 原因 |
|------|----------|------|
| Motion / IMU 進主流程 | 2026/03/05 | 訊號雜、與 BCI 概念衝突 |
| Skeleton detection | 2026/03/31 後最終於 2026/05/06 移除 | 同上、環境依賴大 |
| VR 頭盔（本版） | 全期保留為下階段選項 | 開發優先序、未知變數多 |
| Bandit 自觸發 ONLINE_UPDATE 動作 | 2026/04 後期 | 與 episodic reward 設計衝突 |
| `other/` 資料夾 | 2026/05/06 | 集中試驗碼，整體刪除 |
| Online EWMA early stopping | 2026/05/07 確認 | 不利線上適應 distribution shift |

---

## 16. 系統運作 SOP（record.py 2026/04/11 為基礎，後續微調）

```
1. 戴 EEG 帽，量 Cygnus 阻抗
2. 啟動 LSL → 主程式
3. 選帳號 → 進大廳
4. (首次或漂移) 校正流程：
   - 16 首 calib 歌（0/1/2/3 各 4 首）
   - SKF 切 → BraindecodeTrainer.fit → Calibed_Model_v??
5. 大廳：選歌 / 選模型 / 設定 / 開始
6. 遊玩：EEG → DL prob → 多數決 → 按鍵
7. 結束：
   - 寫紀錄
   - bandit 對每個 trial 取 context / action
   - UpdateScheduler 累積 score；達 15.0 觸發 online_update → RL_Loop_Model_v??
   - 計算 -ΔLoss → 給 bandit 當 reward
   - 若 DL rolling acc > 85% 停止後續 update
8. 結算頁顯示 acc / per-class acc / 是否觸發 update
```

---

## 17. 後續仍開放的議題

- VR 版本完整實作（連 motion sickness 對 EEG artifact 的影響都需重做實驗）。
- Cygnus 失聯時的 fail-safe（目前是 reader 重連）。
- 多人連線 / 跨裝置同步（目前單機）。
- 經濟部競賽 B2B / B2C 落地對應的資料隱私 / 加密層（作品說明書 ver2 第 5 節有提到）。

---

> 本文件補齊 UI / DL / RL 三個主軸以外的工程與系統面演化。對應到 git 的具體提交，可在主資料夾 `git log` 中以日期反查。
