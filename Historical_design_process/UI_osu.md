# UI / osu! 介面 歷史設計流程

> 整理範圍：osu!mania 風格的 4-class MI 校正介面、選歌大廳、選帳號 / 選模型流程、設定頁、暫停 / 結算頁。
> 資料來源：git log（commit 訊息與檔案異動）、`record.py`（每日進度筆記）、Notion「專題小組」會議記錄、`combine_all/docs/Game.md`、`settings.json`。

---

## 0. 整體流程預想（Notion「流程預想」頁）

最初規劃的單次使用流程：

```
戴上 EEG → cygnus 校正 → LSL 串流 →
calib (校正歌)→ train Trained_Model →
進入大廳 → 選歌 → 遊玩 (預測+回饋) →
calib_RL (RL 線上校正) → 產生 new_calib_model →
回大廳
```

UI 部分要支援的關鍵頁面：
1. **登入 / 帳號選擇**：對應 user 資料夾。
2. **大廳 (Lobby)**：歌單列表 + 選歌 + 模型選擇 + 設定按鈕。
3. **遊玩頁 (Game View)**：4 軌 osu!mania，note 從上落下。
4. **暫停頁 (Pause)**：可繼續 / 離開。
5. **結算頁 (Result)**：本次 acc、各 class acc。
6. **設定頁 (Settings)**：音量、speed、offset、解析度、全螢幕。

---

## 1. v0：原型介面（2026/01/22 ~ 01/23）

### 1.1 git 起點
- `2026/01/22  Initial commit`：建立專案骨架。
- `2026/01/23  OSU simple simulator`：第一版可動的 osu!mania 模擬器，僅單機鍵盤輸入測試。

### 1.2 設計討論（Notion 2026/01/23 osu! 會議）
- **歌曲新增方法**：先以手刻譜面為主，之後再考慮自動譜面生成。
- **4 分類介面**：保留 osu!mania 4K 形式，4 軌分別對應 MI 的 4 類（左 / 右 / 下 / 上 = 0/1/2/3）。
- **sample rate**：500 Hz（EEG）。
- **osu marker**：在落鍵時間點打上 marker，用來和 EEG 串流時間軸對齊。

### 1.3 此版本 UI 特徵
- 純鍵盤輸入，沒有 EEG 接入。
- 沒有大廳 / 設定 / 結算頁。
- 譜面以最簡 JSON 表示，純粹用來驗證 note 落下與時序。

---

## 2. v1：合併進主程式 + 介面微調（2026/01/26）

git：`all in combine, 介面微調`

- 把獨立的 osu simulator 合併進主程式（後來成為 `combine_all/`）。
- 開始有「介面微調」這類提交：調整軌道寬度、note 顏色、判定線位置。
- 仍是 standalone 遊玩，沒有 EEG 預測寫進畫面。

---

## 3. v2：Predict / Read 改 thread + 鍵盤重疊問題（2026/01/28）

git：`predict/read thread form`
record.py 2026/01/28：
- 把 `predict` 與 `read` 從序列流程改成 **獨立 thread**，避免遊玩時被 EEG 推論卡住。
- 發現 **時間不對齊** 問題：thread 各自跑時 timestamp 飄移。
- 發現 **鍵重疊** 問題：同一 frame 多軌觸發時畫面與判定錯亂。

UI 影響：
- 落 note 與判定流程改寫成「事件時間 + thread 共享 queue」。
- 當下還沒解決鍵重疊，先以「同一 frame 只取一鍵」暫解。

---

## 4. v3：Calibration 介面成形 + 譜面條件（2026/01/30 ~ 02/02）

### 4.1 譜面條件（Notion 2026/01/30 osu! 會議）
- **calib 歌曲**：每一類（0/1/2/3）各 4 首，共 16 首做為校正歌。
- **key 數**：4 的倍數，便於 SKF（Stratified KFold）切資料。
- **節奏**：避免相鄰 note 過密，留出 EEG window 空間。

### 4.2 git
- `2026/01/30  data load done, train測試完畢`
- `2026/02/02  SKF切, calib完成`
- record.py 2026/02/02：calib 流程跑通，可進到「校正 → 訓練 → 開始遊玩」。

### 4.3 UI 影響
- 加入 **校正模式** 子頁面：依序播放校正歌，畫面上提示「請專心想像〇方向動作」。
- 校正結束後彈出「訓練中…」提示，等模型存檔。
- 遊玩頁開始顯示「目前使用模型版本」。

---

## 5. v4：Fine-tune 切換到當前 active 模型（2026/02/04）

git：`fine tune model active`
record.py 2026/02/04：fine tune model 改成讀「當前 activate」模型（而不是固定路徑）。

UI 影響：
- 大廳開始有「當前模型」狀態顯示。
- 之後 v6 模型選擇頁的雛形。

---

## 6. v5：鍵消失更新（2026/02/06）

git：`key消失更新`

- 重做 note 觸發後的消失動畫，之前是直接消失，會看不出「自己有打到」。
- 改成「擊中時亮一下 + 上飄淡出」。
- 同時補上 hit.mp3 音效（後續才在設定頁加入 sfx_volume）。

---

## 7. v6：帳號 + JSON 紀錄 + 多數決（2026/02/11 ~ 02/23）

### 7.1 record.py 2026/02/11
- **新增帳號功能**：每位使用者一個資料夾，校正資料 / 模型 / 紀錄分開存放。
- **多數決**：同一 window 多個推論結果以多數決決定按鍵。

### 7.2 git
- `2026/02/23  json紀錄version, 多數決`：版本資訊（model version、calib version）寫入 JSON，給介面 + 訓練流程共用。

### 7.3 UI 影響
- 啟動時新增 **登入 / 選帳號** 頁面，列出 `user_data/` 下所有帳號。
- 大廳右上角顯示帳號 + 模型版本。
- 新增 model 選擇頁：列出 `Trained_Model_vXXX` / `Calibed_Model_vXXX` / `RL_Loop_Model_vXXX`。

---

## 8. v7：Data process 加 prob + IMU（2026/02/26）

git：`data process加prob+imu`
record.py 2026/02/26：data process 加入推論 `prob`（softmax 機率）與 `imu`（動作感測），存到每首歌的紀錄裡。

UI 影響：
- 結算頁開始可以畫出「每軌平均機率」、「動作分數隨時間」的曲線（這個之後在 2026/03/05 被部分捨棄）。

---

## 9. v8：Check window 150ms + 捨棄 motion 顯示（2026/03/05 ~ 03/06）

### 9.1 record.py 2026/03/05
- check window 改成 150ms（原本更大），讓判定更貼近視覺感。
- **已捨棄 motion 相關**：不再把 IMU 動作分數當主要 reward / 判定依據（保留紀錄但不再顯示在主介面）。

### 9.2 record.py 2026/03/06（RL 閉環想法的 UI 影響）
- 流程改成：
  ```
  song 1（zero song / 假回饋）→ train →
  use → calib delta acc reward → next song
  ```
- UI 上：每首歌結束後顯示「相對前一首的 ΔAcc」（這個概念之後 2026/04/15 改成 ΔLoss）。

---

## 10. v9：多 thread 檢查 + 維度問題（2026/03/08 ~ 03/11）

- `2026/03/08  multi-thread檢查`：把 `eeg_reader / eeg_predictor / data_logger_worker / rl_loop_worker / game_view` 之間的 queue 重新對齊，避免 deadlock。
- `2026/03/11  EEG_Read維度問題`：EEG raw 維度 (channel, sample) 與遊戲端期望不一致，修正後遊玩頁的「即時頻譜 / alpha power」才能正確繪製。

UI 影響：
- 遊玩頁右側可選擇是否顯示「即時 EEG 條」（debug 用）。
- 結算頁的曲線改用統一的時間戳。

---

## 11. v10：test_model 介面 + parser 修正（2026/03/12）

git：`test_model介面, parser修正`

- 加入 **test_model** 子頁：可獨立載入一個模型對單首歌做離線推論，檢查模型表現，不寫進使用者紀錄。
- `osu_parser.py` 修正：先前在連續同軌 note 過密時會漏 parse 一個 note，導致畫面與判定不一致。

---

## 12. v11：game_view 更新（2026/03/27）

git：`update game_view`

- 軌道視覺重做：判定線改實線、note 形狀拉長、加入命中閃光。
- 為後面的「軌道標示修正」鋪路。

---

## 13. v12：嘗試 skeleton detection（2026/03/31）

git：`try skeleton detection`

- 嘗試在 UI 旁邊加上骨架偵測作為 motion 替代訊號。
- 後續因為效果不顯著＋與 BCI 概念衝突，最終 **沒有納入正式版**（紀錄保留在 `other/` 子資料夾，後來 2026/05/06 整個 `other/` 被刪除）。

詳見 `other_problem.md`。

---

## 14. v13：File structure 大整理（2026/04/08）

git：`file structure更新`
record.py 2026/04/08：file structure 重新排版，UI 對應的資料夾也跟著調整：

- `combine_all/osu/`：osu_interface、osu_parser、game_view。
- `combine_all/utils/`：跨頁面的小工具（音量、設定讀寫）。
- `combine_all/RL/`、`combine_all/DL/`：和 UI 解耦。

同時 record.py 2026/04/08 提到「**zero song 假回饋**」：第一首歌用假回饋讓系統先跑通流程，UI 上顯示「校準中…請專心遊玩」。

---

## 15. v14：軌道標示修正（2026/04/09）

git：`軌道標示修正, RL改神經網路架構`

- **軌道標示修正**：4 軌的「左 / 下 / 上 / 右」對應字 / 顏色重排，避免使用者混淆 MI 想像方向。
- 同一天 RL 改成 NeuralUCB，UI 上模型版本顯示加上 `RL_Loop_Model_vXXX` 命名（詳見 `RL.md`）。

---

## 16. v15：大更新前夕 + RL_pretrain 介面整合（2026/04/11 ~ 04/17）

### 16.1 git
- `2026/04/11  大更新前`：UI 與訓練流程互動方式凍結待整理。
- `2026/04/15  RL_pretrain更新`、`2026/04/17  RL_pretrain更新`。

### 16.2 record.py 2026/04/11（系統運作 SOP）
總結整個 UI 流程：
1. 啟動 → 選帳號 → 進大廳。
2. 大廳：選歌 / 選模型 / 設定 / 開始。
3. （若選 calib）→ 校正流程 → 訓練 → 自動回大廳。
4. （若選遊玩）→ 遊玩頁 → 暫停可離開 → 結算頁。
5. 結算頁：顯示 acc、各 class acc、是否觸發 RL 線上 update。

### 16.3 Notion 2026/04/17 wei 會議對 UI 的指引
- 「DL acc > 85% 停止 RL update」：UI 上會顯示「目前 acc 已達穩定，RL 進入觀望」。
- 「update 從小 trail 累積」：結算頁加入 RL buffer 累積進度條。

---

## 17. v16：log_class + 刪除 other/（2026/05/06）

git：`delete other/, log_class`

- 整理 `other/` 內遺留的試驗 UI（包含 skeleton、VR、Cygnus 試驗），全部移除以保持主程式乾淨。
- `log_class` 把所有 UI 事件（按鍵、頁面切換、模型載入）統一寫進 logger，方便事後回放遊玩過程。

---

## 18. v17：combine_all_new 模組化（2026/05/06 ~ 05/07）

git：`combine_all_new update`、`lsl修正`

- 從 `combine_all/`（單一大 package）切到 **`combine_all_new/`** 模組化版本，UI 對應檔案：
  - `combine_all_new/ui/`：純 UI（pygame 畫面）。
  - `combine_all_new/io/`：LSL / TCP 連線。
  - `combine_all_new/runtime/`：thread 控制。
- record.py 2026/05/07：lsl 連線問題修正後，遊玩頁不再因 LSL 掉包而停頓。

---

## 19. 設定頁（settings.json）

`settings.json` 欄位（v17 為止）：

| 欄位 | 用途 |
|------|------|
| `volume` | 主音量（音樂） |
| `sfx_volume` | 音效音量（hit.mp3 等） |
| `speed` | note 落下速度倍率 |
| `offset` | 音畫 / 判定偏移（ms） |
| `fullscreen` | 是否全螢幕 |
| `resolution` | 視窗解析度 |

設定頁可從大廳進入；修改後即時寫回 `settings.json`，下次啟動讀取。

---

## 20. 重要設計取捨總表

| 取捨 | 結果 | 日期 |
|------|------|------|
| 是否做 motion / IMU 視覺化 | 捨棄主介面顯示，僅保留紀錄 | 2026/03/05 |
| 是否做 skeleton detection | 試做後捨棄 | 2026/03/31 |
| ΔAcc vs ΔLoss 顯示 | 改 ΔLoss | 2026/04/15 之後 |
| `other/` 試驗 UI | 全部刪除 | 2026/05/06 |
| 單體 vs 模組化 | 改 `combine_all_new` 模組化 | 2026/05/06~07 |

---

## 21. 目前架構（參考 `combine_all/docs/Game.md`）

- **進入點**：`main.py` → 帳號選擇 → 大廳。
- **GameView**：核心遊玩畫面，使用 pygame；訂閱 `eeg_predictor` 的推論結果作為「按鍵」。
- **OsuParser**：讀 `.osu` 譜面，輸出時間排序的 note list。
- **OsuInterface**：把 note + 推論結果 + 鍵盤輸入 → 判定 → 畫面。
- **PauseDialog / ResultDialog**：暫停與結算。
- **SettingsDialog**：讀寫 `settings.json`。

---

> 補充：每一版的 commit 對應的具體檔案異動可在 `git log -- combine_all/osu/*` / `git log -- settings.json` 取得；本文件以「設計流程」為主軸，commit 細節僅在關鍵節點列出。
