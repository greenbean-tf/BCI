# 開發紀錄 (Development Record)

## 2026/01/25
- setting.json 的路徑改成變數
- BCIModelInterface 改位置
- raw_eeg 用途?

## 2026/01/27
- records 資料夾可刪除 (已刪)
- log.txt 檔名還沒改

## 2026/01/28
- predict 跟 read 已改成 thread 形式
- 但仍有時間不對齊問題
- 鍵有重疊問題，所以才會有有按到但 acc 沒 + 的情況 → 透過改譜避免
- train 還沒做 → train 時要中斷 predict → train 的時候本來就不會 predict, predict 跟 read 只有在遊玩中
- 選 model 頁面也還沒做 (更 1/29 已完成)

## 2026/01/29
- 選 model 頁面已完成，名稱需美化 → 2026/02/02 已美化完成
- 遊玩介面(線)跑掉
- hit.mp3 沒出來 → 已修正
- 暫停介面偏移 → 已修正
- 鍵按下沒消失

## 2026/01/30
- data load 已完成 → 目前是全 load → train，要實作 load 最新 → fine tune
- train 測試完畢，目前可跑
- 介面需優化彈出 acc
- 時間偏移尚未搞定

## 2026/02/02
- data load 是用 SKF 切的
- calib 應該完成了
- 全部改 thread，時間對齊問題好像改好了? 好像又還沒
- 找到問題了，是切頁面的影響，只要完全不切頁面就沒事

## 2026/02/03
- 要 fine tune 的 model 要改成當前使用的，尚未改 → 02/04 已改，可以從 log 確認是哪個 model calib 來的，ver 只會往上加
- log_dir 做好了

## 2026/02/04
- fine tune 的 model 已改成當前 activate
- 修正大多數路徑問題，TCP client 已傳 training msg，看介面怎麼運用
- 時間對齊問題，大多數是切頁面的影響，小機會不切頁面會發生 → 再看要不要修正

## 2026/02/06
- key 消失已更新
- LSL 接好還沒測試，要 connect 才能測試
- 速度計算應該差不多好了，在 osu_interface 的 calculate_marker 裡
- speed 0.1 還是有點快，可能要再改 speed 下限

## 2026/02/09
- user 帳號名稱管理 model 待實作
- 更新速度計算後，判斷會改時間 → 待改
- esc 問題修正

## 2026/02/11
- model 判斷規則待改 → 目前是以當前結果，看要不要改多數決
- 新增帳號功能
- data 的路徑修好了
- train 的還沒

## 2026/02/13
- 可能要用額外 json 紀錄當前 version

## 2026/02/23
- 可能要用額外 json 紀錄當前 version → 應該好了?
- pred 改用多數決
- 時間 key 間隔問題應該解決了?
- 處理 calib 那首，train 的流程
- pause skip 好像不會終止 thread?
- 測試 model 性能頁面還沒做

## 2026/02/24
- 新紀錄 prob → 但存取結果有問題 → 02/25 問題解決
- pred 改回單一決定
- data process 要重新處理

## 2026/02/25
- data process 重處理 ing

## 2026/02/26
- data process 重處理 ing，加入了 prob 跟 imu

## 2026/02/27
- last acc 要怎麼改? → 03/05 已處理完成

## 2026/03/04
- train 拿全部資料，不走 RL → 預計是第一首用，所以全部也只有第一首的 data
- calib 只取剛完的一首，有 RL → 之後的歌都走這裡
- 要把 train 跟 calib 移到 server 上做的樣子
- `prepare_datasets_v2` 過了
- `prepare_datasets_RL_v1` 還沒測

## 2026/03/05
- check window 改成 150 ms
- first song 要怎麼處理?
- time cost 尚未改成取決於歌曲長度
- 已捨棄所有 motion 相關
- 目前到 training model 前面的部分應該都過了
- training 也過了，calib 還沒測
- parser 還有問題
- test 曲也有問題 → 已修正
- 好像有些會有 key 看不到當前判斷鍵的問題，不確定是不是來不及到 → 已修正完成

## 2026/03/06

### RL 閉環想法

1. song 1 data → train first model → use first model play next song
2. when playing next song (closed loop):
   - keep calib first model by song 1 data
   - get reward (delta acc) update RL [in background]

- 要把 train 跟 calib 移到 server 上做的樣子 → 暫緩，先用柏安的 GPU 代替
- RL loop model 可能會有 multi thread 搶佔問題
- 路徑尚未測試
- 其他閉環應該是好的?
- song 之間更新尚未確定是否使用

## 2026/03/08
- RL loop model 可能會有 multi thread 搶佔問題 → 貌似沒問題? 不知道為啥
- 新增一些 print 資訊
- 如果最後一次 RL loop 還沒結束，要先等結束再 calib 下一個

## 2026/03/11
- first test 修正
- EEG_Read 模擬沒問題
- 但實際好像有維度問題? 或是 thread 問題，CSV 資料出來都是 0
- `pylsl` 取得的 sample 為一維 list，需轉為二維 `(N, 1)` 才可與 `buffer_ori_data` 進行 hstack — 待測試

## 2026/03/12
- test model 會存 CSV → 要改掉
- 修正 parser 解析成變動，方向問題

## 2026/04/08
- zero song 現在是用假回饋
- 先嘗試 pretrain RL

## 2026/04/11

### 系統運作標準流程 (SOP)

1. **資料收集與 RL 決策：** 遊玩一首新歌，產出 Raw Data。RL 根據 Context 決定哪些資料要增強、保留或丟棄（Actions）。
2. **混合緩衝區 (Buffer Mixing)：** 將 RL 認可的新資料，與歷史優質資料混合。
3. **模型微調 (Calibration)：** 將混合資料餵給目前的 $DL_{current}$，經過少量 Epochs 的微調，產出 $DL_{new}$。
4. **計算與更新 (Reward & Update)：** 在固定的、乾淨的驗證集上計算：$Reward = Loss(DL_{current}) - Loss(DL_{new})$。將這筆 Reward 分配給步驟 1 中 RL 做的那些 Actions，執行 `bandit.update()`。
5. **狀態推進 (State Transition)：** $DL_{current} \leftarrow DL_{new}$（覆蓋舊模型，作為下一輪的 Baseline）。

### 固定 Baseline vs 滾動 Baseline

#### 路線一：固定 Baseline
- **優點：** 環境極度穩定，RL 快速收斂；避免 DL 提早飽和（Reward Vanishing）
- **缺點：** 可能過擬合於 Base Model

#### 路線二：滾動 Baseline (Co-evolution)
- **優點：** 高度擬真，模擬 Real-time 情況
- **缺點：** 左腳踩右腳的收斂危機

#### 建議：混合式「回合制預訓練」(Episodic Pretraining)
1. **回合開始（Reset）：** 載入 Base DL Model；RL 模型繼承上一回合
2. **回合內滾動：** 模擬 Real-time，每首歌後 DL 推進
3. **回合結束：** 拋棄進化的 DL，保留變聰明的 RL，回到步驟 1

> 這樣 RL 在每個回合內體驗 DL 推進的真實情況，同時跨回合重置 DL 保證 Reward 訊號強烈。

### EWMA 早停 vs UpdateScheduler

- **EWMA Threshold：** 僅用於離線預訓練，達標就結束程式
- **UpdateScheduler (scoreThreshold)：** 用於即時線上，控制何時觸發 DL 模型微調（無限巡航模式）

## 2026/05/07
- `other/` 刪除
- `combine_all/` 層級也可以刪除了，明天要測試實驗，怕路徑問題，待改
- save DL model, RL_pretrain 也要用，路徑問題待改
- zero_level 必須為初始第一首歌? 待修正
- Calib 的 RL 好像沒有處理 `_rule_based_override`? 待確認
- lsl 從連線開始就會 push 資料，要持續消耗，否則會發生 Lost packet(s). (api) → 好像是無線滑鼠的問題?

## 2026/05/08
- cb_ori 還是原本的
- cb_new = `for_VR_4class.pth` 改成 RL_pretrain 所使用的 model

## 2026/05/12
- zero_level 改成用 calib，但不會啟動 RL_LOOP，作為設定 DL, RL 的 baseline
- 目前流程：`zero_level (no RL_LOOP) → calib → normal level (has RL_LOOP) → calib...`

## 2026/05/13
- `ContextNormalizer` 的 `RuntimeWarning: invalid value encountered in subtract` → 好像只有第一次?
- UpdateScheduler 貌似沒有 reset?
- Dynamic_Buffer 更新正常：`+12 筆, total=12`
- 如果沒有在歌曲期間內 RL_LOOP 的話，會 update model 上去嗎? → 會
- Cygnus Lost packet(s) 發生在歌曲進行中間，懷疑是 CPU 問題 -> 5/13 是
- acc 可能是 epoch 不夠的部分
- RL update 是根據即時的資料，而目前是根據之前的舊資料 `.pt`，把 predict 那邊的腦波拿過來用


# 2026/05/17 
- 修正 context 移除 deltaloss
- RL Loop UpdateScheduler 更新但還沒確認邏輯正不正確
- action 移除 `DISCARD` 及相關實作，LOW_WEIGHT UpdateScheduler score 改成 = 0.7(3 low weight = 2.1 include)
- RL_pretrain 要重 train 
- 目前 DL 與 RL 的 entropy 對齊問題 還在想要怎麼解決


# 2026/05/19 POAN
- 遊玩結束 → intermission 入口 加了一個 pass：用當前最新 DL 模型對所有歷史 .pt 的 x_data 做一次完整 batch inference
- CSV header：移除 config.prob_classes(四個欄位)