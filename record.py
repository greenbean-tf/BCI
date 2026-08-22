
"""
2026/01/25

setting.json 的路徑改成變數
BCIModelInterface 改位置
raw_eeg 用途?


2026/01/27

records 資料夾可刪除 (已刪)
log.txt 檔名還沒改


2026/01/28
predict 跟 read 已改成 thread 形式
但仍有 時間不對齊問題
鍵有重疊問題，所以才會有有按到但 acc 沒 + 的情況 ----> 透過改譜避免
train 還沒做 --> train 時 要中斷 predict --> train 的時候本來就不會predict, predict 跟 read 只有在

選 model 頁面也還沒做 (更 1/29 已完成)



2026/01/29
選 model 頁面已完成，名稱需美化 --> 2026/02/02 已美化完成
遊玩介面(線)跑掉，
hit.mp3 沒出來， ---> 已修正
暫停介面偏移， ----> 已修正
鍵按下沒消失


2026/01/30
data load 已完成 ---> 目前是 全 load -> train, 要實作 load 最新-> fine tune

train 測試完畢，目前可跑

介面需優化彈出 acc
時間偏移尚未搞定


2026/02/02
data load 是用 SKF 切的
calib 應該完成了
全部改 thread, 時間對齊問題好像改好了? 好像又還沒

找到問題了，是切頁面的影響，只要完全不切頁面就沒事


2026/02/03
要 fine tune 的 model 要改成當前使用的，尚未改 -> 02/04 已改，可以從 log 確認是哪個 model calib 來的，ver 只會往上加
log_dir 做好了

2026/02/04
fine tune 的 model 已改成當前 activate
修正大多數路徑問題，TCP client 已傳 training msg，看介面怎麼運用

時間對齊問題，大多數是切頁面的影響，小機會不切頁面會發生 ---> 再看要不要修正


2026/02/06
key 消失已更新
LSL 接好還沒測試， 要 connect 才能測試
速度計算應該差不多好了，在 osu_interface 的 calculate_marker 裡

speed 0.1 還是有點快，可能要再改 speed 下限



2026/02/09
user 帳號名稱管理 model 待實作

更新速度計算後，判斷會改時間 ---> 待改

esc 問題修正


2026/02/11
model 判斷規則待改 ---> 目前是 以當前結果，看要不要改多數決
新增帳號功能

data 的路徑修好了
train 的 還沒

2026/02/13
可能要用額外 json 紀錄當前 version

2026/02/23
可能要用額外 json 紀錄當前 version ---> 應該好了?

pred 改用多數決
時間 key 間隔問題應該解決了?

處理 calib 那首，train 的流程

pause skip 好像不會終止 thread ?

測試 model 性能頁面 還沒做


2026/02/24
新紀錄 prob -> 但存取結果有問題 -> 02/25問題解決
pred 改回單一決定
data process 要重新處理


2026/02/25
data process 重處理ing

2026/02/26
data process 重處理ing，加入了 prob 跟 imu


2026/02/27
last acc 要怎麼改? --> 03/05 已處理完成


2026/03/04
train 拿全部資料，不走 RL 
--> 預計是第一首用，所以全部也只有第一首的 data

calib 只取剛完的一首，有 RL 
--> 之後的歌都走這裡

要把 train 跟 calib 移到 server 上做的樣子
prepare_datasets_v2 過了
prepare_datasets_RL_v1 還沒測



2026/03/05

check window 改成 150 ms
first song 要怎麼處理?

time cost 尚未改成 取決於 歌曲長度

已捨棄所有 motion 相關

目前到 training model 前面的部分應該都過了
training 也過了，calib 還沒測

parser 還有問題
test 曲也有問題 -> 已修正

好像有些會有 key 看不到當前判斷鍵的問題，不確定是不是來不及到 --> 已修正完成



2026/03/06

RL 閉環想法

song 1 data -> train first model -> use first model play next song


when playing next song
closed loop
-> keep calib first model by song 1 data  
-> get reward (delta acc) update RL [in background]


要把 train 跟 calib 移到 server 上做的樣子 -> 暫緩，先用柏安的 GPU 代替



RL loop model 可能會有 multi thread 搶佔問題
路徑尚未測試
其他閉環應該是好的?

song 之間更新尚未 確定是否使用


2026/03/08
RL loop model 可能會有 multi thread 搶佔問題 ---> 貌似沒問題?  不知道為啥

新增一些 print 資訊

如果最後一次 RL loop 還沒結束，要先等結束再 calib 下一個


2026/03/11
first test 修正

EEG_Read 模擬沒問題
但實際 好像有維度問題? 或是 thread 問題，CSV 資料出來都是0

# pylsl 取得的 sample 為一維 list，需轉為二維 (N, 1) 才可與 buffer_ori_data 進行 hstack
sample = np.array(sample).reshape(-1, 1)
待測試



2026/03/12
test model 會存 CSV ---> 要改掉

修正 parser 解析成變動，方向問題



2026/04/08
zero song 現在是用假回饋
先嘗試 pretrain RL



2026/04/11
系統運作標準流程 (SOP) 總結綜合以上設計，您的系統迭代流程應該如下：

資料收集與 RL 決策： 遊玩一首新歌，產出 Raw Data。
RL 根據 Context 決定哪些資料要增強、保留或丟棄（Actions）。
混合緩衝區 (Buffer Mixing)： 將 RL 認可的新資料，與歷史優質資料混合。
模型微調 (Calibration)： 將混合資料餵給目前的 $DL_{current}$，經過少量 Epochs 的微調，產出 $DL_{new}$。
計算與更新 (Reward & Update)：在固定的、乾淨的驗證集上計算：$Reward = Loss(DL_{current}) - Loss(DL_{new})$。
將這筆 Reward 分配給步驟 1 中 RL 做的那些 Actions，執行 bandit.update()。
狀態推進 (State Transition)：$DL_{current} \leftarrow DL_{new}$ （覆蓋舊模型，作為下一輪的 Baseline）。

這樣的架構不僅能保證 DL 模型的訓練效率，也能確保 RL 代理者接收到的反饋始終是「最新、最客觀」的貢獻度評價。


這是一個非常具有深度且直指強化學習核心痛點的架構問題！您所遇到的情況，在 RL 領域被稱為 **「非平穩環境（Non-stationary Environment）與移動目標（Moving Target）問題」**。

簡短的回答是：**不，您並非「絕對不能」做狀態推進，但您目前的直覺（在 Pretrain 階段固定 Base DL Model）是非常正確且保守的安全做法。**

為了解釋這兩種做法的差異，我們來拆解「固定 Baseline」與「滾動 Baseline」在**預訓練（Pretraining）階段**的優缺點與適用場景，並給您最終的架構建議。

---

### 路線一：固定 Baseline (您的假設)
**邏輯：** 每次計算 Reward 時，都用 `Base DL Model` + `新挑選的資料` 微調出一個虛擬的 $DL_2$ 來算 $\Delta Loss$。算完 Reward 丟給 RL 更新後，就把 $DL_2$ 丟掉，下一輪依然從 `Base DL Model` 開始。

* **優點（為什麼您的直覺是對的）：**
    * **環境極度穩定：** RL 代理者面對的是一個規則永遠不變的考官。對於相同的 Context（例如極高的 Entropy），執行 `DISCARD` 永遠會得到相似的 Reward。這能讓 LinUCB 的 $A$ 和 $b$ 矩陣在預訓練階段非常快速且穩定地收斂。
    * **避免 DL 提早飽和：** 如果 DL 一直推進，到預訓練後期 DL 可能已經太強了，Loss 降無可降。這時 RL 不管怎麼挑資料，Reward 都趨近於 0，RL 就會停止學習（這稱為 Reward Vanishing）。
* **缺點（潛在風險）：**
    * **過擬合（Overfitting）於 Base Model：** RL 模型可能會學到一套「只對這個特定 Base DL 有效」的挑選策略。當未來進入 Real-time 階段，DL 模型開始進化了，RL 可能會因為環境改變而不知所措。

### 路線二：滾動 Baseline (狀態推進 Co-evolution)
**邏輯：** 即使在預訓練階段，DL 模型也跟著 RL 一起成長。$DL_1 \rightarrow DL_2 \rightarrow DL_3$，每次都用前一代當作 Baseline 計算 Reward。

* **優點：**
    * **高度擬真：** 這完全模擬了未來 Real-time 系統上線後的真實情況。RL 會學到一個更高級的策略：「如何根據 DL 模型目前的強弱（透過 Context 中的 `delta_acc` 或 `mean_entropy` 反映），動態調整挑資料的嚴格程度」。
* **缺點：**
    * **左腳踩右腳的收斂危機：** DL 和 RL 同時在改變。有時候 $\Delta Loss$ 表現不好，RL 不知道是因為「我剛才資料挑得差」，還是單純因為「DL 模型已經遇到瓶頸了」。這會讓 LinUCB 前期的學習曲線劇烈震盪。

---

### 👑 專家建議：混合式「回合制預訓練」(Episodic Pretraining)

既然您未來的目標是 **Real-time 使用（DL 會動態推進）**，如果您在 Pretrain 階段把 DL 鎖死，RL 上線後肯定會遇到水土不服。

業界在處理這種「RL 輔助 DL 訓練」的預訓練時，最標準的做法是採用 **回合制（Episodic）** 架構，結合兩者的優點：

#### 具體實作流程：
我們把您手邊的歷史訓練資料切分成多個「虛擬玩家 Session（回合）」：

1.  **回合開始（Reset）：**
    * 載入 `Base DL Model` 作為 $DL_{current}$。
    * 載入您正在 Pretrain 的 RL 模型（LinUCB 參數繼承自上一回合，不重置）。
2.  **回合內滾動（模擬 Real-time）：**
    * 餵入第 1 首歌的資料 $\rightarrow$ RL 挑選 $\rightarrow$ 微調出 $DL_{new}$ $\rightarrow$ 計算 Reward 更新 RL $\rightarrow$ **執行狀態推進 $DL_{current} \leftarrow DL_{new}$**。
    * 餵入第 2 首歌的資料 $\rightarrow$ RL 挑選 $\rightarrow$ 微調出 $DL_{new}$ $\rightarrow$ 計算 Reward 更新 RL $\rightarrow$ **執行狀態推進 $DL_{current} \leftarrow DL_{new}$**。
    * ...直到這個虛擬玩家的 5 首歌播完。
3.  **回合結束（防止 DL 飽和）：**
    * **拋棄** 已經進化了 5 次的 $DL_{current}$。
    * 保留已經變聰明的 RL 模型。
    * 換下一個虛擬玩家的資料，**回到步驟 1**（再次從乾淨的 `Base DL Model` 開始）。

#### 為什麼這個架構最好？
* **對 RL 來說：** 它在每個回合內都體驗到了「DL 模型會推進」的真實情況，學會了應對動態環境。同時，因為跨回合會不斷保留參數，它能充分吸收大量資料。
* **對 DL 來說：** 每個回合結束就重置回 Base Model，保證了 RL 在預訓練的每一個階段，都能享受到「容易讓 Loss 下降的紅利期」，確保 Reward 訊號始終強烈且明確。

**總結來說：** 在您的情境下，不要把 DL 永遠鎖死，也不要讓 DL 無限期推進。把您的歷史數據切分成幾個「小遊戲階段」，在每個小階段內部讓 DL 推進，階段結束就重置 DL。這會是培訓出最強 RL 代理者的最佳解法！



2026/05/07

other/ 刪除

combine_all/ 層級也可以刪除了，明天要測試實驗，怕 路徑問題，待改
save DL model, RL_pretrain 也要用, 路徑問題待改

zero_level 必須為初始第一首歌? 待修正
Calib 的 RL 好像沒有處理 _rule_based_override? 待確認

lsl 從連線開始就會push 資料，要持續消耗，否則 會發生 Lost packet(s). (api) ---> 好像是 無線滑鼠的問題?


ewma-threshold（EWMA 早停閾值）這個機制 僅專屬於離線預訓練（Offline Pretrain）階段，在未來的 Real-time（即時線上）系統中是完全不需要、也不存在的。

為您釐清這兩個階段在架構設計上的根本差異：

1. 離線預訓練階段 (Offline Pretrain)
目標： 拿歷史資料反覆訓練（Epochs），讓 Bandit 盡快變聰明。

為什麼需要 EWMA Threshold？ 歷史資料是有限的。如果 Bandit 已經學會了最優策略（Reward 穩定保持正數），卻還強迫它把剩下的 Epoch 跑完，它就會對這批歷史資料過擬合（Overfitting）。

行為： 它是用來「踩煞車」的。達標就直接結束程式，把模型存起來準備上線。

2. 即時線上階段 (Real-time / Online)
目標： 隨著玩家遊玩，無止盡地動態適應玩家「當下」的腦波狀態。

為什麼不需要 EWMA Threshold？ 在真實遊戲中，使用者的狀態是會不斷改變的（從精神飽滿玩到疲勞）。Bandit 必須一直保持學習狀態，永遠不能停下來。如果設定了早停，Bandit 在遊戲玩到一半時「罷工」不學了，後面玩家流汗或分心造成腦波改變時，系統就會崩潰。

線上的替代機制： 在線上系統中，用來控制節奏的不是 EWMA，而是 UpdateScheduler 裡面的 scoreThreshold (預設 15.0)。

這個機制不是用來「結束」RL 訓練的，而是用來「觸發」DL 模型微調的。它負責判斷「我們是不是已經收集到足夠多的高品質資料，可以叫 DL 模型去更新權重了？」

總結來說：
您在 RL_pretrain.py 寫的 EWMA 機制，是一個非常標準且優雅的「訓練終止條件」；而到了實機遊玩時，系統就會切換成「無限巡航模式」，只靠 UpdateScheduler 來決定何時進廠保養（更新 DL 模型）。


2026/05/08

cb_ori 還是原本的
cb_new = for_VR_4class.pth 改成 RL_pretrain 所使用的 model



2026/05/12
zero_level 改成用 calib, 但不會啟動 RL_LOOP, 作為設定 DL, RL 的 baseline ,


目前流程， zero_level (no RL_LOOP) -> calib -> normal level (has RL_LOOP) -> calib....

"""