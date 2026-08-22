# 作法

每次都是 reload DL base model (train.pth) ---> dir=account/Guest/model/base_model/Calibed_Model_v001 `位置待改`
每個 data.pt 是用同一個 DL base，分別 calib



ex 
num of .pt      = 2
DL_calib_epoch  = 4
RL_LOOP_time    = 3


RL total update = 2 * 3

```
RL 0 ->

RL 外層迴圈 (RL_LOOP_time = 3)
   │  
   ├─ LOOP 1
   │  ├─ .pt 檔案 1 → 訓練 EEGNet DL 0 --> (DL_calib_epoch=4) --> DL 1
   │  │                                                             └─ RL update #1
   │  └─ .pt 檔案 2 → 訓練 EEGNet DL 0 --> (DL_calib_epoch=4) --> DL 1
   │                                                                └─ RL update #2
   ├─ LOOP 2
   │  ├─ .pt 檔案 1 → 訓練 EEGNet DL 0 --> (DL_calib_epoch=4) --> DL 1
   │  │                                                             └─ RL update #3
   │  └─ .pt 檔案 2 → RL update #4
   │  
   └─ LOOP 3
      ├─ .pt 檔案 1 → RL update #5
      └─ .pt 檔案 2 → RL update #6

-> RL 1

```

test0 跟 test1 都還是上面的

- `已改待測 @1`: 實際上這邊有問題，每個 trail 都會 對 RL update 造成影響，所以應該要是 RL update  * trail(.pt 檔案 1)

RL total update 應該要是 = [trail(.pt 檔案 1) + trail(.pt 檔案 2)] * 3

```
RL 0 ->

RL 外層迴圈 (RL_LOOP_time = 3)
   │  
   ├─ LOOP 1
   │  ├─ .pt 檔案 1 → 訓練 EEGNet DL 0 --> (DL_calib_epoch=4) --> DL 1
   │  │                                                             └─ RL update #1 * trail(.pt 檔案 1)
   │  └─ .pt 檔案 2 → 訓練 EEGNet DL 0 --> (DL_calib_epoch=4) --> DL 1
   │                                                                └─ RL update #2 * trail(.pt 檔案 2)
   ├─ ...
   │  
   └─ LOOP 3
      └─ ...

-> RL 1
```




# 模型儲存狀態: 

- `no待改`: 多標示出 `load RL` ---> 有 print 在最前面了
Log: [INFO] Pretrained Bandit 已儲存: .../pretrained_bandit.pt

說明: 代表這次的回合制預訓練已經順利完成，並且將 result 存成 .pt 檔案。 `注意: data 跟 RL model 都是 .pt`

未來的 Real-time 系統在啟動時，就可以直接 load 這個檔案，讓您的 Bandit 帶有「過濾記憶」上陣。




# 指數移動平均獎勵 (EWMA Reward):
Log: [INFO] 最終 EWMA reward: -0.0383
說明: EWMA（Exponentially Weighted Moving Average）是一種能平滑劇烈震盪數值的統計方法，它反映了「預訓練後期的平均表現趨勢」。
技術診斷: 這裡的數值是負數 (-0.0383)。這在初期訓練很常見，原因可能是：DL 模型的 Loss 下降幅度（正 Reward）還不夠大。
您公式裡的資料品質懲罰項（$\mu LQP$ 或時間成本）扣了太多分。這暗示您接下來可能需要微調 Reward 公式中的超參數（例如調小 $\mu$ 值），讓系統不會因為過度害怕雜訊而不敢拿資料。



- `公式`: self.ewmaReward = self.ewmaAlpha * reward + (1 - self.ewmaAlpha) * self.ewmaReward
- `threshold`: 如果 EWMA 超過 設定的 threshold，直接停止整個 pretrain，存取最新 RL model


# 總更新次數 (Total Updates):
Log: [INFO] 總更新次數: 20
說明: 代表 bandit.update() 在這次預訓練中被成功呼叫了 20 次。也就是說，系統拿到了 20 次實質的 Reward 來修正 LinUCB 的策略。


# 決策歷史長度 (History Length):
Log: [INFO] History 長度: 26


## History 長度 

通常代表 bandit.update() 成功將資料推入 history 列表的次數。
如果總更新次數是 20，但 History 長度是 26，這代表在這次執行腳本之前，該 Bandit 模型（pretrained_bandit.pt）已經包含了 6 筆歷史紀錄。
當它被載入後，又新增了 20 筆，總計達到 26 筆。


            RL base    +   .pt      * RL_LOOP_time  = History
`test0` is                  2       * 3             = 6
`test1` is  `test0`    +    2       * 10            = 26



## 動作分配比例 (Buffer sizes):
Log: [INFO] Buffer sizes: [15, 7, 2, 2] 

說明: （對應 History 長度）
這表示您的 Bandit 確實做出了多樣化的決策，沒有陷入「只選同一種動作」的死胡同：
Action 0 (INCLUDE) = 15 次：大約 57% 的資料被判定為正常，直接納入訓練。
Action 1 (DISCARD) = 7 次：大約 27% 的資料被判定為雜訊或高 Entropy 而被成功攔截並丟棄。
Action 2 (HIGH_WEIGHT) = 2 次：Bandit 抓到了極少數（約 7%）的高品質黃金數據，決定加重訓練。
Action 3 (LOW_WEIGHT) = 2 次：有 2 筆資料可能狀態偏弱（例如有點疲勞但還堪用），被給予低權重。




# 待辦事項