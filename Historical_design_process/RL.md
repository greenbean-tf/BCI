# RL（Reinforcement Learning / Contextual Bandit）歷史設計流程

> 整理範圍：用 contextual bandit 控制「線上要不要拿這段資料 update DL，以及要用什麼權重」的整套設計演化。
> 資料來源：git log（特別是 `RL/bandit.py` 系列）、`record.py`、`combine_all/RL/RL_pretrain/record.md`、Notion 會議記錄、`combine_all/docs/RL.md`、`combine_all/RL/bandit.py` 原始碼、`combine_all_new/implementation.md`。

---

## 0. RL 在系統中的角色

```
DL 推論 → 取 context（每個 trial / song） →
Bandit.choose_action(context) →
{INCLUDE, DISCARD, HIGH_WEIGHT, LOW_WEIGHT} →
UpdateScheduler 累積樣本 + 權重 →
（達門檻時）DL.online_update(samples, weights) →
觀察 ΔLoss + 計算 reward → Bandit.update(context, action, reward)
```

目標：在不確定 EEG 變動的環境下，自動決定哪些 trial 應拿來 update、以多高的權重，避免 catastrophic forgetting 或 noise 拉壞模型。

---

## 1. v0：早期想法討論（2026/01/19）

Notion 2026/01/19（RL）會議：
- 列出 RL 四元素初稿：reward / policy / environment / agent。
- 仍卡在「reward 怎麼定」、「環境是 EEG 還是任務表現」。
- 結論：先以 contextual bandit 落地，避免完整 MDP 過於複雜。

---

## 2. v1：公式 / 文獻調研（2026/01/26）

Notion 2026/01/26（RL）會議：
- 文獻：bandit / contextual bandit / NeuralUCB / LinUCB。
- 候選 context 特徵：
  - **mean entropy**：softmax entropy 平均。
  - **flip rate**：相鄰 window 預測 class 翻轉率。
  - **mean margin**：top1 - top2 機率差。
  - **motion score**（後來捨棄）。
- 候選 action：拿 / 不拿、加權拿。

---

## 3. v2：閉環雛形（2026/03/06）

record.py 2026/03/06（RL 閉環想法）：

```
song1 → train → use → calib delta acc reward → song2 ...
```

- 第一首歌做 base，後續歌曲用 ΔAcc 當 reward 訊號。
- 後來改成 ΔLoss（更敏感、更平滑）。

---

## 4. v3：LinUCB 落地（~2026/04/08）

依 `combine_all/RL/bandit.py` 早期版本（git history）：

- **演算法**：LinUCB（線性 contextual bandit）。
- **action set**：`INCLUDE / DISCARD / HIGH_WEIGHT / LOW_WEIGHT / ONLINE_UPDATE`。
- **context**：early 版本含 mean entropy / mean margin / flip rate / motion score。
- **更新**：每 trial 拿到 reward 後做 LinUCB 的封閉式 update（A、b 累積）。

---

## 5. v4：Zero song 假回饋 + RL pretrain（2026/04/08）

record.py 2026/04/08：
- 第一首歌（**zero song**）用「假回饋」讓 bandit / DL pipeline 跑通流程；第二首才開始正式 reward。
- 同期 `RL_pretrain/` 子模組成形，目的：在沒有真人受試者的情況下，用既有 session 模擬資料離線預訓練 bandit。

---

## 6. v5：軌道標示修正 + RL 改神經網路架構（2026/04/09）

git：`軌道標示修正, RL改神經網路架構`

關鍵切換：
- LinUCB → **NeuralUCB**：bandit 內部以小型 MLP 表徵 reward 函數，配合 Bayesian-style uncertainty 做 UCB。
- 同步加上 `BANDIT_DEVICE` 設定（CPU/GPU 可切）。

`combine_all/RL/bandit.py`（NeuralUCB）特徵：
- **Networks**：每個 action 一個小 MLP，輸入 context → 估計 reward。
- **UCB**：以 last-layer feature 計算協方差近似（依 NeuralUCB paper）作為 exploration bonus。
- **action 選擇**：argmax(μ + α·σ)。
- **線上更新**：每收到 (context, action, reward) 做小步 SGD。

---

## 7. v6：大更新前 + 系統 SOP（2026/04/11）

record.py 2026/04/11（系統運作 SOP，RL 部分）：

- 流程：Calib → Trained_Model → Gameplay → 每首歌結束累積 RL buffer → 達門檻觸發 online_update → 存 RL_Loop_Model。
- 把 ONLINE_UPDATE 從 action set 拿掉（後續確認）→ update 改由 **UpdateScheduler 集中觸發**，bandit 只負責「樣本級」決策。

---

## 8. v7：RL_pretrain 細節（2026/04/15 ~ 04/17）

依 `combine_all/RL/RL_pretrain/record.md` + Notion 2026/04/17 wei 會議：

### 8.1 Episodic Pretraining 架構
- 把每首歌視為一個 episode。
- 用既有資料模擬「DL 收 trial → 是否 update → 觀察新 loss」整個過程，產生大量 (context, action, reward) 用於離線預訓練 bandit。

### 8.2 Buffer sizes
- 早期實驗 buffer sizes 例：`[15, 7, 2, 2]`（不同 action 的對應 buffer 大小）。
- 後來統一改為單一 Experience Replay Buffer（500 samples，見 §11）。

### 8.3 EWMA reward 公式
- 用 EWMA 平滑 reward 訊號，避免單筆 trial 的 loss noise 主導 bandit。

### 8.4 Notion 2026/04/17 wei 會議重點
- **update 從小 trial 累積**：每次 online_update 不要一次性，而是按 trial 累積到 threshold。
- **DL acc > 85% 停止 update RL**：DL 已穩定就不再 update，bandit 也跟著停。
- **BCI 最終目標**：使用者體感無感、模型自動跟著漂移調整。

---

## 9. v8：完整 NeuralUCB 上線（~2026/04/20 後）

依 `combine_all/RL/bandit.py` 與 `combine_all/docs/RL.md`：

### 9.1 Action set（最終版）
- `INCLUDE`（標準權重 1.0 收進 buffer）
- `DISCARD`（拒收）
- `HIGH_WEIGHT`（權重 2.0 收進 buffer）
- `LOW_WEIGHT`（權重 0.3 收進 buffer）
- 移除：`ONLINE_UPDATE`（改由 UpdateScheduler 統一觸發）

### 9.2 Context（4 維）
```
[MEAN_ENTROPY, MEAN_MARGIN, DELTA_LOSS, ALPHA_POWER_NORM]
```
- `MEAN_ENTROPY`：window 內 softmax entropy 平均。
- `MEAN_MARGIN`：top1 - top2 機率差。
- `DELTA_LOSS`：當下 trial 估計 loss 與基線差。
- `ALPHA_POWER_NORM`：α 頻段 power 標準化（從 `band_power.py`）。

### 9.3 Reward 公式
```
r_t = -ΔLoss_t  -  λ · T_t  -  μ · γ_t · LQP_base_t
```
- `-ΔLoss_t`：負的 loss 變化（loss 降越多、reward 越大）。
- `λ · T_t`：訓練成本懲罰（時間 / 資源）。
- `μ · γ_t · LQP_base_t`：低品質基線懲罰（防 bandit 投機選 HIGH_WEIGHT）。

### 9.4 ContextNormalizer
- 在線 running min-max 正規化，讓 NeuralUCB 訓練穩定。

---

## 10. v9：UpdateScheduler 統一觸發（同步成形）

依 `combine_all/docs/RL.md` 與 implementation.md：

- 維護一個 score：`score = Σ weight(action_i)`，其中：
  - `HIGH_WEIGHT = 2.0`
  - `INCLUDE = 1.0`
  - `LOW_WEIGHT = 0.3`
  - `DISCARD = 0.0`
- 累積到 **threshold = 15.0** 時觸發 `BraindecodeTrainer.online_update(samples, weights)`。
- update 完成後：
  - 計算 ΔLoss → 給 bandit 當 reward。
  - reset score。
  - 若 DL rolling acc > 85% 則停止後續 update（仍照常推論）。

---

## 11. v10：Experience Replay Buffer（500 samples）

- 為了 online_update 不只用「最新一批」，加入 Replay Buffer：
  - 容量：500 樣本。
  - 取樣：依 weight 採樣（HIGH 樣本被抽到的機率較高）。
- 解決：避免最近 distribution shift 把模型拉偏。

---

## 12. v11：移除 ONLINE_UPDATE 動作

- 早期 bandit 動作集合裡有 `ONLINE_UPDATE`（讓 bandit 自己決定要不要立刻 update）。
- 改用 UpdateScheduler + score threshold 之後，這個動作被刪除：
  - 原因：bandit 自己觸發 update 會跟 episodic reward 設計衝突，導致 reward bookkeeping 困難。
  - 結果：bandit 只做 trial 級決策，update 由 scheduler 統一管。

---

## 13. v12：EWMA early stopping 僅 offline pretrain（2026/05/07）

record.py 2026/05/07：
- EWMA early stopping **只用在 offline pretrain**。
- 線上 bandit 持續 update（因為使用者狀態會漂移）。

---

## 14. v13：Refactor band_power（2026/05/04）

git：`refactor band_power`

- α power 取樣由 `band_power.py` 統一提供 → context 第 4 維（ALPHA_POWER_NORM）數值穩定可靠。

---

## 15. v14：Log_class + 移除 other/（2026/05/06）

git：`delete other/, log_class`

- bandit 的決策（action / reward / context）寫進統一 logger，以便事後分析。
- `other/` 中關於 motion / skeleton 的 RL 試驗一併移除（最終 RL 純 EEG 訊號驅動）。

---

## 16. v15：combine_all_new 模組化（2026/05/06 ~ 05/07）

依 `combine_all_new/implementation.md`：

| 模組 | 功能 |
|------|------|
| `rl/bandit.py` | NeuralUCB 主體 |
| `rl/context.py` | context 抽取 + ContextNormalizer |
| `rl/reward.py` | 上述 r_t 公式 |
| `rl/scheduler.py` | UpdateScheduler（score / threshold / 觸發） |
| `rl/replay.py` | Experience Replay Buffer |
| `rl/pretrain/` | episodic pretrain 工具 |

---

## 17. 演化時間軸總覽

| 日期 | 里程碑 |
|------|--------|
| 2026/01/19 | RL 元素初步討論（reward/policy/env/agent） |
| 2026/01/26 | 文獻調研，候選 context / action |
| 2026/03/06 | RL 閉環雛形（song-level reward） |
| 2026/04/08 | LinUCB 落地 + zero song 假回饋 |
| 2026/04/09 | LinUCB → **NeuralUCB**（神經網路 bandit） |
| 2026/04/11 | 系統 SOP，移除 bandit 自觸發 update 的概念 |
| 2026/04/15~17 | RL_pretrain 架構、buffer 設計、EWMA reward |
| 2026/04/17 | wei 會議：trial 累積、acc>85% 停止 update |
| ~2026/04/20+ | 完整 NeuralUCB + 4-dim context + r_t 公式 |
| ~2026/04/?? | UpdateScheduler 統一觸發、Replay 500 samples |
| 2026/05/04 | band_power refactor，α power 來源穩定 |
| 2026/05/06 | log_class、刪除 other/ |
| 2026/05/07 | EWMA early stopping 僅 offline pretrain |

---

## 18. 重要設計取捨總表（RL）

| 取捨 | 結果 | 備註 |
|------|------|------|
| MDP vs Bandit | Contextual Bandit | 工程可行性 |
| LinUCB vs NeuralUCB | NeuralUCB | 2026/04/09 切換 |
| Reward 訊號 | -ΔLoss（含成本/品質懲罰） | 取代 ΔAcc |
| Context 是否含 motion | 不含 | 2026/03/05 起 |
| Bandit 是否自觸發 update | 否 | 改 UpdateScheduler |
| Update threshold | 加權分數 ≥ 15.0 | HIGH=2.0/INC=1.0/LOW=0.3/DIS=0 |
| Replay Buffer | 500 樣本，按 weight 取樣 | |
| Stop 機制 | DL acc > 85% 停止 update | Notion 2026/04/17 |
| Early stopping | 僅 offline pretrain（EWMA） | 2026/05/07 |

---

## 19. 與 DL / UI 的介面整理

- **對 DL**：呼叫 `online_update(samples, weights)`；訂閱 `forward_loss / prob / entropy / margin / band_power`。
- **對 UI**：每首歌結束顯示 RL 的 score 進度（離下次 update 還差多少）；DL acc 達 85% 後 UI 顯示「RL 觀望中」。
- **對 logger**：寫 (timestamp, context, action, reward, song, model_version)。

---

> 詳細數值（α、λ、μ、學習率、MLP 寬度等）請看 `combine_all/RL/bandit.py`、`combine_all/docs/RL.md`；pretrain 設定請看 `combine_all/RL/RL_pretrain/`。
