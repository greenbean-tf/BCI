### python/main/EEG/bandit.py
### 包含 `NeuralContextualBandit`（NeuralUCB）、`BanditAction` enum、`UpdateScheduler`、`ExperienceReplayMemory`

"""
Contextual Bandit for VR-BCI calibration flow
使用 NeuralUCB 演算法（MLP + 梯度 norm 探索項），

Action 定義：
    A0 - INCLUDE:        納入訓練 buffer（預設行為）
    A1 - HIGH_WEIGHT:    高權重納入 buffer
    A2 - LOW_WEIGHT:     低權重納入 buffer

Context 向量 (3 維): 
    [mean_entropy, mean_margin, alpha_power_normalized]
    context 僅包含從 data 濃縮出的狀態描述，delta_loss 不屬於 context（僅用於 reward 計算）。

升級紀錄：
    - Reward 平滑化：ΔAcc → ΔLoss（Cross-Entropy Loss 下降量）
    - NeuralUCB：MLP 取代線性矩陣 A/b
    - 解耦動作空間：移除 ONLINE_UPDATE，由 UpdateScheduler 獨立排程
    - 移除 DISCARD：action 空間縮減為 INCLUDE / HIGH_WEIGHT / LOW_WEIGHT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from enum import IntEnum
from collections import deque


# ─────────────────────────────────────────────
# 設備設定：控制 MLP 使用 CPU 或 GPU
# ─────────────────────────────────────────────
BANDIT_DEVICE = "cpu"  # 可改為 "cuda" 使用 GPU


class BanditAction(IntEnum):
    INCLUDE        = 0  # 納入訓練 buffer（預設行為）
    HIGH_WEIGHT    = 1  # 高權重納入 buffer
    LOW_WEIGHT     = 2  # 低權重納入 buffer

    n_actions      = 3


class ContextIDX(IntEnum):
    MEAN_ENTROPY                = 0
    MEAN_MARGIN                 = 1
    ALPHA_POWER_NORM            = 2

    context_dim                 = 3



# ─────────────────────────────────────────────
# RewardMLP：輕量 MLP 預測 expected reward
# ─────────────────────────────────────────────
class RewardMLP(nn.Module):
    """
    輕量 MLP 用於預測給定 context 下的 expected reward。
    結構：context_dim → 16 → 8 → 1
    加入 Dropout(0.2) + LayerNorm 防止 overfitting。
    每個 action 獨立持有一個 RewardMLP。
    """
    def __init__(self, contextDim: int, dropout: float = 0.2):
        super(RewardMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(contextDim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1),
        )

    def forward(self, contextTensor: torch.Tensor) -> torch.Tensor:
        return self.network(contextTensor)



# ─────────────────────────────────────────────
# NeuralContextualBandit (NeuralUCB)
# ─────────────────────────────────────────────
class NeuralContextualBandit:
    """
    NeuralUCB Contextual Bandit

    每個 action 維護獨立的 MLP：
        MLP_a(x) → predicted reward
    
    UCB score：
        score_a = MLP_a(x) + epsilon * ||grad_lastlayer(MLP_a, x)||

    其中 epsilon 控制 exploration vs exploitation。
    使用 Experience Replay Buffer 進行 mini-batch 更新。
    """

    def __init__(
        self,

        nActions            : int = BanditAction.n_actions,
        contextDim          : int = ContextIDX.context_dim,

        epsilon             : float = 0.5,

        lambdaCost          : float = 0.1,
        mu                  : float = 0.3,
        alp_ent_hyp         : float = 1.0,
        weightEntropy       : float = 0.5,
        rewardClipMin       : float = -2.0,
        rewardClipMax       : float = 2.0,

        replayBufferSize    : int = 500,
        mlpLearningRate     : float = 1e-3,
        miniBatchSize       : int = 32,
        updateEpochs        : int = 5,
    ):
        """
        Args:
            nActions            : action 數量（對應 BanditAction）
            contextDim          : context 向量維度

            epsilon             : NeuralUCB exploration 參數，越大越傾向探索

            lambdaCost          : reward 公式中 TimeCost 的權重 λ
            mu                  : LQP 懲罰項權重
            alp_ent_hyp         : alpha-entropy 交互強度超參數
            rewardClipMin       : reward 下界 clip
            rewardClipMax       : reward 上界 clip


            replayBufferSize:   experience replay buffer 大小
            mlpLearningRate:    MLP 的學習率
            miniBatchSize:      mini-batch 更新的 batch size
            updateEpochs:       每次 update 時 mini-batch 的 epoch 數
        """
        self.nActions               = nActions
        self.contextDim             = contextDim

        self.epsilon                = epsilon


        self.lambdaCost             = lambdaCost
        self.mu                     = mu
        self.alp_ent_hyp            = alp_ent_hyp
        self.weightEntropy          = weightEntropy
        self.rewardClipMin          = rewardClipMin
        self.rewardClipMax          = rewardClipMax


        self.replayBufferSize       = replayBufferSize
        self.mlpLearningRate        = mlpLearningRate
        self.miniBatchSize          = miniBatchSize
        self.updateEpochs           = updateEpochs

        self.device = torch.device(BANDIT_DEVICE)

        # 每個 action 獨立的 MLP 和 optimizer
        self.mlpModels = []
        self.mlpOptimizers = []
        for _ in range(nActions):
            model = RewardMLP(contextDim).to(self.device)
            # weight_decay=1e-3 = L2 正則化，防止 MLP 過擬合
            optimizer = optim.Adam(model.parameters(), lr=mlpLearningRate, weight_decay=1e-3)
            self.mlpModels.append(model)
            self.mlpOptimizers.append(optimizer)

        # Experience Replay Buffer（每個 action 獨立）
        self.replayBuffers = [deque(maxlen=replayBufferSize) for _ in range(nActions)]

        # 紀錄 history（for offline analysis）
        self.history = []  # list of dict {context, action, reward}

        print(f"[Bandit] NeuralContextualBandit 初始化完成")
        print(f"  device            ={self.device}")
        print(f"  nActions          ={nActions}")
        print(f"  contextDim        ={contextDim}")
        print(f"  epsilon           ={epsilon}")
        print(f"  lambdaCost        ={lambdaCost}")
        print(f"  mu                ={mu}")
        print(f"  alp_ent_hyp       ={alp_ent_hyp}")
        print(f"  weightEntropy     ={weightEntropy}")
        print(f"  rewardClipMin     ={rewardClipMin}")
        print(f"  rewardClipMax     ={rewardClipMax}")
        print()

    # ─────────────────────────────────────────────
    # Rule-based 保底機制
    # ─────────────────────────────────────────────
    def _rule_based_override(self, context: np.ndarray):
        """
        Rule-based 保底，優先於 Bandit：
          - entropy 極高 + alpha 極高 → LOW_WEIGHT

        回傳 BanditAction 或 None（None 表示交給 Bandit 決策）
        """
        meanEntropy = context[ContextIDX.MEAN_ENTROPY]
        alphaNorm = context[ContextIDX.ALPHA_POWER_NORM]

        # alpha 極高 + entropy 極高 → 使用者可能疲累
        if alphaNorm > 0.85 and meanEntropy > 0.8:
            print(f"[Bandit] Rule-based override: LOW_WEIGHT (alpha={alphaNorm:.3f}, entropy={meanEntropy:.3f})")
            return BanditAction.LOW_WEIGHT

        return None  # 交給 Bandit

    # ─────────────────────────────────────────────
    # 計算探索項：最後一層梯度 norm
    # ─────────────────────────────────────────────
    def _compute_exploration_bonus(self, actionIdx: int, contextTensor: torch.Tensor) -> float:
        """
        計算 NeuralUCB 探索項：||grad_lastlayer(MLP_a, x)||
        使用最後一層線性層的梯度 norm 作為不確定性估計。
        """
        model = self.mlpModels[actionIdx]
        model.eval()

        contextInput = contextTensor.clone().detach().requires_grad_(False)
        
        # 取得最後一層的參數
        lastLayer = model.network[-1]  # 最後一個 Linear(8, 1)
        
        # 開啟梯度追蹤
        for param in lastLayer.parameters():
            param.requires_grad_(True)

        output = model(contextInput)
        output.backward()

        # 計算梯度 norm
        gradNorm = 0.0
        for param in lastLayer.parameters():
            if param.grad is not None:
                gradNorm += param.grad.norm().item() ** 2
        gradNorm = np.sqrt(gradNorm)

        # 清除梯度
        model.zero_grad()

        return gradNorm

    # ─────────────────────────────────────────────
    # NeuralUCB action 選擇
    # ─────────────────────────────────────────────
    def select_action(self, context: np.ndarray) -> BanditAction:
        """
        給定 context 向量，回傳選擇的 action。
        先過 rule-based，再跑 NeuralUCB。

        Args:
            context: shape (context_dim,)，所有維度已正規化至 [0, 1]

        Returns:
            BanditAction
        """
        # 先檢查 rule-based 保底
        override = self._rule_based_override(context)
        if override is not None:
            return override

        contextTensor = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)

        scores = []
        for actionIdx in range(self.nActions):
            # MLP 預測 expected reward
            self.mlpModels[actionIdx].eval()
            with torch.no_grad():
                exploitScore = self.mlpModels[actionIdx](contextTensor).item()

            # 探索項
            exploreBonus = self._compute_exploration_bonus(actionIdx, contextTensor)

            ucbScore = exploitScore + self.epsilon * exploreBonus
            scores.append(ucbScore)

        chosen = int(np.argmax(scores))
        
        if False:
            print(f"[Bandit] select_action: scores={[f'{s:.4f}' for s in scores]}, chosen={BanditAction(chosen).name}")
        
        return BanditAction(chosen)

    # ─────────────────────────────────────────────
    # NeuralUCB 更新（Experience Replay）
    # ─────────────────────────────────────────────
    def update(self, context: np.ndarray, action: BanditAction, reward: float):
        """
        收到 reward 後更新對應 action 的 MLP。
        將 (context, reward) 加入 replay buffer，然後做 mini-batch 更新。

        Args:
            context: 當時做決策用的 context 向量
            action:  執行的 action
            reward:  計算出的 reward
        """
        actionIdx = int(action)

        # 加入 replay buffer
        self.replayBuffers[actionIdx].append({
            'context': context.copy(),
            'reward': reward,
        })

        # 記錄 history
        self.history.append({
            "context": context.tolist(),
            "action": actionIdx,
            "reward": reward,
        })

        # Mini-batch 更新
        self._train_mlp(actionIdx)

        print(f"[Bandit] update: action={BanditAction(actionIdx).name}, reward={reward:.4f}, "
              f"buffer_size={len(self.replayBuffers[actionIdx])}")

    def update_batch(self, contexts: list[np.ndarray], actions: list[BanditAction], rewards: list[float]):
        """
        批次更新對應 action 的 MLP。
        將所有 (context, reward) 分類加入各自的 replay buffer，然後對有變動的 action 呼叫一次 _train_mlp。
        """
        modifiedActions = set()
        for context, action, reward in zip(contexts, actions, rewards):
            actionIdx = int(action)
            # 加入 replay buffer
            self.replayBuffers[actionIdx].append({
                'context': context.copy(),
                'reward': reward,
            })
            # 記錄 history
            self.history.append({
                "context": context.tolist(),
                "action": actionIdx,
                "reward": reward,
            })
            modifiedActions.add(actionIdx)
        
        # 對有變動的 action 進行一次 mini-batch 更新
        for actionIdx in modifiedActions:
            self._train_mlp(actionIdx)
            print(f"[Bandit] update_batch: action={BanditAction(actionIdx).name}, "
                  f"buffer_size={len(self.replayBuffers[actionIdx])}")


    def _train_mlp(self, actionIdx: int):
        """
        對指定 action 的 MLP 做 mini-batch experience replay 更新。
        """
        buffer = self.replayBuffers[actionIdx]
        if len(buffer) < 2:
            return  # 資料太少，跳過

        model = self.mlpModels[actionIdx]
        optimizer = self.mlpOptimizers[actionIdx]
        criterion = nn.MSELoss()

        model.train()

        for _ in range(self.updateEpochs):
            # 從 buffer 隨機取樣 mini-batch
            batchSize = min(self.miniBatchSize, len(buffer))
            indices = np.random.choice(len(buffer), size=batchSize, replace=False)

            contexts = torch.tensor(
                np.array([buffer[i]['context'] for i in indices]),
                dtype=torch.float32, device=self.device
            )
            rewards = torch.tensor(
                np.array([buffer[i]['reward'] for i in indices]),
                dtype=torch.float32, device=self.device
            ).unsqueeze(1)

            # Forward + backward
            predictions = model(contexts)
            loss = criterion(predictions, rewards)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ─────────────────────────────────────────────
    # Reward 計算工具
    # ─────────────────────────────────────────────
    
    def _compute_lqp_base(self, meanEntropy: float) -> float:
        """
        LQP^base = w_e * mean_entropy
        """
        return self.weightEntropy * meanEntropy

    
    def _compute_gamma(self, alphaNorm: float, meanEntropy: float) -> float:
        """
        gamma_t = 1 + k * alpha_tilde * entropy_tilde
        alpha 調節因子：放大或縮小品質懲罰
        """
        return 1.0 + self.alp_ent_hyp * alphaNorm * meanEntropy

    def compute_reward(
        self,
        deltaLoss:    float,
        meanEntropy:  float,
        alphaNorm:    float,
        timeCost:     float = 1.0,
    ) -> float:
        """
        完整 reward 公式（平滑化版本）：
            r_t = -ΔLoss_t - λ * T_t - μ * γ_t * LQP^base_t

        ΔLoss = current_loss - last_loss：
            正值 = loss 上升 (模型變差)  → reward 為負
            負值 = loss 下降 (模型進步)  → reward 為正

        Args:
            deltaLoss:              驗證集 Cross-Entropy Loss 變化量
            meanEntropy:            這個 block 的 mean entropy（已正規化至 [0,1]）
            alphaNorm:              使用者 alpha power（已正規化至 [0,1]）
            timeCost:               這個 block 的時間成本（通常視為常數 1.0）

        Returns:
            reward (float)，clip 至 [rewardClipMin, rewardClipMax]
        """
        lqpBase = self._compute_lqp_base(meanEntropy)
        gamma = self._compute_gamma(alphaNorm, meanEntropy)
        lqp = gamma * lqpBase

        # 核心變更：-ΔLoss 使 loss 下降時 reward 為正
        raw_reward = -deltaLoss - self.lambdaCost * timeCost - self.mu * lqp
        reward = float(np.clip(raw_reward, self.rewardClipMin, self.rewardClipMax))

        print(f"[Bandit] compute_reward: "
              f" meanEntropy={meanEntropy:.4f}, alphaNorm={alphaNorm:.2f}, gamma={gamma}"
              f" deltaLoss={deltaLoss:.4f}, timeCost={timeCost:.2f}, lqp={lqp:.4f}"
              f" raw_reward={raw_reward:.4f}, clipped_reward={reward:.4f}")

        return reward

    # ─────────────────────────────────────────────
    # 儲存 / 載入
    # ─────────────────────────────────────────────
    def save(self, path: str):
        """
        儲存 NeuralUCB 參數：MLP state_dict + replay buffers + 超參數
        使用 torch.save 存為 .pt 格式
        """
        saveData = {
            "version": "neural_ucb_v1",
            "mlp_state_dicts": [m.state_dict() for m in self.mlpModels],
            "optimizer_state_dicts": [o.state_dict() for o in self.mlpOptimizers],
            "replay_buffers": [list(buf) for buf in self.replayBuffers],
            "history": self.history,
            "epsilon": self.epsilon,
            "lambda_cost": self.lambdaCost,
            "n_actions": self.nActions,
            "context_dim": self.contextDim,
        }
        torch.save(saveData, path)
        print(f"[Bandit] NeuralUCB saved to {path}")

    def load(self, path: str):
        """
        載入 NeuralUCB 參數。
        若檔案為舊版 LinUCB 格式（.npy），自動丟棄並重新初始化。
        """
        try:
            # 嘗試以 torch.load 載入
            data = torch.load(path, map_location=self.device, weights_only=False)

            # 版本檢查：舊 LinUCB 格式不含 "version" 欄位
            if not isinstance(data, dict) or data.get("version") != "neural_ucb_v1":
                print(f"[Bandit] 偵測到舊版或不相容格式，丟棄舊參數，使用新初始化")
                return

            # 驗證維度一致
            if data.get("n_actions") != self.nActions or data.get("context_dim") != self.contextDim:
                print(f"[Bandit] 維度不匹配（file: nActions={data.get('n_actions')}, "
                      f"contextDim={data.get('context_dim')}），使用新初始化")
                return

            # 載入 MLP 參數
            for i, stateDict in enumerate(data["mlp_state_dicts"]):
                self.mlpModels[i].load_state_dict(stateDict)
                self.mlpModels[i].to(self.device)

            for i, optStateDict in enumerate(data["optimizer_state_dicts"]):
                self.mlpOptimizers[i].load_state_dict(optStateDict)

            # 載入 replay buffers
            for i, bufData in enumerate(data["replay_buffers"]):
                self.replayBuffers[i] = deque(bufData, maxlen=self.replayBufferSize)

            self.history = data.get("history", [])
            self.epsilon = data.get("epsilon", self.epsilon)
            self.lambdaCost = data.get("lambda_cost", self.lambdaCost)

            print(f"[Bandit] NeuralUCB loaded from {path}, "
                  f"history_len={len(self.history)}, "
                  f"buffer_sizes={[len(b) for b in self.replayBuffers]}")

        except Exception as loadError:
            print(f"[Bandit] 載入失敗 ({loadError})，使用新初始化")


# ─────────────────────────────────────────────
# Legacy alias：向後相容舊程式碼的 import
# ─────────────────────────────────────────────
ContextualBandit = NeuralContextualBandit


# ─────────────────────────────────────────────────────
# UpdateScheduler：加權品質分數觸發模型更新
# ─────────────────────────────────────────────────────
class UpdateScheduler:
    """
    策略：加權品質分數 (Weighted Quality Score)

    每筆資料根據 Bandit Action 給予品質分數：
        HIGH_WEIGHT = 2.0
        INCLUDE     = 1.0
        LOW_WEIGHT  = 0.7

    累積分數達到 scoreThreshold 後，通知外部觸發一次模型更新。
    """

    # 每個 Action 對應的品質分數
    ACTION_QUALITY_SCORES = {
        BanditAction.INCLUDE:     1.0,
        BanditAction.HIGH_WEIGHT: 2.0,
        BanditAction.LOW_WEIGHT:  0.7,
    }

    def __init__(self, scoreThreshold: float = 7.0):
        """
        Args:
            scoreThreshold: 累積品質分數達到此閾值時觸發更新
        """
        self.scoreThreshold = scoreThreshold
        self.accumulatedScore = 0.0
        self.totalProcessedCount = 0

        print(f"[Bandit UpdateScheduler] 初始化完成")
        print(f" scoreThreshold={scoreThreshold}")
        print()

    def record(self, action: BanditAction) -> bool:
        """
        記錄一次 Bandit action 的結果。

        Args:
            action: 本次 Bandit 選擇的 Action

        Returns:
            shouldUpdate (bool): 是否應該觸發模型更新
        """
        score = self.ACTION_QUALITY_SCORES.get(action, 0.0)
        self.accumulatedScore += score
        self.totalProcessedCount += 1

        shouldUpdate = self.accumulatedScore >= self.scoreThreshold

        print(f"[UpdateScheduler] record: action={action.name}, score={score:.1f},"
                f" accumulated={self.accumulatedScore:.1f}/{self.scoreThreshold:.1f},"
                f" total={self.totalProcessedCount},"
                f" shouldUpdate={shouldUpdate}")

        return shouldUpdate

    def reset(self):
        """
        觸發更新後重置計數器。
        """
        print(f"[Bandit UpdateScheduler] reset: 清除累計分數 {self.accumulatedScore:.1f}")
        self.accumulatedScore = 0.0
        self.totalProcessedCount = 0


# ─────────────────────────────────────────────────────
# Context 正規化工具
# ─────────────────────────────────────────────────────
class ContextNormalizer:
    """
    負責把 raw context 數值正規化至 [0, 1]。
    使用 EMA（指數移動平均）追蹤 mean 和 variance，避免 running min-max
    在離群值出現時永久發散。
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: EMA 衰減係數，越大越快適應新資料
        """
        self._alpha = alpha
        self._emaMean = None   # shape: (context_dim,)
        self._emaVar = None    # shape: (context_dim,)
        self._count = 0

    def update_and_normalize(self, raw: np.ndarray) -> np.ndarray:
        """
        接受 raw context 向量，更新 EMA 統計量，回傳正規化後的向量。

        正規化方式：z-score → sigmoid 壓縮至 [0, 1]
            z = (raw - emaMean) / sqrt(emaVar + eps)
            normalized = sigmoid(z) = 1 / (1 + exp(-z))

        Args:
            raw: shape (context_dim,)，原始數值

        Returns:
            normalized: shape (context_dim,)，每維壓縮至 [0, 1]
        """
        epsilon = 1e-8

        if self._emaMean is None:
            self._emaMean = raw.copy()
            self._emaVar = np.ones_like(raw)  # 初始 variance = 1
            self._count = 1
            return np.full_like(raw, 0.5)  # 第一筆無法正規化，回傳中性值 0.5

        self._count += 1
        alpha = self._alpha

        # EMA 更新 mean 和 variance
        diff = raw - self._emaMean
        self._emaMean = (1 - alpha) * self._emaMean + alpha * raw
        self._emaVar = (1 - alpha) * self._emaVar + alpha * (diff ** 2)

        # z-score + sigmoid 壓縮
        stdDev = np.sqrt(self._emaVar + epsilon)
        zScore = (raw - self._emaMean) / stdDev
        normalized = 1.0 / (1.0 + np.exp(-zScore))  # sigmoid

        return normalized


# ─────────────────────────────────────────────────────
# ExperienceReplayMemory：雙重緩衝 Experience Replay
# ─────────────────────────────────────────────────────
import os
import glob

class ExperienceReplayMemory:
    """
    雙重緩衝 Experience Replay，防止 DL 在 RL Loop 中過擬合。

    Stable_Buffer:
        從 pretrain 資料（rl/rl_pretrain/data/）一次性載入（只讀）。
        提供穩定的歷史乾淨資料基座。

    Dynamic_Buffer:
        遊玩過程中被 RL 判定為 INCLUDE 或 HIGH_WEIGHT 的資料。
        使用 deque(maxlen=maxDynamicSize) 限制大小。

    抽樣策略：
        每次觸發更新時，從 Stable_Buffer 隨機抽 30%，
        從 Dynamic_Buffer 抽 70% 組成訓練集。
    """

    # 品質分數對照表（與 UpdateScheduler.ACTION_QUALITY_SCORES 一致）
    QUALITY_SCORES = {
        BanditAction.INCLUDE:     1.0,
        BanditAction.HIGH_WEIGHT: 2.0,
        BanditAction.LOW_WEIGHT:  0.7,
    }

    def __init__(self, stableDataDir: str, maxDynamicSize: int = 2000):
        """
        Args:
            stableDataDir:   pretrain .pt 檔案所在目錄
            maxDynamicSize:  Dynamic_Buffer 最大容量
        """
        self.stableSegments = []  # list of (segment_np, label_int)
        self.dynamicBuffer = deque(maxlen=maxDynamicSize)
        self.maxDynamicSize = maxDynamicSize

        self._load_stable_data(stableDataDir)

        print(f"[ExperienceReplayMemory] 初始化完成")
        print(f"  Stable_Buffer 大小: {len(self.stableSegments)}")
        print(f"  Dynamic_Buffer 最大容量: {maxDynamicSize}")
        print()

    def _load_stable_data(self, dataDir: str):
        """
        從 dataDir 載入所有 .pt 檔案作為 Stable_Buffer。
        每個 .pt 檔案預期包含 {'x_data': Tensor, 'y_data': Tensor}。
        """
        import torch

        ptFiles = sorted(glob.glob(os.path.join(dataDir, "**", "*.pt"), recursive=True))
        if not ptFiles:
            print(f"[ExperienceReplayMemory] 警告: {dataDir} 中沒有找到 .pt 檔案，Stable_Buffer 為空")
            return

        for ptFile in ptFiles:
            try:
                data = torch.load(ptFile, weights_only=False, map_location='cpu')
                xData = data['x_data'].numpy()  # shape: (N, channels, samples)
                yData = data['y_data'].numpy()   # shape: (N,)

                # 轉置為 (N, samples, channels) 以配合 sliding window
                xTransposed = np.transpose(xData, (0, 2, 1))

                for i in range(len(xTransposed)):
                    self.stableSegments.append((
                        xTransposed[i],   # shape: (samples, channels)
                        int(yData[i]),
                    ))

                print(f"[ExperienceReplayMemory] 載入 Stable: {ptFile}, 樣本數={len(xData)}")
            except Exception as loadError:
                print(f"[ExperienceReplayMemory] 載入失敗: {ptFile}, error={loadError}")

    def add_dynamic(self, segment: np.ndarray, label: int, action: 'BanditAction'):
        """
        將一筆高品質資料加入 Dynamic_Buffer。

        Args:
            segment:  shape (samples, channels)
            label:    類別標籤 (int)
            action:   Bandit 選擇的 action（用於記錄品質分數）
        """
        qualityScore = self.QUALITY_SCORES.get(action, 1.0)
        self.dynamicBuffer.append({
            'segment': segment.copy(),
            'label': label,
            'qualityScore': qualityScore,
            'action': int(action),
        })
        print(f"[ExperienceReplayMemory] add_dynamic: label={label}, action={BanditAction(action).name}, "
              f"dynamic_size={len(self.dynamicBuffer)}/{self.maxDynamicSize}")

    def add_dynamic_batch(self, segments: list, labels: list, actions: list):
        """
        批次加入多筆資料到 Dynamic_Buffer（所有 action 皆納入，以品質分數區分權重）。

        Args:
            segments: list of np.ndarray, each shape (samples, channels)
            labels:   list of int
            actions:  list of BanditAction
        """
        for seg, lab, act in zip(segments, labels, actions):
            self.add_dynamic(seg, lab, act)

    def sample(self, totalCount: int, stableRatio: float = 0.3):
        """
        從雙重緩衝區抽樣混合資料。
        當某個 buffer 為空時，自動將其配額轉給另一個 buffer。

        Args:
            totalCount:   要抽取的總樣本數
            stableRatio:  Stable_Buffer 的抽樣比例（預設 0.3）

        Returns:
            (segments, labels): 混合後的資料
                segments: list of np.ndarray, each shape (samples, channels)
                labels:   list of int
        """
        stableCount = int(totalCount * stableRatio)
        dynamicCount = totalCount - stableCount

        # 空抽防護：若某 buffer 為空，將配額轉給另一個
        if not self.stableSegments and not self.dynamicBuffer:
            print(f"[ExperienceReplayMemory] sample: 雙 buffer 皆空，回傳空列表")
            return [], []

        if not self.stableSegments:
            dynamicCount = totalCount
            stableCount = 0
            print(f"[ExperienceReplayMemory] sample: Stable_Buffer 為空，全部從 Dynamic 抽取")
        elif not self.dynamicBuffer:
            stableCount = totalCount
            dynamicCount = 0
            print(f"[ExperienceReplayMemory] sample: Dynamic_Buffer 為空，全部從 Stable 抽取")

        sampledSegments = []
        sampledLabels = []

        # 從 Stable_Buffer 抽樣
        if self.stableSegments and stableCount > 0:
            actualStableCount = min(stableCount, len(self.stableSegments))
            stableIndices = np.random.choice(
                len(self.stableSegments), size=actualStableCount, replace=False
            )
            for idx in stableIndices:
                seg, lab = self.stableSegments[idx]
                sampledSegments.append(seg)
                sampledLabels.append(lab)
            print(f"[ExperienceReplayMemory] sample: Stable 抽取 {actualStableCount}/{len(self.stableSegments)}")

        # 從 Dynamic_Buffer 抽樣（品質加權）
        if self.dynamicBuffer and dynamicCount > 0:
            actualDynamicCount = min(dynamicCount, len(self.dynamicBuffer))

            # 品質加權抽樣：qualityScore 越高，被選中機率越大
            qualityWeights = np.array([entry['qualityScore'] for entry in self.dynamicBuffer])
            weightSum = qualityWeights.sum()

            if weightSum > 0:
                qualityWeights = qualityWeights / weightSum
            else:
                # 全部 qualityScore 為 0（不應發生，但防禦）→ 均勻抽樣
                qualityWeights = np.ones(len(self.dynamicBuffer)) / len(self.dynamicBuffer)

            # replace=True 防止 actualDynamicCount > len(buffer) 時報錯
            useReplace = actualDynamicCount > len(self.dynamicBuffer)
            dynamicIndices = np.random.choice(
                len(self.dynamicBuffer), size=actualDynamicCount,
                replace=useReplace, p=qualityWeights
            )
            for idx in dynamicIndices:
                entry = self.dynamicBuffer[idx]
                sampledSegments.append(entry['segment'])
                sampledLabels.append(entry['label'])
            print(f"[ExperienceReplayMemory] sample: Dynamic 抽取 {actualDynamicCount}/{len(self.dynamicBuffer)}")

        print(f"[ExperienceReplayMemory] sample 完成: 總計 {len(sampledSegments)} 筆")
        return sampledSegments, sampledLabels

    def save_buffer(self, savePath: str, topK: int = 100):
        """
        將 Dynamic_Buffer 中品質最高的前 topK 筆資料存入檔案。

        Args:
            savePath:  儲存路徑 (e.g., account/Guest/hq_buffer.pt)
            topK:      保留的最高品質筆數
        """
        import torch

        if not self.dynamicBuffer:
            print(f"[ExperienceReplayMemory] save_buffer: Dynamic_Buffer 為空，跳過")
            return

        # 依 qualityScore 排序，取前 topK
        sortedEntries = sorted(self.dynamicBuffer, key=lambda e: e['qualityScore'], reverse=True)
        topEntries = sortedEntries[:topK]

        saveData = {
            'version': 'hq_buffer_v1',
            'entries': [{
                'segment': entry['segment'],
                'label': entry['label'],
                'qualityScore': entry['qualityScore'],
                'action': entry['action'],
            } for entry in topEntries],
        }

        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        torch.save(saveData, savePath)
        print(f"[ExperienceReplayMemory] save_buffer: 儲存 {len(topEntries)} 筆至 {savePath}")

    def load_buffer(self, loadPath: str):
        """
        從檔案載入先前儲存的高品質資料，預填 Dynamic_Buffer。

        Args:
            loadPath:  載入路徑 (e.g., account/Guest/hq_buffer.pt)
        """
        import torch

        if not os.path.exists(loadPath):
            print(f"[ExperienceReplayMemory] load_buffer: {loadPath} 不存在，跳過")
            return

        try:
            data = torch.load(loadPath, weights_only=False, map_location='cpu')
            if not isinstance(data, dict) or data.get('version') != 'hq_buffer_v1':
                print(f"[ExperienceReplayMemory] load_buffer: 格式不相容，跳過")
                return

            loadedCount = 0
            for entry in data['entries']:
                self.dynamicBuffer.append({
                    'segment': entry['segment'],
                    'label': entry['label'],
                    'qualityScore': entry['qualityScore'],
                    'action': entry['action'],
                })
                loadedCount += 1

            print(f"[ExperienceReplayMemory] load_buffer: 從 {loadPath} 載入 {loadedCount} 筆，"
                  f"dynamic_size={len(self.dynamicBuffer)}")
        except Exception as loadError:
            print(f"[ExperienceReplayMemory] load_buffer 失敗: {loadError}")

    @property
    def dynamicSize(self) -> int:
        return len(self.dynamicBuffer)

    @property
    def stableSize(self) -> int:
        return len(self.stableSegments)
