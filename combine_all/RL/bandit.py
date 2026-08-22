### python/main/EEG/bandit.py
### 包含 `NeuralContextualBandit`（NeuralUCB）、`BanditAction` enum、`UpdateScheduler`

"""
Contextual Bandit for VR-BCI calibration flow
使用 NeuralUCB 演算法（MLP + 梯度 norm 探索項），

Action 定義：
    A0 - INCLUDE:        納入訓練 buffer (預設行為)
    A1 - DISCARD:        丟棄這 5 筆資料
    A2 - HIGH_WEIGHT:    高權重納入 buffer
    A3 - LOW_WEIGHT:     低權重納入 buffer

Context 向量 (4 維): 
    [mean_entropy, mean_margin, delta_loss, alpha_power_normalized]

升級紀錄：
    - Reward 平滑化：ΔAcc → ΔLoss（Cross-Entropy Loss 下降量）
    - NeuralUCB：MLP 取代線性矩陣 A/b
    - 解耦動作空間：移除 ONLINE_UPDATE，由 UpdateScheduler 獨立排程
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
    INCLUDE        = 0
    DISCARD        = 1
    HIGH_WEIGHT    = 2
    LOW_WEIGHT     = 3

    n_actions      = 4


class ContextIDX(IntEnum):
    MEAN_ENTROPY                = 0
    MEAN_MARGIN                 = 1
    DELTA_LOSS                  = 2   # 修改：從 DELTA_ACC 改為 DELTA_LOSS
    ALPHA_POWER_NORM            = 3

    context_dim                 = 4



# ─────────────────────────────────────────────
# RewardMLP：輕量 MLP 預測 expected reward
# ─────────────────────────────────────────────
class RewardMLP(nn.Module):
    """
    輕量 MLP 用於預測給定 context 下的 expected reward。
    結構：context_dim → 64 → 32 → 1
    每個 action 獨立持有一個 RewardMLP。
    """
    def __init__(self, contextDim: int):
        super(RewardMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(contextDim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
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
            optimizer = optim.Adam(model.parameters(), lr=mlpLearningRate)
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
        lastLayer = model.network[-1]  # 最後一個 Linear(32, 1)
        
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
        LOW_WEIGHT  = 0.3
        DISCARD     = 0.0

    累積分數達到 scoreThreshold 後，通知外部觸發一次模型更新。
    """

    # 每個 Action 對應的品質分數
    ACTION_QUALITY_SCORES = {
        BanditAction.INCLUDE:     1.0,
        BanditAction.DISCARD:     0.0,
        BanditAction.HIGH_WEIGHT: 2.0,
        BanditAction.LOW_WEIGHT:  0.3,
    }

    def __init__(self, scoreThreshold: float = 15.0):
        """
        Args:
            scoreThreshold: 累積品質分數達到此閾值時觸發更新
        """
        self.scoreThreshold = scoreThreshold
        self.accumulatedScore = 0.0
        self.totalProcessedCount = 0
        self.highQualityCount = 0  # INCLUDE + HIGH_WEIGHT 的數量

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

        if action in (BanditAction.INCLUDE, BanditAction.HIGH_WEIGHT):
            self.highQualityCount += 1

        shouldUpdate = self.accumulatedScore >= self.scoreThreshold

        if False:
            print(f"[Bandit UpdateScheduler] record: action={action.name}, score={score:.1f},"
                    f" accumulated={self.accumulatedScore:.1f}/{self.scoreThreshold:.1f},"
                    f" total={self.totalProcessedCount}, hq={self.highQualityCount},"
                    f" shouldUpdate={shouldUpdate}")

        return shouldUpdate

    def reset(self):
        """
        觸發更新後重置計數器。
        """
        print(f"[Bandit UpdateScheduler] reset: 清除累計分數 {self.accumulatedScore:.1f}")
        self.accumulatedScore = 0.0
        self.totalProcessedCount = 0
        self.highQualityCount = 0


# ─────────────────────────────────────────────────────
# Context 正規化工具
# ─────────────────────────────────────────────────────
class ContextNormalizer:
    """
    負責把 raw context 數值正規化至 [0, 1]。
    各統計量的 min / max 在第一個 block 後開始動態更新（running min-max）。
    """

    def __init__(self):
        self._min = None
        self._max = None

    def update_and_normalize(self, raw: np.ndarray) -> np.ndarray:
        """
        接受 raw context 向量，更新 running min-max，回傳正規化後的向量。

        Args:
            raw: shape (context_dim,)，原始數值

        Returns:
            normalized: shape (context_dim,)，每維 clip 至 [0, 1]
        """
        if self._min is None:
            self._min = raw.copy()
            self._max = raw.copy()
            return np.zeros_like(raw)  # 第一筆無法正規化，回傳全 0

        self._min = np.minimum(self._min, raw)
        self._max = np.maximum(self._max, raw)

        denom = self._max - self._min
        denom = np.where(denom == 0, 1.0, denom)  # 避免除以 0
        normalized = (raw - self._min) / denom
        return np.clip(normalized, 0.0, 1.0)
