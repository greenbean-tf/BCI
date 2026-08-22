"""
RL_pretrain.py

離線 RL Pretrain 腳本：用 MIEXP 轉換後的 .pt 資料，模擬完整 RL 閉環流程，
預訓練 NeuralUCB Bandit 模型，供 2026_BCI_- 即時系統使用。

流程：
    1. 載入 MIEXP 轉換後的 .pt 檔案
    2. 對每個 trial 模擬 Bandit 決策流程
    3. 訓練 EEGNet 並計算 reward (delta_loss)
    4. 更新 Bandit MLP
    5. 使用 EWMA reward 作為早停指標
    6. 儲存 pretrained bandit.pt

使用方式：
    cd combine_all
    python RL_pretrain.py
    python RL_pretrain.py --pretrain-epochs 5 --ewma-threshold 0.3
    python RL_pretrain.py --dry-run

注意：
    - .pt 檔案須由 EEG_MIEXP/convert_to_bci_format.py 產出
    - 需要 base model (for_VR_4class.pth) 作為 EEGNet 起點
"""

import os
import sys
import glob
import argparse
import shutil
import numpy as np
import random
import time
from datetime import datetime

# For CUDA deterministic reproducibility on CUDA >= 10.2
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from sklearn.model_selection import train_test_split

# 本專案 imports
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

import EEG_DL.Utils.config as config
import EEG_DL.Utils.global_value as global_value

from RL.bandit import (
    NeuralContextualBandit, BanditAction, ContextIDX,
    UpdateScheduler, ContextNormalizer,
)
from RL.RL_interface import RL_interface

from EEG_DL.EEG.MI_train import EEGNet_our_ver, BraindecodeTrainer
from EEG_DL.EEG.data_process_np import EEGDataLoader_v2


from log_class import Tee

# ─────────────────────────────────────────────
# 設定區
# ─────────────────────────────────────────────

# EEGNet 訓練參數（離線 pretrain 用，比即時 calib 可以更長）
PRETRAIN_EEGNET_EPOCHS = 2      # EEGNet fine-tune epoch 數 (Update RL once need)
PRETRAIN_EEGNET_LR = 1e-5
PRETRAIN_EEGNET_BATCH_SIZE = 8
PRETRAIN_PATIENCE = 10

# RL 參數
DEFAULT_PRETRAIN_EPOCHS = 1     # 外層：遍歷所有資料的次數（可由 CLI 調整）(tsai: Update RL times!!!)
DEFAULT_EWMA_THRESHOLD = 0.5    # EWMA reward 早停閾值
DEFAULT_EWMA_ALPHA = 0.3        # EWMA 平滑係數（越大越靈敏）
DEFAULT_SCORE_THRESHOLD = 15.0  # UpdateScheduler 的品質分數閾值
DEFAULT_MIN_RETAIN_RATIO = 0.5  # 最低保留比例：至少 50% train trials 不被 DISCARD

# Sliding window 參數（與 BCI 一致，數據是 3 秒 = 1500 樣本）
TRAIL_SIZE = 1500
SEGMENT_LEN = 500
TRAIN_STRIDE = 20
VALID_STRIDE = 50
VALID_RATIO = 0.25



# 路徑
CONVERTED_PT_DIR = os.path.join(
    CURRENT_PATH,
    "RL", "RL_pretrain", "data"
)
OUTPUT_BANDIT_PATH = os.path.join(
    CURRENT_PATH,
    "RL", "RL_pretrain", "pretrained_bandit.pt"
)
OUTPUT_LOG_PATH = os.path.join(
    CURRENT_PATH, 
    "RL", "RL_pretrain", "log", f"RL_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)


SEED = 42


# ─────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────
def setSeed(seed=SEED):
    """設定全域隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


# ─────────────────────────────────────────────
# 核心 Pretrain 邏輯
# ─────────────────────────────────────────────
class RLPretrainer:
    """
    離線 RL Pretrain 控制器

    模擬 BCI 即時系統的完整 RL 閉環：
        1. 載入 .pt 資料
        2. 對每個 trial 進行 Bandit 決策
        3. 根據 action 調整 sliding window stride
        4. 訓練 EEGNet + 計算 reward
        5. 更新 Bandit
    """

    def __init__(
        self,
        pretrainEpochs      =DEFAULT_PRETRAIN_EPOCHS,
        ewmaThreshold       =DEFAULT_EWMA_THRESHOLD,
        ewmaAlpha           =DEFAULT_EWMA_ALPHA,
        scoreThreshold      =DEFAULT_SCORE_THRESHOLD,
        minRetainRatio      =DEFAULT_MIN_RETAIN_RATIO,
        convertedPtDir      =CONVERTED_PT_DIR,
        outputBanditPath    =OUTPUT_BANDIT_PATH,
        baseModelPath       =None,
        pretrainedBanditPath=None,
        dryRun              =False,
    ):
        self.pretrainEpochs         = pretrainEpochs
        self.ewmaThreshold          = ewmaThreshold
        self.ewmaAlpha              = ewmaAlpha
        self.minRetainRatio         = minRetainRatio
        self.convertedPtDir         = convertedPtDir
        self.outputBanditPath       = outputBanditPath
        
        self.pretrainedBanditPath   = pretrainedBanditPath
        self.dryRun                 = dryRun

        # RL 元件：使用現有的 RL_interface，與即時系統共用同一套代碼
        self.rlInterface = RL_interface(
            bandit=NeuralContextualBandit(epsilon=0.5, lambdaCost=0.1)
        )
        # 覆寫 RL_interface 內部的 bandit 參數
        # self.rlInterface.bandit = NeuralContextualBandit(epsilon=0.5, lambdaCost=0.1)
        
        # 如果提供預訓練 Bandit 模型，則載入（fine-tune 模式）
        if pretrainedBanditPath and os.path.exists(pretrainedBanditPath):
            try:
                self.rlInterface.bandit.load(pretrainedBanditPath)
                print(f"[RL_pretrain INFO] 已載入預訓練 Bandit 模型: {pretrainedBanditPath}")
                print(f"[RL_pretrain INFO]   history 長度={len(self.rlInterface.bandit.history)}")
            except Exception as loadErr:
                print(f"[RL_pretrain WARNING] 無法載入預訓練 Bandit: {loadErr}，使用新模型")
        else:
            print(f"[RL_pretrain INFO] 從頭訓練 Bandit 模型（未指定預訓練模型）")
        
        self.rlInterface.normalizer = ContextNormalizer()
        self.rlInterface.updateScheduler = UpdateScheduler(scoreThreshold=scoreThreshold)

        # 資料處理元件：使用現有的 EEGDataLoader_v2，與即時系統共用 sliding_window / to_dataset
        self.dataLoader = EEGDataLoader_v2(valid_ratio=VALID_RATIO)

        # 訓練狀態
        self.lastValLoss = float('inf')
        self.ewmaReward = 0.0
        self.ewmaInitialized = False
        self.totalUpdateCount = 0           # 同 record.md
        self.totalUpdateCount_trail = 0     # each trail 更新的總次數

        # DL Base model
        self.baseModelPath = baseModelPath or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "RL", "RL_pretrain", "train.pth"
        )

        # 裝置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[RL_pretrain RLPretrainer] 初始化完成")
        print(f"  pretrainEpochs={pretrainEpochs}")
        print(f"  ewmaThreshold={ewmaThreshold} (EWMA alpha={ewmaAlpha})")
        print(f"  scoreThreshold={scoreThreshold}")
        print(f"  minRetainRatio={minRetainRatio}")
        print(f"  baseModelPath={self.baseModelPath}")
        print(f"  pretrainedBanditPath={self.pretrainedBanditPath}")
        print(f"  device={self.device}")
        print(f"  dryRun={dryRun}")
        print()

    def _loadConvertedPtFiles(self):
        """載入所有轉換後的 .pt 檔案"""
        ptPattern = os.path.join(self.convertedPtDir, "**", "*.pt")
        ptFiles = sorted(glob.glob(ptPattern, recursive=True))

        if not ptFiles:
            raise FileNotFoundError(
                f"找不到轉換後的 .pt 檔案: {ptPattern}\n"
            )

        print(f"[RL_pretrain RLPretrainer] 找到 {len(ptFiles)} 個 .pt 檔案")
        for ptFile in ptFiles:
            print(f"  - {ptFile}")

        return ptFiles

    def _loadPtData(self, ptPath):
        """載入單一 .pt 檔案，回傳 (x_data, y_data, probs)"""
        data = torch.load(ptPath, weights_only=False)
        xData = data['x_data'].numpy()  # (T, C, S)
        yData = data['y_data'].numpy()  # (T,)
        probs = data['probs'].numpy()   # (T, n_class)
        print(f"[RL_pretrain DEBUG] _loadPtData: {ptPath}, x={xData.shape}, y={yData.shape}")
        return xData, yData, probs

    def _trainEegnet(self, trainDataset, validDataset):
        """
        訓練 EEGNet 並回傳 val_acc, val_loss

        使用 BCI 的 BraindecodeTrainer 進行訓練。
        """
        if len(trainDataset) == 0 or len(validDataset) == 0:
            print(f"[RL_pretrain WARNING] _trainEegnet: 資料不足，跳過")
            return 0.0, float('inf')

        # 建立模型（使用 dataset 推論參數）
        modelKwargs = {
            'n_chans': trainDataset[0][0].shape[1],    # channels
            'n_outputs': trainDataset[0][1].shape[0],  # n_class
            'n_times': trainDataset[0][0].shape[2],    # time
        }

        # 建立 trainer
        trainer = BraindecodeTrainer(
            model_class=EEGNet_our_ver,
            model_kwargs=modelKwargs,
            dataset=trainDataset,
            val_dataset=validDataset,
            batch_size=PRETRAIN_EEGNET_BATCH_SIZE,
            num_epochs=PRETRAIN_EEGNET_EPOCHS,
            lr=PRETRAIN_EEGNET_LR,
            ft=False,
            tcp_client=None,
        )

        # 嘗試載入 base model
        if os.path.exists(self.baseModelPath):
            try:
                trainer.load_checkpoint(self.baseModelPath)
                print(f"[RL_pretrain DEBUG] _trainEegnet: 載入 base model 成功")
            except Exception as loadErr:
                print(f"[RL_pretrain WARNING] _trainEegnet: 無法載入 base model: {loadErr}")

        # 確保 checkpoint 暫存目錄存在（BraindecodeTrainer.train 會寫入此目錄）
        os.makedirs(global_value.EEG_CHECKPOINT_TMP_BASE_DIR, exist_ok=True)
        print(f"[RL_pretrain DEBUG] _trainEegnet: checkpoint dir={global_value.EEG_CHECKPOINT_TMP_BASE_DIR}")

        # 訓練
        history = trainer.train(freeze_layer=False, patience=PRETRAIN_PATIENCE)

        # 取最佳 epoch
        bestEpoch = np.argmin(history['val_loss'])
        valAcc = history['val_acc'][bestEpoch]
        valLoss = history['val_loss'][bestEpoch]

        print(f"[RL_pretrain DEBUG] _trainEegnet: best_epoch={bestEpoch}, val_acc={valAcc:.4f}, val_loss={valLoss:.4f}")
        return valAcc, valLoss

    def _updateEwmaReward(self, reward):
        """更新 EWMA reward"""
        if not self.ewmaInitialized:
            self.ewmaReward = reward
            self.ewmaInitialized = True
        else:
            self.ewmaReward = self.ewmaAlpha * reward + (1 - self.ewmaAlpha) * self.ewmaReward

        print(f"[RL_pretrain DEBUG] EWMA reward: {self.ewmaReward:.4f}")
        return self.ewmaReward

    def _shouldEarlyStop(self):
        """
        EWMA 早停檢查

        品質標準：
            - EWMA reward > ewmaThreshold 表示 Bandit 已學到穩定正向策略
            - 且至少完成 2 輪更新，避免過早觸發
        """
        if self.totalUpdateCount < 2:
            return False

        if self.ewmaReward > self.ewmaThreshold:
            print(f"[RL_pretrain INFO] EWMA 早停觸發: ewmaReward={self.ewmaReward:.4f} > threshold={self.ewmaThreshold}")
            return True

        return False

    def run(self):
        """
        執行離線 RL Pretrain 主流程
        """

        print(f"\n{'#'*60}")
        print(f"# 離線 RL Pretrain 開始")
        print(f"{'#'*60}\n")

        # 載入資料
        ptFiles = self._loadConvertedPtFiles()

        # Epoch 迴圈
        earlyStopTriggered = False
        for epochIdx in range(self.pretrainEpochs):
            print(f"\n{'='*60}")
            print(f"[Epoch {epochIdx+1}/{self.pretrainEpochs}]")
            print(f"{'='*60}")

            for ptFile in ptFiles:
                if earlyStopTriggered:
                    break

                xData, yData, probs = self._loadPtData(ptFile)

                # 轉置: (T, C, S) → (T, S, C) 與 BCI prepare_datasets 一致
                xDataTransposed = np.transpose(xData, (0, 2, 1))

                # 分 train/valid (stratified)
                indices = np.arange(len(xData))
                try:
                    trainIdx, validIdx = train_test_split(
                        indices, 
                        test_size=VALID_RATIO,
                        stratify=yData, 
                        random_state=SEED
                    )
                except ValueError as splitErr:
                    print(f"[RL_pretrain WARNING] stratified split 失敗: {splitErr}，使用隨機切分")
                    trainIdx, validIdx = train_test_split(
                        indices, 
                        test_size=VALID_RATIO, 
                        random_state=SEED
                    )

                validIdxSet = set(validIdx)

                # ─── 模擬 RL 閉環（Two-Pass） ───
                augTrainSegments = []
                augTrainLabels = []
                augValidSegments = []
                augValidLabels = []
                sessionRecords = []

                # ─── Pass 1：收集所有 Bandit 決策 ───
                trainTrialDecisions = []  # list of (trialIdx, segment, context, action)

                for trialIdx in range(len(xDataTransposed)):
                    segment = xDataTransposed[trialIdx]  # (S, C)

                    if segment.shape[0] < SEGMENT_LEN:
                        continue

                    # Valid set: 不走 RL，直接處理
                    if trialIdx in validIdxSet:
                        windowList = self.dataLoader._sliding_window(VALID_STRIDE, segment)
                        if windowList:
                            labelList = [yData[trialIdx]] * len(windowList)
                            augValidSegments.extend(windowList)
                            augValidLabels.extend(labelList)
                        continue

                    # Train set: 走 RL
                    hitData = {
                        'eeg': segment.T,       # (C, S)
                        'probs': probs[trialIdx],  # (n_class,)
                    }

                    # 1. 提取 context（使用 RL_interface 的方法）
                    self.rlInterface.deltaValLoss = self.lastValLoss if self.lastValLoss != float('inf') else 0.0
                    rawContext = self.rlInterface.extract_per_hit_context(hitData)

                    # 2. 正規化 + Bandit 選擇 action
                    normContext = self.rlInterface.normalizer.update_and_normalize(rawContext)
                    action = self.rlInterface.bandit.select_action(normContext)

                    # 3. UpdateScheduler 記錄
                    shouldTriggerUpdate = self.rlInterface.updateScheduler.record(action)

                    trainTrialDecisions.append({
                        'trialIdx': trialIdx,
                        'segment': segment,
                        'context': normContext,
                        'action': action,
                    })

                # ─── Pass 2：最低保留比例檢查 ───
                totalTrainTrials = len(trainTrialDecisions)
                retainedCount = sum(
                    1 for d in trainTrialDecisions if d['action'] != BanditAction.DISCARD
                )
                currentRetainRatio = retainedCount / max(totalTrainTrials, 1)
                minRequired = max(2, int(totalTrainTrials * self.minRetainRatio))
                overrideCount = 0

                if retainedCount < minRequired:
                    # 找出所有被 DISCARD 的 trial，隨機升級為 NORMAL_WEIGHT
                    discardedIndices = [
                        i for i, d in enumerate(trainTrialDecisions)
                        if d['action'] == BanditAction.DISCARD
                    ]
                    needToRecover = minRequired - retainedCount
                    recoverIndices = random.sample(
                        discardedIndices, min(needToRecover, len(discardedIndices))
                    )
                    for recoverIdx in recoverIndices:
                        trainTrialDecisions[recoverIdx]['action'] = BanditAction.INCLUDE
                        overrideCount += 1

                    print(f"[RL_pretrain DEBUG] 最低保留比例觸發: "
                          f"原保留={retainedCount}/{totalTrainTrials} "
                          f"({currentRetainRatio:.1%}), "
                          f"最低要求={minRequired} ({self.minRetainRatio:.0%}), "
                          f"升級 {overrideCount} 個 DISCARD → INCLUDE")

                # ─── Pass 3：根據最終 action 產生訓練資料 ───
                for decision in trainTrialDecisions:
                    segment = decision['segment']
                    action = decision['action']
                    trialIdx = decision['trialIdx']

                    sessionRecords.append({
                        'context': decision['context'],
                        'action': action,
                    })

                    if action == BanditAction.DISCARD:
                        continue

                    currentStride = TRAIN_STRIDE
                    if action == BanditAction.HIGH_WEIGHT:
                        currentStride = max(1, TRAIN_STRIDE // 2)
                    elif action == BanditAction.LOW_WEIGHT:
                        currentStride = TRAIN_STRIDE * 2

                    windowList = self.dataLoader._sliding_window(currentStride, segment)
                    if not windowList:
                        continue
                    labelList = [yData[trialIdx]] * len(windowList)
                    augTrainSegments.extend(windowList)
                    augTrainLabels.extend(labelList)

                print(f"[RL_pretrain DEBUG] RL 決策統計: {_actionStats(sessionRecords)}, "
                      f"override={overrideCount}, "
                      f"train_windows={len(augTrainSegments)}, "
                      f"valid_windows={len(augValidSegments)}")

                # ─── 訓練 EEGNet ───
                if len(augTrainSegments) < 2 or len(augValidSegments) < 2:
                    print(f"[RL_pretrain WARNING] 資料不足: train={len(augTrainSegments)}, valid={len(augValidSegments)}，跳過")
                    continue

                if self.dryRun:
                    print(f"[RL_pretrain DRY-RUN] 跳過 EEGNet 訓練: train={len(augTrainSegments)}, valid={len(augValidSegments)}")
                    print(f"[RL_pretrain DRY-RUN] session_records: {len(sessionRecords)} 筆")
                    print(f"[RL_pretrain DRY-RUN] action 統計: {_actionStats(sessionRecords)}")

                    # Dry-run: 模擬 reward
                    fakeDeltaLoss = np.random.uniform(-0.3, 0.1)
                    fakeReward = self.rlInterface.bandit.compute_reward(
                        deltaLoss=fakeDeltaLoss,
                        meanEntropy=0.5,
                        alphaNorm=0.3,
                    )
                    if sessionRecords:
                        lastRecord = sessionRecords[-1]
                        self.rlInterface.bandit.update(lastRecord['context'], lastRecord['action'], fakeReward)
                        self.totalUpdateCount += 1
                    self._updateEwmaReward(fakeReward)

                else:
                    trainDataset = self.dataLoader._to_dataset(augTrainSegments, augTrainLabels)
                    validDataset = self.dataLoader._to_dataset(augValidSegments, augValidLabels)

                    valAcc, valLoss = self._trainEegnet(trainDataset, validDataset)

                    # 計算 delta_loss
                    deltaLoss = valLoss - self.lastValLoss if self.lastValLoss != float('inf') else 0.0
                    self.lastValLoss = valLoss

                    # 計算 reward
                    if sessionRecords:
                        for record in sessionRecords:
                            meanEntropy = float(record['context'][ContextIDX.MEAN_ENTROPY])
                            alphaNorm = float(record['context'][ContextIDX.ALPHA_POWER_NORM])

                            # 計算這筆紀錄的專屬 Reward (包含它自己的 lqp 懲罰)
                            reward = self.rlInterface.bandit.compute_reward(
                                deltaLoss=deltaLoss, # 這是整首歌共用的
                                meanEntropy=meanEntropy,
                                alphaNorm=alphaNorm,
                            )
                            
                            # 更新 Bandit
                            self.rlInterface.bandit.update(record['context'], record['action'], reward)
                            self.totalUpdateCount_trail += 1
                            
                            # 更新 EWMA
                            self._updateEwmaReward(reward)

                        # print last trail info
                        last_reward = self.rlInterface.bandit.compute_reward(
                            deltaLoss=deltaLoss,
                            meanEntropy=float(sessionRecords[-1]['context'][ContextIDX.MEAN_ENTROPY]),
                            alphaNorm=float(sessionRecords[-1]['context'][ContextIDX.ALPHA_POWER_NORM]),
                        )
                        print(f"[RL_pretrain INFO] this time last trail reward={last_reward:.4f}")

                        self.totalUpdateCount += 1
                        print(f"[RL_pretrain INFO] Update #{self.totalUpdateCount}: "
                                f"total_Update_trail={self.totalUpdateCount_trail} "
                                f"deltaLoss={deltaLoss:.4f} "
                                f"ewma={self.ewmaReward:.4f}, "
                                f"action_stats={_actionStats(sessionRecords)}")

                        
                        if False: # 看 record.md `已改待測 @1` 取最後一筆
                            lastRecord = sessionRecords[-1]
                            meanEntropy = float(lastRecord['context'][ContextIDX.MEAN_ENTROPY])
                            alphaNorm = float(lastRecord['context'][ContextIDX.ALPHA_POWER_NORM])

                            reward = self.rlInterface.bandit.compute_reward(
                                deltaLoss=deltaLoss,
                                meanEntropy=meanEntropy,
                                alphaNorm=alphaNorm,
                            )

                            # 更新 Bandit
                            self.rlInterface.bandit.update(lastRecord['context'], lastRecord['action'], reward)
                            self.totalUpdateCount += 1

                            # 更新 EWMA
                            self._updateEwmaReward(reward)

                            print(f"[RL_pretrain INFO] Update #{self.totalUpdateCount}: "
                                f"deltaLoss={deltaLoss:.4f}, reward={reward:.4f}, "
                                f"ewma={self.ewmaReward:.4f}, "
                                f"action_stats={_actionStats(sessionRecords)}")


                    self.rlInterface.updateScheduler.reset()

                # 早停檢查
                if self._shouldEarlyStop():
                    earlyStopTriggered = True
                    print(f"[RL_pretrain INFO] 早停觸發，結束 pretrain")
                    break

            if earlyStopTriggered:
                break

        # ─── 儲存 Bandit ───
        self.rlInterface.bandit.save(self.outputBanditPath)
        print(f"\n{'='*60}")
        print(f"[RL_pretrain INFO] Pretrained Bandit 已儲存: {self.outputBanditPath}")
        print(f"[RL_pretrain INFO] 總更新次數: {self.totalUpdateCount}")
        print(f"[RL_pretrain INFO] 最終 EWMA reward: {self.ewmaReward:.4f}")
        print(f"[RL_pretrain INFO] History 長度: {len(self.rlInterface.bandit.history)}")
        print(f"[RL_pretrain INFO] Buffer sizes: {[len(b) for b in self.rlInterface.bandit.replayBuffers]}")
        print(f"{'='*60}")

        return self.outputBanditPath


def _actionStats(sessionRecords):
    """統計 action 分佈"""
    stats = {}
    for record in sessionRecords:
        actionName = BanditAction(record['action']).name
        stats[actionName] = stats.get(actionName, 0) + 1
    return stats


# ─────────────────────────────────────────────
# 遷移工具
# ─────────────────────────────────────────────
def transferBanditToAccount(banditPath, accountName="Guest"):
    """
    將 pretrained bandit.pt 複製到 BCI 帳戶目錄

    Args:
        banditPath: pretrained_bandit.pt 路徑
        accountName: BCI 帳戶名稱
    """
    accountDir = os.path.join("account", accountName)
    targetPath = os.path.join(accountDir, config.bandit_RL)  # "bandit.pt"

    if not os.path.exists(banditPath):
        print(f"[RL_pretrain ERROR] Bandit 檔案不存在: {banditPath}")
        return False

    os.makedirs(accountDir, exist_ok=True)
    shutil.copy2(banditPath, targetPath)
    print(f"[RL_pretrain INFO] Bandit 已遷移: {banditPath} → {targetPath}")

    # 驗證載入
    testBandit = NeuralContextualBandit()
    testBandit.load(targetPath)
    print(f"[RL_pretrain INFO] 載入驗證通過: history={len(testBandit.history)}")

    return True



# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────


def main(args):
    pretrainer = RLPretrainer(
        pretrainEpochs=args.pretrain_epochs,
        ewmaThreshold=args.ewma_threshold,
        ewmaAlpha=args.ewma_alpha,
        scoreThreshold=args.score_threshold,
        minRetainRatio=args.min_retain_ratio,
        convertedPtDir=args.converted_pt_dir,
        outputBanditPath=args.output,
        baseModelPath=args.base_model,
        pretrainedBanditPath=args.pretrained_bandit,
        dryRun=args.dry_run,
    )

    banditPath = pretrainer.run()

    # 遷移
    if args.transfer:
        transferBanditToAccount(banditPath, accountName=args.transfer)





# ─────────────────────────────────────────────
# pre Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    setSeed()


    parser = argparse.ArgumentParser(description="離線 RL Pretrain")
    parser.add_argument("--pretrain-epochs", type=int, default=DEFAULT_PRETRAIN_EPOCHS,
                        help=f"Pretrain 外層 epoch 數 (default: {DEFAULT_PRETRAIN_EPOCHS})")

    parser.add_argument("--ewma-threshold", type=float, default=DEFAULT_EWMA_THRESHOLD,
                        help=f"EWMA reward 早停閾值 (default: {DEFAULT_EWMA_THRESHOLD})")

    parser.add_argument("--ewma-alpha", type=float, default=DEFAULT_EWMA_ALPHA,
                        help=f"EWMA 平滑係數 (default: {DEFAULT_EWMA_ALPHA})")

    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD,
                        help=f"UpdateScheduler 品質分數閾值 (default: {DEFAULT_SCORE_THRESHOLD})")

    parser.add_argument("--min-retain-ratio", type=float, default=DEFAULT_MIN_RETAIN_RATIO,
                        help=f"最低保留比例，防止全 DISCARD (default: {DEFAULT_MIN_RETAIN_RATIO})")

    parser.add_argument("--converted-pt-dir", type=str, default=CONVERTED_PT_DIR,
                        help="轉換後 .pt 資料目錄")

    parser.add_argument("--output", type=str, default=OUTPUT_BANDIT_PATH,
                        help="輸出 bandit.pt 路徑")

    parser.add_argument("--base-model", type=str, default=None,
                        help="EEGNet base model 路徑")

    parser.add_argument("--pretrained-bandit", type=str, default=None,
                        help="預訓練 Bandit 模型路徑 (fine-tune 模式)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Dry run: 不實際訓練 EEGNet，用隨機 reward 測試流程")

    parser.add_argument("--transfer", type=str, default=None,
                        help="Pretrain 完成後遷移至指定帳戶 (e.g. Guest)")

    args = parser.parse_args()

    

    # 紀錄 log
    original_stdout = sys.stdout
    
    with open(OUTPUT_LOG_PATH, "w", encoding="utf-8") as f:
        sys.stdout = Tee(original_stdout, f)
        try:
            main(args)
        finally:
            sys.stdout = original_stdout
            print(f"\n✅ 輸出已保存到 {OUTPUT_LOG_PATH}")


    
