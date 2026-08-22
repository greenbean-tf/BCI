"""
RL Pretrain 離線腳本（combine_all_new 版）。

import 路徑更新為 eeg.* / rl.*。
完整邏輯與原 RL_pretrain.py 一致。
"""
import os, sys, glob, argparse, shutil, random, time
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 無顯示器環境也可存檔
import matplotlib.pyplot as plt
from datetime import datetime

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from sklearn.model_selection import train_test_split
from collections import deque

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

import config
import global_state

from rl.bandit import (NeuralContextualBandit, BanditAction, ContextIDX,
                        UpdateScheduler, ContextNormalizer)
from rl.rl_interface import RL_interface
from eeg.eeg_model import EEGNet_our_ver
from eeg.braindecode_trainer import BraindecodeTrainer
from eeg.data_loader import EEGDataLoader_v2
from utils.log_class import Tee

# ─── 設定區 ───
PRETRAIN_EEGNET_EPOCHS      = 10
PRETRAIN_EEGNET_LR          = 1e-5
PRETRAIN_EEGNET_BATCH_SIZE  = 8
PRETRAIN_PATIENCE           = 10

DEFAULT_PRETRAIN_EPOCHS     = 10
DEFAULT_SCORE_THRESHOLD     = 15.0

TRAIL_SIZE      = 1500 
SEGMENT_LEN     = 500
TRAIN_STRIDE    = 100
VALID_STRIDE    = 200
VALID_RATIO     = 0.25

CONVERTED_PT_DIR    = os.path.join(CURRENT_PATH, "rl", "rl_pretrain", "data")
OUTPUT_BANDIT_PATH  = os.path.join(CURRENT_PATH, "rl", "rl_pretrain", "pretrained_bandit.pt")
OUTPUT_LOG_DIR      = os.path.join(CURRENT_PATH, "rl", "rl_pretrain", "log")

SEED = 42


def setSeed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class RLPretrainer:
    def __init__(self, pretrainEpochs=1,
                 scoreThreshold=15.0,
                 convertedPtDir=CONVERTED_PT_DIR, outputBanditPath=OUTPUT_BANDIT_PATH,
                 baseModelPath=None, pretrainedBanditPath=None, dryRun=False):
        
        self.pretrainEpochs         = pretrainEpochs
        self.convertedPtDir         = convertedPtDir
        self.outputBanditPath       = outputBanditPath
        self.pretrainedBanditPath   = pretrainedBanditPath
        self.dryRun                 = dryRun

        self.rlInterface = RL_interface(bandit=NeuralContextualBandit(epsilon=0.5, lambdaCost=0.1))

        if pretrainedBanditPath and os.path.exists(pretrainedBanditPath):
            try:
                self.rlInterface.bandit.load(pretrainedBanditPath)
                print(f"[RL_pretrain INFO] 已載入預訓練 Bandit: {pretrainedBanditPath}")
                print(f"[RL_pretrain INFO]   history 長度={len(self.rlInterface.bandit.history)}")
            except Exception as e:
                print(f"[RL_pretrain WARNING] 無法載入預訓練 Bandit: {e}，使用新模型")

        self.rlInterface.normalizer = ContextNormalizer()
        self.rlInterface.updateScheduler = UpdateScheduler(scoreThreshold=scoreThreshold)
        self.dataLoader = EEGDataLoader_v2(valid_ratio=VALID_RATIO)

        self.lastValLoss = float('inf')
        self.totalUpdateCount = 0
        self.totalUpdateCount_trail = 0

        # ─── Train/Valid MSE Loss 歷史記錄（per epoch） ───
        self.epochTrainMseHistory: list[float] = []
        self.epochValidMseHistory: list[float] = []
        self.banditValidationBuffers = [deque(maxlen=500) for _ in range(3)]

        # ─── EEGNet Train/Valid Acc & Loss 歷史記錄（per update） ───
        self.eegnetTrainAccHistory: list[float] = []
        self.eegnetValidAccHistory: list[float] = []
        self.eegnetTrainLossHistory: list[float] = []
        self.eegnetValidLossHistory: list[float] = []

        self.baseModelPath = baseModelPath or os.path.join(CURRENT_PATH, "rl", "rl_pretrain", "train.pth")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[RL_pretrain RLPretrainer] 初始化完成")
        print(f"  pretrainEpochs    ={pretrainEpochs}")
        print(f"  scoreThreshold    ={scoreThreshold}")
        print(f"  baseModelPath     ={self.baseModelPath}")
        print(f"  device            ={self.device}")
        print(f"  dryRun            ={dryRun}")
        print()

    # ─── 計算 Bandit MLP 的 MSE Loss ───
    def _computeBanditMseLoss(self, buffers: list):
        """
        計算所有 action MLP 在給定 buffers 上的平均 MSE Loss。
        用於追蹤 Bandit 模型的收斂與泛化程度。
        """
        totalLoss = 0.0
        totalSamples = 0

        for actionIdx in range(self.rlInterface.bandit.nActions):
            buffer = buffers[actionIdx]
            if len(buffer) < 1:
                continue

            model = self.rlInterface.bandit.mlpModels[actionIdx]
            model.eval()

            contexts = torch.tensor(
                np.array([entry['context'] for entry in buffer]),
                dtype=torch.float32, device=self.rlInterface.bandit.device
            )
            rewards = torch.tensor(
                np.array([entry['reward'] for entry in buffer]),
                dtype=torch.float32, device=self.rlInterface.bandit.device
            ).unsqueeze(1)

            with torch.no_grad():
                predictions = model(contexts)
                mse = torch.nn.functional.mse_loss(predictions, rewards).item()

            totalLoss += mse * len(buffer)
            totalSamples += len(buffer)

        if totalSamples == 0:
            return 0.0
        return totalLoss / totalSamples

    # ─── 圖表輸出 ───
    def _savePlot(self, outputDir: str):
        """訓練結束後將 Bandit MSE Loss + EEGNet Acc/Loss 曲線存為 PNG"""
        os.makedirs(outputDir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ── 1. Bandit MSE Loss 圖 ──
        if self.epochTrainMseHistory:
            savePath = os.path.join(outputDir, f"mse_loss_{timestamp}.png")
            epochs = list(range(1, len(self.epochTrainMseHistory) + 1))

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(epochs, self.epochTrainMseHistory,
                    color="dodgerblue", linewidth=2.0, marker='o', markersize=5, label="Train MSE Loss")
            ax.plot(epochs, self.epochValidMseHistory,
                    color="tomato", linewidth=2.0, marker='s', markersize=5, label="Valid MSE Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.set_title("RL Pretrain — Bandit MLP Train/Valid MSE Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(savePath, dpi=150)
            plt.close(fig)
            print(f"[RL_pretrain INFO] MSE Loss 圖表已儲存: {savePath}")
        else:
            print("[RL_pretrain WARNING] 無 MSE Loss 資料可繪圖")

        # ── 2. EEGNet Accuracy 圖 ──
        if self.eegnetTrainAccHistory:
            savePath = os.path.join(outputDir, f"eegnet_acc_{timestamp}.png")
            updates = list(range(1, len(self.eegnetTrainAccHistory) + 1))

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(updates, [a * 100 for a in self.eegnetTrainAccHistory],
                    color="dodgerblue", linewidth=2.0, marker='o', markersize=5, label="Train Acc")
            ax.plot(updates, [a * 100 for a in self.eegnetValidAccHistory],
                    color="tomato", linewidth=2.0, marker='s', markersize=5, label="Valid Acc")
            ax.axhline(y=25, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='Chance (25%)')

            # 標注最佳 valid acc
            bestIdx = int(np.argmax(self.eegnetValidAccHistory))
            bestAcc = self.eegnetValidAccHistory[bestIdx] * 100
            ax.scatter([bestIdx + 1], [bestAcc], color='tomato', s=120, zorder=5,
                       edgecolors='black', linewidths=1.5)
            ax.annotate(f'Best: {bestAcc:.1f}%',
                        xy=(bestIdx + 1, bestAcc),
                        xytext=(bestIdx + 2, bestAcc + 3),
                        fontsize=11, fontweight='bold', color='tomato',
                        arrowprops=dict(arrowstyle='->', color='tomato'))

            ax.set_xlabel("Update #", fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.set_title(f"RL Pretrain — EEGNet Train/Valid Accuracy (stride={TRAIN_STRIDE})",
                         fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(savePath, dpi=150)
            plt.close(fig)
            print(f"[RL_pretrain INFO] EEGNet Acc 圖表已儲存: {savePath}")

        # ── 3. EEGNet Loss 圖 ──
        if self.eegnetTrainLossHistory:
            savePath = os.path.join(outputDir, f"eegnet_loss_{timestamp}.png")
            updates = list(range(1, len(self.eegnetTrainLossHistory) + 1))

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(updates, self.eegnetTrainLossHistory,
                    color="dodgerblue", linewidth=2.0, marker='o', markersize=5, label="Train Loss")
            ax.plot(updates, self.eegnetValidLossHistory,
                    color="tomato", linewidth=2.0, marker='s', markersize=5, label="Valid Loss")

            bestIdx = int(np.argmin(self.eegnetValidLossHistory))
            bestLoss = self.eegnetValidLossHistory[bestIdx]
            ax.scatter([bestIdx + 1], [bestLoss], color='tomato', s=120, zorder=5,
                       edgecolors='black', linewidths=1.5)
            ax.annotate(f'Best: {bestLoss:.4f}',
                        xy=(bestIdx + 1, bestLoss),
                        xytext=(bestIdx + 2, bestLoss + 0.05),
                        fontsize=11, fontweight='bold', color='tomato',
                        arrowprops=dict(arrowstyle='->', color='tomato'))

            ax.set_xlabel("Update #", fontsize=12)
            ax.set_ylabel("Loss", fontsize=12)
            ax.set_title(f"RL Pretrain — EEGNet Train/Valid Loss (stride={TRAIN_STRIDE})",
                         fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(savePath, dpi=150)
            plt.close(fig)
            print(f"[RL_pretrain INFO] EEGNet Loss 圖表已儲存: {savePath}")
        else:
            print("[RL_pretrain WARNING] 無 EEGNet Acc/Loss 資料可繪圖 (dry-run 模式不產生)")

    def run(self):
        ptFiles = sorted(glob.glob(os.path.join(self.convertedPtDir, "**", "*.pt"), recursive=True))
        if not ptFiles:
            raise FileNotFoundError(f"No .pt files in {self.convertedPtDir}")
        print(f"Found {len(ptFiles)} .pt files")

        # ─── 載入並合併所有 .pt 檔案資料 ───
        all_xData, all_yData, all_probs = [], [], []
        for ptFile in ptFiles:
            print(f"[RL_pretrain INFO] Loading {ptFile}...")
            data = torch.load(ptFile, weights_only=False, map_location='cpu')
            all_xData.append(data['x_data'].numpy())
            all_yData.append(data['y_data'].numpy())
            all_probs.append(data['probs'].numpy())
        
        xData = np.concatenate(all_xData, axis=0)
        yData = np.concatenate(all_yData, axis=0)
        probs = np.concatenate(all_probs, axis=0)
        xT = np.transpose(xData, (0, 2, 1))
        
        print(f"[RL_pretrain INFO] Combined Data Shape: xData={xData.shape}, yData={yData.shape}")

        for epoch in range(self.pretrainEpochs):
            print(f"\n[Epoch {epoch+1}/{self.pretrainEpochs}]")
            
            indices = np.arange(len(xData))
            try:
                # 每個 Epoch 都保持相同的 Valid Set (根據 SEED)
                trainIdx, validIdx = train_test_split(indices, test_size=VALID_RATIO, stratify=yData, random_state=SEED)
            except ValueError:
                trainIdx, validIdx = train_test_split(indices, test_size=VALID_RATIO, random_state=SEED)
            validSet = set(validIdx)

            augTrain, augTrainLab = [], []
            augValid, augValidLab = [], []
            records = []
            val_records = []

            for i in range(len(xT)):
                seg = xT[i]
                if seg.shape[0] < SEGMENT_LEN: continue
                if i in validSet:
                    wl = self.dataLoader._sliding_window(VALID_STRIDE, seg)
                    if wl:
                        augValid.extend(wl); augValidLab.extend([yData[i]]*len(wl))
                    
                    # 紀錄 validation trial 的 context
                    rawCtx_val = self.rlInterface.extract_per_hit_context({'eeg': seg.T, 'probs': probs[i]})
                    normCtx_val = self.rlInterface.normalizer.update_and_normalize(rawCtx_val)
                    val_records.append({'context': normCtx_val, 'segment': seg, 'trialIdx': i})
                    continue

                # Train trial
                rawCtx = self.rlInterface.extract_per_hit_context({'eeg': seg.T, 'probs': probs[i]})
                normCtx = self.rlInterface.normalizer.update_and_normalize(rawCtx)
                action = self.rlInterface.bandit.select_action(normCtx)
                records.append({'context': normCtx, 'action': action, 'segment': seg, 'trialIdx': i})

            # 所有 record 皆納入訓練（根據 action 調整 stride）
            for r in records:
                stride = TRAIN_STRIDE
                if r['action'] == BanditAction.HIGH_WEIGHT: stride = max(1, stride//2)
                elif r['action'] == BanditAction.LOW_WEIGHT: stride = stride*2
                wl = self.dataLoader._sliding_window(stride, r['segment'])
                if wl:
                    augTrain.extend(wl); augTrainLab.extend([yData[r['trialIdx']]]*len(wl))

            if len(augTrain) < 2 or len(augValid) < 2: 
                print("[RL_pretrain WARNING] 資料不足，跳過此 Epoch")
                continue

            if self.dryRun:
                fakeDelta = np.random.uniform(-0.3, 0.1)
                fakeReward = self.rlInterface.bandit.compute_reward(deltaLoss=fakeDelta, meanEntropy=0.5, alphaNorm=0.3)
                if records:
                    batch_contexts = [rec['context'] for rec in records]
                    batch_actions = [rec['action'] for rec in records]
                    batch_rewards = [fakeReward] * len(records)
                    self.rlInterface.bandit.update_batch(batch_contexts, batch_actions, batch_rewards)
                    self.totalUpdateCount_trail += len(records)
                
                # 紀錄驗證集的 context-reward 到 validation buffer 中
                if val_records:
                    for rec in val_records:
                        action_val = self.rlInterface.bandit.select_action(rec['context'])
                        actionIdx = int(action_val)
                        self.banditValidationBuffers[actionIdx].append({
                            'context': rec['context'].copy(),
                            'reward': fakeReward
                        })
                    
                self.totalUpdateCount += 1

            else:
                trainDs = self.dataLoader._to_dataset(augTrain, augTrainLab)
                validDs = self.dataLoader._to_dataset(augValid, augValidLab)
                mk = {'n_chans': trainDs[0][0].shape[1], 'n_outputs': trainDs[0][1].shape[0], 'n_times': trainDs[0][0].shape[2]}
                trainer = BraindecodeTrainer(EEGNet_our_ver, mk, trainDs, validDs,
                                             PRETRAIN_EEGNET_BATCH_SIZE, PRETRAIN_EEGNET_EPOCHS,
                                             PRETRAIN_EEGNET_LR, False, None)
                if os.path.exists(self.baseModelPath):
                    try: trainer.load_checkpoint(self.baseModelPath)
                    except: pass
                os.makedirs(global_state.EEG_CHECKPOINT_TMP_BASE_DIR, exist_ok=True)
                hist = trainer.train(freeze_layer=False, patience=PRETRAIN_PATIENCE)
                bestEp = np.argmin(hist['val_loss'])
                valLoss = hist['val_loss'][bestEp]
                deltaLoss = valLoss - self.lastValLoss if self.lastValLoss != float('inf') else 0.0
                self.lastValLoss = valLoss

                # 記錄 EEGNet 的 best epoch acc/loss
                self.eegnetTrainAccHistory.append(hist['acc'][bestEp])
                self.eegnetValidAccHistory.append(hist['val_acc'][bestEp])
                self.eegnetTrainLossHistory.append(hist['loss'][bestEp])
                self.eegnetValidLossHistory.append(hist['val_loss'][bestEp])

                batch_contexts = []
                batch_actions = []
                batch_rewards = []
                for rec in records:
                    reward = self.rlInterface.bandit.compute_reward(
                        deltaLoss=deltaLoss,
                        meanEntropy=float(rec['context'][ContextIDX.MEAN_ENTROPY]),
                        alphaNorm=float(rec['context'][ContextIDX.ALPHA_POWER_NORM]))
                    batch_contexts.append(rec['context'])
                    batch_actions.append(rec['action'])
                    batch_rewards.append(reward)
                    self.totalUpdateCount_trail += 1

                if batch_contexts:
                    self.rlInterface.bandit.update_batch(batch_contexts, batch_actions, batch_rewards)

                # 紀錄驗證集的 context-reward 到 validation buffer 中
                if val_records:
                    for rec in val_records:
                        reward_val = self.rlInterface.bandit.compute_reward(
                            deltaLoss=deltaLoss,
                            meanEntropy=float(rec['context'][ContextIDX.MEAN_ENTROPY]),
                            alphaNorm=float(rec['context'][ContextIDX.ALPHA_POWER_NORM]))
                        action_val = self.rlInterface.bandit.select_action(rec['context'])
                        actionIdx = int(action_val)
                        self.banditValidationBuffers[actionIdx].append({
                            'context': rec['context'].copy(),
                            'reward': reward_val
                        })

                self.totalUpdateCount += 1
                print(f"[RL_pretrain INFO] Update #{self.totalUpdateCount}: "
                      f"total_Update_trail={self.totalUpdateCount_trail} "
                      f"deltaLoss={deltaLoss:.4f}")

            self.rlInterface.updateScheduler.reset()

            # ─── Epoch 結束：計算並記錄 Train/Valid MSE Loss ───
            epochTrainMse = self._computeBanditMseLoss(self.rlInterface.bandit.replayBuffers)
            epochValidMse = self._computeBanditMseLoss(self.banditValidationBuffers)
            self.epochTrainMseHistory.append(epochTrainMse)
            self.epochValidMseHistory.append(epochValidMse)
            print(f"[RL_pretrain INFO] Epoch {epoch+1} "
                  f"Train MSE: {epochTrainMse:.6f} ({epochTrainMse:.4e}) | "
                  f"Valid MSE: {epochValidMse:.6f} ({epochValidMse:.4e})")

        os.makedirs(os.path.dirname(self.outputBanditPath), exist_ok=True)
        self.rlInterface.bandit.save(self.outputBanditPath)
        print(f"\n{'='*60}")
        print(f"[RL_pretrain INFO] Pretrained Bandit saved: {self.outputBanditPath}")
        print(f"[RL_pretrain INFO] Total updates: {self.totalUpdateCount}")
        print(f"[RL_pretrain INFO] History length: {len(self.rlInterface.bandit.history)}")
        print(f"[RL_pretrain INFO] Buffer sizes: {[len(b) for b in self.rlInterface.bandit.replayBuffers]}")
        print(f"[RL_pretrain INFO] Train MSE History: {[f'{v:.6f} ({v:.4e})' for v in self.epochTrainMseHistory]}")
        print(f"[RL_pretrain INFO] Valid MSE History: {[f'{v:.6f} ({v:.4e})' for v in self.epochValidMseHistory]}")
        if self.eegnetTrainAccHistory:
            print(f"[RL_pretrain INFO] EEGNet Train Acc: {[f'{v:.4f}' for v in self.eegnetTrainAccHistory]}")
            print(f"[RL_pretrain INFO] EEGNet Valid Acc: {[f'{v:.4f}' for v in self.eegnetValidAccHistory]}")
            print(f"[RL_pretrain INFO] EEGNet Train Loss: {[f'{v:.4f}' for v in self.eegnetTrainLossHistory]}")
            print(f"[RL_pretrain INFO] EEGNet Valid Loss: {[f'{v:.4f}' for v in self.eegnetValidLossHistory]}")
        print(f"{'='*60}")

        # 輸出 MSE Loss 圖表
        self._savePlot(OUTPUT_LOG_DIR)

        return self.outputBanditPath


if __name__ == "__main__":
    setSeed()
    parser = argparse.ArgumentParser(description="Offline RL Pretrain")
    parser.add_argument("--pretrain-epochs", type=int, default=DEFAULT_PRETRAIN_EPOCHS)
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--min-retain-ratio", type=float, default=0.5, help="(deprecated, no effect)")
    parser.add_argument("--converted-pt-dir", type=str, default=CONVERTED_PT_DIR)
    parser.add_argument("--output", type=str, default=OUTPUT_BANDIT_PATH)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--pretrained-bandit", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUTPUT_LOG_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_LOG_DIR, f"RL_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    original_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(original_stdout, f)
        try:
            pretrainer = RLPretrainer(
                pretrainEpochs=args.pretrain_epochs,
                scoreThreshold=args.score_threshold,
                convertedPtDir=args.converted_pt_dir,
                outputBanditPath=args.output,
                baseModelPath=args.base_model,
                pretrainedBanditPath=args.pretrained_bandit,
                dryRun=args.dry_run)
            pretrainer.run()
        finally:
            sys.stdout = original_stdout
            print(f"\n[RL_pretrain INFO] Log saved to {log_path}")