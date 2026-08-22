"""
RL 閉環背景線程（緩衝型滾動進化版）。

拆自原 main.py 的 rl_loop_worker。
改為兩階段架構：
  Phase 1 — 即時資料蒐集 & 品質評估（不訓練 DL）
  Phase 2 — 條件觸發更新（UpdateScheduler 達標時才訓練）

設計原則：
  - 拒絕高頻更新：歌曲進行中 DL 維持 DL_n 不動
  - 積分制推進：累積滿 7.0 分才觸發 DL_n → DL_{n+1}
  - Experience Replay：混合 Stable_Buffer(30%) + Dynamic_Buffer(70%) 防止過擬合
  - 即時模型同步：更新後使用 Lock 安全切換 predictor 模型
  - 即時資料來源：從 global_state.eeg_buffer + prob_buffer 取 real-time data
"""
import os
import time
import numpy as np
from rl.bandit import ContextIDX, ExperienceReplayMemory
import config
import global_state
import eeg.preprocess as pps


# RL Loop 每次擷取 real-time data 的間隔（秒）
RL_LOOP_INTERVAL = 3.0


class RLLoopWorker:
    """背景 RL 閉環：在遊玩中蒐集即時資料，條件觸發 DL 更新"""

    def __init__(self, model_interface, RL_interface, stableDataDir: str):
        """
        Args:
            model_interface:  EEG 模型管理介面
            RL_interface:     RL 介面（Bandit + Normalizer + UpdateScheduler）
            stableDataDir:    pretrain .pt 資料目錄（Stable_Buffer 來源）
        """
        self.model = model_interface
        self.rl = RL_interface
        self.running = False
        self.total_update_times = 0

        # 雙重緩衝 Experience Replay
        self.replayMemory = ExperienceReplayMemory(
            stableDataDir=stableDataDir,
            maxDynamicSize=2000,
        )

        # hq_buffer.pt 路徑（由 app_controller 在登入後設定）
        self.hqBufferPath = ""

        print(f"[RLLoopWorker] 初始化完成 (即時資料版)")
        print(f"  stableDataDir={stableDataDir}")
        print(f"  RL_LOOP_INTERVAL={RL_LOOP_INTERVAL}s")
        print()

    def load_hq_buffer(self, hqBufferPath: str):
        """
        載入先前儲存的高品質 Dynamic_Buffer 資料。
        應在 app_controller.before_start() 中呼叫。
        """
        self.hqBufferPath = hqBufferPath
        self.replayMemory.load_buffer(hqBufferPath)

    def save_hq_buffer(self):
        """
        將 Dynamic_Buffer 中的高品質資料持久化。
        應在歌曲結束後呼叫。
        """
        if self.hqBufferPath:
            self.replayMemory.save_buffer(self.hqBufferPath, topK=100)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def _snapshot_realtime_data(self):
        """
        從 global_state 擷取即時 EEG + prediction 機率的快照。

        Returns:
            hit_data: {'eeg': (n_channels, n_samples), 'probs': (n_classes,)}
                      若資料不足則回傳 None
            segment:  原始 EEG segment (n_samples, n_channels) 供後續訓練用
            label:    當前預測的 class index
        """
        # 確認 buffer 有足夠資料
        with global_state.buffer_lock:
            if global_state.eeg_buffer.shape[1] < config.BUFFER_SIZE:
                return None, None, None
            eegSnapshot = global_state.eeg_buffer[:, -config.BUFFER_SIZE:].copy()

        # 確認有預測結果
        if len(global_state.prob_buffer) == 0:
            return None, None, None

        # 取最新的 prediction probability
        latestProbs = np.array(global_state.prob_buffer[-1])
        latestPred = int(np.argmax(latestProbs))

        # 前處理（跟 predict_loop 一致）
        eegProcessed = pps.each_channel_demean(eegSnapshot, axis=1)
        eegProcessed = pps.bandpass(eegProcessed, axis=-1)

        # hit_data 格式：eeg = (n_channels, n_samples), probs = (n_classes,)
        hitData = {
            'eeg': eegProcessed,
            'probs': latestProbs,
        }

        # segment 格式：(n_samples, n_channels)，跟 data_loader 的 trial 格式一致
        segment = eegProcessed.T  # (BUFFER_SIZE, CHOOSE_CHANNELS)

        print(f"[RL Loop] snapshot: eeg={eegSnapshot.shape}, "
              f"pred={latestPred}, probs={[f'{p:.3f}' for p in latestProbs]}")

        return hitData, segment, latestPred

    def worker(self):
        """
        RL 閉環主迴圈（在背景線程中執行）。

        兩階段架構：
          Phase 1 — 每次迴圈都跑：擷取即時 EEG → Bandit 決策 → UpdateScheduler 累積
          Phase 2 — shouldUpdate 時才跑：replay 混合 → DL fine-tune → 模型切換
        """
        updateTimes = 0
        pendingRecords = []  # 跨迴圈累積的待處理 records
        hitCount = 0

        print(f"[RL Loop] worker 啟動（即時資料版）")

        while self.running:
            self.rl.bandit.load(global_state.RL_path)

            # ═══════════════════════════════════════════════
            # Phase 1：即時資料蒐集 & 品質評估（不訓練 DL）
            # ═══════════════════════════════════════════════
            hitData, segment, label = self._snapshot_realtime_data()

            if hitData is None:
                # 資料不足（buffer 尚未填滿或無預測結果），稍後重試
                time.sleep(1)
                continue

            # 提取 context → Bandit 選 action
            rawContext = self.rl.extract_per_hit_context(hitData)
            normContext = self.rl.normalizer.update_and_normalize(rawContext)
            action = self.rl.bandit.select_action(normContext)

            # UpdateScheduler 累積
            shouldUpdate = self.rl.updateScheduler.record(action)

            record = {
                'context': normContext,
                'action': action,
                'segment': segment,   # (n_samples, n_channels)
                'label': label,
            }
            pendingRecords.append(record)
            hitCount += 1

            print(f"[RL Loop] Phase 1 Hit#{hitCount}: action={action.name}, "
                  f"shouldUpdate={shouldUpdate}, pending={len(pendingRecords)}")

            if not shouldUpdate:
                # 還沒到觸發門檻，繼續蒐集
                time.sleep(RL_LOOP_INTERVAL)
                continue

            # ═══════════════════════════════════════════════
            # Phase 2：條件觸發更新（shouldUpdate == True）
            # ═══════════════════════════════════════════════
            print(f"[RL Loop] ★ UpdateScheduler 觸發！開始 DL 更新 (update #{updateTimes})")

            # 2a. 從 ExperienceReplayMemory 抽樣混合資料
            newDataCount = len(pendingRecords)
            replayCount = max(10, int(newDataCount * 0.3 / 0.7))

            replaySegments, replayLabels = self.replayMemory.sample(
                totalCount=replayCount, stableRatio=0.3
            )

            # 2b. 執行 DL fine-tune（混合新資料 + replay 資料）
            trainResult = self.model.train_rl_loop_update(
                session_records=pendingRecords,
                replay_segments=replaySegments,
                replay_labels=replayLabels,
                update_times=updateTimes,
            )
            valAcc, valLoss, newCheckpointPath, newSegs, newLabs, newActs = trainResult

            if newCheckpointPath is not None:
                # 2c. 即時切換 predictor 模型（使用 Lock 安全同步）
                print(f"[RL Loop] 切換 predictor 模型: {newCheckpointPath}")
                self.model.eeg_predictor.reload_model(newCheckpointPath)

                # 更新 global_state 的 NOW_FT_CHECKPOINT
                modelBaseName = os.path.splitext(os.path.basename(newCheckpointPath))[0]
                global_state.NOW_FT_CHECKPOINT = os.path.join(
                    os.path.dirname(newCheckpointPath),
                    modelBaseName
                )
                self.model.active_model_name = modelBaseName

                # 同步 delta 值
                self.rl.RL_loop_delta_acc = self.model.RL_loop_delta_acc
                self.rl.rlLoopDeltaValLoss = self.model.rlLoopDeltaValLoss

                # 2d. 計算 reward 並更新 Bandit
                for record in pendingRecords:
                    reward = self.rl.bandit.compute_reward(
                        deltaLoss=self.rl.rlLoopDeltaValLoss,
                        meanEntropy=record['context'][ContextIDX.MEAN_ENTROPY],
                        alphaNorm=record['context'][ContextIDX.ALPHA_POWER_NORM],
                    )
                    self.rl.bandit.update(record['context'], record['action'], reward)

                self.rl.bandit.save(global_state.RL_path)
                updateTimes += 1

                print(f"[RL Loop] DL 更新完成: update #{updateTimes}, "
                      f"val_acc={valAcc:.4f}, val_loss={valLoss:.4f}, "
                      f"highest_acc={self.model.RL_loop_highest_valid_acc:.4f}")
            else:
                print(f"[RL Loop] DL 更新跳過（資料不足）")

            # 2e. 將新高品質資料加入 Dynamic_Buffer
            if newSegs and newLabs and newActs:
                self.replayMemory.add_dynamic_batch(newSegs, newLabs, newActs)
                print(f"[RL Loop] Dynamic_Buffer 更新: +{len(newSegs)} 筆, "
                      f"total={self.replayMemory.dynamicSize}")

            # 2f. 重置 UpdateScheduler + 清空 pending
            self.rl.updateScheduler.reset()
            pendingRecords = []

            time.sleep(RL_LOOP_INTERVAL)

        print(f"[RL Loop] worker 結束, updateTimes={updateTimes}")
        self.total_update_times += updateTimes


# ─── Debug ───
if __name__ == "__main__":
    print("=== core.rl_loop standalone test ===")
    print("RLLoopWorker requires model_interface, RL_interface, and stableDataDir.")
    print("✅ rl_loop import OK")
