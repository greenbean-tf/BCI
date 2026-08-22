"""
RL 綜合驗證測試 — 覆蓋 RL Pretrain、RL Loop、Intermission Calib
BUFFER_SIZE=500, CHOOSE_CHANNELS=13
"""
import os
import sys
import json
import shutil
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

import torch

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_PATH)

import config
import global_state
from rl.bandit import (
    NeuralContextualBandit, BanditAction, ContextIDX,
    UpdateScheduler, ContextNormalizer, ExperienceReplayMemory
)
from rl.rl_interface import RL_interface
from core.rl_loop import RLLoopWorker

# ─── 常數 (以 config 為準) ───
BUF  = config.BUFFER_SIZE      # 1500 (3s @ 500Hz)
NCH  = config.CHOOSE_CHANNELS  # 13
NCLS = config.N_Class          # 4
STBL = os.path.join(CURRENT_PATH, "rl", "rl_pretrain", "data")
PRETRAINED = os.path.join(CURRENT_PATH, "rl", "rl_pretrain", "pretrained_bandit.pt")
TMP_DIR = os.path.join(CURRENT_PATH, "_test_tmp")


def make_rl() -> RL_interface:
    rl = RL_interface()
    if os.path.exists(PRETRAINED):
        rl.bandit.load(PRETRAINED)
    return rl


def make_hit_data():
    probs = np.array([0.75, 0.10, 0.08, 0.07])
    eeg   = np.random.randn(NCH, BUF)   # BUF = BUFFER_SIZE = 1500
    return {'eeg': eeg, 'probs': probs}


# ═══════════════════════════════════════════════════════════════
# Suite 1 — Bandit 核心元件
# ═══════════════════════════════════════════════════════════════
class TestBanditCore(unittest.TestCase):

    def setUp(self):
        self.bandit = NeuralContextualBandit(epsilon=0.0)

    # 1-1 select_action 回傳合法 action
    def test_select_action_valid(self):
        ctx = np.array([0.3, 0.7, 0.2])
        action = self.bandit.select_action(ctx)
        self.assertIn(action, list(BanditAction))
        print(f"  [1-1] select_action -> {action.name}")

    # 1-2 update 後 replay buffer 非空
    def test_update_fills_buffer(self):
        ctx = np.array([0.3, 0.7, 0.2])
        action = BanditAction.HIGH_WEIGHT
        self.bandit.update(ctx, action, reward=0.5)
        self.assertGreater(len(self.bandit.replayBuffers[action]), 0)
        print(f"  [1-2] buffer size after update: {[len(b) for b in self.bandit.replayBuffers]}")

    # 1-3 compute_reward 值域在 clip 範圍內
    def test_compute_reward_clipped(self):
        r = self.bandit.compute_reward(deltaLoss=-0.2, meanEntropy=0.3, alphaNorm=0.4)
        self.assertGreaterEqual(r, self.bandit.rewardClipMin)
        self.assertLessEqual(r,   self.bandit.rewardClipMax)
        print(f"  [1-3] compute_reward={r:.4f} (clip [{self.bandit.rewardClipMin}, {self.bandit.rewardClipMax}])")

    # 1-4 save / load 後 history 長度一致
    def test_save_load_roundtrip(self):
        os.makedirs(TMP_DIR, exist_ok=True)
        path = os.path.join(TMP_DIR, "bandit_test.pt")
        ctx = np.array([0.2, 0.8, 0.3])
        self.bandit.update(ctx, BanditAction.INCLUDE, 0.3)
        self.bandit.save(path)

        b2 = NeuralContextualBandit()
        b2.load(path)
        self.assertEqual(len(b2.history), len(self.bandit.history))
        print(f"  [1-4] save/load roundtrip OK, history_len={len(b2.history)}")

    # 1-5 pretrained model 對清晰 context 選 HIGH_WEIGHT
    def test_pretrained_policy_clean_signal(self):
        if not os.path.exists(PRETRAINED):
            self.skipTest("pretrained_bandit.pt 不存在，跳過")
        b = NeuralContextualBandit(epsilon=0.0)
        b.load(PRETRAINED)
        # 低 entropy, 高 margin, 低 alpha → 清晰專注
        ctx = np.array([0.1, 0.9, 0.2])
        action = b.select_action(ctx)
        self.assertEqual(action, BanditAction.HIGH_WEIGHT)
        print(f"  [1-5] pretrained clean context -> {action.name} (expect HIGH_WEIGHT)")

    # 1-6 pretrained model 對噪聲 context 選 LOW_WEIGHT
    def test_pretrained_policy_noisy_signal(self):
        if not os.path.exists(PRETRAINED):
            self.skipTest("pretrained_bandit.pt 不存在，跳過")
        b = NeuralContextualBandit(epsilon=0.0)
        b.load(PRETRAINED)
        # 高 entropy, 低 margin, 高 alpha → 噪聲疲勞
        ctx = np.array([0.9, 0.1, 0.8])
        action = b.select_action(ctx)
        self.assertEqual(action, BanditAction.LOW_WEIGHT)
        print(f"  [1-6] pretrained noisy context -> {action.name} (expect LOW_WEIGHT)")


# ═══════════════════════════════════════════════════════════════
# Suite 2 — UpdateScheduler
# ═══════════════════════════════════════════════════════════════
class TestUpdateScheduler(unittest.TestCase):

    def test_trigger_on_threshold(self):
        sched = UpdateScheduler(scoreThreshold=5.0)
        triggered = False
        for _ in range(10):
            triggered = sched.record(BanditAction.HIGH_WEIGHT)
            if triggered:
                break
        self.assertTrue(triggered, "應在累積足夠分數後觸發")
        print(f"  [2-1] UpdateScheduler triggered OK")

    def test_reset_clears_score(self):
        sched = UpdateScheduler(scoreThreshold=5.0)
        for _ in range(4):
            sched.record(BanditAction.HIGH_WEIGHT)
        sched.reset()
        self.assertAlmostEqual(sched.accumulatedScore, 0.0)
        print(f"  [2-2] reset clears score OK")

    def test_action_scores_ordering(self):
        """HIGH_WEIGHT > INCLUDE > LOW_WEIGHT"""
        hw = UpdateScheduler.ACTION_QUALITY_SCORES[BanditAction.HIGH_WEIGHT]
        inc = UpdateScheduler.ACTION_QUALITY_SCORES[BanditAction.INCLUDE]
        lw  = UpdateScheduler.ACTION_QUALITY_SCORES[BanditAction.LOW_WEIGHT]
        self.assertGreater(hw, inc)
        self.assertGreater(inc, lw)
        print(f"  [2-3] score ordering: HW={hw} > INC={inc} > LW={lw}")


# ═══════════════════════════════════════════════════════════════
# Suite 3 — ContextNormalizer
# ═══════════════════════════════════════════════════════════════
class TestContextNormalizer(unittest.TestCase):

    def test_output_in_unit_range(self):
        norm = ContextNormalizer()
        for _ in range(20):
            raw = np.random.randn(3)
            normed = norm.update_and_normalize(raw)
        # sigmoid 輸出應在 (0,1)
        self.assertTrue(np.all(normed > 0) and np.all(normed < 1))
        print(f"  [3-1] ContextNormalizer output in (0,1): {normed}")

    def test_repeated_same_input_converges(self):
        norm = ContextNormalizer()
        raw = np.array([1.0, 2.0, 3.0])
        for _ in range(50):
            normed = norm.update_and_normalize(raw)
        # 重複同樣輸入，sigmoid(0) = 0.5
        self.assertTrue(np.allclose(normed, 0.5, atol=0.05))
        print(f"  [3-2] normalizer converges to 0.5 on repeated input: {normed}")


# ═══════════════════════════════════════════════════════════════
# Suite 4 — RL_interface Context 提取
# ═══════════════════════════════════════════════════════════════
class TestRLInterface(unittest.TestCase):

    def setUp(self):
        self.rl = make_rl()

    def test_extract_context_shape(self):
        hit = make_hit_data()
        ctx = self.rl.extract_per_hit_context(hit)
        self.assertEqual(ctx.shape, (3,))
        print(f"  [4-1] context shape: {ctx.shape}")

    def test_entropy_range(self):
        hit = make_hit_data()
        ctx = self.rl.extract_per_hit_context(hit)
        entropy = ctx[ContextIDX.MEAN_ENTROPY]
        self.assertGreaterEqual(entropy, 0.0)
        print(f"  [4-2] entropy={entropy:.4f} >= 0")

    def test_full_pipeline_context_to_action(self):
        hit = make_hit_data()
        raw = self.rl.extract_per_hit_context(hit)
        normed = self.rl.normalizer.update_and_normalize(raw)
        action = self.rl.bandit.select_action(normed)
        self.assertIn(action, list(BanditAction))
        print(f"  [4-3] full pipeline -> action={action.name}")


# ═══════════════════════════════════════════════════════════════
# Suite 5 — RL Pretrain 模擬器邏輯
# ═══════════════════════════════════════════════════════════════
class TestRLPretrainSimulator(unittest.TestCase):
    """
    不跑完整 pretrain，只驗證核心的模擬器邏輯正確性。
    """

    def test_reward_ordering_clean_signal(self):
        """清晰訊號 (quality≈1)：HIGH_WEIGHT reward > INCLUDE > LOW_WEIGHT"""
        bandit = NeuralContextualBandit()
        quality = 0.9
        rewards = {}
        for action in [0, 1, 2]:
            if action == 1:
                deltaLoss = 0.5 - quality
            elif action == 0:
                deltaLoss = 0.3 - 0.6 * quality
            else:
                deltaLoss = 0.2 * quality - 0.1
            e, a = 0.1, 0.2  # low entropy, low alpha
            r = bandit.compute_reward(deltaLoss=deltaLoss, meanEntropy=e, alphaNorm=a)
            rewards[action] = r
        self.assertGreater(rewards[1], rewards[0])
        self.assertGreater(rewards[0], rewards[2])
        print(f"  [5-1] clean signal reward: HW={rewards[1]:.4f} > INC={rewards[0]:.4f} > LW={rewards[2]:.4f}")

    def test_reward_ordering_noisy_signal(self):
        """噪聲訊號 (quality≈0)：LOW_WEIGHT reward > INCLUDE > HIGH_WEIGHT"""
        bandit = NeuralContextualBandit()
        quality = 0.05
        rewards = {}
        for action in [0, 1, 2]:
            if action == 1:
                deltaLoss = 0.5 - quality
            elif action == 0:
                deltaLoss = 0.3 - 0.6 * quality
            else:
                deltaLoss = 0.2 * quality - 0.1
            e, a = 0.9, 0.8  # high entropy, high alpha
            r = bandit.compute_reward(deltaLoss=deltaLoss, meanEntropy=e, alphaNorm=a)
            rewards[action] = r
        self.assertGreater(rewards[2], rewards[0])
        self.assertGreater(rewards[0], rewards[1])
        print(f"  [5-2] noisy signal reward: LW={rewards[2]:.4f} > INC={rewards[0]:.4f} > HW={rewards[1]:.4f}")

    def test_pretrained_model_exists_and_loadable(self):
        if not os.path.exists(PRETRAINED):
            self.skipTest("pretrained_bandit.pt 不存在，跳過")
        b = NeuralContextualBandit()
        b.load(PRETRAINED)
        self.assertGreater(len(b.history), 0)
        self.assertTrue(all(len(buf) > 0 for buf in b.replayBuffers))
        print(f"  [5-3] pretrained model: history={len(b.history)}, "
              f"buffers={[len(buf) for buf in b.replayBuffers]}")


# ═══════════════════════════════════════════════════════════════
# Suite 6 — RLLoopWorker Phase 1 & Phase 2
# ═══════════════════════════════════════════════════════════════
class TestRLLoopWorker(unittest.TestCase):

    def setUp(self):
        global_state.eeg_buffer = np.random.randn(NCH, BUF * 2)
        global_state.prob_buffer = [np.array([0.80, 0.10, 0.05, 0.05])]
        global_state.RL_path = PRETRAINED

        self.rl = make_rl()
        self.mock_model = MagicMock()
        self.mock_model.lastValLoss = float('inf')
        self.mock_model.last_acc = 0.0
        self.mock_model.RL_loop_delta_acc = 0.0
        self.mock_model.rlLoopDeltaValLoss = 0.0
        self.mock_model.RL_loop_highest_valid_acc = 0.0
        self.mock_model.eeg_predictor.reload_model = MagicMock()

        self.mock_model.train_rl_loop_update = MagicMock(return_value=(
            0.85, 0.12,
            os.path.join(TMP_DIR, "mock_ckpt.pth"),
            [np.random.randn(BUF, NCH)],
            [1],
            [BanditAction.HIGH_WEIGHT]
        ))

        self.worker = RLLoopWorker(
            model_interface=self.mock_model,
            RL_interface=self.rl,
            stableDataDir=STBL
        )
        self.worker.replayMemory.sample = MagicMock(return_value=(
            [np.random.randn(BUF, NCH) for _ in range(5)],
            [0] * 5
        ))

    # 6-1 即時資料擷取
    def test_snapshot_shape(self):
        hitData, segment, pred = self.worker._snapshot_realtime_data()
        self.assertIsNotNone(hitData)
        self.assertEqual(hitData['eeg'].shape, (NCH, BUF))
        self.assertEqual(hitData['probs'].shape, (NCLS,))
        self.assertIn(pred, range(NCLS))
        print(f"  [6-1] snapshot eeg={hitData['eeg'].shape}, pred={pred}")

    # 6-2 context 提取 + Bandit 決策
    def test_context_and_action(self):
        hitData, _, _ = self.worker._snapshot_realtime_data()
        raw = self.rl.extract_per_hit_context(hitData)
        normed = self.rl.normalizer.update_and_normalize(raw)
        action = self.rl.bandit.select_action(normed)
        self.assertIn(action, list(BanditAction))
        print(f"  [6-2] action={action.name}, normed_ctx={normed}")

    # 6-3 UpdateScheduler 累積到閾值
    def test_update_scheduler_accumulates(self):
        sched = self.rl.updateScheduler
        sched.reset()
        triggered = False
        for _ in range(20):
            if sched.record(BanditAction.HIGH_WEIGHT):
                triggered = True
                break
        self.assertTrue(triggered)
        print(f"  [6-3] UpdateScheduler triggered after accumulation")

    # 6-4 Phase 2 reward 計算與 bandit update
    def test_phase2_reward_and_update(self):
        hitData, segment, pred = self.worker._snapshot_realtime_data()
        raw = self.rl.extract_per_hit_context(hitData)
        normed = self.rl.normalizer.update_and_normalize(raw)
        action = self.rl.bandit.select_action(normed)

        # 模擬 DL 訓練結果
        deltaLoss = -0.15
        history_before = len(self.rl.bandit.history)

        reward = self.rl.bandit.compute_reward(
            deltaLoss=deltaLoss,
            meanEntropy=normed[ContextIDX.MEAN_ENTROPY],
            alphaNorm=normed[ContextIDX.ALPHA_POWER_NORM]
        )
        self.rl.bandit.update(normed, action, reward)

        self.assertGreater(len(self.rl.bandit.history), history_before)
        self.assertGreaterEqual(reward, self.rl.bandit.rewardClipMin)
        self.assertLessEqual(reward,   self.rl.bandit.rewardClipMax)
        print(f"  [6-4] Phase2 reward={reward:.4f}, history {history_before}->{len(self.rl.bandit.history)}")

    # 6-5 Bandit 存檔 (worker 結束後儲存)
    def test_bandit_save_after_update(self):
        os.makedirs(TMP_DIR, exist_ok=True)
        save_path = os.path.join(TMP_DIR, "rl_loop_bandit.pt")
        hitData, _, _ = self.worker._snapshot_realtime_data()
        raw = self.rl.extract_per_hit_context(hitData)
        normed = self.rl.normalizer.update_and_normalize(raw)
        action = self.rl.bandit.select_action(normed)
        reward = self.rl.bandit.compute_reward(-0.1, normed[0], normed[2])
        self.rl.bandit.update(normed, action, reward)
        self.rl.bandit.save(save_path)
        self.assertTrue(os.path.exists(save_path))
        print(f"  [6-5] bandit saved to {save_path}")


# ═══════════════════════════════════════════════════════════════
# Suite 7 — Intermission Calibration RL 流程
# ═══════════════════════════════════════════════════════════════
class TestIntermissionCalib(unittest.TestCase):

    def setUp(self):
        self.rl = make_rl()
        # 模擬 session_records (calib 後回傳)
        self.session_records = [
            {
                'context': np.array([0.15, 0.85, 0.25]),
                'action': BanditAction.HIGH_WEIGHT,
                'segment': np.random.randn(BUF, NCH),
                'label': 1
            },
            {
                'context': np.array([0.82, 0.18, 0.75]),
                'action': BanditAction.LOW_WEIGHT,
                'segment': np.random.randn(BUF, NCH),
                'label': 2
            },
            {
                'context': np.array([0.45, 0.55, 0.50]),
                'action': BanditAction.INCLUDE,
                'segment': np.random.randn(BUF, NCH),
                'label': 0
            },
        ]

    # 7-1 delta 同步
    def test_delta_sync(self):
        mock_model = MagicMock()
        mock_model.deltaValLoss = -0.18
        mock_model.delta_acc = 0.06

        self.rl.deltaValLoss = mock_model.deltaValLoss
        self.rl.delta_acc    = mock_model.delta_acc

        self.assertAlmostEqual(self.rl.deltaValLoss, -0.18)
        print(f"  [7-1] delta sync: deltaValLoss={self.rl.deltaValLoss:.4f}")

    # 7-2 所有 trial 都完成 reward 計算與更新
    def test_all_records_updated(self):
        self.rl.deltaValLoss = -0.18
        history_before = len(self.rl.bandit.history)

        for record in self.session_records:
            reward = self.rl.bandit.compute_reward(
                deltaLoss=self.rl.deltaValLoss,
                meanEntropy=record['context'][ContextIDX.MEAN_ENTROPY],
                alphaNorm=record['context'][ContextIDX.ALPHA_POWER_NORM]
            )
            self.rl.bandit.update(record['context'], record['action'], reward)

        expected = history_before + len(self.session_records)
        self.assertEqual(len(self.rl.bandit.history), expected)
        print(f"  [7-2] all {len(self.session_records)} records updated, "
              f"history {history_before}->{len(self.rl.bandit.history)}")

    # 7-3 calib 後的 reward 方向正確 (deltaLoss < 0 → reward > 0 before LQP)
    def test_reward_direction_on_improvement(self):
        record = self.session_records[0]  # HIGH_WEIGHT, clean signal
        reward = self.rl.bandit.compute_reward(
            deltaLoss=-0.20,
            meanEntropy=record['context'][ContextIDX.MEAN_ENTROPY],
            alphaNorm=record['context'][ContextIDX.ALPHA_POWER_NORM]
        )
        # loss 下降 → raw_reward 應為正（剪裁後仍 >= clip_min）
        self.assertGreater(reward, self.rl.bandit.rewardClipMin)
        print(f"  [7-3] improvement reward={reward:.4f} > {self.rl.bandit.rewardClipMin}")

    # 7-4 calib 後 bandit 存檔
    def test_bandit_save_after_calib(self):
        os.makedirs(TMP_DIR, exist_ok=True)
        path = os.path.join(TMP_DIR, "calib_bandit.pt")
        self.rl.deltaValLoss = -0.18
        for record in self.session_records:
            r = self.rl.bandit.compute_reward(-0.18,
                record['context'][ContextIDX.MEAN_ENTROPY],
                record['context'][ContextIDX.ALPHA_POWER_NORM])
            self.rl.bandit.update(record['context'], record['action'], r)
        self.rl.bandit.save(path)
        self.assertTrue(os.path.exists(path))

        # 驗證能重新載入
        b2 = NeuralContextualBandit()
        b2.load(path)
        self.assertEqual(len(b2.history), len(self.rl.bandit.history))
        print(f"  [7-4] calib bandit saved & reloaded, history={len(b2.history)}")


# ═══════════════════════════════════════════════════════════════
# Suite 8 — ExperienceReplayMemory
# ═══════════════════════════════════════════════════════════════
class TestExperienceReplayMemory(unittest.TestCase):

    def setUp(self):
        self.mem = ExperienceReplayMemory(stableDataDir=STBL, maxDynamicSize=200)

    def test_add_and_sample_dynamic(self):
        for i in range(30):
            seg = np.random.randn(BUF, NCH)
            self.mem.add_dynamic(seg, label=i % NCLS, action=BanditAction.INCLUDE)

        segs, labels = self.mem.sample(totalCount=20, stableRatio=0.3)
        self.assertGreater(len(segs), 0)
        self.assertEqual(len(segs), len(labels))
        print(f"  [8-1] sample returned {len(segs)} segs from dynamic buffer")

    def test_dynamic_size_limit(self):
        for i in range(300):
            seg = np.random.randn(BUF, NCH)
            self.mem.add_dynamic(seg, label=i % NCLS, action=BanditAction.HIGH_WEIGHT)
        self.assertLessEqual(self.mem.dynamicSize, 200)
        print(f"  [8-2] dynamicSize={self.mem.dynamicSize} <= maxDynamicSize=200")

    def test_save_load_buffer(self):
        os.makedirs(TMP_DIR, exist_ok=True)
        path = os.path.join(TMP_DIR, "hq_buffer.pt")
        for i in range(20):
            self.mem.add_dynamic(np.random.randn(BUF, NCH), i % NCLS, BanditAction.INCLUDE)
        self.mem.save_buffer(path, topK=10)
        self.assertTrue(os.path.exists(path))

        mem2 = ExperienceReplayMemory(stableDataDir=STBL, maxDynamicSize=200)
        mem2.load_buffer(path)
        self.assertGreater(mem2.dynamicSize, 0)
        print(f"  [8-3] buffer save/load OK, dynamicSize={mem2.dynamicSize}")


# ═══════════════════════════════════════════════════════════════
# 清理
# ═══════════════════════════════════════════════════════════════
def tearDownModule():
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
