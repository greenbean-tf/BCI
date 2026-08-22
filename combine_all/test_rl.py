"""
RL 模組獨立測試腳本
測試 NeuralContextualBandit、UpdateScheduler、ContextNormalizer 的核心功能。
不需要 EEG 硬體、pygame、LSL。

執行方式：
    cd combine_all
    python test_rl.py
"""

import sys
import numpy as np
import os

# 確保 import 路徑正確
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RL.bandit import (
    NeuralContextualBandit, ContextualBandit,
    BanditAction, ContextIDX,
    UpdateScheduler, ContextNormalizer,
    BANDIT_DEVICE
)


def test_bandit_action():
    """測試 BanditAction enum 是否正確（移除了 ONLINE_UPDATE）"""
    print("=" * 60)
    print("[TEST 1] BanditAction enum")
    print("=" * 60)

    assert BanditAction.n_actions == 4, f"Expected 4 actions, got {BanditAction.n_actions}"
    assert BanditAction.INCLUDE == 0
    assert BanditAction.DISCARD == 1
    assert BanditAction.HIGH_WEIGHT == 2
    assert BanditAction.LOW_WEIGHT == 3

    # 確認沒有 ONLINE_UPDATE
    assert not hasattr(BanditAction, 'ONLINE_UPDATE'), "ONLINE_UPDATE should be removed!"

    print("  ✅ BanditAction: 4 actions, no ONLINE_UPDATE")
    print()


def test_context_idx():
    """測試 ContextIDX 是否使用 DELTA_LOSS"""
    print("=" * 60)
    print("[TEST 2] ContextIDX")
    print("=" * 60)

    assert ContextIDX.DELTA_LOSS == 2, f"Expected DELTA_LOSS=2, got {ContextIDX.DELTA_LOSS}"
    assert ContextIDX.context_dim == 4
    assert not hasattr(ContextIDX, 'DELTA_ACC'), "DELTA_ACC should be renamed to DELTA_LOSS!"

    print(f"  ✅ ContextIDX: dim={ContextIDX.context_dim}, DELTA_LOSS at idx 2")
    print()


def test_neural_bandit_select_action():
    """測試 NeuralUCB 的 action 選擇"""
    print("=" * 60)
    print("[TEST 3] NeuralContextualBandit.select_action()")
    print("=" * 60)

    bandit = NeuralContextualBandit(epsilon=0.5)

    # 正常 context
    context = np.array([0.3, 0.5, 0.1, 0.2])
    action = bandit.select_action(context)
    assert isinstance(action, BanditAction), f"Expected BanditAction, got {type(action)}"
    assert 0 <= int(action) < 4
    print(f"  ✅ Normal context → action={action.name}")

    # Rule-based override: alpha 極高 + entropy 極高 → LOW_WEIGHT
    context_override = np.array([0.9, 0.5, 0.1, 0.9])
    action_override = bandit.select_action(context_override)
    assert action_override == BanditAction.LOW_WEIGHT, f"Expected LOW_WEIGHT, got {action_override.name}"
    print(f"  ✅ Rule-based override → action={action_override.name}")
    print()


def test_neural_bandit_update():
    """測試 NeuralUCB 的 update（experience replay）"""
    print("=" * 60)
    print("[TEST 4] NeuralContextualBandit.update()")
    print("=" * 60)

    bandit = NeuralContextualBandit(epsilon=0.3, replayBufferSize=100)

    # 模擬多次 update
    for i in range(20):
        context = np.random.rand(4)
        action = bandit.select_action(context)
        reward = np.random.randn() * 0.5
        bandit.update(context, action, reward)

    # 檢查 history 和 replay buffer
    assert len(bandit.history) == 20, f"Expected 20 history, got {len(bandit.history)}"
    totalBufferSize = sum(len(b) for b in bandit.replayBuffers)
    assert totalBufferSize == 20, f"Expected 20 total in buffers, got {totalBufferSize}"

    print(f"  ✅ 20 rounds complete, history={len(bandit.history)}, buffer_total={totalBufferSize}")
    print()


def test_compute_reward():
    """測試 reward 計算（使用 deltaLoss 而非 deltaAcc）"""
    print("=" * 60)
    print("[TEST 5] compute_reward() with deltaLoss")
    print("=" * 60)

    bandit = NeuralContextualBandit(lambdaCost=0.1)

    # Case 1: loss 下降（deltaLoss < 0）→ reward 應為正
    reward1 = bandit.compute_reward(
        deltaLoss=-0.5,   # loss 下降了 0.5
        meanEntropy=0.3,
        alphaNorm=0.2,
    )
    print(f"  deltaLoss=-0.5 → reward={reward1:.4f}")
    assert reward1 > 0, f"Expected positive reward when loss decreases, got {reward1}"

    # Case 2: loss 上升（deltaLoss > 0）→ reward 應為負
    reward2 = bandit.compute_reward(
        deltaLoss=0.5,    # loss 上升了 0.5
        meanEntropy=0.3,
        alphaNorm=0.2,
    )
    print(f"  deltaLoss=+0.5 → reward={reward2:.4f}")
    assert reward2 < 0, f"Expected negative reward when loss increases, got {reward2}"

    # Case 3: 極端值 clip
    reward3 = bandit.compute_reward(
        deltaLoss=-10.0,  # 極端下降
        meanEntropy=0.1,
        alphaNorm=0.1,
    )
    assert reward3 <= 2.0, f"Expected clipped reward <= 2.0, got {reward3}"
    print(f"  deltaLoss=-10.0 → clipped reward={reward3:.4f}")

    print(f"  ✅ Reward signs correct, clipping works")
    print()


def test_save_load():
    """測試儲存與載入（含舊格式偵測）"""
    print("=" * 60)
    print("[TEST 6] save() / load()")
    print("=" * 60)

    testPath = "_test_bandit_save.pt"

    try:
        # 建立並訓練 bandit
        bandit1 = NeuralContextualBandit(epsilon=0.4)
        for i in range(10):
            ctx = np.random.rand(4)
            act = bandit1.select_action(ctx)
            bandit1.update(ctx, act, np.random.randn() * 0.3)

        # 儲存
        bandit1.save(testPath)
        assert os.path.exists(testPath), "Save file not created!"

        # 載入到新 bandit
        bandit2 = NeuralContextualBandit(epsilon=0.9)
        bandit2.load(testPath)

        # 驗證參數被載入
        assert bandit2.epsilon == 0.4, f"Expected epsilon=0.4, got {bandit2.epsilon}"
        assert len(bandit2.history) == 10, f"Expected 10 history, got {len(bandit2.history)}"

        print(f"  ✅ Save/Load round-trip OK (torch.save/.pt), epsilon={bandit2.epsilon}, history={len(bandit2.history)}")

        # 測試舊格式偵測（模擬舊 .npy）
        import torch as _torch
        _torch.save({"A": [], "b": []}, testPath)  # 無 version 欄位
        bandit3 = NeuralContextualBandit()
        bandit3.load(testPath)  # 應該自動跳過，不 crash
        assert len(bandit3.history) == 0, "Old format should be discarded!"
        print(f"  ✅ Old/incompatible format correctly detected and discarded")

    finally:
        if os.path.exists(testPath):
            os.remove(testPath)
    print()


def test_update_scheduler():
    """測試 UpdateScheduler 策略 B（加權品質分數）"""
    print("=" * 60)
    print("[TEST 7] UpdateScheduler (Strategy B)")
    print("=" * 60)

    scheduler = UpdateScheduler(scoreThreshold=5.0)  # 低閾值方便測試

    # HIGH_WEIGHT = 2.0, INCLUDE = 1.0
    assert scheduler.record(BanditAction.HIGH_WEIGHT) == False  # 2.0 < 5.0
    assert scheduler.record(BanditAction.INCLUDE) == False       # 3.0 < 5.0
    assert scheduler.record(BanditAction.DISCARD) == False       # 3.0 < 5.0 (DISCARD=0)
    assert scheduler.record(BanditAction.HIGH_WEIGHT) == True    # 5.0 >= 5.0 ← 觸發！

    print(f"  ✅ Triggered at accumulated score={scheduler.accumulatedScore:.1f}")

    # Reset
    scheduler.reset()
    assert scheduler.accumulatedScore == 0.0
    assert scheduler.totalProcessedCount == 0
    print(f"  ✅ Reset works")

    # LOW_WEIGHT = 0.3, 需要很多筆才觸發
    for _ in range(16):
        scheduler.record(BanditAction.LOW_WEIGHT)
    assert scheduler.accumulatedScore >= 4.7  # 16 * 0.3 ≈ 4.8（浮點精度容差）
    assert scheduler.record(BanditAction.LOW_WEIGHT) == True  # 17 * 0.3 = 5.1 >= 5.0
    print(f"  ✅ LOW_WEIGHT needs 17 hits to trigger (score={scheduler.accumulatedScore:.1f})")
    print()


def test_context_normalizer():
    """測試 ContextNormalizer"""
    print("=" * 60)
    print("[TEST 8] ContextNormalizer")
    print("=" * 60)

    norm = ContextNormalizer()

    # 第一筆：回傳全 0
    raw1 = np.array([1.0, 2.0, 0.5, 3.0])
    result1 = norm.update_and_normalize(raw1)
    assert np.allclose(result1, np.zeros(4)), f"First sample should be all zeros, got {result1}"
    print(f"  ✅ First sample → {result1}")

    # 第二筆：正規化
    raw2 = np.array([2.0, 4.0, 1.0, 6.0])
    result2 = norm.update_and_normalize(raw2)
    assert np.allclose(result2, np.ones(4)), f"Second sample should be all ones, got {result2}"
    print(f"  ✅ Second sample → {result2}")

    # 第三筆：介於中間
    raw3 = np.array([1.5, 3.0, 0.75, 4.5])
    result3 = norm.update_and_normalize(raw3)
    assert np.allclose(result3, np.full(4, 0.5)), f"Third sample should be all 0.5, got {result3}"
    print(f"  ✅ Third sample → {result3}")
    print()


def test_legacy_alias():
    """測試 ContextualBandit 作為 legacy alias"""
    print("=" * 60)
    print("[TEST 9] Legacy alias: ContextualBandit → NeuralContextualBandit")
    print("=" * 60)

    bandit = ContextualBandit()
    assert isinstance(bandit, NeuralContextualBandit)
    print(f"  ✅ ContextualBandit is NeuralContextualBandit")
    print()


def test_end_to_end_simulation():
    """模擬一個完整的 RL 閉環：多個 hit → 更新 → 儲存/載入"""
    print("=" * 60)
    print("[TEST 10] End-to-End Simulation (30 hits)")
    print("=" * 60)

    bandit = NeuralContextualBandit(epsilon=0.3, lambdaCost=0.1)
    normalizer = ContextNormalizer()
    scheduler = UpdateScheduler(scoreThreshold=10.0)

    updateCount = 0
    totalHits = 30

    for hitIdx in range(totalHits):
        # 模擬 raw context
        rawEntropy = np.random.uniform(0.1, 0.9)
        rawMargin = np.random.uniform(0.05, 0.7)
        rawDeltaLoss = np.random.uniform(-0.3, 0.3)
        rawAlpha = np.random.uniform(0.1, 0.8)
        rawContext = np.array([rawEntropy, rawMargin, rawDeltaLoss, rawAlpha])

        # 正規化
        normContext = normalizer.update_and_normalize(rawContext)

        # Bandit 選擇 action
        action = bandit.select_action(normContext)

        # UpdateScheduler 記錄
        shouldUpdate = scheduler.record(action)

        if shouldUpdate:
            # 模擬訓練完成後的 reward 計算
            simulatedDeltaLoss = np.random.uniform(-0.5, 0.2)
            reward = bandit.compute_reward(
                deltaLoss=simulatedDeltaLoss,
                meanEntropy=normContext[ContextIDX.MEAN_ENTROPY],
                alphaNorm=normContext[ContextIDX.ALPHA_POWER_NORM],
            )
            bandit.update(normContext, action, reward)
            scheduler.reset()
            updateCount += 1
            print(f"  [Hit #{hitIdx:02d}] UPDATE triggered! reward={reward:.4f}")

    print(f"\n  ✅ Simulation complete: {totalHits} hits, {updateCount} updates triggered")
    print(f"     history_len={len(bandit.history)}, "
          f"buffer_sizes={[len(b) for b in bandit.replayBuffers]}")
    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'#' * 60}")
    print(f"# RL Module Test Suite")
    print(f"# Device: {BANDIT_DEVICE}")
    print(f"{'#' * 60}\n")

    test_bandit_action()
    test_context_idx()
    test_neural_bandit_select_action()
    test_neural_bandit_update()
    test_compute_reward()
    test_save_load()
    test_update_scheduler()
    test_context_normalizer()
    test_legacy_alias()
    test_end_to_end_simulation()

    print("=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
