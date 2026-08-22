"""
RL 模組獨立測試腳本（combine_all_new 版）。
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl.bandit import (NeuralContextualBandit, ContextualBandit, BanditAction,
                        ContextIDX, UpdateScheduler, ContextNormalizer, BANDIT_DEVICE)

def test_all():
    # TEST 1: BanditAction
    assert BanditAction.n_actions == 4
    print("  ✅ TEST 1: BanditAction OK")

    # TEST 2: ContextIDX
    assert ContextIDX.ALPHA_POWER_NORM == 2 and ContextIDX.context_dim == 3
    print("  ✅ TEST 2: ContextIDX OK")

    # TEST 3: select_action
    b = NeuralContextualBandit(epsilon=0.5)
    a = b.select_action(np.array([0.3, 0.5, 0.2]))
    assert isinstance(a, BanditAction) and 0 <= int(a) < 4
    a2 = b.select_action(np.array([0.9, 0.5, 0.9]))
    assert a2 == BanditAction.LOW_WEIGHT
    print("  ✅ TEST 3: select_action OK")

    # TEST 4: update
    b2 = NeuralContextualBandit(epsilon=0.3, replayBufferSize=100)
    for _ in range(20):
        ctx = np.random.rand(3)
        act = b2.select_action(ctx)
        b2.update(ctx, act, np.random.randn() * 0.5)
    assert len(b2.history) == 20
    print("  ✅ TEST 4: update OK")

    # TEST 5: compute_reward
    b3 = NeuralContextualBandit(lambdaCost=0.1)
    assert b3.compute_reward(deltaLoss=-0.5, meanEntropy=0.3, alphaNorm=0.2) > 0
    assert b3.compute_reward(deltaLoss=0.5, meanEntropy=0.3, alphaNorm=0.2) < 0
    print("  ✅ TEST 5: compute_reward OK")

    # TEST 6: save/load
    tp = "_test_bandit_save.pt"
    try:
        b4 = NeuralContextualBandit(epsilon=0.4)
        for _ in range(10):
            ctx = np.random.rand(3)
            act = b4.select_action(ctx)
            b4.update(ctx, act, np.random.randn() * 0.3)
        b4.save(tp)
        b5 = NeuralContextualBandit(epsilon=0.9)
        b5.load(tp)
        assert b5.epsilon == 0.4 and len(b5.history) == 10
        print("  ✅ TEST 6: save/load OK")
    finally:
        if os.path.exists(tp): os.remove(tp)

    # TEST 7: UpdateScheduler
    s = UpdateScheduler(scoreThreshold=5.0)
    s.record(BanditAction.HIGH_WEIGHT)
    s.record(BanditAction.INCLUDE)
    s.record(BanditAction.DISCARD)
    assert s.record(BanditAction.HIGH_WEIGHT) == True
    s.reset()
    assert s.accumulatedScore == 0.0
    print("  ✅ TEST 7: UpdateScheduler OK")

    # TEST 8: ContextNormalizer (EMA + sigmoid)
    n = ContextNormalizer(alpha=0.05)
    r1 = n.update_and_normalize(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(r1, np.full(3, 0.5))  # 第一筆回傳中性值 0.5
    r2 = n.update_and_normalize(np.array([2.0, 4.0, 6.0]))
    assert np.all(r2 > 0.5)  # 偏高值 → sigmoid > 0.5
    r3 = n.update_and_normalize(np.array([0.5, 1.0, 1.5]))
    assert np.all(r3 < 0.5)  # 偏低值 → sigmoid < 0.5
    print("  ✅ TEST 8: ContextNormalizer OK")

    # TEST 9: Legacy alias
    assert isinstance(ContextualBandit(), NeuralContextualBandit)
    print("  ✅ TEST 9: Legacy alias OK")

    print("\n🎉 ALL TESTS PASSED!")

if __name__ == "__main__":
    print(f"\n{'#'*60}\n# RL Test Suite (combine_all_new)\n# Device: {BANDIT_DEVICE}\n{'#'*60}\n")
    test_all()
