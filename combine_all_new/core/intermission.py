"""
結算畫面（Intermission）。

拆自原 main.py 的 run_intermission。
"""
import sys
import time
import pygame

from rl.bandit import ContextIDX
import config
import global_state


def run_intermission(osu_interface, model_interface, RL_interface, map_notes):
    """結算畫面：顯示準確度，提供 Train / Calib / Skip 操作"""
    waiting = True
    next_action = "skip"

    correct, count, each_count, each_count_acc = osu_interface.compute_notes(map_notes)

    while waiting:
        osu_interface.view.draw_intermission(
            correct=correct,
            count=count,
            each_count=each_count,
            each_count_acc=each_count_acc,
            model_name=model_interface.active_model_name,
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                osu_interface.view.close()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    osu_interface.view.draw_loading("Training...")
                    model_interface.train_or_calib(mode=0, RL_open=False, RL_interface=RL_interface)

                    time.sleep(3)
                    _run_model_select(osu_interface, model_interface)

                    next_action = "train"
                    waiting = False

                elif event.key == pygame.K_c:
                    osu_interface.view.draw_loading("Calibing...")
                    session_records = model_interface.train_or_calib(
                        mode=1, RL_open=True, RL_interface=RL_interface)

                    # 同步 delta 值
                    RL_interface.delta_acc = model_interface.delta_acc
                    RL_interface.deltaValLoss = model_interface.deltaValLoss

                    for record in session_records:
                        reward = RL_interface.bandit.compute_reward(
                            deltaLoss=RL_interface.deltaValLoss,
                            meanEntropy=record['context'][ContextIDX.MEAN_ENTROPY],
                            alphaNorm=record['context'][ContextIDX.ALPHA_POWER_NORM]
                        )
                        RL_interface.bandit.update(record['context'], record['action'], reward)

                    print(f"[Intermission] calib done, deltaValLoss={RL_interface.deltaValLoss:.4f}")

                    time.sleep(3)
                    _run_model_select(osu_interface, model_interface)
                    next_action = "calib"
                    waiting = False

                elif event.key == pygame.K_s:
                    waiting = False

    return next_action


def _run_model_select(osu_interface, model_interface):
    """模型選擇畫面（內部使用）"""
    selecting = True
    model_list = model_interface.get_model_list()
    idx = len(model_list) - 1

    while selecting:
        model_rects = osu_interface.view.draw_model_select(model_list, idx)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                osu_interface.view.close()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    idx = max(0, idx - 1)
                if event.key == pygame.K_DOWN:
                    idx = min(len(model_list) - 1, idx + 1)
                if event.key == pygame.K_RETURN:
                    model_interface.set_active_model(idx)
                    return

            if event.type == pygame.MOUSEBUTTONDOWN:
                for i, rect in enumerate(model_rects):
                    if rect.collidepoint(event.pos):
                        idx = i


# ─── Debug ───
if __name__ == "__main__":
    print("=== core.intermission standalone test ===")
    print("run_intermission requires pygame display and model_interface.")
    print("✅ intermission import OK")
