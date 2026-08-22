"""
測試模型頁面。

拆自原 main.py 的 run_test_model。
"""
import pygame
import config
import global_state


def run_test_model(osu_interface, model_interface):
    """使用當前 activate model 持續預測 EEG 並在畫面上顯示結果"""
    ori_SAVE_CSV = config.SAVE_CSV
    config.SAVE_CSV = False

    print(f"[DEBUG] run_test_model 開始，使用模型: {model_interface.active_model_name}")
    print(f"[DEBUG] checkpoint: {global_state.NOW_FT_CHECKPOINT}")
    print(f"[INFO] 按 ESC 結束測試")

    model_interface.start_read_eeg()
    model_interface.start_pred_eeg()

    global_state.pred_buffer.clear()
    global_state.prob_buffer.clear()

    predictionCount = 0
    latestPrediction = -1
    latestProbability = None
    testing = True

    while testing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                testing = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print(f"[DEBUG] ESC 按下，結束測試")
                    testing = False
                    break

        if not testing:
            break

        # 取出最新預測結果
        while len(global_state.pred_buffer) > 0:
            try:
                predResult = global_state.pred_buffer.popleft()
                probResult = global_state.prob_buffer.popleft()
                predictionCount += 1

                latestPrediction = predResult
                latestProbability = probResult

                probDisplayStr = " | ".join(
                    [f"class{i}: {probResult[i]:.4f}" for i in range(len(probResult))])
                print(f"[PRED #{predictionCount:04d}] prediction: {predResult}  |  prob: [{probDisplayStr}]")
            except IndexError:
                break

        osu_interface.view.draw_test_model(
            model_name=model_interface.active_model_name,
            prediction=latestPrediction,
            probability=latestProbability,
            prediction_count=predictionCount,
        )

        osu_interface.clock.tick(30)

    model_interface.end_pred_eeg()
    model_interface.end_read_eeg()

    global_state.pred_buffer.clear()
    global_state.prob_buffer.clear()

    print(f"[DEBUG] run_test_model 結束，共預測 {predictionCount} 次")
    config.SAVE_CSV = ori_SAVE_CSV


# ─── Debug ───
if __name__ == "__main__":
    print("=== core.test_model_runner standalone test ===")
    print("run_test_model requires pygame and model_interface.")
    print("✅ test_model_runner import OK")
