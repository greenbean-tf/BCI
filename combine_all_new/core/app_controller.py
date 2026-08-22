"""
MainController — 應用程式主控制器。

拆自原 main.py，精簡為組裝各子系統 + 主選單事件迴圈。
"""
import sys
import os
import glob
import json
import time
from datetime import datetime

import pygame
import pylsl

from utils.log_class import Tee

import config
import global_state
from eeg.lsl_utils import setup_lsl_inlet
from eeg.tcp_utils import TCPServer
from eeg.model_interface import model_interface

from osu.osu_interface import osu_interface

from rl.rl_interface import RL_interface
from rl.bandit import NeuralContextualBandit

from core.account_manager import AccountManager
from core.level_runner import LevelRunner
from core.intermission import run_intermission
from core.test_model_runner import run_test_model
from core.rl_loop import RLLoopWorker


class MainController:
    def __init__(self):
        # ─── LSL ───
        self.inlet = self.initialize_lsl()

        # ─── TCP ───
        self.tcp_server = TCPServer(
            host=config.TCP_HOST,
            port=config.TCP_PORT,
            on_message=self.on_tcp_message)

        # ─── 遊戲介面 ───
        self.osu = osu_interface()

        # ─── EEG 子系統 ───
        self.model = model_interface(inlet=self.inlet, valid_ratio=0.25)

        # ─── RL ───
        self.RL_interface = RL_interface()

        # stableDataDir = pretrain 資料目錄
        stableDataDir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      "combine_all_new", "rl", "rl_pretrain", "data")
        # 如果路徑不存在，嘗試相對路徑
        if not os.path.exists(stableDataDir):
            stableDataDir = os.path.join("rl", "rl_pretrain", "data")

        self.rl_loop = RLLoopWorker(self.model, self.RL_interface, stableDataDir=stableDataDir)

        # ─── 帳號 ───
        self.account_mgr = AccountManager()

        # ─── 遊玩控制 ───
        self.level_runner = LevelRunner(self.osu, self.model, self.RL_interface)

    def initialize_lsl(self):
        """初始化 LSL（靜默失敗）"""
        inlet = None
        try:
            inlet = setup_lsl_inlet()
        except RuntimeError as e:
            print(f"{config.TAGS.ERROR.value} LSL init failed: {e}")
            print(f"{config.TAGS.INFO.value} 使用模擬 EEG 模式")
            config.is_simulated_eeg = True
        return inlet

    def on_tcp_message(self, msg, server):
        """TCP 訊息回呼"""
        print(f"[TCP] Received: {msg}")

    # ─── 主選單迴圈 ───
    def start(self):
        while True:
            action = self.osu.run_main_menu()

            if action == "song_select":
                result = self.osu.run_song_select()
                if result == "back":
                    continue

                # 遊玩
                self.rl_loop.start()
                state, map_notes_or_idx = self.level_runner.run_level(
                    result, self.tcp_server, self.rl_loop.worker)

                if state == "exit":
                    continue

                # 停止 RL loop
                self.level_runner.rl_loop_running = False
                self.rl_loop.stop()

                # 結算
                run_intermission(self.osu, self.model, self.RL_interface, map_notes_or_idx)

                # 儲存 Dynamic_Buffer 高品質資料
                self.rl_loop.save_hq_buffer()
                print(f"[MainController] 歌曲結束: hq_buffer 已儲存")

            elif action == "settings":
                self.osu.run_settings_menu()

            elif action == "model_select":
                self._run_model_select()

            elif action == "test_model":
                run_test_model(self.osu, self.model)

    def _run_model_select(self):
        """模型選擇畫面"""
        selecting = True
        model_list = self.model.get_model_list()
        idx = len(model_list) - 1

        while selecting:
            model_rects = self.osu.view.draw_model_select(model_list, idx)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.osu.view.close()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        idx = max(0, idx - 1)
                    if event.key == pygame.K_DOWN:
                        idx = min(len(model_list) - 1, idx + 1)
                    if event.key == pygame.K_RETURN:
                        self.model.set_active_model(idx)
                        return
                    if event.key == pygame.K_ESCAPE:
                        return

                if event.type == pygame.MOUSEBUTTONDOWN:
                    for i, rect in enumerate(model_rects):
                        if rect.collidepoint(event.pos):
                            self.model.set_active_model(i)
                            return

    # ─── 啟動流程 ───
    def before_start(self):
        """完整啟動流程：帳號登入 → 路徑設定 → log → start"""
        
        # 帳號登入
        self.account_mgr.run_account_login(
            self.osu, self.RL_interface, self.model)

        user = self.account_mgr.logged_in_user

        # 設定全域路徑
        global_state.REALTIME_BASE_FILE     = os.path.join(config.account_dir, user, config.data_dir)
        global_state.CSV_FILENAME           = os.path.join(global_state.REALTIME_BASE_FILE, "eeg_record.csv")
        global_state.RL_path                = os.path.join(config.account_dir, user, config.bandit_RL)
        global_state.json_path              = os.path.join(config.account_dir, user, config.ver_json)
        global_state.save_dir_model         = os.path.join(config.account_dir, user, config.model_dir)

        global_state.EEG_CHECKPOINT_BASE_DIR        = os.path.join(
            config.account_dir, user, config.model_dir, config.base_model_dir)
        global_state.EEG_CHECKPOINT_TMP_BASE_DIR    = os.path.join(
            config.account_dir, user, config.model_dir, config.base_model_dir, "Calibed_Model_v001")
        global_state.user_base_model                = os.path.join(
            config.account_dir, user, config.model_dir, config.base_model_dir, config.base_model_name)



        # log
        log_dir = os.path.join(config.account_dir, user, config.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        original_stdout = sys.stdout
        f = open(log_path, "w", encoding="utf-8")
        sys.stdout = Tee(original_stdout, f)

        print(f"[INFO] Logged in as: {user}")
        print(f"[INFO] Log file: {log_path}")

        # 更新 data loader
        self.model.eeg_dataloader.update_dir_path()

        # 設定初始模型
        model_list = self.model.get_model_list()
        if model_list:
            self.model.set_active_model(len(model_list) - 1)

        # 載入先前的高品質 Experience Replay 緩衝區
        hqBufferPath = os.path.join("account", user, "hq_buffer.pt")
        self.rl_loop.load_hq_buffer(hqBufferPath)
        print(f"[MainController] hq_buffer 路徑: {hqBufferPath}")

        # 啟動主迴圈
        self.start()


# ─── Debug ───
if __name__ == "__main__":
    print("=== core.app_controller standalone test ===")
    print("MainController requires LSL + Pygame to fully initialize.")
    print("✅ app_controller import OK")
