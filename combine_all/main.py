import pygame
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512) #加緩衝區 要不然會延遲
import sys
import os
import glob
import zipfile
import json
import math
import socket
import csv
from datetime import datetime
from functools import wraps
from queue import Queue # 用於 multi thread, 內建 lock
import numpy as np
import time
from pathlib import Path
from collections import Counter
from collections import deque

import threading
import shutil                                                                       # use for copy model

import pylsl



import torch
import torch.nn.functional as F



# ourself folder
from osu.osu_parser import parse_osu_map, parse_metadata
from osu.game_view import GameView



from EEG_DL.EEG.model_interface import model_interface


import EEG_DL.Utils.config as config
import EEG_DL.Utils.global_value as global_value
import EEG_DL.Utils.LSL as LSL
from EEG_DL.Utils.TCP_use import TCPServer



from osu.osu_interface import osu_interface


from RL.bandit import ContextIDX
from RL.RL_interface import RL_interface

# 自動儲存 log
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


class MainController:
    def __init__(self):
        
        # account
        self.accounts_dir = config.account_dir
        os.makedirs(self.accounts_dir, exist_ok=True)
        
        self.account_list = self.load_accounts()
        self.current_account_index = 0
        self.logged_in_user = None
        


        # 初始化 LSL 入口
        self.inlet = self.initialize_lsl()

        # --- 多執行緒共用變數 ---
        # data record
        self.recording_active = True
        self.current_game_time_ms = 0.0
        global_value.current_marker = config.stay_marker
        global_value.current_predict = config.stay_marker
        global_value.current_prob = config.stay_prob.copy()



        # rl loop
        self.rl_loop_running = True

        


        self.tcp_server = TCPServer(host=config.TCP_HOST, port=config.TCP_PORT, on_message=self.my_message_handler)
        
        # osu_interface
        self.osu_interface = osu_interface()

        
        # 模擬要改 config.is_simulated_eeg
        # model_interface
        self.model_interface = model_interface(inlet=self.inlet)

        # predictor 更新用
        # -1 marker 表 stay
        self.current_predict = config.stay_marker
        self.current_prob = config.stay_prob.copy()

        # ous_interface 更新用
        self.training_msg = "Training..."



        self.RL_interface = RL_interface()
        self.total_update_times = 0

        



    def init_account(self, account_name):
        
        ac_list = [config.log_dir, config.model_dir, config.data_dir]

        user_path = os.path.join(self.accounts_dir, account_name)

        for i in ac_list:
            os.makedirs(os.path.join(user_path, i), exist_ok=True)

        os.makedirs(os.path.join(user_path, config.model_dir, config.base_model_dir), exist_ok=True)
        

        # 寫 RL 檔
        RL_path = os.path.join(user_path, config.bandit_RL)

        if os.path.exists(RL_path):
            self.RL_interface.bandit.load(RL_path)

        else:
            self.RL_interface.bandit.save(RL_path)

        print(f"[Bandit] Saved to {RL_path}")


        # 寫 json 檔
        json_path = os.path.join(user_path, config.ver_json)

        # 預設 version.json 內容
        default_json = {
            "train_model": 0,
            "calib_model": 0,
        }

        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.model_interface.version_setting(data["train_model"] + data["calib_model"])
            print(f" current version: {self.model_interface.version} ")
        
        else:
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(default_json, f, indent=4, ensure_ascii=False)
                print(f"成功為 {account_name} 創建設定檔")
                
            except Exception as e:
                print(f"創建失敗: {e}")
                return False
            

        return True

    def load_accounts(self):

        # 搜尋 account 資料夾下的所有帳號 (第一層資料夾)
        p = Path("account")
        accounts = [f.name for f in p.iterdir() if f.is_dir()]
        
        # 若無帳號則提供訪客選項
        if not accounts: accounts = ["Guest"]

        print(f"accounts_list: {accounts}")
        return accounts

    def login_as(self, account_name):
        self.logged_in_user = account_name
        self.init_account(account_name)
        print(f"Logged in as: {account_name}")
    
    def run_account_login(self):
        selecting = True
        while selecting:
            self.osu_interface.clock.tick(60)
            account_rects = self.osu_interface.view.draw_account_select(self.account_list, self.current_account_index)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    self.osu_interface.view.close()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        self.current_account_index = (self.current_account_index + 1) % len(self.account_list)
                    elif event.key == pygame.K_UP:
                        self.current_account_index = (self.current_account_index - 1) % len(self.account_list)
                    elif event.key == pygame.K_RETURN:
                        user = self.account_list[self.current_account_index]
                        self.login_as(user)
                        return "main_menu"
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for i, rect in enumerate(account_rects):
                        if rect.collidepoint(event.pos):
                            self.current_account_index = i
                            user = self.account_list[i]
                            self.login_as(user)
                            return "main_menu"



    def initialize_lsl(self):
        """初始化 LSL Inlet，但不代表開始存檔"""
        if config.is_simulated_eeg:
            print(f"{config.TAGS.INFO.value} EEG Mode: Simulated")
            return

        try:
            return LSL.setup_lsl_inlet()
            
        except Exception as e:
            print(f"{config.TAGS.ERROR.value} LSL Initialization failed: {e}")
    


    # setup the message event and TCP Server
    def my_message_handler(self, msg, server):
        parts = msg.split(config.TCP_STR_SPLIT)

        if parts[0] == "pred":
            self.current_predict = int(parts[1])

            for i in range(4):
                self.current_prob[i] = float(parts[i+2])



        elif parts[0] == "training":
            self.training_msg = msg
            print(str(msg))
            


    # data_logger_worker has been removed. Logging is now handled in EEG_Read.py by pull_chunk.

    def rl_loop_worker(self):
        """
        RL 閉環（升級版）

        使用 UpdateScheduler 決定何時觸發模型更新，
        Reward 使用 delta_loss（平滑化信號）。

        session_records: list
        record = {
            'context':np.array --> (ContextIDX.context_dim, ),
            'action':action,
        }
        action 不包含 ONLINE_UPDATE（已移除）

        """
        update_times = 0
        
        while self.rl_loop_running:

            self.RL_interface.bandit.load(global_value.RL_path)

            session_records = self.model_interface.train_or_calib(mode=1, 
                                                                  RL_open=True, 
                                                                  RL_interface=self.RL_interface,
                                                                  RL_LOOP_USE=True,
                                                                  update_times=update_times)
                        
            

            # 同步 delta 值（保留 delta_acc 供 log，使用 delta_loss 做 reward）
            self.RL_interface.RL_loop_delta_acc = self.model_interface.RL_loop_delta_acc
            self.RL_interface.rlLoopDeltaValLoss = self.model_interface.rlLoopDeltaValLoss

            for record in session_records:
                print(f"[RL Loop] session_records len: {len(session_records)}")
                reward = self.RL_interface.bandit.compute_reward(
                    deltaLoss=self.RL_interface.rlLoopDeltaValLoss,
                    meanEntropy=record['context'][ContextIDX.MEAN_ENTROPY],
                    alphaNorm=record['context'][ContextIDX.ALPHA_POWER_NORM]
                )
                self.RL_interface.bandit.update(record['context'], record['action'], reward)
            
            
            
            self.RL_interface.bandit.save(global_value.RL_path)
            update_times += 1

            print(f"[RL Loop] highest valid acc {self.model_interface.RL_loop_highest_valid_acc}")

            time.sleep(3)


        print(f"[RL Loop] finish, update_times: {update_times}")

        self.total_update_times += update_times




            



    def run_pause_menu(self):
        paused = True
        opt = 0

        # 暫停仍然會存，但 marker == 99, predict = -1
        global_value.current_marker = 99
        global_value.current_predict = config.stay_marker
        global_value.current_prob = config.stay_prob.copy()

        while paused:

            rects = self.osu_interface.view.draw_pause_menu(opt)

             # 監聽 pause 時
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.osu_interface.view.close(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        opt -= 1
                        opt = np.clip(opt, 0, 2)
                        
                    if event.key == pygame.K_DOWN:
                        opt += 1
                        opt = np.clip(opt, 0, 2)
                        
                    if event.key == pygame.K_RETURN:
                        if opt == 0: return "continue"
                        elif opt == 1: return "quit"
                        elif opt == 2: return "skip"
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if rects[0].collidepoint(event.pos): return "continue"
                    if rects[1].collidepoint(event.pos): return "quit"
                    if rects[2].collidepoint(event.pos): return "skip"


    def run_level(self, map_file):

        map_data = parse_osu_map(map_file)
        map_title = map_data['meta'].get('Title', 'Unknown')

        print(f"song name: {map_title}")

        if map_title == config.the_zero_song_name:
            return self.run_zero_level(map_file)
        

        state = "finish_song"

        # ... 遊玩面板設置 ...
        self.osu_interface.view.draw_loading("Loading Beatmap...")
        pygame.display.flip()
        pygame.event.pump()
    
        score = 0
        hits = 0
        accuracy = 0
        current_speed = self.osu_interface.settings['speed']
        # global_offset = self.osu_interface.settings['offset']


        # ... 讀取譜面 ...

        """
        meta: 譜面元數據
        notes === > list , in ---> directory
        lane (軌道)：透過公式 math.floor(x * key_count / 512) 計算。在 osu!mania 模式中，x 座標決定了音符落在第幾軌。

        time (時間)：該音符出現的精確時間（毫秒, ms）。

        type (類型)：

        'tap': 普通單點音符。

        'hold': 長按音符（Long Note），此時會多出一個 end 欄位。

        end (結束時間)：若是長按音符，紀錄其結束的毫秒數；若是單點則為 0。

        hit (判定標記)：預設為 False，通常用於遊戲執行時紀錄玩家是否已經按過這個音符。
        """

        

        current_note_index = 0  # 用來追蹤目前比對到地圖的哪一個音符，優化效能
        map_notes = map_data['notes']
        map_end_time = map_notes[-1]['time'] + 2000 if map_notes else 2000

        
        # pygame start
        self.osu_interface.play_music(map_data['meta'])

        
        # control judge
        open_judge = 1

        pred_key = config.stay_marker
        prob_key = config.stay_prob.copy()




        
        # ... lsl ...
        total_pause_time = 0.0
        start_lsl_time = pylsl.local_clock() # 抓取遊戲開始的精確 LSL 時間
        global_value.start_lsl_time = start_lsl_time
        self.model_interface.start_read_eeg()
        self.model_interface.start_pred_eeg()


        # 建立並啟動數據紀錄執行緒 (現在由 EEG_Read 處理)
        if config.SAVE_CSV:
            self.recording_active = True


        # RL 閉環
        if config.USE_RL_LOOP:
            self.rl_loop_running = True
            RL_thread = threading.Thread(target=self.rl_loop_worker, daemon=True)
            RL_thread.start()


        # song_start_time
        song_start_time = pylsl.local_clock()

        playing = True
        
        while playing:
            # 全部改 thread, 時間對齊問題好像改好了? 嗎?
            # 2026/04/08 學長說他也是一個一個抓的，應該可以吧

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # 取得當前 LSL 時間與地圖進度
            current_lsl_ts = pylsl.local_clock()
            game_time_ms = (current_lsl_ts - start_lsl_time - total_pause_time) * 1000.0
            
            self.current_game_time_ms = game_time_ms

            # 獲取 Marker
            current_marker, current_note_index = self.osu_interface.calculate_marker(map_notes, current_note_index, game_time_ms)



            # 預測 key
            # pred_key  ==> 0123
            # lane      ==> 0123

            

            # 有 marker 出現的情況下，speed 0.08 ---> 3.125s + 0.150s
            # check_hit 只會在 +-150ms 生效
            # 每間隔 config.PREDICTION_INTERVAL (02/23 時是 0.01s) 會 predict 一次
                ## 所以理論上 global_value.pred_buffer 長度是 100 時應過 1s，但慢超多，大概只有 10 就過了 3s

            if current_marker >= 0:
                
                n = map_notes[current_note_index]

        
                if abs(n['time'] - game_time_ms) < 150 and open_judge:
                    # method 2: 取出現最多次的 --> 7個 約 3秒
                    # pred_key = Counter(global_value.pred_buffer).most_common(1)[0][0]

                    # method 1: 取 最近一次 prediction 的結果
                    current_len = len(global_value.pred_buffer)
                    pred_key = global_value.pred_buffer[current_len-1]
                    prob_key = global_value.prob_buffer[current_len-1]
                    # print(f"pred_key: {pred_key}")
                    global_value.pred_buffer.clear()
                    global_value.prob_buffer.clear()
                    open_judge = 0
                    
                map_notes, score, hits = self.osu_interface.check_hit(map_notes,
                                                                            current_note_index,
                                                                            pred_key,
                                                                            game_time_ms,
                                                                            score,
                                                                            hits,
                                                                            )
                    

            
            else:
                # 若非 marker 出現期間，維持淨空
                open_judge = 1
                pred_key = config.stay_marker
                prob_key = config.stay_prob.copy()
                global_value.pred_buffer.clear()
                global_value.prob_buffer.clear()


            # pred 設計會影響 record， 但 data 只看 ground truth(current_marker) 所以沒差
            global_value.current_marker = current_marker
            global_value.current_predict = pred_key
            global_value.current_prob = prob_key

           
            

            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # 監聽遊玩時手動鍵盤輸入
            for event in pygame.event.get():
                if event.type == pygame.QUIT:

                    self.model_interface.end_pred_eeg()
                    self.model_interface.end_read_eeg()
                    self.osu_interface.view.close()

                    self.tcp_server.stop()
                    self.model_interface.tcp_client.stop()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    # 可以手動操作鍵盤
                    if event.key in self.osu_interface.key_map:
                        hand_key = self.osu_interface.key_map[event.key]
                        self.osu_interface.view.trigger_effect(hand_key)

                        map_notes, score, hits = self.osu_interface.check_hit(map_notes, 
                                                                              current_note_index,
                                                                              hand_key, 
                                                                              game_time_ms, 
                                                                              score, 
                                                                              hits)
              


                    elif event.key == pygame.K_ESCAPE:    #場中暫停
                        pygame.mixer.music.pause()
                        pause_start = pylsl.local_clock()
                        
                        do = self.run_pause_menu()

                        # 2026/03/05 這兩個好像有問題
                        if do == "quit":
                            state = "exit"
                            return state, current_note_index
                        
                        elif do == "skip":
                            # 測試用，視為 state = "finish_song"
                            playing = False
                            break
                        
                        pygame.mixer.music.unpause()
                        total_pause_time += (pylsl.local_clock() - pause_start)



            #準確度 (perfect之類的 啊我其他還沒寫)
            passed_notes = 0
            for n in map_notes:
                if n['time'] < game_time_ms - 150: passed_notes += 1
                else: break
            accuracy = (hits / passed_notes * 100) if passed_notes > 0 else 100.0


            # ... 處理繪圖與 FPS ...
            if game_time_ms > map_end_time:

                print(f"song_start_time: {song_start_time}")
                print(f"song_end_time: {pylsl.local_clock()}")

                # 停止 Data Thread
                self.recording_active = False 

                # 停止 RL Loop
                self.rl_loop_running = False

                # 停止歌曲遊玩
                playing = False

                

                

            self.osu_interface.view.draw_play_screen(map_notes, 
                                                     game_time_ms, 
                                                     score, 
                                                     self.model_interface.active_model_name, 
                                                     current_speed, 
                                                     accuracy
                                                     )
            
            self.osu_interface.clock.tick(config.SAMPLE_RATE)
        
        self.model_interface.end_pred_eeg()
        self.model_interface.end_read_eeg()
        return state, map_notes

    def run_zero_level(self, map_file):

        human_control_acc = 0.8

        print("now run_zero_level")
        map_data = parse_osu_map(map_file)
        
        current_note_index = 0
        map_notes = map_data['notes']
        map_end_time = map_notes[-1]['time'] + 2000 if map_notes else 2000



        state = "finish_song"

        # ... 遊玩面板設置 ...
        self.osu_interface.view.draw_loading("Loading Beatmap...")
        pygame.display.flip()
        pygame.event.pump()
    
        score = 0
        hits = 0
        accuracy = 0
        current_speed = self.osu_interface.settings['speed']
        
        # pygame start
        self.osu_interface.play_music(map_data['meta'])

        
        # control judge
        open_judge = 1
        pred_key = config.stay_marker
        prob_key = config.stay_prob.copy()


        
        # ... lsl ...
        total_pause_time = 0.0
        start_lsl_time = pylsl.local_clock()
        global_value.start_lsl_time = start_lsl_time
        self.model_interface.start_read_eeg()
        


        # 建立並啟動數據紀錄執行緒 (現在由 EEG_Read 處理)
        if config.SAVE_CSV:
            self.recording_active = True


        # song_start_time
        song_start_time = pylsl.local_clock()

        playing = True
        
        while playing:
            
            current_lsl_ts = pylsl.local_clock()
            game_time_ms = (current_lsl_ts - start_lsl_time - total_pause_time) * 1000.0
            
            self.current_game_time_ms = game_time_ms

            # 獲取 Marker
            current_marker, current_note_index = self.osu_interface.calculate_marker(map_notes, current_note_index, game_time_ms)

            # 預測 key
            # pred_key  ==> 0123
            # lane      ==> 0123

            if current_marker >= 0:
                
                n = map_notes[current_note_index]

                if abs(n['time'] - game_time_ms) < 150 and open_judge:
                    
                    correct_lane = n['lane']

                    # 以 human_control_acc 的機率打對，否則隨機打錯
                    if np.random.random() < human_control_acc:
                        pred_key = correct_lane
                        print(f"[ZeroLevel] Note#{current_note_index} HIT correct lane={correct_lane}")
                    else:
                        # 從其他 lane 中隨機選一個錯誤答案
                        wrong_lanes = [l for l in range(4) if l != correct_lane]
                        pred_key = np.random.choice(wrong_lanes)
                        print(f"[ZeroLevel] Note#{current_note_index} MISS correct={correct_lane}, gave={pred_key}")

                    prob_key = config.stay_prob.copy()
                    prob_key[pred_key] = 1.0

                    open_judge = 0
                    
                map_notes, score, hits = self.osu_interface.check_hit(map_notes,
                                                                            current_note_index,
                                                                            pred_key,
                                                                            game_time_ms,
                                                                            score,
                                                                            hits,
                                                                            )
                    

            
            else:
                # 若非 marker 出現期間，維持淨空
                open_judge = 1
                pred_key = config.stay_marker
                prob_key = config.stay_prob.copy()
                


            # record 用
            global_value.current_marker = current_marker
            global_value.current_predict = pred_key
            global_value.current_prob = prob_key



            # 鍵盤監聽
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.model_interface.end_read_eeg()
                    self.osu_interface.view.close()

                    self.tcp_server.stop()
                    self.model_interface.tcp_client.stop()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    # 可以手動操作鍵盤
                    if event.key in self.osu_interface.key_map:
                        hand_key = self.osu_interface.key_map[event.key]
                        self.osu_interface.view.trigger_effect(hand_key)

                        map_notes, score, hits = self.osu_interface.check_hit(map_notes, 
                                                                              current_note_index,
                                                                              hand_key, 
                                                                              game_time_ms, 
                                                                              score, 
                                                                              hits)
              
                    elif event.key == pygame.K_ESCAPE:    #場中暫停
                        pygame.mixer.music.pause()
                        pause_start = pylsl.local_clock()
                        
                        do = self.run_pause_menu()

                        # 2026/03/05 這兩個好像有問題
                        if do == "quit":
                            state = "exit"
                            return state, current_note_index
                        
                        elif do == "skip":
                            # 測試用，視為 state = "finish_song"
                            playing = False
                            break
                        
                        pygame.mixer.music.unpause()
                        total_pause_time += (pylsl.local_clock() - pause_start)


            #準確度
            passed_notes = 0
            for n in map_notes:
                if n['time'] < game_time_ms - 150: passed_notes += 1
                else: break
            accuracy = (hits / passed_notes * 100) if passed_notes > 0 else 100.0


            # ... 處理繪圖與 FPS ...
            if game_time_ms > map_end_time:
                print(f"song_start_time: {song_start_time}")
                print(f"song_end_time: {pylsl.local_clock()}")

                # 停止 Data Thread
                self.recording_active = False 

                # 停止歌曲遊玩
                playing = False

                

            self.osu_interface.view.draw_play_screen(map_notes, 
                                                     game_time_ms, 
                                                     score, 
                                                     "no model", 
                                                     current_speed, 
                                                     accuracy
                                                     )
            
            self.osu_interface.clock.tick(config.SAMPLE_RATE)
            
        self.model_interface.end_read_eeg()
        return state, map_notes



    def run_intermission(self, map_notes):
        waiting = True
        next_action = "skip"

        correct, count, each_count, each_count_acc = self.osu_interface.compute_notes(map_notes)

        while waiting:
            self.osu_interface.view.draw_intermission(correct=correct, 
                                                      count=count, 
                                                      each_count=each_count, 
                                                      each_count_acc=each_count_acc, 
                                                      model_name=self.model_interface.active_model_name,
                                                      )
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    self.osu_interface.view.close(); 
                    sys.exit()
                    
                if event.type == pygame.KEYDOWN:

                    """
                    train_or_calib 
                    mode: int
                    0 -> train
                    1 -> calib
                    """
                    if event.key == pygame.K_t:
                        self.osu_interface.view.draw_loading("Training...")
                        self.model_interface.train_or_calib(mode=0, RL_open=False, RL_interface=self.RL_interface)
                        
                        time.sleep(3)
                        self.run_model_select()
                        
                        next_action = "train"
                        waiting = False


                    elif event.key == pygame.K_c:
                        self.osu_interface.view.draw_loading("Calibing...")
                        session_records = self.model_interface.train_or_calib(mode=1, RL_open=True, RL_interface=self.RL_interface)
                        
                        """
                        session_records: list
                        record = {
                            'context':np.array --> (ContextIDX.context_dim, ),
                            'action':action,
                        }
                        action 不包含 ONLINE_UPDATE（已移除）
                        """

                        # 同步 delta 值（保留 delta_acc 供 log）
                        self.RL_interface.delta_acc = self.model_interface.delta_acc
                        self.RL_interface.deltaValLoss = self.model_interface.deltaValLoss

                        for record in session_records:
                            reward = self.RL_interface.bandit.compute_reward(
                                deltaLoss=self.RL_interface.deltaValLoss,
                                meanEntropy=record['context'][ContextIDX.MEAN_ENTROPY],
                                alphaNorm=record['context'][ContextIDX.ALPHA_POWER_NORM]
                            )
                            self.RL_interface.bandit.update(record['context'], record['action'], reward)
                        
                        print(f"[Intermission] calib done, deltaValLoss={self.RL_interface.deltaValLoss:.4f}")

                        time.sleep(3)
                        self.run_model_select()
                        next_action = "calib"
                        waiting = False



                    elif event.key == pygame.K_s:
                        waiting = False
                    
        return next_action



    # 按 ESC 結束測試
    def run_test_model(self):
        """
        使用當前 activate model 持續預測 EEG 並在畫面上顯示結果
        按 ESC 結束測試
        """


        ori_SAVE_CSV = config.SAVE_CSV
        config.SAVE_CSV = False

        print(f"[DEBUG] run_test_model 開始，使用模型: {self.model_interface.active_model_name}")
        print(f"[DEBUG] checkpoint: {global_value.NOW_FT_CHECKPOINT}")
        print(f"[INFO] 按 ESC 結束測試")

        # 啟動 EEG 讀取 + 預測 thread
        self.model_interface.start_read_eeg()
        self.model_interface.start_pred_eeg()

        # 清空舊的 buffer
        global_value.pred_buffer.clear()
        global_value.prob_buffer.clear()

        predictionCount = 0
        latestPrediction = -1
        latestProbability = None
        testing = True

        while testing:
            # 監聯鍵盤事件
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

            # 從 pred_buffer 取出最新的預測結果
            while len(global_value.pred_buffer) > 0:
                try:
                    predResult = global_value.pred_buffer.popleft()
                    probResult = global_value.prob_buffer.popleft()
                    predictionCount += 1

                    latestPrediction = predResult
                    latestProbability = probResult

                    # 同時 print 到 console 方便 debug
                    probDisplayStr = " | ".join([f"class{i}: {probResult[i]:.4f}" for i in range(len(probResult))])
                    print(f"[PRED #{predictionCount:04d}] prediction: {predResult}  |  prob: [{probDisplayStr}]")

                except IndexError:
                    break

            # 繪製測試頁面
            self.osu_interface.view.draw_test_model(
                model_name=self.model_interface.active_model_name,
                prediction=latestPrediction,
                probability=latestProbability,
                prediction_count=predictionCount,
            )

            self.osu_interface.clock.tick(30)  # 30 FPS 更新畫面
            # time.sleep(0.5)



        # 停止 EEG 讀取 + 預測 thread
        self.model_interface.end_pred_eeg()
        self.model_interface.end_read_eeg()

        # 清空 buffer
        global_value.pred_buffer.clear()
        global_value.prob_buffer.clear()

        print(f"[DEBUG] run_test_model 結束，共預測 {predictionCount} 次")
        
        config.SAVE_CSV = ori_SAVE_CSV



    def run_model_select(self):
        selecting = True
        model_list = self.model_interface.get_model_list()
        idx = len(model_list) - 1

        while selecting:
            model_rects = self.osu_interface.view.draw_model_select(model_list, idx)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    self.osu_interface.view.close()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP: 
                        idx = max(0, idx - 1)
                    if event.key == pygame.K_DOWN: 
                        idx = min(len(model_list)-1, idx + 1)
                    if event.key == pygame.K_RETURN: 
                        self.model_interface.set_active_model(idx)
                        return 
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for i, rect in enumerate(model_rects):
                        if rect.collidepoint(event.pos): 
                            idx = i

    def start(self):

        print(f"self.logged_in_user: {self.logged_in_user}")
        print(f"global_value.REALTIME_BASE_FILE: {global_value.REALTIME_BASE_FILE}")


        while True:
            action = self.osu_interface.run_main_menu()

            if action == "settings": 
                self.osu_interface.run_settings_menu()

            elif action == "model_select":
                self.run_model_select()

            elif action == "test_model":
                self.run_test_model()

            elif action == "song_select":
                selected_file = self.osu_interface.run_song_select()

                if selected_file == "back": 
                    continue



                if selected_file:
                    
                    self.RL_interface.bandit.load(global_value.RL_path)
                    state, map_notes = self.run_level(selected_file)

                    if state == "finish_song":
                        next_action = self.run_intermission(map_notes)

                        if next_action == "skip":
                            continue
                        elif next_action == "calib":
                            self.RL_interface.bandit.save(global_value.RL_path)

                    elif state == "exit":
                        print("not finish this song")



    def before_start(self):
        
        self.run_account_login()

        
        log_name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        log_path = os.path.join(self.accounts_dir, self.logged_in_user, config.log_dir, log_name)

        global_value.REALTIME_BASE_FILE = os.path.join(self.accounts_dir, self.logged_in_user, config.data_dir)
        global_value.CSV_FILENAME = os.path.join(global_value.REALTIME_BASE_FILE, "eeg_record.csv")
        

        global_value.RL_path = os.path.join(self.accounts_dir, self.logged_in_user, config.bandit_RL)
        global_value.json_path = os.path.join(self.accounts_dir, self.logged_in_user, config.ver_json)


        global_value.save_dir_model = os.path.join(self.accounts_dir, self.logged_in_user, config.model_dir)
        global_value.user_base_model = os.path.join(self.accounts_dir, self.logged_in_user, config.model_dir, config.base_model_dir, config.base_model_name)


        global_value.EEG_CHECKPOINT_BASE_DIR = os.path.join(self.accounts_dir, self.logged_in_user, config.model_dir, config.base_model_dir)
        global_value.EEG_CHECKPOINT_TMP_BASE_DIR = os.path.join(self.accounts_dir, self.logged_in_user, config.model_dir, config.base_model_dir, "Calibed_Model_v001")

        
        shutil.copyfile(config.base_base_model, global_value.user_base_model)
        global_value.NOW_FT_CHECKPOINT = os.path.join(global_value.EEG_CHECKPOINT_BASE_DIR, config.base_model_name[:-4])


        # 紀錄 log
        original_stdout = sys.stdout
        
        with open(log_path, "w", encoding="utf-8") as f:
            sys.stdout = Tee(original_stdout, f)
            try:
                self.start()
            finally:
                sys.stdout = original_stdout
                print(f"\n✅ 輸出已保存到 {log_path}")




def main():
    MainController().before_start()



if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("使用者透過 terminal 中止")

