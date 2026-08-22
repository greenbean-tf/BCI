"""
遊玩流程控制器。

拆自原 main.py 的 run_level / run_zero_level / run_pause_menu。
"""
import sys
import threading
import numpy as np
import pygame
import pylsl

from osu.osu_parser import parse_osu_map
import config
import global_state


class LevelRunner:
    """管理單首歌曲的遊玩迴圈"""

    def __init__(self, osu_interface, model_interface, RL_interface):
        self.osu = osu_interface
        self.model = model_interface
        self.rl = RL_interface

        self.recording_active = True
        self.current_game_time_ms = 0.0
        self.rl_loop_running = True

    # ─── 暫停選單 ───
    def run_pause_menu(self):
        paused = True
        opt = 0

        global_state.current_marker = 99
        global_state.current_predict = config.stay_marker
        global_state.current_prob = config.stay_prob.copy()

        while paused:
            rects = self.osu.view.draw_pause_menu(opt)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.osu.view.close()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        opt -= 1
                        opt = np.clip(opt, 0, 2)
                    if event.key == pygame.K_DOWN:
                        opt += 1
                        opt = np.clip(opt, 0, 2)
                    if event.key == pygame.K_RETURN:
                        if opt == 0:
                            return "continue"
                        elif opt == 1:
                            return "quit"
                        elif opt == 2:
                            return "skip"
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if rects[0].collidepoint(event.pos):
                        return "continue"
                    if rects[1].collidepoint(event.pos):
                        return "quit"
                    if rects[2].collidepoint(event.pos):
                        return "skip"

    # ─── 正常遊玩 ───
    def run_level(self, map_file, tcp_server, rl_loop_worker_fn):
        map_data = parse_osu_map(map_file)
        map_title = map_data['meta'].get('Title', 'Unknown')
        print(f"song name: {map_title}")

        if map_title == config.the_zero_song_name:
            return self.run_zero_level(map_file, tcp_server)

        state = "finish_song"

        self.osu.view.draw_loading("Loading Beatmap...")
        pygame.display.flip()
        pygame.event.pump()

        score, hits, accuracy = 0, 0, 0
        current_speed = self.osu.settings['speed']
        current_note_index = 0
        map_notes = map_data['notes']
        map_end_time = map_notes[-1]['time'] + 2000 if map_notes else 2000

        self.osu.play_music(map_data['meta'])

        open_judge = 1
        pred_key = config.stay_marker
        prob_key = config.stay_prob.copy()

        total_pause_time = 0.0
        start_lsl_time = pylsl.local_clock()
        global_state.start_lsl_time = start_lsl_time
        self.model.start_read_eeg()
        self.model.start_pred_eeg()

        if config.SAVE_CSV:
            self.recording_active = True

        # RL 閉環
        if config.USE_RL_LOOP:
            self.rl_loop_running = True
            RL_thread = threading.Thread(target=rl_loop_worker_fn, daemon=True)
            RL_thread.start()

        song_start_time = pylsl.local_clock()
        playing = True

        while playing:
            current_lsl_ts = pylsl.local_clock()
            game_time_ms = (current_lsl_ts - start_lsl_time - total_pause_time) * 1000.0
            self.current_game_time_ms = game_time_ms

            current_marker, current_note_index = self.osu.calculate_marker(
                map_notes, current_note_index, game_time_ms)

            if current_marker >= 0:
                n = map_notes[current_note_index]

                if abs(n['time'] - game_time_ms) < 150 and open_judge:
                    current_len = len(global_state.pred_buffer)
                    pred_key = global_state.pred_buffer[current_len - 1]
                    prob_key = global_state.prob_buffer[current_len - 1]
                    global_state.pred_buffer.clear()
                    global_state.prob_buffer.clear()
                    open_judge = 0

                map_notes, score, hits = self.osu.check_hit(
                    map_notes, current_note_index, pred_key, game_time_ms, score, hits)
            else:
                open_judge = 1
                pred_key = config.stay_marker
                prob_key = config.stay_prob.copy()
                global_state.pred_buffer.clear()
                global_state.prob_buffer.clear()

            global_state.current_marker = current_marker
            global_state.current_predict = pred_key
            global_state.current_prob = prob_key

            # 鍵盤事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.model.end_pred_eeg()
                    self.model.end_read_eeg()
                    self.osu.view.close()
                    tcp_server.stop()
                    self.model.tcp_client.stop()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key in self.osu.key_map:
                        hand_key = self.osu.key_map[event.key]
                        self.osu.view.trigger_effect(hand_key)
                        map_notes, score, hits = self.osu.check_hit(
                            map_notes, current_note_index, hand_key, game_time_ms, score, hits)

                    elif event.key == pygame.K_ESCAPE:
                        pygame.mixer.music.pause()
                        pause_start = pylsl.local_clock()

                        do = self.run_pause_menu()

                        if do == "quit":
                            state = "exit"
                            return state, current_note_index
                        elif do == "skip":
                            playing = False
                            break

                        pygame.mixer.music.unpause()
                        total_pause_time += (pylsl.local_clock() - pause_start)

            # 準確度
            passed_notes = 0
            for n in map_notes:
                if n['time'] < game_time_ms - 150:
                    passed_notes += 1
                else:
                    break
            accuracy = (hits / passed_notes * 100) if passed_notes > 0 else 100.0

            if game_time_ms > map_end_time:
                print(f"song_start_time: {song_start_time}")
                print(f"song_end_time: {pylsl.local_clock()}")
                self.recording_active = False
                self.rl_loop_running = False
                playing = False

            self.osu.view.draw_play_screen(
                map_notes, game_time_ms, score,
                self.model.active_model_name, current_speed, accuracy)

            self.osu.clock.tick(config.SAMPLE_RATE)

        self.model.end_pred_eeg()
        self.model.end_read_eeg()
        return state, map_notes

    # ─── Zero Level（第零首自動控制） ───
    def run_zero_level(self, map_file, tcp_server):
        human_control_acc = 1.0
        print("now run_zero_level")
        map_data = parse_osu_map(map_file)

        current_note_index = 0
        map_notes = map_data['notes']
        map_end_time = map_notes[-1]['time'] + 2000 if map_notes else 2000

        state = "finish_song"

        self.osu.view.draw_loading("Loading Beatmap...")
        pygame.display.flip()
        pygame.event.pump()

        score, hits, accuracy = 0, 0, 0
        current_speed = self.osu.settings['speed']

        self.osu.play_music(map_data['meta'])

        open_judge = 1
        pred_key = config.stay_marker
        prob_key = config.stay_prob.copy()

        total_pause_time = 0.0
        start_lsl_time = pylsl.local_clock()
        global_state.start_lsl_time = start_lsl_time
        self.model.start_read_eeg()

        if config.SAVE_CSV:
            self.recording_active = True

        song_start_time = pylsl.local_clock()
        playing = True

        while playing:
            current_lsl_ts = pylsl.local_clock()
            game_time_ms = (current_lsl_ts - start_lsl_time - total_pause_time) * 1000.0
            self.current_game_time_ms = game_time_ms

            current_marker, current_note_index = self.osu.calculate_marker(
                map_notes, current_note_index, game_time_ms)

            if current_marker >= 0:
                n = map_notes[current_note_index]

                if abs(n['time'] - game_time_ms) < 150 and open_judge:
                    correct_lane = n['lane']

                    if np.random.random() < human_control_acc:
                        pred_key = correct_lane
                    else:
                        wrong_lanes = [l for l in range(4) if l != correct_lane]
                        pred_key = np.random.choice(wrong_lanes)

                    prob_key = config.stay_prob.copy()
                    prob_key[pred_key] = 1.0
                    open_judge = 0

                map_notes, score, hits = self.osu.check_hit(
                    map_notes, current_note_index, pred_key, game_time_ms, score, hits)
            else:
                open_judge = 1
                pred_key = config.stay_marker
                prob_key = config.stay_prob.copy()

            global_state.current_marker = current_marker
            global_state.current_predict = pred_key
            global_state.current_prob = prob_key

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.model.end_read_eeg()
                    self.osu.view.close()
                    tcp_server.stop()
                    self.model.tcp_client.stop()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key in self.osu.key_map:
                        hand_key = self.osu.key_map[event.key]
                        self.osu.view.trigger_effect(hand_key)
                        map_notes, score, hits = self.osu.check_hit(
                            map_notes, current_note_index, hand_key, game_time_ms, score, hits)

                    elif event.key == pygame.K_ESCAPE:
                        pygame.mixer.music.pause()
                        pause_start = pylsl.local_clock()

                        do = self.run_pause_menu()

                        if do == "quit":
                            state = "exit"
                            return state, current_note_index
                        elif do == "skip":
                            playing = False
                            break

                        pygame.mixer.music.unpause()
                        total_pause_time += (pylsl.local_clock() - pause_start)

            passed_notes = 0
            for n in map_notes:
                if n['time'] < game_time_ms - 150:
                    passed_notes += 1
                else:
                    break
            accuracy = (hits / passed_notes * 100) if passed_notes > 0 else 100.0

            if game_time_ms > map_end_time:
                print(f"song_start_time: {song_start_time}")
                print(f"song_end_time: {pylsl.local_clock()}")
                self.recording_active = False
                playing = False

            self.osu.view.draw_play_screen(
                map_notes, game_time_ms, score, "no model", current_speed, accuracy)

            self.osu.clock.tick(config.SAMPLE_RATE)

        self.model.end_read_eeg()
        return state, map_notes


# ─── Debug ───
if __name__ == "__main__":
    print("=== core.level_runner standalone test ===")
    print("LevelRunner requires osu_interface and model_interface to run.")
    print("✅ level_runner import OK")
