import pygame
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512) #加緩衝區 要不然會延遲
import sys
import os
import glob
import zipfile
import json
import math
import socket
import numpy as np
from datetime import datetime
from functools import wraps

# ourself folder
from osu.osu_parser import parse_osu_map, parse_metadata
from osu.game_view import GameView


import EEG_DL.Utils.config as config
import EEG_DL.Utils.global_value as global_value

class osu_interface:
    def __init__(self):

        # system
        self.clock = pygame.time.Clock()
        self.key_map = {pygame.K_a: 0, pygame.K_d: 1, pygame.K_s: 2, pygame.K_w: 3}  # 左=0, 右=1, 下=2, 上=3


        # settings json
        self.setting_path = './osu/settings.json'
        self.settings = self.load_settings()

        # songs
        self.songs_dir = "./osu/songs"
        self.import_osz_files()
        self.song_list = self.load_songs()
        self.current_song_index = 0


        # UI
        ## GameView
        self.view = GameView()
        self.view.update_display_mode(self.settings.get('fullscreen', False))


        ## sound_effects
        self.sound_effect_dir = "./osu/sound_effect/"

        self.hit_sound = self.load_hit_sound()  #打擊的聲音

    
        pygame.init() 
        pygame.mixer.init()
        pygame.mixer.music.set_volume(self.settings['volume'])
        


    def load_settings(self):
        # 新增解析度
        default = {
            'volume': 0.5, 
            'speed': 0.8, 
            'fullscreen': False, 
            'offset': 0, 
            'sfx_volume': 0.5,
            'resolution': 800 
        }
        if os.path.exists(self.setting_path):
            try:
                with open(self.setting_path, 'r') as f:
                    data = json.load(f)
                    for k, v in default.items():
                        if k not in data: data[k] = v
                    return data
            except:
                return default
        return default

    def save_settings(self):
        with open(self.setting_path, 'w') as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=4)


    def load_hit_sound(self):
        hit_path = os.path.join(self.sound_effect_dir, "hit.mp3")

        if os.path.exists(hit_path):
            sound = pygame.mixer.Sound(hit_path)
            sound.set_volume(self.settings.get('sfx_volume', 0.5))
            return sound
        else:
            print("no hit.mp3")
            return None



    def import_osz_files(self): 

        # 把 songs folder 內的 .osz 自動解壓縮
        if not os.path.exists(self.songs_dir): 
            os.makedirs(self.songs_dir)

        for filename in os.listdir(self.songs_dir):
            if filename.endswith(".osz") or filename.endswith(".zip"):
                path = os.path.join(self.songs_dir, filename)
                folder_name = os.path.splitext(filename)[0]
                extract_path = os.path.join(self.songs_dir, folder_name)
                try:
                    with zipfile.ZipFile(path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    os.remove(path)
                except Exception as e:
                    print(f"fail {filename}: {e}")

    def load_songs(self):
        files = glob.glob(os.path.join("osu", "songs", "**", "*.osu"), recursive=True)
        meta_list = [parse_metadata(f) for f in files]
        meta_list.sort(key=lambda x: x['difficulty'])
        return meta_list

    def play_music(self, song_meta, loop=False):
        if not song_meta['AudioFilename']: return
        base = os.path.dirname(song_meta['filepath'])
        path = os.path.join(base, song_meta['AudioFilename'])
        if os.path.exists(path):
            pygame.mixer.music.load(path)
            pygame.mixer.music.play(-1 if loop else 0)

    def run_main_menu(self):
        while True:
            # self.clock.tick(60)
            buttons = self.view.draw_main_menu()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.view.close(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if buttons['start'].collidepoint(event.pos): return "song_select"
                    if buttons['settings'].collidepoint(event.pos): return "settings"
                    if buttons['model_select'].collidepoint(event.pos): return "model_select"
                    if buttons['test_model'].collidepoint(event.pos): return "test_model"
                    if buttons['quit'].collidepoint(event.pos): self.view.close(); sys.exit()

    def run_settings_menu(self):
        running = True
        while running:
            # self.clock.tick(60)
            buttons = self.view.draw_settings(self.settings)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.save_settings(); self.view.close(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    if buttons['vol_down'].collidepoint(pos):
                        self.settings['volume'] = max(0.0, self.settings['volume'] - 0.1)
                        pygame.mixer.music.set_volume(self.settings['volume'])
                    if buttons['vol_up'].collidepoint(pos):
                        self.settings['volume'] = min(1.0, self.settings['volume'] + 0.1)
                        pygame.mixer.music.set_volume(self.settings['volume'])
                    if buttons['sfx_down'].collidepoint(pos):
                        self.settings['sfx_volume'] = max(0.0, self.settings.get('sfx_volume', 0.5) - 0.1)
                        if self.hit_sound: self.hit_sound.set_volume(self.settings['sfx_volume'])
                    if buttons['sfx_up'].collidepoint(pos):
                        self.settings['sfx_volume'] = min(1.0, self.settings.get('sfx_volume', 0.5) + 0.1)
                        if self.hit_sound: self.hit_sound.set_volume(self.settings['sfx_volume'])

                    if buttons['res_down'].collidepoint(pos):
                        new_res = max(600, self.settings.get('resolution', 800) - 100)
                        self.settings['resolution'] = new_res
                        self.view.resize(new_res)
                    
                    if buttons['res_up'].collidepoint(pos):
                        new_res = min(1200, self.settings.get('resolution', 800) + 100)
                        self.settings['resolution'] = new_res
                        self.view.resize(new_res)

                    if buttons['spd_down'].collidepoint(pos): self.settings['speed'] = max(0.08, self.settings['speed'] - 0.01)
                    if buttons['spd_up'].collidepoint(pos): self.settings['speed'] = min(2.0, self.settings['speed'] + 0.01)
                    if buttons['off_down'].collidepoint(pos): self.settings['offset'] -= 5
                    if buttons['off_up'].collidepoint(pos): self.settings['offset'] += 5
                    if buttons['off_test'].collidepoint(pos): self.run_offset_wizard()
                    if buttons['fullscreen'].collidepoint(pos):
                        self.settings['fullscreen'] = not self.settings['fullscreen']
                        self.view.update_display_mode(self.settings['fullscreen'])
                    if buttons['back'].collidepoint(pos): self.save_settings(); running = False

    def run_offset_wizard(self):
        print("Entering Offset Wizard...")
        running = True
        bpm = 120
        beat_interval = 60000 / bpm
        start_ticks = pygame.time.get_ticks()
        
        last_played_beat = -1
        tap_history = []
        
        while running:
            current_time = pygame.time.get_ticks() - start_ticks
            
            beat_index = int(current_time // beat_interval)
            
            if beat_index > last_played_beat:
                if self.hit_sound: self.hit_sound.play()
                last_played_beat = beat_index

            progress = (current_time % beat_interval) / beat_interval
            
            avg_diff = sum(tap_history) / len(tap_history) if tap_history else 0
            
            self.view.draw_offset_wizard(self.settings['offset'], avg_diff, progress, tap_history)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.view.close(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: 
                        running = False
                    elif event.key == pygame.K_RETURN:
                        self.settings['offset'] += int(avg_diff)
                        self.save_settings()
                        running = False
                    else:
                        beat_time = beat_index * beat_interval
                        next_beat_time = (beat_index + 1) * beat_interval
                        
                        diff_curr = current_time - beat_time
                        diff_next = current_time - next_beat_time
                        
                        if abs(diff_curr) < abs(diff_next):
                            diff = diff_curr
                        else:
                            diff = diff_next
                            
                        tap_history.append(diff)
                        if len(tap_history) > 20: tap_history.pop(0)
                        
                        if self.hit_sound: self.hit_sound.play()
            
            self.clock.tick(120)

    def run_song_select(self):
        if not self.song_list:
            print("no songs")
            return "back"
        selecting = True
        self.play_music(self.song_list[self.current_song_index], loop=True)
        while selecting:
            # self.clock.tick(60)
            song_rects, back_btn = self.view.draw_song_select(self.song_list, self.current_song_index)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.view.close(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        self.current_song_index = (self.current_song_index + 1) % len(self.song_list)
                        self.play_music(self.song_list[self.current_song_index], loop=True)
                    elif event.key == pygame.K_UP:
                        self.current_song_index = (self.current_song_index - 1) % len(self.song_list)
                        self.play_music(self.song_list[self.current_song_index], loop=True)
                    elif event.key == pygame.K_RETURN:
                        pygame.mixer.music.stop()
                        return self.song_list[self.current_song_index]['filepath']
                if event.type == pygame.MOUSEBUTTONDOWN:
                    for i, rect in enumerate(song_rects):
                        if rect.collidepoint(event.pos):
                            if i == self.current_song_index: pygame.mixer.music.stop(); return self.song_list[self.current_song_index]['filepath']
                            else: self.current_song_index = i; self.play_music(self.song_list[self.current_song_index], loop=True)
                    if back_btn.collidepoint(event.pos): pygame.mixer.music.stop(); return "back"

    



    def check_hit(self, 
                  map_notes,
                  current_note_index,
                  key,

                  game_time_ms,

                  score,
                  hits,
                  ):

        # for n in map_notes:

        n = map_notes[current_note_index]
        
        if abs(n['time'] - game_time_ms) < 150:

            if key >= 0 and not n['hit']:
                self.view.trigger_effect(key)

            if n['lane'] == key and not n['hit']:
                
                map_notes[current_note_index]['hit'] = True
                score += 100
                hits += 1
                

                if self.hit_sound: 
                    self.hit_sound.play()

                #  break

        return map_notes, score, hits
    
    def calculate_marker(self, 
                         map_notes, 
                         current_note_index, 
                         game_time_ms
                         ):
        
        # 從出現到 判定後 check_window 時間，以 speed 0.08 就是 3125 + 150 (ms)
        current_marker = -1
        check_window = 150  # (ms)

        # speed 0.08 --> 約3.125s
        # speed 0.1 ---> 約2.5s
        # speed 0.2 ---> 約1.25s
        distance = 0.25 * 1000
        st_to_ed_time = distance / self.settings['speed'] # 單位 ms
        # print(f"st_to_ed_time: {st_to_ed_time}")
        
        # 優化搜尋：從上一個音符位置開始找即可
        for j in range(current_note_index, len(map_notes)):
            n = map_notes[j]

            current_note_index = j 
            
            # 太舊的音符不用看
            if n['time'] < game_time_ms - check_window:
                continue
            
        
            # 新的音符
            if n['time'] >= game_time_ms:

                # 太遠
                if n['time'] > game_time_ms + st_to_ed_time:
                    break
                
                # 出現了
                if n['time'] <= game_time_ms + st_to_ed_time:
                    current_marker = n['lane']
                    return current_marker, current_note_index
                
            elif abs(n['time'] - game_time_ms) <= check_window :
                current_marker = n['lane']
                return current_marker, current_note_index
            
        return current_marker, current_note_index

    def compute_notes(self, map_notes):

        correct = 0
        count = len(map_notes)

        each_count = np.zeros(config.N_Class)
        each_count_acc = np.zeros(config.N_Class)

        for note in map_notes:
            if note['hit']:
                correct += 1
                each_count_acc[note['lane']] += 1

            each_count[note['lane']] += 1


        



        return correct, count, each_count, each_count_acc

