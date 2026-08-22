import pygame
import math

COLORS = {
    'bg': (20, 20, 20),
    'text': (255, 255, 255),
    'highlight': (255, 215, 0),
    'dim': (0, 0, 0, 180),
    'ui_bg': (40, 40, 50),
    'btn': (70, 70, 90),
    'btn_hover': (100, 100, 120),
    'arrows': [
        (255, 80, 80),
        (80, 255, 80),
        (80, 200, 255),
        (255, 100, 255)
    ],
    'receptor': (150, 150, 150)
}

class GameView:
    def __init__(self, size=800):
        pygame.init()
        self.w = size
        self.h = size
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("BCI Mania - Center Out Mode")
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 48)
        self.lane_effects = [0, 0, 0, 0]
        
        self.recalc_geometry()

    def recalc_geometry(self):
        """ 當視窗大小改變時，重新計算中心點與判定圓半徑 """
        self.cx = self.w // 2
        self.cy = self.h // 2
        self.hit_radius = int(self.w * 0.4) 

    def resize(self, new_size):
        """ [新功能] 調整視窗大小 (強制正方形) """
        self.w = new_size
        self.h = new_size
        try:
            self.screen = pygame.display.set_mode((self.w, self.h))
            self.recalc_geometry()
        except pygame.error as e:
            print(f"Resize Error: {e}")

    def update_display_mode(self, fullscreen):
        try:
            if fullscreen:
                flags = pygame.FULLSCREEN
                self.screen = pygame.display.set_mode((self.w, self.h), flags)
            else:
                flags = 0 
                self.screen = pygame.display.set_mode((self.w, self.h), flags)
        except pygame.error as e:
            print(f"Display Error: {e}")
            self.screen = pygame.display.set_mode((self.w, self.h))

    def trigger_effect(self, lane):
        self.lane_effects[lane] = 10 

    def truncate_text(self, text, max_width):
        if self.font.size(text)[0] <= max_width: return text
        ellipsis = "..."
        for i in range(len(text), 0, -1):
            short_text = text[:i] + ellipsis
            if self.font.size(short_text)[0] <= max_width: return short_text
        return text

    def get_difficulty_color(self, note_count):
        if note_count < 300: return (100, 255, 100)
        if note_count < 800: return (100, 200, 255)
        if note_count < 1500: return (255, 255, 100)
        if note_count < 2500: return (255, 128, 0)
        return (255, 50, 50)

    def draw_arrow(self, surface, color, x, y, direction, size=40, filled=True):
        points = [(-size//2, -size//2), (size//2, 0), (-size//2, size//2)]
        angle = 0
        if direction == 0: angle = 180     # 左
        elif direction == 1: angle = 0     # 右
        elif direction == 2: angle = 90    # 下
        elif direction == 3: angle = -90   # 上

        rotated_points = []
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        for px, py in points:
            nx = px * cos_a - py * sin_a
            ny = px * sin_a + py * cos_a
            rotated_points.append((x + nx, y + ny))
            
        if filled:
            pygame.draw.polygon(surface, color, rotated_points)
        else:
            pygame.draw.polygon(surface, color, rotated_points, 3)

    def draw_button(self, text, x, y, w, h, active=False):
        mouse_pos = pygame.mouse.get_pos()
        rect = pygame.Rect(x, y, w, h)
        color = COLORS['btn_hover'] if rect.collidepoint(mouse_pos) or active else COLORS['btn']
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        pygame.draw.rect(self.screen, (200, 200, 200), rect, 2, border_radius=10)
        txt_surf = self.font.render(text, True, COLORS['text'])
        txt_rect = txt_surf.get_rect(center=rect.center)
        self.screen.blit(txt_surf, txt_rect)
        return rect



    def draw_account_select(self, account_list, idx):
        self.screen.fill(COLORS['bg'])
        title = self.big_font.render("SELECT ACCOUNT", True, COLORS['highlight'])
        self.screen.blit(title, title.get_rect(center=(self.cx, 100)))
        
        account_rects = []
        start_y = 200
        for i, name in enumerate(account_list):
            color = COLORS['highlight'] if i == idx else COLORS['text']
            surf = self.font.render(name, True, color)
            rect = surf.get_rect(center=(self.cx, start_y + i * 60))
            
            # 繪製一個簡單的外框背景
            bg_rect = rect.inflate(40, 20)
            pygame.draw.rect(self.screen, (60, 60, 70), bg_rect, border_radius=5)
            if i == idx:
                pygame.draw.rect(self.screen, COLORS['highlight'], bg_rect, 2, border_radius=5)
                
            self.screen.blit(surf, rect)
            account_rects.append(bg_rect)
            
        pygame.display.flip()
        return account_rects




    def draw_main_menu(self):
        self.screen.fill(COLORS['bg'])
        title = self.big_font.render("BCI MANIA", True, COLORS['highlight'])
        self.screen.blit(title, title.get_rect(center=(self.cx, 150)))
        
        btn_w, btn_h = 200, 60
        start_x = self.cx - btn_w // 2
        
        btn_start = self.draw_button("Start Game", start_x, 280, btn_w, btn_h)
        btn_settings = self.draw_button("Settings", start_x, 360, btn_w, btn_h)
        btn_model_select = self.draw_button("Model Select", start_x, 440, btn_w, btn_h)
        btn_test_model = self.draw_button("Test Model", start_x, 520, btn_w, btn_h)
        btn_quit = self.draw_button("Quit", start_x, 620, btn_w, btn_h)
        pygame.display.flip()
        return {'start': btn_start, 'settings': btn_settings, 'model_select': btn_model_select, 'test_model': btn_test_model, 'quit': btn_quit}

    def draw_settings(self, settings):
        self.screen.fill(COLORS['bg'])
        title = self.big_font.render("SETTINGS", True, COLORS['highlight'])
        self.screen.blit(title, title.get_rect(center=(self.cx, 60)))
        
        buttons = {}
        y_offset = 120
        gap = 70
        label_x = self.cx - 150
        btn_start_x = self.cx + 50

        vol_val = int(settings['volume'] * 100)
        self.screen.blit(self.font.render(f"Music: {vol_val}%", True, COLORS['text']), (label_x, y_offset + 10))
        buttons['vol_down'] = self.draw_button("-", btn_start_x, y_offset, 50, 40)
        buttons['vol_up'] = self.draw_button("+", btn_start_x + 70, y_offset, 50, 40)
        
        y_offset += gap
        sfx_val = int(settings.get('sfx_volume', 0.5) * 100)
        self.screen.blit(self.font.render(f"SFX: {sfx_val}%", True, COLORS['text']), (label_x, y_offset + 10))
        buttons['sfx_down'] = self.draw_button("-", btn_start_x, y_offset, 50, 40)
        buttons['sfx_up'] = self.draw_button("+", btn_start_x + 70, y_offset, 50, 40)
        
        y_offset += gap
        res_val = settings.get('resolution', 800)
        self.screen.blit(self.font.render(f"Size: {res_val}x{res_val}", True, COLORS['highlight']), (label_x, y_offset + 10))
        buttons['res_down'] = self.draw_button("-", btn_start_x, y_offset, 50, 40)
        buttons['res_up'] = self.draw_button("+", btn_start_x + 70, y_offset, 50, 40)

        y_offset += gap
        spd_val = settings['speed']
        self.screen.blit(self.font.render(f"Speed: {spd_val:.2f}", True, COLORS['text']), (label_x, y_offset + 10))
        buttons['spd_down'] = self.draw_button("-", btn_start_x, y_offset, 50, 40)
        buttons['spd_up'] = self.draw_button("+", btn_start_x + 70, y_offset, 50, 40)

        y_offset += gap
        offset_val = settings.get('offset', 0)
        sign = "+" if offset_val >= 0 else ""
        self.screen.blit(self.font.render(f"Offset: {sign}{offset_val}ms", True, COLORS['text']), (label_x, y_offset + 10))
        buttons['off_down'] = self.draw_button("-", btn_start_x - 50, y_offset, 50, 40)
        buttons['off_up'] = self.draw_button("+", btn_start_x + 10, y_offset, 50, 40)
        buttons['off_test'] = self.draw_button("Test", btn_start_x + 80, y_offset, 80, 40)

        y_offset += gap
        is_fs = settings.get('fullscreen', False)
        mode_text = "FS: ON" if is_fs else "FS: OFF"
        buttons['fullscreen'] = self.draw_button(mode_text, self.cx - 100, y_offset, 200, 50, active=is_fs)

        buttons['back'] = self.draw_button("Back", self.cx - 100, self.h - 100, 200, 60)
        pygame.display.flip()
        return buttons

    def draw_offset_wizard(self, current_offset, avg_diff, beat_progress, history):
        self.screen.fill(COLORS['bg'])
        title = self.big_font.render("OFFSET WIZARD", True, COLORS['highlight'])
        self.screen.blit(title, title.get_rect(center=(self.cx, 80)))
        pygame.draw.line(self.screen, COLORS['text'], (self.cx, self.cy - 100), (self.cx, self.cy + 100), 4)
        
        cursor_x = (self.cx - 200) + beat_progress * 400
        pygame.draw.circle(self.screen, COLORS['highlight'], (int(cursor_x), self.cy), 15)
        
        txt_cur = self.font.render(f"Offset: {current_offset} ms", True, COLORS['text'])
        self.screen.blit(txt_cur, (50, self.h - 150))
        pygame.display.flip()

    def draw_song_select(self, song_list, idx):
        self.screen.fill(COLORS['bg'])
        self.screen.blit(self.big_font.render("SELECT MUSIC", True, COLORS['highlight']), (50, 50))
        start_y = 150
        song_rects = []
        MAX_TEXT_WIDTH = self.w - 110 
        for i, song in enumerate(song_list):
            note_count = song.get('difficulty', 0)
            if i == idx: color = (255, 255, 255)
            else: color = self.get_difficulty_color(note_count)
            raw_txt = f"[{note_count}] {song['Artist']} - {song['Title']}"
            display_txt = self.truncate_text(raw_txt, MAX_TEXT_WIDTH)
            surf = self.font.render(display_txt, True, color)
            rect = surf.get_rect(topleft=(60, start_y + i * 50))
            self.screen.blit(surf, rect)
            song_rects.append(rect)
            if i == idx: pygame.draw.circle(self.screen, color, (40, start_y + i * 50 + 15), 5)
        back_btn = self.draw_button("Back", 50, self.h - 100, 100, 40)
        pygame.display.flip()
        return song_rects, back_btn

    def draw_model_select(self, model_list, idx):
        self.screen.fill(COLORS['bg'])
        self.screen.blit(self.big_font.render("SELECT MODEL", True, COLORS['highlight']), (50, 50))
        start_y = 150
        model_rects = []
        for i, model_name in enumerate(model_list):
            color = COLORS['highlight'] if i == idx else COLORS['text']
            surf = self.font.render(model_name, True, color)
            rect = surf.get_rect(center=(self.cx, start_y + i * 50))
            self.screen.blit(surf, rect)
            model_rects.append(rect)
            if i == idx: pygame.draw.circle(self.screen, color, (rect.left - 20, rect.centery), 5)
        pygame.display.flip()
        return model_rects

    def draw_play_screen(self, notes, time_ms, score, model_info, speed, accuracy):
        self.screen.fill(COLORS['bg'])
        
        SAFE_MS = 75 
        safe_pixel_offset = SAFE_MS * speed
        
        LINE_LEN = 60 
        JUDGMENT_COLOR = (100, 100, 100)

        active_lanes = [False, False, False, False]
        for n in notes:
            if not n['hit']:
                dist_remaining = (n['time'] - time_ms) * speed
                current_r = self.hit_radius - dist_remaining
                if 0 < current_r < self.hit_radius + 150:
                    active_lanes[n['lane'] % 4] = True

        positions = [
            (self.cx - self.hit_radius, self.cy),
            (self.cx + self.hit_radius, self.cy),
            (self.cx, self.cy + self.hit_radius),
            (self.cx, self.cy - self.hit_radius),
        ]
        
        for pos in positions:
            pygame.draw.line(self.screen, (40, 40, 40), (self.cx, self.cy), pos, 2)
        pygame.draw.circle(self.screen, (60, 60, 70), (self.cx, self.cy), 10)

        r_inner = self.hit_radius - safe_pixel_offset
        r_outer = self.hit_radius + safe_pixel_offset

        pygame.draw.line(self.screen, JUDGMENT_COLOR, (self.cx - r_inner, self.cy - LINE_LEN//2), (self.cx - r_inner, self.cy + LINE_LEN//2), 1)
        pygame.draw.line(self.screen, JUDGMENT_COLOR, (self.cx - r_outer, self.cy - LINE_LEN//2), (self.cx - r_outer, self.cy + LINE_LEN//2), 1)

        pygame.draw.line(self.screen, JUDGMENT_COLOR, (self.cx - LINE_LEN//2, self.cy - r_inner), (self.cx + LINE_LEN//2, self.cy - r_inner), 1)
        pygame.draw.line(self.screen, JUDGMENT_COLOR, (self.cx - LINE_LEN//2, self.cy - r_outer), (self.cx + LINE_LEN//2, self.cy - r_outer), 1)

        pygame.draw.line(self.screen, JUDGMENT_COLOR, (self.cx - LINE_LEN//2, self.cy + r_inner), (self.cx + LINE_LEN//2, self.cy + r_inner), 1)
        pygame.draw.line(self.screen, JUDGMENT_COLOR, (self.cx - LINE_LEN//2, self.cy + r_outer), (self.cx + LINE_LEN//2, self.cy + r_outer), 1)

        pygame.draw.line(self.screen, JUDGMENT_COLOR, (self.cx + r_inner, self.cy - LINE_LEN//2), (self.cx + r_inner, self.cy + LINE_LEN//2), 1)
        pygame.draw.line(self.screen, JUDGMENT_COLOR, (self.cx + r_outer, self.cy - LINE_LEN//2), (self.cx + r_outer, self.cy + LINE_LEN//2), 1)

        for i in range(4):
            pos = positions[i]
            color = COLORS['arrows'][i]
            
            if self.lane_effects[i] > 0:
                self.draw_arrow(self.screen, color, pos[0], pos[1], i, size=45, filled=True)
                pygame.draw.circle(self.screen, (color[0], color[1], color[2], 100), pos, 35, 2)
                self.lane_effects[i] -= 1
            else:
                self.draw_arrow(self.screen, COLORS['receptor'], pos[0], pos[1], i, size=40, filled=False)

            text_str = ""
            text_x = pos[0]
            text_y = pos[1]
            
            if i == 0:  # 左
                text_str = "Left"
                text_x = pos[0] + 150  # 左側 receptor 正右方
                text_y = pos[1] 
            elif i == 1:  # 右
                text_str = "Right"
                text_x = pos[0] - 150  # 右側 receptor 正左方
                text_y = pos[1] 
            elif i == 2:  # 下
                text_str = "Foot"
                text_x = pos[0]
                text_y = pos[1] - 100  # 下方 receptor 正上方
            elif i == 3:  # 上
                text_str = "Tongue"
                text_x = pos[0]
                text_y = pos[1] + 100  # 下方 receptor 正上方


            if text_str:
                if active_lanes[i]:
                    text_color = (255, 255, 255) #觸發的顏色 目前是白色 
                else:
                    text_color = (80, 80, 80) #沒觸發的顏色 目前是灰暗色

                text_surf = self.font.render(text_str, True, text_color)
                text_rect = text_surf.get_rect(center=(text_x, text_y))
                self.screen.blit(text_surf, text_rect)

        for n in notes:
            if n['hit']: continue
            
            dist_remaining = (n['time'] - time_ms) * speed
            current_r = self.hit_radius - dist_remaining
            
            if 0 < current_r < self.hit_radius + 150:
                lane = n['lane'] % 4 
                
                if lane == 0:   x, y = self.cx - current_r, self.cy
                elif lane == 1: x, y = self.cx + current_r, self.cy
                elif lane == 2: x, y = self.cx, self.cy + current_r
                elif lane == 3: x, y = self.cx, self.cy - current_r

                self.draw_arrow(self.screen, COLORS['arrows'][lane], x, y, lane, size=35, filled=True)

        self.screen.blit(self.font.render(f"Score: {score}", True, COLORS['text']), (10, 10))
        self.screen.blit(self.font.render(f"Acc: {accuracy:.2f}%", True, COLORS['text']), (10, 40))
        
        model_surf = self.font.render(f"Model: {model_info}", True, COLORS['highlight'])
        m_rect = model_surf.get_rect(bottomright=(self.w - 10, self.h - 10))
        self.screen.blit(model_surf, m_rect)

        pygame.display.flip()

    def draw_pause_menu(self, opt):
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        overlay.fill(COLORS['dim'])
        self.screen.blit(overlay, (0, 0))
        
        title = self.big_font.render("PAUSED", True, COLORS['highlight'])
        self.screen.blit(title, title.get_rect(center=(self.cx, self.cy - 120)))
        
        opts = ["Continue", "Back to Menu", "Skip Level"]
        
        rects = []
        start_y = self.cy - 20
        gap = 70
        
        for i, txt in enumerate(opts):
            color = COLORS['highlight'] if i == opt else COLORS['text']
            surf = self.font.render(txt, True, color)
            rect = surf.get_rect(center=(self.cx, start_y + i * gap))
            
            self.screen.blit(surf, rect)
            rects.append(rect)
            
        pygame.display.flip()
        return rects

    def draw_intermission(self, correct, count, each_count, each_count_acc, model_name):
        self.screen.fill(COLORS['ui_bg'])
        texts = [
            ("LEVEL COMPLETE", COLORS['highlight']),
            (f"Accuracy (total): {correct} / {count}", COLORS['text']),
            (f"Data Each class: {each_count}", COLORS['text']), 
            (f"Correct Each class: {each_count_acc}", COLORS['text']), 
            (f"Current Model: {model_name}", COLORS['text']), 
            ("", None), 
            ("[T] Train Model", COLORS['highlight']),
            ("[C] Calib Model", COLORS['text']),
            ("[S] Skip", COLORS['text'])
        ]
        start_y = 150
        for i, (txt, col) in enumerate(texts):
            if not txt: continue
            surf = self.big_font.render(txt, True, col)
            self.screen.blit(surf, surf.get_rect(center=(self.cx, start_y + i * 60)))
        pygame.display.flip()


    def draw_test_model(self, model_name, prediction=-1, probability=None, prediction_count=0):
        """測試模型頁面：顯示即時預測結果"""
        self.screen.fill(COLORS['bg'])

        # 半透明遮罩
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        # 標題
        title = self.big_font.render("TEST MODEL", True, COLORS['highlight'])
        self.screen.blit(title, title.get_rect(center=(self.cx, 80)))

        # 模型名稱
        model_surf = self.font.render(f"Model: {model_name}", True, COLORS['text'])
        self.screen.blit(model_surf, model_surf.get_rect(center=(self.cx, 140)))

        # 預測次數
        count_surf = self.font.render(f"Predictions: {prediction_count}", True, (180, 180, 180))
        self.screen.blit(count_surf, count_surf.get_rect(center=(self.cx, 180)))

        # 當前預測結果
        if prediction >= 0 and probability is not None:
            # Now Predict 標題
            pred_title = self.big_font.render("Now Predict", True, COLORS['highlight'])
            self.screen.blit(pred_title, pred_title.get_rect(center=(self.cx, 250)))

            # 預測 class
            class_names = ["Left", "Right", "Down", "Up"]
            pred_label = class_names[prediction] if prediction < len(class_names) else f"Class {prediction}"
            pred_color = COLORS['arrows'][prediction] if prediction < 4 else COLORS['text']

            pred_surf = self.big_font.render(f">> {pred_label} (Class {prediction}) <<", True, pred_color)
            self.screen.blit(pred_surf, pred_surf.get_rect(center=(self.cx, 310)))

            # 機率條
            bar_width = 300
            bar_height = 30
            bar_start_x = self.cx - bar_width // 2
            bar_start_y = 370

            for i, prob in enumerate(probability):
                # NaN 防護
                if prob != prob:  # NaN check (NaN != NaN)
                    prob = 0.0

                y = bar_start_y + i * (bar_height + 15)
                color = COLORS['arrows'][i] if i < 4 else COLORS['text']
                label = class_names[i] if i < len(class_names) else f"C{i}"

                # 標籤
                lbl_surf = self.font.render(f"{label}:", True, color)
                self.screen.blit(lbl_surf, (bar_start_x - 80, y + 2))

                # 背景條
                pygame.draw.rect(self.screen, (60, 60, 70), (bar_start_x, y, bar_width, bar_height), border_radius=5)

                # 填充條
                fill_w = int(bar_width * prob)
                if fill_w > 0:
                    pygame.draw.rect(self.screen, color, (bar_start_x, y, fill_w, bar_height), border_radius=5)

                # 數值
                prob_surf = self.font.render(f"{prob:.2%}", True, COLORS['text'])
                self.screen.blit(prob_surf, (bar_start_x + bar_width + 10, y + 2))

        else:
            # 等待預測
            wait_surf = self.big_font.render("Waiting for prediction...", True, (150, 150, 150))
            self.screen.blit(wait_surf, wait_surf.get_rect(center=(self.cx, 350)))

        # ESC 提示
        esc_surf = self.font.render("Press ESC to return to menu", True, (120, 120, 120))
        self.screen.blit(esc_surf, esc_surf.get_rect(center=(self.cx, self.h - 50)))

        pygame.display.flip()

    def draw_loading(self, msg):
        overlay = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        overlay.fill(COLORS['dim'])
        self.screen.blit(overlay, (0,0))
        surf = self.big_font.render(msg, True, COLORS['highlight'])
        self.screen.blit(surf, surf.get_rect(center=(self.cx, self.cy)))
        pygame.display.flip()

    def close(self):
        pygame.quit()