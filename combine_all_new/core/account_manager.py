"""
帳號管理器。

拆自原 main.py 的 MainController 中帳號相關方法。
"""
import os
import sys
import json
import pygame
import shutil
from pathlib import Path

import config
import global_state


class AccountManager:
    def __init__(self):
        self.accounts_dir = config.account_dir
        os.makedirs(self.accounts_dir, exist_ok=True)

        self.account_list = self.load_accounts()
        self.current_account_index = 0
        self.logged_in_user = None

    def load_accounts(self):
        """掃描 account 資料夾下所有帳號"""
        p = Path("account")
        accounts = [f.name for f in p.iterdir() if f.is_dir()]
        if not accounts:
            accounts = ["Guest"]
        print(f"accounts_list: {accounts}")
        return accounts

    def init_account(self, account_name, RL_interface, model_interface):
        """初始化帳號目錄結構"""
        ac_list = [config.log_dir, config.model_dir, config.data_dir]
        user_path = os.path.join(self.accounts_dir, account_name)

        for i in ac_list:
            os.makedirs(os.path.join(user_path, i), exist_ok=True)

        os.makedirs(os.path.join(user_path, config.model_dir, config.base_model_dir), exist_ok=True)

        # DL 基礎模型
        dl_base_model_path = os.path.join(user_path, config.model_dir, config.base_model_dir, config.base_model_name)
        if not os.path.exists(dl_base_model_path) and os.path.exists(config.base_base_model):
            shutil.copyfile(config.base_base_model, dl_base_model_path)
            print(f"[AccountManager] Copied pretrain DL model {config.base_base_model} to {dl_base_model_path}")

        # RL 檔
        RL_path = os.path.join(user_path, config.bandit_RL)
        if os.path.exists(RL_path):
            RL_interface.bandit.load(RL_path)
        else:
            # 嘗試複製預訓練好的 RL model
            if os.path.exists(config.rl_base_base_model):
                shutil.copyfile(config.rl_base_base_model, RL_path)
                RL_interface.bandit.load(RL_path)
                print(f"[AccountManager, Bandit] Copied pretrain model {config.rl_base_base_model} to {RL_path}")
            else:
                RL_interface.bandit.save(RL_path)
                print(f"[AccountManager, Bandit] Saved empty model to {RL_path}")

        # version.json
        json_path = os.path.join(user_path, config.ver_json)
        default_json = {"train_model": 0, "calib_model": 0}

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            model_interface.version_setting(data["train_model"] + data["calib_model"])
            print(f"version.json exist, current version: {model_interface.version} ")
        else:
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(default_json, f, indent=4, ensure_ascii=False)
                print(f"成功為 {account_name} 創建設定檔")
            except Exception as e:
                print(f"創建失敗: {e}")
                return False

        return True

    def login_as(self, account_name, RL_interface, model_interface):
        """登入指定帳號"""
        self.logged_in_user = account_name
        self.init_account(account_name, RL_interface, model_interface)
        print(f"Logged in as: {account_name}")

    def run_account_login(self, osu_interface, RL_interface, model_interface):
        """帳號選擇畫面"""
        selecting = True
        while selecting:
            osu_interface.clock.tick(60)
            account_rects = osu_interface.view.draw_account_select(
                self.account_list, self.current_account_index)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    osu_interface.view.close()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        self.current_account_index = (self.current_account_index + 1) % len(self.account_list)
                    elif event.key == pygame.K_UP:
                        self.current_account_index = (self.current_account_index - 1) % len(self.account_list)
                    elif event.key == pygame.K_RETURN:
                        user = self.account_list[self.current_account_index]
                        self.login_as(user, RL_interface, model_interface)
                        return "main_menu"

                if event.type == pygame.MOUSEBUTTONDOWN:
                    for i, rect in enumerate(account_rects):
                        if rect.collidepoint(event.pos):
                            self.current_account_index = i
                            user = self.account_list[i]
                            self.login_as(user, RL_interface, model_interface)
                            return "main_menu"


# ─── Debug ───
if __name__ == "__main__":
    print("=== core.account_manager standalone test ===")
    am = AccountManager()
    print(f"accounts: {am.account_list}")
    print("✅ account_manager OK")
