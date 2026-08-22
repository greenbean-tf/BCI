"""
combine_all_new 入口點。

使用方式:
    cd combine_all_new
    python main.py
"""
import os
import sys

# 確保根目錄在 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.app_controller import MainController


def main():
    controller = MainController()
    controller.before_start()


if __name__ == "__main__":
    main()
