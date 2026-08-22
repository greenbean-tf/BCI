import EEG_DL.Utils.config as config
import csv
import threading
import numpy as np
from datetime import datetime
import os


def rename_file_with_time(name):
    # 格式化時間為 'YYYYMMDD_HHMMSS'
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 取得檔案名稱
    file_name, file_extension = os.path.splitext(name)
    
    # 生成新的檔案名稱
    return f"{file_name}_{formatted_time}{file_extension}"


def get_next_version_path(base_path):
    """
    輸入預設路徑 (例如 .../c_000.pth)，
    如果該檔案存在，自動尋找下一個編號 (c_001.pth, c_002.pth...)
    :return 可以的路徑名稱
    """
    # 1. 分離目錄與檔名
    directory, filename = os.path.split(base_path)

    # 2. 分離檔名與副檔名 (例如 name='Calibed_Model_v000', ext='.pth')
    name, ext = os.path.splitext(filename)

    # 設定前綴與初始計數
    prefix = "Calibed_Model_v"
    counter = 1

    while True:
        
        # 3. 組合新檔名，使用 :03d 確保是三位數補零 (例如 0 -> 000, 1 -> 001)
        new_filename = f"{prefix}{counter:03d}{ext}"
        candidate_path = os.path.join(directory, new_filename)

        # 4. 檢查檔案是否存在
        if not os.path.exists(candidate_path):
            # 如果不存在，表示這個檔名可用，回傳此路徑
            return candidate_path

        # 如果存在，計數器 +1，繼續迴圈
        counter += 1