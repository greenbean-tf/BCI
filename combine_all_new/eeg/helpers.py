"""
工具函式。

對應原 EEG_DL/Utils/some_functions.py。
"""
import os
from datetime import datetime

import config


def rename_file_with_time(name):
    """將檔名附加時間戳"""
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name, file_extension = os.path.splitext(name)
    return f"{file_name}_{formatted_time}{file_extension}"


def get_next_version_path(base_path):
    """
    輸入預設路徑 (例如 .../c_000.pth)，
    如果該檔案存在，自動尋找下一個編號 (c_001.pth, c_002.pth...)
    """
    directory, filename = os.path.split(base_path)
    name, ext = os.path.splitext(filename)

    prefix = "Calibed_Model_v"
    counter = 1

    while True:
        new_filename = f"{prefix}{counter:03d}{ext}"
        candidate_path = os.path.join(directory, new_filename)

        if not os.path.exists(candidate_path):
            return candidate_path

        counter += 1


# ─── Debug ───
if __name__ == "__main__":
    print("=== eeg.helpers standalone test ===")
    test_name = "eeg_record.csv"
    result = rename_file_with_time(test_name)
    print(f"rename_file_with_time('{test_name}') = '{result}'")
    print("✅ helpers OK")
