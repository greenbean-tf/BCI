"""
向後相容 shim：config 已提升至專案根目錄。

所有新程式碼請使用 `import config`。
此檔案僅為了讓舊的 `import eeg.config` / `from eeg.config import ...` 不壞。
"""
# Re-export everything from root config
from config import *  # noqa: F401, F403
import config as _config
import sys

# 讓 `import eeg.config as config` 拿到的是根目錄的 config module
sys.modules[__name__] = _config
