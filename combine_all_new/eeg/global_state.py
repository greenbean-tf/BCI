"""
向後相容 shim：global_state 已提升至專案根目錄。

所有新程式碼請使用 `import global_state`。
此檔案僅為了讓舊的 `import eeg.global_state` / `from eeg.global_state import ...` 不壞。
"""
# Re-export everything from root global_state
from global_state import *  # noqa: F401, F403
import global_state as _global_state
import sys

# 讓 `import eeg.global_state as global_state` 拿到的是根目錄的 global_state module
sys.modules[__name__] = _global_state
