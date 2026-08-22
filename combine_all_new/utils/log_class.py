"""
自動儲存 log 的工具類。

對應原 log_class.py。
"""


class Tee:
    """將 stdout 同時輸出到多個 file 物件"""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# ─── Debug ───
if __name__ == "__main__":
    import sys
    print("=== log_class standalone test ===")
    original = sys.stdout
    tee = Tee(original)
    sys.stdout = tee
    print("This should appear on stdout via Tee")
    sys.stdout = original
    print("✅ log_class OK")
