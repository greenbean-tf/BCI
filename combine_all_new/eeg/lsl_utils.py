"""
LSL (Lab Streaming Layer) 初始化工具。

對應原 EEG_DL/Utils/LSL.py。
"""
import config
from pylsl import StreamInfo, StreamOutlet, resolve_byprop, StreamInlet, resolve_streams


def setup_lsl_inlet(stream_name=config.RECEIVE_CYGNUS_LSL_STREAM, timeout=5.0):
    """解析 LSL stream 並回傳 StreamInlet"""
    print(f"{config.TAGS.INFO.value} Resolving streams...")
    streams = resolve_streams()

    for s in streams:
        try:
            print(f"{config.TAGS.INFO.value} Found stream:", s.name())
        except Exception:
            pass

    streams = resolve_byprop('name', stream_name, timeout=timeout)

    if len(streams) == 0:
        raise RuntimeError(f"{config.TAGS.ERROR.value} Could not resolve LSL stream: {stream_name}")

    inlet = StreamInlet(streams[0], recover=True)
    print(f"{config.TAGS.INFO.value} Connected to LSL stream:", streams[0].name())

    return inlet


# ─── Debug ───
if __name__ == "__main__":
    print("=== eeg.lsl_utils standalone test ===")
    print("This module requires a running LSL stream to test.")
    print("Call setup_lsl_inlet() with a valid stream name.")
    print("✅ lsl_utils import OK")
