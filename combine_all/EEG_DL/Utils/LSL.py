import EEG_DL.Utils.config as config
from pylsl import StreamInfo, StreamOutlet, resolve_byprop, StreamInlet, resolve_streams


def setup_lsl_inlet(stream_name=config.RECEIVE_CYGNUS_LSL_STREAM, timeout=5.0):

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


# def setup_lsl_outlet(stream_name=config.TO_UNITY_LSL_STREAM, stream_type="int32"):
#     # stream_type=int32, string ...
#     # info = StreamInfo(stream_name, "Markers", 1, 0, stream_type, "marker_stream_id")
#     info = StreamInfo(stream_name, "Markers", 1, 0, stream_type, stream_name)
#     outlet = StreamOutlet(info)
#     return outlet
