"""
Standalone config — 從 combine_all_new/config.py 複製的必要常數。
"""

SAMPLE_RATE = 500

CHOOSE_CHANNELS = 13
choose_channels_names = [
    "F3",  "Fz",  "F4",
    "FC3", "FCz", "FC4",
    "C3",  "Cz",  "C4",
    "CP3", "CPz", "CP4",
           "Pz",
]

N_Class = 4
choose_marker = list(range(N_Class))

band_pass_low = 1
band_pass_high = 40
