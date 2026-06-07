import pyroomacoustics as pra
import numpy as np


def generate_mix(speaker_signals, mic_locs, room_dim=[5, 5, 3]):
    #初始化房间
    room = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(0.2))

    #添加声源
    for i, sig in enumerate(speaker_signals):
        #简单设定位置，实际可根据需要调整
        pos = [1 + i, 1 + i, 1.5]
        room.add_source(pos, signal=sig)

    # 添加麦克风
    room.add_microphone_array(pra.MicrophoneArray(mic_locs, room.fs))
    room.simulate()
    return room.mic_array.signals  # 返回混合后的双耳信号