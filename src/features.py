import numpy as np
import librosa


def extract_binaural_features(sig_left, sig_right, n_fft=512):
    # STFT 变换
    S_L = librosa.stft(sig_left, n_fft=n_fft)
    S_R = librosa.stft(sig_right, n_fft=n_fft)

    # 计算声级差
    ild = 20 * np.log10((np.abs(S_L) + 1e-6) / (np.abs(S_R) + 1e-6))

    # 计算基于相位差的简单估计
    phase_diff = np.angle(S_L) - np.angle(S_R)
    itd = phase_diff / (2 * np.pi * 16000 / n_fft)

    return np.stack([np.abs(S_L), np.abs(S_R), ild], axis=0)  # [3, F, T]