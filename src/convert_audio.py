import os
import librosa
import soundfile as sf
from pathlib import Path


def convert_flac_to_wav(root_dir, target_dir):
    # 遍历所有 flac 文件
    for path in Path(root_dir).rglob("*.flac"):
        #加载音频，强制重采样为 16kHz
        audio, sr = librosa.load(path, sr=16000)
        #构建目标路径
        relative_path = path.relative_to(root_dir)
        new_path = Path(target_dir) / relative_path.with_suffix(".wav")
        #创建对应的子文件夹
        os.makedirs(new_path.parent, exist_ok=True)
        # 保存为 wav
        sf.write(new_path, audio, sr)
        print(f"已转换: {path.name} -> {new_path.name}")
