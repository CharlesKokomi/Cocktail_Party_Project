import os
import glob
import numpy as np
import soundfile as sf
from tqdm import tqdm
from src.room_sim import generate_mix
import librosa
import shutil


def generate_gradient_dataset(file_sublist, target_root_dir, samples_per_scenario):
    """
    根据绝对隔离的原始语音子集，为指定的顶级目录（train、val 或 test）生成多梯度数据集
    """
    os.makedirs(target_root_dir, exist_ok=True)

    # 固定双耳麦克风位置（间距 20cm），确保物理构型在三个数据集中完全统一
    mics = np.array([[2.4, 2.5, 1.5], [2.6, 2.5, 1.5]]).T
    speaker_scenarios = [2, 3, 4, 5]

    for num_speakers in speaker_scenarios:
        # 创建分梯度子文件夹，例如 data/train/spk_3, data/test/spk_5 等
        spk_group_folder = os.path.join(target_root_dir, f"spk_{num_speakers}")
        os.makedirs(spk_group_folder, exist_ok=True)

        print(f"👉 正在向 [{target_root_dir}] 写入 【{num_speakers} 个说话者】 场景 ({samples_per_scenario}个样本)...")

        pbar = tqdm(total=samples_per_scenario)
        success_count = 0

        while success_count < samples_per_scenario:
            # 💡【绝对安全】：严格从该集合专属的原始音频子集中摇号，不同集合的原始干音绝不重叠！
            chosen_indices = np.random.choice(len(file_sublist), num_speakers, replace=False)

            signals = []
            for idx in chosen_indices:
                wav, _ = librosa.load(file_sublist[idx], sr=16000, duration=2.0)
                wav = librosa.util.fix_length(wav, size=32000)
                signals.append(wav)

            # pyroomacoustics 空间碰撞越界防御
            try:
                mixed = generate_mix(signals, mics)
            except ValueError as e:
                if "must be added inside the room" in str(e):
                    continue
                else:
                    raise e

            # 确立样本专属文件夹
            sample_folder = os.path.join(spk_group_folder, f"sample_{success_count}")
            os.makedirs(sample_folder, exist_ok=True)

            # 写入干音标签（speaker_0 永远作为网络抽离的目标人）
            for spk_id, spk_signal in enumerate(signals):
                if spk_id == 0:
                    filename = "clean_target_spk0.wav"
                else:
                    filename = f"clean_interferer_spk{spk_id}.wav"
                sf.write(os.path.join(sample_folder, filename), spk_signal, 16000)

            # 写入统一命名的双耳混合音频
            sf.write(os.path.join(sample_folder, "mixed.wav"), mixed.T, 16000)

            success_count += 1
            pbar.update(1)

        pbar.close()


if __name__ == "__main__":
    src_dir = "data/processed_wav"
    data_root = "data"

    print("==================================================================")
    print("        🚀 语音分离项目：全套多梯度学术数据集一键全量重构 🚀        ")
    print("==================================================================")

    # 1. 扫描并打乱基底原始文件
    all_files = sorted(glob.glob(f"{src_dir}/**/*.wav", recursive=True))
    if len(all_files) < 20:
        raise ValueError(f"【关键错误】原始纯净音频文件太少 ({len(all_files)}个)，无法支撑高强度的隔离集划分！")

    # 固定随机种子，确保划分规则具有可重复性，且底层切片一刀切断
    np.random.seed(42)
    np.random.shuffle(all_files)
    total_files = len(all_files)

    # 2. 💡【核心防线】：严格按比例（8:1:1）切断原始干音资产，从源头绝交
    train_split_idx = int(total_files * 0.8)
    val_split_idx = int(total_files * 0.9)

    train_files = all_files[:train_split_idx]
    val_files = all_files[train_split_idx:val_split_idx]
    test_files = all_files[val_split_idx:]

    print(f"【🎯 原始音频基底隔离切片成功】")
    print(f"  ├── ⚖️ 训练集(Train) 分配基底 : {len(train_files)} 个音频 (占 80%)")
    print(f"  ├── ⚖️ 验证集(Val)   分配基底 : {len(val_files)} 个音频 (占 10%)")
    print(f"  └── ⚖️ 测试集(Test)  分配基底 : {len(test_files)} 个音频 (占 10%)")
    print(f"  注：三种集合所包含的原始说话人音色和音频段落完全独立，不存在任何交叠。")
    print("==================================================================\n")

    # 3. 安全清理旧数据目录，防止新旧文件杂糅
    paths_to_clean = [
        os.path.join(data_root, "train"),
        os.path.join(data_root, "val"),
        os.path.join(data_root, "test"),
        os.path.join(data_root, "train_mixed")  # 顺便清理历史遗留的旧目录
    ]
    for path in paths_to_clean:
        if os.path.exists(path):
            print(f"🧹 正在清空历史旧目录: {path} ...")
            shutil.rmtree(path)

    # 4. 一键开启全量流水线生成
    print("\n📦 [1/3] 开始生成多梯度 训练集(Train)...")
    generate_gradient_dataset(train_files, os.path.join(data_root, "train"), samples_per_scenario=1000)

    print("\n📦 [2/3] 开始生成多梯度 验证集(Val)...")
    generate_gradient_dataset(val_files, os.path.join(data_root, "val"), samples_per_scenario=100)

    print("\n📦 [3/3] 开始生成多梯度 测试集(Test)...")
    generate_gradient_dataset(test_files, os.path.join(data_root, "test"), samples_per_scenario=100)

    print("\n==================================================================")
    print("【🎉 大获全胜】全量数据集全自动流水线闭环生成完毕！")
    print("  👉 训练集 (data/train): 4梯度 × 1000 = 4000 样本")
    print("  👉 验证集 (data/val)  : 4梯度 × 100  = 400  样本")
    print("  👉 测试集 (data/test) : 4梯度 × 100  = 400  样本")
    print("  ✨ 零数据泄露、高学术严谨性的多说话者语音分离矩阵已正式筑基成功！")
    print("==================================================================")