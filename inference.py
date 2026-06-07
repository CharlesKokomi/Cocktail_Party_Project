import os
import glob
import torch
import librosa
import soundfile as sf
import numpy as np
from src.model import SeparationNet


def run_inference(sample_folder, model, device):
    """
    sample_folder: 单个样本的专属文件夹，如 "data/test/spk_2/sample_0"
    """
    #提取说话者梯度和样本文件夹名字
    parts = sample_folder.replace("\\", "/").split("/")
    display_name = f"{parts[-2]}/{parts[-1]}" if len(parts) >= 2 else os.path.basename(sample_folder)

    mixed_audio_path = os.path.join(sample_folder, "mixed.wav")
    out_sep_path = os.path.join(sample_folder, "processed.wav")

    if not os.path.exists(mixed_audio_path):
        print(f"略过 {display_name}: 未找到输入源文件mixed.wav")
        return

    #读取待分离的双耳音频 (2.0秒)
    mixed_sig, sr = librosa.load(mixed_audio_path, sr=16000, mono=False, duration=2.0)

    #对时域波形做能量归一化
    if mixed_sig.max() > 0:
        mixed_sig = mixed_sig / (np.max(np.abs(mixed_sig)) + 1e-6)

    mixed_sig = librosa.util.fix_length(mixed_sig, size=32000)

    #转换为Tensor并增加Batch维度->[1, 2, 32000]
    mixed_tensor = torch.FloatTensor(mixed_sig).unsqueeze(0).to(device)

    #提取左耳的原始相位和幅度用于最终的声音恢复
    n_fft = 512
    window = torch.hann_window(n_fft).to(device)

    mixed_left = mixed_tensor[:, 0, :]  #提取左耳
    stft_mixed = torch.stft(mixed_left, n_fft=n_fft, window=window, return_complex=True)
    mag_left = torch.abs(stft_mixed)
    phase_left = torch.angle(stft_mixed)

    #模型前向传播预测 Mask
    with torch.no_grad():
        predicted_mask = model(mixed_tensor, None)  # 输出为 [1, T, F]
        predicted_mask = predicted_mask.permute(0, 2, 1)  # 变换为 [1, F, T]

    #维度严格对齐
    min_t = min(mag_left.shape[2], predicted_mask.shape[2])
    mag_left = mag_left[:, :, :min_t]
    phase_left = phase_left[:, :, :min_t]
    predicted_mask = predicted_mask[:, :, :min_t]

    #掩码相乘过滤左耳信号幅值
    sep_mag = mag_left * predicted_mask

    #结合左耳原始相位，恢复复数 STFT 信号
    sep_stft = torch.polar(sep_mag, phase_left)

    #iSTFT 变换回时域波形
    sep_audio = torch.istft(sep_stft, n_fft=n_fft, window=window)
    sep_audio_np = sep_audio.squeeze(0).cpu().numpy()

    #保存分离后的第四轨音频到当前文件夹下
    sf.write(out_sep_path, sep_audio_np, sr)


if __name__ == "__main__":
    data_mixed_dir = os.path.join("data", "test")
    model_weight_path = "weights/separation_net.pth"

    print("================ 语音分离批量推理程序启动 ================")

    #递归检索二级子目录下的所有文件夹，路径：data/test/spk_*/sample_*
    search_path = os.path.join(data_mixed_dir, "spk_*", "sample_*")
    all_sample_folders = glob.glob(search_path)

    # 自适应复杂路径的科学排序函数
    def sort_key(path_str):
        normalized_path = path_str.replace("\\", "/")
        parts = normalized_path.split("/")
        try:
            # 提取 spk_X 的数字
            spk_num = int(parts[-2].split("_")[-1])
            # 提取 sample_Y 的数字
            sample_num = int(parts[-1].split("_")[-1])
            return (spk_num, sample_num)  # 优先按人数排，人数相同按样本序号排
        except (IndexError, ValueError):
            return (0, 0)

    selected_folders = sorted(all_sample_folders, key=sort_key)

    if not selected_folders:
        print(f"【错误】未检测到任何样本，请确认是否存在 spk_X/sample_Y 结构:\n{data_mixed_dir}")
    else:
        #模型初始化移到外部，防止显存随循环累积泄露
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SeparationNet().to(device)
        model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
        model.eval()

        total_samples = len(selected_folders)
        print(f"【提示】共检测到 {total_samples} 个样本，已选择全部样本进行全量推理。")
        print(f"【运行设备】{device} | 正在初始化流水线...")

        #遍历推理
        for idx, folder_path in enumerate(selected_folders):
            #获取当前处理文件的相对显示名称
            parts = folder_path.replace("\\", "/").split("/")
            display_name = f"{parts[-2]}/{parts[-1]}"

            #每隔20个样本或者换梯度时打印一次进度
            if (idx + 1) % 20 == 0 or (idx + 1) == total_samples or (idx + 1) <= 3:
                print(f"推理总进度: [{idx + 1}/{total_samples}] | 正在处理: {display_name}")

            run_inference(folder_path, model, device)

        print("\n======================================================")
        print(f"【推理完毕】共 {total_samples} 个样本的结果'processed.wav' 已成功合流！")
        print("======================================================")