import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
from tqdm import tqdm
from src.model import SeparationNet


#内存映射Dataset
class MultiGradientDataset(Dataset):
    def __init__(self, data_type_dir, target_spk_group=None):
        self.data_type_dir = data_type_dir
        #根据是否指定说话人梯度分组来动态构建搜索路径
        if target_spk_group is not None:
            search_path = os.path.join(data_type_dir, target_spk_group, "sample_*")
            desc = f"预载入验证集 [{target_spk_group}]"
        else:
            search_path = os.path.join(data_type_dir, "spk_*", "sample_*")
            desc = f"预载入全量训练集"

        #获取所有样本文件夹的路径列表并排序
        self.sample_folders = sorted(glob.glob(search_path))

        # 定义内存缓冲区，用于存放预处理后的Tensor数据
        self.mixed_buffers = []
        self.target_buffers = []

        #在 __init__ 阶段进行数据啃食，加载所有音频至 RAM
        for folder in tqdm(self.sample_folders, desc=desc):
            mixed_path = os.path.join(folder, "mixed.wav")
            target_path = os.path.join(folder, "clean_target_spk0.wav")

            #加载并对混合音进行归一化，防止溢出或数值不稳
            mixed_sig, _ = librosa.load(mixed_path, sr=16000, mono=False, duration=2.0)
            if mixed_sig.max() > 0:
                mixed_sig = mixed_sig / (np.max(np.abs(mixed_sig)) + 1e-6)
            #统一采样点数（32000点 = 2秒），保证Tensor维度在Batch处理时完全一致
            mixed_sig = librosa.util.fix_length(mixed_sig, size=32000)

            #加载目标干音并统一长度
            target_sig, _ = librosa.load(target_path, sr=16000, mono=True, duration=2.0)
            target_sig = librosa.util.fix_length(target_sig, size=32000)

            #转换为PyTorch Tensor并存入内存列表
            self.mixed_buffers.append(torch.FloatTensor(mixed_sig))
            self.target_buffers.append(torch.FloatTensor(target_sig))

    def __len__(self):
        #返回总样本数
        return len(self.sample_folders)

    def __getitem__(self, idx):
        #索引提取：直接从内存缓冲区返回
        return self.mixed_buffers[idx], self.target_buffers[idx]


#掩码计算函数将波形转换为时频域的目标特征矩阵
def get_target_mask(target_signal, mixed_input, n_fft=512, hop_length=128):
    #准备汉宁窗，防止截断效应
    window = torch.hann_window(n_fft).to(target_signal.device)
    #计算目标干音的幅度谱
    #return_complex=True输出复数STFT，torch.abs()获取幅度信息，permute调整维度为(Batch, Time, Freq)
    target_stft = torch.stft(target_signal, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    target_mag = torch.abs(target_stft).permute(0, 2, 1)

    #计算混合音频的幅度谱，仅左声道
    mixed_left = mixed_input[:, 0, :]
    mixed_stft = torch.stft(mixed_left, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    mixed_mag = torch.abs(mixed_stft).permute(0, 2, 1)

    #对齐长度
    min_t = min(target_mag.shape[1], mixed_mag.shape[1])
    target_mag = target_mag[:, :min_t, :]
    mixed_mag = mixed_mag[:, :min_t, :]

    # 通过将目标幅度除以混合幅度得到比值，代表每个时频点下目标音所占的比例
    target_mask = target_mag / (mixed_mag + 1e-6)

    #将掩码值限制在 0 到 1 之间
    target_mask = torch.clamp(target_mask, 0.0, 1.0)

    return target_mask


if __name__ == '__main__':
    train_root_dir = os.path.join("data", "train")
    val_root_dir = os.path.join("data", "val")

    print("==================================================================")
    print("开启训练")
    print("==================================================================")

    train_dataset = MultiGradientDataset(train_root_dir)

    TARGET_BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=TARGET_BATCH_SIZE, shuffle=True, num_workers=0)

    val_scenarios = ["spk_2", "spk_3", "spk_4", "spk_5"]
    val_loaders_dict = {}
    for spk_g in val_scenarios:
        val_sub_dataset = MultiGradientDataset(val_root_dir, target_spk_group=spk_g)
        #验证集不需要计算梯度，也可以用大Batch Size
        val_loaders_dict[spk_g] = DataLoader(val_sub_dataset, batch_size=TARGET_BATCH_SIZE, shuffle=False,
                                             num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeparationNet().to(device)
    criterion = nn.MSELoss()

    #初始学习率0.0025
    optimizer = optim.Adam(model.parameters(), lr=0.0025)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    #初始化AMP梯度缩放器，防御半精度下可能出现的梯度下溢
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs("weights", exist_ok=True)
    weight_output_path = os.path.join("weights", "separation_net.pth")

    best_val_loss = float('inf')
    patience_early_stop = 7
    early_stop_counter = 0
    min_lr_threshold = 1e-5  # 伴随初始学习率调大，冰点阈值调整为 1e-5

    print(f"\nBatch Size: {TARGET_BATCH_SIZE} | 初始 LR: 0.0025")
    print(f"每个Epoch步数为: {len(train_loader)} 步")

    num_epochs = 100
    for epoch in range(num_epochs):

        # -------------------训练阶段-------------------
        model.train()
        epoch_train_loss = 0.0

        for i, (mixed_input, target_signal) in enumerate(train_loader):
            mixed_input = mixed_input.to(device)
            target_signal = target_signal.to(device)

            optimizer.zero_grad()

            #用autocast开启半精度自动转换上下文
            with torch.amp.autocast('cuda'):
                target_mask = get_target_mask(target_signal, mixed_input)
                predicted_mask = model(mixed_input, None)

                min_t = min(predicted_mask.shape[1], target_mask.shape[1])
                predicted_mask = predicted_mask[:, :min_t, :]
                target_mask = target_mask[:, :min_t, :]

                loss = criterion(predicted_mask, target_mask)

            #使用Scaler进行反向传播与优化器更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)

        # -------------------验证阶段-------------------
        model.eval()
        val_losses_snapshot = {}
        total_val_loss = 0.0

        #设置梯度计算上下文为关闭，这是验证阶段的标配
        with torch.no_grad():
            #遍历验证集中不同的人数梯度
            for spk_g, val_loader in val_loaders_dict.items():
                sub_val_loss = 0.0
                #处理该梯度下的所有验证样本
                for mixed_input, target_signal in val_loader:
                    mixed_input = mixed_input.to(device)
                    target_signal = target_signal.to(device)

                    #启用混合精度上下文
                    with torch.amp.autocast('cuda'):
                        # 获取真实掩码作为对比标准
                        target_mask = get_target_mask(target_signal, mixed_input)
                        # 模型预测掩码
                        predicted_mask = model(mixed_input, None)

                        #预测掩码与真实掩码在时间轴上长度一致，以便计算损失
                        min_t = min(predicted_mask.shape[1], target_mask.shape[1])
                        #计算MSE Loss
                        loss = criterion(predicted_mask[:, :min_t, :], target_mask[:, :min_t, :])

                    sub_val_loss += loss.item()

                #计算该人数梯度的平均验证 Loss
                avg_sub_val_loss = sub_val_loss / len(val_loader)
                val_losses_snapshot[spk_g] = avg_sub_val_loss
                #累加到总验证损失中
                total_val_loss += avg_sub_val_loss

        #计算全局平均验证 Loss
        #该值综合了所有声学环境，是衡量模型泛化能力的最终金标准
        avg_global_val_loss = total_val_loss / len(val_scenarios)

        # 新学习率调度器，根据当前的全局验证Loss情况，决定是否需要降低学习率
        #如果验证损失进入平原区，调度器会自动衰减LR，引导模型在更小的步长下收敛
        scheduler.step(avg_global_val_loss)

        # -------------------总结日志打印-------------------
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n [Epoch {epoch + 1}/{num_epochs}] 核心成效总结汇报:")
        print(f"  ├── 训练集(Train) 平均 Loss: {avg_train_loss:.4f}")
        print(f"  ├── 全局验证(Val)   平均 Loss: {avg_global_val_loss:.4f}")
        print(f"  │     ├── spk_2 局部 Loss: {val_losses_snapshot['spk_2']:.4f}")
        print(f"  │     ├── spk_3 局部 Loss: {val_losses_snapshot['spk_3']:.4f}")
        print(f"  │     ├── spk_4 局部 Loss: {val_losses_snapshot['spk_4']:.4f}")
        print(f"  │     └── spk_5 局部 Loss: {val_losses_snapshot['spk_5']:.4f}")
        print(f"  └── 当前自适应学习率 LR: {current_lr}")

        if avg_global_val_loss < (best_val_loss - 1e-5):
            best_val_loss = avg_global_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), weight_output_path)
            print(f" [权重更新] 锁定当前 Best 权重。")
        else:
            early_stop_counter += 1
            print(f" [遭遇瓶颈] 连续 {early_stop_counter}/{patience_early_stop} 轮未破纪录。")

        print("=" * 66 + "\n")

        if early_stop_counter >= patience_early_stop:
            print(f"【自动掐断】触发早停机制，平稳退出。")
            break

        if current_lr < min_lr_threshold:
            print(f"【自动掐断】学习率 ({current_lr}) 跌破冰点阈值，安全退出。")
            break

    print(f"泛化权重已安全保存在: {weight_output_path}")