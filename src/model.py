# --- 修改后的完整 src/model.py ---
import torch
import torch.nn as nn


class SeparationNet(nn.Module):
    def __init__(self):
        super(SeparationNet, self).__init__()
        #频率维度 F = 512 // 2 + 1 = 257
        #输入特征包含 [左耳幅度, 右耳幅度, ILD空间差] 3个通道，input_size = 257 * 3 = 771
        self.gru = nn.GRU(input_size=257 * 3, hidden_size=512, num_layers=2, bidirectional=True, batch_first=True)
        #双向GRU隐层输出 256 * 2 = 512，映射回 257 维的 Mask
        self.mask_layer = nn.Linear(1024, 257)

    def forward(self, x, target_emb=None):
        #接收的 x 形状应为原始双耳信号: [B, 2, Samples]
        #创建汉宁窗
        n_fft = 512
        hop_length = 128
        window = torch.hann_window(n_fft).to(x.device)

        stft_left = torch.stft(x[:, 0, :], n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        stft_right = torch.stft(x[:, 1, :], n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

        #提取幅度谱 [B, F, T]
        mag_left = torch.abs(stft_left)
        mag_right = torch.abs(stft_right)

        #计算至关重要的双耳声级差 特征
        ild = 20 * torch.log10((mag_left + 1e-6) / (mag_right + 1e-6))

        #特征拼接与维度变换
        #将三个物理特征组合: [B, 3, F, T]
        features = torch.stack([mag_left, mag_right, ild], dim=1)
        #调整维度适应时序GRU: [B, T, 3, F]
        features = features.permute(0, 3, 1, 2)

        #将通道和频域展平融合: [B, T, 3 * F] -> [B, T, 771]
        B, T, C, F = features.shape
        x_input = features.reshape(B, T, C * F)
        #依靠空间ILD差和频谱趋势进行分离学习
        out, _ = self.gru(x_input)  # out: [B, T, 512]
        #输出标准时频比值掩码，预测左耳对应的目标 Mask
        mask = torch.sigmoid(self.mask_layer(out))  # [B, T, 257]
        return mask