# 🍸 Cocktail Party Speech Separation

**基于双耳空间线索的深度学习语音分离**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Deep Learning-Based Binaural Speech Separation Using Spatial Cues

---

## 📖 目录

- [项目概述](#-项目概述)
- [代码结构](#1-代码结构)
- [环境配置](#2-环境配置)
- [数据准备](#3-数据准备)
- [模型训练](#4-模型训练)
- [推理——生成分离音频](#5-推理--生成分离音频)
- [量化评估](#6-量化评估)
- [预期结果参考](#7-预期结果参考)
- [完整复现流程](#8-完整复现流程)
- [常见问题](#9-常见问题)
- [许可证](#license)
- [致谢](#acknowledgements)

---

## 🎯 项目概述

本项目解决**鸡尾酒会问题（Cocktail Party Problem）**：在多人同时说话的多声源混响环境中，利用**双耳（Binaural）空间线索**（如双耳声级差 ILD）从混合音频中分离出**目标说话人**的语音。

### 核心方法

| 环节 | 说明 |
| --- | --- |
| **房间声学模拟** | 使用 `pyroomacoustics` 模拟真实房间声学环境，生成 2~5 人同时说话的双耳混合音频 |
| **分离模型** | 构建基于**双向 GRU** 的时频域掩码预测网络（**SeparationNet**），输入双耳幅度谱 + ILD 空间特征，输出目标说话人的理想比值掩码（IRM） |
| **量化评估** | 采用 **SI-SDR**、**STOI**、**PESQ** 三项指标进行多维度评估 |

---

## 1. 代码结构

```
Cocktail_Party_Project/
├── src/
│   ├── __init__.py             # 包初始化文件
│   ├── model.py                # 模型定义：SeparationNet（双向GRU掩码预测网络）
│   ├── room_sim.py             # 房间声学模拟：pyroomacoustics 多声源混音
│   ├── convert_audio.py        # 音频格式转换：FLAC → WAV (16kHz)
│   ├── run_convert.py          # 格式转换入口脚本
│   └── features.py             # 双耳特征提取工具（ILD 计算等）
├── data/                       # [本地目录，未上传GitHub]
│   ├── LibriSpeech/            # [需自行下载] 原始 LibriSpeech FLAC 语料
│   ├── processed_wav/          # FLAC 转换后的 16kHz WAV 文件
│   ├── train/                  # 训练集（4梯度 × 1000样本 = 4000）
│   ├── val/                    # 验证集（4梯度 × 100样本 = 400）
│   └── test/                   # 测试集（4梯度 × 100样本 = 400）
├── weights/
│   └── separation_net.pth      # 训练好的模型权重
├── figures/                    # 评估结果可视化图表
│   ├── fig1_overall_performance.png
│   ├── fig2_stoi_noise.png
│   ├── fig3_pesq_noise.png
│   └── fig4_delta_sdr_boxplot.png
├── evaluation_report_spk_2.csv # 2说话人场景评估报告
├── evaluation_report_spk_3.csv # 3说话人场景评估报告
├── evaluation_report_spk_4.csv # 4说话人场景评估报告
├── evaluation_report_spk_5.csv # 5说话人场景评估报告
├── main.py                     # 训练主程序
├── prepare_data.py             # 数据集生成流水线
├── inference.py                # 批量推理（生成分离音频）
├── evaluate.py                 # 量化评估（SDR/STOI/PESQ）
├── draw_figures.py             # 可视化脚本
├── requirements.txt            # Python 依赖
├── .gitignore                  # Git 忽略规则
├── LICENSE                     # MIT 开源许可证
└── README.md                   # 本文件
```

---

## 2. 环境配置

### 2.1 创建虚拟环境（推荐）

```bash
conda create -n speech_sep python=3.10
conda activate speech_sep
```

### 2.2 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**：`pesq` 包在某些平台可能需要额外编译工具。若 `pip install pesq` 失败，可尝试：
> ```bash
> conda install -c conda-forge pystoi
> pip install pesq
> ```
> 或参考 [pesq PyPI 页面](https://pypi.org/project/pesq/) 获取平台特定安装说明。

### 2.3 硬件要求

| 组件 | 最低要求 | 推荐配置 |
| --- | --- | --- |
| GPU | 无（支持 CPU 训练） | NVIDIA GPU ≥ 6GB VRAM |
| RAM | 8 GB | 16 GB+ |
| 磁盘 | 约 2 GB（含 LibriSpeech dev-clean 语料） | 10 GB+ |

---

## 3. 数据准备

### 3.1 下载原始语料 —— LibriSpeech

从 [OpenSLR](https://www.openslr.org/12) 下载 LibriSpeech 数据集。本项目当前使用 **`dev-clean`** 子集（约 342 MB，40 位说话人，2,703 条语音），已可满足训练需求。如需更大规模训练，可替换为 `train-clean-100`（约 6.3 GB，251 位说话人）。

```bash
# 下载 dev-clean.tar.gz（约 337 MB）后解压到 data/LibriSpeech/
# 最终目录结构应为 data/LibriSpeech/dev-clean/<speaker_id>/<chapter_id>/*.flac
```

> **说话人数量要求**：最少需要 20 条以上不同说话人的音频片段，否则训练/验证/测试集的独立划分无法保证（参见 `prepare_data.py` 第 81 行的硬性检查）。

### 3.2 音频格式转换：FLAC → WAV

将 LibriSpeech 的 FLAC 文件统一转换为 16kHz 单声道 WAV：

```bash
cd src
python run_convert.py
```

转换结果输出到 `data/processed_wav/`，保持原始 LibriSpeech 的目录层级。

### 3.3 生成模拟数据集

运行数据生成流水线，利用 `pyroomacoustics` 在模拟房间中混合多个说话人，生成训练/验证/测试集：

```bash
python prepare_data.py
```

**生成后的数据目录结构：**

```
data/
├── train/
│   ├── spk_2/               # 2人同时说话场景
│   │   ├── sample_0/
│   │   │   ├── clean_target_spk0.wav      # 目标说话人干音（单声道）
│   │   │   ├── clean_interferer_spk1.wav  # 干扰说话人干音
│   │   │   └── mixed.wav                  # 双耳混合音频（2声道）
│   │   ├── sample_1/
│   │   └── ... （共1000个样本）
│   ├── spk_3/               # 3人同时说话场景
│   ├── spk_4/               # 4人同时说话场景
│   └── spk_5/               # 5人同时说话场景
├── val/                     # 同上结构，每梯度100样本
└── test/                    # 同上结构，每梯度100样本
```

**关键设计决策：**

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| 数据集划分 | Train:Val:Test = 8:1:1 | 在原始说话人级别严格隔离，零数据泄露 |
| 每梯度样本数 | Train=1000, Val=100, Test=100 | 总计4800样本 |
| 随机种子 | 42 | 确保划分可复现 |
| 房间尺寸 | 5m × 5m × 3m | 模拟典型室内会议室 |
| 麦克风间距 | 20 cm | 模拟人耳双耳间距 |
| RT60 | 约 0.2（墙面吸收系数 0.2） | 低混响场景 |
| 采样率 | 16 kHz | 语音分离标准采样率 |
| 音频时长 | 2.0 秒 | 固定长度，便于批处理 |

---

## 4. 模型训练

### 4.1 启动训练

```bash
python main.py
```

### 4.2 训练配置一览

| 超参数 | 取值 | 说明 |
| --- | --- | --- |
| Batch Size | 128 | 大 batch 稳定梯度估计 |
| 初始学习率 | 0.0025 | Adam 优化器 |
| 学习率调度 | ReduceLROnPlateau | factor=0.5, patience=3 |
| 最低学习率阈值 | 1×10⁻⁵ | 跌破即触发早停 |
| Early Stopping | patience=7 | 验证 loss 连续 7 轮不降即停止 |
| 最大 Epoch | 100 | 安全上限 |
| 损失函数 | MSE Loss | 预测掩码 vs 理想比值掩码 |
| 混合精度 | AMP（autocast + GradScaler） | 加速训练、节省显存 |
| STFT 参数 | n_fft=512, hop_length=128 | 频率分辨率 31.25 Hz |

### 4.3 模型架构详情

```
SeparationNet:
┌─────────────────────────────────────────────┐
│ 输入: [B, 2, 32000] 双耳时域波形              │
│   ├─ 左耳 STFT → mag_left  [B, 257, T]      │
│   ├─ 右耳 STFT → mag_right [B, 257, T]      │
│   └─ ILD = 20·log₁₀(L/R)   [B, 257, T]     │
│                                              │
│ 特征拼接: [B, 3, 257, T] → [B, T, 771]      │
│                                              │
│ 双向 GRU（2层, hidden=512）                   │
│   └─ 输出: [B, T, 1024]                      │
│                                              │
│ 全连接层 + Sigmoid                            │
│   └─ 输出: [B, T, 257]  理想比值掩码          │
└─────────────────────────────────────────────┘
参数量: ~6.3M
```

### 4.4 训练监控

训练过程会逐 epoch 打印：

- 训练集平均 Loss
- 各说话人梯度（spk_2 ~ spk_5）的验证 Loss
- 全局验证 Loss（作为模型选择指标）
- 当前学习率

模型权重保存在 `weights/separation_net.pth`。

---

## 5. 推理 —— 生成分离音频

```bash
python inference.py
```

推理脚本会：

1. 加载 `weights/separation_net.pth` 训练权重
2. 遍历 `data/test/` 下所有 `sample_*` 文件夹
3. 读取 `mixed.wav`，模型预测掩码，结合左耳相位通过 iSTFT 重建时域波形
4. 将分离结果写入各样本文件夹下的 `processed.wav`

> 分离音频为**单声道**，16kHz 采样率。

---

## 6. 量化评估

```bash
python evaluate.py
```

### 6.1 评估指标

| 指标 | 全称 | 衡量维度 | 取值范围 |
| --- | --- | --- | --- |
| **SI-SDR** | Scale-Invariant Signal-to-Distortion Ratio | 波形层面分离精度 | 越高越好（dB） |
| **ΔSDR** | SDR Improvement（SDR_sep − SDR_mix） | 分离增益 | 越高越好（dB） |
| **STOI** | Short-Time Objective Intelligibility | 语音可懂度 | 0~1，越高越好 |
| **PESQ (WB)** | Perceptual Evaluation of Speech Quality（Wideband） | 感知语音质量 | -0.5~4.5，越高越好 |

### 6.2 评估输出

按说话人梯度（spk_2 ~ spk_5）分别输出独立 CSV 报告：

- `evaluation_report_spk_2.csv`
- `evaluation_report_spk_3.csv`
- `evaluation_report_spk_4.csv`
- `evaluation_report_spk_5.csv`

每个 CSV 包含：逐样本明细 + 按初始信噪比（Severe/Moderate/Mild）分组的子平均 + 全局总平均。

### 6.3 初始信噪比分层

评估脚本自动按混合信号 SDR 将测试样本分为三档：

| 分层 | SDR 范围 | 含义 |
| --- | --- | --- |
| Severe Noise | SDR ≤ -10 dB | 目标语音深埋于干扰中 |
| Moderate Noise | -10 dB < SDR ≤ -5 dB | 中等干扰强度 |
| Mild Noise | SDR > -5 dB | 目标语音相对清晰 |

---

## 7. 预期结果参考

基于 LibriSpeech dev-clean 语料训练后的典型表现：

| 场景 | Average ΔSDR | Average STOI | Average PESQ |
| --- | --- | --- | --- |
| 2 说话人 | +8 ~ +12 dB | 0.85 ~ 0.92 | 2.5 ~ 3.2 |
| 3 说话人 | +6 ~ +9 dB | 0.78 ~ 0.86 | 2.0 ~ 2.7 |
| 4 说话人 | +4 ~ +7 dB | 0.70 ~ 0.80 | 1.7 ~ 2.3 |
| 5 说话人 | +3 ~ +5 dB | 0.62 ~ 0.73 | 1.4 ~ 2.0 |

> **注**：实际结果受 LibriSpeech 子集大小、训练时长、硬件条件等因素影响，上述为参考区间。

---

## 8. 完整复现流程

```bash
# Step 0: 环境准备
conda create -n speech_sep python=3.10 -y
conda activate speech_sep
pip install -r requirements.txt

# Step 1: 下载 LibriSpeech dev-clean 并解压到 data/LibriSpeech/

# Step 2: FLAC 转 WAV
cd src && python run_convert.py && cd ..

# Step 3: 生成模拟数据集（房间混音）
python prepare_data.py

# Step 4: 训练模型
python main.py

# Step 5: 推理生成分离音频
python inference.py

# Step 6: 量化评估
python evaluate.py
```

---

## 9. 常见问题

**Q: 训练时显存不足？**

- 减小 `main.py` 中的 `TARGET_BATCH_SIZE`（如改为 64 或 32）。

**Q: 如何更换数据集？**

- 将任意 16kHz 单声道 WAV 文件放入 `data/processed_wav/` 目录，保持子目录结构即可。
- 确保音频文件数量 ≥ 20，保证训练/验证/测试集划分有效。

**Q: 为什么用双耳而非单耳？**

- 双耳线索（ILD、ITD）提供了空间信息，有助于区分同位与空间分离的说话人——这对真实的鸡尾酒会场景至关重要。

---

## License

本项目采用 MIT 许可证 —— 详见 [LICENSE](LICENSE) 文件。

---

## 致谢

- **LibriSpeech** 数据集：[OpenSLR](https://www.openslr.org/12)
- **pyroomacoustics**：房间声学模拟工具包
- **PyTorch**：深度学习框架
- 语音分离评估指标由 `pystoi` 和 `pesq` 提供支持

---

<p align="center">
  <sub>Built with ❤️ for speech separation research</sub>
</p>
