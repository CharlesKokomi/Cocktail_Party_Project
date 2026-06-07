# 📦 数据集说明

本项目使用 **LibriSpeech** 语料库，通过 **pyroomacoustics** 房间声学模拟生成多说话人混合语音数据集，用于训练和评估双耳语音分离模型。

---

## 1. 数据集概览

| 属性 | 说明 |
|------|------|
| 原始语料 | [LibriSpeech](https://www.openslr.org/12) `dev-clean` 子集 |
| 原始大小 | 约 337 MB（压缩包），解压后约 342 MB |
| 音频格式 | 原始 FLAC → 转换 16kHz 单声道 WAV |
| 合成方法 | `pyroomacoustics` 房间脉冲响应卷积混音 |
| 说话人梯度 | 2 / 3 / 4 / 5 人同时说话 |
| 数据集划分 | Train : Val : Test = 8 : 1 : 1（说话人级别严格隔离） |
| 样本总量 | 4800（Train 4000 + Val 400 + Test 400） |
| 每样本内容 | 目标干音 + 干扰干音 + 双耳混合音频 |

---

## 2. 目录结构

```
data/
├── LibriSpeech/                    # [需自行下载] 原始 FLAC 语料
│   └── dev-clean/
│       ├── BOOKS.TXT
│       ├── CHAPTERS.TXT
│       ├── SPEAKERS.TXT
│       └── <speaker_id>/           # 40 位说话人
│           └── <chapter_id>/
│               ├── *.flac          # 原始音频片段
│               └── *.trans.txt     # 文本转录
│
├── processed_wav/                  # [脚本生成] FLAC → 16kHz WAV
│   └── dev-clean/
│       └── <speaker_id>/
│           └── <chapter_id>/
│               └── *.wav           # 转换后的单声道 WAV
│
├── train/                          # [脚本生成] 训练集
│   ├── spk_2/                      # 2 人场景（1000 样本）
│   ├── spk_3/                      # 3 人场景（1000 样本）
│   ├── spk_4/                      # 4 人场景（1000 样本）
│   └── spk_5/                      # 5 人场景（1000 样本）
│
├── val/                            # [脚本生成] 验证集
│   ├── spk_2/                      # 2 人场景（100 样本）
│   ├── spk_3/                      # 3 人场景（100 样本）
│   ├── spk_4/                      # 4 人场景（100 样本）
│   └── spk_5/                      # 5 人场景（100 样本）
│
└── test/                           # [脚本生成] 测试集
    ├── spk_2/                      # 2 人场景（100 样本）
    ├── spk_3/                      # 3 人场景（100 样本）
    ├── spk_4/                      # 4 人场景（100 样本）
    └── spk_5/                      # 5 人场景（100 样本）
```

### 单个样本目录结构

以 `data/train/spk_3/sample_0/` 为例：

```
sample_0/
├── clean_target_spk0.wav           # 目标说话人干净语音（待分离）
├── clean_interferer_spk1.wav       # 干扰说话人 1 干净语音
├── clean_interferer_spk2.wav       # 干扰说话人 2 干净语音
└── mixed.wav                       # 双耳混合音频（2 声道，16kHz）
```

> **命名约定**：`spk0` 始终作为分离目标（target），其余为干扰源（interferer）。模型训练时以 `clean_target_spk0.wav` 为监督标签。

---

## 3. 数据集下载与准备

### 步骤 1：下载 LibriSpeech 原始语料

从 OpenSLR 下载 `dev-clean` 子集：

```bash
# 下载（约 337 MB）
wget https://www.openslr.org/resources/12/dev-clean.tar.gz

# 解压到项目 data/LibriSpeech/ 目录
tar -xzf dev-clean.tar.gz -C data/LibriSpeech/

# 验证目录结构
ls data/LibriSpeech/dev-clean/
# 应看到：BOOKS.TXT  CHAPTERS.TXT  SPEAKERS.TXT  <speaker_id 文件夹> ...
```

> **备选方案**：如需更大规模训练，可下载 `train-clean-100`（约 6.3 GB，251 位说话人）替换 `dev-clean`。操作步骤完全相同，只需将压缩包解压至同一目录即可。

**LibriSpeech 子集一览：**

| 子集 | 大小 | 说话人数 | 音频数 | 适用场景 |
|------|------|----------|--------|----------|
| `dev-clean` | 337 MB | 40 | 2,703 | 快速实验、验证流程（**当前使用**） |
| `test-clean` | 346 MB | 40 | 2,620 | 额外测试评估 |
| `train-clean-100` | 6.3 GB | 251 | 28,539 | 完整训练、论文复现 |
| `train-clean-360` | 23 GB | 921 | 104,014 | 大规模预训练 |

---

### 步骤 2：音频格式转换（FLAC → WAV）

将 LibriSpeech 的 FLAC 文件统一转换为 **16kHz 单声道 WAV**，以便后续处理：

```bash
cd src
python run_convert.py
cd ..
```

**转换逻辑**（`src/convert_audio.py`）：
- 遍历 `data/LibriSpeech/` 下所有 `.flac` 文件
- 使用 `librosa` 重采样至 16kHz
- 保持原始目录层级，输出至 `data/processed_wav/`

输出示例：
```
data/processed_wav/dev-clean/1272/128104/1272-128104-0000.wav
```

---

### 步骤 3：生成模拟数据集

运行数据生成流水线，利用 `pyroomacoustics` 在虚拟房间中混合多个说话人：

```bash
python prepare_data.py
```

**脚本执行流程：**

```
[扫描] 收集 data/processed_wav/ 下所有 .wav 文件
   ↓
[划分] 按 8:1:1 随机切分为 Train / Val / Test（seed=42）
   ↓
[清理] 删除旧的 data/train/, data/val/, data/test/
   ↓
[生成] 每个子集 × 4 个说话人梯度 × N 个样本
```

**生成过程示意图：**

```
对于每个样本：
  1. 从对应集合随机抽取 N 个不同说话人的音频（N ∈ {2,3,4,5}）
  2. 每段音频截取/填充至 2.0 秒（32000 采样点）
  3. 在 5m×5m×3m 虚拟房间中放置声源和双耳麦克风
  4. pyroomacoustics 计算房间脉冲响应并卷积混音
  5. 保存干音标签（clean_target / clean_interferer）和双耳混合音频（mixed.wav）
```

---

## 4. 声学模拟参数

| 参数 | 取值 | 说明 |
|------|------|------|
| 房间尺寸 | 5m × 5m × 3m | 模拟典型室内会议室 |
| 墙面吸收系数 | 0.2 | RT60 ≈ 0.2s（低混响） |
| 采样率 | 16,000 Hz | 语音分离标准采样率 |
| 音频时长 | 2.0 秒 | 固定长度，便于批处理 |
| 麦克风间距 | 20 cm | 模拟成年人双耳间距 |
| 麦克风坐标 | (2.4, 2.5, 1.5) & (2.6, 2.5, 1.5) | 房间中央偏左/右 |
| 声源放置 | 沿对角线分布 | 声源 i 放置在 (1+i, 1+i, 1.5) |

### 数据集划分策略

| 属性 | 说明 |
|------|------|
| 划分比例 | Train : Val : Test = **8 : 1 : 1** |
| 划分层级 | **原始音频文件级别**（非说话人级别，更严格） |
| 随机种子 | 42（确保可复现） |
| 隔离保证 | 三个集合的原始干音完全不重叠，**零数据泄露** |

### 各子集样本量

| 说话人数 | Train | Val | Test | 合计 |
|----------|-------|-----|------|------|
| spk_2（2人） | 1000 | 100 | 100 | 1200 |
| spk_3（3人） | 1000 | 100 | 100 | 1200 |
| spk_4（4人） | 1000 | 100 | 100 | 1200 |
| spk_5（5人） | 1000 | 100 | 100 | 1200 |
| **合计** | **4000** | **400** | **400** | **4800** |

---

## 5. 生成时间参考

| 步骤 | 耗时（估算） | 说明 |
|------|-------------|------|
| 下载 dev-clean | ~2-5 分钟 | 取决于网络带宽 |
| FLAC → WAV 转换 | ~1-2 分钟 | 2,703 个文件顺序处理 |
| 数据集生成 | ~20-40 分钟 | 4800 次房间声学模拟，CPU 密集型 |

> 生成过程依赖 `pyroomacoustics` 的房间脉冲响应计算，主要消耗 CPU 资源。建议在性能较好的 CPU 上运行，或耐心等待。

---

## 6. 前置条件检查

运行 `prepare_data.py` 前，请确保：

```bash
# ✅ 1. LibriSpeech 已下载并解压
ls data/LibriSpeech/dev-clean/   # 应显示说话人文件夹

# ✅ 2. FLAC 已转换为 WAV
ls data/processed_wav/dev-clean/ # 应显示与上一步对应的 WAV 文件

# ✅ 3. WAV 文件数量 ≥ 20（硬性要求）
find data/processed_wav -name "*.wav" | wc -l
# 输出应 ≥ 20

# ✅ 4. Python 依赖已安装
pip install -r requirements.txt
```

---

## 7. 常见问题

**Q: 能否使用自己的音频数据？**

可以。将任意 16kHz 单声道 WAV 文件放入 `data/processed_wav/` 目录（保持子目录结构），然后运行 `prepare_data.py` 即可。确保音频文件数量 ≥ 20。

**Q: 为什么数据集不包含在 GitHub 仓库中？**

数据集总量约 2~3 GB（含原始 FLAC + 转换 WAV + 生成样本），远超 Git 仓库合理容量。`.gitignore` 已配置排除 `data/` 目录下的所有内容。每位使用者需按本文档步骤本地生成。

**Q: 能否调整说话人数量或样本数？**

可以。修改 `prepare_data.py` 中 `speaker_scenarios` 列表和 `samples_per_scenario` 参数即可。例如：

```python
# 仅生成 2~3 人场景，每梯度 500 样本
speaker_scenarios = [2, 3]
samples_per_scenario = 500
```

**Q: 如何调整房间声学参数？**

修改 `src/room_sim.py` 中的默认参数，或在调用 `generate_mix()` 时传入自定义参数：

```python
# 更大房间 + 更高混响
mixed = generate_mix(signals, mics, room_dim=[8, 8, 4])
# 修改墙面吸收系数
room = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(0.4))  # RT60 更短
```

---

## 8. 数据流全景图

```
OpenSLR 服务器
    │
    │ wget dev-clean.tar.gz (337 MB)
    ▼
data/LibriSpeech/dev-clean/*.flac       ← [手动下载 + 解压]
    │
    │ src/run_convert.py
    │ (librosa 加载 + 重采样 16kHz)
    ▼
data/processed_wav/dev-clean/*.wav      ← [脚本生成 · 约 350 MB]
    │
    │ prepare_data.py
    │ (8:1:1 划分 → pyroomacoustics 房间混音)
    ▼
┌─────────────────────────────────────────────┐
│  data/train/  4000 样本   (4 梯度 × 1000)   │
│  data/val/     400 样本   (4 梯度 ×  100)   │
│  data/test/    400 样本   (4 梯度 ×  100)   │  ← [脚本生成 · 约 1.5 GB]
└─────────────────────────────────────────────┘
    │
    │ main.py / inference.py / evaluate.py
    ▼
模型训练 → 语音分离 → 量化评估
```

---

## 参考链接

- [LibriSpeech 数据集（OpenSLR）](https://www.openslr.org/12)
- [pyroomacoustics 文档](https://pyroomacoustics.readthedocs.io/)
- [LibriSpeech 原始论文](https://arxiv.org/abs/1503.01516)
