# 🍸 Cocktail Party Speech Separation

**Deep Learning-Based Binaural Speech Separation Using Spatial Cues**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 鸡尾酒会问题语音分离 —— 基于双耳空间线索的深度学习方案

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Code Structure](#1-code-structure)
- [Environment Setup](#2-environment-setup)
- [Data Preparation](#3-data-preparation)
- [Model Training](#4-model-training)
- [Inference](#5-inference--generate-separated-audio)
- [Evaluation](#6-quantitative-evaluation)
- [Expected Results](#7-expected-results)
- [Quick Start](#8-quick-start)
- [FAQ](#9-faq)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 🎯 Project Overview

This project tackles the **Cocktail Party Problem**: separating a target speaker's voice from multi-talker reverberant mixtures by leveraging **binaural spatial cues** (e.g., Interaural Level Difference, ILD).

### Core Approach

| Component | Description |
| --- | --- |
| **Room Simulation** | `pyroomacoustics` simulates realistic room acoustics with 2–5 concurrent speakers, generating binaural mixtures |
| **Separation Model** | Bidirectional GRU-based time-frequency mask prediction network (**SeparationNet**), taking binaural magnitude spectra + ILD spatial features, outputting the Ideal Ratio Mask (IRM) |
| **Evaluation** | Multi-dimensional assessment via **SI-SDR**, **STOI**, and **PESQ** |

---

## 1. Code Structure

```
Cocktail_Party_Project/
├── src/
│   ├── model.py              # Model definition: SeparationNet (Bi-GRU mask predictor)
│   ├── room_sim.py           # Room acoustics simulation (pyroomacoustics multi-source mixing)
│   ├── convert_audio.py      # Audio format conversion: FLAC → WAV (16kHz)
│   ├── run_convert.py        # Format conversion entry point
│   └── features.py           # Binaural feature extraction (ILD computation, etc.)
├── data/
│   ├── LibriSpeech/          # [Download required] Original LibriSpeech FLAC corpus
│   ├── processed_wav/        # FLAC-converted 16kHz WAV files
│   ├── train/                # Training set (4 gradients × 1000 samples = 4000)
│   ├── val/                  # Validation set (4 gradients × 100 samples = 400)
│   └── test/                 # Test set (4 gradients × 100 samples = 400)
├── weights/
│   └── separation_net.pth    # Trained model weights
├── figures/                  # Evaluation result visualizations
├── main.py                   # Training entry point
├── prepare_data.py           # Dataset generation pipeline
├── inference.py              # Batch inference (generates separated audio)
├── evaluate.py               # Quantitative evaluation (SDR/STOI/PESQ)
├── draw_figures.py           # Visualization script
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## 2. Environment Setup

### 2.1 Create Virtual Environment (Recommended)

```bash
conda create -n speech_sep python=3.10
conda activate speech_sep
```

### 2.2 Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: The `pesq` package may require additional build tools on some platforms. If `pip install pesq` fails, try:
> ```bash
> conda install -c conda-forge pystoi
> pip install pesq
> ```
> Or refer to the [pesq PyPI page](https://pypi.org/project/pesq/) for platform-specific instructions.

### 2.3 Hardware Requirements

| Component | Minimum | Recommended |
| --- | --- | --- |
| GPU | None (CPU training supported) | NVIDIA GPU ≥ 6GB VRAM |
| RAM | 8 GB | 16 GB+ |
| Disk | ~2 GB (with LibriSpeech dev-clean) | 10 GB+ |

---

## 3. Data Preparation

### 3.1 Download Raw Corpus — LibriSpeech

Download LibriSpeech from [OpenSLR](https://www.openslr.org/12). This project uses the **`dev-clean`** subset (~342 MB, 40 speakers, 2,703 utterances), which is sufficient for training. For larger-scale training, replace with `train-clean-100` (~6.3 GB, 251 speakers).

```bash
# Download dev-clean.tar.gz (~337 MB) and extract to data/LibriSpeech/
# Final directory structure: data/LibriSpeech/dev-clean/<speaker_id>/<chapter_id>/*.flac
```

> **Speaker Count Requirement**: At least 20 distinct speaker audio clips are required; otherwise, the train/val/test split independence cannot be guaranteed (see `prepare_data.py` line 81 for the hard check).

### 3.2 Audio Format Conversion: FLAC → WAV

Convert LibriSpeech FLAC files to 16kHz mono WAV:

```bash
cd src
python run_convert.py
```

Output is written to `data/processed_wav/`, preserving the original LibriSpeech directory hierarchy.

### 3.3 Generate Simulated Dataset

Run the data generation pipeline to mix multiple speakers in a simulated room:

```bash
python prepare_data.py
```

**Generated Directory Structure:**

```
data/
├── train/
│   ├── spk_2/               # 2-speaker scenario
│   │   ├── sample_0/
│   │   │   ├── clean_target_spk0.wav      # Target speaker dry signal (mono)
│   │   │   ├── clean_interferer_spk1.wav  # Interferer dry signal
│   │   │   └── mixed.wav                  # Binaural mixture (2-channel)
│   │   ├── sample_1/
│   │   └── ... (1000 samples total)
│   ├── spk_3/               # 3-speaker scenario
│   ├── spk_4/               # 4-speaker scenario
│   └── spk_5/               # 5-speaker scenario
├── val/                     # Same structure, 100 samples per gradient
└── test/                    # Same structure, 100 samples per gradient
```

**Key Design Decisions:**

| Parameter | Value | Notes |
| --- | --- | --- |
| Dataset Split | Train:Val:Test = 8:1:1 | Strict speaker-level isolation, zero leakage |
| Samples per Gradient | Train=1000, Val=100, Test=100 | 4800 samples total |
| Random Seed | 42 | Ensures reproducible splits |
| Room Dimensions | 5m × 5m × 3m | Typical indoor meeting room |
| Mic Spacing | 20 cm | Simulates binaural ear spacing |
| RT60 | ~0.2 (wall absorption 0.2) | Low-reverberation scenario |
| Sampling Rate | 16 kHz | Standard for speech separation |
| Audio Duration | 2.0 sec | Fixed length for batch processing |

---

## 4. Model Training

### 4.1 Start Training

```bash
python main.py
```

### 4.2 Training Configuration

| Hyperparameter | Value | Notes |
| --- | --- | --- |
| Batch Size | 128 | Large batch for stable gradient estimation |
| Initial LR | 0.0025 | Adam optimizer |
| LR Schedule | ReduceLROnPlateau | factor=0.5, patience=3 |
| LR Threshold | 1×10⁻⁵ | Triggers early stopping when reached |
| Early Stopping | patience=7 | Stops if val loss doesn't improve for 7 epochs |
| Max Epochs | 100 | Safety upper bound |
| Loss Function | MSE Loss | Predicted mask vs. Ideal Ratio Mask |
| Mixed Precision | AMP (autocast + GradScaler) | Faster training, reduced memory |
| STFT Params | n_fft=512, hop_length=128 | Freq resolution 31.25 Hz |

### 4.3 Model Architecture

```
SeparationNet:
┌─────────────────────────────────────────────────────┐
│ Input: [B, 2, 32000] binaural time-domain waveform  │
│   ├─ Left-ear STFT  → mag_left   [B, 257, T]       │
│   ├─ Right-ear STFT → mag_right  [B, 257, T]       │
│   └─ ILD = 20·log₁₀(L/R)        [B, 257, T]       │
│                                                      │
│ Feature Concat: [B, 3, 257, T] → [B, T, 771]       │
│                                                      │
│ Bidirectional GRU (2 layers, hidden=512)             │
│   └─ Output: [B, T, 1024]                           │
│                                                      │
│ FC + Sigmoid                                         │
│   └─ Output: [B, T, 257]  Ideal Ratio Mask          │
└─────────────────────────────────────────────────────┘
Parameters: ~6.3M
```

### 4.4 Training Monitoring

Each epoch prints:
- Average training loss
- Per-gradient validation loss (spk_2 ~ spk_5)
- Global validation loss (used for model selection)
- Current learning rate

Model weights are saved to `weights/separation_net.pth`.

---

## 5. Inference — Generate Separated Audio

```bash
python inference.py
```

The inference script:
1. Loads trained weights from `weights/separation_net.pth`
2. Iterates over all `sample_*` directories under `data/test/`
3. Reads `mixed.wav`, predicts the mask, reconstructs the time-domain waveform via iSTFT with left-ear phase
4. Writes the separated result as `processed.wav` in each sample directory

> Separated audio is **mono**, 16kHz sampling rate.

---

## 6. Quantitative Evaluation

```bash
python evaluate.py
```

### 6.1 Evaluation Metrics

| Metric | Full Name | Dimension | Range |
| --- | --- | --- | --- |
| **SI-SDR** | Scale-Invariant Signal-to-Distortion Ratio | Waveform-level separation accuracy | Higher is better (dB) |
| **ΔSDR** | SDR Improvement (SDR_sep − SDR_mix) | Separation gain | Higher is better (dB) |
| **STOI** | Short-Time Objective Intelligibility | Speech intelligibility | 0–1, higher is better |
| **PESQ (WB)** | Perceptual Evaluation of Speech Quality (Wideband) | Perceived speech quality | -0.5–4.5, higher is better |

### 6.2 Evaluation Output

Per-gradient CSV reports are generated:

- `evaluation_report_spk_2.csv`
- `evaluation_report_spk_3.csv`
- `evaluation_report_spk_4.csv`
- `evaluation_report_spk_5.csv`

Each CSV contains: per-sample details + sub-averages grouped by initial SNR level + global average.

### 6.3 Initial SNR Stratification

Test samples are automatically categorized into three tiers by mixed-signal SDR:

| Tier | SDR Range | Meaning |
| --- | --- | --- |
| Severe Noise | SDR ≤ -10 dB | Target speech deeply buried in interference |
| Moderate Noise | -10 dB < SDR ≤ -5 dB | Moderate interference |
| Mild Noise | SDR > -5 dB | Target speech relatively clear |

---

## 7. Expected Results

Typical performance after training on LibriSpeech dev-clean:

| Scenario | Average ΔSDR | Average STOI | Average PESQ |
| --- | --- | --- | --- |
| 2 Speakers | +8 ~ +12 dB | 0.85 ~ 0.92 | 2.5 ~ 3.2 |
| 3 Speakers | +6 ~ +9 dB | 0.78 ~ 0.86 | 2.0 ~ 2.7 |
| 4 Speakers | +4 ~ +7 dB | 0.70 ~ 0.80 | 1.7 ~ 2.3 |
| 5 Speakers | +3 ~ +5 dB | 0.62 ~ 0.73 | 1.4 ~ 2.0 |

> **Note**: Results vary based on LibriSpeech subset size, training duration, and hardware. The above ranges are for reference.

---

## 8. Quick Start

```bash
# Step 0: Environment Setup
conda create -n speech_sep python=3.10 -y
conda activate speech_sep
pip install -r requirements.txt

# Step 1: Download LibriSpeech dev-clean and extract to data/LibriSpeech/

# Step 2: Convert FLAC to WAV
cd src && python run_convert.py && cd ..

# Step 3: Generate simulated dataset (room mixing)
python prepare_data.py

# Step 4: Train the model
python main.py

# Step 5: Run inference to generate separated audio
python inference.py

# Step 6: Quantitative evaluation
python evaluate.py
```

---

## 9. FAQ

**Q: Out of memory during training?**
- Reduce `TARGET_BATCH_SIZE` in `main.py` (e.g., to 64 or 32).

**Q: How to use a different dataset?**
- Place any 16kHz mono WAV files under `data/processed_wav/`, maintaining the subdirectory structure.
- Ensure at least 20 distinct audio files for valid train/val/test splits.

**Q: Why binaural instead of monaural?**
- Binaural cues (ILD, ITD) provide spatial information that helps distinguish co-located vs. spatially separated speakers — critical for realistic cocktail party scenarios.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **LibriSpeech** dataset: [OpenSLR](https://www.openslr.org/12)
- **pyroomacoustics**: Room acoustics simulation toolkit
- **PyTorch**: Deep learning framework
- Speech separation metrics via `pystoi` and `pesq`

---

<p align="center">
  <sub>Built with ❤️ for speech separation research</sub>
</p>
