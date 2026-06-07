"""
论文实验结果可视化脚本
根据 draw.txt 要求，从 4 个 evaluation_report CSV 中读取数据，生成4张图表。
输出：png 文件，dpi=300，Times New Roman 字体，A4 论文排版尺寸。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os

# ============================================================
# 全局样式配置
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 数据加载
# ============================================================
SPEAKER_COUNTS = [2, 3, 4, 5]
CSV_PATHS = {n: f"evaluation_report_spk_{n}.csv" for n in SPEAKER_COUNTS}

# 统一的列名
COL_GROUP = "Group"
COL_SAMPLE = "Sample"
COL_SDR_MIX = "SDR_Mix (dB)"
COL_SDR_IMP = "SDR_Improv (dB)"
COL_STOI = "STOI"
COL_PESQ = "PESQ (WB)"

# 噪声等级与 SubAvg 行的精确匹配 key
NOISE_LABEL_MAP = {
    "Mild":     "SubAvg: Mild Noise (SDR > -5dB)",
    "Moderate": "SubAvg: Moderate Noise (-10dB < SDR <= -5dB)",
    "Severe":   "SubAvg: Severe Noise (SDR <= -10dB)",
}
NOISE_ORDER = ["Mild", "Moderate", "Severe"]  # 从左到右（噪声增强）


def load_data():
    """读取四个CSV，返回 (test_data, summary_data, totals)"""
    test_records = []       # 仅前100条原始测试样本
    summary_records = []    # SubAvg 行
    totals = {}             # TOTAL AVERAGE 行

    for n in SPEAKER_COUNTS:
        df = pd.read_csv(CSV_PATHS[n], encoding="utf-8-sig")

        # 分离：前100条测试样本
        test_rows = df[~df[COL_SAMPLE].str.startswith(("SubAvg", "=="))]
        if len(test_rows) > 100:
            test_rows = test_rows.head(100)
        test_rows = test_rows.copy()
        test_rows["Speakers"] = n
        test_records.append(test_rows)

        # 分离：SubAvg 行
        for label_key, full_label in NOISE_LABEL_MAP.items():
            row = df[df[COL_SAMPLE] == full_label]
            if not row.empty:
                summary_records.append({
                    "Speakers": n,
                    "NoiseLevel": label_key,
                    COL_SDR_MIX: row[COL_SDR_MIX].values[0],
                    COL_SDR_IMP: row[COL_SDR_IMP].values[0],
                    COL_STOI: row[COL_STOI].values[0],
                    COL_PESQ: row[COL_PESQ].values[0],
                })

        # 分离：TOTAL AVERAGE
        total_row = df[df[COL_SAMPLE] == "== TOTAL AVERAGE =="]
        if not total_row.empty:
            totals[n] = {
                COL_SDR_IMP: total_row[COL_SDR_IMP].values[0],
                COL_STOI: total_row[COL_STOI].values[0],
                COL_PESQ: total_row[COL_PESQ].values[0],
            }

    test_df = pd.concat(test_records, ignore_index=True)
    summary_df = pd.DataFrame(summary_records)
    return test_df, summary_df, totals


test_df, summary_df, totals = load_data()

# ============================================================
# 图1：不同说话人数下总体性能趋势图（双Y轴）
# ============================================================
def draw_fig1(totals):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.array(SPEAKER_COUNTS)
    delta_sdr = [totals[n][COL_SDR_IMP] for n in SPEAKER_COUNTS]
    stoi_vals = [totals[n][COL_STOI] for n in SPEAKER_COUNTS]
    pesq_vals = [totals[n][COL_PESQ] for n in SPEAKER_COUNTS]

    color_sdr = "#2B7BBD"
    color_stoi = "#E0532C"
    color_pesq = "#3A923A"

    # 左轴：ΔSDR（柱状图）
    bars = ax1.bar(x - 0.15, delta_sdr, width=0.30,
                   color=color_sdr, alpha=0.85, label=r"$\Delta$SDR (dB)",
                   edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Number of Speakers")
    ax1.set_ylabel(r"$\Delta$SDR (dB)", color=color_sdr)
    ax1.tick_params(axis="y", labelcolor=color_sdr)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{n} Speakers" for n in SPEAKER_COUNTS])

    # 右轴：STOI + PESQ（折线）
    ax2 = ax1.twinx()
    line1, = ax2.plot(x, stoi_vals, color=color_stoi, marker="o",
                      markersize=8, linewidth=2, label="STOI")
    line2, = ax2.plot(x, pesq_vals, color=color_pesq, marker="s",
                      markersize=8, linewidth=2, label="PESQ (WB)")
    ax2.set_ylabel("STOI / PESQ Score")
    ax2.set_ylim(0, max(max(stoi_vals), max(pesq_vals)) * 1.15)

    # 图例合并
    bars_legend = plt.Line2D([0], [0], color=color_sdr, linewidth=6, alpha=0.85)
    lines = [bars_legend, line1, line2]
    labels = [r"$\Delta$SDR (dB)", "STOI", "PESQ (WB)"]
    ax1.legend(lines, labels, loc="upper right", frameon=True,
               fancybox=True, edgecolor="gray")

    ax1.set_title("Overall Performance under Different Numbers of Speakers",
                  fontweight="bold", pad=12)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_xlim(1.3, 5.7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_overall_performance.png"))
    plt.close(fig)
    print("[OK] fig1_overall_performance.png")


# ============================================================
# 图2：STOI 随噪声等级变化曲线
# ============================================================
def draw_fig2(summary_df):
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2B7BBD", "#E0532C", "#3A923A", "#8E44AD"]
    markers = ["o", "s", "D", "^"]

    for i, n in enumerate(SPEAKER_COUNTS):
        sub = summary_df[summary_df["Speakers"] == n]
        stoi_vals = [sub[sub["NoiseLevel"] == lvl][COL_STOI].values[0]
                     for lvl in NOISE_ORDER]
        ax.plot(NOISE_ORDER, stoi_vals,
                color=colors[i], marker=markers[i], markersize=8,
                linewidth=2, label=f"{n} Speakers")

    ax.set_xlabel("Noise Condition")
    ax.set_ylabel("STOI")
    ax.set_title("STOI under Different Noise Conditions",
                 fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, fancybox=True, edgecolor="gray")
    ax.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_stoi_noise.png"))
    plt.close(fig)
    print("[OK] fig2_stoi_noise.png")


# ============================================================
# 图3：PESQ 随噪声等级变化曲线
# ============================================================
def draw_fig3(summary_df):
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#2B7BBD", "#E0532C", "#3A923A", "#8E44AD"]
    markers = ["o", "s", "D", "^"]

    for i, n in enumerate(SPEAKER_COUNTS):
        sub = summary_df[summary_df["Speakers"] == n]
        pesq_vals = [sub[sub["NoiseLevel"] == lvl][COL_PESQ].values[0]
                     for lvl in NOISE_ORDER]
        ax.plot(NOISE_ORDER, pesq_vals,
                color=colors[i], marker=markers[i], markersize=8,
                linewidth=2, label=f"{n} Speakers")

    ax.set_xlabel("Noise Condition")
    ax.set_ylabel("PESQ (WB)")
    ax.set_title("PESQ under Different Noise Conditions",
                 fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, fancybox=True, edgecolor="gray")
    ax.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig3_pesq_noise.png"))
    plt.close(fig)
    print("[OK] fig3_pesq_noise.png")


# ============================================================
# 图4：ΔSDR 箱线图
# ============================================================
def draw_fig4(test_df):
    fig, ax = plt.subplots(figsize=(8, 5))

    box_data = [test_df[test_df["Speakers"] == n][COL_SDR_IMP].values
                for n in SPEAKER_COUNTS]

    box_colors = ["#2B7BBD", "#E0532C", "#3A923A", "#8E44AD"]

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 1.5},
                    whiskerprops={"linewidth": 1.2},
                    capprops={"linewidth": 1.2},
                    boxprops={"linewidth": 1.2},
                    flierprops={"marker": "o", "markerfacecolor": "red",
                                "markersize": 5, "alpha": 0.6})

    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels([f"{n} Speakers" for n in SPEAKER_COUNTS])
    ax.set_ylabel(r"$\Delta$SDR (dB)")
    ax.set_title("Distribution of $\Delta$SDR across Different Speaker Numbers",
                 fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # 在箱体上方标注中位数
    for i, (n, data) in enumerate(zip(SPEAKER_COUNTS, box_data)):
        median = np.median(data)
        ax.annotate(f"{median:.2f}", xy=(i + 1, median),
                    xytext=(i + 1.25, median + 0.3),
                    fontsize=9, color="black", fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "fig4_delta_sdr_boxplot.png"))
    plt.close(fig)
    print("[OK] fig4_delta_sdr_boxplot.png")



# ============================================================
# 主函数：依次生成全部4张图
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  论文实验数据可视化 — 批量生成 5 张图表")
    print("=" * 60)
    print(f"  数据源：{', '.join(CSV_PATHS.values())}")
    print(f"  输出目录：{OUTPUT_DIR}/")
    print("-" * 60)

    draw_fig1(totals)
    draw_fig2(summary_df)
    draw_fig3(summary_df)
    draw_fig4(test_df)

    print("-" * 60)
    print(f"  全部完成！共生成4张图片 → {OUTPUT_DIR}/")
    print("=" * 60)
