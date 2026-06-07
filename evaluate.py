import os
import glob
import numpy as np
import librosa
import pandas as pd
from pystoi import stoi
from pesq import pesq


def calculate_sdr(estimated, reference):
    """
    计算 SI-SDR
    它衡量的是分离音频与纯净音频在波形上的相似度，且不受音量大小的影响。
    """
    #将二维或多维数组展平为一维向量
    estimated = estimated.flatten()
    reference = reference.flatten()
    #投影计算alpha
    alpha = np.dot(estimated, reference) / (np.dot(reference, reference) + 1e-8)

    #target是重构后的目标信号
    target = alpha * reference

    #res是分离误差
    res = estimated - target

    #计算能量比：10 * log10(信号能量 / 失真能量)
    si_sdr = 10 * np.log10(np.sum(target ** 2) / (np.sum(res ** 2) + 1e-8))
    return si_sdr


def load_and_align_audio(est_path, ref_path, sr=16000):
    """
    加载音频并进行时域对齐，解决由于 iSTFT 或模型延迟导致的相位/时间偏移问题。
    """
    #加载两个音频文件
    est_sig, _ = librosa.load(est_path, sr=sr)
    ref_sig, _ = librosa.load(ref_path, sr=sr)

    #截取较短的一端，确保两者长度一致，便于比较
    min_len = min(len(est_sig), len(ref_sig))
    est_sig = est_sig[:min_len]
    ref_sig = ref_sig[:min_len]

    #计算互相关函数，寻找两条波形重合度最高的时刻，即“最佳延迟点”
    correlation = np.correlate(est_sig, ref_sig, mode="full")
    best_delay = np.argmax(correlation) - (len(ref_sig) - 1)

    #通过补零（Padding）将较早到达的信号进行对齐，消除系统引起的时延误差
    if best_delay > 0:
        #est_sig比ref_sig滞后，需要对ref_sig进行前补零
        ref_sig = np.pad(ref_sig, (best_delay, 0), mode='constant')[:min_len]
    elif best_delay < 0:
        #est_sig比ref_sig超前，需要对est_sig进行前补零
        est_sig = np.pad(est_sig, (-best_delay, 0), mode='constant')[:min_len]

    return est_sig, ref_sig


def get_sdr_range_label(sdr_mix):
    """🛠️ 新增：根据初始混合SDR（信噪比）自适应划分学术区间"""
    if sdr_mix <= -10.0:
        return "Severe Noise (SDR <= -10dB)"
    elif -10.0 < sdr_mix <= -5.0:
        return "Moderate Noise (-10dB < SDR <= -5dB)"
    else:
        return "Mild Noise (SDR > -5dB)"


def run_system_evaluation(data_dir=os.path.join("data", "test")):
    print("================语音分离指标评估================")

    search_path = os.path.join(data_dir, "spk_*", "sample_*")
    all_sample_folders = glob.glob(search_path)

    def sort_key(path_str):
        normalized_path = path_str.replace("\\", "/")
        parts = normalized_path.split("/")
        try:
            spk_num = int(parts[-2].split("_")[-1])
            sample_num = int(parts[-1].split("_")[-1])
            return (spk_num, sample_num)
        except (IndexError, ValueError):
            return (0, 0)

    sample_folders = sorted(all_sample_folders, key=sort_key)

    if not sample_folders:
        print(f"【错误】未在 '{data_dir}' 中找到任何测试样本。")
        return

    results = []
    print(f"【系统提示】共检测到 {len(sample_folders)} 个样本，开始进行多梯度量化评估...")

    for folder in sample_folders:
        #加载文件
        parts = folder.replace("\\", "/").split("/")
        spk_group = parts[-2]
        sample_name = parts[-1]

        ref_a_path = os.path.join(folder, "clean_target_spk0.wav")
        mix_c_path = os.path.join(folder, "mixed.wav")
        est_d_path = os.path.join(folder, "processed.wav")

        if not os.path.exists(est_d_path):
            continue

        try:
            #评估分离后的结果processed.wav
            #计算三项核心学术指标：SDR (信噪比增益), STOI (可懂度), PESQ (感知质量)
            sig_d, sig_a = load_and_align_audio(est_d_path, ref_a_path, sr=16000)
            sdr_sep = calculate_sdr(sig_d, sig_a)
            stoi_val = stoi(sig_a, sig_d, 16000, extended=False)
            pesq_val = pesq(16000, sig_a, sig_d, 'wb')

            #计算处理前的基准值
            sig_c, _ = librosa.load(mix_c_path, sr=16000, mono=False)
            sig_c_left = sig_c[0]

            #将当前混合音片段临时保存为WAV文件
            tmp_mix_path = os.path.join(folder, "tmp_mix_l.wav")
            # 归一化
            sf_write_data = sig_c_left / (np.max(np.abs(sig_c_left)) + 1e-6) if sig_c_left.max() > 0 else sig_c_left
            sf_write_data = librosa.util.fix_length(sf_write_data, size=len(sig_c_left))

            import soundfile as sf
            sf.write(tmp_mix_path, sf_write_data, 16000)

            #对齐处理前后的音频，计算初始 SDR
            sig_c_aligned, sig_a_for_c = load_and_align_audio(tmp_mix_path, ref_a_path, sr=16000)
            sdr_mix = calculate_sdr(sig_c_aligned, sig_a_for_c)

            # 清理临时文件
            if os.path.exists(tmp_mix_path):
                os.remove(tmp_mix_path)

            #计算增益指标与分类标签
            #ΔSDR代表干扰被削减的程度
            sdr_imp = sdr_sep - sdr_mix
            #根据初始信噪比环境打标签
            sdr_range = get_sdr_range_label(sdr_mix)
            # 输出当前样本的评估快照，方便实时监控进度
            print(f" [{spk_group} | {sample_name}] ({sdr_range.split(' ')[0]}) "
                  f"ΔSDR: {sdr_imp:+.2f} dB | 分离后SDR: {sdr_sep:.2f} dB | STOI: {stoi_val:.4f} | PESQ: {pesq_val:.2f}")

            # 将所有指标存入结果列表，准备后续生成 CSV 统计报告
            results.append({
                "Group": spk_group,
                "Sample": sample_name,
                "SDR_Mix_Range": sdr_range,
                "SDR_Mix (dB)": sdr_mix,
                "SDR_Sep (dB)": sdr_sep,
                "SDR_Improv (dB)": sdr_imp,
                "STOI": stoi_val,
                "PESQ (WB)": pesq_val
            })

        except Exception as e:
            print(f"【错误】评估 {spk_group}/{sample_name} 时发生异常: {str(e)}")

    if results:
        df_all = pd.DataFrame(results)

        #按说话者数目进行拆分，独立输出各自的表格
        unique_groups = df_all["Group"].unique()

        print("\n======================================================")
        print("正在分流生成各人数梯度的专项学术汇报表...")
        print("======================================================")

        for current_group in unique_groups:
            #过滤出当前说话者人数的所有样本数据
            df_group = df_all[df_all["Group"] == current_group].copy()

            summary_rows = []

            #总结相同人数下，不同初始信噪比的平均表现
            range_grouped = df_group.groupby("SDR_Mix_Range")
            for range_name, range_df in range_grouped:
                summary_rows.append({
                    "Group": current_group,
                    "Sample": f"SubAvg: {range_name}",
                    "SDR_Mix (dB)": range_df["SDR_Mix (dB)"].mean(),
                    "SDR_Sep (dB)": range_df["SDR_Sep (dB)"].mean(),
                    "SDR_Improv (dB)": range_df["SDR_Improv (dB)"].mean(),
                    "STOI": range_df["STOI"].mean(),
                    "PESQ (WB)": range_df["PESQ (WB)"].mean()
                })

            #追加当前表格所有样本的全局最终平均表现
            summary_rows.append({
                "Group": current_group,
                "Sample": "== TOTAL AVERAGE ==",
                "SDR_Mix (dB)": df_group["SDR_Mix (dB)"].mean(),
                "SDR_Sep (dB)": df_group["SDR_Sep (dB)"].mean(),
                "SDR_Improv (dB)": df_group["SDR_Improv (dB)"].mean(),
                "STOI": df_group["STOI"].mean(),
                "PESQ (WB)": df_group["PESQ (WB)"].mean()
            })

            df_summary = pd.DataFrame(summary_rows)

            #合并当前组的明细样本数据和尾部总结行
            df_group_clean = df_group.drop(columns=["SDR_Mix_Range"])
            df_final_group = pd.concat([df_group_clean, df_summary], ignore_index=True)

            #保留四位和小数点规范化
            df_final_group["SDR_Mix (dB)"] = df_final_group["SDR_Mix (dB)"].round(2)
            df_final_group["SDR_Sep (dB)"] = df_final_group["SDR_Sep (dB)"].round(2)
            df_final_group["SDR_Improv (dB)"] = df_final_group["SDR_Improv (dB)"].round(2)
            df_final_group["STOI"] = df_final_group["STOI"].round(4)
            df_final_group["PESQ (WB)"] = df_final_group["PESQ (WB)"].round(2)

            #独立文件名输出
            output_csv = f"evaluation_report_{current_group}.csv"
            df_final_group.to_csv(output_csv, index=False, encoding="utf-8-sig")

            #打印当前表格的收尾摘要快照
            total_avg_row = summary_rows[-1]
            print(f"已导出独立表格：{output_csv}")
            print(
                f"   总计表现 -> ΔSDR: {total_avg_row['SDR_Improv (dB)']:+.2f} dB | STOI: {total_avg_row['STOI']:.4f} | PESQ: {total_avg_row['PESQ (WB)']:.2f}")
            for r_row in summary_rows[:-1]:
                print(f"   {r_row['Sample'].split(': ')[1]}: ΔSDR: {r_row['SDR_Improv (dB)']:+.2f} dB")

        print("======================================================")
        print("【评估成功】所有独立梯度专项表格已全部生成。")


if __name__ == "__main__":
    run_system_evaluation()