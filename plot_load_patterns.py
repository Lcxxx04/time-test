# plot_load_patterns.py
# 作用：
# 1) 典型 7 天负荷曲线对比（24h滚动平均 + Z-score）
# 2) 日负荷平均曲线对比（按均值归一化）
# 3) 周负荷平均曲线对比（按均值归一化）

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def normalize_series(series: pd.Series, mode: str = "mean") -> pd.Series:
    """
    归一化方法：
    - mode="mean"  : x / mean(x)
    - mode="zscore": (x - mean) / std
    """
    s = pd.Series(series).dropna().astype(float)

    if len(s) == 0:
        return s

    if mode == "mean":
        mean_val = s.mean()
        if mean_val == 0:
            return s.copy()
        return s / mean_val

    elif mode == "zscore":
        std_val = s.std()
        if std_val == 0:
            return s - s.mean()
        return (s - s.mean()) / std_val

    else:
        raise ValueError(f"不支持的归一化模式: {mode}")


def plot_7day_load(files, output_dir="analysis_outputs", window=168, rolling_window=24):
    """
    绘制典型 7 天负荷曲线对比：
    - 先取前 168 小时
    - 再做 24h 滚动平均（减弱尖刺）
    - 最后做 Z-score 标准化（比较形态）
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for name, path in files.items():
        df = pd.read_csv(path)
        if "WHE" not in df.columns:
            raise ValueError(f"{name} 缺少 WHE 列")

        s = df["WHE"].dropna().iloc[:window].astype(float)

        # 24h滚动平均，平滑短时尖峰
        s_smooth = s.rolling(window=rolling_window, min_periods=1).mean()

        # Z-score 标准化，比较趋势形态
        s_norm = normalize_series(s_smooth, mode="zscore")

        plt.plot(s_norm.values, label=name, linewidth=2)

    plt.title("典型7天负荷曲线对比（24h滚动平均 + Z-score）")
    plt.xlabel("时间步（小时）")
    plt.ylabel("标准化 WHE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "load_curve_7days.png"), dpi=300)
    plt.close()


def plot_daily_profile(files, output_dir="analysis_outputs"):
    """
    绘制日负荷平均曲线对比（按均值归一化）
    适合比较不同数据集的日内峰谷结构
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for name, path in files.items():
        df = pd.read_csv(path)
        if "WHE" not in df.columns:
            raise ValueError(f"{name} 缺少 WHE 列")

        s = df["WHE"].dropna().reset_index(drop=True).astype(float)
        s_norm = normalize_series(s, mode="mean")

        hours = [i % 24 for i in range(len(s_norm))]
        daily_df = pd.DataFrame({"hour": hours, "WHE_norm": s_norm.values})
        daily_mean = daily_df.groupby("hour")["WHE_norm"].mean()

        plt.plot(
            daily_mean.index,
            daily_mean.values,
            marker='o',
            linewidth=2,
            label=name
        )

    plt.title("日负荷平均曲线对比（按均值归一化）")
    plt.xlabel("小时")
    plt.ylabel("归一化平均 WHE")
    plt.xticks(range(24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "daily_profile.png"), dpi=300)
    plt.close()


def plot_weekly_profile(files, output_dir="analysis_outputs"):
    """
    绘制周负荷平均曲线对比（按均值归一化）
    适合比较周内相对变化趋势
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for name, path in files.items():
        df = pd.read_csv(path)
        if "WHE" not in df.columns:
            raise ValueError(f"{name} 缺少 WHE 列")

        s = df["WHE"].dropna().reset_index(drop=True).astype(float)
        s_norm = normalize_series(s, mode="mean")

        weekdays = [(i // 24) % 7 for i in range(len(s_norm))]
        weekly_df = pd.DataFrame({"weekday": weekdays, "WHE_norm": s_norm.values})
        weekly_mean = weekly_df.groupby("weekday")["WHE_norm"].mean()

        plt.plot(
            weekly_mean.index,
            weekly_mean.values,
            marker='o',
            linewidth=2,
            label=name
        )

    plt.title("周负荷平均曲线对比（按均值归一化）")
    plt.xlabel("星期（0=第1天）")
    plt.ylabel("归一化平均 WHE")
    plt.xticks(range(7))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "weekly_profile.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    files = {
        "London_Block0": "./dataset/London/Block0_Final_1h.csv",
        "AMPds": "./dataset/AMPds/AMPds_Final.csv",
        #"Mexico": "./dataset/Mexico/Mexico_Final.csv",
    }

    plot_7day_load(files, output_dir="analysis_outputs", window=168, rolling_window=24)
    plot_daily_profile(files, output_dir="analysis_outputs")
    plot_weekly_profile(files, output_dir="analysis_outputs")

    print("✅ 负荷模式图已生成：")
    print("   - analysis_outputs/load_curve_7days_smooth_zscore.png")
    print("   - analysis_outputs/daily_profile_norm.png")
    print("   - analysis_outputs/weekly_profile_norm.png")