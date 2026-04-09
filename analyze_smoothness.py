import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calc_metrics(series):
    s = pd.Series(series).dropna()
    diff = s.diff().dropna()

    mean_val = s.mean()
    std_val = s.std()
    cv = std_val / mean_val if mean_val != 0 else np.nan

    diff_std = diff.std()
    rel_diff_std = diff_std / mean_val if mean_val != 0 else np.nan

    return {
        "mean": mean_val,
        "std": std_val,
        "cv": cv,
        "diff_std": diff_std,
        "rel_diff_std": rel_diff_std
    }


def analyze_smoothness(files, output_dir="analysis_outputs"):
    os.makedirs(output_dir, exist_ok=True)

    results = []
    diff_data = {}

    for name, path in files.items():
        df = pd.read_csv(path)
        if "WHE" not in df.columns:
            raise ValueError(f"{name} 缺少 WHE 列")

        metrics = calc_metrics(df["WHE"])
        metrics["name"] = name
        results.append(metrics)

        s = pd.Series(df["WHE"]).dropna()
        diff = s.diff().dropna()
        mean_val = s.mean()
        diff_data[name] = diff / mean_val if mean_val != 0 else diff

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(output_dir, "smoothness_metrics.csv"), index=False, encoding="utf-8-sig")

    # 图1：CV
    plt.figure(figsize=(8, 5))
    bars1 = plt.bar(result_df["name"], result_df["cv"])
    for bar in bars1:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                 ha='center', va='bottom', fontsize=10)
    plt.title("不同数据集的变异系数（CV）对比")
    plt.xlabel("数据集")
    plt.ylabel("CV = Std / Mean")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_bar.png"), dpi=300)
    plt.close()

    # 图2：相对差分标准差
    plt.figure(figsize=(8, 5))
    bars2 = plt.bar(result_df["name"], result_df["rel_diff_std"])
    for bar in bars2:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                 ha='center', va='bottom', fontsize=10)
    plt.title("不同数据集的相对差分标准差对比")
    plt.xlabel("数据集")
    plt.ylabel("Relative Diff Std = Std(diff) / Mean")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rel_diff_std_bar.png"), dpi=300)
    plt.close()

    # 图3：相对一阶差分箱线图
    plt.figure(figsize=(10, 6))
    plt.boxplot([diff_data[k] for k in diff_data.keys()],
                labels=list(diff_data.keys()), showfliers=False)
    plt.title("不同数据集的相对一阶差分分布箱线图")
    plt.xlabel("数据集")
    plt.ylabel("相对一阶差分值")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diff_boxplot.png"), dpi=300)
    plt.close()

    return result_df


if __name__ == "__main__":
    files = {
        "London_Block0": "./dataset/London/Block0_Final_1h.csv",
        "London_Block1": "./dataset/London/Block1_Final_1h.csv",
        "AMPds": "./dataset/AMPds/AMPds_Final.csv",
        "Mexico": "./dataset/Mexico/Mexico_Final.csv",
    }
    analyze_smoothness(files)