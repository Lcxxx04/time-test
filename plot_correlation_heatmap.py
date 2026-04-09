import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_correlation_heatmap(file_path, output_dir="analysis_outputs", title="相关性热力图"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(file_path)
    numeric_df = df.select_dtypes(include=["number"])

    corr = numeric_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_correlation_heatmap(
        "./dataset/London/Block0_Final_1h.csv",
        title="London Block0 多变量相关性热力图"
    )