import os

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def load_gate_artifacts(folder_path):
    gate_path = os.path.join(folder_path, "gate.npy")
    linear_path = os.path.join(folder_path, "linear_out.npy")
    pred_path = os.path.join(folder_path, "pred.npy")
    true_path = os.path.join(folder_path, "true.npy")

    if not os.path.exists(gate_path):
        raise FileNotFoundError(f"找不到 {gate_path}")

    gate = np.load(gate_path)
    linear_out = np.load(linear_path) if os.path.exists(linear_path) else None
    pred = np.load(pred_path) if os.path.exists(pred_path) else None
    true = np.load(true_path) if os.path.exists(true_path) else None
    return gate, linear_out, pred, true


def extract_last_series(array):
    sample = array[-1]
    if sample.ndim == 2:
        return sample[-1]
    if sample.ndim == 1:
        return sample
    raise ValueError(f"不支持的数组形状: {array.shape}")


def extract_last_prediction_pair(pred, true):
    pred_sample = pred[-1]
    true_sample = true[-1]
    if pred_sample.ndim == 2:
        return pred_sample[:, -1], true_sample[:, -1]
    if pred_sample.ndim == 1:
        return pred_sample, true_sample
    raise ValueError(f"不支持的预测数组形状: {pred.shape}")


def plot_gate_series(gate, save_path):
    sample = extract_last_series(gate)
    plt.figure(figsize=(10, 4))
    plt.plot(sample, linewidth=2)
    plt.title("Gate Score Across Forecast Horizon")
    plt.xlabel("Forecast Step")
    plt.ylabel("Gate Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gate_hist(gate, save_path):
    plt.figure(figsize=(8, 4))
    plt.hist(gate.reshape(-1), bins=40)
    plt.title("Gate Score Distribution")
    plt.xlabel("Gate Score")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gate_vs_prediction(gate, pred, true, save_path):
    if pred is None or true is None:
        return

    gate_sample = extract_last_series(gate)
    pred_sample, true_sample = extract_last_prediction_pair(pred, true)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(true_sample, label="真实值", linewidth=2)
    axes[0].plot(pred_sample, label="预测值", linewidth=2)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("负荷")

    axes[1].plot(gate_sample, color="orange", linewidth=2)
    axes[1].set_ylabel("Gate")
    axes[1].set_xlabel("Forecast Step")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_linear_vs_final_prediction(linear_out, pred, true, save_path):
    if linear_out is None or pred is None or true is None:
        return

    linear_sample, _ = extract_last_prediction_pair(linear_out, true)
    pred_sample, true_sample = extract_last_prediction_pair(pred, true)

    plt.figure(figsize=(10, 5))
    plt.plot(true_sample, label="真实值", linewidth=2)
    plt.plot(linear_sample, label="linear_out", linewidth=2)
    plt.plot(pred_sample, label="最终输出", linewidth=2)
    plt.xlabel("Forecast Step")
    plt.ylabel("负荷")
    plt.title("linear_out vs final prediction vs ground truth")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_gate_stats(gate, save_path):
    flat_gate = gate.reshape(-1)
    threshold = np.quantile(flat_gate, 0.8)
    peak_mask = flat_gate >= threshold
    non_peak_mask = flat_gate < threshold

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"gate_mean: {np.mean(flat_gate):.6f}\n")
        f.write(f"gate_std: {np.std(flat_gate):.6f}\n")
        f.write(f"gate_min: {np.min(flat_gate):.6f}\n")
        f.write(f"gate_max: {np.max(flat_gate):.6f}\n")
        f.write(f"gate_mean_peak_region: {np.mean(flat_gate[peak_mask]):.6f}\n")
        f.write(f"gate_mean_non_peak_region: {np.mean(flat_gate[non_peak_mask]):.6f}\n")


def summarize_peak_region_analysis(gate, linear_out, pred, true, quantile=0.8):
    if linear_out is None or pred is None or true is None:
        raise ValueError("linear_out、pred、true 不能为空")

    linear_sample, true_sample = extract_last_prediction_pair(linear_out, true)
    pred_sample, true_sample = extract_last_prediction_pair(pred, true)
    gate_sample = extract_last_series(gate)

    threshold = np.quantile(true_sample, quantile)
    peak_mask = true_sample >= threshold
    non_peak_mask = true_sample < threshold

    linear_abs_err = np.abs(linear_sample - true_sample)
    final_abs_err = np.abs(pred_sample - true_sample)

    def safe_mean(values):
        if values.size == 0:
            return float("nan")
        return float(np.mean(values))

    return {
        "true_peak_threshold": float(threshold),
        "gate_mean_peak_region": safe_mean(gate_sample[peak_mask]),
        "gate_mean_non_peak_region": safe_mean(gate_sample[non_peak_mask]),
        "linear_mae_peak_region": safe_mean(linear_abs_err[peak_mask]),
        "linear_mae_non_peak_region": safe_mean(linear_abs_err[non_peak_mask]),
        "final_mae_peak_region": safe_mean(final_abs_err[peak_mask]),
        "final_mae_non_peak_region": safe_mean(final_abs_err[non_peak_mask]),
    }


def save_peak_region_analysis(gate, linear_out, pred, true, save_path):
    stats = summarize_peak_region_analysis(gate, linear_out, pred, true)
    with open(save_path, "w", encoding="utf-8") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value:.6f}\n")


def create_gate_visualizations(folder_path):
    gate, linear_out, pred, true = load_gate_artifacts(folder_path)
    save_dir = os.path.join(folder_path, "gate_figures")
    os.makedirs(save_dir, exist_ok=True)

    plot_gate_series(gate, os.path.join(save_dir, "1_gate_series.png"))
    plot_gate_hist(gate, os.path.join(save_dir, "2_gate_distribution.png"))
    plot_gate_vs_prediction(gate, pred, true, os.path.join(save_dir, "3_gate_vs_prediction.png"))
    plot_linear_vs_final_prediction(linear_out, pred, true, os.path.join(save_dir, "4_linear_vs_final_prediction.png"))
    save_gate_stats(gate, os.path.join(save_dir, "gate_stats_summary.txt"))
    if linear_out is not None and pred is not None and true is not None:
        save_peak_region_analysis(gate, linear_out, pred, true, os.path.join(save_dir, "peak_region_analysis.txt"))

    print(f"已保存到: {save_dir}")


if __name__ == "__main__":
    folder_input = input("请输入 results 下的实验目录路径: ").strip()
    folder_input = folder_input.replace('"', "").replace("'", "")

    if not os.path.isabs(folder_input) and not folder_input.startswith("./"):
        candidate = os.path.join("./results", folder_input)
        if os.path.exists(candidate):
            folder_input = candidate

    create_gate_visualizations(folder_input)
