import numpy as np
import os

# ==========================================
# 交互式输入路径
# ==========================================
folder_path = input("请输入 results 下的具体实验文件夹路径 (右键粘贴): ").strip()
# 去除可能存在的引号 (Windows 复制路径有时带引号)
folder_path = folder_path.replace('"', '').replace("'", "")

print(f"\n📂 正在读取: {folder_path}")

try:
    # ------------------------------------------------
    # 1. 优先读取 metrics.npy (官方算好的指标)
    # ------------------------------------------------
    metrics_path = os.path.join(folder_path, 'metrics.npy')
    if os.path.exists(metrics_path):
        metrics = np.load(metrics_path)
        print("\n📊【metrics.npy 官方指标】")
        # TSlib 标准顺序通常是: MAE, MSE, RMSE, MAPE, MSPE
        # 如果你发现数值对不上，可能是顺序变了，但通常是一致的
        print(f"MAE : {metrics[0]:.4f}")
        print(f"MSE : {metrics[1]:.4f}")
        print(f"RMSE: {metrics[2]:.4f}")
        print(f"MAPE: {metrics[3]:.4f}")
        print(f"MSPE: {metrics[4]:.4f}")
    else:
        print("⚠️ 未找到 metrics.npy，将跳过。")

    # ------------------------------------------------
    # 2. 读取 pred.npy 和 true.npy (手动复核)
    # ------------------------------------------------
    pred_path = os.path.join(folder_path, 'pred.npy')
    true_path = os.path.join(folder_path, 'true.npy')

    if os.path.exists(pred_path) and os.path.exists(true_path):
        print("\n🧮【手动计算复核 (基于 .npy 文件)】")
        preds = np.load(pred_path)
        trues = np.load(true_path)

        print(f"数据形状: {preds.shape} (Batch x Seq_Len x Channels)")

        # 计算核心指标
        # 1. MSE (均方误差)
        mse = np.mean((preds - trues) ** 2)
        # 2. MAE (平均绝对误差)
        mae = np.mean(np.abs(preds - trues))
        # 3. RMSE (均方根误差)
        rmse = np.sqrt(mse)

        print("-" * 30)
        print(f"✅ 最终确认成绩单:")
        print(f"MSE : {mse:.4f}")
        print(f"MAE : {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print("-" * 30)

        # 💡 简单的评价逻辑 (通用)
        if mse < 0.1:
            print("✨ 评价: 误差极低，拟合效果非常好！")
        elif mse < 0.4:
            print("👌 评价: 效果不错，属于正常范围。")
        else:
            print("🤔 评价: 误差偏大，可能存在削峰或欠拟合现象。")

    else:
        print("❌ 错误: 找不到 pred.npy 或 true.npy，无法计算。")

except Exception as e:
    print(f"\n❌ 发生错误: {e}")
    import traceback

    traceback.print_exc()

# 防止窗口一闪而过 (如果你是双击运行的话)
input("\n按回车键退出...")