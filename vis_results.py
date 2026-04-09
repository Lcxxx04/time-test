import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 环境设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="ticks", font='SimHei')


def create_visualizations(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ 错误：找不到路径 {folder_path}")
        return

    save_path = os.path.join(folder_path, 'paper_figures/')
    os.makedirs(save_path, exist_ok=True)

    print("🚀 正在提取数据并生成图表...")

    try:
        # -------- 加载 loss --------
        train_loss = np.load(os.path.join(folder_path, 'loss_train.npy'))

        vali_path = os.path.join(folder_path, 'loss_vali.npy')
        vali_loss = np.load(vali_path)

        preds = np.load(os.path.join(folder_path, 'pred.npy'))
        trues = np.load(os.path.join(folder_path, 'true.npy'))

        # ---------------------------------------------------------
        # 图1：Loss曲线
        # ---------------------------------------------------------
        plt.figure(figsize=(8, 5))
        plt.plot(train_loss, label='训练集 Loss', lw=2)
        plt.plot(vali_loss, label='验证集 Loss', lw=2)
        plt.title('模型训练收敛过程', fontsize=14)
        plt.xlabel('训练轮次 (Epoch)', fontsize=12)
        plt.ylabel('均方误差 (MSE)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(save_path, '1_loss_curve.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------------
        # 图2：预测对比图（兼容2D/3D）
        # ---------------------------------------------------------
        idx = -1

        if trues.ndim == 3:
            sample_true = trues[idx, :, -1].flatten()
            sample_pred = preds[idx, :, -1].flatten()
        elif trues.ndim == 2:
            sample_true = trues[idx, :].flatten()
            sample_pred = preds[idx, :].flatten()
        else:
            raise ValueError(f"不支持的数组维度: {trues.shape}")

        pred_len = len(sample_true)

        plt.figure(figsize=(10, 5))
        plt.plot(sample_true, label='真实值 (Ground Truth)', linewidth=2.5)
        plt.plot(sample_pred, label='预测值 (Prediction)', linewidth=2.5)
        plt.title(f'电力负荷预测结果对比 (预测窗口长度: {pred_len})', fontsize=14)
        plt.xlabel('时间步 (Time Step)', fontsize=12)
        plt.ylabel('功率 (Standardized)', fontsize=12)
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(save_path, '2_prediction_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------------
        # 图3：散点回归图
        # ---------------------------------------------------------
        preds_flat = preds.reshape(-1)
        trues_flat = trues.reshape(-1)

        plt.figure(figsize=(8, 8))

        if len(trues_flat) > 5000:
            sample_idx = np.random.choice(len(trues_flat), 5000, replace=False)
            x_sample = trues_flat[sample_idx]
            y_sample = preds_flat[sample_idx]
        else:
            x_sample = trues_flat
            y_sample = preds_flat

        sns.regplot(
            x=x_sample, y=y_sample,
            scatter_kws={'alpha': 0.3, 's': 10, 'color': 'gray'},
            line_kws={'color': 'red', 'lw': 2},
            label='拟合回归线'
        )

        all_max = max(x_sample.max(), y_sample.max())
        all_min = min(x_sample.min(), y_sample.min())
        plt.plot([all_min, all_max], [all_min, all_max],
                 color='blue', linestyle='--', label='理想对角线(y=x)')

        plt.title('预测值 vs 真实值 相关性分析', fontsize=14)
        plt.xlabel('真实观测值 (True Value)', fontsize=12)
        plt.ylabel('模型预测值 (Predicted Value)', fontsize=12)
        plt.legend()
        plt.axis('equal')
        plt.savefig(os.path.join(save_path, '3_scatter_plot.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 绘图成功！图片已保存至: {save_path}")

    except Exception as e:
        print(f"❌ 绘图失败: {e}")
        import traceback
        traceback.print_exc()


# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("🏠 毕业论文图表生成工具")
    print("=" * 50)

    folder_input = input("\n请输入 results 下的具体实验文件夹路径: ").strip()
    folder_input = folder_input.replace('"', '').replace("'", "")

    if not os.path.isabs(folder_input) and not folder_input.startswith('./'):
        if os.path.exists(os.path.join('./results', folder_input)):
            folder_input = os.path.join('./results', folder_input)

    create_visualizations(folder_input)

    print("\n" + "=" * 50)
    input("任务结束，按回车键退出...")