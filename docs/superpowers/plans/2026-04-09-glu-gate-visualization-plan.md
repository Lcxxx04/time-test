# GLU Gate Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `iTransformer-GLU` 与 `DC-iTransformer-GLU` 在测试阶段自动导出 `gate` 数据与统计，并提供一个独立的 `vis_gate.py` 用来可视化 gate 行为，帮助分析 GLU 导致性能下降的原因。

**Architecture:** 只对 GLU 模型生效：在模型前向中缓存最新一次 `gate_score`，在长期预测测试阶段自动把它保存到对应 `results/` 目录；新增一个独立脚本 `vis_gate.py` 读取 `gate.npy` 和预测结果文件，生成 gate 时间序列、gate 分布和 gate/预测对照图，不修改现有 `vis_results.py` 主流程。

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, `exp/exp_long_term_forecasting.py`, 现有 GLU 模型文件

---

## File Structure

**Create:**

- `vis_gate.py`
- `tests/test_glu_gate_export.py`
- `docs/superpowers/plans/2026-04-09-glu-gate-visualization-plan.md`

**Modify:**

- `models/iTransformer_GLU.py`
- `models/DC_iTransformer_GLU.py`
- `exp/exp_long_term_forecasting.py`

**Verify against existing assets:**

- `checkpoints/*iTransformer-GLU*/checkpoint.pth`
- `checkpoints/*DC-iTransformer-GLU*/checkpoint.pth`
- `results/*iTransformer-GLU*/`
- `results/*DC-iTransformer-GLU*/`

---

### Task 1: 让 GLU 模型在前向后保留 `gate_score`

**Files:**

- Modify: `models/iTransformer_GLU.py`
- Modify: `models/DC_iTransformer_GLU.py`
- Test: `tests/test_glu_gate_export.py`

- [ ] **Step 1: 先写失败测试，约束 GLU 模型会缓存 gate**

```python
import unittest
from types import SimpleNamespace
import torch

from models.iTransformer_GLU import Model as GLUModel
from models.DC_iTransformer_GLU import Model as DCGLUModel


def make_configs():
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=96,
        pred_len=24,
        enc_in=7,
        d_model=16,
        embed="timeF",
        freq="h",
        dropout=0.1,
        factor=1,
        n_heads=1,
        d_ff=32,
        activation="gelu",
        e_layers=1,
    )


class TestGLUGateExport(unittest.TestCase):
    def test_glu_model_stores_last_gate_score(self):
        cfg = make_configs()
        model = GLUModel(cfg)
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        x_mark_enc = torch.randn(2, cfg.seq_len, 4)
        x_dec = torch.randn(2, 72, cfg.enc_in)
        x_mark_dec = torch.randn(2, 72, 4)
        model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        self.assertTrue(hasattr(model, "last_gate_score"))
        self.assertEqual(tuple(model.last_gate_score.shape), (2, cfg.enc_in, cfg.pred_len))

    def test_dc_glu_model_stores_last_gate_score(self):
        cfg = make_configs()
        model = DCGLUModel(cfg)
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        x_mark_enc = torch.randn(2, cfg.seq_len, 4)
        x_dec = torch.randn(2, 72, cfg.enc_in)
        x_mark_dec = torch.randn(2, 72, 4)
        model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        self.assertTrue(hasattr(model, "last_gate_score"))
        self.assertEqual(tuple(model.last_gate_score.shape), (2, cfg.enc_in, cfg.pred_len))
```

- [ ] **Step 2: 运行测试，确认先失败**

Run: `python -m unittest tests.test_glu_gate_export -v`

Expected: FAIL，因为当前 GLU 模型还没有 `last_gate_score`

- [ ] **Step 3: 在两个 GLU 模型里缓存 gate**

在 `forecast()` 中补这行：

```python
self.last_gate_score = gate_score.detach()
```

同时在 `__init__` 中初始化：

```python
self.last_gate_score = None
```

- [ ] **Step 4: 重新运行测试，确认通过**

Run: `python -m unittest tests.test_glu_gate_export -v`

Expected: PASS

---

### Task 2: 在长期预测测试阶段自动保存 `gate.npy` 与统计文件

**Files:**

- Modify: `exp/exp_long_term_forecasting.py`
- Test: `tests/test_glu_gate_export.py`

- [ ] **Step 1: 写失败测试，约束 gate 统计函数输出字段**

```python
from exp.exp_long_term_forecasting import summarize_gate_scores

gate = np.array([0.1, 0.4, 0.9], dtype=np.float32)
stats = summarize_gate_scores(gate)
assert set(stats) >= {"gate_mean", "gate_std", "gate_min", "gate_max"}
```

- [ ] **Step 2: 运行失败验证**

Run: `python -c "from exp.exp_long_term_forecasting import summarize_gate_scores"`

Expected: `ImportError` 或 `AttributeError`

- [ ] **Step 3: 在 `exp/exp_long_term_forecasting.py` 中新增 gate 汇总函数**

```python
def summarize_gate_scores(gate_array):
    return {
        "gate_mean": float(np.mean(gate_array)),
        "gate_std": float(np.std(gate_array)),
        "gate_min": float(np.min(gate_array)),
        "gate_max": float(np.max(gate_array)),
    }
```

- [ ] **Step 4: 在 `test()` 中只对 GLU 模型自动收集并保存 gate**

关键逻辑：

```python
gate_scores = []

...
outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

inner_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
if hasattr(inner_model, "last_gate_score") and inner_model.last_gate_score is not None:
    gate_scores.append(inner_model.last_gate_score.detach().cpu().numpy())
...

if gate_scores:
    gate_scores = np.concatenate(gate_scores, axis=0)
    np.save(folder_path + 'gate.npy', gate_scores)
    gate_stats = summarize_gate_scores(gate_scores)
    with open(folder_path + 'gate_stats.txt', 'w', encoding='utf-8') as f:
        for k, v in gate_stats.items():
            f.write(f"{k}: {v:.6f}\n")
```

- [ ] **Step 5: 运行单元测试**

Run: `python -m unittest tests.test_glu_gate_export -v`

Expected: PASS

---

### Task 3: 新增独立 `vis_gate.py`

**Files:**

- Create: `vis_gate.py`

- [ ] **Step 1: 新建基础加载逻辑**

```python
import os
import numpy as np
import matplotlib.pyplot as plt


def load_gate_artifacts(folder_path):
    gate_path = os.path.join(folder_path, "gate.npy")
    pred_path = os.path.join(folder_path, "pred.npy")
    true_path = os.path.join(folder_path, "true.npy")
    if not os.path.exists(gate_path):
        raise FileNotFoundError(f"找不到 {gate_path}")
    gate = np.load(gate_path)
    pred = np.load(pred_path) if os.path.exists(pred_path) else None
    true = np.load(true_path) if os.path.exists(true_path) else None
    return gate, pred, true
```

- [ ] **Step 2: 生成 gate 时间序列图**

```python
def plot_gate_series(gate, save_path):
    sample = gate[-1]
    if sample.ndim == 2:
        sample = sample[-1]
    plt.figure(figsize=(10, 4))
    plt.plot(sample, linewidth=2)
    plt.title("Gate Score Across Forecast Horizon")
    plt.xlabel("Forecast Step")
    plt.ylabel("Gate Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
```

- [ ] **Step 3: 生成 gate 分布图**

```python
def plot_gate_hist(gate, save_path):
    plt.figure(figsize=(8, 4))
    plt.hist(gate.reshape(-1), bins=40)
    plt.title("Gate Score Distribution")
    plt.xlabel("Gate Score")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
```

- [ ] **Step 4: 生成 gate 与预测对照图**

```python
def plot_gate_vs_prediction(gate, pred, true, save_path):
    if pred is None or true is None:
        return
    gate_sample = gate[-1]
    if gate_sample.ndim == 2:
        gate_sample = gate_sample[-1]
    pred_sample = pred[-1]
    true_sample = true[-1]
    if pred_sample.ndim == 2:
        pred_sample = pred_sample[:, -1]
        true_sample = true_sample[:, -1]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(true_sample, label="True", linewidth=2)
    axes[0].plot(pred_sample, label="Pred", linewidth=2)
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[1].plot(gate_sample, color="orange", linewidth=2)
    axes[1].set_ylabel("Gate")
    axes[1].set_xlabel("Forecast Step")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 5: 输出统计摘要**

```python
def save_gate_stats(gate, save_path):
    peak_threshold = np.quantile(gate.reshape(-1), 0.8)
    peak_mask = gate >= peak_threshold
    non_peak_mask = gate < peak_threshold
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"gate_mean: {np.mean(gate):.6f}\n")
        f.write(f"gate_std: {np.std(gate):.6f}\n")
        f.write(f"gate_min: {np.min(gate):.6f}\n")
        f.write(f"gate_max: {np.max(gate):.6f}\n")
        f.write(f"gate_mean_peak_region: {np.mean(gate[peak_mask]):.6f}\n")
        f.write(f"gate_mean_non_peak_region: {np.mean(gate[non_peak_mask]):.6f}\n")
```

- [ ] **Step 6: 提供命令行入口**

```python
if __name__ == "__main__":
    folder = input("请输入 results 下的实验目录路径: ").strip().replace('"', "").replace("'", "")
    gate, pred, true = load_gate_artifacts(folder)
    save_dir = os.path.join(folder, "gate_figures")
    os.makedirs(save_dir, exist_ok=True)
    plot_gate_series(gate, os.path.join(save_dir, "1_gate_series.png"))
    plot_gate_hist(gate, os.path.join(save_dir, "2_gate_distribution.png"))
    plot_gate_vs_prediction(gate, pred, true, os.path.join(save_dir, "3_gate_vs_prediction.png"))
    save_gate_stats(gate, os.path.join(save_dir, "gate_stats_summary.txt"))
    print(f"已保存到: {save_dir}")
```

---

### Task 4: 基于现有 checkpoint 补跑 gate 导出

**Files:**

- Verify: `checkpoints/*iTransformer-GLU*/checkpoint.pth`
- Verify: `checkpoints/*DC-iTransformer-GLU*/checkpoint.pth`

- [ ] **Step 1: 先确认现有 GLU checkpoint 存在**

Run: `python -c "from pathlib import Path; roots=list(Path('checkpoints').glob('*iTransformer-GLU*')) + list(Path('checkpoints').glob('*DC-iTransformer-GLU*')); print([p.name for p in roots])"`

Expected: 至少包含现有的 GLU 与 DC-GLU 目录

- [ ] **Step 2: 用已有 checkpoint 补跑测试导出**

示例命令（以用户已有实验为准，重点是 `--is_training 0` 与 `--model` 保持原样）：

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/London/ \
  --data_path Block0_Final_1h.csv \
  --model_id LondonB0_256_24_NoWeather \
  --model iTransformer-GLU \
  --data custom \
  --features S \
  --seq_len 256 \
  --label_len 48 \
  --pred_len 24 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 512 \
  --dropout 0.2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des Exp \
  --itr 1 \
  --target WHE
```

- [ ] **Step 3: 确认结果目录新增 gate 文件**

Run: `python -c "from pathlib import Path; p=Path('results'); print(sorted(str(x) for x in p.rglob('gate.npy'))[:10])"`

Expected: 至少看到 GLU 与 DC-GLU 对应目录下的 `gate.npy`

---

### Task 5: 最终回归检查

**Files:**

- Test: `tests/test_glu_gate_export.py`
- Test: `tests/test_itransformer_glu.py`
- Test: `tests/test_dc_itransformer_glu.py`
- Verify: `vis_gate.py`

- [ ] **Step 1: 运行相关测试**

Run: `python -m unittest tests.test_glu_gate_export tests.test_itransformer_glu tests.test_dc_itransformer_glu -v`

Expected: 全部 PASS

- [ ] **Step 2: 检查 gate 文件生成**

Run: `python -c "from pathlib import Path; print(len(list(Path('results').rglob('gate.npy'))))"`

Expected: 大于 0

- [ ] **Step 3: 运行独立可视化脚本**

Run: `python vis_gate.py`

Expected: 能在指定实验目录下生成 `gate_figures/`

- [ ] **Step 4: 验证输出图和统计文件**

Run: `python -c "from pathlib import Path; p=Path('<你的实验目录>')/'gate_figures'; names={x.name for x in p.iterdir()}; print(names)"`

Expected: 至少包含：

```text
{
  '1_gate_series.png',
  '2_gate_distribution.png',
  '3_gate_vs_prediction.png',
  'gate_stats_summary.txt'
}
```

---

## Self-Review

- 规格覆盖：
  - gate 自动导出：Task 1, Task 2
  - 独立可视化脚本：Task 3
  - 基于现有 checkpoint 补跑：Task 4
  - 统计摘要：Task 3
- Placeholder 扫描：无 `TODO`、`TBD`、`implement later`
- 一致性检查：
  - 仅 GLU 模型自动导出 gate
  - `vis_results.py` 不被修改
  - 新分析入口统一通过 `vis_gate.py`
