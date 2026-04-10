# iTransformer-GLU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在原始基线 `iTransformer` 旁边新增一个 `iTransformer-GLU` 变体，仅替换输出头为 GLU 门控结构，并让它能够通过当前预测入口、脚本和文档体系独立运行与对比。

**Architecture:** 保持 `models/iTransformer.py` 不动，新增 `models/iTransformer_GLU.py` 作为独立模型文件；在 `exp/exp_basic.py` 与 `run.py` 中新增 `iTransformer-GLU` 的显式入口映射；通过新增成对脚本和测试来保证它只改输出头、不破坏基线模型的身份与可复现性。

**Tech Stack:** Python, PyTorch, `run.py`, `exp/exp_basic.py`, `unittest`, PowerShell, 现有 `iTransformer` 预测结构

---

## File Structure

**Create:**

- `models/iTransformer_GLU.py`
- `tests/test_itransformer_glu.py`
- `scripts/long_term_forecast/ETT_script/iTransformer_GLU_ETTh2.sh`
- `scripts/long_term_forecast/Weather_script/iTransformer_GLU.sh`
- `scripts/long_term_forecast/Traffic_script/iTransformer_GLU.sh`
- `scripts/long_term_forecast/ECL_script/iTransformer_GLU.sh`
- `scripts/short_term_forecast/iTransformer_GLU_M4.sh`
- `docs/superpowers/plans/2026-04-09-itransformer-glu-plan.md`

**Modify:**

- `exp/exp_basic.py`
- `run.py`
- `README.md`
- `README_zh.md`

**Verify against existing assets:**

- `models/iTransformer.py`
- `checkpoints/*iTransformer*/checkpoint.pth`

---

### Task 1: 新增 `iTransformer-GLU` 模型文件

**Files:**

- Create: `models/iTransformer_GLU.py`
- Test: `tests/test_itransformer_glu.py`

- [ ] **Step 1: 写失败测试，约束 GLU 头结构**

```python
import unittest
from types import SimpleNamespace
import torch

from models.iTransformer_GLU import Model


def make_configs(task_name="long_term_forecast"):
    return SimpleNamespace(
        task_name=task_name,
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


class TestITransformerGLU(unittest.TestCase):
    def test_builds_glu_heads(self):
        model = Model(make_configs())
        self.assertTrue(hasattr(model, "proj_linear"))
        self.assertTrue(hasattr(model, "proj_gate"))
        self.assertFalse(hasattr(model, "projection"))

    def test_forward_output_shape(self):
        cfg = make_configs()
        model = Model(cfg)
        x_enc = torch.randn(2, cfg.seq_len, cfg.enc_in)
        x_mark_enc = torch.randn(2, cfg.seq_len, 4)
        x_dec = torch.randn(2, 72, cfg.enc_in)
        x_mark_dec = torch.randn(2, 72, 4)
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        self.assertEqual(out.shape, (2, cfg.pred_len, cfg.enc_in))

    def test_rejects_non_forecast_tasks(self):
        with self.assertRaises(ValueError):
            Model(make_configs(task_name="classification"))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试，确认先失败**

Run: `python -m unittest tests.test_itransformer_glu -v`

Expected: FAIL，因为 `models/iTransformer_GLU.py` 还不存在

- [ ] **Step 3: 写最小实现，新增 `models/iTransformer_GLU.py`**

```python
import torch
import torch.nn as nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        if configs.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise ValueError("iTransformer-GLU only supports forecasting tasks in this repository.")

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.proj_linear = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.proj_gate = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, n_vars = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        linear_out = self.proj_linear(enc_out)
        gate_score = torch.sigmoid(self.proj_gate(enc_out))
        dec_out = (linear_out * gate_score).permute(0, 2, 1)[:, :, :n_vars]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `python -m unittest tests.test_itransformer_glu -v`

Expected: PASS（3 tests）

---

### Task 2: 在注册入口中暴露 `iTransformer-GLU`

**Files:**

- Modify: `exp/exp_basic.py`
- Modify: `run.py`

- [ ] **Step 1: 写失败验证脚本**

```python
from exp.exp_basic import ALLOWED_MODELS

assert "iTransformer-GLU" in ALLOWED_MODELS
```

- [ ] **Step 2: 运行失败验证**

Run: `python -c "from exp.exp_basic import ALLOWED_MODELS; assert 'iTransformer-GLU' in ALLOWED_MODELS"`

Expected: `AssertionError`

- [ ] **Step 3: 修改 `exp/exp_basic.py`**

```python
ALLOWED_MODELS = {
    "iTransformer",
    "DC-iTransformer",
    "iTransformer-GLU",
    "PatchTST",
    "TimeXer",
}
```

并在显式映射中加入：

```python
"iTransformer-GLU": "models.iTransformer_GLU",
```

- [ ] **Step 4: 修改 `run.py`**

```python
choices=["iTransformer", "DC-iTransformer", "iTransformer-GLU", "PatchTST", "TimeXer"]
```

- [ ] **Step 5: 验证注册与 CLI**

Run: `python -c "from exp.exp_basic import ALLOWED_MODELS; print(sorted(ALLOWED_MODELS))"`

Expected: 输出包含 `iTransformer-GLU`

Run: `python run.py --help`

Expected: `--model` 选项中包含 `iTransformer-GLU`

---

### Task 3: 新增成对 GLU 脚本

**Files:**

- Create: `scripts/long_term_forecast/ETT_script/iTransformer_GLU_ETTh2.sh`
- Create: `scripts/long_term_forecast/Weather_script/iTransformer_GLU.sh`
- Create: `scripts/long_term_forecast/Traffic_script/iTransformer_GLU.sh`
- Create: `scripts/long_term_forecast/ECL_script/iTransformer_GLU.sh`
- Create: `scripts/short_term_forecast/iTransformer_GLU_M4.sh`

- [ ] **Step 1: 复制对应 iTransformer 基线脚本**

```bash
model_name=iTransformer-GLU
```

- [ ] **Step 2: 保持参数完全一致，只替换模型名**

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1
```

- [ ] **Step 3: 验证脚本并存**

Run: `rg "model_name=iTransformer|model_name=iTransformer-GLU" scripts`

Expected: 能同时看到 baseline 与 GLU 两类脚本

---

### Task 4: 更新 README 与中文文档

**Files:**

- Modify: `README.md`
- Modify: `README_zh.md`

- [ ] **Step 1: 更新支持模型列表**

```markdown
- `iTransformer`
- `DC-iTransformer`
- `iTransformer-GLU`
- `PatchTST`
- `TimeXer`
```

- [ ] **Step 2: 更新模型说明**

```markdown
- `iTransformer`：原始基线
- `DC-iTransformer`：卷积增强版
- `iTransformer-GLU`：GLU 输出头版
```

- [ ] **Step 3: 更新 `models/` 结构说明**

```text
models/
├── iTransformer.py
├── DC_iTransformer.py
├── iTransformer_GLU.py
├── PatchTST.py
├── TimeXer.py
└── __init__.py
```

- [ ] **Step 4: 新增 GLU 示例命令**

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh2_96_96 \
  --model iTransformer-GLU \
  --data ETTh2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des Exp \
  --d_model 128 \
  --d_ff 128
```

- [ ] **Step 5: 验证文档文本**

Run: `python -c "from pathlib import Path; text=(Path('README.md').read_text(encoding='utf-8') + Path('README_zh.md').read_text(encoding='utf-8')); assert 'iTransformer-GLU' in text and 'models/iTransformer_GLU.py' in text"`

Expected: 无报错

---

### Task 5: 验证基线 checkpoint 不会被误用于 GLU 版

**Files:**

- Modify: `tests/test_itransformer_glu.py`

- [ ] **Step 1: 新增 checkpoint 不兼容测试**

```python
def test_glu_model_rejects_baseline_checkpoint_keys(self):
    cfg = make_configs()
    baseline_state = {
        "projection.weight": torch.randn(cfg.pred_len, cfg.d_model),
        "projection.bias": torch.randn(cfg.pred_len),
    }
    model = Model(cfg)
    with self.assertRaises(RuntimeError):
        model.load_state_dict(baseline_state, strict=True)
```

- [ ] **Step 2: 运行测试，确认通过**

Run: `python -m unittest tests.test_itransformer_glu -v`

Expected: PASS，且新增用例证明 GLU 不会误接 baseline 输出头参数

---

### Task 6: 最终回归检查

**Files:**

- Test: `tests/test_itransformer_baseline.py`
- Test: `tests/test_itransformer_dc.py`
- Test: `tests/test_dc_itransformer_alias.py`
- Test: `tests/test_itransformer_glu.py`
- Test: `tests/test_patchtst_forecast_only.py`

- [ ] **Step 1: 运行全量单元测试**

Run: `python -m unittest tests.test_itransformer_baseline tests.test_dc_itransformer_alias tests.test_itransformer_dc tests.test_itransformer_glu tests.test_patchtst_forecast_only -v`

Expected: 全部 PASS

- [ ] **Step 2: 运行入口白名单检查**

Run: `python -c "from exp.exp_basic import ALLOWED_MODELS; print(sorted(ALLOWED_MODELS))"`

Expected: 包含 `iTransformer-GLU`

- [ ] **Step 3: 运行 CLI 帮助检查**

Run: `python run.py --help`

Expected: `--model` 中包含 `iTransformer-GLU`

- [ ] **Step 4: 运行文本一致性检查**

Run: `rg "iTransformer-GLU|iTransformer|DC-iTransformer" README.md README_zh.md scripts`

Expected: baseline / DC / GLU 三类命名同时存在，无覆盖关系

---

## Self-Review

- 规格覆盖：
  - 新模型文件：Task 1
  - GLU 输出头结构：Task 1
  - 模型注册与 CLI：Task 2
  - 成对脚本：Task 3
  - README/README_zh：Task 4
  - checkpoint 影响验证：Task 5
- Placeholder 扫描：无 `TODO`、`TBD`、`implement later`
- 一致性检查：
  - 对外名称统一为 `iTransformer-GLU`
  - 文件名统一为 `models/iTransformer_GLU.py`
  - 基线与 DC 模型保持不变，仅新增 GLU 变体
