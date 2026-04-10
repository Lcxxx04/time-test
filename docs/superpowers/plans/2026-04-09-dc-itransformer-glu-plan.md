# DC-iTransformer-GLU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `DC-iTransformer-GLU` 组合版，使其同时具备前端 DC 局部卷积增强和末端 GLU 门控输出头，并能与 `iTransformer`、`DC-iTransformer`、`iTransformer-GLU` 进行公平对比。

**Architecture:** 保持现有三种 iTransformer 变体不动，新增 `models/DC_iTransformer_GLU.py`；前端复用 DC 版本的 `local_cnn` 路径，末端复用 GLU 版本的 `proj_linear` / `proj_gate` 输出头；在 `exp/exp_basic.py`、`run.py`、脚本和 README 中新增第四个对外模型名 `DC-iTransformer-GLU`。

**Tech Stack:** Python, PyTorch, `run.py`, `exp/exp_basic.py`, `unittest`, PowerShell, 现有 iTransformer / DC / GLU 变体实现

---

## File Structure

**Create:**

- `models/DC_iTransformer_GLU.py`
- `tests/test_dc_itransformer_glu.py`
- `scripts/long_term_forecast/ETT_script/DC_iTransformer_GLU_ETTh2.sh`
- `scripts/long_term_forecast/Weather_script/DC_iTransformer_GLU.sh`
- `scripts/long_term_forecast/Traffic_script/DC_iTransformer_GLU.sh`
- `scripts/long_term_forecast/ECL_script/DC_iTransformer_GLU.sh`
- `scripts/short_term_forecast/DC_iTransformer_GLU_M4.sh`
- `docs/superpowers/plans/2026-04-09-dc-itransformer-glu-plan.md`

**Modify:**

- `exp/exp_basic.py`
- `run.py`
- `README.md`
- `README_zh.md`

**Verify against existing assets:**

- `models/DC_iTransformer.py`
- `models/iTransformer_GLU.py`
- `checkpoints/*iTransformer*/checkpoint.pth`

---

### Task 1: 新增 `DC-iTransformer-GLU` 模型文件

**Files:**

- Create: `models/DC_iTransformer_GLU.py`
- Test: `tests/test_dc_itransformer_glu.py`

- [ ] **Step 1: 写失败测试，约束组合版结构**

```python
import unittest
from types import SimpleNamespace
import torch

from models.DC_iTransformer_GLU import Model


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


class TestDCITransformerGLU(unittest.TestCase):
    def test_has_dc_and_glu_modules(self):
        model = Model(make_configs())
        self.assertTrue(hasattr(model, "local_cnn"))
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

Run: `python -m unittest tests.test_dc_itransformer_glu -v`

Expected: FAIL，因为 `models/DC_iTransformer_GLU.py` 还不存在

- [ ] **Step 3: 新建 `models/DC_iTransformer_GLU.py`**

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
            raise ValueError("DC-iTransformer-GLU only supports forecasting tasks in this repository.")

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.local_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=configs.enc_in,
                out_channels=configs.enc_in,
                kernel_size=3,
                padding=1,
                groups=configs.enc_in,
            ),
            nn.GELU(),
        )
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
        x_enc = self.local_cnn(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
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

- [ ] **Step 4: 补强测试，锁定 GLU 的 `sigmoid` 行为**

```python
def test_gate_path_uses_sigmoid(self):
    class FakeEmbedding(torch.nn.Module):
        def forward(self, x, x_mark):
            return torch.zeros(2, 7, 16)

    class FakeEncoder(torch.nn.Module):
        def forward(self, x, attn_mask=None):
            return x, None

    model = Model(make_configs())
    model.enc_embedding = FakeEmbedding()
    model.encoder = FakeEncoder()
    torch.nn.init.zeros_(model.proj_linear.weight)
    torch.nn.init.ones_(model.proj_linear.bias)
    torch.nn.init.zeros_(model.proj_gate.weight)
    torch.nn.init.constant_(model.proj_gate.bias, -20.0)

    pattern = torch.tensor([-1.0, 1.0]).repeat(48).view(1, 96, 1)
    x_enc = pattern.repeat(2, 1, 7)
    output = model(x_enc, None, None, None)
    self.assertTrue(torch.allclose(output, torch.zeros_like(output), atol=1e-4))
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `python -m unittest tests.test_dc_itransformer_glu -v`

Expected: PASS

---

### Task 2: 在入口中暴露 `DC-iTransformer-GLU`

**Files:**

- Modify: `exp/exp_basic.py`
- Modify: `run.py`

- [ ] **Step 1: 写失败验证**

```python
from exp.exp_basic import ALLOWED_MODELS

assert "DC-iTransformer-GLU" in ALLOWED_MODELS
```

- [ ] **Step 2: 运行失败验证**

Run: `python -c "from exp.exp_basic import ALLOWED_MODELS; assert 'DC-iTransformer-GLU' in ALLOWED_MODELS"`

Expected: `AssertionError`

- [ ] **Step 3: 修改 `exp/exp_basic.py`**

```python
ALLOWED_MODELS = {
    "iTransformer",
    "DC-iTransformer",
    "iTransformer-GLU",
    "DC-iTransformer-GLU",
    "PatchTST",
    "TimeXer",
}
```

并加入显式映射：

```python
"DC-iTransformer-GLU": "models.DC_iTransformer_GLU",
```

- [ ] **Step 4: 修改 `run.py`**

```python
choices=[
    "iTransformer",
    "DC-iTransformer",
    "iTransformer-GLU",
    "DC-iTransformer-GLU",
    "PatchTST",
    "TimeXer",
]
```

- [ ] **Step 5: 验证注册与 CLI**

Run: `python -c "from exp.exp_basic import ALLOWED_MODELS; print(sorted(ALLOWED_MODELS))"`

Expected: 输出包含 `DC-iTransformer-GLU`

Run: `python run.py --help`

Expected: `--model` 中包含 `DC-iTransformer-GLU`

---

### Task 3: 新增成对组合版脚本

**Files:**

- Create: `scripts/long_term_forecast/ETT_script/DC_iTransformer_GLU_ETTh2.sh`
- Create: `scripts/long_term_forecast/Weather_script/DC_iTransformer_GLU.sh`
- Create: `scripts/long_term_forecast/Traffic_script/DC_iTransformer_GLU.sh`
- Create: `scripts/long_term_forecast/ECL_script/DC_iTransformer_GLU.sh`
- Create: `scripts/short_term_forecast/DC_iTransformer_GLU_M4.sh`

- [ ] **Step 1: 复制对应基线 iTransformer 脚本**

```bash
model_name=DC-iTransformer-GLU
```

- [ ] **Step 2: 保持其余参数完全一致**

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

Run: `rg "model_name=iTransformer|model_name=DC-iTransformer|model_name=iTransformer-GLU|model_name=DC-iTransformer-GLU" scripts`

Expected: 四类脚本并存

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
- `DC-iTransformer-GLU`
- `PatchTST`
- `TimeXer`
```

- [ ] **Step 2: 更新模型说明**

```markdown
- `iTransformer`：原始基线
- `DC-iTransformer`：卷积增强版
- `iTransformer-GLU`：GLU 输出头版
- `DC-iTransformer-GLU`：卷积 + GLU 组合版
```

- [ ] **Step 3: 更新 `models/` 结构说明**

```text
models/
├── iTransformer.py
├── DC_iTransformer.py
├── iTransformer_GLU.py
├── DC_iTransformer_GLU.py
├── PatchTST.py
├── TimeXer.py
└── __init__.py
```

- [ ] **Step 4: 增加组合版命令示例**

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh2_96_96 \
  --model DC-iTransformer-GLU \
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

- [ ] **Step 5: 增加组合版脚本示例**

```markdown
- `scripts/long_term_forecast/ETT_script/DC_iTransformer_GLU_ETTh2.sh`
- `scripts/short_term_forecast/DC_iTransformer_GLU_M4.sh`
```

- [ ] **Step 6: 验证文档文本**

Run: `python -c "from pathlib import Path; text=(Path('README.md').read_text(encoding='utf-8') + Path('README_zh.md').read_text(encoding='utf-8')); assert 'DC-iTransformer-GLU' in text and 'models/DC_iTransformer_GLU.py' in text"`

Expected: 无报错

---

### Task 5: 验证组合版不会误加载单模块 checkpoint

**Files:**

- Modify: `tests/test_dc_itransformer_glu.py`

- [ ] **Step 1: 新增基线 checkpoint 不兼容测试**

```python
def test_rejects_baseline_projection_head(self):
    cfg = make_configs()
    model = Model(cfg)
    state = dict(model.state_dict())
    state["projection.weight"] = torch.randn(cfg.pred_len, cfg.d_model)
    state["projection.bias"] = torch.randn(cfg.pred_len)
    with self.assertRaises(RuntimeError):
        model.load_state_dict(state, strict=True)
```

- [ ] **Step 2: 新增 GLU-only checkpoint 不兼容测试**

```python
def test_rejects_glu_only_checkpoint_without_local_cnn(self):
    cfg = make_configs()
    model = Model(cfg)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("local_cnn.")}
    with self.assertRaises(RuntimeError):
        model.load_state_dict(state, strict=True)
```

- [ ] **Step 3: 运行测试，确认通过**

Run: `python -m unittest tests.test_dc_itransformer_glu -v`

Expected: PASS

---

### Task 6: 最终回归检查

**Files:**

- Test: `tests/test_itransformer_baseline.py`
- Test: `tests/test_dc_itransformer_alias.py`
- Test: `tests/test_itransformer_dc.py`
- Test: `tests/test_itransformer_glu.py`
- Test: `tests/test_dc_itransformer_glu.py`
- Test: `tests/test_patchtst_forecast_only.py`

- [ ] **Step 1: 运行全量测试**

Run: `python -m unittest discover -s tests -p "test_*.py" -v`

Expected: 全部 PASS

- [ ] **Step 2: 运行白名单检查**

Run: `python -c "from exp.exp_basic import ALLOWED_MODELS; print(sorted(ALLOWED_MODELS))"`

Expected: 包含 `DC-iTransformer-GLU`

- [ ] **Step 3: 运行 CLI 帮助检查**

Run: `python run.py --help`

Expected: `--model` 中包含 `DC-iTransformer-GLU`

- [ ] **Step 4: 运行文本一致性检查**

Run: `rg "DC-iTransformer-GLU|iTransformer-GLU|DC-iTransformer|iTransformer" README.md README_zh.md scripts`

Expected: 四类名称都存在，无覆盖关系

---

## Self-Review

- 规格覆盖：
  - 组合版模型文件：Task 1
  - 模型注册与 CLI：Task 2
  - 成对脚本：Task 3
  - README/README_zh：Task 4
  - checkpoint 不兼容验证：Task 5
  - 全量回归：Task 6
- Placeholder 扫描：无 `TODO`、`TBD`、`implement later`
- 一致性检查：
  - 对外名称统一为 `DC-iTransformer-GLU`
  - 文件名统一为 `models/DC_iTransformer_GLU.py`
  - 现有三个模型保持不动，仅新增组合版
