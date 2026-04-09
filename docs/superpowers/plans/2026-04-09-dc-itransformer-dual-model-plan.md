# DC-iTransformer Dual Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在同一仓库中同时保留原始 `iTransformer` 基线与改进后的 `DC-iTransformer`，让两者可以在相同数据集、相同参数设置下直接运行并做公平对比，同时验证原始 `iTransformer` 相关 checkpoints 仍可正常加载和前向。

**Architecture:** 保留 `models/iTransformer.py` 作为原始预测版基线；新增 `models/DC_iTransformer.py` 保存当前带局部卷积增强的版本；在模型注册和命令行入口中同时暴露 `iTransformer` 与 `DC-iTransformer` 两个可运行名字；通过新增脚本、更新 README 和独立测试来保证两条模型线清晰分离且都可用。

**Tech Stack:** Python, PyTorch, 当前仓库的 `run.py` / `exp/exp_basic.py` 模型注册机制, `unittest`, PowerShell, checkpoint smoke loading

---

## File Structure

**Create:**

- `models/DC_iTransformer.py`
- `tests/test_itransformer_baseline.py`
- `tests/test_dc_itransformer_alias.py`
- `docs/superpowers/plans/2026-04-09-dc-itransformer-dual-model-plan.md`
- 新增一组 `scripts/**/DC_iTransformer*.sh` 或 `scripts/**/DC-iTransformer*.sh` 对应脚本

**Modify:**

- `models/iTransformer.py`
- `exp/exp_basic.py`
- `run.py`
- `README.md`
- `README_zh.md`
- `tests/test_itransformer_dc.py`
- 现有 `scripts/.../iTransformer*.sh` 仅在必要时调整注释，不覆盖基线语义

**Verify against existing assets:**

- `checkpoints/*iTransformer*/checkpoint.pth`

---

### Task 1: 恢复原始 `iTransformer` 基线并拆分出 `DC-iTransformer`

**Files:**

- Create: `models/DC_iTransformer.py`
- Modify: `models/iTransformer.py`
- Test: `tests/test_itransformer_baseline.py`
- Test: `tests/test_dc_itransformer_alias.py`

- [ ] **Step 1: 写基线行为测试**

```python
import unittest
from types import SimpleNamespace
import torch

from models.iTransformer import Model as BaselineModel


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


class TestITransformerBaseline(unittest.TestCase):
    def test_baseline_has_no_local_cnn(self):
        model = BaselineModel(make_configs())
        self.assertFalse(hasattr(model, "local_cnn"))

    def test_baseline_forecast_shape(self):
        model = BaselineModel(make_configs())
        x_enc = torch.randn(2, 96, 7)
        x_mark_enc = torch.randn(2, 96, 4)
        x_dec = torch.randn(2, 72, 7)
        x_mark_dec = torch.randn(2, 72, 4)
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        self.assertEqual(out.shape, (2, 24, 7))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行基线测试，确认先失败**

Run: `python -m unittest tests.test_itransformer_baseline`

Expected: FAIL，因为当前 `models/iTransformer.py` 仍带 `local_cnn`

- [ ] **Step 3: 新建 `models/DC_iTransformer.py`，复制当前带 `local_cnn` 的实现**

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
            raise ValueError("DC-iTransformer only supports forecasting tasks in this repository.")

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
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, n_vars = x_enc.shape
        x_enc = self.local_cnn(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]
```

- [ ] **Step 4: 将 `models/iTransformer.py` 恢复为原始预测版**

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
            raise ValueError("iTransformer only supports forecasting tasks in this repository.")

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
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        _, _, n_vars = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]
```

- [ ] **Step 5: 新增 DC 版本测试**

```python
import unittest
from types import SimpleNamespace
import torch

from models.DC_iTransformer import Model as DCModel


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


class TestDCITransformerAlias(unittest.TestCase):
    def test_dc_model_has_local_cnn(self):
        model = DCModel(make_configs())
        self.assertTrue(hasattr(model, "local_cnn"))
        self.assertEqual(model.local_cnn[0].groups, 7)

    def test_dc_model_forecast_shape(self):
        model = DCModel(make_configs())
        x_enc = torch.randn(2, 96, 7)
        x_mark_enc = torch.randn(2, 96, 4)
        x_dec = torch.randn(2, 72, 7)
        x_mark_dec = torch.randn(2, 72, 4)
        out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        self.assertEqual(out.shape, (2, 24, 7))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6: 运行测试，确认恢复与拆分通过**

Run: `python -m unittest tests.test_itransformer_baseline tests.test_dc_itransformer_alias tests.test_itransformer_dc -v`

Expected: 所有测试 PASS；其中原始版无 `local_cnn`，DC 版有 `local_cnn`

---

### Task 2: 暴露 `DC-iTransformer` 模型名并保留 `iTransformer`

**Files:**

- Modify: `exp/exp_basic.py`
- Modify: `run.py`

- [ ] **Step 1: 写失败验证脚本，确认当前入口还不能识别 `DC-iTransformer`**

```python
from exp.exp_basic import ALLOWED_MODELS

assert "iTransformer" in ALLOWED_MODELS
assert "DC-iTransformer" in ALLOWED_MODELS
```

- [ ] **Step 2: 运行验证脚本，确认先失败**

Run: `python -c "from exp.exp_basic import ALLOWED_MODELS; assert 'DC-iTransformer' in ALLOWED_MODELS"`

Expected: `AssertionError`

- [ ] **Step 3: 修改 `exp/exp_basic.py`，加入显式别名映射**

```python
ALLOWED_MODELS = {"iTransformer", "DC-iTransformer", "PatchTST", "TimeXer"}

MODEL_IMPORT_ALIASES = {
    "iTransformer": "models.iTransformer",
    "DC-iTransformer": "models.DC_iTransformer",
}


def _scan_models_directory(self):
    model_map = dict(MODEL_IMPORT_ALIASES)
    models_dir = "models"
    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]
                if module_name in {"iTransformer", "DC_iTransformer"}:
                    continue
                if module_name not in ALLOWED_MODELS:
                    continue
                model_map[module_name] = f"{models_dir}.{module_name}"
    return model_map
```

- [ ] **Step 4: 修改 `run.py` 的模型白名单**

```python
parser.add_argument(
    "--model",
    type=str,
    required=True,
    default="iTransformer",
    choices=["iTransformer", "DC-iTransformer", "PatchTST", "TimeXer"],
    help="forecasting model name",
)
```

- [ ] **Step 5: 运行入口检查**

Run: `python run.py --help`

Expected: `--model {iTransformer,DC-iTransformer,PatchTST,TimeXer}`

---

### Task 3: 新增 DC 对比脚本并保留原始 iTransformer 脚本

**Files:**

- Create: `scripts/long_term_forecast/ETT_script/DC_iTransformer_ETTh2.sh`
- Create: `scripts/long_term_forecast/Weather_script/DC_iTransformer.sh`
- Create: `scripts/long_term_forecast/Traffic_script/DC_iTransformer.sh`
- Create: `scripts/long_term_forecast/ECL_script/DC_iTransformer.sh`
- Create: `scripts/short_term_forecast/DC_iTransformer_M4.sh`
- Modify: 现有 `scripts/.../iTransformer*.sh` 仅在注释中明确“baseline”

- [ ] **Step 1: 复制现有 iTransformer 脚本作为 DC 脚本模板**

```bash
model_name=DC-iTransformer
```

- [ ] **Step 2: 为每个数据集脚本保留相同参数，仅改模型名**

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7
```

- [ ] **Step 3: 保留原有 iTransformer 脚本不覆盖**

Run: `rg "model_name=iTransformer|model_name=DC-iTransformer" scripts`

Expected: 能同时看到 baseline 和 DC 两类脚本

---

### Task 4: 更新 README 与中文文档

**Files:**

- Modify: `README.md`
- Modify: `README_zh.md`

- [ ] **Step 1: 更新支持模型列表**

```markdown
- `iTransformer`
- `DC-iTransformer`
- `PatchTST`
- `TimeXer`
```

- [ ] **Step 2: 更新仓库结构说明**

```text
models/
├── iTransformer.py
├── DC_iTransformer.py
├── PatchTST.py
├── TimeXer.py
└── __init__.py
```

- [ ] **Step 3: 更新示例命令**

```bash
python -u run.py --task_name long_term_forecast --model iTransformer ...
python -u run.py --task_name long_term_forecast --model DC-iTransformer ...
```

- [ ] **Step 4: 运行文本检查**

Run: `python -c "from pathlib import Path; text=(Path('README.md').read_text(encoding='utf-8') + Path('README_zh.md').read_text(encoding='utf-8')); assert 'DC-iTransformer' in text and 'models/DC_iTransformer.py' in text"`

Expected: 无报错

---

### Task 5: 验证原始 iTransformer checkpoints 仍可加载与前向

**Files:**

- Modify: `tests/test_itransformer_baseline.py`
- Verify: `checkpoints/*iTransformer*/checkpoint.pth`

- [ ] **Step 1: 添加 checkpoint 加载烟测脚本**

```python
import os
import re
import torch
from types import SimpleNamespace
from models.iTransformer import Model

root = r"d:\Projects\Time-Series-Library-main\checkpoints"
pattern = re.compile(
    r"_sl(?P<sl>\d+)_.*_pl(?P<pl>\d+)_dm(?P<dm>\d+)_nh(?P<nh>\d+)_el(?P<el>\d+)_dl(?P<dl>\d+)_df(?P<df>\d+)_.*_fc(?P<fc>\d+)_"
)

for name in sorted(os.listdir(root)):
    if "iTransformer" not in name:
        continue
    ckpt_path = os.path.join(root, name, "checkpoint.pth")
    if not os.path.exists(ckpt_path):
        continue
    match = pattern.search(name)
    assert match is not None, name
    args = SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=int(match.group("sl")),
        pred_len=int(match.group("pl")),
        enc_in=7,
        d_model=int(match.group("dm")),
        embed="timeF",
        freq="h",
        dropout=0.1,
        factor=int(match.group("fc")),
        n_heads=int(match.group("nh")),
        d_ff=int(match.group("df")),
        activation="gelu",
        e_layers=int(match.group("el")),
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model = Model(args).float()
    model.load_state_dict(state, strict=True)
```

- [ ] **Step 2: 先运行脚本，看恢复后的基线是否能加载旧 checkpoint**

Run: `python -c "<上面的脚本>"`

Expected: 无报错；若失败，输出首个失败目录名并暂停实现继续分析

- [ ] **Step 3: 补前向烟测**

```python
with torch.no_grad():
    x_enc = torch.randn(2, args.seq_len, 7)
    x_mark_enc = torch.randn(2, args.seq_len, 4)
    x_dec = torch.randn(2, max(args.pred_len, 1), 7)
    x_mark_dec = torch.randn(2, max(args.pred_len, 1), 4)
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
assert out.shape == (2, args.pred_len, 7)
```

- [ ] **Step 4: 运行完整 checkpoint 验证**

Run: `python -c "<加载 + 前向的完整脚本>"`

Expected: 每个 `iTransformer` checkpoint 都输出 `FORWARD_OK`

---

### Task 6: 最终回归检查

**Files:**

- Modify: `tests/test_itransformer_dc.py`
- Test: `tests/test_itransformer_baseline.py`
- Test: `tests/test_dc_itransformer_alias.py`

- [ ] **Step 1: 调整现有 DC 测试导入路径**

```python
from models.DC_iTransformer import Model
```

- [ ] **Step 2: 运行所有相关测试**

Run: `python -m unittest tests.test_itransformer_baseline tests.test_dc_itransformer_alias tests.test_itransformer_dc tests.test_patchtst_forecast_only -v`

Expected: 全部 PASS

- [ ] **Step 3: 运行入口级 smoke**

Run: `python -c "from exp.exp_basic import ALLOWED_MODELS; print(sorted(ALLOWED_MODELS))"`

Expected: 包含 `DC-iTransformer`, `iTransformer`, `PatchTST`, `TimeXer`

- [ ] **Step 4: 运行文本检查**

Run: `rg "DC-iTransformer|iTransformer" README.md README_zh.md scripts`

Expected: 能看到 baseline 与 DC 两类命名，并且不出现“DC 覆盖 baseline”的描述

---

## Self-Review

- 规格覆盖：
  - 双模型并存：Task 1, Task 2
  - `DC-iTransformer` 外部命名：Task 2, Task 3, Task 4
  - 原始源码恢复：Task 1
  - 公平对比脚本：Task 3
  - 原始 checkpoint 可运行性检查：Task 5
- Placeholder 扫描：计划中无 `TODO`、`TBD`、`implement later`
- 一致性检查：
  - 文件名统一为 `models/DC_iTransformer.py`
  - 对外模型名统一为 `DC-iTransformer`
  - 基线名统一为 `iTransformer`
