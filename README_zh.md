# 时间序列库（预测版）

这是一个面向预测任务裁剪后的 TSLib 版本。

当前仓库只保留：

- `long_term_forecast`
- `short_term_forecast`
- `iTransformer` / `DC-iTransformer`
- `PatchTST`
- `TimeXer`

当前工作流主要服务于电力负荷预测实验，包括家庭用电量预测场景，同时继续兼容仓库中现有的 benchmark 风格数据通路。

## 支持范围

### 任务

- 长期预测
- 短期预测

### 模型

- `iTransformer`（基线）与 `DC-iTransformer`（改进版），分别对应 `models/iTransformer.py` 与 `models/DC_iTransformer.py`
- `PatchTST`
- `TimeXer`

## 仓库结构

```text
Time-Series-Library/
├── run.py
├── exp/
│   ├── exp_basic.py
│   ├── exp_long_term_forecasting.py
│   └── exp_short_term_forecasting.py
├── models/
│   ├── iTransformer.py
│   ├── DC_iTransformer.py
│   ├── PatchTST.py
│   ├── TimeXer.py
│   └── __init__.py
├── data_provider/
├── layers/
├── utils/
└── scripts/
    ├── long_term_forecast/
    └── short_term_forecast/
```

## 安装

1. 创建环境。

```bash
conda create -n tslib python=3.11
conda activate tslib
```

2. 安装依赖。

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

如果本地 CUDA 版本不同，请按实际环境调整 PyTorch 安装命令。

## 命令行入口

`run.py` 现在只接受以下任务名：

- `long_term_forecast`
- `short_term_forecast`

`run.py` 现在只接受以下模型名：

- `iTransformer`
- `DC-iTransformer`
- `PatchTST`
- `TimeXer`

## 示例命令

下面的命令示例尽量与代表性的脚本配置保持一致；如果你要严格复现实验，优先使用 `scripts/` 目录下成对的 shell 脚本。

### 长期预测

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh2_96_96 \
  --model iTransformer \
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

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh2_96_96 \
  --model DC-iTransformer \
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

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_96_96 \
  --model PatchTST \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des Exp \
  --n_heads 2
```

```bash
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ETTh1_96_96 \
  --model TimeXer \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des exp \
  --d_model 256 \
  --batch_size 4
```

### 短期预测

```bash
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id m4_Monthly \
  --model iTransformer \
  --data m4 \
  --root_path ./dataset/m4 \
  --seasonal_patterns Monthly \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 512 \
  --factor 3 \
  --e_layers 2 \
  --d_layers 1 \
  --batch_size 16 \
  --des Exp \
  --learning_rate 0.001 \
  --loss SMAPE
```

```bash
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id m4_Monthly \
  --model DC-iTransformer \
  --data m4 \
  --root_path ./dataset/m4 \
  --seasonal_patterns Monthly \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 512 \
  --factor 3 \
  --e_layers 2 \
  --d_layers 1 \
  --batch_size 16 \
  --des Exp \
  --learning_rate 0.001 \
  --loss SMAPE
```

## 脚本入口

仓库当前保留的是这些支持模型的预测脚本。下面列的是代表性入口，更多数据集脚本仍保留在 `scripts/long_term_forecast/` 和 `scripts/short_term_forecast/` 下。

示例：

- `scripts/long_term_forecast/ETT_script/iTransformer_ETTh2.sh`
- `scripts/long_term_forecast/ETT_script/DC_iTransformer_ETTh2.sh`
- `scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh`
- `scripts/long_term_forecast/ETT_script/TimeXer_ETTh1.sh`
- `scripts/short_term_forecast/iTransformer_M4.sh`
- `scripts/short_term_forecast/DC_iTransformer_M4.sh`

## 说明

- `data_provider/` 下的数据管线采取保守保留策略，避免误删后影响预测任务运行。
- `layers/` 与 `utils/` 中仍保留了一部分共享公共件，即使它们的表面范围大于当前支持的模型集合。
- 仓库已不再通过维护中的 CLI 与脚本入口暴露插补、异常检测、分类和零样本预测。
- 当前测试套件使用 `python -m unittest` 运行，不依赖 `pytest`。

## 引用

如果本仓库对你的工作有帮助，可引用：

```bibtex
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@article{wang2024tssurvey,
  title={Deep Time Series Models: A Comprehensive Survey and Benchmark},
  author={Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Yong Liu and Mingsheng Long and Jianmin Wang},
  journal={arXiv preprint arXiv:2407.13278},
  year={2024}
}
```
