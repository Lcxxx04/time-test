# Time Series Library (Forecasting Edition)

This repository is a trimmed forecasting-focused version of TSLib.

It now supports only:

- `long_term_forecast`
- `short_term_forecast`
- `iTransformer`, `DC-iTransformer`, `iTransformer-GLU`, `PatchTST`, `TimeXer`

The current workflow is intended for electricity-load forecasting experiments, including household power consumption use cases, while remaining compatible with the existing benchmark-style data pipeline in the repository.

## Supported Scope

### Tasks

- Long-term forecasting
- Short-term forecasting

### Models

- `iTransformer` (baseline), `DC-iTransformer` (convolution-enhanced), `iTransformer-GLU` (GLU output head): `models/iTransformer.py`, `models/DC_iTransformer.py`, `models/iTransformer_GLU.py`
- `PatchTST`
- `TimeXer`

## Repository Layout

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
│   ├── iTransformer_GLU.py
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

## Installation

1. Create an environment.

```bash
conda create -n tslib python=3.11
conda activate tslib
```

2. Install dependencies.

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Adjust the PyTorch build to match your local CUDA setup if needed.

## Command Line Entry

`run.py` now accepts only the following task names:

- `long_term_forecast`
- `short_term_forecast`

`run.py` now accepts only the following model names:

- `iTransformer`
- `DC-iTransformer`
- `iTransformer-GLU`
- `PatchTST`
- `TimeXer`

## Example Commands

The examples below follow representative script configurations. For exact benchmark reproduction, prefer the paired shell scripts under `scripts/`.

### Long-term Forecasting

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

### Short-term Forecasting

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

```bash
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id m4_Monthly \
  --model iTransformer-GLU \
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

## Script Entry Points

The repository keeps forecasting scripts for the supported models. The paths below are representative entry points; additional dataset-specific scripts remain under `scripts/long_term_forecast/` and `scripts/short_term_forecast/`.

Examples:

- `scripts/long_term_forecast/ETT_script/iTransformer_ETTh2.sh`
- `scripts/long_term_forecast/ETT_script/DC_iTransformer_ETTh2.sh`
- `scripts/long_term_forecast/ETT_script/iTransformer_GLU_ETTh2.sh`
- `scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh`
- `scripts/long_term_forecast/ETT_script/TimeXer_ETTh1.sh`
- `scripts/short_term_forecast/iTransformer_M4.sh`
- `scripts/short_term_forecast/DC_iTransformer_M4.sh`
- `scripts/short_term_forecast/iTransformer_GLU_M4.sh`

## Notes

- The public data pipeline under `data_provider/` is intentionally kept conservative to avoid breaking forecasting runs.
- Some shared utilities and layers remain in place even if they are broader than the supported-model surface area.
- This repository no longer exposes imputation, anomaly detection, classification, or zero-shot forecasting through the maintained CLI and script entry points.
- Built-in tests are executed with `python -m unittest`; no `pytest` dependency is required for the current test suite.

## Citation

If this repository is useful in your work, please cite:

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
