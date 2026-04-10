# DC-iTransformer-GLU 设计说明

## 背景

当前仓库中已经同时保留了：

- 原始基线 `iTransformer`
- `DC-iTransformer`
- `iTransformer-GLU`

本次需求是在此基础上继续新增一个**组合版**模型，将 DC 局部卷积增强模块与 GLU 门控输出头同时叠加到同一个变体中，用于和原始基线及另外两个单模块版本做对比。

## 目标

新增一个可运行模型：

- `DC-iTransformer-GLU`

它应满足：

- 同时包含 DC 前端与 GLU 输出头
- 输入输出接口保持与原始 `iTransformer` 一致
- 可以通过当前预测入口直接运行
- 可以与以下三个版本做公平对比：
  - `iTransformer`
  - `DC-iTransformer`
  - `iTransformer-GLU`

## 推荐方案

采用“新增独立模型文件”的方式实现。

文件层：

- `models/iTransformer.py`
  - 原始基线
- `models/DC_iTransformer.py`
  - 只加 DC
- `models/iTransformer_GLU.py`
  - 只加 GLU
- `models/DC_iTransformer_GLU.py`
  - 同时加 DC + GLU

对外模型名层：

- `iTransformer`
- `DC-iTransformer`
- `iTransformer-GLU`
- `DC-iTransformer-GLU`

说明：

- Python 文件名不使用连字符，因此组合版文件名使用 `DC_iTransformer_GLU.py`
- 命令行、README、脚本、实验记录中显示为 `DC-iTransformer-GLU`

## 为什么不用直接覆盖现有 DC 或 GLU 版本

如果直接覆盖现有 `DC-iTransformer` 或 `iTransformer-GLU`，会失去单模块版本，不利于判断：

- 只加 DC 是否有效
- 只加 GLU 是否有效
- 两者叠加是否继续提升

因此这次必须采用“四模型并存”的方式。

## 模块叠加方式

本次采用“直接叠加”的组合方式。

### 前端：DC 模块

沿用 `DC-iTransformer` 的前端局部卷积增强：

- 标准化后
- `x_enc` 从 `[B, L, N]` 转成 `[B, N, L]`
- 进入深度可分离卷积 `local_cnn`
- 再转回 `[B, L, N]`
- 送入 `enc_embedding`

### 中间：Encoder 主体

保持与基线一致：

- `enc_embedding`
- `encoder`

### 末端：GLU 输出头

不用单一 `projection`，改成：

- `self.proj_linear = nn.Linear(configs.d_model, configs.pred_len, bias=True)`
- `self.proj_gate = nn.Linear(configs.d_model, configs.pred_len, bias=True)`

前向逻辑：

1. `linear_out = self.proj_linear(enc_out)`
2. `gate_score = torch.sigmoid(self.proj_gate(enc_out))`
3. `gated_out = linear_out * gate_score`
4. 按原有逻辑 `.permute(0, 2, 1)`、切片、反归一化

## 输入输出接口

接口保持不变：

- 输入：`x_enc`，`[Batch, seq_len, variables]`
- encoder 输出：`enc_out`，`[Batch, variables, d_model]`
- 最终输出：`dec_out`，`[Batch, pred_len, variables]`

这意味着：

- 数据加载器不需要改
- 训练入口调用方式不需要改
- 更利于和其它三个版本做公平对比

## 需要修改的地方

### 1. 模型文件

- 新增 `models/DC_iTransformer_GLU.py`
- 不动现有三个模型文件

### 2. 模型注册

修改 `exp/exp_basic.py`：

- 新增 `DC-iTransformer-GLU`
- 为其提供到 `models.DC_iTransformer_GLU` 的显式映射

### 3. 命令行入口

修改 `run.py`：

- 在 `--model choices` 中加入 `DC-iTransformer-GLU`

后续可直接使用：

- `--model DC-iTransformer-GLU`

### 4. 脚本

保留当前三类脚本不动，再新增一套 `DC-iTransformer-GLU` 脚本：

- 长期预测脚本
- 短期预测脚本

要求与对应 baseline / DC / GLU 脚本参数一致，仅改模型名。

### 5. 文档

更新：

- `README.md`
- `README_zh.md`

文档中明确四条模型线：

- `iTransformer`：基线
- `DC-iTransformer`：卷积增强版
- `iTransformer-GLU`：GLU 输出头版
- `DC-iTransformer-GLU`：卷积 + GLU 组合版

### 6. 测试

建议新增覆盖：

- 组合版同时存在 `local_cnn`、`proj_linear`、`proj_gate`
- 不存在旧的 `projection`
- 前向输出形状保持为 `(2, pred_len, enc_in)`
- 非预测任务报错
- 入口注册能识别 `DC-iTransformer-GLU`

## checkpoint 影响

### 原始 iTransformer checkpoint

不能直接用于组合版，因为组合版相比基线：

- 多了 `local_cnn`
- 去掉了 `projection`
- 增加了 `proj_linear` / `proj_gate`

### DC-iTransformer checkpoint

也不能直接用于组合版，因为虽然前端 `local_cnn` 一致，但末端：

- DC 版仍是 `projection`
- 组合版是 `proj_linear` / `proj_gate`

### iTransformer-GLU checkpoint

同样不能直接用于组合版，因为虽然输出头一致，但前端：

- GLU 版没有 `local_cnn`
- 组合版有 `local_cnn`

### 结论

`DC-iTransformer-GLU` 应被视为一个新的独立模型，需要独立训练。

## 风险控制

为了保证对比有效，实施时应遵循：

- 不覆盖现有三个模型
- 不在单个文件里通过隐藏开关混合切换 DC / GLU
- 使用独立名字、独立脚本、独立结果目录
- 只叠加已确认的两个模块，不顺手改其它主体结构

## 最终建议

采用以下稳定结构：

- `iTransformer` = 原始基线
- `DC-iTransformer` = 只加 DC
- `iTransformer-GLU` = 只加 GLU
- `DC-iTransformer-GLU` = DC + GLU 组合版

这样最适合做四组实验对比：

- 基线
- 单模块 DC
- 单模块 GLU
- 双模块组合

这是当前最清晰、最利于复现和后续分析模块增益来源的方案。
