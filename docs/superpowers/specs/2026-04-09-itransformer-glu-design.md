# iTransformer-GLU 设计说明

## 背景

当前仓库已经同时保留了：

- 原始基线 `iTransformer`
- 改进版 `DC-iTransformer`

本次需求是在**原始基线 `iTransformer`** 的基础上继续派生一个新版本，用于与原始基线做公平对比。这个新版本不引入卷积增强，而是在输出端用 GLU 门控机制替代原来的单一线性预测头。

## 目标

新增一个可运行模型：

- `iTransformer-GLU`

它应满足：

- 输入输出接口保持与原始 `iTransformer` 一致
- 仅替换输出头，不修改 embedding 与 encoder 主体结构
- 可以通过当前预测入口直接运行
- 可以与原始基线 `iTransformer` 用同一数据、同一参数、仅改模型名进行对比

## 推荐方案

采用“新增独立模型文件”的方式实现。

文件层：

- `models/iTransformer.py`
  - 保持原始基线版，不动
- `models/iTransformer_GLU.py`
  - 新增 GLU 门控输出版本

对外模型名层：

- `iTransformer`
- `iTransformer-GLU`

说明：

- Python 文件名不使用连字符，因此新文件名使用 `iTransformer_GLU.py`
- 命令行、README、脚本、实验记录中显示为 `iTransformer-GLU`

## 为什么不用直接改原始 iTransformer

如果直接把 `models/iTransformer.py` 改成 GLU 版，会失去可直接运行的原始基线，不利于后续做公平对比。

因此本次必须采用“双模型并存”的方式：

- `iTransformer` = 原始基线
- `iTransformer-GLU` = 输出头改进版

## 模块改动范围

本次只改输出头，不改输入端与 encoder 主体。

### 原始结构

原始基线在 encoder 输出后使用：

- `self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)`

### GLU 结构

替换为两个并行线性层：

- `self.proj_linear = nn.Linear(configs.d_model, configs.pred_len, bias=True)`
- `self.proj_gate = nn.Linear(configs.d_model, configs.pred_len, bias=True)`

前向流程：

1. `enc_out` 的形状为 `[Batch, variables, d_model]`
2. 计算主预测值：
   - `linear_out = self.proj_linear(enc_out)`
3. 计算门控分数：
   - `gate_score = torch.sigmoid(self.proj_gate(enc_out))`
4. 融合：
   - `gated_out = linear_out * gate_score`
5. 按原有逻辑：
   - `.permute(0, 2, 1)`
   - 切片
   - 进入反归一化流程

## 输入输出接口

保持不变：

- 输入：`x_enc`，`[Batch, seq_len, variables]`
- encoder 输出：`enc_out`，`[Batch, variables, d_model]`
- 最终输出：`dec_out`，`[Batch, pred_len, variables]`

这意味着：

- 训练脚本参数不需要因 GLU 版本而改变
- 数据加载器不需要改
- 与基线的公平对比更容易成立

## 需要修改的地方

### 1. 模型文件

- 新增 `models/iTransformer_GLU.py`
- 保持 `models/iTransformer.py` 不变

### 2. 模型注册

修改 `exp/exp_basic.py`：

- 保留 `iTransformer`
- 新增 `iTransformer-GLU`
- 为 `iTransformer-GLU` 提供到 `models.iTransformer_GLU` 的显式映射

原因：

- 对外模型名带连字符
- Python 模块路径不能直接使用连字符

### 3. 命令行入口

修改 `run.py`：

- 在 `--model choices` 中加入 `iTransformer-GLU`

后续可直接使用：

- `--model iTransformer`
- `--model iTransformer-GLU`

### 4. 脚本

保留现有 `iTransformer` 脚本作为基线。

新增一套对应的 `iTransformer-GLU` 脚本：

- 保持数据集和参数一致
- 仅修改模型名

这样可确保公平对比：

- 同一数据集
- 同一超参数
- 唯一变化是输出头结构

### 5. 文档

更新：

- `README.md`
- `README_zh.md`

文档中明确：

- `iTransformer` 是原始基线
- `iTransformer-GLU` 是 GLU 输出头版本

### 6. 测试

建议新增测试覆盖：

- `iTransformer-GLU` 存在 `proj_linear` 与 `proj_gate`
- `proj_gate` 输出经过 `sigmoid`
- 前向输出形状保持为 `(2, pred_len, enc_in)`
- 入口注册能识别 `iTransformer-GLU`

## checkpoint 影响

### 原始 iTransformer checkpoint

原始基线 `iTransformer` 的 checkpoint 不受影响，只要不修改 `models/iTransformer.py` 结构，就仍可继续加载与前向。

### GLU 版本 checkpoint

原始基线 checkpoint **不能直接作为 `iTransformer-GLU` 的权重使用**，因为：

- 原始版有 `projection`
- GLU 版改成 `proj_linear` 与 `proj_gate`

因此参数名和结构不一致，`strict=True` 加载会失败。

这不影响公平对比，因为：

- 基线版单独训练
- GLU 版单独训练
- 两者按相同训练设置独立比较即可

## 风险控制

为了保证对比有效，实施时应遵循：

- 不覆盖原始 `iTransformer`
- 不在同一个文件里通过隐藏开关切换 GLU
- 独立命名、独立脚本、独立结果目录
- 仅修改输出头，不顺手改 encoder 主体

## 最终建议

采用以下稳定结构：

- `iTransformer` = 原始基线
- `iTransformer-GLU` = 仅替换输出头的门控版本

这样最适合做：

- 同数据集
- 同参数
- 同训练设置
- 只比较“输出头是否改进了性能”

这是当前最清晰、最利于复现和对比实验的方案。
