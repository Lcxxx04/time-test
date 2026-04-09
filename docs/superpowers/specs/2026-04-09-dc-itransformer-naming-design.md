# DC-iTransformer 命名与基线并存设计

## 背景

当前仓库中的 `models/iTransformer.py` 已经不是原始基线版本，而是加入了深度可分离局部卷积增强后的版本。为了在同一数据集、同一参数设置下对比原始 `iTransformer` 与改进后的模型性能，需要让两者在同一仓库中并存，并且对外命名清晰可区分。

## 目标

实现以下对比结构：

- `iTransformer`：原始基线模型，只保留预测相关部分
- `DC-iTransformer`：带局部卷积增强的改进模型

要求：

- 两个模型都能通过当前预测入口直接运行
- 脚本、命令行、文档中的模型名清晰区分
- 保持对比实验可复现，避免覆盖原始基线身份

## 推荐方案

采用“双模型并存，文件名与对外模型名分离”的方案。

文件层：

- `models/iTransformer.py`
  - 恢复为原始预测版基线
- `models/DC_iTransformer.py`
  - 保存当前带 `local_cnn` 的改进版

对外模型名层：

- `iTransformer`
- `DC-iTransformer`

说明：

- Python 文件名不使用连字符，因此改进版文件名使用 `DC_iTransformer.py`
- 训练入口、README、脚本、实验记录中可以对外显示为 `DC-iTransformer`

## 为什么不用只改当前模型名字

如果只是把当前 `iTransformer` 改名为 `DC-iTransformer` 而不恢复原始基线，会出现两个问题：

1. 原始 `iTransformer` 无法在同仓库中直接运行，失去公平对比基础。
2. 历史实验结果、脚本和 checkpoint 中的 `iTransformer` 名字会与当前实现含义不一致。

## 需要修改的地方

### 1. 模型文件

- 新增 `models/DC_iTransformer.py`
- 恢复 `models/iTransformer.py` 为不带 `local_cnn` 的预测版

### 2. 模型注册

修改 `exp/exp_basic.py`：

- 保留 `iTransformer`
- 新增 `DC-iTransformer`
- 为 `DC-iTransformer` 提供到 `models/DC_iTransformer.py` 的显式映射

这里不能完全依赖“文件名即模型名”的简单扫描逻辑，因为对外模型名 `DC-iTransformer` 与 Python 文件名 `DC_iTransformer.py` 不一致。

## 3. 命令行入口

修改 `run.py`：

- 在 `--model choices` 中保留 `iTransformer`
- 新增 `DC-iTransformer`

这样后续可以直接使用：

- `--model iTransformer`
- `--model DC-iTransformer`

## 4. 脚本

不覆盖原有 `iTransformer` 脚本，而是新增对应的 `DC-iTransformer` 脚本。

建议：

- 保留现有 `iTransformer` 脚本作为基线实验
- 复制并新增 `DC-iTransformer` 脚本作为改进版实验

这样原始版和改进版可以在相同数据集、相同参数下并排运行。

## 5. 文档

更新：

- `README.md`
- `README_zh.md`

文档中明确：

- `iTransformer` 是基线
- `DC-iTransformer` 是改进版

## 6. 测试

建议拆分测试职责：

- `iTransformer`：验证其为预测专用基线并能正常前向
- `DC-iTransformer`：验证 `local_cnn` 结构与前向行为

## 原始 iTransformer 源码恢复策略

基于当前仓库上下文，原始基线版本可以通过“移除 `local_cnn` 相关逻辑，并保留预测主路径”来恢复。

这意味着：

- 通常不需要用户额外提供原始源码文件
- 但恢复的是“当前仓库语义下的原始预测版”
- 如果用户过去还对原始版做过额外私有修改，则无法保证逐字节一致恢复

因此：

- 若仅做过 DC 增强，则可直接恢复
- 若曾对原始版有其它私改，建议用户提供那份原文件作为基线来源

## checkpoint 与实验记录影响

### 旧 checkpoint

- 现有目录名中大量包含 `iTransformer`
- 它们应继续视为原始版或历史版实验记录，不应被强行重命名

### 新实验记录

新增 `DC-iTransformer` 后：

- 新训练产生的结果目录、checkpoint 目录名会包含 `DC-iTransformer`
- 这样可以和旧 `iTransformer` 结果自然区分

## 风险控制

为了避免影响已有结果与基线对比，实施时应遵循：

- 不覆盖原始 `iTransformer` 的外部模型名
- 不用 `DC-iTransformer` 去替代 `iTransformer`
- 原始版与改进版并存
- 原始脚本保留，改进版脚本新增

## 最终建议

采用以下稳定结构：

- `iTransformer` = 原始基线
- `DC-iTransformer` = 改进版
- 两者同仓库并存、同入口可跑、同参数可对比

这是当前最适合做公平性能对比、结果归档和后续复现实验的方案。
