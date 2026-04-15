# GLU Gate 导出与可视化设计说明

## 目标

为 `iTransformer-GLU` 与 `DC-iTransformer-GLU` 增加一套轻量的 gate 导出与单独可视化能力，用于分析 GLU 模块为何导致指标下降。

## 方案

仅对 GLU 模型生效：

- `iTransformer-GLU`
- `DC-iTransformer-GLU`

在测试/推理阶段自动保存：

- `gate.npy`
- `gate_stats.txt`（或等价的简单统计文件）

保存内容用于后续分析 gate 行为，不影响非 GLU 模型，也不改动现有 `vis_results.py` 主流程。

## 可视化

单独新增脚本：

- `vis_gate.py`

读取某个实验结果目录中的 `gate.npy`，输出以下内容：

1. gate 时间序列图
2. gate 分布图
3. gate 与预测对照图

并输出简单统计：

- `gate_mean`
- `gate_std`
- `gate_min`
- `gate_max`

如果实现成本合适，再补充：

- 高峰区域 gate 均值
- 非高峰区域 gate 均值

## 运行方式

不重新完整训练。

直接基于现有 `iTransformer-GLU` 与 `DC-iTransformer-GLU` checkpoint：

1. 加载已有 checkpoint
2. 重新补跑一次 `test()`
3. 自动导出 gate 文件
4. 再运行 `vis_gate.py` 生成图

## 边界

- 不改非 GLU 模型的结果导出逻辑
- 不要求 `iTransformer` / `DC-iTransformer` 生成 gate 文件
- 不把 gate 可视化强行整合进现有 `vis_results.py`

## 预期价值

这套方案可以帮助判断：

- gate 是否整体偏小
- gate 是否在峰值时段显著变小
- gate 是否与预测值被压低存在明显对应关系

从而更接近定位 GLU 导致性能下降的原因。
