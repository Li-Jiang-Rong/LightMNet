# LightMNet3 - Lightweight Remote Sensing Change Detection

LightMNet3 是一个轻量级遥感影像变化检测模型，基于 ResNet18 骨干网络，结合参数无关度量增强、鲁棒语义交叉融合和轻量级混合注意力机制，在 CDD 和 LEVIR-CD 数据集上取得了优异的检测性能。

## 实验结果

| 数据集 | F1 | 精确率 (PRE) | 召回率 (RECALL) | 总体精度 (OA) |
|--------|------|------|---------|--------|
| CDD（最优） | **95.88** | **94.73** | **97.05** | **99.00** |
| LEVIR-CD（最优） | **89.36** | 89.84 | **88.88** | **99.11** |

## 环境要求

- Python 3.13+
- PyTorch 2.x
- CUDA 11.8+
- torchvision
- albumentations
- tqdm
- numpy
- matplotlib

## 项目结构

### 模型定义

| 文件 | 说明 |
|------|------|
| [lightmnet3.py](lightmnet3.py) | LightMNet3 完整模型定义，包含四大模块：ResNet18 骨干网络、参数无关度量增强模块（PFME）、鲁棒语义交叉融合模块（RSCF）、轻量级混合注意力模块（LHA） |
| [lightMnet.py](lightMnet.py) | LightMNet 模型定义（早期版本） |

### 消融实验模型

| 文件 | 说明 |
|------|------|
| [lightmnet3_ablation_no_attention.py](lightmnet3_ablation_no_attention.py) | 移除注意力模块的消融模型（验证 LHA 模块有效性） |
| [lightmnet3_ablation_no_param.py](lightmnet3_ablation_no_param.py) | 移除参数无关度量增强模块的消融模型（验证 PFME 模块有效性） |
| [lightmnet3_ablation_no_semantic.py](lightmnet3_ablation_no_semantic.py) | 移除鲁棒语义交叉融合模块的消融模型（验证 RSCF 模块有效性） |

### 公共模块

| 文件 | 说明 |
|------|------|
| [lightmnet3_train_ablation_common.py](lightmnet3_train_ablation_common.py) | 消融实验通用训练/评估模块，被所有消融训练脚本共享 |

### 训练脚本

| 文件 | 说明 |
|------|------|
| [lightmnet3-train2.py](lightmnet3-train2.py) | LightMNet3 在 LEVIR-CD 数据集上的训练脚本 |
| [lightmnet3-train3.py](lightmnet3-train3.py) | LightMNet3 在 CDD 数据集上的训练脚本 |

### CDD 消融训练脚本

| 文件 | 说明 |
|------|------|
| [train_cdd_ablation_no_attention.py](train_cdd_ablation_no_attention.py) | 移除注意力模块后在 CDD 上的消融训练 |
| [train_cdd_ablation_no_param.py](train_cdd_ablation_no_param.py) | 移除参数无关度量增强模块后在 CDD 上的消融训练 |
| [train_cdd_ablation_no_semantic.py](train_cdd_ablation_no_semantic.py) | 移除鲁棒语义交叉融合模块后在 CDD 上的消融训练 |

### LEVIR-CD 消融训练脚本

| 文件 | 说明 |
|------|------|
| [train_levircd_ablation_no_attention.py](train_levircd_ablation_no_attention.py) | 移除注意力模块后在 LEVIR-CD 上的消融训练 |
| [train_levircd_ablation_no_param.py](train_levircd_ablation_no_param.py) | 移除参数无关度量增强模块后在 LEVIR-CD 上的消融训练 |
| [train_levircd_ablation_no_semantic.py](train_levircd_ablation_no_semantic.py) | 移除鲁棒语义交叉融合模块后在 LEVIR-CD 上的消融训练 |

### 参数消融实验

| 文件 | 说明 |
|------|------|
| [param_experiment_train.py](param_experiment_train.py) | 参数消融实验通用训练脚本，支持通过命令行参数配置学习率、正类权重、权重衰减、冻结策略和优化器 |
| [run_param_experiments.py](run_param_experiments.py) | 参数实验队列调度器，串行执行 19 组实验并支持断点续跑 |
| [eval_param_experiments.py](eval_param_experiments.py) | 参数实验结果汇总与评估脚本 |

### 模型评估

| 文件 | 说明 |
|------|------|
| [eval_single_weight_on_test.py](eval_single_weight_on_test.py) | 单权重在测试集上的评估脚本 |
| [eval_ablation_best_on_test.py](eval_ablation_best_on_test.py) | 消融实验最优模型在测试集上的评估 |
| [eval_train2_on_levircd.py](eval_train2_on_levircd.py) | train2 权重在 LEVIR-CD 上的评估 |
| [eval_train2_train3_best_on_test.py](eval_train2_train3_best_on_test.py) | train2 和 train3 最优权重在测试集上的评估 |
| [eval_weight_train_val_speed.py](eval_weight_train_val_speed.py) | 权重训练/验证速度评估 |

### 消融实验调度

| 文件 | 说明 |
|------|------|
| [run_ablation_queue.py](run_ablation_queue.py) | 消融实验队列调度器 |
| [report_ablation_status.py](report_ablation_status.py) | 消融实验状态报告脚本 |

### 其他

| 文件 | 说明 |
|------|------|
| [infer_binary_masks.py](infer_binary_masks.py) | 批量推理生成二值变化图 |
| [param_experiment_latex.tex](param_experiment_latex.tex) | 参数消融实验的 LaTeX 报告源码（用于论文） |

## 快速开始

### 训练 LightMNet3（CDD 数据集）

```bash
python lightmnet3-train3.py
```

### 训练 LightMNet3（LEVIR-CD 数据集）

```bash
python lightmnet3-train2.py
```

### 评估单权重

```bash
python eval_single_weight_on_test.py --weight_path /path/to/weight.pth
```

### 运行参数消融实验

```bash
python run_param_experiments.py
```

## 数据集

- **CDD**（Change Detection Dataset）：Google Earth 遥感影像变化检测数据集
- **LEVIR-CD**：大规模遥感建筑变化检测数据集

数据集目录结构：

```
CDD/
  train/
  val/
  test/
LEVIR-CD/
  train/
  val/
  test/
```

## 消融实验结果（LEVIR-CD）

| 配置 | OA (%) | PRE (%) | RECALL (%) | F1 (%) |
|------|--------|---------|------------|--------|
| w/o PFME | 98.70 | 88.51 | 85.53 | 86.99 |
| w/o RSCF | 98.65 | 88.30 | 85.12 | 86.68 |
| w/o LHA | 98.80 | 89.12 | 86.45 | 87.76 |
| LightMNet（完整） | **99.08** | **89.00** | **89.03** | **89.01** |

## 参数消融实验摘要（CDD）

各超参数维度对性能影响的重要性排序：

**优化器 > 冻结策略 > 学习率 > 正类权重 > 权重衰减**

CDD 最优配置：`lr=1e-4`、`pos_weight=2.0`、`weight_decay=1e-2`、`freeze_layer0_1`、`AdamW` → **F1=95.88**

## 引用

如果您在研究中使用了 LightMNet3，请引用本仓库。

### 论文来源

本实验复现自武汉大学博士学位论文：

**《基于语义特征增强的高分辨率遥感影像变化检测及模型优化研究》**

- 研究生姓名：成洪权
- 指导教师：吴华意 教授
- 学科专业：地图制图学与地理信息工程
- 研究方向：遥感影像变化检测

## 许可证

本项目仅供学术研究使用。
