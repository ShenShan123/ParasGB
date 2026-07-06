# rcg_r 等效电阻预测项目重构需求文档

## 项目背景

将 `rcg_r` 目录下的代码重构为仿照 `rcg` 的风格，用于等效电阻预测任务。数据集路径为 `sram_r/`，与 `rcg` 使用的 `sram/` 数据集结构类似但预测目标不同。

## 核心差异分析

### 1. 数据集差异

| 特性 | rcg (电容预测) | rcg_r (电阻预测) |
|------|---------------|-----------------|
| 数据路径 | `../sram/` | `../sram_r/` |
| 目标边类型 | `cc_p2n`, `cc_p2p`, `cc_n2n` (耦合电容) | `r_p2p` (PIN到PIN电阻) |
| 边类型数量 | 5种 (含3种寄生耦合) | 3种 (含1种电阻边) |
| 标签归一化 | 对数变换 `log10(y * 1e21) / 6` | 百分位数过滤 + Min-Max归一化 |

### 2. 图结构差异

**rcg 边类型子图:**
```python
hg.edge_type_subgraph([
    ('device', 'device-pin', 'pin'),
    ('pin', 'pin-net', 'net'),
    ('pin', 'cc_p2n', 'net'),    # 耦合电容
    ('pin', 'cc_p2p', 'pin'),    # 耦合电容
    ('net', 'cc_n2n', 'net'),    # 耦合电容
])
```

**rcg_r 边类型子图:**
```python
hg.edge_type_subgraph([
    ('device', 'device-pin', 'pin'),
    ('pin', 'pin-net', 'net'),
    ('pin', 'r_p2p', 'pin'),     # 等效电阻
])
```

---

## 用户故事

### US-1: 数据集参数化加载
**作为** 研究人员  
**我希望** 能够通过命令行参数灵活指定训练集和测试集  
**以便** 方便进行不同数据集组合的实验

**验收标准:**
- 1.1 支持 `--data_dir` 参数指定数据集根目录 (默认 `../sram_r/`)
- 1.2 支持 `--train_dataset` 参数指定训练数据集 (如 `ssram+digtime`)
- 1.3 支持 `--test_dataset` 参数指定测试数据集 (如 `sandwich+ultra8t`)
- 1.4 数据集类能够正确解析 `train_cases` 和 `test_cases` 参数
- 1.5 数据集类能够追踪 `num_train` 和 `num_test` 数量

### US-2: 模型架构统一
**作为** 研究人员  
**我希望** rcg_r 支持与 rcg 完全相同的模型架构  
**以便** 能够公平对比电容预测和电阻预测的性能

**验收标准:**
- 2.1 支持基础 GNN 模型: `clustergcn`, `gcn`, `sage`, `gat`, `resgatedgcn`, `gine`, `pna`
- 2.2 支持 Transformer 模型: `sgformer`, `polynormer`
- 2.3 支持自定义 GNN+ 层: `CustomGatedGCN`, `CustomGCNConv`, `CustomGINEConv`
- 2.4 移除 `GPSLayer` 和 `gps_attention` 相关代码
- 2.5 复制 `rcg/layer.py` 到 `rcg_r/layer.py`

### US-3: 动态训练/测试集划分
**作为** 研究人员  
**我希望** 采样模块能够根据数据集配置动态划分训练和测试集  
**以便** 避免硬编码导致的灵活性问题

**验收标准:**
- 3.1 使用 `dataset.num_train` 动态获取训练图数量
- 3.2 训练集循环: `for i in range(num_train)`
- 3.3 测试集循环: `for i in range(num_train, len(dataset))`
- 3.4 移除硬编码的 `Batch.from_data_list([train_graph_0, train_graph_1])`

### US-4: 训练日志记录
**作为** 研究人员  
**我希望** 训练过程能够自动保存日志文件  
**以便** 追踪实验结果和复现实验

**验收标准:**
- 4.1 日志保存到 `logs/` 目录
- 4.2 日志文件名包含时间戳和关键参数
- 4.3 记录训练/验证/测试的损失和指标

### US-5: t-SNE 可视化 (可选)
**作为** 研究人员  
**我希望** 能够可视化训练/验证/测试集的特征分布  
**以便** 分析数据集的分布差异

**验收标准:**
- 5.1 支持 `--plot_tsne` 参数开启可视化
- 5.2 支持 `--tsne_max_samples` 参数控制采样数量
- 5.3 生成的图片保存到 `imgs/` 目录

---

## 需要修改的文件清单

### 1. `sram_dataset.py` - 数据集加载模块

#### 1.1 类初始化参数
- **当前状态**: 使用单一 `name` 参数，数据路径硬编码
- **目标状态**: 仿照 rcg 使用 `train_cases` 和 `test_cases` 分离训练/测试集

#### 1.2 `sram_graph_load()` 方法
- **边类型过滤**: 保持 `'r'` 关键字来识别电阻边
- **标签过滤逻辑**: 保持百分位数过滤 `torch.quantile(tar_edge_y, 0.9)`

#### 1.3 `norm_nfeat()` 方法
- **边标签归一化**: 保持按数据集分别做 Min-Max 归一化

#### 1.4 `performat_SramDataset()` 函数
- 参数签名需要与 rcg 保持一致，支持 `train_cases` 和 `test_cases`

---

### 2. `main.py` - 主程序入口

#### 2.1 命令行参数
- **添加参数**:
  - `--data_dir`: 数据集根目录 (默认 `../sram_r/`)
  - `--train_dataset`: 训练数据集名称
  - `--test_dataset`: 测试数据集名称
  - `--plot_tsne`: t-SNE 可视化开关
  - `--tsne_max_samples`: t-SNE 采样数量

- **移除/简化参数**:
  - `--dataset`: 改为 `--train_dataset` + `--test_dataset`
  - GSL 相关参数
  - SGRL 相关参数
  - GPS 相关参数 (`local_gnn_type`, `global_model_type`, `attn_dropout`, `layer_norm`, `batch_norm`, `num_heads`)

#### 2.2 数据集加载调用
```python
# 当前 rcg_r
dataset = performat_SramDataset(
    name=args.dataset, 
    dataset_dir='./datasets_r/',
    ...
)

# 目标 (仿照 rcg)
dataset = performat_SramDataset(
    dataset_dir=args.data_dir,
    train_cases=args.train_dataset,
    test_cases=args.test_dataset,
    ...
)
```

---

### 3. `sampling.py` - 数据采样模块

#### 3.1 训练图选择逻辑
- **当前**: 硬编码合并前两个图 `Batch.from_data_list([train_graph_0, train_graph_1])`
- **目标**: 使用 `dataset.num_train` 动态获取训练图数量

#### 3.2 测试集循环
- **当前**: `for i in range(2, len(dataset.names))`
- **目标**: `for i in range(num_train, len(dataset))`

#### 3.3 添加 t-SNE 可视化
- 从 rcg 移植 `plot_tsne_visualization()` 函数

---

### 4. `model.py` - 模型定义

#### 4.1 边类型嵌入维度
- 确保 `edge_type_vocab_size` 参数正确设置 (rcg_r 只有3种边类型)

#### 4.2 模型架构 (与 rcg 完全一致)
- **移除**: `GPSLayer` 相关代码和 `gps_attention` 模型选项
- **保留/添加**: 与 rcg 完全一致的模型支持列表:
  - `clustergcn`, `gcn`, `sage`, `gat`, `resgatedgcn`, `gine`, `pna`
  - `sgformer`, `polynormer`
  - `CustomGatedGCN`, `CustomGCNConv`, `CustomGINEConv`
- **依赖**: 需要从 rcg 复制 `layer.py` 文件到 rcg_r 目录

#### 4.3 移除的参数
- `local_gnn_type`, `global_model_type`, `attn_dropout`, `layer_norm`, `batch_norm`, `num_heads` (GPS相关)
- `g_bn`, `g_drop`, `g_ffn` (改用 rcg 的 `residual`, `ffn`, `use_bn`)

---

### 5. `downstream_train.py` - 训练逻辑

#### 5.1 日志保存
- **当前**: 无日志文件保存
- **目标**: 添加训练日志保存到 `logs/` 目录

#### 5.2 最佳模型追踪
- 添加 `best_epoch` 追踪和测试结果记录

---

### 6. 新增文件: `layer.py`

从 `rcg/layer.py` 复制到 `rcg_r/layer.py`，包含:
- `GatedGCNLayer`: 残差门控图卷积层
- `GCNConvLayer`: 带 FFN 的 GCN 层
- `GINEConvLayer`: 带边特征的 GIN 层

---

## 建议的修改优先级

### 高优先级 (核心功能)
1. `layer.py` - 复制自定义层文件
2. `sram_dataset.py` - 数据加载是基础
3. `main.py` - 参数接口统一
4. `model.py` - 模型架构统一
5. `sampling.py` - 训练/测试集划分

### 中优先级 (功能完善)
6. `downstream_train.py` - 日志和模型保存

### 低优先级 (可选增强)
7. t-SNE 可视化

---

## 预期修改后的使用方式

```bash
# 训练命令示例
python main.py \
    --data_dir ../sram_r/ \
    --train_dataset ssram+digtime \
    --test_dataset sandwich+ultra8t \
    --task_level edge \
    --task regression \
    --model gcn \
    --epochs 200
```

---

## 注意事项

1. **数据格式兼容性**: 确保 `sram_r/*.pt` 文件包含 `r_p2p` 边类型
2. **标签范围**: 电阻值的物理范围与电容不同，归一化策略需要调整
3. **边类型映射**: `g._e2type` 字典中的边类型编号需要与模型嵌入层对应
4. **不使用 GPSLayer**: rcg_r 不需要 GPS 注意力机制，只使用自定义的 GNN+ 层
