# Requirements Document: 统一任务支持（节点/边 × 回归/分类）

## Introduction

本文档描述了让 `Cirgps` 和 `paragraph` 两个项目支持节点级和边级任务（回归和分类）的需求。参照 `rcg` 的数据处理方式，统一标签存储格式和预处理逻辑。

## 当前状态分析

### 四个项目的任务支持情况

| 项目 | 节点任务 | 边任务 | 回归 | 分类 | task_level参数 |
|------|---------|--------|------|------|---------------|
| **rcg** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **CircuitGCL** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **paragraph** | ❌ | ✅ | ✅ | ✅ | ❌ |
| **Cirgps** | ❌ | ✅ | ✅ | ✅ | ❌ |

### rcg 的标签存储格式（参照标准）

```python
# 边任务标签: edge_label shape = [num_edges, 2]
# - edge_label[:, 0]: 归一化后的回归标签 (0~1)
# - edge_label[:, 1]: 分桶后的分类标签 (0~num_classes-1)

# 节点任务标签: y shape = [num_nodes, 2]
# - y[:, 0]: 归一化后的回归标签 (0~1)
# - y[:, 1]: 分桶后的分类标签 (0~num_classes-1)
```

### rcg 的数据预处理逻辑

1. **归一化**: `label = torch.log10(label * 1e21) / 6` (边) 或 `torch.log10(label * 1e20) / 6` (节点)
2. **裁剪**: `label[label < 0] = 0.0`, `label[label > 1] = 1.0`
3. **分桶**: `label_c = torch.bucketize(label, class_boundaries)`
4. **合并**: `label = torch.stack([label, label_c], dim=1)`

### Cirgps 的 DSPD 特殊处理

- **边任务**: DSPD 存储源节点和目标节点到锚点的距离，shape = `[num_nodes, 2]`
  - `dspd[:, 0]`: 到源节点(anchor 0)的距离
  - `dspd[:, 1]`: 到目标节点(anchor 1)的距离
  
- **节点任务**: DSPD 需要存储两个相同的距离（因为只有一个目标节点）
  - `dspd[:, 0]`: 到目标节点的距离
  - `dspd[:, 1]`: 到目标节点的距离（相同值）

## Glossary

- **task_level**: 任务级别，`node`（节点级）或 `edge`（边级）
- **task**: 任务类型，`regression`（回归）或 `classification`（分类）
- **DSPD**: Double Shortest Path Distance，双源最短路径距离
- **class_boundaries**: 分类边界，如 `[0.2, 0.4, 0.6, 0.8]` 将标签分为5类

## Requirements

### Requirement 1: 为 paragraph 添加节点任务支持

**User Story:** 作为开发者，我希望 paragraph 项目能支持节点级的回归和分类任务。

#### Acceptance Criteria

1. THE paragraph/main.py SHALL 添加 `--task_level` 参数，支持 `node` 和 `edge` 选项
2. THE paragraph/main.py SHALL 添加 `--class_boundaries` 参数用于分类任务
3. THE paragraph/main.py SHALL 添加 `--net_only` 参数用于节点任务
4. THE paragraph/sram_dataset.py SHALL 修改 `SealSramDataset` 类以支持 `task_level` 参数
5. THE paragraph/sram_dataset.py SHALL 在 `single_g_process` 中添加节点任务的数据处理逻辑
6. THE paragraph/sram_dataset.py SHALL 在 `norm_nfeat` 中添加节点标签的归一化和分桶逻辑
7. WHEN task_level='node' THEN 标签 y 的 shape SHALL 为 `[num_nodes, 2]`
8. WHEN task_level='edge' THEN 标签 edge_label 的 shape SHALL 为 `[num_edges, 2]`

### Requirement 2: 为 Cirgps 添加节点任务支持

**User Story:** 作为开发者，我希望 Cirgps 项目能支持节点级的回归和分类任务。

#### Acceptance Criteria

1. THE Cirgps/main.py SHALL 添加 `--task_level` 参数，支持 `node` 和 `edge` 选项
2. THE Cirgps/main.py SHALL 添加 `--class_boundaries` 参数用于分类任务
3. THE Cirgps/main.py SHALL 添加 `--net_only` 参数用于节点任务
4. THE Cirgps/sram_dataset.py SHALL 修改 `SealSramDataset` 类以支持 `task_level` 参数
5. THE Cirgps/sram_dataset.py SHALL 在 `single_g_process` 中添加节点任务的数据处理逻辑
6. THE Cirgps/sram_dataset.py SHALL 在 `norm_nfeat` 中添加节点标签的归一化和分桶逻辑
7. WHEN task_level='node' THEN 标签 y 的 shape SHALL 为 `[num_nodes, 2]`
8. WHEN task_level='edge' THEN 标签 edge_label 的 shape SHALL 为 `[num_edges, 2]`

### Requirement 3: Cirgps 节点任务的 DSPD 计算

**User Story:** 作为开发者，我希望 Cirgps 在节点任务中能正确计算 DSPD。

#### Acceptance Criteria

1. THE Cirgps/sampling_and_pe.py SHALL 添加节点任务的 DSPD 计算函数
2. WHEN task_level='node' THEN DSPD 的两列 SHALL 存储相同的距离值（到目标节点的距离）
3. WHEN task_level='edge' THEN DSPD 的两列 SHALL 分别存储到源节点和目标节点的距离
4. THE Cirgps/sampling_and_pe.py SHALL 添加 `pe_encoding_for_node_graph` 函数处理节点任务
5. THE Cirgps/sampling_and_pe.py SHALL 修改 `dataset_sampling_and_pe_calculation` 以支持 task_level 参数

### Requirement 4: 统一标签存储格式

**User Story:** 作为开发者，我希望所有项目使用统一的标签存储格式。

#### Acceptance Criteria

1. THE 回归标签 SHALL 存储在第一维度（index 0）
2. THE 分类标签 SHALL 存储在第二维度（index 1）
3. THE 归一化逻辑 SHALL 使用 `log10(label * scale) / 6` 公式
4. THE 分桶逻辑 SHALL 使用 `torch.bucketize(label, class_boundaries)`
5. THE 标签范围 SHALL 裁剪到 [0, 1]

### Requirement 5: 修改下游训练逻辑

**User Story:** 作为开发者，我希望下游训练能根据任务类型正确处理标签。

#### Acceptance Criteria

1. THE paragraph/downstream_train.py SHALL 根据 task_level 选择正确的标签
2. THE Cirgps/downstream_train.py SHALL 根据 task_level 选择正确的标签
3. WHEN task='regression' THEN 训练 SHALL 使用 label[:, 0]
4. WHEN task='classification' THEN 训练 SHALL 使用 label[:, 1].long()
5. THE 模型输出维度 SHALL 根据任务类型调整（回归=1，分类=num_classes）

### Requirement 6: 修改模型以支持节点任务

**User Story:** 作为开发者，我希望模型能处理节点级任务。

#### Acceptance Criteria

1. THE paragraph/model.py SHALL 添加节点任务的前向传播逻辑
2. THE Cirgps/model.py SHALL 添加节点任务的前向传播逻辑
3. WHEN task_level='node' THEN 模型 SHALL 直接输出节点表示
4. WHEN task_level='edge' THEN 模型 SHALL 聚合源节点和目标节点表示

## 修改文件清单

### paragraph 项目

| 文件 | 修改内容 |
|------|---------|
| paragraph/main.py | 添加 task_level, class_boundaries, net_only 参数 |
| paragraph/sram_dataset.py | 添加节点任务数据处理，统一标签格式 |
| paragraph/downstream_train.py | 添加节点任务训练逻辑 |
| paragraph/model.py | 添加节点任务前向传播 |
| paragraph/sampling.py | 添加节点任务采样逻辑 |

### Cirgps 项目

| 文件 | 修改内容 |
|------|---------|
| Cirgps/main.py | 添加 task_level, class_boundaries, net_only 参数 |
| Cirgps/sram_dataset.py | 添加节点任务数据处理，统一标签格式 |
| Cirgps/downstream_train.py | 添加节点任务训练逻辑 |
| Cirgps/model.py | 添加节点任务前向传播 |
| Cirgps/sampling_and_pe.py | 添加节点任务 DSPD 计算（两列相同距离） |

## 数据处理流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                        原始数据加载                              │
│  - tar_node_y: 节点接地电容                                      │
│  - tar_edge_y: 边耦合电容                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        数据过滤                                  │
│  - 节点: 1e-21 < tar_node_y < 1e-15                             │
│  - 边: 1e-21 < tar_edge_y < 1e-15                               │
│  - 非法值替换为 1e-30                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        归一化处理                                │
│  - 节点: y = log10(tar_node_y * 1e20) / 6                       │
│  - 边: edge_label = log10(tar_edge_y * 1e21) / 6                │
│  - 裁剪到 [0, 1]                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        分桶处理                                  │
│  - label_c = bucketize(label, [0.2, 0.4, 0.6, 0.8])             │
│  - 生成 0~4 的分类标签                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        标签合并                                  │
│  - 节点: y = stack([y_reg, y_cls], dim=1)  # [N, 2]             │
│  - 边: edge_label = stack([e_reg, e_cls], dim=1)  # [E, 2]      │
└─────────────────────────────────────────────────────────────────┘
```

## Cirgps DSPD 计算说明

### 边任务 DSPD（现有逻辑）

```python
# 对于每条目标边 (src, dst)
# 采样以 src 和 dst 为锚点的子图
# 计算所有节点到 src 和 dst 的最短路径距离
dspd = get_double_spd(subgraph, anchor_indices=[0, 1], max_dist=max_dist)
# dspd shape: [num_nodes_in_subgraph, 2]
# dspd[:, 0]: 到 src 的距离
# dspd[:, 1]: 到 dst 的距离
```

### 节点任务 DSPD（需要添加）

```python
# 对于每个目标节点 node
# 采样以 node 为锚点的子图
# 计算所有节点到 node 的最短路径距离
dspd = get_single_spd(subgraph, anchor_index=0, max_dist=max_dist)
# 复制为两列以保持与边任务相同的格式
dspd = torch.stack([dspd, dspd], dim=1)
# dspd shape: [num_nodes_in_subgraph, 2]
# dspd[:, 0]: 到 node 的距离
# dspd[:, 1]: 到 node 的距离（相同值）
```

## 训练时标签选择

```python
# 在 downstream_train.py 中
if args.task == 'regression':
    if args.task_level == 'node':
        true = batch.y[:, 0]  # 节点回归标签
    else:
        true = batch.edge_label[:, 0]  # 边回归标签
elif args.task == 'classification':
    if args.task_level == 'node':
        true = batch.y[:, 1].long()  # 节点分类标签
    else:
        true = batch.edge_label[:, 1].long()  # 边分类标签
```
