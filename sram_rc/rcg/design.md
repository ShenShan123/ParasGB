# Design Document

## Overview

本设计文档描述了对 `rcg` 目录下模型代码的重构方案。主要目标是：
1. 添加 SGFormer 和 Polynormer 两个新的图神经网络模型
2. 移除图对比学习（SGRL）相关代码
3. 移除 GPS Layer 模块
4. 添加自定义层（CustomGatedGCN、CustomGCNConv、CustomGINEConv）的直接调用支持

## Architecture

### 重构后的模块结构

```
rcg/
├── model.py          # 主模型文件（GraphHead 类）
├── layer.py          # 自定义层（GatedGCNLayer, GCNConvLayer, GINEConvLayer）
├── downstream_train.py
├── main.py
├── sampling.py
├── sram_dataset.py
├── utils.py
├── balanced_mse.py
└── plot.py
```

### 删除的文件
- `rcg/sgrl_models.py` - SGRL 模型定义
- `rcg/sgrl_train.py` - SGRL 训练逻辑
- `rcg/gps_layer.py` - GPS Layer 实现

## Components and Interfaces

### 1. model.py 修改

#### 导入变更

```python
# 移除
from gps_layer import GPSLayer

# 添加
from torch_geometric.nn.models import SGFormer, Polynormer
from layer import GatedGCNLayer, GCNConvLayer, GINEConvLayer
```

#### GraphHead 类修改

**移除的属性和参数：**
- `use_cl` - 对比学习标志
- `cl_linear` - 对比学习线性层
- GPS Layer 相关参数：`local_gnn_type`, `global_model_type`, `attn_dropout`, `layer_norm`, `num_heads`, `g_bn`, `g_drop`, `g_ffn`

**新增的模型选项：**
- `sgformer` - SGFormer 模型
- `polynormer` - Polynormer 模型
- `CustomGatedGCN` - 自定义 Gated GCN 层
- `CustomGCNConv` - 自定义 GCN 卷积层
- `CustomGINEConv` - 自定义 GINE 卷积层

### 2. 新模型接口

#### SGFormer 参数
```python
SGFormer(
    in_channels=hidden_dim,
    hidden_channels=hidden_dim,
    out_channels=hidden_dim,
    trans_num_layers=args.num_gnn_layers,
    trans_num_heads=getattr(args, 'num_heads', 1),
    trans_dropout=args.dropout,
    gnn_num_layers=args.num_gnn_layers,
    gnn_dropout=args.dropout,
)
```

#### Polynormer 参数
```python
Polynormer(
    in_channels=hidden_dim,
    hidden_channels=hidden_dim,
    out_channels=hidden_dim,
    local_layers=args.num_gnn_layers,
    global_layers=getattr(args, 'global_layers', 2),
    dropout=args.dropout,
)
```

#### 自定义层参数
```python
GatedGCNLayer(
    in_dim=hidden_dim,
    out_dim=hidden_dim,
    dropout=args.dropout,
    residual=getattr(args, 'residual', True),
    ffn=getattr(args, 'ffn', True),
    batch_norm=args.use_bn,
)

GCNConvLayer(
    dim_in=hidden_dim,
    dim_out=hidden_dim,
    dropout=args.dropout,
    residual=getattr(args, 'residual', True),
    ffn=getattr(args, 'ffn', True),
    batch_norm=args.use_bn,
)

GINEConvLayer(
    dim_in=hidden_dim,
    dim_out=hidden_dim,
    dropout=args.dropout,
    residual=getattr(args, 'residual', True),
    ffn=getattr(args, 'ffn', True),
    batch_norm=args.use_bn,
)
```

## Data Models

### 支持的模型类型

| 模型名称 | 类型 | 说明 |
|---------|------|------|
| gcn | 基础 GNN | PyG GCNConv |
| sage | 基础 GNN | PyG SAGEConv |
| gat | 基础 GNN | PyG GATConv |
| gine | 基础 GNN | PyG GINEConv |
| resgatedgcn | 基础 GNN | PyG ResGatedGraphConv |
| clustergcn | 基础 GNN | PyG ClusterGCNConv |
| sgformer | Transformer | PyG SGFormer |
| polynormer | Transformer | PyG Polynormer |
| CustomGatedGCN | 自定义层 | layer.py GatedGCNLayer |
| CustomGCNConv | 自定义层 | layer.py GCNConvLayer |
| CustomGINEConv | 自定义层 | layer.py GINEConvLayer |

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: 模型选择正确性
*For any* 有效的模型类型参数（gcn, sage, gat, gine, resgatedgcn, clustergcn, sgformer, polynormer, CustomGatedGCN, CustomGCNConv, CustomGINEConv），GraphHead 初始化后应使用对应的模型类进行前向传播。
**Validates: Requirements 1.1, 2.1, 6.2, 6.3, 6.4, 7.1**

### Property 2: 前向传播输出形状正确性
*For any* 有效的输入 batch（包含 node_type, edge_type, edge_index, y），GraphHead 的前向传播应返回正确形状的预测结果（pred, true_class, true_label）。
**Validates: Requirements 1.4, 2.4, 6.6, 7.2**

### Property 3: Circuit Statistics 编码器兼容性
*For any* 启用 use_stats 的配置，GraphHead 应正确处理 node_attr 并生成正确维度的节点嵌入。
**Validates: Requirements 7.3**

### Property 4: SGRL 代码完全移除
*For any* GraphHead 实例，不应存在 use_cl 或 cl_linear 属性。
**Validates: Requirements 4.1, 4.2, 4.4**

## Error Handling

1. **无效模型类型**: 当用户指定不支持的模型类型时，抛出 `ValueError` 并提示支持的模型列表
2. **参数缺失**: 对于可选参数使用 `getattr` 提供默认值
3. **维度不匹配**: 在初始化时验证 hidden_dim 与 use_stats 的兼容性

## Testing Strategy

### 单元测试
- 测试每种模型类型的初始化
- 测试前向传播的输出形状
- 测试 SGRL 相关代码已被移除
- 测试文件删除操作

### 属性测试
- 使用 pytest 和 hypothesis 进行属性测试
- 最少 100 次迭代验证每个属性
- 测试标签格式: **Feature: model-refactoring, Property N: {property_text}**
