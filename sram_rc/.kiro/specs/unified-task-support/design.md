# Design Document: 统一任务支持（节点/边 × 回归/分类）

## Overview

本设计文档描述了如何修改 `Cirgps` 和 `paragraph` 项目以支持节点级和边级任务（回归和分类）。核心思想是参照 `rcg` 的数据处理方式，统一标签存储格式（回归标签在第一维度，分类标签在第二维度），并为 Cirgps 的节点任务实现特殊的 DSPD 计算逻辑。

## Architecture

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              main.py                                     │
│  - 添加 task_level, class_boundaries, net_only 参数                      │
│  - 根据 task_level 调用不同的数据加载和训练逻辑                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           sram_dataset.py                                │
│  - SealSramDataset 支持 task_level 参数                                  │
│  - single_g_process 根据 task_level 处理节点/边数据                      │
│  - norm_nfeat 统一归一化和分桶逻辑                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────┐     ┌─────────────────────────────┐
│   sampling.py (paragraph)    │     │ sampling_and_pe.py (Cirgps) │
│   - 节点任务: NeighborLoader │     │ - 节点任务: 单锚点DSPD      │
│   - 边任务: LinkNeighborLoader│    │ - 边任务: 双锚点DSPD        │
└─────────────────────────────┘     └─────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              model.py                                    │
│  - 节点任务: 直接输出节点表示                                            │
│  - 边任务: 聚合源节点和目标节点表示                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         downstream_train.py                              │
│  - 根据 task_level 和 task 选择正确的标签维度                            │
│  - 回归: label[:, 0], 分类: label[:, 1].long()                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. main.py 参数扩展

```python
# 新增参数
parser.add_argument("--task_level", type=str, default="edge", 
    choices=['node', 'edge'], help="Task level: 'node' or 'edge'")
parser.add_argument("--class_boundaries", type=list, 
    default=[0.2, 0.4, 0.6, 0.8], help="Boundaries for classification")
parser.add_argument("--net_only", type=int, default=0, 
    help="Only use net nodes for node level task")
parser.add_argument("--num_classes", type=int, default=5, 
    help="Number of classes for classification")
```

### 2. SealSramDataset 类修改

```python
class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        name,
        root,
        neg_edge_ratio=1.0,
        to_undirected=True,
        sample_rates=[1.0],
        task_level='edge',      # 新增
        task_type='regression',
        net_only=False,         # 新增
        class_boundaries=[0.2, 0.4, 0.6, 0.8],  # 新增
        transform=None,
        pre_transform=None
    ):
        self.task_level = task_level
        self.net_only = net_only
        self.class_boundaries = torch.tensor(class_boundaries)
        # ... 其余初始化逻辑
```

### 3. 数据处理接口

```python
def single_g_process(self, idx: int):
    """处理单个图数据"""
    graph = self.sram_graph_load(self.names[idx], self.raw_paths[idx])
    
    if self.task_level == 'edge':
        # 边任务处理（现有逻辑）
        # ...
        graph.edge_label_index = links
        graph.edge_label = labels  # shape: [num_edges]
        
    elif self.task_level == 'node':
        # 节点任务处理（新增）
        graph.y = graph.tar_node_y.squeeze()  # shape: [num_nodes]
        
    return graph

def norm_nfeat(self, ntypes):
    """归一化特征和标签"""
    # 节点特征归一化（现有逻辑）
    # ...
    
    if self.task_level == 'edge':
        # 边标签归一化和分桶
        self._data.edge_label = torch.log10(self._data.edge_label * 1e21) / 6
        self._data.edge_label = torch.clamp(self._data.edge_label, 0.0, 1.0)
        edge_label_c = torch.bucketize(self._data.edge_label, self.class_boundaries)
        self._data.edge_label = torch.stack([self._data.edge_label, edge_label_c], dim=1)
        
    elif self.task_level == 'node':
        # 节点标签归一化和分桶
        self._data.y = torch.log10(self._data.y * 1e20) / 6
        self._data.y = torch.clamp(self._data.y, 0.0, 1.0)
        node_label_c = torch.bucketize(self._data.y, self.class_boundaries)
        self._data.y = torch.stack([self._data.y, node_label_c], dim=1)
```

### 4. Cirgps DSPD 计算接口

```python
def get_single_spd(data, anchor_index, max_dist):
    """计算单锚点最短路径距离（用于节点任务）"""
    # 返回 shape: [num_nodes]
    pass

def pe_encoding_for_node_graph(args, graph, node_indices, node_labels, processed_pe_path):
    """节点任务的 PE 编码"""
    # 对每个目标节点，计算到该节点的距离
    # DSPD 两列存储相同的距离值
    # 返回 loader, dspd_per_batch
    pass

def dataset_sampling_and_pe_calculation(args, train_dataset, test_dataset):
    """根据 task_level 选择不同的采样和 PE 计算逻辑"""
    if args.task_level == 'node':
        # 使用节点采样和单锚点 DSPD
        pass
    else:
        # 使用边采样和双锚点 DSPD（现有逻辑）
        pass
```

### 5. 模型前向传播接口

```python
class GraphHead(nn.Module):
    def __init__(self, args):
        self.task_level = args.task_level
        # ...
        
        # 根据 task_level 调整 head 输入维度
        if self.task_level == 'node':
            head_input_dim = hidden_dim  # 单节点表示
        else:
            head_input_dim = hidden_dim * 2 if src_dst_agg == 'concat' else hidden_dim
    
    def forward(self, batch):
        # GNN 编码
        x = self.encode(batch)
        
        if self.task_level == 'node':
            # 节点任务：直接使用目标节点的表示
            batch_size = batch.y.size(0) if batch.y.ndim == 1 else batch.y.size(0)
            node_emb = x[:batch_size, :]
            pred = self.head_layers(node_emb)
            return pred, batch.y
        else:
            # 边任务：聚合源节点和目标节点表示
            batch_size = batch.edge_label.size(0)
            src_emb = x[:batch_size, :]
            dst_emb = x[batch_size:batch_size*2, :]
            graph_emb = self.aggregate(src_emb, dst_emb)
            pred = self.head_layers(graph_emb)
            return pred, batch.edge_label
```

### 6. 训练时标签选择

```python
def compute_loss(args, pred, true, criterion):
    """根据任务类型选择正确的标签维度"""
    if true.ndim == 2:
        if args.task == 'regression':
            true = true[:, 0]  # 回归标签
        else:
            true = true[:, 1].long()  # 分类标签
    
    # 计算损失
    return criterion(pred, true), pred
```

## Data Models

### 标签数据格式

```python
# 边任务标签
edge_label: torch.Tensor  # shape: [num_edges, 2]
# edge_label[:, 0]: 归一化回归标签 (float, 0~1)
# edge_label[:, 1]: 分桶分类标签 (long, 0~num_classes-1)

# 节点任务标签
y: torch.Tensor  # shape: [num_nodes, 2]
# y[:, 0]: 归一化回归标签 (float, 0~1)
# y[:, 1]: 分桶分类标签 (long, 0~num_classes-1)
```

### DSPD 数据格式

```python
# 边任务 DSPD
dspd: torch.Tensor  # shape: [num_nodes_in_subgraph, 2]
# dspd[:, 0]: 到源节点的距离
# dspd[:, 1]: 到目标节点的距离

# 节点任务 DSPD
dspd: torch.Tensor  # shape: [num_nodes_in_subgraph, 2]
# dspd[:, 0]: 到目标节点的距离
# dspd[:, 1]: 到目标节点的距离（相同值）
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: 标签格式一致性
*For any* 处理后的数据集，标签的 shape 的第二维度应该为 2，第一维度存储回归标签（0~1范围），第二维度存储分类标签（0~num_classes-1范围）。
**Validates: Requirements 1.7, 1.8, 4.1, 4.2**

### Property 2: 归一化范围约束
*For any* 归一化后的标签值，其值应该在 [0, 1] 范围内。
**Validates: Requirements 4.5**

### Property 3: 分桶标签有效性
*For any* 分桶后的分类标签，其值应该是 0 到 num_classes-1 之间的整数。
**Validates: Requirements 4.4**

### Property 4: 节点任务 DSPD 对称性
*For any* 节点任务的 DSPD 张量，其两列的值应该完全相同。
**Validates: Requirements 3.2**

### Property 5: 边任务 DSPD 锚点正确性
*For any* 边任务的 DSPD 张量，锚点节点（index 0 和 1）到自身的距离应该为 0。
**Validates: Requirements 3.3**

## Error Handling

### 数据加载错误
- 当数据文件不存在时，抛出 `FileNotFoundError` 并提示正确的文件路径
- 当数据格式不正确时，抛出 `ValueError` 并说明期望的格式

### 参数验证错误
- 当 `task_level` 不是 'node' 或 'edge' 时，抛出 `ValueError`
- 当 `class_boundaries` 不是递增序列时，抛出 `ValueError`

### 内存错误处理
- 对于大型图，使用分块处理避免内存溢出
- 在 DSPD 计算中添加内存监控和垃圾回收

## Testing Strategy

### 单元测试
- 测试参数解析是否正确
- 测试数据集类的初始化
- 测试归一化和分桶函数的输出格式

### 属性测试
- 使用 pytest 和 hypothesis 进行属性测试
- 验证标签格式、范围约束、DSPD 对称性等属性

### 集成测试
- 测试完整的数据加载和训练流程
- 验证节点任务和边任务的端到端正确性
