# rcg_r 等效电阻预测项目重构 - 设计文档

## 概述

本文档描述将 `rcg_r` 项目重构为与 `rcg` 风格一致的详细设计方案。重构后的代码将支持灵活的数据集配置、统一的模型架构和完善的日志记录功能。

## 技术方案

### 1. 文件复制: `layer.py`

**操作**: 将 `rcg/layer.py` 复制到 `rcg_r/layer.py`

**包含的类**:
- `GatedGCNLayer`: 残差门控图卷积层
- `GCNConvLayer`: 带 FFN 的 GCN 层  
- `GINEConvLayer`: 带边特征的 GIN 层

---

### 2. `sram_dataset.py` 重构

#### 2.1 类签名修改

**当前签名**:
```python
class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        name,
        root,
        neg_edge_ratio=1.0,
        ...
    )
```

**目标签名**:
```python
class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        train_cases,    # e.g., "ssram+digtime"
        test_cases,     # e.g., "sandwich+ultra8t"
        neg_edge_ratio=1.0,
        ...
    )
```

#### 2.2 数据集名称解析

```python
# 解析训练和测试数据集名称
self.train_names = [c.strip() for c in train_cases.split('+') if c.strip()]
self.test_names = [c.strip() for c in test_cases.split('+') if c.strip()]
self.names = self.train_names + self.test_names

# 追踪训练/测试数量
self.num_train = len(self.train_names)
self.num_test = len(self.test_names)
```

#### 2.3 `performat_SramDataset()` 函数修改

**当前签名**:
```python
def performat_SramDataset(dataset_dir, name, neg_edge_ratio, ...)
```

**目标签名**:
```python
def performat_SramDataset(dataset_dir, train_cases, test_cases, neg_edge_ratio, ...)
```

#### 2.4 保持不变的部分

- `sram_graph_load()` 方法中的边类型过滤逻辑 (`'r'` 关键字)
- 百分位数过滤逻辑
- `norm_nfeat()` 方法中的 Min-Max 归一化

---

### 3. `main.py` 重构

#### 3.1 命令行参数修改

**移除的参数**:
```python
# 移除 GPS 相关参数
--local_gnn_type
--global_model_type
--attn_dropout
--layer_norm
--batch_norm (GPS相关)
--num_heads (GPS相关)
--g_bn
--g_drop
--g_ffn

# 移除 GSL 相关参数
--use_gsl
--gsl_layers
--gsl_heads
--gsl_weight
--gsl_memory_efficient
--gsl_checkpoint_freq

# 移除 SGRL 相关参数
--sgrl
--e1_lr
--e2_lr
--momentum
--weight_decay
--cl_epochs
--cl_model
--cl_act_fn
--cl_gnn_layers
--cl_hid_dim
--cl_batch_size
--cl_num_neighbors
--cl_dropout

# 移除单一数据集参数
--dataset
```

**添加的参数**:
```python
# 数据集参数
parser.add_argument("--data_dir", type=str, default="../sram_r/")
parser.add_argument("--train_dataset", type=str, default="ssram+digtime")
parser.add_argument("--test_dataset", type=str, default="sandwich+ultra8t")

# 自定义层参数
parser.add_argument('--residual', type=int, default=1)
parser.add_argument('--ffn', type=int, default=1)

# SGFormer/Polynormer 参数
parser.add_argument("--num_heads", type=int, default=2)
parser.add_argument("--global_layers", type=int, default=2)

# t-SNE 可视化参数
parser.add_argument('--plot_tsne', type=int, default=1)
parser.add_argument('--tsne_max_samples', type=int, default=100000)
parser.add_argument('--tsne_perplexity', type=int, default=30)
```

#### 3.2 模型选择修改

```python
parser.add_argument("--model", type=str, default='gcn', 
    choices=['clustergcn', 'resgatedgcn', 'gat', 'gcn', 'sage', 'gine', 'pna',
             'sgformer', 'polynormer', 'CustomGatedGCN', 'CustomGCNConv', 'CustomGINEConv'])
```

#### 3.3 数据集加载调用修改

```python
dataset = performat_SramDataset(
    dataset_dir=args.data_dir,
    train_cases=args.train_dataset,
    test_cases=args.test_dataset,
    neg_edge_ratio=args.neg_edge_ratio,
    to_undirected=True,
    small_dataset_sample_rates=args.small_dataset_sample_rates,
    large_dataset_sample_rates=args.large_dataset_sample_rates,
    task_level=args.task_level,
    net_only=args.net_only,
    class_boundaries=args.class_boundaries
)
```

---

### 4. `model.py` 重构

#### 4.1 导入修改

```python
# 移除
from gps_layer import GPSLayer

# 添加
from layer import GatedGCNLayer, GCNConvLayer, GINEConvLayer
from torch_geometric.nn.models import SGFormer, Polynormer
from torch_geometric.nn import PNAConv
```

#### 4.2 `GraphHead` 类重构

**简化的 `__init__` 参数**:
```python
def __init__(self, args):
    # 移除 GPS 相关参数
    # local_gnn_type = args.local_gnn_type  # 移除
    # global_model_type = args.global_model_type  # 移除
    # attn_dropout = args.attn_dropout  # 移除
    # layer_norm = args.layer_norm  # 移除
    # batch_norm = args.batch_norm  # 移除 (GPS相关)
    # g_bn = args.g_bn  # 移除
    # g_drop = args.g_drop  # 移除
    # g_ffn = args.g_ffn  # 移除
    
    # 移除 SGRL/对比学习相关
    # self.use_cl = False  # 移除
    # self.cl_linear = nn.Linear(args.cl_hid_dim, node_embed_dim)  # 移除
    
    # 使用统一的参数
    residual = getattr(args, 'residual', True)
    ffn = getattr(args, 'ffn', True)
```

#### 4.3 模型层构建

```python
self.sgformer_model = None
self.polynormer_model = None

if args.model == 'sgformer':
    self.sgformer_model = SGFormer(
        in_channels=hidden_dim, hidden_channels=hidden_dim,
        out_channels=hidden_dim, trans_num_layers=args.num_gnn_layers,
        trans_num_heads=getattr(args, 'num_heads', 1),
        trans_dropout=dropout, gnn_num_layers=args.num_gnn_layers,
        gnn_dropout=dropout,
    )
elif args.model == 'polynormer':
    self.polynormer_model = Polynormer(
        in_channels=hidden_dim, hidden_channels=hidden_dim,
        out_channels=hidden_dim, local_layers=args.num_gnn_layers,
        global_layers=getattr(args, 'global_layers', 2), dropout=dropout,
    )
else:
    for _ in range(args.num_gnn_layers):
        if args.model == 'clustergcn':
            self.layers.append(ClusterGCNConv(hidden_dim, hidden_dim))
        elif args.model == 'gcn':
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        # ... 其他模型
        elif args.model == 'pna':
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity', 'amplification', 'attenuation']
            deg = getattr(args, 'pna_deg', None)
            self.layers.append(PNAConv(hidden_dim, hidden_dim,
                aggregators=aggregators, scalers=scalers, deg=deg,
                edge_dim=hidden_dim, towers=1, pre_layers=1, post_layers=1))
        elif args.model == 'CustomGatedGCN':
            self.layers.append(GatedGCNLayer(in_dim=hidden_dim, out_dim=hidden_dim,
                dropout=dropout, residual=residual, ffn=ffn, batch_norm=use_bn))
        elif args.model == 'CustomGCNConv':
            self.layers.append(GCNConvLayer(dim_in=hidden_dim, dim_out=hidden_dim,
                dropout=dropout, residual=residual, ffn=ffn, batch_norm=use_bn))
        elif args.model == 'CustomGINEConv':
            self.layers.append(GINEConvLayer(dim_in=hidden_dim, dim_out=hidden_dim,
                dropout=dropout, residual=residual, ffn=ffn, batch_norm=use_bn))
```

#### 4.4 `forward` 方法修改

```python
def forward(self, batch):
    x = self.node_encoder(batch.node_type)
    xe = self.edge_encoder(batch.edge_type)
    
    if self.use_stats:
        # ... 电路统计特征编码
    
    if self.model == 'sgformer':
        batch_idx = batch.batch if batch.batch is not None else torch.zeros(batch.num_nodes, dtype=torch.long, device=x.device)
        x = self.sgformer_model(x, batch.edge_index, batch_idx)
    elif self.model == 'polynormer':
        batch_idx = batch.batch if batch.batch is not None else torch.zeros(batch.num_nodes, dtype=torch.long, device=x.device)
        x = self.polynormer_model(x, batch.edge_index, batch_idx)
    elif self.model in ['CustomGatedGCN', 'CustomGCNConv', 'CustomGINEConv']:
        batch.x = x
        batch.edge_attr = xe
        for conv in self.layers:
            batch = conv(batch)
        x = batch.x
    else:
        for conv in self.layers:
            # ... 标准 GNN 层前向传播
```

---

### 5. `sampling.py` 重构

#### 5.1 动态训练图获取

```python
def dataset_sampling(args, dataset, train_idx=None, val_idx=None):
    # 从数据集获取训练图数量
    num_train = getattr(dataset, 'num_train', 1)
    
    # 获取训练/测试图名称
    train_names = getattr(dataset, 'train_names', [])
    test_names = getattr(dataset, 'test_names', [])
    
    print(f"Training graphs ({num_train}): {train_names}")
    print(f"Test graphs ({len(test_names)}): {test_names}")
    
    # 使用第一个训练图作为主训练图
    train_graph = dataset[0]
```

#### 5.2 测试集循环修改

```python
# 测试集：从索引 num_train 开始到末尾
for i in range(num_train, len(dataset)):
    test_graph = dataset[i]
    graph_name = test_graph.name if hasattr(test_graph, 'name') else f'test_graph_{i}'
    
    print(f"Setting up test loader for graph '{graph_name}' (index {i})")
    # ... 创建测试 loader
```

#### 5.3 添加 t-SNE 可视化函数

从 `rcg/sampling.py` 复制 `plot_tsne_visualization()` 函数。

---

### 6. `downstream_train.py` 日志功能

#### 6.1 日志文件创建

```python
import datetime
import sys

# 创建日志目录
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir, exist_ok=True)

# 生成日志文件名
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if args.task == 'classification':
    log_filename = os.path.join(args.log_dir, 
        f"{args.task_level}_{args.task}_{timestamp}_loss{args.class_loss}.txt")
else:
    log_filename = os.path.join(args.log_dir, 
        f"{args.task_level}_{args.task}_{timestamp}_loss{args.regress_loss}.txt")
```

#### 6.2 输出重定向

```python
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(log_filename, 'w')
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, log_file)
```

---

## 文件修改清单

| 文件 | 操作 | 优先级 |
|------|------|--------|
| `rcg_r/layer.py` | 新建 (从 rcg 复制) | 高 |
| `rcg_r/sram_dataset.py` | 修改 (移除 SGRL 相关) | 高 |
| `rcg_r/main.py` | 修改 (移除 GPS/GSL/SGRL) | 高 |
| `rcg_r/model.py` | 修改 (移除 GPS/SGRL) | 高 |
| `rcg_r/sampling.py` | 修改 | 高 |
| `rcg_r/downstream_train.py` | 修改 | 中 |
| `rcg_r/gps_layer.py` | 删除 | 高 |
| `rcg_r/gsl_module.py` | 删除 | 高 |
| `rcg_r/sgrl_models.py` | 删除 | 高 |
| `rcg_r/sgrl_train.py` | 删除 | 高 |

---

## 测试计划

### 功能测试

1. **数据集加载测试**
   - 验证 `train_cases` 和 `test_cases` 参数正确解析
   - 验证 `num_train` 和 `num_test` 正确计算

2. **模型测试**
   - 验证所有支持的模型能够正常初始化
   - 验证前向传播不报错

3. **训练测试**
   - 验证训练循环正常运行
   - 验证日志文件正确生成

### 命令行测试

```bash
# 基础测试
python main.py --data_dir ../sram_r/ --train_dataset ssram --test_dataset digtime --model gcn --epochs 10

# 多数据集测试
python main.py --data_dir ../sram_r/ --train_dataset ssram+digtime --test_dataset sandwich+ultra8t --model CustomGatedGCN --epochs 10

# t-SNE 可视化测试
python main.py --data_dir ../sram_r/ --train_dataset ssram --test_dataset digtime --model gcn --epochs 10 --plot_tsne 1
```

---

## 正确性属性

### P1: 数据集划分正确性
- 训练图数量等于 `train_cases` 中指定的数据集数量
- 测试图数量等于 `test_cases` 中指定的数据集数量
- `dataset[0:num_train]` 为训练图，`dataset[num_train:]` 为测试图

### P2: 模型架构一致性
- rcg_r 支持的模型列表与 rcg 完全一致
- 相同模型配置下，模型参数数量相同

### P3: 边类型处理正确性
- 电阻边 (`r_p2p`) 被正确识别和处理
- 边类型嵌入维度与实际边类型数量匹配
