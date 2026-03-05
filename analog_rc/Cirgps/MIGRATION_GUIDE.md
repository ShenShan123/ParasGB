# Cirgps RC 数据集适配指南

## 概述

本文档描述如何将 Cirgps 项目从原始 SRAM 电路数据集适配到新的 RC 电路数据集。

## 数据格式对比

### 原始 SRAM 数据格式

| 属性 | 原始格式 |
|------|----------|
| 节点类型 | NET=0, DEV=1, PIN=2 |
| 边类型 | device-pin, pin-net, cc_p2n, cc_p2p, cc_n2n |
| 节点特征维度 | 17 维 (统一 padding) |
| 边标签 | 耦合电容 (1e-21 ~ 1e-15 F) |
| 节点标签 | 接地电容 (tar_node_y) |
| 归一化方式 | log10(y * 1e21) / 6 |

### 新 RC 数据格式

| 属性 | 新格式 |
|------|--------|
| 节点类型 | dev=0, pin=1, net=2 |
| 边类型 | connects_to, belongs_to, pair_to |
| 节点特征维度 | dev: 16维, pin: 6维, net: 10维 |
| 边标签 | 电阻值 (0 ~ 700 Ω) |
| 节点标签 | net.y 电容 (~1e-13 F) |
| 归一化方式 | log1p(y) / log1p(700) |

## 需要修改的文件

### 1. sram_dataset.py

**主要修改:**

1. **节点类型常量更新:**
```python
# 原始
NET = 0
DEV = 1
PIN = 2

# 新格式
DEV = 0
PIN = 1
NET = 2
```

2. **添加 `rc_graph_load()` 函数:**
   - 加载 `case{N}_RC.pt` 格式的数据
   - 处理新的边类型: `connects_to`, `belongs_to`, `pair_to`
   - 提取 `pair_to` 边的电阻标签
   - 移除电源网络 (VDD/VSS)

3. **边标签归一化:**
```python
# 原始 (电容)
edge_label = torch.log10(edge_label * 1e21) / 6

# 新格式 (电阻)
edge_label = torch.log1p(edge_label) / np.log1p(700)
```

4. **节点特征处理:**
   - 统一 padding 到 16 维
   - dev: 16维, pin: 6维, net: 10维

### 2. main.py

**主要修改:**

1. **添加 `--data_dir` 参数:**
```python
parser.add_argument("--data_dir", type=str, default="../data", 
                    help="数据目录路径")
```

2. **更新默认数据集名称:**
```python
# 原始
parser.add_argument("--train_dataset", type=str, default="sandwich+ultra8t")
parser.add_argument("--test_dataset", type=str, default="ssram+digtime+timing_ctrl+array_128_32_8t")

# 新格式 (使用 case ID)
parser.add_argument("--train_dataset", type=str, default="1+5+7")
parser.add_argument("--test_dataset", type=str, default="10+11")
```

3. **更新数据集加载调用:**
```python
train_dataset = performat_SramDataset(
    name=args.train_dataset, 
    dataset_dir=args.data_dir,  # 使用命令行参数
    ...
)
```

### 3. model.py

**主要修改:**

1. **更新节点类型常量:**
```python
# 原始
NET = 0
DEV = 1
PIN = 2

# 新格式
DEV = 0
PIN = 1
NET = 2
```

2. **更新节点特征维度:**
```python
# 原始
self.net_attr_layers = nn.Linear(17, node_embed_dim, bias=True)
self.dev_attr_layers = nn.Linear(17, node_embed_dim, bias=True)

# 新格式
self.dev_attr_layers = nn.Linear(16, node_embed_dim, bias=True)
self.pin_attr_layers = nn.Linear(16, node_embed_dim, bias=True)  # 改为 Linear
self.net_attr_layers = nn.Linear(16, node_embed_dim, bias=True)
```

### 4. utils.py

**主要修改:**

1. **更新 `get_pos_neg_edges()` 中的节点类型判断:**
```python
# 原始 (NET=0, PIN=2)
if ntypes == {0, 2}:  # net-pin
    neg_edge_type[i] = 2

# 新格式 (PIN=1, NET=2)
# 只有 pair_to 边 (pin-pin)
if ntypes == {1}:  # pin-pin
    neg_edge_type[i] = 1
```

### 5. downstream_train.py

**主要修改:**

1. **更新节点类型常量:**
```python
DEV = 0
PIN = 1
NET = 2
```

## 数据文件位置

- 原始数据: `/local/hsl/datasets-cirgps/sram/raw/`
- 新数据: `../data/case{N}_RC.pt`

## 运行命令示例

```bash
# 回归任务 (预测电阻值)
python main.py --task regression --train_dataset 1+5+7 --test_dataset 10+11 --data_dir ../data

# 分类任务 (边存在性预测)
python main.py --task classification --train_dataset 1+5+7 --test_dataset 10+11 --data_dir ../data

# 使用 DSPD 位置编码
python main.py --task regression --use_pe 1 --train_dataset 1+5+7 --test_dataset 10+11 --data_dir ../data

# 使用电路统计特征
python main.py --task regression --use_stats 1 --train_dataset 1+5+7 --test_dataset 10+11 --data_dir ../data

# 完整参数示例
python main.py --task regression --train_dataset 1+5+7 --test_dataset 10+11 --data_dir ../data --epochs 200 --batch_size 128 --lr 0.0001 --model clustergcn --num_gnn_layers 4 --hid_dim 144 --use_pe 1 --use_stats 0
```

## 可用的数据文件

数据目录 `../data/` 下的 RC 电路文件:
- case1_RC.pt, case5_RC.pt, case7_RC.pt, case10_RC.pt, case11_RC.pt
- case15_RC.pt, case17_RC.pt, case23_RC.pt, case29_RC.pt, case39_RC.pt
- case42_RC.pt, case44_RC.pt, case45_RC.pt, case55_RC.pt, case58_RC.pt
- case71_RC.pt, case72_RC.pt, case74_RC.pt, case75_RC.pt, case78_RC.pt

## Cirgps 特有功能

### DSPD (Double Source Path Distance)

Cirgps 使用 DSPD 作为位置编码，计算每个节点到源节点和目标节点的最短路径距离。这个功能在新数据集上仍然适用，不需要修改。

### 电路统计特征编码器

原始代码中的电路统计特征编码器需要根据新的节点特征维度进行调整:
- dev: 16维 → Linear(16, embed_dim)
- pin: 6维 → Linear(16, embed_dim) (padding 后)
- net: 10维 → Linear(16, embed_dim) (padding 后)

## 注意事项

1. **节点类型顺序变化**: 原始 NET=0, DEV=1, PIN=2 → 新格式 DEV=0, PIN=1, NET=2
2. **边标签范围变化**: 电容 (1e-21~1e-15 F) → 电阻 (0~700 Ω)
3. **归一化方式变化**: log10 → log1p
4. **目标边类型**: 原始有 3 种 (cc_p2n, cc_p2p, cc_n2n) → 新格式只有 1 种 (pair_to)
