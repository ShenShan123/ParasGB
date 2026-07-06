# paragraph-simple 适配新数据集修改指南

## 概述

将 paragraph-simple 从原始 SRAM 数据集迁移到 `data/` 目录下的 RC 电路数据集。

**状态: ✅ 已完成**

### 已完成的修改

1. **`sram_dataset.py`** - 完全重写，添加 `rc_graph_load()` 函数
2. **`main.py`** - 添加 `--data_dir` 参数，更新默认数据集名称
3. **`utils.py`** - 更新节点类型映射 (dev=0, pin=1, net=2)
4. **`downstream_train.py`** - 更新节点类型常量

### 任务对比

| 任务类型 | paragraph-simple 原始 | 新数据集 |
|----------|----------------------|----------|
| **节点任务** | 对地电容 (Cg, ~1e-15 F) | 电容 (~1e-13 F) |
| **边任务** | 耦合电容 (Cc, 1e-21~1e-15 F) | 电阻 (0~700 Ω) |
| **分类方式** | 按电容值分桶 | 按标签值分桶 |

---

## 运行方式

```bash
cd paragraph-simple

# 边回归任务 (预测电阻值)
python main.py --task regression --train_dataset "1+5+7+15+23" --test_dataset "11+55+78" --data_dir "../data"

# 边分类任务 (预测电阻分桶)
python main.py --task classification --train_dataset "1+5+7+15+23" --test_dataset "11+55+78" --data_dir "../data"
```

---

## 需要修改的文件

### 1. `sram_dataset.py` - 核心数据处理

#### 1.1 节点类型映射

```python
# 原始
NODE_TYPE_ORDER = ['device', 'pin', 'net']  # 隐式

# 修改为
NODE_TYPE_ORDER = ['dev', 'pin', 'net']
NODE_TYPE_TO_ID = {'dev': 0, 'pin': 1, 'net': 2}
```

#### 1.2 边类型映射

```python
# 原始边类型
EDGE_TYPES = [
    ('device', 'device-pin', 'pin'),
    ('pin', 'pin-net', 'net'),
    ('pin', 'cc_p2n', 'net'),      # 耦合电容边
    ('pin', 'cc_p2p', 'pin'),      # 耦合电容边
    ('net', 'cc_n2n', 'net'),      # 耦合电容边
]

# 修改为
EDGE_TYPES = [
    ('dev', 'connects_to', 'pin'),
    ('dev', 'connects_to', 'net'),
    ('pin', 'belongs_to', 'net'),
    ('pin', 'pair_to', 'pin'),     # 目标边 (电阻)
]
```

#### 1.3 新增 `rc_graph_load()` 函数

替换原有的 `sram_graph_load()` 函数，主要变化：
- 移除硬编码的 `power_net_ids`，改为根据特征检测电源网络
- 目标边从 `cc_*` 改为 `pair_to`
- 标签范围从耦合电容 (1e-21~1e-15) 改为电阻 (0~700)

#### 1.4 `norm_nfeat()` 函数修改

```python
# 原始 - 耦合电容归一化
self._data.edge_label = torch.log10(self._data.edge_label * 1e21) / 6

# 修改为 - 电阻归一化
MAX_EDGE_LABEL = 700.0
self._data.edge_label = torch.log1p(self._data.edge_label) / np.log1p(MAX_EDGE_LABEL)
```

#### 1.5 特征维度修改

```python
# 原始
max_feat_dim = 17

# 修改为
max_feat_dim = 16
```

#### 1.6 `raw_file_names` 属性修改

```python
# 原始
raw_file_names.append(name+'.pt')

# 修改为 (适配 caseX_RC.pt 格式)
raw_file_names.append(f'case{name}_RC.pt')
```

---

### 2. `main.py` - 数据路径和参数

#### 2.1 新增数据目录参数

```python
parser.add_argument("--data_dir", type=str, default="../data", help="数据目录路径")
```

#### 2.2 数据集名称格式

```python
# 原始
parser.add_argument("--train_dataset", type=str, default="sandwich+ultra8t")
parser.add_argument("--test_dataset", type=str, default="ssram+digtime")

# 修改为 (使用 case ID)
parser.add_argument("--train_dataset", type=str, default="1+5+7+15+23+29+39+42+44+71+72+74")
parser.add_argument("--test_dataset", type=str, default="11+55+78")
```

#### 2.3 数据集路径

```python
# 原始
dataset_dir='/local/hsl/datasets-para'

# 修改为
dataset_dir=args.data_dir
```

---

### 3. `utils.py` - 辅助函数

#### 3.1 `get_pos_neg_edges()` 函数

需要适配新的节点类型：
```python
# 原始
if ntypes == {0, 2}:  # net=0, pin=2
    neg_edge_type[i] = 2

# 修改为 (dev=0, pin=1, net=2)
if ntypes == {1, 2}:  # pin=1, net=2
    neg_edge_type[i] = 2
elif ntypes == {1}:   # pin=1
    neg_edge_type[i] = 3
elif ntypes == {2}:   # net=2
    neg_edge_type[i] = 4
```

---

## 数据格式对比

### 原始 SRAM 数据 (`/local/hsl/datasets-para/sram/raw/`)

```
sandwich.pt, ultra8t.pt, ssram.pt, ...
```

**异构图结构**:
- 节点: `device`, `pin`, `net`
- 边: `device-pin`, `pin-net`, `cc_p2n`, `cc_p2p`, `cc_n2n`
- 节点标签: `net.y` (对地电容)
- 边标签: `cc_*.y` (耦合电容)

### 新 RC 数据 (`data/`)

```
case1_RC.pt, case5_RC.pt, case7_RC.pt, ...
```

**异构图结构**:
- 节点: `dev`, `pin`, `net`
- 边: `connects_to`, `belongs_to`, `pair_to`
- 节点标签: `net.y` (电容, ~1e-13 F)
- 边标签: `pair_to.y` (电阻, 0~700 Ω)

---

## 归一化方式

### 节点标签 (电容)

```python
# 原始 paragraph-simple
normalized = torch.log10(cap * 1e21) / 6

# 新数据集
MAX_NODE_LABEL = 8e-13
normalized = torch.log1p(cap * 1e15) / np.log1p(MAX_NODE_LABEL * 1e15)
```

### 边标签 (电阻)

```python
# 原始 paragraph-simple (耦合电容)
normalized = torch.log10(cc * 1e21) / 6

# 新数据集 (电阻)
MAX_EDGE_LABEL = 700.0
normalized = torch.log1p(res) / np.log1p(MAX_EDGE_LABEL)
```

---

## 推荐的数据划分

基于相似度分析和标签数量：

| 类型 | Case ID | 说明 |
|------|---------|------|
| **剔除** | 10, 17, 45, 75 | 标签太少或重复 |
| **测试集** | 11, 55, 78 | 离群点，测试泛化 |
| **验证集** | 58 | 中等相似度 |
| **训练集** | 1, 5, 7, 15, 23, 29, 39, 42, 44, 71, 72, 74 | 核心训练数据 |

---

## 修改步骤

1. 备份原始文件
2. 修改 `sram_dataset.py` 中的图加载逻辑
3. 修改 `main.py` 中的数据路径和参数
4. 修改 `utils.py` 中的节点类型映射
5. 删除 `processed/` 缓存目录，重新处理数据
6. 测试运行

---

## 与 DLPL-CAP 的区别

paragraph-simple 和 DLPL-CAP 的主要区别：

| 特性 | paragraph-simple | DLPL-CAP |
|------|------------------|----------|
| **模型架构** | GraphHead (SAGEConv) | CapClassifier + CapRegressor |
| **训练方式** | 单模型 / Ensemble | 分类器 + 回归器两阶段 |
| **采样方式** | LinkNeighborLoader | 自定义采样 |
| **数据处理** | 合并多图训练 | 分图处理 |

两者的数据加载逻辑 (`sram_dataset.py`) 基本相同，可以复用 DLPL-CAP 的 `rc_graph_load()` 实现。
