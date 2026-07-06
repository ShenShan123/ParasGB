# CircuitGCL 迁移指南 - 适配 RC 电路数据集

## 1. CircuitGCL 核心方法

### 1.1 整体流程
```
┌─────────────────────────────────────────────────────────────────┐
│                        CircuitGCL 流程                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ 数据加载  │ -> │ SGRL 预训练  │ -> │ 下游任务训练          │  │
│  │          │    │ (可选)       │    │ (边/节点 回归/分类)   │  │
│  └──────────┘    └──────────────┘    └──────────────────────┘  │
│       │                │                       │                │
│       v                v                       v                │
│  异构图->同构图    节点 Embedding         最终预测结果          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 SGRL 的作用 (Self-supervised Graph Representation Learning)

**SGRL 就是一个预训练模块，用于计算节点 embedding**：

```python
# main.py 中的调用逻辑
if args.sgrl == 1:
    # 1. 预训练: 通过对比学习获取节点 embedding
    cl_embeds = sgrl_train(args, dataset, device)  # [N, cl_hid_dim]
    
# 2. 下游任务: 将 embedding 作为额外特征
downstream_train(args, dataset, device, cl_embeds)
```

**SGRL 内部结构**：
- Online Encoder + Target Encoder (双编码器)
- 通过对比损失训练，让相似节点的 embedding 更接近
- 输出: 每个节点一个 `cl_hid_dim` 维的向量

**下游模型如何使用 embedding**：
```python
# model.py GraphHead.forward()
if self.use_cl:
    xcl = self.cl_linear(batch.x)  # batch.x 就是 SGRL 的 embedding
    x = torch.cat((x, xcl), dim=1)  # 拼接到节点特征上
```

### 1.3 下游任务模型

下游模型 `GraphHead` 支持:
- **边任务**: 用 LinkNeighborLoader 采样，预测边标签
- **节点任务**: 用 NeighborLoader 采样，预测节点标签
- **回归/分类**: 通过 `--task` 参数切换

---

## 2. 数据格式对比

| 属性 | SRAM (原始) | RC 电路 (目标) |
|------|-------------|----------------|
| 节点类型 | net=0, dev=1, pin=2 | dev=0, pin=1, net=2 |
| 节点特征维度 | 17 | 16 (padding) |
| 边标签 | 耦合电容 (1e-21~1e-15 F) | 电阻 (0~700 Ω) |
| 节点标签 | 接地电容 (~1e-20 F) | 电容 (~1e-13 F) |
| 目标边类型 | cc_p2n, cc_p2p, cc_n2n | pair_to |
| 数据文件 | {name}.pt | case{N}_RC.pt |

---

## 3. 修改计划

### 3.1 sram_dataset.py (主要修改)

| 修改点 | 原始 | 修改后 |
|--------|------|--------|
| 节点类型常量 | NET=0, DEV=1, PIN=2 | DEV=0, PIN=1, NET=2 |
| 特征维度 | 17 | 16 |
| 边类型 | cc_p2n, cc_p2p, cc_n2n | pair_to |
| 边标签归一化 | log10(y*1e21)/6 | log1p(y)/log1p(700) |
| 节点标签归一化 | log10(y*1e20)/6 | log1p(y*1e15)/log1p(MAX*1e15) |
| 电源网络移除 | 硬编码 power_net_ids | 根据特征判断 VDD/VSS |
| 数据路径 | ./datasets/{name}.pt | ../data/case{N}_RC.pt |

**核心函数修改**:
1. `sram_graph_load()` → 重写为 `rc_graph_load()`
2. `norm_nfeat()` → 修改归一化公式
3. `raw_file_names` → 修改文件名格式
4. `processed_dir` → 修改处理目录

### 3.2 model.py (小修改)

```python
# 修改节点类型常量
DEV = 0  # 原: NET = 0
PIN = 1  # 原: DEV = 1
NET = 2  # 原: PIN = 2

# 修改特征维度
max_feat_dim = 16  # 原: 17
```

### 3.3 sgrl_models.py (小修改)

```python
# 同样修改节点类型常量
DEV = 0
PIN = 1
NET = 2
```

### 3.4 downstream_train.py (小修改)

```python
# 同样修改节点类型常量
DEV = 0
PIN = 1
NET = 2
```

### 3.5 main.py (参数修改)

```python
# 修改默认数据集名称
--dataset "1+5+7+10+11"  # 原: "ssram+digtime+..."

# 修改数据目录
--dataset_dir "../data/"  # 原: "./datasets/"
```

### 3.6 sampling.py (无需修改)

采样逻辑是通用的，不依赖具体数据格式。

---

## 4. 测试命令

```bash
# 边任务 - 回归 (不使用 SGRL)
python main.py --task_level edge --task regression --dataset 1+5+7 --sgrl 0

# 边任务 - 回归 (使用 SGRL)
python main.py --task_level edge --task regression --dataset 1+5+7 --sgrl 1

# 节点任务 - 分类
python main.py --task_level node --task classification --dataset 1+5+7
```

---

## 5. 工作量估计

| 文件 | 修改量 | 说明 |
|------|--------|------|
| sram_dataset.py | 大 | 重写数据加载和归一化 |
| model.py | 小 | 修改常量 |
| sgrl_models.py | 小 | 修改常量 |
| downstream_train.py | 小 | 修改常量 |
| main.py | 小 | 修改默认参数 |
| sampling.py | 无 | 不需要修改 |

**总预计时间**: 1-2 小时
