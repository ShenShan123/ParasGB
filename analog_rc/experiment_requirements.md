# 实验要求文档

## 1. 实验目标

需要完成共 `18` 组回归实验，并将结果整理为论文中的表格。

- 项目/模型共 `9` 组：
  - `change` 项目中的 `GCN`
  - `change` 项目中的 `GAT`
  - `change` 项目中的 `GraphSAGE`
  - `change` 项目中的 `PNA`
  - `change` 项目中的 `SGFormer`
  - `change` 项目中的 `PolyNormer`
  - `paragraph-simple`
  - `Cirgps`
  - `CircuitGCL`
- 每组实验都要跑两类任务：
  - `节点回归`
  - `边回归`

## 2. 统一数据划分

所有实验统一使用以下数据划分，不再使用原命令中旧的数据集设置。

- 训练集：`1+2+3+6+8+9+10+11+12+15+16+17+18`
- 测试集：`5+14+20`

## 3. 结果记录要求

最终只记录以下两个指标：

- `MAE`
- `R^2`

结果需要整理为两张表：

1. `节点回归`结果表
2. `边回归`结果表

每张表的格式要求如下：

- 每个模型一行
- 包含训练集整体结果
- 包含测试集 `5`、`14`、`20` 三个 case 的结果
- 每个数据块只保留 `MAE` 和 `R^2`
- 表格风格按照目标论文截图中的格式组织

## 4. 结果写入要求

实验完成后，需要将整理好的两张结果表填入飞书论文第 `9` 点 `Results` 部分。

## 5. 执行说明

- 优先按下面给出的命令执行
- 仅将其中训练集和测试集替换为本文件第 2 节中的统一划分
- 未在命令中显式指定的参数，保持项目默认值
- 最终以日志中可提取的 `MAE` 和 `R^2` 为准

---

## 6. 节点回归实验

统一任务设置：

- `task_level = node`
- `task = regression`

### 6.1 `change` 项目

工作目录：`D:\desktop\github_push\analog_rc\change`

#### PolyNormer

```bash
python main.py --model polynormer --task_level node --task regression --train_cases "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_cases "5+14+20" --num_classes 5 --hid_dim 96 --num_layers 4 --num_hops 3 --lr 0.00005 --dropout 0.4 --epochs 200 --activation leakyrelu
```

#### SGFormer

```bash
python main.py --model sgformer --task_level node --task regression --train_cases "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_cases "5+14+20" --num_classes 5 --hid_dim 96 --num_layers 4 --num_hops 3 --lr 0.00005 --dropout 0.4 --epochs 200 --activation leakyrelu
```

#### GAT

```bash
python main.py --model gat --task_level node --task regression --train_cases "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_cases "5+14+20" --num_classes 5 --hid_dim 96 --num_layers 4 --num_hops 3 --lr 0.00005 --dropout 0.4 --epochs 200 --activation leakyrelu
```

#### PNA

```bash
python main.py --model pna --task_level node --task regression --train_cases "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_cases "5+14+20" --num_classes 5 --hid_dim 96 --num_layers 4 --num_hops 3 --lr 0.00005 --dropout 0.4 --epochs 200 --activation leakyrelu
```

#### GCN

```bash
python main.py --model gcn --task_level node --task regression --train_cases "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_cases "5+14+20" --num_classes 5 --hid_dim 96 --num_layers 4 --num_hops 3 --lr 0.00005 --dropout 0.4 --epochs 200 --activation leakyrelu
```

#### GraphSAGE

```bash
python main.py --model sage --task_level node --task regression --train_cases "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_cases "5+14+20" --num_classes 5 --hid_dim 96 --num_layers 4 --num_hops 3 --lr 0.00005 --dropout 0.4 --epochs 200 --activation leakyrelu
```

### 6.2 `paragraph-simple` 项目

工作目录：`D:\desktop\github_push\analog_rc\paragraph-simple`

```bash
python main.py --task_level node --task regression --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --num_classes 5 --num_hops 4 --lr 0.0001 --dropout 0.4 --act_fn leakyrelu
```

### 6.3 `CircuitGCL` 项目

工作目录：`D:\desktop\github_push\analog_rc\CircuitGCL`

```bash
python main.py --model sage --task_level node --task regression --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --num_classes 5 --hid_dim 144 --num_gnn_layers 4 --num_hops 3 --lr 0.0001 --dropout 0.4 --epochs 200 --act_fn leakyrelu
```

### 6.4 `Cirgps` 项目

工作目录：`D:\desktop\github_push\analog_rc\Cirgps`

```bash
python main.py --model clustergcn --task_level node --task regression --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --num_classes 5 --hid_dim 84 --num_hops 3 --lr 0.0001 --dropout 0.4 --act_fn leakyrelu
```

---

## 7. 边回归实验

统一任务设置：

- `task_level = edge`
- `task = regression`

### 7.1 `change` 项目

工作目录：`D:\desktop\github_push\analog_rc\change`

#### PolyNormer

```bash
python main.py --model polynormer --task_level edge --task regression --train_cases "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_cases "5+14+20" --num_classes 5 --hid_dim 96 --num_layers 4 --num_hops 3 --lr 0.00005 --dropout 0.4 --epochs 200 --activation leakyrelu
```

#### SGFormer

```bash
python main.py --model sgformer --task_level edge --task regression --train_cases "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_cases "5+14+20" --num_classes 5 --hid_dim 96 --num_layers 4 --num_hops 3 --lr 0.00005 --dropout 0.4 --epochs 200 --activation leakyrelu
```

#### GAT

```bash
python main.py --model gat --task_level edge --task regression --train_cases "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_cases "5+14+20" --num_classes 5 --hid_dim 96 --num_layers 4 --num_hops 3 --lr 0.00005 --dropout 0.4 --epochs 200 --activation leakyrelu
```

#### PNA

```bash
python main.py --model pna --task_level edge --task regression --train_cases "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_cases "5+14+20" --num_classes 5 --hid_dim 96 --num_layers 4 --num_hops 3 --lr 0.00005 --dropout 0.4 --epochs 200 --activation leakyrelu
```

#### GCN

```bash
python main.py --model gcn --task_level edge --task regression --train_cases "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_cases "5+14+20" --num_classes 5 --hid_dim 96 --num_layers 4 --num_hops 3 --lr 0.00005 --dropout 0.4 --epochs 200 --activation leakyrelu
```

#### GraphSAGE

```bash
python main.py --model sage --task_level edge --task regression --train_cases "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_cases "5+14+20" --num_classes 5 --hid_dim 96 --num_layers 4 --num_hops 3 --lr 0.00005 --dropout 0.4 --epochs 200 --activation leakyrelu
```

### 7.2 `paragraph-simple` 项目

工作目录：`D:\desktop\github_push\analog_rc\paragraph-simple`

```bash
python main.py --task_level edge --task regression --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --num_classes 5 --num_hops 4 --lr 0.0001 --dropout 0.4 --act_fn leakyrelu
```

### 7.3 `CircuitGCL` 项目

工作目录：`D:\desktop\github_push\analog_rc\CircuitGCL`

```bash
python main.py --model sage --task_level edge --task regression --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --num_classes 5 --hid_dim 144 --num_gnn_layers 4 --num_hops 3 --lr 0.0001 --dropout 0.4 --epochs 200 --act_fn leakyrelu
```

### 7.4 `Cirgps` 项目

工作目录：`D:\desktop\github_push\analog_rc\Cirgps`

```bash
python main.py --model clustergcn --task_level edge --task regression --train_dataset "1+2+3+6+8+9+10+11+12+15+16+17+18" --test_dataset "5+14+20" --num_classes 5 --hid_dim 84 --num_hops 3 --lr 0.0001 --dropout 0.4 --act_fn leakyrelu
```

---

## 8. 最终交付物

最终需要交付以下内容：

1. `18` 组实验的原始日志
2. `节点回归`结果表
3. `边回归`结果表
4. 将两张表填入飞书论文第 `9` 点 `Results`

## 9. 结果表字段说明

每张表建议采用如下列结构：

- `Metric / Model`
- `Train: MAE`
- `Train: R^2`
- `Case 5: MAE`
- `Case 5: R^2`
- `Case 14: MAE`
- `Case 14: R^2`
- `Case 20: MAE`
- `Case 20: R^2`

只允许出现 `MAE` 与 `R^2` 两类指标，不填入其他指标。
