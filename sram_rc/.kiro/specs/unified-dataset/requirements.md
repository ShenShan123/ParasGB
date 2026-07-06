# Requirements Document: 统一数据集路径配置

## Introduction

本文档描述了让 `rcg`、`paragraph` 和 `Cirgps` 三个项目都能读取 `sram` 目录下数据集的需求。目前三个项目的数据集加载逻辑各不相同，需要统一配置以支持共享数据集路径。

## 实现状态: ✅ 已完成

## 当前状态分析

### sram 目录数据文件
```
sram/
├── array_128_32_8t.pt
├── digtime.pt
├── sandwich.pt
├── sp8192w.pt
├── ssram.pt
├── timing_ctrl.pt
└── ultra8t.pt
```

### 修改后的统一配置

| 项目 | dataset_dir | raw_dir | 数据集名称格式 |
|------|-------------|---------|---------------|
| rcg | D:/desktop/github_push/sram_rc | {root}/sram | 直接名称 (如 `sandwich`) |
| paragraph | D:/desktop/github_push/sram_rc | {root}/sram | 直接名称 (如 `sandwich`) |
| Cirgps | D:/desktop/github_push/sram_rc | {root}/sram | 直接名称 (如 `sandwich`) |

## 已完成的修改

### 1. rcg/main.py
- `--dataset_dir` 默认值改为 `D:/desktop/github_push/sram_rc`
- `--train_dataset` 默认值改为 `sandwich+ultra8t`
- `--test_dataset` 默认值改为 `ssram+digtime+timing_ctrl+array_128_32_8t`

### 2. rcg/sram_dataset.py
- `folder` 路径改为 `os.path.join(root, 'sram')`
- `raw_dir` 属性返回 `self.folder`
- 数据集名称解析改为直接使用文件名（不再使用 `case{id}_RC` 格式）

### 3. paragraph/main.py
- `dataset_dir` 改为 `D:/desktop/github_push/sram_rc`

### 4. paragraph/sram_dataset.py
- 添加 `raw_dir` 属性，返回 `self.folder`

### 5. Cirgps/main.py
- `dataset_dir` 改为 `D:/desktop/github_push/sram_rc`

### 6. Cirgps/sram_dataset.py
- 添加 `raw_dir` 属性，返回 `self.folder`
