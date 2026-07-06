# Requirements Document

## Introduction

本文档定义了一个统一的数据集API接口系统，用于RC电路图数据的加载、处理、评估和测试。目前项目中存在多个模块（change、CircuitGCL、Cirgps、paragraph-simple），每个模块都有自己的数据集实现，导致代码重复和维护困难。统一的API将提供一致的接口来处理数据集的创建、划分、加载和评估。

## Glossary

- **Dataset**: 数据集，包含RC电路图数据，分为sram和analog两种类型
- **DataLoader**: 数据加载器，用于批量加载数据
- **Evaluator**: 评估器，用于计算模型性能指标，根据数据集类型和任务组合选择
- **Task_Level**: 任务级别，包括节点任务(node)和边任务(edge)
- **Task_Type**: 任务类型，包括回归(regression)和分类(classification)
- **Task_Combination**: 任务组合，格式为{level}_{type}的缩写，如cg_class表示coarse-grained classification
- **Split**: 数据划分，包括训练集(train)、验证集(val)、测试集(test)
- **RC_Graph**: RC电路图，包含器件(dev)、引脚(pin)、网络(net)节点和它们之间的边
- **Normalization**: 归一化，将标签值映射到[0,1]区间
- **SRAM_Dataset**: SRAM电路数据集，train_cases和test_cases已预定义
- **Analog_Dataset**: 模拟电路数据集，train_cases和test_cases已预定义

## Requirements

### Requirement 1: 数据集创建

**User Story:** 作为开发者，我想要通过统一的接口创建数据集，以便在不同模块中使用相同的数据加载方式。

#### Acceptance Criteria

1. WHEN 用户指定数据集名称(sram/analog) THEN THE System SHALL 加载对应类型的数据集
2. WHEN 用户指定数据目录和case ID列表 THEN THE System SHALL 加载对应的RC电路图数据
3. WHEN 用户指定任务级别(node/edge) THEN THE System SHALL 提取对应的标签数据
4. WHEN 用户指定任务类型(regression/classification) THEN THE System SHALL 对标签进行相应的处理
5. THE System SHALL 支持预定义的数据集划分(train_cases和test_cases已固定)
6. THE System SHALL 支持数据缓存机制以加快重复加载速度
7. THE System SHALL 过滤掉电源网络节点(VDD/VSS/GND)
8. THE System SHALL 支持节点特征的padding和归一化

### Requirement 2: 数据集划分

**User Story:** 作为开发者，我想要将数据集划分为训练集、验证集和测试集，以便进行模型训练和评估。

#### Acceptance Criteria

1. WHEN 用户指定训练集和测试集的case ID THEN THE System SHALL 创建对应的数据集对象
2. WHEN 用户指定验证集比例 THEN THE System SHALL 从训练集中划分出验证集
3. THE System SHALL 支持自定义的训练/测试划分策略
4. THE System SHALL 确保训练集、验证集、测试集之间没有数据泄漏
5. THE System SHALL 支持多个测试集的同时评估

### Requirement 3: DataLoader创建

**User Story:** 作为开发者，我想要创建DataLoader对象，以便批量加载数据进行训练。

#### Acceptance Criteria

1. WHEN 用户指定batch size THEN THE System SHALL 创建对应批次大小的DataLoader
2. WHEN 任务级别为node THEN THE System SHALL 使用NeighborLoader进行邻居采样
3. WHEN 任务级别为edge THEN THE System SHALL 使用LinkNeighborLoader进行边采样
4. THE System SHALL 支持自定义采样参数(num_neighbors, num_hops)
5. THE System SHALL 支持数据增强和负采样
6. THE System SHALL 在训练集上启用shuffle，在验证/测试集上禁用shuffle

### Requirement 4: 标签归一化

**User Story:** 作为开发者，我想要对标签进行归一化处理，以便模型能够更好地学习。

#### Acceptance Criteria

1. WHEN 任务级别为node THEN THE System SHALL 使用log归一化处理电容值(~1e-13 F)
2. WHEN 任务级别为edge THEN THE System SHALL 使用log归一化处理电阻值(0-700 Ω)
3. WHEN 任务类型为classification THEN THE System SHALL 将归一化后的值分桶为离散类别
4. THE System SHALL 支持自定义分类边界
5. THE System SHALL 同时保存回归标签和分类标签(格式: [N, 2])
6. THE System SHALL 提供反归一化函数以恢复原始标签值

### Requirement 5: 模型评估

**User Story:** 作为开发者，我想要使用统一的评估器计算模型性能，以便比较不同模型的效果。

#### Acceptance Criteria

1. WHEN 用户指定数据集类型(sram/analog)和任务组合 THEN THE Evaluator SHALL 使用对应的评估器
2. THE System SHALL 支持以下任务组合: cg_class, cg_regr, cc_class, cc_regr, r_class, r_regr
3. WHEN 任务类型为regression THEN THE Evaluator SHALL 计算MAE、MSE、RMSE、R2指标
4. WHEN 任务类型为classification THEN THE Evaluator SHALL 计算Accuracy、F1、Precision、Recall指标
5. THE Evaluator SHALL 支持批量评估
6. THE Evaluator SHALL 支持多个测试集的同时评估
7. THE Evaluator SHALL 返回结构化的评估结果字典
8. THE Evaluator SHALL 支持自定义评估指标

### Requirement 6: 配置管理

**User Story:** 作为开发者，我想要通过配置文件管理数据集参数，以便快速切换不同的实验设置。

#### Acceptance Criteria

1. THE System SHALL 支持从配置文件加载数据集参数
2. THE System SHALL 支持命令行参数覆盖配置文件
3. THE System SHALL 验证配置参数的有效性
4. THE System SHALL 提供默认配置模板
5. THE System SHALL 记录使用的配置参数以便复现实验

### Requirement 7: 错误处理

**User Story:** 作为开发者，我想要系统能够优雅地处理错误情况，以便快速定位问题。

#### Acceptance Criteria

1. WHEN 数据文件不存在 THEN THE System SHALL 抛出清晰的错误信息
2. WHEN 标签数据缺失或无效 THEN THE System SHALL 过滤掉这些样本并记录日志
3. WHEN 配置参数无效 THEN THE System SHALL 抛出参数验证错误
4. THE System SHALL 在关键步骤记录日志信息
5. THE System SHALL 提供调试模式以输出详细信息

### Requirement 8: 性能优化

**User Story:** 作为开发者，我想要系统能够高效地加载和处理数据，以便加快训练速度。

#### Acceptance Criteria

1. THE System SHALL 使用缓存机制避免重复处理数据
2. THE System SHALL 支持多进程数据加载
3. THE System SHALL 支持数据预取(prefetch)
4. THE System SHALL 支持内存映射(memory mapping)以处理大规模数据
5. THE System SHALL 提供性能分析工具以识别瓶颈
