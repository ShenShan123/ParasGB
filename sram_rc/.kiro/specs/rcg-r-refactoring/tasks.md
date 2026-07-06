# rcg_r 等效电阻预测项目重构 - 任务清单

## 任务列表

- [x] 1. 复制 layer.py 文件
  - [x] 1.1 将 `rcg/layer.py` 复制到 `rcg_r/layer.py`
  - [x] 1.2 验证文件包含 `GatedGCNLayer`, `GCNConvLayer`, `GINEConvLayer` 类

- [x] 2. 删除不需要的文件
  - [x] 2.1 删除 `rcg_r/gps_layer.py`
  - [x] 2.2 删除 `rcg_r/gsl_module.py`
  - [x] 2.3 删除 `rcg_r/sgrl_models.py`
  - [x] 2.4 删除 `rcg_r/sgrl_train.py`

- [x] 3. 重构 sram_dataset.py
  - [x] 3.1 修改 `SealSramDataset` 类签名，使用 `train_cases` 和 `test_cases` 参数
  - [x] 3.2 添加 `num_train` 和 `num_test` 属性追踪
  - [x] 3.3 修改 `performat_SramDataset()` 函数签名
  - [x] 3.4 移除 `adaption_for_sgrl()` 函数和 `set_cl_embeds()` 方法
  - [x] 3.5 保持电阻边过滤逻辑 (`'r'` 关键字) 和 Min-Max 归一化不变

- [x] 4. 重构 main.py
  - [x] 4.1 移除 GPS 相关参数 (`local_gnn_type`, `global_model_type`, `attn_dropout`, `layer_norm`, `batch_norm`, `num_heads`, `g_bn`, `g_drop`, `g_ffn`)
  - [x] 4.2 移除 GSL 相关参数 (`use_gsl`, `gsl_layers`, `gsl_heads`, `gsl_weight`, `gsl_memory_efficient`, `gsl_checkpoint_freq`)
  - [x] 4.3 移除 SGRL 相关参数和代码 (`sgrl`, `e1_lr`, `e2_lr`, `momentum`, `weight_decay`, `cl_epochs`, `cl_model`, `cl_act_fn`, `cl_gnn_layers`, `cl_hid_dim`, `cl_batch_size`, `cl_num_neighbors`, `cl_dropout`)
  - [x] 4.4 移除 `sgrl_train` 导入和调用
  - [x] 4.5 添加 `--data_dir`, `--train_dataset`, `--test_dataset` 参数
  - [x] 4.6 添加 `--residual`, `--ffn` 自定义层参数
  - [x] 4.7 添加 `--num_heads`, `--global_layers` Transformer 参数
  - [x] 4.8 添加 t-SNE 可视化参数
  - [x] 4.9 修改模型选择列表，移除 `gps_attention`，添加完整模型列表
  - [x] 4.10 修改数据集加载调用

- [x] 5. 重构 model.py
  - [x] 5.1 移除 `GPSLayer` 导入和相关代码
  - [x] 5.2 移除 `use_cl` 和对比学习相关代码 (`cl_linear`, `xcl` 等)
  - [x] 5.3 添加 `layer.py` 中自定义层的导入
  - [x] 5.4 添加 `SGFormer` 和 `Polynormer` 支持
  - [x] 5.5 添加 `PNAConv` 支持
  - [x] 5.6 简化 `GraphHead.__init__` 参数
  - [x] 5.7 修改 `forward` 方法支持所有模型类型

- [x] 6. 重构 sampling.py
  - [x] 6.1 使用 `dataset.num_train` 动态获取训练图数量
  - [x] 6.2 修改测试集循环为 `range(num_train, len(dataset))`
  - [x] 6.3 移除硬编码的 `Batch.from_data_list` 调用
  - [x] 6.4 添加 `plot_tsne_visualization()` 函数

- [x] 7. 添加日志功能到 downstream_train.py
  - [x] 7.1 添加日志目录创建逻辑
  - [x] 7.2 添加日志文件名生成逻辑
  - [x] 7.3 添加输出重定向 (Tee 类)

- [ ]* 8. 功能测试
  - [ ]* 8.1 测试基础 GNN 模型 (gcn, sage, gat)
  - [ ]* 8.2 测试自定义 GNN+ 层 (CustomGatedGCN, CustomGCNConv, CustomGINEConv)
  - [ ]* 8.3 测试 Transformer 模型 (sgformer, polynormer)
  - [ ]* 8.4 测试多数据集训练/测试配置
