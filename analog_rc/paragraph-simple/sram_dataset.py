"""
RC Circuit Dataset - 适配 data/ 目录下的 RC 电路数据
原始 paragraph-simple 的 SRAM 数据集加载器，已修改为支持新的 RC 数据格式

任务类型:
- 节点任务: 预测 net 节点的电容值 (~1e-13 F)
- 边任务: 预测 pair_to 边的电阻值 (0~700 Ω)
"""
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import os
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
import logging
import time
from pathlib import Path
from utils import collated_data_separate
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from node_encoder import DEFAULT_FEAT_CONFIG, NODE_TYPE_ID_TO_NAME, expand_indices

# 节点类型顺序 (与 newgraph.py 一致)
NODE_TYPE_ORDER = ['dev', 'pin', 'net']
NODE_TYPE_TO_ID = {ntype: i for i, ntype in enumerate(NODE_TYPE_ORDER)}

# 边类型顺序 (与 newgraph.py 一致)
EDGE_TYPE_ORDER = [
    ('dev', 'connects_to', 'pin'),
    ('dev', 'connects_to', 'net'),
    ('pin', 'belongs_to', 'net'),
    ('pin', 'pair_to', 'pin'),
]

# 标签范围
MAX_EDGE_LABEL = 700.0  # 电阻最大值 (Ω)
MAX_NODE_LABEL = 8e-13  # 电容最大值 (F)


class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        name,
        root,
        to_undirected=True,
        sample_rates=[1.0],
        task_type='classification',
        num_classes=5,
        transform=None, 
        pre_transform=None
    ) -> None:
        """RC Circuit Dataset
        
        Args:
            name (str): Case ID，多个用 '+' 连接，如 "1+5+7"
            root (str): 数据根目录
            to_undirected (bool): 是否转为无向图
            sample_rates (list): 采样率
            task_type (str): 任务类型 ('classification', 'regression', 'node_regression', 'node_classification')
            num_classes (int): 分类任务的类别数
        """
        self.name = 'rc_circuit'

        # 解析多个数据集名称
        if '+' in name:
            self.names = name.split('+')
        else:
            self.names = [name]
            
        print(f"RCDataset includes cases: {self.names}")

        self.sample_rates = sample_rates
        assert len(self.names) == len(self.sample_rates), \
            f"len of dataset:{len(self.names)}, len of sample_rate: {len(self.sample_rates)}"
        
        self.folder = root  # 直接使用 root 作为数据目录
        self.to_undirected_flag = to_undirected
        self.data_lengths = {}
        self.data_offsets = {}
        self.task_type = task_type
        self.is_node_task = task_type.startswith('node_')  # 判断是否为节点任务
        self._num_classes = num_classes
        self.max_net_node_feat = torch.ones((1, 16))
        self.max_dev_node_feat = torch.ones((1, 16))
        
        super().__init__(self.folder, transform, pre_transform)
        data_list = []

        for i, name in enumerate(self.names):
            loaded_data, loaded_slices = torch.load(self.processed_paths[i])
            
            self.data_offsets[name] = len(data_list)
            # 根据任务类型获取数据长度
            if self.is_node_task:
                self.data_lengths[name] = loaded_data.node_label.size(0) if hasattr(loaded_data, 'node_label') else 0
            else:
                self.data_lengths[name] = loaded_data.edge_label.size(0) if hasattr(loaded_data, 'edge_label') else 0
            
            if loaded_slices is not None:
                data_list += collated_data_separate(loaded_data, loaded_slices)
            else:
                data_list.append(loaded_data)
            
            print(f"load processed case{name}, " +
                  f"len(data_list)={self.data_lengths[name]}, " +
                  f"data_offset={self.data_offsets[name]}")
    
        self.data, self.slices = self.collate(data_list)

    def _get_data_store(self):
        data_store = getattr(self, '_data', None)
        if data_store is not None:
            return data_store

        if self.slices is not None:
            return self.data

        if getattr(self, '_data_list', None) is not None:
            self.data, self.slices = self.collate(self._data_list)
            self._data_list = None
            return self.data

        raise RuntimeError("Dataset data storage is not initialized.")

    def norm_nfeat(self, ntypes):
        """归一化节点特征和标签
        
        Args:
            ntypes: 节点类型列表
        
        标签处理:
            - 归一化原始标签值
            - 分桶生成分类标签
            - 最终格式: [N, 2], 第0列回归，第1列分类
        """
        self._data = self._get_data_store()

        # 归一化节点特征
        for ntype in ntypes:
            node_mask = self._data.node_type == ntype
            if node_mask.sum() == 0:
                continue
            type_name = NODE_TYPE_ID_TO_NAME.get(int(ntype))
            feat_config = DEFAULT_FEAT_CONFIG.get(type_name, {})
            continuous_indices = expand_indices(feat_config.get('continuous', []))
            if not continuous_indices:
                print(f"skip continuous node_attr normalization for type {ntype}: no continuous dims")
                continue

            type_attr = self._data.node_attr[node_mask].clone()
            cont_attr = type_attr[:, continuous_indices]
            max_node_feat, _ = cont_attr.max(dim=0, keepdim=True)
            max_node_feat[max_node_feat == 0.0] = 1.0

            print(
                f"normalizing continuous node_attr type {ntype}: "
                f"max={max_node_feat.max().item():.4f}, dims={continuous_indices}"
            )
            type_attr[:, continuous_indices] = cont_attr / max_node_feat
            self._data.node_attr[node_mask] = type_attr
        
        # 分类边界
        n_classes = self._num_classes
        boundaries = torch.linspace(0, 1, n_classes + 1)[1:-1]
        
        if self.is_node_task:
            # 归一化节点标签: 原始 paragraph-simple 逻辑
            self._data.node_label = torch.log10(self._data.node_label * 1e21)
            self._data.node_label /= 6
            self._data.node_label[self._data.node_label < 0] = 0.0
            self._data.node_label[self._data.node_label > 1] = 1.0
            
            # 分类标签: 对归一化后的值分桶
            node_label_c = torch.bucketize(self._data.node_label, boundaries)
            self._data.node_label = torch.stack(
                [self._data.node_label.squeeze(), node_label_c.squeeze().float()], dim=1
            )

            if self.slices is not None and "x" in self.slices and "node_label" in self.slices:
                global_target_indices = []
                num_graphs = self.slices["x"].numel() - 1
                for graph_idx in range(num_graphs):
                    node_offset = int(self.slices["x"][graph_idx])
                    label_start = int(self.slices["node_label"][graph_idx])
                    label_end = int(self.slices["node_label"][graph_idx + 1])
                    if label_end <= label_start:
                        continue
                    local_indices = self._data.node_label_index[label_start:label_end]
                    global_target_indices.append(local_indices + node_offset)
                global_target_indices = (
                    torch.cat(global_target_indices, dim=0)
                    if global_target_indices
                    else torch.empty(0, dtype=torch.long)
                )
            else:
                global_target_indices = self._data.node_label_index

            full_y = torch.zeros(
                (self._data.num_nodes, self._data.node_label.size(1)),
                dtype=self._data.node_label.dtype,
            )
            target_node_mask = torch.zeros(self._data.num_nodes, dtype=torch.bool)
            full_y[global_target_indices] = self._data.node_label
            target_node_mask[global_target_indices] = True
            self._data.y = full_y
            self._data.target_node_mask = target_node_mask
            if self.slices is not None and "x" in self.slices:
                self.slices["y"] = self.slices["x"].clone()
                self.slices["target_node_mask"] = self.slices["x"].clone()
            
            print(f"node_label normalized: shape={self._data.node_label.shape}")
            print(f"  回归标签范围: [{self._data.node_label[:, 0].min():.4f}, {self._data.node_label[:, 0].max():.4f}]")
            print(f"  分类标签类别: {self._data.node_label[:, 1].unique().long().tolist()}")
        else:
            # 归一化边标签: 原始 paragraph-simple 逻辑
            self._data.edge_label = torch.log10(self._data.edge_label * 1e21)
            self._data.edge_label /= 6
            self._data.edge_label[self._data.edge_label < 0] = 0.0
            self._data.edge_label[self._data.edge_label > 1] = 1.0
            
            # 分类标签: 对归一化后的值分桶
            edge_label_c = torch.bucketize(self._data.edge_label, boundaries)
            self._data.edge_label = torch.stack(
                [self._data.edge_label.squeeze(), edge_label_c.squeeze().float()], dim=1
            )
            
            print(f"edge_label normalized: shape={self._data.edge_label.shape}")
            print(f"  回归标签范围: [{self._data.edge_label[:, 0].min():.4f}, {self._data.edge_label[:, 0].max():.4f}]")
            print(f"  分类标签类别: {self._data.edge_label[:, 1].unique().long().tolist()}")
        
        self._data_list = None


    def rc_graph_load(self, name, raw_path):
        """
        加载 RC 电路图数据
        
        数据格式 (来自 newgraph.py):
        - 节点: dev (16维), pin (6维), net (10维)
        - 边: connects_to, belongs_to, pair_to
        - 节点标签: net.y (电容, ~1e-13 F)
        - 边标签: pair_to.y (电阻, 0~700 Ω)
        
        Returns:
            g: 处理后的同构图
        """
        logging.info(f"Loading RC graph from: {raw_path}")
        hg = torch.load(raw_path)
        if isinstance(hg, list):
            hg = hg[0]
        
        print(f"  原始异构图: {hg}")
        
        # 根据特征移除电源网络 (net.x[:, 0]=VDD, net.x[:, 1]=VSS)
        subset_dict = {}
        for ntype in hg.node_types:
            subset_dict[ntype] = torch.ones(hg[ntype].num_nodes, dtype=torch.bool)
            if ntype == 'net' and hasattr(hg['net'], 'x'):
                net_x = hg['net'].x
                is_power_net = (net_x[:, 0] == 1.0) | (net_x[:, 1] == 1.0)
                power_count = is_power_net.sum().item()
                print(f"  移除 {power_count} 个电源网络 (VDD/VSS)")
                subset_dict[ntype] = ~is_power_net
        
        hg = hg.subgraph(subset_dict)
        
        # 保留需要的边类型
        edge_types_to_keep = [
            ('dev', 'connects_to', 'pin'),
            ('dev', 'connects_to', 'net'),
            ('pin', 'belongs_to', 'net'),
            ('pin', 'pair_to', 'pin'),
        ]
        existing_edge_types = [et for et in edge_types_to_keep if et in hg.edge_types]
        if existing_edge_types:
            hg = hg.edge_type_subgraph(existing_edge_types)
        
        print(f"  过滤后异构图: {hg}")
        
        # 计算边偏移量 (用于定位 pair_to 边)
        edge_offset = 0
        for etype in [('dev', 'connects_to', 'pin'), ('dev', 'connects_to', 'net'), ('pin', 'belongs_to', 'net')]:
            if etype in hg.edge_types:
                edge_offset += hg[etype].edge_index.shape[1]
        
        # 获取 pair_to 边的标签
        pair_to_etype = ('pin', 'pair_to', 'pin')
        tar_edge_y = None
        tar_edge_count = 0
        if pair_to_etype in hg.edge_types and hasattr(hg[pair_to_etype], 'y'):
            tar_edge_y = hg[pair_to_etype].y.squeeze()
            tar_edge_count = hg[pair_to_etype].edge_index.shape[1]
        
        # 转换为同构图
        g = hg.to_homogeneous()
        g.name = f"case{name}"
        
        # 节点类型映射
        g._n2type = NODE_TYPE_TO_ID.copy()
        g._e2type = {etype: i for i, etype in enumerate(existing_edge_types)}
        g._num_ntypes = len(g._n2type)
        g._num_etypes = len(g._e2type)
        
        # 构建节点特征 (padding 到统一维度)
        max_feat_dim = 16
        node_feat = []
        for ntype in NODE_TYPE_ORDER:
            if ntype in hg.node_types and hasattr(hg[ntype], 'x'):
                feat = hg[ntype].x
                if feat.size(1) < max_feat_dim:
                    feat = torch.nn.functional.pad(feat, (0, max_feat_dim - feat.size(1)))
                elif feat.size(1) > max_feat_dim:
                    feat = feat[:, :max_feat_dim]
                node_feat.append(feat)
        
        g.x = g.node_type.view(-1, 1)
        g.node_attr = torch.cat(node_feat, dim=0) if node_feat else torch.zeros((g.num_nodes, max_feat_dim))
        
        # 节点标签 (net 节点的电容值)
        tar_node_y_list = []
        for ntype in NODE_TYPE_ORDER:
            if ntype in hg.node_types:
                num_nodes = hg[ntype].num_nodes
                if ntype == 'net' and hasattr(hg['net'], 'y'):
                    tar_node_y_list.append(hg['net'].y)
                else:
                    tar_node_y_list.append(torch.zeros((num_nodes, 1)))
        g.tar_node_y = torch.cat(tar_node_y_list, dim=0) if tar_node_y_list else torch.zeros((g.num_nodes, 1))
        
        # 处理边标签
        if tar_edge_y is not None and tar_edge_count > 0:
            tar_edge_index = g.edge_index[:, edge_offset:edge_offset + tar_edge_count]
            
            # 过滤有效标签 (0 < y <= 700)
            valid_mask = (tar_edge_y > 0) & (tar_edge_y <= MAX_EDGE_LABEL)
            valid_count = valid_mask.sum().item()
            filtered_count = tar_edge_count - valid_count
            print(f"  有效边标签: {valid_count}, 过滤掉: {filtered_count}")
            
            g.tar_edge_y = tar_edge_y[valid_mask]
            g.tar_edge_index = tar_edge_index[:, valid_mask]
            g.tar_edge_type = torch.zeros(valid_count, dtype=torch.long)  # 只有一种目标边类型
            g.tar_edge_dist = torch.tensor([valid_count])
            
            # 移除目标边，只保留结构边
            g.edge_index = g.edge_index[:, :edge_offset]
            g.edge_type = g.edge_type[:edge_offset]
        else:
            g.tar_edge_y = torch.empty(0)
            g.tar_edge_index = torch.empty((2, 0), dtype=torch.long)
            g.tar_edge_type = torch.empty(0, dtype=torch.long)
            g.tar_edge_dist = torch.tensor([0])
        
        # 删除原始 y 属性
        if hasattr(g, 'y'):
            del g.y
        
        # 转为无向图
        if self.to_undirected_flag:
            g.edge_index, g.edge_type = to_undirected(
                g.edge_index, g.edge_type, g.num_nodes, reduce='mean'
            )
            g.edge_type = g.edge_type.long()
        
        print(f"  处理后同构图: nodes={g.num_nodes}, edges={g.edge_index.shape[1]}, tar_edges={g.tar_edge_index.shape[1]}")
        
        return g

    def single_g_process(self, idx: int):
        """处理单个图"""
        logging.info(f"Processing case {self.names[idx]} with sample_rate {self.sample_rates[idx]}...")
        
        graph = self.rc_graph_load(self.names[idx], self.raw_paths[idx])
        logging.info(f"Loaded graph: {graph}")
        
        # 根据任务类型处理
        if self.is_node_task:
            # 节点任务: 预测 net 节点的电容值 (只保存原始标签，归一化和分桶在 norm_nfeat 中处理)
            NET = 2  # net 节点类型 ID
            net_mask = graph.node_type == NET
            net_indices = torch.where(net_mask)[0]
            net_labels = graph.tar_node_y[net_mask].squeeze()
            
            # 过滤有效标签 (0 < y <= MAX_NODE_LABEL)
            if net_labels.dim() == 0:
                net_labels = net_labels.unsqueeze(0)
            valid_mask = (net_labels > 0) & (net_labels <= MAX_NODE_LABEL)
            valid_count = valid_mask.sum().item()
            filtered_count = net_labels.size(0) - valid_count
            print(f"  有效节点标签: {valid_count}, 过滤掉: {filtered_count}")
            
            graph.node_label_index = net_indices[valid_mask]
            raw_labels = net_labels[valid_mask]
            
            # 保存原始标签 (归一化和分桶在 norm_nfeat 中处理)
            graph.node_label = raw_labels
            
            print(f"  节点标签: shape={graph.node_label.shape}, 原始范围=[{raw_labels.min():.6e}, {raw_labels.max():.6e}]")
            
            # 清理边任务相关属性
            if hasattr(graph, 'tar_edge_y'):
                del graph.tar_edge_y
            if hasattr(graph, 'tar_edge_index'):
                del graph.tar_edge_index
            if hasattr(graph, 'tar_edge_type'):
                del graph.tar_edge_type
            if hasattr(graph, 'tar_node_y'):
                del graph.tar_node_y
                
            # 设置空的边标签 (兼容性)
            graph.edge_label_index = torch.empty((2, 0), dtype=torch.long)
            graph.edge_label = torch.empty(0)
            
        else:
            # 边任务: 预测 pair_to 边的电阻值 (只保存原始标签，归一化和分桶在 norm_nfeat 中处理)
            pos_edge_index = graph.tar_edge_index
            pos_edge_y = graph.tar_edge_y
            
            # 采样
            sample_rate = self.sample_rates[idx]
            if sample_rate < 1.0:
                num_edges = pos_edge_index.size(1)
                indices = torch.randperm(num_edges)[:int(num_edges * sample_rate)]
                pos_edge_index = pos_edge_index[:, indices]
                pos_edge_y = pos_edge_y[indices]
            
            print(f"  边任务: 使用 {pos_edge_index.size(1)} 条正边 (采样率={sample_rate})")
            
            # 保存原始标签 (归一化和分桶在 norm_nfeat 中处理)
            graph.edge_label = pos_edge_y
            graph.edge_label_index = pos_edge_index
            
            print(f"  边标签: shape={graph.edge_label.shape}, 原始范围=[{pos_edge_y.min():.4f}, {pos_edge_y.max():.4f}]")
            
            # 清理临时属性
            if hasattr(graph, 'tar_node_y'):
                del graph.tar_node_y
            if hasattr(graph, 'tar_edge_index'):
                del graph.tar_edge_index
            if hasattr(graph, 'tar_edge_type'):
                del graph.tar_edge_type
            if hasattr(graph, 'tar_edge_y'):
                del graph.tar_edge_y
            
            # 设置空的节点标签 (兼容性)
            graph.node_label_index = torch.empty(0, dtype=torch.long)
            graph.node_label = torch.empty(0)

        torch.save((graph, None), self.processed_paths[idx])
        
        if self.is_node_task:
            return graph.node_label.size(0)
        else:
            return graph.edge_label.size(0)

    def process(self):
        """处理所有图"""
        for i, name in enumerate(self.names):
            if os.path.exists(self.processed_paths[i]):
                logging.info(f"Found processed file for case{name}, skipping...")
                continue 
            self.single_g_process(i)

    @property
    def raw_dir(self) -> str:
        return self.folder

    @property
    def raw_file_names(self):
        return [f'case{name}_RC.pt' for name in self.names]
    
    @property
    def processed_dir(self) -> str:
        base_dir = os.path.join(self.folder, 'processed_paragraph')
        if self.is_node_task:
            # 节点任务: node_regression 或 node_classification
            task_dir = f"{self.task_type}_c{self._num_classes}"
        else:
            # 边任务: regression 或 classification (统一格式)
            task_dir = f"{self.task_type}_c{self._num_classes}"
        return os.path.join(base_dir, task_dir)

    @property
    def processed_file_names(self):
        processed_names = []
        for i, name in enumerate(self.names):
            fname = f"case{name}"
            if self.sample_rates[i] < 1.0:
                fname += f"_s{self.sample_rates[i]}"
            processed_names.append(fname + "_processed.pt")
        return processed_names


def performat_SramDataset(dataset_dir, name, 
                          to_undirected, 
                          sample_rates, task_type, num_classes=5):
    """创建数据集的便捷函数"""
    start = time.perf_counter()
    num_datasets = len(name.split('+'))

    # 如果sample_rates是单个值，则为所有数据集使用相同的采样率
    if isinstance(sample_rates, (int, float)):
        sample_rates = [sample_rates] * num_datasets

    dataset = SealSramDataset(
        name=name, 
        root=dataset_dir,
        to_undirected=to_undirected,
        sample_rates=sample_rates,
        task_type=task_type,
        num_classes=num_classes,
    )

    elapsed = time.perf_counter() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) + f'{elapsed:.2f}'[-3:]
    print(f"PID = {os.getpid()}")
    print(f"Building dataset {name} took {timestr}")
    print('Dataloader: Loading success.')

    return dataset
