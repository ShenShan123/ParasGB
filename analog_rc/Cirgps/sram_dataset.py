"""
RC Circuit Dataset - 适配 data/ 目录下的 RC 电路数据
原始 Cirgps 的 SRAM 数据集加载器，已修改为支持新的 RC 数据格式

任务类型:
- 边任务: 预测 pair_to 边的电阻值 (0~700 Ω)
- 节点任务: 预测 net 节点的电容值 (~1e-13 F)
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
from utils import (
    get_pos_neg_edges, add_tar_edges_to_g, get_balanced_edges, 
    collated_data_separate)
from torch.utils.data import Dataset
from torch_geometric.data import Batch

# 节点类型顺序 (与 newgraph.py 一致)
NODE_TYPE_ORDER = ['dev', 'pin', 'net']
NODE_TYPE_TO_ID = {ntype: i for i, ntype in enumerate(NODE_TYPE_ORDER)}
DEV = 0
PIN = 1
NET = 2

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
        add_target_edges=False,
        neg_edge_ratio=1.0,
        to_undirected=True,
        sample_rates=[1.0],
        task_type='regression',
        num_classes=5,  # 添加 num_classes 参数
        transform=None, 
        pre_transform=None
    ) -> None:
        """RC Circuit Dataset for Cirgps
        
        Args:
            name (str): Case ID，多个用 '+' 连接，如 "1+5+7"
            root (str): 数据根目录
            add_target_edges (bool): 是否将目标边添加到图中
            neg_edge_ratio (float): 负边比例 (已废弃，保留兼容性)
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
        
        self.folder = root
        self.add_target_edges = add_target_edges
        self.neg_edge_ratio = neg_edge_ratio  # 保留但不使用
        self.to_undirected_flag = to_undirected
        self.data_lengths = {}
        self.data_offsets = {}
        self.task_type = task_type
        self.task_level = 'node' if task_type.startswith('node_') else 'edge'
        self._num_classes = num_classes
        self.max_net_node_feat = torch.ones((1, 16))
        self.max_dev_node_feat = torch.ones((1, 16))
        
        super().__init__(self.folder, transform, pre_transform)
        data_list = []

        for i, name in enumerate(self.names):
            loaded_data, loaded_slices = torch.load(self.processed_paths[i], weights_only=False)

            self.data_offsets[name] = len(data_list)
            # 根据任务类型获取数据长度
            if self.task_level == 'node':
                self.data_lengths[name] = loaded_data.node_label.size(0)
            else:
                self.data_lengths[name] = loaded_data.edge_label.size(0)
            if loaded_slices is not None:
                data_list += collated_data_separate(loaded_data, loaded_slices)
            else:
                data_list.append(loaded_data)
            
            print(f"load processed case{name}, "+
                  f"len(data_list)={self.data_lengths[name]}, "+
                  f"data_offset={self.data_offsets[name]} ")
    
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

    def norm_nfeat(self, ntypes, num_classes=None):
        """归一化节点特征和标签
        
        Args:
            ntypes: 节点类型列表
            num_classes: 分类任务的类别数量 (如果为 None，使用 self._num_classes)
        
        标签格式 (在 process 阶段已处理好):
            - label[:, 0]: 归一化后的连续值 (用于回归)
            - label[:, 1]: 离散类别索引 (用于分类)
        """
        if num_classes is None:
            num_classes = self._num_classes
            
        data_store = self._get_data_store()
        self._data = data_store

        for ntype in ntypes:
            node_mask = data_store.node_type == ntype
            if node_mask.sum() == 0:
                continue
            max_node_feat, _ = data_store.node_attr[node_mask].max(dim=0, keepdim=True)
            max_node_feat[max_node_feat == 0.0] = 1.0

            print(f"normalizing node_attr type {ntype}: max={max_node_feat.max().item():.4f}")
            data_store.node_attr[node_mask] /= max_node_feat
        
        # 打印标签信息 (标签已在 process 阶段处理好)
        if self.task_level == 'edge':
            print(f"edge_label shape: {data_store.edge_label.shape}")
            print(f"  回归标签范围: [{self._data.edge_label[:, 0].min():.4f}, {self._data.edge_label[:, 0].max():.4f}]")
            print(f"  分类标签类别: {self._data.edge_label[:, 1].unique().long().tolist()}")
        elif self.task_level == 'node':
            print(f"node_label shape: {data_store.node_label.shape}")
            print(f"  回归标签范围: [{self._data.node_label[:, 0].min():.4f}, {self._data.node_label[:, 0].max():.4f}]")
            print(f"  分类标签类别: {self._data.node_label[:, 1].unique().long().tolist()}")
        
        self._data_list = None

    def set_cl_embeds(self, embeds):
        """设置对比学习嵌入"""
        self._data = self._get_data_store()

        self._data.x = embeds
        print("Setting CL embeddings to x...")
        print('self._data', self._data)


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
        hg = torch.load(raw_path, weights_only=False)
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
        
        logging.info(f"g._n2type {g._n2type}")
        logging.info(f"g._e2type {g._e2type}")
        
        # 构建节点特征 (padding 到统一维度 16)
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
            tar_edge_type_tensor = g.edge_type[edge_offset:edge_offset + tar_edge_count]
            
            # 过滤有效标签 (0 < y <= 700)
            valid_mask = (tar_edge_y > 0) & (tar_edge_y <= MAX_EDGE_LABEL)
            valid_count = valid_mask.sum().item()
            filtered_count = tar_edge_count - valid_count
            print(f"  有效边标签: {valid_count}, 过滤掉: {filtered_count}")
            
            g.tar_edge_y = tar_edge_y[valid_mask]
            g.tar_edge_index = tar_edge_index[:, valid_mask]
            g.tar_edge_type = tar_edge_type_tensor[valid_mask]
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
        logging.info(f"processing dataset case{self.names[idx]} "+ 
                     f"with sample_rate {self.sample_rates[idx]}...")
        
        graph = self.rc_graph_load(self.names[idx], self.raw_paths[idx])
        logging.info(f"loaded graph {graph}")
        
        n_classes = self._num_classes
        
        if self.task_level == 'node':
            # 节点任务: 提取 net 节点的标签
            net_mask = graph.node_type == NET
            net_indices = torch.where(net_mask)[0]
            
            # 获取有效的节点标签 (电容值 > 0)
            node_labels = graph.tar_node_y[net_indices].squeeze()
            if node_labels.dim() == 0:
                node_labels = node_labels.unsqueeze(0)
            valid_mask = (node_labels > 0) & (node_labels <= MAX_NODE_LABEL)
            valid_indices = net_indices[valid_mask]
            valid_labels = node_labels[valid_mask]
            
            # 采样
            if self.sample_rates[idx] < 1.0:
                num_samples = int(len(valid_indices) * self.sample_rates[idx])
                perm = torch.randperm(len(valid_indices))[:num_samples]
                valid_indices = valid_indices[perm]
                valid_labels = valid_labels[perm]
            
            print(f"  节点任务: 有效 net 节点数={len(valid_indices)}")
            
            # 归一化回归标签: 原始 paragraph-simple 逻辑
            reg_labels = torch.log10(valid_labels * 1e21)
            reg_labels /= 6
            reg_labels[reg_labels < 0] = 0.0
            reg_labels[reg_labels > 1] = 1.0
            
            # 分类标签: 对归一化后的值分桶
            boundaries = torch.linspace(0, 1, n_classes + 1)[1:-1]
            class_labels = torch.bucketize(reg_labels, boundaries).float()
            
            # 标签格式: [N, 2], 第0列回归，第1列分类
            graph.node_label = torch.stack([reg_labels, class_labels], dim=1)
            graph.node_label_index = valid_indices
            graph.n_classes = n_classes
            
            print(f"  节点标签: shape={graph.node_label.shape}, 回归范围=[{reg_labels.min():.4f}, {reg_labels.max():.4f}]")
            print(f"  分类分布 ({n_classes}类): {torch.bincount(class_labels.long(), minlength=n_classes).tolist()}")
            
            # 清理临时属性
            for attr in ['tar_node_y', 'tar_edge_index', 'tar_edge_type', 'tar_edge_y']:
                if hasattr(graph, attr):
                    delattr(graph, attr)
            
            # 设置空的边标签 (兼容性)
            graph.edge_label_index = torch.empty((2, 0), dtype=torch.long)
            graph.edge_label = torch.empty((0, 2))
            
            torch.save((graph, None), self.processed_paths[idx])
            return graph.node_label.size(0)
        
        else:
            # 边任务: 预测 pair_to 边的电阻值 (多分类或回归，不加负边)
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
            
            # 归一化回归标签: 原始 paragraph-simple 逻辑
            reg_labels = torch.log10(pos_edge_y * 1e21)
            reg_labels /= 6
            reg_labels[reg_labels < 0] = 0.0
            reg_labels[reg_labels > 1] = 1.0
            
            # 分类标签: 对归一化后的值分桶
            boundaries = torch.linspace(0, 1, n_classes + 1)[1:-1]
            class_labels = torch.bucketize(reg_labels, boundaries).float()
            
            # 标签格式: [N, 2], 第0列回归，第1列分类
            graph.edge_label = torch.stack([reg_labels, class_labels], dim=1)
            graph.edge_label_index = pos_edge_index
            graph.n_classes = n_classes
            
            print(f"  边标签: shape={graph.edge_label.shape}, 回归范围=[{reg_labels.min():.4f}, {reg_labels.max():.4f}]")
            print(f"  分类分布 ({n_classes}类): {torch.bincount(class_labels.long(), minlength=n_classes).tolist()}")
            
            # 清理临时属性
            for attr in ['tar_node_y', 'tar_edge_index', 'tar_edge_type', 'tar_edge_y', 'tar_edge_dist']:
                if hasattr(graph, attr):
                    delattr(graph, attr)
            
            # 设置空的节点标签 (兼容性)
            graph.node_label_index = torch.empty(0, dtype=torch.long)
            graph.node_label = torch.empty((0, 2))
            
            torch.save((graph, None), self.processed_paths[idx])
            return graph.edge_label.size(0)

    def process(self):
        """处理所有图"""
        for i, name in enumerate(self.names):
            if os.path.exists(self.processed_paths[i]):
                logging.info(f"Found process file of case{name} in {self.processed_paths[i]}, skipping process()")
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
        base_dir = os.path.join(self.folder, 'processed_cirgps')
        # 添加类别数到路径中
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


class LinkPredictionDataset(Dataset):
    """Link prediction dataset (保留兼容性)"""
    def __init__(self, node_embeddings, graph, test=False):
        assert node_embeddings.size(0) == graph.num_nodes
        self.len = graph.edge_label_index.size(1)
        self.x_data = node_embeddings
        self.y_data = graph.edge_label.long()
        self.links = graph.edge_label_index
        if test:
            self.test_idx = np.arange(self.len)
        else:
            self.train_idx, self.val_idx = self.get_idx_split()

    def __getitem__(self, index):
        src = self.links[0, index]
        dst = self.links[1, index]
        y = self.y_data[index]
        return self.x_data[src], self.x_data[dst], y

    def __len__(self):
        return self.len
    
    def get_idx_split(self):
        return train_test_split(
            np.arange(self.len), 
            test_size=0.2, 
            random_state=123, 
            shuffle=True,
        )
    
def collate_fn(dataList):
    """Collate function for LinkPredictionDataset"""
    num_items = len(dataList)
    dim = dataList[0][0].size(-1)
    src_data = torch.zeros((num_items, dim), dtype=torch.float32)
    dst_data = torch.zeros((num_items, dim), dtype=torch.float32)
    y = torch.zeros((num_items), dtype=torch.long)
    for i in range(num_items):
        src_data[i], dst_data[i], y[i] = dataList[i]
    batch = Data()
    batch.x = torch.stack([src_data, dst_data], dim=0)
    batch.y = y
    return batch

def adaption_for_sgrl(dataset):
    """对比学习适配函数"""
    data_list = []

    for i, name in enumerate(dataset.names):
        single_graph = Data(
            x=dataset[i].node_type, edge_index=dataset[i].edge_index, 
            edge_attr=dataset[i].edge_type
        )
        data_list.append(single_graph)

    batch = Batch.from_data_list(data_list)
    batch.x = batch.x.view(-1, 1)
    batch.edge_type = batch.edge_attr
    del batch.edge_attr

    print("attributes in big batch", batch)
    print("batch.ptr", batch.ptr)

    return batch

def performat_SramDataset(dataset_dir, name, 
                          add_target_edges=False,
                          neg_edge_ratio=1.0, 
                          to_undirected=True, 
                          sample_rates=1.0, 
                          task_type='regression',
                          num_classes=5):
    """创建数据集的便捷函数"""
    start = time.perf_counter()
    num_datasets = len(name.split('+'))

    dataset = SealSramDataset(
        name=name, 
        root=dataset_dir,
        add_target_edges=add_target_edges,
        neg_edge_ratio=neg_edge_ratio,
        to_undirected=to_undirected,
        sample_rates=[sample_rates] * num_datasets,
        task_type=task_type,
        num_classes=num_classes,
    )

    elapsed = time.perf_counter() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) + f'{elapsed:.2f}'[-3:]
    print(f"PID = {os.getpid()}")
    print(f"Building dataset case{name} took {timestr}")
    print('Dataloader: Loading success.')

    return dataset
