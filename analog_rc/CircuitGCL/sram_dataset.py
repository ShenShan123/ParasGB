"""
RC Circuit Dataset - 适配 data/ 目录下的 RC 电路数据
支持 CircuitGCL 的对比学习预训练和下游任务

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
from utils import (
    get_pos_neg_edges, get_balanced_edges, 
    collated_data_separate)
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph

# 节点类型顺序 (与 newgraph.py 一致)
NODE_TYPE_ORDER = ['dev', 'pin', 'net']
NODE_TYPE_TO_ID = {ntype: i for i, ntype in enumerate(NODE_TYPE_ORDER)}

# 边类型顺序
EDGE_TYPE_ORDER = [
    ('dev', 'connects_to', 'pin'),
    ('dev', 'connects_to', 'net'),
    ('pin', 'belongs_to', 'net'),
    ('pin', 'pair_to', 'pin'),  # 目标边
]

# 标签范围
MAX_EDGE_LABEL = 700.0  # 电阻最大值 (Ω)
MAX_NODE_LABEL = 8e-13  # 电容最大值 (F)

# 节点类型常量
DEV = 0
PIN = 1
NET = 2


class SealSramDataset(InMemoryDataset):
    def __init__(
        self,
        name,
        root,
        neg_edge_ratio=1.0,
        to_undirected=True,
        sample_rates=[1.0], 
        task_level='edge',
        net_only=True,
        transform=None, 
        pre_transform=None,
        class_boundaries=[0.2, 0.4, 0.6, 0.8],
        train_names=None,
        test_names=None,
    ) -> None:
        """RC Circuit Dataset for CircuitGCL
        
        Args:
            name (str): Case ID，多个用 '+' 连接，如 "1+5+7"
            root (str): 数据根目录
            neg_edge_ratio (float): 负边比例
            to_undirected (bool): 是否转为无向图
            sample_rates (list): 采样率
            task_level (str): 任务级别 ('edge' 或 'node')
            net_only (bool): 节点任务是否只预测 net 节点
            class_boundaries (list): 分类边界
            train_names (list): 训练集名称列表
            test_names (list): 测试集名称列表
        """
        self.name = 'rc_circuit'
        self.class_boundaries = torch.tensor(class_boundaries)
        print("self.class_boundaries", self.class_boundaries)
        
        # 解析多个数据集名称
        if '+' in name:
            self.names = name.split('+')
        else:
            self.names = [name]
        
        # 保存训练集和测试集名称
        self.train_names = train_names if train_names else self.names[:1]
        self.test_names = test_names if test_names else self.names[1:]
            
        print(f"RCDataset includes cases: {self.names}")
        print(f"  Train cases: {self.train_names}")
        print(f"  Test cases: {self.test_names}")

        self.sample_rates = sample_rates
        assert len(self.names) == len(self.sample_rates), \
            f"len of dataset:{len(self.names)}, len of sample_rate: {len(self.sample_rates)}"
        
        self.folder = root  # 直接使用 root 作为数据目录
        self.neg_edge_ratio = neg_edge_ratio
        self.to_undirected_flag = to_undirected
        self.data_lengths = {}
        self.data_offsets = {}

        self.task_level = task_level
        self.net_only = net_only
    
        self.max_net_node_feat = torch.ones((1, 16))
        self.max_dev_node_feat = torch.ones((1, 16))

        super().__init__(self.folder, transform, pre_transform)
        data_list = []

        for i, name in enumerate(self.names):
            loaded_data, loaded_slices = torch.load(self.processed_paths[i], weights_only=False)

            self.data_offsets[name] = len(data_list)
            if self.task_level == 'node':
                self.data_lengths[name] = loaded_data.y.size(0)
            elif self.task_level == 'edge':
                self.data_lengths[name] = loaded_data.edge_label.size(0)
            else:
                raise ValueError(f"Invalid task level: {self.task_level}")

            if loaded_slices is not None:
                data_list += collated_data_separate(loaded_data, loaded_slices)
            else:
                data_list.append(loaded_data)
            
            print(f"load processed case{name}, "+
                  f"len(data_list)={self.data_lengths[name]}, "+
                  f"data_offset={self.data_offsets[name]} ")
    
        self.data, self.slices = self.collate(data_list)

    def norm_nfeat(self, ntypes):
        """归一化节点特征和标签
        
        Args:
            ntypes (list): 要归一化的节点类型 {DEV=0, NET=2}
        """
        if self._data is None or self.slices is None:
            self.data, self.slices = self.collate(self._data_list)
            self._data_list = None

        # 归一化节点特征
        for ntype in ntypes:
            node_mask = self._data.node_type == ntype
            if node_mask.sum() == 0:
                continue
            max_node_feat, _ = self._data.node_attr[node_mask].max(dim=0, keepdim=True)
            max_node_feat[max_node_feat == 0.0] = 1.0

            print(f"normalizing node_attr type {ntype}: max={max_node_feat.max().item():.4f}")
            self._data.node_attr[node_mask] /= max_node_feat

        if self.task_level == 'edge':
            # 归一化边标签: 原始 paragraph-simple 逻辑
            self._data.edge_label = torch.log10(self._data.edge_label * 1e21)
            self._data.edge_label /= 6
            self._data.edge_label[self._data.edge_label < 0] = 0.0
            self._data.edge_label[self._data.edge_label > 1] = 1.0
            
            # 分类标签
            edge_label_c = torch.bucketize(self._data.edge_label, self.class_boundaries)
            self._data.edge_label = torch.stack(
                [self._data.edge_label, edge_label_c.float()], dim=1
            )
            print(f"edge_label normalized: shape={self._data.edge_label.shape}")
            self._data_list = None

        elif self.task_level == 'node':
            # 归一化节点标签: 原始 paragraph-simple 逻辑
            self._data.y = torch.log10(self._data.y * 1e21)
            self._data.y /= 6
            self._data.y[self._data.y < 0] = 0.0
            self._data.y[self._data.y > 1] = 1.0
            
            # 分类标签
            node_label_c = torch.bucketize(self._data.y, self.class_boundaries)
            self._data.y = torch.stack(
                [self._data.y, node_label_c.float()], dim=1
            )
            print(f"node_label (y) normalized: shape={self._data.y.shape}")
            self._data_list = None

       
           

    def set_cl_embeds(self, embeds):
        """设置 SGRL 对比学习的节点嵌入
        
        Args:
            embeds (torch.Tensor): SGRL 学习的节点嵌入 [N, cl_hid_dim]
        """
        if self._data is None or self.slices is None:
            self.data, self.slices = self.collate(self._data_list)
            self._data_list = None

        self._data.x = embeds
        self._data_list = None
        print("Setting CL embeddings to x...")
        print('self._data', self._data)

    def rc_graph_load(self, name, raw_path):
        """加载 RC 电路图数据
        
        数据格式 (来自 newgraph.py):
        - 节点: dev (16维), pin (6维), net (10维)
        - 边: connects_to, belongs_to, pair_to
        - 节点标签: net.y (电容, ~1e-13 F)
        - 边标签: pair_to.y (电阻, 0~700 Ω)
        
        Args:
            name (str): 数据集名称 (case ID)
            raw_path (str): 原始数据文件路径
            
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
        
        if self.task_level == 'node':
            if self.net_only:
                net_mask = g.node_type == NET
                g.tar_node_y = torch.zeros((g.num_nodes, 1))
                net_nodes = torch.where(net_mask)[0]
                if 'net' in hg.node_types and hasattr(hg['net'], 'y'):
                    g.tar_node_y[net_nodes] = hg['net'].y
            else:
                g.tar_node_y = torch.cat(tar_node_y_list, dim=0) if tar_node_y_list else torch.zeros((g.num_nodes, 1))
            g.y = g.tar_node_y
        else:
            g.tar_node_y = torch.cat(tar_node_y_list, dim=0) if tar_node_y_list else torch.zeros((g.num_nodes, 1))
        
        # 处理边标签
        if tar_edge_y is not None and tar_edge_count > 0:
            tar_edge_index = g.edge_index[:, edge_offset:edge_offset + tar_edge_count]
            tar_edge_type = g.edge_type[edge_offset:edge_offset + tar_edge_count]
            
            # 过滤有效标签 (0 < y <= 700)
            valid_mask = (tar_edge_y > 0) & (tar_edge_y <= MAX_EDGE_LABEL)
            valid_count = valid_mask.sum().item()
            filtered_count = tar_edge_count - valid_count
            print(f"  有效边标签: {valid_count}, 过滤掉: {filtered_count}")
            
            g.tar_edge_y = tar_edge_y[valid_mask]
            g.tar_edge_index = tar_edge_index[:, valid_mask]
            g.tar_edge_type = tar_edge_type[valid_mask]
            
            _, g.tar_edge_dist = g.tar_edge_type.unique(return_counts=True)
            
            # 移除目标边，只保留结构边
            g.edge_index = g.edge_index[:, :edge_offset]
            g.edge_type = g.edge_type[:edge_offset]
        else:
            g.tar_edge_y = torch.empty(0)
            g.tar_edge_index = torch.empty((2, 0), dtype=torch.long)
            g.tar_edge_type = torch.empty(0, dtype=torch.long)
            g.tar_edge_dist = torch.tensor([0])
        
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
        print(f"processing case {self.names[idx]} with sample_rate {self.sample_rates[idx]}...")
        
        graph = self.rc_graph_load(self.names[idx], self.raw_paths[idx])
        print(f"loaded graph {graph}")
        
        # 生成负边
        neg_edge_index, neg_edge_type = get_pos_neg_edges(
            graph, neg_ratio=self.neg_edge_ratio)
        
        # 采样正负边
        (
            pos_edge_index, pos_edge_type, pos_edge_y,
            neg_edge_index, neg_edge_type
        ) = get_balanced_edges(
            graph, neg_edge_index, neg_edge_type, 
            self.neg_edge_ratio, self.sample_rates[idx]
        )

        if self.task_level == 'edge':
            # 边任务: 只使用正边
            links = pos_edge_index
            labels = pos_edge_y
        elif self.task_level == 'node':
            # 节点任务
            graph.y = graph.tar_node_y.squeeze()
        else:
            raise ValueError(f"No definition of task {self.task_level}!")
        
        # 清理临时属性
        if hasattr(graph, 'tar_node_y'):
            del graph.tar_node_y
        if hasattr(graph, 'tar_edge_index'):
            del graph.tar_edge_index
        if hasattr(graph, 'tar_edge_type'):
            del graph.tar_edge_type
        if hasattr(graph, 'tar_edge_y'):
            del graph.tar_edge_y

        if self.task_level == 'node':
            torch.save((graph, None), self.processed_paths[idx])
            return graph.y.size(0)
        elif self.task_level == 'edge':
            graph.edge_label_index = links
            graph.edge_label = labels
            torch.save((graph, None), self.processed_paths[idx])
            return graph.edge_label.size(0)
        else:
            raise ValueError(f"No definition of task {self.task_level}!")

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
        base_dir = os.path.join(self.folder, 'processed_circuitgcl')
        if self.task_level == 'edge':
            return os.path.join(base_dir, 'edge')
        elif self.task_level == 'node':
            return os.path.join(base_dir, 'node')
        else:
            return base_dir

    @property
    def processed_file_names(self):
        processed_names = []
        for i, name in enumerate(self.names):
            fname = f"case{name}"
            if self.sample_rates[i] < 1.0:
                fname += f"_s{self.sample_rates[i]}"
            if self.neg_edge_ratio < 1.0:
                fname += f"_nr{self.neg_edge_ratio:.1f}"
            processed_names.append(fname + "_processed.pt")
        return processed_names

def adaption_for_sgrl(dataset):
    """为 SGRL 对比学习准备数据
    
    将数据集中的图合并为一个大图，用于对比学习预训练
    """
    data_list = []

    for i, name in enumerate(dataset.names):
        single_graph = Data(
            x=dataset[i].node_type, 
            edge_index=dataset[i].edge_index, 
            edge_attr=dataset[i].edge_type
        )
        single_graph.node_attr = dataset[i].node_attr
        data_list.append(single_graph)

    # 合并所有图为一个大图
    batch = Batch.from_data_list(data_list)

    batch.x = batch.x.view(-1, 1)
    batch.edge_type = batch.edge_attr

    del batch.edge_attr

    print("attributes in big batch", batch)
    print("batch.ptr", batch.ptr)

    return batch


def performat_SramDataset(dataset_dir, name, 
                          neg_edge_ratio, to_undirected, 
                          sample_rate,
                          task_level,
                          net_only,
                          class_boundaries,
                          train_names=None,
                          test_names=None,
                          ):
    """创建数据集的便捷函数
    
    Args:
        dataset_dir: 数据目录
        name: 所有数据集名称，用 '+' 连接
        neg_edge_ratio: 负边比例
        to_undirected: 是否转为无向图
        sample_rate: 采样率
        task_level: 任务级别 ('edge' 或 'node')
        net_only: 节点任务是否只预测 net 节点
        class_boundaries: 分类边界
        train_names: 训练集名称列表
        test_names: 测试集名称列表
    """
    start = time.perf_counter()
    names = name.split('+')
    
    # 所有数据集使用相同的采样率
    sr_list = [sample_rate] * len(names)

    dataset = SealSramDataset(
            name=name, root=dataset_dir,
            neg_edge_ratio=neg_edge_ratio,
            to_undirected=to_undirected,
            sample_rates=sr_list,
            task_level=task_level,              
            net_only=net_only,
            class_boundaries=class_boundaries,
            train_names=train_names,
            test_names=test_names,
        )

    elapsed = time.perf_counter() - start
    timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
            + f'{elapsed:.2f}'[-3:]
    print(f"PID = {os.getpid()}")
    print(f"Building dataset {name} took {timestr}")
    print('Dataloader: Loading success.')

    return dataset
