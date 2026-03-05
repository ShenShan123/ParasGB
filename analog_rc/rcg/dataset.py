"""
Dataset loading and preprocessing for RC graphs.
硬编码节点和边的位置索引，支持节点任务(net节点)和边任务(pin-pair_to-pin边)
支持数据缓存，处理后的数据会保存到processed目录，下次加载时直接读取
"""
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import os
import matplotlib.pyplot as plt
import numpy as np


class RCDataset:
    """Utility class for loading and processing RC graph data."""
    
    # 硬编码节点类型顺序 (与 newgraph.py 一致: dev, pin, net)
    NODE_TYPE_ORDER = ['dev', 'pin', 'net']
    NODE_TYPE_TO_ID = {ntype: i for i, ntype in enumerate(NODE_TYPE_ORDER)}
    
    # 硬编码边类型顺序 (与 newgraph.py 一致)
    EDGE_TYPE_ORDER = [
        ('pin', 'belongs_to', 'net'),
        ('dev', 'connects_to', 'pin'),
        ('dev', 'connects_to', 'net'),
        ('pin', 'pair_to', 'pin'),
    ]
    EDGE_TYPE_TO_ID = {etype: i for i, etype in enumerate(EDGE_TYPE_ORDER)}
    
    # 分类任务的默认分桶边界 (基于归一化后的标签 0-1)
    DEFAULT_CLASS_BOUNDARIES = [0.2, 0.4, 0.6, 0.8]
    # 归一化方式: 'minmax' 或 'log'
    NORMALIZE_METHOD = 'log'
    # 边标签最大值 (电阻，单位 Ω)
    MAX_EDGE_LABEL = 700.0
    # 节点标签最大值 (电容，单位 F，约 1e-12 量级)
    MAX_NODE_LABEL = 8 * 1e-13  
    
    @staticmethod
    def normalize_edge(y):
        """归一化边标签 (电阻): minmax (y/700) 或 log (log(1+y)/log(701))"""
        if RCDataset.NORMALIZE_METHOD == 'log':
            return (torch.log1p(y) / np.log1p(RCDataset.MAX_EDGE_LABEL)).clamp(0, 1)
        return (y / RCDataset.MAX_EDGE_LABEL).clamp(0, 1)
    
    @staticmethod
    def normalize_node(y):
        """归一化节点标签 (电容): 使用 log 归一化处理极小值"""
        if RCDataset.NORMALIZE_METHOD == 'log':
            return (torch.log1p(y * 1e15) / np.log1p(RCDataset.MAX_NODE_LABEL * 1e15)).clamp(0, 1)
        return (y / RCDataset.MAX_NODE_LABEL).clamp(0, 1)
    
    @staticmethod
    def label_to_class(labels: torch.Tensor, boundaries: list = None) -> torch.Tensor:
        """
        将归一化后的连续标签转换为类别标签（分桶）。
        使用 torch.bucketize 实现。
        
        Args:
            labels: 归一化后的连续标签 (范围 0-1)
            boundaries: 分桶边界列表，如 [0.2, 0.4, 0.6, 0.8] 表示5个类别
        
        Returns:
            类别标签 tensor (long类型，值为 0 到 num_classes-1)
        """
        if boundaries is None:
            boundaries = RCDataset.DEFAULT_CLASS_BOUNDARIES
        
        boundaries_tensor = torch.tensor(boundaries, dtype=labels.dtype, device=labels.device)
        classes = torch.bucketize(labels.squeeze(), boundaries_tensor)
        
        return classes.float()  # 转为float以便与回归标签拼接
    
    @staticmethod
    def load_and_process(filepath: str, task_level: str = 'node', use_cache: bool = True) -> Data:
        """
        Load a single .pt file and convert to homogeneous graph.
        支持缓存：处理后的数据会保存，下次直接加载。
        
        Args:
            filepath: Path to the .pt file
            task_level: 'node' (预测net节点) or 'edge' (预测pin-pair_to-pin边)
            use_cache: 是否使用缓存（默认True）
            
        Returns:
            Processed PyG Data object with position indices
        """
        # 构建缓存路径
        data_dir = os.path.dirname(filepath)
        name = os.path.basename(filepath).replace('.pt', '')
        cache_dir = os.path.join(data_dir, f'processed_for_{task_level}')
        cache_path = os.path.join(cache_dir, f'{name}_processed.pt')
        
        # 尝试从缓存加载
        if use_cache and os.path.exists(cache_path):
            print(f"  [缓存] 从 {cache_path} 加载已处理的数据")
            g = torch.load(cache_path)
            return g
        
        # 没有缓存，进行完整处理
        print(f"  [处理] 从原始文件处理数据...")
        hg = torch.load(filepath)
        if isinstance(hg, list):
            hg = hg[0]
        print("==================== 原始异构图 ==================")
        print(hg)        
        
        # 移除电源网络 (根据特征判断: net.x[:, 0]=VDD, net.x[:, 1]=VSS/GND)
        subset_dict = {}
        for ntype in hg.node_types:
            subset_dict[ntype] = torch.ones(hg[ntype].num_nodes, dtype=torch.bool)
            if ntype == 'net' and hasattr(hg['net'], 'x'):
                net_x = hg['net'].x
                # 特征维度0=is_power(VDD), 维度1=is_ground(VSS/GND)
                is_power_net = (net_x[:, 0] == 1.0) | (net_x[:, 1] == 1.0)
                power_count = is_power_net.sum().item()
                print(f"  [过滤] 移除 {power_count} 个电源网络 (VDD/VSS/GND)")
                subset_dict[ntype] = ~is_power_net
        
        hg = hg.subgraph(subset_dict)
        
        # 只保留需要的边类型 (与 newgraph.py 一致)
        edge_types_to_keep = [
            ('dev', 'connects_to', 'pin'),
            ('dev', 'connects_to', 'net'),
            ('pin', 'belongs_to', 'net'),
            ('pin', 'pair_to', 'pin'),
        ]
        existing_edge_types = [et for et in edge_types_to_keep if et in hg.edge_types]
        if existing_edge_types:
            hg = hg.edge_type_subgraph(existing_edge_types)
        
        # 计算各节点类型的数量和偏移
        node_counts = {}
        node_offsets = {}
        offset = 0
        for ntype in RCDataset.NODE_TYPE_ORDER:
            if ntype in hg.node_types:
                node_counts[ntype] = hg[ntype].num_nodes
                node_offsets[ntype] = offset
                offset += node_counts[ntype]
            else:
                node_counts[ntype] = 0
                node_offsets[ntype] = offset
        
        # 计算各边类型的数量和偏移
        edge_counts = {}
        edge_offsets = {}
        edge_offset = 0
        for etype in RCDataset.EDGE_TYPE_ORDER:
            if etype in hg.edge_types:
                edge_counts[etype] = hg[etype].edge_index.shape[1]
                edge_offsets[etype] = edge_offset
                edge_offset += edge_counts[etype]
            else:
                edge_counts[etype] = 0
                edge_offsets[etype] = edge_offset
        
        # 转换为同构图
        g = hg.to_homogeneous()
        print("==================== 转换为同构图 ==================")
        print(g)
        
        g.name = name
        
        # 存储位置信息
        g._node_counts = node_counts
        g._node_offsets = node_offsets
        g._edge_counts = edge_counts
        g._edge_offsets = edge_offsets
        g._n2type = RCDataset.NODE_TYPE_TO_ID
        g._e2type = RCDataset.EDGE_TYPE_TO_ID
        
        # 构建节点特征
        max_feat_dim = 16
        node_feat_list = []
        for ntype in RCDataset.NODE_TYPE_ORDER:
            if ntype in hg.node_types and hasattr(hg[ntype], 'x'):
                feat = hg[ntype].x
                feat = torch.nn.functional.pad(feat, (0, max_feat_dim - feat.size(1)))
                node_feat_list.append(feat)
        
        if node_feat_list:
            g.node_attr = torch.cat(node_feat_list, dim=0)
        
        # 处理节点任务标签 (net节点上的y)
        if task_level == 'node':
            g = RCDataset._process_node_labels(g, hg, node_counts, node_offsets, name=name)
        # 处理边任务标签 (pin-pair_to-pin边上的y)
        elif task_level == 'edge':
            g = RCDataset._process_edge_labels(g, hg, edge_counts, edge_offsets, name=name)
        
        # 转为无向图 (只对结构边)
        g.edge_index, g.edge_type = to_undirected(
            g.edge_index, g.edge_type, g.num_nodes, reduce='mean'
        )
        # 确保edge_type是long类型 (to_undirected可能会转成float)
        g.edge_type = g.edge_type.long()
        
        # 保存到缓存
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save(g, cache_path)
            print(f"  [缓存] 已保存处理后的数据到 {cache_path}")
        
        return g

    @staticmethod
    def _plot_label_distribution(labels, name="unknown", title="Label Distribution", task_level="node", normalized_labels=None):
        """
        画出标签的分布图（直方图）。原始和归一化后分别保存。
        样式：灰色背景，橙黄色柱状图，y轴为density
        """
        if task_level == 'node':
            save_dir = 'imgs/node'
        elif task_level == 'edge':
            save_dir = 'imgs/edge'
        else:
            save_dir = 'imgs'
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"  [{name}] Label min~max: {labels.min().item():.6e} ~ {labels.max().item():.6e}")
        
        # 样式设置：灰色背景，橙黄色柱状图
        bar_color = '#E8A838'
        labels_np = labels.cpu().numpy().flatten()
        
        # 画原始标签分布
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        ax.set_facecolor('lightgray')
        ax.hist(labels_np, bins=50, density=True, color=bar_color, edgecolor=bar_color, alpha=0.9)
        ax.set_xlabel('label', fontsize=25)
        ax.set_ylabel('density', fontsize=25)
        ax.grid(True, alpha=0.3, color='white')
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        plt.tight_layout()
        save_path = f'{save_dir}/{name}_original.png'
        plt.savefig(save_path, dpi=300, facecolor='white')
        plt.close()
        print(f"  原始标签分布图已保存到 {save_path}")
        
        # 画归一化后标签分布
        if normalized_labels is not None:
            norm_np = normalized_labels.cpu().numpy().flatten()
            fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
            ax.set_facecolor('lightgray')
            ax.hist(norm_np, bins=50, density=True, color=bar_color, edgecolor=bar_color, alpha=0.9)
            ax.set_xlabel('normalized label', fontsize=25)
            ax.set_ylabel('density', fontsize=25)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3, color='white')
            plt.tight_layout()
            save_path = f'{save_dir}/{name}_normalized.png'
            plt.savefig(save_path, dpi=300, facecolor='white')
            plt.close()
            print(f"  归一化标签分布图已保存到 {save_path}")
        
    @staticmethod
    def _process_node_labels(g, hg, node_counts, node_offsets, name="unknown"):
        """
        处理节点级任务的标签。
        只预测net节点的标签，使用硬编码的节点类型ID来确定目标节点。
        
        标签格式: g.y [N, 2]
        - 第0列: 归一化后的回归标签 (0-1)
        - 第1列: 分桶后的分类标签 (0 到 num_classes-1)
        
        最终保留的属性:
        - g.y: 标签 [N, 2]
        - g.train_node_mask: 有效训练节点的mask
        """
        total_nodes = g.num_nodes
        
        # 获取net节点的位置范围 (硬编码: net的type_id=2)
        net_offset = node_offsets.get('net', 0)
        net_count = node_counts.get('net', 0)
        
        # 初始化标签 [N, 2]: 第0列回归，第1列分类
        g.y = torch.zeros((total_nodes, 2))
        # 使用局部变量存储有效标签mask (不再作为图属性)
        valid_label_mask = torch.zeros(total_nodes, dtype=torch.bool)
        
        if net_count > 0 and 'net' in hg.node_types and hasattr(hg['net'], 'y'):
            net_y = hg['net'].y
            
            # 过滤有效标签: 电容值 > 0 (排除无效的零值或负值)
            valid_mask = (net_y > 0) & (net_y <= RCDataset.MAX_NODE_LABEL)
            valid_mask = valid_mask.squeeze()
            
            valid_count = valid_mask.sum().item()
            filtered_count = net_count - valid_count
            print(f"  [{name}] 有效标签 (电容 > 0): {valid_count}, 过滤掉: {filtered_count}")
            
            # 设置有效标签mask (局部变量)
            valid_label_mask[net_offset:net_offset + net_count] = valid_mask
            
            # 归一化处理 (使用节点专用的归一化方法)
            normalized_y = RCDataset.normalize_node(net_y)
            
            # 画出原始和归一化后的标签分布图
            print(f"==================== 绘制标签分布图====================")
            RCDataset._plot_label_distribution(net_y, name=name, title=f"{name} Net Node Label", 
                                               task_level="node", normalized_labels=normalized_y)
            
            # 第0列: 回归标签，第1列: 分类标签
            g.y[net_offset:net_offset + net_count, 0:1] = normalized_y
            g.y[net_offset:net_offset + net_count, 1] = RCDataset.label_to_class(normalized_y)
        
        # 使用node_type来创建目标节点mask (局部变量，node_type==2 表示net节点)
        target_node_type_id = RCDataset.NODE_TYPE_TO_ID['net']
        target_node_mask = (g.node_type == target_node_type_id)
        
        # 最终的训练mask: 既是net节点，又有有效标签 (这个保留为图属性)
        g.train_node_mask = target_node_mask & valid_label_mask
        
        return g
    
    @staticmethod
    def _process_edge_labels(g, hg, edge_counts, edge_offsets, name="unknown"):
        """
        处理边级任务的标签。
        标签在(pin, pair_to, pin)边上。
        
        标签格式: edge_label_y [E, 2]
        - 第0列: 归一化后的回归标签 (0-1)
        - 第1列: 分桶后的分类标签 (0 到 num_classes-1)
        """
        pair_to_etype = ('pin', 'pair_to', 'pin')
        total_edges = g.edge_index.shape[1]
        
        pair_to_offset = edge_offsets.get(pair_to_etype, 0)
        pair_to_count = edge_counts.get(pair_to_etype, 0)
        
        print(f"  [{name}] pair_to边: offset={pair_to_offset}, count={pair_to_count}")
        
        # 初始化边标签 [E, 2]
        g.edge_label_full = torch.zeros((total_edges, 2))
        g.valid_edge_mask = torch.zeros(total_edges, dtype=torch.bool)
        g.target_edge_mask = torch.zeros(total_edges, dtype=torch.bool)
        
        if pair_to_count > 0 and pair_to_etype in hg.edge_types:
            store = hg[pair_to_etype]
            
            g.target_edge_mask[pair_to_offset:pair_to_offset + pair_to_count] = True
            
            if hasattr(store, 'y'):
                edge_y = store.y
                
                valid_mask = (edge_y >= 0) & (edge_y <= RCDataset.MAX_EDGE_LABEL)
                valid_mask = valid_mask.squeeze()
                
                valid_count = valid_mask.sum().item()
                filtered_count = pair_to_count - valid_count
                print(f"  [{name}] 有效边标签 (0-{RCDataset.MAX_EDGE_LABEL}): {valid_count}, 过滤掉: {filtered_count}")
                
                g.valid_edge_mask[pair_to_offset:pair_to_offset + pair_to_count] = valid_mask
                
                # 归一化 (使用边专用的归一化方法)
                normalized_edge_y = RCDataset.normalize_edge(edge_y)
                
                # 画出原始和归一化后的标签分布图
                RCDataset._plot_label_distribution(edge_y, name=f"{name}_edge", title=f"{name} Edge Label", 
                                                   task_level="edge", normalized_labels=normalized_edge_y)
                
                # 第0列: 回归标签，第1列: 分类标签
                g.edge_label_full[pair_to_offset:pair_to_offset + pair_to_count, 0:1] = normalized_edge_y
                g.edge_label_full[pair_to_offset:pair_to_offset + pair_to_count, 1] = RCDataset.label_to_class(normalized_edge_y)
        
        # 计算最终的训练mask
        train_edge_mask = g.target_edge_mask & g.valid_edge_mask
        
        # 提取有效边的索引和标签
        valid_edge_indices = torch.where(train_edge_mask)[0]
        g.edge_label_index = g.edge_index[:, valid_edge_indices]
        g.edge_label_y = g.edge_label_full[valid_edge_indices]  # [E_valid, 2]
        
        print(f"  [{name}] 有效目标边数量: {g.edge_label_index.shape[1]}")
        
        # 删除临时属性，只保留 edge_label_index 和 edge_label_y
        del g.edge_label_full
        del g.valid_edge_mask
        del g.target_edge_mask
        
        return g


def get_node_type_indices(g, node_type: str):
    offset = g._node_offsets.get(node_type, 0)
    count = g._node_counts.get(node_type, 0)
    return offset, offset + count


def get_edge_type_indices(g, edge_type: tuple):
    offset = g._edge_offsets.get(edge_type, 0)
    count = g._edge_counts.get(edge_type, 0)
    return offset, offset + count
