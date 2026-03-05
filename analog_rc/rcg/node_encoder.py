import torch
import torch.nn as nn
from typing import Dict, List, Union, Tuple
"""
节点特征编码器模块。
对不同类型节点的特征分别处理：连续值用Linear，离散值用Embedding。

配置格式说明：
每个节点类型包含 'discrete' 和 'continuous' 两个键：
- discrete: 离散特征索引，支持两种格式：
    - [idx1, idx2, ...] 索引列表，vocab_size使用默认值
    - [(idx, vocab_size), ...] 带vocab_size的元组列表
    - (start, end) 范围元组
- continuous: 连续特征索引，支持两种格式：
    - [idx1, idx2, ...] 索引列表
    - (start, end) 范围元组

示例配置：
{
    'device': {
        'discrete': [],
        'continuous': (0, 17),          # 范围：索引0-16
    },
    'pin': {
        'discrete': [0],                # 第0维离散
        'continuous': [1, 2, 3],        # 第1,2,3维连续
    },
    'net': {
        'discrete': [1, 4, 6, 8, 9],    # 第1,4,6,8,9维离散
        'continuous': [0, 2, 3, 5, 7],  # 第0,2,3,5,7维连续
    },
}
"""
class NodeFeatureEncoder(nn.Module):
    """
    节点特征编码器，对不同类型节点的特征分别编码。
    """
    
    # 默认配置 (与 newgraph.py 一致: dev, pin, net)
    DEFAULT_CONFIG = {
        'dev': {
            'discrete': [(0, 2), (1, 2), (2, 2), (5, 2), (6, 2), (15, 2)],
            'continuous': [3, 4, 7, 8, 9, 10, 11, 12, 13, 14],      
        },
        'pin': {
            'discrete': [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)],            
            'continuous': [],   
        },
        'net': {
            'discrete': [(0, 2), (1, 2), (2, 2), (3, 2), (4, 550), (5, 313), (6, 385), (7, 547)],
            'continuous': [8, 9],      
        },
    }
    
    # 节点类型ID映射 (与 newgraph.py 一致: dev, pin, net)
    NODE_TYPE_TO_ID = {'dev': 0, 'pin': 1, 'net': 2}
    
    # 离散特征默认vocab_size
    DEFAULT_VOCAB_SIZE = 100
    
    def __init__(self, out_dim: int, feat_config: Dict = None, default_vocab_size: int = 100):
        """
        Args:
            out_dim: 输出嵌入维度
            feat_config: 特征配置字典
            default_vocab_size: 离散特征默认词表大小
        """
        super().__init__()
        
        self.out_dim = out_dim
        self.feat_config = feat_config or self.DEFAULT_CONFIG
        self.default_vocab_size = default_vocab_size
        
        # 为每种节点类型创建编码器
        self.encoders = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        
        for node_type, type_config in self.feat_config.items():
            # 解析配置
            discrete_indices, discrete_vocab_sizes = self._parse_discrete_config(type_config.get('discrete', []))
            continuous_indices = self._parse_indices(type_config.get('continuous', []))
            
            # 存储解析后的索引
            if not hasattr(self, '_parsed_configs'):
                self._parsed_configs = {}
            self._parsed_configs[node_type] = {
                'discrete_indices': discrete_indices,
                'discrete_vocab_sizes': discrete_vocab_sizes,
                'continuous_indices': continuous_indices,
            }
            
            # 创建编码器
            type_encoders, encoder_out_dim = self._create_type_encoder(
                discrete_indices, discrete_vocab_sizes, continuous_indices
            )
            self.encoders[node_type] = type_encoders
            
            # 投影层
            if encoder_out_dim > 0:
                self.projections[node_type] = nn.Linear(encoder_out_dim, out_dim)
    
    def _parse_indices(self, spec) -> List[int]:
        """
        解析索引配置。
        
        Args:
            spec: (start, end) 范围元组 或 [idx1, idx2, ...] 索引列表
            
        Returns:
            索引列表
        """
        if isinstance(spec, tuple) and len(spec) == 2:
            # (start, end) 范围
            start, end = spec
            return list(range(start, end))
        elif isinstance(spec, list):
            return spec
        else:
            return []
    
    def _parse_discrete_config(self, spec) -> Tuple[List[int], List[int]]:
        """
        解析离散特征配置。
        
        Args:
            spec: 支持多种格式：
                - [idx1, idx2, ...] 索引列表
                - [(idx, vocab_size), ...] 带vocab_size的元组列表
                - (start, end) 范围元组
            
        Returns:
            (indices, vocab_sizes) 两个列表
        """
        indices = []
        vocab_sizes = []
        
        if isinstance(spec, tuple) and len(spec) == 2:
            # (start, end) 范围
            start, end = spec
            indices = list(range(start, end))
            vocab_sizes = [self.default_vocab_size] * len(indices)
        elif isinstance(spec, list):
            for item in spec:
                if isinstance(item, tuple) and len(item) == 2:
                    # (idx, vocab_size)
                    idx, vocab_size = item
                    indices.append(idx)
                    vocab_sizes.append(vocab_size)
                elif isinstance(item, int):
                    # 单个索引
                    indices.append(item)
                    vocab_sizes.append(self.default_vocab_size)
        
        return indices, vocab_sizes
    
    def _create_type_encoder(self, discrete_indices: List[int], 
                             discrete_vocab_sizes: List[int],
                             continuous_indices: List[int]) -> Tuple[nn.ModuleDict, int]:
        """
        为单个节点类型创建编码器。
        """
        encoders = nn.ModuleDict()
        total_out_dim = 0
        
        # 离散特征编码器
        for i, (idx, vocab_size) in enumerate(zip(discrete_indices, discrete_vocab_sizes)):
            key = f"discrete_{i}"
            embed_dim = max(self.out_dim // 4, 16)  # 至少16维
            encoders[key] = nn.Embedding(vocab_size, embed_dim)
            total_out_dim += embed_dim
        
        # 连续特征编码器
        if continuous_indices:
            feat_dim = len(continuous_indices)
            cont_out_dim = max(self.out_dim // 2, 32)  # 至少32维
            encoders['continuous'] = nn.Sequential(
                nn.Linear(feat_dim, cont_out_dim),
                nn.ReLU(),
            )
            total_out_dim += cont_out_dim
        
        return encoders, total_out_dim
    
    def forward(self, node_attr: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            node_attr: 节点特征 [N, feat_dim]
            node_type: 节点类型 [N]
            
        Returns:
            编码后的节点特征 [N, out_dim]
        """
        device = node_attr.device
        N = node_attr.size(0)
        output = torch.zeros(N, self.out_dim, device=device)
        
        for type_name, type_id in self.NODE_TYPE_TO_ID.items():
            if type_name not in self.feat_config:
                continue
            
            mask = (node_type == type_id)
            if not mask.any():
                continue
            
            parsed = self._parsed_configs[type_name]
            discrete_indices = parsed['discrete_indices']
            continuous_indices = parsed['continuous_indices']
            
            type_attr = node_attr[mask]
            encoders = self.encoders[type_name]
            
            encoded_parts = []
            
            # 编码离散特征
            for i, idx in enumerate(discrete_indices):
                key = f"discrete_{i}"
                feat = type_attr[:, idx].long()
                encoded = encoders[key](feat)
                encoded_parts.append(encoded)
            
            # 编码连续特征
            if continuous_indices and 'continuous' in encoders:
                feat = type_attr[:, continuous_indices]
                encoded = encoders['continuous'](feat)
                encoded_parts.append(encoded)
            
            if encoded_parts:
                type_encoded = torch.cat(encoded_parts, dim=1)
                type_output = self.projections[type_name](type_encoded)
                output[mask] = type_output
        
        return output
