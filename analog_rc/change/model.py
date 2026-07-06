"""
GNN Model for node and edge level tasks.
支持硬编码位置索引来提取目标节点/边的预测。
支持节点特征编码：节点类型嵌入 + 节点特征嵌入 拼接。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINEConv, PNAConv
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.nn.models import Polynormer, SGFormer
from torch_geometric.data import Batch
from torch_geometric.utils import degree

from layer import GatedGCNLayer, GCNConvLayer, GINEConvLayer
from node_encoder import NodeFeatureEncoder


def compute_degree_histogram(graphs):
    """
    计算图列表的 degree histogram，用于 PNA 模型
    
    Args:
        graphs: 图对象列表，每个图需要有 edge_index 和 num_nodes 属性
    
    Returns:
        deg: degree histogram tensor
    """
    max_degree = 0
    for g in graphs:
        d = degree(g.edge_index[1], num_nodes=g.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))
    
    # 计算 degree histogram
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for g in graphs:
        d = degree(g.edge_index[1], num_nodes=g.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    
    return deg


class GNNModel(nn.Module):
    """
    GNN model for node/edge regression or classification.
    - 节点任务: 预测net节点的标签
    - 边任务: 预测(pin, pair_to, pin)边的标签
    
    标签格式: y [N, 2] 或 edge_label_y [E, 2]
    - 第0列: 回归标签
    - 第1列: 分类标签
    """
    
    def __init__(self, args):
        super().__init__()
        
        self.task_level = args.task_level
        self.task = args.task
        self.use_node_attr = getattr(args, 'use_node_attr', False)
        hid_dim = args.hid_dim
        
        if self.use_node_attr:
            type_embed_dim = hid_dim // 2
            feat_embed_dim = hid_dim // 2
            
            self.node_type_encoder = nn.Embedding(num_embeddings=4, embedding_dim=type_embed_dim)
            
            feat_config = getattr(args, 'feat_config', None)
            self.node_feat_encoder = NodeFeatureEncoder(out_dim=feat_embed_dim, feat_config=feat_config)
            
            self.fusion = nn.Linear(hid_dim, hid_dim)
        else:
            self.node_type_encoder = nn.Embedding(num_embeddings=4, embedding_dim=hid_dim)
            self.node_feat_encoder = None
            self.fusion = None
        
        self.edge_encoder = nn.Embedding(num_embeddings=8, embedding_dim=hid_dim)
        
        self.convs = nn.ModuleList()
        self.model_type = args.model
        
        # SGFormer 和 Polynormer 使用独立的模型结构
        if args.model == 'sgformer':
            # SGFormer 参数: in_channels, hidden_channels, out_channels, 
            # trans_num_layers, trans_num_heads, trans_dropout, 
            # gnn_num_layers, gnn_dropout, graph_weight, aggregate
            self.gnn_model = SGFormer(
                in_channels=hid_dim,
                hidden_channels=hid_dim,
                out_channels=hid_dim,
                trans_num_layers=getattr(args, 'trans_num_layers', 2),
                trans_num_heads=getattr(args, 'trans_num_heads', 1),
                trans_dropout=getattr(args, 'trans_dropout', args.dropout),
                gnn_num_layers=getattr(args, 'gnn_num_layers', 3),
                gnn_dropout=getattr(args, 'gnn_dropout', args.dropout),
                graph_weight=getattr(args, 'graph_weight', 0.5),
                aggregate=getattr(args, 'aggregate', 'add'),
            )
        elif args.model == 'polynormer':
            # Polynormer 参数: in_channels, hidden_channels, out_channels,
            # local_layers, global_layers, in_dropout, dropout, global_dropout,
            # heads, beta, qk_shared, pre_ln, post_bn, local_attn
            self.gnn_model = Polynormer(
                in_channels=hid_dim,
                hidden_channels=hid_dim,
                out_channels=hid_dim,
                local_layers=getattr(args, 'local_layers', 7),
                global_layers=getattr(args, 'global_layers', 2),
                in_dropout=getattr(args, 'in_dropout', 0.15),
                dropout=args.dropout,
                global_dropout=getattr(args, 'global_dropout', args.dropout),
                heads=getattr(args, 'poly_heads', 1),
                beta=getattr(args, 'beta', 0.9),
                local_attn=getattr(args, 'local_attn', False),
            )
        else:
            self.gnn_model = None
            for _ in range(args.num_layers):
                if args.model == 'gcn':
                    self.convs.append(GCNConv(hid_dim, hid_dim))
                elif args.model == 'sage':
                    self.convs.append(SAGEConv(hid_dim, hid_dim))
                elif args.model == 'gat':
                    self.convs.append(GATConv(hid_dim, hid_dim, heads=1))
                elif args.model == 'gine':
                    mlp = MLP(in_channels=hid_dim, hidden_channels=hid_dim, 
                             out_channels=hid_dim, num_layers=2)
                    self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=hid_dim))
                elif args.model == 'CustomGatedGCN':
                    self.convs.append(GatedGCNLayer(hid_dim, hid_dim,
                                                   dropout=args.dropout,
                                                   residual=True,
                                                   ffn=True,
                                                   batch_norm=True))
                elif args.model == 'CustomGCNConv':
                    self.convs.append(GCNConvLayer(hid_dim, hid_dim,
                                                  dropout=args.dropout,
                                                  residual=True,
                                                  ffn=True,
                                                  batch_norm=True))
                elif args.model == 'CustomGINEConv':
                    self.convs.append(GINEConvLayer(hid_dim, hid_dim,
                                                   dropout=args.dropout,
                                                   residual=True,
                                                   ffn=True,
                                                   batch_norm=True))
                elif args.model == 'pna':
                    # PNA 需要预计算的 degree 信息
                    # aggregators: 聚合方式 (mean, min, max, std)
                    # scalers: 缩放方式 (identity, amplification, attenuation)
                    aggregators = ['mean', 'min', 'max', 'std']
                    scalers = ['identity', 'amplification', 'attenuation']
                    deg = getattr(args, 'deg', None)  # 需要从数据集预计算
                    if deg is None:
                        # 默认 degree histogram，实际使用时应该从数据集计算
                        deg = torch.ones(100, dtype=torch.long)
                    self.convs.append(PNAConv(
                        in_channels=hid_dim,
                        out_channels=hid_dim,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=deg,
                        edge_dim=hid_dim,
                        towers=getattr(args, 'pna_towers', 4),
                        pre_layers=1,
                        post_layers=1,
                        divide_input=False,
                    ))
        
        self.bns = nn.ModuleList([nn.BatchNorm1d(hid_dim) for _ in range(args.num_layers)])
        
        if self.task_level == 'edge':
            head_in_dim = hid_dim * 2
        else:
            head_in_dim = hid_dim
        
        out_dim = args.num_classes if args.task == 'classification' else 1
        
        self.head = MLP(
            in_channels=head_in_dim,
            hidden_channels=hid_dim,
            out_channels=out_dim,
            num_layers=2,
            dropout=args.dropout,
        )
        
        self.dropout = args.dropout
        self.hid_dim = hid_dim
        
        activation_name = getattr(args, 'activation', 'prelu')
        self.activation = self._get_activation(activation_name)
    
    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'elu':
            return nn.ELU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'leakyrelu':
            return nn.LeakyReLU(0.2)
        elif name == 'prelu':
            return nn.PReLU()
        else:
            return nn.PReLU()
    
    def _encode_nodes(self, batch):
        if self.use_node_attr and hasattr(batch, 'x'):
            type_emb = self.node_type_encoder(batch.node_type)
            feat_emb = self.node_feat_encoder(batch.x, batch.node_type)
            x = torch.cat([type_emb, feat_emb], dim=1)
            x = self.fusion(x)
        else:
            x = self.node_type_encoder(batch.node_type)
        
        return x
    
    def forward(self, batch):
        
        x = self._encode_nodes(batch)
        edge_attr = self.edge_encoder(batch.edge_type)
        
        # 获取 batch tensor，如果不存在则创建（所有节点属于同一个图）
        batch_tensor = getattr(batch, 'batch', None)
        if batch_tensor is None:
            batch_tensor = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # SGFormer 和 Polynormer 使用独立的前向传播
        if self.model_type == 'sgformer':
            x = self.gnn_model(x, batch.edge_index, batch_tensor)
        elif self.model_type == 'polynormer':
            x = self.gnn_model(x, batch.edge_index, batch_tensor)
        else:
            for i, conv in enumerate(self.convs):
                if self.model_type in ['CustomGatedGCN', 'CustomGINEConv']:
                    temp_batch = Batch(x=x, edge_index=batch.edge_index, edge_attr=edge_attr,
                                      batch=batch.batch if hasattr(batch, 'batch') else None)
                    temp_batch = conv(temp_batch)
                    x = temp_batch.x
                    edge_attr = temp_batch.edge_attr
                elif self.model_type == 'CustomGCNConv':
                    temp_batch = Batch(x=x, edge_index=batch.edge_index,
                                      batch=batch.batch if hasattr(batch, 'batch') else None)
                    temp_batch = conv(temp_batch)
                    x = temp_batch.x
                elif self.model_type in ['gine', 'gat']:
                    x = conv(x, batch.edge_index, edge_attr=edge_attr)
                    x = self.bns[i](x)
                    x = self.activation(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                elif self.model_type == 'pna':
                    x = conv(x, batch.edge_index, edge_attr=edge_attr)
                    x = self.bns[i](x)
                    x = self.activation(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
                else:
                    x = conv(x, batch.edge_index)
                    x = self.bns[i](x)
                    x = self.activation(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
        
        if self.task_level == 'node':
            return self._node_output(x, batch)
        else:
            return self._edge_output(x, batch)
    
    def _node_output(self, x, batch):
        """
        节点级任务输出。
        y格式: [N, 2], 第0列回归，第1列分类
        """
        pred = self.head(x)
        
        if hasattr(batch, 'batch_size'):
            batch_size = batch.batch_size
            pred = pred[:batch_size]
            
            if hasattr(batch, 'train_node_mask'):
                valid_mask = batch.train_node_mask[:batch_size]
            elif hasattr(batch, 'target_node_type_id'):
                target_type_id = batch.target_node_type_id
                node_types = batch.node_type[:batch_size]
                valid_mask = (node_types == target_type_id)
            else:
                valid_mask = batch.target_node_mask[:batch_size] if hasattr(batch, 'target_node_mask') else torch.ones(batch_size, dtype=torch.bool, device=pred.device)
            
            pred = pred[valid_mask]
            y_full = batch.y[:batch_size][valid_mask]
        else:
            if hasattr(batch, 'train_node_mask'):
                valid_mask = batch.train_node_mask
            elif hasattr(batch, 'target_node_type_id'):
                target_type_id = batch.target_node_type_id
                valid_mask = (batch.node_type == target_type_id)
            elif hasattr(batch, 'target_node_mask'):
                valid_mask = batch.target_node_mask
            else:
                valid_mask = torch.ones(batch.num_nodes, dtype=torch.bool, device=pred.device)
            
            pred = pred[valid_mask]
            y_full = batch.y[valid_mask]
        
        # 根据任务类型选择标签列
        if self.task == 'classification':
            y = y_full[:, 1].long()  # 第1列: 分类标签
        else:
            y = y_full[:, 0]  # 第0列: 回归标签
        
        return pred, y
    
    def _edge_output(self, x, batch):
        """
        边级任务输出。
        edge_label格式: [E, 2], 第0列回归，第1列分类
        """
        if hasattr(batch, 'edge_label_index'):
            src_idx = batch.edge_label_index[0]
            dst_idx = batch.edge_label_index[1]
            
            src_emb = x[src_idx]
            dst_emb = x[dst_idx]
            
            edge_emb = torch.cat([src_emb, dst_emb], dim=1)
            pred = self.head(edge_emb)
            
            edge_label = batch.edge_label
            
            # 根据任务类型选择标签列
            if edge_label.dim() == 2 and edge_label.size(1) == 2:
                if self.task == 'classification':
                    y = edge_label[:, 1].long()  # 第1列: 分类标签
                else:
                    y = edge_label[:, 0]  # 第0列: 回归标签
            else:
                y = edge_label  # 兼容旧格式
        else:
            pred = torch.zeros((0, 1), device=x.device)
            y = torch.zeros(0, device=x.device)
        
        return pred, y
