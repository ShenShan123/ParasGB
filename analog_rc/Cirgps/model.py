import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, ResGatedGraphConv, 
    GINConv, ChebConv, GINEConv, ClusterGCNConv, SSGConv
)
from torch_geometric.nn.models.mlp import MLP

# 节点类型 (与 newgraph.py 一致)
DEV = 0
PIN = 1
NET = 2

class GraphHead(nn.Module):
    """ GNN head for graph-level prediction (边任务).
    
    支持多分类和回归任务:
    - 分类任务: dim_out = num_classes
    - 回归任务: dim_out = 1
    
    标签格式: edge_label 是 [N, 2]
    - 第0列: 回归标签
    - 第1列: 分类标签
    """
    def __init__(self, args):
        super().__init__()

        self.use_pe = args.use_pe
        self.use_stats = args.use_stats
        self.task = args.task
        hidden_dim = args.hid_dim
        
        self.use_cl = getattr(args, 'use_cl', False)
        self.num_classes = getattr(args, 'num_classes', 5)

        # 重新设计维度分配：确保所有组件加起来等于 hidden_dim
        if self.use_pe and self.use_stats:
            # 三部分: node_type, pe, stats
            node_embed_dim = hidden_dim // 3
            pe_dim = hidden_dim // 3
            stats_dim = hidden_dim - node_embed_dim - pe_dim  # 确保总和正确
        elif self.use_pe:
            # 两部分: node_type, pe
            node_embed_dim = hidden_dim // 2
            pe_dim = hidden_dim - node_embed_dim
            stats_dim = 0
        elif self.use_stats:
            # 两部分: node_type, stats
            node_embed_dim = hidden_dim // 2
            pe_dim = 0
            stats_dim = hidden_dim - node_embed_dim
        else:
            # 只有 node_type
            node_embed_dim = hidden_dim
            pe_dim = 0
            stats_dim = 0

        self.node_embed_dim = node_embed_dim
        self.pe_dim = pe_dim
        self.stats_dim = stats_dim

        ## Circuit Statistics encoder
        if self.use_stats:
            self.dev_attr_layers = nn.Linear(16, stats_dim, bias=True)
            self.pin_attr_layers = nn.Linear(16, stats_dim, bias=True)
            self.net_attr_layers = nn.Linear(16, stats_dim, bias=True)
            self.c_embed_dim = stats_dim

        ## PE encoder
        if self.use_pe:
            # pe_dim 需要被 2 整除（因为有 src 和 dst 两个锚点）
            pe_single_dim = pe_dim // 2
            self.pe_encoder = nn.Embedding(num_embeddings=args.max_dist+1,
                                           embedding_dim=pe_single_dim)

        ## Node / Edge type encoders
        self.node_encoder = nn.Embedding(num_embeddings=4,
                                         embedding_dim=node_embed_dim)
        self.edge_encoder = nn.Embedding(num_embeddings=4,
                                         embedding_dim=hidden_dim)
        
        # GNN layers
        self.layers = nn.ModuleList()
        self.model = args.model

        for _ in range(args.num_gnn_layers):
            if args.model == 'clustergcn':
                self.layers.append(ClusterGCNConv(hidden_dim, hidden_dim))
            elif args.model == 'gcn':
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            elif args.model == 'sage':
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif args.model == 'gat':
                self.layers.append(GATConv(hidden_dim, hidden_dim, heads=1))
            elif args.model == 'resgatedgcn':
                self.layers.append(ResGatedGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim))
            elif args.model == 'gine':
                mlp = MLP(
                    in_channels=hidden_dim, 
                    hidden_channels=hidden_dim, 
                    out_channels=hidden_dim, 
                    num_layers=2, 
                    norm=None,
                )
                self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            else:
                raise ValueError(f'Unsupported GNN model: {args.model}')
        
        self.src_dst_agg = args.src_dst_agg

        if args.src_dst_agg == 'pooladd':
            self.pooling_fun = pygnn.pool.global_add_pool
        elif args.src_dst_agg == 'poolmean':
            self.pooling_fun = pygnn.pool.global_mean_pool
        
        ## The head configuration
        head_input_dim = hidden_dim * 2 if self.src_dst_agg == 'concat' else hidden_dim
        
        # 根据任务类型设置输出维度
        if self.task == 'classification':
            dim_out = self.num_classes
        else:
            dim_out = 1
        
        self.head_layers = MLP(
            in_channels=head_input_dim, 
            hidden_channels=hidden_dim, 
            out_channels=dim_out, 
            num_layers=args.num_head_layers, 
            use_bn=False, dropout=0.0, 
            activation=args.act_fn,
        )

        ## Batch normalization
        self.use_bn = args.use_bn
        self.bn_node_x = nn.BatchNorm1d(hidden_dim)
        if self.use_bn and self.use_cl:
            print("[Warning] Using batch normalization with contrastive learning may cause performance degradation.")

        ## activation setting
        if args.act_fn == 'relu':
            self.activation = nn.ReLU()
        elif args.act_fn == 'elu':
            self.activation = nn.ELU()
        elif args.act_fn == 'tanh':
            self.activation = nn.Tanh()
        elif args.act_fn == 'leakyrelu':
            self.activation = nn.LeakyReLU()    
        else:
            raise ValueError('Invalid activation')
        
        self.drop_out = args.dropout
        
        # 打印维度信息用于调试
        print(f"[GraphHead] hidden_dim={hidden_dim}, node_embed_dim={node_embed_dim}, pe_dim={pe_dim}, stats_dim={stats_dim}")
        print(f"[GraphHead] use_pe={self.use_pe}, use_stats={self.use_stats}")

    def forward(self, batch):
        ## Node type encoding
        x = self.node_encoder(batch.node_type)
        xe = self.edge_encoder(batch.edge_type)

        ## DSPD encoding
        if self.use_pe:
            dspd_emb = self.pe_encoder(batch.dspd)
            if dspd_emb.ndim == 3 and dspd_emb.size(1) == 2:
                dspd_emb = torch.cat((dspd_emb[:, 0, :], dspd_emb[:, 1, :]), dim=1)
            else:
                raise ValueError(
                    f"Dimension number of DSPD embedding is" + 
                    f" {dspd_emb.ndim}, size {dspd_emb.size()}")
            x = torch.cat((x, dspd_emb), dim=1)

        ## Circuit statistics encoding
        if self.use_stats:
            dev_node_mask = batch.node_type == DEV
            pin_node_mask = batch.node_type == PIN
            net_node_mask = batch.node_type == NET
            node_attr_emb = torch.zeros(
                (batch.num_nodes, self.c_embed_dim), device=x.device
            )
            node_attr_emb[dev_node_mask] = \
                self.dev_attr_layers(batch.node_attr[dev_node_mask])
            node_attr_emb[pin_node_mask] = \
                self.pin_attr_layers(batch.node_attr[pin_node_mask])
            node_attr_emb[net_node_mask] = \
                self.net_attr_layers(batch.node_attr[net_node_mask])
            x = torch.cat((x, node_attr_emb), dim=1)

        for conv in self.layers:
            if self.model == 'gine' or self.model == 'resgatedgcn':
                x = conv(x, batch.edge_index, edge_attr=xe)
            else:
                x = conv(x, batch.edge_index)

            if self.use_bn:
                x = self.bn_node_x(x)
            
            x = self.activation(x)

            if self.drop_out > 0.0:
                x = F.dropout(x, p=self.drop_out, training=self.training)

        ## Head layers
        if self.src_dst_agg[:4] == 'pool':
            graph_emb = self.pooling_fun(x, batch.batch)
        else:
            batch_size = batch.edge_label.size(0)
            src_emb = x[:batch_size, :]
            dst_emb = x[batch_size:batch_size*2, :]
            if self.src_dst_agg == 'concat':
                graph_emb = torch.cat((src_emb, dst_emb), dim=1)
            else:
                graph_emb = src_emb + dst_emb

        pred = self.head_layers(graph_emb)

        # 处理标签: edge_label 是 [N, 2] 格式
        # 第0列: 回归标签，第1列: 分类标签
        edge_label = batch.edge_label
        if edge_label.dim() == 2 and edge_label.size(1) == 2:
            if self.task == 'classification':
                true = edge_label[:, 1].long()  # 分类用第1列
            else:
                true = edge_label[:, 0]  # 回归用第0列
        else:
            true = edge_label  # 兼容旧格式

        return pred, true


class NodeHead(nn.Module):
    """GNN head for node-level prediction (节点任务).
    
    支持多分类和回归任务:
    - 分类任务: dim_out = num_classes
    - 回归任务: dim_out = 1
    
    标签格式: node_label 是 [N, 2]
    - 第0列: 回归标签
    - 第1列: 分类标签
    """
    def __init__(self, args):
        super().__init__()
        
        self.use_pe = args.use_pe
        self.use_stats = args.use_stats
        self.task = args.task
        hidden_dim = args.hid_dim
        
        self.use_cl = getattr(args, 'use_cl', False)
        self.num_classes = getattr(args, 'num_classes', 5)
        
        # 重新设计维度分配：确保所有组件加起来等于 hidden_dim
        if self.use_pe and self.use_stats:
            # 三部分: node_type, pe, stats
            node_embed_dim = hidden_dim // 3
            pe_dim = hidden_dim // 3
            stats_dim = hidden_dim - node_embed_dim - pe_dim
        elif self.use_pe:
            # 两部分: node_type, pe
            node_embed_dim = hidden_dim // 2
            pe_dim = hidden_dim - node_embed_dim
            stats_dim = 0
        elif self.use_stats:
            # 两部分: node_type, stats
            node_embed_dim = hidden_dim // 2
            pe_dim = 0
            stats_dim = hidden_dim - node_embed_dim
        else:
            # 只有 node_type
            node_embed_dim = hidden_dim
            pe_dim = 0
            stats_dim = 0

        self.node_embed_dim = node_embed_dim
        self.pe_dim = pe_dim
        self.stats_dim = stats_dim
        
        # Circuit Statistics encoder
        if self.use_stats:
            self.dev_attr_layers = nn.Linear(16, stats_dim, bias=True)
            self.pin_attr_layers = nn.Linear(16, stats_dim, bias=True)
            self.net_attr_layers = nn.Linear(16, stats_dim, bias=True)
            self.c_embed_dim = stats_dim
        
        # PE encoder
        if self.use_pe:
            pe_single_dim = pe_dim // 2
            self.pe_encoder = nn.Embedding(
                num_embeddings=args.max_dist + 1,
                embedding_dim=pe_single_dim
            )
        
        # Node / Edge type encoders
        self.node_encoder = nn.Embedding(num_embeddings=4, embedding_dim=node_embed_dim)
        self.edge_encoder = nn.Embedding(num_embeddings=4, embedding_dim=hidden_dim)
        
        # GNN layers
        self.layers = nn.ModuleList()
        self.model = args.model
        
        for _ in range(args.num_gnn_layers):
            if args.model == 'clustergcn':
                self.layers.append(ClusterGCNConv(hidden_dim, hidden_dim))
            elif args.model == 'gcn':
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            elif args.model == 'sage':
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif args.model == 'gat':
                self.layers.append(GATConv(hidden_dim, hidden_dim, heads=1))
            elif args.model == 'resgatedgcn':
                self.layers.append(ResGatedGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim))
            elif args.model == 'gine':
                mlp = MLP(
                    in_channels=hidden_dim,
                    hidden_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_layers=2,
                    norm=None,
                )
                self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            else:
                raise ValueError(f'Unsupported GNN model: {args.model}')
        
        # 根据任务类型设置输出维度
        if self.task == 'classification':
            dim_out = self.num_classes
        else:
            dim_out = 1
            
        self.head_layers = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=dim_out,
            num_layers=args.num_head_layers,
            use_bn=False, dropout=0.0,
            activation=args.act_fn,
        )
        
        # Batch normalization
        self.use_bn = args.use_bn
        self.bn_node_x = nn.BatchNorm1d(hidden_dim)
        
        # Activation
        if args.act_fn == 'relu':
            self.activation = nn.ReLU()
        elif args.act_fn == 'elu':
            self.activation = nn.ELU()
        elif args.act_fn == 'tanh':
            self.activation = nn.Tanh()
        elif args.act_fn == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError('Invalid activation')
        
        self.drop_out = args.dropout
        
        # 打印维度信息用于调试
        print(f"[NodeHead] hidden_dim={hidden_dim}, node_embed_dim={node_embed_dim}, pe_dim={pe_dim}, stats_dim={stats_dim}")
        print(f"[NodeHead] use_pe={self.use_pe}, use_stats={self.use_stats}")
    
    def forward(self, batch, node_labels):
        """
        Args:
            batch: 子图批次数据
            node_labels: 目标节点的标签 [N, 2]，第0列回归，第1列分类
        Returns:
            pred: 预测值
            true: 真实标签 (根据任务类型选择)
        """
        # Node type encoding
        x = self.node_encoder(batch.node_type)
        xe = self.edge_encoder(batch.edge_type)
        
        # DSPD encoding
        if self.use_pe:
            dspd_emb = self.pe_encoder(batch.dspd)
            if dspd_emb.ndim == 3 and dspd_emb.size(1) == 2:
                dspd_emb = torch.cat((dspd_emb[:, 0, :], dspd_emb[:, 1, :]), dim=1)
            else:
                raise ValueError(f"DSPD embedding dimension error: {dspd_emb.size()}")
            x = torch.cat((x, dspd_emb), dim=1)
        
        # Circuit statistics encoding
        if self.use_stats:
            dev_node_mask = batch.node_type == DEV
            pin_node_mask = batch.node_type == PIN
            net_node_mask = batch.node_type == NET
            
            node_attr_emb = torch.zeros(
                (batch.num_nodes, self.c_embed_dim), device=x.device
            )
            node_attr_emb[dev_node_mask] = self.dev_attr_layers(batch.node_attr[dev_node_mask])
            node_attr_emb[pin_node_mask] = self.pin_attr_layers(batch.node_attr[pin_node_mask])
            node_attr_emb[net_node_mask] = self.net_attr_layers(batch.node_attr[net_node_mask])
            x = torch.cat((x, node_attr_emb), dim=1)
        
        # GNN layers
        for conv in self.layers:
            if self.model == 'gine' or self.model == 'resgatedgcn':
                x = conv(x, batch.edge_index, edge_attr=xe)
            else:
                x = conv(x, batch.edge_index)
            
            if self.use_bn:
                x = self.bn_node_x(x)
            
            x = self.activation(x)
            
            if self.drop_out > 0.0:
                x = F.dropout(x, p=self.drop_out, training=self.training)
        
        # 获取目标节点的嵌入
        batch_size = node_labels.size(0)
        target_emb = x[:batch_size, :]
        
        pred = self.head_layers(target_emb)
        
        # 处理标签: node_labels 是 [N, 2] 格式
        # 第0列: 回归标签，第1列: 分类标签
        if node_labels.dim() == 2 and node_labels.size(1) == 2:
            if self.task == 'classification':
                true = node_labels[:, 1].long()  # 分类用第1列
            else:
                true = node_labels[:, 0]  # 回归用第0列
        else:
            true = node_labels  # 兼容旧格式
        
        return pred, true
