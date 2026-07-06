import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, ResGatedGraphConv, 
    GINEConv, ClusterGCNConv, PNAConv
)
from torch_geometric.nn.models import SGFormer, Polynormer
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.nn.aggr import AttentionalAggregation
from layer import GatedGCNLayer, GCNConvLayer, GINEConvLayer


NET = 0
DEV = 1
PIN = 2


class GraphHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_stats = args.use_stats
        hidden_dim = args.hid_dim
        node_embed_dim = hidden_dim
        self.task = args.task
        self.task_level = args.task_level
        self.net_only = args.net_only
        self.num_classes = args.num_classes
        self.class_boundaries = args.class_boundaries
        act_fn = args.act_fn
        use_bn = args.use_bn
        dropout = args.dropout
        residual = getattr(args, 'residual', True)
        ffn = getattr(args, 'ffn', True)

        if self.use_stats:
            assert hidden_dim % 2 == 0
            node_embed_dim = hidden_dim // 2
            self.net_attr_layers = nn.Linear(17, node_embed_dim, bias=True)
            self.dev_attr_layers = nn.Linear(17, node_embed_dim, bias=True)
            self.pin_attr_layers = nn.Embedding(17, node_embed_dim)
            self.c_embed_dim = node_embed_dim

        node_type_vocab_size = getattr(args, 'node_type_vocab_size', 4)
        self.node_encoder = nn.Embedding(node_type_vocab_size, node_embed_dim)
        edge_type_vocab_size = getattr(args, 'edge_type_vocab_size', 4)
        self.edge_encoder = nn.Embedding(edge_type_vocab_size, hidden_dim)

        self.layers = nn.ModuleList()
        self.model = args.model
        self.sgformer_model = None
        self.polynormer_model = None

        if args.model == 'sgformer':
            self.sgformer_model = SGFormer(
                in_channels=hidden_dim, hidden_channels=hidden_dim,
                out_channels=hidden_dim, trans_num_layers=args.num_gnn_layers,
                trans_num_heads=getattr(args, 'num_heads', 1),
                trans_dropout=dropout, gnn_num_layers=args.num_gnn_layers,
                gnn_dropout=dropout,
            )
        elif args.model == 'polynormer':
            self.polynormer_model = Polynormer(
                in_channels=hidden_dim, hidden_channels=hidden_dim,
                out_channels=hidden_dim, local_layers=args.num_gnn_layers,
                global_layers=getattr(args, 'global_layers', 2), dropout=dropout,
            )
        else:
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
                    mlp = MLP(in_channels=hidden_dim, hidden_channels=hidden_dim,
                              out_channels=hidden_dim, num_layers=2, norm=None)
                    self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
                elif args.model == 'pna':
                    aggregators = ['mean', 'max', 'sum']
                    scalers = ['identity', 'amplification', 'attenuation']
                    deg = getattr(args, 'pna_deg', None)
                    self.layers.append(PNAConv(hidden_dim, hidden_dim,
                        aggregators=aggregators, scalers=scalers, deg=deg,
                        edge_dim=hidden_dim, towers=1, pre_layers=1, post_layers=1))
                elif args.model == 'CustomGatedGCN':
                    self.layers.append(GatedGCNLayer(in_dim=hidden_dim, out_dim=hidden_dim,
                        dropout=dropout, residual=residual, ffn=ffn, batch_norm=use_bn))
                elif args.model == 'CustomGCNConv':
                    self.layers.append(GCNConvLayer(dim_in=hidden_dim, dim_out=hidden_dim,
                        dropout=dropout, residual=residual, ffn=ffn, batch_norm=use_bn))
                elif args.model == 'CustomGINEConv':
                    self.layers.append(GINEConvLayer(dim_in=hidden_dim, dim_out=hidden_dim,
                        dropout=dropout, residual=residual, ffn=ffn, batch_norm=use_bn))
                else:
                    raise ValueError(f'Unsupported GNN model: {args.model}')
        
        self.src_dst_agg = args.src_dst_agg
        if args.src_dst_agg == 'pooladd':
            self.pooling_fun = pygnn.pool.global_add_pool
        elif args.src_dst_agg == 'poolmean':
            self.pooling_fun = pygnn.pool.global_mean_pool
        elif args.src_dst_agg == 'globalattn':
            self.attn_nn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            self.pooling_fun = AttentionalAggregation(gate_nn=self.attn_nn)

        head_input_dim = hidden_dim * 2 if self.src_dst_agg == 'concat' and self.task_level == 'edge' else hidden_dim
        dim_out = 1 if self.task == 'regression' else args.num_classes
        self.head_layers = MLP(in_channels=head_input_dim, hidden_channels=hidden_dim,
            out_channels=dim_out, num_layers=args.num_head_layers, use_bn=use_bn, dropout=dropout, activation=act_fn)

        self.use_bn = use_bn
        self.bn_node_x = nn.BatchNorm1d(hidden_dim)
        if act_fn == 'relu': self.activation = nn.ReLU()
        elif act_fn == 'elu': self.activation = nn.ELU()
        elif act_fn == 'tanh': self.activation = nn.Tanh()
        elif act_fn == 'leakyrelu': self.activation = nn.LeakyReLU()
        elif act_fn == 'prelu': self.activation = nn.PReLU()
        else: raise ValueError('Invalid activation')
        self.drop_out = dropout

    def forward(self, batch):
        x = self.node_encoder(batch.node_type)
        xe = self.edge_encoder(batch.edge_type)
        
        if self.use_stats:
            net_node_mask = batch.node_type == NET
            dev_node_mask = batch.node_type == DEV
            pin_node_mask = batch.node_type == PIN
            node_attr_emb = torch.zeros((batch.num_nodes, self.c_embed_dim), device=batch.x.device)
            node_attr_emb[net_node_mask] = self.net_attr_layers(batch.node_attr[net_node_mask])
            node_attr_emb[dev_node_mask] = self.dev_attr_layers(batch.node_attr[dev_node_mask])
            node_attr_emb[pin_node_mask] = self.pin_attr_layers(batch.node_attr[pin_node_mask, 0].long())
            x = torch.cat((x, node_attr_emb), dim=1)
            
        if self.model == 'sgformer':
            # SGFormer需要batch参数，如果为None则创建一个全0的batch（表示所有节点属于同一个图）
            batch_idx = batch.batch if batch.batch is not None else torch.zeros(batch.num_nodes, dtype=torch.long, device=x.device)
            x = self.sgformer_model(x, batch.edge_index, batch_idx)
        elif self.model == 'polynormer':
            batch_idx = batch.batch if batch.batch is not None else torch.zeros(batch.num_nodes, dtype=torch.long, device=x.device)
            x = self.polynormer_model(x, batch.edge_index, batch_idx)
        elif self.model in ['CustomGatedGCN', 'CustomGCNConv', 'CustomGINEConv']:
            batch.x = x
            batch.edge_attr = xe
            for conv in self.layers:
                batch = conv(batch)
            x = batch.x
        else:
            for conv in self.layers:
                if self.model in ['gine', 'resgatedgcn', 'pna']:
                    x = conv(x, batch.edge_index, edge_attr=xe)
                else:
                    x = conv(x, batch.edge_index)
                if self.use_bn:
                    x = self.bn_node_x(x)
                x = self.activation(x)
                if self.drop_out > 0.0:
                    x = F.dropout(x, p=self.drop_out, training=self.training)

        if self.task_level == 'node':
            if self.net_only:
                net_node_mask = batch.node_type == NET
                pred = self.head_layers(x[net_node_mask])
                true_class = batch.y[:, 1][net_node_mask].long()
                true_label = batch.y[net_node_mask]
            else:
                pred = self.head_layers(x)
                true_class = batch.y[:, 1].long()
                true_label = batch.y
        elif self.task_level == 'edge':
            if self.src_dst_agg[:4] == 'pool':
                graph_emb = self.pooling_fun(x, batch.batch)
            else:
                batch_size = batch.edge_label.size(0)
                src_emb = x[:batch_size, :]
                dst_emb = x[batch_size:batch_size*2, :]
                graph_emb = torch.cat((src_emb, dst_emb), dim=1) if self.src_dst_agg == 'concat' else src_emb + dst_emb
            pred = self.head_layers(graph_emb)
            true_class = batch.edge_label[:, 1].long()
            true_label = batch.edge_label
        else:
            raise ValueError('Invalid task level')
        return pred, true_class, true_label
