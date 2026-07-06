import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINEConv,
    SAGEConv,
    ClusterGCNConv,
    ResGatedGraphConv,
)
from torch_geometric.nn.models.mlp import MLP

from node_encoder import NodeFeatureEncoder


DEV = 0
PIN = 1
NET = 2


def split_dims(total_dim, num_parts):
    base_dim = total_dim // num_parts
    dims = [base_dim] * num_parts
    dims[-1] = total_dim - base_dim * (num_parts - 1)
    return dims


class NodeInputEncoder(nn.Module):
    def __init__(self, node_type_dim, node_feat_dim, cl_dim=0, cl_hid_dim=0):
        super().__init__()
        self.use_cl = cl_dim > 0
        self.node_type_encoder = nn.Embedding(num_embeddings=4, embedding_dim=node_type_dim)
        self.node_feature_encoder = NodeFeatureEncoder(out_dim=node_feat_dim)
        self.cl_linear = nn.Linear(cl_hid_dim, cl_dim) if self.use_cl else None

    def forward(self, batch):
        parts = [
            self.node_type_encoder(batch.node_type),
            self.node_feature_encoder(batch.node_attr, batch.node_type),
        ]
        if self.use_cl:
            parts.append(self.cl_linear(batch.x))
        return torch.cat(parts, dim=1)


class GraphHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_cl = bool(args.sgrl)
        self.task = args.task
        self.task_level = args.task_level
        self.net_only = args.net_only
        self.num_classes = args.num_classes
        self.class_boundaries = args.class_boundaries
        hidden_dim = args.hid_dim

        if self.use_cl:
            node_type_dim, node_feat_dim, cl_dim = split_dims(hidden_dim, 3)
            self.node_input_encoder = NodeInputEncoder(
                node_type_dim=node_type_dim,
                node_feat_dim=node_feat_dim,
                cl_dim=cl_dim,
                cl_hid_dim=args.cl_hid_dim,
            )
        else:
            node_type_dim, node_feat_dim = split_dims(hidden_dim, 2)
            self.node_input_encoder = NodeInputEncoder(
                node_type_dim=node_type_dim,
                node_feat_dim=node_feat_dim,
            )

        self.edge_encoder = nn.Embedding(num_embeddings=4, embedding_dim=hidden_dim)

        self.layers = nn.ModuleList()
        self.model = args.model
        for _ in range(args.num_gnn_layers):
            if args.model == "clustergcn":
                self.layers.append(ClusterGCNConv(hidden_dim, hidden_dim))
            elif args.model == "gcn":
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            elif args.model == "sage":
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            elif args.model == "gat":
                self.layers.append(GATConv(hidden_dim, hidden_dim, heads=1))
            elif args.model == "resgatedgcn":
                self.layers.append(
                    ResGatedGraphConv(hidden_dim, hidden_dim, edge_dim=hidden_dim)
                )
            elif args.model == "gine":
                mlp = MLP(
                    in_channels=hidden_dim,
                    hidden_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_layers=2,
                    norm=None,
                )
                self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
            else:
                raise ValueError(f"Unsupported GNN model: {args.model}")

        self.src_dst_agg = args.src_dst_agg
        if args.src_dst_agg == "pooladd":
            self.pooling_fun = pygnn.pool.global_add_pool
        elif args.src_dst_agg == "poolmean":
            self.pooling_fun = pygnn.pool.global_mean_pool

        head_input_dim = (
            hidden_dim * 2
            if self.src_dst_agg == "concat" and self.task_level == "edge"
            else hidden_dim
        )

        if self.task == "regression":
            dim_out = 1
        elif self.task == "classification":
            dim_out = args.num_classes
        else:
            raise ValueError("Invalid task")

        self.head_layers = MLP(
            in_channels=head_input_dim,
            hidden_channels=hidden_dim,
            out_channels=dim_out,
            num_layers=args.num_head_layers,
            use_bn=False,
            dropout=0.0,
            activation=args.act_fn,
        )

        self.use_bn = args.use_bn
        self.bn_node_x = nn.BatchNorm1d(hidden_dim)
        if self.use_bn and self.use_cl:
            print("[Warning] Using batch normalization with contrastive learning may cause performance degradation.")

        if args.act_fn == "relu":
            self.activation = nn.ReLU()
        elif args.act_fn == "elu":
            self.activation = nn.ELU()
        elif args.act_fn == "tanh":
            self.activation = nn.Tanh()
        elif args.act_fn == "leakyrelu":
            self.activation = nn.LeakyReLU()
        elif args.act_fn == "prelu":
            self.activation = nn.PReLU()
        else:
            raise ValueError("Invalid activation")

        self.drop_out = args.dropout

    def _encode_nodes(self, batch):
        return self.node_input_encoder(batch)

    def forward(self, batch):
        x = self._encode_nodes(batch)
        xe = self.edge_encoder(batch.edge_type)

        for conv in self.layers:
            if self.model in {"gine", "resgatedgcn"}:
                x = conv(x, batch.edge_index, edge_attr=xe)
            else:
                x = conv(x, batch.edge_index)

            if self.use_bn:
                x = self.bn_node_x(x)

            x = self.activation(x)

            if self.drop_out > 0.0:
                x = F.dropout(x, p=self.drop_out, training=self.training)

        if self.task_level == "node":
            if self.net_only:
                net_node_mask = batch.node_type == NET
                pred = self.head_layers(x[net_node_mask])
                true_class = batch.y[:, 1][net_node_mask].long()
                true_label = batch.y[net_node_mask]
            else:
                pred = self.head_layers(x)
                true_class = batch.y[:, 1].long()
                true_label = batch.y
        elif self.task_level == "edge":
            if self.src_dst_agg[:4] == "pool":
                graph_emb = self.pooling_fun(x, batch.batch)
            else:
                batch_size = batch.edge_label.size(0)
                src_emb = x[:batch_size, :]
                dst_emb = x[batch_size : batch_size * 2, :]
                if self.src_dst_agg == "concat":
                    graph_emb = torch.cat((src_emb, dst_emb), dim=1)
                else:
                    graph_emb = src_emb + dst_emb

            pred = self.head_layers(graph_emb)
            true_class = batch.edge_label[:, 1].long()
            true_label = batch.edge_label
        else:
            raise ValueError("Invalid task level")

        return pred, true_class, true_label
