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


class DualPEEncoder(nn.Module):
    def __init__(self, max_dist, out_dim):
        super().__init__()
        self.max_dist = max_dist
        self.out_dim = out_dim
        self.src_dim = out_dim // 2
        self.dst_dim = out_dim - self.src_dim
        self.src_encoder = nn.Embedding(max_dist + 1, self.src_dim) if self.src_dim > 0 else None
        self.dst_encoder = nn.Embedding(max_dist + 1, self.dst_dim) if self.dst_dim > 0 else None

    def forward(self, dspd):
        if dspd.ndim != 2 or dspd.size(1) != 2:
            raise ValueError(f"Expected dspd with shape [num_nodes, 2], got {tuple(dspd.size())}")

        parts = []
        if self.src_encoder is not None:
            src_idx = dspd[:, 0].long().clamp(0, self.max_dist)
            parts.append(self.src_encoder(src_idx))
        if self.dst_encoder is not None:
            dst_idx = dspd[:, 1].long().clamp(0, self.max_dist)
            parts.append(self.dst_encoder(dst_idx))

        if not parts:
            return torch.zeros(dspd.size(0), 0, device=dspd.device)

        return torch.cat(parts, dim=1)


class NodeInputEncoder(nn.Module):
    def __init__(self, node_type_dim, node_feat_dim, pe_dim=0, max_dist=0):
        super().__init__()
        self.use_pe = pe_dim > 0
        self.node_type_encoder = nn.Embedding(num_embeddings=4, embedding_dim=node_type_dim)
        self.node_feature_encoder = NodeFeatureEncoder(out_dim=node_feat_dim)
        self.pe_encoder = DualPEEncoder(max_dist=max_dist, out_dim=pe_dim) if self.use_pe else None

    def forward(self, batch):
        parts = [
            self.node_type_encoder(batch.node_type),
            self.node_feature_encoder(batch.node_attr, batch.node_type),
        ]
        if self.use_pe:
            parts.append(self.pe_encoder(batch.dspd))
        return torch.cat(parts, dim=1)


class BaseHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_pe = bool(args.use_pe)
        self.task = args.task
        self.model = args.model
        self.use_bn = args.use_bn
        self.drop_out = args.dropout
        self.hidden_dim = args.hid_dim
        self.num_classes = getattr(args, "num_classes", 5)

        if self.use_pe:
            node_type_dim, node_feat_dim, pe_dim = split_dims(self.hidden_dim, 3)
            self.node_input_encoder = NodeInputEncoder(
                node_type_dim=node_type_dim,
                node_feat_dim=node_feat_dim,
                pe_dim=pe_dim,
                max_dist=args.max_dist,
            )
        else:
            node_type_dim, node_feat_dim = split_dims(self.hidden_dim, 2)
            self.node_input_encoder = NodeInputEncoder(
                node_type_dim=node_type_dim,
                node_feat_dim=node_feat_dim,
            )

        self.edge_encoder = nn.Embedding(num_embeddings=4, embedding_dim=self.hidden_dim)
        self.layers = nn.ModuleList()

        for _ in range(args.num_gnn_layers):
            if args.model == "clustergcn":
                self.layers.append(ClusterGCNConv(self.hidden_dim, self.hidden_dim))
            elif args.model == "gcn":
                self.layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
            elif args.model == "sage":
                self.layers.append(SAGEConv(self.hidden_dim, self.hidden_dim))
            elif args.model == "gat":
                self.layers.append(GATConv(self.hidden_dim, self.hidden_dim, heads=1))
            elif args.model == "resgatedgcn":
                self.layers.append(
                    ResGatedGraphConv(self.hidden_dim, self.hidden_dim, edge_dim=self.hidden_dim)
                )
            elif args.model == "gine":
                mlp = MLP(
                    in_channels=self.hidden_dim,
                    hidden_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    num_layers=2,
                    norm=None,
                )
                self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=self.hidden_dim))
            else:
                raise ValueError(f"Unsupported GNN model: {args.model}")

        self.bn_node_x = nn.BatchNorm1d(self.hidden_dim)

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

    def _build_output_head(self, head_input_dim, args):
        dim_out = self.num_classes if self.task == "classification" else 1
        return MLP(
            in_channels=head_input_dim,
            hidden_channels=self.hidden_dim,
            out_channels=dim_out,
            num_layers=args.num_head_layers,
            use_bn=False,
            dropout=0.0,
            activation=args.act_fn,
        )

    def _encode_nodes(self, batch):
        return self.node_input_encoder(batch)

    def _run_backbone(self, batch):
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

        return x


class GraphHead(BaseHead):
    def __init__(self, args):
        super().__init__(args)
        self.src_dst_agg = args.src_dst_agg
        if args.src_dst_agg == "pooladd":
            self.pooling_fun = pygnn.pool.global_add_pool
        elif args.src_dst_agg == "poolmean":
            self.pooling_fun = pygnn.pool.global_mean_pool

        head_input_dim = self.hidden_dim * 2 if self.src_dst_agg == "concat" else self.hidden_dim
        self.head_layers = self._build_output_head(head_input_dim=head_input_dim, args=args)

    def forward(self, batch):
        x = self._run_backbone(batch)

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
        edge_label = batch.edge_label
        if edge_label.dim() == 2 and edge_label.size(1) == 2:
            true = edge_label[:, 1].long() if self.task == "classification" else edge_label[:, 0]
        else:
            true = edge_label
        return pred, true


class NodeHead(BaseHead):
    def __init__(self, args):
        super().__init__(args)
        self.head_layers = self._build_output_head(head_input_dim=self.hidden_dim, args=args)

    def forward(self, batch, node_labels):
        x = self._run_backbone(batch)
        batch_size = node_labels.size(0)
        target_emb = x[:batch_size, :]
        pred = self.head_layers(target_emb)

        if node_labels.dim() == 2 and node_labels.size(1) == 2:
            true = node_labels[:, 1].long() if self.task == "classification" else node_labels[:, 0]
        else:
            true = node_labels
        return pred, true
