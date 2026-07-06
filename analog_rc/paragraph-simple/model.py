import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.nn import GINEConv
from torch_geometric.nn.models.mlp import MLP

from node_encoder import NodeFeatureEncoder


class NodeInputEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        type_embed_dim = max(embed_dim // 2, 1)
        feat_embed_dim = embed_dim - type_embed_dim
        if feat_embed_dim <= 0:
            raise ValueError("embed_dim must be at least 2.")

        self.node_type_encoder = nn.Embedding(num_embeddings=4, embedding_dim=type_embed_dim)
        self.node_feat_encoder = NodeFeatureEncoder(out_dim=feat_embed_dim)
        self.fusion = nn.Linear(embed_dim, embed_dim)

    def forward(self, node_type, node_attr):
        type_emb = self.node_type_encoder(node_type)
        feat_emb = self.node_feat_encoder(node_attr, node_type)
        return self.fusion(torch.cat((type_emb, feat_emb), dim=1))


def build_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU()
    if name == "elu":
        return nn.ELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError("Invalid activation")


class GraphHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dim_out,
        num_layers=2,
        num_head_layers=2,
        use_bn=False,
        drop_out=0.0,
        activation="relu",
        src_dst_agg="concat",
        max_dist=400,
        task="classification",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task = task

        node_embed_dim = min(hidden_dim, 64)
        self.input_encoder = NodeInputEncoder(node_embed_dim)
        self.edge_encoder = nn.Embedding(num_embeddings=8, embedding_dim=node_embed_dim)

        self.layers = nn.ModuleList()
        self.drop_out = min(drop_out, 0.3)
        self.use_bn = use_bn
        for _ in range(num_layers):
            mlp = MLP(
                in_channels=node_embed_dim,
                hidden_channels=node_embed_dim,
                out_channels=node_embed_dim,
                num_layers=2,
                norm=None,
                activation=activation,
            )
            self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=node_embed_dim))

        self.src_dst_agg = src_dst_agg
        head_input_dim = node_embed_dim * 2 if src_dst_agg == "concat" else node_embed_dim
        if src_dst_agg == "pool":
            self.pooling_fun = pygnn.pool.global_mean_pool

        self.head_layers = MLP(
            in_channels=head_input_dim,
            hidden_channels=node_embed_dim,
            out_channels=dim_out,
            num_layers=num_head_layers,
            use_bn=False,
            dropout=0.0,
            activation=activation,
        )
        self.bn_node_x = nn.BatchNorm1d(node_embed_dim)
        self.activation = build_activation(activation)

    def forward(self, batch):
        try:
            z = self.input_encoder(batch.node_type, batch.node_attr)
            edge_attr = self.edge_encoder(batch.edge_type)

            for conv in self.layers:
                edge_index = batch.edge_index.to(z.device)
                z = conv(z, edge_index, edge_attr=edge_attr)

                if self.use_bn:
                    z = self.bn_node_x(z)
                z = self.activation(z)
                if self.drop_out > 0.0:
                    z = F.dropout(z, p=self.drop_out, training=self.training)

            if self.src_dst_agg == "pool":
                graph_emb = self.pooling_fun(z, batch.batch)
            else:
                batch_size = batch.edge_label.size(0)
                src_emb = z[:batch_size, :]
                dst_emb = z[batch_size : batch_size * 2, :]
                if self.src_dst_agg == "concat":
                    graph_emb = torch.cat((src_emb, dst_emb), dim=1)
                else:
                    graph_emb = src_emb + dst_emb

            pred = self.head_layers(graph_emb)
            edge_labels = batch.edge_label
            if edge_labels.dim() == 2 and edge_labels.size(1) == 2:
                if self.task == "classification":
                    y = edge_labels[:, 1].long()
                else:
                    y = edge_labels[:, 0]
            else:
                y = edge_labels
            return pred, y

        except RuntimeError as e:
            print(f"Error in forward pass: {str(e)}")
            print(
                f"Input shapes - node_type: {batch.node_type.shape}, "
                f"node_attr: {batch.node_attr.shape}, edge_index: {batch.edge_index.shape}"
            )
            if hasattr(batch, "edge_label"):
                print(f"Edge label shape: {batch.edge_label.shape}")
            raise


class NodeHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dim_out=1,
        num_layers=2,
        num_head_layers=2,
        use_bn=False,
        drop_out=0.0,
        activation="relu",
        task="regression",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task = task

        node_embed_dim = min(hidden_dim, 64)
        self.input_encoder = NodeInputEncoder(node_embed_dim)
        self.edge_encoder = nn.Embedding(num_embeddings=8, embedding_dim=node_embed_dim)

        self.layers = nn.ModuleList()
        self.drop_out = min(drop_out, 0.3)
        self.use_bn = use_bn
        for _ in range(num_layers):
            mlp = MLP(
                in_channels=node_embed_dim,
                hidden_channels=node_embed_dim,
                out_channels=node_embed_dim,
                num_layers=2,
                norm=None,
                activation=activation,
            )
            self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=node_embed_dim))

        self.head_layers = MLP(
            in_channels=node_embed_dim,
            hidden_channels=node_embed_dim,
            out_channels=dim_out,
            num_layers=num_head_layers,
            norm=None,
            use_bn=False,
            dropout=0.0,
            activation=activation,
        )

        self.bn_node_x = nn.BatchNorm1d(node_embed_dim)
        self.activation = build_activation(activation)

    def forward(self, batch):
        try:
            z = self.input_encoder(batch.node_type, batch.node_attr)
            edge_attr = self.edge_encoder(batch.edge_type)

            for conv in self.layers:
                edge_index = batch.edge_index.to(z.device)
                z = conv(z, edge_index, edge_attr=edge_attr)

                if self.use_bn:
                    z = self.bn_node_x(z)
                z = self.activation(z)
                if self.drop_out > 0.0:
                    z = F.dropout(z, p=self.drop_out, training=self.training)

            batch_size = batch.batch_size if hasattr(batch, "batch_size") else batch.num_nodes
            node_emb = z[:batch_size, :]
            target_mask = getattr(
                batch,
                "target_node_mask",
                torch.ones(batch_size, dtype=torch.bool, device=z.device),
            )[:batch_size]
            node_emb = node_emb[target_mask]
            pred = self.head_layers(node_emb)

            node_labels = batch.y[:batch_size][target_mask]
            if node_labels.dim() == 2 and node_labels.size(1) == 2:
                if self.task == "classification":
                    y = node_labels[:, 1].long()
                else:
                    y = node_labels[:, 0]
            else:
                y = node_labels
            return pred, y

        except RuntimeError as e:
            print(f"Error in NodeHead forward pass: {str(e)}")
            print(
                f"Input shapes - node_type: {batch.node_type.shape}, "
                f"node_attr: {batch.node_attr.shape}, edge_index: {batch.edge_index.shape}"
            )
            raise
