import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ResGatedGraphConv, GINConv, ChebConv, GINEConv, ClusterGCNConv, SSGConv
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.data import Data

NET = 0
DEV = 1
PIN = 2

class GraphHead(nn.Module):
    """ GNN head for graph-level prediction.

    Implementation adapted from the transductive GraphGPS.

    Args:
        hidden_dim (int): Hidden features' dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        num_layers (int): Number of layers of GNN model
        layers_post_mp (int): number of layers of head MLP
        use_bn (bool): whether to use batch normalization
        drop_out (float): dropout rate
        activation (str): activation function
        src_dst_agg (str): the way to aggregate src and dst nodes, which can be 'concat' or 'add' or 'pool'
        task_level (str): 'node' or 'edge'
    """
    def __init__(self, hidden_dim, dim_out, num_layers=2, num_head_layers=2, 
                 use_bn=False, drop_out=0.0, activation='relu', 
                 src_dst_agg='concat',  max_dist=400, 
                 task='classification', task_level='edge'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task = task
        self.task_level = task_level
        
        # 减小节点嵌入维度
        node_embed_dim = min(hidden_dim, 64)  # 限制最大嵌入维度

        ## Node / Edge type encoders
        self.node_encoder = nn.Embedding(num_embeddings=4,
                                         embedding_dim=node_embed_dim)
        self.edge_encoder = nn.Embedding(num_embeddings=10,
                                         embedding_dim=node_embed_dim)
        
        # GNN layers - 使用更轻量级的配置
        self.layers = nn.ModuleList()
        self.drop_out = min(drop_out, 0.3)  # 限制最大dropout
        self.use_bn = use_bn
        for _ in range(num_layers):
            self.layers.append(SAGEConv(node_embed_dim, node_embed_dim, 
                                      normalize=True,  # 添加归一化
                                      bias=True))
        
        ## The head configuration
        head_input_dim = node_embed_dim
        self.src_dst_agg = src_dst_agg
        
        if task_level == 'edge':
            head_input_dim = node_embed_dim * 2 if src_dst_agg == 'concat' else node_embed_dim
        else:
            head_input_dim = node_embed_dim
            
        if src_dst_agg == 'pool':
            self.pooling_fun = pygnn.pool.global_mean_pool

        self.num_head_layers = num_head_layers

        # head MLP layers
        self.head_layers = MLP(
            in_channels=head_input_dim, 
            hidden_channels=node_embed_dim, 
            out_channels=dim_out, 
            num_layers=num_head_layers, 
            use_bn=False, dropout=0.0, activation=activation
        )
        self.bn_node_x = nn.BatchNorm1d(node_embed_dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation')

    def forward(self, batch):
        try:
            # 原有的 forward 逻辑
            z = self.node_encoder(batch.x[:, 0])

            for conv in self.layers:
                # 确保边索引在正确的设备上
                edge_index = batch.edge_index.to(z.device)
                z = conv(z, edge_index)

                if self.use_bn:
                    z = self.bn_node_x(z)
                z = self.activation(z)
                if self.drop_out > 0.0:
                    z = F.dropout(z, p=self.drop_out, training=self.training)

            if self.task_level == 'node':
                # 节点级任务: NeighborLoader 中前 batch_size 个节点是目标节点
                # 使用 input_id 或 n_id 来确定 batch_size
                if hasattr(batch, 'input_id') and batch.input_id is not None:
                    batch_size = batch.input_id.size(0)
                elif hasattr(batch, 'batch') and batch.batch is not None:
                    batch_size = batch.batch.max().item() + 1
                else:
                    batch_size = batch.y.size(0)
                # 获取目标节点的嵌入 (前batch_size个)
                node_emb = z[:batch_size, :]
                pred = self.head_layers(node_emb)
                return pred, batch.y[:batch_size]
            else:
                # 边级任务
                if self.src_dst_agg == 'pool':
                    graph_emb = self.pooling_fun(z, batch.batch)
                else:
                    batch_size = batch.edge_label.size(0)
                    src_emb = z[:batch_size, :]
                    dst_emb = z[batch_size:batch_size*2, :]
                    if self.src_dst_agg == 'concat':
                        graph_emb = torch.cat((src_emb, dst_emb), dim=1)
                    else:
                        graph_emb = src_emb + dst_emb

                pred = self.head_layers(graph_emb)
                return pred, batch.edge_label
            
        except RuntimeError as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shapes - x: {batch.x.shape}, edge_index: {batch.edge_index.shape}")
            raise

