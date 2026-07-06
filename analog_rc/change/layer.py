"""
Custom GNN Layers: GatedGCN, GCNConv, GINEConv with optional FFN and batch norm.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_scatter import scatter


class GINEConvESLapPE(pyg_nn.conv.MessagePassing):
    """GINEConv Layer with EquivStableLapPE implementation.
    ICLR 2022 https://openreview.net/pdf?id=e95i1IHcWj
    """
    def __init__(self, nn_module, eps=0., train_eps=False, edge_dim=None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn_module
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if hasattr(self.nn[0], 'in_features'):
                in_channels = self.nn[0].in_features
            else:
                in_channels = self.nn[0].in_channels
            self.lin = pyg_nn.Linear(edge_dim, in_channels)
        else:
            self.lin = None
        self.reset_parameters()

        if hasattr(self.nn[0], 'in_features'):
            out_dim = self.nn[0].out_features
        else:
            out_dim = self.nn[0].out_channels

        self.mlp_r_ij = torch.nn.Sequential(
            torch.nn.Linear(1, out_dim), torch.nn.ReLU(),
            torch.nn.Linear(out_dim, 1),
            torch.nn.Sigmoid())

    def reset_parameters(self):
        pyg_nn.inits.reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()
        pyg_nn.inits.reset(self.mlp_r_ij)

    def forward(self, x, edge_index, edge_attr=None, pe_LapPE=None, size=None):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             PE=pe_LapPE, size=size)
        x_r = x[1] if isinstance(x, tuple) else x
        if x_r is not None:
            out += (1 + self.eps) * x_r
        return self.nn(out)

    def message(self, x_j, edge_attr, PE_i, PE_j):
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not match.")
        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
        r_ij = self.mlp_r_ij(r_ij)
        return ((x_j + edge_attr).relu()) * r_ij


class GatedGCNLayer(pyg_nn.conv.MessagePassing):
    """GatedGCN layer - Residual Gated Graph ConvNets
    https://arxiv.org/pdf/1711.07553.pdf
    """
    def __init__(self, in_dim, out_dim, dropout=0.0, residual=True, ffn=False, 
                 batch_norm=True, act='relu', equivstable_pe=False, **kwargs):
        super().__init__(**kwargs)
        self.activation = nn.ReLU if act == 'relu' else nn.PReLU
        self.A = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.B = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.C = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.D = pyg_nn.Linear(in_dim, out_dim, bias=True)
        self.E = pyg_nn.Linear(in_dim, out_dim, bias=True)

        self.EquivStablePE = equivstable_pe
        if self.EquivStablePE:
            self.mlp_r_ij = nn.Sequential(
                nn.Linear(1, out_dim),
                self.activation(),
                nn.Linear(out_dim, 1),
                nn.Sigmoid())

        self.act_fn_x = self.activation()
        self.act_fn_e = self.activation()
        self.dropout = dropout
        self.residual = residual
        self.e = None
        self.batch_norm = batch_norm
        self.ffn = ffn
        
        if self.batch_norm:
            self.bn_node_x = nn.BatchNorm1d(out_dim)
            self.bn_edge_e = nn.BatchNorm1d(out_dim)
            
        if self.ffn:
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(out_dim)
            self.ff_linear1 = nn.Linear(out_dim, out_dim*2)
            self.ff_linear2 = nn.Linear(out_dim*2, out_dim)
            self.act_fn_ff = self.activation()
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(out_dim)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, batch):
        x, e, edge_index = batch.x, batch.edge_attr, batch.edge_index
        if self.residual:
            x_in = x
            e_in = e

        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        pe_LapPE = batch.pe_EquivStableLapPE if self.EquivStablePE and hasattr(batch, 'pe_EquivStableLapPE') else None
        x, e = self.propagate(edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, e=e, Ax=Ax, PE=pe_LapPE)
        
        if self.batch_norm:
            x = self.bn_node_x(x)
            e = self.bn_edge_e(e)

        x = self.act_fn_x(x)
        e = self.act_fn_e(e)
        x = F.dropout(x, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        if self.residual:
            x = x_in + x
            e = e_in + e

        batch.x = x
        batch.edge_attr = e
        
        if self.ffn:
            if self.batch_norm:
                batch.x = self.norm1_local(batch.x)
            batch.x = batch.x + self._ff_block(batch.x)
            if self.batch_norm:
                batch.x = self.norm2(batch.x)

        return batch

    def message(self, Dx_i, Ex_j, PE_i, PE_j, Ce):
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)
        if self.EquivStablePE and PE_i is not None and PE_j is not None:
            r_ij = ((PE_i - PE_j) ** 2).sum(dim=-1, keepdim=True)
            r_ij = self.mlp_r_ij(r_ij)
            sigma_ij = sigma_ij * r_ij
        self.e = e_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Bx_j, Bx):
        dim_size = Bx.shape[0]
        sum_sigma_x = sigma_ij * Bx_j
        numerator_eta_xj = scatter(sum_sigma_x, index, 0, None, dim_size, reduce='sum')
        sum_sigma = sigma_ij
        denominator_eta_xj = scatter(sum_sigma, index, 0, None, dim_size, reduce='sum')
        out = numerator_eta_xj / (denominator_eta_xj + 1e-6)
        return out

    def update(self, aggr_out, Ax):
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out


class GCNConvLayer(nn.Module):
    """GCN Convolution Layer with optional FFN and batch norm."""
    def __init__(self, dim_in, dim_out, dropout=0.0, residual=True, ffn=False, batch_norm=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = batch_norm
        self.ffn = ffn
        
        if self.batch_norm:
            self.bn_node_x = nn.BatchNorm1d(dim_out)
        
        self.act = nn.Sequential(nn.ReLU(), nn.Dropout(self.dropout))
        self.model = pyg_nn.GCNConv(dim_in, dim_out, bias=True)
        
        if self.ffn:
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(dim_out)
            self.ff_linear1 = nn.Linear(dim_out, dim_out*2)
            self.ff_linear2 = nn.Linear(dim_out*2, dim_out)
            self.act_fn_ff = nn.ReLU()
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(dim_out)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)
        
    def _ff_block(self, x):
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def forward(self, batch):
        x_in = batch.x
        batch.x = self.model(batch.x, batch.edge_index)
        if self.batch_norm:
            batch.x = self.bn_node_x(batch.x)
        batch.x = self.act(batch.x)
        if self.residual:
            batch.x = x_in + batch.x
        if self.ffn:
            if self.batch_norm:
                batch.x = self.norm1_local(batch.x)
            batch.x = batch.x + self._ff_block(batch.x)
            if self.batch_norm:
                batch.x = self.norm2(batch.x)
        return batch


class GINEConvLayer(nn.Module):
    """GINE Convolution Layer with optional FFN and batch norm."""
    def __init__(self, dim_in, dim_out, dropout=0.0, residual=True, ffn=False, batch_norm=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = batch_norm
        self.ffn = ffn
        
        gin_nn = nn.Sequential(
            pyg_nn.Linear(dim_in, dim_out), nn.ReLU(),
            pyg_nn.Linear(dim_out, dim_out))
        self.model = pyg_nn.GINEConv(gin_nn)
        
        if self.ffn:
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(dim_out)
            self.ff_linear1 = nn.Linear(dim_out, dim_out*2)
            self.ff_linear2 = nn.Linear(dim_out*2, dim_out)
            self.act_fn_ff = nn.ReLU()
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(dim_out)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))
    
    def forward(self, batch):
        x_in = batch.x
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)
        if self.residual:
            batch.x = x_in + batch.x
        if self.ffn:
            if self.batch_norm:
                batch.x = self.norm1_local(batch.x)
            batch.x = batch.x + self._ff_block(batch.x)
            if self.batch_norm:
                batch.x = self.norm2(batch.x)
        return batch
