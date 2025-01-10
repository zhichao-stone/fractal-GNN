import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

from .layers import ProjectionHead


class ResGCN(nn.Module):
    def __init__(self, 
        input_dim: int, 
        num_classes: int, 
        hidden_dim: int = 128, 
        dropout: float = 0.0, 
        num_conv_layers: int = 3, 
        num_fc_layers: int = 2, 
        pooling_type: str = "sum", 
        norm_type: str = "both", 
        residual: bool = False, 
        batch_norm: bool = True, 
        head: bool = True, 
        mlp: bool = False
    ) -> None:
        super(ResGCN, self).__init__()

        self.pooling_type = pooling_type
        if self.pooling_type == 'sum':
            self.pool = SumPooling()
        elif self.pooling_type == 'mean':
            self.pool = AvgPooling()
        elif self.pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

        self.conv_residual = residual
        self.batch_norm = batch_norm
        self.head = head
        self.mlp = mlp

        self.dropout = dropout

        self.feat_fc = nn.Linear(input_dim, hidden_dim)
        
        self.num_conv_layers = num_conv_layers
        self.conv_bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for _ in range(self.num_conv_layers):
            self.conv_bns.append(nn.BatchNorm1d(hidden_dim))
            self.convs.append(GraphConv(
                in_feats=hidden_dim, 
                out_feats=hidden_dim, 
                norm=norm_type, 
                allow_zero_in_degree=True
            ))

        self.num_fc_layers = num_fc_layers
        self.fc_bns = nn.ModuleList()
        self.linears = nn.ModuleList()
        for _ in range(self.num_fc_layers - 1):
            self.fc_bns.append(nn.BatchNorm1d(hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))

        self.linear_class = nn.Linear(hidden_dim, num_classes)
        self.proj_head = ProjectionHead(dim=hidden_dim)

        self.bn_feat = nn.BatchNorm1d(input_dim)
        self.bn_hidden = nn.BatchNorm1d(hidden_dim)

    def forward(self, 
        graph: dgl.DGLGraph, 
        h: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor: 
        with graph.local_scope():
            h = self.bn_feat(h)
            h = F.relu(self.feat_fc(h))

            for i in range(self.num_conv_layers):
                h_ = self.conv_bns[i](h)
                h_ = F.relu(self.convs[i](graph, h_))
                h = h + h_ if self.conv_residual else h_
            
            h = self.pool(graph, h)

            for i, fc in enumerate(self.linears):
                h = self.fc_bns[i](h)
                h = F.relu(fc(h))
            h = self.bn_hidden(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.mlp:
                return self.linear_class(h)
            else:
                if self.head:
                    return self.proj_head(h)
                else:
                    return h

