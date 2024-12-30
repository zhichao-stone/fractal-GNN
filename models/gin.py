import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

from .layers import MLP, ProjectionHead


class GINLayer(nn.Module):
    """
    [!] code adapted from dgl implementation of GINConv

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggr_type :
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    out_dim :
        Rquired for batch norm layer; should match out_dim of apply_func if not None.
    dropout :
        Required for dropout of output features.
    graph_norm : 
        boolean flag for output features normalization w.r.t. graph sizes.
    batch_norm :
        boolean flag for batch_norm layer.
    residual :
        boolean flag for using residual connection.
    init_eps : optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter.
    
    """
    def __init__(self, 
        apply_func: MLP, 
        aggr_type: str, 
        dropout: float, 
        graph_norm: bool, 
        batch_norm: bool, 
        residual: bool = False, 
        init_eps: float = 0, 
        learn_eps: bool = False
    ) -> None:
        super(GINLayer, self).__init__()

        self.apply_func = apply_func
        
        if aggr_type == 'sum':
            self._reducer = fn.sum
        elif aggr_type == 'max':
            self._reducer = fn.max
        elif aggr_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggr_type))
            
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout
        
        in_dim = apply_func.input_dim
        out_dim = apply_func.output_dim
        
        if in_dim != out_dim:
            self.residual = False
            
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))
            
        self.bn_node_h = nn.BatchNorm1d(out_dim)

    def forward(self, g:dgl.DGLGraph, h:torch.Tensor, snorm_n:torch.Tensor):
        with g.local_scope():
            h_in = h

            g.ndata["h"] = h
            g.update_all(fn.copy_u("h", "m"), self._reducer("m", "neigh"))
            h = (1 + self.eps) * h + g.ndata["neigh"]

            if self.apply_func is not None:
                h = self.apply_func(h)
            if self.graph_norm:
                h = h * snorm_n
            if self.batch_norm:
                h = self.bn_node_h(h)

            h = F.relu(h)
            if self.residual:
                h = h_in + h
            h = F.dropout(h, self.dropout, training=self.training)

            return h



class GIN(nn.Module):
    def __init__(self, 
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.0, 
        num_layers: int = 4,
        mlp_num_layers: int = 2,
        neighbor_aggr_type: str = "sum",
        pooling_type: str = "sum",
        learn_eps: bool = True,
        graph_norm: bool = True,
        batch_norm: bool = True,
        residual: bool = True
    ) -> None:
        super(GIN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.num_layers = num_layers
        self.neighbor_aggr_type = neighbor_aggr_type
        self.pooling_type = pooling_type

        self.learn_eps = learn_eps
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        

        self.gin_layers = nn.ModuleList()
        self.node_embeddings = nn.Linear(input_dim, hidden_dim)
        for _ in range(self.num_layers):
            self.gin_layers.append(GINLayer(
                apply_func=MLP(mlp_num_layers, hidden_dim, hidden_dim, hidden_dim),
                aggr_type=self.neighbor_aggr_type, 
                dropout=self.dropout, 
                graph_norm=self.graph_norm, 
                batch_norm=self.batch_norm, 
                residual=self.residual, 
                init_eps=0, 
                learn_eps=self.learn_eps
            ))

        self.linears = nn.ModuleList()
        for _ in range(self.num_layers + 1):
            self.linears.append(nn.Linear(hidden_dim, self.num_classes))

        self.proj_head = ProjectionHead(dim=self.hidden_dim)

        if self.pooling_type == 'sum':
            self.pool = SumPooling()
        elif self.pooling_type == 'mean':
            self.pool = AvgPooling()
        elif self.pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, 
        graph: dgl.DGLGraph, 
        h: torch.Tensor, 
        snorm_n: torch.Tensor, 
        mlp: bool = True, 
        head: bool = False
    ):
        with graph.local_scope():
            h = self.node_embeddings(h)
            hidden_rep = [h]

            for i in range(self.num_layers):
                h = self.gin_layers[i](graph, h, snorm_n)
                hidden_rep.append(h)

            score_over_layer, vector_over_layer = 0, 0
            for i, h in enumerate(hidden_rep):
                pooled_h = self.pool(graph, h)
                vector_over_layer += pooled_h
                score_over_layer += self.linears[i](pooled_h)

            if mlp:
                return score_over_layer
            else:
                if head:
                    return self.proj_head(vector_over_layer)
                else:
                    return vector_over_layer

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss