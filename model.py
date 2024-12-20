import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import DGLGraph
import dgl.function as fn


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)
        self.attn = nn.Linear(in_features=2*out_dim, out_features=1, bias=False)

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn(z2)
        a = F.leaky_relu(a)
        return {"e": a}

    def message_func(self, edges):
        return {"z": edges.src["z"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    def forward(self, graph:DGLGraph, h):
        with graph.local_scope():
            z = self.fc(h)
            graph.ndata["z"] = z
            graph.apply_edges(self.edge_attention)
            graph.update_all(self.message_func, self.reduce_func)
            return graph.ndata.pop("h")


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1, merge="cat") -> None:
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, graph:DGLGraph, h):
        head_out = [attn_head(graph, h) for attn_head in self.heads]
        if self.merge == "cat":
            return torch.cat(head_out, dim=1)
        else:
            return torch.mean(torch.stack(head_out))


class FractalLayer(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, scales:list=[1,2,3], concat_type:str="concat") -> None:
        super(FractalLayer, self).__init__()
        self.scales = scales
        self.dropout = nn.Dropout(p=0.6)
        self.frac_fcs = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in scales
        ])
        self.concat_type = concat_type
        if concat_type == "concat":
            self.final_fc = nn.Linear(out_dim*len(scales), out_dim, bias=False)
        else:
            self.final_fc = nn.Linear(out_dim, out_dim, bias=False)

    def forward(self, graph:DGLGraph, h):
        h_all = []
        covering_matrix = graph.ndata["frac_cover_mat"]

        for s in self.scales:
            center_nodes = covering_matrix[:, s-1]
            h_frac = self.frac_fcs[s-1](h[center_nodes, :])
            h_all.append(h_frac)
        
        if self.concat_type == "concat":
            h_final = torch.cat(h_all, dim=-1)
        else:
            h_final = torch.sum(torch.stack(h_all, dim=-1), dim=-1)
        h_final = self.dropout(self.final_fc(h_final))
        return h_final


### Node Representation
class GAT(nn.Module):
    def __init__(self,  
        in_dim:int, 
        hidden_dim:int, 
        out_dim:int,
        num_heads:int=1,
        scales:list=[1, 2, 3],
        multiview:bool=False,
        fractal:bool=True,
        fractal_concat:str="concat"
    ) -> None:
        super(GAT, self).__init__()
        self.multiview = multiview
        self.fractal = fractal

        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads=num_heads)
        if multiview or fractal:
            self.layer2 = MultiHeadGATLayer(hidden_dim*num_heads, hidden_dim, num_heads=1)
        else:
            self.layer2 = MultiHeadGATLayer(hidden_dim*num_heads, out_dim, num_heads=1)      # base

        if fractal:
            self.frac_layer = FractalLayer(in_dim, hidden_dim, scales=scales, concat_type=fractal_concat)

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)

        if multiview and fractal:
            self.fc2 = nn.Linear(hidden_dim*3, out_dim, bias=False)
        elif multiview or fractal:
            self.fc2 = nn.Linear(hidden_dim*2, out_dim, bias=False)
        
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, graph:DGLGraph, h):
        # GAT embedding
        h_nei = self.layer1(graph, h)
        h_nei = F.elu(h_nei)
        h_nei = self.layer2(graph, h_nei)

        ### concatenate base embedding and fractal embedding
        if self.fractal:
            h_frac = self.frac_layer(graph, h)

        if self.multiview:
            h0 = self.dropout(self.fc1(h))
            if self.fractal:
                h_global = torch.cat([h0, h_nei, h_frac], dim=-1)
            else:
                h_global = torch.cat([h0, h_nei], dim=-1)
        else:
            if self.fractal:
                h_global = torch.cat([h_nei, h_frac], dim=-1)
            else:
                h_global = h_nei
        
        if self.multiview or self.fractal:
            h_global = self.dropout(self.fc2(h_global))
        
        return h_global



### link prediction
class DotProductPredictor(nn.Module):
    def forward(self, graph:DGLGraph, h):
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            return graph.edata["score"]


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes) -> None:
        super(MLPPredictor, self).__init__()
        self.W = nn.Linear(in_features*2, out_classes)
    
    def edge_score(self, edges):
        h_u, h_v = edges.src["h"], edges.dst["h"]
        score = self.W(torch.cat([h_u, h_v], 1))
        return {"score": score}
    
    def forward(self, graph:DGLGraph, h):
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(self.edge_score)
            return graph.edata["score"]


class LinkPredModel(nn.Module):
    def __init__(self, 
        in_dim:int, 
        hidden_dim:int, 
        out_dim:int,
        num_heads:int=1,
        scales:list=[1, 2, 3],
        multiview:bool=False,
        fractal:bool=True,
        fractal_concat:str="concat"
    ) -> None:
        super(LinkPredModel, self).__init__()

        self.gat = GAT(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            scales=scales,
            multiview=multiview,
            fractal=fractal,
            fractal_concat=fractal_concat
        )
        
        # self.pred = MLPPredictor(hidden_dim, out_dim)
        self.pred = DotProductPredictor()

    def forward(self, graph:DGLGraph, neg_graph:DGLGraph, h):
        h = self.gat(graph, h)
        return self.pred(graph, h), self.pred(neg_graph, h)
