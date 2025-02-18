
'''
functions for data augmentation
'''
import warnings
warnings.filterwarnings("ignore")

import random
import torch
import dgl
import networkx as nx
import numpy as np

import math
import copy
from typing import List



def aug_drop_node(graph: dgl.DGLGraph, drop_percent: float = 0.2, weighted: bool = False):
    # aug_graph = copy.deepcopy(graph)
    num = graph.number_of_nodes()
    aug_graph = dgl.graph(graph.edges(), num_nodes=num)
    aug_graph.ndata["feat"] = graph.ndata["feat"]
    drop_num = int(num * drop_percent)
    all_node_list = [i for i in range(num)]
    drop_node_list = random.sample(all_node_list, drop_num)
    aug_graph.remove_nodes(drop_node_list)
    if weighted:
        aug_graph.edata["e"] = torch.ones((aug_graph.number_of_edges(), 1), dtype=torch.float)
    return aug_graph


def aug_drop_fractal_box(graph: dgl.DGLGraph, radius: int = 2, drop_percent: float = 0.2, weighted: bool = False):
    node_num = graph.number_of_nodes()

    center_nodes = graph.ndata["frac_cover_mat"][:, radius-1]
    balls = {int(c):[] for c in set(center_nodes)}
    for n, c in enumerate(center_nodes):
        balls[int(c)].append(n)
    boxes = sorted(list(balls.values()), key=lambda x:len(x), reverse=True)[1:]
    random.shuffle(boxes)

    min_drop_num = int(node_num * drop_percent)
    drop_nodes = []

    for b in boxes:
        drop_nodes += b
        if len(drop_nodes) >= min_drop_num:
            break
    if len(drop_nodes) >= 0.4*node_num or len(drop_nodes) < 0.05*node_num:
        drop_nodes = random.sample(list(range(node_num)), min_drop_num)
    
    aug_graph = copy.deepcopy(graph)
    aug_graph.remove_nodes(drop_nodes)
    if weighted:
        aug_graph.edata["e"] = torch.ones((aug_graph.number_of_edges(), 1), dtype=torch.float)
    return aug_graph



def simple_random_walk(graph: dgl.DGLGraph, weighted: bool = False) -> dgl.DGLGraph:
    G = dgl.to_networkx(graph)
    num_nodes = len(G.nodes)

    start_node = random.choice(list(G.nodes))
    visited_nodes, visited_edges = set([start_node]), set()

    current_node = start_node
    for _ in range(num_nodes):
        neighbors = list(G.neighbors(current_node))
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        visited_nodes.add(next_node)
        visited_edges.add((current_node, next_node))
        current_node = next_node

    subgraph = nx.Graph()
    subgraph.add_nodes_from(visited_nodes)
    subgraph.add_edges_from(visited_edges)
    subgraph = dgl.from_networkx(subgraph)
    subgraph.ndata["feat"] = torch.stack([graph.ndata["feat"][n] for n in visited_nodes])
    if weighted:
        subgraph.edata["e"] = torch.ones((subgraph.number_of_edges(), 1), dtype=torch.float)

    return subgraph


def renormalization_graph(
    graph:dgl.DGLGraph, 
    radius: int, 
    device: torch.device, 
    min_edges: int = 1, 
    weighted: bool = False
) -> dgl.DGLGraph:

    if radius <= 0:
        g = dgl.graph(graph.edges(), num_nodes=graph.number_of_nodes())
        g.ndata["feat"] = graph.ndata["feat"]
        return g

    num_nodes = graph.number_of_nodes()
    center_nodes: torch.Tensor = graph.ndata["frac_cover_mat"][:, radius-1]
    
    supernodes_idx_dict = {}
    for i, c in enumerate(set(center_nodes.tolist())):
        supernodes_idx_dict[int(c)] = i
    num_supernodes = len(supernodes_idx_dict)
    
    supernodes_features = torch.zeros((num_supernodes, graph.ndata["feat"].size(-1)))
    for n, c in enumerate(center_nodes):
        supernodes_features[supernodes_idx_dict[int(c)]] += graph.ndata["feat"][n]

    # calculate supernode edges
    supernodes = [supernodes_idx_dict[int(c)] for c in center_nodes]
    
    S = torch.sparse_coo_tensor(torch.tensor([supernodes, list(range(num_nodes))]), torch.ones(num_nodes), size=(num_supernodes, num_nodes)).to(device)
    Adj = torch.sparse_coo_tensor(torch.stack(graph.edges()), torch.ones(graph.number_of_edges()), size=(num_nodes, num_nodes)).to(device)

    A = torch.matmul(torch.matmul(S, Adj), S.T).to_dense()
    A = A - torch.diag_embed(A.diag()).to(device)
    renorm_edges = torch.where(A.cpu() >= min_edges)

    renorm_graph = dgl.graph(renorm_edges, num_nodes=num_supernodes)
    renorm_graph.ndata["feat"] = supernodes_features.to(renorm_graph.device)

    if weighted:
        renorm_graph.edata["e"] = torch.log(A[renorm_edges].unsqueeze(-1) + 1).to(renorm_graph.device)

    return renorm_graph


def renormalization_graph_random_center(
    graph:dgl.DGLGraph, 
    radius: int, 
    device: torch.device, 
    min_edges: int = 1, 
    weighted: bool = False
) -> dgl.DGLGraph:

    if radius <= 0:
        g = dgl.graph(graph.edges(), num_nodes=graph.number_of_nodes())
        g.ndata["feat"] = graph.ndata["feat"]
        return g
    
    num_nodes = graph.number_of_nodes()

    # cluster supernodes
    center_nodes = np.zeros(num_nodes, dtype=int)

    edges = torch.stack(graph.edges())
    values = torch.ones(graph.number_of_edges())
    Adj = torch.sparse_coo_tensor(edges, values, size=(num_nodes, num_nodes)).to(device)
    # N_Adj = torch.sparse_coo_tensor(edges, values, size=(num_nodes, num_nodes)).to(device)
    N_Adj = Adj.clone().to(device)
    for _ in range(1, radius):
        N_Adj = torch.matmul(N_Adj, Adj) + N_Adj
    N_Adj = N_Adj.to_dense()
    if radius > 1:
        # N_Adj = N_Adj.masked_fill(torch.eye(num_nodes, dtype=torch.bool).to(device), 0)
        N_Adj = N_Adj - torch.diag_embed(N_Adj.diag()).to(device)

    all_nodes = list(range(num_nodes))
    remaining_nodes = set(all_nodes)
    num_supernodes = 0
    # supernode_size = []

    while len(remaining_nodes) > 0:
        c = random.choice(list(remaining_nodes))

        balls = torch.where(N_Adj[c] > 0)[0].cpu().tolist()
        balls.append(c)
        center_nodes[balls] = num_supernodes

        remaining_nodes = remaining_nodes.difference(balls)

        # supernode_size.append(len(balls))
        num_supernodes += 1

        N_Adj[balls, :] = 0
        N_Adj[:, balls] = 0

    # calculate features of supernodes
    supernodes_features = torch.zeros((num_supernodes, graph.ndata["feat"].size(-1)))
    for n, c in enumerate(center_nodes):
        # supernodes_features[c] += graph.ndata["feat"][n] / supernode_size[c]
        supernodes_features[c] += graph.ndata["feat"][n]

    # calculate supernode edges
    S = torch.sparse_coo_tensor(torch.tensor([center_nodes.tolist(), all_nodes]), torch.ones(num_nodes), size=(num_supernodes, num_nodes)).to(device)
    A = torch.matmul(torch.matmul(S, Adj), S.T).to_dense()
    A = A - torch.diag_embed(A.diag()).to(device)
    renorm_edges = torch.where(A.cpu() >= min_edges)

    renorm_graph = dgl.graph(renorm_edges, num_nodes=num_supernodes)
    renorm_graph.ndata["feat"] = supernodes_features.to(renorm_graph.device)
    if weighted:
        renorm_graph.edata["e"] = torch.log(A[renorm_edges].unsqueeze(-1) + 1).to(renorm_graph.device)

    return renorm_graph


def collate_batched_graph(graphs:List[dgl.DGLGraph]):
    tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
    tab_snorm_n = [torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n]
    snorm_n = torch.cat(tab_snorm_n).sqrt()

    tab_sizes_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
    while 0 in tab_sizes_e:
        tab_sizes_e[tab_sizes_e.index(0)] = 1
    tab_snorm_e = [torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e]
    snorm_e = torch.cat(tab_snorm_e).sqrt()

    batched_graph = dgl.batch(graphs)
    return batched_graph, snorm_n, snorm_e



class DataAugmentator:
    def __init__(self, 
        drop_ratio: float = 0.2, 
        aug_fractal_threshold: float = 0.95, 
        renorm_min_edges: int = 1, 
        weighted: bool = False, 
        device: torch.device = torch.device("cuda")
    ) -> None:
        self.drop_ratio = drop_ratio
        self.aug_fractal_threshold = aug_fractal_threshold
        self.renorm_min_edges = renorm_min_edges
        self.weighted = weighted
        self.device = device

        self.non_fractal_aug_types = ["drop_node", "simple_random_walk"]

    def augment_graphs(self, 
        graphs: List[dgl.DGLGraph], 
        is_fractals: List[bool], 
        fractal_attrs: List[float], 
        diameters: List[int], 
        aug_type: str
    ):
        aug_graphs_1: List[dgl.DGLGraph] = []
        aug_graphs_2: List[dgl.DGLGraph] = []
        for i in range(len(graphs)):
            g, is_fractal, r2, diameter = graphs[i], is_fractals[i], fractal_attrs[i], diameters[i]
            if aug_type not in self.non_fractal_aug_types:
                if is_fractal and r2 >= self.aug_fractal_threshold:
                    if aug_type == "renormalization_random_center":
                        radius = 1
                        aug_graphs_1.append(renormalization_graph_random_center(g, radius, self.device, self.renorm_min_edges, self.weighted))
                        aug_graphs_2.append(renormalization_graph_random_center(g, radius, self.device, self.renorm_min_edges, self.weighted))
                    elif aug_type == "renormalization_rc_rr":
                        radius_scales = list(range(1, max(2, int(math.log2(diameter/2)))))
                        radius = random.choice(radius_scales)
                        aug_graphs_1.append(renormalization_graph_random_center(g, radius, self.device, self.renorm_min_edges, self.weighted))
                        aug_graphs_2.append(renormalization_graph_random_center(g, radius, self.device, self.renorm_min_edges, self.weighted))
                    elif aug_type == "renormalization_rc_r2prob":
                        if random.random() < r2:
                            radius = 1
                            aug_graphs_1.append(renormalization_graph_random_center(g, radius, self.device, self.renorm_min_edges, self.weighted))
                            aug_graphs_2.append(renormalization_graph_random_center(g, radius, self.device, self.renorm_min_edges, self.weighted))
                        else:
                            aug_graphs_1.append(aug_drop_fractal_box(g, radius, self.drop_ratio, self.weighted))
                            aug_graphs_2.append(aug_drop_fractal_box(g, radius, self.drop_ratio, self.weighted))
                    elif aug_type == "mix":
                        radius = 1
                        if random.random() < 0.5:
                            aug_graphs_1.append(renormalization_graph_random_center(g, radius, self.device, self.renorm_min_edges, self.weighted))
                        else:
                            aug_graphs_1.append(aug_drop_fractal_box(g, radius, self.drop_ratio, self.weighted))
                        if random.random() < 0.5:
                            aug_graphs_2.append(renormalization_graph_random_center(g, radius, self.device, self.renorm_min_edges, self.weighted))
                        else:
                            aug_graphs_2.append(aug_drop_fractal_box(g, radius, self.drop_ratio, self.weighted))
                    elif aug_type == "mix_sep":
                        radius = 1
                        if random.random() < 0.5:
                            aug_graphs_1.append(renormalization_graph_random_center(g, radius, self.device, self.renorm_min_edges, self.weighted))
                            aug_graphs_2.append(aug_drop_node(g, self.drop_ratio, self.weighted))
                        else:
                            aug_graphs_1.append(aug_drop_node(g, self.drop_ratio, self.weighted))
                            aug_graphs_2.append(renormalization_graph_random_center(g, radius, self.device, self.renorm_min_edges, self.weighted))
                    elif aug_type == "renorm_drop":
                        radius_scales = list(range(1, max(2, int(math.log2(diameter/2)))))
                        radius = random.choice(radius_scales)
                        aug_g1 = renormalization_graph(g, radius, self.device, self.renorm_min_edges, self.weighted)
                        aug_graphs_1.append(aug_drop_node(aug_g1, self.drop_ratio, self.weighted))
                        aug_g2 = renormalization_graph(g, radius, self.device, self.renorm_min_edges, self.weighted)
                        aug_graphs_2.append(aug_drop_node(aug_g2, self.drop_ratio, self.weighted))
                    elif aug_type == "drop_fractal_box":
                        radius_scales = list(range(1, max(2, int(math.log2(diameter/2)))))
                        radius = random.choice(radius_scales)
                        aug_graphs_1.append(aug_drop_fractal_box(g, radius, self.drop_ratio, self.weighted))
                        aug_graphs_2.append(aug_drop_fractal_box(g, radius, self.drop_ratio, self.weighted))
                    else:
                        raise NotImplementedError(f"Augmentation method {aug_type} is not supported!")
                else:
                    # default method is drop_node
                    aug_graphs_1.append(aug_drop_node(g, self.drop_ratio, self.weighted))
                    aug_graphs_2.append(aug_drop_node(g, self.drop_ratio, self.weighted))
            else:
                if aug_type == "drop_node":
                    aug_graphs_1.append(aug_drop_node(g, self.drop_ratio, self.weighted))
                    aug_graphs_2.append(aug_drop_node(g, self.drop_ratio, self.weighted))
                elif aug_type == "simple_random_walk":
                    aug_graphs_1.append(simple_random_walk(g, self.weighted))
                    aug_graphs_2.append(simple_random_walk(g, self.weighted))
                else:
                    raise NotImplementedError(f"Augmentation method {aug_type} is not supported!")
        
        return aug_graphs_1, aug_graphs_2