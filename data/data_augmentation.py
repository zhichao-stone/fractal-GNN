
'''
functions for data augmentation
'''
import warnings
warnings.filterwarnings("ignore")

import random
import torch
import dgl
import networkx as nx

import copy
from typing import List



def aug_drop_node(graph:dgl.DGLGraph, drop_percent:float=0.2):
    aug_graph = copy.deepcopy(graph)
    num = graph.number_of_nodes()
    drop_num = int(num * drop_percent)
    all_node_list = [i for i in range(num)]
    drop_node_list = random.sample(all_node_list, drop_num)
    aug_graph.remove_nodes(drop_node_list)
    return aug_graph


def covering_subgraph(graph:dgl.DGLGraph, radius:int) -> List[dgl.DGLGraph]:
    G = dgl.to_networkx(graph)
    ### adjust radius based on the diameter of G
    largest_cc = max(nx.connected_components(nx.to_undirected(G.copy())), key=len)
    diameter = nx.diameter(G.subgraph(largest_cc))
    radius = min(radius, diameter//2)
    ### get subgraphs
    subgraphs = []
    if radius > 2:
        F = G.copy()
        while F.nodes:
            center_node = max(F.degree, key=lambda x:x[1])[0]
            subgraph_nodes = list(nx.single_source_shortest_path_length(F, center_node, cutoff=radius).keys())
            subgraph = F.subgraph(subgraph_nodes)
            subgraphs.append(dgl.from_networkx(subgraph))
    return subgraphs


def simple_random_walk(graph:dgl.DGLGraph) -> dgl.DGLGraph:
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

    return subgraph


def renormalization_graph(graph:dgl.DGLGraph, radius:int=2) -> dgl.DGLGraph:
    if radius <= 0:
        return copy.deepcopy(graph)

    G = dgl.to_networkx(graph)
    remaining_nodes = set(G.nodes)
    supernodes = []
    supernodes_features = []
    while remaining_nodes:
        max_degree_node = max(remaining_nodes, key=lambda node:G.degree(node))
        ball = {node for node in remaining_nodes if nx.shortest_path_length(G, max_degree_node, node) <= radius}
        supernodes.append(ball)
        supernodes_features.append(torch.mean(torch.stack([graph.ndata["feat"][n] for n in ball]), dim=0))
        remaining_nodes -= ball

    renormalization_graph = nx.Graph()
    for i, supernode in enumerate(supernodes):
        renormalization_graph.add_node(i, members=supernode)
    for i, supernode1 in enumerate(supernodes):
        for j, supernode2 in enumerate(supernodes):
            if i < j:
                if any(G.has_edge(u, v) for u in supernode1 for v in supernode2):
                    renormalization_graph.add_edge(i, j)
    
    renormalization_graph = dgl.from_networkx(renormalization_graph)
    renormalization_graph.ndata["feat"] = torch.stack(supernodes_features)
    return renormalization_graph


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


def aug_renormalization_graphs(
    graphs: List[dgl.DGLGraph], 
    is_fractals: List[bool], 
    fractal_attrs: List[float], 
    diameters: List[int], 
    aug_type:str, 
    aug_fractal_threshold: float, 
):
    aug_graphs_1, aug_graphs_2 = [], []
    for i in range(len(graphs)):
        g, is_fractal, r2, diameter = graphs[i], is_fractals[i], float(fractal_attrs[i]), int(diameters[i])
        if aug_type == "renormalization" and is_fractal and r2 >= aug_fractal_threshold:
            radius_scales = list(range(1, max(1, diameter//4)))
            if len(radius_scales) >= 2:
                radius_1, radius_2 = random.sample(radius_scales, 2)
            else:
                radius_1, radius_2 = 0, radius_scales[0]
            aug_graphs_1.append(renormalization_graph(g, radius_1))
            aug_graphs_2.append(renormalization_graph(g, radius_2))
        elif aug_type == "simple random walk":
            aug_graphs_1.append(simple_random_walk(g))
            aug_graphs_2.append(simple_random_walk(g))
        else:
            aug_graphs_1.append(aug_drop_node(g))
            aug_graphs_2.append(aug_drop_node(g))
        
    return aug_graphs_1, aug_graphs_2


def sim_matrix2(ori_vector:torch.Tensor, arg_vector:torch.Tensor, temperature=0.5):
    for i in range(len(ori_vector)):
        sim = torch.cosine_similarity(ori_vector[i].unsqueeze(0), arg_vector, dim=1) * (1/temperature)
        if i == 0:
            sim_tensor = sim.unsqueeze(0)
        else:
            sim_tensor = torch.cat((sim_tensor, sim.unsqueeze(0)), dim=0)
    return sim_tensor


def compute_diag_sum(tensor:torch.Tensor):
    num = len(tensor)
    diag_sum = 0
    for i in range(num):
        diag_sum += tensor[i][i]
    return diag_sum
