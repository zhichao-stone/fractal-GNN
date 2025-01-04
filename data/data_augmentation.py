
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


def aug_drop_fractal_box(graph:dgl.DGLGraph, radius:int=2, drop_percent:float=0.2):
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
    return aug_graph



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


def renormalization_graph(graph:dgl.DGLGraph, radius: int, device: torch.device, min_edges: int=1) -> dgl.DGLGraph:
    if radius <= 0:
        g = dgl.graph(graph.edges(), num_nodes=graph.number_of_nodes())
        g.ndata["feat"] = graph.ndata["feat"]
        return g

    # center_nodes = graph.ndata["frac_cover_mat"][:, radius-1]
    # balls = {int(c):[] for c in set(center_nodes)}
    # for n, c in enumerate(center_nodes):
    #     balls[int(c)].append(n)
    # supernodes = list(balls.values())
    # supernodes_features = [torch.mean(torch.stack([graph.ndata["feat"][n] for n in ball]), dim=0) for ball in supernodes]

    # G = dgl.to_networkx(graph)
    # renormalization_graph = nx.Graph()
    # for i, supernode in enumerate(supernodes):
    #     renormalization_graph.add_node(i, members=supernode)
    # for i, supernode1 in enumerate(supernodes):
    #     for j, supernode2 in enumerate(supernodes):
    #         if i < j:
    #             if any(G.has_edge(u, v) for u in supernode1 for v in supernode2):
    #                 renormalization_graph.add_edge(i, j)
    
    # renormalization_graph = dgl.from_networkx(renormalization_graph)
    # renormalization_graph.ndata["feat"] = torch.stack(supernodes_features)

    num_nodes = graph.number_of_nodes()
    center_nodes = graph.ndata["frac_cover_mat"][:, radius-1]
    
    # calculate features of supernodes
    features = {int(c): [] for c in center_nodes}
    for n, c in enumerate(center_nodes):
        features[int(c)].append(graph.ndata["feat"][n])
    supernodes_idx_dict, supernodes_features = {}, []
    for i, (c, f) in enumerate(features.items()):
        supernodes_features.append(torch.mean(torch.stack(f), dim=0))
        supernodes_idx_dict[c] = i

    # calculate supernode edges
    num_supernodes = len(supernodes_idx_dict)
    for idx, c in enumerate(supernodes_idx_dict.keys()):
        supernodes_idx_dict[c] = idx
    supernodes = [supernodes_idx_dict[int(c)] for c in center_nodes]
    
    S = torch.sparse_coo_tensor(torch.tensor([supernodes, list(range(num_nodes))]), torch.ones(num_nodes), size=(num_supernodes, num_nodes)).to(device)
    Adj = torch.sparse_coo_tensor(torch.stack(graph.edges()), torch.ones(graph.number_of_edges()), size=(num_nodes, num_nodes)).to(device)

    A = torch.matmul(torch.matmul(S, Adj), S.T).to_dense()
    A = A - torch.diag_embed(A.diag()).to(device)
    renorm_edges = torch.where(A.cpu() >= min_edges)

    renormalization_graph = dgl.graph(renorm_edges, num_nodes=num_supernodes)
    renormalization_graph.ndata["feat"] = torch.stack(supernodes_features)

    return renormalization_graph


def renormalization_graph_random_center(
    graph:dgl.DGLGraph, 
    radius: int, 
    device: torch.device, 
    min_edges: int = 1
) -> dgl.DGLGraph:
    if radius <= 0:
        g = dgl.graph(graph.edges(), num_nodes=graph.number_of_nodes())
        g.ndata["feat"] = graph.ndata["feat"]
        return g
    
    num_nodes = graph.number_of_nodes()
    # cluster supernodes
    center_nodes = [0 for _ in range(num_nodes)]
    # F = dgl.to_networkx(graph)
    # visited = [False for _ in range(num_nodes)]
    # num_supernodes = 0
    # while len(F) > 0:
    #     c = random.choice(list(F.nodes))
    #     visited[c] = True
    #     # get neighbors
    #     balls = {c}
    #     neighbors, next_neighbors = {c}, set()
    #     for _ in range(1, radius+1):
    #         for neigh in neighbors:
    #             for n in F.neighbors(neigh):
    #                 if not visited[n]:
    #                     visited[n] = True
    #                     next_neighbors.add(n)
    #         neighbors = next_neighbors
    #         next_neighbors = set()
    #         balls.update(neighbors)
    #     for n in balls:
    #         center_nodes[n] = num_supernodes
    #     num_supernodes += 1
    #     F.remove_nodes_from(balls)
    Adj = torch.sparse_coo_tensor(torch.stack(graph.edges()), torch.ones(graph.number_of_edges()), size=(num_nodes, num_nodes)).to(device)
    N_Adj = torch.sparse_coo_tensor(torch.stack(graph.edges()), torch.ones(graph.number_of_edges()), size=(num_nodes, num_nodes)).to(device)
    for _ in range(1, radius):
        N_Adj = torch.matmul(N_Adj, Adj) + N_Adj
    N_Adj = N_Adj.to_dense()
    visited = [False for _ in range(num_nodes)]
    remaining_nodes = {i for i in range(num_nodes)}
    num_supernodes = 0
    while len(remaining_nodes) > 0:
        c = random.choice(remaining_nodes)
        visited[c] = True
        balls = {c}
        N_neighbors = torch.where(N_Adj[c]>0)[0]
        for n in N_neighbors:
            if not visited[n]:
                visited[n] = True
                balls.add(n)
        for n in balls:
            center_nodes[n] = num_supernodes

        remaining_nodes -= balls
        num_supernodes += 1

    # calculate features of supernodes
    features = [[] for _ in range(num_supernodes)]
    for n, c in enumerate(center_nodes):
        features[c].append(graph.ndata["feat"][n])
    supernodes_features = [torch.mean(torch.stack(f), dim=0) for f in features]

    # calculate supernode edges
    S = torch.sparse_coo_tensor(torch.tensor([center_nodes, list(range(num_nodes))]), torch.ones(num_nodes), size=(num_supernodes, num_nodes)).to(device)

    A = torch.matmul(torch.matmul(S, Adj), S.T).to_dense()
    A = A - torch.diag_embed(A.diag()).to(device)
    renorm_edges = torch.where(A.cpu() >= min_edges)

    renormalization_graph = dgl.graph(renorm_edges, num_nodes=num_supernodes)
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
    device: torch.device
):
    aug_graphs_1, aug_graphs_2 = [], []
    for i in range(len(graphs)):
        g, is_fractal, r2, diameter = graphs[i], is_fractals[i], float(fractal_attrs[i]), int(diameters[i])
        if aug_type == "renormalization":
            if is_fractal and r2 >= aug_fractal_threshold:
                radius_scales = list(range(1, max(1, diameter//4)+1))
                if len(radius_scales) >= 2:
                    radius_1, radius_2 = random.sample(radius_scales, 2)
                else:
                    radius_1, radius_2 = 0, radius_scales[0]
                aug_graphs_1.append(renormalization_graph(g, radius_1, device))
                aug_graphs_2.append(renormalization_graph(g, radius_2, device))
            else:
                aug_g1 = aug_drop_node(g, 0.2)
                g1 = dgl.graph(aug_g1.edges(), num_nodes=aug_g1.number_of_nodes())
                g1.ndata["feat"] = aug_g1.ndata["feat"]
                aug_graphs_1.append(g1)
                aug_g2 = aug_drop_node(g, 0.2)
                g2 = dgl.graph(aug_g2.edges(), num_nodes=aug_g2.number_of_nodes())
                g2.ndata["feat"] = aug_g2.ndata["feat"]
                aug_graphs_2.append(g2)
        
        elif aug_type == "renormalization_random_center":
            if is_fractal and r2 >= aug_fractal_threshold:
                radius = 1
                aug_graphs_1.append(renormalization_graph(g, radius, device))
                aug_graphs_2.append(renormalization_graph(g, radius, device))
            else:
                aug_g1 = aug_drop_node(g, 0.2)
                g1 = dgl.graph(aug_g1.edges(), num_nodes=aug_g1.number_of_nodes())
                g1.ndata["feat"] = aug_g1.ndata["feat"]
                aug_graphs_1.append(g1)
                aug_g2 = aug_drop_node(g, 0.2)
                g2 = dgl.graph(aug_g2.edges(), num_nodes=aug_g2.number_of_nodes())
                g2.ndata["feat"] = aug_g2.ndata["feat"]
                aug_graphs_2.append(g2)

        elif aug_type == "simple_random_walk":
            aug_graphs_1.append(simple_random_walk(g))
            aug_graphs_2.append(simple_random_walk(g))

        elif aug_type == "drop_fractal_box":
            if is_fractal and r2 >= aug_fractal_threshold:
                # radius = random.choice(list(range(1, max(1, diameter//4)+1)))
                radius = random.choice([1, 2])
                aug_graphs_1.append(aug_drop_fractal_box(g, radius, 0.2))
                aug_graphs_2.append(aug_drop_fractal_box(g, radius, 0.2))
            else:
                aug_graphs_1.append(aug_drop_node(g, 0.2))
                aug_graphs_2.append(aug_drop_node(g, 0.2))

        else:
            aug_graphs_1.append(aug_drop_node(g, 0.2))
            aug_graphs_2.append(aug_drop_node(g, 0.2))
        
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
