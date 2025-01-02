import dgl
import torch
import networkx as nx

from typing import List


# functions for fractal graph
def get_fractal_covering_vectors(G:dgl.DGLGraph, scale:int):
    F = dgl.to_networkx(G)
    covering_vectors = [0 for _ in F.nodes]

    largest_cc = max(nx.connected_components(nx.to_undirected(F.copy())), key=len)
    diameter = nx.diameter(F.subgraph(largest_cc))
    if scale < diameter//2:
        while len(F) > 0:
            center_node = max(F.degree, key=lambda x: x[1])[0]
            subgraph_nodes = [node for node, distance in nx.single_source_shortest_path_length(F, center_node).items() if distance <= scale]
            for n in subgraph_nodes:
                covering_vectors[n] = center_node
            F.remove_nodes_from(subgraph_nodes)

    return covering_vectors


def add_fractal_covering_matrix(G:dgl.DGLGraph, scales:List[int]=[1, 2]):
    covering_matrix = []
    for s in scales:
        covering_vectors = get_fractal_covering_vectors(G, s)
        covering_matrix.append(torch.LongTensor(covering_vectors, device=G.device))
    covering_matrix = torch.stack(covering_matrix).T.to(G.device)
    G.ndata["frac_cover_mat"] = covering_matrix