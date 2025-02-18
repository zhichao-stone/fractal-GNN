import os
import gc
import math
import argparse
from tqdm import tqdm

import dgl
import torch
import networkx as nx

from data.loading import LOAD_FUNCTION_MAP
from utils import load_json



def get_fractal_covering_vectors(G:dgl.DGLGraph, scale:int, diameter:int):
    covering_vectors = [0 for _ in range(G.number_of_nodes())]

    if scale < diameter//2:
        F = dgl.to_networkx(G)
        while len(F) > 0:
            center_node = max(F.degree, key=lambda x: x[1])[0]
            subgraph_nodes = [node for node, distance in nx.single_source_shortest_path_length(F, center_node).items() if distance <= scale]
            for n in subgraph_nodes:
                covering_vectors[n] = center_node
            F.remove_nodes_from(subgraph_nodes)

    return covering_vectors



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="redditbinary")
    args = parser.parse_args()
    dataset_name = args.data

    load_data = LOAD_FUNCTION_MAP[dataset_name]()
    graphs = load_data[0]

    fractal_results = load_json(os.path.join("fractal_results", f"linear_regression_{dataset_name}.json"))
    diameters = [r["Statistics of Graph"]["Diameter"] for r in fractal_results]
    # scales = list(range(1, max(1, max(diameters)//3)+1))
    scales = list(range(1, max(2, int(math.log2(max(diameters)/2)))))


    covering_matrix = []
    for i in tqdm(range(len(graphs)), desc=dataset_name):
        cov_mat_g = []
        
        for s in scales:
            cov_vec = get_fractal_covering_vectors(graphs[i], s, diameters[i])
            cov_mat_g.append(torch.LongTensor(cov_vec))

        cov_mat_g = torch.stack(cov_mat_g).T
        covering_matrix.append(cov_mat_g)

        if (i+1) % 10 == 0:
            torch.save(covering_matrix, os.path.join("fractal_results", f"fractal_covering_matrix_{dataset_name}.pt"))

    torch.save(covering_matrix, os.path.join("fractal_results", f"fractal_covering_matrix_{dataset_name}.pt"))