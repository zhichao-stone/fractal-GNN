import os
import json

import dgl
import torch
import networkx as nx



# functions for load json/jsonline files
def load_json(path:str, **kwargs):
    encoding = kwargs.pop("encoding", "utf-8")
    with open(path, "r", encoding=encoding, **kwargs) as fr:
        data = load_json(fr)
    return data

def dump_json(obj, path:str, **kwargs):
    file_dir = os.path.split(path)[0]
    if file_dir != "":
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
    encoding = kwargs.pop("encoding", "utf-8")
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    indent = kwargs.pop("indent", 4)
    with open(path, "w", encoding=encoding, **kwargs) as fw:
        json.dump(obj, fw, ensure_ascii=ensure_ascii, indent=indent)

def load_jsonlines(path:str, **kwargs):
    encoding = kwargs.pop("encoding", "utf-8")
    data = []
    with open(path, "r", encoding=encoding, **kwargs) as fr:
        for line in fr:
            row = json.loads(line)
            data.append(row)
    return data

def dump_jsonlines(obj, path:str, **kwargs):
    file_dir = os.path.split(path)[0]
    if file_dir != "":
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
    encoding = kwargs.pop("encoding", "utf-8")
    with open(path, "w", encoding=encoding, **kwargs) as fw:
        for row in obj:
            json.dump(row, fw)
            fw.write("\n")


# functions for fractal graph
def get_fractal_covering_vectors(G:dgl.DGLGraph, scale:int):
    F = dgl.to_networkx(G)
    covering_vectors = [0 for _ in F.nodes]
    while len(F) > 0:
        center_node = max(F.degree, key=lambda x: x[1])[0]
        subgraph_nodes = [node for node, distance in nx.single_source_shortest_path_length(F, center_node).items() if distance <= scale]
        for n in subgraph_nodes:
            covering_vectors[n] = center_node
        F.remove_nodes_from(subgraph_nodes)
    return covering_vectors


def add_fractal_covering_matrix(G:dgl.DGLGraph, scales:list=[1, 2]):
    covering_matrix = []
    for s in scales:
        covering_vectors = get_fractal_covering_vectors(G, s)
        covering_matrix.append(torch.LongTensor(covering_vectors, device=G.device))
    covering_matrix = torch.stack(covering_matrix).T.to(G.device)
    G.ndata["frac_cover_mat"] = covering_matrix


# functions for link prediction
def construct_negative_graph(graph:dgl.DGLGraph, k:int):
    src, dst = graph.edges()
    src, dst = torch.tensor(src), torch.tensor(dst)
    
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src)*k))

    neg_graph = dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())

    return neg_graph