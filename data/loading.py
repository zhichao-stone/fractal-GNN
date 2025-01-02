import torch
import dgl
import dgl.data as data

import random

import warnings
warnings.filterwarnings("ignore")


RAW_DIR = "/data/FinAi_Mapping_Knowledge/shizhichao/DGL_data"


def load_citeseer_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    dataset = data.citation_graph.load_citeseer(raw_dir=raw_dir)
    if return_origin_dataset:
        return dataset

    g = dataset[0]
    num_classes = dataset.num_classes
    train_mask = torch.BoolTensor(g.ndata["train_mask"])
    val_mask = torch.BoolTensor(g.ndata["val_mask"])
    test_mask = torch.BoolTensor(g.ndata["test_mask"])
    
    return g, g.ndata["feat"], g.ndata["label"], train_mask, val_mask, test_mask, num_classes


def load_cora_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    dataset = data.citation_graph.load_cora(raw_dir=raw_dir)
    if return_origin_dataset:
        return dataset
        
    g = dataset[0]
    num_classes = dataset.num_classes
    train_mask = torch.BoolTensor(g.ndata["train_mask"])
    val_mask = torch.BoolTensor(g.ndata["val_mask"])
    test_mask = torch.BoolTensor(g.ndata["test_mask"])
    return g, g.ndata["feat"], g.ndata["label"], train_mask, val_mask, test_mask, num_classes


def load_pubmed_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    dataset = data.citation_graph.load_pubmed(raw_dir=raw_dir)
    if return_origin_dataset:
        return dataset

    g = dataset[0]
    num_classes = dataset.num_classes
    train_mask = torch.BoolTensor(g.ndata["train_mask"])
    val_mask = torch.BoolTensor(g.ndata["val_mask"])
    test_mask = torch.BoolTensor(g.ndata["test_mask"])
    return g, g.ndata["feat"], g.ndata["label"], train_mask, val_mask, test_mask, num_classes


def load_reddit_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    dataset = data.RedditDataset(raw_dir=raw_dir)
    if return_origin_dataset:
        return dataset
        
    g = dataset[0]
    num_classes = dataset.num_classes
    train_mask = torch.BoolTensor(g.ndata["train_mask"])
    val_mask = torch.BoolTensor(g.ndata["val_mask"])
    test_mask = torch.BoolTensor(g.ndata["test_mask"])
    return g, g.ndata["feat"], g.ndata["label"], train_mask, val_mask, test_mask, num_classes


# link prediction
def load_fb15k237_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    dataset = data.FB15k237Dataset(raw_dir=raw_dir)
    if return_origin_dataset:
        return dataset

    g = dataset[0]
    num_classes = dataset.num_rels

    etype = g.edata["etype"]
    train_mask = torch.BoolTensor(g.edata["train_mask"])
    val_mask = torch.BoolTensor(g.edata["val_mask"])
    test_mask = torch.BoolTensor(g.edata["test_mask"])

    features = torch.randn(g.num_nodes(), 768)

    train_set = torch.arange(g.num_edges())[train_mask]
    val_set = torch.arange(g.num_edges())[val_mask]
    test_set = torch.arange(g.num_edges())[test_mask]

    train_edges = train_set
    train_graph = dgl.edge_subgraph(g, train_edges, relabel_nodes=False)
    train_graph.edata["etype"] = etype[train_edges]

    val_edges = torch.cat([train_edges, val_set])
    val_graph = dgl.edge_subgraph(g, val_edges, relabel_nodes=False)
    val_graph.edata["etype"] = etype[val_edges]

    test_edges = torch.cat([val_edges, test_set])
    test_graph = dgl.edge_subgraph(g, test_edges, relabel_nodes=False)
    test_graph.edata["etype"] = etype[test_edges]

    return g, features, train_graph, val_graph, test_graph, num_classes


def load_wn18_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    dataset = data.WN18Dataset(raw_dir=raw_dir)
    if return_origin_dataset:
        return dataset

    g = dataset[0]
    num_classes = dataset.num_rels

    etype = g.edata["etype"]
    train_mask = torch.BoolTensor(g.edata["train_mask"])
    val_mask = torch.BoolTensor(g.edata["val_mask"])
    test_mask = torch.BoolTensor(g.edata["test_mask"])

    features = torch.randn(g.num_nodes(), 768)

    train_set = torch.arange(g.num_edges())[train_mask]
    val_set = torch.arange(g.num_edges())[val_mask]
    test_set = torch.arange(g.num_edges())[test_mask]

    train_edges = train_set
    train_graph = dgl.edge_subgraph(g, train_edges, relabel_nodes=False)
    train_graph.edata["etype"] = etype[train_edges]

    val_edges = torch.cat([train_edges, val_set])
    val_graph = dgl.edge_subgraph(g, val_edges, relabel_nodes=False)
    val_graph.edata["etype"] = etype[val_edges]

    test_edges = torch.cat([val_edges, test_set])
    test_graph = dgl.edge_subgraph(g, test_edges, relabel_nodes=False)
    test_graph.edata["etype"] = etype[test_edges]

    return g, features, train_graph, val_graph, test_graph, num_classes


### graph classification
def load_gindataset_data(name:str, raw_dir=RAW_DIR, return_origin_dataset=False):
    dataset = data.GINDataset(name=name, raw_dir=raw_dir, self_loop=False)
    if return_origin_dataset:
        return dataset

    num_classes = dataset.num_classes
    graphs, labels = zip(*[dataset[i] for i in range(len(dataset))])
    return graphs, labels, num_classes

def load_mutag_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    return load_gindataset_data(name="MUTAG", raw_dir=raw_dir, return_origin_dataset=return_origin_dataset)

def load_collab_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    return load_gindataset_data(name="COLLAB", raw_dir=raw_dir, return_origin_dataset=return_origin_dataset)

def load_imdbbinary_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    return load_gindataset_data(name="IMDBBINARY", raw_dir=raw_dir, return_origin_dataset=return_origin_dataset)

def load_imdbmulti_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    return load_gindataset_data(name="IMDBMULTI", raw_dir=raw_dir, return_origin_dataset=return_origin_dataset)

def load_nci1_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    return load_gindataset_data(name="NCI1", raw_dir=raw_dir, return_origin_dataset=return_origin_dataset)

def load_proteins_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    return load_gindataset_data(name="PROTEINS", raw_dir=raw_dir, return_origin_dataset=return_origin_dataset)

def load_ptc_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    return load_gindataset_data(name="PTC", raw_dir=raw_dir, return_origin_dataset=return_origin_dataset)

def load_redditbinary_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    return load_gindataset_data(name="REDDITBINARY", raw_dir=raw_dir, return_origin_dataset=return_origin_dataset)

def load_redditmulti5k_data(raw_dir=RAW_DIR, return_origin_dataset=False):
    return load_gindataset_data(name="REDDITMULTI5K", raw_dir=raw_dir, return_origin_dataset=return_origin_dataset)


LOAD_FUNCTION_MAP = {
    # node prediction
    "citeseer": load_citeseer_data,
    "cora": load_cora_data,
    "pubmed": load_pubmed_data,
    "reddit": load_reddit_data,
    # link prediction
    "fb15k237": load_fb15k237_data,
    "wn18": load_wn18_data,
    # graph classification
    "mutag": load_mutag_data,
    "collab": load_collab_data,
    "imdbbinary": load_imdbbinary_data,
    "imdbmulti": load_imdbmulti_data,
    "nci1": load_nci1_data,
    "proteins": load_proteins_data,
    "ptc": load_ptc_data,
    "redditbinary": load_redditbinary_data,
    "redditmulti5k": load_redditmulti5k_data
}


NODE_PRED_DATA = [
    "citeseer",
    "cora",
    "pubmed",
    "reddit"
]

LINK_PRED_DATA = [
    "fb15k237",
    "wn18"
]

GRAPH_PRED_DATA = [
    "mutag",
    "collab",
    "imdbbinary",
    "imdbmulti"
    "nci1",
    "proteins",
    "ptc",
    "redditbinary",
    "redditmulti5k"
]