import warnings
warnings.filterwarnings("ignore")

import os
import gc
import torch
from torch.utils.data import Dataset
import dgl
import numpy as np
from sklearn.model_selection import StratifiedKFold

import random
from typing import List, Dict



class SimpleGCDataset(Dataset):
    def __init__(self, 
        graphs: List[dgl.DGLGraph],
        labels: List[torch.Tensor],
        is_fractal: List[bool],
        fractal_attr: List[float], 
        diameters: List[int]
    ) -> None:
        self.graphs = graphs
        self.labels = torch.LongTensor(labels)
        self.is_fractal = torch.BoolTensor(is_fractal)
        self.fractal_attr = torch.FloatTensor(fractal_attr)
        self.diameters = torch.LongTensor(diameters)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index], self.labels[index], self.is_fractal[index], self.fractal_attr[index], self.diameters[index]


def split_into_SimpleGCDataset(
    graphs: List[dgl.DGLGraph], 
    labels: List[torch.Tensor], 
    is_fractals: List[bool], 
    fractal_attrs: List[float], 
    diameters: List[int], 
    train_idxs: List[List[int]], 
    val_idxs: List[List[int]], 
    test_idxs: List[List[int]]
):
    if len(train_idxs) > 0:
        train_graphs, train_labels = [graphs[i] for i in train_idxs], [labels[i] for i in train_idxs]
        train_is_fractals, train_fractal_attrs = [is_fractals[i] for i in train_idxs], [fractal_attrs[i] for i in train_idxs]
        train_diameters = [diameters[i] for i in train_idxs]
    else:
        train_graphs, train_labels, train_is_fractals, train_fractal_attrs, train_diameters = [], [], [], [], []

    if len(val_idxs) > 0:
        val_graphs, val_labels = [graphs[i] for i in val_idxs], [labels[i] for i in val_idxs]
        val_is_fractals, val_fractal_attrs = [is_fractals[i] for i in val_idxs], [fractal_attrs[i] for i in val_idxs]
        val_diameters = [diameters[i] for i in val_idxs]
    else:
        val_graphs, val_labels, val_is_fractals, val_fractal_attrs, val_diameters = [], [], [], [], []

    if len(test_idxs) > 0:
        test_graphs, test_labels = [graphs[i] for i in test_idxs], [labels[i] for i in test_idxs]
        test_is_fractals, test_fractal_attrs = [is_fractals[i] for i in test_idxs], [fractal_attrs[i] for i in test_idxs]
        test_diameters = [diameters[i] for i in test_idxs]
    else:
        test_graphs, test_labels, test_is_fractals, test_fractal_attrs, test_diameters = [], [], [], [], []

    train = SimpleGCDataset(graphs=train_graphs, labels=train_labels, is_fractal=train_is_fractals, fractal_attr=train_fractal_attrs, diameters=train_diameters)
    val = SimpleGCDataset(graphs=val_graphs, labels=val_labels, is_fractal=val_is_fractals, fractal_attr=val_fractal_attrs, diameters=val_diameters)
    test = SimpleGCDataset(graphs=test_graphs, labels=test_labels, is_fractal=test_is_fractals, fractal_attr=test_fractal_attrs, diameters=test_diameters)

    return train, val, test


def split_train_val_test_GIN(
    graphs: List[dgl.DGLGraph], 
    labels: List[torch.Tensor], 
    fractal_results: List[Dict[str, str]] = None, 
    train_ratio: float = 0.55, 
    val_ratio: float = 0.05
):
    if train_ratio + val_ratio > 1:
        raise Exception(f"Error: allocate dataset train_ratio + val_ratio = {train_ratio:.2f} + {val_ratio:.2f} = {train_ratio+val_ratio:.2f} > 1.00")

    dataset_size = len(graphs)
    train_size, val_size = int(dataset_size*train_ratio), int(dataset_size*val_ratio)

    if fractal_results is None or len(fractal_results) == 0:
        is_fractals, fractal_attrs, diameters = [False for _ in range(dataset_size)], [0.0 for _ in range(dataset_size)], [0 for _ in range(dataset_size)]
    else:
        is_fractals, fractal_attrs, diameters = [], [], []
        for r in fractal_results:
            diameters.append(r["Statistics of Graph"]["Diameter"])
            res = r["Linear Regression"]["Origin Graph"]
            if "Can Test Fractality" in res:
                is_fractals.append(False)
                fractal_attrs.append(0.0)
            else:
                is_fractals.append(True)
                fractal_attrs.append(res["R²"])

    indexs = list(range(dataset_size))
    random.shuffle(indexs)
    
    train, val, test = split_into_SimpleGCDataset(
        graphs=graphs, 
        labels=labels,
        is_fractals=is_fractals, 
        fractal_attrs=fractal_attrs, 
        diameters=diameters, 
        train_idxs=indexs[:train_size], 
        val_idxs=indexs[train_size:train_size+val_size], 
        test_idxs=indexs[train_size+val_size:]
    )

    return train, val, test


def k_fold(
    graphs: List[dgl.DGLGraph], 
    labels: List[torch.Tensor],  
    fractal_results: List[Dict[str, str]] = None, 
    folds: int = 1, 
    semi_split: int = 10, 
): 
    if folds <= 1:
        raise Exception("Error: k-fold <= 1")

    dataset_size = len(graphs)
    labels = torch.tensor(labels)
    # k-fold indices
    train_indices, val_indices, test_indices = [], [], []
    skf = StratifiedKFold(folds, shuffle=True)
    for _, idxs in skf.split(torch.zeros(dataset_size), labels):
        test_indices.append([int(idx) for idx in idxs])

    val_indices = [test_indices[i-1] for i in range(folds)]
    # val_indices = [test_indices[i] for i in range(folds)]

    if semi_split > 1:
        skf_semi = StratifiedKFold(semi_split, shuffle=True)
        
    for i in range(folds):
        train_mask = torch.ones(dataset_size, dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        idx_train = train_mask.nonzero().view(-1)

        if semi_split > 1:
            for _, idxs in skf_semi.split(torch.zeros(idx_train.size(0)), labels[idx_train]):
                train_idx = [int(idx) for idx in idx_train[idxs]]
                break
        else:
            train_idx = [int(idx) for idx in idx_train]

        train_indices.append(train_idx)

    if fractal_results is None or len(fractal_results) == 0:
        is_fractals, fractal_attrs, diameters = [False for _ in range(dataset_size)], [0.0 for _ in range(dataset_size)], [0 for _ in range(dataset_size)]
    else:
        is_fractals, fractal_attrs, diameters = [], [], []
        for r in fractal_results:
            diameters.append(r["Statistics of Graph"]["Diameter"])
            res = r["Linear Regression"]["Origin Graph"]
            if "Can Test Fractality" in res:
                is_fractals.append(False)
                fractal_attrs.append(0.0)
            else:
                is_fractals.append(True)
                fractal_attrs.append(res["R²"])

    trains, vals, tests = [], [], []
    for fold in range(folds):
        train_idxs, val_idxs, test_idxs = train_indices[fold], val_indices[fold], test_indices[fold]
        train, val, test = split_into_SimpleGCDataset(
            graphs=graphs, 
            labels=labels, 
            is_fractals=is_fractals, 
            fractal_attrs=fractal_attrs, 
            diameters=diameters, 
            train_idxs=train_idxs, 
            val_idxs=val_idxs, 
            test_idxs=test_idxs
        )
        trains.append(train)
        vals.append(val)
        tests.append(test)
        
    return trains, vals, tests


class GraphPredDataset(Dataset):
    def __init__(self, 
        dataset_name: str, 
        graphs: List[dgl.DGLGraph], 
        labels: List[torch.Tensor], 
        embed_dim: int = 512, 
        train_ratio: float = 0.55, 
        val_ratio: float = 0.05, 
        folds: int = 1, 
        semi_split: int = 10, 
        fractal_results: List[Dict[str, str]] = [], 
    ) -> None:

        self.name = dataset_name

        ## add node features to dataset
        feat_dir = f"data/features"
        feat_path = os.path.join(feat_dir, f"{dataset_name.upper()}_node_features.pt")
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)
        if os.path.exists(feat_path):
            features = torch.load(feat_path, map_location=torch.device(graphs[0].device))
        else:
            features = [torch.randn(graph.number_of_nodes(), embed_dim) for graph in graphs]

        for idx, graph in enumerate(graphs):
            if "feat" not in graph.ndata:
                graphs[idx].ndata["feat"] = features[idx].to(graph.device)
            else:
                features[idx] = graphs[idx].ndata["feat"]
        torch.save(features, feat_path)
        del features

        # split train / val / test
        self.trains: List[SimpleGCDataset] = []
        self.vals: List[SimpleGCDataset] = []
        self.tests: List[SimpleGCDataset] = []
        if folds == 1:
            train, val, test = split_train_val_test_GIN(
                graphs=graphs, 
                labels=labels, 
                fractal_results=fractal_results, 
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )

            print('train, test, val sizes :',len(train),len(test),len(val))
            self.trains.append(train)
            self.vals.append(val)
            self.tests.append(test)
        else:
            self.trains, self.vals, self.tests = k_fold(
                graphs=graphs, 
                labels=labels, 
                fractal_results=fractal_results, 
                folds=folds, 
                semi_split=semi_split
            )

    def collate(self, samples):
        graphs, labels, is_fractal, fractal_attr, diameters = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))

        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1.0/float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()

        tab_sizes_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [torch.FloatTensor(size, 1).fill_(1.0/float(size)) for size in tab_sizes_e]
        snorm_e = torch.cat(tab_snorm_e).sqrt()

        batched_graphs = dgl.batch(graphs)
        return batched_graphs, labels, snorm_n, snorm_e, is_fractal, fractal_attr, diameters