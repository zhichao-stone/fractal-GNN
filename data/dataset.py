import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
import dgl
import dgl.data as dgldata
import numpy as np

import random
from typing import List, Dict



class GINDGL(Dataset):
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


def split_train_val_test_GIN(
    dataset: dgldata.GINDataset, 
    fractal_results: List[Dict[str, str]] = None, 
    train_ratio: float = 0.55, 
    val_ratio: float = 0.05
):
    if train_ratio + val_ratio > 1:
        raise Exception(f"Error: allocate dataset train_ratio + val_ratio = {train_ratio:.2f} + {val_ratio:.2f} = {train_ratio+val_ratio:.2f} > 1.00")

    dataset_size = len(dataset)
    train_size, val_size = int(dataset_size*train_ratio), int(dataset_size*val_ratio)
    test_size = dataset_size - train_size - val_size

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
    train_idxs, val_idxs, test_idxs = indexs[:train_size], indexs[train_size:train_size+val_size], indexs[train_size+val_size:]
    
    train_graphs, train_labels = zip(*[dataset[i] for i in train_idxs])
    train_is_fractals, train_fractal_attrs = [is_fractals[i] for i in train_idxs], [fractal_attrs[i] for i in train_idxs]
    train_diameters = [diameters[i] for i in train_idxs]

    if val_size > 0:
        val_graphs, val_labels = zip(*[dataset[i] for i in val_idxs])
        val_is_fractals, val_fractal_attrs = [is_fractals[i] for i in val_idxs], [fractal_attrs[i] for i in val_idxs]
        val_diameters = [diameters[i] for i in val_idxs]
    else:
        val_graphs, val_labels, val_is_fractals, val_fractal_attrs, val_diameters = [], [], [], [], []

    if test_size > 0:
        test_graphs, test_labels = zip(*[dataset[i] for i in test_idxs])
        test_is_fractals, test_fractal_attrs = [is_fractals[i] for i in test_idxs], [fractal_attrs[i] for i in test_idxs]
        test_diameters = [diameters[i] for i in test_idxs]
    else:
        test_graphs, test_labels, test_is_fractals, test_fractal_attrs, test_diameters = [], [], [], [], []

    train = GINDGL(graphs=train_graphs, labels=train_labels, is_fractal=train_is_fractals, fractal_attr=train_fractal_attrs, diameters=train_diameters)
    val = GINDGL(graphs=val_graphs, labels=val_labels, is_fractal=val_is_fractals, fractal_attr=val_fractal_attrs, diameters=val_diameters)
    test = GINDGL(graphs=test_graphs, labels=test_labels, is_fractal=test_is_fractals, fractal_attr=test_fractal_attrs, diameters=test_diameters)

    return train, val, test



class GraphPredGINDataset(Dataset):
    def __init__(self, 
        dataset_name: str, 
        raw_dir: str, 
        self_loop: bool = False, 
        embed_dim: int = 768, 
        train_ratio: float = 0.55, 
        val_ratio: float = 0.05, 
        fractal_results: List[Dict[str, str]] = [], 
        covering_matrix: torch.Tensor = None
    ) -> None:

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        dataset = dgldata.GINDataset(name=dataset_name.upper(), raw_dir=raw_dir, self_loop=self_loop)
        if covering_matrix is not None:
            for idx in range(len(dataset)):
                dataset.graphs[idx].ndata["frac_cover_mat"] = covering_matrix[idx]

        self.name = dataset.name
        self.num_classes = dataset.num_classes
        
        self.train, self.val, self.test = split_train_val_test_GIN(
            dataset=dataset, 
            fractal_results=fractal_results, 
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio
        )

        self.process_feature(self.train, embed_dim)
        self.process_feature(self.val, embed_dim)
        self.process_feature(self.test, embed_dim)

        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))

    def process_feature(self, dataset: GINDGL, embed_dim: int):
        for idx, graph in enumerate(dataset.graphs):
            if "feat" not in graph.ndata:
                dataset.graphs[idx].ndata["feat"] = torch.randn(graph.number_of_nodes(), embed_dim)

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