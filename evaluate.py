import torch
import torch.nn as nn
import dgl


def evaluate(model:nn.Module, graph:dgl.DGLGraph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)[mask]
        labels = labels[mask]
        _, preds = torch.max(logits, dim=1)
        correct = torch.sum(preds == labels)
        acc = correct.item() / len(labels)
    return acc