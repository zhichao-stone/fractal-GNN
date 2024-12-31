import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl


def evaluate(
    model: nn.Module, 
    graph: dgl.DGLGraph, 
    features, 
    labels, 
    mask
):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)[mask]
        labels = labels[mask]
        _, preds = torch.max(logits, dim=1)
        correct = torch.sum(preds == labels)
        acc = correct.item() / len(labels)
    return acc


def evaluate_with_dataloader(
    model:nn.Module, 
    data_loader: DataLoader, 
    device: torch.device,  
    criterion: nn.Module = None,
    head: bool = True
): 
    model.eval()
    loss = 0
    acc = 0
    nb_data = 0
    with torch.no_grad():
        for batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e, batch_is_fractal, batch_fractal_attrs, batch_diameters in data_loader:
            batch_graphs = batch_graphs.to(device)
            batch_h = batch_graphs.ndata["feat"].to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            batch_labels = batch_labels.to(device)

            batch_scores = model.forward(batch_graphs, batch_h, batch_snorm_n, mlp=False, head=head)

            if criterion is not None:
                batch_loss = criterion(batch_scores, batch_labels)
                loss += batch_loss.detach().item()
            
            _, batch_preds = torch.max(batch_scores, dim=1)
            batch_correct = torch.sum(batch_preds == batch_labels).item()
            acc += batch_correct
            nb_data += batch_labels.size(0)
        
        loss = loss / len(data_loader)
        acc = acc / nb_data
    
    return loss, acc