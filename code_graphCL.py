import os
import time
import random
import argparse
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from logging import Logger
from typing import List, Dict

from data.dataset import GraphPredGINDataset
from data.data_augmentation import aug_renormalization_graphs, collate_batched_graph, sim_matrix2, compute_diag_sum
from models.gin import GIN
from logger import ModelLogger
from utils import load_json


DATA_RAW_DIR = "/data/FinAi_Mapping_Knowledge/shizhichao/DGL_data"



def compute_contrastive_loss(vec1:torch.Tensor, vec2:torch.Tensor):
    sim_matrix = sim_matrix2(vec1, vec2)
    row_softmax = nn.LogSoftmax(dim=1)
    row_softmax_matrix = - row_softmax(sim_matrix)
    col_softmax = nn.LogSoftmax(dim=0)
    col_softmax_matrix = - col_softmax(sim_matrix)

    row_diag_sum = compute_diag_sum(row_softmax_matrix)
    col_diag_sum = compute_diag_sum(col_softmax_matrix)
    contrastive_loss = (row_diag_sum + col_diag_sum) / (2*len(row_softmax_matrix))

    return contrastive_loss


def train_contrastive_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    data_loader: DataLoader, 
    epoch: int, 
    aug_scales: List[int], 
    head: bool, 
    aug_type: str, 
    aug_fractal_threshold: float, 
    logger: Logger
):
    model.train()
    epoch_loss = 0.0

    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e, batch_is_fractal, batch_fractal_attrs) in enumerate(data_loader):
        aug_batch_graphs = dgl.unbatch(batch_graphs)
        aug_graphs_1, aug_graphs_2 = aug_renormalization_graphs(aug_batch_graphs, aug_scales, batch_is_fractal, batch_fractal_attrs)

        batch_graphs, batch_snorm_n, _ = collate_batched_graph(aug_graphs_1)
        aug_batch_graphs, aug_batch_snorm_n, _ = collate_batched_graph(aug_graphs_2)
        
        batch_h = batch_graphs.ndata["attr"].to(device)
        batch_snorm_n = batch_snorm_n.to(device)
        aug_batch_h = aug_batch_graphs.ndata["attr"].to(device)
        aug_batch_snorm_n = aug_batch_snorm_n.to(device)

        optimizer.zero_grad()
        ori_vector = model.forward(batch_graphs, batch_h, batch_snorm_n, mlp=False, head=head)
        aug_vector = model.forward(aug_batch_graphs, aug_batch_h, aug_batch_snorm_n, mlp=False, head=head)

        contrastive_loss = compute_contrastive_loss(ori_vector, aug_vector)
        contrastive_loss.backward()
        optimizer.step()
        epoch_loss += float(contrastive_loss.detach().cpu().item())

        if (iter + 1) % 20 == 0:
            logger.info(f"##### Epoch: {epoch+1}/{epochs} , Iter: {iter+1}/{len(data_loader)} , Loss: {contrastive_loss:.4f}")
    
    epoch_loss = epoch_loss / len(data_loader)

    return epoch_loss, optimizer



if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    ### train settings
    parser.add_argument("--gpu", type=int, default=0, help="id of gpu to use. -1 if cpu used")
    parser.add_argument("--dataset", type=str, default="redditbinary")
    parser.add_argument("--seed", type=int, default=41, help="random seed")
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5)
    parser.add_argument("--lr_schedule_patience", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--aug_type", type=str, default="rr", help="type of data augmentation, n (drop node) or r (renormalization). aug_type should be selected from [ nn, nr, rn, rr ]")
    parser.add_argument("--aug_scales", nargs="*", help="scales of renormalized graph")
    parser.add_argument("--aug_fractal_threshold", type=float, default=0.8)
    parser.add_argument("--log_epoch_interval", type=int, default=5)
    parser.add_argument("--is_pretrain", action="store_true")
    # model settings
    parser.add_argument("--model", type=str, default="GIN")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--neighbor_aggr", type=str, default="sum")
    parser.add_argument("--pooling_type", type=str, default="sum")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--head", action="store_true", help="use projection head or not")
    parser.add_argument("--learn_eps", action="store_true")
    parser.add_argument("--graph_norm", action="store_true")
    parser.add_argument("--batch_norm", action="store_true")
    parser.add_argument("--residual", action="store_true")
    
    args = parser.parse_args()

    if args.config != "":
        try:
            configs = load_json(args.config)
            assert "train_params" in configs and "model_params" in configs
        except Exception as e:
            print(str(e))
            configs = None

    if configs is not None:
        gpu_id = configs["train_params"].pop("gpu", 0)
        dataset_name = configs["train_params"].pop("dataset", "redditbinary")
        random_seed = configs["train_params"].pop("seed", 41)
        epochs = configs["train_params"].pop("epoch", 80)
        batch_size = configs["train_params"].pop("batch_size", 128)
        lr = configs["train_params"].pop("lr", 0.001)
        lr_reduce_factor = configs["train_params"].pop("lr_reduce_factor", 0.5)
        lr_schedule_patience = configs["train_params"].pop("lr_schedule_patience", 5)
        min_lr = configs["train_params"].pop("min_lr", 1e-5)
        weight_decay = configs["train_params"].pop("weight_decay", 1e-6)
        aug_type = configs["train_params"].pop("aug_type", "nn")
        aug_scales = [int(s) for s in configs["train_params"]["aug_scales"]] if "aug_scales" in configs["train_params"] else [1, 2]
        aug_fractal_threshold = configs["train_params"].pop("aug_fractal_threshold", 0.8)
        log_epoch_interval = configs["train_params"].pop("log_epoch_interval", 5)
        is_pretrain = configs["train_params"].pop("is_pretrain", True)

        model_name = configs["model_params"].pop("model", "GIN")
        num_layers = configs["model_params"].pop("num_layers", 4)
        mlp_layers = configs["model_params"].pop("mlp_layers", 2)
        hidden_dim = configs["model_params"].pop("hidden_dim", 128)
        neighbor_aggr_type = configs["model_params"].pop("neighbor_aggr_type", "sum")
        pooling_type = configs["model_params"].pop("pooling_type", "sum")
        dropout = configs["model_params"].pop("dropout", 0.0)
        head = configs["model_params"].pop("head", True)
        learn_eps = configs["model_params"].pop("learn_eps", True)
        graph_norm = configs["model_params"].pop("graph_norm", True)
        batch_norm = configs["model_params"].pop("batch_norm", True)
        residual = configs["model_params"].pop("residual", True)
    else:
        gpu_id = args.gpu
        dataset_name = args.dataset
        random_seed = args.seed
        epochs = args.epoch
        batch_size = args.batch_size
        lr = args.lr
        lr_reduce_factor = args.lr_reduce_factor
        lr_schedule_patience = args.lr_schedule_patience
        min_lr = args.min_lr
        weight_decay = args.weight_decay
        aug_type = args.aug_type
        aug_scales = [int(s) for s in args.aug_scales] if args.aug_scales is not None else [1, 2]
        aug_fractal_threshold = args.aug_fractal_threshold
        log_epoch_interval = args.log_epoch_interval
        is_pretrain = args.is_pretrain

        model_name = args.model
        num_layers = args.num_layers
        mlp_layers = args.mlp_layers
        hidden_dim = args.hidden_dim
        neighbor_aggr_type = args.neighbor_aggr_type
        pooling_type = args.pooling_type
        dropout = args.dropout
        head = args.head
        learn_eps = args.learn_eps
        graph_norm = args.graph_norm
        batch_norm = args.batch_norm
        residual = args.residual

    # random setting
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  # set seed for cpu
    torch.cuda.manual_seed(random_seed)  # set seed for current gpu
    torch.cuda.manual_seed_all(random_seed)  # set seed for all gpu
    dgl.random.seed(random_seed)

    # log setting
    log_file_name = f"{args.config}" if args.config != "" else f"GIN_{dataset_name}"
    logger = ModelLogger(log_file_name, "log", backupCount=7).get_logger()
    logger.info(f"Logging to {log_file_name}.log")

    # device
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")
    
    # load data
    dataset = GraphPredGINDataset(dataset_name=dataset_name, raw_dir=DATA_RAW_DIR, self_loop=False)
    input_dim = dataset.train[0][0].ndata["attr"].size(0)
    num_classes = dataset.num_classes
    logger.info(f"Load Data : {dataset_name}")

    # model
    model = GIN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        num_layers=num_layers,
        mlp_num_layers=mlp_layers,
        neighbor_aggr_type=neighbor_aggr_type,
        pooling_type=pooling_type,
        learn_eps=learn_eps,
        graph_norm=graph_norm,
        batch_norm=batch_norm,
        residual=residual
    )
    model = model.to(device)

    save_model_dir = os.path.join("./contrastive_models", model_name, f"{dataset_name}_{aug_type}")
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)


    # training
    logger.info("=============== Train Argument ===============")
    logger.info(f"Learnig Rate: {lr}")
    logger.info(f"Weight Decay: {weight_decay}")
    logger.info(f"Epochs: {epochs}")
    logger.info("=============== Start Training ===============")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        factor=lr_reduce_factor, 
        patience=lr_schedule_patience, 
        verbose=True
    )

    train_loader = DataLoader(dataset.train, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate)

    train_time = 0
    for epoch in range(epochs):
        st = time.time()

        epoch_train_loss, optimizer = train_contrastive_epoch(
            model=model, 
            optimizer=optimizer, 
            device=device, 
            data_loader=train_loader, 
            epoch=epoch, 
            head=head, 
            aug_type=aug_type, 
            aug_fractal_threshold=aug_fractal_threshold, 
            logger=logger
        )

        epoch_time = time.time() - st
        train_time += epoch_time

        scheduler.step(epoch_train_loss)
        logger.info(f"# Epoch: {epoch+1} , Loss: {epoch_train_loss:.4f} , Cost Time: {epoch_time:.2f}s , Total Time: {train_time:.2f}")

        ### save model
        save_ckpt_path = os.path.join(save_model_dir, f"epoch_{epoch}.pkl")
        torch.save(model.state_dict(), save_ckpt_path)
