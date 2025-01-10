import os
import time
import random
import glob
import argparse
import numpy as np
import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import GraphPredGINDataset
from data.data_augmentation import aug_renormalization_graphs, collate_batched_graph, sim_matrix2, compute_diag_sum
from models import GIN, ResGCN
from evaluate import evaluate_with_dataloader
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


def train_epoch_contrastive_learning(
    model: nn.Module,
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    data_loader: DataLoader, 
    aug_type: str, 
    aug_fractal_threshold: float
):
    model.train()
    epoch_loss = 0.0

    for batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e, batch_is_fractal, batch_fractal_attrs, batch_diameters in data_loader:
        aug_batch_graphs = dgl.unbatch(batch_graphs)
        aug_graphs_1, aug_graphs_2 = aug_renormalization_graphs(
            graphs=aug_batch_graphs, 
            is_fractals=batch_is_fractal, 
            fractal_attrs=batch_fractal_attrs, 
            diameters=batch_diameters, 
            aug_type=aug_type, 
            aug_fractal_threshold=aug_fractal_threshold,
            device=device
        )

        batch_graphs, batch_snorm_n, _ = collate_batched_graph(aug_graphs_1)
        aug_batch_graphs, aug_batch_snorm_n, _ = collate_batched_graph(aug_graphs_2)
        
        batch_graphs = batch_graphs.to(device)
        batch_h = batch_graphs.ndata["feat"].to(device)
        batch_snorm_n = batch_snorm_n.to(device)
        aug_batch_graphs = aug_batch_graphs.to(device)
        aug_batch_h = aug_batch_graphs.ndata["feat"].to(device)
        aug_batch_snorm_n = aug_batch_snorm_n.to(device)

        optimizer.zero_grad()
        ori_vector = model.forward(graph=batch_graphs, h=batch_h, snorm_n=batch_snorm_n)
        aug_vector = model.forward(graph=aug_batch_graphs, h=aug_batch_h, snorm_n=aug_batch_snorm_n)

        contrastive_loss = compute_contrastive_loss(ori_vector, aug_vector)
        contrastive_loss.backward()
        optimizer.step()
        epoch_loss += float(contrastive_loss.detach().cpu().item())
    
    epoch_loss = epoch_loss / len(data_loader)

    return epoch_loss, optimizer


def train_epoch_graph_classification(
    model: nn.Module,
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    data_loader: DataLoader,
):
    model.train()
    epoch_loss = 0.0

    for batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e, batch_is_fractal, batch_fractal_attrs, batch_diameters in data_loader:

        batch_graphs = batch_graphs.to(device)
        batch_h = batch_graphs.ndata["feat"].to(device)
        batch_snorm_n = batch_snorm_n.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        batch_scores = model.forward(graph=batch_graphs, h=batch_h, snorm_n=batch_snorm_n)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().cpu().item()
    
    epoch_loss = epoch_loss / len(data_loader)

    return epoch_loss, optimizer


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    if args.config is not None:
        try:
            configs: dict = load_json(args.config)
            train_params: dict = configs.pop("train_params", {})
            data_params: dict = configs.pop("data_params", {})
            model_params: dict = configs.pop("model_params", {})
        except Exception as e:
            print(str(e))
            configs = None
    if configs is None:
        raise Exception(f"There is error in configurations of {args.config}")

    gpu_id = train_params.pop("gpu", -1)
    if args.gpu is not None:
        gpu_id = args.gpu
    dataset_name = train_params.pop("dataset", "redditbinary")
    random_seed = train_params.pop("seed", 41)
    epochs = train_params.pop("epoch", 80)
    batch_size = train_params.pop("batch_size", 128)
    lr = train_params.pop("lr", 0.001)
    lr_reduce_factor = train_params.pop("lr_reduce_factor", 0.5)
    lr_schedule_patience = train_params.pop("lr_schedule_patience", 5)
    min_lr = train_params.pop("min_lr", 1e-5)
    weight_decay = train_params.pop("weight_decay", 1e-6)
    aug_type = train_params.pop("aug_type", "drop_node")
    aug_fractal_threshold = train_params.pop("aug_fractal_threshold", 0.8)
    is_pretrain = train_params.pop("is_pretrain", False)
    test_mode = train_params.pop("test", False)
    load_model = train_params.pop("load_model", False)

    embed_dim = data_params.pop("embed_dim", 768)
    train_ratio = data_params.pop("train_ratio", 0.55)
    val_ratio = data_params.pop("val_ratio", 0.05)

    model_name = model_params.pop("model", "GIN")

    # random setting
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)  # set seed for cpu
    torch.cuda.manual_seed(random_seed)  # set seed for current gpu
    torch.cuda.manual_seed_all(random_seed)  # set seed for all gpu
    dgl.random.seed(random_seed)

    # log setting
    log_file_name = f"{str(os.path.split(args.config)[-1]).split('.')[0]}" if args.config != "" else f"GIN_{dataset_name}"
    if test_mode:
        log_file_name += "_test"
    logger = ModelLogger(log_file_name, "log", backupCount=7).get_logger()
    logger.info(f"Logging to {log_file_name}.log")
    logger.info("=============== Configurations ===============")
    logger.info(f"model: {model_name}")
    logger.info(f"embedding dim: {embed_dim}")
    logger.info(f"model_params: ")
    for k, v in model_params.items():
        logger.info(f"\t{k}: {v}")

    # device
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")
    
    # load data
    if is_pretrain and aug_type in ["renormalization", "drop_fractal_box"]:
        try:
            fractal_results = load_json(os.path.join("fractal_results", f"linear_regression_{dataset_name}.json"))
        except:
            fractal_results = []
        try:
            covering_matrix = torch.load(os.path.join("fractal_results", f"fractal_covering_matrix_{dataset_name}.pt"))
        except:
            covering_matrix = None
    else:
        fractal_results = []
        covering_matrix = None

    dataset = GraphPredGINDataset(
        dataset_name=dataset_name, 
        raw_dir=DATA_RAW_DIR, 
        self_loop=False, 
        embed_dim=embed_dim, 
        train_ratio=train_ratio,
        val_ratio=val_ratio, 
        fractal_results=fractal_results, 
        covering_matrix=covering_matrix
    )
    input_dim = embed_dim
    num_classes = dataset.num_classes
    logger.info(f"Load Data: {dataset_name} , input_dim={input_dim} , num_classes={num_classes}")

    # model
    save_model_dir = os.path.join("./contrastive_models", model_name, f"{dataset_name}_{aug_type}")
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    if model_name == "GIN":
        model = GIN(
            input_dim=input_dim,
            num_classes=num_classes,
            **model_params
        )
    elif model_name == "ResGCN":
        model = ResGCN(
            input_dim=input_dim, 
            num_classes=num_classes, 
            **model_params
        )
    else:
        raise Exception(f"Model {model_name} is not supported!")

    if load_model:
        load_model_path = sorted(glob.glob(save_model_dir + "/*.pkl"), key=lambda x:int(os.path.split(x)[-1].replace("epoch_", "").replace(".pkl", "")))
        checkpoint = torch.load(load_model_path[-1])
        model_dict = model.state_dict()
        state_dict = {k:v for k, v in checkpoint.items() if k in model_dict.keys()}
        model.load_state_dict(state_dict)
        logger.info(f"Success load pre-trained model : {load_model_path[-1]}")
        if is_pretrain:
            current_epoch = 1 + int(os.path.splitext(load_model_path[-1])[-1].replace("epoch_", "").replace(".pkl", ""))
        else:
            current_epoch = 0
    else:
        logger.info("Train Base Model.")
        current_epoch = 0

    model = model.to(device)

    # training
    logger.info("=============== Train Argument ===============")
    logger.info(f"batch size: {batch_size}")
    logger.info(f"learnig rate: {lr}")
    logger.info(f"lr reduce factor: {lr_reduce_factor} , schedule patience: {lr_schedule_patience}")
    logger.info(f"min learning rate: {min_lr}")
    logger.info(f"weight decay: {weight_decay}")
    logger.info(f"epochs: {epochs}")
    if is_pretrain:
        logger.info(f"augmentation type: {aug_type}")
        logger.info(f"fractal threshold: {aug_fractal_threshold}")

    logger.info("=============== Start Training ===============")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min", 
        factor=lr_reduce_factor, 
        patience=lr_schedule_patience, 
        verbose=True
    )

    train_loader = DataLoader(
        dataset.train, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=dataset.collate
    )
    if not is_pretrain:
        val_loader = DataLoader(
            dataset.val, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=dataset.collate
        )
        test_loader = DataLoader(
            dataset.test, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=dataset.collate
        )

    if is_pretrain:
        train_time = 0
        for epoch in range(current_epoch, epochs):
            st = time.time()

            epoch_train_loss, optimizer = train_epoch_contrastive_learning(
                model=model, 
                optimizer=optimizer, 
                device=device, 
                data_loader=train_loader, 
 
                aug_type=aug_type, 
                aug_fractal_threshold=aug_fractal_threshold
            )

            epoch_time = time.time() - st
            train_time += epoch_time

            scheduler.step(epoch_train_loss)
            logger.info(f"# Epoch: {epoch+1:04d} , Loss: {epoch_train_loss:.4f} , Cost Time: {epoch_time:.2f} s")

            ### save model
            save_ckpt_path = os.path.join(save_model_dir, f"epoch_{epoch}.pkl")
            torch.save(model.state_dict(), save_ckpt_path)
        logger.info(f"Pre-training finished, Totally cost : {train_time:.2f} s ({train_time/60:.2f} min)")
    else:
        if test_mode:
            best_test_acc, best_test_epoch = 0.0, 0

        for epoch in range(current_epoch, epochs):
            epoch_train_loss, optimizer = train_epoch_graph_classification(
                model=model,
                optimizer=optimizer,
                device=device, 
                data_loader=train_loader
            )

            epoch_val_loss, epoch_val_acc = evaluate_with_dataloader(model, val_loader, device=device, criterion=nn.CrossEntropyLoss())
            if test_mode:
                _, epoch_test_acc = evaluate_with_dataloader(model, test_loader, device=device)

            scheduler.step(epoch_val_loss)
            if test_mode:
                if epoch_test_acc >= best_test_acc:
                    best_test_acc, best_test_epoch = epoch_test_acc, epoch+1
                logger.info(f"# Epoch: {epoch+1:04d} | Train_Loss: {epoch_train_loss:.4f} | Val_Loss: {epoch_val_loss:.4f} , Val_Acc: {epoch_val_acc:.4f} | Test_Acc: {epoch_test_acc:.4f}")
            else:
                logger.info(f"# Epoch: {epoch+1:04d} | Train_Loss: {epoch_train_loss:.4f} | Val_Loss: {epoch_val_loss:.4f} , Val_Acc: {epoch_val_acc:.4f}")

        logger.info("=============== Start Evaluating ===============")
        if not test_mode:
            _, test_acc = evaluate_with_dataloader(model, test_loader, device=device)
            logger.info(f"Test Accuracy: {test_acc:.4f}\n")
        else:
            logger.info(f"Best Test Accuracy: {best_test_acc:.4f} , Best Test Epoch: {best_test_epoch}\n")