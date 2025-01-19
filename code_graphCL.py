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
from tqdm import tqdm

from data.loading import load_gindataset_data
from data.dataset import GraphPredDataset
from data.data_augmentation import DataAugmentator, collate_batched_graph
from models import GIN, ResGCN, GAT
from models.loss import ContrastiveLearningLoss, CLLoss1, CLLoss2, CLLoss3
from evaluate import evaluate_with_dataloader
from logger import ModelLogger
from utils import load_json


DATA_RAW_DIR = "/data/FinAi_Mapping_Knowledge/shizhichao/DGL_data"



def load_graphcl_model(
    model_name: str, 
    checkpoint: dict, 
    device: torch.device, 
    input_dim: int, 
    num_classes: int, 
    **model_params
) -> nn.Module:
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
    elif model_name == "GAT":
        model = GAT(
            input_dim=input_dim, 
            num_classes=num_classes, 
            **model_params
        )
    else:
        raise Exception(f"Model {model_name} is not supported!")

    if checkpoint is not None:
        model_dict = model.state_dict()
        state_dict = {k:v for k, v in checkpoint.items() if k in model_dict.keys()}
        model.load_state_dict(state_dict)

    model = model.to(device)

    return model


def train_epoch_contrastive_learning(
    model: nn.Module,
    optimizer: torch.optim.Optimizer, 
    device: torch.device, 
    data_loader: DataLoader, 
    data_augmentor: DataAugmentator, 
    aug_type: str, 
    loss_fn: ContrastiveLearningLoss
):
    model.train()
    epoch_loss = 0.0

    for batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e, batch_is_fractal, batch_fractal_attrs, batch_diameters in tqdm(data_loader):
        aug_batch_graphs = dgl.unbatch(batch_graphs)
        aug_graphs_1, aug_graphs_2 = data_augmentor.augment_graphs(
            graphs=aug_batch_graphs, 
            is_fractals=batch_is_fractal, 
            fractal_attrs=batch_fractal_attrs, 
            diameters=batch_diameters, 
            aug_type=aug_type
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

        contrastive_loss = loss_fn(ori_vector, aug_vector)
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
        loss: torch.Tensor = criterion(batch_scores, batch_labels)
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
    load_model_epoch = train_params.pop("load_model_epoch", -1)
    loss_type = train_params.pop("loss_type", 1)
    loss_temperature = train_params.pop("loss_temperature", 0.5)
    renorm_min_edges = train_params.pop("renorm_min_edges", 1)
    drop_ratio = train_params.pop("drop_ratio", 0.2)

    embed_dim = data_params.pop("embed_dim", 768)
    train_ratio = data_params.pop("train_ratio", 0.55)
    val_ratio = data_params.pop("val_ratio", 0.05)
    folds = data_params.pop("folds", 1)
    semi_split = data_params.pop("semi_split", 10)

    model_name = model_params.pop("model", "GIN")
    pooling_type = model_params["pooling_type"] if "pooling_type" in model_params else "sum"

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
    if loss_type != 1:
        if f"_loss{loss_type}" not in log_file_name:
            log_file_name += f"_loss{loss_type}"
    if renorm_min_edges > 1:
        if f"_me{renorm_min_edges}" not in log_file_name:
            log_file_name += f"_me{renorm_min_edges}"
    if pooling_type != "sum":
        if f"_{pooling_type}pooling" not in log_file_name:
            log_file_name += f"_{pooling_type}pooling" 
    if not is_pretrain:
        if folds > 1:
            if f"_f{folds}_semi{semi_split}" not in log_file_name:
                log_file_name += f"_{folds}_{semi_split}"
        if test_mode:
            if "_test" not in log_file_name:
                log_file_name += "_test"

    logger = ModelLogger(log_file_name, "log").get_logger()
    logger.info(f"Logging to {log_file_name}.log")
    logger.info("=============== Configurations ===============")
    logger.info(f"model: {model_name}")
    logger.info(f"embedding dim: {embed_dim}")
    logger.info(f"model_params: ")
    for k, v in model_params.items():
        logger.info(f"\t{k}: {v}")

    # device
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")
    logger.info(f"Device: {device}")
    
    # load data
    if is_pretrain and aug_type in ["renormalization", "renormalization_random_center", "drop_fractal_box"]:
        try:
            fractal_results = load_json(os.path.join("fractal_results", f"linear_regression_{dataset_name}.json"))
        except Exception as e:
            logger.info(e)
            fractal_results = []
    else:
        fractal_results = []

    graphs, labels, num_classes = load_gindataset_data(dataset_name, raw_dir=DATA_RAW_DIR)
    dataset = GraphPredDataset(
        dataset_name=dataset_name, 
        graphs=graphs, 
        labels=labels,  
        embed_dim=embed_dim, 
        train_ratio=train_ratio,
        val_ratio=val_ratio, 
        folds=folds, 
        semi_split=semi_split, 
        fractal_results=fractal_results
    )
    input_dim = embed_dim
    logger.info(f"Load Data: {dataset_name} , input_dim={input_dim} , num_classes={num_classes}")

    # model
    model_last_dir = f"{dataset_name}_{aug_type}"
    if loss_type != 1:
        model_last_dir += f"_loss{loss_type}"
    if "aug_sum" in log_file_name:
        model_last_dir += f"_aug_sum"
    if renorm_min_edges > 1:
        model_last_dir += f"_me{renorm_min_edges}"
    if pooling_type != "sum":
        model_last_dir += f"_{pooling_type}pooling"
    save_model_dir = os.path.join("./contrastive_models", model_name, model_last_dir)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    if load_model:
        load_model_path = sorted(glob.glob(save_model_dir + "/*.pkl"), key=lambda x:int(os.path.split(x)[-1].replace("epoch_", "").replace(".pkl", "")))
        if load_model_epoch < 0:
            checkpoint = torch.load(load_model_path[-1])
        else:
            checkpoint = torch.load(load_model_path[load_model_epoch])
        
        logger.info(f"Success load pre-trained model : {load_model_path[-1]}")
        if is_pretrain:
            current_epoch = 1 + int(os.path.splitext(load_model_path[-1])[-1].replace("epoch_", "").replace(".pkl", ""))
        else:
            current_epoch = 0
    else:
        checkpoint = None
        logger.info("Train Base Model.")
        current_epoch = 0

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

    if is_pretrain:
        model = load_graphcl_model(model_name, checkpoint, device, input_dim, num_classes, **model_params)
        loss_fn = None
        if loss_type == 1:
            loss_fn = CLLoss1(temperature=loss_temperature)
        elif loss_type == 2:
            loss_fn = CLLoss2(temperature=loss_temperature)
        elif loss_type == 3:
            loss_fn = CLLoss3(temperature=loss_temperature)
        else:
            raise NotImplementedError()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=lr_reduce_factor, 
            patience=lr_schedule_patience, 
            verbose=True
        )

        train_loader = DataLoader(
            dataset.trains[0], 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=dataset.collate
        )
        if not is_pretrain:
            val_loader = DataLoader(
                dataset.vals[0], 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=dataset.collate
            )
            test_loader = DataLoader(
                dataset.tests[0], 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=dataset.collate
            )

        augmentor = DataAugmentator(
            drop_ratio=drop_ratio, 
            aug_fractal_threshold=aug_fractal_threshold, 
            renorm_min_edges=renorm_min_edges, 
            device=device
        )

        total_st = time.time()
        for epoch in range(current_epoch, epochs):
            st = time.time()

            epoch_train_loss, optimizer = train_epoch_contrastive_learning(
                model=model, 
                optimizer=optimizer, 
                device=device, 
                data_loader=train_loader, 
                data_augmentor=augmentor, 
                aug_type=aug_type, 
                loss_fn = loss_fn
            )

            epoch_time = time.time() - st

            scheduler.step(epoch_train_loss)
            logger.info(f"# Epoch: {epoch+1:04d} , Loss: {epoch_train_loss:.4f} , Cost Time: {epoch_time:.2f} s , Totally: {time.time() - total_st:.2f} s")

            ### save model
            save_ckpt_path = os.path.join(save_model_dir, f"epoch_{epoch}.pkl")
            torch.save(model.state_dict(), save_ckpt_path)
        
        train_time = time.time() - total_st
        logger.info(f"Pre-training finished, Totally cost : {train_time:.2f} s ({train_time/60:.2f} min)")
    else:
        if test_mode:
            best_fold, best_test_acc, best_test_epoch = 0, 0.0, 0

        test_accs = []
        for fold in range(folds):
            model = load_graphcl_model(model_name, checkpoint, device, input_dim, num_classes, **model_params)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode="min", 
                factor=lr_reduce_factor, 
                patience=lr_schedule_patience, 
                verbose=True
            )

            train_loader = DataLoader(
                dataset.trains[fold], 
                batch_size=batch_size, 
                shuffle=True, 
                collate_fn=dataset.collate
            )

            val_loader = DataLoader(
                dataset.vals[fold], 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=dataset.collate
            )
            test_loader = DataLoader(
                dataset.tests[fold], 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=dataset.collate
            )

            logger.info(f"Train Size: {len(train_loader.dataset)} , Val Size: {len(val_loader.dataset)} , Test Size: {len(test_loader.dataset)}")

            fold_best_test_acc, fold_best_test_epoch = 0.0, 0
            for epoch in range(epochs):
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
                    if epoch_test_acc >= fold_best_test_acc:
                        fold_best_test_acc, fold_best_test_epoch = epoch_test_acc, epoch+1
                    logger.info(f"# Fold: {fold}, Epoch: {epoch+1:04d} | Train_Loss: {epoch_train_loss:.4f} | Val_Loss: {epoch_val_loss:.4f} , Val_Acc: {epoch_val_acc:.4f} | Test_Acc: {epoch_test_acc:.4f}")
                else:
                    logger.info(f"# Fold: {fold}, Epoch: {epoch+1:04d} | Train_Loss: {epoch_train_loss:.4f} | Val_Loss: {epoch_val_loss:.4f} , Val_Acc: {epoch_val_acc:.4f}")

            logger.info(f"=============== Fold: {fold} , Start Evaluating ===============")
            if not test_mode:
                _, test_acc = evaluate_with_dataloader(model, test_loader, device=device)
                fold_best_test_acc, fold_best_test_epoch = test_acc, epoch+1
                logger.info(f"Test Accuracy: {test_acc:.4f}\n")
            else:
                logger.info(f"Best Test Accuracy: {fold_best_test_acc:.4f} , Best Test Epoch: {fold_best_test_epoch}\n")

            test_accs.append(fold_best_test_acc)
            if fold_best_test_acc >= best_test_acc:
                best_fold, best_test_acc, best_test_epoch = fold+1, fold_best_test_acc, fold_best_test_epoch
            

        logger.info(f"=============== Best Experiment Results ===============")
        logger.info(f"Test Accs of All Fold: {[round(a, 4) for a in test_accs]}")  
        logger.info(f"Best Fold: {best_fold:2d}, Best Test Accuracy: {best_test_acc:.4f} , Best Test Epoch: {best_test_epoch}")
        test_accs = sorted(test_accs)
        logger.info(f"Acc Statistic: min={test_accs[0]:.4f} , max={test_accs[-1]:.4f}")
        medium_index = int((folds-1)/2)
        logger.info(f"Acc Statistic: medium_l={test_accs[medium_index]:.4f} , medium_r={test_accs[-medium_index]:.4f} , medium={(test_accs[medium_index]+test_accs[-medium_index])/2:.4f}")
        logger.info(f"Average Accs of {folds} Folds: {sum(test_accs)/folds:.4f}\n\n")