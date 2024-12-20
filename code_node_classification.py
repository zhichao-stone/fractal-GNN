import os
import random
import numpy as np
import argparse

import dgl
import torch
import torch.nn.functional as F

from model import GAT
from loading import *
from evaluate import *
from logger import ModelLogger
from utils import add_fractal_covering_matrix



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0005, help="weight decay")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8, help="number of GAT's heads")
    parser.add_argument("--seed", type=int, default=12306)
    parser.add_argument("--data", type=str, default="citeseer", choices=["citeseer", "cora", "pubmed"])
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--fractal", action="store_true")
    parser.add_argument("--fractal_concat", type=str, default="concat", choices=["concat", "sum"])
    parser.add_argument("--max_scale", type=int, default=3, help="max scale of fractal subgraph")
    args = parser.parse_args()


    SEED = args.seed
    EPOCH = args.epoch
    LR = args.lr
    WEIGHT_DECAY = args.wd
    
    ON_MULTIVIEW = args.multiview
    ON_FRACTAL = args.fractal

    DATA = str(args.data).lower()
    SCALES = [i+1 for i in range(args.max_scale)]

    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)  # set seed for cpu
    torch.cuda.manual_seed(SEED)  # set seed for current gpu
    torch.cuda.manual_seed_all(SEED)  # set seed for all gpu
    dgl.random.seed(SEED)

    if ON_FRACTAL:
        method_postfix = "frac"
        if ON_MULTIVIEW:
            method_postfix += "_multiview"
        method_postfix += f"_{args.fractal_concat}"
    else:
        if ON_MULTIVIEW:
            method_postfix = "multivew"
        else:
            method_postfix = "base"

    log_file_name = f"GAT_{method_postfix}_{DATA}"
    logger = ModelLogger(log_file_name, "log", backupCount=7).get_logger()
    logger.info(f"Logging to {log_file_name}.log")


    device = torch.device("cuda")
    if DATA in ["citeseer", "cora", "pubmed"]:
        g, features, labels, train_mask, val_mask, test_mask, num_classes = LOAD_FUNCTION_MAP[DATA](raw_dir=RAW_DIR)
    else:
        raise Exception(f"No loading function for data {DATA}")
    features, labels = features.to(device), labels.to(device)
    logger.info(f"Load Graph Data: {DATA}")

    if ON_FRACTAL:
        add_fractal_covering_matrix(g, scales=SCALES)
        logger.info("Add Fractal Covering Matrix")

    g = g.to(device)

    model = GAT(
        in_dim=features.size()[1], 
        hidden_dim=args.hidden_size, 
        out_dim=num_classes,
        num_heads=args.num_heads,
        scales=SCALES,
        multiview=ON_MULTIVIEW,
        fractal=ON_FRACTAL,
        fractal_concat=args.fractal_concat
    )
    model = model.to(device)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    ### training
    logger.info("=============== Train Argument ===============")
    logger.info(f"Learnig Rate: {LR}")
    logger.info(f"Weight Decay: {WEIGHT_DECAY}")
    logger.info(f"Epochs: {EPOCH}")
    logger.info("=============== Start Training ===============")
    # for epoch in tqdm(range(EPOCH)):
    for epoch in range(EPOCH):
        logits = model(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_train = evaluate(model, g, features, labels, train_mask)
        acc_val = evaluate(model, g, features, labels, val_mask)
        logger.info(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Acc_train {acc_train:.4f} | Acc_Val {acc_val:.4f}")

    ### evaluate
    logger.info("=============== Start Evaluating ===============")
    acc_test = evaluate(model, g, features, labels, test_mask)
    logger.info(f"Test Accuracy {acc_test:.4f}\n")
