import os
import time
import numpy as np
import hydra
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from rec.models import NCF
import rec.evaluate as evaluate
from fedlib.data import FedDataModule
from tqdm import tqdm
import random
from stats import TimeStats, Logger
from fedlib.comm import ClientSampler



cudnn.benchmark = True
device = torch.device("cuda")
os.environ['EXP_DIR'] = str(Path.cwd())

@hydra.main(config_path=str(Path.cwd() / 'configs'), config_name='centraltrain.yaml', version_base="1.2")
def main(cfg):
    print(cfg)
    feddm = FedDataModule(cfg)
    feddm.setup()
    num_items = feddm.num_items
    num_users = feddm.num_users
    all_train_loader = feddm.train_dataloader()
    test_loader = feddm.test_dataloader()
    print("Num users", num_users)
    print("Num items", num_items)


    model = hydra.utils.instantiate(cfg.net.init, 
                                     item_num=num_items, 
                                    user_num=num_users)
    model.to(device)
    loss_function = nn.BCEWithLogitsLoss(reduction='sum')

    ########################### TRAINING #####################################
    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.lr)
    train_loader = all_train_loader
    print("len dataset", len(train_loader.dataset))
    print("len loader", len(train_loader))

    global_step = 0
    mylogger = Logger(cfg, model, wandb=cfg.TRAIN.wandb)
    timestat = TimeStats()

    pbar = tqdm(range(cfg.TRAIN.num_epochs))
    
    # client_sampler = ClientSampler(num_users)
    # client_sampler.initialize_clients(model, feddm, loss_fn=None, shuffle_seed=42, reinit=False)
    def train_epoch(train_loader, optimizer):
        model.train()	
        total_loss = 1
        count = 0
        log_dict = {"epoch": epoch}
        training_step_pbar = tqdm(train_loader, leave=False, disable=True)
        for user, item, label in training_step_pbar:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)

            optimizer.zero_grad()
            prediction = model(user, item, mask_zero_user_index=False)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # global_step += 1
            count += 1
        total_loss /= count
        log_dict['loss'] = total_loss
        return log_dict
    
    for epoch in pbar:
        # total_loss = 0
        # count = 0
        # log_dict = {"epoch": epoch}

        # for i in range(num_users):
        #     with timestat.timer("prepare data"):
        #         train_loader = feddm.train_dataloader(cid=[i])
            
            # count += 1
            # client = client_sampler.next_round(1)[0]
            # if cfg.TRAIN.optimizer == 'sgd':
            #     optimizer = torch.optim.SGD(client._model.parameters(), lr=cfg.TRAIN.lr)
            # else:
            #     raise ValueError("Optimizer not supported")
            # # optimizer = torch.optim.Adam(client._model.parameters(), lr=cfg.TRAIN.lr, weight_decay=cfg.TRAIN.weight_decay)
            # metrics = client._fit(train_loader, optimizer, loss_function, num_epochs=1, device=device, mask_zero_user_index=False)
            # total_loss += np.mean(metrics['loss'])

        # total_loss /= count
        # log_dict.update({"loss": total_loss})
        log_dict = train_epoch(train_loader, optimizer)
        with torch.no_grad():
            model.eval()
            HR, NDCG = evaluate.metrics(model, test_loader, cfg.EVAL.topk, device=device)
            # print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
        metrics = {"HR": np.mean(HR), "NDCG": np.mean(NDCG), "loss": log_dict['loss']}
        pbar.set_postfix(metrics)
        log_dict.update(metrics)
        mylogger.log(log_dict, term_out=True)
        print(timestat._time_dict)
        timestat.reset()
        
main()