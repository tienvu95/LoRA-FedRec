from typing import Dict, List, Any, Optional, Tuple
import hydra
from omegaconf import OmegaConf
import torch.nn
import os
import pandas as pd
import numpy as np
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
import copy
import tqdm
import logging
import rec
from stats import TimeStats, Logger
import torch.nn.functional as F
import fedlib
from fedlib.comm import ClientSampler

import wandb

os.environ['EXP_DIR'] = str(Path.cwd())

def run_server(
    cfg,
) -> pd.DataFrame:

    ############################## PREPARE DATASET ##########################
    feddm = fedlib.data.FedDataModule(cfg)
    feddm.setup()
    num_items = feddm.num_items
    num_users = feddm.num_users
    all_train_loader = feddm.train_dataloader(for_eval=True)
    
    val_loader = feddm.val_dataloader()
    test_loader = feddm.test_dataloader()

    logging.info("Num users: %d" % num_users)
    logging.info("Num items: %d" % num_items)
    
    # define server side model
    logging.info("Init model")
    model = hydra.utils.instantiate(cfg.net.init, item_num=num_items)
    mylogger = Logger(cfg, model, wandb=cfg.TRAIN.wandb)
    
    model.to(cfg.TRAIN.device)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    # loss_fn = torch.nn.BCEWithLogitsLoss()

    logging.info("Init clients")
    client_sampler = ClientSampler(feddm.num_users, n_workers=1)
    client_sampler.initialize_clients(model, feddm, loss_fn=loss_fn, shuffle_seed=42)
    client_sampler.prepare_dataloader(n_clients_per_round=cfg.FED.num_clients*10)

    logging.info("Init server")
    server = fedlib.server.SimpleServer(cfg, model, client_sampler)

    for epoch in range(cfg.FED.agg_epochs):
        log_dict = {"epoch": epoch}
        log_dict.update(server.train_round(epoch_idx=epoch))
        if (cfg.EVAL.interval > 0) and ((epoch % cfg.EVAL.interval == 0) or (epoch == cfg.FED.agg_epochs - 1)):                
            test_metrics = server.evaluate(val_loader, test_loader, train_loader=all_train_loader)
            log_dict.update(test_metrics)

        log_dict.update(server._timestats._time_dict)
        if (epoch % cfg.TRAIN.log_interval == 0) or (epoch == cfg.FED.agg_epochs - 1):
            mylogger.log(log_dict, term_out=True)
        # server._timestats.reset()
    client_sampler.close()
    hist_df = mylogger.finish(quiet=True)
    pca_var_df = pd.DataFrame(data=server._timestats._pca_vars)
    return hist_df, pca_var_df, log_dict

def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        logging.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name]
    logging.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

@hydra.main(config_path=str(Path.cwd() / 'configs'), config_name='fedtrain.yaml', version_base="1.2")
def main(cfg):
    OmegaConf.resolve(cfg)
    logging.info(cfg)
    out_dir = Path(cfg.paths.output_dir)
    hist_df, pca_var_df, log_dict = run_server(cfg)
    hist_df.to_csv(out_dir / "hist.csv", index=False)
    pca_var_df.to_csv(out_dir / "pca_var.csv", index=False)
    
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=log_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value

if __name__ == '__main__':
    main()