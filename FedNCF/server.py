from typing import List, Any, Tuple
from client import Client, NCFClient
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
from models import FedNCFModel, FedLoraNCF
from data import FedMovieLen1MDataset
import evaluate
from stats import TimeStats

class SimpleAvgAggregator:
    def __init__(self, sample_params) -> None:
       self.aggregated_params = [np.zeros_like(p) for p in sample_params]
       self.count = 0
        
    def collect(self, params, weight=1):
       self.aggregated_params = [(p0 + p1*weight) for p0, p1 in zip(self.aggregated_params, params)]
       self.count += weight
    
    def finallize(self):
        return [p / self.count for p in self.aggregated_params]

class SimpleServer:
    def __init__(self, clients: List[Client], cfg, model: FedNCFModel, train_dataset: FedMovieLen1MDataset):
        self.client_set = clients
        self.model = model
        _, self.server_params = self.model._get_splited_params()
        self.cfg = cfg
        self.train_dataset = train_dataset
        self._circulated_client_count = 0
        self._timestats = TimeStats()
        random.shuffle(self.client_set)
        self.sorted_client_set = sorted(self.client_set, key=lambda t: t.cid)


    def sample_clients(
        self,
    ) -> Tuple[List[Client], List[Client]]:
        """
        :param clients: list of all available clients
        :param num_clients: number of clients to sample

        sample `num_clients` clients and return along with their respective data
        """
        num_clients = self.cfg.FED.num_clients
        sample = self.client_set[:num_clients]
        # rotate the list by `num_clients`
        self.client_set =  self.client_set[num_clients:] + sample

        self._circulated_client_count += num_clients
        if self._circulated_client_count >= len(self.client_set):
            logging.info("Resample negative items")
            self.train_dataset.sample_negatives()
            self._circulated_client_count = 0

        return sample

    def train_round(self):
        participants: List[Client] = self.sample_clients()
        pbar = tqdm.tqdm(participants, desc='Training')
        total_loss = 0
        aggregator = SimpleAvgAggregator(self.server_params['weights'])
        for client in pbar:
            # Prepare client dataset
            self.train_dataset.set_client(client.cid)
            train_loader = DataLoader(self.train_dataset, **self.cfg.DATALOADER)

            # Fit client model
            client_params, data_size, metrics = client.fit(train_loader, self.server_params, self.cfg, self.cfg.TRAIN.device, self._timestats)
            
            aggregator.collect(client_params['weights'], weight=data_size)
            client_loss = np.mean(metrics['loss'])
            log_dict = {"client_loss": client_loss}
            total_loss += client_loss

            pbar.set_postfix(log_dict)
        self.server_params['weights'] = aggregator.finallize()
        return {"train_loss": total_loss / len(participants)}

    
    @torch.no_grad()
    def evaluate(self, test_loader):
        self._timestats.mark_start("evaluate")
        # sorted_client_set = sorted(self.client_set, key=lambda t: t.cid)
        sorted_client_set = self.sorted_client_set
        # print(sorted_client_set[0].cid, sorted_client_set[1].cid, sorted_client_set[-1].cid)
        client_weights = [c._private_params['weights'] for c in sorted_client_set]
        client_weights = [torch.tensor(np.concatenate(w, axis=0)).to(cfg.TRAIN.device) for w in zip(*client_weights)]
        client_weights = {k: v for k,v in zip(sorted_client_set[0]._private_params['keys'], client_weights)}
        eval_model = copy.deepcopy(self.model)
        eval_model._set_state_from_splited_params([sorted_client_set[0]._private_params, self.server_params])
        eval_model.embed_user_GMF = torch.nn.Embedding.from_pretrained(client_weights['embed_user_GMF.weight'])
        eval_model.embed_user_MLP = torch.nn.Embedding.from_pretrained(client_weights['embed_user_MLP.weight'])
        # evaluate the model
        # eval_model = self.model
        eval_model.eval()
        HR, NDCG = evaluate.metrics(eval_model, test_loader, cfg.EVAL.topk, device=self.cfg.TRAIN.device)
        self._timestats.mark_end("evaluate")
        return {"HR": HR, "NDCG": NDCG}

def run_server(
    cfg,
) -> pd.DataFrame:

    ############################## PREPARE DATASET ##########################
    train_dataset = FedMovieLen1MDataset(cfg.DATA.root, train=True, num_negatives=cfg.DATA.num_negatives)
    test_dataset = FedMovieLen1MDataset(cfg.DATA.root, train=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.DATA.test_num_ng+1, shuffle=False, num_workers=0)
    # define server side model
    logging.info("Init model")
    if cfg.MODEL.use_lora:
        logging.info("use lora model")
        model = FedLoraNCF(
                train_dataset.num_items,
                factor_num=cfg.MODEL.factor_num,
                num_layers=cfg.MODEL.num_layers,
                dropout=cfg.MODEL.dropout,
                lora_rank=cfg.MODEL.lora_r,
                lora_alpha=cfg.MODEL.lora_alpha,
            )
    else:
        model = FedNCFModel(
            train_dataset.num_items,
            factor_num=cfg.MODEL.factor_num,
            num_layers=cfg.MODEL.num_layers,
            dropout=cfg.MODEL.dropout,
            # user_num=train_dataset.num_users,
        )
    # summary(server_model, *[torch.LongTensor((1,1)), torch.LongTensor((1,1)), None], layer_modules=(lora.Embedding, torch.nn.Parameter))
    model.to(cfg.TRAIN.device)
    logging.info("Init clients")
    clients = initialize_clients(cfg, model, train_dataset.num_users)
    logging.info("Init server")
    server = SimpleServer(clients, cfg, model, train_dataset)
    hist = []
    for epoch in range(cfg.FED.aggregation_epochs):
        log_dict = {"epoch": epoch}
        logging.info(f"Aggregation Epoch: {epoch}")
        log_dict.update(server.train_round())
        logging.info("Evaluate model")
        test_metrics = server.evaluate(test_loader)
        log_dict.update(test_metrics)
        log_dict.update(server._timestats._time_dict)
        server._timestats.reset()
        hist.append(log_dict)
        logging.info(log_dict)
    hist_df = pd.DataFrame(hist)
    return hist_df

def initialize_clients(cfg, model, num_users) -> List[Client]:
    """
    creates `Client` instance for each `client_id` in dataset
    :param dataset: `Dataset` object to load train data
    :return: list of `Client` objects
    """
    clients = list()
    for client_id in range(num_users):
        c = NCFClient(client_id, model=model)
        model._reinit_private_params()
        clients.append(c)
    return clients

def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

if __name__ == '__main__':
    # from pyrootutils import setup_root
    # setup_root(__file__, ".git", pythonpath=True)
    import datetime
    from config import setup_cfg, get_parser
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    out_dir = Path(cfg.EXP.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logfilename = os.path.join(out_dir, current_time+'.txt')
    initLogging(logfilename)
    logging.info(cfg.dump())

    hist_df = run_server(cfg)
    hist_df.to_csv(out_dir / "hist.csv", index=False)