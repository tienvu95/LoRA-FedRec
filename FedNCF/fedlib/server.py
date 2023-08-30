from typing import List, Tuple
from .client import Client, NCFClient
import torch.nn
import numpy as np
import random
import torch
import tqdm
import logging
from fedlib.data import FedDataModule 
import rec.evaluate as evaluate
from stats import TimeStats


class SimpleAvgAggregator:
    def __init__(self, sample_params) -> None:
       self.aggregated_params = [torch.zeros_like(p) for p in sample_params]
       self.count = 0
        
    def collect(self, params, weight=1):
       self.aggregated_params = [(p0 + p1*weight) for p0, p1 in zip(self.aggregated_params, params)]
       self.count += weight
    
    def finallize(self):
        return [p / self.count for p in self.aggregated_params]

class SimpleServer:
    def __init__(self, clients: List[Client], cfg, model, datamodule: FedDataModule):
        self.client_set = clients
        self.model = model
        _, self.server_params = self.model._get_splited_params(compress=False)
        self.cfg = cfg
        self.datamodule = datamodule
        self._circulated_client_count = 0
        self._timestats = TimeStats()
        random.seed(cfg.EXP.seed)
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
        return sample

    @torch.no_grad()
    def _prepare_global_params(self):
        if 'lora' in self.model.__class__.__name__.lower() and self.model.freeze_B:
            self.model._reinit_B()
            _, self.server_params = self.model._get_splited_params(keep_B=True, merge_weights=False)

    def train_round(self, epoch_idx: int = 0):
        participants: List[Client] = self.sample_clients()
        total_loss = 0

        self._prepare_global_params()
        # print(self.embed_item_GMF.lora_B)
        self._timestats.set_aggregation_epoch(epoch_idx)
        pbar = tqdm.tqdm(participants, desc='Training')
        aggregator = SimpleAvgAggregator(self.server_params['weights'])
        for client in pbar:
            # Prepare client dataset
            train_loader = self.datamodule.train_dataloader([client.cid])
            client_params, data_size, metrics = client.fit(train_loader, self.server_params, self.cfg, self.cfg.TRAIN.device, self._timestats)
            aggregator.collect(client_params['weights'], weight=data_size)
            client_loss = np.mean(metrics['loss'])
            log_dict = {"client_loss": client_loss}
            total_loss += client_loss
            pbar.set_postfix(log_dict)
        updated_weight = aggregator.finallize()
        self.server_params['weights'] = updated_weight

        self.model._set_state_from_splited_params([self.sorted_client_set[0]._private_params, self.server_params])
        return {"train_loss": total_loss / len(participants)}

    
    @torch.no_grad()
    def evaluate(self, test_loader):
        self._timestats.mark_start("evaluate")
        sorted_client_set = self.sorted_client_set
        eval_model = self.model.merge_client_params(sorted_client_set, self.server_params, self.model, self.cfg.TRAIN.device)
        eval_model.eval()
        HR, NDCG = evaluate.metrics(eval_model, test_loader, self.cfg.EVAL.topk, device=self.cfg.TRAIN.device)
        self._timestats.mark_end("evaluate")
        return {"HR": HR, "NDCG": NDCG}

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
