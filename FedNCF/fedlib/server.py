from typing import List, Tuple
from .client import Client, Client
import torch.nn
import numpy as np
import random
import torch
import tqdm
import logging
import rec.evaluate as evaluate
from stats import TimeStats
from fedlib.comm import AvgAggregator, ClientSampler

class SimpleServer:
    def __init__(self, cfg, model, client_sampler: ClientSampler):
        self.cfg = cfg
        self.client_sampler = client_sampler
        
        self.model = model
        self._dummy_private_params, self.server_params = self.model._get_splited_params()
        
        self._timestats = TimeStats()
    
    def _step_server_optim(self, delta_params):
        self.server_params.add_(delta_params)
        self.model._set_state_from_splited_params([self._dummy_private_params, self.server_params])

    def train_round(self, epoch_idx: int = 0):
        participants = self.client_sampler.next_round(self.cfg.FED.num_clients)
        aggregator = AvgAggregator(self.server_params)
        total_loss = 0

        self._timestats.set_aggregation_epoch(epoch_idx)
        pbar = tqdm.tqdm(participants, desc='Training')
        for client in pbar:
            update, data_size, metrics = client.fit(self.server_params, self.cfg, self.cfg.TRAIN.device, self._timestats)
            aggregator.collect(update, weight=data_size)
            
            client_loss = np.mean(metrics['loss'])
            log_dict = {"client_loss": client_loss}
            total_loss += client_loss
            pbar.set_postfix(log_dict)
        aggregated_update = aggregator.finallize()
        self._step_server_optim(aggregated_update)

        return {"train_loss": total_loss / len(participants)}

    
    @torch.no_grad()
    def evaluate(self, test_loader):
        self._timestats.mark_start("evaluate")
        sorted_client_set = self.client_sampler.sorted_client_set
        eval_model = self.model.merge_client_params(sorted_client_set, self.server_params, self.model, self.cfg.TRAIN.device)
        eval_model.eval()
        HR, NDCG = evaluate.metrics(eval_model, test_loader, self.cfg.EVAL.topk, device=self.cfg.TRAIN.device)
        self._timestats.mark_end("evaluate")
        return {"HR": HR, "NDCG": NDCG}
