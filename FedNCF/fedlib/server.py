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
        # self._dummy_private_params, self.server_params = self.model._get_splited_params(server_init=True)
        
        self._timestats = TimeStats()
    
    def _step_server_optim(self, delta_params):
        self.server_params.server_step_(delta_params)
        self.model._set_state_from_splited_params([self._dummy_private_params, self.server_params])
    
    def prepare(self):
        # reinit
        self.model.server_prepare()
        self._dummy_private_params, self.server_params = self.model._get_splited_params(server_init=True)


    def train_round(self, epoch_idx: int = 0):
        self.prepare()
        participants = self.client_sampler.next_round(self.cfg.FED.num_clients)
        aggregator = AvgAggregator(self.server_params)
        total_loss = 0

        if epoch_idx > 0:
            self.server_params.compress(**self.cfg.FED.compression_kwargs)
            self.server_params.decompress()

        self._timestats.set_aggregation_epoch(epoch_idx)
        pbar = tqdm.tqdm(participants, desc='Training')
        update_numel = 0
        all_data_size = 0
        # B_0 = self.server_params['embed_item_GMF.lora_B'].clone()
        for client in pbar:
            update, data_size, metrics = client.fit(self.server_params, self.cfg, self.cfg.TRAIN.device, self._timestats)
            update_numel += sum([t.numel() for t in update.values()])
            aggregator.collect(update, weight=data_size)
            
            client_loss = np.mean(metrics['loss'])
            log_dict = {"client_loss": client_loss}
            total_loss += client_loss
            all_data_size += data_size
            pbar.set_postfix(log_dict)
        aggregated_update = aggregator.finallize()
        self._step_server_optim(aggregated_update)
        # B_1 = self.server_params['embed_item_GMF.lora_B'].clone()
        # print(torch.linalg.norm(B_1 - B_0))

        return {"client_loss": total_loss / len(participants), "update_numel": update_numel / len(participants), "data_size": all_data_size}

    
    @torch.no_grad()
    def evaluate(self, test_loader, train_loader=None):
        sorted_client_set = self.client_sampler.sorted_client_set
        with self._timestats.timer("evaluate"):
            eval_model = self.model.merge_client_params(sorted_client_set, self.server_params, self.model, self.cfg.TRAIN.device)
            if train_loader is not None:
                train_loss = evaluate.cal_loss(eval_model, train_loader, loss_function=torch.nn.BCEWithLogitsLoss(),device=self.cfg.TRAIN.device)
            eval_model.eval()
            HR, NDCG = evaluate.metrics(eval_model, test_loader, self.cfg.EVAL.topk, device=self.cfg.TRAIN.device)
        return {"HR": HR, "NDCG": NDCG, "train_loss": train_loss}
