from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .standard.models import FedNCFModel, TransferedParams
from fedlib.data import FedDataModule
from stats import TimeStats

class Client:
    def __init__(
        self,
        cid,
        model: FedNCFModel,
        datamodule,
        loss_fn,
        central_train=False,
    ) -> None:
        self._cid = cid
        self.datamodule = datamodule
        self._model = model
        if not central_train:
            self._private_params = self._model._get_splited_params()[0]
        self.loss_fn = loss_fn

    @property
    def cid(self):
        return self._cid

    def get_parameters(self, config, old_shared_params) -> TransferedParams:
        private_params, sharable_params = self._model._get_splited_params(old_shared_params=old_shared_params)
        self._private_params = private_params
        return sharable_params

    def set_parameters(self, global_params: List[np.ndarray]) -> None:
        self._model._set_state_from_splited_params([self._private_params, global_params])
    
    
    def prepare_dataloader_mp(self, config):
        # print(f'*Preparing client {self.cid}')
        train_loader = self.datamodule.train_dataloader([self.cid])
        return train_loader
    
    def prepare_dataloader(self, config):
        self.train_loader = self.datamodule.train_dataloader([self.cid])
        return len(self.train_loader.dataset)

    def fit(
        self, 
        server_params: TransferedParams, 
        local_epochs: int,
        config: Dict[str, str], 
        device, 
        stats_logger: TimeStats,
        **forward_kwargs
    ) -> Tuple[TransferedParams, int, Dict]:
        # Preparing train dataloader
        try:
            train_loader = self.train_loader
        except AttributeError as e:
            print("Please call prepare_dataloader() first. CID: %d" % self._cid)
            raise e

        # Set model parameters, train model, return updated model parameters
        with torch.no_grad():
            if server_params is not None:
                with stats_logger.timer('set_parameters'):
                    self.set_parameters(server_params)
        # item_emb_params, params_1  = self._model._get_splited_params_for_optim()
        # opt_params = {
        #     [
        #         {'params': item_emb_params, 'lr': config.TRAIN.lr*train_loader.batch_size},
        #         {'params': params_1, 'lr': config.TRAIN.lr},
        #     ]
        # }
        opt_params = self._model.parameters()
        if config.TRAIN.optimizer == 'sgd':
            optimizer = torch.optim.SGD(opt_params, lr=config.TRAIN.lr, weight_decay=config.TRAIN.weight_decay)
        elif config.TRAIN.optimizer == 'adam':
            optimizer = torch.optim.Adam(opt_params, lr=config.TRAIN.lr, weight_decay=config.TRAIN.weight_decay)


        with stats_logger.timer('fit'):
            metrics = self._fit(train_loader, optimizer, self.loss_fn, num_epochs=local_epochs, device=device, **forward_kwargs)
        
        with torch.no_grad():
            with stats_logger.timer('get_parameters'):
                sharable_params = self.get_parameters(None, server_params)
                update = None
                if server_params is not None:
                    update = sharable_params.diff(server_params)

                    with stats_logger.timer('compress'):
                        update.compress(**config.FED.compression_kwargs)

        # stats_logger.stats_transfer_params(cid=self._cid, stat_dict=self._model.stat_transfered_params(update))
        return update, len(train_loader.dataset), metrics
    
    def _fit(self, train_loader, optimizer, loss_fn, num_epochs, device, **forward_kwargs):
        self._model.train() # Enable dropout (if have).
        loss_hist = []
        for e in range(num_epochs):
            total_loss = 0
            count_example = 0
            for user, item, label in train_loader:
                user = user.to(device)
                item = item.to(device)
                label = label.float().to(device)

                optimizer.zero_grad()
                prediction = self._model(user, item, **forward_kwargs)
                loss = loss_fn(prediction, label)
                loss.backward()
                optimizer.step()

                count_example += 1
                total_loss += loss.item()
            total_loss /= count_example
            loss_hist.append(total_loss)
        return {
            "loss": loss_hist
        }