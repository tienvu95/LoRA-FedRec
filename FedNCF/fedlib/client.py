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
        datamodule
    ) -> None:
        self._cid = cid
        self.datamodule = datamodule
        self._model = model
        self._private_params = self._model._get_splited_params()[0]
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    @property
    def cid(self):
        return self._cid

    def get_parameters(self, config, old_shared_params) -> TransferedParams:
        private_params, sharable_params = self._model._get_splited_params(old_shared_params=old_shared_params)
        self._private_params = private_params
        return sharable_params

    def set_parameters(self, global_params: List[np.ndarray]) -> None:
        self._model._set_state_from_splited_params([self._private_params, global_params])

    def fit(
        self, server_params: TransferedParams, config: Dict[str, str], device, stats_logger: TimeStats
    ) -> Tuple[TransferedParams, int, Dict]:
        # Preparing train dataloader
        train_loader = self.datamodule.train_dataloader([self.cid])

        # Set model parameters, train model, return updated model parameters
        with torch.no_grad():
            with stats_logger.timer('set_parameters'):
                self.set_parameters(server_params)
            # param_groups = self._model._get_splited_params_for_optim()
            # optimizer = torch.optim.Adam([{'params': list(param_groups[0].values()),},
            #                               {'params': list(param_groups[1].values()), 'lr': config.TRAIN.lr*train_loader.batch_size}, ], lr=config.TRAIN.lr)
            optimizer = torch.optim.Adam(self._model.parameters(), lr=config.TRAIN.lr)
            # optimizer = torch.optim.SGD(self._model.parameters(), lr=config.TRAIN.lr)

        with stats_logger.timer('fit'):
            metrics = self._fit(train_loader, optimizer, self.loss_fn, num_epochs=config.FED.local_epochs, device=device)
        
        with torch.no_grad():
            with stats_logger.timer('get_parameters'):
                sharable_params = self.get_parameters(None, server_params)
                update = sharable_params.diff(server_params)

            with stats_logger.timer('compress'):
                update.compress(**config.FED.compression_kwargs)

        # timestats.stats_transfer_params(cid=self._cid, stat_dict=self._model.stat_transfered_params(update))
        return update, len(train_loader.dataset), metrics
    
    def _fit(self, train_loader, optimizer, loss_fn, num_epochs, device):
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
                prediction = self._model(user, item)
                loss = loss_fn(prediction, label)
                loss.backward()
                # tmp = self._model.embed_item_GMF.lora_A.detach().clone()
                # print(self._model.embed_user_GMF.weight.grad.data)
                # tmp = self._model.embed_item_GMF.lora_A.grad.data
                # print("lora_A grad", torch.norm(tmp))
                # grad_gmf_A = self._model.embed_item_GMF.lora_A.grad.data.detach().clone()
                # grad_gmf_B = self._model.embed_item_GMF.lora_B.grad
                # print(grad_gmf_B, self._model.embed_item_MLP.lora_B.grad)
                # print("gmf", grad_gmf_A.sum().item(), grad_gmf_B.norm().item())

                # self._model.embed_item_GMF.lora_B.grad.data.zero_()
                # self._model.embed_item_MLP.lora_B.grad.data.zero_()

                optimizer.step()
                # print(torch.linalg.norm(self._model.embed_user_GMF.weight.detach().cpu() - torch.tensor(self._private_params['weights'][0])))
                # print("Lora B grad", torch.norm(tmp))

                count_example += label.shape[0]
                total_loss += loss.item()* label.shape[0]
            total_loss /= count_example
            loss_hist.append(total_loss)
        return {
            "loss": loss_hist
        }