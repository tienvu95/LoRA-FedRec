from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from abc import ABC, abstractmethod
from .models import FedNCFModel

class Client(ABC):
    @abstractmethod
    def get_parameters(self, config):
        pass

    @abstractmethod
    def set_parameters(self, config):
        pass

    @abstractmethod
    def fit(self, train_loader: DataLoader,  parameters: List[np.ndarray], config: Dict[str, str], device):
        pass

class NCFClient(Client):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        cid,
        model: FedNCFModel,
    ) -> None:
        self._cid = cid
        self._model = model
        self._private_params = self._model._get_splited_params()[0]
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    @property
    def cid(self):
        return self._cid

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        private_params, sharable_params = self._model._get_splited_params()
        # print(self._private_params['keys'])
        # print("Update on private params", private_params['weights'][0].shape, np.linalg.norm(private_params['weights'][1] - self._private_params['weights'][1]))
        # print("Update on private params", private_params['weights'][1].shape, np.linalg.norm(private_params['weights'][0] - self._private_params['weights'][0]))
        self._private_params = private_params

        return sharable_params

    def set_parameters(self, global_params: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self._model._set_state_from_splited_params([self._private_params, global_params])

    def fit(
        self, train_loader: DataLoader,  server_params: List[np.ndarray], config: Dict[str, str], device, timestats
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        with torch.no_grad():
            timestats.mark_start("set_parameters")
            self.set_parameters(server_params)
            timestats.mark_end("set_parameters")
            optimizer = torch.optim.Adam(self._model.parameters(), lr=config.TRAIN.lr)
            # optimizer = torch.optim.SGD(self._model.parameters(), lr=config.TRAIN.lr)

        timestats.mark_start("fit")
        metrics = self._fit(train_loader, optimizer, self.loss_fn, num_epochs=config.FED.local_epochs, device=device)
        timestats.mark_end("fit")

        timestats.mark_start("get_parameters")
        sharable_params = self.get_parameters(None)
        timestats.mark_end("get_parameters")

        if timestats is not None:
            name2id = {n: i for i, n in enumerate(sharable_params['keys'])}
            id1 = name2id['embed_item_GMF.weight']
            id2 = name2id['embed_item_MLP.weight']
            timestats.count_num_important_component(self._cid, 'embed_item_GMF.weights', sharable_params['weights'][id1] - server_params['weights'][id1])
            timestats.count_num_important_component(self._cid, 'embed_item_MLP.weights', sharable_params['weights'][id2] - server_params['weights'][id2])

        # name2id = {n: i for i, n in enumerate(server_params['keys'])}
        # id1 = name2id['embed_item_GMF.lora_B']
        # print(torch.abs(sharable_params['weights'][id1]).sum())
        # print(torch.abs(sharable_params['weights'][id1] - server_params['weights'][id1]).sum())
        # id1 = name2id['embed_item_MLP.lora_B']
        # print(torch.abs(sharable_params['weights'][id1]).sum())
        # print(torch.abs(sharable_params['weights'][id1] - server_params['weights'][id1]).sum())

        return sharable_params, len(train_loader.dataset), metrics

    # def evaluate(
    #     self, parameters: List[np.ndarray], config: Dict[str, str]
    # ) -> Tuple[float, int, Dict]:
    #     # Set model parameters, evaluate model on local test dataset, return result
    #     self.set_parameters(parameters)
    #     return

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