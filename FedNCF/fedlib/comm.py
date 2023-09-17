from typing import Dict, List, Tuple
import random
from fedlib.client import Client
from fedlib.standard.models import TransferedParams

class ClientSampler:
    def __init__(self, num_users) -> None:
        # self._client_set = client_set
        self.num_users = num_users
        self._round_count = 0
        self._client_count = 0
    
    def initialize_clients(self, model, dm, loss_fn, shuffle_seed, reinit=True) -> None:
        """
        creates `Client` instance for each `client_id` in dataset
        :param cfg: configuration dict
        :return: list of `Client` objects
        """
        clients = list()
        for client_id in range(self.num_users):
            c = Client(client_id, model=model, datamodule=dm, loss_fn=loss_fn)
            if reinit:
                model._reinit_private_params()
            clients.append(c)
        self._client_set = clients
        self._suffle_client_set(shuffle_seed)

    def _suffle_client_set(self, seed):
        random.seed(seed)
        random.shuffle(self._client_set)
        self.sorted_client_set = sorted(self._client_set, key=lambda t: t.cid)

    def next_round(self, num_clients) -> List[Client]:
        sample = self._client_set[:num_clients]
        # rotate the list by `num_clients`
        self._client_set =  self._client_set[num_clients:] + sample
        self._client_count += num_clients
        self._round_count += 1
        return sample

class AvgAggregator:
    def __init__(self, sample_params: TransferedParams, strategy='fedavg') -> None:
       self.aggregated_params = sample_params.zero_likes()
       self.count = 0
       self.strategy = strategy
        
    def collect(self, params: TransferedParams, weight=1):
        params.decompress()
        if self.strategy == 'fedavg':
            self.aggregated_params = self.aggregated_params.add_(params, alpha=weight)
            self.count += weight
        elif self.strategy == 'simpleavg':
            self.aggregated_params = self.aggregated_params.add_(params)
            self.count += 1
        else:
            raise NotImplementedError(f'Aggregation strategy {self.strategy} not implemented')

    def finallize(self):
        return self.aggregated_params.div_scalar_(self.count)