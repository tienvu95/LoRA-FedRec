from typing import Dict, List, Tuple
import random
from fedlib.client import Client
from fedlib.standard.models import TransferedParams
from multiprocessing import Process, Queue
import pickle


def _prepare_dataloader(participants, pid, n_workers, queue):
    i = pid
    step_size = n_workers
    n_participants = len(participants)
    while True:
        client = participants[i % n_participants]
        # print(f'Preparing client {client.cid}')
        train_loader = client.prepare_dataloader_mp(None)
        train_loader = pickle.dumps(train_loader)
        queue.put(train_loader)
        del train_loader   # save memory
        i += step_size

    # for cid in range(pid, len(participants), n_workers):
    #     client = participants[cid]
    #     # print(f'Preparing client {client.cid}')
    #     train_loader = client.prepare_dataloader_mp(None)
    #     queue.put(train_loader)

class ClientSampler:
    def __init__(self, num_users, n_workers=1) -> None:
        # self._client_set = client_set
        self.num_users = num_users
        self._round_count = 0
        self._client_count = 0
        self._n_workers = n_workers
    
    def initialize_clients(self, model, dm, loss_fn, shuffle_seed, reinit=True, central_train=False) -> None:
        """
        creates `Client` instance for each `client_id` in dataset
        :param cfg: configuration dict
        :return: list of `Client` objects
        """
        clients = list()
        for client_id in range(self.num_users):
            c = Client(client_id, model=model, datamodule=dm, loss_fn=loss_fn, central_train=central_train)
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
        participants = self._client_set[:num_clients]
        # rotate the list by `num_clients`
        self._client_set =  self._client_set[num_clients:] + participants
        self._client_count += num_clients
        self._round_count += 1

        total_ds_sizes = 0
        for i in range(len(participants)):
            train_loader = self.queue.get()
            train_loader = pickle.loads(train_loader)
            participants[i].train_loader = train_loader
            total_ds_sizes += len(train_loader.dataset)

        return participants, total_ds_sizes

    def prepare_dataloader(self, n_clients_per_round) -> None:
        self.processors = []
        self.queue = Queue(maxsize=n_clients_per_round)
        for i in range(self._n_workers):
            print(f'Starting worker {i}')
            process = Process(
                target=_prepare_dataloader,
                args=(self._client_set, i, self._n_workers, self.queue)
            )
            self.processors.append(process)
            process.daemon = True
            process.start()
            # process.join()
        # total_ds_sizes = 0
        # for i in range(len(participants)):
        #     train_loader = queue.get()
        #     participants[i].train_loader = train_loader
        #     total_ds_sizes += len(train_loader.dataset)
    
    def close(self):
        for process in self.processors:
            process.terminate()
            process.join()

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