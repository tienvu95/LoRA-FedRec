import time
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils import *
from metrics import MetronAtK
import random
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
from collections import OrderedDict
import client as cl
from mlp import MLP, MLPLoRA
import numpy as np
import multiprocessing as mp
import pandas as pd
import torch.multiprocessing as mp

import logging

logger = logging.getLogger(__name__)


class FedAvgStrategy(object):
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def aggregate(self, round_user_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.
        aggregated_params = list(zip(*round_user_params.values()))
        aggregated_params = [sum(p_list)/len(p_list) for p_list in aggregated_params]
        return aggregated_params

class FedLoRAAvgStrategy(object):
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def aggregate(self, round_user_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # aggregate item embedding and score function via averaged aggregation.
        aggregated_params = list(zip(*round_user_params.values()))
        for layer_i in range(len(aggregated_params)):
            aggregated_params[layer_i] = [(lora_A @ lora_B) * scaling for lora_A, lora_B, scaling in aggregated_params[layer_i]]
        aggregated_params = [sum(p_list)/len(p_list) for p_list in aggregated_params]
        return aggregated_params

class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self.crit = torch.nn.BCELoss()
        # mae metric
        self.mae = torch.nn.L1Loss()
        if config['lora']:
            self.strategy = FedLoRAAvgStrategy()
            self.client_class = cl.LoRAClient
        else:
            self.strategy = FedAvgStrategy()
            self.client_class = cl.Client

        self._device = 'cpu'

        # self.mp_manager = mp.Manager()
        self.server_model_param = {}
        self.client_model_params = {}
        self.personalize_params = {}


    def init_clients(self, sample_generator):
        self.clients = {}
        all_train_ratings = sample_generator.train_ratings
        all_negatives = sample_generator.negatives
        for uid in range(self.config['num_users']):
            train_ratings = all_train_ratings[all_train_ratings.userId == uid]
            negatives = all_negatives[all_negatives.userId == uid]['negative_items'].tolist()[0]
            self.clients[uid] = self.client_class(uid, train_ratings, negatives, self.config)

            # user_train_data = [all_train_data[0][uid], all_train_data[1][uid], all_train_data[2][uid]]
            # user_dataloader = instance_user_train_loader(user_train_data, batch_size=self.config['batch_size'])

    def instance_user_train_loader(self, user_train_data):
        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2]))
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

    def sample_client(self):
        if self.config['clients_sample_ratio'] == 1:
            participants = range(self.config['num_users'])
        elif self.config['clients_sample_ratio'] < 1:
            num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
            participants = random.sample(range(self.config['num_users']), num_participants)
        else:
            participants = random.sample(range(self.config['num_users']), self.config['clients_sample_num'])
        return participants

    def fed_train_a_round(self, round_id):
        """train a round."""
        # sample users participating in single round.
        participants = self.sample_client()
        # store users' model parameters of current round.
        server_params = self.client_class.get_server_params(self.model)

        log_dict = {}
        round_participant_params = {}

        def call_client_proc(uid):
            client = self.clients[uid]
            return client.local_train(self.config, 
                                    self.client_model_params.get(uid, None), 
                                    server_params, 
                                    self._device)

        def callback(user_result):
            user = user_result['uid']
            global_params, local_params, personalize_params = user_result['parameters']
            self.client_model_params[user] = local_params
            self.personalize_params[user] = personalize_params
            round_participant_params[user] = global_params
            log_dict[user] = user_result['log_dict']
            log_dict[user]['uid'] = user
        
        start = time.time()
        # with mp.Pool() as pool:
        #     r = pool.map_async(self.call_client_proc, participants, callback=callback)
        #     r.wait()
        #     print(round_participant_params.keys())
        _ = [callback(call_client_proc(uid)) for uid in participants]
        logging.info("Training time: {}".format(time.time() - start))
        
        aggregated_params = self.strategy.aggregate(round_participant_params)
        self.model = cl.LoRAClient.set_global_parameters(self.model, aggregated_params)
        
        log_dict[-1] = {"matrix_rank": torch.linalg.matrix_rank(torch.tensor(aggregated_params[0] - server_params[0])).item()}
        df = pd.DataFrame(log_dict).T
        df['round'] = round_id
        logging.info('\n' +  df.head().to_string())
        logging.info(f"Set param time: {df['set_param_time'].sum()}", )
        logging.info(f"Prepare data time: {df['prepare_data_time'].sum()}", )
        logging.info(f"Fit time: {df['fit_time'].sum()}")
        # aggregate client models in server side.
        
        return df


    def fed_evaluate(self, evaluate_data):
        # evaluate all client models' performance using testing data.
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]
        # ratings for computing loss.
        temp = [0] * 100
        temp[0] = 1
        ratings = torch.FloatTensor(temp)
        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
            ratings = ratings.cuda()
        # store all users' test item prediction score.
        test_scores = None
        # store all users' negative items prediction scores.
        negative_scores = None
        all_loss = {}
        for user in range(self.config['num_users']):
            # load each user's mlp parameters.
            local_net = copy.deepcopy(self.model)
            client = self.clients[user]
            if user in self.client_model_params:
                local_net = client.set_local_parameters(local_net, self.client_model_params[user])
            if user in self.personalize_params:
                local_net = client.set_global_parameters(local_net, self.personalize_params[user])
            local_net.eval()
            with torch.no_grad():
                # obtain user's positive test information.
                test_user = test_users[user: user + 1]
                test_item = test_items[user: user + 1]
                # obtain user's negative test information.
                negative_item = negative_items[user*99: (user+1)*99]
                # perform model prediction.
                test_score = local_net(test_item)
                negative_score = local_net(negative_item)
                if user == 0:
                    test_scores = test_score
                    negative_scores = negative_score
                else:
                    test_scores = torch.cat((test_scores, test_score))
                    negative_scores = torch.cat((negative_scores, negative_score))
                ratings_pred = torch.cat((test_score, negative_score))
                loss = self.crit(ratings_pred.view(-1), ratings)
            all_loss[user] = loss.item()
        if self.config['use_cuda'] is True:
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
        
        test_users = test_users.view(-1)
        test_items = test_items.view(-1)
        test_scores = test_scores.view(-1)
        negative_users = negative_users.view(-1)
        negative_items = negative_items.view(-1)
        negative_scores = negative_scores.view(-1)

        print(test_users.shape, test_items.shape, test_scores.shape, negative_users.shape, negative_items.shape, negative_scores.shape)
        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        return hit_ratio, ndcg, all_loss


    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)

class MLPEngine(Engine):
    """Engine for training & evaluating GMF model"""
    def __init__(self, config):
        if config['lora']:
            print("Using LoRA model!")
            self.model = MLPLoRA(config)
        else:
            self.model = MLP(config)
        if config['use_cuda'] is True:
            # use_cuda(True, config['device_id'])
            self.model.cuda()
        super(MLPEngine, self).__init__(config)
        print(self.model)