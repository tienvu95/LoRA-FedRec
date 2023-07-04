from collections import OrderedDict
import warnings

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import pandas as pd
from mlp import MLP
import data
import rec
import pickle
import logging

import multiprocessing as mp

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_opts(model, lr, l2_regularization, num_items, lr_eta):
  # Defining optimizers
  # optimizer is responsible for updating score function.
  optimizer = torch.optim.SGD(model.affine_output.parameters(),
                              lr=lr, weight_decay=l2_regularization)  # MLP optimizer
  # optimizer_i is responsible for updating item embedding.
  optimizer_i = torch.optim.SGD(model.embedding_item.parameters(),
                                lr=lr * num_items * lr_eta,
                              # lr=lr * self.config['lr_eta'],
                              weight_decay=l2_regularization)  # Item optimizer
  optimizers = [optimizer, optimizer_i]
  return optimizers

def train(net, criterion, trainloader, epochs, config):
  """Train the model on the training set."""
  optimizers = get_opts(net, config['lr'], config['l2_regularization'], config['num_items'], config['lr_eta'])
  optimizer, optimizer_i = optimizers
  for _ in range(epochs):
    loss = 0 
    num_sample = 0
    for _, items, ratings in trainloader:
      assert isinstance(_, torch.LongTensor)
      ratings = ratings.float()
      items, ratings = items.to(DEVICE), ratings.to(DEVICE)

      # update score function.
      optimizer.zero_grad()
      ratings_pred = net(items)
      loss_score_fn = criterion(ratings_pred.view(-1), ratings)
      loss_score_fn.backward()
      optimizer.step()

      # update item embedding.
      optimizer_i.zero_grad()
      ratings_pred = net(items)
      loss_item_emb = criterion(ratings_pred.view(-1), ratings)
      loss_item_emb.backward()
      optimizer_i.step()

      loss += loss_item_emb.item() * len(_)
      num_sample += len(_)
    loss = loss / num_sample
  return loss

def instance_user_train_loader(bs, user_train_data):
  """instance a user's train loader."""
  dataset = data.UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                  item_tensor=torch.LongTensor(user_train_data[1]),
                                  target_tensor=torch.FloatTensor(user_train_data[2]))
  return DataLoader(dataset, batch_size=bs, shuffle=True)

def load_data(cid, dataset_name, num_negative):
  dataset_dir = "datasets/" + dataset_name + "/" + "ratings.dat"
  if dataset_name == "ml-1m":
      rating = pd.read_csv(dataset_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
  elif dataset_name == "ml-100k":
      rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
  elif dataset_name == "lastfm-2k":
      rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
  elif dataset_name == "amazon":
      rating = pd.read_csv(dataset_dir, sep=",", header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
      rating = rating.sort_values(by='uid', ascending=True)
  else:
      pass
  # Reindex
  user_id = rating[['uid']].drop_duplicates().reindex()
  user_id['userId'] = np.arange(len(user_id))
  rating = pd.merge(rating, user_id, on=['uid'], how='left')
  item_id = rating[['mid']].drop_duplicates()
  item_id['itemId'] = np.arange(len(item_id))
  rating = pd.merge(rating, item_id, on=['mid'], how='left')
  rating = rating[['userId', 'itemId', 'rating', 'timestamp']]
  logging.info('Range of userId is [{}, {}]'.format(rating.userId.min(), rating.userId.max()))
  logging.info('Range of itemId is [{}, {}]'.format(rating.itemId.min(), rating.itemId.max()))
  
  # DataLoader for training
  sample_generator = data.SampleGenerator(ratings=rating)
  validate_data = sample_generator.validate_data
  test_data = sample_generator.test_data
  all_train_data = sample_generator.store_all_train_data(num_negative)
  return all_train_data, validate_data, test_data

  

# #############################################################################
# Federating the pipeline with Flower
# #############################################################################

# Define Flower client
class Client(object):
  def __init__(self, trainloader, valloader, config):
    # self.net = net
    self.criterion = torch.nn.BCELoss()
    self.trainloader = trainloader
    self.valloader = valloader
    self.config = config
    # self._metron = rec.MetronAtK(top_k=10)

  def get_parameters(self, net, config):
    global_params = [val.cpu().numpy() for _, val in net.embedding_item.state_dict().items()]
    local_params = [val.cpu().numpy() for _, val in net.affine_output.state_dict().items()]
    return global_params, local_params

  def set_parameters(self, net, global_params, local_params):
    global_params_dict = zip(net.embedding_item.state_dict().keys(), global_params)
    global_state_dict = OrderedDict({k: torch.tensor(v) for k, v in global_params_dict})
    net.embedding_item.load_global_state_dict(global_state_dict, strict=True)

    local_params_dict = zip(net.affine_output.state_dict().keys(), local_params)
    local_state_dict = OrderedDict({k: torch.tensor(v) for k, v in local_params_dict})
    net.affine_output.load_state_dict(local_state_dict, strict=True)
    return net

  def fit(self, net, parameters, config):
    net = self.set_parameters(net, parameters)
    loss = train(net, self.criterion, self.trainloader, epochs=1, config=self.config)
    return self.get_parameters(net, config={}), len(self.trainloader.dataset), {"loss": loss}
  

#   def evaluate(self, parameters, config):
#     self.set_parameters(parameters)
#     # hit_ratio, ndcg, te_loss = rec.evaluate(self.net, self.criterion, self._metron, self.valloader, self.config["num_users"], DEVICE)
#     # return float(0.0), len(self.testloader.dataset), {"hit_ratio": float(hit_ratio), "ndcg": float(ndcg), }
#     peval_output = rec.personalize_evaluate(self.net, self.criterion, self._metron, self.valloader, self.config['uid'], DEVICE)
#     return float(0.0), len(self.valloader), {"peval_output": pickle.dumps(peval_output)}


def client_fn(cid: str, config):
  # Load model and data (simple CNN, CIFAR-10)
  # net = MLPLoRA(config).to(DEVICE)
  net = MLP(config).to(DEVICE)
  all_train_data, validate_data, test_data = load_data(cid, config['dataset'], config['num_negative'])
  uid = int(cid)
  user_train_data = [all_train_data[0][uid], all_train_data[1][uid], all_train_data[2][uid]]
  user_dataloader = instance_user_train_loader(config['batch_size'], user_train_data)
  del (all_train_data, validate_data)
  config['uid'] = int(cid)
  return FlowerClient(net, user_dataloader, test_data, config)
