import copy
import torch
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F 
import lora
from rec.models import NCF, LoraNCF

class FedNCFModel(NCF):
    def __init__(self, item_num, gmf_emb_size=16, mlp_emb_size=64, mlp_layer_dims=[128, 64, 32, 16], dropout=0., user_num=1):
        super().__init__(user_num, item_num, gmf_emb_size, mlp_emb_size, mlp_layer_dims, dropout)
        self.user_num = user_num

    def forward(self, user, item, mask_zero_user_index=True):
        if mask_zero_user_index:
            user = torch.zeros_like(user)
        return super().forward(user, item)
    
    def _get_splited_params(self):
        sharable_params = {'weights': [], "keys": []}
        private_params = {'weights': [], "keys": []}
        for key, val in self.state_dict().items():
            if 'user' in key:
                private_params['weights'].append(val.detach().clone())
                private_params['keys'].append(key)
            else:
                sharable_params['weights'].append(val.detach().clone())
                sharable_params['keys'].append(key)
        return private_params, sharable_params


    def _set_state_from_splited_params(self, splited_params):
        # Set model parameters from a list of NumPy ndarrays
        private_params, sharable_params = splited_params
        params_dict = list(zip(sharable_params['keys'], sharable_params['weights']))
        # if private_params is not None:
        params_dict += list(zip(private_params['keys'], private_params['weights']))
        state_dict = OrderedDict({k: v for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)
    
    def _reinit_private_params(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
    
    @classmethod
    def merge_client_params(cls, clients, server_params, model, device):
        client_weights = [c._private_params['weights'] for c in clients]
        client_weights = [torch.cat(w, dim=0).to(device) for w in zip(*client_weights)]
        client_weights = {k: v for k,v in zip(clients[0]._private_params['keys'], client_weights)}
        eval_model = copy.deepcopy(model)
        eval_model._set_state_from_splited_params([clients[0]._private_params, server_params])
        eval_model.embed_user_GMF = torch.nn.Embedding.from_pretrained(client_weights['embed_user_GMF.weight'])
        eval_model.embed_user_MLP = torch.nn.Embedding.from_pretrained(client_weights['embed_user_MLP.weight'])
        return eval_model

class FedLoraNCF(LoraNCF):
    def __init__(self, item_num, gmf_emb_size=16, mlp_emb_size=64, mlp_layer_dims=[128, 64, 32, 16], dropout=0., lora_rank=4, lora_alpha=4, freeze_B=False):
        super().__init__(1, item_num, gmf_emb_size, mlp_emb_size, mlp_layer_dims, dropout, lora_rank, lora_alpha, freeze_B)
        if self.freeze_B:
            self.embed_item_GMF.lora_B.requires_grad = False
            self.embed_item_MLP.lora_B.requires_grad = False
    
    def forward(self, user, item, mask_zero_user_index=True):
        if mask_zero_user_index:
            user = torch.zeros_like(user)
        return super().forward(user, item)
    
    @torch.no_grad()
    def _get_splited_params(self, keep_B=False, merge_weights=True):
        if merge_weights:
            self._merge_all_lora_weights()
        self._reset_all_lora_weights(to_zero=True, keep_B=keep_B)
        sharable_params = {'weights': [], "keys": []}
        private_params = {'weights': [], "keys": []}
        for key, val in self.state_dict().items():
            if 'user' in key:
                private_params['weights'].append(val.detach().clone())
                private_params['keys'].append(key)
            else:
                sharable_params['weights'].append(val.detach().clone())
                sharable_params['keys'].append(key)
        return private_params, sharable_params

    @torch.no_grad()
    def _set_state_from_splited_params(self, splited_params):
        # Set model parameters from a list of NumPy ndarrays
        private_params, sharable_params = splited_params
        params_dict = list(zip(sharable_params['keys'], sharable_params['weights']))
        # if private_params is not None:
        params_dict += list(zip(private_params['keys'], private_params['weights']))
        state_dict = OrderedDict({k: v for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)
        self._reset_all_lora_weights(keep_B=self.freeze_B)
    
    def _reinit_private_params(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
    
    def _reinit_B(self):
        print("Reinit B")
        nn.init.normal_(self.embed_item_GMF.lora_B)
        nn.init.normal_(self.embed_item_MLP.lora_B)
    
    @classmethod
    def merge_client_params(cls, clients, server_params, model, device):
        client_weights = [c._private_params['weights'] for c in clients]
        client_weights = [torch.cat(w, dim=0).to(device) for w in zip(*client_weights)]
        client_weights = {k: v for k,v in zip(clients[0]._private_params['keys'], client_weights)}
        eval_model = copy.deepcopy(model)
        eval_model._set_state_from_splited_params([client_weights[0]._private_params, server_params])
        eval_model.embed_user_GMF = torch.nn.Embedding.from_pretrained(client_weights['embed_user_GMF.weight'])
        eval_model.embed_user_MLP = torch.nn.Embedding.from_pretrained(client_weights['embed_user_MLP.weight'])
        return eval_model