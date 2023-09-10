import copy
import torch
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F 
from rec.models import MF, NCF
from stats import cal_explain_variance_ratio
from fedlib.compression import svd_compress, TopKCompressor

class TransferedParams(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compressor = TopKCompressor()

    def diff(self, other):
        diff = TransferedParams()
        for key, val in self.items():
            diff[key] = val - other[key]
        return diff

    def zero_likes(self):
        zero_likes = TransferedParams()
        for key, val in self.items():
            zero_likes[key] = torch.zeros_like(val)
        return zero_likes
    
    def add_(self, other, alpha=1):
        for key, val in other.items():
            self[key] += alpha*val
        return self
    
    def server_step_(self, other, alpha=1):
        return self.add_(other, alpha=alpha)
    
    def div_scalar_(self, scalar):
        for key, val in self.items():
            self[key] /= scalar
        return self

    def compress(self, method='svd', **kwargs):
        if method == 'svd':
            self['embed_item_GMF.weight'] = svd_compress(self['embed_item_GMF.weight'], **kwargs)
        elif method == 'topk': 
            flatten_mess = self.compressor.flatten(self['embed_item_GMF.weight'], name='embed_item_GMF.weight')
            tensor, indexes, values = self.compressor.compress(flatten_mess, name='embed_item_GMF.weight', **kwargs)
            decompressed_mess = self.compressor.decompress_new(values, indexes, name='embed_item_GMF.weight')
            mess = self.compressor.unflatten(decompressed_mess, name='embed_item_GMF.weight')
            self['embed_item_GMF.weight'] = mess

class FedParamSpliter:
    def __init__(self) -> None:
        pass

    def get_server_params(self, **kwargs):
        return self._get_splited_params()[1]
    
    def get_private_params(self, **kwarfs):
        return self._get_splited_params()[0]

    def _get_splited_params(self, **kwarfs):
        submit_params = TransferedParams()  
        private_params = {}
        for key, val in self.state_dict().items():
            if 'user' in key:
                private_params[key] = val.detach().clone()
            else:
                submit_params[key] = val.detach().clone()
        return private_params, submit_params
    
    def _get_splited_params_for_optim(self, **kwarfs):
        submit_params = {}
        private_params = {}
        for key, val in self.state_dict().items():
            if 'item' in key:
                submit_params[key] = val
            else:
                private_params[key] = val
        return private_params, submit_params
    
    def _set_state_from_splited_params(self, splited_params):
        private_params, submit_params = splited_params
        params_dict = dict(private_params, **submit_params)
        state_dict = OrderedDict(params_dict)
        self.load_state_dict(state_dict, strict=True)

class FedMF(MF, FedParamSpliter):
    def __init__(self, item_num, gmf_emb_size=16, user_num=1):
        MF.__init__(self, user_num, item_num, gmf_emb_size)

    def forward(self, user, item, mask_zero_user_index=True):
        if mask_zero_user_index:
            user = torch.zeros_like(user)
        return super().forward(user, item)
    
    def _reinit_private_params(self):
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
    
    def server_prepare(self):
        return
    
    @classmethod
    def merge_client_params(cls, clients, server_params, model, device):
        client_weights = [c._private_params.values() for c in clients]
        client_weights = [torch.cat(w, dim=0).to(device) for w in zip(*client_weights)]
        client_weights = {k: v for k,v in zip(clients[0]._private_params.keys(), client_weights)}
        eval_model = copy.deepcopy(model)
        eval_model._set_state_from_splited_params([clients[0]._private_params, server_params])
        eval_model.embed_user_GMF = torch.nn.Embedding.from_pretrained(client_weights['embed_user_GMF.weight'])
        return eval_model
    
    @classmethod
    def stat_transfered_params(cls, transfer_params: TransferedParams):
        item_emb = transfer_params['embed_item_GMF.weight']
        explain_variance_ratio = cal_explain_variance_ratio(item_emb)
        return {"mf_item_emb_explain_variance_ratio": explain_variance_ratio}

class FedNCFModel(NCF, FedParamSpliter):
    def __init__(self, item_num, gmf_emb_size=16, mlp_emb_size=64, mlp_layer_dims=[128, 64, 32, 16], dropout=0., user_num=1):
        NCF.__init__(self, user_num, item_num, gmf_emb_size, mlp_emb_size, mlp_layer_dims, dropout)
        FedParamSpliter.__init__(self)
        self.user_num = user_num

    def forward(self, user, item, mask_zero_user_index=True):
        if mask_zero_user_index:
            user = torch.zeros_like(user)
        return super().forward(user, item)
    
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
