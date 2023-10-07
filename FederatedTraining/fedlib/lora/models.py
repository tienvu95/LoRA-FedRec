import copy
import torch
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F 
import lora
from rec.models import LoraNCF, LoraMF
from stats import cal_explain_variance_ratio
# from fedlib.compression import TopKCompressor

class LoRATransferedParams(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sub(self, other):
        diff = LoRATransferedParams()
        for key, val in self.items():
            if 'lora' in key:
                diff[key] = val.clone()
            elif key == "private_inter_mask":
                diff[key] = val.clone()
            else:
                diff[key] = val - other[key]
        return diff

    def zero_likes(self):
        zero_likes = LoRATransferedParams()
        for key, val in self.items():
            zero_likes[key] = torch.zeros_like(val)
        return zero_likes
    
    def add_(self, other, alpha=1):
        other_private_inter_mask = other["private_inter_mask"]
        for key, val in other.items():
            if "item" in key and "emb" in key and "lora_A" in key:
                # add item embedding
                # print("add lora A")
                self[key] += val * other_private_inter_mask.unsqueeze(-1)
                # self[key] += val
            elif key == "private_inter_mask":
                self["private_inter_mask"] = self["private_inter_mask"] + other["private_inter_mask"]
            else:
                self[key] += alpha*val
        return self

    def server_step_(self, other, alpha=1):
        for key, val in other.items():
            if 'lora' in key:
                self[key] = alpha*val
            elif key == "private_inter_mask":
                continue
            else:
                self[key] += alpha*val
        return self
    
    def div_scalar_(self, scalar):
        private_inter_mask = self["private_inter_mask"]
        # print("inter track", private_inter_mask, (private_inter_mask == 0).sum().item())
        private_inter_mask[private_inter_mask == 0] = 1
        for key, val in self.items():
            if key == "private_inter_mask":
                continue
            elif "item" in key and "emb" in key and "lora_A" in key:
                self[key] /= private_inter_mask.unsqueeze(-1)
                # self[key] /= 60
            else:
                self[key] /= scalar
        return self

    def compress(self, method='svd', **kwargs):
        return

    def decompress(self):
        return

class FedLoraParamsSplitter:
    def __init__(self, item_num) -> None:
        self.register_buffer('private_inter_mask', torch.zeros(item_num, dtype=torch.long))
        pass

    def get_server_params(self, **kwargs):
        return self._get_splited_params()[1]
    
    def get_private_params(self, **kwarfs):
        return self._get_splited_params()[0]

    def _get_splited_params(self, server_init=False, **kwarfs):
        submit_params = LoRATransferedParams()
        private_params = {}
        # self._merge_all_lora_weights()
        # self._reset_all_lora_weights(init_B_strategy='zeros', keep_B=False)
        for key, val in self.state_dict().items():
            if 'user' in key:
                private_params[key] = val.detach().clone()
            else:
                if server_init:
                    submit_params[key] = val.detach().clone()
                else:
                    # if 'lora_B' in key:
                    #     print(val[0, :10])
                        # val = val.zero_()
                        
                    if any([n in key for n in self.lora_layer_names]):
                        # lora_comp = (self._model.embed_item_GMF.lora_A @ self._model.embed_item_GMF.lora_B) * self._model.embed_item_GMF.lora_scaling
                        if 'weight' in key:
                            continue
                    elif key == "private_inter_mask":
                        submit_params[key] = val.detach().clone().clamp_(max=1)
                # print("p", private_params[key])
                    submit_params[key] = val.detach().clone()
        # print(list(submit_params.keys()))
        return private_params, submit_params
    
    # def unrole_submitted_params(self, submitted_params):
    #     self.load_state_dict(submitted_params, strict=False)
    #     self._merge_all_lora_weights()
    #     self._reset_all_lora_weights(to_zero=True)

    def _set_state_from_splited_params(self, splited_params):
        private_params, submit_params = splited_params
        params_dict = dict(private_params, **submit_params)
        state_dict = OrderedDict(params_dict)
        self.load_state_dict(state_dict, strict=True)
        self._merge_all_lora_weights()
        self._reset_all_lora_weights(init_B_strategy=None, keep_B=self.freeze_B)
        # self._reset_all_lora_weights(init_B_strategy="random-scaling", keep_B=False)
        self.private_inter_mask = torch.zeros_like(self.private_inter_mask)


    def _get_splited_params_for_optim(self, **kwarfs):
        submit_params = []
        private_params = []
        for key, val in self.named_parameters():
            if 'item' in key and 'emb' in key:
                submit_params.append(val)
            else:
                private_params.append(val)
        return submit_params, private_params 

class FedLoraNCF(LoraNCF, FedLoraParamsSplitter):
    def __init__(self, item_num, gmf_emb_size=16, mlp_emb_size=64, mlp_layer_dims=[128, 64, 32, 16], dropout=0., lora_rank=4, lora_alpha=4, freeze_B=False, user_num=1):
        super().__init__(user_num, item_num, gmf_emb_size, mlp_emb_size, mlp_layer_dims, dropout, lora_rank, lora_alpha, freeze_B)
        if self.freeze_B:
            self.embed_item_GMF.lora_B.requires_grad = False
            self.embed_item_MLP.lora_B.requires_grad = False
    
    def forward(self, user, item, mask_zero_user_index=True):
        self.private_inter_mask[item] = 1
        if mask_zero_user_index:
            user = torch.zeros_like(user)
        return super().forward(user, item)
    
    # @torch.no_grad()
    # def _get_splited_params(self, keep_B=False, merge_weights=True,**kwarfs):
    #     if merge_weights:
    #         self._merge_all_lora_weights()
    #     self._reset_all_lora_weights(to_zero=True, keep_B=keep_B)
    #     sharable_params = {'weights': [], "keys": []}
    #     private_params = {'weights': [], "keys": []}
    #     for key, val in self.state_dict().items():
    #         if 'user' in key:
    #             private_params['weights'].append(val.detach().clone())
    #             private_params['keys'].append(key)
    #         else:
    #             sharable_params['weights'].append(val.detach().clone())
    #             sharable_params['keys'].append(key)
    #     return private_params, sharable_params

    # @torch.no_grad()
    # def _set_state_from_splited_params(self, splited_params):
    #     # Set model parameters from a list of NumPy ndarrays
    #     private_params, sharable_params = splited_params
    #     params_dict = list(zip(sharable_params['keys'], sharable_params['weights']))
    #     # if private_params is not None:
    #     params_dict += list(zip(private_params['keys'], private_params['weights']))
    #     state_dict = OrderedDict({k: v for k, v in params_dict})
    #     self.load_state_dict(state_dict, strict=True)
    #     self._reset_all_lora_weights(keep_B=self.freeze_B)

    def server_prepare(self, **kwargs):
        self._reset_all_lora_weights(keep_B=False, **kwargs)
    
    def _reinit_private_params(self):
        self._init_weight_()
    
    # def _reinit_B(self):
    #     print("Reinit B")
    #     nn.init.normal_(self.embed_item_GMF.lora_B)
    #     nn.init.normal_(self.embed_item_MLP.lora_B)
    
    @classmethod
    def merge_client_params(cls, clients, server_params, model, device):
        client_weights = [c._private_params.values() for c in clients]
        client_weights = [torch.cat(w, dim=0).to(device) for w in zip(*client_weights)]
        client_weights = {k: v for k,v in zip(clients[0]._private_params.keys(), client_weights)}
        eval_model = copy.deepcopy(model)
        eval_model._set_state_from_splited_params([clients[0]._private_params, server_params])
        eval_model.embed_user_GMF = torch.nn.Embedding.from_pretrained(client_weights['embed_user_GMF.weight'])
        eval_model.embed_user_MLP = torch.nn.Embedding.from_pretrained(client_weights['embed_user_MLP.weight'])
        return eval_model
    
class FedLoraMF(LoraMF, FedLoraParamsSplitter):
    def __init__(self, item_num, gmf_emb_size=16, lora_rank=4, lora_alpha=4, freeze_B=False, user_num=1):
        LoraMF.__init__(self, user_num, item_num, gmf_emb_size, lora_rank, lora_alpha, freeze_B)
        FedLoraParamsSplitter.__init__(self, item_num)
        if self.freeze_B:
            self.embed_item_GMF.lora_B.requires_grad = False
    
    def forward(self, user, item, mask_zero_user_index=True):
        self.private_inter_mask[item] = 1
        if mask_zero_user_index:
            user = torch.zeros_like(user)
        return super().forward(user, item)
    
    def server_prepare(self, **kwargs):
        self._reset_all_lora_weights(keep_B=False, **kwargs)
    
    def _reinit_private_params(self):
        self._init_weight_()
    
    # def _reinit_B(self):
    #     print("Reinit B")
    #     nn.init.normal_(self.embed_item_GMF.lora_B)
    
    @classmethod
    def merge_client_params(cls, clients, server_params, model, device):
        client_weights = [c._private_params.values() for c in clients]
        client_weights = [torch.cat(w, dim=0).to(device) for w in zip(*client_weights)]
        client_weights = {k: v for k,v in zip(clients[0]._private_params.keys(), client_weights)}
        eval_model = copy.deepcopy(model)
        eval_model._set_state_from_splited_params([clients[0]._private_params, server_params])
        eval_model.embed_user_GMF = torch.nn.Embedding.from_pretrained(client_weights['embed_user_GMF.weight'])
        return eval_model