from math import sqrt
import torch
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F 
import lora

class MF(nn.Module):
    def __init__(self, user_num, item_num, gmf_emb_size=16, ItemEmbedding=nn.Embedding):
        super(MF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        """		

        self.embed_user_GMF = nn.Embedding(user_num, gmf_emb_size)
        self.embed_item_GMF = ItemEmbedding(item_num, gmf_emb_size)
        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        d = self.embed_user_GMF.weight.shape[1]
        std = torch.sqrt(torch.tensor(1 / d))
        # std=2/sqrt(d)
        # std = 1 / sqrt(d)
        # std = 0.01
        nn.init.normal_(self.embed_user_GMF.weight, std=std)
        nn.init.normal_(self.embed_item_GMF.weight, std=std)

        # nn.init.zeros_(self.embed_user_GMF.weight)
        # nn.init.zeros_(self.embed_item_GMF.weight)

        # with torch.no_grad():
        #     self.embed_user_GMF.weight.data += 0.01
        #     self.embed_item_GMF.weight.data += 0.01
        #     print(torch.mean(self.embed_user_GMF.weight @ self.embed_item_GMF.weight.t()))

    def _gmf_forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        return embed_user_GMF * embed_item_GMF

    def forward(self, user, item, **kwargs):
        output_GMF = self._gmf_forward(user, item)
        prediction = output_GMF.sum(dim=-1)
        return prediction.view(-1)

class NCF(nn.Module):
    def __init__(self, user_num, item_num, gmf_emb_size=16, mlp_emb_size=64, mlp_layer_dims=[128, 64, 32, 16],
                    dropout=0., ItemEmbedding=nn.Embedding):
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """		
        self.dropout = dropout

        self.embed_user_GMF = nn.Embedding(user_num, gmf_emb_size)
        self.embed_item_GMF = ItemEmbedding(item_num, gmf_emb_size)
        self.embed_user_MLP = nn.Embedding(
                user_num, mlp_emb_size)
        self.embed_item_MLP = ItemEmbedding(
                item_num, mlp_emb_size)

        MLP_modules = []
        for i in range(len(mlp_layer_dims) - 1):
            # input_size = factor_num * (2 ** (num_layers - i))
            input_size = mlp_layer_dims[i]
            output_size = mlp_layer_dims[i+1]
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, output_size))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        predict_size = gmf_emb_size + mlp_layer_dims[-1]
        # predict_size = factor_num
        self.predict_layer = nn.Linear(predict_size, 1) 

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, 
                                a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def _gmf_forward(self, user, item):
        embed_user_GMF = self.embed_user_GMF(user)
        embed_item_GMF = self.embed_item_GMF(item)
        return embed_user_GMF * embed_item_GMF

    def _mlp_forward(self, user, item):
        embed_user_MLP = self.embed_user_MLP(user)
        embed_item_MLP = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        return self.MLP_layers(interaction)

    def forward(self, user, item, **kwargs):
        output_GMF = self._gmf_forward(user, item)
        output_MLP = self._mlp_forward(user, item)
        concat = torch.cat((output_GMF, output_MLP), -1)
        # concat = output_GMF
        prediction = self.predict_layer(concat)
        return prediction.view(-1)

class LoraNCF(NCF):
    def __init__(self, user_num, item_num, gmf_emb_size=16, mlp_emb_size=64, mlp_layer_dims=[128, 64, 32, 16], dropout=0., lora_rank=4, lora_alpha=4, freeze_B=False):
        ItemEmbLayer = lambda num_emb, emb_dim: lora.Embedding(num_emb, emb_dim, r=lora_rank, lora_alpha=lora_alpha)
        super().__init__(user_num, item_num, gmf_emb_size, mlp_emb_size, mlp_layer_dims, dropout, ItemEmbedding=ItemEmbLayer)
        self.freeze_B = freeze_B
        self.lora_layer_names = ['embed_item_GMF', 'embed_item_MLP']
    
    def _merge_all_lora_weights(self):
        self.embed_item_GMF.merge_lora_weights()
        self.embed_item_MLP.merge_lora_weights()
    
    def _reset_all_lora_weights(self, init_B_strategy, keep_B=False):
        self.embed_item_GMF.reset_lora_parameters(init_B_strategy=init_B_strategy, keep_B=keep_B)
        self.embed_item_MLP.reset_lora_parameters(init_B_strategy=init_B_strategy, keep_B=keep_B)
    
class LoraMF(MF):
    def __init__(self, user_num, item_num, gmf_emb_size=16, lora_rank=4, lora_alpha=4, freeze_B=False):
        ItemEmbLayer = lambda num_emb, emb_dim: lora.Embedding(num_emb, emb_dim, r=lora_rank, lora_alpha=lora_alpha)
        super().__init__(user_num, item_num, gmf_emb_size, ItemEmbedding=ItemEmbLayer)
        self.freeze_B = freeze_B
        self.lora_layer_names = ['embed_item_GMF']
    
    def _merge_all_lora_weights(self):
        self.embed_item_GMF.merge_lora_weights()
    
    def _reset_all_lora_weights(self,  init_B_strategy, keep_B=False):
        self.embed_item_GMF.reset_lora_parameters(init_B_strategy=init_B_strategy, keep_B=keep_B)