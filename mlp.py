import torch

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((num_embeddings, r)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((r, embedding_dim)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self, to_zero=False):
        if hasattr(self, 'lora_A'):
            if to_zero:
                nn.init.zeros_(self.lora_A)
                nn.init.zeros_(self.lora_B)
            else:
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.zeros_(self.lora_A)
                nn.init.normal_(self.lora_B)
    
    def merge_lora_weights(self):
        if self.r > 0:
            self.weight.data += (self.lora_A @ self.lora_B) * self.scaling
            self.merged = True

    # def train(self, mode: bool = True):
    #     nn.Embedding.train(self, mode)
    #     if mode:
    #         if self.merge_weights and self.merged:
    #             # Make sure that the weights are not merged
    #             if self.r > 0:
    #                 self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
    #             self.merged = False
    #     else:
    #         if self.merge_weights and not self.merged:
    #             # Merge the weights and mark it
    #             self.merge_lora_weights()
    
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x, self.lora_A, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse
                )
                result += (after_A @ self.lora_B) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)

class MLP(torch.nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
        item_embedding = self.embedding_item(item_indices)
        logits = self.affine_output(item_embedding)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass

    @property
    def is_lora(self):
        return False

class MLPLoRA(torch.nn.Module):
    def __init__(self, config):
        super(MLPLoRA, self).__init__()
        self.config = config
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_item = Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim, 
                                        r=config['lora_r'], lora_alpha=config['lora_alpha'], merge_weights=True)
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, item_indices):
        item_embedding = self.embedding_item(item_indices)
        logits = self.affine_output(item_embedding)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

    def load_pretrain_weights(self):
        pass

    @property
    def is_lora(self):
        return True

