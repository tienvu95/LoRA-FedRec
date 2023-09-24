import torch

import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
import logging

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
            self.lora_dropout = nn.Identity()
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
            logging.info("Init lora A and B")
            self.lora_A = nn.Parameter(self.weight.new_zeros((num_embeddings, r)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((r, embedding_dim)))
            scaling = self.lora_alpha / self.r
            self.register_buffer('lora_scaling', torch.tensor([scaling]))
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self, to_zero=False, keep_B=False):
        if hasattr(self, 'lora_A'):
            if to_zero:
                nn.init.zeros_(self.lora_A)
                if not keep_B:
                    nn.init.zeros_(self.lora_B)
            else:
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.zeros_(self.lora_A)
                if not keep_B:
                    nn.init.normal_(self.lora_B)
            self.merged = False
    
    def merge_lora_weights(self):
        if self.r > 0:
            lora_comp = (self.lora_A @ self.lora_B) * self.lora_scaling
            # print("lora_comp", lora_comp.shape, torch.linalg.norm(lora_comp))
            # print(self.lora_A.sum())
            self.weight.data += lora_comp
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
        result = F.embedding(
            x, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        if self.r > 0 and not self.merged:
            # result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.lora_A, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B) * self.lora_scaling
            return result
        else:
            return result

def init_ortho(tensor, gain=1):
    torch.nn.init.orthogonal_(tensor)