import time
import numpy as np
import scipy as sp
import torch

class TimeStats(object):
    def __init__(self) -> None:
        self.reset()

    def __str__(self):
        return str(self._time_dict)
    
    def reset(self):
        self.flag_timestem = {}
        self._time_dict = {
            "set_parameters": 0,
            "fit": 0,
            "evaluate": 0,
            "get_parameters": 0,
        }
        self._pca_vars = []
    
    def set_aggregation_epoch(self, aggregation_epoch):
        self._aggregation_epoch = aggregation_epoch

    def mark_start(self, name):
        self.flag_timestem[name] = time.time()
    
    def mark_end(self, name):
        self._time_dict[name] += time.time() - self.flag_timestem[name]
        del self.flag_timestem[name]

    def count_num_important_component(self, cid, weight_name, A: torch.Tensor, percents=[0.99, 0.95, 0.9]):
        num_cps, S = count_num_important_component(A, percents=percents)
        self._pca_vars.append({"cid": cid, "aggregation_epoch": self._aggregation_epoch, "weight_name": weight_name, "num_important_components": num_cps, "S": S})
    
def count_num_important_component(A: torch.Tensor, percents=[0.99, 0.95, 0.9]):
    S = torch.linalg.svdvals(A.T @ A).cpu().numpy() / (A.shape[0] -  1)
    S = S / S.sum()
    S = np.cumsum(S)
    return [np.sum(S < p) + 1 for p in percents], S
