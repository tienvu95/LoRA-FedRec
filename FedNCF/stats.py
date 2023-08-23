import time
import numpy as np
import scipy as sp
import torch

from typing import Any, Dict

from omegaconf import OmegaConf

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



def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]

    hparams["model"] = cfg["net"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["DATA"]
    hparams["dataloader"] = cfg["DATALOADER"]
    hparams["fed"] = cfg["FED"]
    hparams["trainer"] = cfg["TRAIN"]

    # hparams["callbacks"] = cfg.get("callbacks")
    hparams["exp"] = cfg.get("EXP")
    hparams["eval"] = cfg.get("EVAL")

    hparams["task_name"] = cfg.get("task_name")
    # hparams["tags"] = cfg.get("tags")
    # hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    return hparams