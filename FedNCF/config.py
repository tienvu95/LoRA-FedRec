import torch
from yacs.config import CfgNode as CN
import argparse

_C = CN()

_C.seed = 0

_C.FED = CN()
_C.FED.num_clients = 120
_C.FED.aggregation_epochs = 400
_C.FED.local_epochs = 2

_C.DATA = CN()
_C.DATA.root = '../data/Data'
_C.DATA.name = 'movielens'
_C.DATA.num_negatives = 4
_C.DATA.test_num_ng = 99

_C.DATALOADER = CN()
_C.DATALOADER.batch_size = 64
_C.DATALOADER.shuffle = True
_C.DATALOADER.num_workers = 0


_C.MODEL = CN()
_C.MODEL.name = 'ncf'
_C.MODEL.factor_num = 16
_C.MODEL.num_layers = 3
_C.MODEL.dropout = 0.0
_C.MODEL.use_lora = False
_C.MODEL.lora_r = None
_C.MODEL.lora_alpha = None

_C.TRAIN = CN()
_C.TRAIN.lr = 5e-3 # 1e-3
_C.TRAIN.weight_decay = None
_C.TRAIN.lr_scheduler = None
_C.TRAIN.device = 'cuda' if torch.cuda.is_available() else 'cpu'

_C.EXP = CN()
_C.EXP.save = False
_C.EXP.output_dir = 'pretrained/standard/'

_C.EVAL = CN()
_C.EVAL.topk = 10

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`

def get_parser():
    parser = argparse.ArgumentParser(
        prog='federeco',
        description='federated recommendation system',
    )

    # parser.add_argument('-d', default='movielens', metavar='dataset',
    #                     choices=['movielens', 'pinterest', 'yelp'],
    #                     help='which dataset to use, default "movielens"')
    # parser.add_argument('-p', default='pretrained/ncf.h5', metavar='path',
    #                     help='path where trained model is stored, default "pretrained/ncf.h5"')
    # parser.add_argument('-e', default=400, metavar='epochs', type=int,
    #                     help='number of training epochs, default 400')
    # parser.add_argument('-s', '--save', default=True, metavar='save', action=argparse.BooleanOptionalAction,
    #                     help='flag that indicates if trained model should be saved')
    # parser.add_argument('-n', default=50, metavar='sample_size', type=int,
    #                     help='number of clients to sample per epoch')
    # parser.add_argument('-l', default=3, metavar='local_epochs', type=int,
    #                     help='number of local training epochs')
    # parser.add_argument('-lr', default=0.001, type=float, metavar='learning_rate',
    #                     help='learning rate')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg_defaults()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.freeze()
    return cfg