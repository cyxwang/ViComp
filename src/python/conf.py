# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration file (powered by YACS)."""

import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C


# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Check https://github.com/RobustBench/robustbench for available models
_C.MODEL.ARCH = 'Online'
_C.MODEL.PCMODELNAME = 'PANet'
_C.MODEL.GCMODELNAME = 'hmat'
_C.MODEL.LONG = 'casst'
_C.MODEL.SHORT = 'finetune'


_C.MODEL.PCMODEL_DIR = '../../pretrain/Paper003_CompenDUShufflePretrain_l1+l2+ssim_16_16_20000_0.001_0.001_0.5_5000_0.0001.pth'
_C.MODEL.GCMODEL_DIR = '' 

_C.MODEL.UNCMP = False
_C.MODEL.SUPDATE = True

  

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.SURFACE = 'white'
_C.CORRUPTION.DATA_NAME = ['mountrain']
_C.CORRUPTION.DATA_DIR= '../../data/'
# Number of examples to evaluate (10000 for all samples in CIFAR-10)
_C.CORRUPTION.NUM = 200
_C.CORRUPTION.START = 1
# ------------------------------- Batch norm options ------------------------ #
_C.BN = CfgNode()

# BN epsilon
_C.BN.EPS = 1e-5

# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM_CASST = CfgNode()
_C.OPTIM_CASST.LOSS = 'l1+l2+ssim'
_C.OPTIM_CASST.DEVICE = 'cuda'
# Number of updates per batch

# Learning rate
_C.OPTIM_CASST.LR = 1e-3

# Choices: Adam, SGD
_C.OPTIM_CASST.METHOD = 'Adam'

# Beta
_C.OPTIM_CASST.BETA = 0.9

# Momentum
_C.OPTIM_CASST.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM_CASST.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM_CASST.NESTEROV = True

# L2 regularization
_C.OPTIM_CASST.WD = 0.0

# CASST
_C.OPTIM_CASST.MT = 1.0

_C.OPTIM_CASST.RST = 0.1

# ------------------------------
_C.OPTIM_FINETUNE = CfgNode()
_C.OPTIM_FINETUNE.LOSS = 'l1+l2+ssim'
_C.OPTIM_FINETUNE.DEVICE = 'cuda'
# Number of updates per batch

# Learning rate
_C.OPTIM_FINETUNE.LR = 1e-3

# Choices: Adam, SGD
_C.OPTIM_FINETUNE.METHOD = 'Adam'

# Beta
_C.OPTIM_FINETUNE.BETA = 0.9

# Momentum
_C.OPTIM_FINETUNE.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM_FINETUNE.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM_FINETUNE.NESTEROV = True

# L2 regularization
_C.OPTIM_FINETUNE.WD = 0.0

# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Batch size for evaluation (and updates for norm + tent)
_C.TEST.BATCH_SIZE = 1
_C.TEST.IsDraw = True

_C.TEST.Name = True



# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# ---------------------------------- Misc options --------------------------- #

# Optional description of a config
_C.DESC = ""

# Note that non-determinism is still present due to non-deterministic GPU ops
_C.RNG_SEED = 1

# Output directory
_C.SAVE_DIR = "./output"

# Data directory
_C.DATA_DIR = "./data"

# Weight directory
_C.CKPT_DIR = "./ckpt"

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# # Config destination (in SAVE_DIR)
# _C.CFG_DEST = "cfg.yaml"


# --------------------------------- gma config -------------------------- #
_C.gma = CfgNode()
_C.gma.pretrained_model = '../../pretrain/gma-sintel.pth'
_C.gma.mixed_precision = False
_C.gma.alternate_corr = True

_C.gma.num_heads = 1
_C.gma.position_only = False
_C.gma.position_and_content = False
_C.gma.corr_levels = 4
_C.gma.corr_radius = 4
_C.gma.dropout = 0

# --------------------------------- flowformer config -------------------------- #

# latentcostformer
_C.latentcostformer = CfgNode()
_C.latentcostformer.pretrained_model = '../../pretrain/flowformer-sintel.pth'
_C.latentcostformer.pe = 'linear'
_C.latentcostformer.dropout = 0.0
_C.latentcostformer.encoder_latent_dim = 256 # in twins, this is 256
_C.latentcostformer.query_latent_dim = 64
_C.latentcostformer.cost_latent_input_dim = 64
_C.latentcostformer.cost_latent_token_num = 8
_C.latentcostformer.cost_latent_dim = 128
_C.latentcostformer.arc_type = 'transformer'
_C.latentcostformer.cost_heads_num = 1
# encoder
_C.latentcostformer.pretrain = True
_C.latentcostformer.context_concat = False
_C.latentcostformer.encoder_depth = 3
_C.latentcostformer.feat_cross_attn = False
_C.latentcostformer.patch_size = 8
_C.latentcostformer.patch_embed = 'single'
_C.latentcostformer.no_pe = False
_C.latentcostformer.gma = "GMA"
_C.latentcostformer.kernel_size = 9
_C.latentcostformer.rm_res = True
_C.latentcostformer.vert_c_dim = 64
_C.latentcostformer.cost_encoder_res = True
_C.latentcostformer.cnet = 'twins'
_C.latentcostformer.fnet = 'twins'
_C.latentcostformer.no_sc = False
_C.latentcostformer.only_global = False
_C.latentcostformer.add_flow_token = True
_C.latentcostformer.use_mlp = False
_C.latentcostformer.vertical_conv = False

# decoder
_C.latentcostformer.decoder_depth = 32
_C.latentcostformer.critical_params = ['cost_heads_num', 'vert_c_dim', 'cnet', 'pretrain' , 'add_flow_token', 'encoder_depth', 'gma', 'cost_encoder_res']


# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    print(cfg_file)
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_fom_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info(
        "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)
