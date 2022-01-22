import os
import yaml
import random


def set_seed(seed):
    import torch
    import numpy as np

    """Set randon seed for pytorch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # TRue is too slow


def load_config(config):
    """load experiment settings"""
    with open(config, "r") as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)

    from easydict import EasyDict as edict

    cfg = edict(cfg)
    return cfg


def create_folder(cfg):
    output_dir = "./exps/%s/" % (cfg.EXP_NAME)
    if not os.path.exists(output_dir):
        os.system("mkdir -p ./exps/%s/" % (cfg.EXP_NAME))
    # copy all scripts incase future change
    os.system("cp -r ./lib/ ./exps/%s/archive/" % (cfg.EXP_NAME))
    return output_dir


def create_infer_folder(cfg):
    output_path = "./exps/%s/output/" % (cfg.EXP_NAME)
    if not os.path.exists(output_path):
        os.system("mkdir -p %s" % (output_path))


def save_config(cfg, exp_name):
    os.system("cp {} ./exps/{}/{}".format(cfg, exp_name, cfg.split("/")[-1]))
