import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import torch
import argparse

import lib.dataset as dataset
from lib.core.inferer import inference
from lib.models.build_model import TAL_model
from lib.utils.util import set_seed, load_config, create_infer_folder
from lib.utils.logger import setup_logger


def test(cfg, logger):
    # build dataset
    dataset = eval("dataset.%s" % cfg.DATASET.name)
    logger.info("dataset: {}".format(cfg.DATASET.name))

    test_dataset = dataset(mode="infer", subset="valid", cfg=cfg, logger=logger)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.SOLVER.batch_size,
        num_workers=cfg.SOLVER.workers,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    # build model
    model = TAL_model(cfg)

    # load checkpoint
    if cfg.pretrain != "None":  # load argparse epoch
        checkpoint_path = cfg.pretrain
    elif "infer" in cfg.SOLVER:  # load config epoch
        checkpoint_path = "./exps/{}/checkpoint/epoch_{}.pth.tar".format(cfg.EXP_NAME, cfg.SOLVER.infer)
    else:  # load best epoch
        checkpoint_path = "./exps/{}/checkpoint/best.pth.tar".format(cfg.EXP_NAME)
    logger.info("Loading checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    logger.info("Checkpoint is epoch {}".format(checkpoint["epoch"]))
    model.load_state_dict(checkpoint["state_dict"])

    # DataParallel
    model = torch.nn.DataParallel(model, device_ids=list(range(cfg.num_gpus))).cuda()
    model.eval()

    logger.info("Start inference")
    inference(model, test_loader, logger, cfg)
    logger.info("Inference Over\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSI")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("num_gpus", type=int)
    parser.add_argument("--pretrain", type=str, default="None")
    args = parser.parse_args()

    # load settings
    cfg = load_config(args.config)
    cfg.num_gpus = args.num_gpus
    cfg.pretrain = args.pretrain
    output_dir = create_infer_folder(cfg)
    set_seed(2021)

    # setup logger
    logger = setup_logger("infer", output_dir)
    logger.info("Using {} GPUs".format(cfg.num_gpus))
    logger.info(cfg)

    # start inference
    test(cfg, logger)
