import argparse
import os
import sys
import torch

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import lib.dataset as dataset
from lib.models.build_model import TAL_model
from lib.loss.model_loss import TAL_loss
from lib.core.trainer import do_train
from lib.utils.util import set_seed, load_config, create_folder, save_config
from lib.utils.logger import setup_logger
from lib.utils.metric_logger import MetricLogger, print_meter


def train(cfg, logger):
    # build dataset
    dataset = eval("dataset.%s" % cfg.DATASET.name)
    logger.info("dataset: {}".format(cfg.DATASET.name))

    train_dataset = dataset(mode="train", subset="train", cfg=cfg, logger=logger)
    valid_dataset = dataset(mode="train", subset="valid", cfg=cfg, logger=logger)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.SOLVER.batch_size,
        num_workers=cfg.SOLVER.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.SOLVER.batch_size,
        num_workers=cfg.SOLVER.workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    # build model and loss function
    model = TAL_model(cfg)
    criterion = TAL_loss(cfg)

    # build optimizer
    optimizer = model.get_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.SOLVER.step_size, gamma=cfg.SOLVER.gamma)

    # DataParallel
    logger.info("Using DP, batch size is {}, GPU num is {}".format(cfg.SOLVER.batch_size, cfg.num_gpus))
    model = torch.nn.DataParallel(model, device_ids=list(range(cfg.num_gpus))).cuda()

    # training
    logger.info("Start training")
    train_meters = MetricLogger()
    valid_meters = MetricLogger()

    cfg.best_cost = 1e6
    for epoch in range(cfg.SOLVER.epoch):
        cfg.epoch = epoch
        do_train(model, criterion, train_loader, logger, cfg, train_meters, optimizer)
        do_train(model, criterion, valid_loader, logger, cfg, valid_meters, test=True)
        scheduler.step()
    print_meter(logger, train_meters, valid_meters)

    logger.info("Train over. Best_epoch is {}\n".format(cfg.best_epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSI")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("num_gpus", type=int)
    args = parser.parse_args()

    # load settings
    cfg = load_config(args.config)
    cfg.num_gpus = args.num_gpus
    set_seed(2021)

    # creat work folder
    output_dir = create_folder(cfg)

    # setup logger and save config
    logger = setup_logger("train", output_dir)
    logger.info("Using {} GPUs".format(cfg.num_gpus))
    logger.info(cfg)
    save_config(args.config, cfg.EXP_NAME)

    # start traning
    train(cfg, logger)
