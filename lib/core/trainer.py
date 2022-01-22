import time
import torch
import datetime
from ..utils.metric_logger import MetricLogger
from ..utils.checkpoint import save_checkpoint


def do_train(model, criterion, data_loader, logger, cfg, global_meters, optimizer=None, test=False):
    if test:
        model.eval()
    else:
        model.train()

    meters = MetricLogger(delimiter="  ")
    end = time.time()

    max_iteration = len(data_loader)
    max_epoch = cfg.SOLVER.epoch
    last_epoch_iteration = (max_epoch - cfg.epoch - 1) * max_iteration

    for idx, (video_info, video_data, video_gt) in enumerate(data_loader):
        if isinstance(video_data, list):
            video_data = [_data.cuda() for _data in video_data]
        else:
            video_data = video_data.cuda()
        video_gt = [_gt.cuda() for _gt in video_gt]

        if test:
            with torch.no_grad():
                pred = model(video_data)
                cost, loss_dict = criterion(pred, video_gt)
        else:
            pred = model(video_data)
            cost, loss_dict = criterion(pred, video_gt)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        batch_time = time.time() - end
        end = time.time()

        meters.update(time=batch_time)
        meters.update(**loss_dict)

        eta_seconds = meters.time.avg * (max_iteration - idx + last_epoch_iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if ((idx % 200 == 0) and idx != 0) or (idx == max_iteration - 1):
            logger.info(
                meters.delimiter.join(
                    [
                        "{mode}: [E{epoch}/{max_epoch}]",
                        "iter: {iteration}/{max_iteration}",
                        "eta: {eta}",
                        "{meters}",
                        # "max_mem: {memory:.0f}",
                    ]
                ).format(
                    mode="\t Test " if test else "Train",
                    eta=eta_string,
                    epoch=cfg.epoch,
                    max_epoch=max_epoch - 1,
                    iteration=idx,
                    max_iteration=max_iteration - 1,
                    meters=str(meters),
                    # memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

    # updata global meters
    global_meters.update(**meters.data())

    # save checkpoint
    if test:
        if meters.cost.avg < cfg.best_cost:
            cfg.best_cost = meters.cost.avg
            cfg.best_epoch = cfg.epoch
            save_checkpoint(model, cfg.epoch, cfg, best=True)
        else:
            save_checkpoint(model, cfg.epoch, cfg)
