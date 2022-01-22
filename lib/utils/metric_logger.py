# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque

import torch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self):
        self.deque = deque()
        # self.series = []
        # self.total = 0.0
        # self.count = 0

    def update(self, value):
        self.deque.append(value)
        # self.series.append(value)
        # self.count += 1
        # self.total += value

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def sum(self):
        d = torch.tensor(list(self.deque))
        return d.sum().item()

    @property
    def data(self):
        return list(self.deque)


class MetricLogger(object):
    def __init__(self, delimiter=" "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def data(self):
        loss_dict = {}
        for name, meter in self.meters.items():
            if "time" in name:
                loss_dict[name] = meter.sum
            else:
                loss_dict[name] = meter.avg
        return loss_dict

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if name == "time":
                continue
            loss_str.append("{}: {:.4f}".format(name, meter.avg))
        return self.delimiter.join(loss_str)

    def str_epoch(self, idx):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.data[idx]))
        return self.delimiter.join(loss_str)


def print_meter(logger, train_meter, valid_meter):
    epoch = len(train_meter.cost.data)
    meter_dict = {"Train": train_meter, "\tValid": valid_meter}
    names = train_meter.meters.keys()
    for idx in range(epoch):
        for mode, meters in meter_dict.items():
            logger.info(
                "{mode} [E{epoch}]: {meters}".format(
                    mode=mode,
                    epoch=idx,
                    meters=meters.str_epoch(idx),
                )
            )
