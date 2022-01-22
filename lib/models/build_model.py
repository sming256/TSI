import torch
from .detector import detector_model


class TAL_model(torch.nn.Module):
    def __init__(self, cfg):
        super(TAL_model, self).__init__()

        self.detector = detector_model(cfg)

    def forward(self, video_data):
        pred = self.detector(video_data)
        return pred

    @staticmethod
    def get_optimizer(model, cfg):
        BASE_weight = []
        TBD_weight = []
        IMR_weight = []

        for name, p in model.named_parameters():
            if "base" in name:
                BASE_weight.append(p)
            if "tbd" in name:
                TBD_weight.append(p)
            if "imr" in name:
                IMR_weight.append(p)

        optimizer = torch.optim.Adam(
            [
                {"params": BASE_weight, "weight_decay": 1e-3},
                {"params": TBD_weight, "weight_decay": 1e-4},
                {"params": IMR_weight, "weight_decay": 1e-4},
            ],
            lr=cfg.SOLVER.lr,
        )
        return optimizer
