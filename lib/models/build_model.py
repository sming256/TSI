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
        TEM_weight = []
        PEM_weight = []

        for name, p in model.named_parameters():
            if "base" in name:
                BASE_weight.append(p)
            if "tem" in name:
                TEM_weight.append(p)
            if "pem" in name:
                PEM_weight.append(p)

        optimizer = torch.optim.Adam(
            [
                {"params": BASE_weight, "weight_decay": 1e-3},
                {"params": TEM_weight, "weight_decay": 1e-4},
                {"params": PEM_weight, "weight_decay": 1e-4},
            ],
            lr=cfg.SOLVER.lr,
        )
        return optimizer
