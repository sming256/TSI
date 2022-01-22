import torch.nn as nn
from .tbd import TBD
from .pem import PEM


class detector_model(nn.Module):
    def __init__(self, cfg):
        super(detector_model, self).__init__()

        # Basenet
        self.base = nn.Sequential(
            nn.Conv1d(cfg.FEATURE.dim, 256, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
        )

        # Temporal Boundary Detector
        self.tbd = TBD(in_dim=256, tscale=cfg.DATASET.tscale)

        # IoU Map Regressor
        self.imr = PEM(in_dim=256, dscale=cfg.DATASET.dscale, tscale=cfg.DATASET.tscale)

        self.reset_params()

    def forward(self, x):
        x = self.base(x)

        # Temporal Boundary Detector
        tbd_out = self.tbd(x)

        # IoU Map Regressor
        imr_out = self.imr(x)
        return tbd_out, imr_out

    def reset_params(self):
        for m in self.modules():
            if (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv3d)
            ):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
