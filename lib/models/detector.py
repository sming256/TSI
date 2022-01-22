import torch.nn as nn
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

        # Temporal Evaluation Module
        self.tem_s = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.tem_e = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Proposal Evaluation Module
        self.pem = PEM(in_dim=256, dscale=cfg.DATASET.dscale, tscale=cfg.DATASET.tscale)

        self.reset_params()

    def forward(self, x):
        x = self.base(x)

        # Temporal Evaluation Module
        tem_s = self.tem_s(x).squeeze(1)
        tem_e = self.tem_e(x).squeeze(1)
        tem_out = (tem_s, tem_e)

        # Proposal Evaluation Module
        pem_out = self.pem(x)
        return tem_out, pem_out

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
