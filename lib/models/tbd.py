import torch
import torch.nn as nn
import torch.nn.functional as F


class TBD(nn.Module):
    """Temporal Boundary Detector"""

    def __init__(self, in_dim, tscale=100):
        super(TBD, self).__init__()

        # ------ local branch  ------
        self.local_conv1d_s = nn.Conv1d(in_dim, 256, kernel_size=3, padding=1, groups=4)
        self.local_conv1d_s_out = nn.Conv1d(256, 1, kernel_size=1)

        self.local_conv1d_e = nn.Conv1d(in_dim, 256, kernel_size=3, padding=1, groups=4)
        self.local_conv1d_e_out = nn.Conv1d(256, 1, kernel_size=1)

        # ------ global branch ------
        chans = [128, 256, 512, 1024]
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.x1_1 = nn.Conv1d(in_dim, chans[0], kernel_size=3, stride=1, padding=1, groups=4)
        self.x2_1 = nn.Conv1d(chans[0], chans[1], kernel_size=3, stride=1, padding=1, groups=4)
        self.x3_1 = nn.Conv1d(chans[1], chans[2], kernel_size=3, stride=1, padding=1, groups=4)
        self.x4_1 = nn.Conv1d(chans[2], chans[3], kernel_size=3, stride=1, padding=1, groups=4)

        output_padding = 0 if tscale % (2 ** 3) == 0 else 1
        self.upcv41_to_32 = nn.ConvTranspose1d(
            chans[3],
            chans[2],
            kernel_size=2,
            stride=2,
            output_padding=output_padding,
            groups=4,
        )

        self.x3_2 = nn.Conv1d(chans[2] * 2, chans[2], kernel_size=3, stride=1, padding=1, groups=4)

        self.upcv31_to_22 = nn.ConvTranspose1d(chans[2], chans[1], kernel_size=2, stride=2, groups=4)
        self.x2_2 = nn.Conv1d(chans[1] * 2, chans[1], kernel_size=3, stride=1, padding=1, groups=4)

        self.upcv32_to_23 = nn.ConvTranspose1d(chans[2], chans[1], kernel_size=2, stride=2, groups=4)
        self.x2_3 = nn.Conv1d(chans[1] * 3, chans[1], kernel_size=3, stride=1, padding=1, groups=4)

        self.upcv21_to_12 = nn.ConvTranspose1d(chans[1], chans[0], kernel_size=2, stride=2, groups=4)
        self.x1_2 = nn.Conv1d(chans[0] * 2, chans[0], kernel_size=3, stride=1, padding=1, groups=4)

        self.upcv22_to_13 = nn.ConvTranspose1d(chans[1], chans[0], kernel_size=2, stride=2, groups=4)
        self.x1_3 = nn.Conv1d(chans[0] * 3, chans[0], kernel_size=3, stride=1, padding=1, groups=4)

        self.upcv23_to_14 = nn.ConvTranspose1d(chans[1], chans[0], kernel_size=2, stride=2, groups=4)
        self.x1_4 = nn.Conv1d(chans[0] * 4, chans[0], kernel_size=3, stride=1, padding=1, groups=4)

        self.global_s_out = nn.Conv1d(chans[0], 1, kernel_size=1)
        self.global_e_out = nn.Conv1d(chans[0], 1, kernel_size=1)

    def forward(self, x):
        # ------ local branch  ------
        tbd_local_s = F.relu(self.local_conv1d_s(x))
        tbd_local_s_out = torch.sigmoid(self.local_conv1d_s_out(tbd_local_s)).squeeze(1)

        tbd_local_e = F.relu(self.local_conv1d_e(x))
        tbd_local_e_out = torch.sigmoid(self.local_conv1d_e_out(tbd_local_e)).squeeze(1)

        # ------ global branch ------
        x1_1 = self.x1_1(x)
        x2_1 = self.x2_1(self.pool(x1_1))
        x3_1 = self.x3_1(self.pool(x2_1))
        x4_1 = self.x4_1(self.pool(x3_1))

        # layer 3
        x3_2 = self.x3_2(torch.cat((x3_1, self.upcv41_to_32(x4_1)), dim=1))

        # layer 2
        x2_2 = self.x2_2(torch.cat((x2_1, self.upcv31_to_22(x3_1)), dim=1))
        x2_3 = self.x2_3(torch.cat((x2_1, x2_2, self.upcv32_to_23(x3_2)), dim=1))

        # layer 1
        x1_2 = self.x1_2(torch.cat((x1_1, self.upcv21_to_12(x2_1)), dim=1))
        x1_3 = self.x1_3(torch.cat((x1_1, x1_2, self.upcv22_to_13(x2_2)), dim=1))
        x1_4 = self.x1_4(torch.cat((x1_1, x1_2, x1_3, self.upcv23_to_14(x2_3)), dim=1))

        tbd_global_s_out = torch.sigmoid(self.global_s_out(x1_4)).squeeze(1)
        tbd_global_e_out = torch.sigmoid(self.global_e_out(x1_4)).squeeze(1)

        tbd_out = (tbd_local_s_out, tbd_local_e_out, tbd_global_s_out, tbd_global_e_out)
        return tbd_out
