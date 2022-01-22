import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PEM(nn.Module):
    def __init__(self, in_dim=256, dscale=100, tscale=100):
        super(PEM, self).__init__()

        self.bm_layer = BM_layer(in_dim=in_dim, out_dim=128, dscale=dscale, tscale=tscale)
        self.fc = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        iou_2d = self.bm_layer(x)  # [B,C,D,T]
        iou_2d_out = self.fc(iou_2d)  # [B,2,D,T]
        return iou_2d_out


class BM_layer(nn.Module):
    """BM layer, implemented in BMN"""

    def __init__(self, in_dim, out_dim, dscale=100, tscale=100):
        super(BM_layer, self).__init__()

        self.num_sample = 32
        self.num_sample_perbin = 3
        self.prop_boundary_ratio = 0.5

        self.reduce_dim = nn.Conv1d(in_channels=in_dim, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3d = nn.Conv3d(
            128,
            512,
            kernel_size=(self.num_sample, 1, 1),
            stride=(self.num_sample, 1, 1),
        )
        self.conv2d = nn.Conv2d(512, out_dim, kernel_size=1, stride=1, padding=0)
        self._get_interp1d_mask(tscale, dscale)

    def forward(self, x):
        device = x.get_device()
        x = F.relu(self.reduce_dim(x))  # [-1, 128, tscale]
        self.sample_mask = self.sample_mask.cuda(device)
        map_base = torch.tensordot(x, self.sample_mask, dims=([2], [0]))  # [-1, 128, 32, dscale, tscale]
        map_3d = F.relu(self.conv3d(map_base))  # [-1, 512, 1, dscale, tscale]
        map_2d = map_3d.squeeze(2)  # [-1, 512, dscale, tscale]
        map_2d = F.relu(self.conv2d(map_2d))  # [-1, out_dim, dscale, tscale]
        return map_2d

    def _get_interp1d_mask(self, tscale, dscale):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for start_index in range(tscale):
            mask_mat_vector = []
            for duration_index in range(dscale):
                if start_index + duration_index < tscale:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin,
                        sample_xmax,
                        tscale,
                        self.num_sample,
                        self.num_sample_perbin,
                    )
                else:
                    p_mask = np.zeros([tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3).astype(np.float32)
        self.sample_mask = torch.Tensor(mask_mat)

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [seg_xmin + plen_sample * ii for ii in range(num_sample * num_sample_perbin)]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin : (idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask
