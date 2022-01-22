import os
import torch
import numpy as np
import pandas as pd
import json
import scipy.interpolate

from lib.utils.tool import iou_with_anchors, ioa_with_anchors, get_valid_mask

"""
In anet, we don't use sliding window because many actions are quiet long.
Thus, we rescale the video feature to fixed length.
"""


class VideoDataSet(torch.utils.data.Dataset):
    def __init__(self, mode="train", subset="train", cfg=None, logger=None):
        # MODE SETTINGS
        self.mode = mode
        self.subset = subset
        self.datasetlib_path = "./lib/dataset/anet_1_3/data/"
        self.printer = logger.info if logger is not None else print

        # MODEL SETTINGS
        self.tscale = cfg.DATASET.tscale
        self.dscale = cfg.DATASET.dscale
        self.pos_thresh = cfg.LOSS.pos_thresh

        # FEATURE SETTING
        self.feature_path = os.path.join(cfg.FEATURE.root, cfg.FEATURE.name)
        self.feat_dim = cfg.FEATURE.dim
        self.online_resize = cfg.FEATURE.online_resize

        self._getDatasetDict()
        self._get_match_map()
        self._get_valid_mask()

    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.datasetlib_path + "video_info_2020.csv")
        anno_database = json.load(open(self.datasetlib_path + "activity_net_1_3_new.json"))
        anno_database = anno_database["database"]

        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name[2:]]
            video_info["duration_second"] = anno_df.duration_second.values[i]
            video_subset = anno_df.subset.values[i]

            # filter some dirty videos
            gt_cnt = 0
            for gt in video_info["annotations"]:
                tmp_start = max(min(1, gt["segment"][0] / video_info["duration_second"]), 0)
                tmp_end = max(min(1, gt["segment"][1] / video_info["duration_second"]), 0)
                if tmp_end - tmp_start > 0.01:
                    gt_cnt += 1
            if gt_cnt == 0 and video_subset == "training":
                continue

            if self.subset in video_subset:
                self.video_dict[video_name] = video_info

        self.video_list = list(self.video_dict.keys())
        self.printer("{} subset video numbers: {}".format(self.subset, len(self.video_list)))

    def _get_video_label(self, index):
        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        video_second = video_info["duration_second"]
        video_labels = video_info["annotations"]

        gt_bbox = []
        for gt in video_labels:
            tmp_start = max(min(1, gt["segment"][0] / video_second), 0)
            tmp_end = max(min(1, gt["segment"][1] / video_second), 0)
            if tmp_end - tmp_start < 0.01 and self.subset == "train":
                continue
            else:
                gt_bbox.append([tmp_start, tmp_end])
        gt_bbox = np.array(gt_bbox)

        # gt start end
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        bboxes_len = 3.0 / self.tscale
        gt_start_bboxs = np.stack((gt_xmins - bboxes_len / 2, gt_xmins + bboxes_len / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - bboxes_len / 2, gt_xmaxs + bboxes_len / 2), axis=1)

        gt_start = ioa_with_anchors(gt_start_bboxs, self.temporal_anchor)  # [T,N]
        gt_start = np.max(gt_start, axis=1)  # [T]
        gt_end = ioa_with_anchors(gt_end_bboxs, self.temporal_anchor)
        gt_end = np.max(gt_end, axis=1)

        # gt iou map
        iou_map = iou_with_anchors(gt_bbox, self.map_anchor)
        iou_map = iou_map.reshape([self.dscale, self.tscale, -1])  # [D,T,N]
        gt_iou_map = np.max(iou_map, axis=2)  # [D,T]
        gt_iou_map_index = np.argmax(iou_map, axis=2)  # [D,T]

        # gt iou weight
        gt_iou_weight = np.ones((self.dscale, self.tscale))
        pos_mask = (gt_iou_map > self.pos_thresh) * self.valid_mask == 1  # [D, T]
        pos_num = iou_map * np.expand_dims(self.valid_mask, 2).repeat(len(gt_bbox), axis=2)
        pos_num = (pos_num > self.pos_thresh).sum(axis=(0, 1)) + 1  # [N], avoid 0
        gt_iou_weight[pos_mask] = 1 / pos_num[gt_iou_map_index[pos_mask]]

        neg_mask = (gt_iou_map <= self.pos_thresh) * self.valid_mask == 1
        neg_num = np.sum(self.valid_mask) - np.sum(pos_mask)
        gt_iou_weight[neg_mask] = len(gt_bbox) / neg_num
        gt_iou_weight[self.valid_mask == 0] = 0

        # to Tensor
        gt_start = torch.Tensor(gt_start)
        gt_end = torch.Tensor(gt_end)
        gt_iou_map = torch.Tensor(gt_iou_map)
        gt_iou_weight = torch.Tensor(gt_iou_weight)
        return (gt_start, gt_end, gt_iou_map, gt_iou_weight)

    def _get_base_feat(self, index):
        video_info = {}
        video_name = self.video_list[index]

        if self.online_resize:
            video_data = np.load(os.path.join(self.feature_path, "{}.npy".format(video_name)))
            video_data = self._pool_data(video_data, num_prop=self.tscale)
        else:
            df_path = os.path.join(self.feature_path, "csv_mean_{}".format(self.tscale), video_name + ".csv")
            video_df = pd.read_csv(df_path)
            video_data = video_df.values[:, :]

        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data, 0, 1)  # [C,T]
        video_info["video_name"] = video_name
        video_info["indices"] = self.anchor_xmin
        return video_info, video_data

    def _pool_data(self, data, num_prop=100, num_bin=1, num_sample_bin=3, pool_type="mean"):
        if len(data) == 1:
            video_feature = np.stack([data] * num_prop)
            video_feature = np.reshape(video_feature, [num_prop, self.feat_dim])
            return video_feature

        # x is the temporal location corresponding to each location  in feature sequence
        # st = video_second / len(data)
        x = [0.5 + ii for ii in range(len(data))]
        f = scipy.interpolate.interp1d(x, data, axis=0)

        video_feature = []
        zero_sample = np.zeros(num_bin * self.feat_dim)
        tmp_anchor_xmin = [1.0 / num_prop * i for i in range(num_prop)]
        tmp_anchor_xmax = [1.0 / num_prop * i for i in range(1, num_prop + 1)]

        num_sample = num_bin * num_sample_bin
        for idx in range(num_prop):
            xmin = max(x[0] + 0.0001, tmp_anchor_xmin[idx] * len(data))
            xmax = min(x[-1] - 0.0001, tmp_anchor_xmax[idx] * len(data))
            if xmax < x[0]:
                video_feature.append(zero_sample)
                continue
            if xmin > x[-1]:
                video_feature.append(zero_sample)
                continue

            plen = (xmax - xmin) / (num_sample - 1)
            x_new = [xmin + plen * ii for ii in range(num_sample)]
            y_new = f(x_new)
            y_new_pool = []
            for b in range(num_bin):
                tmp_y_new = y_new[num_sample_bin * b : num_sample_bin * (b + 1)]
                if pool_type == "mean":
                    tmp_y_new = np.mean(y_new, axis=0)
                elif pool_type == "max":
                    tmp_y_new = np.max(y_new, axis=0)
                y_new_pool.append(tmp_y_new)
            y_new_pool = np.stack(y_new_pool)
            y_new_pool = np.reshape(y_new_pool, [-1])
            video_feature.append(y_new_pool)
        video_feature = np.stack(video_feature)
        return video_feature

    def _get_match_map(self):
        self.anchor_xmin = np.array([i / self.tscale for i in range(self.tscale)])
        self.anchor_xmax = np.array([i / self.tscale for i in range(1, self.tscale + 1)])

        tmp_xmin = [i / self.tscale for i in range(self.tscale)]
        tmp_xmax = [i / self.tscale for i in range(1, self.tscale + 1)]
        self.temporal_anchor = np.stack([tmp_xmin, tmp_xmax], axis=1)

        map_anchor = []
        for idx in range(self.dscale):
            for jdx in range(self.tscale):
                if jdx + idx < self.tscale:
                    xmin = float(self.anchor_xmin[jdx])
                    xmax = float(self.anchor_xmax[jdx + idx])
                    map_anchor.append([xmin, xmax])
                else:
                    map_anchor.append([0, 0])
        self.map_anchor = np.array(map_anchor)

    def _get_valid_mask(self):
        self.valid_mask = get_valid_mask(self.dscale, self.tscale)

    def __getitem__(self, index):
        video_info, video_data = self._get_base_feat(index)
        if self.mode == "train":
            gts = self._get_video_label(index)
            return video_info, video_data, gts
        elif self.mode == "infer":
            return video_info, video_data

    def __len__(self):
        return len(self.video_list)
