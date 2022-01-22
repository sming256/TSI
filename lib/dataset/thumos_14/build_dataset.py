import torch
import os
import numpy as np
import json
import h5py
import pickle

from lib.utils.tool import iou_with_anchors, ioa_with_anchors, get_valid_mask


"""
The annotation of following videos are wrong, thus we ignore.
['video_test_0000270', 'video_test_0001496', 'video_validation_0000947']
"""


class VideoDataSet(torch.utils.data.Dataset):
    def __init__(self, mode="train", subset="train", cfg=None, logger=None):
        # MODE SETTINGS
        self.mode = mode
        self.subset = subset
        self.datasetlib_path = "./lib/dataset/thumos_14/data/"
        self.printer = logger.info if logger is not None else print

        # MODEL SETTINGS
        self.tscale = cfg.DATASET.tscale
        self.dscale = cfg.DATASET.dscale
        self.pos_thresh = cfg.LOSS.pos_thresh

        # FEATURE SETTING
        self.snippet_stride = cfg.FEATURE.snippet_stride
        self.feature_path = os.path.join(cfg.FEATURE.root, cfg.FEATURE.name)
        self.feat_dim = cfg.FEATURE.dim

        self._getDataset_feat()
        self._get_match_map()
        self._get_valid_mask()

    def _getDataset_feat(self):
        # build data
        snippet_stride = self.snippet_stride  # sliding snippet stride
        window_size = self.tscale  # detector window
        window_stride = window_size // 2  # sliding window stride

        list_gt_bbox = []
        list_gt_label = []
        list_videos = []
        list_indices = []
        list_data = []

        anno_database = json.load(open(os.path.join(self.datasetlib_path, "thumos_14_anno.json")))
        self.anno_database = anno_database["database"]

        for video_name in list(self.anno_database.keys()):
            # pgcn regards this video as wrong
            if "i3d_pgcn" in self.feature_path:
                if video_name in ["video_validation_0000947"]:
                    continue

            video_info = self.anno_database[video_name]
            if self.subset not in video_info["subset"]:
                continue

            if self.mode == "train":
                gt_bbox, gt_label = self._get_gt(video_info)

            df_data = self._get_unresized_feature(video_name)
            num_snippet = df_data.shape[0]
            df_snippet = [self.snippet_stride * i for i in range(num_snippet)]
            num_windows = (num_snippet + window_stride - window_size) // window_stride
            windows_start = [i * window_stride for i in range(num_windows)]
            if num_snippet < window_size:
                windows_start = [0]
                # Add on a bunch of zero data if there aren't enough windows.
                tmp_data = np.zeros((window_size - num_snippet, self.feat_dim))
                df_data = np.concatenate((df_data, tmp_data), axis=0)
                df_snippet.extend(
                    [df_snippet[-1] + self.snippet_stride * (i + 1) for i in range(window_size - num_snippet)]
                )
            elif num_snippet - windows_start[-1] - window_size > int(window_size / snippet_stride):
                windows_start.append(num_snippet - window_size)

            for start in windows_start:
                end = start + window_size
                tmp_snippets = self._get_pad_snippet(df_snippet, start, end)
                tmp_data = self._get_pad_data(df_data, start, end)

                if self.mode == "train":
                    tmp_anchor_xmins = tmp_snippets - self.snippet_stride / 2.0
                    tmp_anchor_xmaxs = tmp_snippets + self.snippet_stride / 2.0
                    anchor = np.array([[tmp_anchor_xmins[0], tmp_anchor_xmaxs[-1]]])

                    tmp_ioa_list = ioa_with_anchors(anchor, gt_bbox)
                    tmp_ioa_list = tmp_ioa_list.reshape(-1)
                    tmp_gt_bbox = gt_bbox[tmp_ioa_list > 0]
                    tmp_gt_label = gt_label[tmp_ioa_list > 0]

                    if len(tmp_gt_bbox) > 0 and max(tmp_ioa_list) > 0.9:
                        list_gt_bbox.append(tmp_gt_bbox)
                        list_gt_label.append(tmp_gt_label)
                        list_videos.append(video_name)
                        list_indices.append(tmp_snippets)
                        list_data.append(np.array(tmp_data).astype(np.float32))
                elif "infer" in self.mode:
                    list_videos.append(video_name)
                    list_indices.append(tmp_snippets)
                    list_data.append(np.array(tmp_data).astype(np.float32))

        self.data = {"video_names": list_videos, "indices": list_indices, "video_data": list_data}
        if self.mode == "train":
            self.data.update({"gt_bbox": list_gt_bbox, "gt_label": list_gt_label})
        self.printer(
            "{} subset: {} videos, trunced as {} windows".format(
                self.subset,
                len(set(list_videos)),
                len(list_videos),
            )
        )

    def _get_unresized_feature(self, video_name):
        # get unresized feature of the video
        if "i3d_pgcn" in self.feature_path:
            rgb_data = torch.load(os.path.join(self.feature_path, "RGB", video_name))
            flow_data = torch.load(os.path.join(self.feature_path, "Flow", video_name))
            num_snippet = min(flow_data.shape[0], rgb_data.shape[0])
            df_data = np.concatenate([flow_data[:num_snippet, :], rgb_data[:num_snippet, :]], axis=1)
        elif "tsn_gtad" in self.feature_path:
            if "train" in self.subset:
                flow_data = h5py.File(os.path.join(self.feature_path, "flow_val.h5"), "r")
                rgb_data = h5py.File(os.path.join(self.feature_path, "rgb_val.h5"), "r")
            elif "val" in self.subset:
                flow_data = h5py.File(os.path.join(self.feature_path, "flow_test.h5"), "r")
                rgb_data = h5py.File(os.path.join(self.feature_path, "rgb_test.h5"), "r")
            feature_h5s = [
                flow_data[video_name][:: self.snippet_stride, ...],
                rgb_data[video_name][:: self.snippet_stride, ...],
            ]
            num_snippet = min([h5.shape[0] for h5 in feature_h5s])
            df_data = np.concatenate([h5[:num_snippet, :] for h5 in feature_h5s], axis=1)
        elif "i3d_butal" in self.feature_path:
            df_data = np.load(os.path.join(self.feature_path, video_name + ".npy"))
        else:
            raise "This feature processing is not supported"
        return df_data

    def _get_gt(self, video_info):
        gt_bbox = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            gt_start = anno["segment"][0] / video_info["duration"] * video_info["frame"]
            gt_end = anno["segment"][1] / video_info["duration"] * video_info["frame"]
            gt_bbox.append([gt_start, gt_end])
            gt_label.append(anno["label"])
        return np.array(gt_bbox), np.array(gt_label)

    def _get_pad_data(self, data, start, end):
        real_start = max(0, start)
        tmp_start = real_start - start

        real_end = min(len(data), end)
        tmp_end = end - real_end

        real_data = data[real_start:real_end]
        pad_data_start = np.zeros((tmp_start,) + data.shape[1:])
        pad_data_end = np.zeros((tmp_end,) + data.shape[1:])
        pad_data = np.concatenate((pad_data_start, real_data, pad_data_end), axis=0)
        return pad_data

    def _get_pad_snippet(self, data, start, end):
        real_start = max(0, start)
        tmp_start = real_start - start

        real_end = min(len(data), end)
        tmp_end = end - real_end

        real_data = np.array(data[real_start:real_end])
        pad_data_start = 0 - np.arange(tmp_start, 0, -1) * self.snippet_stride
        pad_data_end = data[-1] + np.arange(1, tmp_end + 1) * self.snippet_stride
        pad_data = np.concatenate((pad_data_start, real_data, pad_data_end), axis=0)
        return pad_data

    def _get_video_label(self, index):
        video_name = self.data["video_names"][index]
        gt_bbox = self.data["gt_bbox"][index]
        tmp_snippets = self.data["indices"][index]
        offset = int(min(tmp_snippets - self.snippet_stride // 2))
        gt_bbox = gt_bbox - offset

        # gt start end
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.snippet_stride
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)

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
        video_info["video_name"] = self.data["video_names"][index]
        video_info["indices"] = self.data["indices"][index]
        video_data = self.data["video_data"][index]

        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data, 0, 1)
        return video_info, video_data

    def _get_match_map(self):
        self.anchor_xmin = [i * self.snippet_stride for i in range(self.tscale)]
        self.anchor_xmax = [i * self.snippet_stride for i in range(1, self.tscale + 1)]

        tmp_xmin = [i * self.snippet_stride for i in range(self.tscale)]
        tmp_xmax = [i * self.snippet_stride for i in range(1, self.tscale + 1)]
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
        return len(self.data["video_names"])
