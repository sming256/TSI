import os
import sys

path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import numpy as np
import pandas as pd
import json
import time
import argparse
from lib.utils.util import load_config
from lib.utils.tool import get_valid_mask, boundary_choose
import multiprocessing as mp
import pickle

thumos_class = {
    7: "BaseballPitch",
    9: "BasketballDunk",
    12: "Billiards",
    21: "CleanAndJerk",
    22: "CliffDiving",
    23: "CricketBowling",
    24: "CricketShot",
    26: "Diving",
    31: "FrisbeeCatch",
    33: "GolfSwing",
    36: "HammerThrow",
    40: "HighJump",
    45: "JavelinThrow",
    51: "LongJump",
    68: "PoleVault",
    79: "Shotput",
    85: "SoccerPenalty",
    92: "TennisSwing",
    93: "ThrowDiscus",
    97: "VolleyballSpiking",
}


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor


def Soft_NMS(df, iou_threshold, sigma, num_prop=200):
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []

    # frost: I use a trick here, remove the detection XDD
    # which is longer than 300
    for idx in range(0, len(tscore)):
        if tend[idx] - tstart[idx] >= 300:
            tscore[idx] = 0

    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore) > 0:
        max_index = tscore.index(max(tscore))
        max_width = tend[max_index] - tstart[max_index]
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > iou_threshold:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / sigma)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf["score"] = rscore
    newDf["xmin"] = rstart
    newDf["xmax"] = rend
    return newDf


def getDatasetDict():
    json_data = json.load(open("./lib/dataset/thumos_14/data/thumos_14_anno.json"))
    database = json_data["database"]
    train_dict = {}
    val_dict = {}
    for video_name in list(database.keys()):
        video_info = database[video_name]
        video_new_info = {}
        video_new_info["frame"] = video_info["frame"]
        video_new_info["duration"] = video_info["duration"]
        video_subset = video_info["subset"]
        if video_subset == "training":
            train_dict[video_name] = video_new_info
        elif video_subset == "validation":
            val_dict[video_name] = video_new_info
    return train_dict, val_dict


def _gen_detection_video(video_list, cls_data, thu_label_id, video_dict, cfg, num_prop=200, topk=2):
    dscale = cfg.DATASET.dscale
    cols = ["xmin", "xmax", "score"]

    for video_name, video_cls in zip(video_list, cls_data):
        # read all files
        video_folder = os.path.join(cfg.output_path, video_name)
        try:
            files = [os.path.join(video_folder, f) for f in os.listdir(video_folder)]
        except:
            continue
        if len(files) == 0:
            print("Missing result for video {}".format(video_name))
        files.sort()

        # prepare whole video data
        num_frames = video_dict[video_name]["frame"]
        seconds = video_dict[video_name]["duration"]

        snippet_stride = cfg.FEATURE.snippet_stride
        snippet_num = num_frames // snippet_stride + 1
        v_snippets = [snippet_stride * i for i in range(snippet_num)]

        snippet_num = len(v_snippets)
        v_iou_map = np.zeros((cfg.DATASET.dscale, snippet_num))
        v_tem_s = np.zeros(snippet_num)
        v_tem_e = np.zeros(snippet_num)

        # proposal might be duplicated due to sliding window, need to mean by count
        v_iou_map_cnt = np.zeros((cfg.DATASET.dscale, snippet_num))
        v_tem_cnt = np.zeros(snippet_num)

        for snippet_file in files:
            # load output result
            with open(snippet_file, "rb") as infile:
                result_data = pickle.load(infile)

            [tmp_snippet, pred_local_s, pred_local_e, pred_global_s, pred_global_e, pred_iou_map] = result_data

            pred_start = np.sqrt(pred_local_s * pred_global_s)
            pred_end = np.sqrt(pred_local_e * pred_global_e)

            # get true index of current window to aviod invalid time stamp
            start_idx = int(min(np.argwhere(tmp_snippet >= 0)))
            end_idx = int(max(np.argwhere(tmp_snippet <= num_frames)))
            true_start = tmp_snippet[start_idx]
            true_end = tmp_snippet[end_idx]

            # get absolute index of whole video
            v_s_idx = v_snippets.index(true_start)
            v_e_idx = v_snippets.index(true_end)

            # push data to whole data
            v_tem_s[v_s_idx : v_e_idx + 1] += pred_start[start_idx : end_idx + 1]
            v_tem_e[v_s_idx : v_e_idx + 1] += pred_end[start_idx : end_idx + 1]

            iou_mask = get_valid_mask(dscale, end_idx - start_idx + 1)
            pred_iou_map = pred_iou_map[:, :, : end_idx - start_idx + 1]
            pred_iou_map = pred_iou_map[0, :, :] * pred_iou_map[1, :, :]
            v_iou_map[:, v_s_idx : v_e_idx + 1] += pred_iou_map * iou_mask

            # update count
            v_tem_cnt[v_s_idx : v_e_idx + 1] += 1
            v_iou_map_cnt[:, v_s_idx : v_e_idx + 1] += iou_mask

        v_tem_s /= v_tem_cnt + 1e-6
        v_tem_e /= v_tem_cnt + 1e-6
        v_iou_map /= v_iou_map_cnt + 1e-6

        start_mask = boundary_choose(v_tem_s)
        start_mask[0] = 1.0
        end_mask = boundary_choose(v_tem_e)
        end_mask[-1] = 1.0

        score_vector_list = []
        for idx in range(1, dscale):
            for jdx in range(snippet_num):
                start_idx = jdx
                end_idx = start_idx + idx
                if end_idx < snippet_num and start_mask[start_idx] == 1 and end_mask[end_idx] == 1:
                    xmin = v_snippets[start_idx]
                    xmax = v_snippets[end_idx]

                    xmin_score = v_tem_s[start_idx]
                    xmax_score = v_tem_e[end_idx]
                    bm_score = v_iou_map[idx, jdx]
                    conf_score = xmin_score * xmax_score * bm_score
                    score_vector_list.append([xmin, xmax, conf_score])

        score_vector_list = np.stack(score_vector_list)
        df = pd.DataFrame(score_vector_list, columns=cols)

        if len(df) > 1:
            df = Soft_NMS(
                df,
                iou_threshold=cfg.DETECTION_POST.iou_threshold,
                sigma=cfg.DETECTION_POST.sigma,
            )
        df = df.sort_values(by="score", ascending=False)

        # sort video classification
        video_cls_rank = sorted((e, i) for i, e in enumerate(video_cls))
        unet_classes = [thu_label_id[video_cls_rank[-k - 1][1]] + 1 for k in range(topk)]
        unet_scores = [video_cls_rank[-k - 1][0] for k in range(topk)]

        proposal_list = []
        for j in range(min(num_prop, len(df))):
            for k in range(topk):
                tmp_proposal = {}
                tmp_proposal["label"] = thumos_class[int(unet_classes[k])]
                tmp_proposal["score"] = float(round(df.score.values[j] * unet_scores[k], 6))

                tmp_xmin = max(0, df.xmin.values[j] / num_frames) * seconds
                tmp_xmax = min(1, df.xmax.values[j] / num_frames) * seconds
                tmp_proposal["segment"] = [
                    float(round(tmp_xmin, 1)),
                    float(round(tmp_xmax, 1)),
                ]
                proposal_list.append(tmp_proposal)
        result_dict[video_name] = proposal_list


def gen_detection_multicore(cfg):
    # get video list
    train_dict, val_dict = getDatasetDict()
    video_dict = val_dict
    video_list = list(video_dict.keys())

    # load video level classification
    thu_label_id = np.array(list(thumos_class.keys())) - 1  # get thumos class id
    thu_video_id = np.array([int(i[-4:]) - 1 for i in video_list])  # -1 is to match python index

    cls_data = np.load("./lib/dataset/thumos_14/data/uNet_test.npy")
    cls_data = cls_data[thu_video_id, :][:, thu_label_id]  # order by video list, output 213x20

    # detection_result
    global result_dict
    result_dict = mp.Manager().dict()

    # multi processing
    pp_num = 16
    num_videos = len(video_list)
    num_videos_per_thread = num_videos / pp_num
    processes = []
    for tid in range(pp_num - 1):
        tmp_video_list = video_list[int(tid * num_videos_per_thread) : int((tid + 1) * num_videos_per_thread)]
        tmp_cls_data = cls_data[int(tid * num_videos_per_thread) : int((tid + 1) * num_videos_per_thread)]
        p = mp.Process(
            target=_gen_detection_video,
            args=(tmp_video_list, tmp_cls_data, thu_label_id, video_dict, cfg),
        )
        p.start()
        processes.append(p)
    tmp_video_list = video_list[int((pp_num - 1) * num_videos_per_thread) :]
    tmp_cls_data = cls_data[int((pp_num - 1) * num_videos_per_thread) :]
    p = mp.Process(
        target=_gen_detection_video,
        args=(tmp_video_list, tmp_cls_data, thu_label_id, video_dict, cfg),
    )

    p.start()
    processes.append(p)
    for p in processes:
        p.join()

    # write file
    result_dict = dict(result_dict)
    output_dict = {"version": "THUMOS14", "results": result_dict, "external_data": {}}

    with open(cfg.result_path, "w") as out:
        json.dump(output_dict, out)


def detection_post(cfg):
    cfg.output_path = "./exps/%s/output/" % (cfg.EXP_NAME)
    cfg.result_path = "./exps/%s/result_detection.json" % (cfg.EXP_NAME)

    # post processing
    t1 = time.time()
    print("\nDetection task post processing start")
    gen_detection_multicore(cfg)
    t2 = time.time()
    print("Detection task Post processing finished, time=%.1fmins\n" % ((t2 - t1) / 60))

    # evaluation
    from lib.eval.eval_detection import ANETdetection

    tious = [0.3, 0.4, 0.5, 0.6, 0.7]
    anet_detection = ANETdetection(
        ground_truth_filename="./lib/dataset/thumos_14/data/thumos_14_anno.json",
        prediction_filename=cfg.result_path,
        subset="validation",
        tiou_thresholds=tious,
    )
    mAPs, average_mAP = anet_detection.evaluate()
    for (tiou, mAP) in zip(tious, mAPs):
        print("mAP at tIoU {} is {:.2f}%".format(tiou, mAP * 100))

    # save to file
    cfg.eval_path = "./exps/%s/results.txt" % (cfg.EXP_NAME)
    f2 = open(cfg.eval_path, "a")  # 不替换文件，继续写入
    f2.write("Average-mAP: {}\n".format(average_mAP))
    for (tiou, mAP) in zip(tious, mAPs):
        f2.write("mAP at tIoU {} is {:.2f}%\n".format(tiou, mAP * 100))
    f2.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMN")
    parser.add_argument(
        "config",
        default="configs/anet_tsn_scale_100.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()

    # load settings
    cfg = load_config(args.config)

    # post process
    detection_post(cfg)
