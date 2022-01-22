import json
import numpy as np
import pandas as pd
import multiprocessing as mp


def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011."""
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (
        (candidate_segments[:, 1] - candidate_segments[:, 0])
        + (target_segment[1] - target_segment[0])
        - segments_intersection
    )
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


class ANETdetection(object):

    GROUND_TRUTH_FIELDS = ["database", "taxonomy", "version"]
    PREDICTION_FIELDS = ["results", "version", "external_data"]

    def __init__(
        self,
        ground_truth_filename=None,
        prediction_filename=None,
        ground_truth_fields=GROUND_TRUTH_FIELDS,
        prediction_fields=PREDICTION_FIELDS,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
        subset="validation",
        verbose=True,
        blocked_videos=None,
        pp_num=8,
    ):
        if not ground_truth_filename:
            raise IOError("Please input a valid ground truth file.")
        if not prediction_filename:
            raise IOError("Please input a valid prediction file.")
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.pp_num = pp_num  # multiprocess workers

        # Get blocked videos
        if blocked_videos is None:
            self.blocked_videos = list()
        else:
            with open(blocked_videos) as json_file:
                self.blocked_videos = json.load(json_file)

        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print("[INIT] Loaded annotations from {} subset.".format(subset))
            nr_gt = len(self.ground_truth)
            print("\tNumber of ground truth instances: {}".format(nr_gt))
            nr_pred = len(self.prediction)
            print("\tNumber of predictions: {}".format(nr_pred))
            print("\tFixed threshold for tiou score: {}".format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, "r") as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in list(data.keys()) for field in self.gt_fields]):
            raise IOError("Please input a valid ground truth file.")

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data["database"].items():
            if self.subset != v["subset"]:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v["annotations"]:
                if ann["label"] not in activity_index:
                    activity_index[ann["label"]] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(float(ann["segment"][0]))
                t_end_lst.append(float(ann["segment"][1]))
                label_lst.append(activity_index[ann["label"]])

        ground_truth = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
            }
        )
        return ground_truth, activity_index

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, "r") as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in list(data.keys()) for field in self.pred_fields]):
            raise IOError("Please input a valid prediction file.")

        # Read predicitons.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for videoid, v in data["results"].items():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                label = self.activity_index[result["label"]]
                video_lst.append(videoid)
                t_start_lst.append(float(result["segment"][0]))
                t_end_lst.append(float(result["segment"][1]))
                label_lst.append(label)
                score_lst.append(result["score"])
        prediction = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
                "score": score_lst,
            }
        )
        return prediction

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset."""
        ap = np.zeros((len(self.tiou_thresholds), len(list(self.activity_index.items()))))
        for activity, cidx in self.activity_index.items():

            gt_idx = self.ground_truth["label"] == cidx
            pred_idx = self.prediction["label"] == cidx
            ap[:, cidx] = compute_average_precision_detection(
                self.ground_truth.loc[gt_idx].reset_index(drop=True),
                self.prediction.loc[pred_idx].reset_index(drop=True),
                tiou_thresholds=self.tiou_thresholds,
            )
            # print activity,cidx,np.mean(ap[:,cidx])
        return ap

    def sub_wrapper_compute_average_precision(self, cidx_list):
        """Computes average precision for a sub class list."""
        for cidx in cidx_list:
            gt_idx = self.ground_truth["label"] == cidx
            pred_idx = self.prediction["label"] == cidx
            self.result_dict[cidx] = compute_average_precision_detection(
                self.ground_truth.loc[gt_idx].reset_index(drop=True),
                self.prediction.loc[pred_idx].reset_index(drop=True),
                tiou_thresholds=self.tiou_thresholds,
            )

    def multi_wrapper_compute_average_precision(self):
        self.result_dict = mp.Manager().dict()

        num_total = len(self.activity_index.values())
        num_per_thread = num_total / self.pp_num
        processes = []
        for tid in range(self.pp_num - 1):
            tmp_list = list(self.activity_index.values())[int(tid * num_per_thread) : int((tid + 1) * num_per_thread)]
            p = mp.Process(target=self.sub_wrapper_compute_average_precision, args=(tmp_list,))
            p.start()
            processes.append(p)

        tmp_list = list(self.activity_index.values())[int((self.pp_num - 1) * num_per_thread) :]
        p = mp.Process(target=self.sub_wrapper_compute_average_precision, args=(tmp_list,))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()

        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index.items())))
        for idx in range(len(self.activity_index.values())):
            ap[:, idx] = self.result_dict[idx]
        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.multi_wrapper_compute_average_precision()
        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()
        if self.verbose:
            print("[RESULTS] Performance on ActivityNet detection task.")
            print("\tAverage-mAP: {}".format(self.mAP.mean()))
        return self.mAP, self.average_mAP


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction["score"].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred["video-id"])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[["t-start", "t-end"]].values, this_gt[["t-start", "t-end"]].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    ap = np.zeros(len(tiou_thresholds))

    recs = []
    for tidx in range(len(tiou_thresholds)):
        # Computing prec-rec
        this_tp = np.cumsum(tp[tidx, :]).astype(np.float)
        this_fp = np.cumsum(fp[tidx, :]).astype(np.float)
        rec = this_tp / npos
        prec = this_tp / (this_tp + this_fp)
        ap[tidx] = interpolated_prec_rec(prec, rec)

    return ap
