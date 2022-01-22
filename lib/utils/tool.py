import numpy as np


def iou_with_anchors(gt_boxes, anchors):
    """Compute IoU between gt_boxes and anchors.
    gt_boxes: np.array shape [N, 2]
    anchors:  np.array shape [D*T, 2]
    """

    N = gt_boxes.shape[0]
    M = anchors.shape[0]

    gt_areas = (gt_boxes[:, 1] - gt_boxes[:, 0]).reshape(1, N)
    anchors_areas = (anchors[:, 1] - anchors[:, 0]).reshape(M, 1)

    boxes = anchors.reshape(M, 1, 2).repeat(N, axis=1)
    query_boxes = gt_boxes.reshape(1, N, 2).repeat(M, axis=0)

    inter_max = np.minimum(boxes[:, :, 1], query_boxes[:, :, 1])
    inter_min = np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
    inter = np.maximum(inter_max - inter_min, 0.0)

    scores = inter / (anchors_areas + gt_areas - inter + 1e-6)  # shape [D*T, N]
    return scores


def ioa_with_anchors(gt_boxes, anchors):
    """Compute Intersection between gt_boxes and anchors.
    gt_boxes: np.array shape [N, 2]
    anchors:  np.array shape [T, 2]
    """

    N = gt_boxes.shape[0]
    M = anchors.shape[0]

    anchors_areas = (anchors[:, 1] - anchors[:, 0]).reshape(M, 1)

    boxes = anchors.reshape(M, 1, 2).repeat(N, axis=1)
    query_boxes = gt_boxes.reshape(1, N, 2).repeat(M, axis=0)

    inter_max = np.minimum(boxes[:, :, 1], query_boxes[:, :, 1])
    inter_min = np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
    inter = np.maximum(inter_max - inter_min, 0.0)

    scores = inter / (anchors_areas + 1e-6)  # shape [T, N]
    return scores


def boundary_choose(score_list):
    max_score = max(score_list)
    mask_high = score_list > max_score * 0.5
    score_list = list(score_list)
    score_middle = np.array([0.0] + score_list + [0.0])
    score_front = np.array([0.0, 0.0] + score_list)
    score_back = np.array(score_list + [0.0, 0.0])
    mask_peak = (score_middle > score_front) & (score_middle > score_back)
    mask_peak = mask_peak[1:-1]
    mask = (mask_high | mask_peak).astype("float32")
    return mask


def get_valid_mask(dscale, tscale):
    mask = np.zeros((dscale, tscale))
    for idx in range(dscale):
        for jdx in range(tscale):
            if jdx + idx < tscale:
                mask[idx, jdx] = 1
    return mask
