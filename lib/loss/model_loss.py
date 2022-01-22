import torch
import torch.nn.functional as F
from lib.utils.tool import get_valid_mask


class TAL_loss(object):
    def __init__(self, cfg):
        super(TAL_loss, self).__init__()
        self.cfg = cfg
        self.mask = torch.Tensor(get_valid_mask(cfg.DATASET.dscale, cfg.DATASET.tscale))

    def __call__(self, pred, video_gt):
        self.mask = self.mask.cuda()

        # unpack gt and prediction
        gt_start, gt_end, gt_iou_map, gt_iou_weight = video_gt
        (tbd_ls, tbd_le, tbd_gs, tbd_ge), imr_out = pred

        # ------ TBD loss ------
        loss_tbd_ls = bi_loss(gt_start, tbd_ls)
        loss_tbd_le = bi_loss(gt_end, tbd_le)

        loss_tbd_gs = bi_loss(gt_start, tbd_gs)
        loss_tbd_ge = bi_loss(gt_end, tbd_ge)

        loss_tbd_local = loss_tbd_ls + loss_tbd_le
        loss_tbd_global = loss_tbd_gs + loss_tbd_ge

        loss_tbd = 0.5 * (loss_tbd_local + loss_tbd_global)

        # ------  pem loss ------
        # classification loss - scale invariant loss
        loss_imr_cls = scale_invariant_loss(
            imr_out[:, 0, :, :],
            gt_iou_map,
            gt_iou_weight,
            self.mask,
            pos_thresh=self.cfg.LOSS.pos_thresh,
            alpha=self.cfg.LOSS.coef_alpha,
        )
        # regression loss - l2 loss
        loss_imr_reg = l2_loss(
            imr_out[:, 1, :, :],
            gt_iou_map,
            self.mask,
        )
        loss_imr = self.cfg.LOSS.coef_imr_cls * loss_imr_cls + self.cfg.LOSS.coef_imr_reg * loss_imr_reg

        # -------- Total Cost --------
        cost = loss_tbd + loss_imr

        loss_dict = {}
        loss_dict["cost"] = cost
        loss_dict["tbd"] = loss_tbd
        loss_dict["imr"] = loss_imr
        return cost, loss_dict


def scale_invariant_loss(output, gt_iou_map, gt_iou_weight, valid_mask, pos_thresh=0.9, alpha=0.2):
    gt_iou_map = gt_iou_map.cuda()

    pmask = (gt_iou_map > pos_thresh).float()
    nmask = (gt_iou_map <= pos_thresh).float() * valid_mask

    gt_iou_weight = gt_iou_weight.cuda()
    loss_pos = alpha * pmask * torch.log(output + 1e-6)
    loss_neg = (1 - alpha) * nmask * torch.log(1.0 - output + 1e-6)
    loss = -torch.sum((loss_pos + loss_neg) * gt_iou_weight) / output.shape[0]
    return loss


def bl_loss(output, gt_iou_map, valid_mask):
    bl_positive_thresh = 0.9
    gt_iou_map = gt_iou_map.cuda()

    pmask = (gt_iou_map > bl_positive_thresh).float()
    nmask = (gt_iou_map <= bl_positive_thresh).float() * valid_mask

    num_pos = torch.sum(pmask)
    num_entries = num_pos + torch.sum(nmask)
    ratio = num_entries / num_pos
    coef_0 = 0.5 * (ratio) / (ratio - 1)
    coef_1 = 0.5 * (ratio)

    loss = coef_1 * pmask * torch.log(output + 1e-6) + coef_0 * nmask * torch.log(1.0 - output + 1e-6)
    loss = -torch.sum(loss) / num_entries
    return loss


def bi_loss(gt, pred, pos_thresh=0.5):
    gt = gt.view(-1).cuda()
    pred = pred.contiguous().view(-1)

    pmask = (gt > pos_thresh).float().cuda()
    nmask = (gt <= pos_thresh).float().cuda()

    num_pos = torch.sum(pmask)
    num_neg = torch.sum(nmask)

    if num_pos == 0:  # not have positive sample
        loss = nmask * torch.log(1.0 - pred + 1e-6)
        loss = -torch.sum(loss) / num_neg
        return loss

    if num_neg == 0:  # not have negative sample
        loss = pmask * torch.log(pred + 1e-6)
        loss = -torch.sum(loss) / num_pos
        return loss

    coef_pos = 0.5 * (num_pos + num_neg) / num_pos
    coef_neg = 0.5 * (num_pos + num_neg) / num_neg

    loss = coef_pos * pmask * torch.log(pred + 1e-6) + coef_neg * nmask * torch.log(1.0 - pred + 1e-6)
    loss = -torch.mean(loss)
    return loss


def l2_loss(output, gt_iou_map, valid_mask):
    gt_iou_map = gt_iou_map.cuda() * valid_mask

    u_hmask = (gt_iou_map > 0.7).float()
    u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
    u_lmask = (gt_iou_map <= 0.3).float() * valid_mask

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    r_m = num_h / (num_m)
    u_smmask = torch.rand(u_hmask.shape).cuda()
    u_smmask = u_smmask * u_mmask
    u_smmask = (u_smmask > (1 - r_m)).float()

    r_l = num_h / (num_l)
    u_slmask = torch.rand(u_hmask.shape).cuda()
    u_slmask = u_slmask * u_lmask
    u_slmask = (u_slmask > (1 - r_l)).float()

    mask = u_hmask + u_smmask + u_slmask
    loss = F.mse_loss(output, gt_iou_map, reduction="none")
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss
