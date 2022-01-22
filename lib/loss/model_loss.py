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
        gt_start, gt_end, gt_iou_map = video_gt
        (tem_s, tem_e), pem_out = pred

        # ------ TBD loss ------
        loss_tem_s = bi_loss(gt_start, tem_s)
        loss_tem_e = bi_loss(gt_end, tem_e)
        loss_tem = loss_tem_s + loss_tem_e

        # ------  pem loss ------
        # classification loss - scale invariant loss
        loss_pem_cls = bl_loss(pem_out[:, 0, :, :], gt_iou_map, self.mask)

        # regression loss - l2 loss
        loss_pem_reg = l2_loss(pem_out[:, 1, :, :], gt_iou_map, self.mask)

        loss_pem = self.cfg.LOSS.coef_pem_cls * loss_pem_cls + self.cfg.LOSS.coef_pem_reg * loss_pem_reg

        # -------- Total Cost --------
        cost = loss_tem + loss_pem

        loss_dict = {}
        loss_dict["cost"] = cost
        loss_dict["tem"] = loss_tem
        loss_dict["pem"] = loss_pem
        return cost, loss_dict


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
