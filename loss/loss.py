import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

__all__ = ["CrossEntropyAndBCELoss", 
           "CrossEntropyAndDiceLoss", 
           "BCELoss", 
           "CrossEntropyLoss", 
           "SoftDiceLoss", 
           "SoftDiceLossSquared",
           ]


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

    
def get_tp_fp_fn_tn(pred_softmax: Tensor, gt_seg: Tensor, mask=None, square=False):
    """
    Inputs:
        pred_softmax: (bs, num_class, h, w)
        gt_seg: (bs, 1, h, w)
        mask:
        square:
    Outputs:

    """
    num_class = pred_softmax.size(1)
    onehot_gt_seg = F.one_hot(gt_seg.squeeze(1).long(), num_classes=num_class)  # (bs, h, w, num_class)
    onehot_gt_seg = onehot_gt_seg.permute(0, 3, 1, 2)  # (bs, num_class, h, w)
    onehot_gt_seg.requires_grad = False
    
    tp = pred_softmax * onehot_gt_seg
    fp = pred_softmax * (1 - onehot_gt_seg)
    fn = (1 - pred_softmax) * onehot_gt_seg
    tn = (1 - pred_softmax) * (1 - onehot_gt_seg)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    axes = tuple([0] + list(range(2, len(pred_softmax.size()))))  # [0, 2, 3]
    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLossSquared(nn.Module):
    def __init__(self, do_bg=True, smooth=1.):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()
        self.do_bg = do_bg
        self.smooth = smooth

    def forward(self, pred, gt_seg):
        """
        Inputs:
            pred: (bs, num_class, h, w) / prediction
            gt_seg: (bs, 1, h, w) / gt segmentation
        """
        pred_softmax  = F.softmax(pred, dim=1)  # (bs, num_class, h, w)
        shp_x = pred_softmax.shape

        # equal to: onehot_seg = F.one_hot(gt_seg.squeeze(1).long(), num_classes=num_class).permute(0, 3, 1, 2)
        with torch.no_grad():
            gt_seg = gt_seg.long()
            y_onehot = torch.zeros(shp_x)
            if pred_softmax.device.type == "cuda":
                y_onehot = y_onehot.cuda(pred_softmax.device.index)
            y_onehot.scatter_(1, gt_seg, 1).float()  

        intersect   = pred_softmax * y_onehot
        # values in the denominator get smoothed
        denominator = pred_softmax ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after squaring)
        axes = tuple([0] + list(range(2, len(pred_softmax.size()))))  # [0, 2, 3]
        intersect   = sum_tensor(intersect, axes, False)   + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            dc = dc[1:]
           
        dc = dc.mean()
        return -dc


class SoftDiceLoss(nn.Module):
    def __init__(self, do_bg=True, smooth=1.):
        """
        Inputs:
            do_bg: whether compute bg loss
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.smooth = smooth

    def forward(self, pred, gt_seg, loss_mask=None):
        """
        Inputs:
            pred: (bs, num_class, h, w) / prediction
            gt_seg: (bs, 1, h, w) / gt segmentation
        """
        pred_softmax  = F.softmax(pred, dim=1)  # (bs, num_class, h, w)
        tp, fp, fn, _ = get_tp_fp_fn_tn(pred_softmax, gt_seg, loss_mask, False)

        nominator   = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth
        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            dc = dc[1:]

        dc = dc.mean()
        return -dc


class CrossEntropyAndDiceLoss:
    def __init__(self, num_class, weight_dc=1.0, weight_ce=1.0, ignore_index=0) -> None:
        self.num_class = num_class
        self.ignore_index = ignore_index
        self.ce   = nn.CrossEntropyLoss(reduction='mean')
        self.dc = SoftDiceLoss(do_bg=False if ignore_index is not None else True)
        self.weight_dc = weight_dc
        self.weight_ce = weight_ce

    def __call__(self, pred_out_dict, gt_seg):
        ce_loss_dict, dc_loss_dict = {}, {}
        ce_tot, dc_tot = gt_seg.new_zeros(1), gt_seg.new_zeros(1)

        for k, v in pred_out_dict.items():
            # ce
            ce_loss = self.ce(v, gt_seg.squeeze(1).long())
            ce_tot += ce_loss
            ce_loss_dict[k] = ce_loss

            # dice
            dc_loss = self.dc(v, gt_seg)
            dc_tot += dc_loss

        ce_loss_dict["total_loss"] = ce_tot * self.weight_ce
        dc_loss_dict['total_loss'] = dc_tot * self.weight_dc
        return {"total_loss": ce_tot[0] + dc_tot[0]}


class CrossEntropyAndBCELoss:

    def __init__(self, num_class, weight_ce=1.0, weight_bce=1.0, ignore_index=0) -> None:
        super(CrossEntropyAndBCELoss, self).__init__()
        self.num_class = num_class
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
        self.weight_ce = weight_ce
        self.weight_bce = weight_bce
       
    def __call__(self, pred_out_dict, gt_seg):
        """
        Inputs:
            pred_out_dict: 
                {"stage1": torch.sigmoid(out_stage1),   # (bs, out, h, w)
                "stage2": torch.sigmoid(out_stage2),   # (bs, out, h, w)
                "stage3": torch.sigmoid(out_stage3),   # (bs, out, h, w) 
                "stage4": torch.sigmoid(out_stage4),   # (bs, out, h, w) 
                "stage5": torch.sigmoid(out_stage5),   # (bs, out, h, w) 
                "stage6": torch.sigmoid(out_stage6),   # (bs, out, h, w) 
                "concat": torch.sigmoid(out_concat),   # (bs, out, h, w)
                }
            gt_seg: LongTensor of shape (bs, h, w)

        Outputs:
            total_loss
        """
        
        ce_out_dict, bce_out_dict = {}, {}
        ce_tot, bce_tot = gt_seg.new_zeros(1), gt_seg.new_zeros(1)
        batch_size = gt_seg.size(0)
        for k, v in pred_out_dict.items():
            # ce
            ce_loss = self.ce(v, gt_seg.squeeze(1).long())
            ce_tot += ce_loss
            ce_out_dict[k] = ce_loss

            # bce
            with torch.no_grad():
                onehot_seg = F.one_hot(gt_seg.squeeze(1).long(), num_classes=self.num_class)  # (bs, h, w, num_class)
                onehot_seg = onehot_seg.permute(0, 3, 1, 2)  # (bs, num_class, h, w)
                onehot_seg.requires_grad = False

            bce_loss = self.bce(v, onehot_seg.float())  # (bs, num_class, h, w)
            bce_loss = bce_loss[:, 1:, :, :].mean()  # mask值为0的区域不计入loss
            bce_out_dict[k] = bce_loss
            bce_tot += bce_loss

        
        ce_out_dict["total_loss"] = ce_tot * self.weight_ce
        bce_out_dict['total_loss'] = bce_tot * batch_size * self.weight_bce

        return {"total_loss": ce_tot[0] + bce_tot[0]}


class CrossEntropyLoss:
    def __init__(self, num_class, ignore_index=0) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.num_class = num_class
        self.ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
       
    def __call__(self, pred_out_dict, gt_seg):
        """
        Inputs:
            pred_out_dict: 
                {"stage1": torch.sigmoid(out_stage1),   # (bs, out, h, w)
                "stage2": torch.sigmoid(out_stage2),   # (bs, out, h, w)
                "stage3": torch.sigmoid(out_stage3),   # (bs, out, h, w) 
                "stage4": torch.sigmoid(out_stage4),   # (bs, out, h, w) 
                "stage5": torch.sigmoid(out_stage5),   # (bs, out, h, w) 
                "stage6": torch.sigmoid(out_stage6),   # (bs, out, h, w) 
                "concat": torch.sigmoid(out_concat),   # (bs, out, h, w)
                }
            gt_seg: LongTensor of shape (bs, h, w)

        Outputs:
            total_loss
        """
        
        ce_out_dict = {}
        ce_tot = gt_seg.new_zeros(1)
        for k, v in pred_out_dict.items():
            # ce
            ce_loss = self.ce(v, gt_seg.squeeze(1).long())
            ce_tot += ce_loss
            ce_out_dict[k] = ce_loss

        ce_out_dict["total_loss"] = ce_tot
        return {"total_loss": ce_tot[0]}


class BCELoss:

    def __init__(self, num_class, do_bg=False) -> None:
        super(BCELoss, self).__init__()
        self.num_class = num_class
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.do_bg = do_bg
       
    def __call__(self, pred_out_dict, gt_seg):
        """
        Inputs:
            pred_out_dict: 
                {"stage1": torch.sigmoid(out_stage1),   # (bs, out, h, w)
                "stage2": torch.sigmoid(out_stage2),   # (bs, out, h, w)
                "stage3": torch.sigmoid(out_stage3),   # (bs, out, h, w) 
                "stage4": torch.sigmoid(out_stage4),   # (bs, out, h, w) 
                "stage5": torch.sigmoid(out_stage5),   # (bs, out, h, w) 
                "stage6": torch.sigmoid(out_stage6),   # (bs, out, h, w) 
                "concat": torch.sigmoid(out_concat),   # (bs, out, h, w)
                }
            gt_seg: LongTensor of shape (bs, h, w)

        Outputs:
            total_loss
        """
        
        bce_out_dict = {}
        bce_tot = gt_seg.new_zeros(1)
        batch_size = gt_seg.size(0)
        for k, v in pred_out_dict.items():
            # bce
            with torch.no_grad():
                onehot_seg = F.one_hot(gt_seg.squeeze(1).long(), num_classes=self.num_class)  # (bs, h, w, num_class)
                onehot_seg = onehot_seg.permute(0, 3, 1, 2)  # (bs, num_class, h, w)
                onehot_seg.requires_grad = False

            bce_loss = self.bce(v, onehot_seg.float())  # (bs, num_class, h, w)
            if not self.do_bg:
                bce_loss = bce_loss[:, 1:, :, :].mean()  # mask值为0的区域不计入loss
            else:
                bce_loss = bce_loss.mean()  # mask值为0的区域不计入loss

            bce_out_dict[k] = bce_loss
            bce_tot += bce_loss

        bce_out_dict['total_loss'] = bce_tot * batch_size

        return {"total_loss": bce_tot[0]}
