import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from collections import defaultdict

__all__ = ["SegMetirc2D"]


def pixel_acc(argmax_pred_seg, gt_seg, ignore_index=-1):
    """
    Inputs:
        argmax_pred_seg: (bs, h, w) / ndarray
        gt_seg: (bs, h, w) / ndarray
    """
    tpe = argmax_pred_seg.dtype
    valid = (gt_seg != ignore_index).astype(tpe)
    acc_sum = np.sum(valid * (argmax_pred_seg == gt_seg)).astype(tpe)
    pixel_sum = np.sum(valid)
    acc = acc_sum.astype(np.float32) / (pixel_sum.astype(np.float32) + 1e-10)
    return acc

class SegMetirc2D():
    """

    计算基于重叠度和距离等九种分割常见评价指标
    """

    def update(self, real_mask, pred_mask):
        self.real_mask = real_mask
        self.pred_mask = pred_mask

    # 下面的六个指标是基于重叠度的
    def get_dice_coefficient(self):
        """
        dice系数 dice系数的分子
        :return: dice系数 dice系数的分子 dice系数的分母(后两者用于计算dice_global)
        """
        dice_metric = {}
        unique_label = np.unique(self.real_mask)
        for lab in unique_label:
            tmp_real, tmp_pred = self.get_same_lab_real_and_pred_mask(lab)
            intersection = (tmp_pred * tmp_real).sum()
            union = tmp_pred.sum() + tmp_real.sum()
            dice = 2 * intersection / union
            dice_metric[lab] = dice
        return dice_metric

    def get_iou(self):
        iou_metric = {}
        unique_label = np.unique(self.real_mask)
        for lab in unique_label:
            tmp_real, tmp_pred = self.get_same_lab_real_and_pred_mask(lab)
            intersection = (tmp_pred * tmp_real).sum()
            union = (tmp_real.astype(np.int32) | tmp_pred.astype(np.int32)).sum()
            iou_metric[lab] = (intersection + 1e-8) / (union + 1e-8)
        return iou_metric

    def get_VOE(self):
        """
        体素重叠误差 Volumetric Overlap Error
        :return: 体素重叠误差 Volumetric Overlap Error
        """
        voe_metric = {}
        iou = self.get_iou()
        for k, v in iou.items():
            voe_metric[k] = 1 - v
        return voe_metric

    def get_RVD(self):
        """
        体素相对误差 Relative Volume Difference
        :return: 体素相对误差 Relative Volume Difference
        """
        rvd_metric = {}
        unique_label = np.unique(self.real_mask)
        for lab in unique_label:
            tmp_real, tmp_pred = self.get_same_lab_real_and_pred_mask(lab) 
            rvd_metric[lab] = float(tmp_pred.sum() - tmp_real.sum()) / float(tmp_real.sum())
        return rvd_metric

    def get_FNR(self):
        """
        欠分割率 False negative rate
        :return: 欠分割率 False negative rate
        """
        fnr_metric = {}
        unique_label = np.unique(self.real_mask)
        for lab in unique_label:
            tmp_real, tmp_pred = self.get_same_lab_real_and_pred_mask(lab)
            fn = tmp_real.sum() - (tmp_real * tmp_pred).sum()
            union = (tmp_real.astype(np.int32) | tmp_pred.astype(np.int32)).sum()
            fnr_metric[lab] = fn / union
        return fnr_metric

    def get_same_lab_real_and_pred_mask(self, lab):
        tmp_real = np.zeros_like(self.real_mask)
        tmp_real[self.real_mask == lab] = 1
        tmp_pred = np.zeros_like(self.pred_mask)
        tmp_pred[self.pred_mask == lab] = 1
        return tmp_real, tmp_pred

    def get_FPR(self):
        """
        过分割率 False positive rate
        :return: 过分割率 False positive rate
        """
        fdp_metric = {}
        unique_label = np.unique(self.real_mask)
        for lab in unique_label:
            tmp_real, tmp_pred = self.get_same_lab_real_and_pred_mask(lab)
            fn = tmp_pred.sum() - (tmp_real * tmp_pred).sum()
            union = (tmp_real.astype(np.int32) | tmp_pred.astype(np.int32)).sum()
            fdp_metric[lab] = fn / union
        return fdp_metric
        
    def get_pixel_acc(self, ignore_index=-1):
        tpe = self.pred_mask.dtype
        valid = (self.real_mask != ignore_index).astype(tpe)
        acc_sum = np.sum(valid * (self.pred_mask == self.real_mask)).astype(tpe)
        pixel_sum = np.sum(valid)
        acc = acc_sum.astype(np.float32) / (pixel_sum.astype(np.float32) + 1e-10)
        return acc

    @property
    def dice(self):
        dice_dict = self.get_dice_coefficient()
        return np.mean(list(dice_dict.values()))

    @property
    def iou(self):
        iou_dict = self.get_iou()
        return np.mean(list(iou_dict.values()))

    @property
    def fnr(self):
        fnr_dict = self.get_FNR()
        return np.mean(list(fnr_dict.values()))

    @property
    def fpr(self):
        fpr_dict = self.get_FPR()
        return np.mean(list(fpr_dict.values()))

    @property
    def voe(self):
        voe_dict = self.get_VOE()
        return np.mean(list(voe_dict.values()))

    @property
    def rvd(self):
        rvd_dict = self.get_RVD()
        return np.mean(list(rvd_dict.values()))

    @property
    def acc(self):
        acc = self.get_pixel_acc()
        return acc


# segmeantaion metric
def dice_coeff(input: Tensor, target: Tensor):
    input = (input > 0.5).float()
    smooth = 1e-5
    num = target.size(0)
    input = input.view(num, -1).float()
    target = target.view(num, -1).float()
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = dice.sum() / num
    return dice


def iou_coeff(input: Tensor, target: Tensor):
    input = (input > 0.5).float()
    smooth = 1e-5
    num = target.size(0)
    input = input.view(num, -1).float()
    target = target.view(num, -1).float()
    intersection = (input * target)
    union = (intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) - intersection.sum(1) + smooth)
    union = union.sum() / num
    return union


def multiclass_dice_coeff(input: Tensor, target: Tensor):
    Batchsize, Channel = input.shape[0], input.shape[1]
    y_pred = input.float().contiguous().view(Batchsize, Channel, -1)
    y_true = target.long().contiguous().view(Batchsize, -1)
    y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
    y_true = y_true.permute(0, 2, 1)  # H, C, H*W
    assert y_pred.size() == y_true.size()
    dice = 0
    # remove backgroud region
    for channel in range(1, y_true.shape[1]):
        dice += dice_coeff(y_pred[:, channel, ...], y_true[:, channel, ...])
    return dice / (input.shape[1] - 1)


def multiclass_dice_coeffv2(input: Tensor, target: Tensor):
    Batchsize, Channel = input.shape[0], input.shape[1]
    y_pred = input.float().contiguous().view(Batchsize, Channel, -1)
    y_true = target.long().contiguous().view(Batchsize, -1)
    y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
    y_true = y_true.permute(0, 2, 1)  # H, C, H*W
    assert y_pred.size() == y_true.size()
    smooth = 1e-5
    eps = 1e-7
    # remove backgroud region
    y_pred_nobk = y_pred[:, 1:Channel, ...]
    y_true_nobk = y_true[:, 1:Channel, ...]
    intersection = torch.sum(y_true_nobk * y_pred_nobk, dim=(0, 2))
    denominator = torch.sum(y_true_nobk + y_pred_nobk, dim=(0, 2))
    gen_dice_coef = ((2. * intersection + smooth) / (denominator + smooth)).clamp_min(eps)
    return gen_dice_coef.mean()


def multiclass_iou_coeff(input: Tensor, target: Tensor):
    assert input.size() == target.size()
    union = 0
    # remove backgroud region
    for channel in range(1, input.shape[1]):
        union += iou_coeff(input[:, channel, ...], target[:, channel, ...])
    return union / (input.shape[1] - 1)


# classification metric

def calc_accuracy(input: Tensor, target: Tensor):
    n = input.size(0)
    acc = torch.sum(input == target).sum() / n
    return acc




if __name__ == "__main__":
    torch.manual_seed(1234)
    a = torch.rand(1, 1, 448, 448)
    b = torch.rand(1, 1, 448, 448)
    metric = Metric()
    mae = metric.update(a, b)
    print(str(metric))
