import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiBCELoss"]


class MultiBCELoss:

    def __init__(self, num_class, ignore_index=0) -> None:
        super(MultiBCELoss, self).__init__()
        self.num_class = num_class
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
       
    def __call__(self, pred_out_dict, gt_seg):
        """
        Inputs:
            pred_out_dict: {"fuse": pred}  # (bs, num_class, h, w)
            gt_seg: dtype of LongTensor of shape (bs, h, w)
        Outputs:

        """
        
        out_dict = {}
        pred_for_ce = F.softmax(pred_out_dict['fuse'], dim=1)
        out_dict['total'] = self.ce(pred_for_ce, gt_seg.squeeze(1).long())

        # onehot_seg = F.one_hot(gt_seg.squeeze(1).long(), num_classes=self.num_class)  # (bs, h, w, num_class)
        # onehot_seg = onehot_seg.permute(0, 3, 1, 2)  # (bs, num_class, h, w)
        # onehot_seg.requires_grad = False
        # pred_for_bce = F.sigmoid(pred_out_dict['fuse'])
        # bce_loss = self.bce(pred_for_bce, onehot_seg.float())  # (bs, num_class, h, w)
        # out_dict['bce'] = bce_loss[:, 1:, :, :].mean()  # mask值为0的区域不计入loss
        return out_dict
