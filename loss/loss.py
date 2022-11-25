import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiBCELoss"]


class MultiBCELoss:

    def __init__(self, num_class) -> None:
        super(MultiBCELoss, self).__init__()
        self.num_class = num_class
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')
       
    def __call__(self, pred_out_dict, gt_seg):
        """
        Inputs:
            pred_out_dict: {"fuse": pred,   # (bs, num_class, h, w)}
            gt_seg: dtype of LongTensor of shape (bs, h, w)
        Outputs:

        """
        
        out_dict = {}
        gt_seg.requires_grad = False
        out_dict['ce'] = self.ce(pred_out_dict['fuse'], gt_seg.long())
        onehot_seg = F.one_hot(gt_seg.long(), num_classes=self.num_class)  # (bs, h, w, num_class)
        onehot_seg = onehot_seg.permute((0, 3, 1, 2))  # (bs, num_class, h, w)
        onehot_seg.requires_grad = False
        out_dict['bce'] = self.bce(torch.sigmoid(pred_out_dict['fuse']), onehot_seg)
        return out_dict
