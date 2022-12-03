import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiBCELoss"]


# class MultiBCELoss:

#     def __init__(self, num_class, ignore_index=0) -> None:
#         super(MultiBCELoss, self).__init__()
#         self.num_class = num_class
#         self.bce = nn.BCEWithLogitsLoss(reduction='none')
#         self.ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index)
       
#     def __call__(self, pred_out_dict, gt_seg):
#         """
#         Inputs:
#             pred_out_dict: {"fuse": pred}  # (bs, num_class, h, w)
#             gt_seg: dtype of LongTensor of shape (bs, h, w)
#         Outputs:

#         """
        
#         out_dict = {}
#         pred_for_ce = F.softmax(pred_out_dict['fuse'], dim=1)
#         out_dict['total'] = self.ce(pred_for_ce, gt_seg.squeeze(1).long())

#         # onehot_seg = F.one_hot(gt_seg.squeeze(1).long(), num_classes=self.num_class)  # (bs, h, w, num_class)
#         # onehot_seg = onehot_seg.permute(0, 3, 1, 2)  # (bs, num_class, h, w)
#         # onehot_seg.requires_grad = False
#         # pred_for_bce = F.sigmoid(pred_out_dict['fuse'])
#         # bce_loss = self.bce(pred_for_bce, onehot_seg.float())  # (bs, num_class, h, w)
#         # out_dict['bce'] = bce_loss[:, 1:, :, :].mean()  # mask值为0的区域不计入loss
#         return out_dict



class MultiBCELoss:

    def __init__(self, num_class, ignore_index=0) -> None:
        super(MultiBCELoss, self).__init__()
        self.num_class = num_class
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
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
            gt_seg: FloatTensor of shape (bs, 1, h, w)
        Outputs:

        """
        
        assert isinstance(pred_out_dict, dict), f"argument of pred_out_dict must be a dictionary, but got: {type(pred_out_dict)}"
        out_dict = {}
        tot = 0.0
        for k, v in pred_out_dict.items():
            pred_for_ce = F.softmax(v, dim=1)
            loss = self.ce(pred_for_ce, gt_seg.squeeze(1).long())
            tot += loss
            out_dict[k] = loss
        
        out_dict["total"] = tot
        return {'total': tot}