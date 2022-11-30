import torch
import torch.nn.functional as F
import numpy as np

__all__ = ["Metric"]


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



class Metric:

    def __init__(self, thresh=0.5, ignore_index=-1):
        self.ignore_index = ignore_index
        self.__pixel_acc = []
        self.__thresh = thresh

    @property
    def threshold(self):
        return self.__thresh

    @threshold.setter
    def threshold(self, thresh):
        if 0. <= thresh <= 1.0:
            self.__thresh = thresh
        else:
            raise ValueError(f"threshold should be in range [0, 1]")        

    def update(self, pred_seg, gt_seg):
        self.__pixel_acc.append(pixel_acc(pred_seg, gt_seg, self.ignore_index))

    @property
    def pixel_acc(self):
        if len(self.__pixel_acc) > 0:
            return np.mean(self.__pixel_acc)
        else:
            return 0


    def clear(self):
        self.__pixel_acc = []
        

    def __str__(self):
        return f"pixel_acc = {np.mean(self.__pixel_acc)}"



if __name__ == "__main__":
    torch.manual_seed(1234)
    a = torch.rand(1, 1, 448, 448)
    b = torch.rand(1, 1, 448, 448)
    metric = Metric()
    mae = metric.update(a, b)
    print(str(metric))
