import time
import shutil
import warnings
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import numba
import numpy as np
import logging
from tabulate import tabulate
import pprint

__all__ = ["catch_warnnings", "maybe_mkdir", "time_synchronize", "is_exists", "clear_dir", "summary_model", "print_config", "ExponentialMovingAverageModel", "is_parallel"]


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)

def catch_warnnings(fn):
    def wrapper(instance):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(instance)
    return wrapper


def maybe_mkdir(dirname):
    if isinstance(dirname, str):
        dirname = Path(dirname)
    if not dirname.exists():
        dirname.mkdir(parents=True)


def time_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def is_exists(path_str):
    return Path(path_str).exists()


def clear_dir(dirname):
    if isinstance(dirname, str):
        dirname = Path(dirname)
    if dirname.exists():
        shutil.rmtree(str(dirname), ignore_errors=True)  # shutil.rmtree会将传入的文件夹整个删除
    if not dirname.exists():
        dirname.mkdir(parents=True)


def summary_model(model, input_img_size=[640, 640], verbose=False, prefix=""):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        number_params = sum(x.numel() for x in model.parameters())
        number_gradients = sum(x.numel() for x in model.parameters() if x.requires_grad)
        number_layers = len(list(model.modules()))
        try:
            from thop import profile
            dummy_img = torch.rand(1, 3, input_img_size[0], input_img_size[1], device=next(model.parameters()).device)
            flops, params = profile(deepcopy(model), inputs=(dummy_img, ), verbose=verbose)
            flops /= 1e9 * 2
        except (ImportError, Exception) as err:
            print(f"error occur in summary_model: {err}")
            flops = ""
        
        if verbose:
            msg = f"Model Summary: {prefix} {number_layers} layers; {number_params} parameters; {number_gradients} gradients; {flops} GFLOPs"
            print(msg)
        return {'number_params': number_params, "number_gradients": number_gradients, "flops": flops, "number_layers": number_layers}


logger = logging.getLogger(__name__)
def print_config(args):
    table_header = ["keys", "values"]
    if isinstance(args, dict):
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in args.items()
            if not k.startswith("_")
        ]
    else:
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(args).items()
            if not k.startswith("_")
        ]
    return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")    


class ExponentialMovingAverageModel:
    """
    从始至终维持一个model，并不断更新该model的参数，但该mdoel仅仅是为了inference。
    随着训练的进行，越靠后面的模型参数对ema模型的影响越大。
    """
    def __init__(self, model, decay_ratio=0.9999, update_num=0):
        self.ema = deepcopy(model).eval()
        self.update_num = update_num
        self.get_decay_weight = lambda x: decay_ratio * (1 - np.exp(-x / 2000))
        for parm in self.ema.parameters():
            parm.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.update_num += 1
            decay_weight = self.get_decay_weight(self.update_num)
            cur_state_dict = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= decay_weight
                    v += (1 - decay_weight) * cur_state_dict[k].detach()



