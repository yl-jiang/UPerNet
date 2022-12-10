import os
import time
import torch.backends.cudnn as cudnn
import numpy as np
import random
import torch

__all__ = [
    "get_total_and_free_memory_in_Mb",
    "occupy_mem",
    "gpu_mem_usage",
    "init_seed", 
]

def init_seed(seed, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def get_total_and_free_memory_in_Mb(cuda_device):
    devices_info_str = os.popen( "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader")
    devices_info = devices_info_str.read().strip().split("\n")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(',')
        cuda_device = int(visible_devices[cuda_device])
    total, used = devices_info[int(cuda_device)].split(",")
    return int(total), int(used)


def occupy_mem(cuda_device, mem_ratio=0.9):
    """
    pre-allocate gpu memory for training to avoid memory Fragmentation.
    """
    total, used = get_total_and_free_memory_in_Mb(cuda_device)
    max_mem = int(total * mem_ratio)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x
    time.sleep(5)


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)