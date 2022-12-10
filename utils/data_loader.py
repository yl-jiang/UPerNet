import torch
import numpy as np
from .image_transform import letter_resize
from torch.utils.data import Sampler
from torch.utils.data.sampler import BatchSampler as TorchBatchSampler
import itertools
import torch.distributed as dist
import uuid
import random

__all__ = ["fixed_imgsize_collector", "InfiniteSampler", "worker_init_reset_seed"]


def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)

def normal_normalization(img):
    norm_img = torch.from_numpy(img / (np.max(img) + 1e-8)).permute(2, 0, 1).contiguous()
    return norm_img


def fixed_imgsize_collector(data_in, dst_size):
    """
    将Dataset中__getitem__方法返回的每个值进行进一步组装。

    :param data_in: tuple, data[0] is image, data[1] is seg, data[2] is image's id
    :param dst_size: [h, w]
    :return: dictionary
    """
    # 输入图像的格式为(h,w,3)
    assert data_in[0][0].ndim == 3 and data_in[0][0].shape[-1] == 3, f"data's formate should be (h, w, 3), but got {data_in[0].shape}"
    
    batch_size = len(data_in)
    imgs = [d[0] for d in data_in]
    segs = [d[1] for d in data_in]
    img_ids = [d[2] for d in data_in]

    # batch内image的图像拥有相同的shape, batch之间image的shape不一样
    # dst_size = padding(dst_size, 32)
    imgs_out = torch.zeros(batch_size, 3, dst_size[0], dst_size[1])
    segs_out = torch.zeros(batch_size, 1, dst_size[0], dst_size[1])

    for b in range(batch_size):
        img = imgs[b]  # ndarray
        seg = segs[b] 
        if img.shape[0] != dst_size[0] or img.shape[1] != dst_size[1]:
            img, seg = letter_resize(img, seg, dst_size)
        imgs_out[b] = normal_normalization(img)
        segs_out[b] = torch.from_numpy(seg.copy()).permute(2, 0, 1).contiguous()

    return {'img': imgs_out, 'seg': segs_out, 'id': np.asarray(img_ids)}



class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)

        if dist.is_available() and dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = rank
            self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)

    def __len__(self):
        return self._size // self._world_size