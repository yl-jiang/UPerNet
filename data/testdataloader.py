from pathlib import Path
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from utils import get_world_size, worker_init_reset_seed, letter_resize, wait_for_the_master
from torch.utils.data import Sampler
from torch.utils.data.sampler import BatchSampler as TorchBatchSampler
import itertools
from torch.utils.data import DataLoader as TorchDataLoader
import numbers
from functools import partial


__all__ = ["build_testdataloader"]

def letter_resize_img(img, dst_size, stride=64, fill_value=114, only_ds=False):
    """
    only scale down
    :param only_ds: only downsample
    :param fill_value:
    :param img:
    :param dst_size: int or [h, w]
    :param stride:
    :return:
    """
    if isinstance(dst_size, int):
        dst_size = [dst_size, dst_size]

    # 将dst_size调整到是stride的整数倍
    dst_del_h, dst_del_w = np.remainder(dst_size[0], stride), np.remainder(dst_size[1], stride)
    dst_pad_h = stride - dst_del_h if dst_del_h > 0 else 0
    dst_pad_w = stride - dst_del_w if dst_del_w > 0 else 0
    dst_size = [dst_size[0] + dst_pad_h, dst_size[1] + dst_pad_w]

    org_h, org_w = img.shape[:2]  # [height, width]
    scale = float(np.min([dst_size[0] / org_h, dst_size[1] / org_w]))
    if only_ds:
        scale = min(scale, 1.0)  # only scale down for good test performance
    if scale != 1.:
        resize_h, resize_w = int(org_h * scale), int(org_w * scale)
        img_resize = cv2.resize(img.copy(), (resize_w, resize_h), interpolation=0)
    else:
        resize_h, resize_w = img.shape[:2]
        img_resize = img.copy()

    # training时需要一个batch保持固定的尺寸，testing时尽可能少的填充像素以加速inference
    # 例如: 输入的图片大小为(304, 400), dst_size设置为640，这种模式下的letter_resize_img会将图片resize为(512, 640)
    pad_h, pad_w = dst_size[0] - resize_h, dst_size[1] - resize_w
    pad_h, pad_w = np.remainder(pad_h, stride), np.remainder(pad_w, stride)
    top = int(round(pad_h / 2))
    left = int(round(pad_w / 2))
    bottom = pad_h - top
    right = pad_w - left
    if isinstance(fill_value, int):
        fill_value = (fill_value, fill_value, fill_value)

    img_out = cv2.copyMakeBorder(img_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_value)
    letter_info = {'scale': scale, 'pad_top': top, 'pad_left': left, "pad_bottom": bottom, "pad_right": right, "org_shape": (org_h, org_w)}
    return img_out.astype(np.uint8), letter_info


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

class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            out_dict = next(self.loader)
            self.next_input, self.next_info, self.next_img_id = out_dict["img"], out_dict["info"], out_dict['id']
        except StopIteration:
            self.next_input  = None
            self.next_info   = None
            self.next_img_id = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        info = self.next_info
        img_id = self.next_img_id
        if input is not None:
            self.record_stream(input)
        self.preload()
        return {'img':input, 'info':info, 'id': img_id}

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())

class TestDataset(Dataset):

    def __init__(self, datadir, img_size):
        self.img_pathes = []
        for p in Path(datadir).iterdir():
            if p.is_file() and p.suffix in [".png", '.jpg']:
                self.img_pathes.append(str(p))
        self.img_size = img_size
        self.num_class = 0
        self.class2label = ['lab' for _ in range(self.num_class)]
        self.__input_dim = img_size

    @property
    def input_dim(self):
        if hasattr(self, "_input_dim"):
            return self._input_dim

        if isinstance(self.__input_dim, numbers.Number):
            self.__input_dim = [self.__input_dim, self.__input_dim]
        return self.__input_dim


    def __len__(self):
        return len(self.img_pathes)

    def __iter__(self):
        self.count = 0
        return self

    @staticmethod
    def normalization(img):
        # 输入图像的格式为(h,w,3)
        assert len(img.shape) == 3 and img.shape[-1] == 3
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        return transforms(img)

    def __getitem__(self, item):
        img_bgr = cv2.imread(self.img_pathes[item])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized, letter_info = letter_resize_img(img_rgb, self.img_size)
        return img_resized, letter_info, item


def normal_normalization(img):
    norm_img = torch.from_numpy(img / (np.max(img) + 1e-8)).permute(2, 0, 1).contiguous()
    return norm_img


def padding(hw, factor=32):
        if isinstance(hw, numbers.Real):
            hw = [hw, hw]
        else:
            assert len(hw) == 2, f"input image size's format should like (h, w)"
        h, w = hw
        h_mod = h % factor
        w_mod = w % factor
        if h_mod > 0:
            h = (h // factor + 1) * factor
        if w_mod > 0:
            w = (w // factor + 1) * factor
        return h, w

def collector(data_in, dst_size):
    """
    将Dataset中__getitem__方法返回的每个值进行进一步组装。

    :param data_in: tuple, data[0] is image, data[1] is resize_info, data[2] is image's id
    :param dst_size: [h, w]
    :return: dictionary
    """
    # 输入图像的格式为(h,w,3)
    assert data_in[0][0].ndim == 3
    
    batch_size = len(data_in)
    imgs    = [d[0] for d in data_in]
    infos   = [d[1] for d in data_in]
    img_ids = [d[2] for d in data_in]

    # batch内image的图像拥有相同的shape, batch之间image的shape不一样
    dst_size  = padding(dst_size, 32)
    imgs_out  = torch.zeros(batch_size, 3, dst_size[0], dst_size[1])
    infos_out = []

    for b in range(batch_size):
        img = imgs[b]  # ndarray
        if img.shape[0] != dst_size[0] or img.shape[1] != dst_size[1]:
            img = np.ascontiguousarray(cv2.resize(img, dst_size, interpolation=4))
        imgs_out[b] = normal_normalization(img)
        infos_out.append(infos[b])

    return {'img': imgs_out, 'info': infos_out, 'id': np.asarray(img_ids)}


class TestBatchSampler(TorchBatchSampler):
    """
    This batch sampler will generate mini-batches of (mosaic, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will turn on/off the mosaic aug.
    """

    def __init__(self, *args, **kwargs):
        super(TestBatchSampler, self).__init__(*args, **kwargs)

    def __iter__(self):
        for batch in super().__iter__():
            yield [idx for idx in batch]

class TestDataLoader(TorchDataLoader):
    def __init__(self, *args, **kwargs):
        self.__initialized = False
        shuffle = False
        batch_sampler = None
        if "shuffle" in kwargs:
            shuffle = kwargs["shuffle"]
        if "sampler" in kwargs:
            sampler = kwargs["sampler"]
        if "batch_sampler" in kwargs:
            batch_sampler = kwargs["batch_sampler"]

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = torch.utils.data.sampler.RandomSampler(self.dataset)
                else:
                    sampler = torch.utils.data.sampler.SequentialSampler(self.dataset)
            batch_sampler = TestBatchSampler(sampler, self.batch_size, self.drop_last)

        self.batch_sampler = batch_sampler
        
        super(TestDataLoader, self).__init__(*args, **kwargs)
        self.__initialized = True



def build_testdataloader(datadir, img_size=640, batch_size=1, num_workers=6):
    # 因为在inference模式下使用letter_resize_img函数对输入图片进行resize，不会将所有输入的图像都resize到相同的尺寸，而是只要符合输入网络的要求即可
    # assert batch_size == 1, f"use inference mode, so please set batch size to 1"
    with wait_for_the_master():
        dataset = TestDataset(datadir, img_size)

    sampler = InfiniteSampler(len(dataset), shuffle=False)

    # if dist.is_available() and dist.is_initialized():
    #     batch_size = batch_size * 2 // dist.get_world_size()
    batch_sampler = TestBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)

    dataloader_kwargs = {"num_workers": num_workers, 
                         "pin_memory": True, 
                         "batch_sampler": batch_sampler, 
                         "worker_init_fn": worker_init_reset_seed,
                         "collate_fn": partial(collector, dst_size=dataset.input_dim)
                        }
    dataloader = TestDataLoader(dataset, **dataloader_kwargs)
    if torch.cuda.is_available():
        prefetcher = DataPrefetcher(dataloader)
    else:
        prefetcher = None
    return dataset, iter(dataloader), prefetcher


if __name__ == '__main__':
    def test(datadir, img_size, batch_size):
        dataloader, dataset = build_testdataloader(datadir, img_size, batch_size)
        for x in dataloader:
            for i in range(batch_size):
                img = x['img'][i]
                info = x['resize_info'][i]
                img = img.permute(1, 2, 0)
                img *= 255.
                img_mdy = img.numpy().astype('uint8')
                h, w, c = img_mdy.shape
                fig, axes = plt.subplots(1, 2, figsize=[16, 16])
                axes[0].imshow(img_mdy)
                axes[0].set_title(f'{img_mdy.shape[:2]}')
                pad_t, pad_b, pad_l, pad_r = info['pad_top'], info['pad_bottom'], info['pad_left'], info['pad_right']
                img_org = img_mdy[pad_t:h-pad_b, pad_l:w-pad_r, :]
                # cv2.resize(img_arr, (dst_w, dst_h))
                img_org = cv2.resize(img_org, tuple(info['org_shape'][::-1]), interpolation=0)
                axes[1].imshow(img_org)
                axes[1].set_title(f"{img_org.shape[:2]}")
                plt.show()
                plt.close('all')
                plt.clf()

    test('/Users/ylj/Personal/Programs/Dataset/SOD/DUTS-TE-Image', 640, 1)

