from pathlib import Path

import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data import Sampler
from torch.utils.data.sampler import BatchSampler as TorchBatchSampler
import itertools

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
            self.next_input, self.next_info = out_dict["img"], out_dict["resize_info"]
        except StopIteration:
            self.next_input = None
            self.next_info = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        info = self.next_info
        if input is not None:
            self.record_stream(input)
        self.preload()
        return {'img':input, 'resize_info':info}

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
        img_normed = self.normalization(img_resized)
        return img_normed, letter_info

def collector(data_in):
    batch_size = len(data_in)
    imgs = [d[0] for d in data_in]
    infoes = [d[1] for d in data_in]
    h, w = imgs[0].shape[1:]
    img_out = torch.ones(batch_size, 3, h, w)
    resize_infoes_out = []
    for i in range(batch_size):
        img_out[i] = imgs[i]
        resize_infoes_out.append(infoes[i])
    return {'img': img_out, 'resize_info': resize_infoes_out}


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




def build_testdataloader(datadir, img_size=640, batch_size=1):
    # 因为在inference模式下使用letter_resize_img函数对输入图片进行resize，不会将所有输入的图像都resize到相同的尺寸，而是只要符合输入网络的要求即可
    # assert batch_size == 1, f"use inference mode, so please set batch size to 1"
    dataset = TestDataset(datadir, img_size)
    batch_sampler = TorchBatchSampler(sampler=InfiniteSampler(len(dataset), shuffle=False), batch_size=batch_size, drop_last=False)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=True, num_workers=4, collate_fn=collector)
    if torch.cuda.is_available():
        prefetcher = DataPrefetcher(dataloader)
    else:
        prefetcher = None
    return dataset, iter(dataloader), prefetcher


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


if __name__ == '__main__':
    test('/Users/ylj/Personal/Programs/Dataset/SOD/DUTS-TE-Image', 640, 1)

