import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from pathlib import Path
import cv2
from utils import *
from functools import partial
from torch.utils.data import Sampler
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler as TorchBatchSampler
import itertools
import numpy as np
from loguru import logger
import os
import random
import uuid
from functools import wraps
from torch.utils.data import Dataset as TorchDataset
import numbers


__all__ = ["fixed_imgsize_collector", "CitySpaceDataset", "CitySpaceBatchSampler", 
           "InfiniteSampler", "CitySpaceDataLoader", "DataPrefetcher", "build_dataloader", 
           "Transforms"]

    
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


class Transforms:

    def __init__(self, data_aug_hyp) -> None:
        self.hyp = data_aug_hyp

    def __call__(self, img, seg):
        img, seg = RandomShift(img, seg, self.hyp['data_aug_shift_p'], fill_value=self.hyp["data_aug_fill_value"])
        img = RandomBrightness(img, prob=self.hyp['data_aug_brightness_p'])
        img, seg = RandomFlipLR(img, seg, prob=self.hyp['data_aug_fliplr_p'])
        img, seg = RandomFlipUD(img, seg, prob=self.hyp['data_aug_flipud_p'])
        img = RandomBlur(img, prob=self.hyp['data_aug_blur_p'])
        img = RandomSaturation(img, prob=self.hyp['data_aug_saturation_p'])
        img = RandomHSV(img, 
                        prob=self.hyp['data_aug_hsv_p'], 
                        hgain=self.hyp['data_aug_hsv_hgain'], 
                        sgain=self.hyp['data_aug_hsv_sgain'], 
                        vgain=self.hyp['data_aug_hsv_vgain'])
        img, seg = RandomPerspective(img, seg, 
                                     prob=self.hyp['data_aug_perspective_p'],
                                     degree=self.hyp['data_aug_degree'], 
                                     translate=self.hyp['data_aug_translate'], 
                                     shear=self.hyp['data_aug_shear'], 
                                     perspective=self.hyp['data_aug_prespective'], 
                                     fill_value=self.hyp["data_aug_fill_value"], 
                                     dst_size=img.shape[:2])
        img, seg = RandomCrop(img, seg, self.hyp['data_aug_crop_p'])
        img, seg = RandomCutout(img, seg, prob=self.hyp['data_aug_cutout_p'])
        return img, seg


class Dataset(TorchDataset):

    def __init__(self, input_dimension, enable_data_aug=True) -> None:
        super(Dataset, self).__init__()
        self.enable_data_aug = enable_data_aug
        self.__input_dim = input_dimension


    @property
    def input_dim(self):
        """
        Dimension that can be used by transforms to set the correct image size, etc.
        This allows transforms to have a single source of truth
        for the input dimension of the network.

        Return:
            list: Tuple containing the current width,height
        """
        if hasattr(self, "_input_dim"):
            return self._input_dim

        if isinstance(self.__input_dim, numbers.Number):
            self.__input_dim = [self.__input_dim, self.__input_dim]
        return self.__input_dim
        

    @staticmethod
    def aug_getitem(getitem_func):
        @wraps(getitem_func)
        def wrapper(self, index):
            if not isinstance(index, int):
                self.enable_data_aug = index[0]
                index = index[1]
            
            ret = getitem_func(self, index)
            return ret
        return wrapper


class CitySpaceDataset(Dataset):

    def __init__(self, img_dir, seg_dir, img_size, enable_data_aug=True, transform=None, cache_num=0) -> None:
        super(CitySpaceDataset, self).__init__(enable_data_aug=enable_data_aug, input_dimension=img_size)
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.trans = transform
        self.db_img, self.db_seg = self.make_db()
        self.imgs = None
        if cache_num > 0:
            self.cache_num =  cache_num if cache_num <= len(self) else len(self) # len(self)
            self._cache_images()

    @property
    def num_class(self):
        return 20

    def make_db(self):
        assert Path(self.img_dir).exists(), f"directory: {self.img_dir} is not exists!"
        assert Path(self.seg_dir).exists(), f"directory: {self.seg_dir} is not exists!"

        img_filepathes = [p for p in Path(self.img_dir).iterdir() if p.suffix in ([".jpg", ".png", ".tiff"])]
        seg_filepathes = [p for p in Path(self.seg_dir).iterdir() if p.suffix in ([".jpg", ".png", ".tiff"])]
        assert len(img_filepathes) == len(seg_filepathes), f"len(img_filepathes): {len(img_filepathes)}, but len(seg_filenames): {len(seg_filepathes)}"
        #                                                     (aachen              , 000062              , 000019)
        img_filepathes = sorted(img_filepathes, key=lambda x: (x.stem.split("_")[0], x.stem.split("_")[1], x.stem.split("_")[2]))
        seg_filepathes = sorted(seg_filepathes, key=lambda x: (x.stem.split("_")[0], x.stem.split("_")[1], x.stem.split("_")[2]))

        seg_filenames = ['_'.join(p.stem.split("_")[:-2]) for p in seg_filepathes]
        for i, p in enumerate(img_filepathes):
            img_filename = '_'.join(p.stem.split("_")[:-1])
            assert img_filename in seg_filenames, f"image filename: {img_filepathes[i]}, can not found matched segmentation file."
        return img_filepathes, seg_filepathes

    def __len__(self):
        return len(self.db_img)

    def load_resized_data_pair(self, index):
        img_p = self.db_img[index]
        seg_p = self.db_seg[index]
        img_arr = cv2.imread(str(img_p))  # (h, w, 3)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        seg_arr = cv2.imread(str(seg_p), 0)[:, :, None]  # (h, w, 1)
        # cityspace数据集中的背景类mask值为255, 将背景类的mask修改为0
        bg_mask = seg_arr == 255
        seg_arr += 1
        seg_arr[bg_mask] = 0
        img_arr, seg_arr = letter_resize(img_arr, seg_arr, self.input_dim)
        # (h, w, 3) & (h, w, 1) -> (h, w, 4)
        return np.concatenate([img_arr, seg_arr], axis=-1)


    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.input_dim[0]
        max_w = self.input_dim[1]
        
        cache_file = os.path.join(str(Path(self.img_dir).parent), f"img_resized_cache.array")
        print(f"cache_file path: {cache_file}")
        if not os.path.exists(cache_file):
            logger.info("Caching images for the first time. This might take about 20 minutes for COCO")
            self.imgs = np.memmap(
                cache_file,
                shape=(self.cache_num, max_h, max_w, 4),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(lambda x: self.load_resized_data_pair(x), range(self.cache_num))
            pbar = tqdm(enumerate(loaded_images), total=self.cache_num)
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(self.cache_num, max_h, max_w, 4),
            dtype=np.uint8,
            mode="r+",
        )

        
    def pull_item(self, index):
        if self.imgs is not None and index < self.cache_num:
            data_pair = self.imgs[index]
            img_arr = data_pair[..., :3]
            seg_arr = data_pair[..., -1:]
        else:
            img_p = self.db_img[index]
            seg_p = self.db_seg[index]
            img_arr = cv2.imread(str(img_p))  # (h, w, 3)
            seg_arr = cv2.imread(str(seg_p), 0)[:, :, None]  # (h, w, 1)

        # cityspace数据集中的背景类mask值为255, 将背景类的mask修改为0
        if len(seg_arr[seg_arr == 255]) > 0:
            bg_mask = seg_arr == 255
            seg_arr += 1
            seg_arr[bg_mask] = 0

        return img_arr, seg_arr

    @Dataset.aug_getitem
    def __getitem__(self, index):
        img_arr, seg_arr = self.pull_item(index) 

        if self.enable_data_aug and self.trans is not None:
            img_arr, seg_arr = self.trans(img_arr, seg_arr)

        return img_arr, seg_arr, index


class CitySpaceBatchSampler(TorchBatchSampler):
    """
    This batch sampler will generate mini-batches of (mosaic, index) tuples from another sampler.
    It works just like the :class:`torch.utils.data.sampler.BatchSampler`,
    but it will turn on/off the mosaic aug.
    """

    def __init__(self, *args, enable_data_aug=True, **kwargs):
        super(CitySpaceBatchSampler, self).__init__(*args, **kwargs)
        self.enable_data_aug = enable_data_aug

    def __iter__(self):
        for batch in super().__iter__():
            yield [(self.enable_data_aug, idx) for idx in batch]


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


class CitySpaceDataLoader(TorchDataLoader):
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
            batch_sampler = CitySpaceBatchSampler(sampler, self.batch_size, self.drop_last)

        self.batch_sampler = batch_sampler
        
        super(CitySpaceDataLoader, self).__init__(*args, **kwargs)
        self.__initialized = True

    def close_data_aug(self):
        self.batch_sampler.enable_data_aug = False


def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)


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
            self.next_input, self.next_target, _ = out_dict["img"], out_dict["seg"], out_dict['id']
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return {'img':input, 'seg':target}

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


def build_dataloader(img_dir, seg_dir, data_aug_hyp,
                     batch_size=16, 
                     drop_last=False, 
                     dst_size=None, 
                     enable_data_aug=True, 
                     num_workers=0, 
                     pin_memory=True, 
                     seed=42, 
                     cache_num=0):
    if dist.is_available() and dist.is_initialized():
        batch_size = batch_size // dist.get_world_size()
    if enable_data_aug:
        transform = Transforms(data_aug_hyp)
    else:
        transform = None
    dataset = CitySpaceDataset(img_dir, seg_dir, dst_size, enable_data_aug, transform=transform, cache_num=cache_num)
    sampler = InfiniteSampler(len(dataset), seed=seed if seed else 0)
    batch_sampler = CitySpaceBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=drop_last, enable_data_aug=enable_data_aug)
    dataloader_kwargs = {"num_workers": num_workers, 
                         "pin_memory": pin_memory, 
                         "batch_sampler": batch_sampler, 
                         "worker_init_fn": worker_init_reset_seed,
                         "collate_fn": partial(fixed_imgsize_collector, dst_size=dataset.input_dim)
                        }
    dataloader = CitySpaceDataLoader(dataset, **dataloader_kwargs)
    if torch.cuda.is_available():
        prefetcher = DataPrefetcher(dataloader)
    else:
        prefetcher = None

    return dataset, iter(dataloader), prefetcher