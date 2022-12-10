import sys
import numbers
import argparse
from pathlib import Path
from datetime import datetime

current_work_directionary = Path('__file__').parent.absolute()
sys.path.insert(0, str(current_work_directionary))

import cv2
import torch
import numpy as np
from tqdm import trange
import torch.nn.functional as F

from config import Config
from utils import maybe_mkdir
from data import build_testdataloader
from utils import save_seg
from nets import USquareNet
import gc


class Testing:

    def __init__(self, hyp):
        # parameters
        self.hyp = hyp
        self.select_device()
        self.use_cuda = self.hyp['device'] == 'cuda'

        # current work path
        if self.hyp['current_work_dir'] is None:
            self.cwd = Path('./').absolute()
            self.hyp['current_work_dir'] = str(self.cwd)
        else:
            self.cwd = Path(self.hyp['current_work_dir'])

        # 确保输入图片的shape必须能够被32整除(因为网络会对输入的image进行32倍的下采样),如果不满足条件则对设置的输入shape进行调整
        self.hyp['input_img_size'] = self.padding(self.hyp['input_img_size'], 32)
        # dataset
        self.testdataset, self.testdataloader, self.testprefetcher = build_testdataloader(self.hyp['test_img_dir'], self.hyp['input_img_size'])
        # model
        self.hyp['num_class'] = 1
        self.model = USquareNet(in_channel=3, out_channel=self.hyp['num_class']).to(hyp['device'])
        # load pretrained model
        self.load_model('cpu')
        self.data_type = torch.float32
        if self.use_cuda and self.hyp['half']:  # cpu不支持fp16
            self.model = self.model.half()
            self.data_type = torch.float16

    @staticmethod
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

    def step(self):
        for j in trange(len(self.testdataloader)):
            if self.use_cuda:
                y = self.testprefetcher.next()
            else:
                y = next(self.testdataloader)
            img  = y['img'].to(self.data_type)  # (bs, 3, h, w)
            info = y['resize_info']
            pred_seg = self.tta(img)  # (bs, 1, h, w)
            img_numpy, pred_seg_numpy = self.postprocess(img.float().cpu().numpy(), pred_seg.float().cpu().numpy(), info)
            for k in range(len(img)):
                save_path = str(self.cwd / 'result' / 'predictions' / f"{j + k} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png")
                maybe_mkdir(Path(save_path).parent)
                save_seg(img_numpy[k], pred_seg_numpy[k], save_path)
            del y, img, info, pred_seg, img_numpy, pred_seg_numpy
        gc.collect()
   
    def postprocess(self, inp, pred_segs, info):
        """

        :param inp: normalization image (bs, 3, h, w)
        :param outputs: (bs, 1, h, w)
        :param info:
        :return:
        """
        assert isinstance(inp, np.ndarray)
        assert isinstance(inp, np.ndarray)
        processed_segs = []
        processed_imgs = []
        batch_num = inp.shape[0]

        for i in range(batch_num):
            pad_top, pad_left = info[i]['pad_top'], info[i]['pad_left']
            pad_bot, pad_right = info[i]['pad_bottom'], info[i]['pad_right']
            pred = pred_segs[i]
            org_h, org_w = info[i]['org_shape']
            cur_h, cur_w = inp[i].shape[1], inp[i].shape[2]

            img = np.transpose(inp[i], axes=[1, 2, 0])  # (h, w, 3)
            img *= 255.0
            img = np.clip(img, 0, 255.0)
            img = img.astype(np.uint8)
            img = img[pad_top:(cur_h - pad_bot), pad_left:(cur_w - pad_right), :]
            img = cv2.resize(img, (org_w, org_h), interpolation=0)
            processed_imgs.append(img)
            
            pred = np.transpose(pred_segs[i], axes=[1, 2, 0])  # (h, w, 1)
            pred = np.squeeze(pred)  # (h, w)
            pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
            pred = np.clip(pred * 255, 0, 255).astype(np.uint8)
            pred = pred[pad_top:(cur_h - pad_bot), pad_left:(cur_w - pad_right)]
            pred = cv2.resize(pred, (org_w, org_h), interpolation=0)
            processed_segs.append(pred)

        return processed_imgs, processed_segs

    def select_device(self):
        if self.hyp['device'].lower() != 'cpu':
            if torch.cuda.is_available():
                self.hyp['device'] = 'cuda'
                # region (GPU Tags)
                # 获取当前使用的GPU的属性并打印出来
                gpu_num = torch.cuda.device_count()
                cur_gpu_id = torch.cuda.current_device()
                cur_gpu_name = torch.cuda.get_device_name()
                cur_gpu_properties = torch.cuda.get_device_properties(cur_gpu_id)
                gpu_total_memory = cur_gpu_properties.total_memory
                gpu_major = cur_gpu_properties.major
                gpu_minor = cur_gpu_properties.minor
                gpu_multi_processor_count = cur_gpu_properties.multi_processor_count
                # endregion
                msg = f"Use Nvidia GPU {cur_gpu_name}, find {gpu_num} GPU devices, current device id: {cur_gpu_id}, "
                msg += f"total memory={gpu_total_memory/(2**20):.1f}MB, major={gpu_major}, minor={gpu_minor}, multi_processor_count={gpu_multi_processor_count}"
                print(msg)
            else:
                self.hyp['device'] = 'cpu'

    @staticmethod
    def scale_img(img, scale_factor):
        """

        :param img: (bn, 3, h, w)
        :param scale_factor: 输出的img shape必须能被scale_factor整除
        :return:
        """
        h, w = img.shape[2], img.shape[3]
        if scale_factor == 1.0:
            return img, h, w
        else:
            
            new_h, new_w = int(scale_factor * h), int(scale_factor * w)
            img = F.interpolate(img, size=(new_h, new_w), align_corners=False, mode='bilinear')
            out_h, out_w = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
            pad_right, pad_down = out_w - new_w, out_h - new_h
            pad = [0, pad_right, 0, pad_down]  # [left, right, up, down]
            return F.pad(img, pad, value=0.447), new_h, new_w

    @staticmethod
    def flip(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def tta(self, imgs):
        """
        对传入的imgs和相应的targets进行缩放,从而达到输入的每个batch中image shape不同的目的
        :param imgs: input image tensor from dataloader / tensor / (bn, 3, h, w)
        :return:
        """

        org_img_h, org_img_w = imgs.shape[2:]
        scale_facotr = [1   , 0.83]
        flip_axis    = [None, 2  ]
        # scale_facotr = [0.67]
        # flip_axis    = [3]
        tta_preds = []

        for s, f in zip(scale_facotr, flip_axis):
            if f:
                img_aug = imgs.flip(dims=(f,))
            else:
                img_aug = imgs
            img_aug, h, w = self.scale_img(img_aug, s)
            pred_segs = self.do_eval_forward(img_aug)  # (bs, 1, h, w)
            pred_segs = pred_segs[:, :, :h, :w]

            if s!= 1.0:
                pred_segs = F.interpolate(pred_segs, (org_img_h, org_img_w), align_corners=False, mode="bilinear")
            if f:  # restore flip
                pred_segs = pred_segs.flip(dims=(f,)).contiguous()
                # pred_segs = self.flip(pred_segs, f)
            
            # [(bs, 1, h, w), (bs, 1, h, w), (bs, 1, h, w)]
            tta_preds.append(pred_segs)

        pred_segs_out = tta_preds[0] * (1 / len(tta_preds))
        for i in range(1, len(tta_preds)):
            pred_segs_out += tta_preds[i] *  (1 / len(tta_preds))
        
        return pred_segs_out  # (bs, 1, h, w)
        
    def load_model(self, map_location):
        """
        load pretrained model, EMA model, optimizer(注意: __init_weights()方法并不适用于所有数据集)
        """
        # self._init_bias()
        if self.hyp.get("pretrained_model_path", None):
            model_path = self.hyp["pretrained_model_path"]
            if Path(model_path).exists():
                try:
                    state_dict = torch.load(model_path, map_location=map_location)
                    if "ema" in state_dict and self.hyp['use_ema_when_test']:  # load EMA model
                        self.model.load_state_dict(state_dict['ema'])
                        print(f"use pretrained EMA model from {model_path}")
                    else:
                        print(f"can't load EMA model from {model_path}, try to load plain model ... ...")
                        self.model.load_state_dict(state_dict["model_state_dict"])
                        print(f"use pretrained model {model_path}")
                    del state_dict
                except Exception as err:
                    print(err)
            else:
                print(f"{model_path} is not exist!")
    
    def do_eval_forward(self, imgs):
        self.model.eval()
        with torch.no_grad():   
            pred_segs = torch.sigmoid(self.model(imgs)['concat'])  # (bs, 1, h, w)
        self.model.train()
        return pred_segs


if __name__ == '__main__':
    from utils import print_config
    config_ = Config()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', type=str, required=True, dest='cfg', help='path to config file')
    # parser.add_argument('--pretrained_model_path',default="", dest='pretrained_model_path') 
    # parser.add_argument('--batch_size', type=int, default=2, dest='batch_size')
    # parser.add_argument("--input_img_size", default=640, type=int, dest='input_img_size')
    # parser.add_argument('--train_img_dir', required=True, dest='train_img_dir', type=str)
    # parser.add_argument('--train_seg_dir', required=True, dest='train_lab_dir', type=str)
    # parser.add_argument('--val_img_dir', required=True, dest='val_img_dir', type=str)
    # parser.add_argument('--val_seg_dir', required=True, dest='val_lab_dir', type=str)
    # parser.add_argument('--test_img_dir', required=True, dest='test_img_dir', type=str)
    # parser.add_argument('--model_save_dir', default="", type=str, dest='model_save_dir')
    # parser.add_argument('--log_save_path', default="", type=str, dest="log_save_path")
    # parser.add_argument('--aspect_ratio_path', default=None, dest='aspect_ratio_path', type=str, help="aspect ratio list for dataloader sampler, only support serialization object by pickle")
    # parser.add_argument('--cache_num', default=0, dest='cache_num', type=int)
    # parser.add_argument('--total_epoch', default=300, dest='total_epoch', type=int)
    # parser.add_argument('--do_warmup', default=True, type=bool, dest='do_warmup')
    # parser.add_argument('--use_tta', default=True, type=bool, dest='use_tta')
    # parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'], dest='optimizer')
    # parser.add_argument('--init_lr', default=0.01, type=float, dest='init_lr', help='initialization learning rate')
    # args = parser.parse_args()

    class Args:
        def __init__(self) -> None:
            self.cfg = "./config/train.yaml"

    args = Args()

    hyp = config_.get_config(args.cfg, args)
    formated_config = print_config(hyp)
    print(formated_config)
    train = Testing(hyp)
    train.step()
