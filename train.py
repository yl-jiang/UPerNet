import sys
import math
import random
import logging
import numbers
import argparse
from pathlib import Path
from datetime import datetime

current_work_directionary = Path('__file__').parent.absolute()
sys.path.insert(0, str(current_work_directionary))

import cv2
import torch.cuda
import numpy as np
from torch import nn
from tqdm import tqdm
from tqdm import trange
from loguru import logger
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary as ModelSummary
from torchnet.meter import AverageValueMeter
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from config import Config
from loss import MultiBCELoss
from utils import maybe_mkdir, clear_dir
from utils import ExponentialMovingAverageModel
from utils import time_synchronize, summary_model
from data import build_dataloader, build_testdataloader
from utils import catch_warnnings
from utils import Metric, save_seg, resize_segmentation
from models import UPerNet
import gc


class Training:

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
       
        # dataset, scaler, loss_meter
        self.traindataset, self.traindataloader, self.trainprefetcher = self.load_dataset(is_training=True)
        self.valdataset, self.valdataloader, self.valprefetcher       = self.load_dataset(is_training=False)
        self.testdataset, self.testdataloader, self.testprefetcher    = build_testdataloader(self.hyp['test_img_dir'], self.hyp['input_img_size'])
        self.hyp['num_class'] = self.traindataset.num_class
        self.scaler = amp.GradScaler(enabled=self.use_cuda)  # mix precision training
        self.total_loss_meter  = AverageValueMeter()
        self.metric = Metric()
        
        self.writer = SummaryWriter(log_dir=str(self.cwd / 'log'))
        self.init_lr = self.hyp['init_lr']
        
        # model, optimizer, loss, lr_scheduler, ema
        self.model = UPerNet(in_channel=3, num_class=self.hyp['num_class']).to(hyp['device'])
        ModelSummary(self.model, input_size=(1, 3, self.hyp['input_img_size'][0], self.hyp['input_img_size'][1]), device=self.hyp['device'])
        self.optimizer = self._init_optimizer()
        self.optim_scheduler = lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=self._lr_lambda)
        self.loss_fcn = MultiBCELoss(num_class=self.hyp['num_class'])
        self.ema_model = ExponentialMovingAverageModel(self.model)
        # cpu不支持fp16训练
        self.data_type = torch.float16 if self.hyp['fp16'] and self.use_cuda else torch.float32

        # load pretrained model or initialize model's parameters
        self.load_model(False, 'cpu')

        # logger
        self.logger = self._init_logger()
        self._init_tbar()

        # cudnn settings
        if not self.hyp['mutil_scale_training'] and self.hyp['device'] == 'cuda':
            # 对于输入数据的维度恒定的网络,使用如下配置可加速训练
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # config warmup step
        if self.hyp['do_warmup']:
            self.hyp['warmup_steps'] = max(self.hyp.get('warmup_epoch', 3) * len(self.traindataloader), 1000)
        self.accumulate = self.hyp['accumulate_loss_step'] / self.hyp['batch_size']

    def load_dataset(self, is_training):
        if is_training:
            dataset, dataloader, prefetcher = build_dataloader(img_dir=self.hyp['train_img_dir'], 
                                                               seg_dir=self.hyp["train_seg_dir"], 
                                                               batch_size=self.hyp['batch_size'], 
                                                               drop_last=self.hyp['drop_last'], 
                                                               dst_size=self.hyp['input_img_size'], 
                                                               data_aug_hyp=self.hyp, 
                                                               seed=self.hyp['random_seed'], 
                                                               pin_memory=self.hyp['pin_memory'],
                                                               num_workers=self.hyp['num_workers'],
                                                               enable_data_aug=True, 
                                                               cache_num=self.hyp['cache_num']
                                                               )

        else:
            dataset, dataloader, prefetcher = build_dataloader(img_dir=self.hyp['val_img_dir'], 
                                                               seg_dir=self.hyp["val_seg_dir"], 
                                                               batch_size=self.hyp['batch_size'], 
                                                               drop_last=False, 
                                                               dst_size=self.hyp['input_img_size'], 
                                                               enable_data_aug=False, 
                                                               data_aug_hyp=None
                                                               )
        return dataset, dataloader, prefetcher

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

    def _init_logger(self):
        clear_dir(str(self.cwd / 'log'))  # 再写入log文件前先清空log文件夹
        model_summary = summary_model(self.model, self.hyp['input_img_size'], verbose=True)
        logger = logging.getLogger("UPerNet")
        formated_config = print_config(hyp)  # record training parameters in log.txt
        logger.setLevel(logging.INFO)
        if self.hyp['save_log_txt']:
            if self.hyp.get('log_save_path', None) and Path(self.hyp['log_save_path']).exists():
                txt_log_path = self.hyp['log_save_path']
            else:
                txt_log_path = str(self.cwd / 'log' / 'log.txt')
            maybe_mkdir(Path(txt_log_path).parent)
        else:
            return None
        handler = logging.FileHandler(txt_log_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.info("\n" + formated_config)
        msg = f"\n{'=' * 70} Model Summary {'=' * 70}\n"
        msg += f"Model Summary:\tlayers {model_summary['number_layers']}; parameters {model_summary['number_params']}; gradients {model_summary['number_gradients']}; flops {model_summary['flops']}GFLOPs"
        msg += f"\n{'=' * 70}   Training    {'=' * 70}\n"
        logger.info(msg)
        tags = ("all_mem(G)", "cac_mem(G)", "epoch", "step", "batchsz", "img_shape", "tot_loss", "model_saved")
        logger.info("{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}".format(*tags))
        return logger

    def _init_tbar(self):
        tbar_tags = ("epoch", "tot", "imgsz", "lr", 'pixel_acc', "time(s)")
        msg = "%16s" * len(tbar_tags)
        print(msg % tbar_tags)

    def _lr_lambda(self, epoch, scheduler_type='linear'):
        lr_bias = self.hyp['lr_scheculer_bias']  # lr_bias越大lr的下降速度越慢,整个epoch跑完最后的lr值也越大
        if scheduler_type == 'linear':
            return (1 - epoch / (self.hyp['total_epoch'] - 1)) * (1. - lr_bias) + lr_bias
        elif scheduler_type == 'cosine':
            return ((1 + math.cos(epoch * math.pi / self.hyp['total_epoch'])) / 2) * (1. - lr_bias) + lr_bias  # cosine
        else:
            return math.pow(1 - epoch / self.hyp['total_epoch'], 0.9)

    def _init_optimizer(self):
        param_group_weight, param_group_bias, param_group_other = [], [], []
        for m in self.model.modules():
            if hasattr(m, "bias") and isinstance(m.bias, nn.Parameter):
                param_group_bias.append(m.bias)
            
            if isinstance(m, nn.BatchNorm2d):
                param_group_other.append(m.weight)
            elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
                param_group_weight.append(m.weight)

        if self.hyp['optimizer'].lower() == "sgd":
            optimizer = optim.SGD(params=param_group_other, lr=self.hyp['init_lr'], nesterov=True, momentum=self.hyp['momentum'])
        elif self.hyp['optimizer'].lower() == "adam":
            optimizer = optim.Adam(params=param_group_other, lr=self.hyp['init_lr'], betas=(self.hyp['momentum'], 0.999), eps=self.hyp['eps'])
        else:
            RuntimeError(f"Unkown optim_type {self.hyp['optimizer']}")

        optimizer.add_param_group({"params": param_group_weight, "weight_decay": self.hyp['weight_decay']})
        optimizer.add_param_group({"params": param_group_bias})

        del param_group_weight, param_group_bias, param_group_other
        return optimizer

    def _init_bias(self):
        """
        初始化模型参数,主要是对detection layers的bias参数进行特殊初始化,参考RetinaNet那篇论文,这种初始化方法可让网络较容易度过前期训练困难阶段
        (使用该初始化方法可能针对coco数据集有效,在对global wheat数据集的测试中,该方法根本train不起来)
        """
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
            elif isinstance(m, (nn.LeakyReLU, nn.ReLU, nn.ReLU6)):
                m.inplace = True

    @logger.catch
    @catch_warnnings
    def step(self):
        self.optimizer.zero_grad()
        tot_loss_before = float('inf')
        epoch_use_time = 0
        self.model.zero_grad()
        per_epoch_iters = len(self.traindataloader)
        for epoch in range(self.hyp['total_epoch']):
            self.model.train()
            epoch_start = time_synchronize()
            with tqdm(total=per_epoch_iters, file=sys.stdout) as tbar:
                for i in range(per_epoch_iters):
                    if self.use_cuda:
                        x = self.trainprefetcher.next()
                    else:
                        x = next(self.traindataloader)
                    cur_steps = per_epoch_iters * epoch + i + 1
                    img    = x['img'].to(self.data_type)  # (bn, 3, h, w)
                    gt_seg = x['seg'].to(self.data_type)  # (bn, num_class, h, w)
                    
                    gt_seg.requires_grad = False
                    img, gt_seg = self.mutil_scale_training(img, gt_seg)
                    batchsz, inp_c, inp_h, inp_w = img.shape

                    # warmup
                    self.warmup(epoch, cur_steps)

                    # forward
                    with amp.autocast(enabled=self.use_cuda):
                        preds = self.model(img)
                        loss_dict = self.loss_fcn(preds, gt_seg)

                    tot_loss = loss_dict['total'] * (self.hyp['batch_size'] / self.hyp['accumulate_loss_step'])

                    # backward
                    self.scaler.scale(tot_loss).backward()

                    # optimize
                    if cur_steps % self.accumulate == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        # maintain a model and update it every time, only using for inference
                        if self.hyp['do_ema']:
                            self.ema_model.update(self.model)

                    # tensorboard
                    self.update_loss_meter(loss_dict)
                    is_best = tot_loss.item() < tot_loss_before
                    if self.hyp['enable_tensorboard']:
                        self.update_summarywriter(cur_steps, loss_dict, self.metric.pixel_acc)
                    tot_loss_before = tot_loss.item()

                    # verbose
                    self.show_tbar(cur_steps, tbar, epoch+1, i, batchsz, is_best, inp_h, epoch_use_time, loss_dict, self.metric.pixel_acc)
                    # testing
                    self.test(cur_steps)
                    # validation
                    self.cal_metric(cur_steps)
                    # save model
                    self.save_model(cur_steps, tot_loss, epoch+1, cur_steps, True)
                    
                    # save the lastest model
                    if epoch == self.hyp['total_epoch']-1:
                        save_path = str(self.cwd / 'checkpoints' / f'finally.pth') 
                        self.save_model(tot_loss, epoch, cur_steps, True, save_path)

                    del img, gt_seg, tot_loss
                    tbar.update()
                tbar.close()

            epoch_use_time = time_synchronize() - epoch_start
            # update lr
            self.optim_scheduler.step()
            gc.collect()

    def show_tbar(self, steps, tbar, epoch, step, batchsz, is_best, input_dim, epoch_use_time, loss_dict, pixel_acc):
        if steps % int(self.hyp.get('show_tbar_every', 5)) == 0:
            # tbar
            lrs = [x['lr'] for x in self.optimizer.param_groups]
            msg_dict = {"epoch": epoch, "input_dim": input_dim, "epoch_use_time": epoch_use_time, "lr": lrs[0], "pixel_acc": pixel_acc}
            msg_dict.update({k:v.item() for k, v in loss_dict.items()})
            if epoch_use_time == 0.0:  # 不显示第一个epoch的用时
                msg_dict['epoch_use_time'] = ""
                #              ("epoch",         "tot",        "imgsz",      "lr",      'pixel_acc',             "time(s)")
                tbar_msg = "#  {epoch:^19d}{total:^18.3f}{input_dim:^13d}{lr:^16.3e}{pixel_acc:^14.3f}{epoch_use_time:^15s}"
            else:
                #              ("epoch",         "tot",        "imgsz",      "lr",      'iou',           "time(s)")
                tbar_msg = "#  {epoch:^19d}{total:^18.3f}{input_dim:^13d}{lr:^16.3e}{pixel_acc:^14.3f}{epoch_use_time:^15.1f}"
            
            tbar.set_description_str(tbar_msg.format(**msg_dict))

            # maybe save info to log.txt
            if self.hyp['device'].lower() == "cuda" and torch.cuda.is_available():
                allocated_memory = torch.cuda.memory_allocated() / 2 ** 30
                cached_memory = torch.cuda.memory_reserved() / 2 ** 30
            else:
                allocated_memory = 0.
                cached_memory = 0.
            if self.logger is not None:
                #            ("all_mem(G)",               "cac_mem(G)",        "epoch",     "step",    "batchsz",     "img_shape",           "tot_loss",           "model_saved")
                log_msg = f"{allocated_memory:^15.2f}{cached_memory:^15.2f}{(epoch):^15d}{step:^15d}{batchsz:^15d}{str(input_dim):^15s}{loss_dict['total']:^15.5f}"
                self.logger.info(log_msg + f"{'yes' if is_best else 'no':^15s}")

    def warmup(self, epoch, cur_steps):
        if self.hyp['do_warmup'] and cur_steps < self.hyp["warmup_steps"]:
            self.accumulate = max(1, np.interp(cur_steps,
                                               [0., self.hyp['warmup_steps']],
                                               [1, self.hyp['accumulate_loss_step'] / self.hyp['batch_size']]).round())
            # optimizer有3各param_group,分别是parm_other, param_weight, param_bias
            for j, para_g in enumerate(self.optimizer.param_groups):
                if j != 2:  # param_other and param_weight(该部分参数的learning rate逐渐增大)
                    para_g['lr'] = np.interp(cur_steps,
                                             [0., self.hyp['warmup_steps']],
                                             [0., para_g['initial_lr'] * self._lr_lambda(epoch)])
                else:  # param_bias(该部分参数的learning rate逐渐减小,因为warmup_bias_lr大于initial_lr)
                    para_g['lr'] = np.interp(cur_steps,
                                             [0., self.hyp['warmup_steps']],
                                             [self.hyp['warmup_bias_lr'], para_g['initial_lr'] * self._lr_lambda(epoch)])
                if "momentum" in para_g:  # momentum(momentum在warmup阶段逐渐增大)
                    para_g['momentum'] = np.interp(cur_steps,
                                                   [0., self.hyp['warmup_steps']],
                                                   [self.hyp['warmup_momentum'], self.hyp['momentum']])

    def postprocess(self, inp, pred_segs, info):
        """
        Inputs:
            inp: normalization image (bs, 3, h, w)
            outputs: (bs, 1, h, w)
            info:
        Outpus:

        """
        assert isinstance(inp, np.ndarray)
        assert isinstance(inp, np.ndarray)
        processed_segs = []
        processed_imgs = []
        batch_num = inp.shape[0]

        for i in range(batch_num):
            pad_top, pad_left = info[i]['pad_top'], info[i]['pad_left']
            pad_bot, pad_right = info[i]['pad_bottom'], info[i]['pad_right']
            pred = pred_segs[i]  # (num_class, h, w)
            org_h, org_w = info[i]['org_shape']
            cur_h, cur_w = inp[i].shape[1], inp[i].shape[2]

            img = np.transpose(inp[i], axes=[1, 2, 0])  # (h, w, 3)
            img *= 255.0
            img = np.clip(img, 0, 255.0)
            img = img.astype(np.uint8)
            img = img[pad_top:(cur_h - pad_bot), pad_left:(cur_w - pad_right), :]
            img = cv2.resize(img, (org_w, org_h), interpolation=0)
            processed_imgs.append(img)
            
            pred = np.argmax(pred_segs[i], axis=0).astype(np.uint8)  # (h, w)
            pred = pred[pad_top:(cur_h - pad_bot), pad_left:(cur_w - pad_right)]
            pred = resize_segmentation(pred, (org_h, org_w))
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

    def update_summarywriter(self, steps, loss_dict, pixel_acc):
        lrs = [x['lr'] for x in self.optimizer.param_groups]
        self.writer.add_scalar(f'train/{self.hyp["optimizer"]}_lr', lrs[0], steps)
        self.writer.add_scalar('train/pixel_acc', pixel_acc, steps//int(self.hyp['calculate_metric_every'] * len(self.traindataloader)))

        for k in loss_dict.keys():
            mean_value = eval(f"self.{k}_loss_meter.value()[0]")
            self.writer.add_scalar(tag=f'train/{k}', scalar_value=mean_value, global_step=steps)
    
    def mutil_scale_training(self, imgs, segs):
        """
        对传入的imgs和相应的targets进行缩放,从而达到输入的每个batch中image shape不同的目的
        :param imgs: input image tensor from dataloader / tensor / (bn, 3, h, w)
        :param segs: targets of corrding images / tensor
        :return:

        todo: 
            随着训练的进行,image size逐渐增大。
        """
        if self.hyp['mutil_scale_training']:
            input_img_size = max(self.hyp['input_img_size'])
            random_shape = random.randrange(input_img_size * 0.5, input_img_size * 1.5 + 32) // 32 * 32
            scale = random_shape / max(imgs.shape[2:])
            if scale != 1.:
                new_shape = [math.ceil(x * scale / 32) * 32 for x in imgs.shape[2:]]
                imgs = F.interpolate(imgs, size=new_shape, mode='bilinear', align_corners=False)
                segs = F.interpolate(segs, size=new_shape, mode='bilinear', align_corners=False)
        return imgs, segs

    def load_model(self, load_optimizer, map_location):
        """
        load pretrained model, EMA model, optimizer(注意: __init_weights()方法并不适用于所有数据集)
        """
        # self._init_bias()
        if self.hyp.get("pretrained_model_path", None):
            model_path = self.hyp["pretrained_model_path"]
            if Path(model_path).exists():
                try:
                    state_dict = torch.load(model_path, map_location=map_location)
                    if "model_state_dict" not in state_dict:
                        print(f"can't load pretrained model from {model_path}")
    
                    else:  # load pretrained model
                        self.model.load_state_dict(state_dict["model_state_dict"])
                        print(f"use pretrained model {model_path}")

                    if load_optimizer and "optim_state_dict" in state_dict and state_dict.get("optim_type", None) == self.hyp['optimizer']:  # load optimizer
                        self.optimizer.load_state_dict(state_dict['optim_state_dict'])
                        self.optim_scheduler.load_state_dict(state_dict['lr_scheduler_state_dict'])
                        self.scaler.load_state_dict(state_dict['scaler_state_dict'])
                        print(f"use pretrained optimizer {model_path}")

                    if "ema" in state_dict:  # load EMA model
                        self.ema_model.ema.load_state_dict(state_dict['ema'])
                        print(f"use pretrained EMA model from {model_path}")
                    else:
                        print(f"can't load EMA model from {model_path}")
                    if 'ema_update_num' in state_dict:
                        self.ema_model.update_num = state_dict['ema_update_num']

                    del state_dict

                except Exception as err:
                    print(err)
            else:
                print('training from stratch!')
        else:
            print('training from stratch!')

    def save_model(self, steps, loss, epoch, step, save_optimizer, save_path=None):
        if steps % int(self.hyp['save_ckpt_every'] * len(self.traindataloader)) == 0:
            if self.hyp.get('model_save_dir', None) and Path(self.hyp['model_save_dir']).exists():
                save_path = self.hyp['model_save_dir']
            else:
                save_path = str(self.cwd / 'checkpoints' / f'every_upernet.pth')            

            if not Path(save_path).exists():
                maybe_mkdir(Path(save_path).parent)

            hyp = {"hyp": self.hyp}

            optim_state = self.optimizer.state_dict() if save_optimizer else None
            state_dict = {
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": optim_state,
                "scaler_state_dict": self.scaler.state_dict(),
                "lr_scheduler_state_dict": self.optim_scheduler.state_dict(),
                "optim_type": self.hyp['optimizer'], 
                "loss": loss,
                "epoch": epoch,
                "step": step, 
                "ema": self.ema_model.ema.state_dict(), 
                "ema_update_num": self.ema_model.update_num, 
                "hyp": hyp, 
            }
            torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)
            del state_dict, optim_state, hyp

    def update_loss_meter(self, loss_dict):
        for k, v in loss_dict.items():
            if not math.isnan(v):
                eval(f"self.{k}_loss_meter.add(v.item())")
    
    def do_eval_forward(self, imgs):
        self.model.eval()
        with torch.no_grad():   
            pred_segs = F.softmax(self.model(imgs.float())['fuse'], dim=1)  # (bs, num_class, h, w)
        self.model.train()
        return pred_segs

    def cal_metric(self, steps):
        if steps % int(self.hyp.get('calculate_metric_every', 0.5) * len(self.traindataloader)) == 0:
            self.metric.clear()
            for _ in trange(len(self.valdataloader)):
                if self.valprefetcher is not None:
                    x = self.valprefetcher.next()
                else:
                    x = next(self.valdataloader)
                imgs      = x['img'].float()  # (bn, 3, h, w)
                gt_segs   = x['seg'].float()  # (bn, 1, h, w)
                pred_segs = self.do_eval_forward(imgs)   # (bn, num_class, h, w)
                pred_segs_finnal = torch.argmax(pred_segs, dim=1, keepdim=True)  # (bs, h, w)
                self.metric.update(pred_segs_finnal.cpu().numpy(), gt_segs.cpu().numpy())
                del imgs, gt_segs, pred_segs
            gc.collect()

    def test(self, steps):
        if steps % int(self.hyp.get('inference_every', 0.5)*len(self.traindataloader))== 0:
            for j in range(len(self.testdataloader)):
                if self.use_cuda:
                    y = self.testprefetcher.next()
                else:
                    y = next(self.testdataloader)
                img  = y['img'].to(self.data_type)  # (1, 3, h, w)
                info = y['resize_info']
                if self.hyp['use_tta_when_val']:
                    pred_seg = self.tta(img)  # (1, num_class, h, w)
                else:
                    pred_seg = self.do_eval_forward(img)  # (1, num_class, h, w)
                img_numpy, pred_seg_numpy = self.postprocess(img.cpu().numpy(), pred_seg.cpu().numpy(), info)
                for k in range(len(img)):
                    save_path = str(self.cwd / 'result' / 'predictions' / f"{j + k} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png")
                    maybe_mkdir(Path(save_path).parent)
                    save_seg(img_numpy[k], pred_seg_numpy[k], save_path)
                del y, img, info, pred_seg, img_numpy, pred_seg_numpy
        gc.collect()

    @staticmethod
    def scale_img(img, scale_factor):
        """
        Inputs:
            img: (bn, 3, h, w)
            scale_factor: 输出的img shape必须能被scale_factor整除
        Outputs:
            resized_img: 
            new_h: resized_img's height
            new_w: resized_img's width
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

    def tta(self, imgs):
        """
        对传入的imgs和相应的targets进行缩放,从而达到输入的每个batch中image shape不同的目的
        :param imgs: input image tensor from dataloader / tensor / (bn, 3, h, w)
        :return:
        """

        org_img_h, org_img_w = imgs.shape[2:]
        scale_facotr = [1, 0.83, 0.67]
        flip_axis = [None, 2, 3]
        tta_preds = []

        for s, f in zip(scale_facotr, flip_axis):
            if f:
                img_aug = imgs.flip(dims=(f,)).contiguous()
            else:
                img_aug = imgs
            img_aug, h, w = self.scale_img(img_aug, s)
            pred_segs = self.do_eval_forward(img_aug)  # (bs, num_class, h, w)
            pred_segs = pred_segs[:, :, :h, :w]
            if s != 1.:
                pred_segs = resize_segmentation(pred_segs, (org_img_h, org_img_w))
            if f:  # restore flip
                pred_segs = pred_segs.flip(dims=(f,)).contiguous()
            # [(bs, num_class, h, w), (bs, num_class, h, w), (bs, num_class, h, w)]
            tta_preds.append(pred_segs)

        pred_segs_out = tta_preds[0] * (1 / len(tta_preds))
        for i in range(1, len(tta_preds)):
            pred_segs_out += tta_preds[i] *  (1 / len(tta_preds))
        
        return pred_segs_out  # (bs, num_class, h, w)

if __name__ == '__main__':
    from utils import print_config
    config_ = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=False, dest='cfg', help='path to config file')
    parser.add_argument('--pretrained_model_path',default="", dest='pretrained_model_path') 
    parser.add_argument('--batch_size', type=int, default=2, dest='batch_size')
    parser.add_argument("--input_img_size", default=640, type=int, dest='input_img_size')
    parser.add_argument('--train_img_dir', required=False, dest='train_img_dir', type=str)
    parser.add_argument('--train_seg_dir', required=False, dest='train_lab_dir', type=str)
    parser.add_argument('--val_img_dir', required=False, dest='val_img_dir', type=str)
    parser.add_argument('--val_seg_dir', required=False, dest='val_lab_dir', type=str)
    parser.add_argument('--test_img_dir', required=False, dest='test_img_dir', type=str)
    parser.add_argument('--model_save_dir', default="", type=str, dest='model_save_dir')
    parser.add_argument('--log_save_path', default="", type=str, dest="log_save_path")
    parser.add_argument('--aspect_ratio_path', default=None, dest='aspect_ratio_path', type=str, help="aspect ratio list for dataloader sampler, only support serialization object by pickle")
    parser.add_argument('--cache_num', default=0, dest='cache_num', type=int)
    parser.add_argument('--total_epoch', default=300, dest='total_epoch', type=int)
    parser.add_argument('--do_warmup', default=True, type=bool, dest='do_warmup')
    parser.add_argument('--use_tta', default=True, type=bool, dest='use_tta')
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'], dest='optimizer')
    parser.add_argument('--init_lr', default=0.01, type=float, dest='init_lr', help='initialization learning rate')
    args = parser.parse_args()

    # class Args:
    #     def __init__(self) -> None:
    #         self.cfg = "./config/train.yaml"
    # args = Args()

    hyp = config_.get_config(args.cfg, args)
    formated_config = print_config(hyp)
    print(formated_config)
    train = Training(hyp)
    train.step()
