import sys
import math
import random
import logging
import numbers
import argparse
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

import cv2
import time
import torch.cuda
import numpy as np
from torch import nn
from loguru import logger
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F
from torchinfo import summary as ModelSummary
from torchnet.meter import AverageValueMeter
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

current_work_directionary = Path('__file__').parent.absolute()
sys.path.insert(0, str(current_work_directionary))

from config import Config
from loss import CrossEntropyAndBCELoss, CrossEntropyAndDiceLoss
from utils import maybe_mkdir, clear_dir, get_rank, all_reduce_norm, SegMetirc2D
from utils import ExponentialMovingAverageModel, occupy_mem, is_parallel
from utils import summary_model, get_local_rank, adjust_status
from data import build_train_dataloader, build_val_dataloader, build_testdataloader
from utils import catch_warnnings, get_world_size, configure_omp, configure_nccl, print_config
from utils import save_seg, resize_segmentation, configure_module, synchronize, MeterBuffer
from nets import UPerNet, USquareNetExeriment
import gc


class Training:

    def __init__(self, hyp):
        configure_nccl()
        configure_omp()
        # parameters
        self.hyp = hyp
        self.select_device()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.hyp['device'] = self.device
        self.rank = get_rank()
        self.use_cuda = True if torch.cuda.is_available() else False
        self.is_distributed = get_world_size() > 1

        if dist.is_available() and dist.is_initialized():
            batch_size = self.hyp['batch_size'] // dist.get_world_size()
        else:
            batch_size = self.hyp['batch_size']
        self.lr = self.hyp['basic_lr_per_img'] * batch_size
        self.hyp['lr'] = self.lr
        self.cwd = Path('./').absolute()
        self.hyp['current_work_dir'] = str(self.cwd)
        self.meter = MeterBuffer()

        # 确保输入图片的shape必须能够被32整除(因为网络会对输入的image进行32倍的下采样),如果不满足条件则对设置的输入shape进行调整
        self.hyp['input_img_size'] = self.padding(self.hyp['input_img_size'], 32)
        self.dst_size = self.hyp['input_img_size']
        accumulate = self.get_local_accumulate_step()
        self.hyp['weight_decay'] *= self.hyp['batch_size'] *  accumulate / 64  # 当实际等效的batch_size大于64时，增大weight_decay
        
        # cpu不支持fp16训练
        self.data_type = torch.float16 if self.hyp['fp16'] and self.use_cuda else torch.float32
        self.before_train()
        
        # cudnn settings
        if not self.hyp['mutil_scale_training'] and self.hyp['device'] == 'cuda':
            # 对于输入数据的维度恒定的网络,使用如下配置可加速训练
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # config warmup step
        if self.hyp['do_warmup']:
            self.hyp['warmup_steps'] = max(self.hyp.get('warmup_epoch', 3) * len(self.traindataloader), 1000)

    def load_dataset(self, is_training):
        if is_training:
            dataset, dataloader, prefetcher = build_train_dataloader(img_dir=self.hyp['train_img_dir'], 
                                                                   seg_dir=self.hyp["train_seg_dir"], 
                                                                   batch_size=self.hyp['batch_size'], 
                                                                   drop_last=self.hyp['drop_last'], 
                                                                   dst_size=self.hyp['input_img_size'], 
                                                                   data_aug_hyp=self.hyp, 
                                                                   seed=self.hyp['random_seed'], 
                                                                   pin_memory=self.hyp['pin_memory'],
                                                                   num_workers=self.hyp['num_workers'],
                                                                   enable_data_aug=False, 
                                                                   cache_num=self.hyp['cache_num']
                                                                    )

        else:
            dataset, dataloader, prefetcher = build_val_dataloader(img_dir=self.hyp['val_img_dir'], 
                                                               seg_dir=self.hyp["val_seg_dir"], 
                                                               batch_size=self.hyp['batch_size'], 
                                                               dst_size=self.hyp['input_img_size'], 
                                                               num_workers=self.hyp['num_workers'], 
                                                               cache_num=self.hyp['cache_num']
                                                               )
        return dataset, dataloader, prefetcher

    def _init_logger(self, model):
        model_summary = summary_model(model, self.hyp['input_img_size'], verbose=True)
        logger = logging.getLogger(f"UPerNet_Rank_{self.rank}")
        formated_config = print_config(self.hyp)  # record training parameters in log.txt
        logger.setLevel(logging.INFO)
        txt_log_path = str(self.cwd / 'log' / f'log_rank_{self.rank}' / f'log_{self.model.__class__.__name__}_{datetime.now().strftime("%Y%m%d-%H:%M:%S")}_{self.hyp["log_postfix"]}.txt')
        maybe_mkdir(Path(txt_log_path).parent)
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
        return logger

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

    def _init_optimizer(self, model):
        param_group_weight, param_group_bias, param_group_other = [], [], []
        for m in model.modules():
            if hasattr(m, "bias") and isinstance(m.bias, nn.Parameter):
                param_group_bias.append(m.bias)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                param_group_other.append(m.weight)
            elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
                param_group_weight.append(m.weight)

        lr = self.lr
        
        if self.hyp['optimizer_type'].lower() == "sgd":
            optimizer = optim.SGD(params=param_group_other, lr=lr, nesterov=True, momentum=self.hyp['optimizer_momentum'])
        elif self.hyp['optimizer_type'].lower() == "adam":
            optimizer = optim.Adam(params=param_group_other, lr=lr, betas=(self.hyp['optimizer_momentum'], 0.999), eps=self.hyp['eps'])
        elif self.hyp['optimizer_type'].lower() == "adamw":
            optimizer = optim.AdamW(params=param_group_other, lr=lr, betas=(self.hyp['optimizer_momentum'], 0.999), eps=self.hyp['eps'], weight_decay=0.0)
        else:
            RuntimeError(f"Unkown optim_type {self.hyp['optimizer_type']}")

        optimizer.add_param_group({"params": param_group_weight, "weight_decay": self.hyp['weight_decay']})
        optimizer.add_param_group({"params": param_group_bias, "weight_decay": 0.0})

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

    def _init_scheduler(self, optimizer, trainloader):
        if self.hyp['scheduler_type'].lower() == "onecycle":   # onecycle lr scheduler
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=self.hyp['total_epoch'], steps_per_epoch=len(trainloader), three_phase=True)
        elif self.hyp['scheduler_type'].lower() == 'linear':  # linear lr scheduler
            max_ds_rate = 0.01
            linear_lr = lambda epoch: (1 - epoch / (self.hyp['total_epoch'] - 1)) * (1. - max_ds_rate) + max_ds_rate  # lr_bias越大lr的下降速度越慢,整个epoch跑完最后的lr值也越大
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lr)
        else:  # consin lr scheduler
            max_ds_rate = 0.01  # 整个训练过程中lr的最小值等于: max_ds_rate * init_lr
            cosin_lr = lambda epoch: ((1 + math.cos(epoch * math.pi / self.hyp['total_epoch'])) / 2) * (1. - max_ds_rate) + max_ds_rate  # cosine
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=cosin_lr)
        return scheduler

    def before_train(self):
        occupy_mem(self.local_rank)

        # datasets
        self.traindataset, self.traindataloader, self.trainprefetcher = self.load_dataset(is_training=True)
        self.valdataset  , self.valdataloader  , self.valprefetcher   = self.load_dataset(is_training=False)
        self.testdataset , self.testdataloader , self.testprefetcher  = build_testdataloader(self.hyp['test_img_dir'], self.hyp['input_img_size'], num_workers=self.hyp['num_workers'])
        self.hyp['num_class'] = self.traindataset.num_class

        self.scaler = amp.GradScaler(enabled=self.use_cuda)  # mix precision training
        self.total_loss_meter  = AverageValueMeter()
        
        torch.cuda.set_device(self.local_rank)
        model = USquareNetExeriment(in_channel=3, num_class=self.hyp['num_class'])
        ModelSummary(model, input_size=(1, 3, self.hyp['input_img_size'][0], self.hyp['input_img_size'][1]), device=next(model.parameters()).device)
        self.optimizer    = self._init_optimizer(model)
        self.lr_scheduler = self._init_scheduler(self.optimizer, self.traindataloader)
        if self.rank == 0:
            self.writer = SummaryWriter(log_dir=str(self.cwd / 'log' / f'log_rank_{self.rank}'))

        model = model.to(self.device)
        # load pretrained model
        self.load_model(model, None, self.optimizer, self.lr_scheduler, self.scaler, 'cpu')

        # loss function
        self.loss_fcn = CrossEntropyAndDiceLoss(num_class=self.hyp['num_class'])

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)
        if self.hyp['do_ema']:
            self.ema_model = ExponentialMovingAverageModel(model)
        self.model = model
        self.logger = self._init_logger(model)
        self.metric = SegMetirc2D()
            
    def warmup(self, epoch, tot_step):
        lr_bias = self.hyp['warmup_bias_lr']
        linear_lr = lambda epoch: (1. - epoch / (self.hyp['total_epoch'] - 1.)) * (1. - lr_bias) + lr_bias  # lr_bias越大lr的下降速度越慢,整个epoch跑完最后的lr值也越大
        if self.hyp['do_warmup'] and tot_step < self.hyp["warmup_steps"]:
            self.accumulate = int(max(1, np.interp(tot_step,
                                                   [0., self.hyp['warmup_steps']],
                                                   [1, self.get_local_accumulate_step()]).round()))
            # optimizer有3各param_group, 分别是parm_other, param_weight, param_bias
            for j, para_g in enumerate(self.optimizer.param_groups):
                if j != 2:  # param_other and param_weight(该部分参数的learning rate逐渐增大)
                    para_g['lr'] = np.interp(tot_step,
                                             [0., self.hyp['warmup_steps']],
                                             [0., para_g['initial_lr'] * linear_lr(epoch)])
                else:  # param_bias(该部分参数的learning rate逐渐减小,因为warmup_bias_lr大于initial_lr)
                    para_g['lr'] = np.interp(tot_step,
                                             [0., self.hyp['warmup_steps']],
                                             [self.hyp['warmup_bias_lr'], para_g['initial_lr'] * linear_lr(epoch)])
                if "momentum" in para_g:  # momentum(momentum在warmup阶段逐渐增大)
                    para_g['momentum'] = np.interp(tot_step,
                                                   [0., self.hyp['warmup_steps']],
                                                   [self.hyp['warmup_momentum'], self.hyp['optimizer_momentum']])


    @logger.catch
    @catch_warnnings
    def step(self):
        self.optimizer.zero_grad()
        tot_loss_before = float('inf')
        self.model.zero_grad()
        per_epoch_iters = len(self.traindataloader)
        for cur_epoch in range(1, self.hyp['total_epoch']+1):
            torch.cuda.empty_cache()
            self.model.train()
            for i in range(per_epoch_iters):
                start_time = time.time()
                if self.use_cuda:
                    x = self.trainprefetcher.next()
                else:
                    x = next(self.traindataloader)
                data_end_time = time.time()
                cur_step = i + 1
                tot_step = per_epoch_iters * cur_epoch + i + 1
                img    = x['img'].to(self.data_type)  # (bn, 3, h, w)
                gt_seg = x['seg'].to(self.data_type)  # (bn, 1, h, w)
                gt_seg.requires_grad = False

                img, gt_seg = self.mutil_scale_training(img, gt_seg)
                self.warmup(cur_epoch, tot_step)  # 在warmup期间lr_scheduler只有记录作用, 真正改变lr的还是warmup操作
                my_context = self.model.no_sync if self.is_distributed and cur_step % self.accumulate != 0 else nullcontext
                with my_context():
                    with amp.autocast(enabled=self.use_cuda):
                        preds = self.model(img)
                        loss_dict = self.loss_fcn(preds, gt_seg)
                        loss_dict['total_loss'] /= self.accumulate
                        loss_dict['total_loss'] *= get_world_size()

                    iter_end_time = time.time()
                    tot_loss = loss_dict['total_loss']
                    self.scaler.scale(tot_loss).backward()
                    if cur_step % self.accumulate == 0:
                        # self.scaler.unscale_(self.optimizer)
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        if self.hyp['do_ema']:
                            self.ema_model.update(self.model)

                iter_time = iter_end_time - start_time
                data_time = data_end_time - start_time
                is_best = tot_loss.item() < tot_loss_before
                self.update_meter(cur_epoch, cur_step, tot_step, img.size(2), img.size(0), iter_time, data_time, loss_dict, is_best)
                self.update_summarywriter()
                self.show_tbar(tot_step)
                self.cal_metric(tot_step)
                self.save_model(cur_epoch, tot_step, tot_loss, True)
                self.test(tot_step, cur_epoch)
                tot_loss_before = tot_loss.item()
                del x, img, gt_seg, tot_loss, preds, loss_dict
                if self.hyp['scheduler_type'].lower() == "onecycle":
                    self.lr_scheduler.step()  # 因为self.accumulate会从1开始增长, 因此第一次执行训练时self.optimizer.step()一定会在self.lr_scheduler.step()之前被执行
            if self.hyp['scheduler_type'].lower() != "onecycle":
                self.lr_scheduler.step()
            gc.collect()

    def update_meter(self, cur_epoch, cur_step, tot_step, input_dim, batch_size, iter_time, data_time, loss_dict, is_best):
        self.meter.update(iter_time  = iter_time, 
                          data_time  = data_time, 
                          input_dim  = input_dim,
                          cur_step   = cur_step, 
                          tot_step   = tot_step, 
                          cur_epoch  = cur_epoch, 
                          batch_size = batch_size,
                          is_best    = is_best, 
                          accumulate = self.accumulate,   
                          allo_mem   = torch.cuda.memory_allocated() / 2 ** 30 if self.use_cuda else 0.0,
                          cach_mem   = torch.cuda.memory_reserved() / 2 ** 30  if self.use_cuda else 0.0,
                          lr   = [x['lr'] for x in self.optimizer.param_groups][0], 
                          dice = self.meter.get_filtered_meter('dice')['dice'].global_avg if len(self.meter.get_filtered_meter('dice')) > 0 else 0.0, 
                          iou  = self.meter.get_filtered_meter('iou')['iou'].global_avg if len(self.meter.get_filtered_meter('iou')) > 0 else 0.0, 
                          fpr  = self.meter.get_filtered_meter('fpr')['fpr'].global_avg if len(self.meter.get_filtered_meter('fpr')) > 0 else 0.0, 
                          fnr  = self.meter.get_filtered_meter('fnr')['fnr'].global_avg if len(self.meter.get_filtered_meter('fnr')) > 0 else 0.0, 
                          voe  = self.meter.get_filtered_meter('voe')['voe'].global_avg if len(self.meter.get_filtered_meter('voe')) > 0 else 0.0, 
                          rvd  = self.meter.get_filtered_meter('rvd')['rvd'].global_avg if len(self.meter.get_filtered_meter('rvd')) > 0 else 0.0, 
                          acc  = self.meter.get_filtered_meter('acc')['acc'].global_avg if len(self.meter.get_filtered_meter('acc')) > 0 else 0.0, 
                          rank = self.rank,
                          **loss_dict)
    
    def get_local_batch_size(self):
        if dist.is_available() and dist.is_initialized():
            return self.hyp['batch_size'] // dist.get_world_size()
        return self.hyp['batch_size']

    def get_local_accumulate_step(self):
        if dist.is_available() and dist.is_initialized():
            return self.hyp['accu_batch_size'] / dist.get_world_size() / self.get_local_batch_size()
        return self.hyp['accu_batch_size'] / self.get_local_batch_size()

    def show_tbar(self, tot_step):
        tags = ("total_loss", "dice", 'iou'  , 'fnr'  , 'fpr'  , 'voe'  , 'rvd'  , 'acc'  , "accumulate", "iter_time", "lr"  , "cur_epoch", "cur_step", "batch_size", "input_dim", "allo_mem", "cach_mem")
        fmts = ('5.3f'      , '5.3f', '>5.3f', '>5.3f', '>5.3f', '>5.3f', '>5.3f', '>5.3f', '>02d'      , '5.3f'     , '5.3e', '>04d'     , '>05d'    , '>02d'      , '>03d'     , '5.3f'    ,  '5.3f')
        if tot_step % self.hyp['show_tbar_every'] == 0:
            show_dict = {}
            for tag in tags:
                show_dict[tag] = self.meter.get_filtered_meter(tag)[tag].latest
            if not math.isnan(show_dict['total_loss']):
                log_msg = ''
                for tag, fmt in zip(tags, fmts):
                    log_msg += tag + '=' + '{' + tag +  ':' + fmt + '}' + ", "
                self.logger.info(log_msg.format(**show_dict))

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

    def update_summarywriter(self):
        if self.rank == 0 and self.hyp['enable_tensorboard']:
            tot_step = self.meter.get_filtered_meter('tot_step')['tot_step'].latest
            for k, v in self.meter.items():
                self.writer.add_scalar(tag=f'train/{k}', scalar_value=v.latest, global_step=tot_step)
    
    def mutil_scale_training(self, imgs, segs):
        """
        对传入的imgs和相应的targets进行缩放,从而达到输入的每个batch中image shape不同的目的
        :param imgs: input image tensor from dataloader / tensor / (bn, 3, h, w)
        :param segs: targets of corrding images / tensor / (bn, 1, h, w)
        :return:

        todo: 
            随着训练的进行,image size逐渐增大。
        """
        if self.hyp['mutil_scale_training']:
            input_img_size = max(self.hyp['input_img_size'])
            random_shape = random.randrange(int(input_img_size * 0.6), int(input_img_size * 1.2 + 32)) // 32 * 32
            scale = random_shape / max(imgs.shape[2:])
            if scale != 1.:
                new_shape = [math.ceil(x * scale / 32) * 32 for x in imgs.shape[2:]]
                imgs = F.interpolate(imgs, size=new_shape, mode='bilinear', align_corners=False)
                segs_out = segs.new_zeros([segs.size(0), segs.size(1)] + new_shape)
                for i in range(segs.size(0)):
                    segs_out[i, 0] = torch.from_numpy(resize_segmentation(segs[i][0].detach().cpu().numpy(), new_shape, order=1)).to(self.data_type)
                return imgs, segs_out
        return imgs, segs

    def load_model(self, model, ema_model, optimizer, lr_scheduler, scaler, map_location):
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
                        model.load_state_dict(state_dict["model_state_dict"])
                        print(f"use pretrained model {model_path}")

                    if optimizer is not None and "optim_state_dict" in state_dict and state_dict.get("optim_type", None) == self.hyp['optimizer']:  # load optimizer
                        optimizer.load_state_dict(state_dict['optim_state_dict'])
                        lr_scheduler.load_state_dict(state_dict['lr_scheduler_state_dict'])
                        scaler.load_state_dict(state_dict['scaler_state_dict'])
                        print(f"use pretrained optimizer {model_path}")

                    if ema_model is not None and "ema" in state_dict:  # load EMA model
                        ema_model.ema.load_state_dict(state_dict['ema'])
                        print(f"use pretrained EMA model from {model_path}")
                    else:
                        print(f"can't load EMA model from {model_path}")
                    if ema_model is not None and 'ema_update_num' in state_dict:
                        ema_model.update_num = state_dict['ema_update_num']

                    del state_dict

                except Exception as err:
                    print(err)
            else:
                print('training from stratch!')
        else:
            print('training from stratch!')

    def save_model(self, cur_epoch, tot_step, tot_loss, save_optimizer):
        if self.rank == 0 and tot_step % int(self.hyp['save_ckpt_every'] * len(self.traindataloader)) == 0:
            save_path = str(self.cwd / 'checkpoints' / f'every_upernet.pth')            
            if not Path(save_path).exists():
                maybe_mkdir(Path(save_path).parent)

            state_dict = {
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": self.optimizer.state_dict() if save_optimizer else None,
                "scaler_state_dict": self.scaler.state_dict(),
                'lr_scheduler_type': self.hyp['scheduler_type'], 
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "optim_type": self.hyp['optimizer_type'], 
                "loss": tot_loss,
                "epoch": cur_epoch,
                "step": tot_step, 
                "ema": self.ema_model.ema.state_dict(), 
                "ema_update_num": self.ema_model.update_num, 
                "hyp": self.hyp, 
            }
            torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)
            del state_dict
    
    def do_eval_forward(self, imgs):
        self.model.eval()
        with torch.no_grad():   
            pred_segs = F.softmax(self.model(imgs.float())['fuse'], dim=1)  # (bs, num_class, h, w)
        self.model.train()
        return pred_segs

    def cal_metric(self, tot_step):
        if tot_step % int(self.hyp.get('calculate_metric_every', 0.5) * len(self.traindataloader)) == 0:
            # 在计算metric指标之前, 先清空之前的所有metric值
            metric_tags = ('dice', 'iou', 'fnr', 'fpr', 'voe','rvd', 'acc')
            for tag in metric_tags:
                if tag in self.meter.keys():
                    self.meter.get_filtered_meter(tag)[tag].reset()
            try:
                all_reduce_norm(self.model)  # 该函数只对batchnorm和instancenorm有效
            except:
                pass
            if self.hyp['do_ema']:
                eval_model = self.ema_model.ema
            else:
                eval_model = self.model
                if is_parallel(eval_model):
                    eval_model = eval_model.module
            with adjust_status(eval_model, training=False):
                for j in range(len(self.valdataloader)):
                    if self.use_cuda:
                        y = self.valprefetcher.next()
                    else:
                        y = next(self.valdataloader)
                    img  = y['img'].to(self.data_type)  # (bs, 3, h, w)
                    gt_seg_numpy = y['seg'].squeeze().to(self.data_type).detach().cpu().numpy()  # (bs, 1, h, w)
                    pred_seg = self.do_eval_forward(img)  # (bs, num_class, h, w)
                    pred_seg_numpy = torch.argmax(pred_seg, dim=1).detach().cpu().numpy()  # (bs, h, w)
                    for i in range(img.size(0)):
                        self.metric.update(gt_seg_numpy[i], pred_seg_numpy[i])
                        self.meter.update(dice = self.metric.dice, 
                                          iou  = self.metric.iou, 
                                          fnr  = self.metric.fnr, 
                                          fpr  = self.metric.fpr, 
                                          voe  = self.metric.voe,
                                          rvd  = self.metric.rvd, 
                                          acc  = self.metric.acc,
                                          )

    def test(self, cur_step, cur_epoch):
        if cur_step % int(self.hyp.get('inference_every', 1.0)*len(self.traindataloader))== 0:
            torch.cuda.empty_cache()
            try:
                all_reduce_norm(self.model)  # 该函数只对batchnorm和instancenorm有效
            except:
                pass
            if self.hyp['do_ema']:
                eval_model = self.ema_model.ema
            else:
                eval_model = self.model
                if is_parallel(eval_model):
                    eval_model = eval_model.module
            with adjust_status(eval_model, training=False):
                for j in range(len(self.testdataloader)):
                    if self.use_cuda:
                        y = self.testprefetcher.next()
                    else:
                        y = next(self.testdataloader)
                    img  = y['img'].to(self.data_type)  # (bs, 3, h, w)
                    info = y['info']
                    if self.hyp['use_tta_when_val']:
                        pred_seg = self.tta(img)  # (bs, num_class, h, w)
                    else:
                        pred_seg = self.do_eval_forward(img)  # (1, num_class, h, w)
                    img_numpy, pred_seg_numpy = self.postprocess(img.detach().cpu().numpy(), pred_seg.detach().cpu().numpy(), info)
                    for k in range(len(img)):
                        save_path = str(self.cwd / 'result' / f'predictions_rank_{self.rank}' / f"epoch{cur_epoch+1}_img_{j + k} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.png")
                        maybe_mkdir(Path(save_path).parent)
                        save_seg(img_numpy[k], pred_seg_numpy[k], save_path)
                    del y, img, info, img_numpy, pred_seg, pred_seg_numpy
                    
            synchronize()
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
            pred_segs = self.do_eval_forward(img_aug)  # ndarray / (bs, num_class, h, w)
            pred_segs = pred_segs[:, :, :h, :w]
            if s != 1.0:
                pred_segs = F.interpolate(pred_segs, size=(org_img_h, org_img_w), align_corners=False, mode='bilinear')
            if f:  # restore flip
                pred_segs = pred_segs.flip(dims=(f,)).contiguous()
            # [(bs, num_class, h, w), (bs, num_class, h, w), (bs, num_class, h, w)]
            tta_preds.append(pred_segs)

        pred_segs_out = tta_preds[0] * (1 / len(tta_preds))
        for i in range(1, len(tta_preds)):
            pred_segs_out += tta_preds[i] *  (1 / len(tta_preds))
        return pred_segs_out  # (bs, num_class, h, w)


def main(x):
    configure_module()
    from utils import print_config
    config_ = Config()
    class Args:
        def __init__(self) -> None:
            self.cfg = "./config/train_dist.yaml"
    args = Args()

    hyp = config_.get_config(args.cfg, args)
    formated_config = print_config(hyp)
    print(formated_config)
    train = Training(hyp)
    train.step()


if __name__ == '__main__':
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
    
    from utils import launch, get_num_devices
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    num_gpu = get_num_devices()
    # clear_dir(str(current_work_directionary / 'log'))
    launch(
        main, 
        num_gpus_per_machine= num_gpu, 
        num_machines= 1, 
        machine_rank= 0, 
        backend= "nccl", 
        dist_url= "auto", 
        args=(None,),
    )
