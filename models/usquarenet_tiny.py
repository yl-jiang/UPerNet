import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["USquareNetTiny"]


def freeze_bn(m):
    """
    https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/12
    """
    if isinstance(m, nn.BatchNorm2d):
        if hasattr(m, 'weight'):
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias'):
            m.bias.requires_grad_(False)
        m.eval()  # for freeze bn layer's parameters 'running_mean' and 'running_var


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.dim = dimension

    def forward(self, x):
        assert isinstance(x, (list, tuple))
        return torch.cat(x, dim=self.dim)


def autopad(kernel, padding):
    if padding is None:
        return kernel // 2 if isinstance(kernel, int) else [p // 2 for p in kernel]
    else:
        return padding


class Upsample(nn.Module):

    def __init__(self, factor=2) -> None:
        super(Upsample, self).__init__()
        self.factor = factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode="bilinear")


class ConvBnAct(nn.Module):

    def __init__(self, in_channel, num_class, kernel, stride, dilation=1, bias=False, act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channel, num_class, kernel, stride, padding=dilation, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(num_class)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class ResidualUBlock7(nn.Module):
    
    def __init__(self, in_channel, mid_channel, num_class) -> None:
        super(ResidualUBlock7, self).__init__()

        self.input_conv = ConvBnAct(in_channel, num_class, 3, 1, 1)

        self.encoder1_conv = ConvBnAct(num_class, mid_channel, 3, 1, 1)
        self.encoder1_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder2_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder2_pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        
        self.encoder3_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder3_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder4_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder4_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder5_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder5_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder6_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder7_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 2)


        self.decoder6_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder5_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder4_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder3_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder2_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder1_conv = ConvBnAct(2 * mid_channel, num_class, 3, 1, 1)

        self.upsample = Upsample(factor=2)
        self.concat = Concat()

    def forward(self, x):
        """
        Inputs:
            x: (bs, c, h, w)
        Outputs:
            y: (bs, out, h, w)

        """
        xin = self.input_conv(x)                           # (bs, c, h, w)         -> (bs, out, h, w)

        encoder1_feat = self.encoder1_conv(xin)            # (bs, out, h, w)       -> (bs, mid, h, w)
        x = self.encoder1_pool(encoder1_feat)              # (bs, mid, h, w)       -> (bs, mid, h/2, w/2)

        encoder2_feat = self.encoder2_conv(x)              # (bs, mid, h/2, w/2)   -> (bs, mid, h/2, w/2)
        x = self.encoder2_pool(encoder2_feat)              # (bs, mid, h/2, w/2)   -> (bs, mid, h/4, w/4)
        
        encoder3_feat = self.encoder3_conv(x)              # (bs, mid, h/4, w/4)   -> (bs, mid, h/4, w/4)
        x = self.encoder3_pool(encoder3_feat)              # (bs, mid, h/4, w/4)   -> (bs, mid, h/8, w/8)

        encoder4_feat = self.encoder4_conv(x)              # (bs, mid, h/8, w/8)   -> (bs, mid, h/8, w/8)
        x = self.encoder4_pool(encoder4_feat)              # (bs, mid, h/8, w/8)   -> (bs, mid, h/16, w/16)
        
        encoder5_feat = self.encoder5_conv(x)              # (bs, mid, h/16, w/16) -> (bs, mid, h/16, w/16)
        x = self.encoder5_pool(encoder5_feat)              # (bs, mid, h/16, w/16) -> (bs, mid, h/32, w/32)

        encoder6_feat = self.encoder6_conv(x)              # (bs, mid, h/32, w/32) -> (bs, mid, h/32, w/32)
        encoder7_feat = self.encoder7_conv(encoder6_feat)  # (bs, mid, h/32, w/32) -> (bs, mid, h/32, w/32)

        # ============== decoder ==================
        x = self.decoder6_conv(self.concat([encoder7_feat, encoder6_feat]))  # (bs, out, h/32, w/32) -> (bs, mid, h/32, w/32)
        x = self.upsample(x)                                                 # (bs, mid, h/32, w/32) -> (bs, mid, h/16, w/16)

        x = self.decoder5_conv(self.concat([x, encoder5_feat]))              # (bs, out, h/16, w/16) -> (bs, mid, h/16, w/16)
        x = self.upsample(x)                                                 # (bs, mid, h/16, w/16) -> (bs, mid, h/8, w/8)
        
        x = self.decoder4_conv(self.concat([x, encoder4_feat]))              # (bs, out, h/8, w/8)   -> (bs, mid, h/8, w/8)
        x = self.upsample(x)                                                 # (bs, mid, h/8, w/8)   -> (bs, mid, h/4, w/4)

        x = self.decoder3_conv(self.concat([x, encoder3_feat]))              # (bs, out, h/4, w/4)   -> (bs, mid, h/4, w/4)
        x = self.upsample(x)                                                 # (bs, mid, h/4, w/4)   -> (bs, mid, h/2, w/2)

        x = self.decoder2_conv(self.concat([x, encoder2_feat]))              # (bs, out, h/2, w/2)   -> (bs, mid, h/2, w/2)
        x = self.upsample(x)                                                 # (bs, mid, h/2, w/2)   -> (bs, mid, h, w)

        x = self.decoder1_conv(self.concat([x, encoder1_feat]))              # (bs, out, h, w)       -> (bs, out, h, w)
        return x + xin


class ResidualUBlock6(nn.Module):
    
    def __init__(self, in_channel, mid_channel, num_class) -> None:
        super(ResidualUBlock6, self).__init__()

        self.input_conv = ConvBnAct(in_channel, num_class, 3, 1, 1)

        self.encoder1_conv = ConvBnAct(num_class, mid_channel, 3, 1, 1)
        self.encoder1_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder2_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder2_pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        
        self.encoder3_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder3_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder4_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder4_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder5_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder6_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 2)


        self.decoder5_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder4_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder3_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder2_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder1_conv = ConvBnAct(2 * mid_channel, num_class, 3, 1, 1)

        self.upsample = Upsample(factor=2)
        self.concat = Concat()

    def forward(self, x):
        """
        Inputs:
            x: (bs, c, h, w)
        Outputs:
            y: (bs, out, h, w)
        """
        xin = self.input_conv(x)                           # (bs, c, h, w)         -> (bs, out, h, w)

        encoder1_feat = self.encoder1_conv(xin)            # (bs, out, h, w)       -> (bs, mid, h, w)
        x = self.encoder1_pool(encoder1_feat)              # (bs, mid, h, w)       -> (bs, mid, h/2, w/2)

        encoder2_feat = self.encoder2_conv(x)              # (bs, mid, h/2, w/2)   -> (bs, mid, h/2, w/2)
        x = self.encoder2_pool(encoder2_feat)              # (bs, mid, h/2, w/2)   -> (bs, mid, h/4, w/4)
        
        encoder3_feat = self.encoder3_conv(x)              # (bs, mid, h/4, w/4)   -> (bs, mid, h/4, w/4)
        x = self.encoder3_pool(encoder3_feat)              # (bs, mid, h/4, w/4)   -> (bs, mid, h/8, w/8)

        encoder4_feat = self.encoder4_conv(x)              # (bs, mid, h/8, w/8)   -> (bs, mid, h/8, w/8)
        x = self.encoder4_pool(encoder4_feat)              # (bs, mid, h/8, w/8)   -> (bs, mid, h/16, w/16)

        encoder5_feat = self.encoder5_conv(x)              # (bs, mid, h/16, w/16) -> (bs, mid, h/16, w/16)
        encoder6_feat = self.encoder6_conv(encoder5_feat)  # (bs, mid, h/16, w/16) -> (bs, mid, h/16, w/16)

        # ============== decoder ==============
        x = self.decoder5_conv(self.concat([encoder6_feat, encoder5_feat]))  # (bs, out, h/16, w/16) -> (bs, mid, h/16, w/16)
        x = self.upsample(x)                                                 # (bs, mid, h/16, w/16) -> (bs, mid, h/8, w/8)
        
        x = self.decoder4_conv(self.concat([x, encoder4_feat]))              # (bs, out, h/8, w/8)   -> (bs, mid, h/8, w/8)
        x = self.upsample(x)                                                 # (bs, mid, h/8, w/8)   -> (bs, mid, h/4, w/4)

        x = self.decoder3_conv(self.concat([x, encoder3_feat]))              # (bs, out, h/4, w/4)   -> (bs, mid, h/4, w/4)
        x = self.upsample(x)                                                 # (bs, mid, h/4, w/4)   -> (bs, mid, h/2, w/2)

        x = self.decoder2_conv(self.concat([x, encoder2_feat]))              # (bs, out, h/2, w/2)   -> (bs, mid, h/2, w/2)
        x = self.upsample(x)                                                 # (bs, mid, h/2, w/2)   -> (bs, mid, h, w)

        x = self.decoder1_conv(self.concat([x, encoder1_feat]))              # (bs, out, h, w)       -> (bs, out, h, w)
        return x + xin


class ResidualUBlock5(nn.Module):
    
    def __init__(self, in_channel, mid_channel, num_class) -> None:
        super(ResidualUBlock5, self).__init__()

        self.input_conv = ConvBnAct(in_channel, num_class, 3, 1, 1)

        self.encoder1_conv = ConvBnAct(num_class, mid_channel, 3, 1, 1)
        self.encoder1_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder2_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder2_pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        
        self.encoder3_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder3_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder4_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder5_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 2)

        self.decoder4_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder3_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder2_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder1_conv = ConvBnAct(2 * mid_channel, num_class, 3, 1, 1)

        self.upsample = Upsample(factor=2)
        self.concat = Concat()

    def forward(self, x):
        """
        Inputs:
            x: (bs, c, h, w)
        Outputs:
            y: (bs, out, h, w)

        """
        xin = self.input_conv(x)                           # (bs, c, h, w)         -> (bs, out, h, w)

        encoder1_feat = self.encoder1_conv(xin)            # (bs, out, h, w)       -> (bs, mid, h, w)
        x = self.encoder1_pool(encoder1_feat)              # (bs, mid, h, w)       -> (bs, mid, h/2, w/2)

        encoder2_feat = self.encoder2_conv(x)              # (bs, mid, h/2, w/2)   -> (bs, mid, h/2, w/2)
        x = self.encoder2_pool(encoder2_feat)              # (bs, mid, h/2, w/2)   -> (bs, mid, h/4, w/4)
        
        encoder3_feat = self.encoder3_conv(x)              # (bs, mid, h/4, w/4)   -> (bs, mid, h/4, w/4)
        x = self.encoder3_pool(encoder3_feat)              # (bs, mid, h/4, w/4)   -> (bs, mid, h/8, w/8)

        encoder4_feat = self.encoder4_conv(x)              # (bs, mid, h/8, w/8) -> (bs, mid, h/8, w/8)
        encoder5_feat = self.encoder5_conv(encoder4_feat)  # (bs, mid, h/8, w/8) -> (bs, mid, h/8, w/8)

        # ============== decoder ==============
        x = self.decoder4_conv(self.concat([encoder5_feat, encoder4_feat]))  # (bs, out, h/8, w/8)   -> (bs, mid, h/8, w/8)
        x = self.upsample(x)                                                 # (bs, mid, h/8, w/8)   -> (bs, mid, h/4, w/4)

        x = self.decoder3_conv(self.concat([x, encoder3_feat]))              # (bs, out, h/4, w/4)   -> (bs, mid, h/4, w/4)
        x = self.upsample(x)                                                 # (bs, mid, h/4, w/4)   -> (bs, mid, h/2, w/2)

        x = self.decoder2_conv(self.concat([x, encoder2_feat]))              # (bs, out, h/2, w/2)   -> (bs, mid, h/2, w/2)
        x = self.upsample(x)                                                 # (bs, mid, h/2, w/2)   -> (bs, mid, h, w)

        x = self.decoder1_conv(self.concat([x, encoder1_feat]))              # (bs, out, h, w)       -> (bs, out, h, w)
        return x + xin


class ResidualUBlock4(nn.Module):
    
    def __init__(self, in_channel, mid_channel, num_class) -> None:
        super(ResidualUBlock4, self).__init__()

        self.input_conv = ConvBnAct(in_channel, num_class, 3, 1, 1)

        self.encoder1_conv = ConvBnAct(num_class, mid_channel, 3, 1, 1)
        self.encoder1_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder2_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder2_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder3_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 1)
        self.encoder4_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 2)


        self.decoder3_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder2_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 1)
        self.decoder1_conv = ConvBnAct(2 * mid_channel, num_class, 3, 1, 1)

        self.upsample = Upsample(factor=2)
        self.concat = Concat()

    def forward(self, x):
        """
        Inputs:
            x: (bs, c, h, w)
        Outputs:
            y: (bs, out, h, w)
        """
        xin = self.input_conv(x)                           # (bs, c, h, w)         -> (bs, out, h, w)

        encoder1_feat = self.encoder1_conv(xin)            # (bs, out, h, w)       -> (bs, mid, h, w)
        x = self.encoder1_pool(encoder1_feat)              # (bs, mid, h, w)       -> (bs, mid, h/2, w/2)

        encoder2_feat = self.encoder2_conv(x)              # (bs, mid, h/2, w/2)   -> (bs, mid, h/2, w/2)
        x = self.encoder2_pool(encoder2_feat)              # (bs, mid, h/2, w/2)   -> (bs, mid, h/4, w/4)

        encoder3_feat = self.encoder3_conv(x)              # (bs, mid, h/4, w/4) -> (bs, mid, h/4, w/4)
        encoder4_feat = self.encoder4_conv(encoder3_feat)  # (bs, mid, h/4, w/4) -> (bs, mid, h/4, w/4)

        # ============== decoder ==============

        x = self.decoder3_conv(self.concat([encoder4_feat, encoder3_feat]))  # (bs, out, h/4, w/4)   -> (bs, mid, h/4, w/4)
        x = self.upsample(x)                                                 # (bs, mid, h/4, w/4)   -> (bs, mid, h/2, w/2)

        x = self.decoder2_conv(self.concat([x, encoder2_feat]))              # (bs, out, h/2, w/2)   -> (bs, mid, h/2, w/2)
        x = self.upsample(x)                                                 # (bs, mid, h/2, w/2)   -> (bs, mid, h, w)

        x = self.decoder1_conv(self.concat([x, encoder1_feat]))              # (bs, out, h, w)       -> (bs, out, h, w)
        return x + xin


class ResidualUBlock4F(nn.Module):
    
    def __init__(self, in_channel, mid_channel, num_class) -> None:
        super(ResidualUBlock4F, self).__init__()

        self.input_conv = ConvBnAct(in_channel, num_class, 3, 1, 1)

        self.encoder1_conv = ConvBnAct(num_class, mid_channel, 3, 1, 1)
        self.encoder2_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 2)
        self.encoder3_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 4)
        self.encoder4_conv = ConvBnAct(mid_channel, mid_channel, 3, 1, 8)

        self.decoder3_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 4)
        self.decoder2_conv = ConvBnAct(2 * mid_channel, mid_channel, 3, 1, 2)
        self.decoder1_conv = ConvBnAct(2 * mid_channel, num_class, 3, 1, 1)

        self.concat = Concat()

    def forward(self, x):
        """
        全程不对输入特征图的分辨率进行下采样
        Inputs:
            x: (bs, c, h, w)
        Outputs:
            y: (bs, out, h, w)
        """
        xin = self.input_conv(x)                           # (bs, c, h, w)   -> (bs, out, h, w)

        encoder1_feat = self.encoder1_conv(xin)            # (bs, c, h, w)   -> (bs, mid, h, w)
        encoder2_feat = self.encoder2_conv(encoder1_feat)  # (bs, mid, h, w) -> (bs, mid, h, w)
        encoder3_feat = self.encoder3_conv(encoder2_feat)  # (bs, mid, h, w) -> (bs, mid, h, w)
        encoder4_feat = self.encoder4_conv(encoder3_feat)  # (bs, mid, h, w) -> (bs, mid, h, w)

        # ============== decoder ==============

        x = self.decoder3_conv(self.concat([encoder4_feat, encoder3_feat]))  # (bs, out, h, w) -> (bs, mid, h, w)
        x = self.decoder2_conv(self.concat([x, encoder2_feat]))              # (bs, out, h, w) -> (bs, mid, h, w)
        x = self.decoder1_conv(self.concat([x, encoder1_feat]))              # (bs, out, h, w) -> (bs, out, h, w)
        return x + xin

    
class USquareNetTiny(nn.Module):

    def __init__(self, in_channel=3, num_class=1, scale=1.0):
        super(USquareNetTiny, self).__init__()
        self.encoder_stage1_conv = ResidualUBlock7(in_channel     , int(16*scale),  int(64*scale))
        self.encoder_stage1_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder_stage2_conv = ResidualUBlock6(int(64*scale) , int(16*scale),   int(64*scale))
        self.encoder_stage2_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder_stage3_conv = ResidualUBlock5(int(64*scale) , int(16*scale),  int(64*scale))
        self.encoder_stage3_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder_stage4_conv = ResidualUBlock4(int(64*scale) , int(16*scale), int(64*scale))
        self.encoder_stage4_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder_stage5_conv = ResidualUBlock4F(int(64*scale), int(16*scale), int(64*scale))
        self.encoder_stage5_pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.encoder_stage6_conv = ResidualUBlock4F(int(64*scale), int(16*scale), int(64*scale))

        # ===================== decoder =====================
        self.decoder_stage5_conv = ResidualUBlock4F(int(128*scale), int(16*scale), int(64*scale))
        self.decoder_stage4_conv = ResidualUBlock4(int(128*scale) , int(16*scale), int(64*scale))
        self.decoder_stage3_conv = ResidualUBlock5(int(128*scale)  , int(16*scale) , int(64*scale))
        self.decoder_stage2_conv = ResidualUBlock6(int(128*scale)  , int(16*scale) , int(64*scale))
        self.decoder_stage1_conv = ResidualUBlock7(int(128*scale)  , int(16*scale) , int(64*scale))

        self.out_stage1_conv = nn.Conv2d(int(64*scale), num_class, 3, 1, 1)
        self.out_stage2_conv = nn.Conv2d(int(64*scale), num_class, 3, 1, 1)
        self.out_stage3_conv = nn.Conv2d(int(64*scale), num_class, 3, 1, 1)
        self.out_stage4_conv = nn.Conv2d(int(64*scale), num_class, 3, 1, 1)
        self.out_stage5_conv = nn.Conv2d(int(64*scale), num_class, 3, 1, 1)
        self.out_stage6_conv = nn.Conv2d(int(64*scale), num_class, 3, 1, 1)
        self.out_conv = nn.Conv2d(6*num_class, num_class, 1)

        self.concat = Concat()
        self.upsample = Upsample(2)

    def forward(self, x):
        """
        Inputs:
            x: (bs, c, h, w)
        Ouptuts:

        """
        encoder1_feat = self.encoder_stage1_conv(x)  # (bs, c, h, w) -> (bs, 64, h, w)
        x = self.encoder_stage1_pool(encoder1_feat)  # (bs, 64, h, w) -> (bs, 64, h/2, w/2)

        encoder2_feat = self.encoder_stage2_conv(x)  # (bs, 64, h/2, w/2) -> (bs, 128, h/2, w/2)
        x = self.encoder_stage2_pool(encoder2_feat)  # (bs, 128, h/2, w/2) -> (bs, 128, h/4, w/4)

        encoder3_feat = self.encoder_stage3_conv(x)  # (bs, 128, h/4, w/4) -> (bs, 256, h/4, w/4)
        x = self.encoder_stage3_pool(encoder3_feat)  # (bs, 256, h/4, w/4) -> (bs, 256, h/8, w/8)

        encoder4_feat = self.encoder_stage4_conv(x)  # (bs, 256, h/8, w/8) -> (bs, 512, h/8, w/8)
        x = self.encoder_stage4_pool(encoder4_feat)  # (bs, 512, h/8, w/8) -> (bs, 512, h/16, w/16)

        encoder5_feat = self.encoder_stage5_conv(x)  # (bs, 512, h/16, w/16) -> (bs, 512, h/16, w/16)
        x = self.encoder_stage5_pool(encoder5_feat)  # (bs, 512, h/16, w/16) -> (bs, 512, h/32, w/32)

        encoder6_feat = self.encoder_stage6_conv(x)  # (bs, 512, h/32, w/32) -> (bs, 512, h/32, w/32)
        x = self.upsample(encoder6_feat)             # (bs, 512, h/32, w/32) -> (bs, 512, h/16, w/16)

        # ================== decoder ==================
        decoder5_feat = self.decoder_stage5_conv(self.concat([x, encoder5_feat]))  # (bs, 1024, h/16, w/16) -> (bs, 512, h/16, w/16)
        x = self.upsample(decoder5_feat)                                           # (bs, 512, h/16, w/16)  -> (bs, 512, h/8, w/8)

        decoder4_feat = self.decoder_stage4_conv(self.concat([x, encoder4_feat]))  # (bs, 1024, h/8, w/8) -> (bs, 256, h/8, w/8)
        x = self.upsample(decoder4_feat)                                           # (bs, 256, h/8, w/8)  -> (bs, 256, h/4, w/4)
        
        decoder3_feat = self.decoder_stage3_conv(self.concat([x, encoder3_feat]))  # (bs, 512, h/8, w/8)  -> (bs, 128, h/4, w/4)
        x = self.upsample(decoder3_feat)                                           # (bs, 128, h/4, w/4)  -> (bs, 128, h/2, w/2)

        decoder2_feat = self.decoder_stage2_conv(self.concat([x, encoder2_feat]))  # (bs, 256, h/2, w/2)  -> (bs, 64, h/2, w/2)
        x = self.upsample(decoder2_feat)                                           # (bs, 64, h/2, w/2)  -> (bs, 64, h, w)

        decoder1_feat = self.decoder_stage1_conv(self.concat([x, encoder1_feat]))  # (bs, 128, h, w)  -> (bs, 64, h, w)

        # ================== output ==================
        out_stage1 = self.out_stage1_conv(decoder1_feat)        # (bs, 64, h, w) -> (bs, out, h, w)

        out_stage2 = self.out_stage2_conv(decoder2_feat)        # (bs, 64, h/2, w/2) -> (bs, out, h/2, w/2)
        out_stage2 = F.interpolate(out_stage2, scale_factor=2)  # (bs, out, h/2, w/2) -> (bs, out, h, w)

        out_stage3 = self.out_stage3_conv(decoder3_feat)        # (bs, 128, h/4, w/4) -> (bs, out, h/4, w/4)
        out_stage3 = F.interpolate(out_stage3, scale_factor=4)  # (bs, out, h/4, w/4) -> (bs, out, h, w)

        out_stage4 = self.out_stage4_conv(decoder4_feat)        # (bs, 256, h/8, w/8) -> (bs, out, h/8, w/8)
        out_stage4 = F.interpolate(out_stage4, scale_factor=8)  # (bs, out, h/8, w/8) -> (bs, out, h, w)

        out_stage5 = self.out_stage5_conv(decoder5_feat)        # (bs, 512, h/16, w/16) -> (bs, out, h/16, w/16)
        out_stage5 = F.interpolate(out_stage5, scale_factor=16) # (bs, out, h/16, w/16) -> (bs, out, h, w)

        out_stage6 = self.out_stage6_conv(encoder6_feat)        # (bs, 512, h/32, w/32) -> (bs, out, h/32, w/32)
        out_stage6 = F.interpolate(out_stage6, scale_factor=32) # (bs, out, h/32, w/32) -> (bs, out, h, w)

        out_concat = self.out_conv(self.concat([out_stage1, out_stage2, out_stage3, out_stage4, out_stage5, out_stage6]))  # (bs, out*6, h, w) -> (bs, out, h, w)

        out_dict = {"stage1": out_stage1,   # (bs, out, h, w)
                    "stage2": out_stage2,   # (bs, out, h, w)
                    "stage3": out_stage3,   # (bs, out, h, w) 
                    "stage4": out_stage4,   # (bs, out, h, w) 
                    "stage5": out_stage5,   # (bs, out, h, w) 
                    "stage6": out_stage6,   # (bs, out, h, w) 
                    "fuse": out_concat,   # (bs, out, h, w)
                    }
        return out_dict




