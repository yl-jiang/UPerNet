import torch.nn as nn
import torch
import torch.nn.functional as F


__all__ = ["UNet"]

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


class ConvGnDropAct(nn.Module):

    def __init__(self, in_channel, out_channel, kernel=3, stride=1, padding=1, dilation=1, bias=False, act=True, drop_prob=0.2):
        super(ConvGnDropAct, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=padding, dilation=dilation, bias=bias)
        self.gn = nn.GroupNorm(num_groups=8, num_channels=out_channel)
        self.dropout = nn.Dropout(p=drop_prob, inplace=True)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.dropout(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class UNet(nn.Module):
    def __init__(self, in_channel, num_class, base_features=16, scale=1.0) -> None:
        super(UNet, self).__init__()
        self.encoder_stage1_conv1 = ConvGnDropAct(in_channel, int(base_features*scale))
        self.encoder_stage1_conv2 = ConvGnDropAct(int(base_features*scale), int(base_features*scale))
        self.encoder_stage1_maxpool = nn.MaxPool2d(2, 2)

        self.encoder_stage2_conv1 = ConvGnDropAct(int(base_features*scale), int(base_features*scale*2))
        self.encoder_stage2_conv2 = ConvGnDropAct(int(base_features*scale*2), int(base_features*scale*2))
        self.encoder_stage2_maxpool = nn.MaxPool2d(2, 2)

        self.encoder_stage3_conv1 = ConvGnDropAct(int(base_features*scale*2), int(base_features*scale*4))
        self.encoder_stage3_conv2 = ConvGnDropAct(int(base_features*scale*4), int(base_features*scale*4))
        self.encoder_stage3_maxpool = nn.MaxPool2d(2, 2)

        self.encoder_stage4_conv1 = ConvGnDropAct(int(base_features*scale*4), int(base_features*scale*8))
        self.encoder_stage4_conv2 = ConvGnDropAct(int(base_features*scale*8), int(base_features*scale*8))
        self.encoder_stage4_maxpool = nn.MaxPool2d(2, 2)

        self.neck_conv1 = ConvGnDropAct(int(base_features*scale*8), int(base_features*scale*16))
        self.neck_conv2 = ConvGnDropAct(int(base_features*scale*16), int(base_features*scale*16))

        self.decoder_stage4_upsample = nn.ConvTranspose2d(int(base_features*scale*16), int(base_features*scale*8), 2, 2)
        self.decoder_stage4_conv1 = ConvGnDropAct(int(base_features*scale*8*2), int(base_features*scale*8))
        self.decoder_stage4_conv2 = ConvGnDropAct(int(base_features*scale*8), int(base_features*scale*8))

        self.decoder_stage3_upsample = nn.ConvTranspose2d(int(base_features*scale*8), int(base_features*scale*4), 2, 2)
        self.decoder_stage3_conv1 = ConvGnDropAct(int(base_features*scale*4*2), int(base_features*scale*4))
        self.decoder_stage3_conv2 = ConvGnDropAct(int(base_features*scale*4), int(base_features*scale*4))

        self.decoder_stage2_upsample = nn.ConvTranspose2d(int(base_features*scale*4), int(base_features*scale*2), 2, 2)
        self.decoder_stage2_conv1 = ConvGnDropAct(int(base_features*scale*2*2), int(base_features*scale*2))
        self.decoder_stage2_conv2 = ConvGnDropAct(int(base_features*scale*2), int(base_features*scale*2))

        self.decoder_stage1_upsample = nn.ConvTranspose2d(int(base_features*scale*2), int(base_features*scale), 2, 2)
        self.decoder_stage1_conv1 = ConvGnDropAct(int(base_features*scale*2), int(base_features*scale))
        self.decoder_stage1_conv2 = ConvGnDropAct(int(base_features*scale), int(base_features*scale))

        self.out_conv = nn.Conv2d(int(base_features*scale*(1+2+4+8)), out_channels=num_class, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out_size = (x.size(2), x.size(3))
        encoder1_x = self.encoder_stage1_conv2(self.encoder_stage1_conv1(x))
        encoder2_x = self.encoder_stage2_conv2(self.encoder_stage2_conv1(self.encoder_stage1_maxpool(encoder1_x)))
        encoder3_x = self.encoder_stage3_conv2(self.encoder_stage3_conv1(self.encoder_stage2_maxpool(encoder2_x)))
        encoder4_x = self.encoder_stage4_conv2(self.encoder_stage4_conv1(self.encoder_stage3_maxpool(encoder3_x)))

        neck_x     = self.neck_conv2(self.neck_conv1(self.encoder_stage4_maxpool(encoder4_x)))

        decoder4_x = self.decoder_stage4_upsample(neck_x)
        decoder4_x = torch.cat((decoder4_x, encoder4_x), dim=1)
        decoder4_x = self.decoder_stage4_conv2(self.decoder_stage4_conv1(decoder4_x))
        decoder4_s = F.interpolate(decoder4_x, size=out_size, mode='bilinear', align_corners=False)  # c: base_features*scale*8

        decoder3_x = self.decoder_stage3_upsample(decoder4_x)
        decoder3_x = torch.cat((decoder3_x, encoder3_x), dim=1)
        decoder3_x = self.decoder_stage3_conv2(self.decoder_stage3_conv1(decoder3_x))
        decoder3_s = F.interpolate(decoder3_x, size=out_size, mode='bilinear', align_corners=False)  # c: base_features*scale*4

        decoder2_x = self.decoder_stage2_upsample(decoder3_x)
        decoder2_x = torch.cat((decoder2_x, encoder2_x), dim=1)
        decoder2_x = self.decoder_stage2_conv2(self.decoder_stage2_conv1(decoder2_x))
        decoder2_s = F.interpolate(decoder2_x, size=out_size, mode='bilinear', align_corners=False)  # c: base_features*scale*2

        decoder1_x = self.decoder_stage1_upsample(decoder2_x)
        decoder1_x = torch.cat((decoder1_x, encoder1_x), dim=1)
        decoder1_x = self.decoder_stage1_conv2(self.decoder_stage1_conv1(decoder1_x))  # c: base_features*scale*1
        
        out        = self.out_conv(torch.cat((decoder1_x, decoder2_s, decoder3_s, decoder4_s), dim=1))
        return {'fuse': out}



if __name__ == "__main__":
    dummy = torch.rand(8, 3, 448, 448).float()
    net = UNet(3, 10)
    pred = net(dummy)
    for k, v in pred.items():
        print(k, v.shape)
