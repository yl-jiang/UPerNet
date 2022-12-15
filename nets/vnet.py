import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['VNet']

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


class ConvTransposeGnDropAct(nn.Module):

    def __init__(self, in_channel, out_channel, kernel=2, stride=2, act=True, drop_prob=0.2):
        super(ConvTransposeGnDropAct, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride)
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



class VNet(nn.Module):
    def __init__(self, in_channel, num_class, layers=[2, 3, 3, 3], base_features=16, scale=1.0) -> None:
        super(VNet, self).__init__()
        # ------------------------------------------- encoder layers -------------------------------------------
        self.stem_conv1 = ConvGnDropAct(in_channel, int(base_features * scale))
        self.stem_conv2 = ConvGnDropAct(in_channel, int(base_features * scale))

        self.encoder_stage1_conv1 = ConvGnDropAct(int(base_features*scale), int(base_features*scale*2), 3, 2, 1)
        self.encoder_stage1_conv2 = nn.Sequential(*[ConvGnDropAct(int(base_features*scale*2), int(base_features*scale*2)) for _ in range(layers[0])])

        self.encoder_stage2_conv1 = ConvGnDropAct(int(base_features*scale*2), int(base_features*scale*4), 3, 2, 1)
        self.encoder_stage2_conv2 = nn.Sequential(*[ConvGnDropAct(int(base_features*scale*4), int(base_features*scale*4)) for _ in range(layers[1])])

        self.encoder_stage3_conv1 = ConvGnDropAct(int(base_features*scale*4), int(base_features*scale*8), 3, 2, 1)
        self.encoder_stage3_conv2 = nn.Sequential(*[ConvGnDropAct(int(base_features*scale*8), int(base_features*scale*8)) for _ in range(layers[2])])

        self.encoder_stage4_conv1 = ConvGnDropAct(int(base_features*scale*8), int(base_features*scale*16), 3, 2, 1)
        self.encoder_stage4_conv2 = nn.Sequential(*[ConvGnDropAct(int(base_features*scale*16), int(base_features*scale*16)) for _ in range(layers[3])])

        # ------------------------------------------- decoder layers -------------------------------------------
        self.decoder_stage4_upsample = ConvTransposeGnDropAct(int(base_features*scale*16), int(base_features*scale*8), 2, 2)
        self.decoder_stage4_conv1    = ConvGnDropAct(int(base_features*scale*8)*2, int(base_features*scale*8))
        self.decoder_stage4_conv2    = nn.Sequential(*[ConvGnDropAct(int(base_features*scale*8), int(base_features*scale*8)) for _ in range(layers[-1])])

        self.decoder_stage3_upsample = ConvTransposeGnDropAct(int(base_features*scale*8), int(base_features*scale*4), 2, 2)
        self.decoder_stage3_conv1    = ConvGnDropAct(int(base_features*scale*4)*2, int(base_features*scale*4))
        self.decoder_stage3_conv2    = nn.Sequential(*[ConvGnDropAct(int(base_features*scale*4), int(base_features*scale*4)) for _ in range(layers[-2])])

        self.decoder_stage2_upsample = ConvTransposeGnDropAct(int(base_features*scale*4), int(base_features*scale*2), 2, 2)
        self.decoder_stage2_conv1    = ConvGnDropAct(int(base_features*scale*2)*2, int(base_features*scale*2))
        self.decoder_stage2_conv2    = nn.Sequential(*[ConvGnDropAct(int(base_features*scale*2), int(base_features*scale*2)) for _ in range(layers[-3])])

        self.decoder_stage1_upsample = ConvTransposeGnDropAct(int(base_features*scale*2), int(base_features*scale), 2, 2)
        self.decoder_stage1_conv1    = ConvGnDropAct(int(base_features*scale)*2, int(base_features*scale))
        self.decoder_stage1_conv2    = nn.Sequential(*[ConvGnDropAct(int(base_features*scale), int(base_features*scale)) for _ in range(layers[-4])])

        # ------------------------------------------- output layers -------------------------------------------
        self.out_conv = nn.Conv2d(int(base_features*scale*(1+2+4+8)), num_class, 1, 1, 0, bias=True)

    def forward(self, x):
        out_size = (x.size(2), x.size(3))
        stem_x = self.stem_conv1(x) + self.stem_conv2(x)

        encoder1_f = self.encoder_stage1_conv1(stem_x)
        encoder1_x = self.encoder_stage1_conv2(encoder1_f) + encoder1_f

        encoder2_f = self.encoder_stage2_conv1(encoder1_x)
        encoder2_x = self.encoder_stage2_conv2(encoder2_f) + encoder2_f
        
        encoder3_f = self.encoder_stage3_conv1(encoder2_x)
        encoder3_x = self.encoder_stage3_conv2(encoder3_f) + encoder3_f

        encoder4_f = self.encoder_stage4_conv1(encoder3_x)
        encoder4_x = self.encoder_stage4_conv2(encoder4_f) + encoder4_f

        decoder4_u = self.decoder_stage4_upsample(encoder4_x)
        decoder4_c = self.decoder_stage4_conv1(torch.cat((decoder4_u, encoder3_x), dim=1))
        decoder4_x = self.decoder_stage4_conv2(decoder4_c) + decoder4_u
        decoder4_o = F.interpolate(decoder4_x, size=out_size, mode='bilinear', align_corners=False)

        decoder3_u = self.decoder_stage3_upsample(decoder4_x)
        decoder3_c = self.decoder_stage3_conv1(torch.cat((decoder3_u, encoder2_x), dim=1))
        decoder3_x = self.decoder_stage3_conv2(decoder3_c) + decoder3_u
        decoder3_o = F.interpolate(decoder3_x, size=out_size, mode='bilinear', align_corners=False)

        decoder2_u = self.decoder_stage2_upsample(decoder3_x)
        decoder2_c = self.decoder_stage2_conv1(torch.cat((decoder2_u, encoder1_x), dim=1))
        decoder2_x = self.decoder_stage2_conv2(decoder2_c) + decoder2_u
        decoder2_o = F.interpolate(decoder2_x, size=out_size, mode='bilinear', align_corners=False)

        decoder1_u = self.decoder_stage1_upsample(decoder2_x)
        decoder1_c = self.decoder_stage1_conv1(torch.cat((decoder1_u, stem_x), dim=1))
        decoder1_x = self.decoder_stage1_conv2(decoder1_c) + decoder1_u

        out        = self.out_conv(torch.cat((decoder1_x, decoder2_o, decoder3_o, decoder4_o), dim=1))

        return {'fuse': out}


if __name__ == "__main__":
    with torch.no_grad():
        dummy = torch.rand(4, 3, 256, 256).float()
        net = VNet(3, 10)
        pred = net(dummy)
        for k, v in pred.items():
            print(k, v.shape)