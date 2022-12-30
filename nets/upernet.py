import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["UPerNet"]


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

    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation=1, bias=False, act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(PyramidPoolingModule, self).__init__()
        inter_channels = in_channels //4
        self.cba1 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba2 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba3 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba4 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.out  = ConvBnAct(in_channels * 2, out_channels, 1, 1, 0)

    def pool(self, x, size):
        return nn.AdaptiveAvgPool2d(size)(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode="bilinear", align_corners=True)

    def forward(self, x):
        size = x.shape[2:]
        f1 = self.upsample(self.cba1(self.pool(x, 1)), size)
        f2 = self.upsample(self.cba2(self.pool(x, 2)), size)
        f3 = self.upsample(self.cba3(self.pool(x, 3)), size)
        f4 = self.upsample(self.cba4(self.pool(x, 6)), size)
        f = torch.cat([x, f1, f2, f3, f4], dim=1)
        return self.out(f)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride, downsample):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvBnAct(inplanes, planes, 1, 1, 0, act=True)
        self.conv2 = ConvBnAct(planes, planes, 3, stride, 1, act=True)
        self.conv3 = ConvBnAct(planes, planes*self.expansion, 1, 1, 0, act=False)
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)  # conv with 1x1 kernel and stride 2
        out = out + residual
        return self.act(out)



class ResidualNet(nn.Module):
    
    def __init__(self, in_channel, block, layers) -> None:
        super(ResidualNet, self).__init__()
        self.stem = nn.Sequential(ConvBnAct(in_channel, out_channel=64, kernel=3, stride=2, padding=1, dilation=1),   # /2; c64
                                  ConvBnAct(64, 64, 1, 1, 0),                                                         # c64
                                  ConvBnAct(64, 128, 1, 1, 0),                                                        # c128
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                                    # /2; c128
                                  )
        self.inplanes = 128

        self.layer1 = self.make_layer(block, 64 , layers[0])             # c256
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)  # /2; c512
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)  # /2; c1024
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)  # /2; c2048


    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = ConvBnAct(self.inplanes, planes*block.expansion, 1, stride, 0, act=False)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Inputs:
            x: (bs, c, h, w)
        Outputs:
            y: (bs, out, h, w)

        """
        out_dict = {}
        x = self.stem(x)
        x = self.layer1(x)
        out_dict['resnet_layer1'] = x  # c256
        x = self.layer2(x)
        out_dict['resnet_layer2'] = x  # c512
        x = self.layer3(x)
        out_dict['resnet_layer3'] = x  # c1024  
        x = self.layer4(x)
        out_dict['resnet_layer4'] = x  # c2048
        return out_dict



class FeaturePyramidNet(nn.Module):

    def __init__(self, fpn_dim=256):
        self.fpn_dim = fpn_dim
        super(FeaturePyramidNet, self).__init__()
        self.fpn_in = nn.ModuleDict({'fpn_layer1': ConvBnAct(256 , self.fpn_dim, 1, 1, 0), 
                                     "fpn_layer2": ConvBnAct(512 , self.fpn_dim, 1, 1, 0), 
                                     "fpn_layer3": ConvBnAct(1024, self.fpn_dim, 1, 1, 0), 
                                    })
        self.fpn_out = nn.ModuleDict({'fpn_layer1': ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1), 
                                      "fpn_layer2": ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1), 
                                      "fpn_layer3": ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1), 
                                      })

    def forward(self, pyramid_features):
        """
        
        """
        fpn_out = {}
        
        f = pyramid_features['resnet_layer4']
        fpn_out['fpn_layer4'] = f
        x = self.fpn_in['fpn_layer3'](pyramid_features['resnet_layer3'])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer3'] = self.fpn_out['fpn_layer3'](f)

        x = self.fpn_in['fpn_layer2'](pyramid_features['resnet_layer2'])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer2'] = self.fpn_out['fpn_layer2'](f)

        x = self.fpn_in['fpn_layer1'](pyramid_features['resnet_layer1'])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer1'] = self.fpn_out['fpn_layer1'](f)

        return fpn_out


class UPerNet(nn.Module):

    def __init__(self, in_channel=3, layers=[3, 4, 6, 3], num_class=20, fpn_dim=256):
        super(UPerNet, self).__init__()
        self.fpn_dim = fpn_dim
        self.backbone = ResidualNet(in_channel, Bottleneck, layers)
        self.ppm = PyramidPoolingModule(2048, self.fpn_dim)
        self.fpn = FeaturePyramidNet(self.fpn_dim)
        self.fuse = ConvBnAct(fpn_dim*4, fpn_dim, 1, 1, 0)
        self.seg = nn.Sequential(ConvBnAct(fpn_dim, fpn_dim, 1, 1, 0), nn.Conv2d(fpn_dim, num_class, 1, 1, 0, bias=True))
        self.out = nn.Conv2d(num_class, num_class, 3, 1, 1)

    def forward(self, x):
        seg_size = x.shape[2:]
        resnet_features = self.backbone(x)
        ppm = self.ppm(resnet_features['resnet_layer4'])
        resnet_features.update({'resnet_layer4': ppm})
        fpn = self.fpn(resnet_features)
        out_size = fpn['fpn_layer1'].shape[2:]
        list_f = []
        list_f.append(fpn['fpn_layer1'])
        list_f.append(F.interpolate(fpn['fpn_layer2'], out_size, mode='bilinear', align_corners=False))
        list_f.append(F.interpolate(fpn['fpn_layer3'], out_size, mode='bilinear', align_corners=False))
        list_f.append(F.interpolate(fpn['fpn_layer4'], out_size, mode='bilinear', align_corners=False))
        x = self.seg(self.fuse(torch.cat(list_f, dim=1)))
        pred = self.out(F.interpolate(x, seg_size, mode='bilinear', align_corners=False))
        
        return {'fuse': pred}
        
        
        

if __name__ == "__main__":
    img = torch.rand(8, 3, 448, 448).float()
    net = UPerNet(3, [3, 4, 6, 3])
    out = net(img)
    print(f"{out['fuse'].shape}")

