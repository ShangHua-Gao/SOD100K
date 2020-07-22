import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from .gOctConv import gOctaveConv, gOctaveCBR
import os
affine_par = True



model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation_ = 1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, dilation= dilation_, padding=dilation_, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        for j in range(self.nums):
            for i in self.bns[j].parameters():
                i.requires_grad = False

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth = 26, scale = 4):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_pretrained_model(self, model):
        self.load_state_dict(model, strict=False)

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, 
                    ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample,
                            stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        tmp_x = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # tmp_x.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        tmp_x.append(x)
        x = self.layer2(x)
        tmp_x.append(x)
        x = self.layer3(x)
        tmp_x.append(x)
        x = self.layer4(x)
        tmp_x.append(x)

        return tmp_x

def res2net50_std():
    model = Res2Net(Bottleneck, [3, 4, 6, 3], baseWidth = 26, scale = 4)
    return model

class PallMSBlock(nn.Module):
    def __init__(self,in_channels, out_channels, alpha=[0.5,0.5], bias=False):
        super(PallMSBlock, self).__init__()
        self.std_conv = False
        self.convs = nn.ModuleList()

        for i in range(len(alpha)):
            self.convs.append(MSBlock(int(round(in_channels*alpha[i])), int(round(out_channels*alpha[i]))))
        self.outbranch = len(alpha)

    def forward(self, xset):
        if isinstance(xset,torch.Tensor):
            xset = [xset,]
        yset = []
        for i in range(self.outbranch):
            yset.append(self.convs[i](xset[i]))
        return yset


class MSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations = [1,2,4,8,16]):
        super(MSBlock,self).__init__()
        self.dilations = dilations
        each_out_channels = out_channels//5
        self.msconv = nn.ModuleList()
        for i in range(len(dilations)):
            if i != len(dilations)-1:
                this_outc = each_out_channels
            else:
                this_outc = out_channels - each_out_channels*(len(dilations)-1)
            self.msconv.append(nn.Conv2d(in_channels, this_outc,3, padding=dilations[i], dilation=dilations[i], bias=False))
        self.bn = nn.GroupNorm(32, out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        outs = []
        for i in range(len(self.dilations)):
            outs.append(self.msconv[i](x))
        out = torch.cat(outs, dim=1)
        del outs
        out = self.prelu(self.bn(out))
        return out


class CSFNet(nn.Module):
    def __init__(self, num_classes=1):
        super(CSFNet, self).__init__()
        self.base = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth = 26, scale = 4)
        # self.base.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))

        fuse_in_channel = 256+512+1024+2048
        fuse_in_split = [1/15,2/15,4/15,8/15]
        fuse_out_channel = 128+256+512+512
        fuse_out_split = [1/11,2/11,4/11,4/11]

        self.fuse = gOctaveCBR(fuse_in_channel, fuse_out_channel, kernel_size=(1,1), padding=0, 
                                alpha_in = fuse_in_split, alpha_out = fuse_out_split, stride = 1)
        self.ms = PallMSBlock(fuse_out_channel, fuse_out_channel, alpha = fuse_out_split)
        self.fuse1x1 = gOctaveCBR(fuse_out_channel, fuse_out_channel, kernel_size=(1, 1), padding=0, 
                                alpha_in = fuse_out_split, alpha_out = [1,], stride = 1)
        self.cls_layer = nn.Conv2d(fuse_out_channel, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.base(x)
        fuse = self.fuse(features)
        fuse = self.ms(fuse)
        fuse = self.fuse1x1(fuse)
        output = self.cls_layer(fuse[0])
        output = F.interpolate(output, x.size()[2:], mode='bilinear', align_corners=False)

        return output

def build_model():
    return CSFNet()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = build_model()
    model = model.cuda(0)
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.4fM' % (total / 1e6))
    print(model(images).size())
    print('Memory useage: %.4fM' % ( torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
