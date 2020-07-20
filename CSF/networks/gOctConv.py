import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from torch.nn import init
USE_BALANCE = False
up_kwargs = {'mode': 'bilinear'}
class gOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=[0.5,0.5], alpha_out=[0.5,0.5], stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = up_kwargs):
        super(gOctaveConv, self).__init__()
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.weights = nn.Parameter(torch.Tensor(out_channels, round(in_channels/self.groups), kernel_size[0], kernel_size[1]))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.up_kwargs = up_kwargs
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.alpha_in = [0]
        tmpsum = 0
        for i in range(len(alpha_in)):
            tmpsum += alpha_in[i]
            self.alpha_in.append(tmpsum)
        self.alpha_out = [0]
        tmpsum = 0
        for i in range(len(alpha_out)):
            tmpsum += alpha_out[i]
            self.alpha_out.append(tmpsum)
        self.inbranch = len(alpha_in)
        self.outbranch = len(alpha_out)
        global USE_BALANCE
        self.use_balance = USE_BALANCE
        if self.use_balance:
            self.bals = nn.Parameter(torch.Tensor(self.outbranch, out_channels))
            init.normal_(self.bals, mean=1.0, std=0.05)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    def forward(self, xset):
        yset = []
        ysets = []
        for j in range(self.outbranch):
            ysets.append([])
        if isinstance(xset,torch.Tensor):
            xset = [xset,]
        if USE_BALANCE:
            bals_norm = torch.abs(self.bals)/(torch.abs(self.bals).sum(dim=0)+1e-14)
        for i in range(self.inbranch):
            if xset[i] is None:
                continue
            if self.stride == 2:
                x = F.avg_pool2d(xset[i], (2,2), stride=2)
            else:
                x = xset[i]
            begin_x = int(round(self.in_channels*self.alpha_in[i]/self.groups))
            end_x = int(round(self.in_channels*self.alpha_in[i+1]/self.groups))
            if begin_x == end_x:
                continue
            for j in range(self.outbranch):
                begin_y = int(round(self.out_channels*self.alpha_out[j]))
                end_y = int(round(self.out_channels*self.alpha_out[j+1]))
                if begin_y == end_y:
                    continue
                scale_factor = 2**(i-j)
                this_output_shape = xset[j].shape[2:4]
                if self.bias is not None:
                    this_bias = self.bias[begin_y:end_y]
                else:
                    this_bias = None

                if self.use_balance:
                    this_weight = self.weights[begin_y:end_y, begin_x:end_x, :,:]
                    this_weight = this_weight*bals_norm[j,begin_y:end_y].view(this_weight.shape[0],1,1,1)
                else:
                    this_weight = self.weights[begin_y:end_y, begin_x:end_x, :,:]

                if scale_factor > 1:
                    y = F.conv2d(x, this_weight, this_bias, 1, self.padding, self.dilation, self.groups)
                    y = F.interpolate(y, size=this_output_shape, mode=up_kwargs['mode'])
                elif scale_factor < 1:
                    x_resize = F.interpolate(x, size=this_output_shape, mode=up_kwargs['mode'])
                    y = F.conv2d(x_resize, this_weight, this_bias, 1, self.padding, self.dilation, self.groups)
                else:
                    y = F.conv2d(x, this_weight, this_bias, 1, self.padding, self.dilation, self.groups)
                ysets[j].append(y)

        for j in range(self.outbranch):
            if len(ysets[j])!=0:
                yset.append(sum(ysets[j]))
            else:
                yset.append(None)
        del ysets
        return yset

class gOctaveCBR(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=(3,3),alpha_in=[0.5,0.5], alpha_out=[0.5,0.5], stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = up_kwargs):
        super(gOctaveCBR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.std_conv = False
        if len(alpha_in)==1 and len(alpha_out)==1:
            self.std_conv = True
            self.conv = Conv2dX100(in_channels,out_channels,kernel_size, stride, padding, dilation, groups, bias)
        else:
            self.conv = gOctaveConv(in_channels,out_channels,kernel_size, alpha_in,alpha_out, stride, padding, dilation, groups, bias, up_kwargs)
        
        self.bns = nn.ModuleList()
        self.prelus = nn.ModuleList()
        for i in range(len(alpha_out)):
            if int(round(out_channels*alpha_out[i]))!=0:
                self.bns.append(nn.GroupNorm(32, int(round(out_channels*alpha_out[i]))))
                self.prelus.append(nn.PReLU(int(round(out_channels*alpha_out[i]))))
            else:
                self.bns.append(None)
                self.prelus.append(None)
        self.outbranch = len(alpha_out)
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, xset):
        if self.std_conv:
            if isinstance(xset,torch.Tensor):
                xset = [xset,]
            xset = self.conv(xset[0])
            xset = self.prelus[0](self.bns[0](xset))
        else:
            xset = self.conv(xset)
            for i in range(self.outbranch):
                if xset[i] is not None:
                    xset[i] = self.prelus[i](self.bns[i](xset[i]))
        return xset