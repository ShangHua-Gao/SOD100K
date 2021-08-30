'''
'''
import math
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .conv2d import Conv2dX100


# Model define #
class ILBlock(nn.Module):
    def __init__(self,
                 inlist,
                 outlist,
                 stride=1,
                 nextstride=1,
                 nextoutlist=None,
                 first=False):
        super(ILBlock, self).__init__()
        ninput = int(round(sum(inlist)))
        noutput = int(round(sum(outlist)))
        alpha_in = inlist * 1.0 / ninput
        alpha_out = outlist * 1.0 / noutput
        alpha_in = alpha_in.tolist()
        alpha_out = alpha_out.tolist()
        self.first = first
        if self.first or stride == 2:
            self.conv1x1 = gOctaveCBR(ninput,
                                      noutput,
                                      kernel_size=(3, 3),
                                      padding=1,
                                      alpha_in=alpha_in,
                                      alpha_out=alpha_out,
                                      stride=stride)
        else:
            self.conv1x1 = gOctaveCBR(ninput,
                                      noutput,
                                      kernel_size=(1, 1),
                                      padding=0,
                                      alpha_in=alpha_in,
                                      alpha_out=alpha_out,
                                      stride=1)

        self.conv3x3_1 = SimplifiedGOctConvBR(noutput,
                                              noutput,
                                              stride=1,
                                              kernel_size=(3, 3),
                                              padding=1,
                                              alpha=alpha_out,
                                              groups=noutput)
        self.conv3x3_2 = SimplifiedGOctConvBR(noutput,
                                              noutput,
                                              stride=1,
                                              kernel_size=(3, 3),
                                              padding=1,
                                              alpha=alpha_out,
                                              groups=noutput)
        self.all_flops = 0
        self.stride = stride
        self.nextstride = nextstride
        self.nextoutlist = nextoutlist

        self.baseflop = None
        self.expandflop = None

    def forward(self, x):
        output = self.conv1x1(x)
        output = self.conv3x3_1(output)
        output = self.conv3x3_2(output)
        return output


class PallMSBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dil_channels,
                 alpha_in=[0.5, 0.5],
                 alpha_out=[0.5, 0.5],
                 bias=False,
                 norm_layer=nn.BatchNorm2d):
        super(PallMSBlock, self).__init__()
        self.std_conv = False
        self.convs = nn.ModuleList()

        for i in range(len(alpha_in)):
            if max(dil_channels[i]) != 0:
                self.convs.append(
                    MSBlock(int(round(in_channels * alpha_in[i])),
                            int(round(out_channels * alpha_out[i])),
                            dil_channels[i]))
            else:
                self.convs.append(None)
        self.outbranch = len(alpha_in)

    def forward(self, xset):
        if isinstance(xset, torch.Tensor):
            xset = [
                xset,
            ]
        yset = []
        for i in range(self.outbranch):
            if self.convs[i] is not None:
                yset.append(self.convs[i](xset[i]))
            else:
                yset.append(None)
        return yset


class MSBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dil_channels,
                 dilations=[1, 2, 4, 8, 16]):
        super(MSBlock, self).__init__()
        self.dilations = dilations
        each_out_channels = out_channels // len(dilations)
        self.msconv = nn.ModuleList()
        self.real_dil_branch = len(dilations)
        for i in range(len(dilations)):
            if dil_channels[i] != 0:
                self.msconv.append(
                    Conv2dX100(in_channels,
                               int(dil_channels[i]),
                               3,
                               padding=dilations[i],
                               dilation=dilations[i],
                               bias=False))
            else:
                self.msconv.append(None)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        outs = []
        for i in range(self.real_dil_branch):
            if self.msconv[i] != None:
                outs.append(self.msconv[i](x))
        out = torch.cat(outs, dim=1)
        del outs
        out = self.prelu(self.bn(out))
        return out


class CSFHead(nn.Module):
    def __init__(self, fuse_layer_config):
        super(CSFHead, self).__init__()
        index = 0
        self.layer_config = fuse_layer_config
        fuse_in_channel = int(round(sum(self.layer_config[index][0])))
        fuse_in_split = self.layer_config[index][0] * 1.0 / fuse_in_channel
        index = index + 1
        fuse_mid_in_channel = int(round(sum(self.layer_config[index][0])))
        fuse_mid_in_split = self.layer_config[index][
            0] * 1.0 / fuse_mid_in_channel
        fuse_mid_out_channel = int(round(sum(self.layer_config[index][1])))
        fuse_mid_out_split = self.layer_config[index][
            1] * 1.0 / fuse_mid_out_channel
        dils = self.layer_config[index][2]
        index = index + 1
        fuse_out_channel = int(round(sum(self.layer_config[index][1])))

        print(fuse_in_channel, fuse_in_split, fuse_in_channel * fuse_in_split)
        print(fuse_mid_in_channel, fuse_mid_in_split,
              fuse_mid_in_channel * fuse_mid_in_split)
        print(fuse_mid_out_channel, fuse_mid_out_split,
              fuse_mid_out_channel * fuse_mid_out_split)
        print(fuse_out_channel)
        fuse_in_split = fuse_in_split.tolist()
        fuse_mid_in_split = fuse_mid_in_split.tolist()
        fuse_mid_out_split = fuse_mid_out_split.tolist()

        self.fuse = gOctaveCBR(fuse_in_channel,
                               fuse_mid_in_channel,
                               kernel_size=(1, 1),
                               padding=0,
                               alpha_in=fuse_in_split,
                               alpha_out=fuse_mid_in_split,
                               stride=1)
        self.ms = PallMSBlock(fuse_mid_in_channel,
                              fuse_mid_out_channel,
                              alpha_in=fuse_mid_in_split,
                              alpha_out=fuse_mid_out_split,
                              dil_channels=dils)
        self.fuse1x1 = gOctaveCBR(fuse_mid_out_channel,
                                  fuse_out_channel,
                                  kernel_size=(1, 1),
                                  padding=0,
                                  alpha_in=fuse_mid_out_split,
                                  alpha_out=[
                                      1,
                                  ],
                                  stride=1)

    def forward(self, xset):
        fuse = self.fuse(xset)
        fuse = self.ms(fuse)
        fuse = self.fuse1x1(fuse)
        return fuse


class CSNet(nn.Module):
    def __init__(self, layer_config, num_classes=1):
        super(CSNet, self).__init__()

        self.stages = layer_config[-1]
        self.layer_config = layer_config
        fuse_in = np.zeros(3)
        index = 0
        print(self.layer_config)
        self.stage0 = nn.ModuleList()
        self.stage0.append(
            ILBlock(np.array([3]),
                    self.layer_config[index][1],
                    nextoutlist=self.layer_config[index + 1][1],
                    stride=1,
                    first=True))

        index = index + 1
        self.stage1 = nn.ModuleList()
        self.stage1.append(
            ILBlock(self.layer_config[index][0],
                    self.layer_config[index][1],
                    nextoutlist=self.layer_config[index + 1][1]))
        index = index + 1
        for i in range(1, self.stages[0]):
            if i == self.stages[0] - 1:
                nextstride = 2
            else:
                nextstride = 1
            self.stage1.append(
                ILBlock(self.layer_config[index][0],
                        self.layer_config[index][1],
                        nextoutlist=self.layer_config[index + 1][1],
                        nextstride=nextstride))
            index = index + 1

        self.stage2 = nn.ModuleList()
        self.stage2.append(
            ILBlock(self.layer_config[index][0],
                    self.layer_config[index][1],
                    nextoutlist=self.layer_config[index + 1][1],
                    stride=2))
        index = index + 1
        for i in range(1, self.stages[1]):
            if i == self.stages[1] - 1:
                nextstride = 2
            else:
                nextstride = 1
            self.stage2.append(
                ILBlock(self.layer_config[index][0],
                        self.layer_config[index][1],
                        nextoutlist=self.layer_config[index + 1][1],
                        nextstride=nextstride))
            index = index + 1
        fuse_in[0] = int(round(sum(self.layer_config[index - 1][1])))

        self.stage3 = nn.ModuleList()
        self.stage3.append(
            ILBlock(self.layer_config[index][0],
                    self.layer_config[index][1],
                    nextoutlist=self.layer_config[index + 1][1],
                    stride=2))
        index = index + 1
        for i in range(1, self.stages[2]):
            if i == self.stages[2] - 1:
                nextstride = 2
            else:
                nextstride = 1
            self.stage3.append(
                ILBlock(self.layer_config[index][0],
                        self.layer_config[index][1],
                        nextoutlist=self.layer_config[index + 1][1],
                        nextstride=nextstride))
            index = index + 1
        fuse_in[1] = int(round(sum(self.layer_config[index - 1][1])))

        self.stage4 = nn.ModuleList()
        self.stage4.append(
            ILBlock(self.layer_config[index][0],
                    self.layer_config[index][1],
                    nextoutlist=self.layer_config[index + 1][1],
                    stride=2))
        index = index + 1
        for i in range(1, self.stages[3]):
            if i == self.stages[3] - 1:
                nextstride = 0
            else:
                nextstride = 1
            self.stage4.append(
                ILBlock(self.layer_config[index][0],
                        self.layer_config[index][1],
                        nextoutlist=None))
            index = index + 1
        fuse_in[2] = int(round(sum(self.layer_config[index - 1][1])))

        self.oct_fuse = CSFHead(self.layer_config[index:index + 3])
        fuse_out_channel = int(round(sum(self.layer_config[-2][1])))
        self.cls_layer = nn.Conv2d(fuse_out_channel,
                                   num_classes,
                                   kernel_size=1)

        self.all_flops = 0
        self.batchsize = 0

    def set_batchsize(self, batchsize):
        self.batchsize = batchsize

    def clear_flops(self):
        self.all_flops = 0
        for m in self.modules():
            if isinstance(m, ILBlock):
                m.conv1x1.all_flops = 0
                m.conv3x3_1.all_flops = 0
                m.conv3x3_2.all_flops = 0

    def get_flops(self):
        for m in self.modules():
            if isinstance(m, ILBlock):
                self.all_flops = m.conv1x1.all_flops + m.conv3x3_1.all_flops + m.conv3x3_2.all_flops + self.all_flops
        # print("self.all_flops",self.all_flops/self.batchsize)
        # return self.all_flops
        return self.all_flops / self.batchsize

    def flops_hook(self, expandflop=2):
        baseflop = expandflop**(len(self.stages) - 1)
        stage = 0
        in_stage = 0
        print(self.stages)
        real_stages = self.stages.copy()
        real_stages[0] += 1  # add for stage0
        for m in self.modules():
            if isinstance(m, ILBlock):
                m.conv1x1.baseflop = baseflop
                m.conv1x1.expandflop = expandflop
                m.conv1x1.register_forward_hook(Oct_bn_hook)
                m.conv3x3_1.baseflop = baseflop
                m.conv3x3_1.expandflop = expandflop
                m.conv3x3_1.register_forward_hook(Oct_bn_hook)
                m.conv3x3_2.baseflop = baseflop
                m.conv3x3_2.expandflop = expandflop
                m.conv3x3_2.register_forward_hook(Oct_bn_hook)
                in_stage += 1
                if in_stage == real_stages[stage]:
                    baseflop /= expandflop
                    stage += 1
                    in_stage = 0
                print(m.conv1x1.baseflop, end=" ")

    def updateWeight(self, s=0.001):
        for m in self.modules():
            if isinstance(m, gOctaveCBR):
                for n in list(m.modules()):
                    if isinstance(n, nn.BatchNorm2d):
                        n.weight.grad.data.add_(
                            s * torch.sign(n.weight.data))  # L1

    def forward(self, x):

        x0 = self.stage0[0](x)
        x1 = x0
        for i in range(self.stages[0]):
            x1 = self.stage1[i](x1)
        x2 = x1
        for i in range(self.stages[1]):
            x2 = self.stage2[i](x2)
        x3 = x2
        for i in range(self.stages[2]):
            x3 = self.stage3[i](x3)
        x4 = x3
        for i in range(self.stages[3]):
            x4 = self.stage4[i](x4)
        fuse = self.oct_fuse([x2[0], x3[0], x4[0]])
        output = self.cls_layer(fuse[0])
        output = F.interpolate(output,
                               x.size()[2:],
                               mode='bilinear',
                               align_corners=False)

        return output


# hook for dynamic weight decay #
def Oct_bn_hook(module, input, output):
    all_weights = []
    branches = len(output)
    this_flop_weight = []
    init_flop = module.baseflop * (module.expandflop**(branches - 1))
    for k in range(branches):
        this_flop_weight.append(init_flop)
        init_flop /= module.expandflop
    gap_id = 0
    for name, m in module.named_modules():
        # print(name)
        if isinstance(m, nn.BatchNorm2d):
            gap_vet = torch.nn.functional.adaptive_avg_pool2d(
                output[gap_id].detach(), 1).squeeze().abs()
            gap_id += 1
            bn_id = int(name.split('.')[-1])
            all_weights.append((this_flop_weight[bn_id] * gap_vet *
                                torch.pow(m.weight, 2)).sum())  #l2 reg.
        # print(all_weights)
    module.all_flops += 0.5 * sum(all_weights)


# define config file for the network #
def init_layers(basewidth, basic_split=[
    1,
]):
    # [24/32,8/32]  [16/32,16/32]  [8/32,24/32]
    layer_config = []
    basic_split = [float(x) for x in basic_split]
    stages = [3, 4, 6, 4]
    layer_config.append([np.array([
        3,
    ]), basewidth * np.array(basic_split)])
    layer_config.append(
        [basewidth * np.array(basic_split), basewidth * np.array(basic_split)])
    # stage 1
    for i in range(1, stages[0]):
        layer_config.append([
            basewidth * np.array(basic_split),
            basewidth * np.array(basic_split)
        ])
    # stage 2
    layer_config.append([
        basewidth * np.array(basic_split),
        basewidth * 2 * np.array(basic_split)
    ])
    for i in range(1, stages[1] - 1):
        layer_config.append([
            basewidth * 2 * np.array(basic_split),
            basewidth * 2 * np.array(basic_split)
        ])
    layer_config.append([
        basewidth * 2 * np.array(basic_split), basewidth * 2 * np.array([
            1,
        ])
    ])
    # stage 3
    layer_config.append([
        basewidth * 2 * np.array([
            1,
        ]), basewidth * 4 * np.array(basic_split)
    ])
    for i in range(1, stages[2] - 1):
        layer_config.append([
            basewidth * 4 * np.array(basic_split),
            basewidth * 4 * np.array(basic_split)
        ])
    layer_config.append([
        basewidth * 4 * np.array(basic_split), basewidth * 4 * np.array([
            1,
        ])
    ])
    # stage 4
    layer_config.append([
        basewidth * 4 * np.array([
            1,
        ]), basewidth * 4 * np.array(basic_split)
    ])
    for i in range(1, stages[3] - 1):
        layer_config.append([
            basewidth * 4 * np.array(basic_split),
            basewidth * 4 * np.array(basic_split)
        ])
    layer_config.append([
        basewidth * 4 * np.array(basic_split), basewidth * 4 * np.array([
            1,
        ])
    ])
    # side_fuse
    side2 = basewidth * 2
    side3 = basewidth * 4
    side4 = basewidth * 4
    layer_config.append([
        np.array([side2, side3, side4]),
        np.array([side2 // 3, side3 // 3, side4 // 3])
    ])  # gOctaveCBR
    ## define dilations:
    dil_out_channels = []
    # print(layer_config[-1][1])
    for this_br in layer_config[-1][1]:
        dilations = [1, 2, 4, 8, 16]
        each_out_channels = this_br // len(dilations)
        this_dil_out_channels = []
        for i in range(len(dilations)):
            if i != len(dilations) - 1:
                this_dil_out_channels.append(each_out_channels)
            else:
                this_dil_out_channels.append(this_br - each_out_channels *
                                             (len(dilations) - 1))
        dil_out_channels.append(this_dil_out_channels)
    # print("dil_out_channels", dil_out_channels)
    layer_config.append([
        np.array([side2 // 3, side3 // 3, side4 // 3]),
        np.array([side2 // 3, side3 // 3, side4 // 3]),
        np.array(dil_out_channels)
    ])  # PallMSBlock
    layer_config.append([
        np.array([side2 // 3, side3 // 3, side4 // 3]),
        np.array([
            side2 // 3 + side3 // 3 + side4 // 3,
        ])
    ])  # gOctaveCBR

    for i in range(len(layer_config)):
        layer_config[i][0] = np.round(layer_config[i][0]).astype(np.int32)
        layer_config[i][1] = np.round(layer_config[i][1]).astype(np.int32)
    layer_config.append(stages)
    return layer_config


def load_layer_config(predefine):
    with open(predefine, "rb") as data:
        return pickle.load(data)


def get_CSFHead_dliconf(mask, old_split):
    new_split = np.zeros(old_split.shape)
    for i in range(len(mask)):
        offset = 0
        this_mask = mask[i]
        this_dil_split = old_split[i]
        # print(this_mask,this_dil_split)
        for j in range(len(this_dil_split)):
            this_dia_c = np.count_nonzero(this_mask[offset:offset +
                                                    this_dil_split[j]])
            offset += this_dil_split[j]
            new_split[i][j] = int(this_dia_c)
    return new_split


def save_layer_config(layer_config,
                      save_path,
                      epoch,
                      latest=False,
                      finetune=False):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if finetune:
        with open(
                os.path.join(save_path,
                             "layer_config_finetune_" + str(epoch) + ".bin"),
                "wb") as output:
            pickle.dump(layer_config, output)
        print(
            "Saved in:",
            os.path.join(save_path,
                         "layer_config_finetune_" + str(epoch) + ".bin"))
    else:
        with open(
                os.path.join(save_path, "layer_config_" + str(epoch) + ".bin"),
                "wb") as output:
            pickle.dump(layer_config, output)
        if latest:
            with open(os.path.join(save_path, "layer_config_latest.bin"),
                      "wb") as output:
                pickle.dump(layer_config, output)
        print("Saved in:",
              os.path.join(save_path, "layer_config_" + str(epoch) + ".bin"))


def load_gOctaveCBR_weight(module, module_name, this_mask, last_mask, newdict):

    relu_id = 0
    for name, m in module.named_modules():
        if len(name) == 0:
            continue
        if isinstance(m, gOctaveConv):
            v = m.weight.data
            out_c, in_c, kh, kw = v.shape
            this_name = module_name + '.' + name + '.weight'
            new_out_c, new_in_c, _, _ = newdict[this_name].data.shape
            newv = torch.zeros([new_out_c, new_in_c, kh, kw],
                               requires_grad=False)
            tmp_in_mask = np.concatenate(last_mask)
            in_idxs = np.nonzero(tmp_in_mask)[0]
            tmp_out_mask = np.concatenate(this_mask)
            out_idxs = np.nonzero(tmp_out_mask)[0]
            for i in range(len(out_idxs)):
                for j in range(len(in_idxs)):
                    newv[i, j, :, :] = v[out_idxs[i], in_idxs[j], :, :]
            newdict[this_name].data = newv
            # del newdict[this_name]
            # print(this_name, "done")
        elif isinstance(m, nn.BatchNorm2d):
            this_name = module_name + '.' + name
            bn_id = int(name.split('.')[-1])
            load_BN_weight(m, this_name, this_mask[bn_id], newdict)
            # bn_id+=1
            # print(name, "done")
        elif isinstance(m, nn.PReLU):
            this_name = module_name + '.' + name
            relu_id = int(name.split('.')[-1])
            load_PReLU_weight(m, this_name, this_mask[relu_id], newdict)
            # relu_id+=1

            # print(name, "done")
        # else:
        #     print(name, "not done")


def load_cls_weight(module, module_name, last_mask, newdict):
    v = module.weight.data

    out_c, in_c, kh, kw = v.shape
    this_name = module_name + '.weight'
    new_out_c, new_in_c, _, _ = newdict[this_name].data.shape
    newv = torch.zeros([new_out_c, new_in_c, kh, kw], requires_grad=False)
    tmp_in_mask = np.concatenate(last_mask)
    in_idxs = np.nonzero(tmp_in_mask)[0]
    assert new_out_c == out_c, "channels for cls must be the same."
    for i in range(out_c):
        for j in range(len(in_idxs)):
            newv[i, j, :, :] = v[i, in_idxs[j], :, :]
    newdict[this_name].data = newv
    # del newdict[this_name]
    if module.bias is not None:
        b = module.bias.data
        bias_name = module_name + '.bias'
        newdict[bias_name].data = module.bias.data
        # del newdict[bias_name]


def load_BN_weight(module, module_name, this_mask, newdict):
    v = module.weight.data
    b = module.bias.data
    name = module_name + '.weight'
    if name in newdict.keys():
        out_c = len(newdict[name].data)
    else:
        print(name, "Not in new dict!")
        return
    newv = torch.zeros(out_c, requires_grad=False)
    newb = torch.zeros(out_c, requires_grad=False)
    out_idxs = np.nonzero(this_mask)[0]
    for i in range(len(out_idxs)):
        newv[i] = v[out_idxs[i]]
        newb[i] = b[out_idxs[i]]
    newdict[module_name + '.weight'].data = newv
    newdict[module_name + '.bias'].data = newb
    # del newdict[module_name+'.weight']
    # del newdict[module_name+'.bias']


def load_PReLU_weight(module, module_name, this_mask, newdict):
    v = module.weight.data
    name = module_name + '.weight'
    if name in newdict.keys():
        out_c = len(newdict[name].data)
    else:
        print(name, "Not in new dict!")
        return
    out_c = len(newdict[name].data)
    newv = torch.zeros(out_c, requires_grad=False)
    out_idxs = np.nonzero(this_mask)[0]
    for i in range(len(out_idxs)):
        newv[i] = v[out_idxs[i]]
    newdict[module_name + '.weight'].data = newv
    # del newdict[module_name+'.weight']


def load_SimplifiedGOctConvBR_weight(module, module_name, this_mask, newdict):
    # conv_id = 0
    # bn_id = 0
    # relu_id = 0

    for name, m in module.named_modules():
        if isinstance(m, Conv2dX100):
            v = m.weight.data
            out_c, in_c, kh, kw = v.shape
            this_name = module_name + '.' + name + '.weight'
            conv_id = int(name.split('.')[-1])
            if this_name in newdict.keys():
                new_out_c, new_in_c, _, _ = newdict[this_name].data.shape
            else:
                print(this_name, "Not in new dict!")
                continue
            newv = torch.zeros([new_out_c, new_in_c, kh, kw],
                               requires_grad=False)
            out_idxs = np.nonzero(this_mask[conv_id])[0]
            for i in range(len(out_idxs)):
                newv[i, 0, :, :] = v[out_idxs[i], 0, :, :]
            newdict[this_name].data = newv
            # del newdict[this_name]
            # print(this_name, "done")
        elif isinstance(m, nn.BatchNorm2d):
            this_name = module_name + '.' + name
            bn_id = int(name.split('.')[-1])
            load_BN_weight(m, this_name, this_mask[bn_id], newdict)
            # print(name, "done")
        elif isinstance(m, nn.PReLU):
            this_name = module_name + '.' + name
            relu_id = int(name.split('.')[-1])
            load_PReLU_weight(m, this_name, this_mask[relu_id], newdict)
            # print(name, "done")
        # else:
        #     print(name, "not done")


def load_PallMSBlock_weight(module, module_name, this_mask, last_mask,
                            newdict):
    # split_id = 0
    # relu_id = 0
    for name, m in module.named_modules():
        if isinstance(m, MSBlock):
            this_name = module_name + '.' + name
            split_id = int(name.split('.')[-1])
            # print("name",name)
            load_MSBlock_weight(m, this_name, this_mask[split_id],
                                last_mask[split_id], newdict)
            # split_id+=1
        elif isinstance(m, nn.BatchNorm2d):
            this_name = module_name + '.' + name
            bn_id = int(name.split('.')[-2])
            load_BN_weight(m, this_name, this_mask[bn_id], newdict)
            # print(name, "done")
        elif isinstance(m, nn.PReLU):
            this_name = module_name + '.' + name
            relu_id = int(name.split('.')[-2])
            load_PReLU_weight(m, this_name, this_mask[relu_id], newdict)
            # relu_id+=1


def load_MSBlock_weight(module, module_name, this_mask, last_mask, newdict):
    offset = 0
    for name, m in module.named_modules():
        if len(name) == 0:
            continue
        if isinstance(m, Conv2dX100):
            v = m.weight.data
            out_c, in_c, kh, kw = v.shape
            tmp_out_mask = this_mask[offset:offset + out_c]
            offset += out_c
            this_name = module_name + '.' + name + '.weight'
            if this_name in newdict.keys():
                new_out_c, new_in_c, _, _ = newdict[this_name].data.shape
            else:
                print(this_name, "Not in new dict!")
                continue

            newv = torch.zeros([new_out_c, new_in_c, kh, kw],
                               requires_grad=False)
            tmp_in_mask = last_mask
            in_idxs = np.nonzero(tmp_in_mask)[0]
            out_idxs = np.nonzero(tmp_out_mask)[0]
            for i in range(len(out_idxs)):
                for j in range(len(in_idxs)):
                    newv[i, j, :, :] = v[out_idxs[i], in_idxs[j], :, :]
            newdict[this_name].data = newv
            # del newdict[this_name]
            # print(this_name, "done")


def build_model_with_weight(layer_config, old_model, mask, verbose=False):
    model = CSNet(layer_config=layer_config)
    model_dict = model.state_dict()
    mask = mask.copy()
    mask_id = 0
    stages = layer_config[-1]
    isfirstOct = True
    # for k, v in model_dict.items():
    #     print(k)
    # exit()
    for name, m in old_model.named_modules():
        if isinstance(m, ILBlock):
            for mm_name, mm in m.named_modules():
                full_mm_name = name + '.' + mm_name
                if isinstance(mm, gOctaveCBR):
                    if mask_id == 0:
                        last_mask = [np.array([1, 1, 1])]
                    else:
                        last_mask = mask[mask_id - 1]
                    load_gOctaveCBR_weight(mm, full_mm_name, mask[mask_id],
                                           last_mask, model_dict)
                elif isinstance(mm, SimplifiedGOctConvBR):
                    load_SimplifiedGOctConvBR_weight(mm, full_mm_name,
                                                     mask[mask_id], model_dict)
            mask_id += 1
            # print(name, "done")
        elif isinstance(m, CSFHead):
            for mm_name, mm in m.named_modules():
                full_mm_name = name + '.' + mm_name
                if isinstance(mm, gOctaveCBR):
                    if isfirstOct:
                        mask_side4 = mask[mask_id - 1]
                        mask_side3 = mask[mask_id - stages[3] - 1]
                        mask_side2 = mask[mask_id - stages[3] - stages[2] - 1]
                        isfirstOct = False
                        last_mask = np.array(
                            [mask_side2[0], mask_side3[0], mask_side4[0]])
                    else:
                        last_mask = mask[mask_id - 1]
                    load_gOctaveCBR_weight(mm, full_mm_name, mask[mask_id],
                                           last_mask, model_dict)
                    mask_id += 1
                elif isinstance(mm, PallMSBlock):
                    last_mask = mask[mask_id - 1]
                    load_PallMSBlock_weight(mm, full_mm_name, mask[mask_id],
                                            last_mask, model_dict)
                    mask_id += 1
            # print(name, "done")
        elif isinstance(m, nn.Conv2d) and name == 'cls_layer':
            load_cls_weight(m, name, mask[mask_id - 1], model_dict)
            # print(name, "done")

    # for k, v in model_dict.items():
    #     print(k,v)
    model.load_state_dict(model_dict)
    return model


def finetune_model(model, save_path, base_layer_config, thres, maxpram=None):
    '''
    target config must not be wider than base_layer_config.
    '''
    # base_layer_config = load_layer_config(os.path.join(save_path,"layer_config_latest.bin"))
    layer = 0
    new_layer_config = [None for _ in range(len(base_layer_config))]
    stages = base_layer_config[-1]
    out_mask = [None for _ in range(len(base_layer_config))]
    for m in model.modules():
        if isinstance(m, gOctaveCBR) or isinstance(m, PallMSBlock):
            this_out = base_layer_config[layer][1]
            # new_width = int(round(sum(target_config[layer][1])))
            newsplit = np.zeros(len(this_out))
            this_width = sum(base_layer_config[layer][1])
            bn_weights = []
            for n in list(m.modules()):
                if isinstance(n, nn.BatchNorm2d):
                    weight = n.weight.data.cpu().numpy()
                    bn_weights.append(weight)
            cat_bn_weights = np.concatenate(bn_weights, axis=0)
            mask = np.ones(len(cat_bn_weights))
            mask[abs(cat_bn_weights) < thres] = 0

            old_split_list = []
            tmpsum = 0
            for idx in range(len(this_out) - 1):
                tmpsum += int(this_out[idx])
                old_split_list.append(tmpsum)
            # split mask by old_split_list
            mask = np.split(mask, old_split_list)
            for idx in range(len(this_out)):
                newsplit[idx] = np.count_nonzero(mask[idx])
            out_mask[layer] = mask
            if layer == 0:
                new_layer_config[layer] = [3, newsplit]
            elif layer == len(new_layer_config) - 4:
                side4 = sum(new_layer_config[layer - 1][1])
                side3 = sum(new_layer_config[layer - stages[3] - 1][1])
                side2 = sum(new_layer_config[layer - stages[3] - stages[2] -
                                             1][1])
                # print("redefine",side2,side3,side4)
                new_layer_config[layer] = [
                    np.array([side2, side3, side4]), newsplit
                ]
            elif layer == len(new_layer_config) - 3:
                new_layer_config[layer] = [
                    new_layer_config[layer - 1][1], newsplit,
                    get_CSFHead_dliconf(
                        mask, base_layer_config[layer][2].astype(np.int32))
                ]
            else:
                new_layer_config[layer] = [
                    new_layer_config[layer - 1][1], newsplit
                ]
            layer += 1

    new_layer_config[-1] = stages
    return new_layer_config, out_mask


def build_model(epoch=0,
                predefine='',
                basic_split=[
                    1,
                ],
                save_path='tmp',
                model=None,
                expand=1.0,
                load_weight="NO",
                finetune_thres='1e-20',
                finetune=False):
    basewidth = 20
    out_mask = None
    if expand > 1:
        real_width = int(round(basewidth * expand))
        print("Expand the basewidth from", basewidth, "to", real_width)
    else:
        real_width = basewidth
    if finetune:
        print("Finetune to slim model.")
        layer_config = load_layer_config(predefine)
        layer_config, out_mask = finetune_model(model,
                                                base_layer_config=layer_config,
                                                save_path=save_path,
                                                thres=finetune_thres)
        save_layer_config(layer_config, save_path, epoch, finetune=True)
    else:
        if os.path.isfile(predefine):
            print("predefine.")
            layer_config = load_layer_config(predefine)
        else:
            print("init.")
            if epoch == 0:
                layer_config = init_layers(real_width, basic_split)
            else:
                total = sum([param.nelement() for param in model.parameters()])
                layer_config, out_mask = redefine_model(
                    model,
                    save_path=save_path,
                    thres=finetune_thres,
                    target_config=init_layers(real_width, basic_split))
            save_layer_config(layer_config, save_path, epoch, latest=True)

    # print("epoch:", epoch, layer_config)
    if out_mask == None or load_weight == "NO":
        newmodel = CSNet(layer_config=layer_config)
    elif load_weight == 'FINETUNE' and epoch != 0:
        print("finetune from epoch:", epoch)
        newmodel = build_model_with_weight(layer_config=layer_config,
                                           old_model=model,
                                           mask=out_mask)
    if epoch == 0:
        init_save_file = os.path.join(save_path, 'checkpoint',
                                      'checkpoint_init.pth.tar')
        if not os.path.isdir(os.path.join(save_path, 'checkpoint')):
            os.mkdir(os.path.join(save_path, 'checkpoint'))
        torch.save(
            {
                'epoch': -1,
                'arch': "CSNet",
                'state_dict': newmodel.state_dict(),
            }, init_save_file)
        print("Save init model:", init_save_file)
    total = sum([param.nelement() for param in newmodel.parameters()])
    print('  + Number of params: %.4fM' % (total / 1e6))
    return newmodel


# gOctaveConv
up_kwargs = {'mode': 'bilinear'}


class gOctaveConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 alpha_in=[0.5, 0.5],
                 alpha_out=[0.5, 0.5],
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 up_kwargs=up_kwargs):
        super(gOctaveConv, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        # print("in_channels",in_channels)
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, round(in_channels / self.groups),
                         kernel_size[0], kernel_size[1]))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.up_kwargs = up_kwargs
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # print("alpha_in", alpha_in)
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

        self.reset_parameters()

    def reset_parameters(self):
        # n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, xset):
        # X_h, X_l = x
        yset = []
        ysets = []
        for j in range(self.outbranch):
            ysets.append([])

        if isinstance(xset, torch.Tensor):
            xset = [
                xset,
            ]

        for i in range(self.inbranch):
            if xset[i] is None:
                continue
            if self.stride == 2:
                x = F.avg_pool2d(xset[i], (2, 2), stride=2)
            else:
                x = xset[i]
            begin_x = int(
                round(self.in_channels * self.alpha_in[i] / self.groups))
            end_x = int(
                round(self.in_channels * self.alpha_in[i + 1] / self.groups))
            if begin_x == end_x:
                continue
            for j in range(self.outbranch):
                begin_y = int(round(self.out_channels * self.alpha_out[j]))
                end_y = int(round(self.out_channels * self.alpha_out[j + 1]))
                if begin_y == end_y:
                    continue
                scale_factor = 2**(i - j)
                if self.bias is not None:
                    this_bias = self.bias[begin_y:end_y]
                else:
                    this_bias = None

                this_weight = self.weight[begin_y:end_y, begin_x:end_x, :, :]

                if scale_factor > 1:
                    y = F.conv2d(x, this_weight, this_bias, 1, self.padding,
                                 self.dilation, self.groups)
                    y = F.interpolate(y,
                                      scale_factor=scale_factor,
                                      mode=up_kwargs['mode'])
                elif scale_factor < 1:
                    x_resize = F.max_pool2d(x,
                                            int(round(1.0 / scale_factor)),
                                            stride=int(
                                                round(1.0 / scale_factor)))
                    y = F.conv2d(x_resize, this_weight, this_bias, 1,
                                 self.padding, self.dilation, self.groups)
                else:
                    y = F.conv2d(x, this_weight, this_bias, 1, self.padding,
                                 self.dilation, self.groups)
                ysets[j].append(y)

        for j in range(self.outbranch):
            if len(ysets[j]) != 0:
                yset.append(sum(ysets[j]))
            else:
                yset.append(None)
        del ysets
        return yset


class gOctaveCBR(nn.Module):
    '''
    gOctConv + BatchNorm2d + Relu
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 alpha_in=[0.5, 0.5],
                 alpha_out=[0.5, 0.5],
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 up_kwargs=up_kwargs,
                 norm_layer=nn.BatchNorm2d):
        super(gOctaveCBR, self).__init__()
        # print(alpha_in,alpha_out)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.std_conv = False
        if len(alpha_in) == 1 and len(alpha_out) == 1:
            self.std_conv = True
            self.conv = Conv2dX100(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        else:
            self.conv = gOctaveConv(in_channels, out_channels, kernel_size,
                                    alpha_in, alpha_out, stride, padding,
                                    dilation, groups, bias, up_kwargs)

        self.bns = nn.ModuleList()
        self.prelus = nn.ModuleList()
        for i in range(len(alpha_out)):
            if int(round(out_channels * alpha_out[i])) != 0:
                self.bns.append(
                    norm_layer(int(round(out_channels * alpha_out[i]))))
                self.prelus.append(
                    nn.PReLU(int(round(out_channels * alpha_out[i]))))
            else:
                self.bns.append(None)
                self.prelus.append(None)
        self.outbranch = len(alpha_out)
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.all_flops = 0
        self.baseflop = None
        self.expandflop = None

    def forward(self, xset):
        if self.std_conv:
            if isinstance(xset, torch.Tensor):
                xset = [
                    xset,
                ]

            xset = self.conv(xset[0])
            xset = self.prelus[0](self.bns[0](xset))
        else:
            xset = self.conv(xset)
            for i in range(self.outbranch):
                if xset[i] is not None:
                    xset[i] = self.prelus[i](self.bns[i](xset[i]))
        return xset


class SimplifiedGOctConvBR(nn.Module):
    '''
    The Simplified version of gOctConv + BatchNorm2d + Relu
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 alpha=[0.5, 0.5],
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 norm_layer=nn.BatchNorm2d):
        super(SimplifiedGOctConvBR, self).__init__()
        self.std_conv = False
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.prelus = nn.ModuleList()
        for i in range(len(alpha)):
            if int(round(in_channels * alpha[i])) >= 1:
                self.convs.append(
                    Conv2dX100(int(round(in_channels * alpha[i])),
                               int(round(out_channels * alpha[i])),
                               kernel_size=(3, 3),
                               groups=int(round(out_channels * alpha[i])),
                               padding=padding,
                               dilation=dilation,
                               bias=bias))
                self.bns.append(norm_layer(int(round(out_channels *
                                                     alpha[i]))))
                self.prelus.append(
                    nn.PReLU(int(round(out_channels * alpha[i]))))
            else:
                self.convs.append(None)
                self.bns.append(None)
                self.prelus.append(None)
        self.outbranch = len(alpha)
        self.all_flops = 0
        self.baseflop = None
        self.expandflop = None

    def forward(self, xset):
        if isinstance(xset, torch.Tensor):
            xset = [
                xset,
            ]
        yset = []
        for i in range(self.outbranch):
            if xset[i] is not None:
                yset.append(self.prelus[i](self.bns[i](self.convs[i](
                    xset[i]))))
            else:
                yset.append(None)

        return yset


if __name__ == '__main__':
    #images = torch.rand(2, 3, 224, 224)
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = build_model(epoch=0, basic_split=[0.5, 0.5], expand=1)
