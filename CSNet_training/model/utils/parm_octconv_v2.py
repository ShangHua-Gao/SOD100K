#coding:utf-8
import torch
from torch.autograd import Variable


def print_model_parm_nums(model):
    #model = models.alexnet()
    total = sum([param.numel() for param in model.parameters()])
    print('  + Number of params: %.4fM' % (total / 1e6))
    return total


def print_model_parm_flops(model, inputsize, device=-1):

    multiply_adds = False
    list_conv = []

    def conv_hook(self, input, output):
        # print('input', input[0].size())
        # print('output', output.size())
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (
            self.in_channels / self.groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    def max_pool_calc(kernel_size, output_shape):
        if isinstance(kernel_size, int):
            kernel_ops = kernel_size * kernel_size
        else:
            kernel_ops = kernel_size[0] * kernel_size[1]
        return output_shape[0] * output_shape[1] * output_shape[
            2] * output_shape[3] * kernel_ops

    def avg_pool_calc(kernel_size, output_shape):
        if isinstance(kernel_size, int):
            kernel_ops = kernel_size * kernel_size
        else:
            kernel_ops = kernel_size[0] * kernel_size[1]
        kernel_ops += 1
        return output_shape[0] * output_shape[1] * output_shape[
            2] * output_shape[3] * kernel_ops

    def interpolate_calc(output_shape):
        kernel_ops = 9
        flops = output_shape[0] * output_shape[1] * output_shape[
            2] * output_shape[3] * kernel_ops
        return flops

    def conv_calc(input_channels, output_shape, kernel_size, bias, groups):
        if isinstance(kernel_size, int):
            kernel_ops = kernel_size * kernel_size
        else:
            kernel_ops = kernel_size[0] * kernel_size[1]

        kernel_ops = kernel_ops * (input_channels /
                                   groups) * (2 if multiply_adds else 1)
        bias_ops = 1 if bias is not None else 0

        flops = (kernel_ops + bias_ops) * output_shape[0] * output_shape[
            1] * output_shape[2] * output_shape[3]
        return flops

    list_octconv = []

    def octconv_hook(self, input, output):
        input = input[0]  # input:tuple len:1
        if isinstance(input, torch.Tensor):
            input = [
                input,
            ]
        flops = 0
        # kernel_size = self.weights.shape[-1]
        kernel_size = self.weight.shape[-1]
        for i in range(self.inbranch):
            if input[i] is None:
                continue
            x_shape = input[i].shape
            if self.stride == 2:
                flops += avg_pool_calc(2, x_shape)
                x_shape = (x_shape[0], x_shape[1], x_shape[2] / 2,
                           x_shape[3] / 2)
            for j in range(self.outbranch):
                begin_x = int(self.in_channels * self.alpha_in[i] /
                              self.groups)
                end_x = int(self.in_channels * self.alpha_in[i + 1] /
                            self.groups)
                begin_y = int(self.out_channels * self.alpha_out[j])
                end_y = int(self.out_channels * self.alpha_out[j + 1])
                scale_factor = 2**(i - j)

                if scale_factor > 1:
                    flops += conv_calc(input_channels=end_x - begin_x,
                                       output_shape=(x_shape[0],
                                                     end_y - begin_y,
                                                     x_shape[2], x_shape[3]),
                                       kernel_size=kernel_size,
                                       bias=self.bias,
                                       groups=self.groups)
                    flops += interpolate_calc(
                        output_shape=(x_shape[0], end_y - begin_y,
                                      x_shape[2] * scale_factor,
                                      x_shape[3] * scale_factor))
                elif scale_factor < 1:
                    flops += max_pool_calc(
                        kernel_size,
                        output_shape=(x_shape[0], end_x - begin_x,
                                      x_shape[2] * scale_factor,
                                      x_shape[3] * scale_factor))
                    flops += conv_calc(
                        input_channels=end_x - begin_x,
                        output_shape=(x_shape[0], end_y - begin_y,
                                      x_shape[2] * scale_factor,
                                      x_shape[3] * scale_factor),
                        kernel_size=kernel_size,
                        bias=self.bias,
                        groups=self.groups)
                else:
                    flops += conv_calc(input_channels=end_x - begin_x,
                                       output_shape=(x_shape[0],
                                                     end_y - begin_y,
                                                     x_shape[2], x_shape[3]),
                                       kernel_size=kernel_size,
                                       bias=self.bias,
                                       groups=self.groups)
        list_octconv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.numel() * (2 if multiply_adds else 1)
        bias_ops = self.bias.numel()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bilinear = []

    def bilinear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops_first = self.weight.numel() * (2 if multiply_adds else 1)
        weight_ops_second = input[1].size(1) * output.size(0) * (
            2 if multiply_adds else 1)
        weight_ops = weight_ops_first + weight_ops_second
        bias_ops = self.bias.numel()
        flops = batch_size * (weight_ops + bias_ops)
        list_bilinear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].numel() * 4)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].numel())

    list_prelu = []

    def prelu_hook(self, input, output):
        list_prelu.append(input[0].numel() * 3)

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()[-3:]

        if isinstance(self.kernel_size, int):
            kernel_ops = self.kernel_size * self.kernel_size
        else:
            kernel_ops = self.kernel_size[0] * self.kernel_size[1]
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        net_name = net.__class__.__name__
        # print(net_name)
        childrens = list(net.children())
        # print(type(net))
        if not childrens:
            if net_name in ['Conv2d', 'Conv2dX100']:
                net.register_forward_hook(conv_hook)
            if net_name in ['Linear']:  # isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if net_name in ['Bilinear']:  # isinstance(net, torch.nn.Bilinear):
                net.register_forward_hook(bilinear_hook)
            if net_name in ['BatchNorm2d'
                            ]:  # isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if net_name in ['ReLU']:  # isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if net_name in ['PReLU']:  # isinstance(net, torch.nn.PReLU):
                net.register_forward_hook(prelu_hook)
            if net_name in [
                    'MaxPool2d', 'AvgPool2d'
            ]:  # isinstance(net, (torch.nn.MaxPool2d, torch.nn.AvgPool2d)):
                net.register_forward_hook(pooling_hook)
            return
        else:
            if net_name == 'gOctaveConv':
                net.register_forward_hook(octconv_hook)
            else:
                for c in childrens:
                    foo(c)

    foo(model)
    if device >= 0 and torch.cuda.is_available():
        input = Variable(torch.rand(inputsize).unsqueeze(0),
                         requires_grad=True).cuda(device)
    else:
        input = Variable(torch.rand(inputsize).unsqueeze(0),
                         requires_grad=True)

    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bilinear) +
                   sum(list_bn) + sum(list_relu) + sum(list_prelu) +
                   sum(list_pooling) + sum(list_octconv))

    print('  + Number of FLOPs: %.4fG' % (total_flops / 1e9))
    return total_flops
