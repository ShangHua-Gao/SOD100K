#coding:utf-8
from .parm_octconv_v2 import print_model_parm_flops, print_model_parm_nums


def simplesum(model, inputsize=(3, 224, 224), device=-1):
    parms = print_model_parm_nums(model)
    flops = print_model_parm_flops(model, inputsize=inputsize, device=device)
    return parms, flops


#  + Number of params: 25.56M
#  + Number of FLOPs: 4.11G
