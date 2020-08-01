import os
import math
import argparse
import importlib

import numpy as np
import torch
import skimage
from configs import cfg
from skimage import io
from skimage.transform import resize
from model.utils.simplesum_octconv import simplesum

parser = argparse.ArgumentParser(description='PyTorch SOD')

parser.add_argument(
    "--config",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
args = parser.parse_args()
assert os.path.isfile(args.config)
cfg.merge_from_file(args.config)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU)
if cfg.TASK == '':
    cfg.TASK = cfg.MODEL.ARCH

print(cfg)


def main():
    global cfg
    model_lib = importlib.import_module("model." + cfg.MODEL.ARCH)
    predefine_file = cfg.TEST.MODEL_CONFIG
    model = model_lib.build_model(predefine=predefine_file)
    model.cuda()
    prams, flops = simplesum(model, inputsize=(3, 224, 224), device=0)
    print('  + Number of params: %.4fM' % (prams / 1e6))
    print('  + Number of FLOPs: %.4fG' % (flops / 1e9))
    this_checkpoint = cfg.TEST.CHECKPOINT
    if os.path.isfile(this_checkpoint):
        print("=> loading checkpoint '{}'".format(this_checkpoint))
        checkpoint = torch.load(this_checkpoint)
        loadepoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(
            this_checkpoint, checkpoint['epoch']))
        test(model, cfg.TEST.DATASETS, loadepoch)
        eval(cfg.TASK, loadepoch)
    else:
        print(this_checkpoint, "Not found.")


def test(model, test_datasets, epoch):
    model.eval()
    print("Start testing.")
    for dataset in test_datasets:
        sal_save_dir = os.path.join(cfg.DATA.SAVEDIR, cfg.TASK,
                                    dataset + '_' + str(epoch))
        os.makedirs(sal_save_dir, exist_ok=True)
        img_dir = os.path.join(cfg.TEST.DATASET_PATH, dataset, 'images')
        img_list = os.listdir(img_dir)
        count = 0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        with torch.no_grad():
            for img_name in img_list:
                img = skimage.img_as_float(
                    io.imread(os.path.join(img_dir, img_name)))
                h, w = img.shape[:2]
                if cfg.TEST.IMAGE_H != 0 and cfg.TEST.IMAGE_W != 0:
                    img = resize(img, (cfg.TEST.IMAGE_H, cfg.TEST.IMAGE_W),
                                 mode='reflect',
                                 anti_aliasing=False)
                else:
                    if h % 16 != 0 or w % 16 != 0:
                        img = resize(
                            img,
                            (math.ceil(h / 16) * 16, math.ceil(w / 16) * 16),
                            mode='reflect',
                            anti_aliasing=False)
                img = np.transpose((img - mean) / std, (2, 0, 1))
                img = torch.unsqueeze(torch.FloatTensor(img), 0)
                input_var = torch.autograd.Variable(img)
                input_var = input_var.cuda()
                predict = model(input_var)
                predict = predict[0]
                predict = torch.sigmoid(predict.squeeze(0).squeeze(0))
                predict = predict.data.cpu().numpy()
                predict = (resize(
                    predict, (h, w), mode='reflect', anti_aliasing=False) *
                           255).astype(np.uint8)
                save_file = os.path.join(sal_save_dir, img_name[0:-4] + '.png')
                io.imsave(save_file, predict)
                count += 1
        print('Dataset: {}, {} images'.format(dataset, len(img_list)))


def eval(method, epoch):
    evalcmd = "python eval.py --method " + method + " --range " + str(
        epoch) + "," + str(epoch + 1)
    print(evalcmd)
    content = os.popen(evalcmd).read()
    print(content)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
