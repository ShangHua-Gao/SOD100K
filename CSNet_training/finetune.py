import argparse
import datetime
import importlib
import math
import os
import time

import numpy as np
import skimage
import torch
import torch.nn.functional as F
from configs import cfg
from model.utils.simplesum_octconv import simplesum
from skimage import io
from skimage.transform import resize
from torch.optim import lr_scheduler
from utils.prepare_data import SalData, val_collate
from utils.utils import load_pretrained

parser = argparse.ArgumentParser(description='PyTorch SOD')

parser.add_argument(
    "--config",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "--epoch",
    default=-1,
    help="epoch to infer, start with 1",
    type=int,
)
args = parser.parse_args()
assert os.path.isfile(args.config)
cfg.merge_from_file(args.config)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU)
if cfg.TASK == '':
    cfg.TASK = cfg.MODEL.ARCH

timenow = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
check_point_dir = os.path.join(cfg.DATA.SAVEDIR, cfg.TASK, 'checkpoint')
finetune_check_point_dir = os.path.join(
    cfg.DATA.SAVEDIR, cfg.TASK, 'finetune_checkpoint_epoch' + str(args.epoch))
logname = 'logfinetune_' + cfg.TASK + "_epoch_" + str(
    args.epoch) + "_" + timenow + '.txt'
val_logname = 'logfinetuneval_' + cfg.TASK + "_epoch_" + str(
    args.epoch) + "_" + timenow + '.txt'
logging_dir = os.path.join(cfg.DATA.SAVEDIR, cfg.TASK)
if not os.path.isdir(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
if not os.path.isdir(check_point_dir):
    os.mkdir(check_point_dir)
if not os.path.isdir(finetune_check_point_dir):
    os.mkdir(finetune_check_point_dir)
LOG_FOUT = open(os.path.join(logging_dir, logname), 'w')
VAL_LOG_FOUT = open(os.path.join(logging_dir, val_logname), 'w')


def log_string(out_str, display=True):
    out_str = str(out_str)
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    if display:
        print(out_str)


def val_log_string(out_str, display=False):
    out_str = str(out_str)
    VAL_LOG_FOUT.write(out_str + '\n')
    VAL_LOG_FOUT.flush()
    if display:
        print(out_str)


log_string(cfg)

best_mae = 1000000
best_epoch = -1


def main():
    global cfg, best_mae, best_epoch

    model_lib = importlib.import_module("model." + cfg.MODEL.ARCH)
    if cfg.PRUNE.BNS:
        global gOctaveCBR
        gOctaveCBR = model_lib.gOctaveCBR
    model_lib = importlib.import_module("model." + cfg.MODEL.ARCH)
    if cfg.AUTO.ENABLE:
        layer_config_dir = os.path.join(cfg.DATA.SAVEDIR, cfg.TASK,
                                        'layer_configs')
        if cfg.AUTO.ENABLE:
            predefine_file = os.path.join(layer_config_dir,
                                          "layer_config_0.bin")
            model = model_lib.build_model(predefine=predefine_file)
        else:
            model = model_lib.build_model(predefine=cfg.AUTO.PREDEFINE,
                                          save_path=layer_config_dir)
    else:
        model = model_lib.build_model(basic_split=cfg.MODEL.BASIC_SPLIT)
    model.cuda()

    prams, flops = simplesum(model, inputsize=(3, 224, 224), device=0)
    print("basic_split: " + str(cfg.MODEL.BASIC_SPLIT))
    log_string('  + Number of params: %.4fM' % (prams / 1e6), display=False)
    log_string('  + Number of FLOPs: %.4fG' % (flops / 1e9), display=False)
    if cfg.FINETUNE.SOLVER.METHOD == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            cfg.FINETUNE.SOLVER.LR,
            momentum=cfg.FINETUNE.SOLVER.MOMENTUM,
            weight_decay=cfg.FINETUNE.SOLVER.WEIGHT_DECAY)
    elif cfg.FINETUNE.SOLVER.METHOD == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.FINETUNE.SOLVER.LR,
            betas=(0.9, 0.99),
            eps=1e-08,
            weight_decay=cfg.FINETUNE.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = None
        print("WARNING: Method not implmented.")
    if cfg.DATA.PRETRAIN != '':
        model = load_pretrained(model, cfg.DATA.PRETRAIN)
    start_epoch = 0
    check_point_dir = os.path.join(cfg.DATA.SAVEDIR, cfg.TASK, 'checkpoint')
    this_checkpoint = os.path.join(
        check_point_dir, 'checkpoint_epoch{}.pth.tar'.format(args.epoch))
    if os.path.isfile(this_checkpoint):
        log_string("=> loading checkpoint '{}'".format(this_checkpoint))
        checkpoint = torch.load(this_checkpoint)
        from_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        log_string("=> loaded checkpoint '{}' (epoch {})".format(
            this_checkpoint, checkpoint['epoch']))
    else:
        log_string("=> no checkpoint found at '{}'".format(this_checkpoint))
        exit()

    train_loader, val_loader = prepare_data(cfg.DATA.DIR, cfg.VAL.DIR)
    if cfg.FINETUNE.SOLVER.ADJUST_STEP:
        if cfg.FINETUNE.SOLVER.LR_SCHEDULER == 'step':
            print("step", cfg.FINETUNE.SOLVER.STEPS)
            print(cfg.FINETUNE.SOLVER.LR)
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 cfg.FINETUNE.SOLVER.STEPS,
                                                 gamma=0.1)
        elif cfg.FINETUNE.SOLVER.LR_SCHEDULER == 'cosine':
            max_epoch = cfg.FINETUNE.SOLVER.MAX_EPOCHS
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                       max_epoch,
                                                       eta_min=0)
        else:
            raise ValueError("Unsupported scheduler.")

    model = model_lib.build_model(epoch=from_epoch,
                                  basic_split=cfg.MODEL.BASIC_SPLIT,
                                  model=model,
                                  save_path=layer_config_dir,
                                  predefine=predefine_file,
                                  finetune=True,
                                  finetune_thres=cfg.FINETUNE.THRES,
                                  load_weight='FINETUNE')
    model.cuda()
    start_epoch = 0
    prams, flops = simplesum(model, inputsize=(3, 224, 224), device=0)
    print("After finetune: basic_split: " + str(cfg.MODEL.BASIC_SPLIT))
    log_string('  + Number of params: %.4fM' % (prams / 1e6), display=False)
    log_string('  + Number of FLOPs: %.4fG' % (flops / 1e9), display=False)
    for epoch in range(start_epoch, cfg.FINETUNE.SOLVER.MAX_EPOCHS):
        if cfg.FINETUNE.SOLVER.ADJUST_STEP:
            scheduler.step()
            lr = scheduler.get_lr()[0]
            log_string("lr: " + str(lr))

        train(train_loader, model, optimizer, epoch)
        mae = val(val_loader, model, epoch)
        if cfg.TEST.ENABLE and (epoch + 1) >= cfg.TEST.BEGIN and (
                epoch + 1) % cfg.TEST.INTERVAL == 0:
            test(model, cfg.TEST.DATASETS, epoch + 1)
        is_best = mae < best_mae
        best_mae = min(mae, best_mae)
        if is_best:
            best_epoch = epoch + 1
        log_string(" epoch: " + str(epoch + 1) + " mae: " + str(mae) +
                   " best_epoch: " + str(best_epoch) + " best_mae: " +
                   str(best_mae))
        val_log_string(" epoch: " + str(epoch + 1) + " mae: " + str(mae) +
                       " best_epoch: " + str(best_epoch) + " best_mae: " +
                       str(best_mae))
        # Save checkpoint
        save_file = os.path.join(
            finetune_check_point_dir,
            'checkpoint_epoch{}.pth.tar'.format(epoch + 1))
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': cfg.MODEL.ARCH,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            filename=save_file)


def train(train_loader, model, optimizer, epoch):
    log_string('Memory useage: %.4fM' %
               (torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()
    if cfg.PRUNE.BNS and cfg.PRUNE.SHOW:
        foo_bns(model)
    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # prepare input
        input = data['img'].float()
        target = data['gt'].float()
        input = torch.autograd.Variable(input).cuda()
        target = target.cuda()
        output = model(input)
        loss = 0
        if cfg.LOSS.MLOSS > 1:
            for k in range(cfg.LOSS.MLOSS):
                loss += F.binary_cross_entropy_with_logits(output[k], target)
        else:
            loss += F.binary_cross_entropy_with_logits(output, target)
        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % cfg.PRINT_FREQ == 0:
            log_string('Epoch: [{0}][{1}/{2}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                           epoch,
                           i,
                           len(train_loader),
                           batch_time=batch_time,
                           data_time=data_time,
                           loss=losses))


def val(val_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    maes = AverageMeter()
    # switch to eval mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, [img, target, h, w] in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # prepare input
            input = img.float().cuda()
            output = model(input)
            output_sig = torch.sigmoid(output)
            for idx in range(input.size(0)):
                if cfg.LOSS.MLOSS > 1:
                    for k in range(cfg.LOSS.MLOSS):
                        output_resize = (F.interpolate(output_sig[k][idx],
                                                    size=(h[idx], w[idx]),
                                                    mode='bilinear') *
                                        255.0).int().float() / 255.0
                else:
                    output_resize = (F.interpolate(
                        output_sig[idx].unsqueeze(dim=0),
                        size=(h[idx], w[idx]),
                        mode='bilinear') * 255.0).int().float() / 255.0
                this_target = target[idx].float().cuda().unsqueeze(dim=0)
                mae = F.l1_loss(output_resize, this_target, reduction="mean")
                maes.update(mae.item(), 1)
            batch_time.update(time.time() - end)
            end = time.time()
            if i % cfg.VAL.PRINT_FREQ == 0:
                print('ValEpoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'MAE {mae.val:.4f} ({mae.avg:.4f})\t'.format(
                          epoch + 1,
                          i,
                          len(val_loader),
                          batch_time=batch_time,
                          data_time=data_time,
                          mae=maes))
    return maes.avg


def test(model, test_datasets, epoch):
    model.eval()
    log_string("Start testing.")
    for dataset in test_datasets:
        sal_save_dir = os.path.join(cfg.DATA.SAVEDIR, cfg.TASK,
                                    'finetune_' + dataset + '_' + str(epoch))
        os.makedirs(sal_save_dir, exist_ok=True)
        img_dir = os.path.join(cfg.TEST.DATASET_PATH, dataset, 'images')
        img_list = os.listdir(img_dir)
        start_time = time.time()
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
                # print("Finished:",count,"/",len(img_list))
                count += 1
        duration = time.time() - start_time
        log_string(
            'Dataset: {}, {} images, test time: {}, speed: {}fps'.format(
                dataset, len(img_list), duration,
                len(img_list) / duration))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def prepare_data(data_dir, val_data_dir):
    # prepare dataloader for training
    dataset = SalData(data_dir, (cfg.DATA.IMAGE_H, cfg.DATA.IMAGE_W))
    train_loader = torch.utils.data.DataLoader(dataset,
                                               shuffle=True,
                                               batch_size=cfg.DATA.BATCH_SIZE,
                                               num_workers=cfg.DATA.WORKERS,
                                               drop_last=True)

    val_dataset = SalData(val_data_dir, (cfg.DATA.IMAGE_H, cfg.DATA.IMAGE_W),
                          mode='val')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             shuffle=False,
                                             batch_size=cfg.DATA.BATCH_SIZE,
                                             collate_fn=val_collate,
                                             num_workers=cfg.DATA.WORKERS,
                                             drop_last=False)
    return train_loader, val_loader


def foo_bns(net, pre=''):
    # print(str(type(net)))
    log_string(pre + type(net).__name__, display=False)
    childrens = list(net.children())
    # if not childrens:
    if isinstance(net, gOctaveCBR):
        # print("here.")
        # print(list(net.children()))
        for n in list(net.modules()):
            if isinstance(n, torch.nn.BatchNorm2d):
                log_string(' ' * len(pre) + ' ' + str((n.weight.data)),
                           display=False)
        # return
    for i in range(len(childrens)):
        foo_bns(childrens[i], pre + str(i) + '**|')


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
