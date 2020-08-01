# -*- coding: utf-8 -*-
import os
import argparse
from os.path import join

parser = argparse.ArgumentParser(description='Forward all testing images.')
parser.add_argument('--method',
                    type=str,
                    default='hhrnet_mix_low_nores_auto_nopretrain')
parser.add_argument('--epoch', type=int, default=75)
parser.add_argument('--traverse', type=int, default=1)
parser.add_argument('--range', type=str, default='110,165')
parser.add_argument('--gt_dir', type=str, default='/media/da/Datasets/sal/')
parser.add_argument(
    '--save_dir',
    type=str,
    default='CSNet/results/')
args = parser.parse_args()

modelprefix = args.method

outputs = ['DUTS-TE', 'ECSSD']
LOG_FOUT = open(
    join(args.save_dir, modelprefix,
         "FmeasureResults_" + str(args.method) + '_' + str(outputs) + '.txt'),
    'a')


def log_string(out_str):
    out_str = str(out_str)
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


# #########eval########### #
if args.traverse:
    testrange = args.range.split(',')
    startepoch = int(testrange[0])
    endepoch = int(testrange[1])
else:
    startepoch = args.epoch
    endepoch = args.epoch + 1
bestF = 0
best_epoch = 0
print("Start:", startepoch, "End:", endepoch)
for thisepoch in range(startepoch, endepoch):
    for output in outputs:
        save_dir = join(args.save_dir, modelprefix,
                        output + '_' + str(thisepoch))
        if not os.path.isdir(save_dir):
            continue
        else:
            log_string(output + " Eval epoch: " + str(thisepoch))
        gt = join(args.gt_dir, output, 'GT')
        res = os.listdir(save_dir)
        vallist = join(args.save_dir, modelprefix,
                       "val_" + output + '_' + str(thisepoch) + '.txt')
        fobj = open(vallist, 'w')
        for sub in res:
            fobj.write(join(save_dir, sub) + ' ' + join(gt, sub) + '\n')
        fobj.close()
        valreslist = join(
            args.save_dir, modelprefix,
            "FmeasureResult_" + output + '_' + str(thisepoch) + ".txt")
        valresobj = open(valreslist, 'w')
        # print(vallist+" generate done.")
        valcmd = "SalMetric/build/salmetric " + vallist + " 8"
        content = os.popen(valcmd).read()
        valresobj.write(content)
        results = content.split('\n')[-8:]
        log_string(results)
        thisMaxF = float(results[0].split()[1])
        if bestF < thisMaxF:
            bestF = thisMaxF
            best_epoch = thisepoch
        print(valreslist + " eval done.")
        valresobj.close()
log_string("BestF: " + str(bestF) + " in Epoch: " + str(best_epoch))
