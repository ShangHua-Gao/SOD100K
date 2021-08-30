# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST
# _C.TASK = CN()
_C = CN()

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C.TASK = ""

_C.GPU = 0

_C.PRINT_FREQ = 10

_C.MODEL = CN()
_C.MODEL.ARCH = 'csnet'
_C.MODEL.BASIC_SPLIT = [
    1,
]

_C.LOSS = CN()
_C.LOSS.MLOSS = 4

# # -----------------------------------------------------------------------------
# Dataset
# # -----------------------------------------------------------------------------
_C.DATA = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATA.DIR = ''
_C.DATA.BATCH_SIZE = 32
_C.DATA.WORKERS = 4
_C.DATA.SAVEDIR = 'results/'
_C.DATA.PRETRAIN = ''
_C.DATA.RESUME = ''
_C.DATA.IMAGE_H = 224
_C.DATA.IMAGE_W = 224
_C.DATA.AUG = False
# List of the dataset names for testing, as present in paths_catalog.py
_C.VAL = CN()
_C.VAL.DIR = ''
_C.VAL.PRINT_FREQ = 20

_C.TEST = CN()
_C.TEST.DATASET_PATH = ''
_C.TEST.BEGIN = 200
_C.TEST.INTERVAL = 5
_C.TEST.DATASETS = ['ECSSD']
_C.TEST.CHECKPOINT = ''
_C.TEST.ENABLE = True
_C.TEST.IMAGE_H = 0
_C.TEST.IMAGE_W = 0
_C.TEST.TESTALL = False
_C.TEST.MODEL_CONFIG = ''
# # ---------------------------------------------------------------------------- #
# # Solver
# # ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.METHOD = 'Adam'
_C.SOLVER.MAX_EPOCHS = 100
_C.SOLVER.LR = 1e-4
_C.SOLVER.MOMENTUM = 0.95
_C.SOLVER.WEIGHT_DECAY = 5 * 1e-3
_C.SOLVER.ADJUST_STEP = False
_C.SOLVER.STEPS = [200, 250]
_C.SOLVER.WARMUPLR = 0
_C.SOLVER.STEPSIZE = 20
_C.SOLVER.GAMMA = 0.5
_C.SOLVER.LR_SCHEDULER = 'step'

_C.SOLVER.FINETUNE = CN()
_C.SOLVER.FINETUNE.METHOD = 'Adam'
_C.SOLVER.FINETUNE.LR = 1e-4
_C.SOLVER.FINETUNE.MOMENTUM = 0.95
_C.SOLVER.FINETUNE.WEIGHT_DECAY = 5 * 1e-3
_C.SOLVER.FINETUNE.GAMMA = 0.5
_C.SOLVER.FINETUNE.ADJUST_STEP = False
_C.SOLVER.FINETUNE.STEPS = [5, 10]
_C.SOLVER.FINETUNE.LR_SCHEDULER = 'step'
# learning rate decay parameter: Gamma

_C.PRUNE = CN()
_C.PRUNE.BNS = False
_C.PRUNE.SHOW = True

_C.AUTO = CN()
_C.AUTO.ENABLE = False
_C.AUTO.PREDEFINE = ''
_C.AUTO.FINETUNE = 300
_C.AUTO.FLOPS = CN()
_C.AUTO.FLOPS.ENABLE = False
_C.AUTO.FLOPS.WEIGHT = 0.0
_C.AUTO.FLOPS.EXPAND = -1.0
_C.AUTO.EXPAND = 1.0
_C.AUTO.LOAD_WEIGHT = "NO"

_C.FINETUNE = CN()
_C.FINETUNE.ENABLE = False
_C.FINETUNE.THRES = 1e-40
_C.FINETUNE.SOLVER = CN()
_C.FINETUNE.SOLVER.METHOD = 'Adam'
_C.FINETUNE.SOLVER.MAX_EPOCHS = 20
_C.FINETUNE.SOLVER.LR = 1e-7
_C.FINETUNE.SOLVER.MOMENTUM = 0.95
_C.FINETUNE.SOLVER.WEIGHT_DECAY = 5 * 1e-3
_C.FINETUNE.SOLVER.ADJUST_STEP = False
_C.FINETUNE.SOLVER.STEPS = [50, 100]
_C.FINETUNE.SOLVER.WARMUPLR = 0
_C.FINETUNE.SOLVER.STEPSIZE = 20
_C.FINETUNE.SOLVER.GAMMA = 0.5
_C.FINETUNE.SOLVER.LR_SCHEDULER = 'step'
