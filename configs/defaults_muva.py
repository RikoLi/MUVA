from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Loss weights
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.I2TCE_LOSS_WEIGHT = 1.0
_C.MODEL.PCL_LOSS_WEIGHT = 1.0
_C.MODEL.MASK_LOSS_WEIGHT = 1.0
# Transformer setting
_C.MODEL.STRIDE_SIZE = [16, 16]
# freeze patch projection
_C.MODEL.FREEZE_PATCH_PROJ = True
# L2 normalized i2tce loss
_C.MODEL.NORM_I2TCE = False
# Temperature for i2tce loss
_C.MODEL.I2TCE_TAU = 1.0

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501',)
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')
# Evaluate on another dataset
_C.DATASETS.EVAL_DATASET = ''
# Use new multi-source test protocol
_C.DATASETS.USE_NEW_MSDG_PROTOCOL = True

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
_C.SOLVER = CN()
_C.SOLVER.SEED = 1234

_C.SOLVER.STAGE1 = CN()
_C.SOLVER.STAGE1.PRETRAIN = ''
_C.SOLVER.STAGE1.IMS_PER_BATCH = 64
_C.SOLVER.STAGE1.BASE_LR = 3.5e-4
_C.SOLVER.STAGE1.WEIGHT_DECAY = 1e-4
_C.SOLVER.STAGE1.OPTIMIZER_NAME = "Adam"
_C.SOLVER.STAGE1.MAX_EPOCHS = 120
_C.SOLVER.STAGE1.LR_MIN = 1e-6
_C.SOLVER.STAGE1.WARMUP_LR_INIT = 1e-5
_C.SOLVER.STAGE1.CHECKPOINT_PERIOD = 120
_C.SOLVER.STAGE1.LOG_PERIOD = 50
_C.SOLVER.STAGE1.WARMUP_EPOCHS = 5

_C.SOLVER.STAGE2 = CN()
_C.SOLVER.STAGE2.IMS_PER_BATCH = 64
_C.SOLVER.STAGE2.BASE_LR = 5e-6
_C.SOLVER.STAGE2.WEIGHT_DECAY = 1e-4
_C.SOLVER.STAGE2.OPTIMIZER_NAME = "Adam"
_C.SOLVER.STAGE2.MAX_EPOCHS = 60
_C.SOLVER.STAGE2.WARMUP_METHOD = 'linear'
_C.SOLVER.STAGE2.WARMUP_ITERS = 10
_C.SOLVER.STAGE2.WARMUP_FACTOR = 0.1
_C.SOLVER.STAGE2.CHECKPOINT_PERIOD = 60
_C.SOLVER.STAGE2.LOG_PERIOD = 50
_C.SOLVER.STAGE2.EVAL_PERIOD = 60
_C.SOLVER.STAGE2.STEPS = [30, 50]
_C.SOLVER.STAGE2.GAMMA = 0.1

# PCL memory
_C.PCL = CN()
_C.PCL.CLUSTER_NCE_TEMP = 1.0
_C.PCL.MEMORY_MOMENTUM = 0.2
_C.PCL.HARD_MEMORY_UPDATE = True
_C.PCL.MEMORY_TEMP = 0.01

# MGD
_C.MGD = CN()
_C.MGD.DISABLE_RANDOM_CROP_ERASE = True
_C.MGD.PART_INDEX = [0, 1, 2] # 0: head, 1: upperbody, 2: leg. Values should be in ordered.
_C.MGD.CONTEXT_DIM = 512
_C.MGD.CONTEXT_NUM = 4
_C.MGD.CONTEXT_INIT_STD = 0.002
_C.MGD.LOCAL_TOKEN_INIT_STD = 0.02
_C.MGD.LOCAL_TOKEN = 'random'
_C.MGD.RMP_NUM_HEADS = 2
_C.MGD.RMP_LR_MULT = 100.0
_C.MGD.RMP_USE_LAYER_NORM = False
_C.MGD.RMP_USE_INNER_SHORTCUT = False
_C.MGD.RMP_SIGMOID_TAU = 1.0
_C.MGD.MASK_GATING_THRESH = 0.5
_C.MGD.MASK_FILE_PATH = ''
_C.MGD.MASK_LOSS_TYPE = 'mse' # mse, bce+dice
_C.MGD.MASK_GATING_TYPE = 'offline' # offline, online
_C.MGD.MASK_MAX_VALUE = 0.0 # effective when gating type is 'online'
_C.MGD.DOMAIN_CONTEXT = '' # domain-aware context
_C.MGD.TEMPLATE_TYPE = 'universal' # universal, part-agnostic, 

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 64
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
_C.TEST.FEAT_NORM = True
_C.TEST.AFTER_BN = True

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""