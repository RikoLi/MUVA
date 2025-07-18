import os.path as osp
import torch
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader
from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .msmt17_v2 import MSMT17_V2
from .cuhk03_np import CUHK03_NP
from .viper import VIPeR
from .ilids import iLIDS
from .grid import GRID
from .prid import PRID
from .multi_source_dg import MultiSourceDG, ClassicalMultiSourceDG
from .preprocessing import RandomErasing
from .dataset import ImageDataset, ImageDatasetGroundingMask
from .sampler import RandomIdentitySampler

FACTORY = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17_V2,
    'cuhk03np': CUHK03_NP,
    'viper': VIPeR,
    'ilids': iLIDS,
    'grid': GRID,
    'prid': PRID,
    'msdg': MultiSourceDG,
    'classical_msdg': ClassicalMultiSourceDG
}

def collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids

def collate_fn_grounding_mask(batch):
    imgs, pids, camids, viewids, masks = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    masks = torch.stack(masks, dim=0).type_as(imgs)
    return imgs, pids, camids, viewids, masks

def make_dataloader(cfg):
    """
    It returns 3 dataloaders: training loader, cluster loader and validation loader.
    - For training loader, PK sampling is applied to select K instances from P classes.
    - For cluster loader, a plain loader is returned with validation augmentation but on
      training samples.
    - For validation loader, a validation loader is returned on test samples.
    
    Args:
    - dataset: dataset object.
    - all_iters: if `all_iters=True`, number training iteration is decided by `num_samples//batchsize`
    """
    
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = FACTORY[cfg.DATASETS.NAMES[0]](root=cfg.DATASETS.ROOT_DIR) # single-source
    elif len(cfg.DATASETS.NAMES) > 1:
        dataset = FACTORY['msdg'](root=cfg.DATASETS.ROOT_DIR, cuhk_protocol='detected',
                                  train_datasets=cfg.DATASETS.NAMES,
                                  test_dataset=cfg.DATASETS.EVAL_DATASET) # multi-source
    else:
        raise ValueError(f'Invalid dataset input type {type(cfg.DATASETS.NAMES)}!')
    
    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids
    
    # train loader
    if cfg.INPUT.DISABLE_RANDOM_CROP_ERASE:
        print('Disable random crop & erase preprocessing.')
        train_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ])
        print('Train transforms:')
        print(train_transforms)
    else:
        print('Enable random crop & erase preprocessing.')
        train_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
        print('Train transforms:')
        print(train_transforms)
    
    
    
    train_set = ImageDataset(dataset.train, train_transforms)
    sampler = RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # val loader
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    val_set = ImageDataset(dataset.query+dataset.gallery, val_transforms)
    num_queries = len(dataset.query)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers
    )
    
    
    # cluster loader
    cluster_set = ImageDataset(dataset.train, transform=val_transforms)
    cluster_loader = DataLoader(
        cluster_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, cluster_loader, num_queries, num_classes, cam_num, view_num


def make_coop_dataloader(cfg):
    if cfg.MGD.DISABLE_RANDOM_CROP_ERASE:
        print('Disable random crop & erase preprocessing.')
        train_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                # T.RandomHorizontalFlip(p=cfg.INPUT.PROB), # img & mask flipping in data fetch
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            ])
        print('Train transforms:')
        print(train_transforms)
    else:
        print('Enable random crop & erase preprocessing.')
        train_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                # T.RandomHorizontalFlip(p=cfg.INPUT.PROB), # img & mask flipping in data fetch
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
        print('Train transforms:')
        print(train_transforms)

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = FACTORY[cfg.DATASETS.NAMES[0]](root=cfg.DATASETS.ROOT_DIR) # single-source
    elif cfg.DATASETS.USE_NEW_MSDG_PROTOCOL:
        assert len(cfg.DATASETS.NAMES) > 1, 'Should contain more than one dataset under new MSDG protocol!'
        dataset = FACTORY['msdg'](root=cfg.DATASETS.ROOT_DIR, cuhk_protocol='detected',
                                train_datasets=cfg.DATASETS.NAMES,
                                test_dataset=cfg.DATASETS.EVAL_DATASET) # multi-source
    else:
        dataset = FACTORY['classical_msdg'](root=cfg.DATASETS.ROOT_DIR,
                                            all_for_train=True,
                                            test_dataset=cfg.DATASETS.EVAL_DATASET)
    
    train_set = ImageDatasetGroundingMask(dataset.train, train_transforms, mask_file_path=cfg.MGD.MASK_FILE_PATH,
                                          fliplr_p=cfg.INPUT.PROB,
                                          h_patch=int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1),
                                          w_patch=int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1))
    train_set_normal = ImageDatasetGroundingMask(dataset.train, val_transforms, mask_file_path=cfg.MGD.MASK_FILE_PATH)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    train_loader_stage2 = DataLoader(
        train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
        sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, collate_fn=collate_fn_grounding_mask
    )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn
    )
    train_loader_stage1 = DataLoader(
        train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn_grounding_mask
    )
    return train_loader_stage1, train_loader_stage2, val_loader, len(dataset.query), num_classes, cam_num, view_num






def make_val_dataloader(cfg):
    """Only return a dataloader for test split."""
    
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = FACTORY[cfg.DATASETS.EVAL_DATASET](root=cfg.DATASETS.ROOT_DIR)
    val_set = ImageDataset(dataset.query+dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn
    )
    return val_loader, len(dataset.query)
