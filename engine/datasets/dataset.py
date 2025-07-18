from PIL import Image, ImageFile

from torch.utils.data import Dataset
from .preprocessing import RandomMaskedHorizontalFlip
import os
import os.path as osp
import random
import json
import pickle
import numpy as np
import cv2
import torch
import torchvision.transforms as T
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, return_path=False):
        self.dataset = dataset
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.return_path:
            return img_path, img, pid, camid, trackid
        else:
            return img, pid, camid, trackid
        
class ImageDatasetGroundingMask(Dataset):
    def __init__(self, dataset, transform=None, return_path=False, mask_file_path=None,
                 fliplr_p=None, h_patch=None, w_patch=None):
        self.dataset = dataset
        self.transform = transform
        self.return_path = return_path
        self.parse_mask_file(mask_file_path)
        
        self.fliplr_p = fliplr_p
        if self.fliplr_p is not None:
            self.random_flip_lr = RandomMaskedHorizontalFlip(h_patch, w_patch, p=fliplr_p)
            print(f'Enable consistent horizontal flipping with prob={fliplr_p}')
        
    def parse_mask_file(self, path):
        with open(path, 'rb') as f:
            self.mask_dict = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        
        name = osp.basename(img_path)
        mask = self.mask_dict[name] # [R, L]
        mask = torch.from_numpy(mask)

        if self.transform is not None:
            img = self.transform(img)
            
        if self.fliplr_p is not None:
            img, mask = self.random_flip_lr(img, mask) # consistently flip

        if self.return_path:
            return img_path, img, pid, camid, trackid, mask
        else:
            return img, pid, camid, trackid, mask
        
class ImageDatasetLLM(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return index, img_path, img, pid, camid, trackid
    
class PseudoLabelImageDataset(ImageDataset):
    def __init__(self, dataset, transform=None):
        super().__init__(dataset, transform)
        
    def __getitem__(self, index):
        # override to return pseudo ID
        img_path, pid, camid, trackid, pseudo_id = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, pseudo_id
    
   
from .preprocessing import MultiGrainedSlice
class TeacherImageDataset(Dataset):
    def __init__(self, dataset, size_before_slice, num_regions, teacher_transform,
                 student_resize):
        self.dataset = dataset
        self.teacher_transform = teacher_transform
        self.student_resize = student_resize
        self.size_before_slice = size_before_slice
        self.slicer = MultiGrainedSlice(size_before_slice, num_regions)
        self.to_pillow = T.ToPILImage()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        
        # image regions for teacher
        img_global = self.teacher_transform(img) # [C, tH, tW]
        img_local = self.slicer(self.student_resize(img)) # [num_regions, teacherH, teacherW, C]
        img_t = [self.teacher_transform(self.to_pillow(region)) for region in img_local]
        img_t = torch.stack([img_global]+img_t, dim=0) # [1+num_regions, C, tH, tW]
        
        return img_t, pid, camid, trackid

class KDImageDatasetPLIP(Dataset):
    def __init__(self, dataset, teacher_transform, student_transform):
        self.dataset = dataset
        self.teacher_transform = teacher_transform
        self.student_transform = student_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        
        # image regions for teacher
        img_t = self.teacher_transform(img) # [C, tH, tW]
        
        # image for student
        img_s = self.student_transform(img) # [C, H, W]

        return img_t, img_s, pid, camid, trackid

class KDImageDataset(Dataset):
    def __init__(self, dataset, size_before_slice, num_regions, teacher_transform,
                 student_transform, student_resize):
        self.dataset = dataset
        self.teacher_transform = teacher_transform
        self.student_transform = student_transform
        self.size_before_slice = size_before_slice
        self.student_resize = student_resize
        self.slicer = MultiGrainedSlice(size_before_slice, num_regions)
        self.to_pillow = T.ToPILImage()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)
        
        # image regions for teacher
        img_global = self.teacher_transform(img) # [C, tH, tW]
        img_local = self.slicer(self.student_resize(img)) # [num_regions, teacherH, teacherW, C]
        img_t = [self.teacher_transform(self.to_pillow(region)) for region in img_local]
        img_t = torch.stack([img_global]+img_t, dim=0) # [1+num_regions, C, tH, tW]
        
        # image for student
        img_s = self.student_transform(img) # [C, H, W]

        return img_t, img_s, pid, camid, trackid
    
    
class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)
