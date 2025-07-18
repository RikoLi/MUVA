import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

class MultiGrainedSlice(object):
    def __init__(self, img_size, num_regions):
        self.img_size = img_size
        self.num_regions = num_regions
        assert img_size[0] % self.num_regions == 0, f"Image height={self.img_size[0]} cannot be divided by num_regions={self.num_regions}!"
        
    def __call__(self, img):
        """
        img: PIL.Image
        """
        img = np.asarray(img)
        imgs = np.stack(np.split(img, self.num_regions, axis=0), axis=0) # [num_regions, h, W, C]
        return imgs

class RandomMaskedHorizontalFlip(object):
    def __init__(self, h_patch, w_patch, p=0.5):
        self.h_patch = h_patch
        self.w_patch = w_patch
        self.p = p
    
    def __call__(self, img, mask):
        """
        img: [C, H, W] Tensor
        mask: [R, h_patch * w_patch] Tensor
        """
        
        if torch.rand(1) < self.p:
            R = mask.shape[0]
            mask = mask.reshape(R, self.h_patch, self.w_patch) # [R, h_patch, w_patch]
            mask = torch.flip(mask, dims=[-1]).reshape(R, -1) # [R, h_patch * w_patch]
            img = torch.flip(img, dims=[-1])
            return img, mask
        return img, mask
    
    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'