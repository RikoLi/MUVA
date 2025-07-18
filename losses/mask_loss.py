import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        self.dice_fn = None
        
        if loss_type == 'mse':
            self.loss_fn = MSEMaskLoss()
        elif loss_type == 'bce+dice':
            self.loss_fn = BCEMaskLoss()
            self.dice_fn = DiceLoss()
        else:
            raise ValueError(f'Invalid loss type {loss_type}!')
        
    def forward(self, pred_mask, pred_mask_logit, gt_mask):
        mask_loss = self.loss_fn(pred_mask, pred_mask_logit, gt_mask)
        dice_loss = torch.tensor(0.0).type_as(mask_loss)
        
        if self.dice_fn is not None:
            dice_loss = self.dice_fn(pred_mask, gt_mask)
            
        return mask_loss, dice_loss

class MSEMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_mask, pred_mask_logit, gt_mask):
        """
        pred_mask: List[Tensor], shape=[B, R, L]
        gt_mask: [B, R, L]
        """
        gt_mask = gt_mask.bool().logical_not().float().unsqueeze(0) # [1, B, R, L]
        loss = (pred_mask - gt_mask) ** 2
        return loss.mean()

class BCEMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_mask, pred_mask_logit, gt_mask):
        gt_mask = gt_mask.bool().logical_not().float().unsqueeze(0).expand(len(pred_mask_logit), -1, -1, -1) # [n_layers, B, R, L]
        bce_loss = F.binary_cross_entropy_with_logits(pred_mask_logit.reshape(-1), gt_mask.reshape(-1))
        return bce_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, scale_factor=1000):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.scale_factor = scale_factor

    def forward(self, pred_mask, gt_mask):
        # import ipdb; ipdb.set_trace()
        L = pred_mask.shape[-1]
        gt_mask = gt_mask.bool().logical_not().float().unsqueeze(0).expand(len(pred_mask), -1, -1, -1) # [n_layers, B, R, L]
        
        pred_mask = pred_mask.reshape(-1, L)
        gt_mask = gt_mask.reshape(-1, L)
        
        intersection = pred_mask * gt_mask / self.scale_factor
        union = (pred_mask / self.scale_factor).sum(dim=1) + (gt_mask / self.scale_factor).sum(dim=1)
        dice = (2 * intersection.sum(dim=1) + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        return dice_loss