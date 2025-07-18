import torch
import torch.nn as nn
from model.clip_pat.model import QuickGELU
from .utils import *

class MaskPredictor(nn.Module):
    def __init__(self, dim, num_heads, num_patches, use_layer_norm=False, use_inner_shortcut=False):
        super().__init__()
        self.cross_attn = CrossAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, num_patches),
            QuickGELU(),
            nn.Linear(num_patches, num_patches)
        )
        self.use_inner_shortcut = use_inner_shortcut
        self.use_layer_norm = use_layer_norm
        
        if self.use_layer_norm:
            self.ln_attn1 = nn.LayerNorm(dim)
            self.ln_attn2 = nn.LayerNorm(dim)
            self.ln_mlp = nn.LayerNorm(dim)
        else:
            self.ln_attn1 = nn.Identity()
            self.ln_attn2 = nn.Identity()
            self.ln_mlp = nn.Identity()
        
        self.mlp.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        
    def forward(self, x_part, x_patch):
        x_out = self.cross_attn(self.ln_attn1(x_part), self.ln_attn2(x_patch)) # [B, R, D]
        if self.use_inner_shortcut:
            x_out = x_out + x_part
        mask = self.mlp(self.ln_mlp(x_out)) # [B, R, L]
        return mask
    
class ResidualMaskPredictor(nn.Module):
    def __init__(self, num_layers, num_parts, num_patches, dim, num_heads,
                 tau=1.0, use_layer_norm=False, use_inner_shortcut=False):
        super().__init__()
        self.num_layers = num_layers
        self.num_parts = num_parts
        self.dim = dim
        self.tau = tau
        self.blocks = nn.ModuleList([MaskPredictor(dim, num_heads, num_patches, use_layer_norm, use_inner_shortcut) for _ in range(self.num_layers)])
        
    def forward(self, x, layer_idx, prev_mask=None):
        """
        prev_mask: None (for first block) or [B, R, L]
        x: [1+R+L, B, D], sequence first
        """
        x = x.permute(1, 0, 2) # batch first
        x_part = x[:, 1:1+self.num_parts, :] # [B, R, D]
        x_patch = x[:, 1+self.num_parts:, :] # [B, L, D]
        
        mask = self.blocks[layer_idx](x_part, x_patch) # [B, R, L]
        
        if prev_mask is not None:
            mask = mask + prev_mask
            
        pred_mask = torch.sigmoid(mask.div(self.tau))
        
        return pred_mask, mask