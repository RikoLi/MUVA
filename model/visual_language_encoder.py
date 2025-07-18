import logging
import torch
import torch.nn as nn
from model.clip_pat import clip
from model.ammsa import ResidualMaskPredictor
from .utils import *

logger = logging.getLogger("MGD")

class MultiGrainedPromptLearner(nn.Module):
    def __init__(self, cfg, num_classes, num_parts, dtype, token_embedding):
        super().__init__()
        ctx_dim = cfg.MGD.CONTEXT_DIM # default: 512
        ctx_num = cfg.MGD.CONTEXT_NUM # default: 4
        
        # part-aware context with part names
        prefix = "A photo of a"
        global_desc, part_desc = self.get_description(cfg.MGD.TEMPLATE_TYPE, cfg.MGD.DOMAIN_CONTEXT, prefix, ctx_num, cfg.MGD.PART_INDEX)
        logger.info(f'Full body description: {global_desc}')
        for p in part_desc:
            logger.info(f'Local body description: {p}')

        # use given words to initialize context vectors
        self.tokenized_prompts, embedding = self.tokenize_description([global_desc] + part_desc, dtype, token_embedding)
        
        self.embedding = embedding
        self.ctx_start_idx = len(prefix.split(' ')) + 1 # consider <sos> token
        
        # learnable context tokens
        learnable_ctx = torch.empty(num_classes, 1+num_parts, ctx_num, ctx_dim, dtype=dtype) # [N, 1+R, ctx_num, D]
        nn.init.normal_(learnable_ctx, std=cfg.MGD.CONTEXT_INIT_STD) # 0.002 seems to perform better
        self.learnable_ctx = nn.Parameter(learnable_ctx) # [N, 1+R, ctx_num, D]
        
        # others
        self.num_classes = num_classes
        self.ctx_num = ctx_num
        self.num_parts = num_parts
        
    def tokenize_description(self, texts, dtype, token_embedding):
        texts = list(map(lambda t: t.replace('_', ' '), texts))
        tokenized_prompts = clip.tokenize(texts).cuda()
        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        return tokenized_prompts, embedding
        
    def get_description(self, template_type, domain_context, prefix, ctx_num, part_idx):
        assert template_type in ('universal', 'part-agnostic')
        logger.info(f'Template type: {template_type}')
        
        prefix = f"{prefix} {' '.join(ctx_num * ['*'])}"
        if template_type == 'universal':
            global_desc = f"{prefix} person"
            part_desc = self.get_part_description(prefix, part_idx)
        elif template_type == 'part-agnostic':
            global_desc = f"{prefix}"
            part_desc = [global_desc] * len(part_idx)
            
        if domain_context == '':
            logger.info('No domain context is used.')
        else:
            global_desc = f"{global_desc}, {domain_context}"
            part_desc = [f"{desc}, {domain_context}" for desc in part_desc]

        return global_desc, part_desc
        
    
    def get_part_description(self, prefix, part_idx):
        assert isinstance(part_idx, list), 'Part index should be a list of integers!'
        part_idx.sort()
        
        part_desc = []
        if 0 in part_idx:
            part_desc.append(f"{prefix} head of a person")
        if 1 in part_idx:
            part_desc.append(f"{prefix} upper body of a person")
        if 2 in part_idx:
            part_desc.append(f"{prefix} legs of a person")
        
        return part_desc        

    def forward(self, label):
        b = label.shape[0]
        prompt = self.embedding.unsqueeze(0).repeat(b, 1, 1, 1) # [B, 1+R, 77, D]
        
        # replace with learnable context tokens
        ctx = self.learnable_ctx[label] # [B, 1+R, ctx_num, D]
        prompt[:, :, self.ctx_start_idx:self.ctx_start_idx+self.ctx_num, :] = ctx
        
        return prompt

class MultiGrainedTextEncoder(nn.Module):
    def __init__(self, clip_model, num_parts, prompt_learner):
        super().__init__()
        eos_idx = prompt_learner.tokenized_prompts.argmax(dim=-1) # [1+R]
        self.num_parts = num_parts
        self.eos_idx = eos_idx
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding # [77, D]
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
    def forward(self, prompts):
        B, R, L, D = prompts.shape
        prompts = prompts.reshape(B*R, L, D) # [B, 1+R, 77, D] -> [B*(1+R), 77, D]
        x = prompts + self.positional_embedding.type(self.dtype) # [B*(1+R), 77, D]
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) # [B*(1+R), 77, D]

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x.reshape(B, R, L, D) # [B, 1+R, L, D]
        x = x[:, torch.arange(R), self.eos_idx, :] @ self.text_projection # [B, 1+R, D]
        return x


class ViLaMDGroundingMask(nn.Module):
    def __init__(self, cfg, num_classes, cam_num, view_num) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.num_cams = cam_num
        self.num_views = view_num
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.num_parts = len(cfg.MGD.PART_INDEX)
        self.num_patches = self.h_resolution * self.w_resolution
        
        # attention mask
        # use no default fixed attention mask
        # mask should be dynamically fed in forwarding
        clip_model = load_clip_to_cpu_pat(cfg.MODEL.NAME, self.h_resolution, self.w_resolution, cfg.MODEL.STRIDE_SIZE[0], attn_mask=None).cuda()
        
        # encoders
        self.image_encoder = clip_model.visual
        self.prompt_learner = MultiGrainedPromptLearner(cfg, self.num_classes, self.num_parts, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = MultiGrainedTextEncoder(clip_model, self.num_parts, self.prompt_learner)
        
        # local token
        # use totally new initialized param
        if cfg.MGD.LOCAL_TOKEN == 'random':
            self.local_token = nn.Parameter(torch.zeros(self.num_parts, self.image_encoder.transformer.width)) # [R, D]
            trunc_normal_(self.local_token, std=cfg.MGD.LOCAL_TOKEN_INIT_STD) # default: 0.02, maybe 0.002 is better?
            logger.info('Local token is initialized by trunc_normal.')
        elif cfg.MGD.LOCAL_TOKEN == 'patch_avg':
            self.local_token = nn.Parameter(torch.zeros(self.num_parts, self.image_encoder.transformer.width)) # placeholder
            logger.info('Local token will be initialized in forwarding with patches.')
        elif cfg.MGD.LOCAL_TOKEN == 'cls':
            self.local_token = nn.Parameter(self.image_encoder.class_embedding.data.clone().view(1,-1).repeat(self.num_parts, 1))
            logger.info(f'Local token is copied from image CLS token with new shape {self.local_token.shape}')
        
        # rebuild PE
        pe_data = clip_model.visual.positional_embedding.data # [1+L, D]
        local_emb = pe_data[0].unsqueeze(0).repeat(self.num_parts, 1) # [R, D]
        pe_data = torch.cat([pe_data[:1], local_emb, pe_data[1:]], dim=0) # [1+R+L, D]
        clip_model.visual.positional_embedding = nn.Parameter(pe_data).cuda() # [1+R+L, D]
        logger.info(f'Positional embedding is rebuild with shape {clip_model.visual.positional_embedding.shape}')
        
        self.mask_template = self.create_mask_template()
        self.residual_mask_predictor = ResidualMaskPredictor(len(self.image_encoder.transformer.resblocks),
                                                             self.num_parts,
                                                             self.num_patches,
                                                             self.image_encoder.class_embedding.shape[-1],
                                                             tau=cfg.MGD.RMP_SIGMOID_TAU,
                                                             num_heads=cfg.MGD.RMP_NUM_HEADS,
                                                             use_layer_norm=cfg.MGD.RMP_USE_LAYER_NORM,
                                                             use_inner_shortcut=cfg.MGD.RMP_USE_INNER_SHORTCUT)
        
        if self.cfg.MGD.MASK_GATING_TYPE == 'offline':
            logger.info(f'Using offline mask gating, threshold={self.cfg.MGD.MASK_GATING_THRESH}')
        elif self.cfg.MGD.MASK_GATING_TYPE == 'online':
            logger.info(f'Using online mask gating, max value={self.cfg.MGD.MASK_MAX_VALUE}')
        else:
            raise(f'Invalid mask gating type {self.cfg.MGD.MASK_GATING_TYPE}!')
        
        # classification head
        self.fc = nn.Linear((1+self.num_parts)*self.image_encoder.output_dim, num_classes, bias=False)
        self.fc.apply(weights_init_classifier)
        logger.info(f'FC classifier check: {self.fc}')
        
        
        
        # bnneck
        self.bnneck = nn.BatchNorm1d((1+self.num_parts)*self.image_encoder.output_dim)
        self.bnneck.apply(self._init_bnneck)
        logger.info(f'BN neck check: {self.bnneck}')
        
        
        
        self.prepare_patch_projection()
        
    def get_param_num(self, name=None):
        """
        Get the number of parameters in the model.
        """
        if name is not None:
            if hasattr(self, name):
                num_params = sum(p.numel() for p in getattr(self, name).parameters())
                return num_params
            else:
                return 0
        else:
            total_params = sum(p.numel() for p in self.parameters())
            return total_params
        
    def prepare_patch_projection(self):
        """
        Freeze patch projection for improved stability as a trick. This should be set in the config file.
        Paper: https://arxiv.org/pdf/2104.02057.pdf
        """
        
        if self.cfg.MODEL.FREEZE_PATCH_PROJ:
            for _, v in self.image_encoder.conv1.named_parameters():
                v.requires_grad_(False)
            logger.info('freeze patch projection layer with shape {}'.format(self.image_encoder.conv1.weight.shape))
        else:
            logger.info('do not freeze patch projection layer')
        
        
    def get_local_token_from_patches(self, x):
        """x: BLD"""
        B, L, D = x.shape
        patches = x.unsqueeze(2).reshape(B, self.h_resolution, -1, D) # [B, H, W, D]
        W = patches.shape[2]
        patches = patches.unsqueeze(1).reshape(B, self.num_parts, -1, W, D) # [B, R, H/R, W, D]
        patches = patches.reshape(B, self.num_parts, -1, D) # [B, R, HW/R, D]
        local_tokens = patches.mean(dim=2) # [B, R, D]
        return local_tokens
    
    def mask_gating(self, pred_mask):
        """
        pred_mask: [B, R, L]
        """
        gating_type = self.cfg.MGD.MASK_GATING_TYPE
        if gating_type == 'offline':
            # mask is gated without gradient
            thresh = self.cfg.MGD.MASK_GATING_THRESH
            pred_mask = pred_mask.detach() # cut off gradient
            gated_mask = torch.full(pred_mask.shape, float('-inf')).type_as(pred_mask)
            gated_mask[pred_mask > thresh] = 0
        elif gating_type == 'online':
            # mask is gated with gradient
            mask_max_val = self.cfg.MGD.MASK_MAX_VALUE
            gated_mask = pred_mask * 2 * mask_max_val - mask_max_val # range: [-max_val, max_val]
        else:
            raise ValueError(f'Invalid gating type {gating_type}!')
        
        return gated_mask
    
    def create_mask_template(self):
        R = self.num_parts
        L = self.num_patches
        
        grounding_mask_placeholder = torch.full((R, L), fill_value=float('-inf'))
        
        cls2cls = torch.zeros(1, 1) # [1, 1]
        cls2local = torch.zeros(1, R + L) # [1, R+L]
        diag = torch.full((R, R), fill_value=float('-inf')) # [R, R]
        for i in range(R):
            diag[i, i] = 0.0
        full = torch.zeros(L, L) # [L, L]
        
        tmp1 = torch.cat([diag, grounding_mask_placeholder], dim=1) # [R, R+L]
        tmp2 = torch.cat([grounding_mask_placeholder.t(), full], dim=1) # [L, R+L]
        tmp3 = torch.cat([tmp1, tmp2], dim=0) # [R+L, R+L]
        tmp4 = torch.cat([cls2local.t(), tmp3], dim=1) # [R+L, 1+R+L]
        mask_template = torch.cat([torch.cat([cls2cls, cls2local], dim=1), tmp4], dim=0) # [1+R+L, 1+R+L]
        
        return mask_template
        
    
    def convert_to_attn_mask(self, mask):
        """
        Convert [B, R, L] -> [B*n_heads, 1+R+L, 1+R+L]
        mask: [B, R, L]
        """
        B, R, L = mask.shape
        n_heads = self.image_encoder.transformer.heads
        
        out_mask = self.mask_template.unsqueeze(0).repeat(B, 1, 1).type_as(mask) # [B, 1+R+L, 1+R+L]
        out_mask[:, 1:R+1, 1+R:] = mask
        out_mask[:, 1+R:, 1:R+1] = mask.transpose(1, 2)
        out_mask = out_mask.unsqueeze(1).repeat(1, n_heads, 1, 1).view(B*n_heads, 1+R+L, -1) # [B*n_heads, 1+R+L, 1+R+L]
        
        return out_mask
        
    def forward_image(self, img, gt_mask=None):
        x = self.image_encoder.conv1(img)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        cls_token = self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device) # [B, 1, D]
        if self.cfg.MGD.LOCAL_TOKEN == 'patch_avg':
            # average patches as local token
            local_token = self.get_local_token_from_patches(x) # [B, R, D]
        else:
            local_token = self.local_token.expand(x.shape[0], -1, -1) # [B, R, D]
        x = torch.cat([cls_token, local_token, x], dim=1)  # shape = [B, 1+R+L, D]
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        num_layers = self.image_encoder.transformer.layers
        all_pred_masks = []
        all_pred_masks_logit = []
        
        if gt_mask is not None:
            # use ground truth mask
            gt_mask = self.convert_to_attn_mask(gt_mask)
            for i in range(num_layers):
                x = self.image_encoder.transformer.resblocks[i](x, gt_mask)
        else:
            # use predicted mask
            for i in range(num_layers):
                if i == 0:
                    mask = None
                pred_mask, mask = self.residual_mask_predictor(x, i, mask)
                gated_mask = self.mask_gating(pred_mask)
                gated_mask = self.convert_to_attn_mask(gated_mask)
                x = self.image_encoder.transformer.resblocks[i](x, gated_mask)
                all_pred_masks.append(pred_mask)
                all_pred_masks_logit.append(mask)
        
        x = x.permute(1, 0, 2)  # LND -> NLD  

        x = self.image_encoder.ln_post(x)  

        if self.image_encoder.proj is not None:
            xproj = x @ self.image_encoder.proj
            
        if len(all_pred_masks) > 0:
            all_pred_masks = torch.stack(all_pred_masks, dim=0)
            all_pred_masks_logit = torch.stack(all_pred_masks_logit, dim=0)

        return x, xproj, all_pred_masks, all_pred_masks_logit
    
    def forward_text(self, label):
        B = label.shape[0]
        prompts = self.prompt_learner(label)
        text_features = self.text_encoder(prompts) # [B, 1+R, D]
        text_features = text_features.view(B, -1) # [B, (1+R)*D]
        return text_features

    
    def forward(self, img=None, label=None, gt_mask=None, get_image=False, get_text=False):
        """
        img: [B, C, H, W]
        """
        assert get_image or get_text
        
        if get_image:
            _, xproj, all_pred_masks, all_pred_masks_logit = self.forward_image(img, gt_mask) # [B, 1+R+L, D]
            B = xproj.shape[0]
            xproj = xproj[:,:1+self.num_parts,:] # [B, 1+R, D]
            xproj = xproj.view(B, -1) # [B, (1+R)*D]
            
            bn_xproj = self.bnneck(xproj)
            logit = self.fc(bn_xproj)
            
            return logit, xproj, all_pred_masks, all_pred_masks_logit
        
        if get_text:
            return self.forward_text(label)
    
    @torch.no_grad()
    def infer_image(self, img, gt_mask=None, after_bn=True):
        """
        img: [B, C, H, W]
        """
        _, xproj, _, _ = self.forward_image(img, gt_mask) # [B, 1+R+L, D]
        B = xproj.shape[0]
        xproj = xproj[:,:1+self.num_parts,:] # [B, 1+R, D]
        xproj = xproj.view(B, -1) # [B, (1+R)*D]
        
        bn_xproj = self.bnneck(xproj)
        
        if after_bn:
            return bn_xproj
        else:
            return xproj
        
    @torch.no_grad()
    def predict_mask(self, img):
        _, _, all_pred_masks, _ = self.forward_image(img, None)
        return all_pred_masks
        
        
    @torch.no_grad()
    def infer_text(self, label):
        return self.forward_text(label)
    
    def load_pretrained_prompt_learner(self, path):
        param_dict = torch.load(path, map_location='cpu')
        for k, v in param_dict.items():
            if 'prompt_learner' in k:
                self.state_dict()[k].copy_(v)
                logger.info('============ Load Pretrained Prompt Learner ============')
                logger.info(f'Key: {k}, Value shape: {v.shape}')
    
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        logger.info('Loading pretrained model from {}'.format(trained_path))
    
    def load_param_inference(self, trained_path):
        cnt = 0
        param_dict = torch.load(trained_path, map_location='cpu')
        for i in param_dict:
            if 'text_encoder' in i or 'prompt_learner' in i or 'fc' in i:
                if 'image_encoder' not in i:
                    cnt += 1
                    continue # ignore num_class related layers
            self.state_dict()[i].copy_(param_dict[i])
        logger.info(f'Loading pretrained model from {trained_path}, ignore {cnt} layers.')
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def _init_bnneck(self, m):
        if isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
            m.bias.requires_grad_(False)
            
def make_vilamd_grounding_mask(cfg, num_classes, cam_num, view_num):
    model = ViLaMDGroundingMask(cfg, num_classes, cam_num, view_num)
    return model