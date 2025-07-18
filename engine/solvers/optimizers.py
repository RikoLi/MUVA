import logging
import torch
from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger("MGD")

def make_optimizer(cfg, model):
    params = []
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        
    return optimizer

def make_optimizer_lora(cfg, model):
    params = []
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    
    # train LoRA adapters only for the image encoder
    for key, value in model.image_encoder.named_parameters():
        if 'lora_module' in key:
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            logger.info(key)
        else:
            value.requires_grad_(False)
            
    # train other params in classifier & bnneck and so on
    for key, value in model.named_parameters():
        if 'image_encoder' in key or not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        logger.info(key)
        
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        
    return optimizer

def make_cascade_clip_encoder_optimizer(cfg, model):
    params = []
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    
    for key, value in model.named_parameters():
        if 'text_encoder' in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        
    return optimizer

def make_optimizer_coop_1stage(cfg, model):
    params = []
    learnable_keys = []
    for key, value in model.named_parameters():
        if "prompt_learner" in key:
            lr = cfg.SOLVER.STAGE1.BASE_LR
            weight_decay = cfg.SOLVER.STAGE1.WEIGHT_DECAY
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            learnable_keys.append(key)
    logger.info(f"Learnable params in stage1: {learnable_keys}")
    if cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE1.MOMENTUM)
    elif cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif cfg.SOLVER.STAGE1.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE1.OPTIMIZER_NAME)(params)
    return optimizer

def make_optimizer_coop_2stage(cfg, model):
    params = []
    lr = cfg.SOLVER.STAGE2.BASE_LR
    weight_decay = cfg.SOLVER.STAGE2.WEIGHT_DECAY
    for key, value in model.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue   
        elif "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        elif "residual_mask_predictor" in key:
            params += [{"params": [value], "lr": lr * cfg.MGD.RMP_LR_MULT, "weight_decay": 0}]
        elif not value.requires_grad:
            continue
        else:
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.STAGE2.MOMENTUM)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif cfg.SOLVER.STAGE2.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.STAGE2.OPTIMIZER_NAME)(params)

    return optimizer

def make_optimizer_warmup(cfg, model):
    params = []
    trained_keys = []
    lr = cfg.SOLVER.WARMUP.BASE_LR
    weight_decay = cfg.SOLVER.WARMUP.WEIGHT_DECAY
    
    lora_config = LoraConfig(
        task_type=None,
        r=cfg.SOLVER.WARMUP.LORA_RANK,
        lora_alpha=cfg.SOLVER.WARMUP.LORA_ALPHA,
        target_modules=[f'image_encoder.transformer.resblocks.{i}.attn' for i in range(12)] + \
            [f'image_encoder.transformer.resblocks.{i}.mlp.c_fc' for i in range(12)] + \
            [f'image_encoder.transformer.resblocks.{i}.mlp.c_proj' for i in range(12)]
    )
    model = get_peft_model(model, lora_config)
    
    for key, value in model.named_parameters():
        if '.fc.' in key or 'bnneck.weight' in key:
            value.requires_grad_(True) # classifier and bnneck are trainable in warmup
        if value.requires_grad:
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            trained_keys.append(key)
    logger.info(f"Learnable params in warmup: {trained_keys}")
    
    if cfg.SOLVER.WARMUP.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.WARMUP.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.WARMUP.MOMENTUM)
    elif cfg.SOLVER.WARMUP.OPTIMIZER_NAME == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif cfg.SOLVER.WARMUP.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.WARMUP.OPTIMIZER_NAME)(params)
    
    return optimizer, model