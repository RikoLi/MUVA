import os
import tqdm
import logging
import time
import torch
import torch.amp as amp
import torch.nn.functional as F
import tensorboardX as tbx
from datetime import timedelta
from termcolor import colored
from .utils import compute_cluster_centroids, pk_sampling
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.perf_monitor import Timer
from losses.ce_loss import CrossEntropyLabelSmooth
from losses.cm import ClusterMemoryAMP
from losses.mask_loss import MaskLoss

def train_coop_stage1_memory(
        cfg,
        model,
        train_loader,
        optimizer,
        scheduler
    ):
    
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH

    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS

    logger = logging.getLogger("MGD")
    logger.info('start training')
    
    model.to(device)
    part_idx = torch.LongTensor(cfg.MGD.PART_INDEX).to(device)
    
    
    meters = {}
    meters['total'] = AverageMeter()
    meters['cluster_nce'] = AverageMeter()
    meters['i2t_acc'] = AverageMeter()
    
    summary = tbx.SummaryWriter(cfg.OUTPUT_DIR)
    scaler = amp.GradScaler()
    
    all_start_time = time.monotonic()
    
    # ######################################################################
    # Step1: Extract all image features & Create memory bank
    # ######################################################################
    model.eval()
    image_features_list = []
    gt_labels_list = []
    with torch.no_grad():
        for _, (img, pid, camid, _, mask) in enumerate(tqdm.tqdm(train_loader, desc='Extract image features')):
            img = img.to(device)
            target = pid.to(device)
            mask = mask.to(device)[:, part_idx, :] # select chosen parts
            with amp.autocast('cuda', torch.float16, enabled=True):
                out = model.infer_image(img, gt_mask=mask, after_bn=False) # [B, (1+R)*D]
            image_features_list.append(out)
            gt_labels_list.append(target)
    image_features_list = torch.cat(image_features_list, dim=0) # [N, (1+R)*D]
    gt_labels_list = torch.cat(gt_labels_list, dim=0)
    
    image_centroids = compute_cluster_centroids(image_features_list, gt_labels_list).to(device)
    logger.info(f'Image feature memory shape: {image_centroids.shape}')
    
    
    
    # ######################################################################
    # Step2: Fine-tune prompt learner
    # ######################################################################
    timer = Timer('stage1')
    all_indices = torch.arange(len(gt_labels_list))
    cnt = 0
    for epoch in range(1, epochs+1):
        timer.reset()
        for v in meters.values():
            v.reset() # reset each meter
        scheduler.step(epoch)
        logger.info('Learning rate is changed to {:.2e}'.format(scheduler._get_lr(epoch)[0]))
        
        model.train()
        
        iterator, iters = pk_sampling(batch, k=cfg.DATALOADER.NUM_INSTANCE, pseudo_labels=gt_labels_list, samples=all_indices)
        tloader = tqdm.tqdm(range(iters), total=iters)
        torch.cuda.reset_max_memory_allocated()
        for n_iter in tloader:
            optimizer.zero_grad()
            
            indices = next(iterator) # grab a batch
            indices = indices.to(device)
            target = gt_labels_list[indices]
                            
            loss_dict = {}
            with timer:
                with amp.autocast('cuda', torch.float16, enabled=True):
                    text_features = model(label=target, get_text=True) # [B, (1+R)*D]
                    text_features = F.normalize(text_features, dim=1) # l2 norm
                    score = text_features @ image_centroids.t()
                    score = score / cfg.PCL.CLUSTER_NCE_TEMP
                    loss_dict['cluster_nce'] = F.cross_entropy(score, target)
                    
                loss = sum([v for v in loss_dict.values()])
                loss_dict['total'] = loss
                
                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()
            
            # udpate meters
            for k in meters.keys():
                if k == 'i2t_acc':
                    acc = (score.max(1)[1] == target).float().mean()
                    meters[k].update(acc, 1)
                else:
                    meters[k].update(loss_dict[k].item(), img.shape[0])

            torch.cuda.synchronize()
            
            tloader.set_description('Epoch [{}/{}]: ClusterNCE loss={:.4e}, i2t acc={:.1%}'.format(
                epoch,
                epochs,
                meters['cluster_nce'].avg,
                meters['i2t_acc'].avg
            ))
            
            for k, v in loss_dict.items():
                if k == 'i2t_acc':
                    summary.add_scalar(f'{k}', v.item(), global_step=cnt)
                else:
                    summary.add_scalar(f'{k}_loss', v.item(), global_step=cnt)
            cnt += 1
            
        logger.info("Epoch {} done.".format(epoch))
        logger.info(colored(f'Elapsed time per batch: {1000 * timer.average():.1f}ms', 'green'))
        logger.info(colored(f'Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 3):.1f}GB', 'green'))
        
        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
        
            
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info(f'Stage1 training done in {total_time}.')
    print(cfg.OUTPUT_DIR)    

def train_coop_stage2_memory(
    cfg,
    model,
    train_loader,
    val_loader,
    cluster_loader,
    optimizer,
    scheduler,
    num_query
    ):
    
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    iters = model.num_classes // batch

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("MGD")
    logger.info('start training')
    
    model.to(device)
    part_idx = torch.LongTensor(cfg.MGD.PART_INDEX,).to(device)
    
    
    meters = {}
    for k in ['total', 'ce', 'pcl', 'i2tce', 'mask', 'dice', 'i2tce_acc', 'ce_acc']:
        meters[k] = AverageMeter()
    xent = CrossEntropyLabelSmooth(model.num_classes)
    mask_loss_fn = MaskLoss(cfg.MGD.MASK_LOSS_TYPE)
    logger.info(f'Using mask loss = {cfg.MGD.MASK_LOSS_TYPE}')
    
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    summary = tbx.SummaryWriter(cfg.OUTPUT_DIR)
    scaler = amp.GradScaler()
    
    all_start_time = time.monotonic()
    
    # ######################################################################
    # Step1: Extract text prototypes
    # ######################################################################
    left = model.num_classes - batch * (model.num_classes//batch)
    if left != 0 :
        iters = iters+1
    text_features = []
    with torch.no_grad():
        for i in range(iters):
            if i+1 != iters:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, model.num_classes)
            with amp.autocast('cuda', torch.float16, enabled=True):
                text_feature = model.infer_text(label=l_list)
            text_features.append(text_feature)
        text_features = torch.cat(text_features, dim=0) # [N, (1+R)*D]
    
    
    # ######################################################################
    # Step2: Fine-tune image encoder
    # ######################################################################
    cnt = 0
    timer = Timer('stage2')
    for epoch in range(1, epochs+1):
        timer.reset()
        for v in meters.values():
            v.reset() # reset each meter
        evaluator.reset()
        scheduler.step(epoch)
        logger.info('Learning rate is changed to {:.2e}'.format(scheduler._get_lr(epoch)[0]))
        
        # Create PCL memory bank
        model.eval()
        image_features = []
        gt_labels = []
        with torch.no_grad():
            for _, (img, pid, camid, _, mask) in enumerate(tqdm.tqdm(cluster_loader, desc='Extract image features')):
                img = img.to(device)
                target = pid.to(device)
                mask = mask.to(device)[:, part_idx, :] # select chosen parts
                with amp.autocast('cuda', torch.float16, enabled=True):
                    out = model.infer_image(img, gt_mask=mask, after_bn=True)
                image_features.append(out)
                gt_labels.append(target)
        image_features = torch.cat(image_features, dim=0)
        gt_labels = torch.cat(gt_labels, dim=0)
        image_features = image_features.float()
        image_features = F.normalize(image_features, dim=1)
        memory = ClusterMemoryAMP(temp=cfg.PCL.MEMORY_TEMP,
                                  momentum=cfg.PCL.MEMORY_MOMENTUM,
                                  use_hard=cfg.PCL.HARD_MEMORY_UPDATE).to(device)
        memory.features = compute_cluster_centroids(image_features, gt_labels).to(device)
        logger.info(f'PCL memory shape: {memory.features.shape}')
        
        model.train()
        tloader = tqdm.tqdm(train_loader, total=len(train_loader))
        torch.cuda.reset_max_memory_allocated()
        for n_iter, (img, vid, _, _, mask) in enumerate(tloader):
            optimizer.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            mask = mask.to(device)[:, part_idx, :] # select chosen parts
                            
            loss_dict = {}
            with timer:
                with amp.autocast('cuda', torch.float16, enabled=True):
                    fc_logit, proj_feat, all_pred_masks, all_pred_masks_logit = model(img=img, label=target, gt_mask=None, get_image=True) # use predicted mask, not GT mask
                    if cfg.MODEL.NORM_I2TCE:
                        i2t_logit = F.normalize(proj_feat, dim=1) @ F.normalize(text_features, dim=1).t() # [B, N]
                    else:
                        i2t_logit = proj_feat @ text_features.t() # [B, N]
                    i2t_logit = i2t_logit.div(cfg.MODEL.I2TCE_TAU)
                    proj_feat_bn = model.bnneck(proj_feat) # after bn
                    proj_feat_bn = F.normalize(proj_feat_bn, dim=1) # l2 norm
                    loss_dict['ce'] = xent(fc_logit, target) * cfg.MODEL.ID_LOSS_WEIGHT
                    loss_dict['pcl'] = memory(proj_feat_bn, target) * cfg.MODEL.PCL_LOSS_WEIGHT
                    loss_dict['i2tce'] = xent(i2t_logit, target) * cfg.MODEL.I2TCE_LOSS_WEIGHT
                    
                    loss_mask, loss_dice = mask_loss_fn(all_pred_masks, all_pred_masks_logit, mask)
                    loss_dict['mask'] = loss_mask * cfg.MODEL.MASK_LOSS_WEIGHT
                    loss_dict['dice'] = loss_dice * cfg.MODEL.MASK_LOSS_WEIGHT
                    
                    
                    loss = sum([v for v in loss_dict.values()])
                    loss_dict['total'] = loss
                
                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()
            
            # udpate meters
            for k in meters.keys():
                if k == 'i2tce_acc':
                    acc = (i2t_logit.max(1)[1] == target).float().mean()
                    meters[k].update(acc, 1)
                elif k == 'ce_acc':
                    acc = (fc_logit.max(1)[1] == target).float().mean()
                    meters[k].update(acc, 1)
                else:
                    meters[k].update(loss_dict[k].item(), img.shape[0])

            torch.cuda.synchronize()
            
            tloader.set_description('Epoch [{}/{}]: ReID loss={:.4e}, i2t loss={:.4e}, mask loss={:.4e}, dice loss={:.4e}, ce acc={:.1%}, i2tce acc={:.1%}'.format(
                epoch,
                epochs,
                meters['ce'].avg + meters['pcl'].avg,
                meters['i2tce'].avg,
                meters['mask'].avg,
                meters['dice'].avg,
                meters['ce_acc'].avg,
                meters['i2tce_acc'].avg
            ))
            
            for k, v in loss_dict.items():
                if 'acc' in k:
                    summary.add_scalar(f'{k}', v.item(), global_step=cnt)
                else:
                    summary.add_scalar(f'{k}_loss', v.item(), global_step=cnt)
            cnt += 1
            
        logger.info("Epoch {} done.".format(epoch))
        logger.info(colored(f'Elapsed time per batch: {1000 * timer.average():.1f}ms', 'green'))
        logger.info(colored(f'Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 3):.1f}GB', 'green'))
        
        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage2_{}.pth'.format(epoch)))
            
        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, _) in enumerate(tqdm.tqdm(val_loader, 'Extraction')):
                with torch.no_grad():
                    with amp.autocast('cuda', torch.float16, enabled=True):
                        img = img.to(device)
                        feat_bn = model.infer_image(img, gt_mask=None, after_bn=cfg.TEST.AFTER_BN)
                    evaluator.update((feat_bn, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info(f'Stage2 training done in {total_time}.')
    print(cfg.OUTPUT_DIR)