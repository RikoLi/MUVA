import logging
import os
import time
from datetime import timedelta
import torch
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
import torch.amp as amp
from .utils import *
from losses.ce_loss import CrossEntropyLabelSmooth
from losses.triplet_loss import TripletLoss
from losses.cm import ClusterMemoryAMP
import tensorboardX as tbx
import torch.nn as nn

def train_baseline_pcl(cfg,
            model,
            train_loader,
            cluster_loader,
            val_loader,
            optimizer,
            scheduler,
            num_query):
    
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("MGD")
    logger.info('start training')
    
    model.to(device)
    
    
    meters = {}
    for k in ['total', 'ce', 'pcl', 'ce_acc']:
        meters[k] = AverageMeter()
    xent = CrossEntropyLabelSmooth(model.num_classes)
    
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    summary = tbx.SummaryWriter(cfg.OUTPUT_DIR)
    scaler = amp.GradScaler()
    
    all_start_time = time.monotonic()
    
    
    # ######################################################################
    # Train image encoder
    # ######################################################################
    cnt = 0
    for epoch in range(1, epochs+1):
        for v in meters.values():
            v.reset() # reset each meter
        evaluator.reset()
        scheduler.step(epoch)
        logger.info('Learning rate is changed to {:.2e}'.format(scheduler._get_lr(epoch)[0]))
        
        # create memory bank
        model.eval()
        image_features = []
        gt_labels = []
        with torch.no_grad():
            for _, (img, pid, camid, _) in enumerate(tqdm.tqdm(cluster_loader, desc='Extract image features')):
                img = img.cuda()
                target = pid.cuda()
                with amp.autocast('cuda', torch.float16, enabled=True):
                    out = model.infer_image(img, after_bn=True)
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
        logger.info(f'PCL Memory shape: {memory.features.shape}')
        
        
        model.train()
        tloader = tqdm.tqdm(train_loader, total=len(train_loader))
        for n_iter, (img, target, _, _) in enumerate(tloader):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device)
            
            loss_dict = {}             
            with amp.autocast('cuda', torch.float16, enabled=True):
                fc_logit, feat = model(img)
                feat_bn = F.normalize(model.bnneck(feat), dim=1)
                
                loss_dict['ce'] = xent(fc_logit, target) * cfg.MODEL.ID_LOSS_WEIGHT
                loss_dict['pcl'] = memory(feat_bn, target) * cfg.MODEL.PCL_LOSS_WEIGHT
                
                loss = sum([v for v in loss_dict.values()])
                loss_dict['total'] = loss
                
                
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            # udpate meters
            for k in meters.keys():
                if k == 'ce_acc':
                    acc = (fc_logit.max(1)[1] == target).float().mean()
                    meters[k].update(acc, 1)
                else:
                    meters[k].update(loss_dict[k].item(), img.shape[0])

            torch.cuda.synchronize()
            
            tloader.set_description('Epoch [{}/{}]: ReID loss={:.4e}, ce acc={:.1%}'.format(
                epoch,
                epochs,
                meters['ce'].avg+meters['pcl'].avg,
                meters['ce_acc'].avg
            ))
            
            for k, v in loss_dict.items():
                summary.add_scalar(f'{k}_loss', v.item(), global_step=cnt)
            cnt += 1
            
            
        logger.info("Epoch {} done.".format(epoch))
        
        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            
        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, _) in enumerate(val_loader):
                with torch.no_grad():
                    with amp.autocast('cuda', torch.float16, enabled=True):
                        img = img.to(device)
                        feat_bn = model.infer_image(img, after_bn=True)
                    evaluator.update((feat_bn, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info(f'Image encoder training done in {total_time}.')
    print(cfg.OUTPUT_DIR)
    
def train_baseline_pcl_lora(cfg,
            model,
            train_loader,
            cluster_loader,
            val_loader,
            optimizer,
            scheduler,
            num_query):
    
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("logger")
    logger.info('start training')
    
    model.to(device)
    
    
    meters = {}
    for k in ['total', 'ce', 'pcl', 'ce_acc']:
        meters[k] = AverageMeter()
    xent = CrossEntropyLabelSmooth(model.num_classes)
    
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    summary = tbx.SummaryWriter(cfg.OUTPUT_DIR)
    scaler = amp.GradScaler()
    
    all_start_time = time.monotonic()
    
    
    # ######################################################################
    # Train image encoder
    # ######################################################################
    cnt = 0
    for epoch in range(1, epochs+1):
        for v in meters.values():
            v.reset() # reset each meter
        evaluator.reset()
        scheduler.step(epoch)
        logger.info('Learning rate is changed to {:.2e}'.format(scheduler._get_lr(epoch)[0]))
        
        # create memory bank
        model.eval()
        image_features = []
        gt_labels = []
        with torch.no_grad():
            for _, (img, pid, camid, _) in enumerate(tqdm.tqdm(cluster_loader, desc='Extract image features')):
                img = img.cuda()
                target = pid.cuda()
                with amp.autocast('cuda', torch.float16, enabled=True):
                    out = model.infer_image(img, after_bn=True)
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
        logger.info(f'PCL Memory shape: {memory.features.shape}')
        
        
        model.train()
        tloader = tqdm.tqdm(train_loader, total=len(train_loader))
        for n_iter, (img, target, _, _) in enumerate(tloader):
            optimizer.zero_grad()
            img = img.to(device)
            target = target.to(device)
            
            loss_dict = {}             
            with amp.autocast('cuda', torch.float16, enabled=True):
                fc_logit, feat = model(img)
                feat_bn = F.normalize(model.bnneck(feat), dim=1)
                
                loss_dict['ce'] = xent(fc_logit, target) * cfg.MODEL.ID_LOSS_WEIGHT
                loss_dict['pcl'] = memory(feat_bn, target) * cfg.MODEL.PCL_LOSS_WEIGHT
                
                loss = sum([v for v in loss_dict.values()])
                loss_dict['total'] = loss
                
                
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            # udpate meters
            for k in meters.keys():
                if k == 'ce_acc':
                    acc = (fc_logit.max(1)[1] == target).float().mean()
                    meters[k].update(acc, 1)
                else:
                    meters[k].update(loss_dict[k].item(), img.shape[0])

            torch.cuda.synchronize()
            
            tloader.set_description('Epoch [{}/{}]: ReID loss={:.4e}, ce acc={:.1%}'.format(
                epoch,
                epochs,
                meters['ce'].avg+meters['pcl'].avg,
                meters['ce_acc'].avg
            ))
            
            for k, v in loss_dict.items():
                summary.add_scalar(f'{k}_loss', v.item(), global_step=cnt)
            cnt += 1
            
            
        logger.info("Epoch {} done.".format(epoch))
        
        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            
        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    feat_bn = model.infer_image(img, after_bn=True)
                    evaluator.update((feat_bn, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info(f'Image encoder training done in {total_time}.')
    print(cfg.OUTPUT_DIR)