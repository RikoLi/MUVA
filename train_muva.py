from utils.logger import setup_logger
import random
import torch
import numpy as np
import os
import sys
import argparse
from configs import cfg as cfg
from engine.coop_trainers import train_coop_stage1_memory, train_coop_stage2_memory
from model.visual_language_encoder import make_muva
from engine.datasets.dataloader import make_coop_dataloader, make_val_dataloader
from engine.solvers.optimizers import make_optimizer_coop_1stage, make_optimizer_coop_2stage, make_optimizer_warmup
from engine.solvers.schedulers import WarmupMultiStepLR, create_cosine_scheduler
from termcolor import colored

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="MGD Training")
    parser.add_argument(
        "--config_file", default="configs/pcl_baseline.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)
    
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger = setup_logger("MGD", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))
    
    train_loader1, train_loader2, val_loader, num_queries, num_classes, cam_num, view_num = make_coop_dataloader(cfg)
    
    # evaluate on given dataset only in single-source DG
    if len(cfg.DATASETS.NAMES) == 1 and cfg.DATASETS.EVAL_DATASET != '':
        logger.info(f'Use {cfg.DATASETS.EVAL_DATASET} for evaluation.')
        val_loader, num_queries = make_val_dataloader(cfg)
    
    # Create models
    model = make_muva(cfg, num_classes, cam_num, view_num)
    num_params = model.get_param_num()
    logger.info(colored(f'Number of parameters in the model: {num_params / 1e6:.1f}M', 'green'))
           
    # optimizers for stage1
    stage1_pretrain_path = cfg.SOLVER.STAGE1.PRETRAIN
    if stage1_pretrain_path == '':
        optimizer1 = make_optimizer_coop_1stage(cfg, model)
        scheduler1 = create_cosine_scheduler(optimizer1,
                                            num_epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS,
                                            lr_min = cfg.SOLVER.STAGE1.LR_MIN,
                                            warmup_lr_init = cfg.SOLVER.STAGE1.WARMUP_LR_INIT,
                                            warmup_t = cfg.SOLVER.STAGE1.WARMUP_EPOCHS,
                                            noise_range = None)
        
        train_coop_stage1_memory(
            cfg,
            model,
            train_loader1,
            optimizer1,
            scheduler1
        )
    else:
        assert os.path.exists(stage1_pretrain_path), f'Weight {stage1_pretrain_path} doesn\'t exist!'
        logger.info(f'Load stage 1 pre-trained weight from: {stage1_pretrain_path}')
        model.load_pretrained_prompt_learner(stage1_pretrain_path)
    
    # optimizer for stage2
    optimizer2 = make_optimizer_coop_2stage(cfg, model)
    scheduler2 = WarmupMultiStepLR(optimizer2, milestones=cfg.SOLVER.STAGE2.STEPS, gamma=cfg.SOLVER.STAGE2.GAMMA,
                                  warmup_factor=cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                  warmup_iters=cfg.SOLVER.STAGE2.WARMUP_ITERS,
                                  warmup_method=cfg.SOLVER.STAGE2.WARMUP_METHOD)
    
    train_coop_stage2_memory(
        cfg,
        model,
        train_loader2,
        val_loader,
        train_loader1,
        optimizer2,
        scheduler2,
        num_queries
    )
    
    # save running config
    new_cfg = cfg.clone()
    new_cfg.defrost()
    new_cfg.OUTPUT_DIR = 'logs/debug'
    new_cfg.freeze()
    save_path = os.path.join(output_dir, 'running_config.yml')
    with open(save_path, 'w') as f:
        old_stdout = sys.stdout
        sys.stdout = f
        print(new_cfg)
        sys.stdout = old_stdout
    print('Running config is saved.')