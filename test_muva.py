from utils.logger import setup_logger
import random
import torch
import numpy as np
import os
import argparse
import tqdm
from configs import cfg_grounding_mask as cfg
from utils.metrics import R1_mAP_eval
from utils.perf_monitor import Timer
from model.visual_language_encoder import make_vilamd_grounding_mask
from engine.datasets.dataloader import make_coop_dataloader, make_val_dataloader
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
    model = make_vilamd_grounding_mask(cfg, num_classes, cam_num, view_num)
    model.load_param_inference(cfg.TEST.WEIGHT)
    num_params = model.get_param_num()
    logger.info(colored(f'Number of parameters in the model: {num_params / 1e6:.1f}M', 'green'))
    
    model.eval()
    model = model.to('cuda')
    timer = Timer('inference')
    evaluator = R1_mAP_eval(num_queries, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    torch.cuda.reset_max_memory_allocated()
    for n_iter, (img, vid, camid, _) in enumerate(tqdm.tqdm(val_loader, 'Extraction')):
        with timer:
            with torch.no_grad():
                with torch.amp.autocast('cuda', torch.float16, enabled=True):
                    img = img.to('cuda')
                    feat_bn = model.infer_image(img, gt_mask=None, after_bn=cfg.TEST.AFTER_BN)
        evaluator.update((feat_bn, vid, camid))
    logger.info(colored(f"Average inference time: {1000 * timer.average():.1f}ms per batch ({cfg.TEST.IMS_PER_BATCH} samples)", 'green'))
    logger.info(colored(f'Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 3):.1f}GB', 'green'))
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))