#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import sys
import json
import glob
import torch
import shutil
import argparse
import subprocess
from loguru import logger
from omegaconf import OmegaConf

sys.path.append('.')

from hugs.trainer import GaussianTrainer
from hugs.utils.general import safe_state
from hugs.cfg.config import cfg as default_cfg


def get_logger(cfg):
    logdir = cfg.logdir
    
    mode = 'render_canon'
    logger.add(os.path.join(cfg.logdir, f'{mode}.log'), level='INFO')
    logger.info(f'Logging to {logdir}')
    logger.info(OmegaConf.to_yaml(cfg))


@torch.no_grad()
def main(cfg):
    safe_state(seed=cfg.seed)
    
    get_logger(cfg)
    
    latest_human_ckpt = None
    human_ckpt_files = glob.glob(cfg.logdir_ckpt + '/*human*.pth')
    human_ckpt_files += glob.glob(cfg.logdir_ckpt + '/ckpt/*human*.pth')
    human_ckpt_files = sorted(human_ckpt_files)
    
    if len(human_ckpt_files) > 0:
        latest_human_ckpt = human_ckpt_files[-1]
        logger.info(f'Found human ckpt: {latest_human_ckpt}')
        cfg.human.ckpt = latest_human_ckpt
    else:
        if cfg.mode in ['human', 'human_scene']:
            logger.error(f'Human ckpt is required for {cfg.mode} mode.')
            exit()
        else:
            logger.warning('No human ckpt found')
            
    latest_scene_ckpt = None
    scene_ckpt_files = sorted(glob.glob(cfg.logdir_ckpt + '/*scene*.pth'))
    if len(scene_ckpt_files) > 0:
        latest_scene_ckpt = scene_ckpt_files[-1]
        logger.info(f'Found scene ckpt: {latest_scene_ckpt}')
        cfg.scene.ckpt = latest_scene_ckpt
    else:
        if cfg.mode in ['scene', 'human_scene']:
            logger.error(f'Scene ckpt is required for {cfg.mode} mode.')
            exit()
        else:
            logger.warning('No scene ckpt found')
    
    # get trainer
    trainer = GaussianTrainer(cfg)
 
    trainer.validate()
                
    # run animation
    trainer.animate()
    
    mode = 'eval' if cfg.eval else 'train'
    with open(os.path.join(cfg.logdir, f'results_{mode}.json'), 'w') as f:
        json.dump(trainer.eval_metrics, f, indent=4)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", default=None, help="path to the output directory")
    
    args, extras = parser.parse_known_args()
        
    cfg_file = args.output_dir + '/config_train.yaml'
    
    cfg_file = OmegaConf.load(cfg_file)
    
    cfg = OmegaConf.merge(default_cfg, cfg_file, OmegaConf.from_cli(extras))
    cfg.eval = True
    
    if args.output_dir is not None:
        cfg.logdir = args.output_dir
        cfg.logdir_ckpt = args.output_dir
    
    main(cfg)
            