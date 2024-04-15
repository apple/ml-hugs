#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import glob
import json
import os
import subprocess
import sys
import time
import argparse
from loguru import logger
from omegaconf import OmegaConf

sys.path.append('.')

from hugs.trainer import GaussianTrainer
from hugs.utils.config import get_cfg_items
from hugs.cfg.config import cfg as default_cfg
from hugs.utils.general import safe_state, find_cfg_diff


def get_logger(cfg):
    output_path = cfg.output_path
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    mode = 'eval' if cfg.eval else 'train'
    
    
    if cfg.mode in ['human', 'human_scene']:
        logdir = os.path.join(
            output_path, cfg.mode, cfg.dataset.name,
            cfg.dataset.seq, cfg.human.name, cfg.exp_name, 
            time_str,
        )
    else:
        logdir = os.path.join(
            output_path, cfg.mode, cfg.dataset.name,
            cfg.dataset.seq, cfg.exp_name, # exp_diff_str, 
            time_str,
        )
    cfg.logdir = logdir
    cfg.logdir_ckpt = os.path.join(logdir, 'ckpt')
    
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(cfg.logdir_ckpt, exist_ok=True)
    os.makedirs(os.path.join(logdir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'anim'), exist_ok=True)
    os.makedirs(os.path.join(logdir, 'meshes'), exist_ok=True)
    
    logger.add(os.path.join(logdir, f'{mode}.log'), level='INFO')
    logger.info(f'Logging to {logdir}')
    logger.info(OmegaConf.to_yaml(cfg))
    
    with open(os.path.join(logdir, f'config_{mode}.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg)) 
    
    
def main(cfg):
    safe_state(seed=cfg.seed)
    # create loggers
    get_logger(cfg)
    
    # get trainer
    trainer = GaussianTrainer(cfg)
    
    if not cfg.eval:
        trainer.train()
        trainer.save_ckpt()
    
    # run evaluation
    trainer.validate()
    
    mode = 'eval' if cfg.eval else 'train'
    with open(os.path.join(cfg.logdir, f'results_{mode}.json'), 'w') as f:
        json.dump(trainer.eval_metrics, f, indent=4)
        
    # run animation
    if cfg.mode in ['human', 'human_scene']:
        trainer.animate()
        trainer.render_canonical(pose_type='a_pose')
        trainer.render_canonical(pose_type='da_pose')


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", required=True, help="path to the yaml config file")
    parser.add_argument("--cfg_id", type=int, default=-1, help="id of the config to run")
    args, extras = parser.parse_known_args()
    
    cfg_file = OmegaConf.load(args.cfg_file)
    list_of_cfgs, hyperparam_search_keys = get_cfg_items(cfg_file)
    
    logger.info(f'Running {len(list_of_cfgs)} experiments')
    
    if args.cfg_id >= 0:
        cfg_item = list_of_cfgs[args.cfg_id]
        logger.info(f'Running experiment {args.cfg_id} -- {cfg_item.exp_name}')
        default_cfg.cfg_file = args.cfg_file
        cfg = OmegaConf.merge(default_cfg, cfg_item, OmegaConf.from_cli(extras))
        main(cfg)
    else:
        for exp_id, cfg_item in enumerate(list_of_cfgs):
            logger.info(f'Running experiment {exp_id} -- {cfg_item.exp_name}')
            default_cfg.cfg_file = args.cfg_file
            cfg = OmegaConf.merge(default_cfg, cfg_item, OmegaConf.from_cli(extras))
            main(cfg)
            