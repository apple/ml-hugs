#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import sys
import torch

from hugs.cfg.config import cfg as default_cfg


def optimize_init(model, lr: float = 1e-3, num_steps: int = 2000):
    model.train()

   
    lr = 1e-3
    default_cfg.human.lr.appearance = lr
    default_cfg.human.lr.geometry = lr
    default_cfg.human.lr.vembed = lr
    default_cfg.human.lr.deformation = 5e-4
    model.setup_optimizer(default_cfg.human.lr)
    optim = model.optimizer
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1000, verbose=True, factor=0.5)
    fn = torch.nn.MSELoss()
    
    body_pose = torch.zeros((69)).to("cuda").float()
    global_orient = torch.zeros((3)).to("cuda").float()
    betas = torch.zeros((10)).to("cuda").float()
    
    gt_vals = model.initialize()
    
    print("===== Ground truth values: =====")
    for k, v in gt_vals.items():
        print(k, v.shape)
        gt_vals[k] = v.detach().clone().to("cuda").float()
    print("================================")
    
    losses = []

    for i in range(num_steps):
        
        if hasattr(model, 'canon_forward'):
            model_out = model.canon_forward()
        else:
            model_out = model.forward(global_orient, body_pose, betas)
        
        if i % 1000 == 0:
            continue
            
        loss_dict = {}
        for k, v in gt_vals.items():
            if k in ['faces', 'deformed_normals', 'edges']:
                continue
            if k in model_out:
                if model_out[k] is not None:
                    loss_dict['loss_' + k] = fn(model_out[k], v)
            
        loss = sum(loss_dict.values())
        loss.backward()
        loss_str = ", ".join([f"{k}: {v.item():.7f}" for k, v in loss_dict.items()])
        print(f"Step {i:04d}: {loss.item():.7f} ({loss_str})", end='\r')
        
        optim.step()
        optim.zero_grad(set_to_none=True)
        lr_scheduler.step(loss.item())
            
        losses.append(loss.item())
    
    return model
