# Code based on 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/general_utils.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


import os
import torch
import random
import itertools
import subprocess
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont


def debug_tensor(tensor, name):
    print(f'{name}: {tensor.shape} {tensor.dtype} {tensor.device}')
    print(f'{name}: min: {tensor.min().item():.5f} \
        max: {tensor.max().item():.5f} \
            mean: {tensor.mean().item():.5f} \
                std: {tensor.std().item():.5f}')


def load_human_ckpt(human_gs, ckpt_path):
    ckpt = torch.load(ckpt_path)
    persistent_buffers = {k: v for k, v in human_gs._buffers.items() if k not in human_gs._non_persistent_buffers_set}
    local_name_params = itertools.chain(human_gs._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    for k, v in local_state.items():
        if v.shape != ckpt[k].shape:
            logger.warning(f'Warning: shape mismatch for {k}: {v.shape} vs {ckpt[k].shape}')
            if (isinstance(v, torch.nn.Parameter) and
                    not isinstance(ckpt[k], torch.nn.Parameter)):
                setattr(human_gs, k, torch.nn.Parameter(ckpt[k]))
            else:
                setattr(human_gs, k, ckpt[k])

    human_gs.load_state_dict(ckpt, strict=False)
    logger.info(f'Loaded human model from {ckpt_path}')
    return human_gs


class RandomIndexIterator:
    def __init__(self, max_index):
        self.max_index = max_index
        self.indices = list(range(max_index))
        random.shuffle(self.indices)
        self.current_index = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= self.max_index:
            self.current_index = 0
            random.shuffle(self.indices)
        index = self.indices[self.current_index]
        self.current_index += 1
        return index


def find_cfg_diff(default_cfg, cfg, delimiter='_'):
    default_cfg_list = OmegaConf.to_yaml(default_cfg).split('\n')
    cfg_str_list = OmegaConf.to_yaml(cfg).split('\n')
    diff_str = ''
    nlines = len(default_cfg_list)
    for lnum in range(nlines):
        if default_cfg_list[lnum] != cfg_str_list[lnum]:
            diff_str += cfg_str_list[lnum].replace(': ', '-').replace(' ', '')
            diff_str += delimiter
    diff_str = diff_str[:-1]
    return diff_str
        

def create_video(img_folder, output_fname, fps=20):
    os.makedirs(os.path.dirname(output_fname), exist_ok=True)
    cmd = f"/usr/bin/ffmpeg -hide_banner -loglevel error -framerate {fps} -pattern_type glob -i '{img_folder}/*.png' \
        -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" \
            -c:v libx264 -pix_fmt yuv420p {output_fname} -y"
    logger.info(f"Video is saved under {output_fname}")
    subprocess.call(cmd, shell=True)


def save_images(img, img_fname, txt_label=None):
    if not os.path.isdir(os.path.dirname(img_fname)):
        os.makedirs(os.path.dirname(img_fname), exist_ok=True)
    im = Image.fromarray(img)
    if txt_label is not None:
        draw = ImageDraw.Draw(im)
        txt_font = ImageFont.load_default()
        draw.text((10, 10), txt_label, fill=(0, 0, 0), font=txt_font)
    im.save(img_fname)
        
        
def eps_denom(denom, eps=1e-17):
    """ Prepare denominator for division """
    denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
    denom = denom_sign * torch.clamp(denom.abs(), eps)
    return denom


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def safe_state(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.set_device(torch.device("cuda:0"))


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def torch_rotation_matrix_from_vectors(vec1: torch.Tensor, vec2: torch.Tensor):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector of shape N,3
    :param vec2: A 3d "destination" vector of shape N,3
    :return mat: A transform matrix (Nx3x3) which when applied to vec1, aligns it with vec2.
    """
    a = vec1 / torch.norm(vec1, dim=-1, keepdim=True)
    b = vec2 / torch.norm(vec2, dim=-1, keepdim=True)
    
    v = torch.cross(a, b, dim=-1)
    c = torch.matmul(a.unsqueeze(1), b.unsqueeze(-1)).squeeze(-1)
    s = torch.norm(v, dim=-1, keepdim=True)
    kmat = torch.zeros(v.shape[0], 3, 3, device=v.device, dtype=v.dtype)
    kmat[:, 0, 1] = -v[:, 2]
    kmat[:, 0, 2] = v[:, 1]
    kmat[:, 1, 0] = v[:, 2]
    kmat[:, 1, 2] = -v[:, 0]
    kmat[:, 2, 0] = -v[:, 1]
    kmat[:, 2, 1] = v[:, 0]
    rot_mat = torch.eye(3, device=v.device, dtype=v.dtype).unsqueeze(0)
    rot_mat = rot_mat + kmat + torch.matmul(kmat, kmat) * ((1 - c) / (s ** 2)).unsqueeze(-1)
    return rot_mat


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward (ctx, input):
        ctx.save_for_backward(input)
        return torch.clamp(input, -1, 1)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        tanh = torch.tanh(input)
        grad_input[input <= -1] = (1.0 - tanh[input <= -1]**2.0) * grad_output[input <= -1]
        grad_input[input >= 1] = (1.0 - tanh[input >= 1]**2.0) * grad_output[input >= 1]
        max_norm = 1.0  # set the maximum gradient norm value
        torch.nn.utils.clip_grad_norm_(grad_input, max_norm)
        return grad_input