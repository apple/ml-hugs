#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import cv2
import numpy as np
import torch
from pytorch3d.renderer import look_at_view_transform

from hugs.utils.graphics import fov2focal, get_projection_matrix
from hugs.utils.rotations import batch_look_at_th


def get_static_camera(img_size=512, fov=0.4, device='cuda'):
    fovx = fov
    fovy = fov
    zfar = 100.0
    znear = 0.01
    world_view_transform = torch.eye(4)
    
    cam_int = torch.eye(3)
    
    # convert fov to focal length (in pixels
    fx = fov2focal(fovx, img_size)
    fy = fov2focal(fovy, img_size)
    
    cam_int[0, 0] = fx
    cam_int[1, 1] = fy
    cam_int[0, 2] = img_size / 2
    cam_int[1, 2] = img_size / 2
    
    projection_matrix = get_projection_matrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    
    datum = {
        "fovx": fovx,
        "fovy": fovy,
        "image_height": img_size,
        "image_width": img_size,
        "world_view_transform": world_view_transform,
        "full_proj_transform": full_proj_transform,
        "camera_center": camera_center,
        "cam_int": cam_int,
        "cam_ext": world_view_transform,
        "near": znear,
        "far": zfar,
    }
    for k, v in datum.items():
        if isinstance(v, torch.Tensor):
            datum[k] = v.to(device)
    return datum

def rot_z(angle):
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    Rz = torch.tensor([[cos_theta, 0, sin_theta],
                        [0, 1, 0],
                        [-sin_theta, 0, cos_theta]])
    return Rz


def get_rotating_camera(img_size=512, fov=0.4, dist=5.0, device='cuda', nframes=40, angle_limit=2*torch.pi):
    # img size in height, width format
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
        
    fovx = fov
    fovy = fov
    zfar = 100.0
    znear = 0.01
    
    w2c_mats = []
    for idx, azim in enumerate(torch.linspace(0, angle_limit, nframes)):
        nRz = rot_z(-azim)
        vec = torch.tensor([[0., 0, dist]])
        vec = nRz @ vec.T
        vec = vec.T
        Rt = torch.eye(4)
        t = vec
        R = rot_z(azim)[None]
        R[:, 1:3] *= -1
        Rt[:3, :3] = R[0].T
        Rt[:3, 3] = t[0].squeeze()
        Rt = Rt.inverse().T
        w2c_mats.append(Rt)

    cam_int = torch.eye(3)

    fx = fov2focal(fovx, img_size[0])
    fy = fx
    
    cam_int[0, 0] = fx
    cam_int[1, 1] = fy
    cam_int[0, 2] = img_size[1] / 2
    cam_int[1, 2] = img_size[0] / 2
    
    data = []
    for i in range(len(w2c_mats)):
        world_view_transform = w2c_mats[i]
        projection_matrix = get_projection_matrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        
        datum = {
            "fovx": fovx,
            "fovy": fovy,
            "image_height": img_size[0],
            "image_width": img_size[1],
            "world_view_transform": world_view_transform,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
            "cam_int": cam_int,
            "cam_ext": world_view_transform,
            "near": znear,
            "far": zfar,
        }
        for k, v in datum.items():
            if isinstance(v, torch.Tensor):
                datum[k] = v.to(device)
        data.append(datum)
        
    return data


def get_predefined_pose(pose_type):
    if pose_type == 'da_pose':
        body_pose = torch.zeros((1, 69), dtype=torch.float32)
        body_pose[:, 2] = 1.0
        body_pose[:, 5] = -1.0
    elif pose_type == 'a_pose':
        body_pose = torch.zeros((1, 69), dtype=torch.float32)
        body_pose[:, 2] = 0.2
        body_pose[:, 5] = -0.2
        body_pose[:, 47] = -0.8
        body_pose[:, 50] = 0.8
    elif pose_type == 't_pose':
        body_pose = torch.zeros((1, 69), dtype=torch.float32)
        
    return body_pose


def get_smpl_static_params(betas, pose_type='da_pose', device='cuda'):
    global_orient = torch.zeros(3, dtype=torch.float32)

    body_pose = get_predefined_pose(pose_type)[0]
    
    transl = torch.tensor([0.0, 0.0, 0.0]).float()
    if betas.shape != (10,):
        betas = betas.reshape(10)
        
    scale = torch.ones(1, dtype=torch.float32)
    
    datum = {
        'betas': betas,
        'global_orient': global_orient,
        'body_pose': body_pose,
        'transl': transl,
        'smpl_scale': scale,
    }
    for k, v in datum.items():
        if isinstance(v, torch.Tensor):
            datum[k] = v.to(device)
            
    return datum


def get_smpl_canon_params(betas, nframes=40, pose_type='da_pose', device='cuda'):
    
    global_orient = torch.zeros((nframes, 3), dtype=torch.float32)
    for idx in range(nframes):
        angle = 2 * np.pi * idx / nframes
        R = cv2.Rodrigues(np.array([0, angle, 0]))[0]
        R_gt = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
        R_gt = R @ R_gt
        R_gt = cv2.Rodrigues(R_gt)[0].astype(np.float32)
        global_orient[idx] = torch.tensor(R_gt.reshape(3)).float()

    body_pose = get_predefined_pose(pose_type).repeat_interleave(nframes, dim=0)
    
    transl = torch.tensor([[0., 0.05, 5]]).float().repeat(nframes, 1)
    if betas.shape == (10,):
        betas = betas.unsqueeze(0).repeat(nframes, 1)
    else:
        betas = betas.repeat(nframes, 1)
        
    scale = torch.ones((nframes, 1), dtype=torch.float32)
    
    datum = {
        'betas': betas,
        'global_orient': global_orient,
        'body_pose': body_pose,
        'transl': transl,
        'smpl_scale': scale,
    }
    for k, v in datum.items():
        if isinstance(v, torch.Tensor):
            datum[k] = v.to(device)
            
    return datum