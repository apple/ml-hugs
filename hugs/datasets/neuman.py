#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
import os
import cv2
import glob
import copy
from loguru import logger
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from .neuman_utils import neuman_helper
from .neuman_utils.geometry import transformations
from .neuman_utils.cameras.camera_pose import CameraPose
from .neuman_utils.geometry.basics import Translation, Rotation
from hugs.cfg.constants import AMASS_SMPLH_TO_SMPL_JOINTS, NEUMAN_PATH
from hugs.utils.graphics import get_projection_matrix, BasicPointCloud


def get_center_and_diag(cam_centers):
    cam_centers = np.vstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=0, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=1, keepdims=True)
    diagonal = np.max(dist)
    return center.flatten(), diagonal


def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": smpl_params["betas"].astype(np.float32).reshape(1, 10),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }
    

def get_data_splits(scene):
    scene_length = len(scene.captures)
    num_val = scene_length // 5
    length = int(1 / (num_val) * scene_length)
    offset = length // 2
    val_list = list(range(scene_length))[offset::length]
    train_list = list(set(range(scene_length)) - set(val_list))
    test_list = val_list[:len(val_list) // 2]
    val_list = val_list[len(val_list) // 2:]
    assert len(train_list) > 0
    assert len(test_list) > 0
    assert len(val_list) > 0    
    return train_list, val_list, test_list


def mocap_path(scene_name):
    # ./data/MoSh/MPI_mosh/50027/misc_dancing_hiphop_poses.npz
    if os.path.basename(scene_name) == 'seattle': # and opt.motion_name == 'moonwalk':
        # return './data/SFU/0018/0018_Moonwalk001_poses.npz', 0, 400, 4
        # return './data/SFU/0005/0005_Stomping001_poses.npz', 0, 800, 4
        return './data/SFU/0005/0005_SideSkip001_poses.npz', 0, 800, 4
    elif os.path.basename(scene_name) == 'citron': # and opt.motion_name == 'speedvault':
        # return './data/SFU/0008/0008_ChaCha001_poses.npz', 0, 1000, 4
        return './data/MPI_mosh/00093/irish_dance_poses.npz', 0, 1000, 4
        # return './data/SFU/0012/0012_SpeedVault001_poses.npz', 0, 340, 2
        # return './data/MPI_mosh/50027/misc_dancing_hiphop_poses.npz', 0, 2000, 4
        # return './data/SFU/0017/0017_ParkourRoll001_poses.npz', 140, 500, 4
    elif os.path.basename(scene_name) == 'parkinglot': # and opt.motion_name == 'yoga':
        return './data/SFU/0005/0005_2FeetJump001_poses.npz', 0, 1200, 4
        # return './data/SFU/0008/0008_Yoga001_poses.npz', 300, 1900, 8
    elif os.path.basename(scene_name) == 'bike': # and opt.motion_name == 'jumpandroll':
        return './data/MPI_mosh/50002/misc_poses.npz', 0, 250, 1
        # return './data/SFU/0018/0018_Moonwalk001_poses.npz', 0, 600, 4
        # return './data/SFU/0012/0012_JumpAndRoll001_poses.npz', 100, 400, 3
    elif os.path.basename(scene_name) == 'jogging': # and opt.motion_name == 'cartwheel':
        return './data/SFU/0007/0007_Cartwheel001_poses.npz', 200, 1000, 8
    elif os.path.basename(scene_name) == 'lab': # and opt.motion_name == 'chacha':
        return './data/SFU/0008/0008_ChaCha001_poses.npz', 0, 1000, 4
    else:
        raise ValueError('Define new elif branch')


def alignment(scene_name, motion_name=None):
    if os.path.basename(scene_name) == 'seattle':
        manual_trans = np.array([-2.25, 1.08, 8.18])
        manual_rot = np.array([90.4, -4.2, -1]) / 180 * np.pi
        manual_scale = 1.8
    elif os.path.basename(scene_name) == 'citron':
        manual_trans = np.array([6.33, 1.7, 10.7])
        manual_rot = np.array([72.4, 168.2, -4.4]) / 180 * np.pi
        manual_scale = 2.5
    elif os.path.basename(scene_name) == 'parkinglot':
        manual_trans = np.array([-0.8, 2.35, 12.67])
        manual_rot = np.array([94, -85, -363]) / 180 * np.pi
        manual_scale = 3.0
    elif os.path.basename(scene_name) == 'bike':
        manual_trans = np.array([0.0, 0.88, 3.89])
        manual_rot = np.array([88.8, 180, 1.8]) / 180 * np.pi
        manual_scale = 1.0
    elif os.path.basename(scene_name) == 'jogging':
        manual_trans = np.array([0.0, 0.24, 0.33])
        manual_rot = np.array([95.8, -1.2, -2.2]) / 180 * np.pi
        manual_scale = 0.25
    elif os.path.basename(scene_name) == 'lab':
        manual_trans = np.array([5.76, 3.03, 11.69])
        manual_rot = np.array([90.4, -4.2, -1.8]) / 180 * np.pi
        manual_scale = 3.0
    else:
        manual_trans = np.array([0, 0, 0])
        manual_rot = np.array([0, 0, 0]) / 180 * np.pi
        manual_scale = 1
    return manual_trans, manual_rot, manual_scale


def rendering_caps(scene_name, nframes, scene):
    if os.path.basename(scene_name) == 'seattle':
        dummy_caps = []
        for i in range(nframes):
            temp = copy.deepcopy(scene.captures[20])
            ellipse_a = 0.15 * 10
            ellipse_b = 0.05 #* 1
            x_offset= temp.cam_pose.right * ellipse_a * np.cos(i/nframes * 2 * np.pi)
            y_offset= temp.cam_pose.up * ellipse_b * np.sin(i/nframes * 2 * np.pi)
            temp.cam_pose.camera_center_in_world = temp.cam_pose.camera_center_in_world + x_offset + y_offset
            dummy_caps.append(temp)
    elif os.path.basename(scene_name) == 'citron':
        dummy_caps = []
        for i in range(nframes):
            temp = copy.deepcopy(scene.captures[33])
            ellipse_a = 0.15 * 3
            ellipse_b = 0.03 * 3 
            x_offset= temp.cam_pose.right * (ellipse_a * np.cos(2 * i/nframes * 2 * np.pi) + 0.2)
            y_offset= temp.cam_pose.up * ellipse_b * np.sin(2 * i/nframes * 2 * np.pi)
            temp.cam_pose.camera_center_in_world = temp.cam_pose.camera_center_in_world + x_offset + y_offset
            dummy_caps.append(temp)
    elif os.path.basename(scene_name) == 'parkinglot':
        dummy_caps = []
        for i in range(nframes):
            temp = copy.deepcopy(scene.captures[23])
            ellipse_a = 0.15 * 10
            ellipse_b = 0.03 * 5
            x_offset= temp.cam_pose.right * (ellipse_a * np.cos(2 * i/nframes * 2 * np.pi) + 0.2)
            y_offset= temp.cam_pose.up * ellipse_b * np.sin(2 * i/nframes * 2 * np.pi)
            temp.cam_pose.camera_center_in_world = temp.cam_pose.camera_center_in_world + x_offset + y_offset
            dummy_caps.append(temp)
    elif os.path.basename(scene_name) == 'bike':
        dummy_caps = []
        start_id = 25
        interval = 0.005 * 2
        for i in range(nframes):
            temp = copy.deepcopy(scene.captures[start_id])
            temp.cam_pose.camera_center_in_world += interval * i * temp.cam_pose.right
            dummy_caps.append(temp)
    elif os.path.basename(scene_name) == 'jogging':
        dummy_caps = []
        start_id = 67
        interval = 0.01
        for i in range(nframes):
            temp = copy.deepcopy(scene.captures[start_id])
            temp.cam_pose.camera_center_in_world -= interval * i * temp.cam_pose.right
            dummy_caps.append(temp)
    elif os.path.basename(scene_name) == 'lab':
        dummy_caps = []
        start_id = 39
        ellipse_a = 0.15 * 10
        ellipse_b = 0.03
        for i in range(nframes):
            temp = copy.deepcopy(scene.captures[start_id])
            x_offset= temp.cam_pose.right * (ellipse_a * np.cos(i/nframes * 2 * np.pi))
            y_offset= temp.cam_pose.up * ellipse_b * np.sin(i/nframes * 2 * np.pi)
            temp.cam_pose.camera_center_in_world = temp.cam_pose.camera_center_in_world + x_offset + y_offset
            temp.cam_pose.camera_center_in_world += temp.cam_pose.forward * 0.2
            dummy_caps.append(temp)
    return dummy_caps


class NeumanDataset(torch.utils.data.Dataset):
    def __init__(
        self, seq, split, 
        render_mode='human_scene',
        add_bg_points=False, 
        num_bg_points=204_800,
        bg_sphere_dist=5.0,
        clean_pcd=False,
    ):
        dataset_path = f"{NEUMAN_PATH}/{seq}"
        scene = neuman_helper.NeuManReader.read_scene(
            dataset_path,
            tgt_size=None,
            normalize=False,
            smpl_type='optimized'
        )
        
        smpl_params_path = f'{dataset_path}/4d_humans/smpl_optimized_aligned_scale.npz'        
        smpl_params = np.load(smpl_params_path)
        smpl_params = {f: smpl_params[f] for f in smpl_params.files}
        
        if split == 'anim':
            motion_path, start_idx, end_idx, skip = mocap_path(seq)
            motions = np.load(motion_path)
            poses = motions['poses'][start_idx:end_idx:skip, AMASS_SMPLH_TO_SMPL_JOINTS]
            transl = motions['trans'][start_idx:end_idx:skip]
            betas = smpl_params['betas'][0]
            smpl_params = {
                'global_orient': poses[:, :3],
                'body_pose': poses[:, 3:],
                'transl': transl,
                'scale': np.array([1.0] * poses.shape[0]),
                'betas': betas[None].repeat(poses.shape[0], 0)[:, :10],
            }
            
            manual_trans, manual_rot, manual_scale = alignment(seq)
            manual_rotmat = transformations.euler_matrix(*manual_rot)[:3, :3]
            self.manual_rotmat = torch.from_numpy(manual_rotmat).float().unsqueeze(0)
            self.manual_trans = torch.from_numpy(manual_trans).float().unsqueeze(0)
            self.manual_scale = torch.tensor([manual_scale]).float().unsqueeze(0)
            nframes = poses.shape[0]
            caps = rendering_caps(seq, nframes, scene)
            scene.captures = caps
        else:
            self.train_split, _, self.val_split = get_data_splits(scene)
        
        self.scene = scene
        
        pcd_xyz = self.scene.point_cloud[:, :3]
        pcd_col = self.scene.point_cloud[:, 3:6] / 255.
        
        if clean_pcd:
            import open3d as o3d
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(pcd_xyz)

            logger.debug(f'Num points before outlier removal: {len(pcd_xyz)}')
            _, inlier_ind = scene_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.5)

            print(f'Num points after outlier removal: {len(inlier_ind)}')
            pcd_xyz = pcd_xyz[inlier_ind]
            pcd_col = pcd_col[inlier_ind]

        if add_bg_points:
            # find the scene center and size
            point_max_coordinate = np.max(pcd_xyz, axis=0)
            point_min_coordinate = np.min(pcd_col, axis=0)
            scene_center = (point_max_coordinate + point_min_coordinate) / 2
            scene_size = np.max(point_max_coordinate - point_min_coordinate)
            # build unit sphere points
            n_points = num_bg_points
            samples = np.arange(n_points)
            y = 1 - (samples / float(n_points - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y
            phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians
            theta = phi * samples  # golden angle increment
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            unit_sphere_points = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)
            # build background sphere
            bg_sphere_point_xyz = (unit_sphere_points * scene_size * bg_sphere_dist) + scene_center
            bg_sphere_point_rgb = np.asarray(np.random.random(bg_sphere_point_xyz.shape))
            # add background sphere to scene
            pcd_xyz = np.concatenate([pcd_xyz, bg_sphere_point_xyz], axis=0)
            pcd_col = np.concatenate([pcd_col, bg_sphere_point_rgb], axis=0)

            import open3d as o3d
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_xyz))
            pcd.colors = o3d.utility.Vector3dVector(pcd_col)
            o3d.io.write_point_cloud(f'./output/{seq}_bg_sphere.ply', pcd)
            logger.debug(f"Added {len(bg_sphere_point_xyz)} background points, saved to output/{seq}_bg_sphere.ply")
        
        self.init_pcd = BasicPointCloud(
            points=pcd_xyz, 
            colors=pcd_col, 
            normals=np.zeros_like(pcd_xyz), 
            faces=None
        )
        
        self.smpl_params = {}
        for k in smpl_params.keys():
            self.smpl_params[k] = torch.from_numpy(smpl_params[k]).float()
        
        self.sam_mask_dir = f'{dataset_path}/4d_humans/sam_segmentations'
        self.msk_lists = sorted(glob.glob(f"{self.sam_mask_dir}/*.png"))
        
        _, diag = get_center_and_diag([cap.cam_pose.camera_center_in_world for cap in scene.captures])
    
        self.radius = diag * 1.1
        
        self.split = split
        self.mode = render_mode
        
        self.num_frames = len(self.scene.captures)    

        self.cached_data = None
        if self.cached_data is None:
            self.load_data_to_cuda()

    def __len__(self):
        if self.split == "train":
            return len(self.train_split)
        elif self.split == "val":
            return len(self.val_split)
        elif self.split == "anim":
            return self.num_frames
    
    def get_single_item(self, i):
        
        if self.split == "train":
            idx = self.train_split[i]
        elif self.split == "val":
            idx = self.val_split[i]
        elif self.split == "anim":
            idx = i
        
        cap = self.scene.captures[idx]
        
        datum = {}
        if self.split in ['train', 'val']:
            img = cap.captured_image.image # cv2.imread(self.img_lists[idx])
            
            msk = cv2.imread(self.msk_lists[idx], cv2.IMREAD_GRAYSCALE) / 255
            if self.mode == 'scene':
                msk = cv2.dilate(msk, np.ones((20,20), np.uint8), msk, iterations=1)
            
            # get bbox from mask
            rows = np.any(msk, axis=0)
            cols = np.any(msk, axis=1)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            bbox = np.array([xmin, ymin, xmax, ymax])

            img = (img[..., :3] / 255).astype(np.float32)
            msk = msk.astype(np.float32)
            
            img = img.transpose(2, 0, 1)
            datum.update({
                "rgb": torch.from_numpy(img).float(),
                "mask": torch.from_numpy(msk).float(),
                "bbox": torch.from_numpy(bbox).float(),
            })
        
        K = cap.intrinsic_matrix
        width = cap.size[1]
        height = cap.size[0]
        
        fovx = 2 * np.arctan(width / (2 * K[0, 0]))
        fovy = 2 * np.arctan(height / (2 * K[1, 1]))
        # zfar = max(cap.far['human'], cap.near['bkg']) + 1.0
        # znear = min(cap.near['human'], cap.near['bkg'])
        zfar = 100.0 # max(zfar, 100.0)
        znear = 0.01 # min(znear, 0.01)
        
        world_view_transform = torch.from_numpy(cap.cam_pose.world_to_camera).T # torch.eye(4)
        c2w = torch.from_numpy(cap.cam_pose.camera_to_world)
        
        projection_matrix = get_projection_matrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        cam_intrinsics = torch.from_numpy(cap.intrinsic_matrix).float()
        
        datum.update({
            "fovx": fovx,
            "fovy": fovy,
            "image_height": height,
            "image_width": width,
            "world_view_transform": world_view_transform,
            "c2w": c2w,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
            "cam_intrinsics": cam_intrinsics,
            
            "betas": self.smpl_params["betas"][idx],
            "global_orient": self.smpl_params["global_orient"][idx],
            "body_pose": self.smpl_params["body_pose"][idx],
            "transl": self.smpl_params["transl"][idx],
            "smpl_scale": self.smpl_params["scale"][idx],
            "near": znear,
            "far": zfar,
        })
        
        if self.split == 'anim':
            datum.update({
                "manual_rotmat": self.manual_rotmat,
                "manual_scale": self.manual_scale,
                "manual_trans": self.manual_trans,
            })
        
        return datum
    
    def load_data_to_cuda(self):
        self.cached_data = []
        for i in tqdm(range(self.__len__())):
            datum = self.get_single_item(i)
            for k, v in datum.items():
                if isinstance(v, torch.Tensor):
                    datum[k] = v.to("cuda")
            self.cached_data.append(datum)
                
    def __getitem__(self, idx):
        if self.cached_data is None:
            return self.get_single_item(idx, is_src=True)
        else:
            return self.cached_data[idx]
