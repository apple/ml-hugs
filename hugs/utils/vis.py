#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import cv2
import math
import torch
import trimesh
import numpy as np
import open3d as o3d
from PIL import Image
from plyfile import PlyData, PlyElement

from hugs.utils.general import inverse_sigmoid

from .rotations import quaternion_to_matrix
from .spherical_harmonics import SH2RGB

sphere = trimesh.primitives.Sphere(subdivisions=0)
coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
coord.compute_vertex_normals()


def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(1*3):
        l.append('f_dc_{}'.format(i))
    for i in range(15*3):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l


@torch.no_grad()
def save_ply(human_gs_out, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = human_gs_out['xyz_canon'].cpu().numpy()
    normals = np.zeros_like(xyz)

    f_dc = human_gs_out['shs'][:, :1].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = human_gs_out['shs'][:, 1:].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities =  inverse_sigmoid(human_gs_out['opacity']).cpu().numpy()
    scale = torch.log(human_gs_out['scales_canon']).cpu().numpy()
    rotation = human_gs_out['rotq_canon'].cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def save_ellipsoid_meshes(human_gs_out, out_fname):
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    
    colors = torch.clamp(SH2RGB(human_gs_out['shs'][:, 0]), 0.0, 1.0)
    colors = torch.cat([colors, human_gs_out['opacity']], dim=-1)
    
    sp_meshes_deformed = get_vis_mesh(
        mean=human_gs_out['xyz'],
        scale=human_gs_out['scales'],
        rotation=human_gs_out['rotq'],
        colors=colors,
    )
    
    sp_meshes_canon = get_vis_mesh(
        mean=human_gs_out['xyz_canon'],
        scale=human_gs_out['scales_canon'],
        rotation=human_gs_out['rotq_canon'],
        colors=colors,
    )

    sp_meshes_deformed.rotate(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))
    sp_meshes_canon.translate([1.5, 0, 0])
    
    sp_meshes_deformed.compute_vertex_normals()
    sp_meshes_canon.compute_vertex_normals()

    o3d.io.write_triangle_mesh(f'{out_fname}_deformed_rgb.ply', sp_meshes_deformed)
    o3d.io.write_triangle_mesh(f'{out_fname}_canon_rgb.ply', sp_meshes_canon)


def get_bbox_from_smpl(vs, factor=1.2):
    assert vs.shape[0] == 1
    min_vert = vs.min(dim=1).values
    max_vert = vs.max(dim=1).values

    c = (max_vert + min_vert) / 2
    s = (max_vert - min_vert) / 2
    s = s.max(dim=-1).values * factor

    min_vert = c - s[:, None]
    max_vert = c + s[:, None]
    return torch.cat([min_vert, max_vert], dim=0)


def get_predefined_rest_pose(cano_pose, device="cuda"):
    body_pose_t = torch.zeros((1, 69), device=device)
    if cano_pose.lower() == "da_pose":
        body_pose_t[:, 2] = torch.pi / 6
        body_pose_t[:, 5] = -torch.pi / 6
    elif cano_pose.lower() == "a_pose":
        body_pose_t[:, 2] = 0.2
        body_pose_t[:, 5] = -0.2
        body_pose_t[:, 47] = -0.8
        body_pose_t[:, 50] = 0.8
    else:
        raise ValueError("Unknown cano_pose: {}".format(cano_pose))
    return body_pose_t


@torch.no_grad()
def get_vis_mesh(mean, scale, rotation, colors, transl=None, normals=None):
    sp_faces = sphere.faces[None].repeat(rotation.shape[0], axis=0)
    sp_verts = sphere.vertices[None].repeat(rotation.shape[0], axis=0)
    
    sp_verts = sp_verts * scale[:, None].cpu().numpy()
    sp_verts = (quaternion_to_matrix(rotation).cpu().numpy()[:, None] @ sp_verts[..., None])[..., 0]
    if transl is not None:
        trans = (mean-transl.unsqueeze(0)).cpu().numpy()
    else:
        trans = mean.cpu().numpy()
        
    if normals is not None:
        for ni in range(len(normals)):
            normals[ni] = normals[ni].cpu().numpy()
        
    sp_verts = sp_verts + trans[:, None]

    sp_meshes = None
    for i in range(sp_verts.shape[0]):
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(sp_verts[i]),
            o3d.utility.Vector3iVector(sp_faces[i])
        )
        c = np.clip(colors[i].cpu().numpy()[:3], 0.0, 1.0)
        m.paint_uniform_color(c)
            
        if i == 0:
            sp_meshes = m
        else:
            sp_meshes += m
        
        if normals is not None:
            for ni, n in enumerate(normals):
                o = trans[i]
                e = o + (n[i] * 0.05)
                cn = [[1, 0, 0], [0, 1, 0], [0, 0, 1]][ni]
                arrow = get_arrow(end=e, origin=o, scale=0.5)
                arrow.paint_uniform_color(cn)
                sp_meshes += arrow
            
    return sp_meshes


def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([
                    [np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                ])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([
                    [np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                ])
    return Rz, Ry


def get_line(start, end, color=[1, 0, 0]):
    l = o3d.geometry.LineSet()
    l.points = o3d.utility.Vector3dVector([start, end])
    l.lines = o3d.utility.Vector2iVector([[0, 1]])
    l.paint_uniform_color(color)
    return l


def get_arrow(end, origin=np.array([0, 0, 0]), scale=1):
    assert(not np.all(end == origin))
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=size/17.5 * scale,
        cone_height=size*0.2 * scale,
        cylinder_radius=size/30 * scale,
        cylinder_height=size*(1 - 0.2*scale),
        resolution=2,
        cylinder_split=1,
        cone_split=1,
    )
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return(mesh)


@torch.no_grad()
def save_depthmap(depth, filename, min=None, max=None):
    if depth.shape[0] == 1:
        depth = depth[0]
    
    if min is None:
        min = depth.min()

    if max is None:
        max = depth.max()
        
    depth = (depth - min) / (max - min)
    depth = 1.0 - depth
    
    depth = depth[None].repeat_interleave(3, 0)
    depth = (depth.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(depth).save(filename)
    
    
def draw_bodypose(joints_2d, width=512, height=512):
    canvas = np.zeros((height, width, 3), np.uint8)
    
    H, W, C = canvas.shape

    stickwidth = 4

    joints_2d = joints_2d[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13,14,15,16,17,18]]
    
    limbSeq = [
        [2, 3], # 0
        [2, 6], # 1
        [3, 4], # 2
        [4, 5], # 3
        [6, 7], # 4
        [7, 8], # 5
        [2, 9], # 6
        [9, 10], # 7
        [10, 11], # 8
        [2, 12], # 9
        [12, 13], # 10
        [13, 14], # 11
        [2, 1], # 12
        [1, 15], # 13 --
        [15, 17], # 14
        [1, 16], # 15
        [16, 18], # 16
        [3, 17], # 17
        [6, 18] # 18
        ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(18):
        x, y = joints_2d[i]
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    
    for i in range(17):
        index = np.array(limbSeq[i]) - 1
        cur_canvas = canvas.copy()
        Y = joints_2d[index.astype(int), 0]
        X = joints_2d[index.astype(int), 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas
