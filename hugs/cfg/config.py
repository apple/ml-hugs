#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from omegaconf import OmegaConf

# general configuration
cfg = OmegaConf.create()
cfg.seed = 0
cfg.mode = 'human' # 'human_scene' or 'scene'
cfg.output_path = 'output'
cfg.cfg_file = ''
cfg.exp_name = 'test'
cfg.dataset_path = ''
cfg.detect_anomaly = False
cfg.debug = False
cfg.wandb = False
cfg.logdir = ''
cfg.logdir_ckpt = ''
cfg.eval = False
cfg.bg_color = 'white'

# human dataset configuration
cfg.dataset = OmegaConf.create()
cfg.dataset.name = 'neuman' # 'zju', 'colmap', 'people_snapshot', 'itw'
cfg.dataset.seq = 'citron'

# training configuration
cfg.train = OmegaConf.create()
cfg.train.batch_size = 1
cfg.train.num_workers = 0
cfg.train.num_steps = 30_000
cfg.train.save_ckpt_interval = 4000
cfg.train.val_interval = 2000
cfg.train.anim_interval = 4000
cfg.train.optim_scene = True
cfg.train.save_progress_images = False
cfg.train.progress_save_interval = 10

# human model configuration
cfg.human = OmegaConf.create()
cfg.human.name = 'hugs'
cfg.human.ckpt = None
cfg.human.sh_degree = 3
cfg.human.n_subdivision = 0
cfg.human.only_rgb = False
cfg.human.use_surface = False
cfg.human.use_deformer = False
cfg.human.init_2d = False
cfg.human.disable_posedirs = False

cfg.human.res_offset = False
cfg.human.rotate_sh = False
cfg.human.isotropic = False
cfg.human.init_scale_multiplier = 1.0
cfg.human.run_init = False
cfg.human.estimate_delta = True
cfg.human.triplane_res = 256

cfg.human.optim_pose = False
cfg.human.optim_betas = False
cfg.human.optim_trans = False
cfg.human.optim_eps_offsets = False
cfg.human.activation = 'relu'

cfg.human.canon_nframes = 60
cfg.human.canon_pose_type = 'da_pose'
cfg.human.knn_n_hops = 3

# human model learning rate configuration
cfg.human.lr = OmegaConf.create()
cfg.human.lr.wd = 0.0
cfg.human.lr.position = 0.00016
cfg.human.lr.position_init = 0.00016
cfg.human.lr.position_final = 0.0000016
cfg.human.lr.position_delay_mult = 0.01
cfg.human.lr.position_max_steps = 30_000
cfg.human.lr.opacity = 0.05
cfg.human.lr.scaling = 0.005
cfg.human.lr.rotation = 0.001
cfg.human.lr.feature = 0.0025
cfg.human.lr.smpl_spatial = 2.0
cfg.human.lr.smpl_pose = 0.0001
cfg.human.lr.smpl_betas = 0.0001
cfg.human.lr.smpl_trans = 0.0001
cfg.human.lr.smpl_eps_offset = 0.0001
cfg.human.lr.lbs_weights = 0.0
cfg.human.lr.posedirs = 0.0
cfg.human.lr.percent_dense = 0.01

cfg.human.lr.appearance = 1e-3
cfg.human.lr.geometry = 1e-3
cfg.human.lr.vembed = 1e-3
cfg.human.lr.deformation = 1e-4
# scale
cfg.human.lr.scale_lr_w_npoints = False

# human model loss coefficients
cfg.human.loss = OmegaConf.create()
cfg.human.loss.ssim_w = 0.2
cfg.human.loss.l1_w = 0.8
cfg.human.loss.lpips_w = 1.0
cfg.human.loss.lbs_w = 0.0
cfg.human.loss.humansep_w = 0.0
cfg.human.loss.num_patches = 4
cfg.human.loss.patch_size = 128
cfg.human.loss.use_patches = 1

# human model densification configuration
cfg.human.densification_interval = 100
cfg.human.opacity_reset_interval = 3000
cfg.human.densify_from_iter = 500
cfg.human.densify_until_iter = 15_000
cfg.human.densify_grad_threshold = 0.0002
cfg.human.prune_min_opacity = 0.005
cfg.human.densify_extent = 2.0
cfg.human.max_n_gaussians = 2e5

# scene model configuration
cfg.scene = OmegaConf.create()
cfg.scene.name = 'scene_gs'
cfg.scene.ckpt = None
cfg.scene.sh_degree = 3
cfg.scene.add_bg_points = False
cfg.scene.num_bg_points = 204_800
cfg.scene.bg_sphere_dist = 5.0
cfg.scene.clean_pcd = False
cfg.scene.opt_start_iter = -1
cfg.scene.lr = OmegaConf.create()
cfg.scene.lr.percent_dense = 0.01
cfg.scene.lr.spatial_scale = 1.0
cfg.scene.lr.position_init = 0.00016
cfg.scene.lr.position_final = 0.0000016
cfg.scene.lr.position_delay_mult = 0.01
cfg.scene.lr.position_max_steps = 30_000
cfg.scene.lr.opacity = 0.05
cfg.scene.lr.scaling = 0.005
cfg.scene.lr.rotation = 0.001
cfg.scene.lr.feature = 0.0025

# scene model densification configuration
cfg.scene.percent_dense = 0.01
cfg.scene.densification_interval = 100
cfg.scene.opacity_reset_interval = 3000
cfg.scene.densify_from_iter = 500
cfg.scene.densify_until_iter = 15_000
cfg.scene.densify_grad_threshold = 0.0002
cfg.scene.prune_min_opacity = 0.005
cfg.scene.max_n_gaussians = 2e6

# scene model loss coefficients
cfg.scene.loss = OmegaConf.create()
cfg.scene.loss.ssim_w = 0.2
cfg.scene.loss.l1_w = 0.8
