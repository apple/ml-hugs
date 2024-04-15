# Code based on 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/loss_utils.py
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

import torch
from math import exp
import torch.nn.functional as F
import torch.distributions as dist
from torch.autograd import Variable
from pytorch3d.ops import laplacian


def total_variation_loss(img, mask=None):
    """
    Compute the scale-invariant total variation loss for an image.

    Parameters:
        img (torch.Tensor): Input image tensor.
        mask (torch.Tensor, optional): Optional mask tensor to apply the loss only on certain regions.

    Returns:
        torch.Tensor: Scale-invariant total variation loss.
    """
    assert len(img.size()) == 3, "Input image tensor must be 3D (H W C)"
    assert img.size(0) == 3, "Input image tensor must have 3 channels"
    # Calculate the total variation loss
    # Shift the image to get the differences in both x and y directions
    d_x = img[:, :, 1:] - img[:, :, :-1]
    d_y = img[:, 1:, :] - img[:, :-1, :]

    # Compute the L1 norm of the differences
    tv_loss = torch.sum(torch.abs(d_x)) + torch.sum(torch.abs(d_y))

    if mask is not None:
        tv_loss = tv_loss / mask.sum()
    else:
        # Normalize by the size of the image
        img_size = img.size(-1) * img.size(-2)
        tv_loss = tv_loss / img_size

    return tv_loss


def l1_loss(network_output, gt, mask=None):
    if mask is not None:
        return torch.abs((network_output - gt)).sum() / mask.sum()
    else:
        return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)


def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def multivariate_normal_kl(mu_0, cov_0, mu_1, cov_1):
    # Create multivariate normal distributions
    mvn_0 = dist.MultivariateNormal(mu_0, covariance_matrix=cov_0)
    mvn_1 = dist.MultivariateNormal(mu_1, covariance_matrix=cov_1)

    # Calculate KL divergence
    kl_divergence = torch.distributions.kl.kl_divergence(mvn_0, mvn_1)

    return kl_divergence


def multivariate_normal_kl_v2(mu_0, cov_0, mu_1, cov_1):
    """
    Calculate the KL divergence between two batches of multivariate normal distributions.

    Parameters:
    - mu_0: Mean of the first distribution, shape (batch_size, n)
    - cov_0: Covariance matrix of the first distribution, shape (batch_size, n, n)
    - mu_1: Mean of the second distribution, shape (batch_size, n)
    - cov_1: Covariance matrix of the second distribution, shape (batch_size, n, n)

    Returns:
    - kl_divergence: KL divergence between the two batches of distributions, shape (batch_size,)
    """

    # Ensure covariance matrices are positive definite
    eye_like = torch.eye(3).to(cov_0)
    
    cov_0 = (cov_0 + cov_0.transpose(-2, -1)) / 2.0 + 1e-6 * eye_like.unsqueeze(0)
    cov_1 = (cov_1 + cov_1.transpose(-2, -1)) / 2.0 + 1e-6 * eye_like.unsqueeze(0)
    
    # Calculate KL divergence using the formula
    trace = lambda x : torch.einsum("...ii", x)
    term1 = 0.5 * (trace(cov_1.inverse() @ cov_0) + torch.sum((mu_1 - mu_0).unsqueeze(-1).transpose(-2, -1) @ cov_1.inverse() @ (mu_1 - mu_0).unsqueeze(-1), dim=[-2, -1]))
    term2 = -0.5 * mu_0.size(-1) + 0.5 * torch.log(cov_1.det() / cov_0.det())

    kl_divergence = term1 + term2

    return kl_divergence


def pcd_laplacian_smoothing(verts, edges, method: str = "uniform"):
    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        L = laplacian(verts, edges)
        
    loss = L.mm(verts)
    loss = loss.norm(dim=1)

    return loss.mean()


def gmof(x, rho=2.0):
    return (x**2 / (rho**2 + x**2)) * rho**2