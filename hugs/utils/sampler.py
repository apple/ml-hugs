#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import numpy as np
import torch
import cv2


class PatchSampler():
    def __init__(self, num_patch=4, patch_size=20, ratio_mask=0.9, dilate=0):
        self.n = num_patch
        self.patch_size = patch_size
        self.p = ratio_mask
        self.dilate = dilate
        assert self.patch_size % 2 == 0, "patch size has to be even"

    def sample(self, mask, *args):
        patch = (self.patch_size, self.patch_size)
        shape = mask.shape[1:]
        mask_np = mask.squeeze(0).detach().cpu().numpy()
        
        if np.random.rand() < self.p:
            o = patch[0] // 2
            if self.dilate > 0:
                m = cv2.dilate(mask_np, np.ones((self.dilate, self.dilate))) > 0
            else:
                m = mask_np
            valid = m[o:-o, o:-o] > 0
            (xs, ys) = np.where(valid)
            idx = np.random.choice(len(xs), size=self.n, replace=False)
            x, y = xs[idx], ys[idx]
        else:
            x = np.random.randint(0, shape[0] - patch[0], size=self.n)
            y = np.random.randint(0, shape[1] - patch[1], size=self.n)
        
        
        output = []
        for d in [mask, *args]:
            patches = []
            for xi, yi in zip(x, y):
                p = d[:, xi:xi + patch[0], yi:yi + patch[1]]
                patches.append(p)
            patches = torch.stack(patches, dim=0)
            output.append(patches)
        return output
