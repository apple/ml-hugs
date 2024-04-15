#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import numpy as np


SMPL_PATH = 'data/smpl'

AMASS_SMPLH_TO_SMPL_JOINTS = np.arange(0,156).reshape((-1,3))[[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 
    19, 20, 21, 22, 37
]].reshape(-1)

NEUMAN_PATH = 'data/neuman/dataset'