# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from . import Camera
import numpy as np



def projection(fovy, n=1.0, f=50.0, near_plane=None):
    focal = np.tan(fovy / 180.0 * np.pi * 0.5)
    if near_plane is None:
        near_plane = n
    return np.array(
        [[n / focal, 0, 0, 0],
         [0, n / -focal, 0, 0],
         [0, 0, -(f + near_plane) / (f - near_plane), -(2 * f * near_plane) / (f - near_plane)],
         [0, 0, -1, 0]]).astype(np.float32)

def projection_2(opt):
    zfar= opt.zfar
    znear= opt.znear
    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (zfar + znear) / (zfar - znear)
    proj_matrix[3, 2] = - (zfar * znear) / (zfar - znear)
    proj_matrix[2, 3] = 1
    
    return proj_matrix


class PerspectiveCamera(Camera):
    def __init__(self, opt, device='cuda'):
        super(PerspectiveCamera, self).__init__()
        self.device = device
        self.proj_mtx = torch.from_numpy(projection(opt.fovy, f=1000.0, n=1.0, near_plane=0.1)).to(self.device).unsqueeze(dim=0)
        #self.proj_mtx= projection_2(opt).to(self.device).unsqueeze(dim=0)
        

    def project(self, points_bxnx4):
        out = torch.matmul(
            points_bxnx4,
            torch.transpose(self.proj_mtx, 1, 2))
        return out
