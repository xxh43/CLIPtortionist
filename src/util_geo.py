
import sys
from matplotlib.pyplot import xscale

from torch._C import dtype
sys.path.append('..')
import torch
import torch.nn
from scipy.spatial.distance import cdist
from config import *
import copy
import numpy as np
from util_mesh_volume import *


def intersection_1d(min0, max0, min1, max1):

    if max0 - min0 < max1 - min1:
        new_min0 = min0
        new_max0 = max0
        new_min1 = min1
        new_max1 = max1
    else:
        new_min0 = min1
        new_max0 = max1
        new_min1 = min0
        new_max1 = max0
    
    if new_max0 <= new_min1:
        #print('case1')
        return 0.0
    elif new_min0 < new_min1 and new_max0 > new_min1 and new_max0 <= new_max1:
        #print('case2')
        return new_max0 - new_min1
    elif new_min0 > new_min1 and new_max0 <= new_max1:
        #print('case3')
        return new_max0 - new_min0
    elif new_max0 > new_max1 and new_min0 <= new_max1 and new_min0 > new_min1:
        #print('case4')
        return new_max1-new_min0
    elif new_min0 > new_max1:
        #print('case5')
        return 0.0
    else:
        print('impossible 1d intersection')
        exit() 

def intersection_2d(dir0_min0, dir0_max0, dir0_min1, dir0_max1, dir1_min0, dir1_max0, dir1_min1, dir1_max1):
    dir0_inter = intersection_1d(dir0_min0, dir0_max0, dir0_min1, dir0_max1)
    dir1_inter = intersection_1d(dir1_min0, dir1_max0, dir1_min1, dir1_max1)
    return dir0_inter, dir1_inter

def project_to_1d(pc, dir_index):

    if len(pc) == 0:
        return np.array([])

    if dir_index == 0:
        pc_1d = to_numpy(pc[:, 0])
    if dir_index == 1:
        pc_1d = to_numpy(pc[:, 1])

    return pc_1d

def project_to_2d(pc, dir_index):

    pc = to_numpy(pc)

    #if len(pc) > 500:
        #display_pcs([pc])

    #projected_pc = pc - np.mean(pc, axis=0)   
    #projected_pc = np.matmul(np.linalg.inv(box.rotmat), projected_pc.transpose(1,0)).transpose(1,0)
    if len(pc) == 0:
        return np.array([])

    if dir_index == 0:
        pc_2d = np.concatenate((np.expand_dims(pc[:, 1], axis=1), np.expand_dims(pc[:, 2], axis=1)), axis=1)
    if dir_index == 1:
        pc_2d = np.concatenate((np.expand_dims(pc[:, 0], axis=1), np.expand_dims(pc[:, 2], axis=1)), axis=1)
    if dir_index == 2:
        pc_2d = np.concatenate((np.expand_dims(pc[:, 0], axis=1), np.expand_dims(pc[:, 1], axis=1)), axis=1)

    return pc_2d

def get_2d_voxel(pc):
    
    x_min = -1.1
    x_max = 1.1
    y_min = -1.1
    y_max = 1.1
    delta = 0.04

    voxel = np.zeros((int((x_max-x_min)/delta)+1, int((y_max-y_min)/delta)+1))

    inside_pc = []
    outside_pc = []
    for p in pc:
        x_idx = int((p[0] - x_min) / delta)
        y_idx = int((p[1] - y_min) / delta)
        voxel[x_idx][y_idx] = 1.0
    
    for x_idx in range(len(voxel)):
        for y_idx in range(len(voxel[0])):
            x = x_min + x_idx * delta
            y = y_min + y_idx * delta
            if voxel[x_idx][y_idx] == 1.0:
                inside_pc.append([x, y])
            else:
                outside_pc.append([x, y])

    return voxel, inside_pc, outside_pc

def get_3d_voxel(mesh):
    
    x_min = -1.1
    x_max = 1.1
    y_min = -1.1
    y_max = 1.1
    z_min = -1.1
    z_max = 1.1

    delta = 0.01
    voxel = np.zeros((int((x_max-x_min)/delta)+1, int((y_max-y_min)/delta)+1, int((z_max-z_min)/delta)+1))

    inside_pc = []
    outside_pc = []

    voxel_centers = []
    for x_idx in range(len(voxel)):
        for y_idx in range(len(voxel[0])):
            for z_idx in range(len(voxel[0][0])):
                x = x_min + x_idx * delta + 0.5 * delta
                y = y_min + y_idx * delta + 0.5 * delta
                z = z_min + z_idx * delta + 0.5 * delta
                cell_center = np.array([x, y, z])
                voxel_centers.append(cell_center)
    voxel_centers = np.array(voxel_centers)
    labels = points_in_mesh(mesh, voxel_centers)
    inside_pc = voxel_centers[labels == True]
    outside_pc = voxel_centers[labels == False]
        
    return None, inside_pc, outside_pc