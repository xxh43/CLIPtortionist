
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

def reflect_pc_batched(pcs, normals, centers):
    normals = normals.unsqueeze(dim=1)
    #normals = torch.tensor(normal, device=device, dtype=torch.float).unsqueeze(dim=0).repeat_interleave(len(pcs), dim=0).unsqueeze(dim=1)
    #centers = torch.zeros(pcs.shape, device=device, dtype=torch.float)
    vecs = (pcs - centers).permute(0, 2, 1)
    dots = torch.bmm(normals, vecs)
    temp = torch.bmm(dots.permute(0, 2, 1), normals)
    reflected_pcs = pcs - 2 * temp
    return reflected_pcs

def transform_with_homo_matrix_batched(pcs, mats):
    pad = torch.ones((pcs.shape[0], pcs.shape[1], 1), device='cuda:0')
    pcs_homo = torch.cat((pcs, pad), dim=2)
    pcs_transformed = torch.bmm(mats, pcs_homo.transpose(1,2)).transpose(1,2)
    return pcs_transformed

def transform_with_homo_matrix(pc, mat):
    pad = torch.nn.ConstantPad1d((0, 1), 1.0)
    pc_homo = pad(pc)
    pc_homo = pc_homo.transpose(0, 1)
    pc_transformed = torch.matmul(mat, pc_homo)
    pc_transformed = pc_transformed.transpose(0, 1)
    return pc_transformed

def rotate_with_axis_center_angle_batched(pcs, axes, centers, angles):
    axes = axes/torch.norm(axes, dim=1).unsqueeze(dim=1)
    mats = getRotMatrixHomo_batched(axes, centers, angles)
    return transform_with_homo_matrix_batched(pcs, mats)

def rotate_with_axis_center_angle(pc, axis, center, angle):
    axis = axis/torch.norm(axis)
    mat_homo = getRotMatrixHomo(axis, center, angle)
    return transform_with_homo_matrix(pc, mat_homo)

def getRotMatrixHomo_batched(axes, centers, angles):
    u = axes[:,0]
    v = axes[:,1]
    w = axes[:,2]
    a = centers[:,0]
    b = centers[:,1]
    c = centers[:,2]
    sin_theta = torch.sin(angles)
    cos_theta = torch.cos(angles)

    m00 = u*u + (v*v + w*w)*cos_theta
    m01 = u*v*(torch.ones(cos_theta.shape, device='cuda:0')-cos_theta)-w*sin_theta
    m02 = u*w*(torch.ones(cos_theta.shape, device='cuda:0')-cos_theta)+v*sin_theta
    m03 = (a*(v*v+w*w) - u*(b*v+c*w))*(torch.ones(cos_theta.shape, device='cuda:0')-cos_theta)+(b*w-c*v)*sin_theta

    m10 = u*v*(torch.ones(cos_theta.shape, device='cuda:0')-cos_theta)+w*sin_theta
    m11 = v*v+(u*u+w*w)*cos_theta
    m12 = v*w*(torch.ones(cos_theta.shape, device='cuda:0')-cos_theta)-u*sin_theta
    m13 = (b*(u*u+w*w) - v*(a*u+c*w))*(torch.ones(cos_theta.shape, device='cuda:0')-cos_theta)+(c*u-a*w)*sin_theta

    m20 = u*w*(torch.ones(cos_theta.shape, device='cuda:0')-cos_theta)-v*sin_theta
    m21 = v*w*(torch.ones(cos_theta.shape, device='cuda:0')-cos_theta)+u*sin_theta
    m22 = w*w+(u*u+v*v)*cos_theta
    m23 = (c*(u*u+v*v) - w*(a*u+b*v))*(torch.ones(cos_theta.shape, device='cuda:0')-cos_theta)+(a*v-b*u)*sin_theta

    rot_mats = torch.stack((m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23)).transpose(0, 1)
    rot_mats = rot_mats.reshape(rot_mats.shape[0], 3, 4)

    return rot_mats


def getRotMatrixHomo(axis, center, angle):
    u = axis[0]
    v = axis[1]
    w = axis[2]
    a = center[0]
    b = center[1]
    c = center[2]
    sin_theta = torch.sin(angle)
    cos_theta = torch.cos(angle)

    m00 = u*u + (v*v + w*w)*cos_theta
    m01 = u*v*(1-cos_theta)-w*sin_theta
    m02 = u*w*(1-cos_theta)+v*sin_theta
    m03 = (a*(v*v+w*w) - u*(b*v+c*w))*(1-cos_theta)+(b*w-c*v)*sin_theta
    
    m10 = u*v*(1-cos_theta)+w*sin_theta
    m11 = v*v+(u*u+w*w)*cos_theta
    m12 = v*w*(1-cos_theta)-u*sin_theta
    m13 = (b*(u*u+w*w) - v*(a*u+c*w))*(1-cos_theta)+(c*u-a*w)*sin_theta

    m20 = u*w*(1-cos_theta)-v*sin_theta
    m21 = v*w*(1-cos_theta)+u*sin_theta
    m22 = w*w+(u*u+v*v)*cos_theta
    m23 = (c*(u*u+v*v) - w*(a*u+b*v))*(1-cos_theta)+(a*v-b*u)*sin_theta
    
    rot_mat = torch.stack((

        torch.stack((m00, m01, m02, m03)),

        torch.stack((m10, m11, m12, m13)),

        torch.stack((m20, m21, m22, m23)),
    ))
    
    return rot_mat


def diag(a, b):
  return 1 - 2 * a * a - 2 * b * b

def tr_add(a, b, c, d):
  return 2 * a * b + 2 * c * d

def tr_sub(a, b, c, d):
  return 2 * a * b - 2 * c * d

def quaternion_to_rotation_matrix(qs):
    normalized_qs = qs/(torch.norm(qs, dim=1).unsqueeze(dim=1))
    ws = normalized_qs[:,0]
    xs = normalized_qs[:,1]
    ys = normalized_qs[:,2]
    zs = normalized_qs[:,3]
    mats = torch.zeros((len(qs),3,3), device=device, dtype=torch.float)
    mats[:,0,0] = diag(ys, zs)  
    mats[:,0,1] = tr_sub(xs, ys, zs, ws)
    mats[:,0,2] = tr_add(xs, zs, ys, ws)
    mats[:,1,0] = tr_add(xs, ys, zs, ws) 
    mats[:,1,1] = diag(xs, zs)
    mats[:,1,2] = tr_sub(ys, zs, xs, ws)
    mats[:,2,0] = tr_sub(xs, zs, ys, ws)  
    mats[:,2,1] = tr_add(ys, zs, xs, ws)  
    mats[:,2,2] = diag(xs, ys)
    return mats

def axisAngle_to_quaternion(axisAngles):

    axes = axisAngles[:, 0:3]
    axes = axes/torch.norm(axes, dim=1).unsqueeze(dim=1)
    angles = axisAngles[:, 3]

    getRotMatrixHomo_batched(axes, centers, angles)

    print('angles', angles)

    xs = axes[:, 0] * torch.sin(angles/2)
    ys = axes[:, 1] * torch.sin(angles/2)
    zs = axes[:, 2] * torch.sin(angles/2)
    ws = torch.cos(angles/2)

    quaternions = torch.cat((xs.unsqueeze(dim=1), ys.unsqueeze(dim=1), zs.unsqueeze(dim=1), ws.unsqueeze(dim=1)), dim=1)

    print('quaternions', quaternions)

    return quaternions
