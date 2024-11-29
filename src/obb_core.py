
import joblib
import torch

import random
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff 

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 

from config import *
import matplotlib.colors as mcolors
import numpy as np
from sklearn.decomposition import PCA
import networkx as nx
from scipy.spatial import distance
from shapely import geometry, ops
from util_motion import axisAngle_to_quaternion, quaternion_to_rotation_matrix
from util_geo import *
alpha = 220
color_palette = ['red', 'green', 'blue', 'orange', 'purple']

import numpy as np
import random
import copy
import torchvision
import os
from PIL import Image
import argparse
from pathlib import Path
from torchvision import transforms
from mesh import *
from util_vis import *
from collections import deque
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from util_mesh_volume import *
from util_mesh_surface import *
from util_motion import *
import time

class BBox3D:
    def __init__(self):
        self.pc_indices = None
        self.rotmat = []
        self.size = []
        self.center = []

        self.surface_points = None
        self.face_center_points = None
        self.corner_points = None

        self.pose_rotation_axis = None

        #self.cage = None
        #self.face_to_cage_control = []
        #self.face_to_normal = []
        #self.active_cage_face = [True]*6

    def initialize(self):
        self.center = torch.tensor(self.center, device=device, dtype=torch.float)
        self.size = torch.tensor(self.size, device=device, dtype=torch.float)
        self.rotmat = torch.tensor(self.rotmat, device=device, dtype=torch.float)
        self.surface_points = None
        self.face_center_points = None
        self.corner_points = None
        self.face_to_normal = [-self.rotmat[:, 0], self.rotmat[:, 0], -self.rotmat[:, 1], self.rotmat[:, 1], -self.rotmat[:, 2], self.rotmat[:, 2]]
        self.get_face_center_points()
        self.get_corner_points()
        self.get_surface_points()
        #self.get_cage_points()

    def update_size(self, size_change):
        self.size = self.size * (1 + size_change)
        self.surface_points = perferm_point_size_change(self.surface_points, self, size_change)
        self.face_center_points = perferm_point_size_change(self.face_center_points, self, size_change)
        self.corner_points = perferm_point_size_change(self.corner_points, self, size_change)
        #self.cage.vertices = to_numpy(perferm_point_size_change(to_torch(self.cage.vertices), self, size_change))

    def update_pose(self, pose_change):
        self.rotmat = perferm_direction_pose_change(self.rotmat.transpose(0, 1), self, pose_change).transpose(0, 1)
        self.surface_points = perferm_point_pose_change(self.surface_points, self, pose_change)
        self.face_center_points = perferm_point_pose_change(self.face_center_points, self, pose_change)
        self.corner_points = perferm_point_pose_change(self.corner_points, self, pose_change)
        #self.cage.vertices = to_numpy(perferm_point_pose_change(to_torch(self.cage.vertices), self, pose_change))

    def update_loc(self, loc_change):
        self.center = perferm_loc_change(self.center.unsqueeze(dim=0), self, loc_change)[0]
        self.surface_points = perferm_loc_change(self.surface_points, self, loc_change)
        self.face_center_points = perferm_loc_change(self.face_center_points, self, loc_change)
        self.corner_points = perferm_loc_change(self.corner_points, self, loc_change)
        #self.cage.vertices = to_numpy(perferm_loc_change(to_torch(self.cage.vertices), self, loc_change))

    def get_corner_points(self):

        if self.corner_points is not None:
            return self.corner_points

        center = self.center
        size = self.size
        dir0 = self.rotmat[:, 0]
        dir1 = self.rotmat[:, 1]
        dir2 = self.rotmat[:, 2]
    
        p0 = center - dir0*size[0] - dir1*size[1] + dir2*size[2]
        p1 = center - dir0*size[0] - dir1*size[1] - dir2*size[2]
        p2 = center - dir0*size[0] + dir1*size[1] - dir2*size[2]
        p3 = center - dir0*size[0] + dir1*size[1] + dir2*size[2]
        
        p4 = center + dir0*size[0] - dir1*size[1] + dir2*size[2]
        p5 = center + dir0*size[0] - dir1*size[1] - dir2*size[2]
        p6 = center + dir0*size[0] + dir1*size[1] - dir2*size[2]
        p7 = center + dir0*size[0] + dir1*size[1] + dir2*size[2]
        self.corner_points = [p0, p1, p2, p3, p4, p5, p6, p7]
        
        self.corner_points = torch.stack(self.corner_points)
        return self.corner_points

    def get_surface_points(self):
        
        if self.surface_points is not None:
            return self.surface_points

        self.surface_points = []

        center = self.center
        size = self.size
        dir0 = self.rotmat[:, 0]
        dir1 = self.rotmat[:, 1]
        dir2 = self.rotmat[:, 2]

        delta = 0.025
        sample_count0 = max(5, min(30, int(2*size[0]/delta) + 1))
        sample_count1 = max(5, min(30, int(2*size[1]/delta) + 1))
        sample_count2 = max(5, min(30, int(2*size[2]/delta) + 1))

        delta0 = 2*size[0]*(1.0/(sample_count0))
        delta1 = 2*size[1]*(1.0/(sample_count1))
        delta2 = 2*size[2]*(1.0/(sample_count2))

        min_corner = center - dir0*size[0] - dir1*size[1] - dir2*size[2]
        
        for i in range(1, sample_count0):
            for j in range(1, sample_count1):
                p = min_corner + dir0*delta0*i + dir1*delta1*j
                self.surface_points.append(p)        
        
        for i in range(1, sample_count0):
            for j in range(1, sample_count2):
                p = min_corner + dir0*delta0*i + dir2*delta2*j
                self.surface_points.append(p)
        
        for i in range(1, sample_count1):
            for j in range(1, sample_count2):
                p = min_corner + dir1*delta1*i + dir2*delta2*j
                self.surface_points.append(p)
    
        max_corner = center + dir0*size[0] + dir1*size[1] + dir2*size[2]

        for i in range(1, sample_count0):
            for j in range(1, sample_count1):
                p = max_corner - dir0*delta0*i - dir1*delta1*j
                self.surface_points.append(p)        
        
        for i in range(1, sample_count0):
            for j in range(1, sample_count2):
                p = max_corner - dir0*delta0*i - dir2*delta2*j
                self.surface_points.append(p)
        
        for i in range(1, sample_count1):
            for j in range(1, sample_count2):
                p = max_corner - dir1*delta1*i - dir2*delta2*j
                self.surface_points.append(p)
        
        #display_pcs_and_cubes([self.surface_points], [self])
        #exit()
        self.surface_points = torch.stack(self.surface_points)
        return self.surface_points

    def get_face_center_points(self):

        if self.face_center_points is not None:
            return self.face_center_points
        
        self.face_center_points = []
        dir0 = self.rotmat[:, 0]
        dir1 = self.rotmat[:, 1]
        dir2 = self.rotmat[:, 2]
        c0_pos = self.center + dir0 * self.size[0]
        c0_neg = self.center - dir0 * self.size[0]
        c1_pos = self.center + dir1 * self.size[1]
        c1_neg = self.center - dir1 * self.size[1]
        c2_pos = self.center + dir2 * self.size[2]
        c2_neg = self.center - dir2 * self.size[2]
        self.face_center_points = [c0_pos, c0_neg, c1_pos, c1_neg, c2_pos, c2_neg]

        self.face_center_points = torch.stack(self.face_center_points)
        return self.face_center_points

    def get_edges(self):
        corners = self.get_corner_points()
        p0 = corners[0]
        p1 = corners[1]
        p2 = corners[2]
        p3 = corners[3]
        p4 = corners[4]
        p5 = corners[5]
        p6 = corners[6]
        p7 = corners[7]
        edges = [[p0, p1], [p1, p2], [p2, p3], [p3, p0], [p0, p4], [p1, p5], [p2, p6], [p3, p7], [p4, p5], [p5, p6], [p6, p7], [p7, p4]]
        return edges

    def surface_point_to_face(self, p):

        eps = 0.0001
        vec = p - self.center 
        
        min_projection_diff = np.inf
        associated_dir_index = -1
        for i in range(3):
            projection_diff = torch.abs(torch.abs(torch.dot(vec, self.rotmat[:, i])) - self.size[i])
            if projection_diff < min_projection_diff:
                min_projection_diff = projection_diff
                associated_dir_index = i
        
        if associated_dir_index == 0:
            #display_pc_and_dirs_and_cubes([p, p, p], [self.rotmat[:, 0], self.rotmat[:, 1], self.rotmat[:, 2]], [self] )
            return 0, 1, 2
        elif associated_dir_index == 1:
            #display_pc_and_dirs_and_cubes([p, p, p], [self.rotmat[:, 1], self.rotmat[:, 0], self.rotmat[:, 2]], [self] )
            return 1, 0, 2
        else:
            #display_pc_and_dirs_and_cubes([p, p, p], [self.rotmat[:, 2], self.rotmat[:, 0], self.rotmat[:, 1]], [self] )
            return 2, 0, 1

    def get_volume(self):
        return 2*self.size[0] * 2*self.size[1] * 2*self.size[2]

    def to_triangle_mesh(self):

        center = self.center
        size = self.size * 1.05
        dir0 = self.rotmat[:, 0]
        dir1 = self.rotmat[:, 1]
        dir2 = self.rotmat[:, 2]
    
        p0 = center - dir0*size[0] - dir1*size[1] + dir2*size[2]
        p1 = center - dir0*size[0] - dir1*size[1] - dir2*size[2]
        p2 = center - dir0*size[0] + dir1*size[1] - dir2*size[2]
        p3 = center - dir0*size[0] + dir1*size[1] + dir2*size[2]
        
        p4 = center + dir0*size[0] - dir1*size[1] + dir2*size[2]
        p5 = center + dir0*size[0] - dir1*size[1] - dir2*size[2]
        p6 = center + dir0*size[0] + dir1*size[1] - dir2*size[2]
        p7 = center + dir0*size[0] + dir1*size[1] + dir2*size[2]
        vertices = to_numpy(torch.stack([p0, p1, p2, p3, p4, p5, p6, p7]))
        faces = np.array([[0, 2, 1], [3, 2, 0], [4, 7, 6], [4, 6, 5], [3, 2, 6], [3, 6, 7], [0, 1, 5], [0, 5, 4], [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5]])
        return (vertices, faces)

    def to_quad_mesh(self):
        vertices = to_numpy(self.get_corner_points())
        faces = [(0,1,5,4), (3,2,6,7), (4,7,6,5), (0,3,2,1), (0,3,7,4), (1,2,6,5)]
        return (vertices, faces)


def get_3d_axis_aligned_bbox(pc):

    pc = to_numpy(pc)

    x_min = np.min(pc[:, 0])
    x_max = np.max(pc[:, 0])
    y_min = np.min(pc[:, 1])
    y_max = np.max(pc[:, 1])
    z_min = np.min(pc[:, 2])
    z_max = np.max(pc[:, 2])
    size = np.array([x_max-x_min, y_max-y_min, z_max-z_min])*0.5

    xdir = np.array([1, 0, 0])
    ydir = np.array([0, 1, 0])
    zdir = np.array([0, 0, 1])
    rotmat = np.vstack([xdir, ydir, zdir]).T

    center = np.array([(x_min+x_max)*0.5, (y_min+y_max)*0.5, (z_min+z_max)*0.5])

    box = BBox3D()
    box.pc = pc
    box.center = np.array(center)
    box.rotmat = np.array(rotmat)
    box.size = np.array(size)

    return box

def get_3d_upward_aligned_bbox(pc):

    try:

        pc_2d = np.concatenate((np.expand_dims(pc[:, 0], axis=1), np.expand_dims(pc[:,2], axis=1)), axis=1)    
        to_origin, extents = trimesh.bounds.oriented_bounds(pc_2d, angle_digits=10)
        t_xz = to_origin[:2, :2].transpose().dot(-to_origin[:2, 2])
        size_y = np.max(pc[:,1]) - np.min(pc[:,1])
        center = np.array([t_xz[0], np.min(pc[:,1])+size_y*0.5, t_xz[1]])
        size = np.array([extents[0]*0.5, size_y*0.5, extents[1]*0.5])
        xdir = np.array([to_origin[0, 0], 0, to_origin[0, 1]])
        zdir = np.array([to_origin[1, 0], 0, to_origin[1, 1]])
        ydir = np.cross(xdir, zdir)
        rotmat = np.vstack([xdir, ydir, zdir]).T

    except:

        center = pc.mean(axis=0, keepdims=True)
        points = pc - center
        center = center[0, :]
        pca = PCA()
        pca.fit(points)
        pcomps = pca.components_
        points_local = np.matmul(pcomps, points.transpose()).transpose()
        size = (points_local.max(axis=0) - points_local.min(axis=0))*0.5
        xdir = pcomps[0, :]
        ydir = pcomps[1, :]
        
        xdir /= np.linalg.norm(xdir)
        ydir /= np.linalg.norm(ydir)
        zdir = np.cross(xdir, ydir)
        rotmat = np.vstack([xdir, ydir, zdir]).T

    rotmat[:, 0] = rotmat[:, 0]/np.linalg.norm(rotmat[:, 0])
    if np.dot(rotmat[:, 0], np.array([1, 0, 0])) < 0:
        rotmat[:, 0] = -rotmat[:, 0]
    rotmat[:, 1] = rotmat[:, 1]/np.linalg.norm(rotmat[:, 1])
    if np.dot(rotmat[:, 1], np.array([0, 1, 0])) < 0:
        rotmat[:, 1] = -rotmat[:, 1]
    rotmat[:, 2] = rotmat[:, 2]/np.linalg.norm(rotmat[:, 2])
    if np.dot(rotmat[:, 2], np.array([0, 0, 1])) < 0:
        rotmat[:, 2] = -rotmat[:, 2]
    
    box = BBox3D()
    box.pc = pc
    box.center = np.array(center)
    box.rotmat = np.array(rotmat)
    box.size = np.array(size)
    return box

def get_3d_object_aligned_bbox(pc):

    try:
        to_origin, extents = trimesh.bounds.oriented_bounds(pc, angle_digits=10)
        t = to_origin[:3, :3].transpose().dot(-to_origin[:3, 3])
        size = extents*0.5
        xdir = to_origin[0, :3]
        ydir = to_origin[1, :3]
    except:
        center = pc.mean(axis=0, keepdims=True)
        points = pc - center
        t = center[0, :]
        pca = PCA()
        pca.fit(points)
        pcomps = pca.components_
        points_local = np.matmul(pcomps, points.transpose()).transpose()
        size = (points_local.max(axis=0) - points_local.min(axis=0))*0.5
        xdir = pcomps[0, :]
        ydir = pcomps[1, :]

    xdir /= np.linalg.norm(xdir)
    ydir /= np.linalg.norm(ydir)
    zdir = np.cross(xdir, ydir)
    rotmat = np.vstack([xdir, ydir, zdir]).T

    rotmat[:, 0] = rotmat[:, 0]/np.linalg.norm(rotmat[:, 0])
    rotmat[:, 1] = rotmat[:, 1]/np.linalg.norm(rotmat[:, 1])
    rotmat[:, 2] = rotmat[:, 2]/np.linalg.norm(rotmat[:, 2])


    box = BBox3D()
    box.pc = pc
    box.center = np.array(center)
    box.rotmat = np.array(rotmat)
    box.size = np.array(size)
    return box


    
def clamp_batched(input, min, max):
    clamped_input = torch.max(torch.min(input, max), min)
    return clamped_input

def get_decay(v):
    eps = 0.00001
    decayed_v = 1.0/(v*v + eps)
    return decayed_v

def get_pc_to_cubes_distance(pc, zs, ts, rotmats, option):
    
    repeated_pc = pc.unsqueeze(dim=0).repeat_interleave(len(zs), dim=0)

    axis0s = rotmats[:, :, 0]
    z0s = zs[:, 0]

    axis1s = rotmats[:, :, 1]
    z1s = zs[:, 1]

    axis2s = rotmats[:, :, 2]
    z2s = zs[:, 2]

    repeated_ts = ts.unsqueeze(dim=1).repeat_interleave(len(pc), dim=1)
    repeated_z0s = z0s.unsqueeze(dim=1).repeat_interleave(len(pc), dim=1)
    repeated_axis0s = axis0s.unsqueeze(dim=1)
    #axis0_sign_distance = torch.abs(torch.bmm((repeated_pc - repeated_ts), repeated_axis0s.transpose(1, 2)).squeeze(dim=2)) - repeated_z0s

    axis0_sign_distance = torch.abs((repeated_pc[:, :, 0] - repeated_ts[:, :, 0])) - repeated_z0s

    repeated_ts = ts.unsqueeze(dim=1).repeat_interleave(len(pc), dim=1)
    repeated_z1s = z1s.unsqueeze(dim=1).repeat_interleave(len(pc), dim=1)
    repeated_axis1s = axis1s.unsqueeze(dim=1)
    #axis1_sign_distance = torch.abs(torch.bmm((repeated_pc - repeated_ts), repeated_axis1s.transpose(1, 2)).squeeze(dim=2)) - repeated_z1s
    axis1_sign_distance = torch.abs((repeated_pc[:, :, 1] - repeated_ts[:, :, 1])) - repeated_z1s

    repeated_ts = ts.unsqueeze(dim=1).repeat_interleave(len(pc), dim=1)
    repeated_z2s = z2s.unsqueeze(dim=1).repeat_interleave(len(pc), dim=1)
    repeated_axis2s = axis2s.unsqueeze(dim=1)
    #axis2_sign_distance = torch.abs(torch.bmm((repeated_pc - repeated_ts), repeated_axis2s.transpose(1, 2)).squeeze(dim=2)) - repeated_z2s
    axis2_sign_distance = torch.abs((repeated_pc[:, :, 2] - repeated_ts[:, :, 2])) - repeated_z2s

    if option == 'outside':
        #distances = axis2_sign_distance
        distances = torch.square(torch.relu(axis0_sign_distance)) + torch.square(torch.relu(axis1_sign_distance)) + torch.square(torch.relu(axis2_sign_distance))
        #distances = torch.min(torch.stack((torch.relu(axis0_sign_distance).unsqueeze(dim=2), torch.relu(axis1_sign_distance).unsqueeze(dim=2), torch.relu(axis2_sign_distance).unsqueeze(dim=2)), dim=2), dim=2).values.squeeze(dim=2)

        #distances = torch.sqrt(distances)
        distances = distances.permute(1, 0)
    elif option == 'inside':
        axis0_sign_distance = -axis0_sign_distance.permute(1, 0)
        axis1_sign_distance = -axis1_sign_distance.permute(1, 0)
        axis2_sign_distance = -axis2_sign_distance.permute(1, 0)
        stacked_sign_distance = torch.stack([axis0_sign_distance, axis1_sign_distance, axis2_sign_distance], dim=1)    
        distances = torch.max(torch.relu(stacked_sign_distance), dim=1).values
        #distances = distances.permute(1, 0)
    else:
        print('wrong point to cube distance option')
        exit()

    return distances

def get_box_deformation_batched(deform_zs):

    xscales = deform_zs[:, 0]
    yscales = deform_zs[:, 1]
    zscales = deform_zs[:, 2]

    x_identity_matrix = torch.eye(3, dtype=torch.float, device=device)
    x_identity_matrix[1][1] = 0.0
    x_identity_matrix[2][2] = 0.0
    x_identity_matrices = x_identity_matrix.repeat(len(deform_zs), 1, 1)
    repeated_xscales = (xscales.unsqueeze(dim=1).unsqueeze(dim=1)).repeat(1, 3, 3) 
    x_scale_matrices = x_identity_matrices * repeated_xscales

    y_identity_matrix = torch.eye(3, dtype=torch.float, device=device)
    y_identity_matrix[0][0] = 0.0
    y_identity_matrix[2][2] = 0.0
    y_identity_matrices = y_identity_matrix.repeat(len(deform_zs), 1, 1)
    repeated_yscales = (yscales.unsqueeze(dim=1).unsqueeze(dim=1)).repeat(1, 3, 3) 
    y_scale_matrices = y_identity_matrices * repeated_yscales

    z_identity_matrix = torch.eye(3, dtype=torch.float, device=device)
    z_identity_matrix[0][0] = 0.0
    z_identity_matrix[1][1] = 0.0
    z_identity_matrices = z_identity_matrix.repeat(len(deform_zs), 1, 1)
    repeated_zscales = (zscales.unsqueeze(dim=1).unsqueeze(dim=1)).repeat(1, 3, 3) 
    z_scale_matrices = z_identity_matrices * repeated_zscales

    scale_matrices = x_scale_matrices + y_scale_matrices + z_scale_matrices

    return scale_matrices

def perform_cube_deformation_core(pcs, pc_normals, zs, ts, rotmats, size_changes, pose_changes, loc_changes):
    
    deformed_pcs = pcs
    deformed_pc_normals = pc_normals

    deformed_pcs = deformed_pcs - ts.unsqueeze(dim=1)

    if size_changes is not None:
        scale_mats = get_box_deformation_batched(1 + size_changes)
        deformed_pcs = torch.bmm(rotmats.inverse(), deformed_pcs.transpose(1,2)).transpose(1,2)
        deformed_pcs = torch.bmm(scale_mats, deformed_pcs.transpose(1,2)).transpose(1,2)    
        deformed_pcs = torch.bmm(rotmats, deformed_pcs.transpose(1,2)).transpose(1,2)
        if pc_normals is not None:
            deformed_pc_normals = torch.bmm(scale_mats.inverse(), pc_normals.transpose(1,2)).transpose(1,2)

    if pose_changes is not None:
        axes = pose_changes[:, 0:3]
        axes = axes/torch.norm(axes, dim=1).unsqueeze(dim=1)
        centers = torch.zeros(axes.shape, device=device)
        angles = pose_changes[:, 3]
        pose_mats = getRotMatrixHomo_batched(axes, centers, angles)[:, :, 0:3]
        deformed_pcs = torch.bmm(pose_mats, deformed_pcs.transpose(1,2)).transpose(1,2) 
    
    if loc_changes is not None:
        deformed_pcs = deformed_pcs + loc_changes.unsqueeze(dim=1).repeat_interleave(deformed_pcs.shape[1], dim=1)

    deformed_pcs = deformed_pcs + ts.unsqueeze(dim=1)

    return deformed_pcs, deformed_pc_normals
    
def perform_cube_deformation(pc, pc_normals, cubes, size_changes, pose_changes, loc_changes):

    zs = []
    ts = []
    rotmats = []
    for i in range(len(cubes)):
        zs.append(torch.tensor(cubes[i].size, device=device, dtype=torch.float))
        ts.append(torch.tensor(cubes[i].center, device=device, dtype=torch.float))
        #ts.append(torch.tensor(cubes[i].pivot, device=device, dtype=torch.float))
        rotmats.append(torch.tensor(cubes[i].rotmat, device=device, dtype=torch.float))
    zs = torch.stack(zs)
    ts = torch.stack(ts)
    rotmats = torch.stack(rotmats)

    dists = get_pc_to_cubes_distance(pc, zs, ts, rotmats, 'outside')  
    weights = get_decay(dists)
    largest_indices = torch.topk(weights, k=1, dim=1, largest=True).indices.squeeze(dim=1)
    one_hot = torch.nn.functional.one_hot(largest_indices, num_classes=len(cubes))
    one_hot = one_hot.unsqueeze(dim=2).repeat_interleave(3, dim=2)
    repeated_pc = pc.unsqueeze(dim=0).repeat_interleave(len(cubes), dim=0)
    repeated_pc_normals = pc_normals.unsqueeze(dim=0).repeat_interleave(len(cubes), dim=0)
    deformed_pc, deformed_pc_normals = perform_cube_deformation_core(repeated_pc, repeated_pc_normals, zs, ts, rotmats, size_changes, pose_changes, loc_changes)
    deformed_pc = deformed_pc.permute(1, 0, 2)
    deformed_pc_normals = deformed_pc_normals.permute(1, 0, 2)
    deformed_pc = torch.sum(deformed_pc * one_hot, dim=1)
    deformed_pc_normals = torch.sum(deformed_pc_normals * one_hot, dim=1)
    deformed_pc_normals = deformed_pc_normals/torch.norm(deformed_pc_normals, dim=1).unsqueeze(dim=1)
    return deformed_pc, deformed_pc_normals

def perferm_point_size_change(pc, box, size_change):
    deformed_pc = pc
    #deformed_pc = deformed_pc - box.center
    scale_mat = get_box_deformation_batched(1 + size_change.unsqueeze(dim=0))[0]
    #print('scale_mat', scale_mat)
    #deformed_pc = torch.matmul(box.rotmat.inverse(), deformed_pc.transpose(0,1)).transpose(0,1)
    deformed_pc = torch.matmul(scale_mat, deformed_pc.transpose(0,1)).transpose(0,1)    
    #deformed_pc = torch.matmul(box.rotmat, deformed_pc.transpose(0,1)).transpose(0,1)
    #deformed_pc = deformed_pc + box.center
    return deformed_pc

def perferm_direction_size_change(pc_normal, box, size_change):
    deformed_pc_normal = pc_normal
    scale_mat = get_box_deformation_batched(1 + size_change.unsqueeze(dim=0))[0]
    deformed_pc_normal = torch.matmul(scale_mat.inverse(), deformed_pc_normal.transpose(0,1)).transpose(0,1)
    deformed_pc_normal = deformed_pc_normal/torch.norm(deformed_pc_normal, dim=1).unsqueeze(dim=1)
    return  deformed_pc_normal

def perferm_point_pose_change(pc, box, pose_change):

    if box.pose_rotation_axis is None:
        return pc

    deformed_pc = pc
    deformed_pc = deformed_pc - box.center
    axes = box.pose_rotation_axis.unsqueeze(dim=0)
    #axes = pose_change[0:3].unsqueeze(dim=0)
    axes = axes/torch.norm(axes, dim=1).unsqueeze(dim=1)
    centers = torch.zeros(axes.shape, device=device)
    angles = torch.stack([pose_change[0]])
    pose_mat = getRotMatrixHomo_batched(axes, centers, angles)[0, 0:3, 0:3]
    deformed_pc = torch.matmul(pose_mat, deformed_pc.transpose(0,1)).transpose(0,1) 
    deformed_pc = deformed_pc + box.center
    return deformed_pc

def perferm_direction_pose_change(dir, box, pose_change):

    if box.pose_rotation_axis is None:
        return dir

    deformed_dir = dir
    axes = box.pose_rotation_axis.unsqueeze(dim=0)
    #axes = pose_change[0:3].unsqueeze(dim=0)
    axes = axes/torch.norm(axes, dim=1).unsqueeze(dim=1)
    centers = torch.zeros(axes.shape, device=device)
    angles = torch.stack([pose_change[0]])
    pose_mat = getRotMatrixHomo_batched(axes, centers, angles)[0, 0:3, 0:3]
    deformed_dir = torch.matmul(pose_mat, deformed_dir.transpose(0,1)).transpose(0,1) 
    return deformed_dir

def perferm_loc_change(pc, box, loc_change):
    deformed_pc = pc
    deformed_pc = deformed_pc + loc_change.unsqueeze(dim=0).repeat_interleave(len(deformed_pc), dim=0)
    return deformed_pc

def get_obb_overlap_loss(obbs):

    zs = []
    ts = []
    rotmats = []
    for i in range(len(obbs)):
        zs.append(obbs[i].size)
        ts.append(obbs[i].center)
        rotmats.append(obbs[i].rotmat)
    zs = torch.stack(zs)
    ts = torch.stack(ts)
    rotmats = torch.stack(rotmats)
    
    distances = []
    for i in range(len(obbs)):
        vertices_i = obbs[i].get_surface_points()

        other_zs = []
        other_ts = []
        other_rotmats = []
        for j in range(len(zs)):
            if j != i:
                other_zs.append(zs[j])
                other_ts.append(ts[j])
                other_rotmats.append(rotmats[j])
        other_zs = torch.stack(other_zs)
        other_ts = torch.stack(other_ts)
        other_rotmats = torch.stack(other_rotmats)
        dis_i = get_pc_to_cubes_distance(vertices_i, other_zs, other_ts, other_rotmats, 'inside')
        distances.append(torch.mean(torch.mean(dis_i, dim=1)))

    return torch.mean(torch.stack(distances))

def get_obb_compact_loss(pc, obbs):

    zs = []
    ts = []
    rotmats = []
    for i in range(len(obbs)):
        zs.append(obbs[i].size)
        ts.append(obbs[i].center)
        rotmats.append(obbs[i].rotmat)
    zs = torch.stack(zs)
    ts = torch.stack(ts)
    rotmats = torch.stack(rotmats)
    distances = get_pc_to_cubes_distance(pc, zs, ts, rotmats, 'inside')
    return torch.mean(torch.max(distances, dim=1).values) 

def get_target_coverage_loss(pc, obbs):

    zs = []
    ts = []
    rotmats = []
    for i in range(len(obbs)):
        zs.append(obbs[i].size)
        ts.append(obbs[i].center)
        rotmats.append(obbs[i].rotmat)
    zs = torch.stack(zs)
    ts = torch.stack(ts)
    rotmats = torch.stack(rotmats)

    outside_distances = get_pc_to_cubes_distance(pc, zs, ts, rotmats, 'outside')
    #print('outside_distances shape', outside_distances.shape)
    min_outside_distances = torch.min(outside_distances, dim=1).values

    #inside_distances = get_pc_to_cubes_distance(pc, zs, ts, rotmats, 'inside')
    #print('inside_distances shape', inside_distances.shape)
    #max_inside_distances = torch.max(inside_distances, dim=1).values

    return torch.mean(min_outside_distances)

def assign_pc_voronoi(pc, obbs):

    pc = torch.tensor(pc, device=device, dtype=torch.float)

    zs = []
    ts = []
    rotmats = []
    for i in range(len(obbs)):
        zs.append(torch.tensor(obbs[i].size, device=device, dtype=torch.float))
        ts.append(torch.tensor(obbs[i].center, device=device, dtype=torch.float))
        rotmats.append(torch.tensor(obbs[i].rotmat, device=device, dtype=torch.float))
    zs = torch.stack(zs)
    ts = torch.stack(ts)
    rotmats = torch.stack(rotmats)

    dists = get_pc_to_cubes_distance(pc, zs, ts, rotmats, 'outside')  
    #weights = get_decay(dists)
    #largest_indices = torch.topk(weights, k=1, dim=1, largest=True).indices.squeeze(dim=1)
    
    min_indices = torch.min(dists, dim=1).indices
    #print('min_indices', min_indices.shape)
    #exit()

    #print('(largest_indices == obb_index).nonzero()', (largest_indices == obb_index).nonzero().shape)
    



    all_pcs = []
    all_pc_indices = []
    for obb_index in range(len(obbs)):
        pc_indices = (min_indices == obb_index).nonzero().squeeze(dim=1)
        all_pc_indices.append(pc_indices)
        all_pcs.append(pc[pc_indices])

    #display_pcs(all_pcs)
    #exit()

    for i in range(len(all_pc_indices)):
        for j in range(len(all_pc_indices)):
            inter = set(all_pc_indices[i]).intersection(set(all_pc_indices[j]))
            if len(inter) > 0:
                display_pcs([pc[inter]])
                exit()


    #display_pcs([pc[all_pc_indices[1]]])
    #exit()

    return all_pc_indices


    return [index.item() for index in assigned_pc_indices]

def assign_pc(pc, obb):

    obbs = [obb]
    pc = torch.tensor(pc, device=device, dtype=torch.float)

    zs = []
    ts = []
    rotmats = []
    for i in range(len(obbs)):
        zs.append(torch.tensor(obbs[i].size, device=device, dtype=torch.float))
        ts.append(torch.tensor(obbs[i].center, device=device, dtype=torch.float))
        rotmats.append(torch.tensor(obbs[i].rotmat, device=device, dtype=torch.float))
    zs = torch.stack(zs)
    ts = torch.stack(ts)
    rotmats = torch.stack(rotmats)

    eps = 0.001
    dists = get_pc_to_cubes_distance(pc, zs, ts, rotmats, 'outside').squeeze(dim=1)
    #print('dists', dists.shape)
    assigned_pc_indices = (dists < eps).nonzero().squeeze(dim=1)
    #print('assigned_pc_indices shape', assigned_pc_indices.shape)

    return assigned_pc_indices

def obb_distance(obb_a, obb_b):

    obb_a_points = obb_a.get_surface_points()
    obb_a_points = torch.tensor(obb_a_points,device=device)

    obb_b_size = torch.tensor(obb_b.size, device=device)
    obb_b_center = torch.tensor(obb_b.center, device=device)
    obb_b_rotmat = torch.tensor(obb_b.rotmat, device=device)
    
    dists_a2b = get_pc_to_cubes_distance(obb_a_points, torch.stack([obb_b_size]), torch.stack([obb_b_center]), torch.stack([obb_b_rotmat]), 'outside')
    min_dist_a2b = torch.min(dists_a2b.squeeze())
    #print('dists_a2b shape', dists_a2b.shape)
    #print('min_dist_a2b', min_dist_a2b)

    obb_b_points = obb_b.get_surface_points()
    obb_b_points = torch.tensor(obb_b_points,device=device)

    obb_a_size = torch.tensor(obb_a.size, device=device)
    obb_a_center = torch.tensor(obb_a.center, device=device)
    obb_a_rotmat = torch.tensor(obb_a.rotmat, device=device)
    
    dists_b2a = get_pc_to_cubes_distance(obb_b_points, torch.stack([obb_a_size]), torch.stack([obb_a_center]), torch.stack([obb_a_rotmat]), 'outside')
    min_dist_b2a = torch.min(dists_b2a.squeeze())

    distance = min(min_dist_a2b, min_dist_b2a)
    
    return distance


def is_intersected(box_a, box_b):

    tol = 0.00

    points_a = box_a.get_surface_points()
    points_b = box_b.get_surface_points()

    #print('torch.abs(torch.matmul(points_a-box_b.center, box_b.rotmat[:, 0].unsqueeze(dim=1))) shape', torch.abs(torch.matmul(points_a-box_b.center, box_b.rotmat[:, 0].unsqueeze(dim=1))).shape)
    #print('box_b.size[0] - tol shape', (box_b.size[0] - tol).shape)
    #exit()

    a2b_axis0_dists = ((box_b.size[0] - tol) - torch.abs(torch.matmul(points_a-box_b.center, box_b.rotmat[:, 0].unsqueeze(dim=1)))).squeeze(dim=1)
    a2b_axis1_dists = ((box_b.size[1] - tol) - torch.abs(torch.matmul(points_a-box_b.center, box_b.rotmat[:, 1].unsqueeze(dim=1)))).squeeze(dim=1)
    a2b_axis2_dists = ((box_b.size[2] - tol) - torch.abs(torch.matmul(points_a-box_b.center, box_b.rotmat[:, 2].unsqueeze(dim=1)))).squeeze(dim=1)
    
    #print('a2b_axis0_dists', a2b_axis0_dists)
    #print('a2b_axis0_dists shape', a2b_axis0_dists.shape)
    a2b_axis0_overlap_dist = torch.max(torch.relu(a2b_axis0_dists))
    a2b_axis1_overlap_dist = torch.max(torch.relu(a2b_axis1_dists))
    a2b_axis2_overlap_dist = torch.max(torch.relu(a2b_axis2_dists))
    a2b_overlap_dist = torch.min(torch.stack([a2b_axis0_overlap_dist, a2b_axis1_overlap_dist, a2b_axis2_overlap_dist]))
    #print('a2b_overlap_dist', a2b_overlap_dist)

    return a2b_overlap_dist


    #b2a_axis0_dists = ((box_a.size[0] - tol) - torch.abs(torch.matmul(points_b-box_a.center, box_a.rotmat[:, 0].unsqueeze(dim=1)))).squeeze(dim=1)
    #b2a_axis1_dists = ((box_a.size[1] - tol) - torch.abs(torch.matmul(points_b-box_a.center, box_a.rotmat[:, 1].unsqueeze(dim=1)))).squeeze(dim=1)
    #b2a_axis1_dists = ((box_a.size[2] - tol) - torch.abs(torch.matmul(points_b-box_a.center, box_a.rotmat[:, 2].unsqueeze(dim=1)))).squeeze(dim=1)
    #torch.max(torch.relu(b2a_axis0_dists))

    '''
    a_in_b = True
    if (torch.abs(torch.matmul(points_a-box_b.center, box_b.rotmat[:, 0].unsqueeze(dim=1))) > box_b.size[0] - tol).squeeze(dim=1).all() or \
        (torch.abs(torch.matmul(points_a-box_b.center, box_b.rotmat[:, 1].unsqueeze(dim=1))) > box_b.size[1] - tol).squeeze(dim=1).all() or \
          (torch.abs(torch.matmul(points_a-box_b.center, box_b.rotmat[:, 2].unsqueeze(dim=1))) > box_b.size[2] - tol).squeeze(dim=1).all():
            a_in_b = False
            
    b_in_a = True
    if (torch.abs(torch.matmul(points_b-box_a.center, box_a.rotmat[:, 0].unsqueeze(dim=1))) > box_a.size[0] - tol).squeeze(dim=1).all() or \
        (torch.abs(torch.matmul(points_b-box_a.center, box_a.rotmat[:, 1].unsqueeze(dim=1))) > box_a.size[1] - tol).squeeze(dim=1).all() or \
          (torch.abs(torch.matmul(points_b-box_a.center, box_a.rotmat[:, 2].unsqueeze(dim=1))) > box_a.size[2] - tol).squeeze(dim=1).all() :
            b_in_a = False

    if a_in_b is True or b_in_a is True:
        #display_pcs_and_cubes([],[box_a, box_b], 'inter')
        return True
    else:
        #display_pcs_and_cubes([],[box_a, box_b], 'not inter')
        return False
    '''




    