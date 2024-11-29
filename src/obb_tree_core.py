
from collections import defaultdict
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
from obb_core import *
from util_geo import *

class ObbNode():
    def __init__(self):
        self.index = -1
        self.layer = -1
        self.pc_indices = None
        self.childs = []
        self.child_loc_follow_constraints = []
        self.child_size_follow_constraints = []

def pick_root_index(obbs):

    max_volume = 0
    root_index = 0
    for i in range(len(obbs)):
        volume = obbs[i].get_volume()
        if volume > max_volume:
            max_volume = volume
            root_index = i

    return root_index


def is_neighbor(obbs, mesh, obb_index_a, obb_index_b):
    if len(compute_bound_vertices(mesh, obbs, obb_index_a, obb_index_b))>0:
        return True
    else:
        return False

def get_neighbor_obbs(obbs, obb_index, mesh):
    nb_indices = []
    search_range = list(np.arange(0, len(obbs), 1))
    for i in search_range:
        if i != obb_index:
            if is_neighbor(obbs, mesh, obb_index, i):
                nb_indices.append(i)
    return nb_indices

def compute_bound_vertices(mesh, obbs, obb_a_index, obb_b_index):

    vertices = mesh.vertices
    faces = mesh.faces

    obb_a_vertex_indices = to_numpy(obbs[obb_a_index].pc_indices)
    obb_b_vertex_indices = to_numpy(obbs[obb_b_index].pc_indices)

    temp_mesh = trimesh.Trimesh(to_numpy(vertices), to_numpy(faces), process=False)
    adj_graph = trimesh.graph.vertex_adjacency_graph(temp_mesh)
    ret = list(nx.edge_boundary(adj_graph, obb_a_vertex_indices, obb_b_vertex_indices))
    return ret

def get_merged_obb(mesh, obbs):
    
    #display_pcs_and_cubes([mesh.vertices], obbs, 'before_merge')

    merged_pc_indices = None
    for i in range(len(obbs)):
        #obb_pc_indices = assign_pc(mesh.vertices, obbs[i])
        if merged_pc_indices is None:
            merged_pc_indices = obbs[i].pc_indices
        else:
            merged_pc_indices = torch.cat((merged_pc_indices, obbs[i].pc_indices), dim=0)

    merged_pc = mesh.vertices[merged_pc_indices]
    merged_obb = get_3d_axis_aligned_bbox(merged_pc)
    merged_obb.initialize()
    merged_obb.pc_indices = merged_pc_indices

    #display_pcs_and_cubes([mesh.vertices], [merged_obb], 'after_merge')
    #exit()

    return merged_obb

def compute_parallel_face_distance(obb_a, obb_b, main_dir, side_dir0, side_dir1, use_parent_union=False):
    
    obb_a_min0 = obb_a.center[side_dir0] - obb_a.size[side_dir0]
    obb_a_max0 = obb_a.center[side_dir0] + obb_a.size[side_dir0]
    obb_a_min1 = obb_a.center[side_dir1] - obb_a.size[side_dir1]
    obb_a_max1 = obb_a.center[side_dir1] + obb_a.size[side_dir1]

    obb_b_min0 = obb_b.center[side_dir0] - obb_b.size[side_dir0]
    obb_b_max0 = obb_b.center[side_dir0] + obb_b.size[side_dir0]
    obb_b_min1 = obb_b.center[side_dir1] - obb_b.size[side_dir1]
    obb_b_max1 = obb_b.center[side_dir1] + obb_b.size[side_dir1]

    inter0, inter1 = intersection_2d(obb_a_min0, obb_a_max0, obb_b_min0, obb_b_max0, obb_a_min1, obb_a_max1, obb_b_min1, obb_b_max1)
    if inter0 * inter1 > 0:
        obb_a_min_to_obb_b_min = abs((obb_a.center[main_dir] - obb_a.size[main_dir]) - (obb_b.center[main_dir] - obb_b.size[main_dir]))
        obb_a_min_to_obb_b_max = abs((obb_a.center[main_dir] - obb_a.size[main_dir]) - (obb_b.center[main_dir] + obb_b.size[main_dir]))
        obb_a_max_to_obb_b_min = abs((obb_a.center[main_dir] + obb_a.size[main_dir]) - (obb_b.center[main_dir] - obb_b.size[main_dir]))
        obb_a_max_to_obb_b_max = abs((obb_a.center[main_dir] + obb_a.size[main_dir]) - (obb_b.center[main_dir] + obb_b.size[main_dir]))
        distances = torch.stack([obb_a_min_to_obb_b_min, obb_a_min_to_obb_b_max, obb_a_max_to_obb_b_min, obb_a_max_to_obb_b_max])

        min_distance_index = torch.argmin(distances)
        min_distance = distances[min_distance_index]

        if use_parent_union:
            union0 = 2*obb_a.size[side_dir0]
            union1 = 2*obb_a.size[side_dir1]
        else:
            union0 = max(2*obb_a.size[side_dir0], 2*obb_b.size[side_dir0])
            union1 = max(2*obb_a.size[side_dir1], 2*obb_b.size[side_dir1])
        
        ratio0 = inter0/union0
        ratio1 = inter1/union1

    else:
        min_distance = np.inf
        min_distance_index = None
        ratio0 = 0
        ratio1 = 0

    return min_distance, min_distance_index, ratio0, ratio1

def is_valid_rot_ratio(ratio):
    low_rotation_threshold = ratio_low_threshold
    high_rotation_threshold = ratio_high_threshold
    if ratio >= low_rotation_threshold and ratio < high_rotation_threshold:
        return True
    else:
        return False

ratio_low_threshold = 0.15
ratio_high_threshold = 0.3
def compute_pose_rotation_axis(obbs, obb_a_index, obb_b_index, mid_sep):  
    
    obb_a = obbs[obb_a_index]
    obb_b = obbs[obb_b_index]

    dir0_distance, _, dir0_ratio0, dir0_ratio1 = compute_parallel_face_distance(obb_a, obb_b, 0, 1, 2, use_parent_union=True)
    dir1_distance, _, dir1_ratio0, dir1_ratio1 = compute_parallel_face_distance(obb_a, obb_b, 1, 0, 2, use_parent_union=True)
    dir2_distance, _, dir2_ratio0, dir2_ratio1 = compute_parallel_face_distance(obb_a, obb_b, 2, 0, 1, use_parent_union=True)

    if mid_sep is not None:
        if dir0_ratio0 < ratio_low_threshold or dir0_ratio1 < ratio_low_threshold or (obb_a_index<mid_sep and obb_b_index<mid_sep):
            dir0_ratio0 = -1
            dir0_ratio1 = -1
            dir0_distance = np.inf
    else:
        if dir0_ratio0 < ratio_low_threshold or dir0_ratio1 < ratio_low_threshold:
            dir0_ratio0 = -1
            dir0_ratio1 = -1
            dir0_distance = np.inf
    
    if dir1_ratio0 < ratio_low_threshold or dir1_ratio1 < ratio_low_threshold:
        dir1_ratio0 = -1
        dir1_ratio1 = -1
        dir1_distance = np.inf

    if dir2_ratio0 < ratio_low_threshold or dir2_ratio1 < ratio_low_threshold:
        dir2_ratio0 = -1
        dir2_ratio1 = -1
        dir2_distance = np.inf
    
    if mid_sep is not None:
        if (obb_a_index<mid_sep and obb_b_index<mid_sep):
            if is_valid_rot_ratio(dir0_ratio0) or is_valid_rot_ratio(dir0_ratio1):
                pose_dir = obb_b.rotmat[:, 0]
            else:
                pose_dir = None

    if dir0_distance == np.inf and dir1_distance == np.inf and dir2_distance == np.inf:
        pose_dir = None
    else:
        if dir0_distance < dir1_distance and dir0_distance < dir2_distance:
            if is_valid_rot_ratio(dir0_ratio0) or is_valid_rot_ratio(dir0_ratio1):
                if obb_b.size[1] > obb_b.size[2]:
                    pose_dir = obb_b.rotmat[:, 1]
                else:
                    pose_dir = obb_b.rotmat[:, 2]
            else:
                pose_dir = None
        if dir1_distance < dir0_distance and dir1_distance < dir2_distance:
            if is_valid_rot_ratio(dir1_ratio0) or is_valid_rot_ratio(dir1_ratio1):
                if obb_b.size[0] > obb_b.size[2]:
                    pose_dir = obb_b.rotmat[:, 0]
                else:
                    pose_dir = obb_b.rotmat[:, 2]
            else:
                pose_dir = None
        if dir2_distance < dir0_distance and dir2_distance < dir1_distance:
            if is_valid_rot_ratio(dir2_ratio0) or is_valid_rot_ratio(dir2_ratio1):
                if obb_b.size[0] > obb_b.size[1]:
                    pose_dir = obb_b.rotmat[:, 0]
                else:
                    pose_dir = obb_b.rotmat[:, 1]
            else:
                pose_dir = None
    
    return pose_dir


def compute_loc_follow_constraint_info(obbs, obb_a_index, obb_b_index, mid_sep):
    min_dis = np.inf
    obb_a_points = obbs[obb_a_index].get_surface_points()
    obb_b_points = obbs[obb_b_index].get_face_center_points()
    
    for f_i in range(len(obb_a_points)):
        for f_j in range(len(obb_b_points)):

            if obb_a_index < mid_sep and obb_b_index < mid_sep:
                if f_j < 2:
                    continue

            dis = torch.norm(obb_a_points[f_i] - obb_b_points[f_j])
            if dis < min_dis:
                min_dis = dis
                min_obb_a_point_index = f_i
                min_obb_b_point_index = f_j
    
    constraint = (obb_b_index, min_obb_a_point_index, min_obb_b_point_index)

    return constraint

def get_size_follow_constraint_vector(obbs, parent_index, size_follow_constraint):

    return torch.zeros(3, device=device)

    if size_follow_constraint is None:
        return torch.zeros(3, device=device)
    else:
        child_index = size_follow_constraint[0]
        child_independent_size_index = size_follow_constraint[1]
        child_dependent_size_index0 = size_follow_constraint[2]
        child_dependent_size_ratio0 = size_follow_constraint[3]
        child_dependent_size_index1 = size_follow_constraint[4]
        child_dependent_size_ratio1 = size_follow_constraint[5]

        size_follow_vector = torch.zeros(3, device=device)
        size_follow_vector[child_dependent_size_index0] = torch.tensor(child_dependent_size_ratio0, device=device, dtype=torch.float)
        size_follow_vector[child_dependent_size_index1] = torch.tensor(child_dependent_size_ratio1, device=device, dtype=torch.float)

        return size_follow_vector

def get_loc_follow_constraint_vector(obbs, parent_index, loc_follow_constraint):
    child_index = loc_follow_constraint[0]
    parent_surface_point_index = loc_follow_constraint[1]
    child_face_point_index = loc_follow_constraint[2]
    constraint_vector = obbs[parent_index].get_surface_points()[parent_surface_point_index] - obbs[child_index].get_face_center_points()[child_face_point_index]
    return constraint_vector


def visulize_obb_tree(root, obbs, mesh, filename):

    layered_obbs = defaultdict(list)
    
    all_constraint_points = []
    
    queue = [root]
    while len(queue) > 0:
        cur_node = queue.pop(0)
        #display_pcs_and_cubes([obbs[cur_node.index].get_surface_points()], [obbs[cur_node.index]], filename=None)
        layered_obbs[cur_node.layer].append(obbs[cur_node.index])
        for i, child_node in enumerate(cur_node.childs):
            queue.append(child_node)
            #display_pcs_and_cubes([mesh.vertices], [obbs[cur_node.index], obbs[child_node.index]], str(cur_node.child_size_follow_constraints[i]))

    obb_groups = []
    for k, v in layered_obbs.items():
        obb_groups.append(v)
    
    display_pcs_and_cube_groups([], obb_groups, filename, True)

    #bound_vertices = []
    #for pair in root.bound_vertex_mat:
        #bound_vertices.append(to_numpy(mesh.vertices[pair[0]]))
        #bound_vertices.append(to_numpy(mesh.vertices[pair[1]]))
    #display_pcs([bound_vertices])
