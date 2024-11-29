
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
from obb_core import *
from util_geo import *


def get_split_score(local_ratio, dir_range):

    if local_ratio == 0:
        return 0.0

    eps = 0.03
    #return local_ratio * 1.0/(dir_range+eps)
    return local_ratio * 1.0/(dir_range+eps)
        
def split_along_dir(box, dir_index, vis=False):

    standard_pc = box.pc - box.center
    standard_pc = np.matmul(np.linalg.inv(box.rotmat), standard_pc.transpose(1,0)).transpose(1,0)
    dir_min = np.min(standard_pc[:, dir_index])
    dir_max = np.max(standard_pc[:, dir_index])

    if vis:
        display_pcs([standard_pc])
    
    split_delta = 0.1
    voxels = []
    
    section_pc_2ds = []
    inside_pc_2ds = []
    outside_pc_2ds = []
    section_pc_3ds = []

    dir_range = dir_max - dir_min
    section_count = int(np.floor(dir_range/split_delta))

    sections = []
    section = dir_min

    #print('dir_min', dir_min)
    #print('dir_max', dir_max)
    #print('section_count', section_count)

    count = 0
    while count < section_count:
        section = dir_min + count * split_delta
        section_pc_3d = standard_pc[np.logical_and(standard_pc[:, dir_index] >= section, standard_pc[:, dir_index] < section+split_delta)]
        section_pc_2d = project_to_2d(section_pc_3d, dir_index)
        voxel, inside_pc_2d, outside_pc_2d = get_2d_voxel(section_pc_2d)
        voxels.append(voxel)
        sections.append(section)
        section_pc_3ds.append(section_pc_3d)
        section_pc_2ds.append(section_pc_2d)
        inside_pc_2ds.append(inside_pc_2d)
        outside_pc_2ds.append(outside_pc_2d)
        count += 1

    min_split_score = 9999
    split_section = 0
    window_size = 1
    if len(sections) > 0:
        for i in range(0, len(sections)):
            if i > 0:
                voxel = voxels[i]
                voxel_area = np.sum(voxel)
                
                min_local_ratio = np.inf
                for j in range(max(0, i-window_size), i):
                    prev_voxel = voxels[j]
                    prev_voxel_area = np.sum(prev_voxel)
                    if voxel_area == 0 and prev_voxel_area == 0:
                        local_ratio = 1.0
                    else:
                        local_ratio = min(voxel_area, prev_voxel_area)/max(voxel_area, prev_voxel_area)
                    
                    if local_ratio < min_local_ratio:
                        min_local_ratio = local_ratio
                       
                split_score = get_split_score(min_local_ratio, dir_range)

                if split_score <= min_split_score:
                    min_split_score = split_score
                    split_section = sections[i] 

    return split_section, min_split_score


def double_split(box, sym_dir_index):

    eps = 0.001

    score, pc0, pc1, split_dir_index, split_section = split(box)

    standard_pc = box.pc - box.center
    standard_pc = np.matmul(np.linalg.inv(box.rotmat), standard_pc.transpose(1,0)).transpose(1,0)

    #print('split_section', split_section)

    if split_dir_index == sym_dir_index:

        if split_section > 0:
            split_section_neg = -split_section
            split_section_pos = split_section 
        else:
            split_section_neg = split_section
            split_section_pos = -split_section 

        split_section_neg += eps
        split_section_pos -= eps

        #print('split_section_neg', split_section_neg)
        #print('split_section_pos', split_section_pos)
    
        values = standard_pc[:, split_dir_index]

        standard_pc_neg = copy.deepcopy(standard_pc[values < split_section_neg])
        pc_neg = np.matmul(box.rotmat, standard_pc_neg.transpose(1,0)).transpose(1,0)
        pc_neg = pc_neg + box.center
        #display_pcs([pc_neg])
        #exit()

        standard_pc_mid = copy.deepcopy(standard_pc[np.logical_and(values >= split_section_neg, values <= split_section_pos)])

        pc_mid = np.matmul(box.rotmat, standard_pc_mid.transpose(1,0)).transpose(1,0)
        pc_mid = pc_mid + box.center
        #display_pcs([pc_mid])

        standard_pc_pos = copy.deepcopy(standard_pc[values > split_section_pos])
        pc_pos = np.matmul(box.rotmat, standard_pc_pos.transpose(1,0)).transpose(1,0)
        pc_pos = pc_pos + box.center
        #display_pcs([pc_pos])
        
        #display_pcs([standard_pc_neg, standard_pc_mid, standard_pc_pos], 'standard' + 'pos' + str(split_section_pos) + 'neg' + str(split_section_neg))
        #display_pcs([pc_neg, pc_mid, pc_pos], str(score))

        if len(pc_mid) <= min_pc_size and min(pc_pos[:, split_dir_index]) - max(pc_neg[:, split_dir_index]) < 0.1:
            return np.inf, pc_mid, pc_pos, pc_neg
        else:
            
            return score, pc_mid, pc_pos, pc_neg 
    else:
        #display_pcs([pc0, pc1], 'score '+str(score))
        #exit()
        return score, pc0, pc1, None


def split(box, vis=False):

    split_ratio = np.inf
    split_section = None
    split_dir_index = None

    for i in range(len(box.size)):
        if box.size[i]/max(box.size) > split_axis_ratio:
            split_section_i, split_ratio_i = split_along_dir(box, i, vis)
            
            if split_ratio_i < split_ratio:
                split_ratio = split_ratio_i
                split_section = split_section_i
                split_dir_index = i

    standard_pc = box.pc - box.center
    standard_pc = np.matmul(np.linalg.inv(box.rotmat), standard_pc.transpose(1,0)).transpose(1,0)
    #if split_ratio < 1.0:
    values = standard_pc[:, split_dir_index]
    #pc0 = copy.deepcopy(box.pc[values >= split_section])
    #pc1 = copy.deepcopy(box.pc[values < split_section])

    standard_pc0 = standard_pc[values >= split_section]
    pc0 = np.matmul(box.rotmat, standard_pc0.transpose(1,0)).transpose(1,0)
    pc0 = pc0 + box.center 

    standard_pc1 = standard_pc[values < split_section]
    pc1 = np.matmul(box.rotmat, standard_pc1.transpose(1,0)).transpose(1,0)
    pc1 = pc1 + box.center 

    return split_ratio, pc0, pc1, split_dir_index, split_section
    #else:
        #return False, None, None, None, None


def same(box0, box1):
    eps = 0.00001
    if np.linalg.norm(np.array(box0.center) - np.array(box1.center)) < eps and np.linalg.norm(np.array(box0.size) - np.array(box1.size)) < eps:
        return True
    else:
        return False

def include(box, mother_box):

    corners = box.get_corner_points()
    for i in range(len(corners)):
        if np.abs(np.dot(corners[i]-mother_box.center, mother_box.rotmat[:, 0])) > mother_box.size[0] * 1.5 or \
            np.abs(np.dot(corners[i]-mother_box.center, mother_box.rotmat[:, 1])) > mother_box.size[1] * 1.5 or \
            np.abs(np.dot(corners[i]-mother_box.center, mother_box.rotmat[:, 2])) > mother_box.size[2] * 1.5:
            return False
    return True            

def filter_boxes(boxes, mother_boxes):
    
    to_continue = True
    while to_continue:
        to_continue = False
        for i in range(len(boxes)):

            axis_indices = [0, 1, 2]
            sorted_sizes = [x for x, y in sorted(zip(boxes[i].size, axis_indices), reverse=True)]
            sorted_indices = [y for x, y in sorted(zip(boxes[i].size, axis_indices), reverse=True)]

            if sorted_sizes[2]/sorted_sizes[0] < 1/10:
                to_continue = True
                break

            for j in range(len(mother_boxes)):
                if same(boxes[i], mother_boxes[j]) is False and include(boxes[i], mother_boxes[j]):
                    to_continue = True
                    break
            
            if to_continue:
                break

        if to_continue:
            boxes.pop(i)

    return boxes


min_pc_size = 20

def try_split_mid_obb(obb, sym_dir_index):
    score, child_pc_0, child_pc_1, child_pc_2 = double_split(obb, sym_dir_index)
    return score 

def try_split_pos_obb(obb):
    score, pos_child_pc_0, pos_child_pc_1, _, _ = split(obb)
    return score

def split_mid_obb(obb, queue_mid, queue_pos, sym_dir_index):

    print('split_mid_obb')

    mid_obbs = []

    _, child_pc_0, child_pc_1, child_pc_2 = double_split(obb, sym_dir_index)
    
    if child_pc_0 is not None and child_pc_1 is not None and child_pc_2 is not None:
        
        if len(child_pc_0) > min_pc_size:
            child_box_mid = get_3d_axis_aligned_bbox(child_pc_0)
            queue_mid.append(child_box_mid)
        
        if len(child_pc_1) > min_pc_size :
            child_box_pos = get_3d_axis_aligned_bbox(child_pc_1)
            queue_pos.append(child_box_pos)
        
        if len(child_pc_0) < min_pc_size and len(child_pc_1) < min_pc_size:
            mid_obbs.append(obb)
    else:
        #display_pcs([box_mid.pc])
        #display_pcs([child_pc_0, child_pc_1])
        
        if len(child_pc_0) > min_pc_size: 
            child_box_mid_0 = get_3d_axis_aligned_bbox(child_pc_0)
            queue_mid.append(child_box_mid_0)
        if len(child_pc_1) > min_pc_size:
            child_box_mid_1 = get_3d_axis_aligned_bbox(child_pc_1)
            queue_mid.append(child_box_mid_1)
        
        if len(child_pc_0) < min_pc_size and len(child_pc_1) < min_pc_size:
            mid_obbs.append(obb)

    return mid_obbs

def split_pos_obb(obb, queue_pos):

    print('split_pos_obb')

    pos_obbs = []
    _, pos_child_pc_0, pos_child_pc_1, _, _ = split(obb)
    
    if len(pos_child_pc_0) > min_pc_size:
        pos_child_box_0 = get_3d_axis_aligned_bbox(pos_child_pc_0)
        queue_pos.append(pos_child_box_0)
    if len(pos_child_pc_1) > min_pc_size:
        pos_child_box_1 = get_3d_axis_aligned_bbox(pos_child_pc_1)
        queue_pos.append(pos_child_box_1)

    if len(pos_child_pc_0) < min_pc_size and len(pos_child_pc_1) < min_pc_size:
        pos_obbs.append(obb)

    return pos_obbs

def get_obb_to_split(queue_mid, queue_pos, sym_dir_index):
    
    mid_scores = []
    for obb in queue_mid:
        score = try_split_mid_obb(obb, sym_dir_index)
        mid_scores.append(score)

    pos_scores = []
    for obb in queue_pos:
        score = try_split_pos_obb(obb)
        pos_scores.append(score)

    print('mid_scores', mid_scores)
    print('pos_scores', pos_scores)
    
    min_mid_index = -1
    min_mid_score = np.inf
    min_pos_index = -1
    min_pos_score = np.inf

    if len(mid_scores) > 0:
        min_mid_index = np.argmin(mid_scores)
        min_mid_score = mid_scores[min_mid_index]

    if len(pos_scores) > 0:
        min_pos_index = np.argmin(pos_scores)
        min_pos_score = pos_scores[min_pos_index]

    if min_mid_score == np.inf and min_pos_score == np.inf:
        return None, None

    if min_mid_score <= min_pos_score:
        box_to_split = copy.deepcopy(queue_mid[min_mid_index])
        queue_mid.pop(min_mid_index)
        return box_to_split, True
    else:
        box_to_split = copy.deepcopy(queue_pos[min_pos_index])
        queue_pos.pop(min_pos_index)
        return box_to_split, False

def build_obbs_by_level_sym(split_num, mesh, exp_folder=None):
    #temp_mesh = copy.deepcopy(mesh)
    #temp_mesh.vertices = temp_mesh.vertices * 1.1
    _, inside_voxel_pc, outside_voxel_pc = get_3d_voxel(mesh)
    pc = inside_voxel_pc

    sym_dir_index = 0

    mid_obbs = []
    pos_obbs = []
    
    queue_pos = []
    queue_mid = []

    box = get_3d_axis_aligned_bbox(pc)
    queue_mid.append(box)

    split_count = 0
    while split_count < split_num:
        obb, is_mid = get_obb_to_split(queue_mid, queue_pos, sym_dir_index)
        if obb is None:
            break

        if is_mid:
            mid_obbs += split_mid_obb(obb, queue_mid, queue_pos, sym_dir_index)
        else:
            pos_obbs += split_pos_obb(obb, queue_pos)
        split_count += 1

    mid_obbs += queue_mid
    pos_obbs += queue_pos

    neg_obbs = []
    for pos_box in pos_obbs:
        neg_box = copy.deepcopy(pos_box)
        neg_box.center[0] = - pos_box.center[0]
        neg_box.rotmat[:, 0][0] = -pos_box.rotmat[:, 0][0]
        neg_box.rotmat[:, 2][0] = -pos_box.rotmat[:, 2][0]
        neg_obbs.append(neg_box)

    #for obb in mid_obbs+pos_obbs+neg_obbs:
        #obb.size = obb.size * 0.9
        #obb.initialize()

    vertices = torch.tensor(mesh.vertices, device=device, dtype=torch.float)

    obbs = mid_obbs + pos_obbs + neg_obbs

    filtered_mid_obbs = []
    for i in range(len(mid_obbs)):
        pc_indices = assign_pc_voronoi(vertices, obbs)[i]
        if len(pc_indices) < 10:
            continue
        filtered_mid_obbs.append(copy.deepcopy(mid_obbs[i]))
    
    obbs = filtered_mid_obbs + pos_obbs + neg_obbs

    filtered_pos_obbs = []
    filtered_neg_obbs = []
    for i in range(len(pos_obbs)):
        pos_pc_indices = assign_pc_voronoi(vertices, obbs)[i+len(filtered_mid_obbs)]
        neg_pc_indices = assign_pc_voronoi(vertices, obbs)[i+len(filtered_mid_obbs)+len(pos_obbs)]
        if len(pos_pc_indices) < 10 or len(neg_pc_indices) < 10:
            continue
        
        filtered_pos_obbs.append(copy.deepcopy(pos_obbs[i]))
        filtered_neg_obbs.append(copy.deepcopy(neg_obbs[i]))

    filtered_obbs = filtered_mid_obbs + filtered_pos_obbs + filtered_neg_obbs
    all_pc_indices = assign_pc_voronoi(vertices, filtered_obbs)

    refined_mid_obbs = []
    for i in range(len(filtered_mid_obbs)):
        pc_indices = all_pc_indices[i]
        #refined_obb = get_3d_axis_aligned_bbox(vertices[pc_indices])
        refined_mid_obb = filtered_mid_obbs[i]
        refined_mid_obb.initialize()
        refined_mid_obb.pc_indices = pc_indices
        refined_mid_obbs.append(refined_mid_obb)

    refined_pos_obbs = []
    refined_neg_obbs = []
    for i in range(len(filtered_pos_obbs)):
        pos_pc_indices = all_pc_indices[i+len(filtered_mid_obbs)]
        neg_pc_indices = all_pc_indices[i+len(filtered_mid_obbs)+len(filtered_pos_obbs)]
        
        refined_pos_obb = filtered_pos_obbs[i]
        refined_pos_obb.pc_indices = pos_pc_indices
        refined_pos_obb.initialize()
        refined_pos_obbs.append(refined_pos_obb)
        
        refined_neg_obb = filtered_neg_obbs[i]
        refined_neg_obb.pc_indices = neg_pc_indices
        refined_neg_obb.initialize()
        refined_neg_obbs.append(refined_neg_obb)

    return refined_mid_obbs, refined_pos_obbs, refined_neg_obbs

def build_obbs_by_level_nosym(split_num, mesh, exp_folder):
    
    _, inside_voxel_pc, outside_voxel_pc = get_3d_voxel(mesh)
    pc = inside_voxel_pc

    sym_dir_index = 0

    obbs = []
    queue = []
    box = get_3d_axis_aligned_bbox(pc)
    queue.append(box)

    split_count = 0
    while split_count < split_num:
        obb, is_mid = get_obb_to_split([], queue, sym_dir_index)
        if is_mid:
            print('error')
            exit()
        else:
            obbs += split_pos_obb(obb, queue)
        split_count += 1

    obbs += queue

    vertices = torch.tensor(mesh.vertices, device=device, dtype=torch.float)
    
    filtered_obbs = []
    for i in range(len(obbs)):
        pc_indices = assign_pc_voronoi(vertices, obbs, i)
        try:
            if len(pc_indices) < 20:
                continue
        except:
            if len(pc_indices) < 20:
                continue
        filtered_obbs.append(copy.deepcopy(obbs[i]))

    refined_obbs = []
    for i in range(len(filtered_obbs)):
        pc_indices = assign_pc_voronoi(vertices, filtered_obbs, i)
        refined_obb = get_3d_axis_aligned_bbox(vertices[pc_indices])
        #refined_obb.size = obbs[i].size
        #refined_obb.center = obbs[i].center
        #refined_obb.rotmat = obbs[i].rotmat
        refined_obb.initialize()
        refined_obb.pc_indices = pc_indices
        refined_obbs.append(refined_obb)

    return refined_obbs

split_axis_ratio = 0.2
#split_ratio_threshold = 0.3
split_min_point_count = 20