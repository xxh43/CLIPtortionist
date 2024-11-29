from cmath import cos
from tabnanny import verbose
from turtle import pen
import clip
from pyparsing import col
from tqdm import tqdm
import kaolin.ops.mesh
import kaolin as kal
import torch
from utils import device 
from render import Renderer
from mesh import Mesh
from Normalization import MeshNormalizer
import numpy as np
import random
import copy
import torchvision
import os
from PIL import Image
import argparse
from pathlib import Path
from torchvision import transforms
import joblib
from obb_core import *
from obb_tree_sym import *
from util_file import *
import random
import numpy as np
import pymeshlab 
import cma
from main_utils_core import *
from main_utils_sym import *

def refine_inter_node_by_range_sym(sim_low, sim_high, old_mid_obb_size_changes, old_pos_obb_size_changes, mid_obb_size_change_low, mid_obb_size_change_high, pos_obb_size_change_low, pos_obb_size_change_high, size_range, mid_obbs, pos_obbs, neg_obbs, mesh, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, augment_transform, clip_model, render, res, background):
    
    #print('sim_low', sim_low)
    #print('sim_high', sim_high)

    mid_obb_pose_changes = torch.zeros((len(mid_obbs), 1), device=device, dtype=torch.float)
    pos_obb_pose_changes = torch.zeros((len(pos_obbs), 1), device=device, dtype=torch.float)
    pose_range = 0.0
    
    mid_obb_loc_changes = torch.zeros((len(mid_obbs), 3), device=device, dtype=torch.float)
    pos_obb_loc_changes = torch.zeros((len(pos_obbs), 3), device=device, dtype=torch.float)
    loc_range = 0.0

    mid_obb_size_change_delta = (mid_obb_size_change_high - mid_obb_size_change_low)*0.5
    pos_obb_size_change_delta = (pos_obb_size_change_high - pos_obb_size_change_low)*0.5

    min_sim_dev = np.inf
    best_img = None
    best_sim = None

    resolution = 20
    for k in range(resolution):
        
        new_mid_obb_size_change = old_mid_obb_size_changes + k/resolution * mid_obb_size_change_delta
        new_pos_obb_size_change = old_pos_obb_size_changes + k/resolution * pos_obb_size_change_delta
        
        k_pos_sim, _, _, rendered_img = eval_result_sym(mesh, None, None, new_mid_obb_size_change, new_pos_obb_size_change, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, pose_range, loc_range, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, augment_transform, clip_model, render, res, background)

        #print('k_pos_sim', k_pos_sim)

        #if k_pos_sim > sim_low and k_pos_sim < sim_high:
            #return new_mid_obb_size_change, new_pos_obb_size_change

        if abs(k_pos_sim-(sim_low+sim_high)*0.5) < min_sim_dev:
            min_sim_dev = abs(k_pos_sim-(sim_low+sim_high)*0.5)
            best_img = rendered_img
            best_sim = k_pos_sim
            best_mid_obb_size_change = new_mid_obb_size_change
            best_pos_obb_size_change = new_pos_obb_size_change

        new_mid_obb_size_change = old_mid_obb_size_changes - k/resolution * mid_obb_size_change_delta
        new_pos_obb_size_change = old_pos_obb_size_changes - k/resolution * pos_obb_size_change_delta
        
        k_neg_sim, _, _, rendered_img = eval_result_sym(mesh, None, None, new_mid_obb_size_change, new_pos_obb_size_change, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, pose_range, loc_range, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, augment_transform, clip_model, render, res, background)

        #print('k_neg_sim', k_neg_sim)

        if k_neg_sim > sim_low and k_neg_sim < sim_high:
            return new_mid_obb_size_change, new_pos_obb_size_change

        if abs(k_neg_sim-(sim_low+sim_high)*0.5) < min_sim_dev:
            min_sim_dev = abs(k_neg_sim-(sim_low+sim_high)*0.5)
            best_img = rendered_img
            best_sim = k_neg_sim
            best_mid_obb_size_change = new_mid_obb_size_change
            best_pos_obb_size_change = new_pos_obb_size_change

    return best_mid_obb_size_change, best_pos_obb_size_change

def generate_sequence_sym(final_mid_obb_size_changes, final_pos_obb_size_changes, size_range, mesh, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges,
    clip_model, augment_transform, encoded_text, render, background, res, option):

    init_mid_obb_size_changes = torch.zeros((len(mid_obbs), 3), device=device, dtype=torch.float)
    init_pos_obb_size_changes = torch.zeros((len(pos_obbs), 3), device=device, dtype=torch.float)

    mid_obb_pose_changes = torch.zeros((len(mid_obbs), 1), device=device, dtype=torch.float)
    pos_obb_pose_changes = torch.zeros((len(pos_obbs), 1), device=device, dtype=torch.float)
    pose_range = 0.0
    
    mid_obb_loc_changes = torch.zeros((len(mid_obbs), 3), device=device, dtype=torch.float)
    pos_obb_loc_changes = torch.zeros((len(pos_obbs), 3), device=device, dtype=torch.float)
    loc_range = 0.0

    start_sim, _, _, rendered_img = eval_result_sym(mesh, None, None, init_mid_obb_size_changes, init_pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, pose_range, loc_range, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, augment_transform, clip_model, render, res, background)
    #torchvision.utils.save_image(rendered_img, 'start.jpg')
    end_sim, _, _, rendered_img = eval_result_sym(mesh, None, None, final_mid_obb_size_changes, final_pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, pose_range, loc_range, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, augment_transform, clip_model, render, res, background)
    #torchvision.utils.save_image(rendered_img, 'end.jpg')

    interp_num = 10
    half_interp_sims = []
    half_interp_mid_obb_size_changes = []
    half_interp_pos_obb_size_changes = []
    interp_sims = []
    interp_mid_obb_size_changes = []
    interp_pos_obb_size_changes = []
    for i in range(interp_num+1):
        #print('interpolation i', i)
        alpha = 1.0/interp_num * i
        interp_sim = (1 - alpha) * start_sim + alpha * end_sim
        interp_sims.append(interp_sim)
        interp_mid_obb_size_change = (1 - alpha) * init_mid_obb_size_changes + alpha * final_mid_obb_size_changes
        interp_mid_obb_size_changes.append(interp_mid_obb_size_change)
        interp_pos_obb_size_change = (1 - alpha) * init_pos_obb_size_changes + alpha * final_pos_obb_size_changes
        interp_pos_obb_size_changes.append(interp_pos_obb_size_change)

        half_alpha = alpha + 0.5/interp_num
        #half_interp_sim = (1 - half_alpha) * start_sim + half_alpha * end_sim
        #half_interp_sims.append(half_interp_sim)
        half_interp_mid_obb_size_change = (1 - half_alpha) * init_mid_obb_size_changes + half_alpha * final_mid_obb_size_changes
        half_interp_mid_obb_size_changes.append(half_interp_mid_obb_size_change)
        half_interp_pos_obb_size_change = (1 - half_alpha) * init_pos_obb_size_changes + half_alpha * final_pos_obb_size_changes
        half_interp_pos_obb_size_changes.append(half_interp_pos_obb_size_change)

    final_parameter_sequence = []
    for i in range(1, len(interp_sims)):

        print('i', i)

        half_i = i - 1

        real_sim, real_mesh, _, rendered_img = eval_result_sym(mesh, None, None, half_interp_mid_obb_size_changes[half_i], half_interp_pos_obb_size_changes[half_i], mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, pose_range, loc_range, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, augment_transform, clip_model, render, res, background)
        interp_sim_low = interp_sims[i-1]
        interp_sim_high = interp_sims[i] 

        if real_sim > interp_sim_low and real_sim < interp_sim_high:
            final_parameter_sequence.append((half_interp_mid_obb_size_changes[half_i], half_interp_pos_obb_size_changes[half_i], rendered_img, real_sim, real_mesh))
        else:
            refined_mid_obb_size_changes, refined_pos_obb_size_changes = refine_inter_node_by_range_sym(interp_sim_low, interp_sim_high, half_interp_mid_obb_size_changes[half_i], half_interp_pos_obb_size_changes[half_i], interp_mid_obb_size_changes[i-1], interp_mid_obb_size_changes[i], interp_pos_obb_size_changes[i-1], interp_pos_obb_size_changes[i], size_range, mid_obbs, pos_obbs, neg_obbs, mesh, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, augment_transform, clip_model, render, res, background)
            refined_sim, refined_mesh, _, rendered_img = eval_result_sym(mesh, None, None, refined_mid_obb_size_changes, refined_pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, pose_range, loc_range, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, augment_transform, clip_model, render, res, background)
            final_parameter_sequence.append((refined_mid_obb_size_changes, refined_pos_obb_size_changes, rendered_img, refined_sim, refined_mesh))
        
    return final_parameter_sequence