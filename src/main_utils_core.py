from cmath import cos
from turtle import pen
import clip
from pyparsing import col
from sklearn.preprocessing import scale
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

import random
import numpy as np
import pymeshlab 
import time

def update_mesh_by_obb_tree(mesh, old_obbs, new_obbs, root_node, nodes, node_orders, graph_edges, size_changes, pose_changes, loc_changes, uv, vis=False):

    #start = time.time()

    queue = []
    queue.append(root_node)

    deformed_vertices = copy.deepcopy(mesh.vertices)
    deformed_vertex_normals = copy.deepcopy(mesh.vertex_normals)

    deformed_vertices[root_node.pc_indices] = perferm_point_size_change(deformed_vertices[root_node.pc_indices], old_obbs[root_node.index], size_changes[root_node.index])
    deformed_vertex_normals[root_node.pc_indices] = perferm_direction_size_change(deformed_vertex_normals[root_node.pc_indices], old_obbs[root_node.index], size_changes[root_node.index])
    new_obbs[root_node.index].update_size(size_changes[root_node.index])

    #print('pose changes', pose_changes)

    #deformed_vertices[root_node.pc_indices] = perferm_point_pose_change(deformed_vertices[root_node.pc_indices], old_obbs[root_node.index], pose_changes[root_node.index])
    #new_obbs[root_node.index].update_pose(pose_changes[root_node.index])
    
    deformed_vertices[root_node.pc_indices] = perferm_loc_change(deformed_vertices[root_node.pc_indices], old_obbs[root_node.index], loc_changes[root_node.index])
    new_obbs[root_node.index].update_loc(loc_changes[root_node.index])
    
    while len(queue) > 0:
        cur_node = queue.pop(0)
        for child_index in range(len(cur_node.childs)):
            child_node = cur_node.childs[child_index]
            
            #size_follow_vector = get_size_follow_constraint_vector(new_obbs, cur_node.index, cur_node.child_size_follow_constraints[child_index])

            #deformed_vertices[child_node.pc_indices] = perferm_point_size_change(deformed_vertices[child_node.pc_indices], new_obbs[child_node.index], size_changes[cur_node.index])
            #deformed_vertex_normals[child_node.pc_indices] = perferm_direction_size_change(deformed_vertex_normals[child_node.pc_indices], new_obbs[child_node.index], size_changes[cur_node.index])
            #new_obbs[child_node.index].update_size(size_changes[cur_node.index])
            
            # --- child size change
            deformed_vertices[child_node.pc_indices] = perferm_point_size_change(deformed_vertices[child_node.pc_indices], new_obbs[child_node.index], size_changes[child_node.index])
            deformed_vertex_normals[child_node.pc_indices] = perferm_direction_size_change(deformed_vertex_normals[child_node.pc_indices], new_obbs[child_node.index], size_changes[child_node.index])
            new_obbs[child_node.index].update_size(size_changes[child_node.index])

            # --- child pose change
            #deformed_vertices[child_node.pc_indices] = perferm_point_pose_change(deformed_vertices[child_node.pc_indices], new_obbs[child_node.index], pose_changes[child_node.index])
            #new_obbs[child_node.index].update_pose(pose_changes[child_node.index])
            
            # --- child loc change accordint to parent size and loc change

            old_child_loc_follow_constraint_vector = get_loc_follow_constraint_vector(old_obbs, cur_node.index, cur_node.child_loc_follow_constraints[child_index])
            new_child_loc_follow_constraint_vector = get_loc_follow_constraint_vector(new_obbs, cur_node.index, cur_node.child_loc_follow_constraints[child_index])
            loc_follow_constraint_change_vector = new_child_loc_follow_constraint_vector - old_child_loc_follow_constraint_vector
            deformed_vertices[child_node.pc_indices] = perferm_loc_change(deformed_vertices[child_node.pc_indices], new_obbs[child_node.index], loc_follow_constraint_change_vector)
            new_obbs[child_node.index].update_loc(loc_follow_constraint_change_vector)

            # --- child loc change
            #deformed_vertices[child_node.pc_indices] = perferm_loc_change(deformed_vertices[child_node.pc_indices], new_obbs[child_node.index], loc_changes[child_node.index])
            #new_obbs[child_node.index].update_loc(loc_changes[child_node.index])
            
            queue.append(child_node)
    
    #queue = []
    #queue.append(root_node)
    #while len(queue) > 0:
        #cur_node = queue.pop(0)
        #for child_index in range(len(cur_node.childs)):
            
    for edge in graph_edges:
        if node_orders[edge[0]] <= node_orders[edge[1]]:
            cur_node_index = edge[0]
            child_node_index = edge[1]
        else: 
            cur_node_index = edge[1]
            child_node_index = edge[0]

        cur_node = nodes[cur_node_index]
        child_node = nodes[child_node_index]
        deformed_vertices[child_node.pc_indices] += 0.5 * get_boundary_vertex_follow_vectors(mesh.vertices, deformed_vertices, cur_node.pc_indices, child_node.pc_indices, vis)

        temp = cur_node_index
        cur_node_index = child_node_index
        child_node_index = temp
        cur_node = nodes[cur_node_index]
        child_node = nodes[child_node_index]
        deformed_vertices[child_node.pc_indices] += 0.5 * get_boundary_vertex_follow_vectors(mesh.vertices, deformed_vertices, cur_node.pc_indices, child_node.pc_indices, vis)
      
    mesh.vertices = deformed_vertices
    mesh.vertex_normals = deformed_vertex_normals
    MeshNormalizer(mesh)() 

    if uv is not None:
        mesh.face_attributes = uv.unsqueeze(0)
    else:
        prior_color = torch.full(size=(mesh.faces.shape[0], 3, 3), fill_value=object_grey_color, device=device)
        pred_rgb = torch.full(size=(len(mesh.vertices), 3), fill_value=0.0, device=device)
        mesh.face_attributes = prior_color + kaolin.ops.mesh.index_vertices_by_faces(
            pred_rgb.unsqueeze(0),
            mesh.faces)

    return new_obbs


def get_loss(encoded_renders, encoded_text, old_mesh, new_mesh, reverse, use_normal_loss=False, shape_normal_error_weight=0.0):
    
    sim = get_similarity(encoded_renders, encoded_text)
    #print('sim', sim)

    #print('normal penalty', normal_error * shape_normal_error_weight)

    if reverse is True:
        loss = sim 
    else:
        loss = -sim 

    if use_normal_loss:

        temp_mesh = trimesh.Trimesh(to_numpy(new_mesh.vertices), to_numpy(new_mesh.faces), process=False)
        temp_mesh.fix_normals()
        new_vertex_normals = trimesh.geometry.mean_vertex_normals(len(new_mesh.vertices), to_numpy(new_mesh.faces), temp_mesh.face_normals)    
        new_vertex_normals = torch.tensor(new_vertex_normals, device=device, dtype=torch.float)
        new_vertex_normals = new_vertex_normals/torch.norm(new_vertex_normals, dim=1).unsqueeze(dim=1)
        #display_pcs_and_vectors([old_mesh.vertices[0:2000]], [0.1 * old_mesh.vertex_normals[0:2000]])
        #display_pcs_and_vectors([new_mesh.vertices[0:2000]], [0.1 * new_vertex_normals[0:2000]])
        #print('old_vertex_normals shape', old_mesh.vertex_normals.shape)
        #print('new_vertex_normals shape', new_vertex_normals.shape)

        normal_cos_sims = torch.cosine_similarity(old_mesh.vertex_normals, new_vertex_normals)
        filtered_normal_cos_sims = torch.where(torch.isnan(normal_cos_sims), torch.zeros_like(normal_cos_sims), normal_cos_sims)
        normal_error = torch.mean(1 - filtered_normal_cos_sims)
        loss = loss + shape_normal_error_weight * normal_error
    
    #print('loss', loss)

    return loss

def render_and_encode(mesh, tex, render, augment_transform, background, res, clip_model, view_option, use_random=False):

    if view_option == 'random':
        elev_ranges, azim_ranges, dists = get_random_views()
    elif view_option == 'front':
        elev_ranges, azim_ranges, dists = get_front_views()
    elif view_option == 'hemisphere':
        elev_ranges, azim_ranges = get_hemisphere_views()
    elif view_option == 'sphere':
        elev_ranges, azim_ranges, dists = get_sphere_views(False)
    elif view_option == 'sphere_random':
        elev_ranges, azim_ranges, dists = get_sphere_views(True)
    elif view_option == 'ring':
        elev_ranges, azim_ranges, dists = get_ring_views()
    elif view_option == 'demo':
        elev_ranges, azim_ranges, dists = get_demo_views()
    elif view_option == 'spiral':
        elev_ranges, azim_ranges = get_spiral_views()
    else:
        print('wrong view option')
        exit()

    if tex is not None:
        rendered_images, elev, azim = render.render_front_views_with_textures(mesh, tex, num_views=0,
                                                                show=False,
                                                                center_elev=None,
                                                                center_azim=None,
                                                                input_elevs = elev_ranges,
                                                                input_azims = azim_ranges,
                                                                dists = dists,
                                                                std=0,
                                                                return_views=True,
                                                                background=background)
    else:
        rendered_images, elev, azim = render.render_front_views(mesh, num_views=0,
                                                                show=False,
                                                                center_elev=None,
                                                                center_azim=None,
                                                                input_elevs = elev_ranges,
                                                                input_azims = azim_ranges,
                                                                dists = dists,
                                                                std=0,
                                                                return_views=True,
                                                                background=background)

    encoded_renders, _ = encode_images(rendered_images, augment_transform, clip_model, res)
    return encoded_renders, rendered_images


def get_similarity(encoded_renders, encoded_text, k=20, seperate=False):

    encoded_renders = encoded_renders/(torch.norm(encoded_renders, dim=1).unsqueeze(dim=1))
    encoded_text = encoded_text.repeat_interleave(len(encoded_renders), dim=0)
    encoded_text = encoded_text/(torch.norm(encoded_text, dim=1).unsqueeze(dim=1))

    if seperate:
        seperate_sims = []
        for i in range(len(encoded_renders)):
            cos_sim = torch.cosine_similarity(encoded_renders[i].unsqueeze(dim=0), encoded_text[i].unsqueeze(dim=0))[0]
            seperate_sims.append(cos_sim)
        return seperate_sims
    else:
        #print('encoded_renders shape', encoded_renders.shape)
        #print('encoded_text', encoded_text.shape)
        cos_sims = torch.cosine_similarity(encoded_renders, encoded_text)
        #sim = torch.mean(torch.topk(cos_sims, k=min(10, len(cos_sims)), largest=True).values)
        sim = torch.mean(cos_sims)

        #render_mean = torch.mean(encoded_renders, dim=0)
        #render_mean = (render_mean/torch.norm(render_mean)).unsqueeze(dim=0)
        #text_mean = torch.mean(encoded_text, dim=0)
        #text_mean = (text_mean/torch.norm(text_mean)).unsqueeze(dim=0)
        #sim = torch.mean(torch.cosine_similarity(render_mean, text_mean))
    
    return sim

#torch.autograd.set_detect_anomaly(True)

def encode_images(rendered_images, augment_transform, clip_model, res):
    aug_count = 1
    all_augmented_rendered_images = []
    for i in range(aug_count):
        augmented_rendered_images = augment_transform(rendered_images)
        #augmented_rendered_images = rendered_images
        all_augmented_rendered_images.append(augmented_rendered_images)
    all_augmented_rendered_images = torch.stack(all_augmented_rendered_images)
    #print('all_augmented_rendered_images shape', all_augmented_rendered_images.shape)
    all_augmented_rendered_images = all_augmented_rendered_images.permute(1, 0, 2, 3, 4)
    #print('all_augmented_rendered_images shape', all_augmented_rendered_images.shape)
    all_augmented_rendered_images = all_augmented_rendered_images.reshape(-1, 3, res, res)
    #print('all_augmented_rendered_images shape', all_augmented_rendered_images.shape)

    encoded_renders = clip_model.encode_image(all_augmented_rendered_images)
    #print('encoded_renders shape', encoded_renders.shape)
    encoded_renders = encoded_renders/torch.norm(encoded_renders, dim=1).unsqueeze(dim=1)
    #print('encoded_renders shape', encoded_renders.shape)
    #encoded_renders = encoded_renders.reshape(-1, aug_count, 512)
    #encoded_renders = torch.mean(encoded_renders, dim=1)
    return encoded_renders, all_augmented_rendered_images

def get_demo_views():
    elev_ranges = []
    azim_ranges = []

    num_views = 8
    azim_delta = -2*np.pi/num_views

    ref_elev = 0.2 * np.pi
    elev_ranges.append(ref_elev)
    ref_azim = 3*azim_delta
    azim_ranges.append(ref_azim)
    dists = [2.5]

    return elev_ranges, azim_ranges, dists

def get_random_views():
    num_views = 24
    elev = (torch.rand(num_views)-0.5) * 2 * np.pi
    azim = (torch.rand(num_views)-0.5) * 2 * np.pi
    dists = [2.5] * num_views
    return elev, azim, dists

def get_spiral_views():
    center = torch.tensor([0,0])
    num_views = 16
    azim = torch.linspace(center[0], 2 * np.pi + center[0], num_views + 1)[:-1]  # since 0 =360 dont include last element
    elev = torch.cat((torch.linspace(center[1], np.pi / 2 + center[1], int((num_views + 1) / 2)),
                    torch.linspace(center[1], -np.pi / 2 + center[1], int((num_views) / 2))))

    return elev, azim

def get_ring_views():
    elev_ranges = []
    azim_ranges = []
    dist_ranges = []

    num_views = 8
    azim_delta = -2*np.pi/num_views
    dist = torch.tensor(2.5, device=device, dtype=torch.float)

    ref_elev = 0.4 * np.pi
    ref_azim = 0
    for i in range(0, num_views):
        elev_range = torch.tensor(ref_elev, device=device, dtype=torch.float)
        elev_ranges.append(elev_range)
        azim_range = torch.tensor(ref_azim + i*azim_delta, device=device, dtype=torch.float)                    
        azim_ranges.append(azim_range)
        dist_ranges.append(dist)
    
    return elev_ranges, azim_ranges, dist_ranges

def compute_pivot_center(mesh, render, prior_color, encoded_text, clip_model, clip_transform, background):

    pivot_azim = None
    pivot_elev = None
    max_sim = 0

    new_mesh = copy.deepcopy(mesh)
    pred_rgb = torch.full(size=(len(new_mesh.vertices), 3), fill_value=0.0, device=device)
    new_mesh.face_attributes = prior_color + kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0),
        new_mesh.faces)
    MeshNormalizer(new_mesh)()

    for azim in np.arange(-np.pi, np.pi, 0.2):
        for elev in np.arange(-np.pi, np.pi, 0.2):

            rendered_images, elev, azim = render.render_front_views(new_mesh, num_views=1,
                                                            show=False,
                                                            center_azim=torch.tensor(azim, dtype=torch.float),
                                                            center_elev=torch.tensor(elev, dtype=torch.float),
                                                            std=0,
                                                            return_views=True,
                                                            background=background)
            clip_image = clip_transform(rendered_images)
            encoded_renders = clip_model.encode_image(clip_image)
            sim = torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
            #print('azim', azim)
            #print('elev', elev)
            #print('sim', sim)
            if sim > max_sim:
                max_sim = sim
                pivot_azim = azim
                pivot_elev = elev


    #rendered_images, elev, azim = render.render_front_views(new_mesh, num_views=1,
                                                            #show=args.show,
                                                            #center_azim=pivot_azim,
                                                            #center_elev=pivot_elev,
                                                            #std=args.frontview_std,
                                                            #return_views=True,
                                                            #background=background)

    #torchvision.utils.save_image(rendered_images, os.path.join(folder, 'pivot_view.jpg'))

    return pivot_azim, pivot_elev

pivot_azim = None
pivot_elev = None


def get_pivot_views(mesh, render, prior_color, encoded_text, clip_model, clip_transform, background):

    global pivot_elev
    global pivot_azim
    if pivot_azim is None and pivot_elev is None:
        pivot_azim, pivot_elev = compute_pivot_center(mesh, render, prior_color, encoded_text, clip_model, clip_transform, background)
    
    elev_ranges = []
    azim_ranges = []

    num_views = 2
    
    azim_delta = 0.5*np.pi/(2*num_views+1)
    elev_delta = 0.5*np.pi/(2*num_views+1)

    for i in range(-num_views, num_views+1):
        for j in range(-num_views, num_views+1):
            elev_range = torch.tensor(pivot_elev + elev_delta * j, device=device, dtype=torch.float)
            elev_ranges.append(elev_range)
            azim_range = torch.tensor(pivot_azim + azim_delta * i, device=device, dtype=torch.float)                   
            azim_ranges.append(azim_range)

    return elev_ranges, azim_ranges

def get_hemisphere_views():
    elev_ranges = []
    azim_ranges = []

    num_views = 8
    azim_delta = -2*np.pi/num_views
    
    ref_elev = 0.4 * np.pi
    ref_azim = 0
    for i in range(0, num_views):
        elev_range = torch.tensor(ref_elev, device=device, dtype=torch.float)
        elev_ranges.append(elev_range)
        azim_range = torch.tensor(ref_azim + i*azim_delta, device=device, dtype=torch.float)                    
        azim_ranges.append(azim_range)

    ref_elev = 0.2 * np.pi
    ref_azim = 0
    for i in range(0, num_views):
        elev_range = torch.tensor(ref_elev, device=device, dtype=torch.float)
        elev_ranges.append(elev_range)
        azim_range = torch.tensor(ref_azim + i*azim_delta, device=device, dtype=torch.float)                    
        azim_ranges.append(azim_range)

    ref_elev = 0.0 * np.pi
    ref_azim = 0
    for i in range(0, num_views):
        elev_range = torch.tensor(ref_elev, device=device, dtype=torch.float)
        elev_ranges.append(elev_range)
        azim_range = torch.tensor(ref_azim + i*azim_delta, device=device, dtype=torch.float)                    
        azim_ranges.append(azim_range)

    return elev_ranges, azim_ranges

def get_front_views(is_random=False):
    elev_ranges = []
    azim_ranges = []
    dist_ranges = []
        
    elev = torch.tensor(0.25 * np.pi, device=device, dtype=torch.float)
    azim = torch.tensor(-0.25 * np.pi, device=device, dtype=torch.float)
    elev_ranges.append(elev)
    azim_ranges.append(azim)
    dist_ranges.append(torch.tensor(2.5, device=device, dtype=torch.float))

    elev = torch.tensor(0.25 * np.pi, device=device, dtype=torch.float)
    azim = torch.tensor(0.25 * np.pi, device=device, dtype=torch.float)
    elev_ranges.append(elev)
    azim_ranges.append(azim)
    dist_ranges.append(torch.tensor(2.5, device=device, dtype=torch.float))

    elev = torch.tensor(0.25 * np.pi, device=device, dtype=torch.float)
    azim = torch.tensor(-0.75 * np.pi, device=device, dtype=torch.float)
    elev_ranges.append(elev)
    azim_ranges.append(azim)
    dist_ranges.append(torch.tensor(2.5, device=device, dtype=torch.float))

    elev = torch.tensor(0.25 * np.pi, device=device, dtype=torch.float)
    azim = torch.tensor(0.75 * np.pi, device=device, dtype=torch.float)
    elev_ranges.append(elev)
    azim_ranges.append(azim)
    dist_ranges.append(torch.tensor(2.5, device=device, dtype=torch.float))

    #print('elev_ranges', elev_ranges)
    #print('azim_ranges', azim_ranges)
    return elev_ranges, azim_ranges, dist_ranges


def get_sphere_views(is_random=False):
    elev_ranges = []
    azim_ranges = []
    dist_ranges = []

    if is_random:
        start_elev = torch.tensor(np.random.uniform(-0.5 * np.pi, 0.5 * np.pi), device=device, dtype=torch.float)
        start_azim = torch.tensor(np.random.uniform(-np.pi, np.pi), device=device, dtype=torch.float)
        #start_elev = torch.tensor(0.0, device=device, dtype=torch.float)
        #start_azim = torch.tensor(0.0, device=device, dtype=torch.float)
        dist = torch.tensor(2.5, device=device, dtype=torch.float)
        #dist = torch.tensor(np.random.uniform(2.0, 3.0), device=device, dtype=torch.float)
    else:
        start_elev = torch.tensor(0.0, device=device, dtype=torch.float)
        start_azim = torch.tensor(0.0, device=device, dtype=torch.float)
        dist = torch.tensor(2.5, device=device, dtype=torch.float)

    num_views = 6
    azim_delta = -2 * np.pi/(num_views)
    elev_delta = -2 * np.pi/(num_views)

    ref_elev = 0.0
    ref_azim = 0.0
    for i in range(0, num_views):
        elev_range = torch.tensor(ref_elev, device=device, dtype=torch.float)
        elev_ranges.append(elev_range)
        azim_range = torch.tensor(ref_azim + i*azim_delta, device=device, dtype=torch.float)                    
        azim_ranges.append(start_azim + azim_range)
        dist_ranges.append(dist)

    ref_elev = 0.0
    ref_azim = 0.5 * np.pi
    for i in range(0, num_views):
        elev_range = torch.tensor(ref_elev + i*elev_delta, device=device, dtype=torch.float)
        elev_ranges.append(start_elev + elev_range)
        azim_range = torch.tensor(ref_azim, device=device, dtype=torch.float)                    
        azim_ranges.append(azim_range)
        dist_ranges.append(dist)

    ref_elev = 0.0
    ref_azim = 1.0 * np.pi
    for i in range(0, num_views):
        elev_range = torch.tensor(ref_elev + i*elev_delta, device=device, dtype=torch.float)
        elev_ranges.append(start_elev + elev_range)
        azim_range = torch.tensor(ref_azim, device=device, dtype=torch.float)                    
        azim_ranges.append(azim_range)
        dist_ranges.append(dist)

    #print('elev_ranges', elev_ranges)
    #print('azim_ranges', azim_ranges)
    return elev_ranges, azim_ranges, dist_ranges

'''
def get_sphere_views(is_random=False):
    elev_ranges = []
    azim_ranges = []
    dist_ranges = []

    if is_random:
        start_elev = torch.tensor(np.random.uniform(-0.5 * np.pi, 0.5 * np.pi), device=device, dtype=torch.float)
        start_azim = torch.tensor(np.random.uniform(-np.pi, np.pi), device=device, dtype=torch.float)
        #start_elev = torch.tensor(0.0, device=device, dtype=torch.float)
        #start_azim = torch.tensor(0.0, device=device, dtype=torch.float)
        dist = torch.tensor(np.random.uniform(2.0, 3.0), device=device, dtype=torch.float)
    else:
        start_elev = torch.tensor(0.0, device=device, dtype=torch.float)
        start_azim = torch.tensor(0.0, device=device, dtype=torch.float)
        dist = torch.tensor(2.5, device=device, dtype=torch.float)

    num_views = 6
    azim_delta = -2 * np.pi/(num_views-1)
    
    ref_elev = 0.5 * np.pi
    ref_azim = 0
    for i in range(0, num_views):
        elev_range = torch.tensor(ref_elev, device=device, dtype=torch.float)
        elev_ranges.append(start_elev + elev_range)
        azim_range = torch.tensor(ref_azim + i*azim_delta, device=device, dtype=torch.float)                    
        azim_ranges.append(start_azim + azim_range)
        dist_ranges.append(dist)

    ref_elev = 0.25 * np.pi
    ref_azim = 0
    for i in range(0, num_views):
        elev_range = torch.tensor(ref_elev, device=device, dtype=torch.float)
        elev_ranges.append(start_elev + elev_range)
        azim_range = torch.tensor(ref_azim + i*azim_delta, device=device, dtype=torch.float)                    
        azim_ranges.append(start_azim + azim_range)
        dist_ranges.append(dist)

    ref_elev = 0.0 * np.pi
    ref_azim = 0
    for i in range(0, num_views):
        elev_range = torch.tensor(ref_elev, device=device, dtype=torch.float)
        elev_ranges.append(start_elev + elev_range)
        azim_range = torch.tensor(ref_azim + i*azim_delta, device=device, dtype=torch.float)                    
        azim_ranges.append(start_azim + azim_range)
        dist_ranges.append(dist)

    ref_elev = -0.25 * np.pi
    ref_azim = 0
    for i in range(0, num_views):
        elev_range = torch.tensor(ref_elev, device=device, dtype=torch.float)
        elev_ranges.append(start_elev + elev_range)
        azim_range = torch.tensor(ref_azim + i*azim_delta, device=device, dtype=torch.float)                    
        azim_ranges.append(start_azim + azim_range)
        dist_ranges.append(dist)

    ref_elev = -0.5 * np.pi
    ref_azim = 0
    for i in range(0, num_views):
        elev_range = torch.tensor(ref_elev, device=device, dtype=torch.float)
        elev_ranges.append(start_elev + elev_range)
        azim_range = torch.tensor(ref_azim + i*azim_delta, device=device, dtype=torch.float)                    
        azim_ranges.append(start_azim + azim_range)
        dist_ranges.append(dist)

    return elev_ranges, azim_ranges, dist_ranges
'''

def get_vertex_follow_weights(dists):
    eps = 0.001
    temperature = 50.0
    weights = 1.0/(dists + eps)
    normalized_weights = torch.clamp(weights/temperature, 0.0, 1.0)
    #normalized_weights[normalized_weights<0.2] = 0.0
    return normalized_weights
    
def get_boundary_vertex_follow_vectors(old_vertices, new_vertices, parent_vertex_indices, child_vertex_indices, vis):

    old_parent_vertices = old_vertices[parent_vertex_indices]
    new_parent_vertices = new_vertices[parent_vertex_indices]
    old_child_vertices = old_vertices[child_vertex_indices]
    new_child_vertices = new_vertices[child_vertex_indices]
    
    
    child_to_parent_dists = torch.cdist(old_child_vertices, old_parent_vertices)
    parent_vertex_movements = new_parent_vertices - old_parent_vertices
    pivot_parent_vertex_dists = torch.min(child_to_parent_dists, dim=1).values
    pivot_parent_vertex_indices = torch.min(child_to_parent_dists, dim=1).indices
    follow_weights = get_vertex_follow_weights(pivot_parent_vertex_dists)
    child_vertex_follow_movements = parent_vertex_movements[pivot_parent_vertex_indices]
    child_vertex_movements = new_child_vertices - old_child_vertices
    complement_child_vertex_vectors = child_vertex_follow_movements - child_vertex_movements
    
    if vis:
        display_pcs_and_vectors([new_child_vertices[0:50], old_parent_vertices[pivot_parent_vertex_indices]][0:50], [complement_child_vertex_vectors[0:50], parent_vertex_movements[pivot_parent_vertex_indices][0:50]])

    complement_child_vertex_vectors = complement_child_vertex_vectors * follow_weights.unsqueeze(dim=1).repeat_interleave(3, dim=1)
    
    if vis:
        display_pcs_and_vectors([new_child_vertices[0:50], old_parent_vertices[pivot_parent_vertex_indices]][0:50], [complement_child_vertex_vectors[0:50], parent_vertex_movements[pivot_parent_vertex_indices][0:50]])

    return complement_child_vertex_vectors


    