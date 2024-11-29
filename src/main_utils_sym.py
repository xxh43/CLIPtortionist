from cmath import cos
from email.policy import default
from tabnanny import verbose
from turtle import pen
import clip
from pyparsing import col
from tqdm import tqdm
import kaolin.ops.mesh
import kaolin as kal
import torch
from obb_graph import get_constraint_graph
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
from obb_graph import *

def eval_result_sym(mesh, tex, uv, init_mid_obb_size_changes, init_pos_obb_size_changes, init_mid_obb_pose_changes, init_pos_obb_pose_changes, init_mid_obb_loc_changes, init_pos_obb_loc_changes, size_range, angle_range, loc_range, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, augment_transform, clip_model, render, res, background, view='front'):
    with torch.no_grad():
        mapped_mid_obb_size_changes, mapped_pos_obb_size_changes, mapped_mid_obb_pose_changes, mapped_pos_obb_pose_changes, mapped_mid_obb_loc_changes, mapped_pos_obb_loc_changes = \
            map_bbox_parameters_sym(init_mid_obb_size_changes, init_pos_obb_size_changes, init_mid_obb_pose_changes, init_pos_obb_pose_changes, init_mid_obb_loc_changes, init_pos_obb_loc_changes, size_range, angle_range, loc_range)
        obbs, obb_size_changes, obb_pose_changes, obb_loc_changes, new_obbs = get_bbox_parameters_sym(mid_obbs, pos_obbs, neg_obbs, mapped_mid_obb_size_changes, mapped_pos_obb_size_changes, mapped_mid_obb_pose_changes, mapped_pos_obb_pose_changes, mapped_mid_obb_loc_changes, mapped_pos_obb_loc_changes)

        new_mesh = copy.deepcopy(mesh)
        #new_obbs = update_mesh_by_obb_graph(new_mesh, obbs, new_obbs, obb_size_changes, obb_pose_changes, obb_loc_changes, prior_color)
        new_obbs = update_mesh_by_obb_tree(new_mesh, obbs, new_obbs, obb_tree_root, nodes, node_orders, graph_edges, obb_size_changes, obb_pose_changes, obb_loc_changes, uv)
        encoded_renders, rendered_images = render_and_encode(new_mesh, tex, render, augment_transform, background, res, clip_model, 'front')
        sim = get_similarity(encoded_renders, encoded_text, k=1, seperate=False)
        return sim.item(), new_mesh, new_obbs, rendered_images

def get_bbox_parameters_sym(mid_obbs, pos_obbs, neg_obbs, mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes):
    
    neg_obb_size_changes = pos_obb_size_changes
    neg_obb_pose_changes = pos_obb_pose_changes
    neg_obb_loc_changes = pos_obb_loc_changes
    
    if len(pos_obb_pose_changes) > 0:    
        neg_obb_pose_changes = []
        for v in pos_obb_pose_changes:
            neg_obb_pose_changes.append(-v)
        neg_obb_pose_changes = torch.stack(neg_obb_pose_changes)

    if len(pos_obb_loc_changes) > 0:
        neg_obb_loc_changes = []
        for v in pos_obb_loc_changes:
            neg_obb_loc_changes.append(torch.stack([-v[0], v[1], v[2]]))
        neg_obb_loc_changes = torch.stack(neg_obb_loc_changes)

    new_mid_obb_pose_changes = []
    for v in mid_obb_pose_changes:
        new_mid_obb_pose_changes.append(v)
    new_mid_obb_pose_changes = torch.stack(new_mid_obb_pose_changes)

    new_mid_obb_loc_changes = []
    for v in mid_obb_loc_changes:
        new_mid_obb_loc_changes.append(torch.stack([torch.tensor(0.0, device=device), v[1], v[2]]))
    new_mid_obb_loc_changes = torch.stack(new_mid_obb_loc_changes)

    obbs = mid_obbs + pos_obbs + neg_obbs
    obb_size_changes = torch.cat((mid_obb_size_changes, pos_obb_size_changes, neg_obb_size_changes), dim=0)
    obb_pose_changes = torch.cat((new_mid_obb_pose_changes, pos_obb_pose_changes, neg_obb_pose_changes), dim=0)
    obb_loc_changes = torch.cat((new_mid_obb_loc_changes, pos_obb_loc_changes, neg_obb_loc_changes), dim=0)
    new_obbs = copy.deepcopy(obbs)

    return obbs, obb_size_changes, obb_pose_changes, obb_loc_changes, new_obbs

def map_bbox_parameters_sym(mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, angle_range, loc_range):

    mapped_mid_obb_size_changes = mid_obb_size_changes * size_range
    mapped_pos_obb_size_changes = pos_obb_size_changes * size_range

    mapped_mid_obb_pose_changes = mid_obb_pose_changes * angle_range
    mapped_pos_obb_pose_changes = pos_obb_pose_changes * angle_range

    mapped_mid_obb_loc_changes = mid_obb_loc_changes * loc_range
    mapped_pos_obb_loc_changes = pos_obb_loc_changes * loc_range

    #mapped_mid_obb_size_changes = torch.sigmoid(mid_obb_size_changes) * size_range
    #mapped_pos_obb_size_changes = torch.sigmoid(pos_obb_size_changes) * size_range

    #mapped_mid_obb_pose_changes = torch.sigmoid(mid_obb_pose_changes) * angle_range
    #mapped_pos_obb_pose_changes = torch.sigmoid(pos_obb_pose_changes) * angle_range

    #mapped_mid_obb_loc_changes = torch.sigmoid(mid_obb_loc_changes) * loc_range
    #mapped_pos_obb_loc_changes = torch.sigmoid(pos_obb_loc_changes) * loc_range

    return mapped_mid_obb_size_changes, mapped_pos_obb_size_changes, mapped_mid_obb_pose_changes, mapped_pos_obb_pose_changes, mapped_mid_obb_loc_changes, mapped_pos_obb_loc_changes

def obb_overlap_graph(obbs):
    overlap_graph = torch.zeros((len(obbs), len(obbs)))
    for i in range(len(obbs)):
        for j in range(i + 1, len(obbs)):
            dist = is_intersected(obbs[i], obbs[j])
            overlap_graph[i][j] = dist

    return overlap_graph

def obb_overlap_degree(obbs, penalty_factor, overlap_graph):    
    dist = 0.0
    for i in range(len(obbs)):
        for j in range(i + 1, len(obbs)):
            dist += torch.relu(is_intersected(obbs[i], obbs[j]) - overlap_graph[i][j]-0.1)
    
    if dist > 0.000001:
        return True, (dist/len(obbs))*penalty_factor
    else:
        return False, torch.tensor(0.0, device=device, dtype=torch.float)

def cma_eval_func_sym(params, size_range, angle_range, loc_range, mesh, mid_obbs, pos_obbs, neg_obbs, overlap_graph, obb_tree_root, nodes, node_orders, graph_edges, tex, uv, encoded_text, augment_transform, clip_model, render, res, background, shape_normal_error_weight, view_option, reverse_optim):

    #print('free cuda mem', get_free_cuda_mem())

    new_mesh = copy.deepcopy(mesh)

    mid_obb_size_changes = torch.tensor(params[0 : len(mid_obbs)*3], device=device, dtype=torch.float).reshape(-1, 3)
    pos_obb_size_changes = torch.tensor(params[len(mid_obbs)*3 : (len(mid_obbs)+len(pos_obbs))*3], device=device, dtype=torch.float).reshape(-1, 3)
    
    if len(params) > len(mid_obbs)*3+len(pos_obbs)*3:
        mid_obb_loc_changes = torch.tensor(params[(len(mid_obbs)+len(pos_obbs))*3 : (len(mid_obbs)+len(pos_obbs))*3+len(mid_obbs)*3], device=device, dtype=torch.float).reshape(-1, 3)
        pos_obb_loc_changes = torch.tensor(params[(len(mid_obbs)+len(pos_obbs))*3+len(mid_obbs)*3 : len(mid_obbs)*3+len(pos_obbs)*3+len(mid_obbs)*3+len(pos_obbs)*3], device=device, dtype=torch.float).reshape(-1, 3)
    else:
        mid_obb_loc_changes = torch.zeros((len(mid_obbs), 3), device=device)
        pos_obb_loc_changes = torch.zeros((len(pos_obbs), 3), device=device)

    if len(params) > len(mid_obbs)*3+len(pos_obbs)*3+len(mid_obbs)*3+len(pos_obbs)*3:
        mid_obb_pose_changes = torch.tensor(params[len(mid_obbs)*3+len(pos_obbs)*3+len(mid_obbs)*3+len(pos_obbs)*3 : len(mid_obbs)*3+len(pos_obbs)*3+len(mid_obbs)*3+len(pos_obbs)*3+len(mid_obbs)*1], device=device, dtype=torch.float).reshape(-1, 1)
        pos_obb_pose_changes = torch.tensor(params[len(mid_obbs)*3+len(pos_obbs)*3+len(mid_obbs)*3+len(pos_obbs)*3+len(mid_obbs)*1 : ], device=device, dtype=torch.float).reshape(-1, 1)
    else:
        mid_obb_pose_changes = torch.zeros((len(mid_obbs), 1), device=device)
        pos_obb_pose_changes = torch.zeros((len(pos_obbs), 1), device=device)
    
    mapped_mid_obb_size_changes, mapped_pos_obb_size_changes, mapped_mid_obb_pose_changes, mapped_pos_obb_pose_changes, mapped_mid_obb_loc_changes, mapped_pos_obb_loc_changes = \
        map_bbox_parameters_sym(mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, angle_range, loc_range)

    obbs, obb_size_changes, obb_pose_changes, obb_loc_changes, new_obbs = get_bbox_parameters_sym(mid_obbs, pos_obbs, neg_obbs, mapped_mid_obb_size_changes, mapped_pos_obb_size_changes, mapped_mid_obb_pose_changes, mapped_pos_obb_pose_changes, mapped_mid_obb_loc_changes, mapped_pos_obb_loc_changes)
    new_obbs = update_mesh_by_obb_tree(new_mesh, obbs, new_obbs, obb_tree_root, nodes, node_orders, graph_edges, obb_size_changes, obb_pose_changes, obb_loc_changes, uv)
    encoded_renders, _ = render_and_encode(new_mesh, tex, render, augment_transform, background, res, clip_model, view_option, use_random=True)
    loss = get_loss(encoded_renders, encoded_text, mesh, new_mesh, reverse_optim, True, shape_normal_error_weight)
    
    #is_overlap, overlap_penalty = obb_overlap_degree(new_obbs, 1.0, overlap_graph)
    #penalty = compute_constraint_graph_penalty(new_obbs, connect_graph, overlap_graph, 0.1)
    #loss += penalty.item()
    #return loss.item() + overlap_penalty.item()
    return loss.item()

def get_best_config_cma_sym(iter, mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, angle_range, loc_range, mesh, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, tex, uv, encoded_text, augment_transform, clip_model, render, res, background, shape_normal_error_weight, view_option, reverse_optim):

    mid_obb_size_changes = to_numpy(mid_obb_size_changes)
    pos_obb_size_changes = to_numpy(pos_obb_size_changes)
    mid_obb_pose_changes = to_numpy(mid_obb_pose_changes)
    pos_obb_pose_changes = to_numpy(pos_obb_pose_changes)
    mid_obb_loc_changes = to_numpy(mid_obb_loc_changes)
    pos_obb_loc_changes = to_numpy(pos_obb_loc_changes)

    eps = 0.000001

    if size_range > eps and loc_range < eps and angle_range < eps :
        parameters = np.concatenate((mid_obb_size_changes.flatten(), pos_obb_size_changes.flatten()), axis=0)
    elif size_range > eps and loc_range > eps and angle_range < eps:
        parameters = np.concatenate((mid_obb_size_changes.flatten(), pos_obb_size_changes.flatten(), mid_obb_loc_changes.flatten(), pos_obb_loc_changes.flatten()), axis=0)
    elif size_range > eps and angle_range > eps and loc_range > eps:    
        parameters = np.concatenate((mid_obb_size_changes.flatten(), pos_obb_size_changes.flatten(), mid_obb_loc_changes.flatten(), pos_obb_loc_changes.flatten(), mid_obb_pose_changes.flatten(), pos_obb_pose_changes.flatten()), axis=0)
    else:
        print('wrong ranges')
        exit()

    overlap_graph = obb_overlap_graph(mid_obbs+pos_obbs+neg_obbs)

    '''
    args = [size_range, angle_range, loc_range, mesh, mid_obbs, pos_obbs, neg_obbs, overlap_graph, obb_tree_root, tex, uv, encoded_text, augment_transform, clip_model, render, res, background, view_option, reverse_optim]
    opts = cma.CMAOptions()
    opts.set('tolfun', 1e-5)
    opts.set('bounds', [-1.0, 1.0])
    opts.set('maxiter', 100)
    opts.set('popsize_factor', 5)

    while True:
        xopt, es = cma.fmin2(cma_eval_func_sym, x0=parameters, sigma0=1.0, options=opts, args=args)
        
        print('xopt', xopt)
        if xopt is not None:
            break
    
    '''
    verbose = True
    x0 = parameters
    
    sigma0 = 1.0/(iter + 1)
    es = cma.CMAEvolutionStrategy(x0, sigma0, {'bounds': [-1.0, 3.0], 'popsize_factor': 3})
    es.opts.set('tolfun', 1e-4)
    es.opts.set('maxiter', 80)
    
    counter = 0
    while not es.stop():
        X = es.ask()  # sample len(X) candidate solutions
        es.tell(X, [cma_eval_func_sym(x, size_range, angle_range, loc_range, mesh, mid_obbs, pos_obbs, neg_obbs, overlap_graph, obb_tree_root, nodes, node_orders, graph_edges, tex, uv, encoded_text, augment_transform, clip_model, render, res, background, shape_normal_error_weight, view_option, reverse_optim) for x in X])
        es.disp()
        xopt = es.result[0]
        fopt = es.result[1]
        
        '''
        if verbose and counter % 2 == 0:
            best_mid_obb_size_changes = torch.tensor(xopt[0:len(mid_obbs)*3], device=device, dtype=torch.float).reshape(-1, 3)
            best_pos_obb_size_changes = torch.tensor(xopt[len(mid_obbs)*3 : (len(mid_obbs)+len(pos_obbs))*3], device=device, dtype=torch.float).reshape(-1, 3)

            #best_mid_obb_size_changes = torch.stack([torch.tensor([2.0, -0.6, 0.0], device=device, dtype=torch.float)]*len(mid_obbs))


            best_mid_obb_pose_changes = torch.tensor(mid_obb_pose_changes, device=device, dtype=torch.float)
            best_pos_obb_pose_changes = torch.tensor(pos_obb_pose_changes, device=device, dtype=torch.float)
            
            best_mid_obb_loc_changes = torch.tensor(mid_obb_loc_changes, device=device, dtype=torch.float)
            best_pos_obb_loc_changes = torch.tensor(pos_obb_loc_changes, device=device, dtype=torch.float)

            #print('best_mid_obb_size_changes', best_mid_obb_size_changes)
            #similarity, new_mesh, new_obbs, rendered_images = eval_result_sym(mesh, tex, uv, best_mid_obb_size_changes, best_pos_obb_size_changes, best_mid_obb_pose_changes, best_pos_obb_pose_changes, best_mid_obb_loc_changes, best_pos_obb_loc_changes, size_range, angle_range, loc_range, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, augment_transform, clip_model, render, res, background)
            #torchvision.utils.save_image(rendered_images, os.path.join('temp', str(counter) + 'eval_sim_' + str(round(similarity, 4)) + 'fopt_' + str(fopt) + '.jpg'))
            
            mapped_mid_obb_size_changes, mapped_pos_obb_size_changes, mapped_mid_obb_pose_changes, mapped_pos_obb_pose_changes, mapped_mid_obb_loc_changes, mapped_pos_obb_loc_changes = \
            map_bbox_parameters_sym(best_mid_obb_size_changes, best_pos_obb_size_changes, best_mid_obb_pose_changes, best_pos_obb_pose_changes, best_mid_obb_loc_changes, best_pos_obb_loc_changes, size_range, angle_range, loc_range)
            obbs, obb_size_changes, obb_pose_changes, obb_loc_changes, new_obbs = get_bbox_parameters_sym(mid_obbs, pos_obbs, neg_obbs, mapped_mid_obb_size_changes, mapped_pos_obb_size_changes, mapped_mid_obb_pose_changes, mapped_pos_obb_pose_changes, mapped_mid_obb_loc_changes, mapped_pos_obb_loc_changes)
            new_mesh = copy.deepcopy(mesh)
            new_obbs = update_mesh_by_obb_tree(new_mesh, obbs, new_obbs, obb_tree_root, nodes, node_orders, graph_edges, obb_size_changes, obb_pose_changes, obb_loc_changes, uv)
            encoded_renders, rendered_images = render_and_encode(new_mesh, tex, render, augment_transform, background, res, clip_model, 'front')
            sims = get_similarity(encoded_renders, encoded_text, k=1, seperate=True)
            for i in range(len(sims)):
                torchvision.utils.save_image(rendered_images[i], os.path.join('temp', str(counter) + 'eval_sim_' + str(round(sims[i].item(), 4)) + '.jpg'))
        ''' 

        counter += 1  
        

    best_mid_obb_size_changes = torch.tensor(xopt[0 : len(mid_obbs)*3], device=device, dtype=torch.float).reshape(-1, 3)
    best_pos_obb_size_changes = torch.tensor(xopt[len(mid_obbs)*3 : (len(mid_obbs)+len(pos_obbs))*3], device=device, dtype=torch.float).reshape(-1, 3)

    if len(xopt) > len(mid_obbs)*3+len(pos_obbs)*3:
        best_mid_obb_loc_changes = torch.tensor(xopt[(len(mid_obbs)+len(pos_obbs))*3 : (len(mid_obbs)+len(pos_obbs))*3+len(mid_obbs)*3], device=device, dtype=torch.float).reshape(-1, 3)
        best_pos_obb_loc_changes = torch.tensor(xopt[(len(mid_obbs)+len(pos_obbs))*3+len(mid_obbs)*3 : (len(mid_obbs)+len(pos_obbs))*3+len(mid_obbs)*3+len(pos_obbs)*3], device=device, dtype=torch.float).reshape(-1, 3)
    else:
        best_mid_obb_loc_changes = torch.tensor(mid_obb_loc_changes, device=device, dtype=torch.float)
        best_pos_obb_loc_changes = torch.tensor(pos_obb_loc_changes, device=device, dtype=torch.float)
        
    if len(xopt) > len(mid_obbs)*3+len(pos_obbs)*3+len(mid_obbs)*3+len(pos_obbs)*3:
        best_mid_obb_pose_changes = torch.tensor(xopt[(len(mid_obbs)+len(pos_obbs))*3+(len(mid_obbs)+len(pos_obbs))*3: (len(mid_obbs)+len(pos_obbs))*3+(len(mid_obbs)+len(pos_obbs))*3+len(mid_obbs)*1], device=device, dtype=torch.float).reshape(-1, 1)
        best_pos_obb_pose_changes = torch.tensor(xopt[(len(mid_obbs)+len(pos_obbs))*3+(len(mid_obbs)+len(pos_obbs))*3+len(mid_obbs)*1 : ], device=device, dtype=torch.float).reshape(-1, 1)
    else:
        best_mid_obb_pose_changes = torch.tensor(mid_obb_pose_changes, device=device, dtype=torch.float)
        best_pos_obb_pose_changes = torch.tensor(pos_obb_pose_changes, device=device, dtype=torch.float)

    return best_mid_obb_size_changes, best_pos_obb_size_changes, best_mid_obb_pose_changes, best_pos_obb_pose_changes, best_mid_obb_loc_changes, best_pos_obb_loc_changes

def gradient_optimize_sym_with_color(iter, mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, angle_range, loc_range, mesh, tex, tex_bin, uv, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, augment_transform, clip_model, render, res, background, shape_normal_error_weight, view_option,  reverse_optim, learning_rate, optimize_shape):

    print('run_ours_gradient')

    #if not os.path.exists('temp'):
        #os.makedirs('temp')

    mid_obb_size_changes_delta = torch.tensor(mid_obb_size_changes, device=device, requires_grad=True)
    pos_obb_size_changes_delta = torch.tensor(pos_obb_size_changes, device=device, requires_grad=True)
    #mid_obb_pose_changes = torch.zeros((len(mid_obbs), 1), device=device, requires_grad=True)
    #pos_obb_pose_changes = torch.zeros((len(pos_obbs), 1), device=device, requires_grad=True)
    #mid_obb_loc_changes = torch.zeros((len(mid_obbs), 3), device=device, requires_grad=True)
    #pos_obb_loc_changes = torch.zeros((len(pos_obbs), 3), device=device, requires_grad=True)
    
    optimizable_parameters = []
    if tex is not None and uv is not None and tex_bin is not None:
        tex_bin_change = torch.zeros((len(tex_bin.keys()), 3), device=device, requires_grad=True)
        optimizable_parameters.append(tex_bin_change)

    if optimize_shape:
        optimizable_parameters.append(mid_obb_size_changes_delta)
        optimizable_parameters.append(pos_obb_size_changes_delta)
    
    #optimizable_parameters.append(mid_obb_pose_changes)
    #optimizable_parameters.append(pos_obb_pose_changes)
    #optimizable_parameters.append(mid_obb_loc_changes)
    #optimizable_parameters.append(pos_obb_loc_changes)
    
    optim = torch.optim.Adam(optimizable_parameters, learning_rate/(iter+1))

    start_time = time.time()

    overlap_graph = obb_overlap_graph(mid_obbs+pos_obbs+neg_obbs)

    max_epoch = 500
    for epoch in range(max_epoch):
        
        new_mesh = copy.deepcopy(mesh)
        
        new_mid_obb_size_changes = torch.clamp(mid_obb_size_changes + mid_obb_size_changes_delta, -1.0, 3.0)
        new_pos_obb_size_changes = torch.clamp(pos_obb_size_changes + pos_obb_size_changes_delta, -1.0, 3.0)

        if tex is not None and uv is not None and tex_bin is not None:
            new_tex = copy.deepcopy(tex).reshape(-1, 3)
            for i in range(0, len(tex_bin.keys())):
                new_tex[tex_bin[i]] = new_tex[tex_bin[i]] + 2 * (torch.sigmoid(tex_bin_change[i]) - 0.5)
            new_tex = new_tex.reshape(tex.shape[0], tex.shape[1], 3)
            new_uv = uv
        else:
            new_tex = None
            new_uv = None

        mapped_mid_obb_size_changes, mapped_pos_obb_size_changes, mapped_mid_obb_pose_changes, mapped_pos_obb_pose_changes, mapped_mid_obb_loc_changes, mapped_pos_obb_loc_changes = \
        map_bbox_parameters_sym(new_mid_obb_size_changes, new_pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, angle_range, loc_range)
        obbs, obb_size_changes, obb_pose_changes, obb_loc_changes, new_obbs = get_bbox_parameters_sym(mid_obbs, pos_obbs, neg_obbs, mapped_mid_obb_size_changes, mapped_pos_obb_size_changes, mapped_mid_obb_pose_changes, mapped_pos_obb_pose_changes, mapped_mid_obb_loc_changes, mapped_pos_obb_loc_changes)
        new_obbs = update_mesh_by_obb_tree(new_mesh, obbs, new_obbs, obb_tree_root, nodes, node_orders, graph_edges, obb_size_changes, obb_pose_changes, obb_loc_changes, new_uv)

        if tex is not None and uv is not None and tex_bin is not None:
            encoded_renders, rendered_images = render_and_encode(new_mesh, new_tex, render, augment_transform, background, res, clip_model, view_option)
        else:
            encoded_renders, rendered_images = render_and_encode(new_mesh, tex, render, augment_transform, background, res, clip_model, view_option)
        
        loss = get_loss(encoded_renders, encoded_text, mesh, new_mesh, reverse_optim, True, shape_normal_error_weight)
        is_overlap, overlap_penalty = obb_overlap_degree(new_obbs, 1.0, overlap_graph)
        
        loss = loss
        
        optim.zero_grad()
        loss.backward(retain_graph=True)
        optim.step()

        print('gradient epoch', epoch, 'loss', loss.item())

        #if time.time() - start_time > max_time:
            #break

        #if activate_scheduler:
            #lr_scheduler.step()
    if tex is not None and uv is not None and tex_bin is not None:
        return new_mid_obb_size_changes.detach(), new_pos_obb_size_changes.detach(), mid_obb_pose_changes.detach(), pos_obb_pose_changes.detach(), mid_obb_loc_changes.detach(), pos_obb_loc_changes.detach(), new_tex.detach()
    else:
        return new_mid_obb_size_changes.detach(), new_pos_obb_size_changes.detach(), mid_obb_pose_changes.detach(), pos_obb_pose_changes.detach(), mid_obb_loc_changes.detach(), pos_obb_loc_changes.detach(), None
