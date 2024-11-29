from cmath import cos
from turtle import shape
import clip
import kaolin.ops.mesh
import kaolin as kal
import torch
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
import matplotlib
from matplotlib import pyplot as plt
import pymeshlab 
from main_utils_core import *
from main_utils_sym import *
from obb_split import *
from load_mesh import *
from interpolation_sym import *

import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional, cast
from torch import Tensor
from collections import OrderedDict 
from torchvision.models import vgg16

print(torch.__version__)


def set_seed(seed):
    print('setting seed', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def visulize_obbs(mesh, obbs, filename):
    labeled_vertices = []
    pose_vectors = []
    rotmat_vectors = []

    for i in range(len(obbs)):
        labeled_vertices.append(mesh.vertices[obbs[i].pc_indices])
        #if obbs[i].pose_rotation_axis is not None:
            #pose_vector = [obbs[i].center, obbs[i].center + obbs[i].pose_rotation_axis]
            #pose_vectors.append(pose_vector)
        #rotmat_vectors.append([obbs[i].center, obbs[i].center + obbs[i].rotmat[:, 0]])
        #rotmat_vectors.append([obbs[i].center, obbs[i].center + obbs[i].rotmat[:, 1]])
        #rotmat_vectors.append([obbs[i].center, obbs[i].center + obbs[i].rotmat[:, 2]])

    #display_pcs_and_cubes(labeled_vertices, obbs, filename, True)
    #display_pc_and_cubes_and_vectors([], obbs, rotmat_vectors, filename, True)
    display_pc_and_cubes_and_vectors(labeled_vertices, obbs, pose_vectors, filename, True)
    #exit()



def run_sym_single_pass(args, split_level, mesh, tex, tex_bin, uv, clip_model, clip_augment_transform, full_augment_transform, encoded_text, render, background, res, shape_folder):

    level_folder = os.path.join(shape_folder, str(split_level))
    if not os.path.exists(level_folder):
        os.makedirs(level_folder) 

    mid_obbs, pos_obbs, neg_obbs = build_obbs_by_level_sym(split_level, trimesh.Trimesh(to_numpy(mesh.vertices), to_numpy(mesh.faces), process=False), shape_folder)
    visulize_obbs(mesh, mid_obbs+pos_obbs+neg_obbs, os.path.join(level_folder, 'original_obbs.png'))
    mid_obbs, pos_obbs, neg_obbs = prepare_obb_tree_sym(mid_obbs, pos_obbs, neg_obbs, mesh)
    visulize_obbs(mesh, mid_obbs+pos_obbs+neg_obbs, os.path.join(level_folder, 'merged_obbs.png'))
    obb_tree_root, nodes, node_orders = build_obb_tree_sym(mid_obbs, pos_obbs, neg_obbs, mesh)
    if obb_tree_root is None:
        return None, None, None, None, None

    graph_edges = compute_obb_graph(mesh, mid_obbs, pos_obbs, neg_obbs)
    print('node_orders', node_orders)
    print('graph_edges', graph_edges)

    #visulize_obbs(mesh, mid_obbs+pos_obbs+neg_obbs, os.path.join(level_folder, 'prepared_obbs_after_tree_build.png'))
    #return None, None, None, None, None, None

    mid_obb_size_changes = torch.zeros((len(mid_obbs), 3), device=device)
    pos_obb_size_changes = torch.zeros((len(pos_obbs), 3), device=device)
    mid_obb_pose_changes = torch.zeros((len(mid_obbs), 1), device=device)
    pos_obb_pose_changes = torch.zeros((len(pos_obbs), 1), device=device)
    mid_obb_loc_changes = torch.zeros((len(mid_obbs), 3), device=device)
    pos_obb_loc_changes = torch.zeros((len(pos_obbs), 3), device=device)

    size_range = args.size_range
    pose_range = args.pose_range * np.pi
    loc_range = args.loc_range
    
    old_similarity, _, old_obbs, old_rendered_images = eval_result_sym(mesh, tex, uv, mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, pose_range, loc_range, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, clip_augment_transform, clip_model, render, res, background, 'demo')
    #torchvision.utils.save_image(old_rendered_images, 'demo_mesh.jpg')
    torchvision.utils.save_image(old_rendered_images, os.path.join(shape_folder, 'original_mesh.jpg'))
    #exit()

    max_iter = 1

    for iter in range(max_iter):
        
        if args.method == 'cma':

            mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes = \
                get_best_config_cma_sym(iter, mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, pose_range, loc_range, mesh, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, tex, uv, encoded_text, clip_augment_transform, clip_model, render, res, background, args.shape_normal_error_weight, args.view, args.reverse)

            #if tex is not None and uv is not None and tex_bin is not None:
                #mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, tex = \
                    #gradient_optimize_sym_with_color(iter, mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, pose_range, loc_range, mesh, tex, tex_bin, uv, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, full_augment_transform, clip_model, render, res, background, args.shape_normal_error_weight, args.view, args.reverse, args.box_learning_rate, False)

        elif args.method == 'gradient':
            mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, tex = \
                gradient_optimize_sym_with_color(iter, mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, pose_range, loc_range, mesh, tex, tex_bin, uv, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, full_augment_transform, clip_model, render, res, background, args.shape_normal_error_weight, args.view, args.reverse, args.box_learning_rate, True)
        else:
            print('wrong method')
            exit()

    similarity, new_mesh, new_obbs, rendered_images = eval_result_sym(mesh, tex, uv, mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, pose_range, loc_range, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges, encoded_text, clip_augment_transform, clip_model, render, res, background, args.view)
    encoded_renders, _ = encode_images(rendered_images, clip_augment_transform, clip_model, res)
    sims = get_similarity(encoded_renders, encoded_text, k=1, seperate=True)
    torchvision.utils.save_image(rendered_images, os.path.join(level_folder, 'final_mesh.jpg'))   
    top_similarity = torch.mean(torch.stack(sims))
    new_mesh.export(os.path.join(level_folder, 'level_final.obj'))
    visulize_obbs(new_mesh, new_obbs, os.path.join(level_folder, ' new_obbs.png'))
        
    return top_similarity, new_mesh, (mesh, mid_obb_size_changes, pos_obb_size_changes, mid_obb_pose_changes, pos_obb_pose_changes, mid_obb_loc_changes, pos_obb_loc_changes, size_range, pose_range, loc_range, mid_obbs, pos_obbs, neg_obbs, obb_tree_root, nodes, node_orders, graph_edges), mid_obbs+pos_obbs+neg_obbs, new_obbs

def run_box(args, mesh, tex, uv, tex_bin, clip_model, clip_augment_transform, full_augment_transform, encoded_text, render, background, res, shape_id, exp_folder, sym):

    shape_folder = os.path.join(exp_folder, shape_id)
    #if os.path.exists(shape_folder):
        #return 

    if not os.path.exists(shape_folder):
        os.makedirs(shape_folder)

    if tex is not None and uv is not None:
        matplotlib.image.imsave(os.path.join(shape_folder, 'tex.png'), to_numpy(tex))
        joblib.dump(uv, os.path.join(shape_folder, 'face_uv.joblib'))
        tex = None
        uv = None
        tex_bin = None

    summary = os.path.join(shape_folder, 'summary.txt')
    if os.path.isfile(summary):
        os.remove(summary)

    mesh.export(os.path.join(shape_folder, 'original.obj'))

    max_similarity = -np.inf
    final_mesh = None
    final_params = None
    final_old_obbs = None
    final_new_obbs = None

    split_levels = [2, 3, 4]
    for split_level in split_levels:
        print('split_level', split_level)
        similarity, deformed_mesh, params, old_obbs, new_obbs = run_sym_single_pass(args, split_level, mesh, tex, tex_bin, uv, clip_model, clip_augment_transform, full_augment_transform, encoded_text, render, background, res, shape_folder)

        append_items_to_file(summary, [split_level, similarity])
        if similarity > max_similarity:
            max_similarity = similarity
            final_mesh = deformed_mesh
            final_params = params
            final_old_obbs = old_obbs
            final_new_obbs = new_obbs

    #MeshNormalizer(final_mesh)() 
    final_mesh.export(os.path.join(shape_folder, 'final.obj'))
    joblib.dump(final_params, os.path.join(shape_folder, 'params.joblib'))
    joblib.dump(final_old_obbs, os.path.join(shape_folder, 'final_old_obbs.joblib'))
    joblib.dump(final_new_obbs, os.path.join(shape_folder, 'final_new_obbs.joblib'))

def run(args):

    prompt = ''.join(args.prompt)
    #prompt = args.prompt
    #prefix = 'a '
    #if prompt[0] in ['a', 'e', 'i', 'o', 'u']:
        #prefix = 'an '
    #prompt = prefix + prompt

    print('args.shape_normal_error_weight,', args.shape_normal_error_weight)

    exp_folder = os.path.join('exps', 'ours_'+str(args.deformer), 'mtl'+str(args.usemtl) + '_' + args.category + '_' + str(prompt.split(' ')[1]) + '_' + str(args.method) \
        + '_' + str(args.size_range) + '_' + str(args.pose_range) + '_' + str(args.loc_range) + '_' + str(args.view) + '_sym_' + str(args.sym) + '_reverse_' + str(args.reverse) + '_normal_weight_' + str(args.shape_normal_error_weight))
    
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    
    clip_model, preprocess = clip.load(args.clipmodel, device, jit=args.jit)
    clip_model.eval()
    
    res = 224
    if args.clipmodel == "ViT-L/14@336px":
        res = 336
    if args.clipmodel == "RN50x4":
        res = 288
    if args.clipmodel == "RN50x16":
        res = 384
    if args.clipmodel == "RN50x64":
        res = 448
        
    render = Renderer(dim=(res, res))

    if args.background == 'white':
        background = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float)
    elif args.background == 'black':
        background = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float)
    else:
        print('wrong background')
        exit()

    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    clip_augment_transform = transforms.Compose([
        transforms.Resize((res, res)),
        clip_normalizer
    ])

    full_augment_transform = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.2),
        transforms.RandomResizedCrop(res, scale=(0.8, 0.8)),
        transforms.Resize((res, res)),
        clip_normalizer
    ])

    #print('prompt', prompt)
    #exit()

    prompt_token = clip.tokenize(prompt).to(device)
    encoded_text = clip_model.encode_text(prompt_token)
    
    data_folder = os.path.join(args.data_dir, args.category)
    folders = os.listdir(data_folder)

    use_mtl = args.usemtl

    for folder in folders:
        shape_id = folder
        folder_name = os.path.join(data_folder, folder)
        
        if use_mtl:
            vertices, faces, face_uvs, tex, color_bin = get_mesh_with_mtl(folder_name, args.category, shape_id)
            
            ms = pymeshlab.MeshSet()
            meshlab_mesh = pymeshlab.Mesh(to_numpy(vertices), to_numpy(faces))
            ms.add_mesh(meshlab_mesh, 'meshlab_mesh')
            ms.meshing_isotropic_explicit_remeshing(iterations=3)
            remeshed = ms.current_mesh()
            vertices = remeshed.vertex_matrix()
            faces = remeshed.face_matrix()

            mesh = Mesh(vertices, faces)
            MeshNormalizer(mesh)()
            uv = torch.tensor(face_uvs, device=device, dtype=torch.float)
            tex = torch.tensor(tex, device=device, dtype=torch.float)
            color_bin = color_bin

        else:
            mesh_filename = os.path.join(args.data_dir, args.category, shape_id)
            mesh = kal.io.obj.import_mesh(mesh_filename, with_normals=True)
            mesh = Mesh(mesh.vertices, mesh.faces)
            MeshNormalizer(mesh)()
            uv = None
            tex = None
            color_bin = None

        if args.deformer == 'box':
            #try:
            run_box(args, mesh, tex, uv, color_bin, clip_model, clip_augment_transform, full_augment_transform, encoded_text, render, background, res, shape_id, exp_folder, sym=True)
            #except:
                #continue
        
    exit()


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--usemtl', default=False, type=boolean_string)
    parser.add_argument('--shape_count', type=int, default=100)
    parser.add_argument('--category', type=str, default='airplane')
    parser.add_argument('--prompt', nargs="+", default='')
    parser.add_argument('--deformer', type=str, default='box')
    parser.add_argument('--clipmodel', type=str, default='ViT-B/32')
    #parser.add_argument('--clipmodel', type=str, default='ViT-L/14@336px')
    parser.add_argument('--jit', action="store_true")
    parser.add_argument('--size_range', type=float, default=0.7)
    parser.add_argument('--loc_range', type=float, default=0.0)
    parser.add_argument('--pose_range', type=float, default=0.00)
    parser.add_argument('--background', type=str, default='white')
    parser.add_argument('--method', type=str, default='cma')
    parser.add_argument('--reverse', default=False, type=boolean_string)
    parser.add_argument('--sequence', default=False, type=boolean_string)
    parser.add_argument('--sym', default=True, type=boolean_string)
    parser.add_argument('--view', type=str, default='front')
    parser.add_argument('--cage_learning_rate', type=float, default=0.01)
    parser.add_argument('--box_learning_rate', type=float, default=0.05)
    parser.add_argument('--shape_normal_error_weight', type=float, default=0.25)
    args = parser.parse_args()
    run(args)


