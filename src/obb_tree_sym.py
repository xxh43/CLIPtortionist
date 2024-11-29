
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
import itertools
from obb_tree_core import *

def build_obb_tree_sym(mid_obbs, pos_obbs, neg_obbs, mesh):
    
    obbs = mid_obbs + pos_obbs + neg_obbs
    root_index = pick_root_index(mid_obbs)   
    
    nodes = []
    for i in range(len(obbs)):
        node = ObbNode()
        node.index = i
        node.layer = -1
        node.pc_indices = obbs[i].pc_indices
        nodes.append(node) 

    front_obb_indices = [root_index]
    back_obb_indices = []
    visited_obb_indices = [root_index]
    
    nodes[root_index].layer = 0

    order = 0
    node_orders = [-1]*len(obbs)

    while True:

        next_obb_indices = []
        for obb_index in front_obb_indices:
            node_orders[obb_index] = order
            cur_node = nodes[obb_index]
            nb_obb_indices = get_neighbor_obbs(mid_obbs+pos_obbs+neg_obbs, obb_index, mesh)
            non_visited_nb_obb_indices = [v for v in nb_obb_indices if v not in visited_obb_indices]
            for nb_obb_index in non_visited_nb_obb_indices:
                nb_node = nodes[nb_obb_index]
                nb_node.layer = cur_node.layer + 1
                cur_node.child_loc_follow_constraints.append(compute_loc_follow_constraint_info(mid_obbs+pos_obbs+neg_obbs, cur_node.index, nb_node.index, len(mid_obbs)))
                cur_node.childs.append(nb_node)
            next_obb_indices.append(non_visited_nb_obb_indices)  
            visited_obb_indices += non_visited_nb_obb_indices      

        #for l in next_obb_indices: 
            #visited_obb_indices += l
        
        if len(next_obb_indices) == 0:
            break

        front_obb_indices = []
        for l in next_obb_indices:
            for obb_index in l:
                if  obb_index < len(mid_obbs): 
                    if obb_index not in front_obb_indices:
                        front_obb_indices.append(obb_index)        
                else:
                    if obb_index not in back_obb_indices:
                        back_obb_indices.append(obb_index)    

        if len(front_obb_indices) == 0 and len(back_obb_indices) > 0:
            front_obb_indices = back_obb_indices
            back_obb_indices = []

        order += 1

    if len(visited_obb_indices) != len(mid_obbs) + len(pos_obbs) + len(neg_obbs):
        return None, None, None
    else:
        return nodes[root_index], nodes, node_orders

def merge_2_obbs_sym(mid_obbs, pos_obbs, neg_obbs, mesh, obb_a_index, obb_b_index, merge_parent):

    # input mid, mid
    # mid + mid -> mid
    if obb_a_index < len(mid_obbs) and obb_b_index < len(mid_obbs):

        return False
        print('mid + mid -> mid')

        mid_remove_index_a = obb_a_index
        mid_remove_index_b = obb_b_index
        obb_a_to_remove = mid_obbs[mid_remove_index_a]
        obb_b_to_remove = mid_obbs[mid_remove_index_b]
        merged_obb = get_merged_obb(mesh, [obb_a_to_remove, obb_b_to_remove])

        mid_obbs.remove(obb_a_to_remove)
        mid_obbs.remove(obb_b_to_remove)
        mid_obbs.append(merged_obb)
       
        return True

    # input pos, pos
    # pos + pos -> pos, neg + neg -> neg
    elif (obb_a_index >= len(mid_obbs) and obb_a_index < len(mid_obbs)+len(pos_obbs)) and (obb_b_index >= len(mid_obbs) and obb_b_index < len(mid_obbs)+len(pos_obbs)):
        print('pos + pos -> pos, neg + neg -> neg')

        return False

        pos_remove_index_a = obb_a_index - len(mid_obbs)
        pos_remove_index_b = obb_b_index - len(mid_obbs)
        pos_obb_to_remove_a = pos_obbs[pos_remove_index_a]
        pos_obb_to_remove_b = pos_obbs[pos_remove_index_b]
        merged_obb = get_merged_obb(mesh, [pos_obb_to_remove_a, pos_obb_to_remove_b])
        pos_obbs.remove(pos_obb_to_remove_a)
        pos_obbs.remove(pos_obb_to_remove_b)
        pos_obbs.append(merged_obb)

        neg_remove_index_a = obb_a_index - len(mid_obbs)
        neg_remove_index_b = obb_b_index - len(mid_obbs)
        neg_obb_to_remove_a = neg_obbs[neg_remove_index_a]
        neg_obb_to_remove_b = neg_obbs[neg_remove_index_b]
        merged_obb = get_merged_obb(mesh, [neg_obb_to_remove_a, neg_obb_to_remove_b])

        neg_obbs.remove(neg_obb_to_remove_a)
        neg_obbs.remove(neg_obb_to_remove_b)
        neg_obbs.append(merged_obb)
        return True
    
    # input pos, neg
    # pos + neg -> mid 
    elif (obb_a_index >= len(mid_obbs) and obb_a_index < len(mid_obbs)+len(pos_obbs)) and (obb_b_index >= len(mid_obbs)+len(pos_obbs) and obb_a_index < len(mid_obbs)+len(pos_obbs)+len(neg_obbs)):
        
        print('pos + neg -> mid')
        pos_remove_index = obb_a_index - len(mid_obbs)
        neg_remove_index = obb_a_index - len(mid_obbs)
        pos_obb_to_remove = pos_obbs[pos_remove_index]
        neg_obb_to_remove = neg_obbs[neg_remove_index]
        merged_obb = get_merged_obb(mesh, [pos_obb_to_remove, neg_obb_to_remove])
        #display_pcs_and_cubes([mesh.vertices], [pos_obb_to_remove, neg_obb_to_remove, merged_obb])
        pos_obbs.remove(pos_obb_to_remove)
        neg_obbs.remove(neg_obb_to_remove)
        mid_obbs.append(merged_obb)
        return True

    # input mid, pos
    #mid + pos + neg -> mid
    elif obb_a_index < len(mid_obbs) and (obb_b_index >= len(mid_obbs) and obb_b_index < len(mid_obbs) + len(pos_obbs)):

        return False

        print('mid + pos + neg -> mid')
        mid_remove_index = obb_a_index
        pos_remove_index = obb_b_index - len(mid_obbs)
        neg_remove_index = obb_b_index - len(mid_obbs)

        mid_obb_to_remove = mid_obbs[mid_remove_index]
        pos_obb_to_remove = pos_obbs[pos_remove_index]
        neg_obb_to_remove = neg_obbs[neg_remove_index]

        merged_obb = get_merged_obb(mesh, [mid_obb_to_remove, pos_obb_to_remove, neg_obb_to_remove])
        #display_pcs_and_cubes([mesh.vertices], [mid_obb_to_remove, pos_obb_to_remove, neg_obb_to_remove, merged_obb])

        mid_obbs.remove(mid_obb_to_remove)
        pos_obbs.remove(pos_obb_to_remove)
        neg_obbs.remove(neg_obb_to_remove)  

        mid_obbs.append(merged_obb)   
        return True

    else: # mid + neg
        return False

        print('impossible case ------------------- ')
        print('len(mid_obbs)', len(mid_obbs))
        print('len(pos_obbs)', len(pos_obbs))
        print('len(neg_obbs)', len(neg_obbs))
        print('obb_a_index', obb_a_index)
        print('obb_b_index', obb_b_index)
        exit()

def merge_parent_layer_sym(mid_obbs, pos_obbs, neg_obbs, mesh, cur_obb_indices, temp_next_obb_indices):
    
    obbs = mid_obbs + pos_obbs + neg_obbs
    for i in range(len(temp_next_obb_indices)):
        for j in range(len(temp_next_obb_indices)):
            if i != j:
                if len(set(temp_next_obb_indices[i]).intersection(set(temp_next_obb_indices[j]))) > 0:
                    #print('i', temp_next_obb_indices[i])
                    #print('j', temp_next_obb_indices[j])
                    #print('i', i, 'j', j)
                    updated = merge_2_obbs_sym(mid_obbs, pos_obbs, neg_obbs, mesh, cur_obb_indices[i], cur_obb_indices[j], True)    
                    if updated:
                        return True
    return False

def merge_child_layer_sym(mid_obbs, pos_obbs, neg_obbs, mesh, cur_obb_indices, temp_next_obb_indices):
    obbs = mid_obbs + pos_obbs + neg_obbs
    #print('len(mid_obbs)', len(mid_obbs))
    #print('len(pos_obbs)', len(pos_obbs))
    #print('len(neg_obbs)', len(neg_obbs))
    for child_obb_indices in temp_next_obb_indices:
        for i in range(len(child_obb_indices)):
            for j in range(len(child_obb_indices)):
                if i != j:
                    obb_a_index = child_obb_indices[i]
                    obb_b_index = child_obb_indices[j]
                    if is_neighbor(obbs, mesh, obb_a_index, obb_b_index):
                        updated = merge_2_obbs_sym(mid_obbs, pos_obbs, neg_obbs, mesh, obb_a_index, obb_b_index, False)
                        if updated:
                            return True
    return False

def update_obbs_sym(mid_obbs, pos_obbs, neg_obbs, mesh, update_parent):

    if update_parent:
        print('update parent---------------')
    else:
        print('update childs---------------')
    
    root_index = pick_root_index(mid_obbs)   
    front_obb_indices = [root_index]
    back_obb_indices = []
    visited_obb_indices = [root_index]

    while True:
        
        next_obb_indices = []
        for obb_index in front_obb_indices:
            nb_obb_indices = get_neighbor_obbs(mid_obbs+pos_obbs+neg_obbs, obb_index, mesh)
            next_obb_indices.append([v for v in nb_obb_indices if v not in visited_obb_indices])
        
        for l in next_obb_indices: 
            visited_obb_indices += l
        
        if len(next_obb_indices) == 0:
            break
        
        #if update_parent:
        has_update = merge_parent_layer_sym(mid_obbs, pos_obbs, neg_obbs, mesh, front_obb_indices, next_obb_indices)
        #else:
            #has_update = merge_child_layer_sym(mid_obbs, pos_obbs, neg_obbs, mesh, front_obb_indices, next_obb_indices)

        if has_update:
            return True

        front_obb_indices = []
        for l in next_obb_indices:
            for obb_index in l:
                if  obb_index < len(mid_obbs): 
                    if obb_index not in front_obb_indices:
                        front_obb_indices.append(obb_index)        
                else:
                    if obb_index not in back_obb_indices:
                        back_obb_indices.append(obb_index)    

        if len(front_obb_indices) == 0 and len(back_obb_indices) > 0:
            front_obb_indices = back_obb_indices
            back_obb_indices = []

    return False

def prepare_obb_tree_sym(mid_obbs, pos_obbs, neg_obbs, mesh):

    #obbs = mid_obbs + pos_obbs + neg_obbs
    #display_pcs_and_cubes([mesh.vertices], obbs)

    final_has_update = True
    while final_has_update:
        final_has_update = False

        has_update = True
        while has_update:
            has_update = update_obbs_sym(mid_obbs, pos_obbs, neg_obbs, mesh, update_parent=True)
            if has_update:
                final_has_update = True

        #has_update = True
        #while has_update:
            #has_update = update_obbs_sym(mid_obbs, pos_obbs, neg_obbs, mesh, update_parent=False)
            #if has_update:
                #final_has_update = True
    
    #obbs = mid_obbs + pos_obbs + neg_obbs
    #display_pcs_and_cubes([mesh.vertices], obbs)
    #exit()

    return mid_obbs, pos_obbs, neg_obbs