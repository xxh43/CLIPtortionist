
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
from cmath import cos
from turtle import pen, pos
import clip
from pyparsing import col
from sklearn.preprocessing import scale
from tqdm import tqdm
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
from obb_tree_core import *
import kaolin

def get_constraint_graph(obbs, mesh):
    
    constraint_graph = torch.zeros((len(obbs), len(obbs), 6, 6, 3), device=device)

    for i in range(len(obbs)):
        for j in range(i+1, len(obbs)):
            if is_neighbor(obbs, mesh, i, j):
                min_dis = np.inf
                min_obb_i_face_index = -1
                min_obb_j_face_index = -1
                obb_i_points = obbs[i].get_face_center_points()
                obb_j_points = obbs[j].get_face_center_points()
                for f_i in range(len(obb_i_points)):
                    for f_j in range(len(obb_j_points)):
                        dis = torch.norm(obb_i_points[f_i] - obb_j_points[f_j])
                        if dis < min_dis:
                            min_dis = dis
                            min_obb_i_face_index = f_i
                            min_obb_j_face_index = f_j

                constraint_graph[i][j][min_obb_i_face_index][min_obb_j_face_index] = obb_i_points[f_i] - obb_j_points[f_j]
                print('constraint_graph[i][j][min_obb_i_face_index][min_obb_j_face_index]', constraint_graph[i][j][min_obb_i_face_index][min_obb_j_face_index])

    return constraint_graph

def compute_constraint_graph_penalty(obbs, connect_graph, overlap_graph, penalty_factor):
    all_penalty = 0
    penalty_count = 0
    
    for i in range(len(obbs)):
        for j in range(i+1, len(obbs)):
            obb_i_points = obbs[i].get_face_center_points()
            obb_j_points = obbs[j].get_face_center_points()
            for f_i in range(len(obb_i_points)):
                for f_j in range(len(obb_j_points)):
                    old_vector = connect_graph[i][j][f_i][f_j]
                    if torch.norm(old_vector) > 0.0001:
                        new_vector = obb_i_points[f_i] - obb_j_points[f_j]
                        detach_penalty = torch.norm(new_vector - old_vector)
                        if detach_penalty > 0.1:
                            all_penalty += detach_penalty - 0.1
            
            all_penalty += torch.relu(is_intersected(obbs[i], obbs[j]) - overlap_graph[i][j])

    return (all_penalty/len(obbs))*penalty_factor

def compute_obb_graph(mesh, mid_obbs, pos_obbs, neg_obbs):

    edges = []
    for i in range(len(mid_obbs+pos_obbs+neg_obbs)):
        for j in range(i+1, len(mid_obbs+pos_obbs+neg_obbs)):
            if i != j:
                if is_neighbor(mid_obbs+pos_obbs+neg_obbs, mesh, i, j):
                    edges.append([i, j])
    return edges


    
