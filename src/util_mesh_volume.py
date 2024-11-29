
import copy
from platform import node
from time import time
import numpy as np
import trimesh
import math
import time
from mesh_contain.inside_mesh import check_mesh_contains

def get_submesh(mesh, face_indices):
    submesh = copy.deepcopy(mesh)
    submesh.faces = mesh.faces[face_indices]
    return submesh

def load_mesh(obj_filename):
    tri_obj = trimesh.load_mesh(obj_filename)
    if tri_obj.is_empty:
        return None
    if type(tri_obj) is trimesh.scene.scene.Scene:
        tri_mesh = tri_obj.dump(True)
    else:
        tri_mesh = tri_obj

    return tri_mesh

def merge2mesh(mesh1, mesh2):
    new_mesh = copy.deepcopy(mesh1)
    shifted_mesh2_faces = copy.deepcopy(mesh2.faces) + copy.deepcopy(mesh1.vertices.shape[0])
    new_mesh.faces = np.concatenate((copy.deepcopy(mesh1.faces), copy.deepcopy(shifted_mesh2_faces)))
    new_mesh.vertices = np.concatenate((copy.deepcopy(mesh1.vertices), copy.deepcopy(mesh2.vertices)))
    return new_mesh

def merge_meshes(meshes):
    if len(meshes) == 0:
        return None
    base_mesh = meshes[0]
    for i in range(1, len(meshes)):
        base_mesh = merge2mesh(base_mesh, meshes[i])
    return base_mesh

def points_in_mesh(mesh, points):
    labels = check_mesh_contains(mesh, points)
    return labels
    