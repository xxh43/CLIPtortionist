

from email.policy import default
import trimesh
from PIL import Image
from config import to_numpy
from util_mesh_surface import *
import imageio.v3 as iio
import os
from matplotlib import pyplot as plt
from PIL import Image
from collections import defaultdict
import joblib
from sklearn.cluster import KMeans
from util_motion import *
from skimage.measure import block_reduce

def parse_mtl(mtl_filename):

    arr = mtl_filename.split('/')[0:-1]
    parent_dir = os.path.join(*arr)
    print('parent_dir', parent_dir)
    
    mtl_file = open(mtl_filename, 'r')
    lines = mtl_file.readlines()

    mtl_count = 0
    for line in lines:
        if 'newmtl' in line:
            mtl_count += 1

    mtl_name_all = [None] * (mtl_count+1)
    kd_all = [None] * (mtl_count+1)
    kd_map_all = [None] * (mtl_count+1)

    mtl_index = -1
    for line in lines:

        if len(line) <= 1:
            continue
        line = line.replace('\n',"")

        if 'newmtl' in line:
            mtl_index += 1
            mtl_name_all[mtl_index] = line.split(' ')[1:][0]

        if 'Kd' in line and 'map' not in line:
            arr = line.split(' ')[1:]
            arr = [i for i in arr if i != '']
            kd_all[mtl_index] = np.array([float(arr[0]), float(arr[1]), float(arr[2])])    
        
        if 'map_Kd' in line:
            map_name = line.split(' ')[1]
            tex_image = iio.imread(os.path.join(parent_dir, map_name))
            kd_map_all[mtl_index] = tex_image/255.0

    kd_all[-1] = np.array([0.5, 0.5, 0.5])  

    max_dim0 = 0
    max_dim1 = 0

    for tex in kd_map_all:
        if tex is not None:
            max_dim0 = max(max_dim0, tex.shape[0])
            max_dim1 = max(max_dim1, tex.shape[1])

    for i in range(len(kd_map_all)):
        if kd_map_all[i] is None:
            tex = np.zeros((max_dim0, max_dim1, 3))
            tex[:, :, 0] = kd_all[i][0]         
            tex[:, :, 1] = kd_all[i][1]
            tex[:, :, 2] = kd_all[i][2]   
            kd_map_all[i] = tex
            #print('kd_all[i]', kd_all[i])
            #plt.imshow(tex/255.0, interpolation='nearest')
            #plt.show()
        #else:
            #repeated_kd = kd_all[i].repeat(kd_map_all[i].shape[0], axis=0)
            #repeated_kd = repeated_kd.repeat(kd_map_all[i].shape[1], axis=1)
            #kd_map_all[i] = kd_map_all[i] * repeated_kd

    global_tex, u_sep = fuse_textures(kd_map_all)

    return mtl_name_all, kd_map_all, global_tex, u_sep

def local_uv_to_global_uv(local_uv, local_index, tex, large_tex, u_sep):

    v = local_uv[1]
    #print('v', v)
    local_pixel0 = int(tex.shape[0]*(1-v))
    #print('tex.shape', tex.shape)
    #print('local_pixel0', local_pixel0)
    global_pixel0 = local_pixel0
    #print('large_tex shape', large_tex.shape)
    global_v = 1 - global_pixel0/large_tex.shape[0]
    #print('global_v', global_v)
    #exit()

    #v = local_uv[0]
    #local_pixel0 = int(tex.shape[0]*(v))
    #global_pixel0 = local_pixel0
    #global_v = global_pixel0/large_tex.shape[0]

    u = local_uv[0]
    #print('u', u)
    local_pixel1 = int(tex.shape[1]*u)
    global_pixel1 = local_index * u_sep + local_pixel1
    global_u = global_pixel1/large_tex.shape[1]

    #print('global_u', global_u, 'global_v', global_v)

    return np.array([global_u, global_v])

def split_polygon_to_triangles(arrs, faces, face_to_mtl, face_to_uv, mtl_index):

    for i in range(len(arrs)-2):

        f0 = arrs[0].split('/')
        if len(f0) == 2:
            f0_vertex_index = int(f0[0])
            f0_vertex_uv_index = int(f0[1])
        if len(f0) == 3:
            f0_vertex_index = int(f0[0])
            if f0[1] == '':
                f0_vertex_uv_index = 1
            else:
                f0_vertex_uv_index = int(f0[1])

        f1 = arrs[i+1].split('/')
        if len(f1) == 2:
            f1_vertex_index = int(f1[0])
            f1_vertex_uv_index = int(f1[1])
        if len(f1) == 3:
            f1_vertex_index = int(f1[0])
            if f1[1] == '':
                f1_vertex_uv_index = 1
            else:
                f1_vertex_uv_index = int(f1[1])

        f2 = arrs[i+2].split('/')
        if len(f2) == 2:
            f2_vertex_index = int(f2[0])
            f2_vertex_uv_index = int(f2[1])
        if len(f2) == 3:
            f2_vertex_index = int(f2[0])
            if f2[1] == '':
                f2_vertex_uv_index = 1
            else:
                f2_vertex_uv_index = int(f2[1])

        face = np.array([f0_vertex_index-1, f1_vertex_index-1, f2_vertex_index-1])
        faces.append(face)        
        face_to_mtl.append(mtl_index)
        face_to_uv.append([f0_vertex_uv_index-1, f1_vertex_uv_index-1, f2_vertex_uv_index-1])


def parse_obj(obj_filename, mtl_name_all):
    
    obj_file = open(obj_filename, 'r')
    lines = obj_file.readlines()
    
    vertices = []
    uvs = []    
    
    for line in lines:

        if len(line) <= 1:
            continue
        line = line.replace('\n',"")

        if line[0:2] == 'v ':
            arrs = line[2:].split(' ')
            arrs = [i for i in arrs if i != '']
            vertex = np.array([float(arrs[0]), float(arrs[1]), float(arrs[2])])
            vertices.append(vertex)

        if line[0:2] == 'vt':
            arrs = line[2:].split(' ')
            arrs = [i for i in arrs if i != '']
            uv = np.array([float(arrs[0]), float(arrs[1])])
            uvs.append(uv)

    face_to_mtl = []
    face_to_uv = []
    faces = []

    for line in lines:

        if len(line) <= 1:
            continue

        line = line.replace('\n',"")

        if 'usemtl' in line:
            mtl_name = line.split(' ')[1:][0]
            if mtl_name not in mtl_name_all:
                mtl_index = -1
            else:
                mtl_index = mtl_name_all.index(mtl_name)

        if line[0:2] == 'f ':
            arrs = line[2:].split(' ')
            arrs = [i for i in arrs if i != '']
            split_polygon_to_triangles(arrs, faces, face_to_mtl, face_to_uv, mtl_index)

    return np.stack(vertices), np.stack(faces), uvs, face_to_mtl, face_to_uv

def fuse_textures(texs):

    texs = texs

    #img = Image.fromarray(texs[0][:,:,0:3], 'RGB')
    #img.save('my.jpg')
    #img.show()
    
    max_dim0 = 0
    max_dim1 = 0

    for tex in texs:
        max_dim0 = max(max_dim0, tex.shape[0])+1
        max_dim1 = max(max_dim1, tex.shape[1])+1

    print('max_dim0', max_dim0, 'max_dim1', max_dim1)
    
    tex_num = len(texs)
    large_tex = np.zeros((max_dim0, max_dim1*tex_num, 3))
    for i in range(len(texs)):
        #print('texs[i][:,:,0:3]', texs[i][:,:,0:3])
        large_tex[0:0+texs[i].shape[0], i*max_dim1:i*max_dim1+texs[i].shape[1], 0:3] = texs[i][:,:,0:3]
        #print('large_tex[0:0+texs[i].shape[0], i*max_dim1:i*max_dim1+texs[i].shape[1], 0:3]', large_tex[0:0+texs[i].shape[0], i*max_dim1:i*max_dim1+texs[i].shape[1], 0:3])
        #exit()
    #plt.imshow(large_tex/255.0, interpolation='nearest')
    #plt.show()
    #exit()
    return large_tex, max_dim1

def map_tex_to_color_bin(tex):

    flat_tex = (tex.reshape(-1, 3)).astype(np.float)
    kmeans = KMeans(n_clusters=10, max_iter=30).fit(flat_tex)
    labels = kmeans.labels_

    print('labels', labels)

    color_bin = {}
    for label in range(1, max(labels)):
        color_bin[label-1] = np.squeeze(np.argwhere(labels==label), axis=1)

    return color_bin


def get_mesh_with_mtl(folder_name, category, shape_id):

    print('folder_name', folder_name)
    files = os.listdir(folder_name)
    print('files', files)
    
    obj_filename = None
    mtl_filename = None
    for file in files:
        if '.obj' == file[-4:]:
            obj_filename = os.path.join(folder_name, file)
        if '.mtl' == file[-4:]:
            mtl_filename = os.path.join(folder_name, file)

    print('obj_filename', obj_filename)
    print('mtl_filename', mtl_filename)
    if obj_filename is None or mtl_filename is None:
        print('None file name')
        exit()

    mtl_name_all, texs, global_tex, v_sep = parse_mtl(mtl_filename)

    print('mtl_name_all', mtl_name_all)

    while global_tex.shape[0] > 3000 or global_tex.shape[1] > 10000:
        for i in range(len(texs)):
            texs[i] = compress_tex(texs[i])
        global_tex = compress_tex(global_tex)
        v_sep *= 0.5

    #color_bin = None
    #if os.path.isfile(os.path.join(folder_name, 'color_bin.joblib')):
        #color_bin = joblib.load(os.path.join(folder_name, 'color_bin.joblib'))
    #else:
    color_bin = map_tex_to_color_bin(global_tex)
    #color_bin = None
    #joblib.dump(color_bin, os.path.join(folder_name, 'color_bin.joblib'))

    vertices, faces, uvs, face_to_mtl, face_to_uv = parse_obj(obj_filename, mtl_name_all)
    
    face_uvs = []
    for i in range(len(faces)):
        local_index = face_to_mtl[i]
        global_uv_0 = local_uv_to_global_uv(uvs[face_to_uv[i][0]], local_index, texs[local_index], global_tex, v_sep)
        global_uv_1 = local_uv_to_global_uv(uvs[face_to_uv[i][1]], local_index, texs[local_index], global_tex, v_sep)
        global_uv_2 = local_uv_to_global_uv(uvs[face_to_uv[i][2]], local_index, texs[local_index], global_tex, v_sep)
        face_uv = np.array([global_uv_0, global_uv_1, global_uv_2])
        face_uvs.append(face_uv)
    face_uvs = np.stack(face_uvs)
    
    print('vertices shape', vertices.shape)
    print('faces shape', faces.shape)
    print('face_uvs shape', face_uvs.shape)

    vertices = rotate_vertices(vertices, category, shape_id)

    print('global_tex shape', global_tex.shape)
    
    #print('compressed_global_tex shape', global_tex.shape)
    #exit()

    return vertices, faces, face_uvs, global_tex, color_bin

def compress_tex(tex):
    compressed_tex = []
    for i in range(3):
        compressed_tex.append(block_reduce(tex[:, :, i], block_size=2, func=np.mean, cval=0, func_kwargs=None))
    compressed_tex = np.stack(compressed_tex, axis=2)
    return compressed_tex

def rotate_vertices(vertices, category, shape_id):



    if category == 'airplane':
        rotate_dict = airplane_rotate_dict
    if category == 'chair':
        rotate_dict = chair_rotate_dict
    if category == 'table':
        rotate_dict = table_rotate_dict

    if shape_id not in rotate_dict:
        angles = [0, 0, 0]
    else:
        angles = rotate_dict[shape_id]    

    axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    centers = [np.zeros(3)]*3

    rotated_vertices = copy.deepcopy(vertices)
    for i in range(len(axes)):
        rotated_vertices = rotate_with_axis_center_angle(torch.tensor(rotated_vertices, dtype=torch.float), torch.tensor(axes[i], dtype=torch.float), torch.tensor(centers[i], dtype=torch.float), torch.tensor(angles[i], dtype=torch.float))

    rotated_vertices = rotated_vertices - torch.mean(rotated_vertices, dim=0)

    return to_numpy(rotated_vertices)

airplane_rotate_dict = {
    '2': [-0.5*np.pi, 0, 0],
    '8': [0, 0.5*np.pi, 0],
    '12': [-0.5*np.pi, -0.5*np.pi, 0],
    '13': [-0.5*np.pi, np.pi, 0],
    '14': [0, np.pi, 0],
    '15': [0, 0.5*np.pi, 0.5*np.pi],
    '16': [0, np.pi, 0],
    '17': [-0.5*np.pi, 0, 0]
    }

chair_rotate_dict = {
    '3': [0, -0.5*np.pi, 0],
    '11': [0, -0.5*np.pi, 0],
    '12': [0, -0.5*np.pi, 0],
    '13': [0, 0.5*np.pi, 0],
    '14': [0, 0.5*np.pi, 0],
    '22': [-0.5*np.pi, np.pi, 0],
    '25': [-0.5*np.pi, 0, 0]
    }

table_rotate_dict = {
    '14': [-0.5*np.pi, 0, 0],
    '16': [-0.5*np.pi, 0, 0],
    '17': [-0.5*np.pi, 0, 0],
    }
