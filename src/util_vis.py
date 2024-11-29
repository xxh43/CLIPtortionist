


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 

from config import *
import matplotlib.colors as mcolors
import numpy as np
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

alpha = 220
color_palette = ['red', 'green', 'blue', 'purple', 'yellow', 'cyan', 'brown', 'pink', 'black', 'blueviolet', 'burlywood', 'cadetblue']

def set_fig(traces, filename, save, has_bg=False):
    
    fig = go.Figure(data=traces)

    for trace in fig['data']: 
        trace['showlegend'] = False

    camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0.5, y=1.0, z=0.5)
    )

    fig.update_layout(scene_camera=camera)

    range_min = -2.0
    range_max = 2.0

    if has_bg:
        fig.update_layout(
            scene = dict(xaxis = dict(range=[range_min,range_max],),
                        yaxis = dict(range=[range_min,range_max],),
                        zaxis = dict(range=[range_min,range_max],),
                        aspectmode='cube'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=filename,
            autosize=False,width=1000,height=1000)
    else:
        fig.update_layout(
            scene = dict(xaxis = dict(range=[range_min,range_max],),
                        yaxis = dict(range=[range_min,range_max],),
                        zaxis = dict(range=[range_min,range_max],),
                        aspectmode='cube'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=filename,
            autosize=False,width=1000,height=1000)

    #fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
    
    if save is True:
        fig.write_image(filename)
    else:
        fig.show()


def display_pcs_and_cubes(pcs, cubes, filename='', save=False):

    pcs = [ to_numpy(pcs[i]) for i in range(len(pcs))]

    pc_traces = []
    all_x = []
    all_y = []
    all_z = []
    for i in range(len(pcs)):
        pcs[i] = to_numpy(pcs[i])
        x = []
        y = []
        z = []
        c = []
        for p in pcs[i]:
            color = color_palette[i%len(color_palette)]
            c.append(color)
            x.append(to_numpy(p[0]))
            y.append(to_numpy(p[1]))
            z.append(to_numpy(p[2]))

        trace = go.Scatter3d(
            x=x, 
            y=y, 
            z=z, 
            mode='markers', 
            marker=dict(
                size=5,
                color=c,                
                colorscale='Viridis',   
                opacity=1.0
            )
        )
        pc_traces.append(trace)
        all_x += x
        all_y += y
        all_z += z
    
    edge_traces = []
    for i in range(len(cubes)):
        color = color_palette[i%len(color_palette)]
        edges = cubes[i].get_edges()
        for j in range(len(edges)):
            edge = to_numpy(edges[j])
            #print('i', i, 'j', j)
            trace = go.Scatter3d(
                x = [to_numpy(edge[0][0]), to_numpy(edge[1][0])], 
                y = [to_numpy(edge[0][1]), to_numpy(edge[1][1])],
                z = [to_numpy(edge[0][2]), to_numpy(edge[1][2])],
                line=dict(
                    color=color,
                    width=3
                )
            )

            edge_traces.append(trace)

    traces = pc_traces+edge_traces
    set_fig(traces, filename, save)

def display_pcs_and_cube_groups(pcs, cube_groups, filename='', save=False):

    pc_traces = []
    all_x = []
    all_y = []
    all_z = []
    for i in range(len(pcs)):
        pcs[i] = to_numpy(pcs[i])
        x = []
        y = []
        z = []
        c = []
        for p in pcs[i]:
            color = 'rgba(0.5, 0.5, 0.5, 0.5)'
            c.append(color)
            x.append(to_numpy(p[0]))
            y.append(to_numpy(p[1]))
            z.append(to_numpy(p[2]))

        trace = go.Scatter3d(
            x=x, 
            y=y, 
            z=z, 
            mode='markers', 
            marker=dict(
                size=5,
                color=c,                
                colorscale='Viridis',   
                opacity=1.0
            )
        )
        pc_traces.append(trace)
        all_x += x
        all_y += y
        all_z += z
    
    edge_traces = []
    for i in range(len(cube_groups)):
        color = color_palette[i%len(color_palette)]
        for j in range(len(cube_groups[i])):
            cube = cube_groups[i][j]
            edges = cube.get_edges()
            for k in range(len(edges)):
                edge = to_numpy(edges[k])
                trace = go.Scatter3d(
                    x = [to_numpy(edge[0][0]), to_numpy(edge[1][0])], 
                    y = [to_numpy(edge[0][1]), to_numpy(edge[1][1])],
                    z = [to_numpy(edge[0][2]), to_numpy(edge[1][2])],
                    line=dict(
                        color=color,
                        width=3
                    )
                )

                edge_traces.append(trace)

    traces = pc_traces+edge_traces
    set_fig(traces, filename, save)

def display_meshes(meshes, filename=' ', save=False):

    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    zmin = np.inf
    zmax = -np.inf

    traces = []
    for mesh_index in range(len(meshes)):
        print('mesh_index', mesh_index)
        color = color_palette[mesh_index]
        mesh = meshes[mesh_index]
        
        for i in range(len(mesh.faces)):
            edge01 = np.array([mesh.vertices[mesh.faces[i][0]], mesh.vertices[mesh.faces[i][1]]])
            edge12 = np.array([mesh.vertices[mesh.faces[i][1]], mesh.vertices[mesh.faces[i][2]]])
            edge20 = np.array([mesh.vertices[mesh.faces[i][2]], mesh.vertices[mesh.faces[i][0]]])

            edges = [edge01, edge12, edge20]
            
            for edge in edges:
                #print('i', i, 'j', j)
                trace = go.Scatter3d(
                    x = [to_numpy(edge[0][0]), to_numpy(edge[1][0])], 
                    y = [to_numpy(edge[0][1]), to_numpy(edge[1][1])],
                    z = [to_numpy(edge[0][2]), to_numpy(edge[1][2])],
                    line=dict(
                        color=color,
                        width=5
                    )
                )

                traces.append(trace)

    set_fig(traces, filename, save)

def display_pcs_and_wireframes(pcs, meshes, filename=' ', save=False):

    traces = []    
    for i in range(len(pcs)):
        pcs[i] = to_numpy(pcs[i])
        x = []
        y = []
        z = []
        c = []
        for p in pcs[i]:
            c.append(color_palette[i%len(color_palette)])
            x.append(to_numpy(p[0]))
            if len(p) > 1:
                y.append(to_numpy(p[1]))
            else:
                y.append(0.0)
            if len(p) > 2:
                z.append(to_numpy(p[2]))
            else:
                z.append(0.0)

        trace = go.Scatter3d(
            x=x, 
            y=y, 
            z=z, 
            mode='markers', 
            marker=dict(
                size=3,
                color=c,                
                colorscale='Viridis',   
                opacity=1.0
            )
        )
        traces.append(trace)
        
    edge_traces = []
    for mesh_index in range(len(meshes)):
        print('mesh_index', mesh_index)
        color = color_palette[mesh_index]
        mesh = meshes[mesh_index]
        
        for i in range(len(mesh.faces)):
            edge01 = np.array([mesh.vertices[mesh.faces[i][0]], mesh.vertices[mesh.faces[i][1]]])
            edge12 = np.array([mesh.vertices[mesh.faces[i][1]], mesh.vertices[mesh.faces[i][2]]])
            edge20 = np.array([mesh.vertices[mesh.faces[i][2]], mesh.vertices[mesh.faces[i][0]]])

            edges = [edge01, edge12, edge20]
            
            for edge in edges:
                #print('i', i, 'j', j)
                trace = go.Scatter3d(
                    x = [to_numpy(edge[0][0]), to_numpy(edge[1][0])], 
                    y = [to_numpy(edge[0][1]), to_numpy(edge[1][1])],
                    z = [to_numpy(edge[0][2]), to_numpy(edge[1][2])],
                    line=dict(
                        color=color,
                        width=5
                    )
                )

                edge_traces.append(trace)

    traces += edge_traces
    set_fig(traces, filename, save)

def display_pcs(pcs, filename=' ', save=False):

    pcs = [ to_numpy(pcs[i]) for i in range(len(pcs))]

    traces = []
    all_x = []
    all_y = []
    all_z = []
    for i in range(len(pcs)):
        pcs[i] = to_numpy(pcs[i])
        x = []
        y = []
        z = []
        c = []
        for p in pcs[i]:
            c.append(color_palette[i%len(color_palette)])
            x.append(p[0])
            if len(p) > 1:
                y.append(p[1])
            else:
                y.append(0.0)
            if len(p) > 2:
                z.append(p[2])
            else:
                z.append(0.0)

        trace = go.Scatter3d(
            x=x, 
            y=y, 
            z=z, 
            mode='markers', 
            marker=dict(
                size=3,
                color=c,                
                colorscale='Viridis',   
                opacity=1.0
            )
        )
        traces.append(trace)
        all_x += x
        all_y += y
        all_z += z
    
    set_fig(traces, filename, save)

def display_pc_and_cubes_and_vectors(pcs, cubes, vectors, filename=' ', save=False):

    pc_traces = []
    for i in range(len(pcs)):
        color = color_palette[i%len(color_palette)]
        pcs[i] = to_numpy(pcs[i])
        x = []
        y = []
        z = []
        c = []
        for p in pcs[i]:
            c.append(color)
            x.append(to_numpy(p[0]))
            y.append(to_numpy(p[1]))
            z.append(to_numpy(p[2]))

        trace = go.Scatter3d(
            x=x, 
            y=y, 
            z=z, 
            mode='markers', 
            marker=dict(
                size=5,
                color=c,                
                colorscale='Viridis',   
                opacity=1.0
            )
        )
        pc_traces.append(trace)

    edge_traces = []
    for i in range(len(cubes)):
        color = color_palette[i%len(color_palette)]
        edges = cubes[i].get_edges()
        for j in range(len(edges)):
            edge = to_numpy(edges[j])
            #print('i', i, 'j', j)
            trace = go.Scatter3d(
                x = [to_numpy(edge[0][0]), to_numpy(edge[1][0])], 
                y = [to_numpy(edge[0][1]), to_numpy(edge[1][1])],
                z = [to_numpy(edge[0][2]), to_numpy(edge[1][2])],
                line=dict(
                    color=color,
                    width=5
                )
            )

            edge_traces.append(trace)

    vector_traces = []
    for i in range(len(vectors)):    
        color = color_palette[i%len(color_palette)]
        dir_start = to_numpy(vectors[i][0])
        dir_end = to_numpy(vectors[i][1])

        trace = go.Scatter3d(
            x = [dir_start[0], dir_end[0]], 
            y = [dir_start[1], dir_end[1]],
            z = [dir_start[2], dir_end[2]],
            line=dict(
                color=color,
                width=5
            )
        )

        vector_traces.append(trace)

    traces = pc_traces + edge_traces + vector_traces
    set_fig(traces, filename, save)



def display_pcs_and_vectors(pcs, vectors, filename=' ', save=False):

    pc_traces = []
    for i in range(len(pcs)):
        color = color_palette[i%len(color_palette)]
        pcs[i] = to_numpy(pcs[i])
        x = []
        y = []
        z = []
        c = []
        for p in pcs[i]:
            c.append(color)
            x.append(to_numpy(p[0]))
            y.append(to_numpy(p[1]))
            z.append(to_numpy(p[2]))

        trace = go.Scatter3d(
            x=x, 
            y=y, 
            z=z, 
            mode='markers', 
            marker=dict(
                size=5,
                color=c,                
                colorscale='Viridis',   
                opacity=1.0
            )
        )
        pc_traces.append(trace)

    vector_traces = []
    for i in range(len(vectors)):    
        color = color_palette[i+len(pcs)]

        for j in range(len(vectors[i])):
            vector_start = pcs[i][j]
            #print('vector_start', vector_start)
            #print('vectors[i][j]', vectors[i][j])
            vector_end = vector_start + to_numpy(vectors[i][j])

            trace = go.Scatter3d(
                x = [vector_start[0], vector_end[0]], 
                y = [vector_start[1], vector_end[1]],
                z = [vector_start[2], vector_end[2]],
                line=dict(
                    color=color,
                    width=1
                ),
                marker=dict(
                    size=1,
                    color=color
                )
            )

            vector_traces.append(trace)

    traces = pc_traces + vector_traces
    set_fig(traces, filename, save)