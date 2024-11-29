# text2mesh

import sys

from mesh import Mesh
import kaolin as kal
from utils import get_camera_from_view2
import matplotlib.pyplot as plt
from utils import device
import torch
import numpy as np
import imageio.v3 as iio

class Renderer():

    def __init__(self, mesh='sample.obj',
                 lights=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                 camera=kal.render.camera.generate_perspective_projection(np.pi / 3).to(device),
                 dim=(224, 224)):

        if camera is None:
            camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device)

        self.lights = lights.unsqueeze(0).to(device)
        self.camera_projection = camera
        self.dim = dim
        self.background_rotation_index = 0

    def render_front_views(self, mesh, num_views=8, std=8, center_elev=0, center_azim=0, input_elevs=[], input_azims=[], dists=[], show=False, lighting=True,
                           background=None, mask=False, return_views=False):
        # Front view with small perturbations in viewing angle
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        # limited view

        # use input view
        #if use_input:
            #elev = torch.tensor([center_elev], dtype=torch.float)
            #azim = torch.tensor([center_azim], dtype=torch.float)
        #else:
        
        #print('(torch.rand(num_views - 1)-0.5)', (torch.rand(num_views - 1)-0.5))
        #exit()

        if len(input_elevs) > 0 and len(input_azims) > 0:
            elev = torch.tensor(input_elevs, device=device, dtype=torch.float)
            azim = torch.tensor(input_azims, device=device, dtype=torch.float)
            num_views = len(input_elevs)
        else:
            elev = center_elev + (torch.rand(num_views)-0.5) * 0.5 * np.pi
            azim = center_azim + (torch.rand(num_views)-0.5) * 0.5 * np.pi


        #print('elev', elev)
        #exit()

        '''
        if len(input_elevs) > 0 and len(input_azims) > 0:
            elev = torch.tensor(input_elevs, device=device, dtype=torch.float)
            azim = torch.tensor(input_azims, device=device, dtype=torch.float)
        else:
                    elev = center_elev + (torch.rand(num_views)-0.5) * 2 * np.pi
        azim = center_azim + (torch.rand(num_views)-0.5) * 2 * np.pi

            elev_ranges = []
            elev_delta = 0.0
            for i in range(0, num_views):
            #for i in range(0, int(num_views)):
                #elev_range = torch.tensor(center_elev, device=device, dtype=torch.float)
                elev_range = torch.tensor(center_elev + i * elev_delta, device=device, dtype=torch.float)
                elev_ranges.append(elev_range)
            elev = torch.stack(elev_ranges)

            azim_ranges = []
            azim_delta = 0.5
            for i in range(0, num_views):
                azim_range = torch.tensor(center_azim + i * azim_delta, device=device, dtype=torch.float)                    
                azim_ranges.append(azim_range)
            azim = torch.stack(azim_ranges)
        '''

        #elev = torch.cat((torch.tensor([center_elev], dtype=torch.float), center_elev + (torch.rand(num_views - 1)-0.5) * 2 * np.pi))
        #azim = torch.cat((torch.tensor([center_azim], dtype=torch.float), center_azim + (torch.rand(num_views - 1)-0.5) * 2 * np.pi))

        #elev = center_elev + (torch.rand(num_views)-0.5) * 2 * np.pi
        #azim = center_azim + (torch.rand(num_views)-0.5) * 2 * np.pi
        
        #print('elev shape', elev.shape)
        #print('azim shpae', azim.shape)
        #exit()
        
        #azim = torch.cat((torch.tensor([0.5*np.pi]), 0.5*np.pi + (torch.rand(num_views - 1)-0.5)*0.2*np.pi))

        # random view
        #elev = torch.cat((torch.tensor([0.0]), (torch.rand(num_views - 1)-0.5)*2*np.pi))
        #azim = torch.cat((torch.tensor([0.0]), (torch.rand(num_views - 1)-0.5)*2*np.pi))

        #elev = torch.randn(num_views) * 2 * np.pi
        #azim = torch.randn(num_views - 1) * 2 * np.pi

        #print('elev', elev)
        #print('azim', azim)
        #exit()

        images = []
        masks = []
        rgb_mask = []

        '''
        if self.background_rotation_index % 3 == 0:
            background = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float)
        elif self.background_rotation_index % 3 == 1:
            background = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float)
        elif self.background_rotation_index % 3 == 2:
            background = torch.tensor([1.0, 0.5, 0.0], device=device, dtype=torch.float)
        else:
            exit()
        self.background_rotation_index += 1
        '''

        black_background = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float)
        white_background = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float)
        orange_background = torch.tensor([1.0, 0.5, 0.0], device=device, dtype=torch.float)
        backgrounds = [black_background, white_background, orange_background]
        #backgrounds = [white_background]

        face_attributes = [
            mesh.face_attributes,
            torch.ones((1, n_faces, 3, 1), device=device)
        ]

        #if background is not None:
            #face_attributes = [
                #mesh.face_attributes,
                #torch.ones((1, n_faces, 3, 1), device=device)
            #]
        #else:
            #face_attributes = mesh.face_attributes

        for background in backgrounds:
            for i in range(num_views):
                camera_transform = get_camera_from_view2(elev[i], azim[i], r=dists[i]).to(device)
                face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                    mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                    camera_transform=camera_transform)

                image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                    self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                    face_vertices_image, face_attributes, face_normals[:, :, -1])
                masks.append(soft_mask)

                # Debugging: color where soft mask is 1
                # tmp_rgb = torch.ones((224, 224, 3))
                # tmp_rgb[torch.where(soft_mask.squeeze() == 1)] = torch.tensor([1, 0, 0]).float()
                # rgb_mask.append(tmp_rgb)

                if background is not None:
                    image_features, mask = image_features

                image = torch.clamp(image_features, 0.0, 1.0)

                if lighting:
                    image_normals = face_normals[:, face_idx].squeeze(0)
                    image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                    image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                    image = torch.clamp(image, 0.0, 1.0)

                if background is not None:
                    background_mask = torch.zeros(image.shape).to(device)
                    mask = mask.squeeze(-1)
                    assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                    #print('background shape', background.shape)
                    #print('background_mask shape', background_mask.shape)
                    #print('background_mask[torch.where(mask == 0)] shape', background_mask[torch.where(mask == 0)].shape)
                    #background = background.unsqueeze(dim=0).repeat_interleave(len(background_mask[torch.where(mask == 0)]), dim=0)
                    background_mask[torch.where(mask == 0)] = background
                    image = torch.clamp(image + background_mask, 0., 1.)
                images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)
        # rgb_mask = torch.cat(rgb_mask, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                plt.show()

        if return_views == True:
            return images, elev, azim
        else:
            return images

    def render_front_views_with_textures(self, mesh, tex, num_views=8, std=8, center_elev=0, center_azim=0, input_elevs=[], input_azims=[], dists=[], show=False, lighting=True,
                           background=None, mask=False, return_views=False):
        # Front view with small perturbations in viewing angle
        verts = mesh.vertices
        faces = mesh.faces
        n_faces = faces.shape[0]

        if len(input_elevs) > 0 and len(input_azims) > 0:
            elev = torch.tensor(input_elevs, device=device, dtype=torch.float)
            azim = torch.tensor(input_azims, device=device, dtype=torch.float)
            num_views = len(input_elevs)
        else:
            elev = center_elev + (torch.rand(num_views)-0.5) * 0.5 * np.pi
            azim = center_azim + (torch.rand(num_views)-0.5) * 0.5 * np.pi

        images = []
        masks = []
        rgb_mask = []

        if background is not None:
            face_attributes = [
                mesh.face_attributes,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        else:
            face_attributes = mesh.face_attributes

        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=dists[i]).to(device)
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                mesh.vertices.to(device), mesh.faces.to(device), self.camera_projection,
                camera_transform=camera_transform)

            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            masks.append(soft_mask)

            texture_coords, mask = image_features
            image = kal.render.mesh.texture_mapping(texture_coords,
                                                tex.permute(2, 0, 1).unsqueeze(dim=0),
                                                mode='bilinear')

            #print('mask shape', mask.shape)
            #print('image shape', image.shape)
            #exit()
            
            #print('image shape', image.shape)
            #exit()

            image = torch.clamp(image * mask, 0., 1.)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clamp(image, 0.0, 1.0)

            #background = torch.rand(3, device=device, dtype=torch.float)
            #mask = 
            #image = torch.clamp(mask.repeat_interleave(3, dim=3), 0., 1.)

            #if self.background_rotation_index % 3 == 0:
            #background = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float)
            #elif self.background_rotation_index % 3 == 1:
                #background = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float)
            #elif self.background_rotation_index % 3 == 2:
                #background = torch.tensor([0.5, 0.5, 0.5], device=device, dtype=torch.float)
            #else:
                #print('? wrong rotation')
                #exit()
            
            self.background_rotation_index += 1

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                assert torch.all(image[torch.where(mask == 0)] == torch.zeros(3).to(device))
                #print('background shape', background.shape)
                #print('background_mask shape', background_mask.shape)
                #print('background_mask[torch.where(mask == 0)] shape', background_mask[torch.where(mask == 0)].shape)
                #background = background.unsqueeze(dim=0).repeat_interleave(len(background_mask[torch.where(mask == 0)]), dim=0)
                #background_mask[torch.where(mask == 0)] = background[torch.where(mask == 0)]
                background_mask[torch.where(mask == 0)] = background
                image = torch.clamp(image + background_mask, 0., 1.)
            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)
        # rgb_mask = torch.cat(rgb_mask, dim=0)

        if show:
            with torch.no_grad():
                fig, axs = plt.subplots(1 + (num_views - 1) // 4, min(4, num_views), figsize=(89.6, 22.4))
                for i in range(num_views):
                    if num_views == 1:
                        ax = axs
                    elif num_views <= 4:
                        ax = axs[i]
                    else:
                        ax = axs[i // 4, i % 4]
                    ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                plt.show()

        if return_views == True:
            return images, elev, azim
        else:
            return images


if __name__ == '__main__':
    mesh = Mesh('sample.obj')
    mesh.set_image_texture('sample_texture.png')
    renderer = Renderer()
    # renderer.render_uniform_views(mesh,show=True,texture=True)
    mesh = mesh.divide()
    renderer.render_uniform_views(mesh, show=True, texture=True)
