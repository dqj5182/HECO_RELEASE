from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class ContactRenderer:
    def __init__(self):
        self.default_mesh_color = [200, 200, 200, 255]
        self.contact_mesh_color = [0, 255, 0, 255]

        path = Path(__file__).resolve().parents[2] / 'data/base_data/human_models/flame/FLAME_template.npz'

        with np.load(path) as mesh:
            self.faces = mesh['faces']
            self.vertices = np.asarray(mesh['vertices'], dtype=np.float64).copy()

        self.vertices -= (self.vertices.max(0) + self.vertices.min(0)) / 2
        self.scale = 1 / max(np.ptp(self.vertices[:, 0]), np.ptp(self.vertices[:, 1]))


    def render_view(self, probability, sideview_angle=0, topview_angle=0, size=500):
        yaw, pitch = np.radians([sideview_angle, topview_angle])
        rotation_y = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ])
        rotation_x = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ])
        vertices = self.vertices @ rotation_y.T @ rotation_x.T

        scale = (size - 24) * self.scale
        pixels = np.stack((size / 2 + scale * vertices[:, 0], size / 2 - scale * vertices[:, 1]), -1)

        triangles = vertices[self.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)

        light = np.array([-.3, .4, 1.])
        light /= np.linalg.norm(light)
        shade = .80 + .20 * np.abs(normals @ light)

        active = (np.asarray(probability)[self.faces] >= .5).mean(1)
        base = (1 - active[:, None]) * np.array(self.default_mesh_color[:3]) + active[:, None] * np.array(self.contact_mesh_color[:3])
        colors = np.clip(base * shade[:, None], 0, 255).astype(np.uint8)

        image = Image.new('RGB', (size, size), 'white')
        draw = ImageDraw.Draw(image)

        for index in np.argsort(triangles[:, :, 2].mean(1)):
            draw.polygon([tuple(x) for x in pixels[self.faces[index]]], fill=tuple(colors[index]))

        return image


    def create_scene(self, image, probability, img_res=500):
        image = image.convert('RGB')
        width = int(img_res * image.width / image.height)
        mesh_images = [image.resize((width, img_res), Image.Resampling.BICUBIC)]

        for angle in [0, 90, 180, 270]:
            mesh_images.append(self.render_view(probability, sideview_angle=angle, size=img_res))

        for angle in [90, 270]:
            mesh_images.append(self.render_view(probability, topview_angle=angle, size=img_res))

        return Image.fromarray(np.hstack([np.asarray(view) for view in mesh_images]))


    def render_contact(self, image, probability):
        return self.create_scene(image, probability)