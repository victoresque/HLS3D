import numpy as np

from camera import CameraBase


class Light(CameraBase):
    def __init__(self):
        world_to_light = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        super().__init__(world_to_light)

        self.translate_z(2000)
        self.translate_y(2000)
        self.translate_x(2000)
        self.rotate_y(90)
        self.rotate_x(45)

        self.ambient = 0.5

        self.shadow_map_dim = 512
        self.shadow_map_depth = 10000
        self.shadow_map_param = [0, 1, 0, 1]

    @property
    def world_to_light(self):
        return self.world_to_camera

    @property
    def light_to_world(self):
        return np.linalg.inv(self.world_to_light)

    def project(self, point):
        xoffset, xscale, yoffset, yscale = self.shadow_map_param
        offset = np.array([xoffset, yoffset, 0.0])
        scale = np.array([xscale, yscale, 1.0])
        return (point - offset) * scale
