import numpy as np

from mesh import Mesh
from transformable import Transformable


class GameObject(Transformable):
    def __init__(self):
        super().__init__(np.eye(4))
        self.mesh = None
        self.rigid_body = None

    def load_mesh(self, obj_file, texture_file=None):
        self.mesh = Mesh(obj_file, texture_file)
