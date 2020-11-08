from pprint import pprint

import cv2
import numpy as np
import pywavefront


class Mesh:
    def __init__(self, obj_file=None, texture_file=None):
        self.mesh = {}
        self.texture = {}

        if obj_file is not None:
            self.load(obj_file, texture_file)

    def load(self, obj_file, texture_file):
        print('+ loading obj file: ', obj_file, '\n')
        obj = pywavefront.Wavefront(obj_file, create_materials=True)

        for name, material in obj.materials.items():
            print('>', name)
            print('   Format:  ', material.vertex_format)
            print('   Vertices:', len(material.vertices))

            texture = None
            texture_name = None
            if material.texture is None:
                if texture_file is not None:
                    print('   Texture: ', texture_file)
                    texture_data = cv2.imread(texture_file)
                    texture_name = texture_file
                else:
                    print('   Texture: ', 'None')
            else:
                print('   Texture: ', material.texture.name)
                texture_data = cv2.imread(material.texture.path)
                texture_name = material.texture.name

            self.mesh[name] = {
                'format': material.vertex_format,
                'vertices': np.array(material.vertices),
                'texture_name': texture_name
            }

            if texture_name is not None:
                self.texture[texture_name] = texture_data


if __name__ == '__main__':
    mesh = Mesh(obj_file='../data/cube.obj')
