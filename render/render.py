from time import time

import cv2
import numpy as np

from scene import Scene
from light import Light
from camera import Camera
from game_object import GameObject


def triangle_area(v0, v1, v2):
    """
    | v01[0] v01[1] |
    | v02[0] v02[1] | = v01[0]*v02[1] - v01[1]*v02[0]
    """
    return (v1[0]-v0[0])*(v2[1]-v0[1]) - (v1[1]-v0[1])*(v2[0]-v0[0])


def geometric_transform(scene):
    cam_trans = scene.camera.world_to_camera
    near_clip = scene.camera.near_clip
    far_clip = scene.camera.far_clip

    light_trans = scene.lights[0].world_to_light

    # for shadow_map_param
    xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf

    for obj in scene.objects:
        obj_trans = obj.transform

        for name, mesh in obj.mesh.mesh.items():
            mesh_format = mesh['format']
            vertices = mesh['vertices']
            assert mesh_format in ['V3F', 'N3F_V3F', 'T2F_V3F', 'T2F_N3F_V3F']

            if mesh_format == 'V3F':
                step = 3
            elif mesh_format == 'N3F_V3F':
                step = 6
            elif mesh_format == 'T2F_V3F':
                step = 5
            elif mesh_format == 'T2F_N3F_V3F':
                step = 8

            mesh['cam_vertices'] = []
            mesh['light_vertices'] = []

            for i in range(0, len(vertices), step*3):
                cam_vertices = []
                cv_tmp = []
                lv_tmp = []

                for j in range(3):
                    K = i+j*step+step
                    vertex = np.ones((4, 1))
                    vertex[:3, 0] = vertices[K-3:K]
                    cam_vertex = (cam_trans @ obj_trans @ vertex).squeeze()[:3]
                    cv_tmp.append(cam_vertex)

                    if 'N3F' in mesh_format:
                        norm = np.ones((3, 1))
                        norm[:, 0] = vertices[K-6:K-3]
                        cam_norm = (cam_trans[:3, :3] @ obj_trans[:3, :3] @ norm).squeeze()
                        cam_vertices.extend([*vertices[K-step:K-6], *cam_norm, *cam_vertex])
                    else:
                        cam_vertices.extend([*vertices[K-step:K-3], *cam_vertex])

                    light_vertex = (light_trans @ obj_trans @ vertex).squeeze()[:3]
                    xmin, xmax = min(xmin, light_vertex[0]), max(xmax, light_vertex[0])
                    ymin, ymax = min(ymin, light_vertex[1]), max(ymax, light_vertex[1])
                    lv_tmp.append(light_vertex)

                # frustum clipping
                if max(-cv_tmp[0][2], -cv_tmp[1][2], -cv_tmp[2][2]) > far_clip or \
                        min(-cv_tmp[0][2], -cv_tmp[1][2], -cv_tmp[2][2]) < near_clip:
                    continue

                if triangle_area(cv_tmp[0], cv_tmp[1], cv_tmp[2]) > 0:
                    mesh['cam_vertices'].extend(cam_vertices)

                if triangle_area(lv_tmp[0], lv_tmp[1], lv_tmp[2]) > 0:
                    mesh['light_vertices'].extend([*lv_tmp[0], *lv_tmp[1], *lv_tmp[2]])

    max_dim = max(xmax - xmin, ymax - ymin)
    shadow_map_scale = scene.lights[0].shadow_map_dim / (max_dim * 1.2)
    scene.lights[0].shadow_map_param = [xmin - (xmax - xmin) * 0.1, shadow_map_scale,
                                        ymin - (ymax - ymin) * 0.1, shadow_map_scale]


def shadow_mapping(scene, shadow_map):
    light = scene.lights[0]

    assert shadow_map.shape[0] == shadow_map.shape[1]
    shadow_map_dim = shadow_map.shape[0]

    for obj in scene.objects:
        for name, mesh in obj.mesh.mesh.items():
            light_vertices = mesh['light_vertices']
            for i in range(0, len(light_vertices), 9):
                v0 = np.round(light.project(light_vertices[i+0:i+3])).astype(int)
                v1 = np.round(light.project(light_vertices[i+3:i+6])).astype(int)
                v2 = np.round(light.project(light_vertices[i+6:i+9])).astype(int)

                area = triangle_area(v0, v1, v2)
                if area <= 0:
                    continue

                ymax, ymin = max(v0[1], v1[1], v2[1]), min(v0[1], v1[1], v2[1])
                xmax, xmin = max(v0[0], v1[0], v2[0]), min(v0[0], v1[0], v2[0])

                for y in range(ymin, ymax+1):
                    for x in range(xmin, xmax+1):
                        w0 = triangle_area(v1, v2, (x, y))
                        w1 = triangle_area(v2, v0, (x, y))
                        w2 = triangle_area(v0, v1, (x, y))

                        if (w0 >= 0) and (w1 >= 0) and (w2 >= 0):
                            w0, w1, w2 = w0 / area, w1 / area, w2 / area
                            depth = -(w0 * v0[2] + w1 * v1[2] + w2 * v2[2])
                            shadow_map[shadow_map_dim-y-1, x] = min(shadow_map[shadow_map_dim-y-1, x], depth)


def rasterization(scene, depth_buffer, raster_buffer, shadow_map):
    camera = scene.camera
    height, width = depth_buffer.shape

    light = scene.lights[0]
    lnorm = (camera.world_to_camera[:3, :3] @ \
        light.light_to_world[:3, :3] @ np.array([[0], [0], [1]])).squeeze()
    lnorm = lnorm / np.linalg.norm(lnorm)
    lambt = light.ambient

    for obj in scene.objects:
        for name, mesh in obj.mesh.mesh.items():
            mesh_format = mesh['format']
            vertices = mesh['cam_vertices']

            if mesh_format == 'V3F':
                step = 3
                vi0, vi1 = 0, 3
            elif mesh_format == 'N3F_V3F':
                step = 6
                ni0, ni1 = 0, 3
                vi0, vi1 = 3, 6
            elif mesh_format == 'T2F_V3F':
                step = 5
                ti0. ti1 = 0, 2
                vi0, vi1 = 2, 5
            elif mesh_format == 'T2F_N3F_V3F':
                step = 8
                ti0, ti1 = 0, 2
                ni0, ni1 = 2, 5
                vi0, vi1 = 5, 8

            for i in range(0, len(vertices), step*3):
                i0, i1, i2 = i+0*step, i+1*step, i+2*step
                v0 = np.array(vertices[i0+vi0:i0+vi1])
                v1 = np.array(vertices[i1+vi0:i1+vi1])
                v2 = np.array(vertices[i2+vi0:i2+vi1])

                # y inverted
                c0 = camera.project(v0)
                c1 = camera.project(v1)
                c2 = camera.project(v2)

                area = triangle_area(c2, c1, c0)
                if area <= 0:
                    continue

                if 'N3F' in mesh_format:
                    n0 = np.array(vertices[i0+ni0:i0+ni1])
                    n1 = np.array(vertices[i1+ni0:i1+ni1])
                    n2 = np.array(vertices[i2+ni0:i2+ni1])
                else:
                    n_default = np.cross(v1-v0, v2-v0)
                    n_default = n_default / np.linalg.norm(n_default)
                    n0 = n_default
                    n1 = n_default
                    n2 = n_default

                ymax = int(np.round(min(height-1, max(c0[1], c1[1], c2[1]))))
                ymin = int(np.round(max(0, min(c0[1], c1[1], c2[1]))))
                xmax = int(np.round(min(width-1, max(c0[0], c1[0], c2[0]))))
                xmin = int(np.round(max(0, min(c0[0], c1[0], c2[0]))))

                l0 = np.clip(np.dot(n0, lnorm), lambt, 1.0)
                l1 = np.clip(np.dot(n1, lnorm), lambt, 1.0)
                l2 = np.clip(np.dot(n2, lnorm), lambt, 1.0)

                z0_1 = 1 / c0[2]
                z1_1 = 1 / c1[2]
                z2_1 = 1 / c2[2]

                for y in range(ymin, ymax+1):
                    for x in range(xmin, xmax+1):
                        w0 = triangle_area((x, y), c2, c1) / area
                        w1 = triangle_area((x, y), c0, c2) / area
                        w2 = triangle_area((x, y), c1, c0) / area

                        z_1 = w0*z0_1 + w1*z1_1 + w2*z2_1
                        z = 1 / z_1

                        if (w0 >= 0) and (w1 >= 0) and (w2 >= 0) and (z < depth_buffer[y, x]):
                                raster_buffer[y, x] = w0*l0 + w1*l1 + w2*l2
                                depth_buffer[y, x] = z


def texture_mapping(scene):
    pass


def render(scene, image_width, image_height):
    shadow_map_dim = scene.lights[0].shadow_map_dim
    shadow_map_depth = scene.lights[0].shadow_map_depth
    shadow_map = np.ones((shadow_map_dim, shadow_map_dim)) * shadow_map_depth
    depth_buffer = np.ones((image_height, image_width)) * scene.camera.far_clip
    raster_buffer = np.zeros((image_height, image_width, 4))  # texture ID, texture Y, texture X, luminance
    frame_buffer = np.zeros((image_height, image_width, 1))

    # geometric (mesh transform, clipping, backface culling)
    t0 = time()
    geometric_transform(scene)
    print('{:25s}:'.format('geometric_transform()'), time() - t0)

    # shadow mapping
    t0 = time()
    shadow_mapping(scene, shadow_map)
    print('{:25s}:'.format('shadow_mapping()'), time() - t0)
    cv2.imshow('', shadow_map / (shadow_map.max() - shadow_map.min() + 1e-9))
    cv2.waitKey()

    # rasterization (z buffer)
    t0 = time()
    rasterization(scene, depth_buffer, frame_buffer, shadow_map)
    print('{:25s}:'.format('rasterization()'), time() - t0)
    # cv2.imshow('', depth_buffer / (depth_buffer.max() - depth_buffer.min() + 1e-9))
    cv2.imshow('', frame_buffer)
    cv2.waitKey()

    # texture mapping
    t0 = time()
    texture_mapping(scene)
    print('{:25s}:'.format('texture_mapping()'), time() - t0)
    cv2.imshow('', frame_buffer[:, :, 3])
    cv2.waitKey()

    pass


if __name__ == '__main__':
    scene = Scene()

    obj = GameObject()
    obj.load_mesh('../data/deer.obj')
    scene.objects += [obj]

    scene.lights += [Light()]
    scene.camera = Camera(0.98, 0.735, 640, 480, 1, 10000, 20, np.eye(4))
    # scene.camera.translate_x(25)
    # scene.camera.translate_y(8)
    # scene.camera.translate_z(25)
    # scene.camera.rotate_y(45)

    scene.camera.translate_x(1500)
    scene.camera.translate_y(800)
    scene.camera.translate_z(1500)
    scene.camera.rotate_y(45)

    render(scene, 640, 480)
