import trimesh
import numpy as np


def voxelize(mesh_path, voxel_dim):
    mesh = trimesh.load(mesh_path)
    return mesh.voxelized(voxel_dim).matrix


def point_cloudify(mesh_path, num_pts):
    mesh = trimesh.load(mesh_path)
    return mesh.sample(num_pts)


def seven_dim_extraction(mesh_path):
    mesh = trimesh.load(mesh_path)
    mesh.fix_normals()
    normals = mesh.face_normals
    centers = mesh.triangles_center
    sizes = mesh.area_faces
    return np.hstack([centers, normals, sizes])
