import trimesh
import numpy as np
import pdb


def voxelize(mesh_path, voxel_dim):
    mesh = trimesh.load(mesh_path)
    return mesh.voxelized(voxel_dim).matrix


def point_cloudify(mesh_path, num_pts=2048):
    mesh = trimesh.load(mesh_path)
    tsfm_matrix = np.eye(4)
    tsfm_matrix[:3, 3] = -mesh.center_mass
    mesh = mesh.apply_transform(tsfm_matrix)
    centered_pts = mesh.sample(num_pts)
    avg_dist = np.average(np.linalg.norm(centered_pts, axis=-1))
    centered_pts /= avg_dist
    # centered_pts = np.hstack([centered_pts, np.ones(len(centered_pts)).reshape(-1, 1)])
    return centered_pts.T


def seven_dim_extraction(mesh_path):
    mesh = trimesh.load(mesh_path)
    tsfm_matrix = np.eye(4)
    tsfm_matrix[:3, 3] = -mesh.center_mass
    mesh = mesh.apply_transform(tsfm_matrix)
    mesh.fix_normals()
    normals = mesh.face_normals
    largest_dist = 1
    centers = mesh.triangles_center / largest_dist
    sizes = mesh.area_faces.reshape((-1, 1)) / (largest_dist**2)
    out = np.hstack([centers, normals, sizes])
    if out.shape[0] >= 512:
        out = out[out[:, -1].argsort()[::-1]]
        out = out[:512], np.zeros(512, dtype=bool)
    elif out.shape[0] < 512:
        mask = np.hstack([np.zeros(out.shape[0]), np.ones(512 - out.shape[0])])
        out = np.vstack([out, np.zeros((512 - out.shape[0], 7))])
        out = out, mask
    return out
