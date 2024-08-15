import trimesh
import numpy as np
import open3d as o3d
import os
import json
import torch
from tqdm import tqdm

X_RES = 64
Y_RES = 64
NUM_SLICE = 64


def mesh_process_pipeline(mesh_path, output_path):
    original_mesh = trimesh.load_mesh(mesh_path)
    original_mesh.vertices -= original_mesh.centroid
    original_mesh.vertices /= np.max(np.linalg.norm(original_mesh.vertices, axis=1))
    centroid = original_mesh.centroid
    total_range = np.max(original_mesh.vertices, axis=0) - np.min(
        original_mesh.vertices, axis=0
    )

    top = None

    depth_images = []
    distance_fields = []
    for bottom in np.linspace(
        original_mesh.bounds[1, 2], original_mesh.bounds[0, 2], NUM_SLICE + 1
    ):
        if top is None:
            top = bottom
            continue
        sliced_mesh = trimesh.intersections.slice_mesh_plane(
            original_mesh,
            plane_normal=[0, 0, -1],
            plane_origin=[
                original_mesh.vertices[:, 0].mean(),
                original_mesh.vertices[:, 1].mean(),
                top,
            ],
        )

        min_bounds = np.min(original_mesh.vertices, axis=0)
        max_bounds = np.max(original_mesh.vertices, axis=0)

        trimesh.exchange.export.export_mesh(sliced_mesh, f"{output_path}/tmp_slice.stl")
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.io.read_triangle_mesh(f"{output_path}/tmp_slice.stl")
        )
        # Create scene and add the cube mesh
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)

        # Rays are 6D vectors with origin and ray direction.
        # Here we use a helper function to create rays for a pinhole camera.

        stride = np.max((max_bounds[:2] - min_bounds[:2]) / np.max([X_RES, Y_RES]))

        x = np.arange(
            centroid[0] - stride * X_RES / 2,
            centroid[0] + stride * X_RES / 2,
            stride,
        )
        y = np.arange(
            centroid[1] - stride * Y_RES / 2,
            centroid[1] + stride * Y_RES / 2,
            stride,
        )

        # full coordinate arrays
        xx, yy = np.meshgrid(x, y)

        current_plane = np.stack(
            [
                xx.flatten(),
                yy.flatten(),
                bottom * np.ones_like(xx.flatten()),
            ],
            axis=1,
        )
        distance = scene.compute_distance(
            o3d.core.Tensor(current_plane, dtype=o3d.core.Dtype.Float32)
        )
        distance_field = distance.cpu().numpy().reshape(Y_RES, X_RES)
        # plt.imshow(distance.cpu().numpy().reshape(Y_RES, X_RES))
        # plt.show()

        ray_center = centroid
        rays = scene.create_rays_pinhole(
            fov_deg=120,
            center=ray_center,
            eye=np.array(
                [ray_center[0], ray_center[1] + 0.000001, top + total_range[2] * 0.5]
            ),
            up=[0, 0, 1],
            width_px=X_RES,
            height_px=Y_RES,
        )

        # Compute the ray intersections.
        ans = scene.cast_rays(rays)

        # Visualize the hit distance (depth)

        depth_image = ans["t_hit"].cpu().numpy().reshape(Y_RES, X_RES)
        depth_image = np.nan_to_num(depth_image, nan=0.0, posinf=0.0, neginf=0.0)
        # plt.imshow(depth_image)
        # plt.show()

        depth_images.append(depth_image)
        distance_fields.append(distance_field)
        top = bottom

    # data/parts_3, files 1 through 5500/rotated_files/4ff20d53-30f2-426b-9f2c-2410e4275d14.stl0-10.stl

    write_path = mesh_path.replace("rotated_files", "depth_image")
    os.makedirs("/".join(write_path.split("/")[:-1]), exist_ok=True)
    torch.save(
        torch.from_numpy(np.array(depth_images)),
        write_path + ".pth",
    )

    write_path = mesh_path.replace("rotated_files", "distance_field")
    os.makedirs("/".join(write_path.split("/")[:-1]), exist_ok=True)
    torch.save(
        torch.from_numpy(np.array(distance_fields)),
        write_path + ".pth",
    )


with open("/global/scratch/users/ethantqiu/data/agg.json") as r:
    run_dict = json.load(r)

for file in tqdm(run_dict.keys()):
    mesh_process_pipeline(f"/global/scratch/users/ethantqiu/{file}", "/global/scratch/users/ethantqiu/data")
